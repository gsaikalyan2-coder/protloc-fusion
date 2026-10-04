"""Phase 13 homology evidence: masking, GO transfer, independence, configuration, the manifest."""

import ast
from collections.abc import Callable
import copy
import math
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from protloc.evidence import homology as hm
from protloc.evidence.homology import Entry, Hit, HitFilter, HomologyError
from protloc.registry import evidence as reg
from protloc.training import baseline_models as bm
from protloc.training import go, homology_eval
from protloc.training.baselines import (
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    digest,
    lopo_folds,
)

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "backend" / "protloc"
YAML = REPO / "configs" / "training" / "homology.yaml"
SHA = "a" * 64
OTHER = "b" * 64
MODULES = [
    PACKAGE / "evidence" / "homology.py",
    PACKAGE / "registry" / "evidence.py",
    PACKAGE / "training" / "homology_eval.py",
]


@pytest.fixture
def np() -> Any:
    """numpy, or skip the test."""
    return pytest.importorskip("numpy")


def _yaml() -> dict[str, Any]:
    doc = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _config() -> homology_eval.HomologyConfig:
    return homology_eval.load_config(YAML)


def _hit(  # noqa: PLR0913
    query: str,
    target: str,
    *,
    identity: float = 0.9,
    evalue: float = 1e-30,
    coverage: float = 0.9,
    bits: float = 100.0,
) -> Hit:
    return Hit(query, target, identity, coverage, coverage, evalue, bits, 100)


# -- configuration ---------------------------------------------------------------------
def test_the_configuration_records_the_gate_decisions() -> None:
    config = _config()
    assert config.seed == _yaml()["seed"] == 20260909
    assert config.engine.release == "18-8cc5c"
    assert config.engine.platform == "win64"
    assert config.engine.license == "MIT"
    assert config.database.release == "2026_03"
    assert config.database.records == 548702
    assert config.masks == hm.MASKS
    assert config.registry_only_diagnostic is True
    assert config.go_config == "configs/training/go.yaml"


def test_the_seed_agrees_with_the_root_config() -> None:
    root = yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert _config().seed == root["seed"]


def test_the_grid_is_108_points_in_the_documented_order() -> None:
    points = homology_eval.grid_points(_config())
    assert len(points) == 4 * 3 * 3 * 3
    assert points[0] == HitFilter(1e-20, 0.0, 0.0, 1)
    assert points[1] == HitFilter(1e-20, 0.0, 0.0, 3)
    assert points[3] == HitFilter(1e-20, 0.0, 0.5, 1)
    assert points[-1] == HitFilter(1e-3, 0.5, 0.8, 5)
    assert len(set(points)) == len(points)


def test_the_engine_pins_are_the_downloaded_release() -> None:
    engine = _config().engine
    assert len(engine.archive_sha256) == len(engine.executable_sha256) == 64
    assert engine.download_url.endswith(f"/{engine.release}/mmseqs-win64.zip")
    assert engine.license_url.endswith("LICENSE.md")


def test_the_search_is_permissive_enough_for_the_whole_grid() -> None:
    config = _config()
    assert config.search.e_value >= max(config.grid["e_value"])


def test_the_tiebreak_weight_cannot_outrank_one_vote() -> None:
    config = _config()
    assert config.transfer.tiebreak_weight < 1.0 / max(config.grid["top_k"])


def test_config_values_are_not_repeated_as_literals_in_the_modules() -> None:
    config = _config()
    forbidden = {
        config.seed,
        config.transfer.tiebreak_weight,
        config.search.sensitivity,
        config.search.e_value,
        config.search.max_seqs,
        config.min_free_bytes,
        config.poll_interval,
        config.transfer.min_rows_per_bin,
        config.smoke.bootstrap_samples,
    }
    for module in MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        literals = {
            node.value
            for function in functions
            for node in ast.walk(function)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        }
        assert not (literals & forbidden), (module.name, sorted(literals & forbidden))


def _edited(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    doc = _yaml()
    edit(doc)
    path = tmp_path / "homology.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(task="go_function"),
        lambda d: d["masking"].update(masks=["self_accession", "self_sequence_sha256"]),
        lambda d: d["masking"].update(masks=list(reversed(d["masking"]["masks"]))),
        lambda d: d["transfer"].update(tiebreak_weight=0.5),
        lambda d: d["transfer"].update(identity_bins=[0.0, 0.5]),
        lambda d: d["transfer"].update(identity_bins=[0.1, 0.5, 1.0]),
        lambda d: d["transfer"].update(identity_bins=[0.0, 0.7, 0.5, 1.0]),
        lambda d: d["search"].update(e_value=1e-30),
        lambda d: d["hit_filter"]["grid"].pop("top_k"),
        lambda d: d["hit_filter"]["grid"].update(min_identity=[1.5]),
        lambda d: d["hit_filter"]["grid"].update(top_k=[0]),
        lambda d: d["engine"].update(executable_sha256="xyz"),
        lambda d: d["database"].update(corpus_sha256="short"),
    ],
)
def test_a_malformed_configuration_is_refused(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    with pytest.raises(BaselineError):
        homology_eval.load_config(_edited(tmp_path, edit))


def test_the_configuration_hash_follows_every_byte(tmp_path: Path) -> None:
    other = _edited(tmp_path, lambda d: d["smoke"].update(query_stride=3))
    assert homology_eval.load_config(other).sha256 != _config().sha256


# -- parsing ---------------------------------------------------------------------------
def test_hits_parse_group_and_round_trip_the_columns() -> None:
    text = "q1\tP1\t0.9\t0.8\t0.7\t1e-30\t120.5\t200\nq1\tP2\t0.4\t0.6\t0.5\t1e-5\t60\t90\n"
    grouped = hm.parse_hits(text)
    assert [h.target for h in grouped["q1"]] == ["P1", "P2"]
    assert grouped["q1"][0].identity == 0.9
    assert grouped["q1"][0].alignment_length == 200


@pytest.mark.parametrize(
    "text",
    [
        "q1\tP1\t0.9\n",
        "q1\tP1\t90.0\t0.8\t0.7\t1e-30\t120\t200\n",
        "q1\tP1\tx\t0.8\t0.7\t1e-30\t120\t200\n",
    ],
)
def test_a_malformed_hit_line_is_refused(text: str) -> None:
    with pytest.raises(HomologyError):
        hm.parse_hits(text)


def test_the_entry_table_round_trips_and_refuses_what_it_does_not_write() -> None:
    entries = [Entry("A", SHA, 1, "c1"), Entry("B", OTHER, None, None)]
    parsed = hm.parse_entries(hm.entries_text(entries))
    assert parsed["A"] == entries[0]
    assert parsed["B"].partition is None
    with pytest.raises(HomologyError):
        hm.parse_entries("accession\tsha\n")
    with pytest.raises(HomologyError):
        hm.parse_entries(hm.entries_text(entries) + "A\t" + SHA + "\t1\tc1\n")


# -- leakage masks ---------------------------------------------------------------------
def _database() -> dict[str, Entry]:
    """The query Q (partition 2, cluster c2) and the sequences planted around it."""
    entries = [
        Entry("Q", SHA, 2, "c2"),
        Entry("SELF_COPY", SHA, 4, "c9"),
        Entry("SAME_CLUSTER", OTHER, 2, "c2"),
        Entry("SAME_PARTITION", "c" * 64, 2, "c7"),
        Entry("OTHER_PARTITION", "d" * 64, 3, "c3"),
        Entry("UNREGISTERED", "e" * 64, None, None),
    ]
    return {e.accession: e for e in entries}


def test_a_planted_self_hit_is_removed_by_accession() -> None:
    database = _database()
    kept, report = hm.mask_hits(
        database["Q"], [_hit("Q", "Q"), _hit("Q", "OTHER_PARTITION")], database
    )
    assert [h.target for h in kept] == ["OTHER_PARTITION"]
    assert report.exclusive[hm.MASK_SELF_ACCESSION] == 1


def test_a_planted_self_sequence_under_another_accession_is_removed_by_hash() -> None:
    database = _database()
    kept, report = hm.mask_hits(database["Q"], [_hit("Q", "SELF_COPY")], database)
    assert kept == []
    assert report.exclusive[hm.MASK_SELF_SEQUENCE] == 1
    assert report.exclusive[hm.MASK_SELF_ACCESSION] == 0


def test_a_planted_same_cluster_hit_is_removed() -> None:
    database = _database()
    kept, report = hm.mask_hits(database["Q"], [_hit("Q", "SAME_CLUSTER")], database)
    assert kept == []
    assert report.exclusive[hm.MASK_SAME_CLUSTER] == 1


def test_a_planted_same_partition_hit_is_removed() -> None:
    database = _database()
    kept, report = hm.mask_hits(database["Q"], [_hit("Q", "SAME_PARTITION")], database)
    assert kept == []
    assert report.exclusive[hm.MASK_SAME_PARTITION] == 1


def test_other_partitions_and_unregistered_sequences_survive() -> None:
    database = _database()
    hits = [_hit("Q", "OTHER_PARTITION"), _hit("Q", "UNREGISTERED")]
    kept, report = hm.mask_hits(database["Q"], hits, database)
    assert [h.target for h in kept] == ["OTHER_PARTITION", "UNREGISTERED"]
    assert report.raw == 2
    assert report.kept == 2


def test_the_counts_attribute_every_removed_hit_to_exactly_one_mask() -> None:
    database = _database()
    hits = [_hit("Q", t) for t in database if t != "Q"] + [_hit("Q", "Q")]
    kept, report = hm.mask_hits(database["Q"], hits, database)
    assert report.raw - report.kept == sum(report.exclusive.values()) == len(hits) - len(kept)
    assert report.exclusive == {
        hm.MASK_SELF_ACCESSION: 1,
        hm.MASK_SELF_SEQUENCE: 1,
        hm.MASK_SAME_CLUSTER: 1,
        hm.MASK_SAME_PARTITION: 1,
    }
    # Q matches three masks by itself; the independent counts show the overlap.
    assert report.independent[hm.MASK_SAME_CLUSTER] == 2
    assert report.independent[hm.MASK_SAME_PARTITION] == 3


def test_masking_is_not_optional_a_planted_hit_never_reaches_the_votes() -> None:
    database = _database()
    terms = {"SAME_CLUSTER": frozenset({"T1"}), "OTHER_PARTITION": frozenset({"T2"})}
    hits = [_hit("Q", "SAME_CLUSTER", evalue=1e-90), _hit("Q", "OTHER_PARTITION", evalue=1e-5)]
    kept, _report = hm.mask_hits(database["Q"], hits, database)
    chosen = hm.candidates(kept, HitFilter(1e-3, 0.0, 0.0, 1), terms)
    assert [h.target for h in chosen] == ["OTHER_PARTITION"]
    assert hm.vote(chosen, terms) == {"T2": 1.0}


def test_an_unregistered_query_cannot_be_scored() -> None:
    database = _database()
    with pytest.raises(HomologyError):
        hm.mask_hits(database["UNREGISTERED"], [], database)


def test_a_hit_target_missing_from_the_table_is_an_error_not_a_pass() -> None:
    database = _database()
    with pytest.raises(HomologyError):
        hm.mask_hits(database["Q"], [_hit("Q", "GHOST")], database)


def test_reports_add_up_over_a_query_set() -> None:
    database = _database()
    _a, first = hm.mask_hits(database["Q"], [_hit("Q", "Q")], database)
    _b, second = hm.mask_hits(database["Q"], [_hit("Q", "SAME_CLUSTER")], database)
    total = hm.add_reports([first, second])
    assert total.raw == 2
    assert total.kept == 0
    assert total.exclusive[hm.MASK_SELF_ACCESSION] == 1
    assert total.exclusive[hm.MASK_SAME_CLUSTER] == 1


def test_the_registry_restriction_keeps_only_registered_targets() -> None:
    database = _database()
    hits = [_hit("Q", "OTHER_PARTITION"), _hit("Q", "UNREGISTERED")]
    assert [h.target for h in hm.in_registry(hits, database)] == ["OTHER_PARTITION"]


# -- hit filter and GO transfer, by hand -----------------------------------------------
TERMS = {
    "H1": frozenset({"A", "B"}),
    "H2": frozenset({"A"}),
    "H3": frozenset(),
    "H4": frozenset({"C"}),
}


def _hits() -> list[Hit]:
    return [
        _hit("Q", "H1", identity=0.95, evalue=1e-60),
        _hit("Q", "H3", identity=0.80, evalue=1e-40),
        _hit("Q", "H2", identity=0.55, evalue=1e-20),
        _hit("Q", "H4", identity=0.35, evalue=1e-8, coverage=0.4),
    ]


def test_the_filter_keeps_hits_by_evalue_identity_and_query_coverage() -> None:
    rule = HitFilter(1e-10, 0.5, 0.5, 5)
    assert [h.target for h in hm.passing(_hits(), rule)] == ["H1", "H3", "H2"]
    assert [h.target for h in hm.passing(_hits(), HitFilter(1e-30, 0.0, 0.0, 5))] == ["H1", "H3"]
    assert [h.target for h in hm.passing(_hits(), HitFilter(1e-3, 0.0, 0.5, 5))] == [
        "H1",
        "H3",
        "H2",
    ]


def test_hits_rank_by_evalue_then_bits_then_accession() -> None:
    tied = [_hit("Q", "Z", bits=50), _hit("Q", "A", bits=50), _hit("Q", "M", bits=90)]
    assert [h.target for h in hm.passing(tied, HitFilter(1.0, 0.0, 0.0, 3))] == ["M", "A", "Z"]


def test_an_unannotated_hit_does_not_take_a_vote_place() -> None:
    rule = HitFilter(1e-3, 0.0, 0.0, 2)
    assert [h.target for h in hm.candidates(_hits(), rule, TERMS)] == ["H1", "H2"]
    unfiltered = hm.candidates(_hits(), rule, TERMS, annotated_only=False)
    assert [h.target for h in unfiltered] == ["H1", "H3"]


def test_a_vote_is_the_fraction_of_the_top_k_hits_carrying_the_term() -> None:
    chosen = hm.candidates(_hits(), HitFilter(1e-3, 0.0, 0.0, 3), TERMS)
    assert [h.target for h in chosen] == ["H1", "H2", "H4"]
    assert hm.vote(chosen, TERMS) == {"A": 2 / 3, "B": 1 / 3, "C": 1 / 3}
    assert hm.vote(chosen, TERMS, ["A", "C"]) == {"A": 2 / 3, "C": 1 / 3}


def test_the_best_hit_arm_transfers_exactly_its_terms() -> None:
    chosen = hm.candidates(_hits(), HitFilter(1e-3, 0.0, 0.0, 1), TERMS)
    assert hm.vote(chosen, TERMS) == {"A": 1.0, "B": 1.0}


def test_no_passing_hit_gives_no_vote() -> None:
    assert hm.vote([], TERMS) == {}
    assert hm.candidates(_hits(), HitFilter(1e-90, 0.0, 0.0, 3), TERMS) == []


def test_transferred_terms_say_homology_and_name_the_hit_they_came_from() -> None:
    chosen = hm.candidates(_hits(), HitFilter(1e-3, 0.0, 0.0, 3), TERMS)
    out = hm.transfer_terms(chosen, TERMS)
    assert [t.term for t in out] == ["A", "B", "C"]
    assert {t.source for t in out} == {"homology"} == {hm.SOURCE_HOMOLOGY}
    assert {t.database for t in out} == {"swissprot"}
    by_term = {t.term: t for t in out}
    assert by_term["A"].source_accession == "H1"
    assert by_term["A"].identity == 0.95
    assert by_term["C"].source_accession == "H4"
    assert by_term["A"].score == 2 / 3


def test_a_homology_term_is_never_labelled_as_a_model_term() -> None:
    assert hm.SOURCE_HOMOLOGY != go.SOURCE_MODEL


def test_nearest_rank_percentiles_by_hand() -> None:
    values = [float(v) for v in range(1, 21)]
    assert hm.nearest_rank(values, 50) == 10.0
    assert hm.nearest_rank(values, 95) == 19.0
    assert hm.nearest_rank([3.0], 95) == 3.0
    with pytest.raises(HomologyError):
        hm.nearest_rank([], 50)
    with pytest.raises(HomologyError):
        hm.nearest_rank(values, 0)


def test_identity_bins_are_half_open_and_the_last_is_closed() -> None:
    edges = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    assert hm.identity_bin(0.0, edges) == 0
    assert hm.identity_bin(0.3, edges) == 1
    assert hm.identity_bin(0.8999, edges) == 3
    assert hm.identity_bin(1.0, edges) == 4
    assert hm.identity_bin(1.2, edges) is None
    assert hm.bin_label(edges, 0) == "[0, 0.3)"
    assert hm.bin_label(edges, 4) == "[0.9, 1]"


# -- votes as arrays -------------------------------------------------------------------
def _vocab(*ids: str) -> go.Vocabulary:
    return go.Vocabulary(
        terms=tuple(go.VocabTerm(i, i.lower(), 30, {}, ()) for i in ids), undefined={}, floor=20
    )


def test_vote_scores_by_hand(np: Any) -> None:
    hits = {"q1": _hits(), "q2": []}
    spec = (("A", "B", "C"), HitFilter(1e-3, 0.0, 0.0, 3), True)
    votes = homology_eval.vote_scores(["q1", "q2"], hits, TERMS, spec)
    assert votes.score.tolist() == [[2 / 3, 1 / 3, 1 / 3], [0.0, 0.0, 0.0]]
    assert votes.any_hit.tolist() == [True, False]
    assert votes.any_candidate.tolist() == [True, False]
    assert votes.best_identity[0] == 0.95
    assert math.isnan(votes.best_identity[1])


def test_a_hit_that_passes_but_has_no_annotation_is_a_hit_without_a_candidate(np: Any) -> None:
    hits = {"q": [_hit("q", "H3")]}
    spec = (("A",), HitFilter(1e-3, 0.0, 0.0, 1), True)
    votes = homology_eval.vote_scores(["q"], hits, TERMS, spec)
    assert votes.any_hit.tolist() == [True]
    assert votes.any_candidate.tolist() == [False]
    assert votes.score.tolist() == [[0.0]]


def test_the_prior_only_breaks_ties_between_equal_votes(np: Any) -> None:
    vocab = _vocab("A", "B", "C")
    config = _config()
    weight = config.transfer.tiebreak_weight
    votes = np.array([[0.4, 0.4, 0.2]])
    prior = np.array([[0.1, 0.9, 1.0]])
    scored = homology_eval.with_prior(votes, prior, weight, vocab)
    assert scored[0, 1] > scored[0, 0] > scored[0, 2]
    gap = 1.0 / max(config.grid["top_k"])
    assert weight * 1.0 < gap


# -- grid search -----------------------------------------------------------------------
def _world(np: Any) -> tuple[go.Rows, dict[str, list[Hit]], dict[str, frozenset[str]]]:
    accession, partition, y = [], [], []
    hits: dict[str, list[Hit]] = {}
    for p in VALIDATION_PARTITIONS:
        for k in range(4):
            name = f"p{p}_{k}"
            accession.append(name)
            partition.append(p)
            y.append([1, 0] if k % 2 == 0 else [0, 1])
            target = "TA" if k % 2 == 0 else "TB"
            hits[name] = [_hit(name, target, identity=0.9, evalue=1e-30)]
    rows = go.Rows(
        accession=accession,
        shas=[a * 2 for a in accession],
        y=np.array(y, dtype=np.int64),
        partition=np.array(partition),
        cluster_code=np.arange(len(accession)),
        seq_index=np.arange(len(accession)),
    )
    return rows, hits, {"TA": frozenset({"T1"}), "TB": frozenset({"T2"})}


def test_the_grid_scores_every_point_on_partitions_one_to_four_only(np: Any) -> None:
    rows, hits, terms = _world(np)
    vocab = _vocab("T1", "T2")
    config = _config()
    prior = go.fold_prior_scores(rows.y, rows.partition, lopo_folds())
    metrics = go.MetricSettings(
        k_values=(1, 3), primary_k=1, per_term_min_positives=20, max_threshold_candidates=1000
    )
    grid = homology_eval.run_grid(rows, hits, terms, (vocab, prior, config, metrics))
    assert len(grid) == 108
    assert {int(p) for p in rows.partition} == set(VALIDATION_PARTITIONS)
    assert TEST_PARTITION not in rows.partition
    chosen = homology_eval.select_row(grid)
    assert chosen.selector == 1.0
    assert chosen.index == 0, "ties go to the earlier grid point"
    assert chosen.hit_coverage == 1.0
    best = homology_eval.select_row([g for g in grid if g.rule.top_k == 1])
    assert best.rule.top_k == 1


def test_a_grid_point_that_loses_the_hits_scores_worse(np: Any) -> None:
    rows, hits, terms = _world(np)
    for name in hits:
        hits[name] = [_hit(name, hits[name][0].target, identity=0.2, evalue=1e-4)]
    vocab = _vocab("T1", "T2")
    prior = go.fold_prior_scores(rows.y, rows.partition, lopo_folds())
    metrics = go.MetricSettings(
        k_values=(1, 3), primary_k=1, per_term_min_positives=20, max_threshold_candidates=1000
    )
    grid = homology_eval.run_grid(rows, hits, terms, (vocab, prior, _config(), metrics))
    strict = next(g for g in grid if g.rule.e_value == 1e-20)
    loose = next(g for g in grid if g.rule == HitFilter(1e-3, 0.0, 0.0, 1))
    assert strict.hit_coverage == 0.0
    assert loose.hit_coverage == 1.0
    assert homology_eval.select_row(grid).rule.e_value == 1e-3


def test_the_coverage_block_counts_rows_with_any_passing_hit(np: Any) -> None:
    votes = homology_eval.Votes(
        score=np.zeros((3, 1)),
        any_hit=np.array([True, True, False]),
        any_candidate=np.array([True, False, False]),
        best_identity=np.array([0.9, 0.4, math.nan]),
    )
    block = homology_eval.coverage_block(votes, np.array([[1], [1], [1]]))
    assert block["rows_with_hit"] == 2
    assert block["hit_coverage"] == 2 / 3
    assert block["annotated_coverage"] == 1 / 3


# -- independence ----------------------------------------------------------------------
def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_the_evidence_module_imports_nothing_that_produces_a_score() -> None:
    names = _imports(PACKAGE / "evidence" / "homology.py")
    forbidden = ("protloc.training", "protloc.calibration", "protloc.models", "numpy", "torch")
    bad = {n for n in names if any(n == f or n.startswith(f + ".") for f in forbidden)}
    assert not bad, bad


def test_the_registry_entry_imports_no_head_or_calibrator() -> None:
    names = _imports(PACKAGE / "registry" / "evidence.py")
    assert not {n for n in names if n.startswith(("protloc.training", "protloc.models", "torch"))}
    assert "protloc.registry.bundle" not in names


def test_no_model_calibrator_or_head_code_imports_the_evidence_layer() -> None:
    allowed = {"homology_eval.py"}
    roots = [PACKAGE / "calibration", PACKAGE / "models", PACKAGE / "training"]
    files = [PACKAGE / "registry" / "bundle.py"]
    for root in roots:
        files += [p for p in root.rglob("*.py") if p.name not in allowed]
    for path in files:
        names = _imports(path)
        bad = {n for n in names if n.startswith(("protloc.evidence", "protloc.registry.evidence"))}
        assert not bad, (path.name, bad)


def test_no_head_bundle_or_calibrator_mentions_homology() -> None:
    for name in ("bundle.py",):
        assert "homology" not in (PACKAGE / "registry" / name).read_text(encoding="utf-8").lower()
    for path in (PACKAGE / "calibration").rglob("*.py"):
        assert "homology" not in path.read_text(encoding="utf-8").lower(), path.name


def test_the_homology_arms_are_built_without_any_model_score() -> None:
    source = (REPO / "scripts" / "homology.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        if function.name not in {"_score_arms", "_arm_votes"}:
            continue
        text = ast.get_source_segment(source, function) or ""
        for word in ("oof", "crossfit", "model", "calibrat", "raw_score"):
            assert word not in text.lower(), (function.name, word)


def test_homology_scores_do_not_change_when_the_model_scores_change(np: Any) -> None:
    rows, hits, terms = _world(np)
    vocab = _vocab("T1", "T2")
    spec = (vocab.ids, HitFilter(1e-3, 0.0, 0.0, 1), True)
    votes = homology_eval.vote_scores(rows.accession, hits, terms, spec)
    prior = np.full(rows.y.shape, 0.5)
    first = homology_eval.with_prior(votes.score, prior, 0.01, vocab)
    again = homology_eval.with_prior(votes.score, prior, 0.01, vocab)
    assert (first == again).all()
    scoring = go.Scoring(
        go.MetricSettings(
            k_values=(1,), primary_k=1, per_term_min_positives=20, max_threshold_candidates=100
        ),
        bm.Bootstrap(0, 1),
    )
    decision = first >= 0.5
    spec_bins = ([0.0, 0.5, 1.0], 1, scoring)

    def run(model_scores: Any) -> Any:
        arms = {
            homology_eval.ARM_VOTE: (first, decision),
            homology_eval.ARM_RAW: (model_scores, model_scores >= 0.5),
        }
        out = homology_eval.bin_reports(arms, rows, votes.best_identity, spec_bins)
        return out["[0.5, 1]"]["arms"][homology_eval.ARM_VOTE]

    assert run(np.zeros(rows.y.shape)) == run(np.ones(rows.y.shape))


def test_the_independence_statement_denies_every_feed() -> None:
    assert homology_eval.INDEPENDENCE == {
        "feeds_model_score": False,
        "feeds_calibrator": False,
        "feeds_threshold": False,
        "homology_arms_read_model_scores": False,
    }


# -- the registry entry ----------------------------------------------------------------
def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "db").write_bytes(b"database")
    (index_dir / "db.idx").write_bytes(b"index bytes")
    files = reg.index_files(index_dir, ["db", "db.idx"])
    sections: dict[str, dict[str, Any]] = {
        "database": {"corpus_sha256": SHA, "fasta_sha256": SHA, "meta_sha256": SHA},
        "engine": {"archive_sha256": SHA, "executable_sha256": SHA},
        "index": {"files": files, "index_sha256": reg.index_digest(files)},
        "go": {"sha256": SHA, "ontology_sha256": OTHER, "vocabulary_sha256": SHA},
        "hit_filter": {},
        "masking": {"masks": list(hm.MASKS)},
        "transfer": {},
        "independence": {
            "feeds_model_score": False,
            "feeds_calibrator": False,
            "feeds_threshold": False,
        },
    }
    document = reg.build_document(
        version="0.0.1", scope={"scope": "subset", "scope_label": "x"}, sections=sections
    )
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / reg.BUNDLE_FILE).write_text(__import__("json").dumps(document), encoding="utf-8")
    return directory, index_dir


def _rewrite(directory: Path, edit: Callable[[dict[str, Any]], None], *, rehash: bool) -> None:
    path = directory / reg.BUNDLE_FILE
    document = __import__("json").loads(path.read_text(encoding="utf-8"))
    edit(document)
    if rehash:
        document["bundle_sha256"] = reg.bundle_digest(document)
    path.write_text(__import__("json").dumps(document), encoding="utf-8")


def test_a_sound_bundle_loads(tmp_path: Path) -> None:
    directory, index_dir = _bundle(tmp_path)
    document = reg.load_evidence_bundle(
        directory, index_directory=index_dir, goa_sha256=SHA, ontology_sha256=OTHER
    )
    assert document["source"] == "homology"
    assert document["task"] == "homology_evidence"


def test_the_loader_refuses_a_changed_index(tmp_path: Path) -> None:
    directory, index_dir = _bundle(tmp_path)
    (index_dir / "db.idx").write_bytes(b"index bytez")
    with pytest.raises(reg.EvidenceBundleError, match="not the one the bundle records"):
        reg.load_evidence_bundle(directory, index_directory=index_dir)


def test_the_loader_refuses_a_missing_index_file(tmp_path: Path) -> None:
    directory, index_dir = _bundle(tmp_path)
    (index_dir / "db").unlink()
    with pytest.raises(reg.EvidenceBundleError, match="missing"):
        reg.load_evidence_bundle(directory, index_directory=index_dir)


def test_the_loader_refuses_a_changed_goa_source_or_ontology(tmp_path: Path) -> None:
    directory, _index = _bundle(tmp_path)
    with pytest.raises(reg.EvidenceBundleError, match="GO annotation source"):
        reg.load_evidence_bundle(directory, goa_sha256=OTHER)
    with pytest.raises(reg.EvidenceBundleError, match="ontology release"):
        reg.load_evidence_bundle(directory, ontology_sha256=SHA)


def test_the_loader_refuses_an_edited_bundle(tmp_path: Path) -> None:
    directory, _index = _bundle(tmp_path)
    _rewrite(directory, lambda d: d["go"].update(sha256=OTHER), rehash=False)
    with pytest.raises(reg.EvidenceBundleError, match="bundle_sha256"):
        reg.load_evidence_bundle(directory)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["masking"].update(masks=["self_accession"]),
        lambda d: d["independence"].update(feeds_model_score=True),
        lambda d: d["independence"].update(feeds_threshold=True),
        lambda d: d["independence"].pop("feeds_calibrator"),
        lambda d: d.update(source="model"),
        lambda d: d["index"].update(index_sha256="0" * 64),
        lambda d: d["go"].update(ontology_sha256="nope"),
    ],
)
def test_a_rehashed_but_unsound_bundle_is_still_refused(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    directory, _index = _bundle(tmp_path)
    _rewrite(directory, edit, rehash=True)
    with pytest.raises(reg.EvidenceBundleError):
        reg.load_evidence_bundle(directory)


# -- the manifest ----------------------------------------------------------------------
KS = (1, 3, 5)


def _arm_report() -> dict[str, Any]:
    entry = {"value": 0.5, "ci_low": 0.4, "ci_high": 0.6}
    return {"pooled": {"metrics": {name: dict(entry) for name in go.metric_names(KS)}}}


def _pins() -> homology_eval.Pins:
    config = _config()
    arms = homology_eval.arm_names(config, with_model=True)
    return homology_eval.Pins(
        registry_sha256="1" * 64,
        corpus_sha256="2" * 64,
        ontology_sha256="3" * 64,
        vocabulary_sha256="4" * 64,
        go_frozen_sha256="5" * 64,
        config_sha256="6" * 64,
        engine_release=config.engine.release,
        engine_commit=config.engine.commit,
        engine_executable_sha256=config.engine.executable_sha256,
        grid_size=108,
        k_values=KS,
        reproduction_tolerance=1e-6,
        arms_validation=arms,
        arms_test=tuple(a for a in arms if a not in homology_eval.MODEL_ARMS),
    )


def _manifest(pins: homology_eval.Pins) -> dict[str, Any]:
    frozen: dict[str, Any] = {
        "protocol": {
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
        },
        "inputs": {
            "registry_sha256": pins.registry_sha256,
            "corpus_sha256": pins.corpus_sha256,
            "ontology_sha256": pins.ontology_sha256,
            "vocabulary_sha256": pins.vocabulary_sha256,
            "go_manifest_frozen_sha256": pins.go_frozen_sha256,
            "homology_config_sha256": pins.config_sha256,
            **dict.fromkeys(
                (
                    "fasta_sha256",
                    "meta_sha256",
                    "index_sha256",
                    "queries_sha256",
                    "hits_sha256",
                    "label_table_sha256",
                    "feature_table_sha256",
                ),
                SHA,
            ),
        },
        "engine": {
            "release": pins.engine_release,
            "commit": pins.engine_commit,
            "executable_sha256": pins.engine_executable_sha256,
        },
        "masking": {
            "masks": list(hm.MASKS),
            "validation": {
                "raw": 10,
                "kept": 4,
                "exclusive": {
                    hm.MASK_SELF_ACCESSION: 1,
                    hm.MASK_SELF_SEQUENCE: 1,
                    hm.MASK_SAME_CLUSTER: 2,
                    hm.MASK_SAME_PARTITION: 2,
                },
            },
        },
        "selected": {"index": 7, "params": dict.fromkeys(homology_eval.GRID_KEYS, 1)},
        "thresholds": {a: 0.3 for a in pins.arms_validation if a not in homology_eval.MODEL_ARMS},
        "validation": {
            **{a: _arm_report() for a in pins.arms_validation},
            "model_reproduction": {"max_abs_difference": 0.0},
            "grid": [],
            "coverage": {},
            "identity_bins": {},
        },
        "measurements": {
            "index_build": {"seconds": 100.0, "index_bytes": 5, "peak_working_set_bytes": 9},
            "query_latency": {
                "p50_seconds": 1.0,
                "p95_seconds": 2.0,
                "cpu_seconds_per_query": 3.0,
            },
        },
        "independence": dict(homology_eval.INDEPENDENCE),
    }
    return {
        "schema_version": homology_eval.SCHEMA_VERSION,
        "phase": homology_eval.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }


def _refrozen(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    return manifest


def _tested(pins: homology_eval.Pins) -> dict[str, Any]:
    manifest = _manifest(pins)
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "earlier_touches": dict.fromkeys(homology_eval.EARLIER_HEADS, 1),
        "metrics": {a: _arm_report() for a in pins.arms_test},
    }
    return manifest


def test_a_sound_manifest_has_no_problems() -> None:
    pins = _pins()
    assert homology_eval.check_homology_manifest(_manifest(pins), pins) == []
    assert homology_eval.check_homology_manifest(_tested(pins), pins) == []


def _mutations() -> list[tuple[str, Callable[[dict[str, Any]], None]]]:
    return [
        ("engine commit", lambda f: f["engine"].update(commit="x")),
        ("engine hash", lambda f: f["engine"].update(executable_sha256="0" * 64)),
        ("registry hash", lambda f: f["inputs"].update(registry_sha256="0" * 64)),
        ("ontology hash", lambda f: f["inputs"].update(ontology_sha256="0" * 64)),
        ("go manifest hash", lambda f: f["inputs"].update(go_manifest_frozen_sha256="0" * 64)),
        ("config hash", lambda f: f["inputs"].update(homology_config_sha256="0" * 64)),
        ("index hash", lambda f: f["inputs"].update(index_sha256="short")),
        ("a mask dropped", lambda f: f["masking"].update(masks=list(hm.MASKS)[:3])),
        ("mask counts", lambda f: f["masking"]["validation"].update(kept=5)),
        (
            "a mask not counted",
            lambda f: f["masking"]["validation"]["exclusive"].pop("same_partition"),
        ),
        ("selected outside the grid", lambda f: f["selected"].update(index=108)),
        ("selected params", lambda f: f["selected"]["params"].pop("top_k")),
        ("independence", lambda f: f["independence"].update(feeds_model_score=True)),
        (
            "independence flag",
            lambda f: f["independence"].update(homology_arms_read_model_scores=True),
        ),
        ("validation partitions", lambda f: f["protocol"].update(validation_partitions=[1, 2, 3])),
        ("test partition", lambda f: f["protocol"].update(test_partition=4)),
        ("threshold", lambda f: f["thresholds"].update(homology_vote=math.nan)),
        ("an arm missing", lambda f: f["validation"].pop(homology_eval.ARM_PRIOR)),
        (
            "reproduction",
            lambda f: f["validation"]["model_reproduction"].update(max_abs_difference=1.0),
        ),
        (
            "a metric missing",
            lambda f: f["validation"][homology_eval.ARM_VOTE]["pooled"]["metrics"].pop("fmax"),
        ),
        (
            "latency not measured",
            lambda f: f["measurements"]["query_latency"].update(p95_seconds=None),
        ),
        (
            "index size not measured",
            lambda f: f["measurements"]["index_build"].update(index_bytes=0),
        ),
        (
            "memory not measured",
            lambda f: f["measurements"]["index_build"].update(peak_working_set_bytes=None),
        ),
    ]


@pytest.mark.parametrize(("name", "edit"), _mutations(), ids=[m[0] for m in _mutations()])
def test_a_mutated_frozen_block_is_refused(
    name: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    pins = _pins()
    manifest = _manifest(pins)
    edit(manifest["frozen"])
    # Re-hash so only the content problem, not the hash mismatch, can be what is reported.
    assert homology_eval.check_homology_manifest(_refrozen(manifest), pins), name


def test_an_edited_frozen_block_without_a_new_hash_is_refused() -> None:
    pins = _pins()
    manifest = _manifest(pins)
    manifest["frozen"]["thresholds"][homology_eval.ARM_VOTE] = 0.9
    assert any("frozen_sha256" in p for p in homology_eval.check_homology_manifest(manifest, pins))


def test_a_wrong_schema_or_phase_is_refused() -> None:
    pins = _pins()
    for key, value in (("schema_version", 99), ("phase", 12)):
        manifest = _manifest(pins)
        manifest[key] = value
        assert homology_eval.check_homology_manifest(manifest, pins)


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.update(touches=2),
        lambda t: t.update(touches=0),
        lambda t: t.update(frozen_sha256="0" * 64),
        lambda t: t["earlier_touches"].update(go=0),
        lambda t: t["earlier_touches"].update(crf=2),
        lambda t: t["earlier_touches"].pop("localization"),
        lambda t: t["metrics"].pop(homology_eval.ARM_PRIOR),
        lambda t: t["metrics"].update(model_raw=_arm_report()),
    ],
)
def test_a_second_or_foreign_test_block_is_refused(edit: Callable[[dict[str, Any]], None]) -> None:
    pins = _pins()
    manifest = _tested(pins)
    edit(manifest["test"])
    assert homology_eval.check_homology_manifest(manifest, pins)


def test_the_manifest_copy_is_independent_of_the_pins() -> None:
    pins = _pins()
    manifest = copy.deepcopy(_manifest(pins))
    assert homology_eval.check_homology_manifest(manifest, pins) == []
    manifest["frozen"]["inputs"]["registry_sha256"] = "0" * 64
    assert homology_eval.check_homology_manifest(manifest, pins) != []


# -- the engine ------------------------------------------------------------------------
def test_engine_commands_use_forward_slashes_and_the_configured_settings(tmp_path: Path) -> None:
    config = _config()
    exe = tmp_path / "bin" / "mmseqs.exe"
    q, t, r, tmp = (tmp_path / n for n in ("q", "t", "r", "tmp"))
    command = hm.search_command(exe, (q, t, r, tmp), config.search)
    assert all("\\" not in part for part in command)
    assert command[command.index("-s") + 1] == str(config.search.sensitivity)
    assert command[command.index("-e") + 1] == str(config.search.e_value)
    assert command[command.index("--max-seqs") + 1] == str(config.search.max_seqs)
    assert command[command.index("--alignment-mode") + 1] == str(config.search.alignment_mode)
    convert = hm.convertalis_command(exe, (q, t, r, tmp / "o.tsv"), 2, 1)
    assert convert[convert.index("--format-output") + 1] == ",".join(hm.HIT_COLUMNS)
    index = hm.createindex_command(exe, t, tmp, 4, 1)
    assert index[index.index("--threads") + 1] == "4"


def test_the_engine_environment_puts_its_own_bin_first(tmp_path: Path) -> None:
    exe = tmp_path / "bin" / "mmseqs.exe"
    env = hm.engine_env(exe)
    assert env["PATH"].split(os.pathsep)[0] == str(exe.parent)
    assert env["CYGWIN"] == "nodosfilewarning"


def test_a_command_is_run_and_measured(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('ok')"]
    measured = hm.run_measured(command, dict(os.environ), tmp_path / "log" / "a.log", 0.05)
    assert measured.returncode == 0
    assert measured.seconds > 0
    assert measured.method in {hm.METHOD_JOB_OBJECT, hm.METHOD_WAIT4, hm.METHOD_NONE}
    if measured.cpu_seconds is not None:
        assert measured.cpu_seconds >= 0
    assert measured.as_dict()["method"] == measured.method
    assert "ok" in (tmp_path / "log" / "a.log").read_text(encoding="utf-8")


def test_a_failing_command_raises_with_the_end_of_its_log(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    with pytest.raises(hm.EngineError, match="boom"):
        hm.run_measured(command, dict(os.environ), tmp_path / "b.log", 0.05)


def test_free_bytes_is_a_positive_number(tmp_path: Path) -> None:
    assert hm.free_bytes(tmp_path) > 0


# -- layering and housekeeping ---------------------------------------------------------
def test_the_modules_exist_where_the_layering_test_expects_them() -> None:
    for module in MODULES:
        assert module.is_file()


def test_the_data_directory_is_ignored_by_git() -> None:
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/evidence/" in ignored
