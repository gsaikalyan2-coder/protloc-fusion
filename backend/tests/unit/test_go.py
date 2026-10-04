"""Phase 12 GO head: config, ontology, labels, scope, rows, metrics, floors, manifest, bundle."""

import ast
from collections.abc import Callable
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from protloc.calibration import calibrator as cb
from protloc.registry import bundle as registry_bundle
from protloc.training import baseline_models as bm
from protloc.training import calibration as cal
from protloc.training import go, go_net, head_net
from protloc.training.baselines import (
    SUBSET_LABEL,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    digest,
    lopo_folds,
)
from protloc.training.head import Architecture, Schedule
from protloc.training.head_net import FitSettings, Net

REPO = Path(__file__).resolve().parents[3]
GO_YAML = REPO / "configs" / "training" / "go.yaml"
MODULES = [
    REPO / "backend" / "protloc" / "training" / "go.py",
    REPO / "backend" / "protloc" / "training" / "go_net.py",
]
SHA = "a" * 64
DIM = 6


@pytest.fixture
def np() -> Any:
    """numpy, or skip the test."""
    return pytest.importorskip("numpy")


@pytest.fixture
def torch() -> Any:
    """Torch (and numpy), or skip the test."""
    pytest.importorskip("numpy")
    return pytest.importorskip("torch")


@pytest.fixture
def sklearn() -> Any:
    """scikit-learn (and numpy), or skip the test."""
    pytest.importorskip("numpy")
    return pytest.importorskip("sklearn")


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _config() -> go.GoConfig:
    return go.load_config(GO_YAML)


# -- configuration ---------------------------------------------------------------------
def test_config_agrees_with_the_root_model_and_evaluation_configs() -> None:
    config = _config()
    assert config.seed == _yaml(REPO / "configs" / "config.yaml")["seed"]
    trunk = _yaml(REPO / "configs" / "model" / "trunk_esm2_150m.yaml")["trunk"]
    assert config.architecture.embedding_dim == trunk["embedding_dim"]
    assert (
        config.architecture.embedding_dim
        == _yaml(REPO / "configs/training/head.yaml")["architecture"]["embedding_dim"]
    )
    evaluation = _yaml(REPO / "configs" / "evaluation" / "default.yaml")
    assert evaluation["reporting"]["bootstrap_samples"] == 1000
    assert evaluation["partitions"]["test"] == [TEST_PARTITION]
    assert (REPO / config.calibration_config).is_file()


def test_the_configuration_records_the_gate_decisions() -> None:
    config = _config()
    assert config.term_floor == 20
    assert config.evidence.policy == ("experimental", "curated")
    assert config.ontology.namespace == "molecular_function"
    assert config.ontology.root == "GO:0003674"
    assert config.ontology.relations == ("is_a", "part_of")
    assert config.ontology.license == "CC-BY-4.0"
    assert config.metrics.k_values == (1, 3, 5)
    assert config.metrics.primary_k in config.metrics.k_values
    assert config.bundle_version == "0.12.0"
    assert config.platt_fallback is False


def test_the_grid_is_eight_points_in_the_documented_order() -> None:
    points = go.grid_points(_config())
    assert len(points) == 8
    assert points[0].as_dict() == {
        "learning_rate": 0.001,
        "dropout": 0.3,
        "hidden": 256,
        "gamma": 0.0,
    }
    assert points[1].hidden == 128
    assert points[2].dropout == 0.5
    assert points[4].learning_rate == 0.0003
    assert len({tuple(p.as_dict().values()) for p in points}) == 8


def test_config_values_are_not_repeated_as_literals_in_the_modules() -> None:
    config = _config()
    arch, schedule = config.architecture, config.schedule
    forbidden = {
        0.5,
        config.seed,
        arch.embedding_dim,
        arch.attention_hidden,
        schedule.batch_size,
        schedule.max_epochs,
        schedule.patience,
        schedule.inner_holdout_fraction,
        config.weight_decay,
        config.pos_weight_cap,
        config.term_floor,
        config.metrics.per_term_min_positives,
        config.metrics.max_threshold_candidates,
        config.reproduction_tolerance,
        config.motif_tiebreak_weight,
        config.smoke.stride,
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
    doc = _yaml(GO_YAML)
    edit(doc)
    path = tmp_path / "go.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(task="localization"),
        lambda d: d["optimiser"].update(name="sgd"),
        lambda d: d["grid"].pop("dropout"),
        lambda d: d["grid"].update(dropout=[1.0]),
        lambda d: d["grid"].update(gamma=[-1.0]),
        lambda d: d["grid"].update(hidden=[0]),
        lambda d: d["schedule"].update(inner_holdout_fraction=1.0),
        lambda d: d["labels"].update(evidence_policy=["nonsense"]),
        lambda d: d["labels"]["evidence_classes"].update(curated=["EXP"]),
        lambda d: d["labels"].update(term_floor=0),
        lambda d: d["ontology"].update(sha256="xyz"),
        lambda d: d["ontology"].update(propagate_over=["regulates"]),
        lambda d: d["metrics"].update(k_values=[3, 1]),
        lambda d: d["metrics"].update(primary_k=4),
    ],
)
def test_a_malformed_configuration_is_refused(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    with pytest.raises(BaselineError):
        go.load_config(_edited(tmp_path, edit))


def test_the_label_policy_hash_follows_only_the_ontology_and_label_blocks(tmp_path: Path) -> None:
    base = _config()
    other = go.load_config(_edited(tmp_path, lambda d: d["schedule"].update(patience=11)))
    assert other.labels_sha256 == base.labels_sha256
    assert other.sha256 != base.sha256
    changed = go.load_config(_edited(tmp_path, lambda d: d["labels"].update(term_floor=21)))
    assert changed.labels_sha256 != base.labels_sha256


def test_every_evidence_code_is_in_exactly_one_class() -> None:
    rules = _config().evidence
    assert rules.class_of("EXP") == "experimental"
    assert rules.class_of("TAS") == "curated"
    assert rules.class_of("IEA") == "electronic"
    assert "IEA" not in rules.policy_codes
    with pytest.raises(BaselineError, match="no evidence class"):
        rules.class_of("ZZZ")


# -- a small ontology ------------------------------------------------------------------
OBO = """format-version: 1.2
data-version: releases/test

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0000001
name: binding
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0000002
name: ion binding
namespace: molecular_function
is_a: GO:0000001 ! binding

[Term]
id: GO:0000003
name: metal ion binding
namespace: molecular_function
alt_id: GO:0000099
is_a: GO:0000002 ! ion binding

[Term]
id: GO:0000004
name: catalytic complex binding
namespace: molecular_function
relationship: part_of GO:0000001 ! binding

[Term]
id: GO:0000005
name: obsolete old binding
namespace: molecular_function
is_obsolete: true
replaced_by: GO:0000002

[Term]
id: GO:0000006
name: obsolete nothing
namespace: molecular_function
is_obsolete: true

[Term]
id: GO:0000007
name: obsolete two ways
namespace: molecular_function
is_obsolete: true
replaced_by: GO:0000002
replaced_by: GO:0000003

[Term]
id: GO:0100001
name: a process
namespace: biological_process
is_a: GO:0000001 ! an edge that leaves the aspect

[Typedef]
id: part_of
name: part of
"""


def _pin(**overrides: Any) -> go.OntologyPin:
    values: dict[str, Any] = {
        "file": "x.obo",
        "data_version": "releases/test",
        "sha256": SHA,
        "size": len(OBO.encode()),
        "term_stanzas": 9,
        "obsolete_stanzas": 3,
        "namespace": "molecular_function",
        "root": "GO:0003674",
        "relations": ("is_a", "part_of"),
        "license": "CC-BY-4.0",
        "license_url": "https://example.org",
    }
    values.update(overrides)
    return go.OntologyPin(**values)


def _onto() -> go.Ontology:
    return go.parse_obo(OBO)


def _ctx(**config_overrides: Any) -> go.LabelContext:
    onto = _onto()
    pin = _pin()
    config = replace(_config(), ontology=pin, **config_overrides)
    return go.LabelContext(onto, go.Ancestry(onto, pin), config)


def test_the_obo_parser_reads_terms_alt_ids_and_obsolete_stanzas() -> None:
    onto = _onto()
    assert onto.data_version == "releases/test"
    assert onto.stanzas == 9
    assert onto.obsolete == 3
    assert onto.alt == {"GO:0000099": "GO:0000003"}
    assert onto.terms["GO:0000003"].is_a == ("GO:0000002",)
    assert onto.terms["GO:0000004"].part_of == ("GO:0000001",)
    assert onto.terms["GO:0000007"].replaced_by == ("GO:0000002", "GO:0000003")
    assert go.ontology_problems(onto, _pin()) == []


@pytest.mark.parametrize(
    "pin",
    [
        _pin(data_version="releases/other"),
        _pin(term_stanzas=10),
        _pin(obsolete_stanzas=2),
        _pin(namespace="biological_process"),
        _pin(root="GO:9999999"),
    ],
)
def test_a_release_that_differs_from_the_pin_is_reported(pin: go.OntologyPin) -> None:
    assert go.ontology_problems(_onto(), pin)


def test_a_repeated_term_id_is_refused() -> None:
    with pytest.raises(BaselineError, match="repeated id"):
        go.parse_obo("[Term]\nid: GO:1\nname: a\n\n[Term]\nid: GO:1\nname: b\n")


def test_load_ontology_checks_bytes_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "go.obo"
    path.write_text(OBO, encoding="utf-8", newline="\n")
    good = _pin(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert go.load_ontology(path, good).stanzas == 9
    with pytest.raises(BaselineError, match="not the ontology release"):
        go.load_ontology(path, _pin())
    path.write_text(OBO + "\n! tampered\n", encoding="utf-8", newline="\n")
    with pytest.raises(BaselineError, match="not the ontology release"):
        go.load_ontology(path, good)


@pytest.mark.parametrize(
    ("term", "case", "target"),
    [
        ("GO:0000003", "primary", "GO:0000003"),
        ("GO:0000099", "alt_id", "GO:0000003"),
        ("GO:0000005", "replaced_by", "GO:0000002"),
        ("GO:0000006", "obsolete_dropped", None),
        ("GO:0000007", "obsolete_dropped", None),
        ("GO:7777777", "absent", None),
        ("GO:0100001", "other_aspect", None),
    ],
)
def test_a_corpus_term_id_resolves_without_guessing(
    term: str, case: str, target: str | None
) -> None:
    resolution = go.resolve_term(_onto(), term, "molecular_function")
    assert (resolution.case, resolution.term) == (case, target)


def test_the_true_path_rule_follows_is_a_and_part_of_and_drops_the_root() -> None:
    ancestry = _ctx().ancestry
    assert ancestry.of("GO:0000003") == {"GO:0000002", "GO:0000001", "GO:0003674"}
    assert ancestry.of("GO:0000004") == {"GO:0000001", "GO:0003674"}
    assert ancestry.propagate({"GO:0000003"}) == {"GO:0000003", "GO:0000002", "GO:0000001"}
    assert ancestry.propagate({"GO:0000004", "GO:0000002"}) == {
        "GO:0000004",
        "GO:0000002",
        "GO:0000001",
    }
    assert ancestry.of("GO:0000006") == frozenset()
    assert "GO:0003674" not in ancestry.propagate({"GO:0000001"})


def test_an_edge_that_leaves_the_aspect_is_counted_and_never_followed() -> None:
    onto = _onto()
    # GO:0100001 is_a GO:0000001 stays inside the file, but the process term is not in the aspect.
    assert go.cross_aspect_edges(onto, _pin()) == 0
    crossing = go.parse_obo(
        OBO.replace("is_a: GO:0000001 ! binding", "is_a: GO:0100001 ! a process", 1)
    )
    assert go.cross_aspect_edges(crossing, _pin()) == 1
    assert go.Ancestry(crossing, _pin()).of("GO:0000002") == frozenset()


def test_the_mapping_report_counts_occurrences_and_distinct_ids() -> None:
    report = go.mapping_report(
        {"GO:0000003": 5, "GO:0000099": 2, "GO:0000005": 1, "GO:0000006": 4, "GO:7777777": 3},
        _onto(),
        "molecular_function",
    )
    assert report["occurrences"] == {
        "primary": 5,
        "alt_id": 2,
        "replaced_by": 1,
        "obsolete_dropped": 4,
        "absent": 3,
        "other_aspect": 0,
    }
    assert report["distinct_ids"]["primary"] == 1
    assert report["total_occurrences"] == 15
    assert report["total_distinct_ids"] == 5


# -- labels from the corpus ------------------------------------------------------------
def _record(
    accession: str, values: dict[str, tuple[str, ...]], state: str = "annotated"
) -> go.RawRecord:
    return go.RawRecord(
        accession=accession,
        group="eukaryote",
        sequence="MKTAYIAKQR",
        state=state,
        values=tuple(values),
        evidence=values,
    )


def _label(
    ctx: go.LabelContext,
    accession: str,
    values: dict[str, tuple[str, ...]],
    state: str = "annotated",
    motif: tuple[str, ...] = (),
) -> go.LabelRow:
    return go.map_record(ctx, _record(accession, values, state), motif)


def test_an_experimental_term_is_a_label_and_carries_its_ancestors_without_the_root() -> None:
    row = _label(_ctx(), "P1", {"GO:0000003": ("EXP", "IEA")})
    assert row.propagated == ("GO:0000001", "GO:0000002", "GO:0000003")
    assert row.direct == {"GO:0000003": ("EXP", "IEA")}
    assert row.state == "annotated"


def test_an_electronic_only_term_is_not_a_label_and_the_row_is_not_negative() -> None:
    row = _label(_ctx(), "P1", {"GO:0000003": ("IEA",), "GO:0000002": ("ISS",)})
    assert row.propagated == ()
    assert row.state == "annotated"
    assert set(row.direct) == {"GO:0000003", "GO:0000002"}


def test_a_curated_code_counts_under_the_gate_policy() -> None:
    assert _label(_ctx(), "P1", {"GO:0000002": ("TAS",)}).propagated == (
        "GO:0000001",
        "GO:0000002",
    )


def test_the_root_is_never_a_label() -> None:
    assert _label(_ctx(), "P1", {"GO:0003674": ("EXP",)}).propagated == ()


def test_alt_ids_replacements_and_drops_are_applied_and_counted() -> None:
    row = _label(
        _ctx(),
        "P1",
        {
            "GO:0000099": ("EXP",),
            "GO:0000005": ("IDA",),
            "GO:0000006": ("EXP",),
            "GO:7777777": ("EXP",),
            "GO:0100001": ("EXP",),
        },
    )
    assert set(row.direct) == {"GO:0000003", "GO:0000002"}
    assert row.dropped == {"obsolete_dropped": 1, "absent": 1, "other_aspect": 1}
    assert "GO:0000003" in row.propagated


def test_two_corpus_ids_for_one_term_union_their_evidence() -> None:
    row = _label(_ctx(), "P1", {"GO:0000099": ("IEA",), "GO:0000003": ("EXP",)})
    assert row.direct == {"GO:0000003": ("EXP", "IEA")}


def test_an_unknown_state_row_has_no_label() -> None:
    row = _label(_ctx(), "P1", {}, state="unknown")
    assert (row.state, row.propagated) == ("unknown", ())


@pytest.mark.parametrize("state", ["negative", "not_applicable", "absent"])
def test_go_is_open_world_so_a_negative_state_is_refused(state: str) -> None:
    with pytest.raises(BaselineError, match="open-world"):
        _label(_ctx(), "P1", {}, state=state)


def test_an_evidence_code_in_no_class_is_an_error_not_a_guess() -> None:
    with pytest.raises(BaselineError, match="no evidence class"):
        _label(_ctx(), "P1", {"GO:0000003": ("QQQ",)})


def test_motif_hits_are_mapped_and_propagated_like_labels() -> None:
    row = _label(_ctx(), "P1", {}, state="unknown", motif=("GO:0000003", "GO:7777777"))
    assert row.motif == ("GO:0000003", "GO:7777777")
    assert row.motif_propagated == ("GO:0000001", "GO:0000002", "GO:0000003")


def test_the_label_table_round_trips_and_is_deterministic() -> None:
    ctx = _ctx()
    labels = {
        "P2": _label(ctx, "P2", {"GO:0000003": ("EXP", "IEA"), "GO:0000006": ("EXP",)}),
        "P1": _label(ctx, "P1", {}, state="unknown", motif=("GO:0000002",)),
    }
    text = go.label_table_text(labels)
    assert text.splitlines()[1].startswith("P1\t")
    assert go.parse_label_table(text) == dict(sorted(labels.items()))
    assert go.label_table_text(go.parse_label_table(text)) == text


@pytest.mark.parametrize(
    "text",
    ["", "accession\tstate\n", "\t".join(go.LABEL_COLUMNS) + "\nP1\tannotated\n"],
)
def test_a_malformed_label_table_is_refused(text: str) -> None:
    with pytest.raises(BaselineError):
        go.parse_label_table(text)


def _corpus_line(accession: str, state: str, values: dict[str, list[str]]) -> str:
    record = {
        "accession": accession,
        "organism_group": "eukaryote",
        "sequence_normalized": "MKTAYIAKQR",
        "go_function": {"state": state, "values": list(values), "evidence": values},
    }
    return json.dumps(record, separators=(",", ":"))


def test_the_corpus_scan_tallies_everything_and_keeps_only_the_wanted_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            [
                _corpus_line("A1", "annotated", {"GO:0000003": ["EXP", "IEA"]}),
                _corpus_line("A2", "unknown", {}),
                _corpus_line("A3", "annotated", {"GO:0000006": ["IDA"], "GO:0000003": ["IEA"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scan = go.scan_corpus(path, {"A1", "A3"})
    assert scan.lines == 3
    assert dict(scan.states) == {"annotated": 2, "unknown": 1}
    assert scan.term_occurrences == {"GO:0000003": 2, "GO:0000006": 1}
    assert scan.code_occurrences == {"EXP": 1, "IEA": 2, "IDA": 1}
    assert set(scan.records) == {"A1", "A3"}
    assert scan.records["A1"].evidence == {"GO:0000003": ("EXP", "IEA")}
    with pytest.raises(BaselineError, match="not in the corpus"):
        go.scan_corpus(path, {"A1", "NOPE"})


# -- the subset mask decides the vocabulary --------------------------------------------
def _world(np: Any) -> dict[str, Any]:
    """Twelve subset accessions (partitions 1-5) and eight non-subset ones with GO:0000004."""
    ctx = _ctx(term_floor=5)
    accessions: list[str] = []
    partition: list[int] = []
    labels: dict[str, go.LabelRow] = {}
    for i in range(12):
        accession = f"m{i}"
        accessions.append(accession)
        partition.append(1 + i % 5)
        if i == 11:
            labels[accession] = _label(ctx, accession, {}, state="unknown")
        elif i == 10:
            labels[accession] = _label(ctx, accession, {"GO:0000003": ("IEA",)})
        elif i % 2 == 0:
            labels[accession] = _label(ctx, accession, {"GO:0000003": ("EXP",)})
        else:
            labels[accession] = _label(ctx, accession, {"GO:0000002": ("EXP",)})
    for i in range(8):
        accession = f"x{i}"
        accessions.append(accession)
        partition.append(1 + i % 4)
        labels[accession] = _label(ctx, accession, {"GO:0000004": ("EXP",)})
    count = len(accessions)
    table = {
        "accession": np.array(accessions),
        "partition": np.array(partition),
        "cluster_code": np.arange(count, dtype=np.int64),
        "seq_index": np.arange(count, dtype=np.int64),
    }
    member = np.array([a.startswith("m") for a in accessions])
    return {
        "ctx": ctx,
        "table": table,
        "member": member,
        "order": [f"sha{i}" for i in range(count)],
        "labels": labels,
        "partition_of": dict(zip(accessions, partition, strict=True)),
        "subset": {a for a in accessions if a.startswith("m")},
    }


def test_the_vocabulary_and_every_count_come_from_the_subset_mask(np: Any) -> None:
    w = _world(np)
    everything = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])
    members_only = {a: r for a, r in w["labels"].items() if a in w["subset"]}
    assert everything == go.build_vocabulary(members_only, w["subset"], w["partition_of"], w["ctx"])
    # GO:0000004 has 8 positives in partitions 1-4, but only on accessions outside the subset.
    assert "GO:0000004" not in everything.ids
    assert "GO:0000004" not in everything.undefined
    assert everything.ids == ("GO:0000001", "GO:0000002")
    assert {t.id: t.positives for t in everything.terms} == {"GO:0000001": 8, "GO:0000002": 8}
    assert everything.undefined == {"GO:0000003": 4}
    assert everything.terms[1].ancestors == ("GO:0000001",)
    assert everything.terms[0].by_partition == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}


def test_counting_the_whole_working_set_would_have_given_another_vocabulary(np: Any) -> None:
    w = _world(np)
    whole = set(w["labels"])
    wrong = go.build_vocabulary(w["labels"], whole, w["partition_of"], w["ctx"])
    assert "GO:0000004" in wrong.ids
    assert wrong != go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])


def test_the_floor_counts_partitions_one_to_four_only(np: Any) -> None:
    w = _world(np)
    ctx = _ctx(term_floor=4)
    vocab = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], ctx)
    assert "GO:0000003" in vocab.ids
    only_five = {"m4": w["labels"]["m4"], "m9": w["labels"]["m9"]}
    low = go.build_vocabulary(only_five, {"m4", "m9"}, w["partition_of"], _ctx(term_floor=1))
    assert low.ids == ()


def test_rows_are_the_labelled_subset_rows_of_the_requested_partitions(np: Any) -> None:
    w = _world(np)
    vocab = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])
    spec = (vocab, VALIDATION_PARTITIONS, 1)
    rows = go.build_rows(w["table"], w["order"], w["member"], w["labels"], spec)
    # m4/m9 are partition 5, m10 has no policy term, m11 is unknown, x* are outside the subset.
    assert list(rows.accession) == ["m0", "m1", "m2", "m3", "m5", "m6", "m7", "m8"]
    assert rows.shas == [f"sha{i}" for i in (0, 1, 2, 3, 5, 6, 7, 8)]
    assert rows.y.tolist() == [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    assert set(rows.partition.tolist()) == {1, 2, 3, 4}
    every_third = (vocab, VALIDATION_PARTITIONS, 3)
    thinned = go.build_rows(w["table"], w["order"], w["member"], w["labels"], every_third)
    assert list(thinned.accession) == ["m0", "m3", "m7"]
    test_rows = go.build_rows(w["table"], w["order"], w["member"], w["labels"], (vocab, (5,), 1))
    assert list(test_rows.accession) == ["m4", "m9"]


def test_a_row_outside_the_vocabulary_is_a_closed_world_negative(np: Any) -> None:
    w = _world(np)
    vocab = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])
    narrow = go.Vocabulary(terms=(vocab.terms[1],), undefined={}, floor=vocab.floor)
    spec = (narrow, VALIDATION_PARTITIONS, 1)
    rows = go.build_rows(w["table"], w["order"], w["member"], w["labels"], spec)
    assert rows.y.shape == (8, 1)
    assert int(rows.y.sum()) == 8
    labelled_without_term = {
        a: replace(r, propagated=("GO:0000001",)) for a, r in w["labels"].items()
    }
    again = go.build_rows(w["table"], w["order"], w["member"], labelled_without_term, spec)
    assert int(again.y.sum()) == 0
    assert len(again) == 9  # m10 is now labelled too: it is a closed-world negative


def test_a_subset_row_without_a_label_row_is_an_error(np: Any) -> None:
    w = _world(np)
    vocab = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])
    missing = {a: r for a, r in w["labels"].items() if a != "m0"}
    with pytest.raises(BaselineError, match="no GO label row for m0"):
        go.build_rows(w["table"], w["order"], w["member"], missing, (vocab, (1, 2, 3, 4), 1))


def test_the_vocabulary_file_is_bound_to_the_release_and_the_policy(np: Any) -> None:
    w = _world(np)
    config = w["ctx"].config
    vocab = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])
    document = go.vocabulary_document(vocab, config)
    assert document["ontology"]["sha256"] == config.ontology.sha256
    assert document["evidence_policy"] == ["experimental", "curated"]
    assert document["propagation"] == ["is_a", "part_of"]
    assert [t["id"] for t in document["terms"]] == list(vocab.ids)
    rebuilt = go.vocabulary_from_document(json.loads(json.dumps(document)))
    assert rebuilt.ids == vocab.ids
    assert [t.ancestors for t in rebuilt.terms] == [t.ancestors for t in vocab.terms]
    with pytest.raises(BaselineError, match="no terms"):
        go.vocabulary_from_document({**document, "terms": []})


def test_the_label_summary_counts_the_states_and_the_labelled_rows(np: Any) -> None:
    w = _world(np)
    members = {a: r for a, r in w["labels"].items() if a in w["subset"]}
    summary = go.label_summary(members, w["partition_of"])
    assert summary["accessions"] == 12
    assert summary["states"] == {"annotated": 11, "unknown": 1}
    assert summary["labelled"] == 10
    assert summary["annotated_without_policy_term"] == 1
    assert summary["labelled_by_partition"] == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}


# -- scores, propagation, metrics ------------------------------------------------------
def _vocab() -> go.Vocabulary:
    """A (the parent) with two children B and C."""
    return go.Vocabulary(
        terms=(
            go.VocabTerm("A", "a", 30, {}, ()),
            go.VocabTerm("B", "b", 25, {}, ("A",)),
            go.VocabTerm("C", "c", 22, {}, ("A",)),
        ),
        undefined={},
        floor=20,
    )


def _metric_settings(**overrides: Any) -> go.MetricSettings:
    values: dict[str, Any] = {
        "k_values": (1, 3),
        "primary_k": 1,
        "per_term_min_positives": 20,
        "max_threshold_candidates": 1000,
    }
    values.update(overrides)
    return go.MetricSettings(**values)


def _scoring(samples: int = 0, seed: int = 1, **overrides: Any) -> go.Scoring:
    return go.Scoring(_metric_settings(**overrides), bm.Bootstrap(samples, seed))


def test_a_parent_term_scores_at_least_as_high_as_any_descendant(np: Any) -> None:
    score = np.array([[0.1, 0.9, 0.2], [0.8, 0.3, 0.2]])
    out = go.propagate_scores(score, _vocab())
    assert out.tolist() == [[0.9, 0.9, 0.2], [0.8, 0.3, 0.2]]
    assert (out >= score).all()


def test_the_descendant_mask_marks_self_and_descendants_only(np: Any) -> None:
    mask = go.descendant_mask(_vocab())
    assert mask.tolist() == [[True, True, True], [False, True, False], [False, False, True]]


def test_top_k_hits_cap_k_at_the_vocabulary_and_break_ties_by_column(np: Any) -> None:
    score = np.array([[0.9, 0.8, 0.1]])
    y = np.array([[1, 0, 1]])
    assert go.top_k_hits(score, y, 1).tolist() == [1]
    assert go.top_k_hits(score, y, 2).tolist() == [1]
    assert go.top_k_hits(score, y, 5).tolist() == [2]
    tied = np.array([[0.5, 0.5, 0.5]])
    assert go.top_k_hits(tied, np.array([[0, 1, 0]]), 1).tolist() == [0]
    assert go.top_k_hits(tied, np.array([[1, 0, 0]]), 1).tolist() == [1]


def _small() -> tuple[Any, Any]:
    np = pytest.importorskip("numpy")
    y = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0]])
    score = np.array([[0.9, 0.8, 0.1], [0.9, 0.2, 0.7], [0.5, 0.5, 0.5]])
    return score, y


def test_metrics_match_a_hand_computation_on_the_evaluable_rows(np: Any) -> None:
    score, y = _small()
    report = go.metric_report(score, y, score >= 0.6, np.array([0, 1, 2]), _scoring())
    metrics = {k: v["value"] for k, v in report["metrics"].items()}
    assert report["rows"] == 3
    assert report["evaluable_rows"] == 2
    assert metrics["precision_at_1"] == pytest.approx(1.0)
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["precision_at_3"] == pytest.approx(2 / 3)
    assert metrics["recall_at_3"] == pytest.approx(1.0)
    assert metrics["fmax_precision"] == pytest.approx(1.0)
    assert metrics["fmax_recall"] == pytest.approx(1.0)
    assert metrics["fmax"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert set(metrics) == set(go.metric_names((1, 3)))
    assert report["metrics"]["fmax"]["ci_low"] is None


def test_a_threshold_that_calls_nothing_gives_zero_precision_recall_and_coverage(np: Any) -> None:
    score, y = _small()
    report = go.metric_report(score, y, score >= 2.0, np.array([0, 1, 2]), _scoring())
    values = {k: v["value"] for k, v in report["metrics"].items()}
    assert (values["fmax"], values["coverage"], values["fmax_precision"]) == (0.0, 0.0, 0.0)


def test_a_set_without_an_evaluable_row_is_an_error(np: Any) -> None:
    score, y = _small()
    with pytest.raises(BaselineError, match="no evaluable row"):
        go.metric_report(score[2:], y[2:], score[2:] >= 0.5, np.array([0]), _scoring())


def test_the_f_measure_uses_precision_over_called_rows_and_recall_over_all(np: Any) -> None:
    y = np.array([[1, 0], [1, 0]])
    score = np.array([[0.9, 0.1], [0.2, 0.1]])
    # tau 0.5: row 0 calls {A} (tp 1 of 1); row 1 calls nothing. P = 1, R = (1 + 0) / 2.
    assert go.f_measure(score, y, 0.5) == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert go.f_measure(score[:0], y[:0], 0.5) == 0.0


def test_the_best_threshold_maximises_f_over_evaluable_rows_only(np: Any) -> None:
    score, y = _small()
    assert go.best_f_threshold(score, y, 1000) == pytest.approx(0.7)
    with pytest.raises(BaselineError, match="evaluable"):
        go.best_f_threshold(score[2:], y[2:], 1000)


def test_threshold_candidates_are_capped_but_keep_the_extremes(np: Any) -> None:
    values = np.linspace(0.0, 1.0, 5000)
    cut = go.threshold_candidates(values, 100)
    assert len(cut) <= 100
    assert float(cut.min()) == 0.0
    assert float(cut.max()) == 1.0
    assert len(go.threshold_candidates(values[:50], 100)) == 50


def _random_scored(np: Any, n: int = 160) -> tuple[Any, Any, Any, Any]:
    rng = np.random.default_rng(5)
    y = (rng.random((n, 3)) < [0.9, 0.4, 0.3]).astype(int)
    y[:, 1:] *= y[:, :1]
    score = np.clip(0.55 * y + rng.random((n, 3)) * 0.7, 0.0, 1.0)
    partition = 1 + np.arange(n) % 4
    cluster = np.arange(n) // 2
    return go.propagate_scores(score, _vocab()), y, partition, cluster


def test_a_folds_threshold_never_saw_that_folds_rows(np: Any) -> None:
    score, y, partition, _ = _random_scored(np)
    base, frozen = go.nested_thresholds(score, y, partition, 1000)
    held = partition == 1
    rng = np.random.default_rng(9)
    changed_score = score.copy()
    changed_y = y.copy()
    changed_score[held] = rng.random((int(held.sum()), 3))
    changed_y[held] = (rng.random((int(held.sum()), 3)) < 0.5).astype(int)
    again, frozen_again = go.nested_thresholds(changed_score, changed_y, partition, 1000)
    assert again[1] == base[1]
    assert any(again[p] != base[p] for p in (2, 3, 4))
    assert frozen_again != frozen


def test_decisions_use_each_rows_own_fold_threshold(np: Any) -> None:
    score = np.array([[0.4, 0.4], [0.4, 0.4]])
    decision = go.decide(score, np.array([1, 2]), {1: 0.3, 2: 0.5})
    assert decision.tolist() == [[True, True], [False, False]]


def test_auc_and_average_precision_match_known_values(np: Any) -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    assert go.auc_of(score, y) == pytest.approx(0.75)
    assert go.aupr_of(score, y) == pytest.approx(5 / 6)
    flat = np.array([0.3, 0.3, 0.3, 0.3])
    assert go.auc_of(flat, y) == pytest.approx(0.5)
    assert go.aupr_of(flat, y) == pytest.approx(0.5)
    with pytest.raises(BaselineError):
        go.auc_of(score, np.zeros(4, dtype=int))


def test_per_term_scores_are_only_for_terms_with_enough_positives_and_negatives(np: Any) -> None:
    rng = np.random.default_rng(2)
    n = 120
    y = np.zeros((n, 3), dtype=int)
    y[:, 0] = 1
    y[:30, 1] = 1
    y[:5, 2] = 1
    score = np.clip(0.4 * y + rng.random((n, 3)) * 0.6, 0.0, 1.0)
    report = go.per_term_report(score, y, np.arange(n) // 3, _vocab(), _scoring(samples=20))
    assert report["A"]["scored"] is False
    assert (report["A"]["positives"], report["A"]["negatives"]) == (120, 0)
    assert report["C"]["scored"] is False
    assert report["B"]["scored"] is True
    assert report["B"]["auc"]["value"] > 0.5
    assert report["B"]["auc"]["ci_low"] is not None
    assert report["B"]["positive_rate"] == pytest.approx(0.25)
    assert "auc" not in report["C"]


def test_the_cluster_bootstrap_is_seeded_and_resamples_whole_clusters(np: Any) -> None:
    score, y, _partition, cluster = _random_scored(np)
    decision = score >= 0.6
    first = go.metric_report(score, y, decision, cluster, _scoring(samples=60, seed=11))
    again = go.metric_report(score, y, decision, cluster, _scoring(samples=60, seed=11))
    other = go.metric_report(score, y, decision, cluster, _scoring(samples=60, seed=12))
    assert first == again
    assert first != other
    one = go.metric_report(score, y, decision, np.zeros(len(y), dtype=int), _scoring(samples=30))
    for entry in one["metrics"].values():
        assert entry["ci_low"] == pytest.approx(entry["value"])
        assert entry["ci_high"] == pytest.approx(entry["value"])
    for entry in first["metrics"].values():
        assert entry["ci_low"] <= entry["value"] + 1e-9
        assert entry["value"] <= entry["ci_high"] + 1e-9
    assert first["bootstrap"] == {"samples": 60, "seed": 11, "unit": "cluster_id"}


def test_selection_prefers_precision_then_fmax_then_the_earlier_point() -> None:
    assert go.select_point([(0.5, 0.1), (0.6, 0.0), (0.4, 0.9)]) == 1
    assert go.select_point([(0.5, 0.1), (0.5, 0.3), (0.4, 0.9)]) == 1
    assert go.select_point([(0.5, 0.3), (0.5, 0.3)]) == 0


def test_fold_means_skip_a_fold_with_no_evaluable_row_and_use_nested_thresholds(np: Any) -> None:
    score, y, partition, _ = _random_scored(np)
    precision, fmax = go.fold_means(score, y, partition, _metric_settings())
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= fmax <= 1.0
    blank = y.copy()
    blank[partition == 4] = 0
    skipped, _ = go.fold_means(score, blank, partition, _metric_settings())
    assert math.isfinite(skipped)
    with pytest.raises(BaselineError, match="evaluable"):
        go.fold_means(score, np.zeros_like(y), partition, _metric_settings())


def test_fold_summaries_list_every_fold_and_mark_a_blank_one(np: Any) -> None:
    score, y, partition, _ = _random_scored(np)
    blank = y.copy()
    blank[partition == 3] = 0
    summaries = go.fold_summaries(score, blank, score >= 0.6, partition, (1, 3))
    assert [s["validate"] for s in summaries] == [1, 2, 3, 4]
    assert summaries[2]["metrics"] is None
    assert summaries[0]["metrics"]["precision_at_1"] >= 0.0


# -- the floors ------------------------------------------------------------------------
def test_the_frequency_prior_of_a_fold_comes_from_its_training_partitions_only(np: Any) -> None:
    _, y, partition, _ = _random_scored(np)
    folds = lopo_folds()
    prior = go.fold_prior_scores(y, partition, folds)
    expected = y[partition != 2].mean(axis=0)
    assert np.allclose(prior[partition == 2], expected)
    flipped = y.copy()
    flipped[partition == 2] = 1 - flipped[partition == 2]
    again = go.fold_prior_scores(flipped, partition, folds)
    assert np.array_equal(again[partition == 2], prior[partition == 2])
    assert not np.array_equal(again[partition == 1], prior[partition == 1])


def _motif_rows(np: Any) -> tuple[go.Rows, dict[str, go.LabelRow]]:
    ctx = _ctx()
    labels = {
        "a": _label(ctx, "a", {}, state="unknown", motif=("GO:0000003",)),
        "b": _label(ctx, "b", {}, state="unknown", motif=()),
    }
    rows = go.Rows(
        accession=["a", "b"],
        shas=["s0", "s1"],
        y=np.zeros((2, 3), dtype=int),
        partition=np.array([1, 2]),
        cluster_code=np.array([0, 1]),
        seq_index=np.array([0, 1]),
    )
    return rows, labels


def test_motif_hits_name_the_hit_term_and_its_ancestors_in_the_vocabulary(np: Any) -> None:
    rows, labels = _motif_rows(np)
    vocab = go.Vocabulary(
        terms=(
            go.VocabTerm("GO:0000001", "binding", 30, {}, ()),
            go.VocabTerm("GO:0000002", "ion binding", 30, {}, ("GO:0000001",)),
            go.VocabTerm("GO:0000004", "complex", 30, {}, ("GO:0000001",)),
        ),
        undefined={},
        floor=20,
    )
    hits = go.motif_matrix(rows, labels, vocab)
    assert hits.tolist() == [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    prior = np.array([[0.2, 0.9, 0.1], [0.2, 0.9, 0.1]])
    score = go.motif_scores(hits, prior, _config().motif_tiebreak_weight)
    assert score[0].argmax() == 1
    assert score[1].argmax() == 1
    assert (score[0] >= score[1]).all()
    assert float(score[0, 2]) < float(score[0, 0])


def test_logreg_scores_for_a_fold_do_not_depend_on_that_folds_labels(np: Any, sklearn: Any) -> None:
    _score, y, partition, _ = _random_scored(np, 120)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(len(y), 5)) + y[:, [1]]
    spec = (lopo_folds(), 1.0, bm.Runtime(3, 1))
    base = go.logreg_scores(x, y, partition, spec)
    flipped = y.copy()
    flipped[partition == 1] = 1 - flipped[partition == 1]
    again = go.logreg_scores(x, flipped, partition, spec)
    assert base.shape == y.shape
    assert ((base > 0) & (base < 1)).all()
    assert np.array_equal(base[partition == 1], again[partition == 1])
    assert not np.array_equal(base[partition == 2], again[partition == 2])


# -- calibration -----------------------------------------------------------------------
def _calibration_config(platt: bool) -> Any:
    base = cal.load_config(REPO / "configs" / "training" / "calibration.yaml")
    return go.calibration_settings(base, replace(_config(), platt_fallback=platt))


def test_the_platt_fallback_is_switched_off_unless_the_config_keeps_it() -> None:
    off, on = _calibration_config(False), _calibration_config(True)
    assert cal.choose_method(10, 10, off.eligibility) == cb.UNCALIBRATED
    assert cal.choose_method(10, 10, on.eligibility) == cb.PLATT
    assert cal.choose_method(30, 30, off.eligibility) == cb.ISOTONIC
    assert off.eligibility.isotonic_min_positives == on.eligibility.isotonic_min_positives


def _calibration_data(np: Any, n: int = 240) -> go.CalibData:
    rng = np.random.default_rng(8)
    y = (rng.random((n, 3)) < [0.7, 0.6, 0.5]).astype(int)
    y[:, 1:] *= y[:, :1]
    raw = np.clip(0.5 * y + rng.random((n, 3)) * 0.6, 0.0, 1.0)
    partition = 1 + np.arange(n) % 4
    return go.CalibData(raw, y, partition, np.arange(n, dtype=np.int64), _vocab())


def test_crossfit_never_lets_a_folds_labels_move_that_folds_scores_or_threshold(
    np: Any, sklearn: Any
) -> None:
    data = _calibration_data(np)
    cfg = _calibration_config(False)
    fit = go.crossfit(data, lopo_folds(), cfg, 1000)
    held = data.partition == 1
    flipped_y = data.y.copy()
    flipped_y[held] = 1 - flipped_y[held]
    again = go.crossfit(replace(data, y=flipped_y), lopo_folds(), cfg, 1000)
    assert np.array_equal(fit.calibrated[held], again.calibrated[held])
    assert fit.fold_thresholds[1] == again.fold_thresholds[1]
    assert not np.array_equal(fit.calibrated[~held], again.calibrated[~held])


def test_crossfit_freezes_one_global_threshold_into_every_term_calibrator(
    np: Any, sklearn: Any
) -> None:
    data = _calibration_data(np)
    fit = go.crossfit(data, lopo_folds(), _calibration_config(False), 1000)
    assert [c.label for c in fit.frozen] == ["A", "B", "C"]
    assert {c.threshold for c in fit.frozen} == {fit.frozen_threshold}
    assert all(c.method == cb.ISOTONIC for c in fit.frozen)
    assert all(fit.methods["consistent"].values())
    assert fit.propagated.shape == data.raw.shape
    for c in fit.frozen:
        assert c.problems() == []


def test_a_term_below_the_eligibility_minimum_keeps_its_raw_score(np: Any, sklearn: Any) -> None:
    data = _calibration_data(np)
    rare = data.y.copy()
    rare[:, 2] = 0
    rare[:6, 2] = 1
    fit = go.crossfit(replace(data, y=rare), lopo_folds(), _calibration_config(False), 1000)
    assert fit.frozen[2].method == cb.UNCALIBRATED
    assert np.array_equal(fit.calibrated[:, 2], data.raw[:, 2])


def test_crossfit_refuses_a_cluster_that_spans_two_partitions(np: Any, sklearn: Any) -> None:
    data = _calibration_data(np)
    shared = data.cluster_code.copy()
    shared[1] = shared[0]
    with pytest.raises(BaselineError, match="cluster leakage"):
        go.crossfit(replace(data, cluster_code=shared), lopo_folds(), _calibration_config(False), 9)


def test_the_ece_report_covers_every_term_before_and_after(np: Any, sklearn: Any) -> None:
    data = _calibration_data(np)
    cfg = _calibration_config(False)
    fit = go.crossfit(data, lopo_folds(), cfg, 1000)
    report = go.ece_report(data, fit, cfg, bm.Bootstrap(8, 3))
    assert set(report["per_label"]) == {"A", "B", "C"}
    assert {"ece_raw", "ece_calibrated", "ece_delta"} <= set(report["macro"])
    assert report["bootstrap"]["unit"] == "cluster_id"


# -- weights and losses without torch --------------------------------------------------
def test_the_positive_weight_is_negatives_over_positives_capped_and_one_without_positives() -> None:
    y = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]]
    weights = go.capped_pos_weights(y, 10.0)
    assert weights == [pytest.approx(1.0), 1.0, pytest.approx(3.0)]
    assert go.capped_pos_weights([[1, 0]] + [[0, 0]] * 99, 10.0) == [10.0, 1.0]
    with pytest.raises(BaselineError, match="at least one"):
        go.capped_pos_weights([], 10.0)


def test_the_early_stopping_signal_is_the_mean_binary_cross_entropy(np: Any) -> None:
    y = np.array([[1, 0], [0, 1]])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    half = np.full((2, 2), 0.5)
    assert go.holdout_nll(perfect, y) == pytest.approx(0.0, abs=1e-9)
    assert go.holdout_nll(half, y) == pytest.approx(math.log(2.0))


def test_fold_rows_split_by_partition(np: Any) -> None:
    rows = go.Rows(
        accession=list("abcd"),
        shas=list("abcd"),
        y=np.zeros((4, 1), dtype=int),
        partition=np.array([1, 2, 3, 2]),
        cluster_code=np.arange(4),
        seq_index=np.arange(4),
    )
    train, held = go.fold_rows(rows, lopo_folds()[1])
    assert held == [1, 3]
    assert train == [0, 2]


# -- the manifest ----------------------------------------------------------------------
KS = (1, 3, 5)
OTHER = "b" * 64


def _metrics_block() -> dict[str, Any]:
    return {name: {"value": 0.5, "ci_low": 0.4, "ci_high": 0.6} for name in go.metric_names(KS)}


def _arm() -> dict[str, Any]:
    return {
        "source": "model",
        "pooled": {"rows": 10, "evaluable_rows": 9, "metrics": _metrics_block()},
        "folds": [],
        "per_term": {},
    }


def _pins(vocab: go.Vocabulary) -> go.Pins:
    return go.Pins(
        registry_sha256="1" * 64,
        corpus_sha256="2" * 64,
        ontology_sha256="3" * 64,
        embedding_cache_key="cache",
        residue_cache_key="residues",
        config_sha256="4" * 64,
        labels_sha256="5" * 64,
        calibration_config_sha256="6" * 64,
        vocabulary_sha256="7" * 64,
        vocabulary_ids=vocab.ids,
        term_floor=vocab.floor,
        closed_world="closed-world rule",
        k_values=KS,
        reproduction_tolerance=1e-9,
        torch="2.4.0",
    )


def _frozen(pins: go.Pins, vocab: go.Vocabulary) -> dict[str, Any]:
    return {
        "scope": "subset",
        "scope_label": SUBSET_LABEL,
        "subset": {
            "embedding_cache_key": pins.embedding_cache_key,
            "sequences_sha256": "8" * 64,
            "sequences": 500,
        },
        "protocol": {
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
            "k_values": list(KS),
        },
        "inputs": {
            "registry_sha256": pins.registry_sha256,
            "corpus_sha256": pins.corpus_sha256,
            "ontology_sha256": pins.ontology_sha256,
            "residue_cache_key": pins.residue_cache_key,
            "go_config_sha256": pins.config_sha256,
            "label_policy_sha256": pins.labels_sha256,
            "calibration_config_sha256": pins.calibration_config_sha256,
            "vocabulary_sha256": pins.vocabulary_sha256,
            "feature_table_sha256": "9" * 64,
            "label_table_sha256": "9" * 64,
            "weights_sha256": "9" * 64,
            "training_record_sha256": "9" * 64,
            "calibrator_sha256": "9" * 64,
        },
        "label_policy": {"states": ["annotated", "unknown"], "closed_world": pins.closed_world},
        "vocabulary": go.vocabulary_block(vocab),
        "selected": {"trainable_parameters": 1234, "final_epochs": 7},
        "thresholds": dict.fromkeys(go.ARMS, 0.5),
        "software": {"torch": pins.torch},
        "validation": {
            **{arm: _arm() for arm in go.ARMS},
            "reproduction": {"max_abs_difference": 0.0, "decisions_identical": True},
            "calibration": {"macro": {}},
        },
    }


def _manifest(pins: go.Pins, vocab: go.Vocabulary) -> dict[str, Any]:
    frozen = _frozen(pins, vocab)
    return {
        "schema_version": go.SCHEMA_VERSION,
        "phase": go.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }


def _resign(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    if manifest["test"] is not None:
        manifest["test"]["frozen_sha256"] = manifest["frozen_sha256"]
    return manifest


def _subset_vocabulary(np: Any) -> go.Vocabulary:
    w = _world(np)
    return go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])


def test_freeze_logic_accepts_a_manifest_built_from_subset_counts(np: Any) -> None:
    vocab = _subset_vocabulary(np)
    assert vocab.undefined  # a below-floor term exists, and must be allowed to be undefined
    assert go.check_go_manifest(_manifest(_pins(vocab), vocab), _pins(vocab)) == []


def test_counts_from_the_whole_working_set_are_refused_against_the_subset_vocabulary(
    np: Any,
) -> None:
    w = _world(np)
    subset = go.build_vocabulary(w["labels"], w["subset"], w["partition_of"], w["ctx"])
    whole = go.build_vocabulary(w["labels"], set(w["labels"]), w["partition_of"], w["ctx"])
    assert whole.ids != subset.ids
    problems = go.check_go_manifest(_manifest(_pins(subset), whole), _pins(subset))
    assert any("vocabulary.ids differs" in p for p in problems)


def test_a_term_below_the_floor_cannot_be_recorded_as_trained(np: Any) -> None:
    vocab = _subset_vocabulary(np)
    pins = _pins(vocab)
    manifest = _manifest(pins, vocab)
    manifest["frozen"]["vocabulary"]["positives_validation"]["GO:0000001"] = vocab.floor - 1
    problems = go.check_go_manifest(_resign(manifest), pins)
    assert any("fewer than the floor" in p for p in problems)


MUTATIONS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("scope", lambda f: f.update(scope="full")),
    ("scope label", lambda f: f.update(scope_label="whole corpus")),
    ("subset cache", lambda f: f["subset"].update(embedding_cache_key="other")),
    ("ontology hash", lambda f: f["inputs"].update(ontology_sha256=OTHER)),
    ("config hash", lambda f: f["inputs"].update(go_config_sha256=OTHER)),
    ("label policy hash", lambda f: f["inputs"].update(label_policy_sha256=OTHER)),
    ("vocabulary hash", lambda f: f["inputs"].update(vocabulary_sha256=OTHER)),
    ("weights hash", lambda f: f["inputs"].update(weights_sha256="nope")),
    ("counts scope", lambda f: f["vocabulary"].update(counts_scope="whole_corpus")),
    ("floor", lambda f: f["vocabulary"].update(term_floor=1)),
    ("undefined clears the floor", lambda f: f["vocabulary"]["undefined"].update({"GO:9": 99})),
    ("closed world", lambda f: f["label_policy"].update(closed_world="other")),
    ("states", lambda f: f["label_policy"].update(states=["annotated", "negative"])),
    ("partitions", lambda f: f["protocol"].update(validation_partitions=[1, 2, 3])),
    ("test partition", lambda f: f["protocol"].update(test_partition=4)),
    ("k values", lambda f: f["protocol"].update(k_values=[1])),
    ("threshold", lambda f: f["thresholds"].update(logreg="high")),
    ("missing arm", lambda f: f["validation"].pop("motif")),
    ("missing metric", lambda f: f["validation"]["logreg"]["pooled"]["metrics"].pop("fmax")),
    ("missing per-term", lambda f: f["validation"]["motif"].pop("per_term")),
    ("reproduction", lambda f: f["validation"]["reproduction"].update(max_abs_difference=1.0)),
    ("decisions", lambda f: f["validation"]["reproduction"].update(decisions_identical=False)),
    ("calibration", lambda f: f["validation"].pop("calibration")),
    ("torch", lambda f: f["software"].update(torch="1.0.0")),
    ("parameters", lambda f: f["selected"].update(trainable_parameters="many")),
    ("epochs", lambda f: f["selected"].update(final_epochs=0)),
]


@pytest.mark.parametrize(("name", "edit"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_every_way_the_manifest_can_disagree_is_reported(
    np: Any, name: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    vocab = _subset_vocabulary(np)
    pins = _pins(vocab)
    manifest = _manifest(pins, vocab)
    edit(manifest["frozen"])
    assert go.check_go_manifest(_resign(manifest), pins), name


def test_an_edited_frozen_block_no_longer_matches_its_hash(np: Any) -> None:
    vocab = _subset_vocabulary(np)
    pins = _pins(vocab)
    manifest = _manifest(pins, vocab)
    manifest["frozen"]["selected"]["final_epochs"] = 8
    assert any("frozen_sha256" in p for p in go.check_go_manifest(manifest, pins))


def test_the_one_shot_test_block_must_be_touched_once_against_this_frozen_block(np: Any) -> None:
    vocab = _subset_vocabulary(np)
    pins = _pins(vocab)
    manifest = _manifest(pins, vocab)
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "metrics": {arm: _arm() for arm in go.ARMS},
    }
    assert go.check_go_manifest(manifest, pins) == []
    for edit in (
        lambda t: t.update(touches=2),
        lambda t: t.update(frozen_sha256=OTHER),
        lambda t: t["metrics"].pop("logreg"),
        lambda t: t["metrics"]["motif"]["pooled"]["metrics"].pop("coverage"),
    ):
        broken = copy.deepcopy(manifest)
        edit(broken["test"])
        assert go.check_go_manifest(broken, pins)


# -- the registry bundle ---------------------------------------------------------------
ONTOLOGY_SHA = "c" * 64
CLASSES = ["GO:0000001", "GO:0000002"]


def _terms(ontology: str = ONTOLOGY_SHA, ids: list[str] | None = None) -> bytes:
    document = {
        "schema_version": 1,
        "task": "go_function",
        "ontology": {"data_version": "releases/test", "sha256": ontology},
        "terms": [{"id": t, "name": t} for t in (ids or CLASSES)],
    }
    return cb.canonical_json_bytes(document)


def _go_head(terms: bytes, **overrides: Any) -> dict[str, Any]:
    head: dict[str, Any] = {
        "weights_file": "final_weights.npz",
        "weights_sha256": SHA,
        "frozen_sha256": "7" * 64,
        "head_config_sha256": "6" * 64,
        "classes": list(CLASSES),
        "vocabulary_file": "terms.json",
        "vocabulary_sha256": cb.sha256_hex(terms),
        "ontology_sha256": ONTOLOGY_SHA,
        "ontology_data_version": "releases/test",
        "evidence_policy": ["experimental", "curated"],
        "term_floor": 20,
        "propagation": ["is_a", "part_of"],
        "threshold": 0.4,
        "threshold_rule": "rule",
        "source": "model",
    }
    head.update(overrides)
    return head


def _write_go_bundle(
    tmp_path: Path, head: dict[str, Any] | None = None, terms: bytes | None = None
) -> tuple[Path, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rules = {"baseline_class_map": tmp_path / "map.yaml", "consistency": tmp_path / "c.json"}
    rules["baseline_class_map"].write_text("classes: [a]\n", encoding="utf-8")
    rules["consistency"].write_text("{}\n", encoding="utf-8")
    shipped_terms = terms if terms is not None else _terms()
    shipped = cb.CalibratorSet(
        task="go_function",
        calibration_version="0.12.0+test",
        classes=tuple(CLASSES),
        head_weights_sha256=SHA,
        head_frozen_sha256="7" * 64,
        threshold_rule="rule",
        labels=tuple(cb.LabelCalibrator(t, cb.UNCALIBRATED, 30, 200, 0.4) for t in CLASSES),
    )
    document = registry_bundle.build_document(
        version="0.12.0",
        head=head if head is not None else _go_head(shipped_terms),
        calibrator=shipped,
        rules={f"{key}_sha256": cb.sha256_hex(path.read_bytes()) for key, path in rules.items()},
        scope={"scope": "subset", "scope_label": "label"},
    )
    directory = tmp_path / "registry" / "go_function" / "0.12.0"
    directory.mkdir(parents=True)
    (directory / registry_bundle.BUNDLE_FILE).write_bytes(cb.canonical_json_bytes(document))
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(shipped.to_bytes())
    (directory / "terms.json").write_bytes(shipped_terms)
    return directory, rules


def test_a_go_bundle_loads_and_is_bound_to_its_vocabulary_and_ontology(tmp_path: Path) -> None:
    directory, rules = _write_go_bundle(tmp_path)
    loaded = registry_bundle.load_bundle(directory, rules)
    assert loaded.document["task"] == "go_function"
    assert loaded.calibrators.classes == tuple(CLASSES)
    again = registry_bundle.load_bundle(directory, rules, ontology=ONTOLOGY_SHA)
    assert again.document["bundle_sha256"] == loaded.document["bundle_sha256"]


@pytest.mark.parametrize("key", registry_bundle.TASK_HEAD_KEYS["go_function"])
def test_a_go_bundle_without_a_head_definition_is_refused(tmp_path: Path, key: str) -> None:
    head = _go_head(_terms())
    del head[key]
    directory, rules = _write_go_bundle(tmp_path, head=head)
    with pytest.raises(registry_bundle.BundleError, match=f"head.{key}"):
        registry_bundle.load_bundle(directory, rules)


def test_a_changed_vocabulary_file_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_go_bundle(tmp_path)
    (directory / "terms.json").write_bytes(_terms() + b" ")
    with pytest.raises(registry_bundle.BundleError, match="vocabulary file does not match"):
        registry_bundle.load_bundle(directory, rules)


def test_a_missing_or_escaping_vocabulary_file_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_go_bundle(tmp_path)
    (directory / "terms.json").unlink()
    with pytest.raises(registry_bundle.BundleError, match="missing from the bundle directory"):
        registry_bundle.load_bundle(directory, rules)
    escaping = _go_head(_terms(), vocabulary_file="../terms.json")
    other, rules = _write_go_bundle(tmp_path / "other", head=escaping)
    with pytest.raises(registry_bundle.BundleError, match="missing from the bundle directory"):
        registry_bundle.load_bundle(other, rules)


def test_a_bundle_on_another_ontology_release_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_go_bundle(tmp_path)
    with pytest.raises(registry_bundle.BundleError, match="another ontology release"):
        registry_bundle.load_bundle(directory, rules, ontology=OTHER)


def test_a_vocabulary_that_names_another_ontology_than_the_head_is_refused(
    tmp_path: Path,
) -> None:
    terms = _terms(ontology=OTHER)
    directory, rules = _write_go_bundle(tmp_path, terms=terms)
    with pytest.raises(registry_bundle.BundleError, match="names another ontology release"):
        registry_bundle.load_bundle(directory, rules)


def test_a_vocabulary_in_another_term_order_than_the_head_is_refused(tmp_path: Path) -> None:
    terms = _terms(ids=list(reversed(CLASSES)))
    directory, rules = _write_go_bundle(tmp_path, terms=terms)
    with pytest.raises(registry_bundle.BundleError, match="disagree on the term order"):
        registry_bundle.load_bundle(directory, rules)


@pytest.mark.parametrize("name", [registry_bundle.CALIBRATOR_FILE, registry_bundle.BUNDLE_FILE])
def test_a_tampered_go_bundle_file_is_refused(tmp_path: Path, name: str) -> None:
    directory, rules = _write_go_bundle(tmp_path)
    path = directory / name
    document = json.loads(path.read_bytes())
    if name == registry_bundle.BUNDLE_FILE:
        document["head"]["threshold"] = 0.01
    else:
        document["labels"][0]["threshold"] = 0.01
    path.write_bytes(cb.canonical_json_bytes(document))
    with pytest.raises(registry_bundle.BundleError):
        registry_bundle.load_bundle(directory, rules)


def test_other_tasks_are_not_asked_for_a_vocabulary(tmp_path: Path) -> None:
    directory, rules = _write_sp_free_bundle(tmp_path)
    assert registry_bundle.load_bundle(directory, rules, ontology=OTHER).version == "0.10.0"


def _write_sp_free_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    rules = {"baseline_class_map": tmp_path / "map.yaml", "consistency": tmp_path / "c.json"}
    rules["baseline_class_map"].write_text("classes: [a]\n", encoding="utf-8")
    rules["consistency"].write_text("{}\n", encoding="utf-8")
    shipped = cb.CalibratorSet(
        task="localization",
        calibration_version="0.10.0+test",
        classes=("nucleus",),
        head_weights_sha256=SHA,
        head_frozen_sha256="7" * 64,
        threshold_rule="rule",
        labels=(cb.LabelCalibrator("nucleus", cb.UNCALIBRATED, 30, 200, 0.5),),
    )
    head = {
        "weights_sha256": SHA,
        "frozen_sha256": "7" * 64,
        "head_config_sha256": "6" * 64,
        "classes": ["nucleus"],
    }
    document = registry_bundle.build_document(
        version="0.10.0",
        head=head,
        calibrator=shipped,
        rules={f"{key}_sha256": cb.sha256_hex(path.read_bytes()) for key, path in rules.items()},
        scope={"scope": "subset", "scope_label": "label"},
    )
    directory = tmp_path / "registry" / "localization" / "0.10.0"
    directory.mkdir(parents=True)
    (directory / registry_bundle.BUNDLE_FILE).write_bytes(cb.canonical_json_bytes(document))
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(shipped.to_bytes())
    return directory, rules


# -- the network and the training loop -------------------------------------------------
class _Reader:
    """Deterministic per-residue states keyed by the sequence hash."""

    def __init__(self, np: Any) -> None:
        self._np = np

    def states(self, sha256: str) -> Any:
        number = int(sha256[3:])
        rng = self._np.random.default_rng(number)
        return rng.normal(size=(4 + number % 5, DIM)).astype(self._np.float32)


def _plan(epochs: int = 3) -> go_net.Plan:
    return go_net.Plan(
        settings=FitSettings(
            net=Net(Architecture(heads=1, attention_hidden=8, embedding_dim=DIM), 8, 0.0, 3),
            gamma=0.0,
            learning_rate=0.01,
            weight_decay=0.0,
            schedule=Schedule(
                batch_size=4, max_epochs=epochs, patience=2, inner_holdout_fraction=0.3
            ),
            seed=3,
        ),
        pos_weight_cap=10.0,
    )


def _toy(np: Any, n: int = 24) -> tuple[_Reader, go.Rows]:
    rng = np.random.default_rng(1)
    y = (rng.random((n, 3)) < 0.5).astype(int)
    y[0] = 1
    rows = go.Rows(
        accession=[f"a{i}" for i in range(n)],
        shas=[f"sha{i}" for i in range(n)],
        y=y,
        partition=1 + np.arange(n) % 4,
        cluster_code=np.arange(n, dtype=np.int64),
        seq_index=np.arange(n, dtype=np.int64),
    )
    return _Reader(np), rows


def _fields(rows: go.Rows) -> dict[str, Any]:
    return {name: getattr(rows, name) for name in rows.__dataclass_fields__}


def test_a_plan_takes_everything_from_the_config() -> None:
    config = _config()
    point = go.grid_points(config)[3]
    plan = go_net.plan_for(point, config, 12)
    assert plan.settings.net.hidden == point.hidden
    assert plan.settings.net.dropout == point.dropout
    assert plan.settings.net.classes == 12
    assert plan.settings.learning_rate == point.learning_rate
    assert plan.settings.schedule == config.schedule
    assert plan.pos_weight_cap == config.pos_weight_cap


def test_a_fit_needs_exactly_one_of_a_holdout_or_an_epoch_count(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 12)
    source = go_net.Source(reader, rows)
    with pytest.raises(BaselineError, match="either"):
        go_net.fit_head(source, [0, 1], None, _plan())
    with pytest.raises(BaselineError, match="either"):
        go_net.fit_head(source, [0, 1], [2], _plan(), epochs=2)


def test_a_fixed_epoch_fit_runs_exactly_that_many_epochs(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 12)
    result = go_net.fit_head(go_net.Source(reader, rows), list(range(8)), None, _plan(), epochs=2)
    assert (result.epochs_run, result.best_epoch, result.hold_nll) == (2, 2, None)


def test_early_stopping_keeps_the_best_holdout_epoch(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 24)
    result = go_net.fit_head(
        go_net.Source(reader, rows), list(range(16)), list(range(16, 24)), _plan(6)
    )
    assert 1 <= result.best_epoch <= result.epochs_run <= 6
    assert result.hold_nll is not None
    assert math.isfinite(result.hold_nll)


def test_a_cluster_shared_across_partitions_stops_the_fit_before_it_starts(
    np: Any, torch: Any
) -> None:
    reader, rows = _toy(np)
    leaky = go.Rows(**{**_fields(rows), "cluster_code": np.zeros(len(rows), dtype=np.int64)})
    with pytest.raises(BaselineError, match=r"whole cluster|at least two clusters|leakage"):
        go_net.run_grid_point(
            go_net.Source(reader, leaky), lopo_folds(), _plan(), set(), lambda _m: None
        )


def test_a_test_cluster_in_the_training_rows_stops_the_refit(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 16)
    with pytest.raises(BaselineError, match="cluster leakage"):
        go_net.refit_final(go_net.Source(reader, rows), _plan(), 2, {0})


def test_a_grid_point_scores_every_validation_row_and_flipping_a_fold_changes_nothing(
    np: Any, torch: Any
) -> None:
    reader, rows = _toy(np)
    folds = lopo_folds()
    result = go_net.run_grid_point(
        go_net.Source(reader, rows), folds, _plan(), {99}, lambda _m: None
    )
    assert ((result.oof > 0.0) & (result.oof < 1.0)).all()
    assert set(result.states) == {1, 2, 3, 4}
    assert [r.validate for r in result.runs] == [1, 2, 3, 4]
    inverted = np.where(rows.partition[:, None] == 1, 1 - rows.y, rows.y)
    flipped = go.Rows(**{**_fields(rows), "y": inverted})
    again = go_net.run_grid_point(
        go_net.Source(reader, flipped), folds, _plan(), {99}, lambda _m: None
    )
    held = rows.partition == 1
    assert np.array_equal(result.oof[held], again.oof[held])
    assert not np.array_equal(result.oof[~held], again.oof[~held])


def test_saved_fold_weights_reproduce_the_scores_to_the_configured_tolerance(
    np: Any, torch: Any
) -> None:
    reader, rows = _toy(np)
    plan = _plan()
    result = go_net.run_grid_point(
        go_net.Source(reader, rows), lopo_folds(), plan, {99}, lambda _m: None
    )
    worst = 0.0
    for fold in lopo_folds():
        _train, held = go.fold_rows(rows, fold)
        model = head_net.build_model(
            plan.settings.net, plan.settings.seed, result.states[fold.validate]
        )
        again, _ = head_net.predict(
            model, reader, [rows.shas[i] for i in held], plan.settings.schedule.batch_size
        )
        worst = max(worst, float(np.abs(again - result.oof[held]).max()))
    assert worst <= _config().reproduction_tolerance


# -- the script ------------------------------------------------------------------------
def _script() -> Any:
    spec = importlib.util.spec_from_file_location("go_script", REPO / "scripts" / "go.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_has_exactly_the_documented_modes() -> None:
    script = _script()
    parser = script._parser()
    for mode in ("labels", "train", "evaluate", "calibrate", "freeze", "check"):
        assert getattr(parser.parse_args([f"--{mode}"]), mode) is True
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--train", "--check"])


@pytest.mark.parametrize("argv", [["--train", "--test"], ["--check", "--force"]])
def test_the_script_refuses_flags_that_do_not_go_with_the_mode(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _script().main(argv)


@pytest.mark.parametrize("argv", [["--freeze", "--smoke"], ["--evaluate", "--test", "--smoke"]])
def test_smoke_output_can_never_be_frozen_or_touch_the_test_partition(argv: list[str]) -> None:
    assert _script().main(argv) == 1


def test_the_script_check_passes_on_the_repository_without_data_bytes() -> None:
    script = _script()
    if not script.EMBEDDING_MANIFEST.exists() or not script.SPLIT_MANIFEST.exists():
        pytest.skip("the committed manifests are not present")
    assert script.main(["--check"]) == 0
