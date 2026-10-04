"""Phase 11 signal-peptide CRF: config, tags, labels, rows, metrics, protocol, manifest, fit."""

import ast
from collections.abc import Callable
import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from protloc.calibration import calibrator as cb
from protloc.registry import bundle as registry_bundle
from protloc.training import baseline_models as bm
from protloc.training import crf, crf_net
from protloc.training.baselines import (
    FEATURE_NAMES,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    digest,
    lopo_folds,
)

REPO = Path(__file__).resolve().parents[3]
CRF_YAML = REPO / "configs" / "training" / "crf.yaml"
MODULES = [
    REPO / "backend" / "protloc" / "training" / "crf.py",
    REPO / "backend" / "protloc" / "training" / "crf_net.py",
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


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _config() -> crf.CrfConfig:
    return crf.load_config(CRF_YAML)


# -- configuration ---------------------------------------------------------------------
def test_config_agrees_with_the_root_model_and_evaluation_configs() -> None:
    config = _config()
    assert config.seed == _yaml(REPO / "configs" / "config.yaml")["seed"]
    trunk = _yaml(REPO / "configs" / "model" / "trunk_esm2_150m.yaml")["trunk"]
    assert config.architecture.embedding_dim == trunk["embedding_dim"]
    evaluation = _yaml(REPO / "configs" / "evaluation" / "default.yaml")
    assert evaluation["reporting"]["bootstrap_samples"] == 1000
    assert evaluation["partitions"]["test"] == [TEST_PARTITION]
    assert (REPO / config.calibration_config).is_file()


def test_the_grid_is_eight_points_in_the_documented_order() -> None:
    points = crf.grid_points(_config())
    assert len(points) == 8
    assert [p.encoder for p in points] == ["cnn"] * 4 + ["bilstm"] * 4
    assert points[0].as_dict() == {"encoder": "cnn", "learning_rate": 0.001, "dropout": 0.3}
    assert points[1].dropout == 0.5
    assert points[2].learning_rate == 0.0003


def test_config_values_are_not_repeated_as_literals_in_the_modules() -> None:
    config = _config()
    arch, schedule = config.architecture, config.schedule
    forbidden = {
        0.5,
        config.seed,
        arch.length_cap,
        arch.cnn_hidden,
        arch.cnn_kernel,
        arch.lstm_hidden,
        schedule.batch_size,
        schedule.max_epochs,
        schedule.patience,
        schedule.inner_holdout_fraction,
        config.weight_decay,
        config.tags.forbidden_score,
        config.tags.min_cleavage_positives,
        config.evaluation.group_min_support,
        config.evaluation.reproduction_tolerance,
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
    doc = _yaml(CRF_YAML)
    edit(doc)
    path = tmp_path / "crf.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(task="localization"),
        lambda d: d["architecture"]["cnn"].update(kernel=4),
        lambda d: d["tags"].update(signal="X"),
        lambda d: d["tags"].update(forbidden_score=1.0),
        lambda d: d["tags"].update(forbidden_transitions=[["O", "Z"]]),
        lambda d: d["grid"].update(encoder=["transformer"]),
        lambda d: d["grid"].pop("dropout"),
        lambda d: d["grid"].update(dropout=[1.0]),
        lambda d: d["selection"].update(tolerance=7),
        lambda d: d["evaluation"].update(tolerances=[3, 1]),
        lambda d: d["schedule"].update(inner_holdout_fraction=1.0),
        lambda d: d["optimiser"].update(name="sgd"),
    ],
)
def test_malformed_config_is_refused(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    with pytest.raises(BaselineError):
        crf.load_config(_edited(tmp_path, edit))


def test_the_signal_tag_is_forbidden_after_outside_only() -> None:
    tags = _config().tags
    allowed = tags.allowed()
    o, s = tags.outside_index, tags.signal_index
    assert allowed[o][s] is False
    assert allowed[s][s]
    assert allowed[s][o]
    assert allowed[o][o]


# -- tags from labels ------------------------------------------------------------------
def test_tags_follow_the_cleavage_site_convention_of_adr_0006() -> None:
    tags = _config().tags
    o, s = tags.outside_index, tags.signal_index
    # cleavage_site is the 1-based LAST residue of the peptide: residues 1..3 are signal.
    assert crf.tags_from_cleavage(6, 3, 100, tags) == [s, s, s, o, o, o]
    assert crf.tags_from_cleavage(4, 0, 100, tags) == [o, o, o, o]
    assert crf.tags_from_cleavage(5, 5, 100, tags) == [s] * 5


def test_the_cap_truncates_the_path_and_tags_nothing_beyond_it() -> None:
    tags = _config().tags
    path = crf.tags_from_cleavage(50, 8, 4, tags)
    assert len(path) == 4
    assert crf.cleavage_of_path(path, tags.signal_index) == 4
    assert crf.cleavage_of_path([tags.outside_index] * 3, tags.signal_index) == 0
    assert crf.tags_from_cleavage(3, 2, 100, tags) == [tags.signal_index] * 2 + [tags.outside_index]


# -- the label table -------------------------------------------------------------------
def _record(accession: str, sp: tuple[str, int | None], mtp: str = "negative") -> dict[str, Any]:
    labels = {
        "signal_peptide": {"state": sp[0], "cleavage_site": sp[1]},
        "mitochondrial_transit_peptide": {"state": mtp, "cleavage_site": None},
        "chloroplast_transit_peptide": {"state": "unknown", "cleavage_site": None},
    }
    return {
        "accession": accession,
        "organism_group": "eukarya",
        "length": 120,
        "labels": labels,
    }


def test_the_corpus_scan_keeps_only_the_wanted_accessions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        _record("P1", ("annotated", 22)),
        _record("P2", ("negative", None)),
        _record("P3", ("unknown", None)),
    ]
    corpus.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records) + "\n"
    )
    found = crf.scan_corpus(corpus, ["P1", "P3"])
    assert set(found) == {"P1", "P3"}
    assert found["P1"].cleavage["signal_peptide"] == 22
    assert found["P3"].states["signal_peptide"] == "unknown"
    with pytest.raises(BaselineError, match="not in the corpus"):
        crf.scan_corpus(corpus, ["P1", "P9"])


def test_the_label_table_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [_record("B", ("annotated", 9)), _record("A", ("negative", None))]
    corpus.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records) + "\n"
    )
    labels = crf.scan_corpus(corpus, ["A", "B"])
    text = crf.label_table_text(labels)
    assert text.splitlines()[1].startswith("A\t")
    assert crf.parse_label_table(text) == labels
    assert crf.label_table_text(crf.parse_label_table(text)) == text
    with pytest.raises(BaselineError, match="header"):
        crf.parse_label_table("wrong\n")


def test_label_counts_split_positives_by_partition_and_group() -> None:
    rows = {
        "P1": crf.LabelRow("P1", "eukarya", 100, _states("annotated"), _sites(20)),
        "P2": crf.LabelRow("P2", "eukarya", 100, _states("annotated"), _sites(None)),
        "P3": crf.LabelRow("P3", "eukarya", 100, _states("negative"), _sites(None)),
    }
    report = crf.label_counts(rows, {"P1": 1, "P2": 2, "P3": 5})
    sp = report["signal_peptide"]
    assert sp["positives_with_cleavage"] == 1
    assert sp["positives_without_cleavage"] == 1
    assert sp["by_partition"] == {"1": 1}
    assert sp["states"] == {"annotated": 2, "negative": 1}


def _states(state: str) -> dict[str, str]:
    return dict.fromkeys(crf.LABEL_TASKS, state)


def _sites(site: int | None) -> dict[str, int | None]:
    return dict.fromkeys(crf.LABEL_TASKS, site)


# -- rows ------------------------------------------------------------------------------
def _toy_table(np: Any) -> tuple[dict[str, Any], list[str], dict[str, crf.LabelRow]]:
    accessions = ["A1", "A2", "A3", "A4", "A5"]
    states = ["annotated", "negative", "annotated", "unknown", "annotated"]
    sites = [12, None, None, None, 300]
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 0], [1, 0]], dtype=np.int8)
    mask = np.array([True, True, True, False, True])
    table = {
        "accession": np.array(accessions),
        "partition": np.array([1, 1, 2, 2, 5], dtype=np.int8),
        "cluster_code": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        "seq_index": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        "M_signal_peptide": mask,
        "Y_signal_peptide": y,
    }
    labels = {
        a: crf.LabelRow(a, "eukarya", 400, _states(s), _sites(site))
        for a, s, site in zip(accessions, states, sites, strict=True)
    }
    return table, [f"sha{i}" for i in range(5)], labels


def test_rows_drop_unknown_and_mask_a_positive_without_a_usable_cleavage_site(np: Any) -> None:
    table, order, labels = _toy_table(np)
    spec = crf.RowSpec(VALIDATION_PARTITIONS, 1, 100)
    rows = crf.build_rows(table, order, labels, spec)
    assert list(rows.accession) == ["A1", "A2", "A3"]
    assert rows.y.tolist() == [1, 0, 1]
    assert rows.cleavage.tolist() == [12, 0, 0]
    assert rows.tagged.tolist() == [True, True, False]
    assert list(rows.shas) == ["sha0", "sha1", "sha2"]


def test_a_cleavage_site_beyond_the_cap_is_detection_only_and_test_rows_are_separate(
    np: Any,
) -> None:
    table, order, labels = _toy_table(np)
    rows = crf.build_rows(table, order, labels, crf.RowSpec((TEST_PARTITION,), 1, 100))
    assert list(rows.accession) == ["A5"]
    assert rows.tagged.tolist() == [False]


def test_a_disagreement_between_the_label_table_and_the_feature_table_raises(np: Any) -> None:
    table, order, labels = _toy_table(np)
    table["Y_signal_peptide"][1] = [1, 0]
    with pytest.raises(BaselineError, match="differ"):
        crf.build_rows(table, order, labels, crf.RowSpec(VALIDATION_PARTITIONS, 1, 100))


def test_a_stride_keeps_every_nth_row(np: Any) -> None:
    table, order, labels = _toy_table(np)
    rows = crf.build_rows(table, order, labels, crf.RowSpec(VALIDATION_PARTITIONS, 2, 100))
    assert list(rows.accession) == ["A1", "A3"]


# -- metrics ---------------------------------------------------------------------------
def _scored_rows(np: Any) -> tuple[crf.Rows, crf.Scored]:
    cleavage = np.array([20, 20, 20, 20, 0, 0, 0, 0], dtype=np.int64)
    y = (cleavage > 0).astype(np.int64)
    rows = crf.Rows(
        accession=list("abcdefgh"),
        shas=list("abcdefgh"),
        group=["eukarya"] * 8,
        seq_index=np.arange(8),
        length=np.full(8, 100),
        cleavage=cleavage,
        y=y,
        tagged=np.ones(8, dtype=bool),
        partition=np.array([1, 1, 2, 2, 3, 3, 4, 4]),
        cluster_code=np.arange(8),
    )
    scored = crf.Scored(
        score=np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.6, 0.05]),
        decision=np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=bool),
        cleavage=np.array([20, 21, 18, 30, 9, 9, 22, 9]),
    )
    return rows, scored


def test_metrics_match_a_hand_computation(np: Any) -> None:
    rows, scored = _scored_rows(np)
    values = crf.compute_metrics(rows, scored, np.arange(8), (1, 2, 3))
    # detection: tp 3, fp 1, fn 1, tn 3
    assert values["detection_f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))
    assert values["detection_mcc"] == pytest.approx((9 - 1) / math.sqrt(4 * 4 * 4 * 4))
    # AUC: positives 0.9 0.8 0.7 0.2 vs negatives 0.1 0.3 0.6 0.05 -> 14 of 16 pairs ordered
    assert values["detection_auc"] == pytest.approx(14 / 16)
    # cleavage errors on the four positives are 0, 1, 2, 10
    assert values["cleavage_exact"] == pytest.approx(0.25)
    assert values["cleavage_within_1"] == pytest.approx(0.5)
    assert values["cleavage_within_2"] == pytest.approx(0.75)
    assert values["cleavage_within_3"] == pytest.approx(0.75)
    assert values["cleavage_mae"] == pytest.approx(13 / 4)
    # detected positives are the first three: errors 0, 1, 2
    assert values["cleavage_detected_exact"] == pytest.approx(1 / 3)
    assert values["cleavage_detected_within_2"] == pytest.approx(1.0)
    assert values["cleavage_detected_mae"] == pytest.approx(1.0)
    # residues: truth 20,20,20,20 and 0s; guess 20,21,18,0 and 22 on the false positive
    overlap, extra, missed = 20 + 20 + 18 + 0, 1 + 22, 2 + 20
    assert values["residue_f1"] == pytest.approx(2 * overlap / (2 * overlap + extra + missed))
    assert set(values) == set(crf.metric_names((1, 2, 3)))


def test_auc_gives_tied_scores_their_mean_rank(np: Any) -> None:
    y = np.array([1, 0, 1, 0])
    assert crf.auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert crf.auc_score(y, np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
    assert crf.auc_score(np.array([0, 0]), np.array([0.1, 0.2])) == 0.0


def test_an_undefined_slice_reports_zero_and_is_marked(np: Any) -> None:
    rows, scored = _scored_rows(np)
    idx = np.array([0, 4, 5, 6])
    report = crf.metric_report(rows, scored, idx, (_config().evaluation, bm.Bootstrap(0, 0)))
    assert report["positives"] == 1
    assert report["undefined"] == {"detection": True, "cleavage": True}
    metrics = report["metrics"]
    assert metrics["detection_mcc"]["value"] == 0.0
    assert metrics["detection_auc"]["value"] == 0.0
    assert metrics["cleavage_exact"]["value"] == 0.0
    assert metrics["detection_mcc"]["ci_low"] is None


def test_the_bootstrap_resamples_clusters_and_is_deterministic(np: Any) -> None:
    rows, scored = _scored_rows(np)
    settings = (_config().evaluation, bm.Bootstrap(40, _config().seed))
    first = crf.metric_report(rows, scored, np.arange(8), settings)
    second = crf.metric_report(rows, scored, np.arange(8), settings)
    assert first == second
    assert first["bootstrap"] == {"samples": 40, "seed": _config().seed, "unit": "cluster_id"}
    f1 = first["metrics"]["detection_f1"]
    assert f1["ci_low"] <= f1["value"] <= f1["ci_high"] or f1["ci_low"] <= f1["ci_high"]


def test_groups_below_the_support_floor_are_listed_not_scored(np: Any) -> None:
    rows, scored = _scored_rows(np)
    report = crf.group_reports(rows, scored, (_config().evaluation, bm.Bootstrap(0, 0)))
    assert report["eukarya"]["reported"] is False
    assert report["eukarya"]["positives"] == 4


# -- protocol --------------------------------------------------------------------------
def test_a_fold_threshold_never_sees_its_own_labels(np: Any) -> None:
    rows, scored = _scored_rows(np)
    folds = lopo_folds()
    decision, thresholds, frozen = crf.nested_decisions(scored.score, rows, folds)
    flipped = crf.Rows(**{**_fields(rows), "y": np.where(rows.partition == 1, 1 - rows.y, rows.y)})
    again, thresholds_again, frozen_again = crf.nested_decisions(scored.score, flipped, folds)
    assert thresholds[1] == thresholds_again[1]
    assert decision[rows.partition == 1].tolist() == again[rows.partition == 1].tolist()
    assert thresholds[2] != thresholds_again[2] or frozen != frozen_again


def _fields(rows: crf.Rows) -> dict[str, Any]:
    return {name: getattr(rows, name) for name in rows.__slots__}


def test_selection_prefers_cleavage_then_mcc_then_the_earlier_point() -> None:
    assert crf.select_point([(0.5, 0.9), (0.6, 0.1), (0.4, 0.99)]) == 1
    assert crf.select_point([(0.6, 0.1), (0.6, 0.3), (0.6, 0.3)]) == 1
    assert crf.select_point([(0.6, 0.3), (0.6, 0.3)]) == 0


def test_the_selector_is_the_mean_fold_accuracy_within_the_tolerance(np: Any) -> None:
    rows, scored = _scored_rows(np)
    folds = lopo_folds()
    # fold 1 positives: errors 0 and 1; fold 2: errors 2 and 10; folds 3 and 4 have none
    assert crf.fold_cleavage_accuracy(rows, scored.cleavage, 1, 1) == 1.0
    assert crf.fold_cleavage_accuracy(rows, scored.cleavage, 2, 1) == 0.0
    assert crf.fold_cleavage_accuracy(rows, scored.cleavage, 3, 1) == 0.0
    assert crf.selection_value(rows, scored.cleavage, folds, 1) == pytest.approx(0.25)


def test_the_protocol_keeps_partition_five_out(np: Any) -> None:
    for fold in lopo_folds():
        assert TEST_PARTITION not in fold.train
        assert fold.validate in VALIDATION_PARTITIONS


def test_the_motif_floor_reads_the_engine_score_and_its_own_cleavage_site(np: Any) -> None:
    column = FEATURE_NAMES.index("cleavage_position")
    x = np.zeros((3, len(FEATURE_NAMES)))
    x[:, column] = [0.0, 24.0, 31.0]
    table = {"R_signal_peptide": np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]), "X": x}
    rows, _ = _scored_rows(np)
    small = crf.Rows(**{**_fields(rows), "seq_index": np.array([2, 1])})
    score, cleavage = crf.motif_arrays(table, small)
    assert score.tolist() == [0.7, 0.8]
    assert cleavage.tolist() == [31, 24]


# -- the manifest ----------------------------------------------------------------------
def _pins() -> crf.Pins:
    config = _config()
    return crf.Pins(
        registry_sha256="1" * 64,
        corpus_sha256="2" * 64,
        embedding_cache_key="3" * 64,
        residue_cache_key="4" * 64,
        config_sha256=config.sha256,
        calibration_config_sha256="5" * 64,
        tags=config.tags,
        tolerances=config.evaluation.tolerances,
        reproduction_tolerance=config.evaluation.reproduction_tolerance,
        torch="2.14.1",
    )


def _report() -> dict[str, Any]:
    names = crf.metric_names(_config().evaluation.tolerances)
    return {"pooled": {"metrics": {n: {"value": 0.0} for n in names}}}


def _frozen() -> dict[str, Any]:
    pins = _pins()
    return {
        "protocol": {"validation_partitions": [1, 2, 3, 4], "test_partition": 5},
        "scope": "subset",
        "scope_label": (
            "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"
        ),
        "subset": {"embedding_cache_key": "3" * 64, "sequences_sha256": SHA, "sequences": 500},
        "inputs": {
            "registry_sha256": pins.registry_sha256,
            "corpus_sha256": pins.corpus_sha256,
            "residue_cache_key": pins.residue_cache_key,
            "crf_config_sha256": pins.config_sha256,
            "calibration_config_sha256": pins.calibration_config_sha256,
            "feature_table_sha256": SHA,
            "label_table_sha256": SHA,
            "weights_sha256": SHA,
            "training_record_sha256": SHA,
            "calibrator_sha256": SHA,
        },
        "tag_scope": {
            "tags": list(pins.tags.names),
            "undefined": {
                task: {"positives_with_cleavage_validation": 9, "trained": False}
                for task in pins.tags.undefined_tasks
            },
        },
        "selected": {"trainable_parameters": 1234, "final_epochs": 7},
        "thresholds": {"crf_raw": 0.4, "crf_calibrated": 0.5, "motif": 0.3},
        "validation": {
            "crf_raw": _report(),
            "crf_calibrated": _report(),
            "motif_floor": _report(),
            "reproduction": {"max_abs_difference": 0.0, "decisions_identical": True},
        },
        "software": {"torch": "2.14.1"},
    }


def _manifest(test: Any = None) -> dict[str, Any]:
    frozen = _frozen()
    return {"schema_version": 1, "frozen": frozen, "frozen_sha256": digest(frozen), "test": test}


def _test_block(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "metrics": {n: _report()["pooled"] for n in ("crf_raw", "crf_calibrated", "motif_floor")},
    }


def test_a_sound_manifest_has_no_problems_with_or_without_a_test_block() -> None:
    manifest = _manifest()
    assert crf.check_crf_manifest(manifest, _pins()) == []
    manifest["test"] = _test_block(manifest)
    assert crf.check_crf_manifest(manifest, _pins()) == []


def _rehash(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    return manifest


@pytest.mark.parametrize(
    ("edit", "fragment"),
    [
        (lambda f: f["inputs"].update(registry_sha256="9" * 64), "registry_sha256"),
        (lambda f: f["inputs"].update(crf_config_sha256="9" * 64), "crf_config_sha256"),
        (lambda f: f["inputs"].update(weights_sha256="nope"), "weights_sha256"),
        (lambda f: f["protocol"].update(test_partition=4), "test partition"),
        (lambda f: f["tag_scope"].update(tags=["O"]), "tag_scope.tags"),
        (
            lambda f: f["tag_scope"]["undefined"]["mitochondrial_transit_peptide"].update(
                positives_with_cleavage_validation=20
            ),
            "enough positives",
        ),
        (
            lambda f: f["tag_scope"]["undefined"]["chloroplast_transit_peptide"].update(
                trained=True
            ),
            "not trained",
        ),
        (lambda f: f["thresholds"].update(crf_raw=float("nan")), "thresholds.crf_raw"),
        (lambda f: f["validation"]["reproduction"].update(max_abs_difference=1e-3), "reproduce"),
        (
            lambda f: f["validation"]["reproduction"].update(decisions_identical=False),
            "decisions",
        ),
        (lambda f: f["validation"].pop("motif_floor"), "motif_floor"),
        (lambda f: f["software"].update(torch="9.9.9"), "torch"),
    ],
)
def test_every_disagreement_is_reported(
    edit: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    manifest = _manifest()
    edit(manifest["frozen"])
    _rehash(manifest)
    problems = crf.check_crf_manifest(manifest, _pins())
    assert any(fragment in p for p in problems), problems


def test_a_hand_edited_frozen_block_is_detected() -> None:
    manifest = _manifest()
    manifest["frozen"]["selected"]["final_epochs"] = 99
    assert any("frozen_sha256" in p for p in crf.check_crf_manifest(manifest, _pins()))


def test_a_test_block_without_the_hash_or_with_two_touches_is_refused() -> None:
    manifest = _manifest()
    block = _test_block(manifest)
    del block["frozen_sha256"]
    manifest["test"] = block
    assert any("different frozen block" in p for p in crf.check_crf_manifest(manifest, _pins()))
    manifest["test"] = {**_test_block(manifest), "touches": 2}
    assert any("exactly once" in p for p in crf.check_crf_manifest(manifest, _pins()))
    manifest["test"] = {**_test_block(manifest), "metrics": {"crf_raw": _report()["pooled"]}}
    assert any("do not cover" in p for p in crf.check_crf_manifest(manifest, _pins()))


def test_a_manifest_missing_a_section_is_refused_not_crashed() -> None:
    manifest = copy.deepcopy(_manifest())
    del manifest["frozen"]["inputs"]
    assert crf.check_crf_manifest(manifest, _pins())[0].startswith("manifest is missing")


# -- the CRF against brute force -------------------------------------------------------
def _tensors(torch: Any, seed: int, steps: int, size: int, banned: set[tuple[int, int]]) -> Any:
    generator = torch.Generator().manual_seed(seed)
    emissions = torch.randn(1, steps, size, generator=generator, dtype=torch.float64)
    transitions = torch.randn(size, size, generator=generator, dtype=torch.float64)
    for a, b in banned:
        transitions[a, b] = _config().tags.forbidden_score
    crf_t = crf_net.CrfTensors(
        transitions,
        torch.randn(size, generator=generator, dtype=torch.float64),
        torch.randn(size, generator=generator, dtype=torch.float64),
    )
    return emissions, crf_t


def _brute(emissions: Any, crf_t: Any) -> tuple[float, list[int], list[float], Any]:
    steps, size = emissions.shape[1], emissions.shape[2]
    scores: dict[tuple[int, ...], float] = {}
    for path in itertools.product(range(size), repeat=steps):
        total = float(crf_t.start[path[0]]) + float(crf_t.end[path[-1]])
        total += sum(float(emissions[0, t, path[t]]) for t in range(steps))
        total += sum(float(crf_t.transitions[a, b]) for a, b in itertools.pairwise(path))
        scores[path] = total
    top = max(scores.values())
    log_z = top + math.log(sum(math.exp(v - top) for v in scores.values()))
    best = max(scores, key=lambda p: scores[p])
    marginal = [[0.0] * size for _ in range(steps)]
    for path, value in scores.items():
        for t, tag in enumerate(path):
            marginal[t][tag] += math.exp(value - log_z)
    return log_z, list(best), scores[best], marginal


CASES = [
    (3, 2, {(0, 1)}),
    (6, 2, {(0, 1)}),
    (6, 2, set()),
    (5, 3, {(0, 2), (1, 2)}),
    (1, 2, set()),
]


@pytest.mark.parametrize(("steps", "size", "banned"), CASES)
def test_the_crf_matches_brute_force_enumeration(
    torch: Any, steps: int, size: int, banned: set[tuple[int, int]]
) -> None:
    emissions, crf_t = _tensors(torch, 7, steps, size, banned)
    mask = torch.ones(1, steps, dtype=torch.bool)
    log_z, best, best_score, marginal = _brute(emissions, crf_t)
    assert float(crf_net.log_partition(emissions, mask, crf_t)) == pytest.approx(log_z, abs=1e-9)
    path, score = crf_net.viterbi(emissions, mask, crf_t)
    assert path[0].tolist() == best
    assert float(score) == pytest.approx(best_score, abs=1e-9)
    assert float(crf_net.path_score(emissions, mask, path, crf_t)) == pytest.approx(
        best_score, abs=1e-9
    )
    soft = crf_net.marginals(emissions, mask, crf_t)[0]
    assert soft.sum(dim=1).tolist() == pytest.approx([1.0] * steps, abs=1e-9)
    for t in range(steps):
        assert soft[t].tolist() == pytest.approx(marginal[t], abs=1e-9)


def test_padding_never_changes_the_crf_result(torch: Any) -> None:
    banned = {(0, 1)}
    short, crf_t = _tensors(torch, 3, 4, 2, banned)
    other, _ = _tensors(torch, 4, 6, 2, banned)
    batch = torch.cat([torch.cat([short, torch.randn(1, 2, 2, dtype=torch.float64)], 1), other])
    mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    alone = torch.ones(1, 4, dtype=torch.bool)
    z = crf_net.log_partition(batch, mask, crf_t)
    assert float(z[0]) == pytest.approx(float(crf_net.log_partition(short, alone, crf_t)), abs=1e-9)
    path, _ = crf_net.viterbi(batch, mask, crf_t)
    assert path[0, :4].tolist() == crf_net.viterbi(short, alone, crf_t)[0][0].tolist()
    assert path[0, 4:].tolist() == [0, 0]
    soft = crf_net.marginals(batch, mask, crf_t)
    assert soft[0, 4:].abs().sum() == 0
    assert soft[0, :4].tolist() == crf_net.marginals(short, alone, crf_t)[0].tolist() or (
        (soft[0, :4] - crf_net.marginals(short, alone, crf_t)[0]).abs().max() < 1e-9
    )
    gold = torch.zeros(2, 6, dtype=torch.long)
    scored = crf_net.path_score(batch, mask, gold, crf_t)
    assert float(scored[0]) == pytest.approx(
        float(crf_net.path_score(short, alone, gold[:1, :4], crf_t)), abs=1e-9
    )


def test_a_forbidden_transition_is_never_decoded_or_given_probability(torch: Any) -> None:
    config = _config()
    tags = config.tags
    o, s = tags.outside_index, tags.signal_index
    emissions, crf_t = _tensors(torch, 11, 6, 2, {(o, s)})
    mask = torch.ones(1, 6, dtype=torch.bool)
    emissions[0, 3, s] = 50.0
    path = crf_net.viterbi(emissions, mask, crf_t)[0][0].tolist()
    assert all(not (a == o and b == s) for a, b in itertools.pairwise(path))
    log_z = float(crf_net.log_partition(emissions, mask, crf_t))
    brute = _brute(emissions, crf_t)[0]
    assert log_z == pytest.approx(brute, abs=1e-9)


def test_detection_is_the_probability_of_any_signal_and_the_residue_one_marginal(
    torch: Any,
) -> None:
    tags = _config().tags
    o, s = tags.outside_index, tags.signal_index
    emissions, crf_t = _tensors(torch, 5, 6, 2, {(o, s)})
    mask = torch.ones(1, 6, dtype=torch.bool)
    outside = crf_net.outside_log_prob(emissions, mask, crf_t, o)
    detection = float(-torch.expm1(outside))
    first = float(crf_net.marginals(emissions, mask, crf_t)[0, 0, s])
    assert detection == pytest.approx(first, abs=1e-9)
    # brute force: the sum over every path with a signal tag
    steps = 6
    scores = {}
    for path in itertools.product(range(2), repeat=steps):
        scores[path] = (
            float(crf_t.start[path[0]])
            + float(crf_t.end[path[-1]])
            + sum(float(emissions[0, t, path[t]]) for t in range(steps))
            + sum(float(crf_t.transitions[a, b]) for a, b in itertools.pairwise(path))
        )
    log_z = float(crf_net.log_partition(emissions, mask, crf_t))
    summed = sum(math.exp(v - log_z) for p, v in scores.items() if s in p)
    assert detection == pytest.approx(summed, abs=1e-9)
    # the alternative score (best single cleavage) can only be smaller than the sum
    longest = max(math.exp(v - log_z) for p, v in scores.items() if s in p)
    assert longest <= detection
    assert float(crf_net.log_one_minus_exp(outside)) == pytest.approx(math.log(detection), abs=1e-9)


def test_log_one_minus_exp_is_stable_at_both_ends(torch: Any) -> None:
    value = torch.tensor([-1e-12, -1e-3, -1.0, -50.0, -800.0], dtype=torch.float64)
    out = crf_net.log_one_minus_exp(value)
    assert torch.isfinite(out).all()
    assert float(out[2]) == pytest.approx(math.log(1 - math.exp(-1.0)))
    assert float(out[3]) == pytest.approx(math.log1p(-math.exp(-50.0)))


def test_gold_paths_use_the_last_residue_convention(torch: Any) -> None:
    tags = _config().tags
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    gold = crf_net.gold_paths(torch.tensor([2, 0]), mask, tags)
    s, o = tags.signal_index, tags.outside_index
    assert gold.tolist() == [[s, s, o, o, o], [o] * 5]
    capped = crf_net.gold_paths(torch.tensor([9, 9]), mask, tags)
    assert capped.tolist() == [[s] * 5, [s] * 3 + [o] * 2]


# -- the network and the fit -----------------------------------------------------------
def _arch(config: crf.CrfConfig) -> crf.Architecture:
    base = config.architecture
    return crf.Architecture(
        embedding_dim=DIM,
        length_cap=base.length_cap,
        cnn_hidden=8,
        cnn_kernel=base.cnn_kernel,
        cnn_layers=base.cnn_layers,
        lstm_hidden=8,
        lstm_layers=base.lstm_layers,
    )


def _net(encoder: str, dropout: float = 0.0) -> crf_net.Net:
    config = _config()
    return crf_net.Net(_arch(config), config.tags, encoder, dropout)


def _batch(np: Any, torch: Any, lengths: list[int]) -> tuple[Any, Any]:
    rng = np.random.default_rng(1)
    widest = max(lengths)
    x = np.zeros((len(lengths), widest, DIM), dtype=np.float32)
    mask = np.zeros((len(lengths), widest), dtype=bool)
    for i, n in enumerate(lengths):
        x[i, :n] = rng.normal(0.0, 1.0, (n, DIM))
        mask[i, :n] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


@pytest.mark.parametrize("encoder", ["cnn", "bilstm"])
def test_padding_never_changes_the_emissions_of_a_real_residue(
    np: Any, torch: Any, encoder: str
) -> None:
    model = crf_net.build_model(_net(encoder), 3)
    x, mask = _batch(np, torch, [4, 9])
    with torch.no_grad():
        batched = model.emissions(x, mask)
        alone = model.emissions(x[:1, :4], mask[:1, :4])
    assert float((batched[0, :4] - alone[0]).abs().max()) < 1e-5


@pytest.mark.parametrize("encoder", ["cnn", "bilstm"])
def test_the_network_is_deterministic_and_depends_on_the_seed(
    np: Any, torch: Any, encoder: str
) -> None:
    x, mask = _batch(np, torch, [5, 7])
    outs = []
    for seed in (1, 1, 2):
        model = crf_net.build_model(_net(encoder), seed)
        with torch.no_grad():
            outs.append(model.emissions(x, mask))
    assert torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[0], outs[2])


def test_the_two_encoders_have_different_parameter_counts(np: Any, torch: Any) -> None:
    cnn = crf_net.SequenceTagger(_net("cnn"), 1).parameter_count()
    lstm = crf_net.SequenceTagger(_net("bilstm"), 1).parameter_count()
    assert cnn > 0
    assert lstm > 0
    assert cnn != lstm
    with pytest.raises(BaselineError, match="encoder"):
        crf_net.SequenceTagger(_net("rnn"), 1)


class FakeReader:
    """Per-residue states held in memory, read by hash like the real cache."""

    def __init__(self, states: dict[str, Any]) -> None:
        """Keep the states."""
        self._states = states

    def states(self, sha256: str) -> Any:
        """The (length, dim) states of one sequence."""
        return self._states[sha256]


def _toy(np: Any, n: int = 48) -> tuple[FakeReader, crf.Rows]:
    """A separable problem: the first ``cleavage`` residues of a positive carry a signal."""
    rng = np.random.default_rng(0)
    shas = [f"s{i:03d}" for i in range(n)]
    cleavage = np.zeros(n, dtype=np.int64)
    states, lengths = {}, []
    for i, sha in enumerate(shas):
        length = int(rng.integers(12, 20))
        lengths.append(length)
        array = rng.normal(0.0, 0.3, (length, DIM))
        if i % 3 == 0:
            cleavage[i] = int(rng.integers(3, 7))
            array[: cleavage[i], 0] += 3.0
        states[sha] = array.astype(np.float16)
    partition = np.array([1 + (i % 4) for i in range(n)])
    rows = crf.Rows(
        accession=shas,
        shas=shas,
        group=["eukarya"] * n,
        seq_index=np.arange(n),
        length=np.array(lengths),
        cleavage=cleavage,
        y=(cleavage > 0).astype(np.int64),
        tagged=np.ones(n, dtype=bool),
        partition=partition,
        cluster_code=partition - 1,
    )
    return FakeReader(states), rows


def _settings(**overrides: Any) -> crf_net.FitSettings:
    base = _config()
    values: dict[str, Any] = {
        "net": _net("cnn"),
        "learning_rate": 0.05,
        "weight_decay": base.weight_decay,
        "schedule": crf.Schedule(
            batch_size=8, max_epochs=30, patience=5, inner_holdout_fraction=0.25
        ),
        "seed": 7,
    }
    values.update(overrides)
    return crf_net.FitSettings(**values)


def test_a_fit_learns_a_separable_problem_and_is_reproducible(np: Any, torch: Any) -> None:
    reader, rows = _toy(np)
    source = crf_net.Source(reader, rows)
    everything = list(range(len(rows)))
    first = crf_net.fit_tagger(source, everything, None, _settings(), epochs=25)
    second = crf_net.fit_tagger(source, everything, None, _settings(), epochs=25)
    for name in first.state:
        assert np.array_equal(first.state[name], second.state[name])
    model = crf_net.build_model(_settings().net, 7, first.state)
    prediction = crf_net.predict(model, reader, list(rows.shas), 8, keep_marginals=True)
    assert crf.auc_score(rows.y, prediction.score) > 0.95
    positives = rows.y == 1
    error = np.abs(prediction.cleavage[positives] - rows.cleavage[positives])
    assert float(np.mean(error <= 1)) > 0.8
    assert prediction.marginals is not None
    assert len(prediction.marginals) == len(rows)
    assert all(-1e-6 <= float(v) <= 1.0 + 1e-6 for m in prediction.marginals for v in m)


def test_predict_restores_the_order_asked_and_never_exceeds_the_cap(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 12)
    model = crf_net.build_model(_settings().net, 3)
    shas = list(rows.shas)[::-1]
    forward = crf_net.predict(model, reader, list(rows.shas), 4, keep_marginals=True)
    backward = crf_net.predict(model, reader, shas, 5, keep_marginals=True)
    assert np.allclose(forward.score[::-1], backward.score, atol=1e-6)
    assert forward.cleavage.min() >= 1
    assert int(forward.cleavage.max()) <= _config().architecture.length_cap
    assert forward.marginals is not None
    assert [len(m) for m in forward.marginals] == [int(n) for n in rows.length]


def test_a_positive_without_a_cleavage_site_is_masked_in_the_tag_loss(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 8)
    model = crf_net.build_model(_settings().net, 3)
    x, mask = crf_net.gather(reader, list(rows.shas), DIM, 100)
    cleavage = torch.from_numpy(rows.cleavage)
    tagged = torch.ones(8, dtype=torch.bool)
    positive = int(np.argmax(rows.cleavage > 0))
    wrong = cleavage.clone()
    wrong[positive] = 1
    tagged[positive] = False
    with torch.no_grad():
        masked_a = float(crf_net.batch_loss(model, x, mask, (cleavage, tagged)))
        masked_b = float(crf_net.batch_loss(model, x, mask, (wrong, tagged)))
        full = float(
            crf_net.batch_loss(model, x, mask, (cleavage, torch.ones(8, dtype=torch.bool)))
        )
    assert masked_a == pytest.approx(masked_b)
    assert math.isfinite(masked_a)
    assert masked_a != pytest.approx(full)


def test_a_fit_needs_exactly_one_of_a_holdout_or_an_epoch_count(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 12)
    source = crf_net.Source(reader, rows)
    with pytest.raises(BaselineError, match="either"):
        crf_net.fit_tagger(source, [0, 1], None, _settings())
    with pytest.raises(BaselineError, match="either"):
        crf_net.fit_tagger(source, [0, 1], [2], _settings(), epochs=2)


def test_a_cluster_shared_across_partitions_stops_the_fit_before_it_starts(
    np: Any, torch: Any
) -> None:
    reader, rows = _toy(np)
    leaky = crf.Rows(**{**_fields(rows), "cluster_code": np.zeros(len(rows), dtype=np.int64)})
    with pytest.raises(BaselineError, match=r"whole cluster|at least two clusters|leakage"):
        crf_net.run_grid_point(
            crf_net.Source(reader, leaky), lopo_folds(), _settings(), set(), lambda _m: None
        )


def test_a_test_cluster_in_the_training_rows_stops_the_refit(np: Any, torch: Any) -> None:
    reader, rows = _toy(np, 16)
    with pytest.raises(BaselineError, match="cluster leakage"):
        crf_net.refit_final(crf_net.Source(reader, rows), _settings(), 2, {0})


def test_a_grid_point_scores_every_validation_row_and_flipping_a_fold_changes_nothing(
    np: Any, torch: Any
) -> None:
    reader, rows = _toy(np)
    folds = lopo_folds()
    settings = _settings()
    result = crf_net.run_grid_point(
        crf_net.Source(reader, rows), folds, settings, {99}, lambda _m: None
    )
    assert (result.score >= 0).all()
    assert set(result.states) == {1, 2, 3, 4}
    # Label-flip: change every label of partition 1. The scores the model gives partition 1
    # come from weights fitted on partitions 2-4 only, so they cannot move.
    flipped_y = np.where(rows.partition == 1, 1 - rows.y, rows.y)
    flipped_cs = np.where(rows.partition == 1, np.where(rows.y == 1, 0, 4), rows.cleavage)
    flipped = crf.Rows(**{**_fields(rows), "y": flipped_y, "cleavage": flipped_cs})
    again = crf_net.run_grid_point(
        crf_net.Source(reader, flipped), folds, settings, {99}, lambda _m: None
    )
    held = rows.partition == 1
    assert np.array_equal(result.score[held], again.score[held])
    assert np.array_equal(result.cleavage[held], again.cleavage[held])
    assert not np.array_equal(result.score[~held], again.score[~held])


def test_saved_fold_weights_reproduce_the_scores_to_the_configured_tolerance(
    np: Any, torch: Any
) -> None:
    reader, rows = _toy(np)
    settings = _settings()
    result = crf_net.run_grid_point(
        crf_net.Source(reader, rows), lopo_folds(), settings, {99}, lambda _m: None
    )
    tolerance = _config().evaluation.reproduction_tolerance
    worst = 0.0
    for fold in lopo_folds():
        held = [i for i, p in enumerate(rows.partition) if int(p) == fold.validate]
        model = crf_net.build_model(settings.net, settings.seed, result.states[fold.validate])
        again = crf_net.predict(model, reader, [rows.shas[i] for i in held], 8)
        worst = max(worst, float(np.abs(again.score - result.score[held]).max()))
    assert worst <= tolerance


# -- the signal-peptide registry bundle ------------------------------------------------
def _sp_head(**overrides: Any) -> dict[str, Any]:
    head = {
        "weights_file": "final_weights.npz",
        "weights_sha256": SHA,
        "frozen_sha256": "7" * 64,
        "head_config_sha256": "6" * 64,
        "classes": ["signal_peptide"],
        "tags": ["O", "S"],
        "length_cap": 100,
        "encoder": "cnn",
        "detection_score": "p_any_signal",
        "cleavage_rule": "viterbi given residue 1 is signal",
        "source": "model",
    }
    head.update(overrides)
    return head


def _write_sp_bundle(tmp_path: Path, head: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    rules = {"baseline_class_map": tmp_path / "map.yaml", "consistency": tmp_path / "c.json"}
    rules["baseline_class_map"].write_text("classes: [a]\n", encoding="utf-8")
    rules["consistency"].write_text("{}\n", encoding="utf-8")
    shipped = cb.CalibratorSet(
        task="signal_peptide",
        calibration_version="0.11.0+test",
        classes=("signal_peptide",),
        head_weights_sha256=SHA,
        head_frozen_sha256="7" * 64,
        threshold_rule="rule",
        labels=(cb.LabelCalibrator("signal_peptide", cb.UNCALIBRATED, 30, 200, 0.5),),
    )
    document = registry_bundle.build_document(
        version="0.11.0",
        head=head,
        calibrator=shipped,
        rules={f"{key}_sha256": cb.sha256_hex(path.read_bytes()) for key, path in rules.items()},
        scope={"scope": "subset", "scope_label": "label"},
    )
    directory = tmp_path / "registry" / "signal_peptide" / "0.11.0"
    directory.mkdir(parents=True)
    (directory / registry_bundle.BUNDLE_FILE).write_bytes(cb.canonical_json_bytes(document))
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(shipped.to_bytes())
    return directory, rules


def test_a_signal_peptide_bundle_loads_with_the_localization_loader(tmp_path: Path) -> None:
    directory, rules = _write_sp_bundle(tmp_path, _sp_head())
    loaded = registry_bundle.load_bundle(directory, rules)
    assert loaded.document["task"] == "signal_peptide"
    assert loaded.calibrators.classes == ("signal_peptide",)
    assert loaded.document["head"]["source"] == "model"


@pytest.mark.parametrize(
    "key",
    ["tags", "length_cap", "encoder", "detection_score", "cleavage_rule", "source"],
)
def test_a_signal_peptide_bundle_without_a_head_definition_is_refused(
    tmp_path: Path, key: str
) -> None:
    head = _sp_head()
    del head[key]
    directory, rules = _write_sp_bundle(tmp_path, head)
    with pytest.raises(registry_bundle.BundleError, match=f"head.{key}"):
        registry_bundle.load_bundle(directory, rules)


def test_the_localization_head_contract_is_unchanged() -> None:
    assert registry_bundle.TASK_HEAD_KEYS["localization"] == ()
    assert set(registry_bundle.TASK_HEAD_KEYS) == {
        "localization",
        "signal_peptide",
        "go_function",
    }
    assert registry_bundle.RULE_KEYS == ("baseline_class_map", "consistency")
