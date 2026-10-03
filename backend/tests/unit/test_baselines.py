"""Phase 8 baselines: vectors, labels, folds, metrics, thresholds and the manifest check.

Toy data appears only here. Every expected number in the metric tests is worked out by
hand in the comment above it; no figure from a real run is asserted.
"""

import copy
import math
from pathlib import Path
import re
from typing import Any

import pytest
import yaml

from protloc.data import reference_demo
from protloc.features.classical import ORGANISM_GROUPS, extract, raw_scores
from protloc.training import baseline_models
from protloc.training import baselines as bl

ROOT = Path(__file__).resolve().parents[3]
CLASS_MAP = ROOT / "backend/rules/baseline_class_map.yaml"
LABEL_MAP = ROOT / "backend/rules/uniprot_label_map.yaml"
PYPROJECT = ROOT / "backend/pyproject.toml"
MANIFEST = ROOT / "data/manifests/baseline_metrics.json"
SPLIT_MANIFEST = ROOT / "data/manifests/split_manifest.json"
SEQUENCES = [e["sequence"] for e in reference_demo.EXAMPLES]


@pytest.fixture(scope="module")
def cmap() -> bl.ClassMap:
    return bl.load_class_map(CLASS_MAP)


# -- the vector ------------------------------------------------------------------------
def test_the_vector_has_one_fixed_length_for_every_sequence() -> None:
    lengths = {len(bl.vectorise(extract(seq))) for seq in SEQUENCES}
    assert lengths == {len(bl.FEATURE_NAMES)}
    assert len(set(bl.FEATURE_NAMES)) == len(bl.FEATURE_NAMES)


def test_the_vector_is_finite_and_deterministic() -> None:
    for seq in [*SEQUENCES, "M", "MK", "A" * 5]:
        first = bl.vectorise(extract(seq))
        assert first == bl.vectorise(extract(seq))
        assert all(math.isfinite(v) for v in first)


def test_motif_columns_mean_what_their_names_say() -> None:
    kdel = bl.vectorise(extract("MKTLLLLAVVLSAFSSAHA" + "G" * 40 + "KDEL"))
    plain = bl.vectorise(extract("MKTLLLLAVVLSAFSSAHA" + "G" * 40 + "GGGG"))
    index = bl.FEATURE_NAMES.index("has_kdel")
    assert kdel[index] == 1.0
    assert plain[index] == 0.0


# -- the class map and the labels ------------------------------------------------------
def test_the_localization_classes_equal_the_label_map(cmap: bl.ClassMap) -> None:
    label_map = yaml.safe_load(LABEL_MAP.read_text(encoding="utf-8"))
    assert list(cmap.classes["localization"]) == label_map["localization"]["classes"]
    assert set(cmap.from_registry) == set(label_map["membrane_type"]["classes"])


def test_every_engine_key_exists_in_the_engine(cmap: bl.ClassMap) -> None:
    group_keys = set(ORGANISM_GROUPS[cmap.engine_group]["locs"])
    assert set(cmap.engine_keys["localization"].values()) == group_keys


def test_only_annotated_localization_rows_are_used(cmap: bl.ClassMap) -> None:
    names = cmap.classes["localization"]
    row = bl.label_vector("localization", "annotated", ["nucleus", "cytoplasm"], cmap)
    assert row is not None
    assert [n for n, v in zip(names, row, strict=True) if v] == ["cytoplasm", "nucleus"]
    for state in ("unknown", "negative", "not_applicable"):
        assert bl.label_vector("localization", state, [], cmap) is None
    with pytest.raises(bl.BaselineError):
        bl.label_vector("localization", "annotated", ["not_a_class"], cmap)


def test_unknown_is_never_a_negative(cmap: bl.ClassMap) -> None:
    for task in ("membrane_type", "signal_peptide"):
        assert bl.label_vector(task, "unknown", [], cmap) is None
        assert bl.label_vector(task, "not_applicable", [], cmap) is None


def test_membrane_and_signal_labels_follow_adr_0006(cmap: bl.ClassMap) -> None:
    membrane = cmap.classes["membrane_type"]
    topology = bl.label_vector("membrane_type", "annotated", ["single_pass_type_ii"], cmap)
    soluble = bl.label_vector("membrane_type", "negative", [], cmap)
    assert topology == [1 if n == "transmembrane" else 0 for n in membrane]
    assert soluble == [1 if n == "soluble" else 0 for n in membrane]
    assert bl.label_vector("signal_peptide", "annotated", ["signal_peptide"], cmap) == [1, 0]
    assert bl.label_vector("signal_peptide", "negative", [], cmap) == [0, 1]


def test_rule_vectors_have_one_score_per_class(cmap: bl.ClassMap) -> None:
    features = extract(SEQUENCES[0])
    for task in bl.TASKS:
        vector = bl.rule_vector(task, features, cmap)
        assert len(vector) == len(cmap.classes[task])
        assert all(0.0 <= v <= 1.0 for v in vector)
    scores = raw_scores(features, cmap.engine_group)
    nucleus = cmap.classes["localization"].index("nucleus")
    assert bl.rule_vector("localization", features, cmap)[nucleus] == scores["NUC"]
    sp = bl.rule_vector("signal_peptide", features, cmap)
    assert sp[0] + sp[1] == pytest.approx(1.0)


def test_a_malformed_class_map_is_rejected(tmp_path: Path) -> None:
    broken = yaml.safe_load(CLASS_MAP.read_text(encoding="utf-8"))
    del broken["tasks"]["signal_peptide"]
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(bl.BaselineError):
        bl.load_class_map(path)


# -- the folds -------------------------------------------------------------------------
def test_folds_never_overlap_and_never_contain_the_test() -> None:
    folds = bl.lopo_folds()
    assert [f.validate for f in folds] == [1, 2, 3, 4]
    for fold in folds:
        assert fold.validate not in fold.train
        assert bl.TEST_PARTITION not in fold.train
        assert set(fold.train) | {fold.validate} == set(bl.VALIDATION_PARTITIONS)
    bl.assert_protocol(folds)


def test_a_fold_that_reads_the_test_is_refused() -> None:
    with pytest.raises(bl.BaselineError):
        bl.lopo_folds((1, 2, 3, 5))
    with pytest.raises(bl.BaselineError):
        bl.assert_protocol([bl.Fold((1, 2, 5), 3)])
    with pytest.raises(bl.BaselineError):
        bl.assert_protocol([bl.Fold((1, 2, 3), 3)])


def test_the_grids_are_small_and_named() -> None:
    assert set(bl.GRIDS) == set(bl.MODEL_NAMES)
    assert all(0 < len(grid) <= 4 for grid in bl.GRIDS.values())


# -- the pins --------------------------------------------------------------------------
def test_every_baselines_pin_is_exact() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"^baselines = \[(.*?)^\]", text, re.M | re.S)
    assert block is not None
    entries = re.findall(r'"([^"]+)"', block.group(1))
    assert {e.split("==")[0] for e in entries} >= {"scikit-learn", "numpy", "scipy", "joblib"}
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", e) for e in entries)


def test_the_installed_scikit_learn_is_the_pinned_one() -> None:
    sklearn = pytest.importorskip("sklearn")
    assert sklearn.__version__ == bl.pinned_sklearn(PYPROJECT.read_text(encoding="utf-8"))


def test_an_unpinned_scikit_learn_is_an_error() -> None:
    with pytest.raises(bl.BaselineError):
        bl.pinned_sklearn('baselines = [\n  "scikit-learn>=1.0",\n]\n')


# -- metrics, by hand ------------------------------------------------------------------
@pytest.fixture(scope="module")
def models() -> Any:
    pytest.importorskip("numpy")
    return baseline_models


def _toy(models: Any) -> tuple[Any, Any]:
    np = models._np()
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 1]])
    p = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
    return y, p


def test_counts_f1_mcc_and_jaccard_match_the_hand_worked_values(models: Any) -> None:
    # Class 0: y = 1,1,0,0 and p = 1,1,1,0 -> TP 2 FP 1 FN 0 TN 1.
    #   F1 = 2*2 / (4+1+0) = 0.8; MCC = (2*1 - 1*0) / sqrt(3*2*2*1) = 2/sqrt(12).
    # Class 1: y = 0,1,1,1 and p = 0,0,1,1 -> TP 2 FP 0 FN 1 TN 1.
    #   F1 = 4 / (4+0+1) = 0.8; MCC = (2*1 - 0*1) / sqrt(2*3*1*2) = 2/sqrt(12).
    # Jaccard rows: {0}/{0} = 1, {0}/{0,1} = 0.5, {0,1}/{1} = 0.5, {1}/{1} = 1 -> mean 0.75.
    y, p = _toy(models)
    report = models.metrics_report(y, p, models._np().arange(4), ["a", "b"], models.Bootstrap(0, 1))
    assert report["macro_f1"]["value"] == pytest.approx(0.8)
    assert report["jaccard"]["value"] == pytest.approx(0.75)
    assert report["per_class"]["a"]["mcc"]["value"] == pytest.approx(2 / math.sqrt(12))
    assert report["per_class"]["b"]["mcc"]["value"] == pytest.approx(2 / math.sqrt(12))
    assert report["per_class"]["a"]["support"] == 2
    assert report["per_class"]["b"]["support"] == 3
    assert report["macro_f1"]["ci_low"] is None


def test_a_class_with_no_predictions_has_zero_f1_and_mcc(models: Any) -> None:
    np = models._np()
    y = np.array([[1], [0]])
    p = np.array([[0], [0]])
    report = models.metrics_report(y, p, np.arange(2), ["a"], models.Bootstrap(0, 1))
    assert report["per_class"]["a"]["f1"]["value"] == 0.0
    assert report["per_class"]["a"]["mcc"]["value"] == 0.0


def test_two_empty_sets_have_jaccard_one(models: Any) -> None:
    np = models._np()
    empty = np.zeros((2, 2), dtype=int)
    assert list(models.sample_jaccard(empty, empty)) == [1.0, 1.0]


def test_the_bootstrap_is_deterministic_and_resamples_clusters(models: Any) -> None:
    np = models._np()
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=(60, 2))
    p = rng.integers(0, 2, size=(60, 2))
    cluster = np.repeat(np.arange(20), 3)
    first = models.metrics_report(y, p, cluster, ["a", "b"], models.Bootstrap(200, 5))
    again = models.metrics_report(y, p, cluster, ["a", "b"], models.Bootstrap(200, 5))
    assert first == again
    low, high = first["macro_f1"]["ci_low"], first["macro_f1"]["ci_high"]
    assert low <= high
    assert low >= 0.0
    assert high <= 1.0


# -- thresholds ------------------------------------------------------------------------
def test_the_best_threshold_is_a_score_found_by_hand(models: Any) -> None:
    np = models._np()
    # Cutting at 0.8 calls the two positives and nothing else: MCC 1. Any other cut is worse.
    assert models.best_threshold(np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 1, 0, 0])) == 0.8
    # Tied scores cannot be split: cut at 0.5 gives TP 1 FP 1 (MCC 0.5), at 0.2 TP 1 FP 2 (MCC 0).
    assert models.best_threshold(np.array([0.5, 0.5, 0.2]), np.array([1, 0, 0])) == 0.5
    # No positives: every MCC is 0, so the highest (most conservative) threshold is kept.
    assert models.best_threshold(np.array([0.9, 0.4]), np.array([0, 0])) == 0.9


def test_a_fold_is_scored_with_thresholds_that_never_saw_its_labels(models: Any) -> None:
    np = models._np()
    rng = np.random.default_rng(11)
    partition = np.repeat([1, 2, 3, 4], 25)
    scores = rng.random((100, 2))
    y = (scores + rng.normal(0, 0.3, size=scores.shape) > 0.5).astype(int)
    split = models.Split(y, partition, np.arange(100))
    boot = models.Bootstrap(0, 1)
    base = models.validate_oof(scores, split, bl.lopo_folds(), ["a", "b"], boot)
    flipped_y = y.copy()
    flipped_y[partition == 2] = 1 - flipped_y[partition == 2]
    flipped = models.validate_oof(
        scores,
        models.Split(flipped_y, partition, np.arange(100)),
        bl.lopo_folds(),
        ["a", "b"],
        boot,
    )
    assert [f["rows"] for f in base["folds"]] == [25, 25, 25, 25]
    # Flipping fold 2's labels cannot change what fold 2's own thresholds were tuned on...
    held = models.tune_thresholds(scores[partition != 2], y[partition != 2])
    assert held == models.tune_thresholds(scores[partition != 2], flipped_y[partition != 2])
    # ...but it does change the frozen thresholds, which are tuned on every validation row.
    assert base["frozen_thresholds"] != flipped["frozen_thresholds"]


# -- the table and the models ----------------------------------------------------------
def test_the_table_masks_unknown_rows_and_the_test_partition(models: Any) -> None:
    raw = models.RawTable(
        accessions=["A", "B", "C", "D"],
        partitions=[1, 1, 2, 5],
        clusters=["c1", "c1", "c2", "c3"],
        seq_index=[0, 0, 1, 2],
        x_rows=[[0.0], [1.0], [2.0]],
        rules={"t": [[0.1], [0.2], [0.3]]},
        labels={"t": [[1], None, [0], [1]]},
        widths={"t": 1},
    )
    table = models.assemble(raw)
    assert list(table["M_t"]) == [True, False, True, True]
    assert list(models.task_rows(table, "t", (1, 2, 3, 4), 1)) == [0, 2]
    assert list(models.task_rows(table, "t", (5,), 1)) == [3]
    assert list(table["cluster_code"][:2]) == [0, 0]


@pytest.mark.parametrize("model", bl.MODEL_NAMES)
def test_every_model_fits_and_scores_one_column_per_class(models: Any, model: str) -> None:
    pytest.importorskip("sklearn")
    np = models._np()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 4))
    y = np.stack([(x[:, 0] > 0), (x[:, 1] > 0.3)], axis=1).astype(int)
    params = dict(next(bl.grid_items(model))[1])
    fitted = models.fit_model(model, params, x, y, models.Runtime(7, 1))
    assert models.model_scores(fitted, x).shape == (80, 2)
    again = models.fit_model(model, params, x, y, models.Runtime(7, 1))
    assert (models.model_scores(fitted, x) == models.model_scores(again, x)).all()


def test_the_scaler_is_fit_on_the_training_rows_only(models: Any) -> None:
    pytest.importorskip("sklearn")
    np = models._np()
    train = np.array([[0.0], [2.0], [4.0], [6.0]])
    labels = np.array([[0], [1], [0], [1]])
    fitted = models.fit_model("logreg", {"C": 1.0}, train, labels, models.Runtime(1, 1))
    assert fitted["scaler"].mean_[0] == pytest.approx(3.0)


# -- the manifest ----------------------------------------------------------------------
REGISTRY = "a" * 64
CORPUS = "b" * 64
PIN = "1.2.3"


def _entry(classes: int) -> dict[str, Any]:
    return {"thresholds": [0.25] * classes, "validation": {"pooled": {}}}


def _manifest() -> dict[str, Any]:
    tasks = {
        task: {
            "classes": ["x", "y"],
            "models": {m: _entry(2) for m in bl.MODEL_NAMES},
            "rules": _entry(2),
        }
        for task in bl.TASKS
    }
    frozen = {
        "protocol": {
            "validation_partitions": [1, 2, 3, 4],
            "test_partition": 5,
        },
        "inputs": {
            "registry_sha256": REGISTRY,
            "corpus_sha256": CORPUS,
            "feature_table_sha256": "c" * 64,
            "engine_sha256": "d" * 64,
            "class_map_sha256": "e" * 64,
        },
        "software": {"scikit_learn": PIN},
        "tasks": tasks,
    }
    return {
        "schema_version": bl.SCHEMA_VERSION,
        "frozen": frozen,
        "frozen_sha256": bl.digest(frozen),
        "test": None,
    }


def _problems(manifest: dict[str, Any]) -> list[str]:
    return bl.check_manifest(
        manifest, registry_sha256=REGISTRY, corpus_sha256=CORPUS, sklearn_pin=PIN
    )


def _scored(manifest: dict[str, Any]) -> dict[str, Any]:
    tasks = {task: {m: {} for m in (*bl.MODEL_NAMES, bl.RULES_NAME)} for task in bl.TASKS}
    manifest["test"] = {"touches": 1, "frozen_sha256": manifest["frozen_sha256"], "tasks": tasks}
    return manifest


def test_a_sound_manifest_has_no_problems_before_and_after_the_test() -> None:
    assert _problems(_manifest()) == []
    assert _problems(_scored(_manifest())) == []


def test_editing_a_frozen_value_breaks_the_digest() -> None:
    manifest = _manifest()
    manifest["frozen"]["tasks"]["localization"]["models"]["logreg"]["thresholds"][0] = 0.9
    assert any("frozen_sha256" in p for p in _problems(manifest))


def test_the_test_partition_cannot_be_scored_twice_or_against_another_freeze() -> None:
    twice = _scored(_manifest())
    twice["test"]["touches"] = 2
    assert any("exactly once" in p for p in _problems(twice))
    other = _scored(_manifest())
    other["test"]["frozen_sha256"] = "0" * 64
    assert any("different frozen block" in p for p in _problems(other))
    partial = _scored(_manifest())
    del partial["test"]["tasks"]["localization"]["logreg"]
    assert any("do not cover" in p for p in _problems(partial))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("protocol", "validation_partitions"), [0, 1, 2, 3], "validation partitions"),
        (("protocol", "test_partition"), 4, "test partition"),
        (("inputs", "registry_sha256"), "f" * 64, "registry hash"),
        (("inputs", "corpus_sha256"), "f" * 64, "corpus hash"),
        (("software", "scikit_learn"), "9.9.9", "scikit-learn"),
        (("inputs", "engine_sha256"), "short", "engine_sha256"),
    ],
)
def test_each_pin_disagreement_is_reported(path: tuple[str, str], value: Any, message: str) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["frozen"][path[0]][path[1]] = value
    manifest["frozen_sha256"] = bl.digest(manifest["frozen"])
    assert any(message in p for p in _problems(manifest))


def test_a_missing_threshold_is_reported() -> None:
    manifest = _manifest()
    manifest["frozen"]["tasks"]["signal_peptide"]["rules"]["thresholds"] = [0.5]
    manifest["frozen_sha256"] = bl.digest(manifest["frozen"])
    assert any("one threshold per class" in p for p in _problems(manifest))


def test_the_committed_manifest_agrees_with_the_repository() -> None:
    if not MANIFEST.exists():
        pytest.skip("data/manifests/baseline_metrics.json is written by --freeze")
    split = bl.read_manifest(SPLIT_MANIFEST)
    problems = bl.check_manifest(
        bl.read_manifest(MANIFEST),
        registry_sha256=split["registry"]["sha256"],
        corpus_sha256=split["corpus_sha256"],
        sklearn_pin=bl.pinned_sklearn(PYPROJECT.read_text(encoding="utf-8")),
    )
    assert problems == []
