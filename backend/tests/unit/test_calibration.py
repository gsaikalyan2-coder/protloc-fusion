"""Phase 10 calibration: ECE and Brier, calibrators, eligibility, cross-fitting, manifest, bundle.

Pure-Python tests run everywhere. Tests that need numpy or scikit-learn skip themselves when the
package is absent, so CI without the ``baselines`` extra still runs the rest. Nothing here reads
data bytes or the test partition.
"""

import ast
from itertools import pairwise
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from protloc.calibration import calibrator as cb
from protloc.data.manifest import canonical_json_bytes as manifest_canonical_json_bytes
from protloc.registry import bundle as registry_bundle
from protloc.training import baseline_models as bm
from protloc.training import calibration as cal
from protloc.training.baselines import (
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    digest,
    lopo_folds,
)

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "training" / "calibration.yaml"
MODULE = REPO / "backend" / "protloc" / "training" / "calibration.py"
SHA = "a" * 64
SHA_B = "b" * 64
ECE_TWO = cal.EceSettings(bins=2, min_bin_rows=1)


@pytest.fixture
def np() -> Any:
    """numpy, or skip the test."""
    return pytest.importorskip("numpy")


@pytest.fixture
def sklearn() -> Any:
    """scikit-learn (and numpy), or skip the test."""
    pytest.importorskip("numpy")
    return pytest.importorskip("sklearn")


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


# -- ECE and Brier, computed by hand ---------------------------------------------------
def test_brier_by_hand() -> None:
    # (0.1^2 + 0.2^2 + 0.4^2 + 0.6^2) / 4 = 0.57 / 4
    assert cal.brier([0.9, 0.2, 0.6, 0.4], [1, 0, 1, 1]) == pytest.approx(0.1425)


def test_brier_of_a_perfect_and_a_wrong_forecast() -> None:
    assert cal.brier([1.0, 0.0], [1, 0]) == 0.0
    assert cal.brier([0.0, 1.0], [1, 0]) == 1.0


def test_ece_by_hand_two_equal_mass_bins() -> None:
    # bin A: scores 0.1, 0.2 (mean 0.15), outcomes 0, 1 (mean 0.5): gap 0.35, weight 1/2
    # bin B: scores 0.8, 0.9 (mean 0.85), outcomes 1, 1 (mean 1.0): gap 0.15, weight 1/2
    value = cal.ece([0.1, 0.2, 0.8, 0.9], [0, 1, 1, 1], ECE_TWO)
    assert value == pytest.approx(0.25)


def test_ece_of_a_perfectly_calibrated_bin_is_zero() -> None:
    assert cal.ece([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], ECE_TWO) == pytest.approx(0.0)


def test_tied_scores_are_never_split_across_bins() -> None:
    members = cal.bin_members([0.5, 0.5, 0.5, 0.5, 0.9, 0.9], cal.EceSettings(3, 1))
    assert sorted(len(m) for m in members) == [2, 4]
    assert sorted(i for m in members for i in m) == list(range(6))


def test_bin_count_is_lowered_so_each_bin_has_enough_rows() -> None:
    scores = [i / 40 for i in range(40)]
    assert len(cal.bin_members(scores, cal.EceSettings(10, 20))) == 2
    assert len(cal.bin_members(scores, cal.EceSettings(10, 4))) == 10
    assert len(cal.bin_members(scores[:5], cal.EceSettings(10, 20))) == 1


def test_reliability_bins_report_rows_mean_score_and_fraction_positive() -> None:
    bins = cal.reliability([0.1, 0.2, 0.8, 0.9], [0, 1, 1, 1], ECE_TWO)
    assert [b["rows"] for b in bins] == [2, 2]
    assert bins[0]["mean_score"] == pytest.approx(0.15)
    assert bins[1]["fraction_positive"] == pytest.approx(1.0)


def test_empty_input_is_an_error_not_a_zero() -> None:
    with pytest.raises(BaselineError):
        cal.brier([], [])
    with pytest.raises(BaselineError):
        cal.ece([], [], ECE_TWO)


# -- calibrators -----------------------------------------------------------------------
def _isotonic(label: str = "x") -> cb.LabelCalibrator:
    return cb.LabelCalibrator(
        label, cb.ISOTONIC, 30, 70, 0.4, knots_x=(0.1, 0.5, 0.9), knots_y=(0.0, 0.5, 1.0)
    )


def test_isotonic_apply_interpolates_and_clips() -> None:
    c = _isotonic()
    assert c.apply(0.0) == 0.0
    assert c.apply(1.0) == 1.0
    assert c.apply(0.3) == pytest.approx(0.25)
    assert c.apply(0.7) == pytest.approx(0.75)
    assert c.apply(0.5) == pytest.approx(0.5)


def test_uncalibrated_is_the_identity_and_says_so() -> None:
    c = cb.LabelCalibrator("x", cb.UNCALIBRATED, 2, 100, 0.7)
    assert c.apply(0.123) == 0.123
    assert c.calibrated is False
    assert _isotonic().calibrated is True


def test_platt_apply_matches_the_formula() -> None:
    c = cb.LabelCalibrator("x", cb.PLATT, 8, 40, 0.5, slope=2.0, intercept=-0.5, logit_eps=1e-6)
    z = math.log(0.8 / 0.2)
    assert c.apply(0.8) == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * z - 0.5))))
    assert 0.0 < c.apply(0.0) < c.apply(1.0) < 1.0


def test_json_round_trip_without_pickle() -> None:
    c = _isotonic("cytoplasm")
    text = json.dumps(c.to_dict())
    assert cb.label_from_dict(json.loads(text)) == c
    assert "pickle" not in text.lower()


def _set(labels: tuple[cb.LabelCalibrator, ...], weights: str = SHA) -> cb.CalibratorSet:
    return cb.CalibratorSet(
        task="localization",
        calibration_version="0.10.0+test",
        classes=tuple(c.label for c in labels),
        head_weights_sha256=weights,
        head_frozen_sha256=SHA_B,
        threshold_rule="rule",
        labels=labels,
    )


def test_set_round_trip_and_decide_uses_each_labels_own_threshold() -> None:
    labels = (
        _isotonic("a"),
        cb.LabelCalibrator("b", cb.UNCALIBRATED, 3, 90, 0.9),
    )
    original = _set(labels)
    rebuilt = cb.set_from_dict(json.loads(original.to_bytes()))
    assert rebuilt == original
    assert original.decide([0.7, 0.8]) == [True, False]
    assert original.decide([0.2, 0.95]) == [False, True]
    with pytest.raises(cb.CalibratorError):
        original.calibrate([0.1])


def test_set_from_dict_refuses_unsound_calibrators() -> None:
    good = _set((_isotonic("a"),)).to_dict()
    bad_knots = json.loads(json.dumps(good))
    bad_knots["labels"][0]["knots_y"] = [0.0, 0.9, 0.5]
    with pytest.raises(cb.CalibratorError, match="monotone"):
        cb.set_from_dict(bad_knots)
    nan_threshold = json.loads(json.dumps(good))
    nan_threshold["labels"][0]["threshold"] = None
    with pytest.raises(cb.CalibratorError):
        cb.set_from_dict(nan_threshold)
    wrong_schema = {**good, "schema_version": 99}
    with pytest.raises(cb.CalibratorError, match="schema_version"):
        cb.set_from_dict(wrong_schema)


def test_canonical_json_equals_the_manifest_helper() -> None:
    obj = {"b": [1, 2.5, "x"], "a": {"z": None, "y": True}}
    assert cb.canonical_json_bytes(obj) == manifest_canonical_json_bytes(obj)


def test_isotonic_is_monotone_and_matches_scikit_learn(sklearn: Any, np: Any) -> None:
    rng = np.random.default_rng(3)
    scores = [float(v) for v in rng.uniform(0.0, 1.0, size=200)]
    y = [int(rng.uniform() < s * s) for s in scores]
    xs, ys = cal.fit_isotonic(scores, y)
    assert all(b > a for a, b in pairwise(xs))
    assert all(b >= a for a, b in pairwise(ys))
    calibrator = cb.LabelCalibrator("x", cb.ISOTONIC, sum(y), len(y) - sum(y), 0.5, xs, ys)
    grid = [i / 100 for i in range(101)]
    values = [calibrator.apply(g) for g in grid]
    assert all(b >= a for a, b in pairwise(values))
    model = sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(scores, y)
    assert values == pytest.approx([float(v) for v in model.predict(grid)], abs=1e-9)
    assert cb.label_from_dict(json.loads(json.dumps(calibrator.to_dict()))) == calibrator


def test_platt_recovers_a_monotone_map_and_beats_the_identity(np: Any) -> None:
    rng = np.random.default_rng(5)
    scores = [float(v) for v in rng.uniform(0.02, 0.98, size=400)]
    y = [int(rng.uniform() < s**3) for s in scores]
    settings = cal.load_config(CONFIG).platt
    slope, intercept = cal.fit_platt(scores, y, settings)
    c = cb.LabelCalibrator(
        "x",
        cb.PLATT,
        sum(y),
        len(y) - sum(y),
        0.5,
        slope=slope,
        intercept=intercept,
        logit_eps=settings.logit_eps,
    )
    assert slope > 0.0
    after = [c.apply(s) for s in scores]
    assert cal.brier(after, y) < cal.brier(scores, y)


def test_platt_stays_finite_when_the_classes_separate() -> None:
    settings = cal.load_config(CONFIG).platt
    scores = [0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    slope, intercept = cal.fit_platt(scores, y, settings)
    assert math.isfinite(slope)
    assert math.isfinite(intercept)
    assert slope > 0.0


# -- eligibility and fallback ----------------------------------------------------------
def test_eligibility_boundaries_come_from_the_config() -> None:
    rule = cal.load_config(CONFIG).eligibility
    iso_p, iso_n = rule.isotonic_min_positives, rule.isotonic_min_negatives
    fb_p, fb_n = rule.fallback_min_positives, rule.fallback_min_negatives
    assert cal.choose_method(iso_p, iso_n, rule) == cb.ISOTONIC
    assert cal.choose_method(iso_p - 1, iso_n, rule) == cb.PLATT
    assert cal.choose_method(iso_p, iso_n - 1, rule) == cb.PLATT
    assert cal.choose_method(fb_p, fb_n, rule) == cb.PLATT
    assert cal.choose_method(fb_p - 1, 1000, rule) == cb.UNCALIBRATED
    assert cal.choose_method(1000, fb_n - 1, rule) == cb.UNCALIBRATED


def test_a_label_below_every_minimum_is_marked_not_dropped(sklearn: Any) -> None:
    config = cal.load_config(CONFIG)
    scores = [0.1 * (i % 10) for i in range(60)]
    y = [1 if i < 2 else 0 for i in range(60)]
    fitted = cal.fit_label("peroxisome", scores, y, config)
    assert fitted.method == cb.UNCALIBRATED
    assert fitted.label == "peroxisome"
    assert (fitted.positives, fitted.negatives) == (2, 58)
    assert fitted.apply(0.3) == 0.3


def test_a_forced_method_is_respected_unless_a_class_is_missing(sklearn: Any) -> None:
    config = cal.load_config(CONFIG)
    scores = [i / 40 for i in range(40)]
    y = [i % 2 for i in range(40)]
    assert cal.fit_label("x", scores, y, config, forced=cb.PLATT).method == cb.PLATT
    assert cal.fit_label("x", scores, y, config, forced=cb.ISOTONIC).method == cb.ISOTONIC
    assert (
        cal.fit_label("x", scores, [0] * 40, config, forced=cb.ISOTONIC).method == cb.UNCALIBRATED
    )


# -- cross-fitting ---------------------------------------------------------------------
def _synthetic(np: Any, rows_per_partition: int = 80) -> cal.CalibrationData:
    """Four partitions, 3 labels (common, mid, rare), signal in the score, unique clusters."""
    rng = np.random.default_rng(11)
    n = rows_per_partition * 4
    partition = np.repeat(np.array(VALIDATION_PARTITIONS), rows_per_partition)
    base = np.array([0.5, 0.25, 0.04])
    y = (rng.uniform(size=(n, 3)) < base).astype(int)
    scores = np.clip(0.15 + 0.6 * y + rng.normal(0.0, 0.18, size=(n, 3)), 0.01, 0.99)
    return cal.CalibrationData(
        scores=scores,
        y=y,
        partition=partition,
        cluster_code=np.arange(n),
        classes=("common", "mid", "rare"),
    )


def test_crossfit_never_lets_a_folds_labels_into_its_own_calibrator_or_thresholds(
    sklearn: Any, np: Any
) -> None:
    config = cal.load_config(CONFIG)
    data = _synthetic(np)
    folds = lopo_folds()
    before = cal.crossfit(data, folds, config, held_out=set())
    flipped = data.y.copy()
    target = data.partition == 2
    flipped[target] = 1 - flipped[target]
    after = cal.crossfit(
        cal.CalibrationData(data.scores, flipped, data.partition, data.cluster_code, data.classes),
        folds,
        config,
        held_out=set(),
    )
    assert np.array_equal(before.calibrated[target], after.calibrated[target])
    assert before.fold_thresholds[2] == after.fold_thresholds[2]
    # the other folds and the frozen fit do depend on partition 2's labels
    assert not np.array_equal(before.calibrated[~target], after.calibrated[~target])
    assert before.frozen_thresholds != after.frozen_thresholds


def test_frozen_thresholds_are_tuned_on_the_cross_fitted_calibrated_scores(
    sklearn: Any, np: Any
) -> None:
    config = cal.load_config(CONFIG)
    data = _synthetic(np)
    fitted = cal.crossfit(data, lopo_folds(), config, held_out=set())
    assert fitted.frozen_thresholds == bm.tune_thresholds(fitted.calibrated, data.y)
    assert [c.threshold for c in fitted.frozen] == fitted.frozen_thresholds
    assert all(math.isfinite(t) for t in fitted.frozen_thresholds)
    for j, threshold in enumerate(fitted.frozen_thresholds):
        assert threshold in set(fitted.calibrated[:, j].tolist())


def test_methods_follow_the_counts_and_the_report_flags_inconsistency(
    sklearn: Any, np: Any
) -> None:
    config = cal.load_config(CONFIG)
    fitted = cal.crossfit(_synthetic(np), lopo_folds(), config, held_out=set())
    methods = dict(zip(("common", "mid", "rare"), fitted.methods["frozen"], strict=True))
    assert methods["common"] == cb.ISOTONIC
    assert methods["rare"] in {cb.PLATT, cb.UNCALIBRATED}
    assert set(fitted.methods["consistent"]) == {"common", "mid", "rare"}
    assert set(fitted.methods["fold"]) == {"1", "2", "3", "4"}
    assert set(fitted.methods["inner"]["1"]) == {"2", "3", "4"}


def test_leakage_assert_fires_before_any_fit(sklearn: Any, np: Any) -> None:
    config = cal.load_config(CONFIG)
    data = _synthetic(np)
    shared = {int(data.cluster_code[0])}
    with pytest.raises(BaselineError, match="cluster leakage"):
        cal.crossfit(data, lopo_folds(), config, held_out=shared)
    clash = data.cluster_code.copy()
    clash[data.partition == 1] = 7
    clash[data.partition == 2] = 7
    broken = cal.CalibrationData(data.scores, data.y, data.partition, clash, data.classes)
    with pytest.raises(BaselineError, match="cluster leakage"):
        cal.crossfit(broken, lopo_folds(), config, held_out=set())


# -- decisions and the undefined-label rule --------------------------------------------
def test_a_label_with_one_positive_in_a_fold_is_undefined_not_dropped(
    sklearn: Any, np: Any
) -> None:
    config = cal.load_config(CONFIG)
    data = _synthetic(np)
    y = data.y.copy()
    y[:, 2] = 0
    y[(data.partition == 3).nonzero()[0][0], 2] = 1
    sparse = cal.CalibrationData(data.scores, y, data.partition, data.cluster_code, data.classes)
    folds = lopo_folds()
    fitted = cal.crossfit(sparse, folds, config, held_out=set())
    report = cal.validate_calibrated(fitted, sparse, folds, bm.Bootstrap(0, 0))
    fold3 = next(f for f in report["folds"] if f["validate"] == 3)
    rare = fold3["per_class"]["rare"]
    assert rare["undefined"] is True
    assert rare["mcc"] == 0.0
    assert rare["support"] == 1
    assert set(fold3["per_class"]) == {"common", "mid", "rare"}


def test_decisions_use_each_labels_own_threshold_and_never_a_literal(sklearn: Any, np: Any) -> None:
    config = cal.load_config(CONFIG)
    data = _synthetic(np)
    folds = lopo_folds()
    fitted = cal.crossfit(data, folds, config, held_out=set())
    report = cal.validate_calibrated(fitted, data, folds, bm.Bootstrap(20, 1))
    assert report["frozen_thresholds"] == fitted.frozen_thresholds
    assert set(report["fold_thresholds"]) == {"1", "2", "3", "4"}
    assert report["pooled"]["rows"] == len(data.y)
    assert 0.0 <= report["pooled"]["macro_f1"]["value"] <= 1.0


def test_calibration_report_structure_and_determinism(sklearn: Any, np: Any) -> None:
    config = cal.load_config(CONFIG)
    data = _synthetic(np)
    fitted = cal.crossfit(data, lopo_folds(), config, held_out=set())
    rows = cal.ScoredRows(data.scores, fitted.calibrated, data.y, data.cluster_code, data.classes)
    first = cal.calibration_report(rows, config.ece, bm.Bootstrap(30, config.seed))
    second = cal.calibration_report(rows, config.ece, bm.Bootstrap(30, config.seed))
    assert first == second
    entry = first["per_label"]["common"]["ece_calibrated"]
    assert entry["ci_low"] <= entry["value"] <= entry["ci_high"] or entry["ci_low"] is not None
    assert set(first["per_label"]["mid"]) == {
        "ece_raw",
        "ece_calibrated",
        "ece_delta",
        "brier_raw",
        "brier_calibrated",
        "brier_delta",
    }
    point = cal.calibration_report(rows, config.ece, bm.Bootstrap(0, 0))
    assert point["macro"]["ece_raw"]["ci_low"] is None
    assert point["per_label"]["common"]["ece_raw"]["value"] == pytest.approx(
        cal.ece([float(v) for v in data.scores[:, 0]], [int(v) for v in data.y[:, 0]], config.ece)
    )
    assert first["reliability"]["rare"]["raw"]
    assert first["bootstrap"] == {"samples": 30, "seed": config.seed, "unit": "cluster_id"}


# -- configuration agrees with code ----------------------------------------------------
def test_config_agrees_with_the_root_and_evaluation_configs() -> None:
    config = cal.load_config(CONFIG)
    assert config.seed == _yaml(REPO / "configs" / "config.yaml")["seed"]
    evaluation = _yaml(REPO / "configs" / "evaluation" / "default.yaml")
    assert evaluation["reporting"]["bootstrap_samples"] == 1000
    assert evaluation["partitions"]["test"] == [TEST_PARTITION]
    assert _yaml(CONFIG)["test_partition"]["policy"] == cal.TEST_POLICY


def test_config_values_are_not_repeated_as_literals_in_the_module() -> None:
    config = cal.load_config(CONFIG)
    doc = _yaml(CONFIG)
    forbidden = {
        0.5,
        config.eligibility.isotonic_min_positives,
        config.eligibility.isotonic_min_negatives,
        config.eligibility.fallback_min_positives,
        config.eligibility.fallback_min_negatives,
        config.ece.bins,
        config.ece.min_bin_rows,
        config.platt.logit_eps,
        config.platt.max_iter,
        config.platt.max_halvings,
        config.platt.tolerance,
        config.seed,
    }
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    # Module-level names (PHASE, SCHEMA_VERSION, ...) are labels, not settings; the settings
    # could only leak in through the bodies of functions.
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    literals = {
        node.value
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    assert not (literals & forbidden), sorted(literals & forbidden)
    assert doc["eligibility"]["fallback"] == cal.FALLBACK


def test_config_rejects_a_fallback_above_the_isotonic_minimum(tmp_path: Path) -> None:
    doc = _yaml(CONFIG)
    doc["eligibility"]["fallback_min_positives"] = doc["eligibility"]["isotonic_min_positives"] + 1
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(BaselineError, match="fallback"):
        cal.load_config(path)


def test_config_rejects_a_touched_test_partition_policy(tmp_path: Path) -> None:
    doc = _yaml(CONFIG)
    doc["test_partition"]["policy"] = "scored"
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(BaselineError, match="policy"):
        cal.load_config(path)


# -- the manifest ----------------------------------------------------------------------
CLASSES = ("a", "b")


def _pins() -> cal.Pins:
    return cal.Pins(
        head_frozen_sha256=SHA,
        head_weights_sha256=SHA_B,
        registry_sha256="c" * 64,
        corpus_sha256="d" * 64,
        embedding_cache_key="e" * 64,
        residue_cache_key="f" * 64,
        class_map_sha256="1" * 64,
        config_sha256="2" * 64,
        classes=CLASSES,
    )


def _manifest() -> dict[str, Any]:
    pins = _pins()
    frozen: dict[str, Any] = {
        "protocol": {
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
            "test_partition_policy": cal.TEST_POLICY,
        },
        "inputs": {
            "head_frozen_sha256": pins.head_frozen_sha256,
            "head_weights_sha256": pins.head_weights_sha256,
            "registry_sha256": pins.registry_sha256,
            "corpus_sha256": pins.corpus_sha256,
            "embedding_cache_key": pins.embedding_cache_key,
            "residue_cache_key": pins.residue_cache_key,
            "class_map_sha256": pins.class_map_sha256,
            "calibration_config_sha256": pins.config_sha256,
            "oof_sha256": "3" * 64,
            "feature_table_sha256": "4" * 64,
            "calibrator_sha256": "5" * 64,
        },
        "classes": list(CLASSES),
        "labels": [
            {"label": "a", "method": cb.ISOTONIC, "threshold": 0.4},
            {"label": "b", "method": cb.UNCALIBRATED, "threshold": 0.9},
        ],
        "validation": {"calibration": {}, "raw": {}, "calibrated": {}},
    }
    return {
        "schema_version": cal.SCHEMA_VERSION,
        "phase": cal.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }


def test_a_sound_manifest_has_no_problems() -> None:
    assert cal.check_manifest(_manifest(), _pins()) == []


def test_a_tampered_frozen_block_is_detected() -> None:
    manifest = _manifest()
    manifest["frozen"]["labels"][0]["threshold"] = 0.41
    assert any("frozen_sha256" in p for p in cal.check_manifest(manifest, _pins()))


def test_a_test_block_is_refused_because_phase_10_never_reads_partition_5() -> None:
    manifest = _manifest()
    manifest["test"] = {"touches": 2}
    problems = cal.check_manifest(manifest, _pins())
    assert any("untouched" in p for p in problems)


def test_a_policy_other_than_untouched_is_refused() -> None:
    manifest = _manifest()
    manifest["frozen"]["protocol"]["test_partition_policy"] = "scored"
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    assert any("policy" in p for p in cal.check_manifest(manifest, _pins()))


def test_inputs_that_differ_from_the_repository_are_refused() -> None:
    manifest = _manifest()
    manifest["frozen"]["inputs"]["head_frozen_sha256"] = SHA_B
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    assert any("head_frozen_sha256" in p for p in cal.check_manifest(manifest, _pins()))


def test_a_non_finite_or_unknown_label_entry_is_refused() -> None:
    manifest = _manifest()
    manifest["frozen"]["labels"][1]["threshold"] = None
    manifest["frozen"]["labels"][0]["method"] = "temperature"
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    problems = cal.check_manifest(manifest, _pins())
    assert any("threshold" in p for p in problems)
    assert any("method" in p for p in problems)


# -- the registry bundle ---------------------------------------------------------------
def _write_bundle(
    tmp_path: Path, calibrators: cb.CalibratorSet | None = None
) -> tuple[Path, dict[str, Path]]:
    rules = {"baseline_class_map": tmp_path / "map.yaml", "consistency": tmp_path / "c.json"}
    rules["baseline_class_map"].write_text("classes: [a]\n", encoding="utf-8")
    rules["consistency"].write_text("{}\n", encoding="utf-8")
    labels = (cb.LabelCalibrator("a", cb.UNCALIBRATED, 5, 50, 0.6),)
    fitted = _set(labels, weights=SHA)
    document = registry_bundle.build_document(
        version="0.10.0",
        head={
            "weights_file": "w.npz",
            "weights_sha256": SHA,
            "frozen_sha256": SHA_B,
            "head_config_sha256": "6" * 64,
            "classes": ["a"],
        },
        calibrator=fitted,
        rules={f"{key}_sha256": cb.sha256_hex(path.read_bytes()) for key, path in rules.items()},
        scope={"scope": "subset", "scope_label": "label"},
    )
    directory = tmp_path / "registry" / "localization" / "0.10.0"
    directory.mkdir(parents=True)
    (directory / registry_bundle.BUNDLE_FILE).write_bytes(cb.canonical_json_bytes(document))
    shipped = calibrators or fitted
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(shipped.to_bytes())
    return directory, rules


def test_a_consistent_bundle_loads(tmp_path: Path) -> None:
    directory, rules = _write_bundle(tmp_path)
    loaded = registry_bundle.load_bundle(directory, rules)
    assert loaded.version == "0.10.0"
    assert loaded.calibrators.classes == ("a",)


def test_the_weights_file_is_hashed_when_it_is_supplied(tmp_path: Path) -> None:
    directory, rules = _write_bundle(tmp_path)
    weights = tmp_path / "w.npz"
    weights.write_bytes(b"not the head")
    with pytest.raises(registry_bundle.BundleError, match="weights file"):
        registry_bundle.load_bundle(directory, rules, weights)


def test_a_calibrator_fitted_on_other_head_weights_is_refused(tmp_path: Path) -> None:
    other = _set((cb.LabelCalibrator("a", cb.UNCALIBRATED, 5, 50, 0.6),), weights=SHA_B)
    directory, rules = _write_bundle(tmp_path, calibrators=other)
    with pytest.raises(registry_bundle.BundleError, match="different head weights"):
        registry_bundle.load_bundle(directory, rules)


def test_a_tampered_calibrator_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_bundle(tmp_path)
    path = directory / registry_bundle.CALIBRATOR_FILE
    doc = json.loads(path.read_bytes())
    doc["labels"][0]["threshold"] = 0.01
    path.write_bytes(cb.canonical_json_bytes(doc))
    with pytest.raises(registry_bundle.BundleError, match="SHA-256"):
        registry_bundle.load_bundle(directory, rules)


def test_a_changed_rule_file_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_bundle(tmp_path)
    rules["consistency"].write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(registry_bundle.BundleError, match="consistency"):
        registry_bundle.load_bundle(directory, rules)


def test_a_tampered_bundle_document_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_bundle(tmp_path)
    path = directory / registry_bundle.BUNDLE_FILE
    doc = json.loads(path.read_bytes())
    doc["bundle_version"] = "9.9.9"
    path.write_bytes(cb.canonical_json_bytes(doc))
    with pytest.raises(registry_bundle.BundleError, match="bundle_sha256"):
        registry_bundle.load_bundle(directory, rules)


def test_a_missing_calibrator_file_is_refused(tmp_path: Path) -> None:
    directory, rules = _write_bundle(tmp_path)
    (directory / registry_bundle.CALIBRATOR_FILE).unlink()
    with pytest.raises(registry_bundle.BundleError, match="cannot be read"):
        registry_bundle.load_bundle(directory, rules)


# -- the frozen artifacts, when they exist ---------------------------------------------
def test_the_committed_bundle_directory_matches_the_config_when_present() -> None:
    config = cal.load_config(CONFIG)
    directory = REPO / config.bundle_directory / config.bundle_version
    if not directory.exists():
        pytest.skip("the bundle is created by scripts/calibrate.py --freeze")
    rules = {
        "baseline_class_map": REPO / "backend" / "rules" / "baseline_class_map.yaml",
        "consistency": REPO / "backend" / "rules" / "consistency.json",
    }
    loaded = registry_bundle.load_bundle(directory, rules)
    assert loaded.version == config.bundle_version
    assert not loaded.calibrators.problems()
