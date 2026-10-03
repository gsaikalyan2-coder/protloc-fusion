"""Phase 10 calibration of the localization head: config, fitting, metrics and the manifest.

Pure Python where possible: the calibrators, ECE, Brier, Platt scaling and the eligibility
rule need only the standard library. numpy (cluster bootstrap, array slicing) and
scikit-learn (isotonic regression) are imported lazily inside the functions that need them,
so CI can import this module and check the manifest on a runner with neither. The fitted
objects themselves are ``protloc.calibration.calibrator`` types, which the serving image can
load without either package.

The protocol (ADR-0012), on the 408 out-of-fold validation rows of Phase 9:

* Cross-fitting. Fold k is scored with a calibrator fitted on the other validation partitions'
  out-of-fold scores; the frozen calibrator is fitted on all four.
* Doubly nested thresholds. Fold k's per-label thresholds are tuned on calibrated scores of
  the other partitions whose calibrators never saw fold k either (a calibrator fitted on the
  two partitions that are neither k nor the one being scored). The frozen thresholds are tuned
  on the cross-fitted scores of all four, never on in-sample calibrated scores.
* Labels (ADR-0006): annotated rows only; a label with fewer than 2 positives in the rows being
  scored has MCC 0.0, is marked undefined and is never dropped.

``training/`` is an entry-point layer: nothing imports it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

import yaml

from protloc.calibration.calibrator import (
    ISOTONIC,
    METHODS,
    PLATT,
    UNCALIBRATED,
    LabelCalibrator,
    is_sha256,
    logit,
)
from protloc.training import baseline_models as bm
from protloc.training import head
from protloc.training.baselines import (
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    Fold,
    digest,
)

Array = Any
SCHEMA_VERSION = 1
PHASE = 10
TASK = "localization"
TEST_POLICY = "untouched"
FALLBACK = PLATT
ECE_KIND = "equal_mass"
THRESHOLD_RULE = (
    "per label, the calibrated score maximising MCC; fold k is scored with thresholds tuned on "
    "cross-fitted calibrated scores of the other validation partitions whose calibrators never "
    "saw fold k; the frozen thresholds are tuned on the cross-fitted scores of all rows"
)
CI_PERCENTILES = (2.5, 97.5)


@dataclass(frozen=True, slots=True)
class Eligibility:
    """The counts a label needs for each calibrator (configs/training/calibration.yaml)."""

    isotonic_min_positives: int
    isotonic_min_negatives: int
    fallback_min_positives: int
    fallback_min_negatives: int


@dataclass(frozen=True, slots=True)
class PlattSettings:
    """Platt scaling numerics."""

    logit_eps: float
    max_iter: int
    max_halvings: int
    tolerance: float


@dataclass(frozen=True, slots=True)
class EceSettings:
    """Equal-mass binning: at most ``bins`` bins, each with at least ``min_bin_rows`` rows."""

    bins: int
    min_bin_rows: int


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """``configs/training/calibration.yaml``, parsed and checked."""

    seed: int
    eligibility: Eligibility
    platt: PlattSettings
    ece: EceSettings
    bundle_version: str
    bundle_directory: str
    smoke_stride: int
    sha256: str


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{where} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BaselineError(f"{where} must be a positive integer")
    return value


def _positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise BaselineError(f"{where} must be a positive number")
    return float(value)


def load_config(path: Path) -> CalibrationConfig:
    """Parse the calibration configuration and reject anything malformed before a fit starts."""
    raw = path.read_bytes()
    doc = _mapping(yaml.safe_load(raw), str(path))
    if doc.get("name") != "calibration" or doc.get("task") != TASK:
        raise BaselineError("calibration.yaml must be name 'calibration' for task 'localization'")
    el = _mapping(doc.get("eligibility"), "eligibility")
    if el.get("fallback") != FALLBACK:
        raise BaselineError(f"eligibility.fallback must be '{FALLBACK}'")
    platt = _mapping(doc.get("platt"), "platt")
    ece = _mapping(doc.get("ece"), "ece")
    if ece.get("kind") != ECE_KIND:
        raise BaselineError(f"ece.kind must be '{ECE_KIND}'")
    bundle = _mapping(doc.get("bundle"), "bundle")
    if _mapping(doc.get("test_partition"), "test_partition").get("policy") != TEST_POLICY:
        raise BaselineError(f"test_partition.policy must be '{TEST_POLICY}'")
    eligibility = Eligibility(
        isotonic_min_positives=_count(el.get("isotonic_min_positives"), "isotonic_min_positives"),
        isotonic_min_negatives=_count(el.get("isotonic_min_negatives"), "isotonic_min_negatives"),
        fallback_min_positives=_count(el.get("fallback_min_positives"), "fallback_min_positives"),
        fallback_min_negatives=_count(el.get("fallback_min_negatives"), "fallback_min_negatives"),
    )
    if eligibility.fallback_min_positives > eligibility.isotonic_min_positives:
        raise BaselineError("the fallback minimum cannot exceed the isotonic minimum")
    eps = _positive(platt.get("logit_eps"), "platt.logit_eps")
    if not eps < 1.0:
        raise BaselineError("platt.logit_eps must be below 1")
    return CalibrationConfig(
        seed=_count(doc.get("seed"), "seed"),
        eligibility=eligibility,
        platt=PlattSettings(
            logit_eps=eps,
            max_iter=_count(platt.get("max_iter"), "platt.max_iter"),
            max_halvings=_count(platt.get("max_halvings"), "platt.max_halvings"),
            tolerance=_positive(platt.get("tolerance"), "platt.tolerance"),
        ),
        ece=EceSettings(
            bins=_count(ece.get("bins"), "ece.bins"),
            min_bin_rows=_count(ece.get("min_bin_rows"), "ece.min_bin_rows"),
        ),
        bundle_version=str(bundle.get("version", "")),
        bundle_directory=str(bundle.get("directory", "")),
        smoke_stride=_count(_mapping(doc.get("smoke"), "smoke").get("stride"), "smoke.stride"),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


# -- eligibility and fitting -----------------------------------------------------------
def choose_method(positives: int, negatives: int, rule: Eligibility) -> str:
    """The calibrator a label is eligible for, from the counts of the rows it is fitted on."""
    if positives >= rule.isotonic_min_positives and negatives >= rule.isotonic_min_negatives:
        return ISOTONIC
    if positives >= rule.fallback_min_positives and negatives >= rule.fallback_min_negatives:
        return FALLBACK
    return UNCALIBRATED


def _nll(a: float, b: float, z: Sequence[float], t: Sequence[float]) -> float:
    """Negative log-likelihood of Platt's smoothed targets under ``sigmoid(a z + b)``."""
    total = 0.0
    for zi, ti in zip(z, t, strict=True):
        u = a * zi + b
        total += max(u, 0.0) + math.log1p(math.exp(-abs(u))) - ti * u
    return total


def fit_platt(
    scores: Sequence[float], y: Sequence[int], settings: PlattSettings
) -> tuple[float, float]:
    """Platt scaling by Newton's method with step halving; returns (slope, intercept).

    Targets are Platt's smoothed ones, ``(N+ + 1) / (N+ + 2)`` and ``1 / (N- + 2)``, which keep
    the likelihood bounded when the two classes separate.
    """
    z = [logit(s, settings.logit_eps) for s in scores]
    positives = sum(y)
    negatives = len(y) - positives
    high = (positives + 1.0) / (positives + 2.0)
    low = 1.0 / (negatives + 2.0)
    t = [high if v else low for v in y]
    a, b = 0.0, math.log((positives + 1.0) / (negatives + 1.0))
    value = _nll(a, b, z, t)
    for _ in range(settings.max_iter):
        step = _newton_step(a, b, z, t)
        if step is None:
            break
        moved = _backtrack((a, b), step, value, (z, t), settings.max_halvings)
        if moved is None:
            break
        gain = value - moved[2]
        a, b, value = moved
        if gain < settings.tolerance:
            break
    return a, b


def _newton_step(
    a: float, b: float, z: Sequence[float], t: Sequence[float]
) -> tuple[float, float] | None:
    grad_a = grad_b = h_aa = h_ab = h_bb = 0.0
    for zi, ti in zip(z, t, strict=True):
        u = a * zi + b
        p = 1.0 / (1.0 + math.exp(-u)) if u >= 0.0 else math.exp(u) / (1.0 + math.exp(u))
        w = p * (1.0 - p)
        grad_a += (p - ti) * zi
        grad_b += p - ti
        h_aa += w * zi * zi
        h_ab += w * zi
        h_bb += w
    det = h_aa * h_bb - h_ab * h_ab
    if det <= 0.0:
        return None
    return (h_bb * grad_a - h_ab * grad_b) / det, (h_aa * grad_b - h_ab * grad_a) / det


def _backtrack(
    origin: tuple[float, float],
    step: tuple[float, float],
    value: float,
    data: tuple[Sequence[float], Sequence[float]],
    halvings: int,
) -> tuple[float, float, float] | None:
    scale = 1.0
    for _ in range(halvings):
        na, nb = origin[0] - scale * step[0], origin[1] - scale * step[1]
        nv = _nll(na, nb, data[0], data[1])
        if nv <= value:
            return na, nb, nv
        scale /= 2.0
    return None


def fit_isotonic(
    scores: Sequence[float], y: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Isotonic regression (scikit-learn) reduced to its fitted knots, clipped to [0, 1]."""
    isotonic = importlib.import_module("sklearn.isotonic")
    model = isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(list(scores), list(y))
    return (
        tuple(float(v) for v in model.X_thresholds_),
        tuple(float(v) for v in model.y_thresholds_),
    )


def fit_label(
    label: str,
    scores: Sequence[float],
    y: Sequence[int],
    cfg: CalibrationConfig,
    forced: str | None = None,
) -> LabelCalibrator:
    """Fit one label's calibrator; the threshold stays NaN until thresholds are tuned.

    The method is the eligibility rule's choice unless ``forced`` (an inner fit reuses its
    outer calibrator's method so the score scale matches). A label whose fit rows hold no
    positive or no negative cannot be fitted and falls back to the identity.
    """
    positives = int(sum(y))
    negatives = len(y) - positives
    method = forced or choose_method(positives, negatives, cfg.eligibility)
    if method not in METHODS:
        raise BaselineError(f"unknown calibration method {method!r}")
    if positives == 0 or negatives == 0:
        method = UNCALIBRATED
    base = LabelCalibrator(label, method, positives, negatives, math.nan)
    if method == ISOTONIC:
        xs, ys = fit_isotonic(scores, y)
        return replace(base, knots_x=xs, knots_y=ys)
    if method == PLATT:
        slope, intercept = fit_platt(scores, y, cfg.platt)
        return replace(base, slope=slope, intercept=intercept, logit_eps=cfg.platt.logit_eps)
    return base


# -- metrics ---------------------------------------------------------------------------
def brier(scores: Sequence[float], y: Sequence[int]) -> float:
    """Mean squared difference between the probability and the 0/1 outcome."""
    if not scores:
        raise BaselineError("Brier score needs at least one row")
    return math.fsum((s - v) ** 2 for s, v in zip(scores, y, strict=True)) / len(scores)


def bin_members(scores: Sequence[float], settings: EceSettings) -> list[list[int]]:
    """Row indices per equal-mass bin, in score order; a run of tied scores is never split.

    The bin of a run is set by the rank at which the run starts, so ties (an isotonic
    calibrator produces many) stay together. Empty bins are omitted.
    """
    n = len(scores)
    if n == 0:
        raise BaselineError("binning needs at least one row")
    bins = min(settings.bins, max(1, n // settings.min_bin_rows))
    order = sorted(range(n), key=scores.__getitem__)
    members: dict[int, list[int]] = {}
    start = 0
    while start < n:
        end = start
        while end + 1 < n and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        members.setdefault(min(bins - 1, start * bins // n), []).extend(order[start : end + 1])
        start = end + 1
    return [members[slot] for slot in sorted(members)]


def ece(scores: Sequence[float], y: Sequence[int], settings: EceSettings) -> float:
    """Expected calibration error over equal-mass bins: sum of (rows/n) * |mean y - mean score|."""
    n = len(scores)
    total = 0.0
    for rows in bin_members(scores, settings):
        mean_score = math.fsum(scores[i] for i in rows) / len(rows)
        mean_outcome = math.fsum(y[i] for i in rows) / len(rows)
        total += len(rows) / n * abs(mean_outcome - mean_score)
    return total


def reliability(
    scores: Sequence[float], y: Sequence[int], settings: EceSettings
) -> list[dict[str, Any]]:
    """The reliability diagram's data: rows, mean score and fraction positive per bin."""
    return [
        {
            "rows": len(rows),
            "mean_score": math.fsum(scores[i] for i in rows) / len(rows),
            "fraction_positive": math.fsum(y[i] for i in rows) / len(rows),
        }
        for rows in bin_members(scores, settings)
    ]


def _columns(matrix: Array) -> list[list[float]]:
    return [[float(v) for v in matrix[:, j]] for j in range(matrix.shape[1])]


def _int_columns(matrix: Array) -> list[list[int]]:
    return [[int(v) for v in matrix[:, j]] for j in range(matrix.shape[1])]


def _interval(point: float, draws: Sequence[float] | None) -> dict[str, float | None]:
    if draws is None:
        return {"value": point, "ci_low": None, "ci_high": None}
    np = importlib.import_module("numpy")
    low, high = np.percentile(np.array(draws), CI_PERCENTILES)
    return {"value": point, "ci_low": float(low), "ci_high": float(high)}


@dataclass(frozen=True, slots=True)
class ScoredRows:
    """Raw and cross-fitted calibrated scores with labels, clusters and class names."""

    raw: Array
    calibrated: Array
    y: Array
    cluster_code: Array
    classes: tuple[str, ...]


def _point_metrics(
    raw: Sequence[Sequence[float]],
    cal: Sequence[Sequence[float]],
    y: Sequence[Sequence[int]],
    settings: EceSettings,
) -> dict[str, list[float]]:
    """Per-label ECE and Brier of the raw and calibrated columns, in class order."""
    return {
        "ece_raw": [ece(c, v, settings) for c, v in zip(raw, y, strict=True)],
        "ece_calibrated": [ece(c, v, settings) for c, v in zip(cal, y, strict=True)],
        "brier_raw": [brier(c, v) for c, v in zip(raw, y, strict=True)],
        "brier_calibrated": [brier(c, v) for c, v in zip(cal, y, strict=True)],
    }


def _with_deltas(metrics: dict[str, list[float]]) -> dict[str, list[float]]:
    full = dict(metrics)
    full["ece_delta"] = [
        c - r for r, c in zip(metrics["ece_raw"], metrics["ece_calibrated"], strict=True)
    ]
    full["brier_delta"] = [
        c - r for r, c in zip(metrics["brier_raw"], metrics["brier_calibrated"], strict=True)
    ]
    return full


def calibration_report(
    rows: ScoredRows, settings: EceSettings, boot: bm.Bootstrap
) -> dict[str, Any]:
    """ECE and Brier before and after, per label and macro, with cluster-bootstrap intervals.

    The bootstrap resamples whole ``cluster_id`` clusters of the pooled out-of-fold rows and
    recomputes every metric on the SAME cross-fitted scores (it does not refit the calibrators).
    ``delta`` is calibrated minus raw, so a negative value is an improvement. ``boot.samples`` = 0
    skips the intervals.
    """
    np = importlib.import_module("numpy")
    raw, cal, y = _columns(rows.raw), _columns(rows.calibrated), _int_columns(rows.y)
    point = _with_deltas(_point_metrics(raw, cal, y, settings))
    draws: dict[str, list[list[float]]] = {key: [] for key in point}
    by_cluster: dict[int, list[int]] = {}
    for index, code in enumerate(rows.cluster_code):
        by_cluster.setdefault(int(code), []).append(index)
    keys = sorted(by_cluster)
    rng = np.random.default_rng(boot.seed)
    for _ in range(boot.samples):
        picks = rng.integers(0, len(keys), size=len(keys))
        chosen = [i for pick in picks for i in by_cluster[keys[int(pick)]]]
        resampled = _with_deltas(
            _point_metrics(
                [[col[i] for i in chosen] for col in raw],
                [[col[i] for i in chosen] for col in cal],
                [[col[i] for i in chosen] for col in y],
                settings,
            )
        )
        for key, values in resampled.items():
            draws[key].append(values)
    return _report_body(rows.classes, point, draws, boot, (raw, cal, y, settings))


def _report_body(
    classes: Sequence[str],
    point: Mapping[str, list[float]],
    draws: Mapping[str, list[list[float]]],
    boot: bm.Bootstrap,
    columns: tuple[
        Sequence[Sequence[float]], Sequence[Sequence[float]], Sequence[Sequence[int]], EceSettings
    ],
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    have = boot.samples > 0
    per_label: dict[str, Any] = {}
    macro: dict[str, Any] = {}
    for key, values in point.items():
        stack = np.array(draws[key]) if have else None
        macro[key] = _interval(
            math.fsum(values) / len(values), None if stack is None else stack.mean(axis=1)
        )
        for j, name in enumerate(classes):
            per_label.setdefault(name, {})[key] = _interval(
                values[j], None if stack is None else stack[:, j]
            )
    raw, cal, y, settings = columns
    return {
        "per_label": per_label,
        "macro": macro,
        "reliability": {
            name: {
                "raw": reliability(raw[j], y[j], settings),
                "calibrated": reliability(cal[j], y[j], settings),
            }
            for j, name in enumerate(classes)
        },
        "bootstrap": {"samples": boot.samples, "seed": boot.seed, "unit": "cluster_id"},
    }


# -- cross-fitting ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CalibrationData:
    """The out-of-fold rows being calibrated: scores, 0/1 labels, partition, cluster per row."""

    scores: Array
    y: Array
    partition: Array
    cluster_code: Array
    classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossFit:
    """What cross-fitting produced; none of it used a row's own labels to score that row."""

    calibrated: Array
    fold_thresholds: dict[int, list[float]]
    frozen_thresholds: list[float]
    frozen: list[LabelCalibrator]
    methods: dict[str, Any]


def _clusters(data: CalibrationData, mask: Array) -> set[int]:
    return {int(c) for c in data.cluster_code[mask]}


def _fit_all(
    data: CalibrationData, fit: Array, cfg: CalibrationConfig, forced: Sequence[str] | None
) -> list[LabelCalibrator]:
    """One calibrator per label from the rows selected by the boolean mask ``fit``."""
    scores, labels = data.scores[fit], data.y[fit]
    return [
        fit_label(
            name,
            [float(v) for v in scores[:, j]],
            [int(v) for v in labels[:, j]],
            cfg,
            None if forced is None else forced[j],
        )
        for j, name in enumerate(data.classes)
    ]


def _apply_all(calibrators: Sequence[LabelCalibrator], scores: Array) -> Array:
    np = importlib.import_module("numpy")
    out = np.zeros(scores.shape)
    for j, cal in enumerate(calibrators):
        out[:, j] = [cal.apply(float(v)) for v in scores[:, j]]
    return out


def _guard(
    data: CalibrationData,
    sets: Mapping[str, Array],
    held_out: Collection[int],
) -> None:
    """Raise unless the calibrator's rows, the scored rows and the test clusters are disjoint."""
    named: dict[str, Collection[int]] = {k: _clusters(data, m) for k, m in sets.items()}
    named["test"] = set(held_out)
    head.assert_no_overlap(named)


def crossfit(
    data: CalibrationData,
    folds: Sequence[Fold],
    cfg: CalibrationConfig,
    held_out: Collection[int],
) -> CrossFit:
    """Cross-fit the calibrators and the thresholds over the leave-one-partition-out folds."""
    np = importlib.import_module("numpy")
    calibrated = np.zeros(data.scores.shape)
    outer: dict[int, list[LabelCalibrator]] = {}
    methods: dict[str, Any] = {"fold": {}, "inner": {}}
    for fold in folds:
        scored = data.partition == fold.validate
        fit = ~scored
        _guard(data, {"calibration": fit, "scored": scored}, held_out)
        cals = _fit_all(data, fit, cfg, None)
        outer[fold.validate] = cals
        calibrated[scored] = _apply_all(cals, data.scores[scored])
        methods["fold"][str(fold.validate)] = [c.method for c in cals]
    fold_thresholds: dict[int, list[float]] = {}
    for fold in folds:
        pooled_scores, pooled_y = [], []
        forced = [c.method for c in outer[fold.validate]]
        inner_methods: dict[str, list[str]] = {}
        for other in folds:
            if other.validate == fold.validate:
                continue
            scored = data.partition == other.validate
            fit = (data.partition != other.validate) & (data.partition != fold.validate)
            _guard(
                data,
                {
                    "calibration": fit,
                    "scored": scored,
                    "outer_fold": data.partition == fold.validate,
                },
                held_out,
            )
            cals = _fit_all(data, fit, cfg, forced)
            inner_methods[str(other.validate)] = [c.method for c in cals]
            pooled_scores.append(_apply_all(cals, data.scores[scored]))
            pooled_y.append(data.y[scored])
        methods["inner"][str(fold.validate)] = inner_methods
        fold_thresholds[fold.validate] = bm.tune_thresholds(
            np.concatenate(pooled_scores), np.concatenate(pooled_y)
        )
    everything = np.ones(len(data.partition), dtype=bool)
    _guard(data, {"calibration": everything}, held_out)
    frozen_thresholds = bm.tune_thresholds(calibrated, data.y)
    frozen = [
        replace(cal, threshold=threshold)
        for cal, threshold in zip(
            _fit_all(data, everything, cfg, None), frozen_thresholds, strict=True
        )
    ]
    methods["frozen"] = [c.method for c in frozen]
    methods["consistent"] = {
        name: all(methods["fold"][k][j] == frozen[j].method for k in methods["fold"])
        for j, name in enumerate(data.classes)
    }
    return CrossFit(calibrated, fold_thresholds, frozen_thresholds, frozen, methods)


def validate_calibrated(
    fit: CrossFit, data: CalibrationData, folds: Sequence[Fold], boot: bm.Bootstrap
) -> dict[str, Any]:
    """Decision metrics at the new thresholds, laid out like ``baseline_models.validate_oof``."""
    np = importlib.import_module("numpy")
    predicted = np.zeros(data.y.shape, dtype=int)
    per_fold = []
    off = bm.Bootstrap(0, 0)
    for fold in folds:
        held = data.partition == fold.validate
        predicted[held] = fit.calibrated[held] >= np.array(fit.fold_thresholds[fold.validate])
        report = bm.metrics_report(
            data.y[held], predicted[held], data.cluster_code[held], data.classes, off
        )
        per_fold.append(
            {
                "validate": fold.validate,
                "rows": report["rows"],
                "macro_f1": report["macro_f1"]["value"],
                "per_class": {
                    name: {
                        "f1": entry["f1"]["value"],
                        "mcc": entry["mcc"]["value"],
                        "support": entry["support"],
                        "undefined": entry["undefined"],
                    }
                    for name, entry in report["per_class"].items()
                },
            }
        )
    return {
        "folds": per_fold,
        "mean_fold_macro_f1": float(np.mean([f["macro_f1"] for f in per_fold])),
        "fold_thresholds": {str(k): v for k, v in fit.fold_thresholds.items()},
        "frozen_thresholds": fit.frozen_thresholds,
        "pooled": bm.metrics_report(data.y, predicted, data.cluster_code, data.classes, boot),
    }


# -- the manifest ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Pins:
    """What a calibration manifest is checked against; read from the repository."""

    head_frozen_sha256: str
    head_weights_sha256: str
    registry_sha256: str
    corpus_sha256: str
    embedding_cache_key: str
    residue_cache_key: str
    class_map_sha256: str
    config_sha256: str
    classes: tuple[str, ...]


def _object(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise BaselineError(f"manifest is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def _input_problems(inputs: Mapping[str, Any], pins: Pins) -> list[str]:
    expected = {
        "head_frozen_sha256": pins.head_frozen_sha256,
        "head_weights_sha256": pins.head_weights_sha256,
        "registry_sha256": pins.registry_sha256,
        "corpus_sha256": pins.corpus_sha256,
        "embedding_cache_key": pins.embedding_cache_key,
        "residue_cache_key": pins.residue_cache_key,
        "class_map_sha256": pins.class_map_sha256,
        "calibration_config_sha256": pins.config_sha256,
    }
    found = [
        f"inputs.{k} differs from the repository" for k, v in expected.items() if inputs.get(k) != v
    ]
    for key in ("oof_sha256", "feature_table_sha256", "calibrator_sha256"):
        if not is_sha256(inputs.get(key)):
            found.append(f"inputs.{key} is not a SHA-256")
    return found


def _label_problems(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    found: list[str] = []
    labels = frozen.get("labels")
    if not isinstance(labels, list) or [e.get("label") for e in labels] != list(pins.classes):
        return ["labels do not cover the classes in order"]
    for entry in labels:
        if entry.get("method") not in METHODS:
            found.append(f"{entry.get('label')}: unknown calibration method")
        threshold = entry.get("threshold")
        if not isinstance(threshold, float) or not math.isfinite(threshold):
            found.append(f"{entry.get('label')}: threshold must be a finite number")
    return found


def check_manifest(manifest: Mapping[str, Any], pins: Pins) -> list[str]:
    """Every way the calibration manifest disagrees with itself or the repository."""
    try:
        frozen = _object(manifest, "frozen")
        protocol = _object(frozen, "protocol")
        inputs = _object(frozen, "inputs")
        validation = _object(frozen, "validation")
    except BaselineError as error:
        return [str(error)]
    problems = _input_problems(inputs, pins)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version is not the one this code writes")
    if protocol.get("validation_partitions") != list(VALIDATION_PARTITIONS):
        problems.append("validation partitions are not [1, 2, 3, 4]")
    if protocol.get("test_partition_policy") != TEST_POLICY:
        problems.append("the test partition policy is not 'untouched'")
    if frozen.get("classes") != list(pins.classes):
        problems.append("classes differ from baseline_class_map.yaml")
    problems += _label_problems(frozen, pins)
    if not {"calibration", "raw", "calibrated"} <= set(validation):
        problems.append("validation must hold calibration, raw and calibrated")
    if manifest.get("frozen_sha256") != digest(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
    if manifest.get("test") is not None:
        problems.append(
            f"the test partition ({TEST_PARTITION}) must stay untouched by Phase {PHASE}; "
            "the manifest carries a test block"
        )
    return problems
