"""Phase 8 numerics: metrics, thresholds, bootstrap and the scikit-learn models.

numpy and scikit-learn are imported lazily, inside the functions that need them, so that
importing this module costs nothing on a runner that has neither (CI runs the manifest
check without them). Arrays are typed ``Any`` for the same reason: the packages are an
optional extra (``baselines``), not a dependency of the served image.

Metric conventions, each stated once here and repeated in ADR-0010:

* a class's F1 and MCC are 0.0 when their denominator is 0;
* sample-wise Jaccard is |predicted AND true| / |predicted OR true| per row, 1.0 for a
  row whose predicted and true sets are both empty, then the mean over rows;
* a positive call is ``score >= threshold`` and the threshold is a score, never a literal;
* confidence intervals resample whole clusters (``cluster_id``), not rows, because
  homologous rows inside one cluster are not independent.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import platform
import time
from typing import Any

from protloc.training.baselines import (
    N_ESTIMATORS,
    BaselineError,
    Fold,
)

Array = Any
CI_PERCENTILES = (2.5, 97.5)
BOOTSTRAP_CHUNK = 50
LOGREG_MAX_ITER = 2000
SVC_MAX_ITER = 20000
PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """Cluster-bootstrap settings; ``samples`` = 0 skips the intervals."""

    samples: int
    seed: int


@dataclass(frozen=True, slots=True)
class Split:
    """The rows being scored: 0/1 labels, partition number and cluster code per row."""

    y: Array
    partition: Array
    cluster_code: Array


@dataclass(frozen=True, slots=True)
class Runtime:
    """Seed and parallelism of a model fit; neither changes what the fit computes."""

    seed: int
    n_jobs: int


def _np() -> Any:
    return importlib.import_module("numpy")


# -- the table -------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RawTable:
    """The feature table as plain lists: one row per accession, one X row per distinct sequence."""

    accessions: Sequence[str]
    partitions: Sequence[int]
    clusters: Sequence[str]
    seq_index: Sequence[int]
    x_rows: Sequence[Sequence[float]]
    rules: Mapping[str, Sequence[Sequence[float]]]
    labels: Mapping[str, Sequence[Sequence[int] | None]]
    widths: Mapping[str, int]


def assemble(raw: RawTable) -> dict[str, Array]:
    """Pack the per-accession and per-distinct-sequence lists into one dict of arrays."""
    np = _np()
    _, cluster_code = np.unique(np.array(list(raw.clusters)), return_inverse=True)
    table: dict[str, Array] = {
        "accession": np.array(list(raw.accessions)),
        "partition": np.array(raw.partitions, dtype=np.int8),
        "cluster_code": cluster_code.astype(np.int32),
        "seq_index": np.array(raw.seq_index, dtype=np.int32),
        "X": np.array(raw.x_rows, dtype=np.float64),
    }
    for task, width in raw.widths.items():
        table[f"R_{task}"] = np.array(raw.rules[task], dtype=np.float64)
        y = np.zeros((len(raw.accessions), width), dtype=np.int8)
        mask = np.zeros(len(raw.accessions), dtype=bool)
        for row, label in enumerate(raw.labels[task]):
            if label is not None:
                y[row] = label
                mask[row] = True
        table[f"Y_{task}"] = y
        table[f"M_{task}"] = mask
    return table


def save_table(path: Path, table: Mapping[str, Array]) -> None:
    """Write the table as an uncompressed ``.npz`` (no pickle)."""
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **table)


def load_table(path: Path) -> dict[str, Array]:
    """Read the table back; ``allow_pickle`` stays off."""
    np = _np()
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def save_array(path: Path, array: Array) -> None:
    """Write one array as ``.npy``."""
    _np().save(path, array)


def load_array(path: Path) -> Array:
    """Read one ``.npy`` array; ``allow_pickle`` stays off."""
    return _np().load(path, allow_pickle=False)


def dump_model(path: Path, fitted: Mapping[str, Any]) -> None:
    """Persist a fitted scaler and estimator (gitignored; never served)."""
    importlib.import_module("joblib").dump(dict(fitted), path)


def load_model(path: Path) -> dict[str, Any]:
    """Load a model written by ``dump_model``."""
    loaded: dict[str, Any] = importlib.import_module("joblib").load(path)
    return loaded


def task_rows(table: Mapping[str, Array], task: str, partitions: Sequence[int], stride: int) -> Any:
    """Row indices of a task in the given partitions, every ``stride``-th (1 = all)."""
    np = _np()
    keep = table[f"M_{task}"] & np.isin(table["partition"], list(partitions))
    return np.flatnonzero(keep)[::stride]


def task_xy(table: Mapping[str, Array], task: str, rows: Array) -> tuple[Array, Array]:
    """Motif features and 0/1 labels of the given rows."""
    return table["X"][table["seq_index"][rows]], table[f"Y_{task}"][rows].astype(int)


def rule_scores(table: Mapping[str, Array], task: str, rows: Array) -> Array:
    """The untrained motif engine's scores for the given rows."""
    return table[f"R_{task}"][table["seq_index"][rows]]


# -- metrics ---------------------------------------------------------------------------
def _ratio(num: Array, den: Array) -> Array:
    np = _np()
    safe = np.where(den == 0, 1.0, den)
    return np.where(den == 0, 0.0, num / safe)


def f1_from_counts(tp: Array, fp: Array, fn: Array) -> Array:
    """F1 = 2TP / (2TP + FP + FN), 0.0 where the denominator is 0."""
    return _ratio(2.0 * tp, 2.0 * tp + fp + fn)


def mcc_from_counts(tp: Array, fp: Array, fn: Array, tn: Array) -> Array:
    """Matthews correlation coefficient, 0.0 where the denominator is 0."""
    np = _np()
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _ratio(tp * tn - fp * fn, den)


def _counts(y: Array, p: Array) -> tuple[Array, Array, Array, Array]:
    y = y.astype(float)
    p = p.astype(float)
    tp = (y * p).sum(axis=0)
    fp = ((1 - y) * p).sum(axis=0)
    fn = (y * (1 - p)).sum(axis=0)
    tn = ((1 - y) * (1 - p)).sum(axis=0)
    return tp, fp, fn, tn


def sample_jaccard(y: Array, p: Array) -> Array:
    """Per-row Jaccard of the predicted and true label sets (1.0 when both are empty)."""
    np = _np()
    inter = (y * p).sum(axis=1).astype(float)
    union = np.maximum(y, p).sum(axis=1).astype(float)
    return np.where(union == 0, 1.0, inter / np.where(union == 0, 1.0, union))


def best_threshold(scores: Array, y: Array) -> float:
    """The score that maximises MCC when ``score >= threshold`` is called positive.

    Candidates are the distinct score values. Ties between equal MCCs go to the highest
    threshold, the more conservative call. A class with no positives gets the top score.
    """
    np = _np()
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    positive = y[order].astype(float)
    tp = np.cumsum(positive)
    fp = np.cumsum(1.0 - positive)
    total = positive.sum()
    fn = total - tp
    tn = (len(s) - total) - fp
    mcc = mcc_from_counts(tp, fp, fn, tn)
    last_of_value = np.append(s[1:] != s[:-1], True)
    mcc = np.where(last_of_value, mcc, -np.inf)
    return float(s[int(np.argmax(mcc))])


def tune_thresholds(scores: Array, y: Array) -> list[float]:
    """One MCC-optimal threshold per class column."""
    return [best_threshold(scores[:, j], y[:, j]) for j in range(scores.shape[1])]


def _value(point: float, interval: Sequence[float] | None) -> dict[str, float | None]:
    return {
        "value": float(point),
        "ci_low": None if interval is None else float(interval[0]),
        "ci_high": None if interval is None else float(interval[1]),
    }


def metrics_report(
    y: Array, p: Array, cluster_code: Array, classes: Sequence[str], boot: Bootstrap
) -> dict[str, Any]:
    """Macro F1, per-class F1 and MCC, sample-wise Jaccard and support, with cluster CIs.

    ``boot.samples`` = 0 skips the bootstrap (the intervals are then None).
    """
    np = _np()
    samples = boot.samples
    tp, fp, fn, tn = _counts(y, p)
    f1 = f1_from_counts(tp, fp, fn)
    mcc = mcc_from_counts(tp, fp, fn, tn)
    jac = sample_jaccard(y, p)
    point_macro, point_jac = float(f1.mean()), float(jac.mean())
    intervals: dict[str, Any] = {"macro": None, "jac": None, "f1": None, "mcc": None}
    if samples > 0:
        _, inverse = np.unique(cluster_code, return_inverse=True)
        n_clusters = int(inverse.max()) + 1
        per_cluster = [
            np.stack(
                [
                    np.bincount(inverse, weights=arr[:, j], minlength=n_clusters)
                    for j in range(len(classes))
                ],
                axis=1,
            )
            for arr in (y * p, (1 - y) * p, y * (1 - p), (1 - y) * (1 - p))
        ]
        jac_sum = np.bincount(inverse, weights=jac, minlength=n_clusters)
        size = np.bincount(inverse, minlength=n_clusters).astype(float)
        rng = np.random.default_rng(boot.seed)
        draws: dict[str, list[Array]] = {"macro": [], "jac": [], "f1": [], "mcc": []}
        for start in range(0, samples, BOOTSTRAP_CHUNK):
            width = min(BOOTSTRAP_CHUNK, samples - start)
            weights = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=width)
            weights = weights.astype(float)
            b_tp, b_fp, b_fn, b_tn = (weights @ part for part in per_cluster)
            b_f1 = f1_from_counts(b_tp, b_fp, b_fn)
            draws["f1"].append(b_f1)
            draws["macro"].append(b_f1.mean(axis=1))
            draws["mcc"].append(mcc_from_counts(b_tp, b_fp, b_fn, b_tn))
            draws["jac"].append(_ratio(weights @ jac_sum, weights @ size))
        for key, parts in draws.items():
            stacked = np.concatenate(parts, axis=0)
            intervals[key] = np.percentile(stacked, CI_PERCENTILES, axis=0)
    per_class = {}
    for j, name in enumerate(classes):
        per_class[name] = {
            "f1": _value(f1[j], None if intervals["f1"] is None else intervals["f1"][:, j]),
            "mcc": _value(mcc[j], None if intervals["mcc"] is None else intervals["mcc"][:, j]),
            "support": int(y[:, j].sum()),
        }
    return {
        "rows": int(y.shape[0]),
        "macro_f1": _value(point_macro, intervals["macro"]),
        "jaccard": _value(point_jac, intervals["jac"]),
        "per_class": per_class,
    }


# -- validation with nested thresholds -------------------------------------------------
def validate_oof(
    scores: Array, split: Split, folds: Sequence[Fold], classes: Sequence[str], boot: Bootstrap
) -> dict[str, Any]:
    """Score out-of-fold scores with thresholds that never saw the fold they are applied to.

    For fold k the thresholds are tuned on the out-of-fold scores of the other validation
    partitions. ``frozen_thresholds`` are tuned on all validation rows and are what the
    test partition will use. The pooled metrics concatenate the four validation folds.
    """
    np = _np()
    y = split.y
    predicted = np.zeros(y.shape, dtype=int)
    per_fold = []
    for fold in folds:
        held = split.partition == fold.validate
        thresholds = tune_thresholds(scores[~held], y[~held])
        predicted[held] = scores[held] >= np.array(thresholds)
        fold_f1 = f1_from_counts(*_counts(y[held], predicted[held])[:3]).mean()
        per_fold.append(
            {"validate": fold.validate, "rows": int(held.sum()), "macro_f1": float(fold_f1)}
        )
    return {
        "folds": per_fold,
        "mean_fold_macro_f1": float(np.mean([f["macro_f1"] for f in per_fold])),
        "frozen_thresholds": tune_thresholds(scores, y),
        "pooled": metrics_report(y, predicted, split.cluster_code, classes, boot),
    }


def score_test(
    scores: Array,
    split: Split,
    thresholds: Sequence[float],
    classes: Sequence[str],
    boot: Bootstrap,
) -> dict[str, Any]:
    """Metrics of test scores under thresholds frozen on validation."""
    np = _np()
    predicted = (scores >= np.array(list(thresholds))).astype(int)
    return metrics_report(split.y, predicted, split.cluster_code, classes, boot)


# -- models ----------------------------------------------------------------------------
def make_estimator(model: str, params: Mapping[str, Any], seed: int, n_jobs: int) -> Any:
    """A one-vs-rest estimator with balanced class weights; ``n_jobs`` spans the classes."""
    multiclass = importlib.import_module("sklearn.multiclass")
    if model == "logreg":
        linear = importlib.import_module("sklearn.linear_model")
        base = linear.LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=LOGREG_MAX_ITER, random_state=seed
        )
    elif model == "linear_svc":
        svm = importlib.import_module("sklearn.svm")
        base = svm.LinearSVC(
            C=params["C"],
            class_weight="balanced",
            dual=False,
            max_iter=SVC_MAX_ITER,
            random_state=seed,
        )
    elif model == "random_forest":
        ensemble = importlib.import_module("sklearn.ensemble")
        base = ensemble.RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    else:
        raise BaselineError(f"unknown model {model!r}")
    return multiclass.OneVsRestClassifier(base, n_jobs=n_jobs)


def fit_model(
    model: str, params: Mapping[str, Any], x: Array, y: Array, runtime: Runtime
) -> dict[str, Any]:
    """Fit the scaler on ``x`` (the training rows only) and the estimator on the scaled rows."""
    preprocessing = importlib.import_module("sklearn.preprocessing")
    scaler = preprocessing.StandardScaler().fit(x)
    estimator = make_estimator(model, params, runtime.seed, runtime.n_jobs)
    estimator = estimator.fit(scaler.transform(x), y)
    return {"model": model, "params": dict(params), "scaler": scaler, "estimator": estimator}


def model_scores(fitted: Mapping[str, Any], x: Array) -> Array:
    """Per-class scores: log-odds or margins for the linear models, probabilities for RF."""
    scaled = fitted["scaler"].transform(x)
    estimator = fitted["estimator"]
    if fitted["model"] == "random_forest":
        return estimator.predict_proba(scaled)
    return estimator.decision_function(scaled)


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """What one ``train_model`` call fits: the task, the model and how to run it."""

    task: str
    model: str
    classes: Sequence[str]
    seed: int
    n_jobs: int
    stride: int


def train_model(
    table: Mapping[str, Array],
    spec: TrainSpec,
    grid: Sequence[tuple[int, Mapping[str, Any]]],
    folds: Sequence[Fold],
    log: Callable[[str], None],
) -> tuple[dict[str, Any], Array, dict[str, Any]]:
    """Run the grid by leave-one-partition-out, pick the best, refit it on every validation row.

    Returns (report, out-of-fold scores of the selected configuration, the fitted model).
    The test partition is never read here.
    """
    np = _np()
    task, model, classes, seed = spec.task, spec.model, spec.classes, spec.seed
    runtime = Runtime(seed, spec.n_jobs)
    split_boot = Bootstrap(0, seed)
    parts = sorted({p for fold in folds for p in (*fold.train, fold.validate)})
    rows = task_rows(table, task, parts, spec.stride)
    x, y = task_xy(table, task, rows)
    partition = table["partition"][rows]
    cluster = table["cluster_code"][rows]
    entries: list[dict[str, Any]] = []
    best: tuple[float, int, Array] | None = None
    for index, params in grid:
        oof = np.zeros(y.shape, dtype=float)
        fit_seconds = []
        for fold in folds:
            train = np.isin(partition, list(fold.train))
            held = partition == fold.validate
            began = time.perf_counter()
            fitted = fit_model(model, params, x[train], y[train], runtime)
            fit_seconds.append(time.perf_counter() - began)
            oof[held] = model_scores(fitted, x[held])
        scored = validate_oof(oof, Split(y, partition, cluster), folds, classes, split_boot)
        entries.append(
            {
                "index": index,
                "params": dict(params),
                "mean_fold_macro_f1": scored["mean_fold_macro_f1"],
                "fold_macro_f1": [f["macro_f1"] for f in scored["folds"]],
                "fit_seconds": fit_seconds,
            }
        )
        mean_f1 = scored["mean_fold_macro_f1"]
        log(f"  {task}/{model} cfg {index} {dict(params)}: mean fold macro F1 {mean_f1:.4f}")
        if best is None or scored["mean_fold_macro_f1"] > best[0]:
            best = (scored["mean_fold_macro_f1"], index, oof)
    if best is None:
        raise BaselineError(f"{model} has an empty grid")
    selected = dict(grid)[best[1]]
    began = time.perf_counter()
    final = fit_model(model, selected, x, y, runtime)
    report = {
        "grid": entries,
        "selected_index": best[1],
        "selected_params": dict(selected),
        "train_rows": len(rows),
        "final_fit_seconds": time.perf_counter() - began,
    }
    return report, best[2], final


# -- environment -----------------------------------------------------------------------
def versions() -> dict[str, str]:
    """Installed versions of the numerical packages, by distribution name."""
    metadata = importlib.import_module("importlib.metadata")
    return {name.replace("-", "_"): str(metadata.version(name)) for name in PACKAGES}


def hardware() -> dict[str, Any]:
    """Where a stage ran: enough to say 'this machine, this device only'."""
    os = importlib.import_module("os")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }
