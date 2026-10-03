"""Phase 9 localization head: configuration, rows, protocol guards and the frozen manifest.

Everything here runs without torch. ``protloc.training.head_net`` holds the network and the
training loop and imports torch lazily; this module holds what CI can import and test on a
runner with neither torch nor a GPU:

* the configuration (``configs/training/head.yaml``) parsed and checked, and the grid,
* the rows the head is trained and scored on (ADR-0006: annotated labels only),
* the loss weights, the early-stopping rule, the batch order and the inner holdout,
* the leakage guards (no ``cluster_id`` shared between train, validation and test),
* the frozen manifest: how it is checked (ADR-0011 section 2.8).

``training/`` is an entry-point layer: nothing imports it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import itertools
import math
from pathlib import Path
import re
from typing import Any

import yaml

from protloc.training import baseline_models as bm
from protloc.training.baselines import (
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    Fold,
    check_subset_block,
    digest,
)

Array = Any
SCHEMA_VERSION = 1
TASK = "localization"
LOSS_KIND = "weighted_focal"
#: Order in which the grid expands: the first key varies slowest.
GRID_KEYS: tuple[str, ...] = ("learning_rate", "dropout", "hidden", "gamma")
#: Chunks of this many batches are length-sorted before they are cut (padding waste).
BUCKET_BATCHES = 8
#: An inner holdout needs a cluster to fit on and a cluster to hold out.
MIN_CLUSTERS = 2
_SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class Architecture:
    """Attention pooling settings; the MLP width is a grid axis, not an architecture setting."""

    heads: int
    attention_hidden: int
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class Schedule:
    """Optimisation schedule shared by every grid point."""

    batch_size: int
    max_epochs: int
    patience: int
    inner_holdout_fraction: float


@dataclass(frozen=True, slots=True)
class SmokeSettings:
    """What ``--smoke`` shrinks: a code check, never a result."""

    stride: int
    max_epochs: int
    grid_points: int


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One hyperparameter setting."""

    learning_rate: float
    dropout: float
    hidden: int
    gamma: float

    def as_dict(self) -> dict[str, float | int]:
        """The setting as plain JSON-able values, in grid order."""
        return {
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "hidden": self.hidden,
            "gamma": self.gamma,
        }


@dataclass(frozen=True, slots=True)
class HeadConfig:
    """``configs/training/head.yaml``, parsed and checked."""

    seed: int
    architecture: Architecture
    grid: Mapping[str, tuple[float, ...]]
    weight_decay: float
    schedule: Schedule
    smoke: SmokeSettings
    sha256: str


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{where} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise BaselineError(f"{where} must be a positive number")
    return float(value)


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BaselineError(f"{where} must be a positive integer")
    return value


def load_config(path: Path) -> HeadConfig:
    """Parse the head configuration and reject anything malformed before a fit starts."""
    raw = path.read_bytes()
    doc = _mapping(yaml.safe_load(raw), str(path))
    if doc.get("name") != "head" or doc.get("task") != TASK:
        raise BaselineError("head.yaml must be name 'head' for task 'localization'")
    arch = _mapping(doc.get("architecture"), "architecture")
    if arch.get("pooling") != "attention":
        raise BaselineError("architecture.pooling must be 'attention'")
    if _mapping(doc.get("loss"), "loss").get("kind") != LOSS_KIND:
        raise BaselineError(f"loss.kind must be '{LOSS_KIND}'")
    optimiser = _mapping(doc.get("optimiser"), "optimiser")
    if optimiser.get("name") != "adamw":
        raise BaselineError("optimiser.name must be 'adamw'")
    grid_doc = _mapping(doc.get("grid"), "grid")
    if set(grid_doc) != set(GRID_KEYS):
        raise BaselineError(f"grid must have exactly the keys {list(GRID_KEYS)}")
    grid: dict[str, tuple[float, ...]] = {}
    for key in GRID_KEYS:
        values = grid_doc[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise BaselineError(f"grid.{key} must be a non-empty list of distinct values")
        grid[key] = tuple(float(v) for v in values)
    if any(not 0.0 <= d < 1.0 for d in grid["dropout"]):
        raise BaselineError("grid.dropout values must be in [0, 1)")
    if any(g < 0.0 for g in grid["gamma"]):
        raise BaselineError("grid.gamma values must be >= 0")
    if any(h < 1 or h != int(h) for h in grid["hidden"]):
        raise BaselineError("grid.hidden values must be positive integers")
    if any(lr <= 0.0 for lr in grid["learning_rate"]):
        raise BaselineError("grid.learning_rate values must be positive")
    schedule = _mapping(doc.get("schedule"), "schedule")
    fraction = _positive(schedule.get("inner_holdout_fraction"), "schedule.inner_holdout_fraction")
    if fraction >= 1.0:
        raise BaselineError("schedule.inner_holdout_fraction must be below 1")
    smoke = _mapping(doc.get("smoke"), "smoke")
    return HeadConfig(
        seed=_count(doc.get("seed"), "seed"),
        architecture=Architecture(
            heads=_count(arch.get("heads"), "architecture.heads"),
            attention_hidden=_count(arch.get("attention_hidden"), "architecture.attention_hidden"),
            embedding_dim=_count(arch.get("embedding_dim"), "architecture.embedding_dim"),
        ),
        grid=grid,
        weight_decay=float(_positive(optimiser.get("weight_decay"), "optimiser.weight_decay")),
        schedule=Schedule(
            batch_size=_count(schedule.get("batch_size"), "schedule.batch_size"),
            max_epochs=_count(schedule.get("max_epochs"), "schedule.max_epochs"),
            patience=_count(schedule.get("patience"), "schedule.patience"),
            inner_holdout_fraction=fraction,
        ),
        smoke=SmokeSettings(
            stride=_count(smoke.get("stride"), "smoke.stride"),
            max_epochs=_count(smoke.get("max_epochs"), "smoke.max_epochs"),
            grid_points=_count(smoke.get("grid_points"), "smoke.grid_points"),
        ),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def grid_points(config: HeadConfig) -> list[GridPoint]:
    """Every grid point in expansion order (the first key varies slowest)."""
    return [
        GridPoint(
            learning_rate=lr,
            dropout=dropout,
            hidden=int(hidden),
            gamma=gamma,
        )
        for lr, dropout, hidden, gamma in itertools.product(*(config.grid[k] for k in GRID_KEYS))
    ]


# -- rows ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Rows:
    """The labelled accessions being used: their sequence hash, 0/1 labels, partition, cluster."""

    shas: Sequence[str]
    y: Array
    partition: Array
    cluster_code: Array

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.shas)


def build_rows(
    table: Mapping[str, Array], order: Sequence[str], partitions: Sequence[int], stride: int
) -> Rows:
    """The annotated localization rows of the given partitions (ADR-0006), every stride-th.

    ``order`` is the distinct sequence hashes in the feature table's own order. Rows whose
    label state is ``unknown`` or ``not_applicable`` carry no label and are not here: the
    ``M_localization`` mask the table was built with excludes them, they are never negatives.
    """
    rows = bm.task_rows(table, TASK, partitions, stride)
    return Rows(
        shas=[order[int(i)] for i in table["seq_index"][rows]],
        y=table[f"Y_{TASK}"][rows].astype(int),
        partition=table["partition"][rows],
        cluster_code=table["cluster_code"][rows],
    )


def held_out_clusters(table: Mapping[str, Array]) -> set[int]:
    """The cluster codes of the test partition: their identities only, never labels or scores."""
    return {int(c) for c in table["cluster_code"][table["partition"] == TEST_PARTITION]}


def clusters_of(rows: Rows, indices: Sequence[int]) -> set[int]:
    """The cluster codes of the given rows."""
    return {int(rows.cluster_code[i]) for i in indices}


def assert_no_overlap(sets: Mapping[str, Collection[int]]) -> None:
    """Raise unless the named cluster sets are pairwise disjoint (run before every fit)."""
    names = list(sets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = set(sets[first]) & set(sets[second])
            if shared:
                raise BaselineError(
                    f"cluster leakage: {len(shared)} cluster_id values are in both "
                    f"'{first}' and '{second}'"
                )


# -- loss weights, early stopping, batches, inner holdout ------------------------------
def pos_weights(labels: Sequence[Sequence[int]]) -> list[float]:
    """Per-class ``negatives / positives`` over the TRAINING rows passed in; 1.0 if no positive.

    Callers pass the training rows and nothing else; a class with no positive in them gets
    1.0 rather than an infinite weight.
    """
    if not labels:
        raise BaselineError("pos_weights needs at least one training row")
    width = len(labels[0])
    positives = [0] * width
    for row in labels:
        if len(row) != width:
            raise BaselineError("label rows differ in width")
        for j, value in enumerate(row):
            positives[j] += int(value)
    total = len(labels)
    return [(total - p) / p if p > 0 else 1.0 for p in positives]


class EarlyStopper:
    """Stop when the monitored score has not strictly improved for ``patience`` epochs."""

    def __init__(self, patience: int) -> None:
        """Start with no epoch seen."""
        if patience < 1:
            raise BaselineError("patience must be at least 1")
        self.patience = patience
        self.epochs = 0
        self.best_score = -math.inf
        self.best_epoch = 0

    def update(self, score: float) -> bool:
        """Record one epoch's score; True if it is the best so far (strictly)."""
        self.epochs += 1
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epochs
            return True
        return False

    @property
    def should_stop(self) -> bool:
        """True once ``patience`` epochs have passed since the best one."""
        return self.epochs - self.best_epoch >= self.patience


class Shuffler:
    """Deterministic shuffling by SHA-256 rank: the same seed gives the same orders everywhere.

    Every call to :meth:`shuffle` uses a fresh call number, so successive shuffles differ.
    Nothing here is random in the cryptographic sense, and nothing needs to be.
    """

    def __init__(self, seed: int) -> None:
        """Start at call 0."""
        self._seed = seed
        self._calls = 0

    def shuffle(self, items: list[Any]) -> None:
        """Reorder ``items`` in place."""
        self._calls += 1
        prefix = f"{self._seed}:{self._calls}:"
        items.sort(key=lambda item: hashlib.sha256((prefix + repr(item)).encode()).digest())


def epoch_batches(
    lengths: Sequence[int], indices: Sequence[int], batch_size: int, rng: Shuffler
) -> list[list[int]]:
    """One epoch of batches over ``indices``: shuffled, length-bucketed, batches shuffled.

    ``lengths`` is indexed by row. Each row appears exactly once. Sorting inside chunks of
    ``BUCKET_BATCHES`` batches keeps padding small without making the batches deterministic.
    """
    if batch_size < 1:
        raise BaselineError("batch_size must be at least 1")
    shuffled = list(indices)
    rng.shuffle(shuffled)
    chunk = batch_size * BUCKET_BATCHES
    batches: list[list[int]] = []
    for start in range(0, len(shuffled), chunk):
        block = sorted(shuffled[start : start + chunk], key=lambda i: lengths[i])
        batches.extend(block[k : k + batch_size] for k in range(0, len(block), batch_size))
    rng.shuffle(batches)
    return batches


def inner_split(
    cluster_code: Sequence[int], rows: Sequence[int], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split ``rows`` into (fit, holdout) by WHOLE clusters, for early stopping.

    ``cluster_code`` is indexed by row. About ``fraction`` of the clusters (at least one,
    never all) go to the holdout, chosen by ``Shuffler(seed)``; the result is
    deterministic and no cluster has rows on both sides.
    """
    unique = sorted({int(cluster_code[i]) for i in rows})
    if len(unique) < MIN_CLUSTERS:
        raise BaselineError("an inner holdout needs at least two clusters")
    Shuffler(seed).shuffle(unique)
    count = min(len(unique) - 1, max(1, round(fraction * len(unique))))
    held = set(unique[:count])
    fit = [i for i in rows if int(cluster_code[i]) not in held]
    hold = [i for i in rows if int(cluster_code[i]) in held]
    return fit, hold


def held_macro_f1(scores: Array, y: Array) -> float:
    """Macro F1 of ``scores`` on ``y`` with MCC-optimal thresholds tuned on those same rows.

    This is the early-stopping signal. Tuning on the rows it scores is optimistic (ADR-0011
    section 4); it only picks an epoch and is never reported.
    """
    np = importlib.import_module("numpy")
    thresholds = bm.tune_thresholds(scores, y)
    return bm.macro_f1(y, (scores >= np.array(thresholds)).astype(int))


def fold_rows(rows: Rows, fold: Fold) -> tuple[list[int], list[int]]:
    """Row indices of the training partitions and of the validation partition of a fold."""
    train = [i for i, p in enumerate(rows.partition) if int(p) in fold.train]
    held = [i for i, p in enumerate(rows.partition) if int(p) == fold.validate]
    return train, held


def mean_fold_macro_f1(
    oof: Array, rows: Rows, folds: Sequence[Fold], classes: Sequence[str]
) -> float:
    """Mean per-fold macro F1 of out-of-fold scores under nested thresholds (the selector)."""
    split = bm.Split(rows.y, rows.partition, rows.cluster_code)
    report = bm.validate_oof(oof, split, folds, classes, bm.Bootstrap(0, 0))
    return float(report["mean_fold_macro_f1"])


# -- the manifest ----------------------------------------------------------------------
_PIN_TEMPLATE = r"^\s*\"{name}==([0-9][^\"]*)\""


def pinned_version(pyproject_text: str, package: str) -> str:
    """The exact version of ``package`` pinned in pyproject.toml, or an error."""
    match = re.search(_PIN_TEMPLATE.format(name=re.escape(package)), pyproject_text, re.M)
    if match is None:
        raise BaselineError(f"pyproject.toml has no exact {package}== pin")
    return match.group(1)


@dataclass(frozen=True, slots=True)
class Pins:
    """What a head manifest is checked against; read from the repository, not from the manifest."""

    registry_sha256: str
    corpus_sha256: str
    embedding_cache_key: str
    residue_cache_key: str
    config_sha256: str
    classes: tuple[str, ...]
    torch: str


def _object(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise BaselineError(f"manifest is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def _check_inputs(frozen: Mapping[str, Any], inputs: Mapping[str, Any], pins: Pins) -> list[str]:
    problems = check_subset_block(frozen, cache_key=pins.embedding_cache_key)
    expected = {
        "registry_sha256": pins.registry_sha256,
        "corpus_sha256": pins.corpus_sha256,
        "residue_cache_key": pins.residue_cache_key,
        "head_config_sha256": pins.config_sha256,
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            problems.append(f"inputs.{key} differs from the repository")
    for key in ("feature_table_sha256", "class_map_sha256", "weights_sha256"):
        if not _SHA.fullmatch(str(inputs.get(key, ""))):
            problems.append(f"inputs.{key} is not a SHA-256")
    if frozen.get("classes") != list(pins.classes):
        problems.append("classes differ from baseline_class_map.yaml")
    thresholds = frozen.get("thresholds", [])
    if len(thresholds) != len(pins.classes):
        problems.append("one threshold per class is required")
    if not all(isinstance(t, float) and math.isfinite(t) for t in thresholds):
        problems.append("thresholds must be finite numbers")
    return problems


def check_head_manifest(manifest: Mapping[str, Any], pins: Pins) -> list[str]:
    """Every way the head manifest disagrees with itself or the repository; empty if sound."""
    try:
        frozen = _object(manifest, "frozen")
        protocol = _object(frozen, "protocol")
        inputs = _object(frozen, "inputs")
        selected = _object(frozen, "selected")
        software = _object(frozen, "software")
        validation = _object(frozen, "validation")
    except BaselineError as error:
        return [str(error)]
    problems = _check_inputs(frozen, inputs, pins)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version is not the one this code writes")
    if protocol.get("validation_partitions") != list(VALIDATION_PARTITIONS):
        problems.append("validation partitions are not [1, 2, 3, 4]")
    if protocol.get("test_partition") != TEST_PARTITION:
        problems.append("test partition is not 5")
    if not isinstance(selected.get("trainable_parameters"), int):
        problems.append("selected.trainable_parameters must be an integer")
    if not isinstance(selected.get("final_epochs"), int) or selected["final_epochs"] < 1:
        problems.append("selected.final_epochs must be a positive integer")
    if software.get("torch") != pins.torch:
        problems.append("torch version differs from the pyproject pin")
    if "pooled" not in validation:
        problems.append("no pooled validation metrics")
    if manifest.get("frozen_sha256") != digest(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
    test = manifest.get("test")
    if test is not None:
        problems += _check_test(test, str(manifest.get("frozen_sha256")), pins.classes)
    return problems


def _check_test(test: object, frozen_sha256: str, classes: Sequence[str]) -> list[str]:
    if not isinstance(test, dict):
        return ["test must be an object or null"]
    problems: list[str] = []
    if test.get("frozen_sha256") != frozen_sha256:
        problems.append("test was scored against a different frozen block")
    if test.get("touches") != 1:
        problems.append("the test partition must be touched exactly once")
    metrics = test.get("metrics")
    if not isinstance(metrics, dict) or set(metrics.get("per_class", {})) != set(classes):
        problems.append("test metrics do not cover every class")
    return problems
