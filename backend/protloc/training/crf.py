"""Phase 11 signal-peptide CRF head: configuration, labels, rows, metrics, protocol and manifest.

Everything here runs without torch. ``protloc.training.crf_net`` holds the network, the CRF and the
training loop and imports torch lazily; this module holds what CI can import and test on a
runner with neither torch nor a GPU:

* the configuration (``configs/training/crf.yaml``) parsed and checked, and the grid,
* the per-residue tag derivation from a ``cleavage_site`` (ADR-0006: the 1-based last residue of the
  peptide) and the label table read from the corpus,
* the rows the head is trained and scored on (annotated or negative only, ADR-0006),
* the metrics: detection F1, MCC and AUC; cleavage exact-match and within-tolerance accuracy; mean
  absolute cleavage error; per-residue F1; cluster-bootstrap intervals,
* the protocol (nested thresholds, the selection rule) and the frozen manifest checks (ADR-0013).

``training/`` is an entry-point layer: nothing imports it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from protloc.training import baseline_models as bm
from protloc.training.baselines import (
    FEATURE_NAMES,
    MIN_POSITIVES,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    Fold,
    check_subset_block,
    digest,
)
from protloc.training.head import Schedule, SmokeSettings

Array = Any
SCHEMA_VERSION = 1
PHASE = 11
TASK = "signal_peptide"
SOURCE_MODEL = "model"
SOURCE_MOTIF = "motif"
STATE_ANNOTATED = "annotated"
STATE_NEGATIVE = "negative"
#: The corpus tasks whose label state and cleavage site the label table carries.
LABEL_TASKS: tuple[str, ...] = (
    "signal_peptide",
    "mitochondrial_transit_peptide",
    "chloroplast_transit_peptide",
)
#: Order in which the grid expands: the first key varies slowest.
GRID_KEYS: tuple[str, ...] = ("encoder", "learning_rate", "dropout")
ENCODERS: tuple[str, ...] = ("cnn", "bilstm")
LABEL_HEADER = ("accession", "organism_group", "length")
NO_CLEAVAGE = ""
LENGTH_COLUMN = LABEL_HEADER.index("length")
#: A transition is a (from, to) pair.
PAIR_WIDTH = 2
#: Divisor for parity and for the mean-rank AUC terms.
EVEN_DIVISOR = 2
_SHA = re.compile(r"[0-9a-f]{64}")
_ACCESSION_KEY = b'"accession":"'


# -- configuration ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Architecture:
    """Network sizes and the length cap (configs/training/crf.yaml)."""

    embedding_dim: int
    length_cap: int
    cnn_hidden: int
    cnn_kernel: int
    cnn_layers: int
    lstm_hidden: int
    lstm_layers: int


@dataclass(frozen=True, slots=True)
class Tags:
    """The tag set, its transition constraints and the tasks marked undefined."""

    names: tuple[str, ...]
    outside: str
    signal: str
    forbidden: tuple[tuple[str, str], ...]
    forbidden_score: float
    undefined_tasks: tuple[str, ...]
    min_cleavage_positives: int

    @property
    def outside_index(self) -> int:
        """Index of the outside tag."""
        return self.names.index(self.outside)

    @property
    def signal_index(self) -> int:
        """Index of the signal tag."""
        return self.names.index(self.signal)

    def allowed(self) -> list[list[bool]]:
        """``allowed[i][j]`` is False where the transition tag i -> tag j is forbidden."""
        banned = {(self.names.index(a), self.names.index(b)) for a, b in self.forbidden}
        size = len(self.names)
        return [[(i, j) not in banned for j in range(size)] for i in range(size)]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """How the head is scored."""

    tolerances: tuple[int, ...]
    group_min_support: int
    reproduction_tolerance: float


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One hyperparameter setting."""

    encoder: str
    learning_rate: float
    dropout: float

    def as_dict(self) -> dict[str, float | str]:
        """The setting as plain JSON-able values, in grid order."""
        return {
            "encoder": self.encoder,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
        }


@dataclass(frozen=True, slots=True)
class CrfConfig:
    """``configs/training/crf.yaml``, parsed and checked."""

    seed: int
    architecture: Architecture
    tags: Tags
    grid: Mapping[str, tuple[Any, ...]]
    weight_decay: float
    schedule: Schedule
    selection_tolerance: int
    evaluation: Evaluation
    bundle_version: str
    bundle_directory: str
    calibration_config: str
    targetp: Mapping[str, Any]
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


def _names(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise BaselineError(f"{where} must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise BaselineError(f"{where} must not repeat a name")
    return tuple(value)


def _tags(doc: Mapping[str, Any]) -> Tags:
    names = _names(doc.get("names"), "tags.names")
    outside, signal = str(doc.get("outside")), str(doc.get("signal"))
    if outside not in names or signal not in names or outside == signal:
        raise BaselineError("tags.outside and tags.signal must be two different tag names")
    pairs = doc.get("forbidden_transitions")
    if not isinstance(pairs, list) or not all(
        isinstance(p, list) and len(p) == PAIR_WIDTH and set(p) <= set(names) for p in pairs
    ):
        raise BaselineError("tags.forbidden_transitions must be a list of [from, to] tag pairs")
    score = doc.get("forbidden_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not score < 0:
        raise BaselineError("tags.forbidden_score must be a negative number")
    return Tags(
        names=names,
        outside=outside,
        signal=signal,
        forbidden=tuple((str(a), str(b)) for a, b in pairs),
        forbidden_score=float(score),
        undefined_tasks=_names(doc.get("undefined_tasks"), "tags.undefined_tasks"),
        min_cleavage_positives=_count(
            doc.get("min_cleavage_positives"), "tags.min_cleavage_positives"
        ),
    )


def _architecture(doc: Mapping[str, Any]) -> Architecture:
    cnn = _mapping(doc.get("cnn"), "architecture.cnn")
    lstm = _mapping(doc.get("bilstm"), "architecture.bilstm")
    kernel = _count(cnn.get("kernel"), "architecture.cnn.kernel")
    if kernel % EVEN_DIVISOR == 0:
        raise BaselineError("architecture.cnn.kernel must be odd so that 'same' padding is exact")
    return Architecture(
        embedding_dim=_count(doc.get("embedding_dim"), "architecture.embedding_dim"),
        length_cap=_count(doc.get("length_cap"), "architecture.length_cap"),
        cnn_hidden=_count(cnn.get("hidden"), "architecture.cnn.hidden"),
        cnn_kernel=kernel,
        cnn_layers=_count(cnn.get("layers"), "architecture.cnn.layers"),
        lstm_hidden=_count(lstm.get("hidden"), "architecture.bilstm.hidden"),
        lstm_layers=_count(lstm.get("layers"), "architecture.bilstm.layers"),
    )


def _grid(doc: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    if set(doc) != set(GRID_KEYS):
        raise BaselineError(f"grid must have exactly the keys {list(GRID_KEYS)}")
    grid: dict[str, tuple[Any, ...]] = {}
    for key in GRID_KEYS:
        values = doc[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise BaselineError(f"grid.{key} must be a non-empty list of distinct values")
        grid[key] = tuple(values)
    if not set(grid["encoder"]) <= set(ENCODERS):
        raise BaselineError(f"grid.encoder values must be among {list(ENCODERS)}")
    if any(not 0.0 <= float(d) < 1.0 for d in grid["dropout"]):
        raise BaselineError("grid.dropout values must be in [0, 1)")
    if any(float(lr) <= 0.0 for lr in grid["learning_rate"]):
        raise BaselineError("grid.learning_rate values must be positive")
    return grid


def _evaluation(doc: Mapping[str, Any], selected: int) -> Evaluation:
    tolerances = doc.get("tolerances")
    if (
        not isinstance(tolerances, list)
        or not tolerances
        or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 1 for t in tolerances)
        or sorted(set(tolerances)) != tolerances
    ):
        raise BaselineError("evaluation.tolerances must be increasing positive integers")
    if selected not in tolerances:
        raise BaselineError("selection.tolerance must be one of evaluation.tolerances")
    return Evaluation(
        tolerances=tuple(tolerances),
        group_min_support=_count(doc.get("group_min_support"), "evaluation.group_min_support"),
        reproduction_tolerance=_positive(
            doc.get("reproduction_tolerance"), "evaluation.reproduction_tolerance"
        ),
    )


def load_config(path: Path) -> CrfConfig:
    """Parse the CRF configuration and reject anything malformed before a fit starts."""
    raw = path.read_bytes()
    doc = _mapping(yaml.safe_load(raw), str(path))
    if doc.get("name") != "crf" or doc.get("task") != TASK:
        raise BaselineError("crf.yaml must be name 'crf' for task 'signal_peptide'")
    if _mapping(doc.get("optimiser"), "optimiser").get("name") != "adamw":
        raise BaselineError("optimiser.name must be 'adamw'")
    schedule = _mapping(doc.get("schedule"), "schedule")
    fraction = _positive(schedule.get("inner_holdout_fraction"), "schedule.inner_holdout_fraction")
    if fraction >= 1.0:
        raise BaselineError("schedule.inner_holdout_fraction must be below 1")
    selected = _count(_mapping(doc.get("selection"), "selection").get("tolerance"), "tolerance")
    bundle = _mapping(doc.get("bundle"), "bundle")
    smoke = _mapping(doc.get("smoke"), "smoke")
    context = _mapping(doc.get("context"), "context")
    return CrfConfig(
        seed=_count(doc.get("seed"), "seed"),
        architecture=_architecture(_mapping(doc.get("architecture"), "architecture")),
        tags=_tags(_mapping(doc.get("tags"), "tags")),
        grid=_grid(_mapping(doc.get("grid"), "grid")),
        weight_decay=_positive(
            _mapping(doc.get("optimiser"), "optimiser").get("weight_decay"), "weight_decay"
        ),
        schedule=Schedule(
            batch_size=_count(schedule.get("batch_size"), "schedule.batch_size"),
            max_epochs=_count(schedule.get("max_epochs"), "schedule.max_epochs"),
            patience=_count(schedule.get("patience"), "schedule.patience"),
            inner_holdout_fraction=fraction,
        ),
        selection_tolerance=selected,
        evaluation=_evaluation(_mapping(doc.get("evaluation"), "evaluation"), selected),
        bundle_version=str(bundle.get("version", "")),
        bundle_directory=str(bundle.get("directory", "")),
        calibration_config=str(_mapping(doc.get("calibration"), "calibration").get("config", "")),
        targetp=_mapping(context.get("targetp"), "context.targetp"),
        smoke=SmokeSettings(
            stride=_count(smoke.get("stride"), "smoke.stride"),
            max_epochs=_count(smoke.get("max_epochs"), "smoke.max_epochs"),
            grid_points=_count(smoke.get("grid_points"), "smoke.grid_points"),
        ),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def grid_points(config: CrfConfig) -> list[GridPoint]:
    """Every grid point in expansion order (the first key varies slowest)."""
    return [
        GridPoint(encoder=str(encoder), learning_rate=float(lr), dropout=float(dropout))
        for encoder, lr, dropout in itertools.product(*(config.grid[k] for k in GRID_KEYS))
    ]


# -- tags from labels ------------------------------------------------------------------
def tags_from_cleavage(length: int, cleavage: int, cap: int, tags: Tags) -> list[int]:
    """Per-residue tag indices of one sequence: residues 1..cleavage are signal, the rest outside.

    ``cleavage`` is the 1-based last residue of the peptide (ADR-0006); 0 means no peptide. The
    path covers the first ``min(length, cap)`` residues, so a residue beyond the cap is not in it
    (the decoder tags it outside) and a cleavage beyond the cap simply fills the whole path.
    """
    width = min(length, cap)
    return [
        tags.signal_index if position < cleavage else tags.outside_index
        for position in range(width)
    ]


def cleavage_of_path(path: Sequence[int], signal_index: int) -> int:
    """The 1-based last residue tagged signal in a decoded path (0 if none)."""
    last = 0
    for position, tag in enumerate(path, start=1):
        if tag == signal_index:
            last = position
    return last


# -- the label table -------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LabelRow:
    """What the corpus says about one accession: its group, length and three peptide tasks."""

    accession: str
    group: str
    length: int
    states: Mapping[str, str]
    cleavage: Mapping[str, int | None]


def _label_columns() -> tuple[str, ...]:
    return LABEL_HEADER + tuple(
        column for task in LABEL_TASKS for column in (f"{task}_state", f"{task}_cleavage")
    )


def _record_row(record: Mapping[str, Any]) -> LabelRow:
    labels = record["labels"]
    return LabelRow(
        accession=str(record["accession"]),
        group=str(record["organism_group"]),
        length=int(record["length"]),
        states={task: str(labels[task]["state"]) for task in LABEL_TASKS},
        cleavage={
            task: None
            if labels[task]["cleavage_site"] is None
            else int(labels[task]["cleavage_site"])
            for task in LABEL_TASKS
        },
    )


def scan_corpus(path: Path, accessions: Collection[str]) -> dict[str, LabelRow]:
    """Stream the canonical corpus once and keep the requested accessions (pure Python).

    A line is parsed only when its accession is wanted, so a 1 GB corpus costs one pass of byte
    searches. Every requested accession must be found or the call raises.
    """
    wanted = set(accessions)
    found: dict[str, LabelRow] = {}
    with path.open("rb") as handle:
        for line in handle:
            start = line.find(_ACCESSION_KEY)
            if start < 0:
                continue
            start += len(_ACCESSION_KEY)
            accession = line[start : line.find(b'"', start)].decode("ascii")
            if accession in wanted:
                row = _record_row(json.loads(line))
                found[row.accession] = row
    missing = sorted(wanted - set(found))
    if missing:
        raise BaselineError(f"{len(missing)} accessions are not in the corpus, e.g. {missing[0]}")
    return found


def label_table_text(labels: Mapping[str, LabelRow]) -> str:
    """The label table as TSV text, sorted by accession (deterministic bytes)."""
    lines = ["\t".join(_label_columns())]
    for accession in sorted(labels):
        row = labels[accession]
        cells = [row.accession, row.group, str(row.length)]
        for task in LABEL_TASKS:
            site = row.cleavage[task]
            cells += [row.states[task], NO_CLEAVAGE if site is None else str(site)]
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def parse_label_table(text: str) -> dict[str, LabelRow]:
    """Read a table written by :func:`label_table_text`; anything else is refused."""
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != _label_columns():
        raise BaselineError("the label table has an unexpected header")
    labels: dict[str, LabelRow] = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(_label_columns()):
            raise BaselineError("a label table row has the wrong number of columns")
        states: dict[str, str] = {}
        cleavage: dict[str, int | None] = {}
        for k, task in enumerate(LABEL_TASKS):
            first = len(LABEL_HEADER) + PAIR_WIDTH * k
            states[task] = cells[first]
            cleavage[task] = None if cells[first + 1] == NO_CLEAVAGE else int(cells[first + 1])
        labels[cells[0]] = LabelRow(cells[0], cells[1], int(cells[LENGTH_COLUMN]), states, cleavage)
    return labels


def label_counts(labels: Mapping[str, LabelRow], partition_of: Mapping[str, int]) -> dict[str, Any]:
    """Per task: label states, and positives that carry a cleavage site by partition and group."""
    report: dict[str, Any] = {}
    for task in LABEL_TASKS:
        states: dict[str, int] = {}
        by_partition: dict[str, int] = {}
        by_group: dict[str, int] = {}
        without_site = 0
        for accession, row in labels.items():
            state = row.states[task]
            states[state] = states.get(state, 0) + 1
            if state != STATE_ANNOTATED:
                continue
            if row.cleavage[task] is None:
                without_site += 1
                continue
            key = str(partition_of[accession])
            by_partition[key] = by_partition.get(key, 0) + 1
            by_group[row.group] = by_group.get(row.group, 0) + 1
        validation = sum(n for p, n in by_partition.items() if int(p) in VALIDATION_PARTITIONS)
        report[task] = {
            "states": dict(sorted(states.items())),
            "positives_with_cleavage": sum(by_partition.values()),
            "positives_with_cleavage_validation": validation,
            "positives_without_cleavage": without_site,
            "by_partition": dict(sorted(by_partition.items())),
            "by_group": dict(sorted(by_group.items())),
        }
    return report


# -- rows ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RowSpec:
    """Which rows to build: the partitions, every stride-th, and the length cap."""

    partitions: Sequence[int]
    stride: int
    cap: int


@dataclass(frozen=True, slots=True)
class Rows:
    """The labelled accessions used. ``cleavage`` is the 1-based last peptide residue (0 = none).

    ``tagged`` is False for a positive without a usable cleavage site: its tag loss is masked and
    only detection is learned from it.
    """

    accession: Sequence[str]
    shas: Sequence[str]
    group: Sequence[str]
    seq_index: Array
    length: Array
    cleavage: Array
    y: Array
    tagged: Array
    partition: Array
    cluster_code: Array

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.shas)


def build_rows(
    table: Mapping[str, Array],
    order: Sequence[str],
    labels: Mapping[str, LabelRow],
    spec: RowSpec,
) -> Rows:
    """The annotated-or-negative signal-peptide rows of the given partitions (ADR-0006).

    ``order`` is the distinct sequence hashes in the feature table's own order. The corpus label
    table and the feature table must agree on every row's detection label or the call raises.
    """
    np = _np()
    index = bm.task_rows(table, TASK, spec.partitions, spec.stride).tolist()
    cleavage, y, tagged, group, length = [], [], [], [], []
    for i in index:
        record = labels.get(str(table["accession"][i]))
        if record is None:
            raise BaselineError(f"no label row for {table['accession'][i]}")
        state, site = record.states[TASK], record.cleavage[TASK]
        positive = state == STATE_ANNOTATED
        if not positive and state != STATE_NEGATIVE:
            raise BaselineError(f"{record.accession}: state {state!r} should have been masked")
        if bool(table[f"Y_{TASK}"][i, 0]) != positive:
            raise BaselineError(f"{record.accession}: the label table and the feature table differ")
        usable = site is not None and 1 <= site <= spec.cap
        y.append(int(positive))
        cleavage.append(int(site or 0))
        tagged.append(not positive or usable)
        group.append(record.group)
        length.append(record.length)
    seq_index = table["seq_index"][index]
    return Rows(
        accession=[str(table["accession"][i]) for i in index],
        shas=[order[int(s)] for s in seq_index],
        group=group,
        seq_index=seq_index,
        length=np.array(length, dtype=np.int64),
        cleavage=np.array(cleavage, dtype=np.int64),
        y=np.array(y, dtype=np.int64),
        tagged=np.array(tagged, dtype=bool),
        partition=table["partition"][index],
        cluster_code=table["cluster_code"][index],
    )


def fold_rows(rows: Rows, fold: Fold) -> tuple[list[int], list[int]]:
    """Row indices of the training partitions and of the validation partition of a fold."""
    train = [i for i, p in enumerate(rows.partition) if int(p) in fold.train]
    held = [i for i, p in enumerate(rows.partition) if int(p) == fold.validate]
    return train, held


def clusters_of(rows: Rows, indices: Sequence[int]) -> set[int]:
    """The cluster codes of the given rows."""
    return {int(rows.cluster_code[i]) for i in indices}


# -- the motif floor -------------------------------------------------------------------
def motif_arrays(table: Mapping[str, Array], rows: Rows) -> tuple[Array, Array]:
    """The untrained motif detector's score and cleavage site for each row (floor, never trained).

    Score is the engine's signal-peptide score; cleavage is the h-region detector's AxA-like
    site (0 where it found none), already in the 1-based last-residue convention.
    """
    column = FEATURE_NAMES.index("cleavage_position")
    score = table[f"R_{TASK}"][rows.seq_index, 0]
    cleavage = table["X"][rows.seq_index, column].round().astype("int64")
    return score, cleavage


# -- metrics ---------------------------------------------------------------------------
def _np() -> Any:
    return importlib.import_module("numpy")


@dataclass(frozen=True, slots=True)
class Scored:
    """Predictions aligned to the rows: score, decision and predicted cleavage (0 = none)."""

    score: Array
    decision: Array
    cleavage: Array


def auc_score(y: Array, score: Array) -> float:
    """Rank-based area under the ROC curve (Mann-Whitney, ties share their mean rank)."""
    np = _np()
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / EVEN_DIVISOR)[inverse]
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / EVEN_DIVISOR) / (positives * negatives)


def metric_names(tolerances: Sequence[int]) -> list[str]:
    """Names of every metric :func:`compute_metrics` returns."""
    cleavage = ["exact", *(f"within_{t}" for t in tolerances), "mae"]
    return [
        "detection_f1",
        "detection_mcc",
        "detection_auc",
        *(f"cleavage_{c}" for c in cleavage),
        *(f"cleavage_detected_{c}" for c in cleavage),
        "residue_f1",
    ]


def _cleavage_metrics(error: Array, tolerances: Sequence[int], prefix: str) -> dict[str, float]:
    np = _np()
    if len(error) < MIN_POSITIVES:
        return {
            f"{prefix}_{name}": 0.0
            for name in ["exact", *(f"within_{t}" for t in tolerances), "mae"]
        }
    values = {f"{prefix}_exact": float(np.mean(error == 0))}
    for t in tolerances:
        values[f"{prefix}_within_{t}"] = float(np.mean(error <= t))
    values[f"{prefix}_mae"] = float(np.mean(error))
    return values


def compute_metrics(
    rows: Rows, scored: Scored, idx: Array, tolerances: Sequence[int]
) -> dict[str, float]:
    """Every metric on the rows ``idx``.

    Undefined rule: with fewer than ``MIN_POSITIVES`` positives, detection MCC and AUC and the
    cleavage metrics are 0.0 (the report marks them undefined). Cleavage metrics use the true
    positives that carry a usable cleavage site; ``cleavage_detected_*`` further restricts them
    to positives the decision also calls positive. A predicted cleavage of 0 is position 0.
    """
    np = _np()
    y = rows.y[idx]
    decision = scored.decision[idx].astype(int)
    tp, fp = int(((y == 1) & (decision == 1)).sum()), int(((y == 0) & (decision == 1)).sum())
    fn, tn = int(((y == 1) & (decision == 0)).sum()), int(((y == 0) & (decision == 0)).sum())
    defined = int(y.sum()) >= MIN_POSITIVES
    metrics = {
        "detection_f1": float(bm.f1_from_counts(tp, fp, fn)),
        "detection_mcc": float(bm.mcc_from_counts(tp, fp, fn, tn)) if defined else 0.0,
        "detection_auc": auc_score(y, scored.score[idx]) if defined else 0.0,
    }
    cleaved = (y == 1) & rows.tagged[idx]
    error = np.abs(scored.cleavage[idx][cleaved] - rows.cleavage[idx][cleaved])
    metrics.update(_cleavage_metrics(error, tolerances, "cleavage"))
    caught = decision[cleaved] == 1
    metrics.update(_cleavage_metrics(error[caught], tolerances, "cleavage_detected"))
    kept = rows.tagged[idx]
    truth = rows.cleavage[idx][kept]
    guess = np.where(decision[kept] == 1, scored.cleavage[idx][kept], 0)
    overlap = float(np.minimum(truth, guess).sum())
    extra = float(np.maximum(guess - truth, 0).sum())
    missed = float(np.maximum(truth - guess, 0).sum())
    metrics["residue_f1"] = float(bm.f1_from_counts(overlap, extra, missed))
    return metrics


def _counts_of(rows: Rows, idx: Array) -> dict[str, int]:
    y = rows.y[idx]
    cleaved = (y == 1) & rows.tagged[idx]
    return {
        "rows": len(idx),
        "positives": int(y.sum()),
        "positives_with_cleavage": int(cleaved.sum()),
    }


def metric_report(
    rows: Rows,
    scored: Scored,
    idx: Array,
    settings: tuple[Evaluation, bm.Bootstrap],
) -> dict[str, Any]:
    """Point values with percentile intervals, bootstrapped over whole ``cluster_id`` clusters."""
    np = _np()
    evaluation, boot = settings
    point = compute_metrics(rows, scored, idx, evaluation.tolerances)
    counts = _counts_of(rows, idx)
    draws: dict[str, list[float]] = {name: [] for name in point}
    if boot.samples > 0:
        _, inverse = np.unique(rows.cluster_code[idx], return_inverse=True)
        members = [idx[inverse == k] for k in range(int(inverse.max()) + 1)]
        rng = np.random.default_rng(boot.seed)
        for _ in range(boot.samples):
            picks = rng.integers(0, len(members), size=len(members))
            chosen = np.concatenate([members[int(p)] for p in picks])
            for name, value in compute_metrics(rows, scored, chosen, evaluation.tolerances).items():
                draws[name].append(value)
    metrics: dict[str, Any] = {}
    for name, value in point.items():
        low = high = None
        if draws[name]:
            low, high = (float(v) for v in np.percentile(draws[name], bm.CI_PERCENTILES))
        metrics[name] = {"value": float(value), "ci_low": low, "ci_high": high}
    return {
        **counts,
        "undefined": {
            "detection": counts["positives"] < MIN_POSITIVES,
            "cleavage": counts["positives_with_cleavage"] < MIN_POSITIVES,
        },
        "metrics": metrics,
        "bootstrap": {"samples": boot.samples, "seed": boot.seed, "unit": "cluster_id"},
    }


def group_reports(
    rows: Rows, scored: Scored, settings: tuple[Evaluation, bm.Bootstrap]
) -> dict[str, Any]:
    """One report per organism group with enough positives; the others are listed, not scored."""
    np = _np()
    evaluation, _boot = settings
    out: dict[str, Any] = {}
    for group in sorted(set(rows.group)):
        idx = np.array([i for i, g in enumerate(rows.group) if g == group])
        positives = int(rows.y[idx].sum())
        if positives >= evaluation.group_min_support:
            out[group] = {"reported": True, **metric_report(rows, scored, idx, settings)}
        else:
            out[group] = {
                "reported": False,
                "positives": positives,
                "reason": f"fewer than {evaluation.group_min_support} positives",
            }
    return out


# -- protocol --------------------------------------------------------------------------
def nested_decisions(
    score: Array, rows: Rows, folds: Sequence[Fold]
) -> tuple[Array, dict[int, float], float]:
    """Decisions in which fold k's threshold was tuned on the other partitions' scores only.

    Returns the decisions, the per-fold thresholds and the frozen threshold (tuned on every
    validation row), each the MCC-optimal score (``score >= threshold`` is positive).
    """
    np = _np()
    decision = np.zeros(len(rows), dtype=bool)
    fold_thresholds: dict[int, float] = {}
    for fold in folds:
        held = rows.partition == fold.validate
        threshold = bm.best_threshold(score[~held], rows.y[~held])
        decision[held] = score[held] >= threshold
        fold_thresholds[fold.validate] = float(threshold)
    return decision, fold_thresholds, float(bm.best_threshold(score, rows.y))


def fold_cleavage_accuracy(rows: Rows, cleavage: Array, validate: int, tolerance: int) -> float:
    """Share of a fold's true positives (usable cleavage site) predicted within ``tolerance``."""
    np = _np()
    keep = (rows.partition == validate) & (rows.y == 1) & rows.tagged
    if not keep.any():
        return 0.0
    return float(np.mean(np.abs(cleavage[keep] - rows.cleavage[keep]) <= tolerance))


def selection_value(rows: Rows, cleavage: Array, folds: Sequence[Fold], tolerance: int) -> float:
    """The grid selector: mean over folds of cleavage accuracy within the tolerance (Gate C)."""
    values = [fold_cleavage_accuracy(rows, cleavage, f.validate, tolerance) for f in folds]
    return math.fsum(values) / len(values)


def mean_fold_mcc(rows: Rows, score: Array, folds: Sequence[Fold]) -> float:
    """Mean over folds of detection MCC under nested thresholds (the selector's tie-break)."""
    np = _np()
    decision, _thresholds, _frozen = nested_decisions(score, rows, folds)
    values = []
    for fold in folds:
        held = rows.partition == fold.validate
        y, p = rows.y[held], decision[held].astype(int)
        tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
        fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
        defined = int(y.sum()) >= MIN_POSITIVES
        values.append(float(bm.mcc_from_counts(tp, fp, fn, tn)) if defined else 0.0)
    return float(np.mean(values))


def select_point(values: Sequence[tuple[float, float]]) -> int:
    """Index of the best grid point: higher selector, then higher MCC, then the earlier point."""
    return max(range(len(values)), key=lambda i: (values[i][0], values[i][1], -i))


# -- the manifest ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Pins:
    """What a CRF manifest is checked against; read from the repository, not from the manifest."""

    registry_sha256: str
    corpus_sha256: str
    embedding_cache_key: str
    residue_cache_key: str
    config_sha256: str
    calibration_config_sha256: str
    tags: Tags
    tolerances: tuple[int, ...]
    reproduction_tolerance: float
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
        "crf_config_sha256": pins.config_sha256,
        "calibration_config_sha256": pins.calibration_config_sha256,
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            problems.append(f"inputs.{key} differs from the repository")
    for key in (
        "feature_table_sha256",
        "label_table_sha256",
        "weights_sha256",
        "training_record_sha256",
        "calibrator_sha256",
    ):
        if not _SHA.fullmatch(str(inputs.get(key, ""))):
            problems.append(f"inputs.{key} is not a SHA-256")
    return problems


def _check_scope(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    scope = frozen.get("tag_scope")
    if not isinstance(scope, dict):
        return ["manifest is missing the object 'tag_scope'"]
    problems: list[str] = []
    if scope.get("tags") != list(pins.tags.names):
        problems.append("tag_scope.tags differs from crf.yaml")
    undefined = scope.get("undefined")
    if not isinstance(undefined, dict) or set(undefined) != set(pins.tags.undefined_tasks):
        return [*problems, "tag_scope.undefined does not name exactly the undefined tasks"]
    for task, entry in undefined.items():
        count = entry.get("positives_with_cleavage_validation") if isinstance(entry, dict) else None
        if not isinstance(count, int) or count >= pins.tags.min_cleavage_positives:
            problems.append(f"{task} is marked undefined but has enough positives to train")
        if isinstance(entry, dict) and entry.get("trained") is not False:
            problems.append(f"{task} must be recorded as not trained")
    return problems


def _check_report(report: object, name: str, pins: Pins) -> list[str]:
    if not isinstance(report, dict) or not isinstance(report.get("pooled"), dict):
        return [f"validation.{name} has no pooled report"]
    metrics = report["pooled"].get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(metric_names(pins.tolerances)):
        return [f"validation.{name} does not report every metric"]
    return []


def _check_validation(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    validation = frozen.get("validation")
    if not isinstance(validation, dict):
        return ["manifest is missing the object 'validation'"]
    problems: list[str] = []
    for name in ("crf_raw", "crf_calibrated", "motif_floor"):
        problems += _check_report(validation.get(name), name, pins)
    reproduction = validation.get("reproduction")
    if not isinstance(reproduction, dict):
        return [*problems, "validation.reproduction is missing"]
    difference = reproduction.get("max_abs_difference")
    if not isinstance(difference, float) or not difference <= pins.reproduction_tolerance:
        problems.append("the raw detection scores did not reproduce from the saved weights")
    if reproduction.get("decisions_identical") is not True:
        problems.append("the raw detection decisions did not reproduce from the saved weights")
    return problems


def check_crf_manifest(manifest: Mapping[str, Any], pins: Pins) -> list[str]:
    """Every way the CRF manifest disagrees with itself or the repository; empty if sound."""
    try:
        frozen = _object(manifest, "frozen")
        protocol = _object(frozen, "protocol")
        inputs = _object(frozen, "inputs")
        selected = _object(frozen, "selected")
        software = _object(frozen, "software")
        thresholds = _object(frozen, "thresholds")
    except BaselineError as error:
        return [str(error)]
    problems = _check_inputs(frozen, inputs, pins) + _check_scope(frozen, pins)
    problems += _check_validation(frozen, pins)
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
    for key in ("crf_raw", "crf_calibrated", "motif"):
        value = thresholds.get(key)
        if not isinstance(value, float) or not math.isfinite(value):
            problems.append(f"thresholds.{key} must be a finite number")
    if software.get("torch") != pins.torch:
        problems.append("torch version differs from the pyproject pin")
    if manifest.get("frozen_sha256") != digest(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
    test = manifest.get("test")
    if test is not None:
        problems += _check_test(test, str(manifest.get("frozen_sha256")), pins)
    return problems


def _check_test(test: object, frozen_sha256: str, pins: Pins) -> list[str]:
    if not isinstance(test, dict):
        return ["test must be an object or null"]
    problems: list[str] = []
    if test.get("frozen_sha256") != frozen_sha256:
        problems.append("test was scored against a different frozen block")
    if test.get("touches") != 1:
        problems.append("the test partition must be touched exactly once")
    metrics = test.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "crf_raw",
        "crf_calibrated",
        "motif_floor",
    }:
        return [*problems, "test metrics do not cover the CRF, the calibrated CRF and the floor"]
    for name, report in metrics.items():
        if not isinstance(report, dict) or set(report.get("metrics", {})) != set(
            metric_names(pins.tolerances)
        ):
            problems.append(f"test.{name} does not report every metric")
    return problems
