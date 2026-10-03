"""Phase 8: classical baselines, the metric floor. Everything here runs without numpy.

``training/`` is an entry-point layer: nothing imports it. This module holds the parts
that need only the standard library, so CI can import and test them on a runner with no
numpy and no scikit-learn:

* the fixed-length motif vector (``vectorise``) and its names,
* the class map and the label and rule-score readers (ADR-0006 states only),
* the leave-one-partition-out folds and the protocol guards (ADR-0010),
* the frozen manifest: how it is built, hashed and checked.

The numerical half (metrics, thresholds, bootstrap, scikit-learn) is
``protloc.training.baseline_models`` and imports numpy lazily.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from protloc.data.manifest import ManifestError, canonical_json_bytes
from protloc.features.classical import Features, membrane_type, raw_scores

# -- protocol (ADR-0010) ---------------------------------------------------------------
# Registry partitions are numbered 1 to 5. configs/evaluation/default.yaml numbers them
# 0 to 4 (validation [0,1,2,3], test [4]); ADR-0010 records the +1 mapping and Phase 9
# owns that file.
VALIDATION_PARTITIONS: tuple[int, ...] = (1, 2, 3, 4)
TEST_PARTITION = 5
MIN_VALIDATION_PARTITIONS = 2
TASKS: tuple[str, ...] = ("localization", "membrane_type", "signal_peptide")
MODEL_NAMES: tuple[str, ...] = ("logreg", "linear_svc", "random_forest")
RULES_NAME = "motif_rules"
SCHEMA_VERSION = 1
VECTORISER_VERSION = "1"
# A class with fewer positives than this in the rows being scored has no defined MCC: it is
# reported as 0.0 and marked "undefined", never dropped (ADR-0011 section 2.4).
MIN_POSITIVES = 2
# The 500-sequence prototype scope (ADR-0009, ADR-0011). Every number computed on it carries
# this label.
SUBSET_SCOPE = "subset"
SUBSET_LABEL = "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"
# Residues at each terminus summarised by the hydropathy end means. A vectorisation
# parameter (how much of the profile one number covers), not a biological threshold.
TERMINAL_WINDOW = 30
N_ESTIMATORS = 200

# Small fixed grids, chosen by the validation folds only. Order is the tie-break.
GRIDS: dict[str, tuple[dict[str, float | int | None], ...]] = {
    "logreg": tuple({"C": c} for c in (0.01, 0.1, 1.0, 10.0)),
    "linear_svc": tuple({"C": c} for c in (0.01, 0.1, 1.0)),
    "random_forest": tuple(
        {"max_depth": depth, "min_samples_leaf": leaf} for depth in (8, 16) for leaf in (1, 5)
    ),
}

FEATURE_NAMES: tuple[str, ...] = (
    "log1p_length",
    "hydropathy_mean",
    "hydropathy_std",
    "hydropathy_min",
    "hydropathy_max",
    "hydropathy_argmax_fraction",
    "hydropathy_n_terminal_mean",
    "hydropathy_c_terminal_mean",
    "sp_score",
    "mtp_score",
    "ctp_score",
    "has_h_region",
    "h_region_start",
    "h_region_length",
    "has_cleavage",
    "cleavage_position",
    "n_tm_segments",
    "tm_total_length",
    "tm_first_start_fraction",
    "tm_last_end_fraction",
    "n_nls",
    "nls_first_start_fraction",
    "has_kdel",
    "has_pts1",
    "has_lpxtg",
    "has_myristoyl",
    "has_c_aromatic",
    "has_basic_tail",
    "n_go_hits",
)

_SHA = re.compile(r"[0-9a-f]{64}")
_PIN = re.compile(r"^\s*\"scikit-learn==([0-9][^\"]*)\"", re.M)


class BaselineError(ManifestError):
    """A Phase 8 input, label or manifest is inconsistent."""


# -- the motif vector ------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    centre = _mean(values)
    return math.sqrt(math.fsum((v - centre) ** 2 for v in values) / len(values))


def _fraction(position: int, length: int) -> float:
    return position / length if length else 0.0


def vectorise(f: Features) -> list[float]:
    """The motif Features as a fixed-length numeric vector (``FEATURE_NAMES`` order).

    Hydropathy is summarised by statistics, never passed as a profile, so the length is
    the same for every sequence. Motif presence is 0/1, motif counts are counts, and
    positions are fractions of the sequence length (0.0 when the motif is absent; the
    matching ``has_`` or count column says whether 0.0 means absent).
    """
    profile = f.hydropathy
    length = f.length
    window = min(TERMINAL_WINDOW, len(profile))
    argmax = max(range(len(profile)), key=profile.__getitem__) if profile else 0
    tm = sorted(f.tm_segments)
    h_start, h_end = f.h_region if f.h_region else (0, 0)
    vector = [
        math.log1p(length),
        _mean(profile),
        _std(profile),
        min(profile) if profile else 0.0,
        max(profile) if profile else 0.0,
        _fraction(argmax, len(profile)),
        _mean(profile[:window]),
        _mean(profile[len(profile) - window :]),
        f.sp_score,
        f.mtp_score,
        f.ctp_score,
        float(f.h_region is not None),
        float(h_start),
        float(h_end - h_start),
        float(f.cleavage is not None),
        float(f.cleavage or 0),
        float(len(tm)),
        float(sum(end - start for start, end in tm)),
        _fraction(tm[0][0], length) if tm else 0.0,
        _fraction(tm[-1][1], length) if tm else 0.0,
        float(len(f.nls)),
        _fraction(min(f.nls)[0], length) if f.nls else 0.0,
        float(f.kdel is not None),
        float(f.pts1 is not None),
        float(f.lpxtg is not None),
        float(f.myristoyl),
        float(f.c_aromatic),
        float(f.basic_tail),
        float(len(f.go_hits)),
    ]
    if len(vector) != len(FEATURE_NAMES):
        raise BaselineError("vectorise() and FEATURE_NAMES disagree on the vector length")
    if not all(math.isfinite(v) for v in vector):
        raise BaselineError("vectorise() produced a non-finite value")
    return vector


# -- class map, labels and rule scores -------------------------------------------------
@dataclass(frozen=True, slots=True)
class ClassMap:
    """``backend/rules/baseline_class_map.yaml``, parsed and checked."""

    version: str
    engine_group: str
    classes: Mapping[str, tuple[str, ...]]
    engine_keys: Mapping[str, Mapping[str, str]]
    from_registry: Mapping[str, str]
    negative_class: Mapping[str, str]
    positive_class: Mapping[str, str]
    sha256: str


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{where} must be a mapping")
    return {str(k): v for k, v in value.items()}


def load_class_map(path: Path) -> ClassMap:
    """Parse the class map and reject anything malformed before a label is read."""
    raw = path.read_bytes()
    doc = _mapping(yaml.safe_load(raw), str(path))
    tasks = _mapping(doc.get("tasks"), "tasks")
    if set(tasks) != set(TASKS):
        raise BaselineError(f"tasks must be exactly {TASKS}, found {sorted(tasks)}")
    classes: dict[str, tuple[str, ...]] = {}
    keys: dict[str, dict[str, str]] = {}
    negative: dict[str, str] = {}
    positive: dict[str, str] = {}
    from_registry: dict[str, str] = {}
    for name in TASKS:
        task = _mapping(tasks[name], f"tasks.{name}")
        names = tuple(str(c) for c in task["classes"])
        if len(set(names)) != len(names) or not names:
            raise BaselineError(f"tasks.{name}.classes must be unique and non-empty")
        classes[name] = names
        if "engine_keys" in task:
            mapped = {str(k): str(v) for k, v in _mapping(task["engine_keys"], name).items()}
            if set(mapped) != set(names):
                raise BaselineError(f"tasks.{name}.engine_keys must cover every class")
            keys[name] = mapped
        if "negative_class" in task:
            negative[name] = str(task["negative_class"])
        if "positive_class" in task:
            positive[name] = str(task["positive_class"])
    if "from_registry" in tasks["membrane_type"]:
        mapping = _mapping(tasks["membrane_type"]["from_registry"], "from_registry")
        from_registry = {str(k): str(v) for k, v in mapping.items()}
    unknown = set(from_registry.values()) - set(classes["membrane_type"])
    if unknown:
        raise BaselineError(f"from_registry targets unknown classes {sorted(unknown)}")
    return ClassMap(
        version=str(doc["version"]),
        engine_group=str(doc["engine_group"]),
        classes=classes,
        engine_keys=keys,
        from_registry=from_registry,
        negative_class=negative,
        positive_class=positive,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def label_vector(task: str, state: str, values: Sequence[str], cmap: ClassMap) -> list[int] | None:
    """The 0/1 label row for one record, or None when the record is not used.

    Only ``annotated`` rows count as positives. ``negative`` is read where ADR-0006 §3.3
    makes it an informative absence (``soluble``, ``no_signal_peptide``). ``unknown`` and
    ``not_applicable`` are never negatives: the record is dropped from the task.
    """
    names = cmap.classes[task]
    row = [0] * len(names)
    if task == "localization":
        if state != "annotated":
            return None
        for value in values:
            if value not in names:
                raise BaselineError(f"localization value {value!r} is not a class")
            row[names.index(value)] = 1
        return row
    if state == "annotated":
        if task == "signal_peptide":
            row[names.index(cmap.positive_class[task])] = 1
            return row
        for value in values:
            if value not in cmap.from_registry:
                raise BaselineError(f"membrane_type value {value!r} is not in from_registry")
            row[names.index(cmap.from_registry[value])] = 1
        return row
    if state == "negative":
        row[names.index(cmap.negative_class[task])] = 1
        return row
    return None


def rule_vector(task: str, f: Features, cmap: ClassMap) -> list[float]:
    """The untrained motif engine's score for every class of a task, never a model input."""
    names = cmap.classes[task]
    if task == "localization":
        scores = raw_scores(f, cmap.engine_group)
        return [float(scores.get(cmap.engine_keys[task][c], 0.0)) for c in names]
    if task == "membrane_type":
        scores = membrane_type(f)
        return [float(scores[cmap.engine_keys[task][c]]) for c in names]
    return [float(f.sp_score), 1.0 - float(f.sp_score)]


# -- the protocol (ADR-0010) -----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Fold:
    """Train on every validation partition except one; validate on that one."""

    train: tuple[int, ...]
    validate: int


def lopo_folds(
    partitions: Sequence[int] = VALIDATION_PARTITIONS, test: int = TEST_PARTITION
) -> list[Fold]:
    """Leave-one-partition-out folds over the validation partitions; the test never joins."""
    if test in partitions:
        raise BaselineError(f"test partition {test} is among the validation partitions")
    if len(set(partitions)) != len(partitions) or len(partitions) < MIN_VALIDATION_PARTITIONS:
        raise BaselineError("validation partitions must be at least two distinct numbers")
    return [Fold(tuple(p for p in partitions if p != held), held) for held in partitions]


def assert_protocol(folds: Sequence[Fold], test: int = TEST_PARTITION) -> None:
    """Raise unless no fold trains on its own validation partition or on the test."""
    for fold in folds:
        if fold.validate in fold.train:
            raise BaselineError(f"fold {fold.validate} trains on its own validation partition")
        if test in fold.train or fold.validate == test:
            raise BaselineError(f"fold {fold.validate} touches test partition {test}")


def grid_items(model: str) -> Iterator[tuple[int, dict[str, float | int | None]]]:
    """The hyperparameter grid of a model as (index, params)."""
    for index, params in enumerate(GRIDS[model]):
        yield index, dict(params)


# -- the 500-sequence subset -----------------------------------------------------------
def cache_lengths(cache_dir: Path) -> dict[str, int]:
    """Sequence hash and length of every sequence in an embedding cache directory.

    Reads only the committed ``shard_*.tsv`` tables (``sha256``, ``length``, ``windows``),
    never the vector bytes, so the subset is defined by what the cache actually holds.
    """
    found: dict[str, int] = {}
    for table in sorted(cache_dir.glob("shard_*.tsv")):
        for row in table.read_text(encoding="ascii").splitlines()[1:]:
            sha, length, _windows = row.split("\t")
            if not _SHA.fullmatch(sha):
                raise BaselineError(f"{table.name}: {sha!r} is not a sequence hash")
            found[sha] = int(length)
    if not found:
        raise BaselineError(f"{cache_dir} holds no committed shard tables")
    return found


def subset_digest(shas: Sequence[str]) -> str:
    """SHA-256 of the sorted sequence hashes that define the subset."""
    return hashlib.sha256("\n".join(sorted(shas)).encode("ascii")).hexdigest()


def check_subset_block(frozen: Mapping[str, Any], *, cache_key: str) -> list[str]:
    """Problems with the ``scope`` and ``subset`` entries of a subset manifest; empty if sound."""
    problems: list[str] = []
    if frozen.get("scope") != SUBSET_SCOPE:
        problems.append("a subset manifest must say scope 'subset'")
    if frozen.get("scope_label") != SUBSET_LABEL:
        problems.append("scope_label is not the required subset label")
    block = frozen.get("subset")
    if not isinstance(block, dict):
        return [*problems, "manifest is missing the object 'subset'"]
    if block.get("embedding_cache_key") != cache_key:
        problems.append("subset embedding_cache_key differs from the embedding manifest")
    if not _SHA.fullmatch(str(block.get("sequences_sha256", ""))):
        problems.append("subset.sequences_sha256 is not a SHA-256")
    if not isinstance(block.get("sequences"), int) or block["sequences"] <= 0:
        problems.append("subset.sequences must be a positive count")
    return problems


# -- the manifest ----------------------------------------------------------------------
def digest(obj: object) -> str:
    """SHA-256 of the canonical JSON bytes of an object."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def pinned_sklearn(pyproject_text: str) -> str:
    """The scikit-learn version pinned in pyproject.toml, or an error if it is not exact."""
    match = _PIN.search(pyproject_text)
    if match is None:
        raise BaselineError("pyproject.toml has no exact scikit-learn== pin")
    return match.group(1)


def _section(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise BaselineError(f"manifest is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def check_manifest(
    manifest: Mapping[str, Any],
    *,
    registry_sha256: str,
    corpus_sha256: str,
    sklearn_pin: str,
) -> list[str]:
    """Every way the baseline manifest disagrees with itself or the repo; empty if sound."""
    problems: list[str] = []
    try:
        frozen = _section(manifest, "frozen")
        protocol = _section(frozen, "protocol")
        inputs = _section(frozen, "inputs")
        software = _section(frozen, "software")
        tasks = _section(frozen, "tasks")
    except BaselineError as error:
        return [str(error)]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version is not the one this code writes")
    if protocol.get("validation_partitions") != list(VALIDATION_PARTITIONS):
        problems.append("validation partitions are not [1, 2, 3, 4]")
    if protocol.get("test_partition") != TEST_PARTITION:
        problems.append("test partition is not 5")
    if inputs.get("registry_sha256") != registry_sha256:
        problems.append("registry hash differs from the split manifest")
    if inputs.get("corpus_sha256") != corpus_sha256:
        problems.append("corpus hash differs from the dataset manifest")
    for key in ("feature_table_sha256", "engine_sha256", "class_map_sha256"):
        if not _SHA.fullmatch(str(inputs.get(key, ""))):
            problems.append(f"inputs.{key} is not a SHA-256")
    if software.get("scikit_learn") != sklearn_pin:
        problems.append("scikit-learn version differs from the pyproject pin")
    problems += _check_tasks(tasks)
    if manifest.get("frozen_sha256") != digest(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
    test = manifest.get("test")
    if test is not None:
        problems += _check_test(test, str(manifest.get("frozen_sha256")), tasks)
    return problems


def _check_tasks(tasks: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if set(tasks) != set(TASKS):
        problems.append(f"tasks are not exactly {list(TASKS)}")
    for name, body in tasks.items():
        classes = body.get("classes", [])
        models = body.get("models", {})
        if set(models) != set(MODEL_NAMES):
            problems.append(f"{name}: models are not exactly {list(MODEL_NAMES)}")
        rows = {**models, RULES_NAME: body.get("rules", {})}
        for model, entry in rows.items():
            thresholds = entry.get("thresholds", [])
            if len(thresholds) != len(classes):
                problems.append(f"{name}/{model}: one threshold per class is required")
            if not all(isinstance(t, float) and math.isfinite(t) for t in thresholds):
                problems.append(f"{name}/{model}: thresholds must be finite numbers")
            if "validation" not in entry:
                problems.append(f"{name}/{model}: no validation metrics")
    return problems


def _check_test(test: object, frozen_sha256: str, tasks: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(test, dict):
        return ["test must be an object or null"]
    if test.get("frozen_sha256") != frozen_sha256:
        problems.append("test was scored against a different frozen block")
    if test.get("touches") != 1:
        problems.append("the test partition must be touched exactly once")
    scored = test.get("tasks", {})
    for name, body in tasks.items():
        expected = {*body.get("models", {}), RULES_NAME}
        if set(scored.get(name, {})) != expected:
            problems.append(f"{name}: test metrics do not cover every row of the table")
    return problems


def read_manifest(path: Path) -> dict[str, Any]:
    """Read the baseline manifest as a JSON object."""
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise BaselineError(f"{path}: top level must be a JSON object")
    return {str(k): v for k, v in data.items()}
