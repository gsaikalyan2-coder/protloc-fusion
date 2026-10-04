"""Phase 13 homology evidence: configuration, scoring under the Phase 12 protocol, manifest checks.

Everything here runs without torch. The GO label machinery, the metrics, the nested thresholds, the
selection rule and the cluster bootstrap are the Phase 12 ones and are imported from
``protloc.training.go``, not copied. This module adds only what homology needs:

* ``configs/training/homology.yaml`` parsed and checked (the masks cannot be switched off, the
  tie-break weight cannot outrank a vote, the search e-value cannot be tighter than the grid),
* the vote matrix of a hit filter, the grid search over the filters on partitions 1-4 only, the
  per-identity bins and the coverage figures,
* the frozen manifest checks (ADR-0015).

Independence: nothing in this module reads a model score to build a homology score. The model arms
(raw and calibrated Phase 12 scores) are passed in beside the homology arms to be compared and are
never an input to them; ``tests/unit/test_homology.py`` fails if that changes.

``training/`` is an entry-point layer: nothing imports it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import itertools
import math
from pathlib import Path
import re
from typing import Any

import yaml

from protloc.evidence import homology
from protloc.evidence.homology import Hit, HitFilter
from protloc.training import go
from protloc.training.baselines import (
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    Fold,
    digest,
)

Array = Any
PHASE = 13
SCHEMA_VERSION = 1
TASK = "homology_evidence"
GRID_KEYS: tuple[str, ...] = ("e_value", "min_identity", "min_query_coverage", "top_k")
ARM_VOTE = "homology_vote"
ARM_BEST = "homology_best"
ARM_REGISTRY = "homology_registry_only"
ARM_PRIOR = "frequency_prior"
ARM_RAW = "model_raw"
ARM_CALIBRATED = "model_calibrated"
MODEL_ARMS: tuple[str, ...] = (ARM_RAW, ARM_CALIBRATED)
NO_HIT_BIN = "no_hit"
#: What the layer promises: none of these may ever be true (a test and the manifest check say so).
INDEPENDENCE: dict[str, bool] = {
    "feeds_model_score": False,
    "feeds_calibrator": False,
    "feeds_threshold": False,
    "homology_arms_read_model_scores": False,
}
EARLIER_HEADS: tuple[str, ...] = ("localization", "crf", "go")
_SHA = re.compile(r"[0-9a-f]{64}")
_UNIT = (0.0, 1.0)
_MIN_EDGES = 2


def _np() -> Any:
    return importlib.import_module("numpy")


# -- configuration ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DatabasePin:
    """The Swiss-Prot snapshot the index is built from (the training corpus's release)."""

    source: str
    release: str
    records: int
    corpus_file: str
    corpus_sha256: str
    license: str
    license_url: str
    directory: str
    fasta: str
    meta: str
    mmseqs_directory: str
    mmseqs_name: str


@dataclass(frozen=True, slots=True)
class EnginePin:
    """The MMseqs2 binary: release, commit, hashes and how many threads it may use."""

    name: str
    release: str
    commit: str
    platform: str
    archive_sha256: str
    executable_sha256: str
    executable: str
    license: str
    license_url: str
    download_url: str
    threads: int
    verbosity: int


@dataclass(frozen=True, slots=True)
class TransferRule:
    """How hits become term scores and how the per-identity report is cut."""

    annotated_hits_only: bool
    tiebreak_weight: float
    identity_bins: tuple[float, ...]
    min_rows_per_bin: int


@dataclass(frozen=True, slots=True)
class Smoke:
    """``--smoke`` settings: a code check, never frozen."""

    query_stride: int
    latency_samples: int
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class HomologyConfig:
    """``configs/training/homology.yaml``, parsed and checked."""

    seed: int
    database: DatabasePin
    engine: EnginePin
    min_free_bytes: int
    search: homology.SearchSettings
    latency_samples: int
    poll_interval: float
    masks: tuple[str, ...]
    registry_only_diagnostic: bool
    transfer: TransferRule
    grid: Mapping[str, tuple[float, ...]]
    go_config: str
    go_labels: str
    go_vocabulary: str
    annotation_source: str
    smoke: Smoke
    bundle_version: str
    bundle_directory: str
    sha256: str


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{where} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineError(f"{where} must be a non-empty string")
    return value


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BaselineError(f"{where} must be a positive integer")
    return value


def _positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise BaselineError(f"{where} must be a positive number")
    return float(value)


def _unit(value: object, where: str) -> float:
    number = (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else -1.0
    )
    if not _UNIT[0] <= number <= _UNIT[1]:
        raise BaselineError(f"{where} must be a number in [0, 1]")
    return number


def _sha(value: object, where: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise BaselineError(f"{where} must be a lowercase SHA-256")
    return value


def _database(doc: Mapping[str, Any]) -> DatabasePin:
    return DatabasePin(
        source=_text(doc.get("source"), "database.source"),
        release=_text(doc.get("release"), "database.release"),
        records=_count(doc.get("records"), "database.records"),
        corpus_file=_text(doc.get("corpus_file"), "database.corpus_file"),
        corpus_sha256=_sha(doc.get("corpus_sha256"), "database.corpus_sha256"),
        license=_text(doc.get("license"), "database.license"),
        license_url=_text(doc.get("license_url"), "database.license_url"),
        directory=_text(doc.get("directory"), "database.directory"),
        fasta=_text(doc.get("fasta"), "database.fasta"),
        meta=_text(doc.get("meta"), "database.meta"),
        mmseqs_directory=_text(doc.get("mmseqs_directory"), "database.mmseqs_directory"),
        mmseqs_name=_text(doc.get("mmseqs_name"), "database.mmseqs_name"),
    )


def _engine(doc: Mapping[str, Any]) -> EnginePin:
    commit = _text(doc.get("commit"), "engine.commit")
    return EnginePin(
        name=_text(doc.get("name"), "engine.name"),
        release=_text(doc.get("release"), "engine.release"),
        commit=commit,
        platform=_text(doc.get("platform"), "engine.platform"),
        archive_sha256=_sha(doc.get("archive_sha256"), "engine.archive_sha256"),
        executable_sha256=_sha(doc.get("executable_sha256"), "engine.executable_sha256"),
        executable=_text(doc.get("executable"), "engine.executable"),
        license=_text(doc.get("license"), "engine.license"),
        license_url=_text(doc.get("license_url"), "engine.license_url"),
        download_url=_text(doc.get("download_url"), "engine.download_url"),
        threads=_count(doc.get("threads"), "engine.threads"),
        verbosity=_count(doc.get("verbosity"), "engine.verbosity"),
    )


def _grid(doc: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    if set(doc) != set(GRID_KEYS):
        raise BaselineError(f"hit_filter.grid must have exactly the keys {list(GRID_KEYS)}")
    grid: dict[str, tuple[float, ...]] = {}
    for key in GRID_KEYS:
        values = doc[key]
        if not isinstance(values, list) or not values:
            raise BaselineError(f"hit_filter.grid.{key} must be a non-empty list")
        if key == "e_value":
            grid[key] = tuple(_positive(v, f"grid.{key}") for v in values)
        elif key == "top_k":
            grid[key] = tuple(float(_count(v, f"grid.{key}")) for v in values)
        else:
            grid[key] = tuple(_unit(v, f"grid.{key}") for v in values)
    return grid


def _transfer(doc: Mapping[str, Any], grid: Mapping[str, tuple[float, ...]]) -> TransferRule:
    flag = doc.get("annotated_hits_only")
    if not isinstance(flag, bool):
        raise BaselineError("transfer.annotated_hits_only must be true or false")
    weight = _positive(doc.get("tiebreak_weight"), "transfer.tiebreak_weight")
    if weight * 1.0 >= 1.0 / max(grid["top_k"]):
        raise BaselineError(
            "transfer.tiebreak_weight must stay below 1 / max(top_k): it only breaks ties"
        )
    edges = doc.get("identity_bins")
    if not isinstance(edges, list) or len(edges) < _MIN_EDGES:
        raise BaselineError("transfer.identity_bins needs at least two edges")
    bins = tuple(_unit(e, "transfer.identity_bins") for e in edges)
    if bins[0] != _UNIT[0] or bins[-1] != _UNIT[1] or list(bins) != sorted(set(bins)):
        raise BaselineError("transfer.identity_bins must ascend strictly from 0 to 1")
    return TransferRule(
        annotated_hits_only=flag,
        tiebreak_weight=weight,
        identity_bins=bins,
        min_rows_per_bin=_count(doc.get("min_rows_per_bin"), "transfer.min_rows_per_bin"),
    )


def load_config(path: Path) -> HomologyConfig:
    """Parse the homology configuration and reject anything malformed before a stage starts."""
    raw = path.read_bytes()
    doc = _mapping(yaml.safe_load(raw), str(path))
    if doc.get("name") != "homology" or doc.get("task") != TASK:
        raise BaselineError("homology.yaml must be name 'homology' for task 'homology_evidence'")
    engine = _engine(_mapping(doc.get("engine"), "engine"))
    search = _mapping(doc.get("search"), "search")
    grid = _grid(_mapping(_mapping(doc.get("hit_filter"), "hit_filter").get("grid"), "grid"))
    e_value = _positive(search.get("e_value"), "search.e_value")
    if e_value < max(grid["e_value"]):
        raise BaselineError(
            "search.e_value is below the largest e-value of the grid: hits are lost"
        )
    masking = _mapping(doc.get("masking"), "masking")
    masks = masking.get("masks")
    if masks != list(homology.MASKS):
        raise BaselineError(
            f"masking.masks must be exactly {list(homology.MASKS)}: no leakage mask may be dropped"
        )
    diagnostic = masking.get("registry_only_diagnostic")
    if not isinstance(diagnostic, bool):
        raise BaselineError("masking.registry_only_diagnostic must be true or false")
    go_block = _mapping(doc.get("go"), "go")
    smoke = _mapping(doc.get("smoke"), "smoke")
    bundle = _mapping(doc.get("bundle"), "bundle")
    latency = _mapping(doc.get("latency"), "latency")
    return HomologyConfig(
        seed=_count(doc.get("seed"), "seed"),
        database=_database(_mapping(doc.get("database"), "database")),
        engine=engine,
        min_free_bytes=_count(
            _mapping(doc.get("index"), "index").get("min_free_bytes"), "index.min_free_bytes"
        ),
        search=homology.SearchSettings(
            sensitivity=_positive(search.get("sensitivity"), "search.sensitivity"),
            e_value=e_value,
            max_seqs=_count(search.get("max_seqs"), "search.max_seqs"),
            alignment_mode=_count(search.get("alignment_mode"), "search.alignment_mode"),
            threads=engine.threads,
            verbosity=engine.verbosity,
        ),
        latency_samples=_count(latency.get("samples"), "latency.samples"),
        poll_interval=_positive(
            latency.get("poll_interval_seconds"), "latency.poll_interval_seconds"
        ),
        masks=tuple(str(m) for m in masks),
        registry_only_diagnostic=diagnostic,
        transfer=_transfer(_mapping(doc.get("transfer"), "transfer"), grid),
        grid=grid,
        go_config=_text(go_block.get("config"), "go.config"),
        go_labels=_text(go_block.get("labels"), "go.labels"),
        go_vocabulary=_text(go_block.get("vocabulary"), "go.vocabulary"),
        annotation_source=_text(go_block.get("annotation_source"), "go.annotation_source"),
        smoke=Smoke(
            query_stride=_count(smoke.get("query_stride"), "smoke.query_stride"),
            latency_samples=_count(smoke.get("latency_samples"), "smoke.latency_samples"),
            bootstrap_samples=_count(smoke.get("bootstrap_samples"), "smoke.bootstrap_samples"),
        ),
        bundle_version=_text(bundle.get("version"), "bundle.version"),
        bundle_directory=_text(bundle.get("directory"), "bundle.directory"),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def grid_points(config: HomologyConfig) -> list[HitFilter]:
    """Every hit filter in expansion order (the first key varies slowest)."""
    return [
        HitFilter(e_value=e, min_identity=i, min_query_coverage=c, top_k=int(k))
        for e, i, c, k in itertools.product(*(config.grid[key] for key in GRID_KEYS))
    ]


def arm_names(config: HomologyConfig, *, with_model: bool) -> tuple[str, ...]:
    """The arms of a report: homology (vote, best hit, optional registry-only), prior, model."""
    names = [ARM_VOTE, ARM_BEST]
    if config.registry_only_diagnostic:
        names.append(ARM_REGISTRY)
    names.append(ARM_PRIOR)
    if with_model:
        names += list(MODEL_ARMS)
    return tuple(names)


# -- votes, grid, coverage, bins -------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Votes:
    """What a hit filter does for each row: vote fractions and which rows found anything."""

    score: Array
    any_hit: Array
    any_candidate: Array
    best_identity: Array


def vote_scores(
    accessions: Sequence[str],
    hits_by_query: Mapping[str, Sequence[Hit]],
    terms_of: Mapping[str, frozenset[str]],
    spec: tuple[Sequence[str], HitFilter, bool],
) -> Votes:
    """Vote fractions per row and term for one hit filter.

    ``spec`` is ``(vocabulary ids, filter, annotated_hits_only)``. The result depends only on the
    hits, the annotation of the hit targets and the filter: never on a model score.
    """
    np = _np()
    vocabulary, rule, annotated_only = spec
    column = {t: j for j, t in enumerate(vocabulary)}
    score = np.zeros((len(accessions), len(vocabulary)), dtype=float)
    any_hit = np.zeros(len(accessions), dtype=bool)
    any_candidate = np.zeros(len(accessions), dtype=bool)
    best = np.full(len(accessions), np.nan, dtype=float)
    for r, accession in enumerate(accessions):
        ranked = homology.passing(hits_by_query.get(accession, ()), rule)
        if ranked:
            any_hit[r] = True
            best[r] = ranked[0].identity
        chosen = homology.choose(ranked, rule.top_k, terms_of, annotated_only)
        if chosen:
            any_candidate[r] = True
        for term, fraction in homology.vote(chosen, terms_of, vocabulary).items():
            score[r, column[term]] = fraction
    return Votes(score, any_hit, any_candidate, best)


def with_prior(votes: Array, prior: Array, weight: float, vocab: go.Vocabulary) -> Array:
    """Vote fractions plus the tie-break prior, made true-path consistent (Phase 12 propagation)."""
    return go.propagate_scores(votes + weight * prior, vocab)


@dataclass(frozen=True, slots=True)
class GridRow:
    """One grid point scored on partitions 1-4 under nested thresholds."""

    index: int
    rule: HitFilter
    selector: float
    mean_fold_fmax: float
    hit_coverage: float


def run_grid(
    rows: go.Rows,
    hits_by_query: Mapping[str, Sequence[Hit]],
    terms_of: Mapping[str, frozenset[str]],
    spec: tuple[go.Vocabulary, Array, HomologyConfig, go.MetricSettings],
) -> list[GridRow]:
    """Every grid point's mean fold precision@k and Fmax; partitions 1-4 only, no model score.

    ``spec`` is ``(vocabulary, fold prior scores, configuration, metric settings)``.
    """
    np = _np()
    vocab, prior, config, metrics = spec
    keep = go.evaluable(rows.y)
    out: list[GridRow] = []
    for index, rule in enumerate(grid_points(config)):
        votes = vote_scores(
            rows.accession,
            hits_by_query,
            terms_of,
            (vocab.ids, rule, config.transfer.annotated_hits_only),
        )
        score = with_prior(votes.score, prior, config.transfer.tiebreak_weight, vocab)
        selector, fmax = go.fold_means(score, rows.y, rows.partition, metrics)
        coverage = float(np.mean(votes.any_hit[keep]))
        out.append(GridRow(index, rule, selector, fmax, coverage))
    return out


def select_row(grid: Sequence[GridRow]) -> GridRow:
    """The Phase 12 selection rule: higher mean fold precision@k, then Fmax, then earlier point."""
    return grid[go.select_point([(g.selector, g.mean_fold_fmax) for g in grid])]


def coverage_block(votes: Votes, y: Array) -> dict[str, Any]:
    """Fraction of evaluable rows with any passing hit, and with any hit that can vote."""
    np = _np()
    keep = go.evaluable(y)
    return {
        "rows": len(y),
        "evaluable_rows": int(keep.sum()),
        "rows_with_hit": int(votes.any_hit.sum()),
        "hit_coverage": float(np.mean(votes.any_hit[keep])) if keep.any() else 0.0,
        "annotated_coverage": float(np.mean(votes.any_candidate[keep])) if keep.any() else 0.0,
    }


def assemble(
    rows: go.Rows, scored: tuple[Array, Array], ks: Sequence[int], scoring: go.Scoring
) -> dict[str, Any]:
    """One arm's report from (propagated scores, decisions): pooled and per fold."""
    score, decision = scored
    return {
        "pooled": go.metric_report(score, rows.y, decision, rows.cluster_code, scoring),
        "folds": go.fold_summaries(score, rows.y, decision, rows.partition, ks),
    }


def nested_arm(
    rows: go.Rows, score: Array, ks: Sequence[int], scoring: go.Scoring
) -> tuple[dict[str, Any], dict[str, Any], Array]:
    """An arm under nested global thresholds: its report, the thresholds and the decisions."""
    limit = scoring.metrics.max_threshold_candidates
    per_fold, frozen = go.nested_thresholds(score, rows.y, rows.partition, limit)
    decision = go.decide(score, rows.partition, per_fold)
    thresholds = {"folds": {str(k): v for k, v in per_fold.items()}, "frozen": frozen}
    return assemble(rows, (score, decision), ks, scoring), thresholds, decision


def bin_reports(
    arms: Mapping[str, tuple[Array, Array]],
    rows: go.Rows,
    best_identity: Array,
    spec: tuple[Sequence[float], int, go.Scoring],
) -> dict[str, Any]:
    """Per-identity-bin metrics: rows binned by the identity of their best passing hit.

    ``arms`` maps an arm name to ``(propagated scores, decisions)`` over the same rows;
    ``spec`` is ``(bin edges, minimum evaluable rows to report, scoring)``. A bin below the
    minimum is listed with its counts and not scored.
    """
    np = _np()
    edges, minimum, scoring = spec
    labels = [homology.bin_label(edges, i) for i in range(len(edges) - 1)] + [NO_HIT_BIN]
    index = np.array(
        [
            len(edges) - 1
            if math.isnan(v)
            else (homology.identity_bin(float(v), edges) or 0)
            if homology.identity_bin(float(v), edges) is not None
            else len(edges) - 1
            for v in best_identity
        ]
    )
    out: dict[str, Any] = {}
    for b, label in enumerate(labels):
        mask = index == b
        evaluable = int((mask & go.evaluable(rows.y)).sum())
        entry: dict[str, Any] = {"rows": int(mask.sum()), "evaluable_rows": evaluable}
        entry["reported"] = evaluable >= minimum and label != NO_HIT_BIN
        if entry["reported"]:
            entry["arms"] = {
                name: go.metric_report(
                    score[mask], rows.y[mask], decision[mask], rows.cluster_code[mask], scoring
                )["metrics"]
                for name, (score, decision) in arms.items()
            }
        out[label] = entry
    return out


# -- manifest checks -------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Pins:
    """What a homology manifest is checked against; read from the repository, not the manifest."""

    registry_sha256: str
    corpus_sha256: str
    ontology_sha256: str
    vocabulary_sha256: str
    go_frozen_sha256: str
    config_sha256: str
    engine_release: str
    engine_commit: str
    engine_executable_sha256: str
    grid_size: int
    k_values: tuple[int, ...]
    reproduction_tolerance: float
    arms_validation: tuple[str, ...]
    arms_test: tuple[str, ...]


def _object(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise BaselineError(f"manifest is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def _check_inputs(inputs: Mapping[str, Any], pins: Pins) -> list[str]:
    expected = {
        "registry_sha256": pins.registry_sha256,
        "corpus_sha256": pins.corpus_sha256,
        "ontology_sha256": pins.ontology_sha256,
        "vocabulary_sha256": pins.vocabulary_sha256,
        "go_manifest_frozen_sha256": pins.go_frozen_sha256,
        "homology_config_sha256": pins.config_sha256,
    }
    problems = [
        f"inputs.{k} differs from the repository" for k, v in expected.items() if inputs.get(k) != v
    ]
    for key in (
        "fasta_sha256",
        "meta_sha256",
        "index_sha256",
        "queries_sha256",
        "hits_sha256",
        "label_table_sha256",
        "feature_table_sha256",
    ):
        if not _SHA.fullmatch(str(inputs.get(key, ""))):
            problems.append(f"inputs.{key} is not a SHA-256")
    return problems


def _check_engine(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    engine = _object(frozen, "engine")
    expected = {
        "release": pins.engine_release,
        "commit": pins.engine_commit,
        "executable_sha256": pins.engine_executable_sha256,
    }
    return [
        f"engine.{k} differs from homology.yaml" for k, v in expected.items() if engine.get(k) != v
    ]


def _check_masking(frozen: Mapping[str, Any]) -> list[str]:
    masking = _object(frozen, "masking")
    problems: list[str] = []
    if masking.get("masks") != list(homology.MASKS):
        problems.append("masking.masks must be exactly the four leakage masks")
    counts = masking.get("validation")
    if not isinstance(counts, dict):
        return [*problems, "masking.validation (hits removed per mask) is missing"]
    exclusive = counts.get("exclusive")
    if not isinstance(exclusive, dict) or set(exclusive) != set(homology.MASKS):
        return [*problems, "masking.validation.exclusive must count every mask"]
    if counts.get("raw", -1) - counts.get("kept", -2) != sum(exclusive.values()):
        problems.append("masking.validation: raw - kept differs from the hits the masks removed")
    return problems


def _check_selected(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    selected = _object(frozen, "selected")
    index = selected.get("index")
    params = selected.get("params")
    problems: list[str] = []
    if not isinstance(index, int) or not 0 <= index < pins.grid_size:
        problems.append("selected.index is outside the grid")
    if not isinstance(params, dict) or set(params) != set(GRID_KEYS):
        problems.append("selected.params must name every grid key")
    return problems


def _check_arm(report: object, name: str, pins: Pins, where: str) -> list[str]:
    if not isinstance(report, dict) or not isinstance(report.get("pooled"), dict):
        return [f"{where}.{name} has no pooled report"]
    metrics = report["pooled"].get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(go.metric_names(pins.k_values)):
        return [f"{where}.{name} does not report every metric"]
    return []


def _check_validation(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    validation = frozen.get("validation")
    if not isinstance(validation, dict):
        return ["manifest is missing the object 'validation'"]
    problems: list[str] = []
    for arm in pins.arms_validation:
        problems += _check_arm(validation.get(arm), arm, pins, "validation")
    reproduction = validation.get("model_reproduction")
    difference = reproduction.get("max_abs_difference") if isinstance(reproduction, dict) else None
    if not isinstance(difference, float) or not difference <= pins.reproduction_tolerance:
        problems.append("the Phase 12 model arm did not reproduce from the saved scores")
    for key in ("grid", "coverage", "identity_bins"):
        if key not in validation:
            problems.append(f"validation.{key} is missing")
    return problems


def _check_measurements(frozen: Mapping[str, Any]) -> list[str]:
    measured = _object(frozen, "measurements")
    problems: list[str] = []
    build = measured.get("index_build")
    latency = measured.get("query_latency")
    if not isinstance(build, dict):
        return ["measurements.index_build is missing"]
    for key in ("seconds", "index_bytes", "peak_working_set_bytes"):
        value = build.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0:
            problems.append(f"measurements.index_build.{key} must be a measured positive number")
    if not isinstance(latency, dict):
        return [*problems, "measurements.query_latency is missing"]
    for key in ("p50_seconds", "p95_seconds", "cpu_seconds_per_query"):
        value = latency.get(key)
        if not isinstance(value, float) or not math.isfinite(value) or value < 0:
            problems.append(f"measurements.query_latency.{key} must be a measured number")
    return problems


def _check_test(test: object, frozen_sha256: str, pins: Pins) -> list[str]:
    if not isinstance(test, dict):
        return ["test must be an object or null"]
    problems: list[str] = []
    if test.get("frozen_sha256") != frozen_sha256:
        problems.append("test was scored against a different frozen block")
    if test.get("touches") != 1:
        problems.append("the homology test partition must be touched exactly once")
    if test.get("earlier_touches") != dict.fromkeys(EARLIER_HEADS, 1):
        problems.append(f"every earlier head must still read touches 1: {list(EARLIER_HEADS)}")
    metrics = test.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(pins.arms_test):
        return [*problems, "test metrics do not cover exactly the homology and prior arms"]
    for name, report in metrics.items():
        problems += _check_arm(report, name, pins, "test")
    return problems


def check_homology_manifest(manifest: Mapping[str, Any], pins: Pins) -> list[str]:
    """Every way the homology manifest disagrees with itself or the repository; empty if sound."""
    try:
        frozen = _object(manifest, "frozen")
        inputs = _object(frozen, "inputs")
        thresholds = _object(frozen, "thresholds")
        independence = _object(frozen, "independence")
        protocol = _object(frozen, "protocol")
    except BaselineError as error:
        return [str(error)]
    problems = _check_inputs(inputs, pins) + _check_engine(frozen, pins)
    try:
        problems += _check_masking(frozen) + _check_selected(frozen, pins)
        problems += _check_validation(frozen, pins) + _check_measurements(frozen)
    except BaselineError as error:
        problems.append(str(error))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("phase") != PHASE:
        problems.append("schema_version or phase is not the one this code writes")
    if independence != INDEPENDENCE:
        problems.append(
            "independence must state that homology feeds no model score, calibrator or threshold"
        )
    if protocol.get("validation_partitions") != list(VALIDATION_PARTITIONS):
        problems.append("validation partitions are not [1, 2, 3, 4]")
    if protocol.get("test_partition") != TEST_PARTITION:
        problems.append("test partition is not 5")
    for arm in (a for a in pins.arms_validation if a not in MODEL_ARMS):
        value = thresholds.get(arm)
        if not isinstance(value, float) or not math.isfinite(value):
            problems.append(f"thresholds.{arm} must be a finite number")
    if manifest.get("frozen_sha256") != digest(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
    test = manifest.get("test")
    if test is not None:
        problems += _check_test(test, str(manifest.get("frozen_sha256")), pins)
    return problems


def folds_of(rows: go.Rows) -> list[Fold]:
    """The leave-one-partition-out folds (Phase 12), for the prior scores."""
    baselines = importlib.import_module("protloc.training.baselines")
    folds: list[Fold] = baselines.lopo_folds()
    return folds
