r"""Phase 11: the signal-peptide CRF head on the 500-sequence subset.

    python scripts/crf.py --train [--smoke] [--force]
        Builds the label table from the corpus (once), then leave-one-partition-out over
        registry partitions 1-4. Every grid point of configs/training/crf.yaml is fitted once
        per fold (early stopping on an inner whole-cluster holdout), scored out of fold, and the
        best by mean fold cleavage accuracy within +-1 residue (tie-break: detection MCC) is refit
        on partitions 1-4. Partition 5 is never read. Safe to kill and rerun: a finished grid
        point is kept. The script records its own wall time; do NOT wrap it in time or
        Measure-Command.

    python scripts/crf.py --evaluate [--smoke]
        Re-derives the out-of-fold detection scores from the saved fold weights (they must match
        to 1e-9 or the stage fails), then scores the CRF (source "model") and the untrained motif
        h-region detector (source "motif", the floor) under the identical protocol, with
        cluster-bootstrap CIs.

    python scripts/crf.py --calibrate [--smoke]
        Phase 10 machinery on the detection score: cross-fitted calibrator, MCC-tuned thresholds,
        ECE and Brier, and the calibrated decision metrics. Writes data/crf/calibration/.

    python scripts/crf.py --freeze
        Writes registry/signal_peptide/<version>/{calibrator.json,bundle.json} and
        data/manifests/crf_metrics.json. Refuses if either exists.

    python scripts/crf.py --evaluate --test
        Scores partition 5 ONCE with the frozen weights, calibrator and thresholds and adds it to
        the manifest. Refuses if the manifest is unsound or the test was already scored.

    python scripts/crf.py --check
        Needs no data bytes (this is what CI runs). Exit 1 unless the configuration, the pins,
        the manifest and the registry bundle (when present) agree.

--smoke runs every stage on every 4th row for 2 epochs and 2 grid points into data/crf/smoke.
Smoke output is a code check only: it can never be frozen and no number from it is reported.

Every number is for the 500-sequence subset and is labelled "subset (500), not comparable to
DeepLoc or to the Phase 8 full-corpus numbers". TargetP's published 83% (signal peptide) and
46-49% (transit peptides) are context only: other data, other protocol, not comparable.

Run from the repository root inside the project venv (pip install -e "backend[head,baselines]").
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import yaml

from protloc.calibration.calibrator import CalibratorSet, label_from_dict
from protloc.data.embeddings import read_pack
from protloc.data.manifest import read_json_object, sha256_file, write_json
from protloc.data.residue_cache import (
    ResidueStore,
    residue_directory,
    verify_residue_cache,
)
from protloc.registry import bundle as registry_bundle
from protloc.training import baseline_models as bm
from protloc.training import calibration as cal
from protloc.training import crf, crf_net, head
from protloc.training.baselines import (
    SUBSET_LABEL,
    SUBSET_SCOPE,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    assert_protocol,
    digest,
    load_class_map,
    lopo_folds,
    read_manifest,
)
from protloc.training.subset import embedding_manifest, pooled_directory, subset_block

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_MANIFEST = REPO_ROOT / "data" / "manifests" / "split_manifest.json"
EMBEDDING_MANIFEST = REPO_ROOT / "data" / "manifests" / "embedding_manifest.json"
RESIDUE_MANIFEST = REPO_ROOT / "data" / "manifests" / "residue_cache_manifest.json"
HEAD_MANIFEST = REPO_ROOT / "data" / "manifests" / "head_metrics.json"
MANIFEST = REPO_ROOT / "data" / "manifests" / "crf_metrics.json"
CLASS_MAP = REPO_ROOT / "backend" / "rules" / "baseline_class_map.yaml"
CONSISTENCY = REPO_ROOT / "backend" / "rules" / "consistency.json"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
ROOT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "trunk_esm2_150m.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "default.yaml"
CRF_CONFIG = REPO_ROOT / "configs" / "training" / "crf.yaml"
BASELINES = REPO_ROOT / "data" / "baselines"
EMBED_ROOT = REPO_ROOT / "data" / "embeddings"
PACK = EMBED_ROOT / "input_pack.tsv"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
CRF_DIR = REPO_ROOT / "data" / "crf"
LABELS_TSV = "labels.tsv"
LABELS_JSON = "labels.json"
WEIGHTS = "final_weights.npz"
OOF_SCORE = "oof_score.npy"
OOF_CLEAVAGE = "oof_cleavage.npy"
FIT_FILE = "fit.json"
CROSSFIT_FILE = "crossfit.npy"
VALIDATION_FILE = "validation.json"
SMOKE_BOOTSTRAP = 50
THRESHOLD_RULE = (
    "the detection score (probability of any signal tag) at which MCC is maximal; fold k is "
    "scored with a threshold tuned on the out-of-fold scores of the other validation "
    "partitions; the frozen threshold is tuned on all validation rows. The calibrated "
    "threshold is tuned the same way on the cross-fitted calibrated scores"
)
SELECTION_RULE = (
    "mean over folds of cleavage accuracy within +-1 residue among true positives with a usable "
    "cleavage site; tie-break mean fold detection MCC under nested thresholds; then the earlier "
    "grid point"
)
CLEAVAGE_RULE = (
    "the last residue of the highest-scoring path given that residue 1 is a signal residue "
    "(Viterbi with the outside tag forbidden at residue 1); reported only when detection fires"
)
DETECTION_SCORE = (
    "P(any signal tag) = 1 - P(every residue outside) = the signal marginal at residue 1"
)
ARMS = ("crf_raw", "crf_calibrated", "motif_floor")


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _np() -> Any:
    return importlib.import_module("numpy")


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise BaselineError(f"{path}: top level must be a mapping")
    return {str(k): v for k, v in doc.items()}


def _out_dir(args: argparse.Namespace) -> Path:
    return CRF_DIR / "smoke" if args.smoke else CRF_DIR


def _pinned_hashes() -> tuple[str, str]:
    manifest = read_manifest(SPLIT_MANIFEST)
    return str(manifest["registry"]["sha256"]), str(manifest["corpus_sha256"])


def _version(name: str) -> str:
    metadata = importlib.import_module("importlib.metadata")
    return str(metadata.version(name))


def _versions() -> dict[str, str]:
    return {name: _version(name) for name in ("torch", "numpy", "pyyaml")}


def _samples(args: argparse.Namespace) -> int:
    configured = int(_yaml(EVAL_CONFIG)["reporting"]["bootstrap_samples"])
    return SMOKE_BOOTSTRAP if args.smoke else configured


def _config(args: argparse.Namespace) -> crf.CrfConfig:
    """The CRF configuration, cross-checked against the root and model configs."""
    config = crf.load_config(CRF_CONFIG)
    if config.seed != int(_yaml(ROOT_CONFIG)["seed"]):
        raise BaselineError("crf.yaml seed differs from configs/config.yaml")
    if config.architecture.embedding_dim != int(_yaml(MODEL_CONFIG)["trunk"]["embedding_dim"]):
        raise BaselineError("crf.yaml embedding_dim differs from the trunk's embedding_dim")
    if args.smoke:
        shrunk = replace(config.schedule, max_epochs=config.smoke.max_epochs)
        return replace(config, schedule=shrunk)
    return config


def _calibration_config(config: crf.CrfConfig) -> cal.CalibrationConfig:
    calibration = cal.load_config(REPO_ROOT / config.calibration_config)
    if calibration.seed != config.seed:
        raise BaselineError("calibration.yaml seed differs from crf.yaml")
    return calibration


def _stride(args: argparse.Namespace, config: crf.CrfConfig) -> int:
    return config.smoke.stride if args.smoke else 1


def _points(args: argparse.Namespace, config: crf.CrfConfig) -> list[crf.GridPoint]:
    points = crf.grid_points(config)
    return points[: config.smoke.grid_points] if args.smoke else points


@dataclass(frozen=True, slots=True)
class Context:
    """Everything a data stage needs, loaded and cross-checked."""

    config: crf.CrfConfig
    table: dict[str, Any]
    order: list[str]
    store: ResidueStore
    labels: dict[str, crf.LabelRow]
    meta: dict[str, Any]
    label_info: dict[str, Any]
    residue_key: str


def _ensure_labels(
    table: dict[str, Any], meta: dict[str, Any], pins: tuple[str, str]
) -> tuple[dict[str, crf.LabelRow], dict[str, Any]]:
    """The label table for the subset's accessions, built from the corpus once and then reused."""
    registry_pin, corpus_pin = pins
    accessions = sorted({str(a) for a in table["accession"]})
    wanted = {
        "corpus_sha256": corpus_pin,
        "registry_sha256": registry_pin,
        "feature_table_sha256": str(meta["table_sha256"]),
        "accessions_sha256": digest(accessions),
    }
    tsv, info = CRF_DIR / LABELS_TSV, CRF_DIR / LABELS_JSON
    if tsv.exists() and info.exists():
        kept = read_manifest(info)
        if {k: kept.get(k) for k in wanted} == wanted:
            if sha256_file(tsv)[0] != kept.get("label_table_sha256"):
                raise BaselineError("labels.tsv does not match labels.json")
            _say("label table: reused")
            return crf.parse_label_table(tsv.read_text(encoding="utf-8")), kept
        _say("label table: inputs changed, rebuilding")
    began = time.perf_counter()
    if sha256_file(CORPUS)[0] != corpus_pin:
        raise BaselineError("data/processed/corpus.jsonl is not the corpus the manifests pin")
    labels = crf.scan_corpus(CORPUS, accessions)
    CRF_DIR.mkdir(parents=True, exist_ok=True)
    tsv.write_text(crf.label_table_text(labels), encoding="utf-8", newline="\n")
    partition_of = {
        str(a): int(p) for a, p in zip(table["accession"], table["partition"], strict=True)
    }
    document = {
        **wanted,
        "label_table_sha256": sha256_file(tsv)[0],
        "accessions": len(accessions),
        "counts": crf.label_counts(labels, partition_of),
        "seconds": time.perf_counter() - began,
    }
    write_json(info, document)
    _say(f"label table: built from the corpus in {document['seconds']:.1f} s")
    return labels, document


def _load(args: argparse.Namespace) -> Context:
    """Feature table (subset), sequence order, labels and per-residue store, checked together."""
    config = _config(args)
    cmap = load_class_map(CLASS_MAP)
    meta = read_manifest(BASELINES / "features.json")
    pins = _pinned_hashes()
    registry_pin, corpus_pin = pins
    if meta["registry_sha256"] != registry_pin or meta["corpus_sha256"] != corpus_pin:
        raise BaselineError("features.json was built from other data than the manifests record")
    if meta["class_map_sha256"] != cmap.sha256:
        raise BaselineError("the class map changed since baselines --features; rerun it")
    table_path = BASELINES / "features.npz"
    if sha256_file(table_path)[0] != meta["table_sha256"]:
        raise BaselineError("features.npz does not match features.json")
    table = bm.load_table(table_path)
    embedding = embedding_manifest(EMBEDDING_MANIFEST)
    indices, block = subset_block(table, embedding, EMBED_ROOT, PACK)
    directory = residue_directory(pooled_directory(embedding, EMBED_ROOT))
    problems = verify_residue_cache(directory)
    if problems:
        raise BaselineError(f"the per-residue cache is not sound: {problems[0]}")
    store = ResidueStore(directory)
    if store.dim != config.architecture.embedding_dim:
        raise BaselineError("the per-residue cache dimension differs from crf.yaml")
    items, _header = read_pack(PACK)
    order = [item.sha256 for item in items]
    if set(store.lengths) != {order[int(i)] for i in indices}:
        raise BaselineError("the per-residue cache and the pooled cache hold different sequences")
    if read_json_object(RESIDUE_MANIFEST).get("cache_key") != store.cache_key:
        raise BaselineError("residue_cache_manifest.json does not describe this cache")
    restricted = bm.restrict_to_sequences(table, indices)
    labels, label_info = _ensure_labels(restricted, meta, pins)
    return Context(
        config=config,
        table=restricted,
        order=order,
        store=store,
        labels=labels,
        meta={**meta, "subset": block},
        label_info=label_info,
        residue_key=store.cache_key,
    )


def _rows(ctx: Context, partitions: Sequence[int], stride: int) -> crf.Rows:
    spec = crf.RowSpec(partitions, stride, ctx.config.architecture.length_cap)
    return crf.build_rows(ctx.table, ctx.order, ctx.labels, spec)


def _announce() -> None:
    _say(f"scope: {SUBSET_LABEL}")


def _fingerprint(ctx: Context) -> dict[str, str]:
    """What a kept result must still agree with: change any of these and it is stale."""
    return {
        "crf_config_sha256": ctx.config.sha256,
        "residue_cache_key": ctx.residue_key,
        "feature_table_sha256": str(ctx.meta["table_sha256"]),
        "class_map_sha256": str(ctx.meta["class_map_sha256"]),
        "label_table_sha256": str(ctx.label_info["label_table_sha256"]),
    }


# -- --train ---------------------------------------------------------------------------
def _grid_path(out: Path, index: int) -> Path:
    return out / "train" / f"grid_{index:02d}.json"


def _fold_weights(out: Path, index: int, validate: int) -> Path:
    return out / "train" / f"grid_{index:02d}__fold_{validate}.npz"


def _kept(path: Path, ctx: Context) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = read_manifest(path)
    if report.get("fingerprint") != _fingerprint(ctx):
        raise BaselineError(f"{path.name} was produced from other inputs; rerun with --force")
    return report


def _train_point(
    ctx: Context, source: crf_net.Source, point: crf.GridPoint, index: int, out: Path
) -> dict[str, Any]:
    """Fit one grid point on every fold, score it out of fold and write its files."""
    np = _np()
    config, rows = ctx.config, source.rows
    folds = lopo_folds()
    leaks = head.held_out_clusters(ctx.table)
    settings = crf_net.settings_for(point, config)
    began = time.perf_counter()
    result = crf_net.run_grid_point(source, folds, settings, leaks, _say)
    selector = crf.selection_value(rows, result.cleavage, folds, config.selection_tolerance)
    mcc = crf.mean_fold_mcc(rows, result.score, folds)
    prefix = out / "train" / f"grid_{index:02d}__"
    bm.save_array(Path(f"{prefix}oof_score.npy"), result.score)
    bm.save_array(Path(f"{prefix}oof_cleavage.npy"), result.cleavage)
    for validate, state in result.states.items():
        np.savez(_fold_weights(out, index, validate), **state)
    report = {
        "index": index,
        "params": point.as_dict(),
        "selector": selector,
        "mean_fold_mcc": mcc,
        "folds": [asdict(r) for r in result.runs],
        "seconds": time.perf_counter() - began,
        "fingerprint": _fingerprint(ctx),
    }
    write_json(_grid_path(out, index), report)
    return report


def run_train(args: argparse.Namespace) -> int:
    """Leave-one-partition-out over partitions 1-4; the test partition is never read."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    config, np = ctx.config, _np()
    folds = lopo_folds()
    assert_protocol(folds)
    rows = _rows(ctx, VALIDATION_PARTITIONS, _stride(args, config))
    source = crf_net.Source(ctx.store, rows)
    out = _out_dir(args)
    (out / "train").mkdir(parents=True, exist_ok=True)
    points = _points(args, config)
    _say(
        f"{len(rows)} signal-peptide rows ({int(rows.y.sum())} positives, "
        f"{int((rows.y * rows.tagged).sum())} with a usable cleavage site), "
        f"{len(points)} grid points x {len(folds)} folds"
    )
    reports: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        kept = None if args.force else _kept(_grid_path(out, index), ctx)
        if kept is not None:
            _say(f"grid {index}: finished earlier, kept")
            reports.append(kept)
            continue
        _say(f"grid {index}: {point.as_dict()}")
        report = _train_point(ctx, source, point, index, out)
        reports.append(report)
        _say(
            f"grid {index}: cleavage within +-{config.selection_tolerance} "
            f"{report['selector']:.4f}, mean fold MCC {report['mean_fold_mcc']:.4f}, "
            f"{report['seconds']:.1f} s"
        )
    best_index = crf.select_point([(r["selector"], r["mean_fold_mcc"]) for r in reports])
    best = reports[best_index]
    chosen = crf.GridPoint(**best["params"])
    epochs = max(1, int(statistics.median_low(f["best_epoch"] for f in best["folds"])))
    _say(f"selected grid {best_index} {chosen.as_dict()}; refitting {epochs} epochs on 1-4")
    settings = crf_net.settings_for(chosen, config)
    final = crf_net.refit_final(source, settings, epochs, head.held_out_clusters(ctx.table))
    model = crf_net.build_model(settings.net, config.seed, final.state)
    np.savez(out / WEIGHTS, **final.state)
    prefix = out / "train" / f"grid_{best_index:02d}__"
    bm.save_array(out / OOF_SCORE, bm.load_array(Path(f"{prefix}oof_score.npy")))
    bm.save_array(out / OOF_CLEAVAGE, bm.load_array(Path(f"{prefix}oof_cleavage.npy")))
    write_json(
        out / "train" / "final.json",
        {
            "selected_index": best_index,
            "selected_params": best["params"],
            "selector": best["selector"],
            "mean_fold_mcc": best["mean_fold_mcc"],
            "final_epochs": epochs,
            "trainable_parameters": model.parameter_count(),
            "final_fit_seconds": final.seconds,
            "grid": [
                {
                    "index": r["index"],
                    "params": r["params"],
                    "selector": r["selector"],
                    "mean_fold_mcc": r["mean_fold_mcc"],
                    "seconds": r["seconds"],
                }
                for r in reports
            ],
            "train_rows": len(rows),
            "fingerprint": _fingerprint(ctx),
        },
    )
    stage = {
        "seconds": time.perf_counter() - began,
        "grid_seconds_total": sum(float(r["seconds"]) for r in reports),
        "hardware": bm.hardware(),
        "torch_threads": importlib.import_module("torch").get_num_threads(),
        "versions": _versions(),
        "seed": config.seed,
        "scope": SUBSET_SCOPE,
        "label": SUBSET_LABEL,
    }
    write_json(out / "train" / "stage.json", stage)
    _say(f"train stage: {stage['seconds']:.1f} s ({model.parameter_count()} trainable parameters)")
    return 0


# -- shared stage inputs ---------------------------------------------------------------
def _final(ctx: Context, out: Path) -> dict[str, Any]:
    final = read_manifest(out / "train" / "final.json")
    if final.get("fingerprint") != _fingerprint(ctx):
        raise BaselineError("train/final.json was produced from other inputs; rerun --train")
    return final


@dataclass(frozen=True, slots=True)
class OutOfFold:
    """The selected grid point's out-of-fold detection scores and cleavage sites."""

    rows: crf.Rows
    score: Any
    cleavage: Any


def _out_of_fold(ctx: Context, args: argparse.Namespace, out: Path) -> OutOfFold:
    rows = _rows(ctx, VALIDATION_PARTITIONS, _stride(args, ctx.config))
    score, cleavage = bm.load_array(out / OOF_SCORE), bm.load_array(out / OOF_CLEAVAGE)
    if score.shape != rows.y.shape or cleavage.shape != rows.y.shape:
        raise BaselineError("the out-of-fold files do not match the rows; rerun --train")
    return OutOfFold(rows, score, cleavage)


def _arm(
    rows: crf.Rows, scored: crf.Scored, settings: tuple[crf.Evaluation, bm.Bootstrap]
) -> dict[str, Any]:
    """Pooled report with intervals, one plain row per fold, and per-group reports."""
    np = _np()
    evaluation, _boot = settings
    everything = np.arange(len(rows))
    per_fold = []
    for fold in lopo_folds():
        held = np.flatnonzero(rows.partition == fold.validate)
        per_fold.append(
            {
                "validate": fold.validate,
                "rows": len(held),
                "metrics": crf.compute_metrics(rows, scored, held, evaluation.tolerances),
            }
        )
    return {
        "pooled": crf.metric_report(rows, scored, everything, settings),
        "folds": per_fold,
        "groups": crf.group_reports(rows, scored, settings),
    }


def _thresholded(
    rows: crf.Rows, score: Any, cleavage: Any, source: str
) -> tuple[crf.Scored, dict[str, Any]]:
    """Nested decisions on a score, with the thresholds that produced them."""
    decision, fold_thresholds, frozen = crf.nested_decisions(score, rows, lopo_folds())
    thresholds = {
        "source": source,
        "folds": {str(k): v for k, v in fold_thresholds.items()},
        "frozen": frozen,
    }
    return crf.Scored(score, decision, cleavage), thresholds


# -- --evaluate ------------------------------------------------------------------------
def _reproduce(ctx: Context, final: dict[str, Any], oof: OutOfFold, out: Path) -> dict[str, Any]:
    """Recompute every fold's scores from its saved weights; they must equal the stored ones."""
    np = _np()
    began = time.perf_counter()
    config, rows = ctx.config, oof.rows
    point = crf.GridPoint(**final["selected_params"])
    settings = crf_net.settings_for(point, config)
    folds = lopo_folds()
    score = np.zeros(len(rows), dtype=np.float64)
    cleavage = np.zeros(len(rows), dtype=np.int64)
    for fold in folds:
        _train, held = crf.fold_rows(rows, fold)
        with np.load(_fold_weights(out, int(final["selected_index"]), fold.validate)) as handle:
            state = {name: handle[name] for name in handle.files}
        model = crf_net.build_model(settings.net, config.seed, state)
        shas = [rows.shas[i] for i in held]
        prediction = crf_net.predict(model, ctx.store, shas, config.schedule.batch_size)
        score[held], cleavage[held] = prediction.score, prediction.cleavage
    difference = float(np.max(np.abs(score - oof.score)))
    stored, _t1, _f1 = crf.nested_decisions(oof.score, rows, folds)
    again, _t2, _f2 = crf.nested_decisions(score, rows, folds)
    identical = bool((stored == again).all()) and bool((cleavage == oof.cleavage).all())
    tolerance = config.evaluation.reproduction_tolerance
    if not difference <= tolerance or not identical:
        raise BaselineError(
            f"the raw detection scores do not reproduce from the saved weights "
            f"(max difference {difference:.3e}, decisions identical {identical})"
        )
    return {
        "max_abs_difference": difference,
        "decisions_identical": identical,
        "tolerance": tolerance,
        "seconds": time.perf_counter() - began,
    }


def _headline(name: str, arm: dict[str, Any]) -> str:
    metrics = arm["pooled"]["metrics"]

    def cell(key: str) -> str:
        m = metrics[key]
        return f"{m['value']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"

    return (
        f"{name}: detection MCC {cell('detection_mcc')}, AUC {cell('detection_auc')}; cleavage "
        f"exact {cell('cleavage_exact')}, +-1 {cell('cleavage_within_1')}; residue F1 "
        f"{cell('residue_f1')}"
    )


def _verdict(model: dict[str, Any], floor: dict[str, Any]) -> str:
    """Plain statement of whether the CRF beats the motif floor, headline by headline."""
    parts = []
    for key in ("detection_mcc", "detection_auc", "cleavage_within_1", "residue_f1"):
        a, b = model["pooled"]["metrics"][key], floor["pooled"]["metrics"][key]
        if a["ci_low"] is None or b["ci_low"] is None:
            parts.append(f"{key}: no intervals")
        elif a["ci_low"] > b["ci_high"]:
            parts.append(f"{key}: CRF above the floor, intervals disjoint")
        elif a["value"] > b["value"]:
            parts.append(f"{key}: CRF higher, intervals overlap (not shown to differ)")
        else:
            parts.append(f"{key}: CRF not above the floor")
    return "; ".join(parts)


def run_validation(args: argparse.Namespace) -> int:
    """The CRF and the motif floor scored out of fold under one protocol."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    out = _out_dir(args)
    final = _final(ctx, out)
    oof = _out_of_fold(ctx, args, out)
    reproduction = _reproduce(ctx, final, oof, out)
    _say(
        f"reproduction: max |difference| {reproduction['max_abs_difference']:.2e}, "
        "decisions identical"
    )
    boot = bm.Bootstrap(_samples(args), ctx.config.seed)
    settings = (ctx.config.evaluation, boot)
    rows = oof.rows
    model_scored, model_thr = _thresholded(rows, oof.score, oof.cleavage, "model")
    motif_score, motif_cleavage = crf.motif_arrays(ctx.table, rows)
    motif_scored, motif_thr = _thresholded(rows, motif_score, motif_cleavage, "motif")
    arms = {
        "crf_raw": {"source": crf.SOURCE_MODEL, **_arm(rows, model_scored, settings)},
        "motif_floor": {
            "source": crf.SOURCE_MOTIF,
            **_arm(rows, motif_scored, settings),
        },
    }
    _say(_headline("CRF  ", arms["crf_raw"]))
    _say(_headline("floor", arms["motif_floor"]))
    _say(f"verdict: {_verdict(arms['crf_raw'], arms['motif_floor'])}")
    _say("intervals are wide: few clusters, one subset; TargetP's 83% / 46-49% are not comparable")
    stage = {
        "seconds": time.perf_counter() - began,
        "samples": boot.samples,
        "seed": boot.seed,
    }
    write_json(
        out / VALIDATION_FILE,
        {
            "fingerprint": _fingerprint(ctx),
            "scope": SUBSET_SCOPE,
            "label": SUBSET_LABEL,
            "stage": stage,
            "validation": {
                **arms,
                "reproduction": reproduction,
                "thresholds": {"crf_raw": model_thr, "motif": motif_thr},
            },
        },
    )
    _say(f"validation stage: {stage['seconds']:.1f} s")
    return 0


# -- --calibrate -----------------------------------------------------------------------
def _calibration_fingerprint(ctx: Context, final: dict[str, Any], out: Path) -> dict[str, str]:
    return {
        **_fingerprint(ctx),
        "training_record_sha256": digest(final),
        "calibration_config_sha256": _calibration_config(ctx.config).sha256,
        "oof_sha256": sha256_file(out / OOF_SCORE)[0],
    }


def run_calibrate(args: argparse.Namespace) -> int:
    """Cross-fit the detection calibrator and its thresholds; the test partition is not read."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    np = _np()
    out = _out_dir(args)
    final = _final(ctx, out)
    fingerprint = _calibration_fingerprint(ctx, final, out)
    oof = _out_of_fold(ctx, args, out)
    rows, folds = oof.rows, lopo_folds()
    calibration = _calibration_config(ctx.config)
    data = cal.CalibrationData(
        scores=oof.score.reshape(-1, 1),
        y=rows.y.reshape(-1, 1),
        partition=rows.partition,
        cluster_code=rows.cluster_code,
        classes=(crf.TASK,),
    )
    fitted = cal.crossfit(data, folds, calibration, head.held_out_clusters(ctx.table))
    calibrated = fitted.calibrated[:, 0]
    decision = np.zeros(len(rows), dtype=bool)
    for fold in folds:
        held = rows.partition == fold.validate
        decision[held] = calibrated[held] >= fitted.fold_thresholds[fold.validate][0]
    boot = bm.Bootstrap(_samples(args), ctx.config.seed)
    settings = (ctx.config.evaluation, boot)
    scored = crf.Scored(calibrated, decision, oof.cleavage)
    arm = {"source": crf.SOURCE_MODEL, **_arm(rows, scored, settings)}
    report = cal.calibration_report(
        cal.ScoredRows(data.scores, fitted.calibrated, data.y, data.cluster_code, data.classes),
        calibration.ece,
        boot,
    )
    folder = out / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    bm.save_array(folder / CROSSFIT_FILE, fitted.calibrated)
    stage = {
        "seconds": time.perf_counter() - began,
        "samples": boot.samples,
        "seed": boot.seed,
        "hardware": bm.hardware(),
        "versions": _versions(),
    }
    write_json(
        folder / FIT_FILE,
        {
            "fingerprint": fingerprint,
            "labels": [entry.to_dict() for entry in fitted.frozen],
            "fold_thresholds": {str(k): v for k, v in fitted.fold_thresholds.items()},
            "frozen_thresholds": fitted.frozen_thresholds,
            "methods": fitted.methods,
        },
    )
    write_json(
        folder / VALIDATION_FILE,
        {
            "fingerprint": fingerprint,
            "scope": SUBSET_SCOPE,
            "label": SUBSET_LABEL,
            "stage": stage,
            "validation": {"crf_calibrated": arm, "calibration": report},
        },
    )
    macro = report["macro"]
    _say(
        f"method {fitted.methods['frozen'][0]}; ECE {macro['ece_raw']['value']:.4f} -> "
        f"{macro['ece_calibrated']['value']:.4f}; Brier {macro['brier_raw']['value']:.4f} -> "
        f"{macro['brier_calibrated']['value']:.4f}"
    )
    _say(_headline("CRF calibrated", arm))
    _say(f"calibrate stage: {stage['seconds']:.1f} s")
    return 0


# -- the manifest ----------------------------------------------------------------------
def _pins() -> crf.Pins:
    registry_pin, corpus_pin = _pinned_hashes()
    embedding = embedding_manifest(EMBEDDING_MANIFEST)
    residue = read_json_object(RESIDUE_MANIFEST) if RESIDUE_MANIFEST.exists() else {}
    config = crf.load_config(CRF_CONFIG)
    return crf.Pins(
        registry_sha256=registry_pin,
        corpus_sha256=corpus_pin,
        embedding_cache_key=str(embedding["cache_key"]),
        residue_cache_key=str(residue.get("cache_key", "")),
        config_sha256=config.sha256,
        calibration_config_sha256=_calibration_config(config).sha256,
        tags=config.tags,
        tolerances=config.evaluation.tolerances,
        reproduction_tolerance=config.evaluation.reproduction_tolerance,
        torch=head.pinned_version(PYPROJECT.read_text(encoding="utf-8"), "torch"),
    )


def _rules() -> dict[str, Path]:
    return {"baseline_class_map": CLASS_MAP, "consistency": CONSISTENCY}


def _protocol(config: crf.CrfConfig, boot: dict[str, Any]) -> dict[str, Any]:
    return {
        "adr": "ADR-0013 (protocol of ADR-0010 and ADR-0011)",
        "validation_partitions": list(VALIDATION_PARTITIONS),
        "test_partition": TEST_PARTITION,
        "folds": [{"train": list(f.train), "validate": f.validate} for f in lopo_folds()],
        "threshold_rule": THRESHOLD_RULE,
        "selection_rule": SELECTION_RULE,
        "detection_score": DETECTION_SCORE,
        "cleavage_rule": CLEAVAGE_RULE,
        "early_stopping": (
            "negative mean NLL on an inner whole-cluster holdout of the training partitions; "
            f"patience {config.schedule.patience}, at most {config.schedule.max_epochs} epochs"
        ),
        "length_cap": config.architecture.length_cap,
        "bootstrap": {
            "samples": boot["samples"],
            "unit": "cluster_id",
            "seed": boot["seed"],
        },
        "label_states": (
            "annotated or negative rows only (ADR-0006); unknown and not_applicable dropped"
        ),
    }


def _tag_scope(ctx: Context) -> dict[str, Any]:
    counts = ctx.label_info["counts"]
    undefined = {
        task: {
            "positives_with_cleavage_validation": counts[task][
                "positives_with_cleavage_validation"
            ],
            "positives_with_cleavage": counts[task]["positives_with_cleavage"],
            "trained": False,
            "reason": (
                f"fewer than {ctx.config.tags.min_cleavage_positives} positives "
                "with a cleavage site"
            ),
        }
        for task in ctx.config.tags.undefined_tasks
    }
    return {
        "tags": list(ctx.config.tags.names),
        "trained": crf.TASK,
        "positives": counts[crf.TASK],
        "undefined": undefined,
    }


def _head_block(ctx: Context, record_sha: str, weights_sha: str) -> dict[str, Any]:
    config = ctx.config
    return {
        "weights_file": WEIGHTS,
        "weights_sha256": weights_sha,
        "frozen_sha256": record_sha,
        "head_config_sha256": config.sha256,
        "classes": [crf.TASK],
        "tags": list(config.tags.names),
        "length_cap": config.architecture.length_cap,
        "encoder": "",
        "detection_score": DETECTION_SCORE,
        "cleavage_rule": CLEAVAGE_RULE,
        "source": crf.SOURCE_MODEL,
    }


def _calibrator_set(
    fit: dict[str, Any], record_sha: str, weights_sha: str, version: str
) -> CalibratorSet:
    return CalibratorSet(
        task=crf.TASK,
        calibration_version=f"{version}+{fit['fingerprint']['oof_sha256'][:12]}",
        classes=(crf.TASK,),
        head_weights_sha256=weights_sha,
        head_frozen_sha256=record_sha,
        threshold_rule=THRESHOLD_RULE,
        labels=tuple(label_from_dict(entry) for entry in fit["labels"]),
    )


def _inputs(ctx: Context, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "registry_sha256": ctx.meta["registry_sha256"],
        "corpus_sha256": ctx.meta["corpus_sha256"],
        "embedding_cache_key": ctx.meta["subset"]["embedding_cache_key"],
        "residue_cache_key": ctx.residue_key,
        "feature_table_sha256": ctx.meta["table_sha256"],
        "class_map_sha256": ctx.meta["class_map_sha256"],
        "label_table_sha256": ctx.label_info["label_table_sha256"],
        "crf_config_sha256": ctx.config.sha256,
        "calibration_config_sha256": _calibration_config(ctx.config).sha256,
        **hashes,
    }


def run_freeze(args: argparse.Namespace) -> int:
    """Write the registry bundle and the manifest: fixed before the test partition is read."""
    if args.smoke:
        _say("--smoke output can never be frozen")
        return 1
    if MANIFEST.exists():
        _say(f"{MANIFEST.name} exists; a frozen manifest is never rewritten")
        return 1
    _announce()
    ctx = _load(args)
    config = ctx.config
    final = _final(ctx, CRF_DIR)
    stage = read_manifest(CRF_DIR / "train" / "stage.json")
    valid = read_manifest(CRF_DIR / VALIDATION_FILE)
    fit = read_manifest(CRF_DIR / "calibration" / FIT_FILE)
    cal_valid = read_manifest(CRF_DIR / "calibration" / VALIDATION_FILE)
    fingerprint = _calibration_fingerprint(ctx, final, CRF_DIR)
    if (
        valid.get("fingerprint") != _fingerprint(ctx)
        or fit.get("fingerprint") != fingerprint
        or cal_valid.get("fingerprint") != fingerprint
    ):
        _say("a stage file was produced from other inputs; rerun --train, --evaluate, --calibrate")
        return 1
    directory = REPO_ROOT / config.bundle_directory / config.bundle_version
    if directory.exists():
        _say(f"{directory} exists; a frozen bundle is never rewritten")
        return 1
    record_sha, weights_sha = digest(final), sha256_file(CRF_DIR / WEIGHTS)[0]
    calibrator = _calibrator_set(fit, record_sha, weights_sha, config.bundle_version)
    head_block = _head_block(ctx, record_sha, weights_sha)
    head_block["encoder"] = final["selected_params"]["encoder"]
    rule_hashes = {f"{key}_sha256": sha256_file(path)[0] for key, path in _rules().items()}
    document = registry_bundle.build_document(
        version=config.bundle_version,
        head=head_block,
        calibrator=calibrator,
        rules=rule_hashes,
        scope={"scope": SUBSET_SCOPE, "scope_label": SUBSET_LABEL},
    )
    thresholds = {
        "crf_raw": float(valid["validation"]["thresholds"]["crf_raw"]["frozen"]),
        "crf_calibrated": float(fit["frozen_thresholds"][0]),
        "motif": float(valid["validation"]["thresholds"]["motif"]["frozen"]),
    }
    validation = {
        **valid["validation"],
        "crf_calibrated": cal_valid["validation"]["crf_calibrated"],
        "calibration": cal_valid["validation"]["calibration"],
    }
    frozen: dict[str, Any] = {
        "protocol": _protocol(config, valid["stage"]),
        "scope": SUBSET_SCOPE,
        "scope_label": SUBSET_LABEL,
        "subset": ctx.meta["subset"],
        "inputs": _inputs(
            ctx,
            {
                "weights_sha256": weights_sha,
                "training_record_sha256": record_sha,
                "calibrator_sha256": document["calibrator"]["sha256"],
                "oof_sha256": fingerprint["oof_sha256"],
            },
        ),
        "tag_scope": _tag_scope(ctx),
        "architecture": {
            "encoder": final["selected_params"]["encoder"],
            "embedding_dim": config.architecture.embedding_dim,
            "length_cap": config.architecture.length_cap,
            "optimiser": "adamw",
            "weight_decay": config.weight_decay,
            "batch_size": config.schedule.batch_size,
            "seed": config.seed,
        },
        "selected": {
            "index": final["selected_index"],
            "params": final["selected_params"],
            "final_epochs": final["final_epochs"],
            "trainable_parameters": final["trainable_parameters"],
            "selector": final["selector"],
            "mean_fold_mcc": final["mean_fold_mcc"],
        },
        "grid": final["grid"],
        "thresholds": thresholds,
        "calibration_methods": fit["methods"],
        "validation": validation,
        "context": {"targetp": dict(config.targetp)},
        "software": {"torch": _version("torch").split("+")[0], "versions": _versions()},
        "hardware": {
            "train": {
                **stage["hardware"],
                "seconds": stage["seconds"],
                "grid_seconds_total": stage["grid_seconds_total"],
                "final_fit_seconds": final["final_fit_seconds"],
                "torch_threads": stage["torch_threads"],
            },
            "evaluate_seconds": valid["stage"]["seconds"],
            "calibrate_seconds": cal_valid["stage"]["seconds"],
        },
        "bundle": {
            "version": config.bundle_version,
            "directory": f"{config.bundle_directory}/{config.bundle_version}",
            "bundle_sha256": document["bundle_sha256"],
            "calibration_version": calibrator.calibration_version,
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": crf.SCHEMA_VERSION,
        "phase": crf.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }
    problems = crf.check_crf_manifest(manifest, _pins()) + calibrator.problems()
    if problems:
        for problem in problems:
            _say(f"  {problem}")
        return 1
    directory.mkdir(parents=True)
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(calibrator.to_bytes())
    write_json(directory / registry_bundle.BUNDLE_FILE, document)
    registry_bundle.load_bundle(directory, _rules(), CRF_DIR / WEIGHTS)
    write_json(MANIFEST, manifest)
    _say(f"froze {directory} and {MANIFEST}, frozen_sha256 {manifest['frozen_sha256'][:12]}")
    return 0


# -- --evaluate --test -----------------------------------------------------------------
def _test_state(frozen: dict[str, Any]) -> dict[str, Any]:
    np = _np()
    with np.load(CRF_DIR / WEIGHTS, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def run_test(args: argparse.Namespace) -> int:
    """Score partition 5 once, with the weights, calibrator and thresholds the manifest froze."""
    if args.smoke:
        _say("--smoke never touches the test partition")
        return 1
    began = time.perf_counter()
    manifest = read_manifest(MANIFEST)
    problems = crf.check_crf_manifest(manifest, _pins())
    if problems or manifest["test"] is not None:
        _say("refusing: the manifest has problems or the test partition was already scored")
        for problem in problems:
            _say(f"  {problem}")
        return 1
    _announce()
    ctx = _load(args)
    frozen = manifest["frozen"]
    if sha256_file(CRF_DIR / WEIGHTS)[0] != frozen["inputs"]["weights_sha256"]:
        _say("refusing: the weights are not the ones the manifest froze")
        return 1
    loaded = registry_bundle.load_bundle(
        REPO_ROOT / frozen["bundle"]["directory"], _rules(), CRF_DIR / WEIGHTS
    )
    config, np = ctx.config, _np()
    settings = crf_net.settings_for(crf.GridPoint(**frozen["selected"]["params"]), config)
    model = crf_net.build_model(settings.net, config.seed, _test_state(frozen))
    rows = _rows(ctx, (TEST_PARTITION,), 1)
    train = _rows(ctx, VALIDATION_PARTITIONS, 1)
    head.assert_no_overlap(
        {
            "train": crf.clusters_of(train, range(len(train))),
            "test": crf.clusters_of(rows, range(len(rows))),
        }
    )
    prediction = crf_net.predict(model, ctx.store, rows.shas, config.schedule.batch_size)
    label = loaded.calibrators.labels[0]
    calibrated = np.array([label.apply(float(v)) for v in prediction.score])
    motif_score, motif_cleavage = crf.motif_arrays(ctx.table, rows)
    thresholds = frozen["thresholds"]
    scored = {
        "crf_raw": crf.Scored(
            prediction.score,
            prediction.score >= thresholds["crf_raw"],
            prediction.cleavage,
        ),
        "crf_calibrated": crf.Scored(
            calibrated, calibrated >= label.threshold, prediction.cleavage
        ),
        "motif_floor": crf.Scored(motif_score, motif_score >= thresholds["motif"], motif_cleavage),
    }
    boot = bm.Bootstrap(_samples(args), config.seed)
    everything = np.arange(len(rows))
    metrics = {
        name: crf.metric_report(rows, value, everything, (config.evaluation, boot))
        for name, value in scored.items()
    }
    for name in ARMS:
        _say(_headline(f"test {name}", {"pooled": metrics[name]}))
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "rows": len(rows),
        "seconds": time.perf_counter() - began,
        "hardware": bm.hardware(),
        "scope": SUBSET_SCOPE,
        "label": SUBSET_LABEL,
        "metrics": metrics,
    }
    write_json(MANIFEST, manifest)
    _say(
        f"test partition scored once in {manifest['test']['seconds']:.1f} s "
        f"({len(rows)} rows, CI is wide)"
    )
    return 0


# -- --check ---------------------------------------------------------------------------
def _bundle_problems(manifest: dict[str, Any]) -> list[str]:
    """The registry entry must verify and be the one the manifest names."""
    frozen = manifest["frozen"]
    directory = REPO_ROOT / frozen["bundle"]["directory"]
    try:
        loaded = registry_bundle.load_bundle(directory, _rules())
    except (registry_bundle.BundleError, OSError) as error:
        return [f"registry bundle: {error}"]
    found: list[str] = []
    if loaded.document["bundle_sha256"] != frozen["bundle"]["bundle_sha256"]:
        found.append("the bundle is not the one the manifest froze")
    if loaded.document["calibrator"]["sha256"] != frozen["inputs"]["calibrator_sha256"]:
        found.append("the calibrator is not the one the manifest froze")
    head_doc = loaded.document["head"]
    if head_doc["weights_sha256"] != frozen["inputs"]["weights_sha256"]:
        found.append("the bundle names other weights than the manifest")
    if head_doc["frozen_sha256"] != frozen["inputs"]["training_record_sha256"]:
        found.append("the bundle names another training record than the manifest")
    return found


def run_check(args: argparse.Namespace) -> int:
    """No data bytes: configuration, pins and, when present, the manifest and the bundle."""
    problems: list[str] = []
    config = _config(args)
    assert_protocol(lopo_folds())
    pins = _pins()
    if MANIFEST.exists():
        manifest = read_manifest(MANIFEST)
        problems += crf.check_crf_manifest(manifest, pins)
        if not problems:
            problems += _bundle_problems(manifest)
        state = f"{MANIFEST.name} present"
    else:
        state = f"{MANIFEST.name} absent (configuration and pins only)"
    for problem in problems:
        _say(f"FAIL {problem}")
    _say(
        f"crf --check: {'FAIL' if problems else 'ok'}; torch=={pins.torch}; "
        f"{len(crf.grid_points(config))} grid points; tags {list(config.tags.names)}; {state}; "
        f"scope: {SUBSET_LABEL}"
    )
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 11 signal-peptide CRF head.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("train", "evaluate", "calibrate", "freeze", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--test", action="store_true", help="with --evaluate: score the test once")
    parser.add_argument("--smoke", action="store_true", help="every 4th row, never frozen")
    parser.add_argument("--force", action="store_true", help="with --train: refit every grid point")
    return parser


def _mode(args: argparse.Namespace) -> int:
    """Run the one mode the arguments name."""
    if args.train:
        return run_train(args)
    if args.evaluate:
        return run_test(args) if args.test else run_validation(args)
    if args.calibrate:
        return run_calibrate(args)
    if args.freeze:
        return run_freeze(args)
    return run_check(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one mode."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.test and not args.evaluate:
        parser.error("--test goes with --evaluate")
    if args.force and not args.train:
        parser.error("--force goes with --train")
    try:
        return _mode(args)
    except BaselineError as error:
        _say(f"error: {error}")
        return 1
    except json.JSONDecodeError as error:
        _say(f"error: unreadable JSON: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
