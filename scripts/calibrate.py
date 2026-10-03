r"""Phase 10: per-label calibration and thresholds for the Phase 9 localization head.

    python scripts/calibrate.py --fit [--smoke] [--force]
        Cross-fits the per-label calibrators on the 408 out-of-fold validation scores of
        data/head/oof.npy (leave-one-partition-out over registry partitions 1-4), tunes the
        doubly-nested per-label thresholds and fits the frozen calibrators on all four
        partitions. Partition 5 is never read. Writes data/calibration/.

    python scripts/calibrate.py --evaluate [--smoke]
        ECE and Brier before and after (per label and macro, cluster-bootstrap CIs), the
        reliability bins, and macro F1, Jaccard and per-label MCC at the new thresholds beside
        the raw Phase 9 decisions.

    python scripts/calibrate.py --freeze
        Writes registry/localization/<version>/{calibrator.json,bundle.json} and
        data/manifests/calibration_metrics.json. Refuses if either exists. There is no test
        mode: Phase 10 leaves partition 5 untouched (ADR-0012).

    python scripts/calibrate.py --check
        Needs no data bytes (this is what CI runs). Exit 1 unless the configuration, the pins,
        the manifest and the registry bundle (when present) agree.

--smoke runs on every 4th row with 50 bootstrap samples into data/calibration/smoke. Smoke
output is a code check only: it can never be frozen and no number from it is reported.
Every script records its own wall time; do NOT wrap it in time or Measure-Command.

Every number is for the 500-sequence subset and is labelled "subset (500), not comparable to
DeepLoc or to the Phase 8 full-corpus numbers".

Run from the repository root inside the project venv (pip install -e "backend[baselines]").
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from protloc.calibration.calibrator import CalibratorSet, label_from_dict
from protloc.data.embeddings import read_pack
from protloc.data.manifest import read_json_object, sha256_file, write_json
from protloc.registry import bundle as registry_bundle
from protloc.training import baseline_models as bm
from protloc.training import calibration as cal
from protloc.training import head
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
from protloc.training.subset import embedding_manifest, subset_block

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_MANIFEST = REPO_ROOT / "data" / "manifests" / "split_manifest.json"
EMBEDDING_MANIFEST = REPO_ROOT / "data" / "manifests" / "embedding_manifest.json"
RESIDUE_MANIFEST = REPO_ROOT / "data" / "manifests" / "residue_cache_manifest.json"
HEAD_MANIFEST = REPO_ROOT / "data" / "manifests" / "head_metrics.json"
MANIFEST = REPO_ROOT / "data" / "manifests" / "calibration_metrics.json"
CLASS_MAP = REPO_ROOT / "backend" / "rules" / "baseline_class_map.yaml"
CONSISTENCY = REPO_ROOT / "backend" / "rules" / "consistency.json"
ROOT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "default.yaml"
CALIBRATION_CONFIG = REPO_ROOT / "configs" / "training" / "calibration.yaml"
BASELINES = REPO_ROOT / "data" / "baselines"
EMBED_ROOT = REPO_ROOT / "data" / "embeddings"
PACK = EMBED_ROOT / "input_pack.tsv"
HEAD_DIR = REPO_ROOT / "data" / "head"
OUT_DIR = REPO_ROOT / "data" / "calibration"
SMOKE_BOOTSTRAP = 50
# The raw decisions must reproduce the Phase 9 validation macro F1 to this absolute difference.
REPRODUCTION_TOLERANCE = 1e-9
FIT_FILE = "fit.json"
CROSSFIT_FILE = "crossfit.npy"
VALIDATION_FILE = "validation.json"
WEIGHTS = "final_weights.npz"


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise BaselineError(f"{path}: top level must be a mapping")
    return {str(k): v for k, v in doc.items()}


def _config() -> cal.CalibrationConfig:
    """The calibration configuration, cross-checked against the root configuration."""
    config = cal.load_config(CALIBRATION_CONFIG)
    if config.seed != int(_yaml(ROOT_CONFIG)["seed"]):
        raise BaselineError("calibration.yaml seed differs from configs/config.yaml")
    return config


def _out(args: argparse.Namespace) -> Path:
    return OUT_DIR / "smoke" if args.smoke else OUT_DIR


def _samples(args: argparse.Namespace) -> int:
    configured = int(_yaml(EVAL_CONFIG)["reporting"]["bootstrap_samples"])
    return SMOKE_BOOTSTRAP if args.smoke else configured


def _version(name: str) -> str:
    metadata = importlib.import_module("importlib.metadata")
    return str(metadata.version(name))


def _pinned_hashes() -> tuple[str, str]:
    manifest = read_manifest(SPLIT_MANIFEST)
    return str(manifest["registry"]["sha256"]), str(manifest["corpus_sha256"])


def _head_manifest() -> dict[str, Any]:
    """The frozen Phase 9 manifest, accepted only if its own hash and its weights name agree."""
    manifest = read_manifest(HEAD_MANIFEST)
    frozen = manifest["frozen"]
    if manifest.get("frozen_sha256") != digest(frozen):
        raise BaselineError(
            "head_metrics.json: frozen_sha256 does not match the frozen block"
        )
    return manifest


def _pins() -> cal.Pins:
    manifest = _head_manifest()
    inputs = manifest["frozen"]["inputs"]
    registry_pin, corpus_pin = _pinned_hashes()
    cmap = load_class_map(CLASS_MAP)
    if (
        inputs["registry_sha256"] != registry_pin
        or inputs["corpus_sha256"] != corpus_pin
    ):
        raise BaselineError(
            "head_metrics.json was built from other data than the manifests record"
        )
    if inputs["class_map_sha256"] != cmap.sha256:
        raise BaselineError("the class map changed since the head was frozen")
    return cal.Pins(
        head_frozen_sha256=str(manifest["frozen_sha256"]),
        head_weights_sha256=str(inputs["weights_sha256"]),
        registry_sha256=registry_pin,
        corpus_sha256=corpus_pin,
        embedding_cache_key=str(embedding_manifest(EMBEDDING_MANIFEST)["cache_key"]),
        residue_cache_key=str(read_json_object(RESIDUE_MANIFEST).get("cache_key", "")),
        class_map_sha256=cmap.sha256,
        config_sha256=cal.load_config(CALIBRATION_CONFIG).sha256,
        classes=tuple(cmap.classes[cal.TASK]),
    )


@dataclass(frozen=True, slots=True)
class Context:
    """The out-of-fold rows and scores being calibrated, loaded and cross-checked."""

    config: cal.CalibrationConfig
    data: cal.CalibrationData
    held_out: set[int]
    pins: cal.Pins
    head: dict[str, Any]
    fingerprint: dict[str, str]


def _load(args: argparse.Namespace) -> Context:
    """Rows of partitions 1-4 and their Phase 9 out-of-fold scores; partition 5 only by cluster."""
    config = _config()
    pins = _pins()
    manifest = _head_manifest()
    head_dir = HEAD_DIR / "smoke" if args.smoke else HEAD_DIR
    if (
        not args.smoke
        and sha256_file(HEAD_DIR / WEIGHTS)[0] != pins.head_weights_sha256
    ):
        raise BaselineError(
            "data/head/final_weights.npz is not the head the manifest froze"
        )
    meta = read_manifest(BASELINES / "features.json")
    if (
        meta["registry_sha256"] != pins.registry_sha256
        or meta["corpus_sha256"] != pins.corpus_sha256
    ):
        raise BaselineError(
            "features.json was built from other data than the manifests record"
        )
    table_path = BASELINES / "features.npz"
    if sha256_file(table_path)[0] != meta["table_sha256"]:
        raise BaselineError("features.npz does not match features.json")
    table = bm.load_table(table_path)
    indices, _block = subset_block(
        table, embedding_manifest(EMBEDDING_MANIFEST), EMBED_ROOT, PACK
    )
    order = [item.sha256 for item in read_pack(PACK)[0]]
    restricted = bm.restrict_to_sequences(table, indices)
    stride = config.smoke_stride if args.smoke else 1
    rows = head.build_rows(restricted, order, VALIDATION_PARTITIONS, stride)
    scores = bm.load_array(head_dir / "oof.npy")
    if scores.shape != rows.y.shape:
        raise BaselineError(
            "oof.npy does not match the validation rows; rerun head.py --train"
        )
    data = cal.CalibrationData(
        scores=scores,
        y=rows.y,
        partition=rows.partition,
        cluster_code=rows.cluster_code,
        classes=pins.classes,
    )
    fingerprint = {
        "head_frozen_sha256": pins.head_frozen_sha256,
        "calibration_config_sha256": config.sha256,
        "oof_sha256": sha256_file(head_dir / "oof.npy")[0],
        "feature_table_sha256": str(meta["table_sha256"]),
        "class_map_sha256": pins.class_map_sha256,
    }
    return Context(
        config, data, head.held_out_clusters(restricted), pins, manifest, fingerprint
    )


def _announce() -> None:
    _say(f"scope: {SUBSET_LABEL}")


# -- --fit -----------------------------------------------------------------------------
def run_fit(args: argparse.Namespace) -> int:
    """Cross-fit calibrators and thresholds; the test partition is never read."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    out = _out(args)
    out.mkdir(parents=True, exist_ok=True)
    if (out / FIT_FILE).exists() and not args.force:
        kept = read_manifest(out / FIT_FILE)
        if kept.get("fingerprint") != ctx.fingerprint:
            raise BaselineError(
                "fit.json was produced from other inputs; rerun with --force"
            )
        _say("fit.json exists for these inputs; kept (use --force to refit)")
        return 0
    folds = lopo_folds()
    assert_protocol(folds)
    _say(f"{len(ctx.data.y)} annotated localization rows, {len(folds)} folds")
    fitted = cal.crossfit(ctx.data, folds, ctx.config, ctx.held_out)
    bm.save_array(out / CROSSFIT_FILE, fitted.calibrated)
    for name, method in zip(ctx.pins.classes, fitted.methods["frozen"], strict=True):
        consistent = fitted.methods["consistent"][name]
        _say(f"  {name}: {method}{'' if consistent else ' (fold methods differ)'}")
    write_json(
        out / FIT_FILE,
        {
            "fingerprint": ctx.fingerprint,
            "labels": [entry.to_dict() for entry in fitted.frozen],
            "fold_thresholds": {str(k): v for k, v in fitted.fold_thresholds.items()},
            "frozen_thresholds": fitted.frozen_thresholds,
            "methods": fitted.methods,
            "stage": {
                "seconds": time.perf_counter() - began,
                "hardware": bm.hardware(),
                "versions": {
                    n: _version(n) for n in ("numpy", "scikit-learn", "pyyaml")
                },
                "seed": ctx.config.seed,
                "scope": SUBSET_SCOPE,
                "label": SUBSET_LABEL,
            },
        },
    )
    _say(f"fit stage: {time.perf_counter() - began:.1f} s")
    return 0


def _read_fit(ctx: Context, out: Path) -> tuple[dict[str, Any], cal.CrossFit]:
    report = read_manifest(out / FIT_FILE)
    if report.get("fingerprint") != ctx.fingerprint:
        raise BaselineError("fit.json was produced from other inputs; rerun --fit")
    frozen = [label_from_dict(entry) for entry in report["labels"]]
    fitted = cal.CrossFit(
        calibrated=bm.load_array(out / CROSSFIT_FILE),
        fold_thresholds={int(k): v for k, v in report["fold_thresholds"].items()},
        frozen_thresholds=report["frozen_thresholds"],
        frozen=frozen,
        methods=report["methods"],
    )
    if fitted.calibrated.shape != ctx.data.y.shape:
        raise BaselineError(
            "crossfit.npy does not match the validation rows; rerun --fit"
        )
    return report, fitted


# -- --evaluate ------------------------------------------------------------------------
def run_evaluate(args: argparse.Namespace) -> int:
    """ECE, Brier and decision metrics before and after calibration, out of fold."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    out = _out(args)
    _report, fitted = _read_fit(ctx, out)
    data, folds = ctx.data, lopo_folds()
    boot = bm.Bootstrap(_samples(args), ctx.config.seed)
    split = bm.Split(data.y, data.partition, data.cluster_code)
    raw = bm.validate_oof(data.scores, split, folds, data.classes, boot)
    reference = ctx.head["frozen"]["validation"]["pooled"]["macro_f1"]["value"]
    gap = abs(raw["pooled"]["macro_f1"]["value"] - reference)
    if not args.smoke and gap > REPRODUCTION_TOLERANCE:
        raise BaselineError(
            "raw decisions do not reproduce the Phase 9 validation macro F1"
        )
    scored = cal.ScoredRows(
        data.scores, fitted.calibrated, data.y, data.cluster_code, data.classes
    )
    calibration = cal.calibration_report(scored, ctx.config.ece, boot)
    calibrated = cal.validate_calibrated(fitted, data, folds, boot)
    _summary(raw, calibrated, calibration)
    stage = {
        "seconds": time.perf_counter() - began,
        "samples": boot.samples,
        "seed": boot.seed,
    }
    write_json(
        out / VALIDATION_FILE,
        {
            "fingerprint": ctx.fingerprint,
            "scope": SUBSET_SCOPE,
            "label": SUBSET_LABEL,
            "stage": stage,
            "validation": {
                "calibration": calibration,
                "raw": raw,
                "calibrated": calibrated,
            },
        },
    )
    _say(f"evaluate stage: {stage['seconds']:.1f} s")
    return 0


def _summary(
    raw: dict[str, Any], calibrated: dict[str, Any], calibration: dict[str, Any]
) -> None:
    macro = calibration["macro"]
    _say(
        f"macro ECE {macro['ece_raw']['value']:.4f} -> {macro['ece_calibrated']['value']:.4f}; "
        f"macro Brier {macro['brier_raw']['value']:.4f} -> {macro['brier_calibrated']['value']:.4f}"
    )
    for label, block in (("raw", raw), ("calibrated", calibrated)):
        pooled = block["pooled"]
        f1, jac = pooled["macro_f1"], pooled["jaccard"]
        _say(
            f"{label}: macro F1 {f1['value']:.4f} [{f1['ci_low']:.4f}, {f1['ci_high']:.4f}] "
            f"Jaccard {jac['value']:.4f}; intervals are wide: few clusters, one subset"
        )


# -- --freeze --------------------------------------------------------------------------
def _frozen_block(
    ctx: Context, fit: dict[str, Any], valid: dict[str, Any]
) -> dict[str, Any]:
    config, head_frozen = ctx.config, ctx.head["frozen"]
    return {
        "protocol": {
            "adr": "ADR-0012 (protocol of ADR-0010 and ADR-0011)",
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "folds": [
                {"train": list(f.train), "validate": f.validate} for f in lopo_folds()
            ],
            "test_partition": TEST_PARTITION,
            "test_partition_policy": cal.TEST_POLICY,
            "head_test_touches": ctx.head["test"]["touches"],
            "split": "cross-fitted on the out-of-fold validation scores of the Phase 9 head",
            "threshold_rule": cal.THRESHOLD_RULE,
            "eligibility": asdict(config.eligibility),
            "fallback": cal.FALLBACK,
            "platt": asdict(config.platt),
            "ece": {"kind": cal.ECE_KIND, **asdict(config.ece)},
            "bootstrap": {
                "samples": valid["stage"]["samples"],
                "seed": valid["stage"]["seed"],
                "unit": "cluster_id",
            },
            "label_states": "annotated rows only (ADR-0006); unknown and not_applicable dropped",
        },
        "scope": SUBSET_SCOPE,
        "scope_label": SUBSET_LABEL,
        "subset": head_frozen["subset"],
        "inputs": {
            "head_frozen_sha256": ctx.pins.head_frozen_sha256,
            "head_weights_sha256": ctx.pins.head_weights_sha256,
            "registry_sha256": ctx.pins.registry_sha256,
            "corpus_sha256": ctx.pins.corpus_sha256,
            "embedding_cache_key": ctx.pins.embedding_cache_key,
            "residue_cache_key": ctx.pins.residue_cache_key,
            "class_map_sha256": ctx.pins.class_map_sha256,
            "calibration_config_sha256": config.sha256,
            "oof_sha256": ctx.fingerprint["oof_sha256"],
            "feature_table_sha256": ctx.fingerprint["feature_table_sha256"],
        },
        "classes": list(ctx.pins.classes),
        "labels": fit["labels"],
        "methods": fit["methods"],
        "fold_thresholds": fit["fold_thresholds"],
        "validation": valid["validation"],
        "phase9_reference": {
            "validation_macro_f1": head_frozen["validation"]["pooled"]["macro_f1"],
            "validation_jaccard": head_frozen["validation"]["pooled"]["jaccard"],
            "thresholds_on_raw_scores": head_frozen["thresholds"],
        },
        "software": fit["stage"]["versions"],
        "hardware": {
            "fit": fit["stage"]["hardware"],
            "fit_seconds": fit["stage"]["seconds"],
        },
    }


def _rules() -> dict[str, Path]:
    return {"baseline_class_map": CLASS_MAP, "consistency": CONSISTENCY}


def _calibrator_set(ctx: Context, labels: list[dict[str, Any]]) -> CalibratorSet:
    version = ctx.config.bundle_version
    return CalibratorSet(
        task=cal.TASK,
        calibration_version=f"{version}+{ctx.fingerprint['oof_sha256'][:12]}",
        classes=ctx.pins.classes,
        head_weights_sha256=ctx.pins.head_weights_sha256,
        head_frozen_sha256=ctx.pins.head_frozen_sha256,
        threshold_rule=cal.THRESHOLD_RULE,
        labels=tuple(label_from_dict(entry) for entry in labels),
    )


def run_freeze(args: argparse.Namespace) -> int:
    """Write the registry bundle and the manifest; both are fixed before anything is scored."""
    if args.smoke:
        _say("--smoke output can never be frozen")
        return 1
    if MANIFEST.exists():
        _say(f"{MANIFEST.name} exists; a frozen manifest is never rewritten")
        return 1
    _announce()
    ctx = _load(args)
    fit = read_manifest(OUT_DIR / FIT_FILE)
    valid = read_manifest(OUT_DIR / VALIDATION_FILE)
    if (
        fit.get("fingerprint") != ctx.fingerprint
        or valid.get("fingerprint") != ctx.fingerprint
    ):
        _say(
            "fit.json or validation.json was produced from other inputs; rerun --fit and --evaluate"
        )
        return 1
    directory = REPO_ROOT / ctx.config.bundle_directory / ctx.config.bundle_version
    if directory.exists():
        _say(f"{directory} exists; a frozen bundle is never rewritten")
        return 1
    calibrator = _calibrator_set(ctx, fit["labels"])
    head_block = {
        "weights_file": WEIGHTS,
        "weights_sha256": ctx.pins.head_weights_sha256,
        "frozen_sha256": ctx.pins.head_frozen_sha256,
        "head_config_sha256": ctx.head["frozen"]["inputs"]["head_config_sha256"],
        "classes": list(ctx.pins.classes),
    }
    rule_hashes = {
        f"{key}_sha256": sha256_file(path)[0] for key, path in _rules().items()
    }
    document = registry_bundle.build_document(
        version=ctx.config.bundle_version,
        head=head_block,
        calibrator=calibrator,
        rules=rule_hashes,
        scope={"scope": SUBSET_SCOPE, "scope_label": SUBSET_LABEL},
    )
    frozen = _frozen_block(ctx, fit, valid)
    frozen["inputs"]["calibrator_sha256"] = document["calibrator"]["sha256"]
    frozen["bundle"] = {
        "version": ctx.config.bundle_version,
        "directory": f"{ctx.config.bundle_directory}/{ctx.config.bundle_version}",
        "bundle_sha256": document["bundle_sha256"],
        "calibration_version": calibrator.calibration_version,
    }
    manifest: dict[str, Any] = {
        "schema_version": cal.SCHEMA_VERSION,
        "phase": cal.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }
    problems = cal.check_manifest(manifest, ctx.pins) + calibrator.problems()
    if problems:
        for problem in problems:
            _say(f"  {problem}")
        return 1
    directory.mkdir(parents=True)
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(calibrator.to_bytes())
    write_json(directory / registry_bundle.BUNDLE_FILE, document)
    registry_bundle.load_bundle(directory, _rules(), HEAD_DIR / WEIGHTS)
    write_json(MANIFEST, manifest)
    _say(
        f"froze {directory} and {MANIFEST}, frozen_sha256 {manifest['frozen_sha256'][:12]}"
    )
    return 0


# -- --check ---------------------------------------------------------------------------
def _bundle_problems(manifest: dict[str, Any], pins: cal.Pins) -> list[str]:
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
    if loaded.document["head"]["weights_sha256"] != pins.head_weights_sha256:
        found.append("the bundle names other head weights than head_metrics.json")
    return found


def run_check(args: argparse.Namespace) -> int:
    """No data bytes: configuration, pins and, when present, the manifest and the bundle."""
    del args
    problems: list[str] = []
    config = _config()
    assert_protocol(lopo_folds())
    pins = _pins()
    if MANIFEST.exists():
        manifest = read_manifest(MANIFEST)
        problems += cal.check_manifest(manifest, pins)
        if not problems:
            problems += _bundle_problems(manifest, pins)
        state = f"{MANIFEST.name} present"
    else:
        state = f"{MANIFEST.name} absent (configuration and pins only)"
    for problem in problems:
        _say(f"FAIL {problem}")
    _say(
        f"calibrate --check: {'FAIL' if problems else 'ok'}; bundle {config.bundle_version}; "
        f"{state}; partition 5 policy: {cal.TEST_POLICY}; scope: {SUBSET_LABEL}"
    )
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 10 calibration and thresholds.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("fit", "evaluate", "freeze", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="every 4th row, never frozen"
    )
    parser.add_argument(
        "--force", action="store_true", help="with --fit: refit even if kept"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one mode."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.force and not args.fit:
        parser.error("--force goes with --fit")
    try:
        if args.fit:
            return run_fit(args)
        if args.evaluate:
            return run_evaluate(args)
        if args.freeze:
            return run_freeze(args)
        return run_check(args)
    except BaselineError as error:
        _say(f"error: {error}")
        return 1
    except json.JSONDecodeError as error:
        _say(f"error: unreadable JSON: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
