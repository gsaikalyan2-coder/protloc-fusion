r"""Phase 9: the localization head (attention pooling + MLP) on the 500-sequence subset.

    python scripts/head.py --train [--smoke] [--force]
        Leave-one-partition-out over registry partitions 1-4. Every grid point of
        configs/training/head.yaml is fitted once per fold (early stopping on an inner
        whole-cluster holdout), scored out of fold, and the best by mean fold macro F1 is refit
        on partitions 1-4. Partition 5 is never read. Safe to kill and rerun: a finished
        grid point is kept. The script records its own wall time; do NOT wrap it in time or
        Measure-Command.

    python scripts/head.py --evaluate [--smoke]
        Validation metrics with cluster-bootstrap CIs; fold k is scored with thresholds tuned
        on the other partitions' out-of-fold scores, the frozen thresholds on all of them.

    python scripts/head.py --freeze
        Writes data/manifests/head_metrics.json: selected hyperparameters, thresholds, the
        weights hash, validation numbers, every input hash. Refuses if it already exists.

    python scripts/head.py --evaluate --test
        Scores partition 5 ONCE with the frozen weights and thresholds and adds it to the
        manifest. Refuses if the manifest is unsound or the test was already scored.

    python scripts/head.py --check
        Needs no data bytes (this is what CI runs). Exit 1 unless the configuration, the pins
        and the manifest (when present) agree.

--smoke runs every stage on every 4th row for 2 epochs and 2 grid points into data/head/smoke.
Smoke output is a code check only: it can never be frozen and no number from it is reported.

Every number is for the 500-sequence subset and is labelled "subset (500), not comparable to
DeepLoc or to the Phase 8 full-corpus numbers".

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

from protloc.data.embeddings import read_pack
from protloc.data.manifest import read_json_object, sha256_file, write_json
from protloc.data.residue_cache import (
    ResidueStore,
    residue_directory,
    verify_residue_cache,
)
from protloc.training import baseline_models as bm
from protloc.training import head, head_net
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
MANIFEST = REPO_ROOT / "data" / "manifests" / "head_metrics.json"
CLASS_MAP = REPO_ROOT / "backend" / "rules" / "baseline_class_map.yaml"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
ROOT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "trunk_esm2_150m.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "default.yaml"
HEAD_CONFIG = REPO_ROOT / "configs" / "training" / "head.yaml"
BASELINES = REPO_ROOT / "data" / "baselines"
EMBED_ROOT = REPO_ROOT / "data" / "embeddings"
PACK = EMBED_ROOT / "input_pack.tsv"
HEAD_DIR = REPO_ROOT / "data" / "head"
WEIGHTS = "final_weights.npz"
SMOKE_BOOTSTRAP = 50


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise BaselineError(f"{path}: top level must be a mapping")
    return {str(k): v for k, v in doc.items()}


def _out_dir(args: argparse.Namespace) -> Path:
    return HEAD_DIR / "smoke" if args.smoke else HEAD_DIR


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


def _config(args: argparse.Namespace) -> head.HeadConfig:
    """The head configuration, cross-checked against the root and model configs."""
    config = head.load_config(HEAD_CONFIG)
    if config.seed != int(_yaml(ROOT_CONFIG)["seed"]):
        raise BaselineError("head.yaml seed differs from configs/config.yaml")
    if config.architecture.embedding_dim != int(
        _yaml(MODEL_CONFIG)["trunk"]["embedding_dim"]
    ):
        raise BaselineError(
            "head.yaml embedding_dim differs from the trunk's embedding_dim"
        )
    if args.smoke:
        shrunk = replace(config.schedule, max_epochs=config.smoke.max_epochs)
        return replace(config, schedule=shrunk)
    return config


def _stride(args: argparse.Namespace, config: head.HeadConfig) -> int:
    return config.smoke.stride if args.smoke else 1


def _points(args: argparse.Namespace, config: head.HeadConfig) -> list[head.GridPoint]:
    points = head.grid_points(config)
    return points[: config.smoke.grid_points] if args.smoke else points


@dataclass(frozen=True, slots=True)
class Context:
    """Everything a data stage needs, loaded and cross-checked."""

    config: head.HeadConfig
    table: dict[str, Any]
    order: list[str]
    store: ResidueStore
    classes: tuple[str, ...]
    meta: dict[str, Any]
    residue_key: str


def _load(args: argparse.Namespace) -> Context:
    """Feature table (subset only), sequence order and per-residue store, checked together."""
    config = _config(args)
    cmap = load_class_map(CLASS_MAP)
    meta = read_manifest(BASELINES / "features.json")
    registry_pin, corpus_pin = _pinned_hashes()
    if meta["registry_sha256"] != registry_pin or meta["corpus_sha256"] != corpus_pin:
        raise BaselineError(
            "features.json was built from other data than the manifests record"
        )
    if meta["class_map_sha256"] != cmap.sha256:
        raise BaselineError(
            "the class map changed since baselines --features; rerun it"
        )
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
        raise BaselineError("the per-residue cache dimension differs from head.yaml")
    items, _header = read_pack(PACK)
    order = [item.sha256 for item in items]
    pooled_shas = {order[int(i)] for i in indices}
    if set(store.lengths) != pooled_shas:
        raise BaselineError(
            "the per-residue cache and the pooled cache hold different sequences"
        )
    residue_manifest = read_json_object(RESIDUE_MANIFEST)
    if residue_manifest.get("cache_key") != store.cache_key:
        raise BaselineError("residue_cache_manifest.json does not describe this cache")
    restricted = bm.restrict_to_sequences(table, indices)
    return Context(
        config=config,
        table=restricted,
        order=order,
        store=store,
        classes=tuple(cmap.classes[head.TASK]),
        meta={**meta, "subset": block},
        residue_key=store.cache_key,
    )


def _announce() -> None:
    _say(f"scope: {SUBSET_LABEL}")


# -- --train ---------------------------------------------------------------------------
def _grid_report_path(out: Path, index: int) -> Path:
    return out / "train" / f"grid_{index:02d}.json"


def _fingerprint(ctx: Context) -> dict[str, str]:
    """What a kept result must still agree with: change any of these and it is stale."""
    return {
        "head_config_sha256": ctx.config.sha256,
        "residue_cache_key": ctx.residue_key,
        "feature_table_sha256": str(ctx.meta["table_sha256"]),
        "class_map_sha256": str(ctx.meta["class_map_sha256"]),
    }


def _kept(path: Path, ctx: Context) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = read_manifest(path)
    if report.get("fingerprint") != _fingerprint(ctx):
        raise BaselineError(
            f"{path.name} was produced from other inputs; rerun with --force"
        )
    return report


def run_train(args: argparse.Namespace) -> int:
    """Leave-one-partition-out over partitions 1-4; the test partition is never read."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    config, np = ctx.config, importlib.import_module("numpy")
    folds = lopo_folds()
    assert_protocol(folds)
    rows = head.build_rows(
        ctx.table, ctx.order, VALIDATION_PARTITIONS, _stride(args, config)
    )
    source = head_net.Source(ctx.store, rows)
    leaks = head.held_out_clusters(ctx.table)
    out = _out_dir(args)
    (out / "train").mkdir(parents=True, exist_ok=True)
    points = _points(args, config)
    _say(
        f"{len(rows)} annotated localization rows, {len(points)} grid points x {len(folds)} folds"
    )
    reports: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        path = _grid_report_path(out, index)
        oof_path = out / "train" / f"grid_{index:02d}__oof.npy"
        kept = None if args.force else _kept(path, ctx)
        if kept is not None:
            _say(f"grid {index}: finished earlier, kept")
            reports.append(kept)
            continue
        _say(f"grid {index}: {point.as_dict()}")
        settings = head_net.settings_for(point, config, len(ctx.classes))
        began_point = time.perf_counter()
        oof, runs = head_net.run_grid_point(source, folds, settings, leaks, _say)
        mean_f1 = head.mean_fold_macro_f1(oof, rows, folds, ctx.classes)
        report = {
            "index": index,
            "params": point.as_dict(),
            "mean_fold_macro_f1": mean_f1,
            "folds": [asdict(r) for r in runs],
            "seconds": time.perf_counter() - began_point,
            "fingerprint": _fingerprint(ctx),
        }
        bm.save_array(oof_path, oof)
        write_json(path, report)
        reports.append(report)
        _say(
            f"grid {index}: mean fold macro F1 {mean_f1:.4f}, {report['seconds']:.1f} s"
        )
    scores = [float(r["mean_fold_macro_f1"]) for r in reports]
    best_index = max(range(len(scores)), key=lambda i: (scores[i], -i))
    best = reports[best_index]
    best_oof = bm.load_array(out / "train" / f"grid_{best_index:02d}__oof.npy")
    chosen = head.GridPoint(**best["params"])
    epochs = max(1, int(statistics.median_low(f["best_epoch"] for f in best["folds"])))
    _say(
        f"selected grid {best_index} {chosen.as_dict()}; refitting {epochs} epochs on 1-4"
    )
    settings = head_net.settings_for(chosen, config, len(ctx.classes))
    final = head_net.refit_final(source, settings, epochs, leaks)
    model = head_net.build_model(settings.net, config.seed, final.state)
    np.savez(out / WEIGHTS, **final.state)
    bm.save_array(out / "oof.npy", best_oof)
    final_report = {
        "selected_index": best_index,
        "selected_params": best["params"],
        "mean_fold_macro_f1": best["mean_fold_macro_f1"],
        "final_epochs": epochs,
        "trainable_parameters": model.parameter_count(),
        "final_fit_seconds": final.seconds,
        "grid": [
            {
                "index": r["index"],
                "params": r["params"],
                "mean_fold_macro_f1": r["mean_fold_macro_f1"],
                "seconds": r["seconds"],
            }
            for r in reports
        ],
        "train_rows": len(rows),
        "fingerprint": _fingerprint(ctx),
    }
    write_json(out / "train" / "final.json", final_report)
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
    _say(
        f"train stage: {stage['seconds']:.1f} s ({model.parameter_count()} trainable parameters)"
    )
    return 0


# -- --evaluate ------------------------------------------------------------------------
def run_validation(args: argparse.Namespace) -> int:
    """Validation metrics for the selected head, from its out-of-fold scores."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    out = _out_dir(args)
    final = read_manifest(out / "train" / "final.json")
    if final.get("fingerprint") != _fingerprint(ctx):
        raise BaselineError(
            "train/final.json was produced from other inputs; rerun --train"
        )
    rows = head.build_rows(
        ctx.table, ctx.order, VALIDATION_PARTITIONS, _stride(args, ctx.config)
    )
    oof = bm.load_array(out / "oof.npy")
    if oof.shape != rows.y.shape:
        raise BaselineError("oof.npy does not match the validation rows; rerun --train")
    split = bm.Split(rows.y, rows.partition, rows.cluster_code)
    boot = bm.Bootstrap(_samples(args), ctx.config.seed)
    block = bm.validate_oof(oof, split, lopo_folds(), ctx.classes, boot)
    pooled = block["pooled"]
    macro = pooled["macro_f1"]
    _say(
        f"validation macro F1 {macro['value']:.4f} [{macro['ci_low']:.4f}, {macro['ci_high']:.4f}] "
        f"Jaccard {pooled['jaccard']['value']:.4f}; the CI is wide: few clusters, one subset"
    )
    stage = {
        "seconds": time.perf_counter() - began,
        "samples": boot.samples,
        "seed": boot.seed,
    }
    write_json(
        out / "validation.json",
        {
            "scope": SUBSET_SCOPE,
            "label": SUBSET_LABEL,
            "stage": stage,
            "validation": block,
        },
    )
    _say(f"validation stage: {stage['seconds']:.1f} s")
    return 0


# -- the manifest ----------------------------------------------------------------------
def _pins() -> head.Pins:
    registry_pin, corpus_pin = _pinned_hashes()
    embedding = embedding_manifest(EMBEDDING_MANIFEST)
    residue = read_json_object(RESIDUE_MANIFEST) if RESIDUE_MANIFEST.exists() else {}
    cmap = load_class_map(CLASS_MAP)
    return head.Pins(
        registry_sha256=registry_pin,
        corpus_sha256=corpus_pin,
        embedding_cache_key=str(embedding["cache_key"]),
        residue_cache_key=str(residue.get("cache_key", "")),
        config_sha256=head.load_config(HEAD_CONFIG).sha256,
        classes=tuple(cmap.classes[head.TASK]),
        torch=head.pinned_version(PYPROJECT.read_text(encoding="utf-8"), "torch"),
    )


def run_freeze(args: argparse.Namespace) -> int:
    """Write the manifest: every value the test will be scored with, fixed before it is read."""
    if args.smoke:
        _say("--smoke output can never be frozen")
        return 1
    if MANIFEST.exists():
        _say(f"{MANIFEST.name} exists; a frozen manifest is never rewritten")
        return 1
    _announce()
    ctx = _load(args)
    out = _out_dir(args)
    validation = read_manifest(out / "validation.json")
    final = read_manifest(out / "train" / "final.json")
    stage = read_manifest(out / "train" / "stage.json")
    if final.get("fingerprint") != _fingerprint(ctx):
        _say(
            "train/final.json was produced from other inputs; rerun --train and --evaluate"
        )
        return 1
    config = ctx.config
    frozen: dict[str, Any] = {
        "protocol": {
            "adr": "ADR-0011 (protocol of ADR-0010)",
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
            "folds": [
                {"train": list(f.train), "validate": f.validate} for f in lopo_folds()
            ],
            "threshold_rule": (
                "per class, the score maximising MCC; fold k is scored with thresholds tuned on "
                "the out-of-fold scores of the other validation partitions; the frozen "
                "thresholds are tuned on all validation rows"
            ),
            "early_stopping": (
                "macro F1 on an inner whole-cluster holdout of the training partitions, "
                "thresholds tuned on that holdout; patience "
                f"{config.schedule.patience}, at most {config.schedule.max_epochs} epochs"
            ),
            "bootstrap": {
                "samples": validation["stage"]["samples"],
                "unit": "cluster_id",
                "seed": validation["stage"]["seed"],
            },
            "label_states": "annotated rows only (ADR-0006); unknown and not_applicable dropped",
        },
        "scope": SUBSET_SCOPE,
        "scope_label": SUBSET_LABEL,
        "subset": ctx.meta["subset"],
        "inputs": {
            "registry_sha256": ctx.meta["registry_sha256"],
            "corpus_sha256": ctx.meta["corpus_sha256"],
            "embedding_cache_key": ctx.meta["subset"]["embedding_cache_key"],
            "residue_cache_key": ctx.residue_key,
            "feature_table_sha256": ctx.meta["table_sha256"],
            "class_map_sha256": ctx.meta["class_map_sha256"],
            "head_config_sha256": config.sha256,
            "weights_sha256": sha256_file(out / WEIGHTS)[0],
        },
        "classes": list(ctx.classes),
        "architecture": {
            "pooling": "attention",
            "heads": config.architecture.heads,
            "attention_hidden": config.architecture.attention_hidden,
            "embedding_dim": config.architecture.embedding_dim,
            "loss": head.LOSS_KIND,
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
            "mean_fold_macro_f1": final["mean_fold_macro_f1"],
        },
        "grid": final["grid"],
        "thresholds": validation["validation"]["frozen_thresholds"],
        "validation": validation["validation"],
        "software": {"torch": _version("torch").split("+")[0], "versions": _versions()},
        "hardware": {
            "train": {
                **stage["hardware"],
                "seconds": stage["seconds"],
                "grid_seconds_total": stage["grid_seconds_total"],
                "final_fit_seconds": final["final_fit_seconds"],
                "torch_threads": stage["torch_threads"],
            }
        },
    }
    manifest = {
        "schema_version": head.SCHEMA_VERSION,
        "phase": 9,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }
    problems = head.check_head_manifest(manifest, _pins())
    if problems:
        for problem in problems:
            _say(f"  {problem}")
        return 1
    write_json(MANIFEST, manifest)
    _say(f"froze {MANIFEST}, frozen_sha256 {manifest['frozen_sha256'][:12]}")
    return 0


# -- --evaluate --test -----------------------------------------------------------------
def run_test(args: argparse.Namespace) -> int:
    """Score partition 5 once, with the weights and thresholds the manifest froze."""
    if args.smoke:
        _say("--smoke never touches the test partition")
        return 1
    began = time.perf_counter()
    manifest = read_manifest(MANIFEST)
    problems = head.check_head_manifest(manifest, _pins())
    if problems or manifest["test"] is not None:
        _say(
            "refusing: the manifest has problems or the test partition was already scored"
        )
        for problem in problems:
            _say(f"  {problem}")
        return 1
    _announce()
    ctx = _load(args)
    frozen = manifest["frozen"]
    weights_path = HEAD_DIR / WEIGHTS
    if sha256_file(weights_path)[0] != frozen["inputs"]["weights_sha256"]:
        _say("refusing: the weights are not the ones the manifest froze")
        return 1
    np = importlib.import_module("numpy")
    config = ctx.config
    chosen = head.GridPoint(**frozen["selected"]["params"])
    settings = head_net.settings_for(chosen, config, len(ctx.classes))
    with np.load(weights_path, allow_pickle=False) as handle:
        state = {name: handle[name] for name in handle.files}
    model = head_net.build_model(settings.net, config.seed, state)
    rows = head.build_rows(ctx.table, ctx.order, (TEST_PARTITION,), 1)
    train = head.build_rows(ctx.table, ctx.order, VALIDATION_PARTITIONS, 1)
    head.assert_no_overlap(
        {
            "train": head.clusters_of(train, range(len(train))),
            "test": head.clusters_of(rows, range(len(rows))),
        }
    )
    scores, _ = head_net.predict(
        model, ctx.store, rows.shas, config.schedule.batch_size
    )
    split = bm.Split(rows.y, rows.partition, rows.cluster_code)
    boot = bm.Bootstrap(_samples(args), config.seed)
    metrics = bm.score_test(scores, split, frozen["thresholds"], ctx.classes, boot)
    macro = metrics["macro_f1"]
    _say(
        f"test macro F1 {macro['value']:.4f} [{macro['ci_low']:.4f}, {macro['ci_high']:.4f}] "
        f"Jaccard {metrics['jaccard']['value']:.4f} on {len(rows)} rows (CI is wide)"
    )
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "seconds": time.perf_counter() - began,
        "hardware": bm.hardware(),
        "scope": SUBSET_SCOPE,
        "label": SUBSET_LABEL,
        "metrics": metrics,
    }
    write_json(MANIFEST, manifest)
    _say(f"test partition scored once in {manifest['test']['seconds']:.1f} s")
    return 0


# -- --check ---------------------------------------------------------------------------
def run_check(args: argparse.Namespace) -> int:
    """No data bytes: configuration, pins and, when present, the manifest."""
    del args
    problems: list[str] = []
    config = _config(argparse.Namespace(smoke=False))
    assert_protocol(lopo_folds())
    pins = _pins()
    if MANIFEST.exists():
        problems += head.check_head_manifest(read_manifest(MANIFEST), pins)
        state = f"{MANIFEST.name} present"
    else:
        state = f"{MANIFEST.name} absent (configuration and pins only)"
    for problem in problems:
        _say(f"FAIL {problem}")
    _say(
        f"head --check: {'FAIL' if problems else 'ok'}; torch=={pins.torch}; "
        f"{len(head.grid_points(config))} grid points; {state}; scope: {SUBSET_LABEL}"
    )
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 9 localization head.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("train", "evaluate", "freeze", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument(
        "--test", action="store_true", help="with --evaluate: score the test once"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="every 4th row, never frozen"
    )
    parser.add_argument(
        "--force", action="store_true", help="with --train: refit every grid point"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one mode."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.test and not args.evaluate:
        parser.error("--test goes with --evaluate")
    if args.force and not args.train:
        parser.error("--force goes with --train")
    try:
        if args.train:
            return run_train(args)
        if args.evaluate:
            return run_test(args) if args.test else run_validation(args)
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
