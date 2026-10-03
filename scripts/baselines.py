r"""Phase 8: classical baselines, the metric floor. One job per mode.

    python scripts/baselines.py --features [--force]
        Verifies the registry and corpus hashes, runs the motif engine once per DISTINCT
        sequence, reads the labels (annotated states only) and writes
        data/baselines/features.npz and features.json. The script records its own time.

    python scripts/baselines.py --train [--smoke] [--tasks T ...] [--models M ...] [--jobs N]
        Leave-one-partition-out over registry partitions 1-4: fits every grid point, picks
        the best by mean fold macro F1, refits it on partitions 1-4, saves out-of-fold
        scores and the fitted model. Partition 5 is never read. Safe to kill and rerun:
        a finished (task, model) is kept. Do NOT wrap it in time or Measure-Command.

    python scripts/baselines.py --evaluate [--smoke]
        Validation metrics with cluster-bootstrap CIs for every model and for the
        untrained motif rules; thresholds tuned on the other folds only.

    python scripts/baselines.py --freeze
        Writes data/manifests/baseline_metrics.json: validation numbers, params,
        thresholds, hashes, versions, hardware. Everything the test will be scored with.

    python scripts/baselines.py --evaluate --test
        Scores partition 5 ONCE under the frozen thresholds and adds it to the manifest.
        Refuses if the manifest is not frozen or the test was already scored.

    python scripts/baselines.py --check
        Needs no data bytes (this is what CI runs). Exit 1 unless the protocol pins, the
        scikit-learn pin and the manifest (when present) agree.

--smoke runs every stage on every 25th row into data/baselines/smoke. Smoke output is a
code check only: it can never be frozen and no number from it is reported.

--subset (with --train, --evaluate, --freeze, --evaluate --test, --check) runs the same protocol
on the 500 sequences of the Phase 7 embedding subset, reading the same feature table with every
other row masked out. Output goes to data/baselines/subset/ and its own manifest
data/baselines/subset/baseline_metrics.json; the frozen Phase 8 manifest is never touched. Every
number it produces is labelled "subset (500), not comparable to DeepLoc or to the Phase 8
full-corpus numbers". It is the floor the Phase 9 head is compared with.

Run from the repository root inside the project venv (pip install -e "backend[baselines]").
"""

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from protloc.data.embeddings import read_pack
from protloc.data.manifest import sha256_file, write_json
from protloc.data.splits import read_registry
from protloc.features.classical import extract
from protloc.training import baseline_models as bm
from protloc.training.baselines import (
    FEATURE_NAMES,
    GRIDS,
    MODEL_NAMES,
    RULES_NAME,
    SCHEMA_VERSION,
    SUBSET_LABEL,
    SUBSET_SCOPE,
    TASKS,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    VECTORISER_VERSION,
    BaselineError,
    ClassMap,
    assert_protocol,
    cache_lengths,
    check_manifest,
    check_subset_block,
    digest,
    grid_items,
    label_vector,
    load_class_map,
    lopo_folds,
    pinned_sklearn,
    read_manifest,
    rule_vector,
    subset_digest,
    vectorise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
REGISTRY = REPO_ROOT / "data" / "processed" / "splits" / "registry.tsv"
SPLIT_MANIFEST = REPO_ROOT / "data" / "manifests" / "split_manifest.json"
DATASET_MANIFEST = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
MANIFEST = REPO_ROOT / "data" / "manifests" / "baseline_metrics.json"
CLASS_MAP = REPO_ROOT / "backend" / "rules" / "baseline_class_map.yaml"
ENGINE = REPO_ROOT / "backend" / "protloc" / "features" / "classical.py"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
ROOT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "default.yaml"
BASELINES = REPO_ROOT / "data" / "baselines"
SUBSET_DIR = "subset"
EMBEDDING_MANIFEST = REPO_ROOT / "data" / "manifests" / "embedding_manifest.json"
EMBED_ROOT = REPO_ROOT / "data" / "embeddings"
PACK = EMBED_ROOT / "input_pack.tsv"
CACHE_DIR_HEX = 16
SMOKE_STRIDE = 25
PROGRESS_EVERY = 5000
MAX_JOBS = 10


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _out_dir(args: argparse.Namespace) -> Path:
    base = BASELINES / SUBSET_DIR if args.subset else BASELINES
    return base / "smoke" if args.smoke else base


def _manifest_path(args: argparse.Namespace) -> Path:
    return BASELINES / SUBSET_DIR / MANIFEST.name if args.subset else MANIFEST


def _stride(args: argparse.Namespace) -> int:
    return SMOKE_STRIDE if args.smoke else 1


def _announce(args: argparse.Namespace) -> None:
    if args.subset:
        _say(f"scope: {SUBSET_LABEL}")


def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise BaselineError(f"{path}: top level must be a mapping")
    return {str(k): v for k, v in doc.items()}


def _seed() -> int:
    return int(_yaml(ROOT_CONFIG)["seed"])


def _bootstrap_samples() -> int:
    return int(_yaml(EVAL_CONFIG)["reporting"]["bootstrap_samples"])


def _jobs(requested: int) -> int:
    return requested if requested > 0 else min(os.cpu_count() or 1, MAX_JOBS)


def _pinned_hashes() -> tuple[str, str]:
    """Registry and corpus SHA-256 as the committed manifests record them."""
    manifest = read_manifest(SPLIT_MANIFEST)
    return str(manifest["registry"]["sha256"]), str(manifest["corpus_sha256"])


def _engine_sha256() -> str:
    return sha256_file(ENGINE)[0]


# -- --features ------------------------------------------------------------------------
def _feature_key(cmap: ClassMap, registry_sha: str, corpus_sha: str) -> str:
    return digest(
        {
            "engine_sha256": _engine_sha256(),
            "corpus_sha256": corpus_sha,
            "registry_sha256": registry_sha,
            "class_map_sha256": cmap.sha256,
            "vectoriser_version": VECTORISER_VERSION,
            "feature_names": list(FEATURE_NAMES),
        }
    )


def run_features(args: argparse.Namespace) -> int:
    """Build the feature table: one engine call per distinct sequence."""
    began = time.perf_counter()
    cmap = load_class_map(CLASS_MAP)
    registry_pin, corpus_pin = _pinned_hashes()
    registry_sha, _ = sha256_file(REGISTRY)
    corpus_sha, _ = sha256_file(CORPUS)
    if registry_sha != registry_pin or corpus_sha != corpus_pin:
        _say(
            "hash mismatch: the registry or corpus is not the one the manifests record"
        )
        return 1
    _say(
        f"registry {registry_sha[:12]} and corpus {corpus_sha[:12]} match the manifests"
    )
    key = _feature_key(cmap, registry_sha, corpus_sha)
    meta_path = BASELINES / "features.json"
    table_path = BASELINES / "features.npz"
    if not args.force and meta_path.exists() and table_path.exists():
        meta = read_manifest(meta_path)
        if (
            meta.get("cache_key") == key
            and meta.get("table_sha256") == sha256_file(table_path)[0]
        ):
            _say(f"feature table is current (cache key {key[:12]}); nothing to do")
            return 0
    rows = read_registry(REGISTRY)
    wanted = {r["accession"]: r for r in rows}
    found: dict[str, dict[str, Any]] = {}
    sequences: dict[str, str] = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            accession = record["accession"]
            if accession in wanted:
                found[accession] = record
                sequences[record["sequence_sha256"]] = record["sequence_normalized"]
    missing = set(wanted) - set(found)
    if missing:
        _say(f"{len(missing)} registry accessions are not in the corpus")
        return 1
    _say(f"read {len(found)} records, {len(sequences)} distinct sequences")
    order = sorted(sequences)
    index_of = {sha: i for i, sha in enumerate(order)}
    x_rows: list[list[float]] = []
    rules: dict[str, list[list[float]]] = {task: [] for task in TASKS}
    for n, sha in enumerate(order, start=1):
        features = extract(sequences[sha])
        x_rows.append(vectorise(features))
        for task in TASKS:
            rules[task].append(rule_vector(task, features, cmap))
        if n % PROGRESS_EVERY == 0:
            _say(f"  {n}/{len(order)} sequences, {time.perf_counter() - began:.0f} s")
    accessions = [r["accession"] for r in rows]
    labels: dict[str, list[list[int] | None]] = {task: [] for task in TASKS}
    for accession in accessions:
        for task in TASKS:
            label = found[accession]["labels"][task]
            labels[task].append(
                label_vector(task, label["state"], label["values"], cmap)
            )
    table = bm.assemble(
        bm.RawTable(
            accessions=accessions,
            partitions=[int(r["partition"]) for r in rows],
            clusters=[r["cluster_id"] for r in rows],
            seq_index=[index_of[found[a]["sequence_sha256"]] for a in accessions],
            x_rows=x_rows,
            rules=rules,
            labels=labels,
            widths={task: len(cmap.classes[task]) for task in TASKS},
        )
    )
    bm.save_table(table_path, table)
    seconds = time.perf_counter() - began
    meta = {
        "schema_version": SCHEMA_VERSION,
        "cache_key": key,
        "table_sha256": sha256_file(table_path)[0],
        "registry_sha256": registry_sha,
        "corpus_sha256": corpus_sha,
        "engine_sha256": _engine_sha256(),
        "class_map_sha256": cmap.sha256,
        "vectoriser_version": VECTORISER_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "accessions": len(accessions),
        "distinct_sequences": len(order),
        "rows_per_task": {t: int(table[f"M_{t}"].sum()) for t in TASKS},
        "rows_by_partition": {
            t: {
                str(p): int(((table["partition"] == p) & table[f"M_{t}"]).sum())
                for p in (*VALIDATION_PARTITIONS, TEST_PARTITION)
            }
            for t in TASKS
        },
        "seconds": seconds,
        "hardware": bm.hardware(),
        "versions": bm.versions(),
    }
    write_json(meta_path, meta)
    _say(
        f"wrote {table_path.name} ({table_path.stat().st_size} bytes) in {seconds:.1f} s"
    )
    return 0


def _embedding_manifest() -> dict[str, Any]:
    manifest = read_manifest(EMBEDDING_MANIFEST)
    if manifest["sequences"]["scope"] != SUBSET_SCOPE:
        raise BaselineError(
            "the embedding manifest does not say 'subset'; --subset needs the subset cache"
        )
    return manifest


def _subset_block(table: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The subset as indices into the feature table, and the facts the manifest records."""
    embedding = _embedding_manifest()
    key = str(embedding["cache_key"])
    sequences = embedding["sequences"]
    directory = EMBED_ROOT / f"{key[:CACHE_DIR_HEX]}_subset{sequences['subset_size']}"
    lengths = cache_lengths(directory)
    if len(lengths) != int(sequences["cached"]):
        raise BaselineError(
            "the cache tables and the embedding manifest disagree on the count"
        )
    if sha256_file(PACK)[0] != embedding["input"]["pack_sha256"]:
        raise BaselineError(
            "input_pack.tsv is not the pack the embedding manifest records"
        )
    items, _header = read_pack(PACK)
    indices = bm.locate_sequences(table, [item.sha256 for item in items], lengths)
    restricted = bm.restrict_to_sequences(table, indices)
    labelled = {t: int(restricted[f"M_{t}"].sum()) for t in TASKS}
    in_subset = bm.in_sequences(table, indices)
    by_partition = {
        str(p): int(((table["partition"] == p) & in_subset).sum())
        for p in (*VALIDATION_PARTITIONS, TEST_PARTITION)
    }
    return indices, {
        "sequences": len(lengths),
        "sequences_sha256": subset_digest(list(lengths)),
        "embedding_cache_key": key,
        "cache_directory": directory.name,
        "accessions_by_partition": by_partition,
        "labelled_rows_per_task": labelled,
    }


def _load(subset: bool) -> tuple[dict[str, Any], dict[str, Any], ClassMap]:
    """The feature table, its metadata and the class map, after checking they belong together.

    With ``subset`` the table is the same one with every row outside the embedding subset
    masked out, and the metadata gains a ``subset`` entry describing it.
    """
    meta = read_manifest(BASELINES / "features.json")
    cmap = load_class_map(CLASS_MAP)
    registry_pin, corpus_pin = _pinned_hashes()
    if meta["registry_sha256"] != registry_pin or meta["corpus_sha256"] != corpus_pin:
        raise BaselineError(
            "features.json was built from other data than the manifests record"
        )
    if meta["cache_key"] != _feature_key(cmap, registry_pin, corpus_pin):
        raise BaselineError(
            "the engine or the class map changed since --features; rerun it"
        )
    table_path = BASELINES / "features.npz"
    if sha256_file(table_path)[0] != meta["table_sha256"]:
        raise BaselineError("features.npz does not match features.json")
    table = bm.load_table(table_path)
    if not subset:
        return table, dict(meta), cmap
    indices, block = _subset_block(table)
    return bm.restrict_to_sequences(table, indices), {**meta, "subset": block}, cmap


# -- --train ---------------------------------------------------------------------------
def run_train(args: argparse.Namespace) -> int:
    """Leave-one-partition-out over partitions 1-4; the test partition is never read."""
    began = time.perf_counter()
    _announce(args)
    table, _meta, cmap = _load(args.subset)
    folds = lopo_folds()
    assert_protocol(folds)
    out = _out_dir(args) / "train"
    out.mkdir(parents=True, exist_ok=True)
    seed, jobs = _seed(), _jobs(args.jobs)
    reports: list[str] = []
    for task in args.tasks:
        for model in args.models:
            stem = f"{task}__{model}"
            report_path = out / f"{stem}.json"
            if report_path.exists() and not args.force:
                _say(f"{stem}: finished earlier, kept")
                reports.append(stem)
                continue
            _say(
                f"{stem}: {len(GRIDS[model])} grid points x {len(folds)} folds, {jobs} jobs"
            )
            spec = bm.TrainSpec(
                task=task,
                model=model,
                classes=cmap.classes[task],
                seed=seed,
                n_jobs=jobs,
                stride=_stride(args),
            )
            report, oof, final = bm.train_model(
                table, spec, list(grid_items(model)), folds, _say
            )
            bm.save_array(out / f"{stem}__oof.npy", oof)
            bm.dump_model(out / f"{stem}__final.joblib", final)
            write_json(report_path, report)
            reports.append(stem)
    stage = {
        "seconds": time.perf_counter() - began,
        "jobs": jobs,
        "seed": seed,
        "hardware": bm.hardware(),
        "versions": bm.versions(),
        "finished": reports,
    }
    if args.subset:
        stage["scope"] = SUBSET_SCOPE
        stage["label"] = SUBSET_LABEL
    write_json(out / "stage.json", stage)
    _say(f"train stage: {stage['seconds']:.1f} s")
    return 0


# -- --evaluate ------------------------------------------------------------------------
def _validation_block(
    table: dict[str, Any], task: str, scores: Any, rows: Any, cmap: ClassMap
) -> dict[str, Any]:
    _x, y = bm.task_xy(table, task, rows)
    split = bm.Split(y, table["partition"][rows], table["cluster_code"][rows])
    boot = bm.Bootstrap(_bootstrap_samples(), _seed())
    return bm.validate_oof(scores, split, lopo_folds(), cmap.classes[task], boot)


def run_validation(args: argparse.Namespace) -> int:
    """Validation metrics for every trained model and for the untrained rules."""
    began = time.perf_counter()
    _announce(args)
    table, _meta, cmap = _load(args.subset)
    out = _out_dir(args)
    samples, seed = _bootstrap_samples(), _seed()
    tasks: dict[str, Any] = {}
    for task in args.tasks:
        rows = bm.task_rows(table, task, VALIDATION_PARTITIONS, _stride(args))
        models: dict[str, Any] = {}
        for model in args.models:
            stem = f"{task}__{model}"
            report = read_manifest(out / "train" / f"{stem}.json")
            oof = bm.load_array(out / "train" / f"{stem}__oof.npy")
            block = _validation_block(table, task, oof, rows, cmap)
            models[model] = {
                "params": report["selected_params"],
                "grid": report["grid"],
                "train_rows": report["train_rows"],
                "fit_seconds": report["final_fit_seconds"],
                "thresholds": block["frozen_thresholds"],
                "validation": block,
            }
            pooled = block["pooled"]
            _say(
                f"{stem}: macro F1 {pooled['macro_f1']['value']:.4f} "
                f"[{pooled['macro_f1']['ci_low']:.4f}, {pooled['macro_f1']['ci_high']:.4f}] "
                f"Jaccard {pooled['jaccard']['value']:.4f}"
            )
        rule_block = _validation_block(
            table, task, bm.rule_scores(table, task, rows), rows, cmap
        )
        tasks[task] = {
            "classes": list(cmap.classes[task]),
            "validation_rows": len(rows),
            "models": models,
            "rules": {
                "thresholds": rule_block["frozen_thresholds"],
                "validation": rule_block,
            },
        }
    stage = {"seconds": time.perf_counter() - began, "samples": samples, "seed": seed}
    document: dict[str, Any] = {"stage": stage, "tasks": tasks}
    if args.subset:
        document["scope"] = SUBSET_SCOPE
        document["label"] = SUBSET_LABEL
    write_json(out / "validation.json", document)
    _say(f"validation stage: {stage['seconds']:.1f} s")
    return 0


# -- --freeze --------------------------------------------------------------------------
def _manifest_problems(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    registry_pin: str,
    corpus_pin: str,
) -> list[str]:
    """Every disagreement of a manifest with the repository; the subset adds its own checks."""
    problems = check_manifest(
        manifest,
        registry_sha256=registry_pin,
        corpus_sha256=corpus_pin,
        sklearn_pin=pinned_sklearn(PYPROJECT.read_text(encoding="utf-8")),
    )
    if args.subset:
        frozen = manifest.get("frozen")
        key = str(read_manifest(EMBEDDING_MANIFEST)["cache_key"])
        if isinstance(frozen, dict):
            problems += check_subset_block(frozen, cache_key=key)
    return problems


def run_freeze(args: argparse.Namespace) -> int:
    """Write the manifest: every value the test will be scored with, fixed before it is read."""
    if args.smoke:
        _say("--smoke output can never be frozen")
        return 1
    manifest_path = _manifest_path(args)
    if manifest_path.exists():
        _say(f"{manifest_path.name} exists; a frozen manifest is never rewritten")
        return 1
    _announce(args)
    out = _out_dir(args)
    table, meta, cmap = _load(args.subset)
    validation = read_manifest(out / "validation.json")
    stage = read_manifest(out / "train" / "stage.json")
    tasks = validation["tasks"]
    if set(tasks) != set(TASKS) or any(
        set(t["models"]) != set(MODEL_NAMES) for t in tasks.values()
    ):
        _say(
            "validation.json must cover every task and model; run --train and --evaluate in full"
        )
        return 1
    versions = bm.versions()
    rows_per_task = meta["rows_per_task"]
    rows_by_partition = meta["rows_by_partition"]
    if args.subset:
        rows_per_task = {t: int(table[f"M_{t}"].sum()) for t in TASKS}
        rows_by_partition = {
            t: {
                str(p): int(((table["partition"] == p) & table[f"M_{t}"]).sum())
                for p in (*VALIDATION_PARTITIONS, TEST_PARTITION)
            }
            for t in TASKS
        }
    frozen: dict[str, Any] = {
        "protocol": {
            "adr": "ADR-0010",
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
            "zero_based_mapping": "registry partition n is evaluation/default.yaml index n-1",
            "folds": [
                {"train": list(f.train), "validate": f.validate} for f in lopo_folds()
            ],
            "threshold_rule": (
                "per class, the score maximising MCC; fold k is scored with thresholds tuned on "
                "the out-of-fold scores of the other validation partitions; the frozen "
                "thresholds are tuned on all validation rows"
            ),
            "bootstrap": {
                "samples": validation["stage"]["samples"],
                "unit": "cluster_id",
                "seed": validation["stage"]["seed"],
            },
            "label_states": (
                "annotated rows only; negative only where ADR-0006 3.3 makes it informative"
            ),
        },
        "inputs": {
            "registry_sha256": meta["registry_sha256"],
            "corpus_sha256": meta["corpus_sha256"],
            "feature_table_sha256": meta["table_sha256"],
            "engine_sha256": meta["engine_sha256"],
            "class_map_sha256": cmap.sha256,
        },
        "features": {
            "count": len(FEATURE_NAMES),
            "names": list(FEATURE_NAMES),
            "vectoriser_version": VECTORISER_VERSION,
            "rows_per_task": rows_per_task,
            "rows_by_partition": rows_by_partition,
        },
        "software": {"scikit_learn": versions["scikit_learn"], "versions": versions},
        "hardware": {
            "features": {**meta["hardware"], "seconds": meta["seconds"]},
            "train": {
                **stage["hardware"],
                "seconds": stage["seconds"],
                "jobs": stage["jobs"],
            },
        },
        "tasks": tasks,
    }
    if args.subset:
        frozen["scope"] = SUBSET_SCOPE
        frozen["scope_label"] = SUBSET_LABEL
        frozen["subset"] = meta["subset"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": 9 if args.subset else 8,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }
    registry_pin, corpus_pin = _pinned_hashes()
    problems = _manifest_problems(args, manifest, registry_pin, corpus_pin)
    if problems:
        for problem in problems:
            _say(f"  {problem}")
        return 1
    write_json(manifest_path, manifest)
    _say(f"froze {manifest_path}, frozen_sha256 {manifest['frozen_sha256'][:12]}")
    return 0


# -- --evaluate --test -----------------------------------------------------------------
def _say_test_row(task: str, model: str, block: dict[str, Any]) -> None:
    macro, jaccard = block["macro_f1"], block["jaccard"]
    _say(
        f"  test {task}/{model}: macro F1 {macro['value']:.4f} "
        f"[{macro['ci_low']:.4f}, {macro['ci_high']:.4f}] Jaccard {jaccard['value']:.4f}"
    )


def run_test(args: argparse.Namespace) -> int:
    """Score partition 5 once, under the thresholds the manifest froze."""
    if args.smoke:
        _say("--smoke never touches the test partition")
        return 1
    began = time.perf_counter()
    manifest_path = _manifest_path(args)
    manifest = read_manifest(manifest_path)
    registry_pin, corpus_pin = _pinned_hashes()
    problems = _manifest_problems(args, manifest, registry_pin, corpus_pin)
    if problems or manifest["test"] is not None:
        _say(
            "refusing: the manifest has problems or the test partition was already scored"
        )
        for problem in problems:
            _say(f"  {problem}")
        return 1
    _announce(args)
    table, _meta, cmap = _load(args.subset)
    boot = bm.Bootstrap(_bootstrap_samples(), _seed())
    scored: dict[str, Any] = {}
    for task in TASKS:
        body = manifest["frozen"]["tasks"][task]
        rows = bm.task_rows(table, task, (TEST_PARTITION,), 1)
        x, y = bm.task_xy(table, task, rows)
        split = bm.Split(y, table["partition"][rows], table["cluster_code"][rows])
        scored[task] = {}
        classes = cmap.classes[task]
        for model in MODEL_NAMES:
            fitted = bm.load_model(
                _out_dir(args) / "train" / f"{task}__{model}__final.joblib"
            )
            thresholds = body["models"][model]["thresholds"]
            scored[task][model] = bm.score_test(
                bm.model_scores(fitted, x), split, thresholds, classes, boot
            )
            _say_test_row(task, model, scored[task][model])
        scored[task][RULES_NAME] = bm.score_test(
            bm.rule_scores(table, task, rows),
            split,
            body["rules"]["thresholds"],
            classes,
            boot,
        )
        _say_test_row(task, RULES_NAME, scored[task][RULES_NAME])
        _say(f"{task}: scored {len(rows)} test rows")
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "seconds": time.perf_counter() - began,
        "hardware": bm.hardware(),
        "tasks": scored,
    }
    if args.subset:
        manifest["test"]["scope"] = SUBSET_SCOPE
        manifest["test"]["label"] = SUBSET_LABEL
    write_json(manifest_path, manifest)
    _say(f"test partition scored once in {manifest['test']['seconds']:.1f} s")
    return 0


# -- --check ---------------------------------------------------------------------------
def run_check(args: argparse.Namespace) -> int:
    """No data bytes: protocol pins, scikit-learn pin and, when present, the manifest."""
    problems: list[str] = []
    cmap = load_class_map(CLASS_MAP)
    pin = pinned_sklearn(PYPROJECT.read_text(encoding="utf-8"))
    assert_protocol(lopo_folds())
    registry_pin, corpus_pin = _pinned_hashes()
    label_map = _yaml(REPO_ROOT / "backend" / "rules" / "uniprot_label_map.yaml")
    if list(cmap.classes["localization"]) != list(label_map["localization"]["classes"]):
        problems.append(
            "baseline_class_map localization classes differ from uniprot_label_map"
        )
    manifest_path = _manifest_path(args)
    if manifest_path.exists():
        problems += _manifest_problems(
            args, read_manifest(manifest_path), registry_pin, corpus_pin
        )
        state = f"{manifest_path.name} present"
    else:
        state = f"{manifest_path.name} absent (pins only)"
    if args.subset:
        state += f"; scope: {SUBSET_LABEL}"
    for problem in problems:
        _say(f"FAIL {problem}")
    _say(
        f"baselines --check: {'FAIL' if problems else 'ok'}; scikit-learn=={pin}; {state}"
    )
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8 classical baselines.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("features", "train", "evaluate", "freeze", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument(
        "--test", action="store_true", help="with --evaluate: score the test once"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="every 25th row, never frozen"
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="the 500-sequence embedding subset; output in data/baselines/subset",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one mode."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.subset and (args.smoke or args.features):
        parser.error("--subset cannot be combined with --smoke or --features")
    try:
        if args.features:
            return run_features(args)
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


if __name__ == "__main__":
    sys.exit(main())
