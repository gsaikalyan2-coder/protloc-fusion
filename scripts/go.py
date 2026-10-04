r"""Phase 12: the GO head (multi-label GO molecular function) on the 500-sequence subset.

    python scripts/go.py --labels
        Reads the pinned ontology release (data/raw/go-basic.obo, SHA-256 checked), maps the
        subset's corpus GO ids onto it (alt_id, replaced_by, obsolete, absent, other aspect),
        propagates over is_a and part_of, applies the evidence policy and counts the vocabulary on
        SUBSET rows only. Writes data/go/{labels.tsv,labels.json,vocabulary.json}.

    python scripts/go.py --train [--smoke] [--force]
        Leave-one-partition-out over registry partitions 1-4. Every grid point of
        configs/training/go.yaml is fitted once per fold (early stopping on an inner whole-cluster
        holdout), scored out of fold, and the best by mean fold precision@k (tie-break: mean fold
        Fmax) is refit on partitions 1-4. Partition 5 is never read. Safe to kill and rerun: a
        finished grid point is kept. The script records its own wall time; do NOT wrap it in time
        or Measure-Command.

    python scripts/go.py --evaluate [--smoke]
        Re-derives the out-of-fold scores from the saved fold weights (they must match to 1e-9 or
        the stage fails), then scores the model and the three floors (training-set frequency
        prior, the Phase 8 logistic regression extended to multi-label, the PROSITE-style motif
        hits) under the identical protocol, with cluster-bootstrap CIs.

    python scripts/go.py --calibrate [--smoke]
        Phase 10 machinery per eligible term: cross-fitted calibrators, one global nested-Fmax
        threshold, ECE and Brier before and after, and the calibrated metrics.

    python scripts/go.py --freeze
        Writes registry/go_function/<version>/{terms.json,calibrator.json,bundle.json} and
        data/manifests/go_metrics.json. Refuses if either exists.

    python scripts/go.py --evaluate --test
        Scores partition 5 ONCE with the frozen weights, calibrators and thresholds and adds it
        to the manifest. Refuses if the manifest is unsound or the test was already scored.

    python scripts/go.py --check
        Needs no data bytes (this is what CI runs). Exit 1 unless the configuration, the pins,
        the manifest and the registry bundle (when present) agree.

--smoke runs every stage on every 4th row for 2 epochs and 2 grid points into data/go/smoke.
Smoke output is a code check only: it can never be frozen and no number from it is reported.

Every number is for the 500-sequence subset and is labelled "subset (500), not comparable to
DeepLoc or to the Phase 8 full-corpus numbers". No published GO benchmark figure is compared with
a subset number.

Run from the repository root inside the project venv (pip install -e "backend[head,baselines]").
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
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
from protloc.training import baselines, go, go_net, head, head_net
from protloc.training import calibration as cal
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
CRF_MANIFEST = REPO_ROOT / "data" / "manifests" / "crf_metrics.json"
MANIFEST = REPO_ROOT / "data" / "manifests" / "go_metrics.json"
CLASS_MAP = REPO_ROOT / "backend" / "rules" / "baseline_class_map.yaml"
CONSISTENCY = REPO_ROOT / "backend" / "rules" / "consistency.json"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
ROOT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "trunk_esm2_150m.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "default.yaml"
GO_CONFIG = REPO_ROOT / "configs" / "training" / "go.yaml"
CLASSICAL = REPO_ROOT / "backend" / "protloc" / "features" / "classical.py"
BASELINES = REPO_ROOT / "data" / "baselines"
EMBED_ROOT = REPO_ROOT / "data" / "embeddings"
PACK = EMBED_ROOT / "input_pack.tsv"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
GO_DIR = REPO_ROOT / "data" / "go"
LABELS_TSV = "labels.tsv"
LABELS_JSON = "labels.json"
VOCABULARY_JSON = "vocabulary.json"
WEIGHTS = "final_weights.npz"
OOF_SCORE = "oof_score.npy"
FIT_FILE = "fit.json"
CROSSFIT_FILE = "crossfit.npy"
VALIDATION_FILE = "validation.json"
SMOKE_BOOTSTRAP = 50
THRESHOLD_RULE = (
    "one global threshold on the propagated score (a term scores at least as high as any "
    "descendant); it is the CAFA-style Fmax maximiser over the labelled rows that have a "
    "vocabulary positive; fold k is scored with a threshold tuned on the out-of-fold scores of "
    "the other validation partitions; the frozen threshold is tuned on all validation rows. The "
    "calibrated threshold is tuned the same way on the cross-fitted calibrated scores"
)
SELECTION_RULE = (
    "mean over folds of precision@k (k = metrics.primary_k) over rows with a vocabulary "
    "positive; tie-break mean fold Fmax under nested thresholds; then the earlier grid point"
)
FMAX_RULE = (
    "precision is averaged over rows with at least one term at or above the threshold, recall "
    "over all rows with a vocabulary positive, F = 2PR/(P+R); the threshold is nested (see "
    "threshold_rule), so Fmax here is F at the nested threshold, not an oracle maximum"
)
ARMS = go.ARMS
FLOOR_SOURCES = {
    "model_raw": go.SOURCE_MODEL,
    "model_calibrated": go.SOURCE_MODEL,
    "frequency_prior": "prior",
    "logreg": "logreg",
    "motif": go.SOURCE_MOTIF,
}


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
    return GO_DIR / "smoke" if args.smoke else GO_DIR


def _pinned_hashes() -> tuple[str, str]:
    manifest = read_manifest(SPLIT_MANIFEST)
    return str(manifest["registry"]["sha256"]), str(manifest["corpus_sha256"])


def _version(name: str) -> str:
    metadata = importlib.import_module("importlib.metadata")
    return str(metadata.version(name))


def _versions() -> dict[str, str]:
    names = ("torch", "numpy", "pyyaml", "scikit-learn")
    return {name: _version(name) for name in names}


def _samples(args: argparse.Namespace) -> int:
    configured = int(_yaml(EVAL_CONFIG)["reporting"]["bootstrap_samples"])
    return SMOKE_BOOTSTRAP if args.smoke else configured


def _config(args: argparse.Namespace) -> go.GoConfig:
    """The GO configuration, cross-checked against the root and model configs."""
    config = go.load_config(GO_CONFIG)
    if config.seed != int(_yaml(ROOT_CONFIG)["seed"]):
        raise BaselineError("go.yaml seed differs from configs/config.yaml")
    if config.architecture.embedding_dim != int(_yaml(MODEL_CONFIG)["trunk"]["embedding_dim"]):
        raise BaselineError("go.yaml embedding_dim differs from the trunk's embedding_dim")
    if args.smoke:
        shrunk = replace(config.schedule, max_epochs=config.smoke.max_epochs)
        return replace(config, schedule=shrunk)
    return config


def _calibration_config(config: go.GoConfig) -> cal.CalibrationConfig:
    base = cal.load_config(REPO_ROOT / config.calibration_config)
    if base.seed != config.seed:
        raise BaselineError("calibration.yaml seed differs from go.yaml")
    return go.calibration_settings(base, config)


def _stride(args: argparse.Namespace, config: go.GoConfig) -> int:
    return config.smoke.stride if args.smoke else 1


def _points(args: argparse.Namespace, config: go.GoConfig) -> list[go.GridPoint]:
    points = go.grid_points(config)
    return points[: config.smoke.grid_points] if args.smoke else points


def _announce() -> None:
    _say(f"scope: {SUBSET_LABEL}")


# -- the subset and the labels ---------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Subset:
    """The feature table, the subset mask and the facts a manifest records about them."""

    config: go.GoConfig
    table: dict[str, Any]
    member: Any
    order: list[str]
    meta: dict[str, Any]
    indices: Any
    embedding: dict[str, Any]


def _subset(args: argparse.Namespace) -> Subset:
    """Feature table (whole working set), the subset mask and the sequence order, checked."""
    config = _config(args)
    cmap = load_class_map(CLASS_MAP)
    meta = read_manifest(BASELINES / "features.json")
    registry_pin, corpus_pin = _pinned_hashes()
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
    items, _header = read_pack(PACK)
    return Subset(
        config=config,
        table=table,
        member=bm.in_sequences(table, indices),
        order=[item.sha256 for item in items],
        meta={**meta, "subset": block},
        indices=indices,
        embedding=embedding,
    )


def _subset_accessions(sub: Subset) -> list[str]:
    """Accessions of the subset, from the feature-table mask and nothing else."""
    return sorted(str(a) for a in sub.table["accession"][sub.member])


def _label_fingerprint(sub: Subset, onto_sha: str) -> dict[str, str]:
    registry_pin, corpus_pin = _pinned_hashes()
    return {
        "corpus_sha256": corpus_pin,
        "registry_sha256": registry_pin,
        "feature_table_sha256": str(sub.meta["table_sha256"]),
        "accessions_sha256": digest(_subset_accessions(sub)),
        "ontology_sha256": onto_sha,
        "label_policy_sha256": sub.config.labels_sha256,
        "motif_engine_sha256": sha256_file(CLASSICAL)[0],
    }


def _term_table(vocab: go.Vocabulary) -> list[dict[str, Any]]:
    return [
        {
            "id": t.id,
            "name": t.name,
            "positives_validation": t.positives,
            "by_partition": dict(t.by_partition),
            "ancestors_in_vocabulary": list(t.ancestors),
        }
        for t in vocab.terms
    ]


def run_labels(args: argparse.Namespace) -> int:
    """Map the subset's corpus GO ids onto the pinned ontology and count the vocabulary."""
    began = time.perf_counter()
    _announce()
    sub = _subset(args)
    config = sub.config
    onto = go.load_ontology(REPO_ROOT / config.ontology.file, config.ontology)
    crossing = go.cross_aspect_edges(onto, config.ontology)
    if crossing:
        raise BaselineError(f"{crossing} propagation edges leave {config.ontology.namespace}")
    if sha256_file(CORPUS)[0] != _pinned_hashes()[1]:
        raise BaselineError("data/processed/corpus.jsonl is not the corpus the manifests pin")
    wanted = _subset_accessions(sub)
    scan = go.scan_corpus(CORPUS, wanted)
    for code in scan.code_occurrences:
        config.evidence.class_of(code)
    ctx = go.LabelContext(onto, go.Ancestry(onto, config.ontology), config)
    labels = go.build_labels(scan, ctx)
    partition_of = {
        str(a): int(p) for a, p in zip(sub.table["accession"], sub.table["partition"], strict=True)
    }
    vocab = go.build_vocabulary(labels, set(wanted), partition_of, ctx)
    GO_DIR.mkdir(parents=True, exist_ok=True)
    tsv = GO_DIR / LABELS_TSV
    tsv.write_text(go.label_table_text(labels), encoding="utf-8", newline="\n")
    vocabulary = GO_DIR / VOCABULARY_JSON
    write_json(vocabulary, go.vocabulary_document(vocab, config))
    subset_terms: Counter[str] = Counter()
    for record in scan.records.values():
        subset_terms.update(record.values)
    namespace = config.ontology.namespace
    info = {
        **_label_fingerprint(sub, config.ontology.sha256),
        "label_table_sha256": sha256_file(tsv)[0],
        "vocabulary_sha256": sha256_file(vocabulary)[0],
        "accessions": len(wanted),
        "corpus_lines": scan.lines,
        "corpus_states": dict(sorted(scan.states.items())),
        "evidence_codes_corpus": {
            code: {"class": config.evidence.class_of(code), "occurrences": n}
            for code, n in sorted(scan.code_occurrences.items())
        },
        "mapping_whole_corpus": go.mapping_report(scan.term_occurrences, onto, namespace),
        "mapping_subset": go.mapping_report(subset_terms, onto, namespace),
        "cross_aspect_edges": crossing,
        "summary": go.label_summary(labels, partition_of),
        "vocabulary": go.vocabulary_block(vocab),
        "terms": _term_table(vocab),
        "seconds": time.perf_counter() - began,
    }
    write_json(GO_DIR / LABELS_JSON, info)
    _print_labels(info)
    return 0


def _print_labels(info: dict[str, Any]) -> None:
    summary = info["summary"]
    _say(
        f"subset accessions {info['accessions']}: states {summary['states']}; labelled "
        f"{summary['labelled']} (by partition {summary['labelled_by_partition']}); annotated "
        f"without a policy term {summary['annotated_without_policy_term']}; with a motif hit "
        f"{summary['with_motif_hit']}"
    )
    for name in ("mapping_whole_corpus", "mapping_subset"):
        m = info[name]
        _say(f"{name}: occurrences {m['occurrences']} (distinct ids {m['distinct_ids']})")
    vocab = info["vocabulary"]
    _say(f"vocabulary: {len(vocab['ids'])} terms with >= {vocab['term_floor']} positives (1-4)")
    for term in info["terms"]:
        _say(
            f"  {term['id']} {term['positives_validation']:4d}  {term['by_partition']}  "
            f"{term['name']}"
        )
    under = sorted(vocab["undefined"].items(), key=lambda kv: -kv[1])[:10]
    _say(f"under the floor: {len(vocab['undefined'])} terms; largest {under}")
    _say(f"labels stage: {info['seconds']:.1f} s")


# -- everything a data stage needs -----------------------------------------------------
@dataclass(frozen=True, slots=True)
class Context:
    """The subset, its labels, the vocabulary and the per-residue store, cross-checked."""

    sub: Subset
    config: go.GoConfig
    store: ResidueStore
    labels: dict[str, go.LabelRow]
    label_info: dict[str, Any]
    vocab: go.Vocabulary
    residue_key: str


def _load(args: argparse.Namespace) -> Context:
    """The subset, the label table built by ``--labels`` and the per-residue cache, checked."""
    sub = _subset(args)
    config = sub.config
    tsv, info_path, vocabulary = GO_DIR / LABELS_TSV, GO_DIR / LABELS_JSON, GO_DIR / VOCABULARY_JSON
    if not (tsv.exists() and info_path.exists() and vocabulary.exists()):
        raise BaselineError("no GO label table: run `python scripts/go.py --labels` first")
    info = read_manifest(info_path)
    wanted = _label_fingerprint(sub, config.ontology.sha256)
    if {k: info.get(k) for k in wanted} != wanted:
        raise BaselineError("the GO label table is stale (inputs changed): rerun --labels")
    if sha256_file(tsv)[0] != info["label_table_sha256"]:
        raise BaselineError("labels.tsv does not match labels.json")
    if sha256_file(vocabulary)[0] != info["vocabulary_sha256"]:
        raise BaselineError("vocabulary.json does not match labels.json")
    labels = go.parse_label_table(tsv.read_text(encoding="utf-8"))
    if set(labels) != set(_subset_accessions(sub)):
        raise BaselineError("the label table does not cover exactly the subset accessions")
    vocab = go.vocabulary_from_document(read_json_object(vocabulary))
    directory = residue_directory(pooled_directory(sub.embedding, EMBED_ROOT))
    problems = verify_residue_cache(directory)
    if problems:
        raise BaselineError(f"the per-residue cache is not sound: {problems[0]}")
    store = ResidueStore(directory)
    if store.dim != config.architecture.embedding_dim:
        raise BaselineError("the per-residue cache dimension differs from go.yaml")
    if set(store.lengths) != {sub.order[int(i)] for i in sub.indices}:
        raise BaselineError("the per-residue cache and the pooled cache hold different sequences")
    if read_json_object(RESIDUE_MANIFEST).get("cache_key") != store.cache_key:
        raise BaselineError("residue_cache_manifest.json does not describe this cache")
    return Context(
        sub=sub,
        config=config,
        store=store,
        labels=labels,
        label_info=info,
        vocab=vocab,
        residue_key=store.cache_key,
    )


def _rows(ctx: Context, partitions: Sequence[int], stride: int) -> go.Rows:
    spec = (ctx.vocab, partitions, stride)
    return go.build_rows(ctx.sub.table, ctx.sub.order, ctx.sub.member, ctx.labels, spec)


def _fingerprint(ctx: Context) -> dict[str, str]:
    """What a kept result must still agree with: change any of these and it is stale."""
    return {
        "go_config_sha256": ctx.config.sha256,
        "residue_cache_key": ctx.residue_key,
        "feature_table_sha256": str(ctx.sub.meta["table_sha256"]),
        "label_table_sha256": str(ctx.label_info["label_table_sha256"]),
        "vocabulary_sha256": str(ctx.label_info["vocabulary_sha256"]),
        "ontology_sha256": ctx.config.ontology.sha256,
    }


def _scoring(ctx: Context, args: argparse.Namespace) -> go.Scoring:
    return go.Scoring(ctx.config.metrics, bm.Bootstrap(_samples(args), ctx.config.seed))


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
    ctx: Context, source: go_net.Source, point: go.GridPoint, index: int, out: Path
) -> dict[str, Any]:
    """Fit one grid point on every fold, score it out of fold and write its files."""
    np = _np()
    config, rows = ctx.config, source.rows
    folds = lopo_folds()
    leaks = head.held_out_clusters(ctx.sub.table)
    plan = go_net.plan_for(point, config, len(ctx.vocab))
    began = time.perf_counter()
    result = go_net.run_grid_point(source, folds, plan, leaks, _say)
    score = go.propagate_scores(result.oof, ctx.vocab)
    selector, fmax = go.fold_means(score, rows.y, rows.partition, config.metrics)
    bm.save_array(out / "train" / f"grid_{index:02d}__oof_score.npy", result.oof)
    for validate, state in result.states.items():
        np.savez(_fold_weights(out, index, validate), **state)
    report = {
        "index": index,
        "params": point.as_dict(),
        "selector": selector,
        "mean_fold_fmax": fmax,
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
    source = go_net.Source(ctx.store, rows)
    out = _out_dir(args)
    (out / "train").mkdir(parents=True, exist_ok=True)
    points = _points(args, config)
    _say(
        f"{len(rows)} labelled rows, {int(go.evaluable(rows.y).sum())} with a vocabulary "
        f"positive, {len(ctx.vocab)} terms, {len(points)} grid points x {len(folds)} folds"
    )
    reports: list[dict[str, Any]] = []
    fitted_now = 0
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
            f"grid {index}: precision@{config.metrics.primary_k} {report['selector']:.4f}, "
            f"mean fold Fmax {report['mean_fold_fmax']:.4f}, {report['seconds']:.1f} s"
        )
        fitted_now += 1
        if fitted_now == 1:
            _say(
                f"projected: {report['seconds'] * len(points) / 60.0:.1f} min for "
                f"{len(points)} grid points (this one took {report['seconds']:.1f} s)"
            )
    best_index = go.select_point([(r["selector"], r["mean_fold_fmax"]) for r in reports])
    best = reports[best_index]
    chosen = go.GridPoint(**best["params"])
    epochs = max(1, int(statistics.median_low(f["best_epoch"] for f in best["folds"])))
    _say(f"selected grid {best_index} {chosen.as_dict()}; refitting {epochs} epochs on 1-4")
    plan = go_net.plan_for(chosen, config, len(ctx.vocab))
    final = go_net.refit_final(source, plan, epochs, head.held_out_clusters(ctx.sub.table))
    model = head_net.build_model(plan.settings.net, config.seed, final.state)
    np.savez(out / WEIGHTS, **final.state)
    prefix = out / "train" / f"grid_{best_index:02d}__"
    bm.save_array(out / OOF_SCORE, bm.load_array(Path(f"{prefix}oof_score.npy")))
    write_json(
        out / "train" / "final.json",
        {
            "selected_index": best_index,
            "selected_params": best["params"],
            "selector": best["selector"],
            "mean_fold_fmax": best["mean_fold_fmax"],
            "final_epochs": epochs,
            "trainable_parameters": model.parameter_count(),
            "final_fit_seconds": final.seconds,
            "grid": [
                {
                    "index": r["index"],
                    "params": r["params"],
                    "selector": r["selector"],
                    "mean_fold_fmax": r["mean_fold_fmax"],
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
    """The validation rows and the selected grid point's raw out-of-fold scores."""

    rows: go.Rows
    raw: Any


def _out_of_fold(ctx: Context, args: argparse.Namespace, out: Path) -> OutOfFold:
    rows = _rows(ctx, VALIDATION_PARTITIONS, _stride(args, ctx.config))
    raw = bm.load_array(out / OOF_SCORE)
    if raw.shape != rows.y.shape:
        raise BaselineError("the out-of-fold file does not match the rows; rerun --train")
    return OutOfFold(rows, raw)


def _assemble(
    source: str, rows: go.Rows, scored: tuple[Any, Any], ctx: Context, scoring: go.Scoring
) -> dict[str, Any]:
    """One arm's report from (scores, decisions): pooled, per fold, per term."""
    score, decision = scored
    ks = ctx.config.metrics.k_values
    return {
        "source": source,
        "pooled": go.metric_report(score, rows.y, decision, rows.cluster_code, scoring),
        "folds": go.fold_summaries(score, rows.y, decision, rows.partition, ks),
        "per_term": go.per_term_report(score, rows.y, rows.cluster_code, ctx.vocab, scoring),
    }


def _nested_arm(
    source: str, rows: go.Rows, score: Any, ctx: Context, scoring: go.Scoring
) -> tuple[dict[str, Any], dict[str, Any]]:
    """An arm scored under nested global thresholds, with the thresholds that produced it."""
    limit = ctx.config.metrics.max_threshold_candidates
    per_fold, frozen = go.nested_thresholds(score, rows.y, rows.partition, limit)
    decision = go.decide(score, rows.partition, per_fold)
    thresholds = {"folds": {str(k): v for k, v in per_fold.items()}, "frozen": frozen}
    return _assemble(source, rows, (score, decision), ctx, scoring), thresholds


# -- --evaluate ------------------------------------------------------------------------
def _reproduce(ctx: Context, final: dict[str, Any], oof: OutOfFold, out: Path) -> dict[str, Any]:
    """Recompute every fold's scores from its saved weights; they must equal the stored ones."""
    np = _np()
    began = time.perf_counter()
    config, rows = ctx.config, oof.rows
    plan = go_net.plan_for(go.GridPoint(**final["selected_params"]), config, len(ctx.vocab))
    raw = np.zeros(rows.y.shape, dtype=np.float64)
    for fold in lopo_folds():
        _train, held = go.fold_rows(rows, fold)
        path = _fold_weights(out, int(final["selected_index"]), fold.validate)
        with np.load(path) as handle:
            state = {name: handle[name] for name in handle.files}
        model = head_net.build_model(plan.settings.net, config.seed, state)
        shas = [rows.shas[i] for i in held]
        scores, _ = head_net.predict(model, ctx.store, shas, config.schedule.batch_size)
        raw[held] = scores
    difference = float(np.max(np.abs(raw - oof.raw)))
    limit = config.metrics.max_threshold_candidates
    a = go.propagate_scores(oof.raw, ctx.vocab)
    b = go.propagate_scores(raw, ctx.vocab)
    t_a, _ = go.nested_thresholds(a, rows.y, rows.partition, limit)
    t_b, _ = go.nested_thresholds(b, rows.y, rows.partition, limit)
    identical = bool((go.decide(a, rows.partition, t_a) == go.decide(b, rows.partition, t_b)).all())
    tolerance = config.reproduction_tolerance
    if not difference <= tolerance or not identical:
        raise BaselineError(
            f"the raw scores do not reproduce from the saved weights "
            f"(max difference {difference:.3e}, decisions identical {identical})"
        )
    return {
        "max_abs_difference": difference,
        "decisions_identical": identical,
        "tolerance": tolerance,
        "seconds": time.perf_counter() - began,
    }


def _cell(entry: dict[str, Any]) -> str:
    low, high = entry["ci_low"], entry["ci_high"]
    interval = "" if low is None else f" [{low:.3f}, {high:.3f}]"
    return f"{entry['value']:.3f}{interval}"


def _headline(name: str, arm: dict[str, Any], ks: Sequence[int]) -> str:
    metrics = arm["pooled"]["metrics"]
    parts = [f"p@{k} {_cell(metrics[f'precision_at_{k}'])}" for k in ks]
    parts += [f"r@{k} {_cell(metrics[f'recall_at_{k}'])}" for k in ks]
    parts += [f"Fmax {_cell(metrics['fmax'])}", f"coverage {_cell(metrics['coverage'])}"]
    return f"{name}: " + ", ".join(parts)


def _verdict(model: dict[str, Any], floor: dict[str, Any], keys: Sequence[str]) -> str:
    """Whether the model beats a floor, metric by metric, judged by the intervals."""
    parts = []
    for key in keys:
        a, b = model["pooled"]["metrics"][key], floor["pooled"]["metrics"][key]
        if a["ci_low"] is None or b["ci_low"] is None:
            parts.append(f"{key}: no intervals")
        elif a["ci_low"] > b["ci_high"]:
            parts.append(f"{key}: model above, intervals disjoint")
        elif a["value"] > b["value"]:
            parts.append(f"{key}: model higher, intervals overlap (not shown to differ)")
        else:
            parts.append(f"{key}: model not above")
    return "; ".join(parts)


def _verdict_keys(config: go.GoConfig) -> list[str]:
    return [f"precision_at_{config.metrics.primary_k}", "fmax"]


def _logreg(ctx: Context, oof: OutOfFold, seed: int) -> tuple[Any, dict[str, Any]]:
    """The Phase 8 logistic regression on multi-label targets; C chosen by the shared rule."""
    rows, folds = oof.rows, lopo_folds()
    x = go.feature_matrix(ctx.sub.table, rows)
    runtime = bm.Runtime(seed, 1)
    candidates = [float(str(p["C"])) for p in baselines.GRIDS["logreg"]]
    scored, values = [], []
    for c in candidates:
        raw = go.logreg_scores(x, rows.y, rows.partition, (folds, c, runtime))
        prop = go.propagate_scores(raw, ctx.vocab)
        scored.append(prop)
        values.append(go.fold_means(prop, rows.y, rows.partition, ctx.config.metrics))
    best = go.select_point(values)
    selection = {
        "grid": [
            {"C": c, "precision_at_primary_k": v[0], "mean_fold_fmax": v[1]}
            for c, v in zip(candidates, values, strict=True)
        ],
        "selected_c": candidates[best],
        "rule": SELECTION_RULE,
    }
    return scored[best], selection


def run_validation(args: argparse.Namespace) -> int:
    """The model and the three floors scored out of fold under one protocol."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    config = ctx.config
    out = _out_dir(args)
    final = _final(ctx, out)
    oof = _out_of_fold(ctx, args, out)
    reproduction = _reproduce(ctx, final, oof, out)
    _say(
        f"reproduction: max |difference| {reproduction['max_abs_difference']:.2e}, "
        "decisions identical"
    )
    scoring = _scoring(ctx, args)
    rows = oof.rows
    prior = go.fold_prior_scores(rows.y, rows.partition, lopo_folds())
    hits = go.motif_matrix(rows, ctx.labels, ctx.vocab)
    logreg_score, logreg_selection = _logreg(ctx, oof, config.seed)
    scores = {
        "model_raw": go.propagate_scores(oof.raw, ctx.vocab),
        "frequency_prior": go.propagate_scores(prior, ctx.vocab),
        "logreg": logreg_score,
        "motif": go.propagate_scores(
            go.motif_scores(hits, prior, config.motif_tiebreak_weight), ctx.vocab
        ),
    }
    arms: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for name, score in scores.items():
        arms[name], thresholds[name] = _nested_arm(FLOOR_SOURCES[name], rows, score, ctx, scoring)
    ks = config.metrics.k_values
    for name, arm in arms.items():
        _say(_headline(name, arm, ks))
    for floor in ("frequency_prior", "logreg", "motif"):
        _say(f"model vs {floor}: {_verdict(arms['model_raw'], arms[floor], _verdict_keys(config))}")
    _say("intervals are wide: few clusters, one subset; no published GO figure is comparable")
    stage = {
        "seconds": time.perf_counter() - began,
        "samples": scoring.boot.samples,
        "seed": scoring.boot.seed,
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
                "thresholds": thresholds,
                "logreg_selection": logreg_selection,
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


def _ece_verdict(report: dict[str, Any]) -> str:
    """Whether calibration lowered the macro ECE, judged by the cluster-bootstrap interval."""
    delta = report["macro"]["ece_delta"]
    if delta["ci_low"] is None:
        return "no interval"
    if delta["ci_high"] < 0.0:
        return "calibration lowers ECE (interval below 0)"
    if delta["ci_low"] > 0.0:
        return "calibration RAISES ECE (interval above 0)"
    return "ECE change not shown to differ from 0 (interval spans 0)"


def run_calibrate(args: argparse.Namespace) -> int:
    """Cross-fit the per-term calibrators and the global threshold; the test is not read."""
    began = time.perf_counter()
    _announce()
    ctx = _load(args)
    config = ctx.config
    out = _out_dir(args)
    final = _final(ctx, out)
    fingerprint = _calibration_fingerprint(ctx, final, out)
    oof = _out_of_fold(ctx, args, out)
    rows, folds = oof.rows, lopo_folds()
    calibration = _calibration_config(config)
    data = go.CalibData(oof.raw, rows.y, rows.partition, rows.cluster_code, ctx.vocab)
    fitted = go.crossfit(data, folds, calibration, config.metrics.max_threshold_candidates)
    scoring = _scoring(ctx, args)
    decision = go.decide(fitted.propagated, rows.partition, fitted.fold_thresholds)
    arm = _assemble(go.SOURCE_MODEL, rows, (fitted.propagated, decision), ctx, scoring)
    report = go.ece_report(data, fitted, calibration, scoring.boot)
    folder = out / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    bm.save_array(folder / CROSSFIT_FILE, fitted.calibrated)
    stage = {
        "seconds": time.perf_counter() - began,
        "samples": scoring.boot.samples,
        "seed": scoring.boot.seed,
        "hardware": bm.hardware(),
        "versions": _versions(),
    }
    write_json(
        folder / FIT_FILE,
        {
            "fingerprint": fingerprint,
            "labels": [entry.to_dict() for entry in fitted.frozen],
            "fold_thresholds": {str(k): v for k, v in fitted.fold_thresholds.items()},
            "frozen_threshold": fitted.frozen_threshold,
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
            "validation": {
                "model_calibrated": arm,
                "calibration": report,
                "thresholds": {
                    "folds": {str(k): v for k, v in fitted.fold_thresholds.items()},
                    "frozen": fitted.frozen_threshold,
                },
            },
        },
    )
    macro = report["macro"]
    calibrated = sum(1 for m in fitted.methods["frozen"] if m != "uncalibrated")
    _say(
        f"{calibrated}/{len(ctx.vocab)} terms calibrated "
        f"({sorted(set(fitted.methods['frozen']))}); "
        f"ECE {macro['ece_raw']['value']:.4f} -> {macro['ece_calibrated']['value']:.4f}; "
        f"Brier {macro['brier_raw']['value']:.4f} -> {macro['brier_calibrated']['value']:.4f}"
    )
    _say(f"verdict: {_ece_verdict(report)}")
    _say(_headline("model_calibrated", arm, config.metrics.k_values))
    _say(f"calibrate stage: {stage['seconds']:.1f} s")
    return 0


# -- the manifest ----------------------------------------------------------------------
def _vocabulary_source(config: go.GoConfig) -> Path | None:
    """The committed terms file when it exists, else the one --labels wrote."""
    for path in (
        REPO_ROOT / config.bundle_directory / config.bundle_version / config.vocabulary_file,
        GO_DIR / VOCABULARY_JSON,
    ):
        if path.exists():
            return path
    return None


def _pins() -> go.Pins:
    registry_pin, corpus_pin = _pinned_hashes()
    embedding = embedding_manifest(EMBEDDING_MANIFEST)
    residue = read_json_object(RESIDUE_MANIFEST) if RESIDUE_MANIFEST.exists() else {}
    config = go.load_config(GO_CONFIG)
    source = _vocabulary_source(config)
    ids: tuple[str, ...] = ()
    sha = ""
    if source is not None:
        raw = source.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        ids = tuple(str(t["id"]) for t in json.loads(raw)["terms"])
    return go.Pins(
        registry_sha256=registry_pin,
        corpus_sha256=corpus_pin,
        ontology_sha256=config.ontology.sha256,
        embedding_cache_key=str(embedding["cache_key"]),
        residue_cache_key=str(residue.get("cache_key", "")),
        config_sha256=config.sha256,
        labels_sha256=config.labels_sha256,
        calibration_config_sha256=_calibration_config(config).sha256,
        vocabulary_sha256=sha,
        vocabulary_ids=ids,
        term_floor=config.term_floor,
        closed_world=config.closed_world,
        k_values=config.metrics.k_values,
        reproduction_tolerance=config.reproduction_tolerance,
        torch=head.pinned_version(PYPROJECT.read_text(encoding="utf-8"), "torch"),
    )


def _rules() -> dict[str, Path]:
    return {"baseline_class_map": CLASS_MAP, "consistency": CONSISTENCY}


def _protocol(config: go.GoConfig, boot: dict[str, Any]) -> dict[str, Any]:
    return {
        "adr": "ADR-0014 (protocol of ADR-0010, ADR-0011 and ADR-0013)",
        "validation_partitions": list(VALIDATION_PARTITIONS),
        "test_partition": TEST_PARTITION,
        "folds": [{"train": list(f.train), "validate": f.validate} for f in lopo_folds()],
        "threshold_rule": THRESHOLD_RULE,
        "selection_rule": SELECTION_RULE,
        "fmax_rule": FMAX_RULE,
        "k_values": list(config.metrics.k_values),
        "primary_k": config.metrics.primary_k,
        "early_stopping": (
            "negative mean binary cross-entropy on an inner whole-cluster holdout of the "
            f"training partitions; patience {config.schedule.patience}, at most "
            f"{config.schedule.max_epochs} epochs"
        ),
        "bootstrap": {
            "samples": boot["samples"],
            "unit": "cluster_id",
            "seed": boot["seed"],
        },
        "label_states": "annotated or unknown (ADR-0006); never negative; " + config.closed_world,
    }


def _inputs(ctx: Context, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "registry_sha256": ctx.sub.meta["registry_sha256"],
        "corpus_sha256": ctx.sub.meta["corpus_sha256"],
        "ontology_sha256": ctx.config.ontology.sha256,
        "embedding_cache_key": ctx.sub.meta["subset"]["embedding_cache_key"],
        "residue_cache_key": ctx.residue_key,
        "feature_table_sha256": ctx.sub.meta["table_sha256"],
        "label_table_sha256": ctx.label_info["label_table_sha256"],
        "vocabulary_sha256": ctx.label_info["vocabulary_sha256"],
        "go_config_sha256": ctx.config.sha256,
        "label_policy_sha256": ctx.config.labels_sha256,
        "calibration_config_sha256": _calibration_config(ctx.config).sha256,
        **hashes,
    }


def _head_block(
    ctx: Context, record_sha: str, weights_sha: str, parts: tuple[str, float]
) -> dict[str, Any]:
    config = ctx.config
    vocabulary_sha, threshold = parts
    return {
        "weights_file": WEIGHTS,
        "weights_sha256": weights_sha,
        "frozen_sha256": record_sha,
        "head_config_sha256": config.sha256,
        "classes": list(ctx.vocab.ids),
        "vocabulary_file": config.vocabulary_file,
        "vocabulary_sha256": vocabulary_sha,
        "ontology_sha256": config.ontology.sha256,
        "ontology_data_version": config.ontology.data_version,
        "evidence_policy": list(config.evidence.policy),
        "term_floor": config.term_floor,
        "propagation": list(config.ontology.relations),
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        "source": go.SOURCE_MODEL,
    }


def _calibrator_set(fit: dict[str, Any], hashes: tuple[str, str], ctx: Context) -> CalibratorSet:
    record_sha, weights_sha = hashes
    return CalibratorSet(
        task=go.TASK,
        calibration_version=f"{ctx.config.bundle_version}+{fit['fingerprint']['oof_sha256'][:12]}",
        classes=ctx.vocab.ids,
        head_weights_sha256=weights_sha,
        head_frozen_sha256=record_sha,
        threshold_rule=THRESHOLD_RULE,
        labels=tuple(label_from_dict(entry) for entry in fit["labels"]),
    )


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
    directory = REPO_ROOT / config.bundle_directory / config.bundle_version
    if directory.exists():
        _say(f"{directory} exists; a frozen bundle is never rewritten")
        return 1
    final = _final(ctx, GO_DIR)
    stage = read_manifest(GO_DIR / "train" / "stage.json")
    valid = read_manifest(GO_DIR / VALIDATION_FILE)
    fit = read_manifest(GO_DIR / "calibration" / FIT_FILE)
    cal_valid = read_manifest(GO_DIR / "calibration" / VALIDATION_FILE)
    fingerprint = _calibration_fingerprint(ctx, final, GO_DIR)
    if (
        valid.get("fingerprint") != _fingerprint(ctx)
        or fit.get("fingerprint") != fingerprint
        or cal_valid.get("fingerprint") != fingerprint
    ):
        _say("a stage file was produced from other inputs; rerun --train, --evaluate, --calibrate")
        return 1
    vocabulary_bytes = (GO_DIR / VOCABULARY_JSON).read_bytes()
    vocabulary_sha = hashlib.sha256(vocabulary_bytes).hexdigest()
    record_sha, weights_sha = digest(final), sha256_file(GO_DIR / WEIGHTS)[0]
    calibrator = _calibrator_set(fit, (record_sha, weights_sha), ctx)
    threshold = float(fit["frozen_threshold"])
    head_block = _head_block(ctx, record_sha, weights_sha, (vocabulary_sha, threshold))
    rule_hashes = {f"{key}_sha256": sha256_file(path)[0] for key, path in _rules().items()}
    document = registry_bundle.build_document(
        version=config.bundle_version,
        head=head_block,
        calibrator=calibrator,
        rules=rule_hashes,
        scope={"scope": SUBSET_SCOPE, "scope_label": SUBSET_LABEL},
    )
    thresholds = {
        "model_raw": float(valid["validation"]["thresholds"]["model_raw"]["frozen"]),
        "model_calibrated": threshold,
        "frequency_prior": float(valid["validation"]["thresholds"]["frequency_prior"]["frozen"]),
        "logreg": float(valid["validation"]["thresholds"]["logreg"]["frozen"]),
        "motif": float(valid["validation"]["thresholds"]["motif"]["frozen"]),
    }
    validation = {
        **valid["validation"],
        "model_calibrated": cal_valid["validation"]["model_calibrated"],
        "calibration": cal_valid["validation"]["calibration"],
    }
    info = ctx.label_info
    frozen: dict[str, Any] = {
        "protocol": _protocol(config, valid["stage"]),
        "scope": SUBSET_SCOPE,
        "scope_label": SUBSET_LABEL,
        "subset": ctx.sub.meta["subset"],
        "inputs": _inputs(
            ctx,
            {
                "weights_sha256": weights_sha,
                "training_record_sha256": record_sha,
                "calibrator_sha256": document["calibrator"]["sha256"],
                "oof_sha256": fingerprint["oof_sha256"],
            },
        ),
        "ontology": {
            "data_version": config.ontology.data_version,
            "sha256": config.ontology.sha256,
            "license": config.ontology.license,
            "mapping_whole_corpus": info["mapping_whole_corpus"],
            "mapping_subset": info["mapping_subset"],
            "cross_aspect_edges": info["cross_aspect_edges"],
        },
        "label_policy": go.label_policy_block(config),
        "label_summary": info["summary"],
        "vocabulary": {**go.vocabulary_block(ctx.vocab), "table": info["terms"]},
        "architecture": {
            "pooling": "attention",
            "embedding_dim": config.architecture.embedding_dim,
            "heads": config.architecture.heads,
            "attention_hidden": config.architecture.attention_hidden,
            "optimiser": "adamw",
            "weight_decay": config.weight_decay,
            "batch_size": config.schedule.batch_size,
            "pos_weight_cap": config.pos_weight_cap,
            "seed": config.seed,
        },
        "selected": {
            "index": final["selected_index"],
            "params": final["selected_params"],
            "final_epochs": final["final_epochs"],
            "trainable_parameters": final["trainable_parameters"],
            "selector": final["selector"],
            "mean_fold_fmax": final["mean_fold_fmax"],
        },
        "grid": final["grid"],
        "thresholds": thresholds,
        "calibration_methods": fit["methods"],
        "validation": validation,
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
        "schema_version": go.SCHEMA_VERSION,
        "phase": go.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }
    problems = go.check_go_manifest(manifest, _pins()) + calibrator.problems()
    if problems:
        for problem in problems:
            _say(f"  {problem}")
        return 1
    directory.mkdir(parents=True)
    (directory / config.vocabulary_file).write_bytes(vocabulary_bytes)
    (directory / registry_bundle.CALIBRATOR_FILE).write_bytes(calibrator.to_bytes())
    write_json(directory / registry_bundle.BUNDLE_FILE, document)
    registry_bundle.load_bundle(
        directory, _rules(), GO_DIR / WEIGHTS, ontology=config.ontology.sha256
    )
    write_json(MANIFEST, manifest)
    _say(f"froze {directory} and {MANIFEST}, frozen_sha256 {manifest['frozen_sha256'][:12]}")
    return 0


# -- --evaluate --test -----------------------------------------------------------------
def _test_state() -> dict[str, Any]:
    np = _np()
    with np.load(GO_DIR / WEIGHTS, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def _touches(path: Path) -> int | None:
    """How many times an earlier phase touched partition 5; None if it was not scored."""
    test = read_manifest(path).get("test")
    return int(test["touches"]) if isinstance(test, dict) else None


def _earlier_touches() -> dict[str, int | None]:
    return {
        "localization": _touches(HEAD_MANIFEST) if HEAD_MANIFEST.exists() else None,
        "crf": _touches(CRF_MANIFEST) if CRF_MANIFEST.exists() else None,
    }


def run_test(args: argparse.Namespace) -> int:
    """Score partition 5 once, with the weights, calibrators and thresholds the manifest froze."""
    if args.smoke:
        _say("--smoke never touches the test partition")
        return 1
    began = time.perf_counter()
    manifest = read_manifest(MANIFEST)
    problems = go.check_go_manifest(manifest, _pins())
    if problems or manifest["test"] is not None:
        _say("refusing: the manifest has problems or the test partition was already scored")
        for problem in problems:
            _say(f"  {problem}")
        return 1
    earlier = _earlier_touches()
    if any(v != 1 for v in earlier.values()):
        _say(f"refusing: an earlier head's test touch count is not 1: {earlier}")
        return 1
    _announce()
    ctx = _load(args)
    frozen = manifest["frozen"]
    if sha256_file(GO_DIR / WEIGHTS)[0] != frozen["inputs"]["weights_sha256"]:
        _say("refusing: the weights are not the ones the manifest froze")
        return 1
    config, np = ctx.config, _np()
    loaded = registry_bundle.load_bundle(
        REPO_ROOT / frozen["bundle"]["directory"],
        _rules(),
        GO_DIR / WEIGHTS,
        ontology=config.ontology.sha256,
    )
    plan = go_net.plan_for(go.GridPoint(**frozen["selected"]["params"]), config, len(ctx.vocab))
    model = head_net.build_model(plan.settings.net, config.seed, _test_state())
    rows = _rows(ctx, (TEST_PARTITION,), 1)
    train = _rows(ctx, VALIDATION_PARTITIONS, 1)
    head.assert_no_overlap(
        {
            "train": go_net.clusters_of(train, range(len(train))),
            "test": go_net.clusters_of(rows, range(len(rows))),
        }
    )
    raw, _ = head_net.predict(model, ctx.store, rows.shas, config.schedule.batch_size)
    calibrated = go.apply_calibrators(loaded.calibrators.labels, raw)
    prior = np.tile(train.y.mean(axis=0), (len(rows), 1))
    hits = go.motif_matrix(rows, ctx.labels, ctx.vocab)
    runtime = bm.Runtime(config.seed, 1)
    chosen_c = float(frozen["validation"]["logreg_selection"]["selected_c"])
    fitted = go.logreg_fit(go.feature_matrix(ctx.sub.table, train), train.y, chosen_c, runtime)
    scores = {
        "model_raw": raw,
        "model_calibrated": calibrated,
        "frequency_prior": prior,
        "logreg": go.logreg_apply(fitted, go.feature_matrix(ctx.sub.table, rows)),
        "motif": go.motif_scores(hits, prior, config.motif_tiebreak_weight),
    }
    scoring = _scoring(ctx, args)
    metrics: dict[str, Any] = {}
    for name in ARMS:
        propagated = go.propagate_scores(scores[name], ctx.vocab)
        decision = propagated >= float(frozen["thresholds"][name])
        metrics[name] = _assemble(FLOOR_SOURCES[name], rows, (propagated, decision), ctx, scoring)
        _say(_headline(f"test {name}", metrics[name], config.metrics.k_values))
    seconds = time.perf_counter() - began
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "rows": len(rows),
        "evaluable_rows": int(go.evaluable(rows.y).sum()),
        "seconds": seconds,
        "hardware": bm.hardware(),
        "scope": SUBSET_SCOPE,
        "label": SUBSET_LABEL,
        "metrics": metrics,
    }
    write_json(MANIFEST, manifest)
    _say(f"test partition scored once in {seconds:.1f} s ({len(rows)} rows, CI is wide)")
    return 0


# -- --check ---------------------------------------------------------------------------
def _bundle_problems(manifest: dict[str, Any]) -> list[str]:
    """The registry entry must verify and be the one the manifest names."""
    frozen = manifest["frozen"]
    directory = REPO_ROOT / frozen["bundle"]["directory"]
    try:
        loaded = registry_bundle.load_bundle(
            directory, _rules(), ontology=frozen["ontology"]["sha256"]
        )
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
    if head_doc["vocabulary_sha256"] != frozen["inputs"]["vocabulary_sha256"]:
        found.append("the bundle names another vocabulary than the manifest")
    return found


def run_check(args: argparse.Namespace) -> int:
    """No data bytes: configuration, pins and, when present, the manifest and the bundle."""
    problems: list[str] = []
    config = _config(args)
    assert_protocol(lopo_folds())
    pins = _pins()
    if MANIFEST.exists():
        manifest = read_manifest(MANIFEST)
        problems += go.check_go_manifest(manifest, pins)
        if not problems:
            problems += _bundle_problems(manifest)
        state = f"{MANIFEST.name} present"
    else:
        state = f"{MANIFEST.name} absent (configuration and pins only)"
    earlier = _earlier_touches()
    problems += [
        f"{name} test touches is {n}, not 1" for name, n in earlier.items() if n not in (None, 1)
    ]
    for problem in problems:
        _say(f"FAIL {problem}")
    _say(
        f"go --check: {'FAIL' if problems else 'ok'}; torch=={pins.torch}; "
        f"{len(go.grid_points(config))} grid points; floor {config.term_floor}; "
        f"policy {list(config.evidence.policy)}; earlier touches {earlier}; {state}; "
        f"scope: {SUBSET_LABEL}"
    )
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 12 GO head.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("labels", "train", "evaluate", "calibrate", "freeze", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--test", action="store_true", help="with --evaluate: score the test once")
    parser.add_argument("--smoke", action="store_true", help="every 4th row, never frozen")
    parser.add_argument("--force", action="store_true", help="with --train: refit every grid point")
    return parser


def _mode(args: argparse.Namespace) -> int:
    """Run the one mode the arguments name."""
    if args.labels:
        return run_labels(args)
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
