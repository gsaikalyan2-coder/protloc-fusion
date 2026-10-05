"""Faithfulness of the attribution methods on the 500-sequence subset (Phase 15, Step 4).

Two steps, in this order, so the protocol is frozen before any result exists::

    python scripts/attribution_eval.py --freeze-protocol   # writes attribution_protocol.json
    python scripts/attribution_eval.py --run --write       # refuses if the protocol changed

Rows: the localization-labelled accessions of registry partitions 1-4, one row per distinct
sequence. Partition 5 is never read (``partition_5_touched: 0``). The heads saw these sequences
in training or validation, so every number is a check of faithfulness to the head, not accuracy.

Deletion test (localization head and GO head, each on its own top class): remove the top-k% of
residues by a method's score from the pooled input, re-score, record the drop in the class's raw
sigmoid. Baselines: random residues of the same count (fixed seeds), the bottom-k% by the same
method, and (localization only) the old motif attribution. Insertion test: keep only the top-k%.
A method passes only if the cluster-bootstrap interval of (its drop - random drop) is above 0 at
every k. Signal peptide: AUROC of the CRF posterior against the annotated cleavage-site mask
(residues 1..site), against the motif hydropathy and a position-only prior; the CRF was trained
on these labels, so this checks coherence with the training signal, not generalisation.

Scope of every number: subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus
numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from protloc.data.embeddings import read_pack  # noqa: E402
from protloc.features.classical import (  # noqa: E402
    extract,
    normalize,
    residue_attribution,
)
from protloc.inference.attribution import (  # noqa: E402
    OcclusionSettings,
    attention_track,
    occlusion_track,
    score_without,
)
from protloc.models.attention_head import AttentionHeadNet  # noqa: E402
from protloc.models.crf_tagger import CrfTaggerNet, TagSpec  # noqa: E402
from protloc.training import crf as crf_train  # noqa: E402
from protloc.training import head as head_train  # noqa: E402
from protloc.training.baselines import load_class_map  # noqa: E402

SCOPE = "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"
PROTOCOL_FILE = REPO / "data" / "manifests" / "attribution_protocol.json"
METRICS_FILE = REPO / "data" / "manifests" / "attribution_metrics.json"
ENGINE = json.loads((REPO / "configs" / "serving" / "engine.json").read_text(encoding="utf-8"))
WEIGHTS = {
    "localization": REPO / "data" / "head" / "final_weights.npz",
    "go_function": REPO / "data" / "go" / "final_weights.npz",
    "signal_peptide": REPO / "data" / "crf" / "final_weights.npz",
}
EPS = float(ENGINE["network"]["layer_norm_eps"])
PARTITIONS = (1, 2, 3, 4)
LOW, HIGH = 2.5, 97.5

PROTOCOL: dict[str, Any] = {
    "scope": SCOPE,
    "rows": (
        "localization-labelled accessions of registry partitions 1-4 (stride 1), one row per "
        "distinct sequence; cluster = the registry cluster_code"
    ),
    "partitions": list(PARTITIONS),
    "partition_5_touched": 0,
    "target": "the head's own top class: argmax of the raw sigmoid over all of its classes",
    "k_grid_percent": [5, 10, 20, 40],
    "residues_removed": "ceil(k/100 * length), never all of them",
    "random_baseline": {"draws": 20, "seed_base": 20261005, "per": "sequence and draw"},
    "occlusion": {"window": 5, "stride": 1, "max_forward_passes": 64},
    "methods": {
        "localization": ["occlusion", "pooling_attention", "motif"],
        "go_function": ["occlusion", "pooling_attention"],
    },
    "baselines": ["random", "bottom_occlusion", "bottom_pooling_attention"],
    "bootstrap": {
        "samples": 1000,
        "seed": 20261005,
        "unit": "cluster_code",
        "interval": [2.5, 97.5],
    },
    "pass_rule": (
        "for every k in the grid the 2.5th percentile of the cluster bootstrap of "
        "(deletion drop of the method - deletion drop of random) is above 0"
    ),
    "failing_method": "dropped from the served path (Gate D); recorded here as failed",
    "signal_peptide": {
        "rows": "annotated signal-peptide accessions of partitions 1-4, usable cleavage site",
        "mask": "residues 1..cleavage within the CRF length cap",
        "scores": ["crf_posterior", "motif_hydropathy", "position_prior"],
        "metric": "pooled-residue AUROC with the same cluster bootstrap",
    },
    "illustrative_motif_check": "not run",
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)  # noqa: T201


def _script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"_protloc_{name}", REPO / "scripts" / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# -- statistics -------------------------------------------------------------------------
class Bootstrap:
    """Whole-cluster bootstrap of per-row metrics, one weight matrix reused for every metric."""

    def __init__(self, cluster: Any, samples: int, seed: int) -> None:  # noqa: D107
        self.codes, self.inverse = np.unique(np.asarray(cluster), return_inverse=True)
        self.counts = np.bincount(self.inverse).astype(np.float64)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.codes), size=(samples, len(self.codes)))
        self.weights = np.stack([np.bincount(d, minlength=len(self.codes)) for d in draws]).astype(
            np.float64
        )

    def interval(self, per_row: Any) -> dict[str, float]:  # noqa: D102
        values = np.asarray(per_row, dtype=np.float64)
        sums = np.bincount(self.inverse, weights=values)
        estimate = (self.weights @ sums) / (self.weights @ self.counts)
        return {
            "mean": float(values.mean()),
            "low": float(np.percentile(estimate, LOW)),
            "high": float(np.percentile(estimate, HIGH)),
        }


def auroc(score: Any, label: Any) -> float:  # noqa: D103
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    ranks = (cumulative - (counts - 1) / 2.0)[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# -- deletion / insertion ----------------------------------------------------------------
def _count(k: int, length: int) -> int:
    return min(max(1, math.ceil(k * length / 100.0)), length - 1)


def _tracks(
    head: AttentionHeadNet, states: Any, cls: int, settings: OcclusionSettings
) -> dict[str, Any]:
    return {
        "occlusion": occlusion_track(head, states, cls, settings).values,
        "pooling_attention": attention_track(head.forward(states)).values,
    }


def _curve(  # noqa: PLR0913, PLR0917
    head: AttentionHeadNet,
    states: Any,
    cls: int,
    full: float,
    order: Any,
    ks: list[int],
) -> tuple[list[float], list[float]]:
    length = int(states.shape[0])
    drops, kept_scores = [], []
    for k in ks:
        n = _count(k, length)
        keep = np.ones(length, dtype=bool)
        keep[order[:n]] = False
        drops.append(full - score_without(head, states, cls, keep))
        only = np.zeros(length, dtype=bool)
        only[order[:n]] = True
        kept_scores.append(score_without(head, states, cls, only))
    return drops, kept_scores


def evaluate_head(  # noqa: D103, PLR0913, PLR0917
    name: str,
    head: AttentionHeadNet,
    shas: list[str],
    store: Any,
    clusters: Any,
    motif_of: Any,
    boot: Bootstrap,
) -> dict[str, Any]:
    ks = list(PROTOCOL["k_grid_percent"])
    settings = OcclusionSettings(**PROTOCOL["occlusion"])
    draws = int(PROTOCOL["random_baseline"]["draws"])
    seed_base = int(PROTOCOL["random_baseline"]["seed_base"])
    methods = list(PROTOCOL["methods"][name])
    rows: dict[str, dict[str, list[list[float]]]] = {}  # label -> {"del": [seq][k], "ins": ...}

    def put(label: str, drops: list[float], kept: list[float]) -> None:
        rows.setdefault(label, {"del": [], "ins": []})
        rows[label]["del"].append(drops)
        rows[label]["ins"].append(kept)

    began = time.perf_counter()
    for i, sha in enumerate(shas):
        states = np.asarray(store.states(sha), dtype=np.float32)
        out = head.forward(states)
        cls = int(np.argmax(out.scores))
        full = float(out.scores[cls])
        tracks = _tracks(head, states, cls, settings)
        if "motif" in methods:
            tracks["motif"] = np.asarray(motif_of(sha, cls), dtype=np.float64)
        for label in methods:
            score = tracks[label]
            put(
                label,
                *_curve(head, states, cls, full, np.argsort(-score, kind="stable"), ks),
            )
            if label != "motif":
                put(
                    f"bottom_{label}",
                    *_curve(head, states, cls, full, np.argsort(score, kind="stable"), ks),
                )
        random_d = np.zeros(len(ks))
        random_i = np.zeros(len(ks))
        for d in range(draws):
            rng = np.random.default_rng(seed_base + 1000 * d + i)
            drops, kept = _curve(head, states, cls, full, rng.permutation(states.shape[0]), ks)
            random_d += drops
            random_i += kept
        put("random", list(random_d / draws), list(random_i / draws))
        if (i + 1) % 50 == 0:
            _say(f"  {name}: {i + 1}/{len(shas)} sequences, {time.perf_counter() - began:.0f} s")

    grid = np.array([0.0, *[k / 100.0 for k in ks]])
    result: dict[str, Any] = {
        "sequences": len(shas),
        "k_grid_percent": ks,
        "methods": {},
    }
    rand_del = np.array(rows["random"]["del"])
    for label, block in rows.items():
        dele, ins = np.array(block["del"]), np.array(block["ins"])
        audc = np.trapezoid(np.hstack([np.zeros((len(dele), 1)), dele]), grid, axis=1)
        entry: dict[str, Any] = {
            "deletion_drop": {str(k): boot.interval(dele[:, j]) for j, k in enumerate(ks)},
            "insertion_kept_score": {str(k): boot.interval(ins[:, j]) for j, k in enumerate(ks)},
            "deletion_auc": boot.interval(audc),
        }
        if label != "random":
            diff = {str(k): boot.interval(dele[:, j] - rand_del[:, j]) for j, k in enumerate(ks)}
            entry["drop_minus_random"] = diff
            if label in methods:
                entry["passes"] = all(v["low"] > 0.0 for v in diff.values())
        result["methods"][label] = entry
    return result


# -- signal peptide ---------------------------------------------------------------------
def evaluate_signal_peptide(  # noqa: D103
    crf_rows: Any, store: Any, sequence_of: Any, cap: int
) -> dict[str, Any]:
    spec = ENGINE["crf"]
    net = CrfTaggerNet(
        _load(WEIGHTS["signal_peptide"]),
        TagSpec(
            tuple(spec["tags"]),
            spec["outside"],
            spec["signal"],
            tuple(tuple(p) for p in spec["forbidden_transitions"]),
            float(spec["forbidden_score"]),
        ),
        cap,
        EPS,
    )
    per_row: list[tuple[int, Any, Any, Any, Any]] = []
    for j, sha in enumerate(crf_rows.shas):
        site = int(crf_rows.cleavage[j])
        if int(crf_rows.y[j]) != 1 or site < 1 or not bool(crf_rows.tagged[j]):
            continue
        out = net.forward(store.states(sha))
        n = min(len(out.signal_posterior), net.length_cap)
        hydro = np.asarray(extract(normalize(sequence_of(sha))).hydropathy[:n], dtype=np.float64)
        mask = np.arange(n) < site
        per_row.append(
            (
                int(crf_rows.cluster_code[j]),
                mask,
                np.asarray(out.signal_posterior[:n]),
                hydro,
                -np.arange(n, dtype=np.float64),
            )
        )
    clusters = np.array([r[0] for r in per_row])
    codes, inverse = np.unique(clusters, return_inverse=True)
    boot_cfg = PROTOCOL["bootstrap"]
    rng = np.random.default_rng(int(boot_cfg["seed"]))
    draws = rng.integers(0, len(codes), size=(int(boot_cfg["samples"]), len(codes)))
    members = [np.flatnonzero(inverse == c) for c in range(len(codes))]
    out_doc: dict[str, Any] = {
        "rows": len(per_row),
        "clusters": len(codes),
        "scores": {},
    }
    for column, label in (
        (2, "crf_posterior"),
        (3, "motif_hydropathy"),
        (4, "position_prior"),
    ):

        def pooled(selected: Any, _col: int = column) -> float:
            y = np.concatenate([per_row[i][1] for i in selected])
            s = np.concatenate([per_row[i][_col] for i in selected])
            return auroc(s, y)

        point = pooled(range(len(per_row)))
        boots = []
        for d in draws:
            chosen = np.concatenate([members[c] for c in d])
            boots.append(pooled(chosen))
        boots_a = np.array([b for b in boots if not math.isnan(b)])
        out_doc["scores"][label] = {
            "auroc": point,
            "low": float(np.percentile(boots_a, LOW)),
            "high": float(np.percentile(boots_a, HIGH)),
        }
    return out_doc


# -- driver -----------------------------------------------------------------------------
def freeze_protocol() -> int:  # noqa: D103
    doc = {"protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256}
    if PROTOCOL_FILE.exists():
        old = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != PROTOCOL_SHA256:
            _say("refusing: a different protocol is already frozen; the file is the record")
            return 1
        _say(f"protocol already frozen: {PROTOCOL_SHA256}")
        return 0
    PROTOCOL_FILE.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _say(f"froze the protocol: {PROTOCOL_SHA256}")
    return 0


def run(args: argparse.Namespace) -> int:  # noqa: D103
    if not args.limit:  # a smoke run (--limit) writes nothing, so it needs no frozen protocol
        if not PROTOCOL_FILE.exists():
            _say("refusing: run --freeze-protocol first")
            return 1
        frozen = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        if frozen["protocol_sha256"] != PROTOCOL_SHA256 or frozen["protocol"] != PROTOCOL:
            _say("refusing: the protocol in this script differs from the frozen one")
            return 1
    crf_script = _script("crf")
    ctx = crf_script._load(argparse.Namespace(smoke=False))
    cmap = load_class_map(crf_script.CLASS_MAP)
    class_names = list(cmap.classes["localization"])
    codes = ENGINE["localization"]["codes"]
    items, _ = read_pack(REPO / "data" / "embeddings" / "input_pack.tsv")
    wanted = set(ctx.store.lengths)
    sequences = {it.sha256: it.sequence for it in items if it.sha256 in wanted}

    rows = head_train.build_rows(ctx.table, ctx.order, PARTITIONS, 1)
    seen: dict[str, int] = {}
    for j, sha in enumerate(rows.shas):
        seen.setdefault(sha, int(rows.cluster_code[j]))
    shas = sorted(seen)
    if args.limit:
        shas = shas[:: max(1, len(shas) // args.limit)][: args.limit]
    clusters = np.array([seen[s] for s in shas])
    _say(f"{len(shas)} distinct sequences, {len(set(clusters.tolist()))} clusters; scope: {SCOPE}")

    boot_cfg = PROTOCOL["bootstrap"]
    boot = Bootstrap(clusters, int(boot_cfg["samples"]), int(boot_cfg["seed"]))

    def motif_of(sha: str, cls: int) -> Any:
        scored = normalize(sequences[sha])
        return residue_attribution(scored, extract(scored), codes[class_names[cls]])

    results: dict[str, Any] = {}
    for name in ("localization", "go_function"):
        head = AttentionHeadNet(_load(WEIGHTS[name]), EPS)
        results[name] = evaluate_head(name, head, shas, ctx.store, clusters, motif_of, boot)
    cap = int(ctx.config.architecture.length_cap)
    crf_rows = crf_train.build_rows(
        ctx.table, ctx.order, ctx.labels, crf_train.RowSpec(PARTITIONS, 1, cap)
    )
    results["signal_peptide"] = evaluate_signal_peptide(
        crf_rows, ctx.store, sequences.__getitem__, cap
    )

    body = {
        "scope": SCOPE,
        "heads_saw_these_sequences": True,
        "partition_5_touched": 0,
        "protocol_sha256": PROTOCOL_SHA256,
        "weights_sha256": {
            n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in WEIGHTS.items()
        },
        "results": results,
        "verdicts": {
            name: {
                m: bool(results[name]["methods"][m]["passes"]) for m in PROTOCOL["methods"][name]
            }
            for name in ("localization", "go_function")
        },
    }
    body["frozen_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False)
    if args.write and not args.limit:
        METRICS_FILE.write_text(text + "\n", encoding="utf-8")
        _say(f"wrote {METRICS_FILE}")
    elif args.write:
        _say("--limit is a smoke run: nothing written")
    print(  # noqa: T201
        json.dumps(
            {"verdicts": body["verdicts"], "signal_peptide": results["signal_peptide"]},
            indent=2,
        )
    )
    return 0


def main() -> int:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="smoke: use about this many sequences")
    args = parser.parse_args()
    if args.freeze_protocol:
        return freeze_protocol()
    if args.run:
        return run(args)
    parser.error("choose --freeze-protocol or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
