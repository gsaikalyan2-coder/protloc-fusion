"""Measure what the pooling attention actually is (Phase 15, Step 2). Read-only.

For every sequence in the 500-sequence residue cache this records, for the localization head
and the GO head: the number of attention heads, the shape, the largest deviation of a head's
weights from summing to 1, the entropy and effective number of residues attended
(exp(entropy)), the largest weight, and the mean weight by relative position (decile) scaled
by length so that a uniform head reads 1.0. It also proves, on every sequence, that the
attention is shared by all classes: a copy of the head with one class's output row replaced
yields bit-identical attention.

Writes ``data/manifests/attribution_facts.json``. Nothing is trained, nothing is scored against
labels, partition 5 is not read. Every number is "subset (500), not comparable to DeepLoc or to
the Phase 8 full-corpus numbers".

Usage (repo root)::

    python scripts/attribution_facts.py --write
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from protloc.data.residue_cache import ResidueStore  # noqa: E402
from protloc.models.attention_head import AttentionHeadNet  # noqa: E402

CACHE = REPO / "data" / "embeddings" / "0e528d4a46831b05_subset500_residues"
WEIGHTS = {
    "localization": REPO / "data" / "head" / "final_weights.npz",
    "go_function": REPO / "data" / "go" / "final_weights.npz",
}
OUT = REPO / "data" / "manifests" / "attribution_facts.json"
EPS = 1e-5
DECILES = 10
LENGTH_BINS = ((1, 150), (151, 300), (301, 500), (501, 800), (801, 10_000))
SCOPE = "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"


def _load(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _summ(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _bin(length: int) -> str:
    for low, high in LENGTH_BINS:
        if low <= length <= high:
            return f"{low}-{high}"
    return "other"


def _entropy(w: np.ndarray) -> float:
    w = np.clip(w, 1e-30, 1.0)
    return float(-(w * np.log(w)).sum())


def measure(task: str, tensors: dict[str, Any], store: ResidueStore) -> dict[str, Any]:
    """Compute attention statistics for one task over every sequence in the residue store."""
    net = AttentionHeadNet(tensors, EPS)
    # A copy whose class-0 output row is replaced: if the attention were class-specific it
    # would change; it is computed before fc2, so it cannot.
    altered = dict(tensors)
    altered["fc2.weight"] = np.array(tensors["fc2.weight"], copy=True)
    altered["fc2.bias"] = np.array(tensors["fc2.bias"], copy=True)
    altered["fc2.weight"][0] = altered["fc2.weight"][0][::-1] * 3.0
    altered["fc2.bias"][0] = altered["fc2.bias"][0] + 5.0
    other = AttentionHeadNet(altered, EPS)

    shas = sorted(store.lengths)
    sum_err, ent, eff, wmax, eff_frac = [], [], [], [], []
    by_bin: dict[str, list[float]] = defaultdict(list)
    decile_ratio: list[list[float]] = [[] for _ in range(DECILES)]
    identical = 0
    score_changed = 0
    for sha in shas:
        states = store.states(sha)
        out = net.forward(states)
        att = np.asarray(out.attention)  # (heads, length)
        length = att.shape[1]
        sum_err.append(float(np.abs(att.sum(axis=1) - 1.0).max()))
        mean_att = att.mean(axis=0)
        h = _entropy(mean_att)
        ent.append(h)
        eff.append(math.exp(h))
        eff_frac.append(math.exp(h) / length)
        wmax.append(float(mean_att.max()))
        by_bin[_bin(length)].append(math.exp(h))
        position = np.minimum((np.arange(length) * DECILES) // length, DECILES - 1)
        for d in range(DECILES):
            members = position == d
            if members.any():  # short sequences leave deciles empty: skip them
                decile_ratio[d].append(float(mean_att[members].sum()) / (members.sum() / length))
        o2 = other.forward(states)
        identical += int(np.array_equal(att, np.asarray(o2.attention)))
        score_changed += int(not np.array_equal(out.scores, o2.scores))
    return {
        "task": task,
        "sequences": len(shas),
        "attention_heads": net.heads,
        "classes": net.classes,
        "shape": "(heads, length), softmax over residues",
        "max_abs_deviation_from_sum_1": float(max(sum_err)),
        "entropy_nats": _summ(ent),
        "effective_residues_attended": _summ(eff),
        "effective_fraction_of_length": _summ(eff_frac),
        "max_weight": _summ(wmax),
        "effective_residues_by_length_bin": {
            k: {"n": len(v), **_summ(v)} for k, v in sorted(by_bin.items())
        },
        "position_bias": {
            "note": (
                "attention mass in each relative-position decile divided by the mass a uniform "
                "head would put there, averaged over sequences; 1.0 = uniform"
            ),
            "deciles": [round(float(np.mean(r)), 4) for r in decile_ratio],
            "sequences_per_decile": [len(r) for r in decile_ratio],
        },
        "shared_across_classes": {
            "method": "replace class 0's fc2 row and bias, recompute",
            "attention_bit_identical": identical,
            "scores_changed": score_changed,
            "of": len(shas),
        },
    }


def main() -> int:
    """Entry point: measure attention and optionally write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the manifest")
    args = parser.parse_args()
    store = ResidueStore(CACHE)
    tasks = {name: measure(name, _load(path), store) for name, path in WEIGHTS.items()}
    doc: dict[str, Any] = {
        "scope": SCOPE,
        "partition_5_touched": 0,
        "labels_read": 0,
        "weights_sha256": {
            n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in WEIGHTS.items()
        },
        "tasks": tasks,
    }
    text = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False)
    print(text)  # noqa: T201
    if args.write:
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT}", file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
