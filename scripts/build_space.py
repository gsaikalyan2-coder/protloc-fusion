"""Build the protein-space map: the cached ESM-2 vectors projected to three dimensions.

The Predict screen and the Protein space screen draw every cached sequence as a point, placed by
a principal-component projection of its mean-pooled ESM-2 150M vector (the shared representation
every head reads), coloured by its Swiss-Prot location. Nothing here is a model output and nothing
here feeds a score: it shows where a sequence sits among the others, and which cached sequences
are its nearest neighbours in the full 640-dimensional space.

Inputs (all already on disk, nothing is fetched):

* the subset embedding cache (``spec.json``, ``shard_*.tsv``, ``shard_*.f32``);
* ``data/gallery/catalog.json`` for accessions, labels and partitions;
* ``configs/serving/engine.json`` for the label-to-code map (no biology is written here).

Output: ``data/space/space.json`` and ``data/space/manifest.json`` (the file's SHA-256 and every
input's identity). The projection is deterministic: PCA by SVD with each component's sign fixed so
its largest-magnitude loading is positive, coordinates rounded to 4 decimals.

    python scripts/build_space.py           # write
    python scripts/build_space.py --check   # exit 1 if the written file would differ
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "embeddings" / "0e528d4a46831b05_subset500"
CATALOG = REPO_ROOT / "data" / "gallery" / "catalog.json"
ENGINE_CONFIG = REPO_ROOT / "configs" / "serving" / "engine.json"
OUT_DIR = REPO_ROOT / "data" / "space"
SPACE_FILE = "space.json"
MANIFEST_FILE = "manifest.json"
COMPONENTS = 3
NEIGHBOURS = 8
DIGITS = 4
SCHEMA_VERSION = 1
SCOPE = "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"
NOTE = (
    "Principal components of mean-pooled ESM-2 150M vectors of the cached sequences. Position "
    "shows similarity in the shared representation, not a prediction; colour is the Swiss-Prot "
    "location, not the model's call. Neighbours are by cosine similarity of the centred vectors "
    "in all 640 dimensions."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cache(directory: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Row hashes in shard order, the float32 matrix, and the cache spec."""
    spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    dim = int(spec["spec"]["embedding_dim"])
    hashes: list[str] = []
    blocks: list[np.ndarray] = []
    for tsv in sorted(directory.glob("shard_*.tsv")):
        rows = [line.split("\t")[0] for line in tsv.read_text(encoding="utf-8").splitlines()[1:]]
        data = np.frombuffer(tsv.with_suffix(".f32").read_bytes(), dtype="<f4")
        if data.size != len(rows) * dim:
            message = f"{tsv.name}: {data.size} floats for {len(rows)} rows of {dim}"
            raise ValueError(message)
        hashes.extend(rows)
        blocks.append(data.reshape(len(rows), dim))
    if not hashes:
        message = f"no shards in {directory}"
        raise ValueError(message)
    return hashes, np.vstack(blocks).astype(np.float64), spec


def pca(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores on the first ``k`` components and each component's share of the variance.

    Signs are fixed (largest-magnitude loading positive) so a rebuild gives the same picture.
    """
    centred = matrix - matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:k]
    for i, row in enumerate(components):
        if row[np.argmax(np.abs(row))] < 0:
            components[i] = -row
    scores = centred @ components.T
    variance = singular**2
    return scores, variance[:k] / variance.sum()


def neighbours(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row, the ``k`` other rows with the highest cosine similarity, best first.

    Vectors are centred first: raw mean-pooled ESM-2 vectors share a large common direction, so
    uncentred cosines all sit near 0.97 and say little about which neighbour is closer.
    """
    centred = matrix - matrix.mean(axis=0)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(similarity, order, axis=1)


def build(cache_dir: Path, catalog_path: Path, engine_path: Path) -> dict[str, Any]:
    """The space document, with every input identified by hash."""
    hashes, matrix, spec = read_cache(cache_dir)
    codes: dict[str, str] = json.loads(engine_path.read_text(encoding="utf-8"))["localization"][
        "codes"
    ]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog["entries"]:
        by_sha.setdefault(entry["sequence_sha256"], []).append(entry)

    keep = [i for i, h in enumerate(hashes) if h in by_sha]
    hashes = [hashes[i] for i in keep]
    matrix = matrix[keep]
    scores, ratio = pca(matrix, COMPONENTS)
    scale = float(np.abs(scores).max()) or 1.0
    order, similarity = neighbours(matrix, NEIGHBOURS)

    points = []
    for row, sha in enumerate(hashes):
        entries = sorted(by_sha[sha], key=lambda e: e["id"])
        first = entries[0]
        labels = first["database_labels"]["localization"]
        points.append(
            {
                "sha256": sha,
                "ids": [e["id"] for e in entries],
                "entry_name": first["entry_name"],
                "organism": first.get("organism"),
                "organism_group": first["organism_group"],
                "length": first["length"],
                "partition": first["partition"],
                "in_training_set": first["in_training_set"],
                "locations": [codes[label] for label in labels if label in codes],
                "xyz": [round(float(v) / scale, DIGITS) for v in scores[row]],
                "neighbours": [
                    [int(j), round(float(s), DIGITS)]
                    for j, s in zip(order[row], similarity[row], strict=True)
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "pca",
        "components": COMPONENTS,
        "explained_variance_ratio": [round(float(r), DIGITS) for r in ratio],
        "neighbour_metric": "cosine of centred vectors, 640 dimensions",
        "embedding": {
            "checkpoint": spec["spec"]["checkpoint"],
            "revision": spec["spec"]["revision"],
            "pooling": spec["spec"]["pooling"],
            "cache_key": spec["cache_key"],
        },
        "catalog_sha256": _sha256(catalog_path),
        "scope_label": SCOPE,
        "note": NOTE,
        "points": points,
    }


def render(document: dict[str, Any]) -> bytes:
    """Canonical bytes: sorted keys, no whitespace, a trailing newline."""
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    """Write the space document and its manifest, or check that they are current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the output is stale")
    args = parser.parse_args(argv)

    body = render(build(CACHE_DIR, CATALOG, ENGINE_CONFIG))
    digest = hashlib.sha256(body).hexdigest()
    target = OUT_DIR / SPACE_FILE
    if args.check:
        current = target.read_bytes() if target.is_file() else b""
        if current != body:
            sys.stderr.write("data/space/space.json is stale. Run: python scripts/build_space.py\n")
            return 1
        sys.stdout.write(f"data/space/space.json is current ({digest[:12]})\n")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    manifest = {"files_sha256": {SPACE_FILE: digest}, "schema_version": SCHEMA_VERSION}
    (OUT_DIR / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    count = len(json.loads(body)["points"])
    sys.stdout.write(f"wrote data/space/space.json: {count} points, sha256 {digest[:12]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
