"""The 500-sequence prototype subset: which of the feature table's sequences the cache holds.

Shared by ``scripts/baselines.py --subset`` (the floor) and ``scripts/head.py`` (the head), so
both resolve the same 500 sequences the same way (ADR-0009 prototype scope, ADR-0011).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from protloc.data.embeddings import read_pack
from protloc.data.manifest import sha256_file
from protloc.training import baseline_models as bm
from protloc.training.baselines import (
    SUBSET_SCOPE,
    TASKS,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    cache_lengths,
    read_manifest,
    subset_digest,
)

CACHE_DIR_HEX = 16
Array = Any


def embedding_manifest(path: Path) -> dict[str, Any]:
    """The embedding manifest, refused unless it describes the subset cache."""
    manifest = read_manifest(path)
    if manifest["sequences"]["scope"] != SUBSET_SCOPE:
        raise BaselineError(
            "the embedding manifest does not say 'subset'; --subset needs the subset cache"
        )
    return manifest


def pooled_directory(embedding: Mapping[str, Any], embed_root: Path) -> Path:
    """The Phase 7 pooled-vector cache directory the embedding manifest describes."""
    key = str(embedding["cache_key"])
    return embed_root / f"{key[:CACHE_DIR_HEX]}_subset{embedding['sequences']['subset_size']}"


def subset_block(
    table: Mapping[str, Array], embedding: Mapping[str, Any], embed_root: Path, pack: Path
) -> tuple[Array, dict[str, Any]]:
    """The subset as indices into the feature table, and the facts a manifest records."""
    directory = pooled_directory(embedding, embed_root)
    lengths = cache_lengths(directory)
    if len(lengths) != int(embedding["sequences"]["cached"]):
        raise BaselineError("the cache tables and the embedding manifest disagree on the count")
    if sha256_file(pack)[0] != embedding["input"]["pack_sha256"]:
        raise BaselineError("input_pack.tsv is not the pack the embedding manifest records")
    items, _header = read_pack(pack)
    indices = bm.locate_sequences(table, [item.sha256 for item in items], lengths)
    restricted = bm.restrict_to_sequences(table, indices)
    in_subset = bm.in_sequences(table, indices)
    return indices, {
        "sequences": len(lengths),
        "sequences_sha256": subset_digest(list(lengths)),
        "embedding_cache_key": str(embedding["cache_key"]),
        "cache_directory": directory.name,
        "accessions_by_partition": {
            str(p): int(((table["partition"] == p) & in_subset).sum())
            for p in (*VALIDATION_PARTITIONS, TEST_PARTITION)
        },
        "labelled_rows_per_task": {t: int(restricted[f"M_{t}"].sum()) for t in TASKS},
    }
