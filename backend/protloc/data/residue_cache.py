"""The per-residue embedding cache for the prototype subset (Phase 9, ADR-0011).

Phase 7 stores one mean-pooled ESM-2 vector per sequence (ADR-0009 section 2.7 chose not to
store per-residue states: 19.1 GB in float16 for the full corpus). The Phase 9 head pools
with attention over residues, so it needs the states. This module stores them for the
500-sequence prototype subset only (about 286 MB in float16) and reads them back.

What a cached state is: for each residue, the last hidden state of the frozen encoder, with
residues seen by more than one sliding window averaged over the windows that saw them
(exactly the average Phase 7's pooled vector is the mean of). So the mean over residues of a
sequence's states reproduces its Phase 7 pooled vector up to float16 rounding, and
:func:`compare_with_pooled` measures that instead of assuming it.

Layout of a cache directory (the same commit discipline as ``embeddings.py``)::

    spec.json              the spec and its key; a different key is refused
    shard_00000.f16        every residue of the shard's sequences in sha order,
                           little-endian float16, row-major (length, dim)
    shard_00000.tsv        sha256, length, windows, offset_residues; written LAST
    index.json             written only when every shard is committed

Writing and verifying need only the standard library. Reading states back needs numpy,
imported lazily, because 143 million float16 values are not something to unpack with
``struct``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import time
from typing import Any, Protocol

from protloc.data.embeddings import (
    CacheSpec,
    EmbeddingError,
    ExtractOptions,
    ExtractReport,
    UniqueSequence,
    plan_shards,
    read_vector,
    shard_name,
    windows,
)
from protloc.data.manifest import canonical_json_bytes, sha256_file

KIND = "per_residue"
DTYPE = "float16"
HALF_BYTES = 2
#: Residues per shard are not what is limited, sequences are: 50 sequences hold about 29 MB
#: of float16 states, so one shard is cheap to recompute after a kill.
SHARD_SEQUENCES = 50
#: float16 keeps 11 significant bits, so one stored state is off by at most about 5e-4 of its
#: magnitude; a mean over hundreds of residues averages those errors down. The comparison
#: with Phase 7's pooled vector passes if no element differs by more than this.
POOLED_TOLERANCE = 5e-3
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEADER = "sha256\tlength\twindows\toffset_residues"


class ResidueEncoder(Protocol):
    """Turns sequences into per-residue states. The torch encoder implements this."""

    def embed_residues(self, sequences: Sequence[str]) -> list[bytes]:
        """Return, per sequence, little-endian float16 bytes of its (length, dim) states."""


@dataclass(frozen=True, slots=True)
class ResidueSpec:
    """Everything that decides what a cached per-residue state is.

    It wraps the Phase 7 :class:`~protloc.data.embeddings.CacheSpec` (checkpoint, revision,
    dimension and window policy decide the states) and adds the storage dtype. The pooled
    cache key is part of the key, so states and pooled vectors can be matched.
    """

    base: CacheSpec
    dtype: str = DTYPE

    def validate(self) -> None:
        """Raise unless the spec could identify real states."""
        self.base.validate()
        if self.dtype != DTYPE:
            raise EmbeddingError(f"only {DTYPE} storage is implemented, not {self.dtype!r}")

    def as_dict(self) -> dict[str, object]:
        """The spec as plain JSON-able values."""
        return {
            "kind": KIND,
            "dtype": self.dtype,
            "pooled_cache_key": self.base.key(),
            "states": self.base.as_dict(),
        }

    def key(self) -> str:
        """SHA-256 of the canonical spec."""
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


def residue_directory(pooled_directory: Path) -> Path:
    """Where the states of a pooled subset cache live, next to it and never mixed into it."""
    return pooled_directory.with_name(pooled_directory.name + "_residues")


def pooled_lengths(directory: Path) -> dict[str, int]:
    """Sequence hash and length of every sequence in a Phase 7 cache, from its shard tables."""
    found: dict[str, int] = {}
    for table in sorted(directory.glob("shard_*.tsv")):
        for row in table.read_text(encoding="ascii").splitlines()[1:]:
            sha, length, _windows = row.split("\t")
            if not _SHA256.fullmatch(sha):
                raise EmbeddingError(f"{table.name}: {sha!r} is not a sequence hash")
            found[sha] = int(length)
    if not found:
        raise EmbeddingError(f"{directory} holds no committed shard tables")
    return found


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _shard_rows(shard: Sequence[UniqueSequence]) -> bytes:
    rows = [_HEADER]
    offset = 0
    for item in shard:
        length = len(item.sequence)
        rows.append(f"{item.sha256}\t{length}\t{len(windows(length))}\t{offset}")
        offset += length
    return ("\n".join(rows) + "\n").encode("ascii")


def _shard_residues(shard: Sequence[UniqueSequence]) -> int:
    return sum(len(item.sequence) for item in shard)


def _shard_done(directory: Path, number: int, shard: Sequence[UniqueSequence], dim: int) -> bool:
    stem = directory / shard_name(number)
    table, data = stem.with_suffix(".tsv"), stem.with_suffix(".f16")
    if not (table.exists() and data.exists()):
        return False
    if table.read_bytes() != _shard_rows(shard):
        return False
    return data.stat().st_size == _shard_residues(shard) * dim * HALF_BYTES


def _prepare(directory: Path, spec: ResidueSpec) -> None:
    spec.validate()
    directory.mkdir(parents=True, exist_ok=True)
    spec_file = directory / "spec.json"
    if spec_file.exists():
        existing = json.loads(spec_file.read_bytes())
        if existing.get("cache_key") != spec.key():
            raise EmbeddingError(
                f"{directory} holds a cache with a different key; use a new directory"
            )
    else:
        document = {"cache_key": spec.key(), "spec": spec.as_dict()}
        _write_atomic(spec_file, canonical_json_bytes(document))


def extract_residues(
    items: Sequence[UniqueSequence],
    directory: Path,
    spec: ResidueSpec,
    encoder: ResidueEncoder,
    options: ExtractOptions | None = None,
) -> ExtractReport:
    """Store the states of every item, one committed shard at a time; skip finished shards.

    The shard plan is the Phase 7 one (consecutive runs of the hash-sorted sequences), so a
    shard's content depends on the inputs and the spec and nothing else, and a killed run
    resumes with the same bytes. ``options.stop_after`` commits at most that many new shards,
    which is how a test interrupts a run at a shard boundary.
    """
    chosen = options or ExtractOptions(shard_size=SHARD_SEQUENCES)
    _prepare(directory, spec)
    plan = plan_shards(items, chosen.shard_size)
    dim = spec.base.embedding_dim
    computed = skipped = sequences = 0
    started = time.perf_counter()
    for number, shard in enumerate(plan):
        if _shard_done(directory, number, shard, dim):
            skipped += 1
            continue
        if chosen.stop_after is not None and computed >= chosen.stop_after:
            break
        shard_started = time.perf_counter()
        stem = directory / shard_name(number)
        stem.with_suffix(".tsv").unlink(missing_ok=True)
        blobs = encoder.embed_residues([item.sequence for item in shard])
        expected = [len(item.sequence) * dim * HALF_BYTES for item in shard]
        if [len(blob) for blob in blobs] != expected:
            raise EmbeddingError("encoder returned states of the wrong number or size")
        _write_atomic(stem.with_suffix(".f16"), b"".join(blobs))
        _write_atomic(stem.with_suffix(".tsv"), _shard_rows(shard))
        computed += 1
        sequences += len(shard)
        if chosen.on_shard is not None:
            chosen.on_shard(number, len(plan), time.perf_counter() - shard_started)
    seconds = time.perf_counter() - started
    complete = computed + skipped == len(plan)
    if complete:
        _write_index(directory, spec, plan)
    return ExtractReport(len(plan), computed, skipped, sequences, seconds, complete)


def _write_index(
    directory: Path, spec: ResidueSpec, plan: Sequence[Sequence[UniqueSequence]]
) -> None:
    shards: list[dict[str, object]] = []
    for number, shard in enumerate(plan):
        stem = directory / shard_name(number)
        data_sha, data_bytes = sha256_file(stem.with_suffix(".f16"))
        table_sha, _ = sha256_file(stem.with_suffix(".tsv"))
        shards.append(
            {
                "name": shard_name(number),
                "sequences": len(shard),
                "residues": _shard_residues(shard),
                "f16_sha256": data_sha,
                "f16_bytes": data_bytes,
                "tsv_sha256": table_sha,
            }
        )
    index = {
        "kind": KIND,
        "cache_key": spec.key(),
        "spec": spec.as_dict(),
        "sequence_count": sum(len(shard) for shard in plan),
        "residue_count": sum(_shard_residues(shard) for shard in plan),
        "shard_count": len(shards),
        "shard_index_sha256": hashlib.sha256(canonical_json_bytes(shards)).hexdigest(),
        "shards": shards,
    }
    _write_atomic(directory / "index.json", canonical_json_bytes(index))


def read_residue_index(directory: Path) -> dict[str, Any]:
    """The cache index, or an error if the cache is incomplete or is not a per-residue cache."""
    path = directory / "index.json"
    if not path.exists():
        raise EmbeddingError(f"{directory} has no index.json: extraction is not complete")
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or document.get("kind") != KIND:
        raise EmbeddingError(f"{path} is not a per-residue cache index")
    return {str(k): v for k, v in document.items()}


def verify_residue_cache(directory: Path) -> list[str]:
    """Problems found by re-hashing every shard against ``index.json``; empty if sound."""
    index = read_residue_index(directory)
    shards = index.get("shards")
    if not isinstance(shards, list):
        return ["index.json has no shard list"]
    problems: list[str] = []
    if hashlib.sha256(canonical_json_bytes(shards)).hexdigest() != index["shard_index_sha256"]:
        problems.append("shard_index_sha256 does not match the shard list")
    for entry in shards:
        stem = directory / str(entry["name"])
        for suffix, field in ((".f16", "f16_sha256"), (".tsv", "tsv_sha256")):
            path = stem.with_suffix(suffix)
            if not path.exists():
                problems.append(f"{path.name} is missing")
            elif sha256_file(path)[0] != entry[field]:
                problems.append(f"{path.name} does not match its recorded hash")
    return problems


def _np() -> Any:
    return importlib.import_module("numpy")


class ResidueStore:
    """Read access to a finished per-residue cache, by sequence hash."""

    def __init__(self, directory: Path) -> None:
        """Open a cache; refuses one whose index is missing or whose shard tables disagree."""
        index = read_residue_index(directory)
        self.directory = directory
        self.cache_key = str(index["cache_key"])
        self.dim = int(index["spec"]["states"]["embedding_dim"])
        self._rows: dict[str, tuple[Path, int, int]] = {}
        for entry in index["shards"]:
            stem = directory / str(entry["name"])
            lines = stem.with_suffix(".tsv").read_text(encoding="ascii").splitlines()
            if lines[0] != _HEADER:
                raise EmbeddingError(f"{stem.name}.tsv has an unexpected header")
            for line in lines[1:]:
                sha, length, _windows, offset = line.split("\t")
                self._rows[sha] = (stem.with_suffix(".f16"), int(offset), int(length))
        if len(self._rows) != int(index["sequence_count"]):
            raise EmbeddingError("shard tables and index.json disagree on the sequence count")
        self._maps: dict[Path, Any] = {}

    @property
    def lengths(self) -> dict[str, int]:
        """Residue count of every cached sequence, by hash."""
        return {sha: length for sha, (_path, _offset, length) in self._rows.items()}

    def states(self, sha256: str) -> Any:
        """The (length, dim) float16 states of one sequence, as a read-only numpy view."""
        if sha256 not in self._rows:
            raise EmbeddingError(f"no residue states for {sha256}")
        path, offset, length = self._rows[sha256]
        if path not in self._maps:
            self._maps[path] = _np().memmap(path, dtype="<f2", mode="r")
        flat = self._maps[path]
        start = offset * self.dim
        return flat[start : start + length * self.dim].reshape(length, self.dim)


def compare_with_pooled(store: ResidueStore, pooled_directory: Path) -> dict[str, object]:
    """How far the mean of each sequence's states is from its Phase 7 pooled vector.

    The two are the same quantity computed once in float32 (pooled) and once from float16
    storage (this cache), so the difference is the float16 rounding and nothing else. Reads
    the pooled vectors through :func:`protloc.data.embeddings.read_vector`.
    """
    np = _np()
    worst = 0.0
    total = 0.0
    worst_sha = ""
    for sha in sorted(store.lengths):
        mean = store.states(sha).astype(np.float64).mean(axis=0)
        pooled = np.array(read_vector(pooled_directory, sha), dtype=np.float64)
        difference = float(np.abs(mean - pooled).max())
        total += difference
        if difference > worst:
            worst, worst_sha = difference, sha
    count = len(store.lengths)
    return {
        "sequences": count,
        "max_abs_diff": worst,
        "mean_of_max_abs_diff": total / count,
        "worst_sha256": worst_sha,
        "tolerance": POOLED_TOLERANCE,
        "within_tolerance": worst <= POOLED_TOLERANCE,
    }


def check_residue_manifest(
    manifest: Mapping[str, Any] | None, spec: ResidueSpec, pooled_manifest_key: str
) -> list[str]:
    """Every way a residue-cache manifest disagrees with the pins; empty if sound or absent."""
    if manifest is None:
        return []
    problems: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("residue manifest schema_version is not the one this code writes")
    if manifest.get("cache_key") != spec.key():
        problems.append("residue manifest cache_key differs from the pinned spec")
    if manifest.get("pooled_cache_key") != pooled_manifest_key:
        problems.append("residue manifest pooled_cache_key differs from the embedding manifest")
    comparison = manifest.get("pooled_comparison")
    if not isinstance(comparison, dict) or comparison.get("within_tolerance") is not True:
        problems.append("residue states were not shown to reproduce the pooled vectors")
    if manifest.get("scope") != "subset":
        problems.append("the residue cache is a subset cache and the manifest must say so")
    return problems
