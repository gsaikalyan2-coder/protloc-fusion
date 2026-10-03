"""The embedding cache: which vectors exist, how they are stored, and how they resume.

Phase 7 turns every distinct working-set sequence into one mean-pooled ESM-2 vector
and stores it. This module is the part of that job that needs neither torch nor
numpy, so CI can test all of it with a fake encoder:

* :class:`CacheSpec` and its :meth:`~CacheSpec.key`: what identifies a vector;
* :func:`windows` and :func:`residue_weights`: the sliding-window policy for
  sequences longer than the encoder context (nothing is truncated);
* :func:`select_unique`: the registry sequences, one per distinct sequence hash;
* :func:`extract`: shard-by-shard, resumable extraction through an
  :class:`Encoder`;
* :func:`read_vector`: lookup by sequence hash.

Layout of a cache directory::

    spec.json              the spec and its key; a different key is refused
    shard_00000.f32        n * dim little-endian float32 values, rows in sha order
    shard_00000.tsv        sha256, length, windows per row; written LAST (commit mark)
    index.json             written only when every shard is committed

A shard holds the sorted hashes ``[i * N, (i + 1) * N)``, so its content depends on
the registry and the spec and on nothing else: a killed run resumes and produces the
same bytes. Vectors are stored as raw bytes produced with :mod:`struct`, so reading
the cache never needs numpy.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from typing import Protocol

from protloc.data.manifest import ManifestError, canonical_json_bytes, sha256_file

POOLING = "mean"
DTYPE = "float32"
LONG_SEQUENCE_POLICY = "sliding_window"
#: ESM-2 holds 1024 tokens: ``<cls>``, ``<eos>`` and at most this many residues.
MAX_RESIDUES = 1022
#: Window start step. Half a window, so every interior residue is seen at least twice.
WINDOW_STRIDE = 511
SHARD_SIZE = 512
_FLOAT_BYTES = 4
_ACCESSION_PREFIX = b'{"accession":"'
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EmbeddingError(ManifestError):
    """The cache, its spec or its inputs are inconsistent."""


class Encoder(Protocol):
    """Turns sequences into pooled vectors. The torch encoder implements this."""

    def embed_batch(self, sequences: Sequence[str]) -> list[list[float]]:
        """Return one vector of ``embedding_dim`` floats per sequence, in order."""


@dataclass(frozen=True, slots=True)
class CacheSpec:
    """Everything that decides what a cached vector is.

    A checkpoint name alone does not identify the weights, and a vector from a
    different pooling or window policy is a different vector, so all of it goes
    into :meth:`key`.
    """

    checkpoint: str
    revision: str
    embedding_dim: int
    pooling: str = POOLING
    dtype: str = DTYPE
    max_residues: int = MAX_RESIDUES
    window_stride: int = WINDOW_STRIDE
    long_sequence_policy: str = LONG_SEQUENCE_POLICY

    def validate(self) -> None:
        """Raise unless the spec could identify real weights."""
        if not _REVISION.fullmatch(self.revision):
            raise EmbeddingError(f"revision must be 40 lowercase hex characters: {self.revision!r}")
        if self.embedding_dim <= 0:
            raise EmbeddingError(f"embedding_dim must be positive: {self.embedding_dim}")
        if not 0 < self.window_stride <= self.max_residues:
            raise EmbeddingError("window_stride must be in (0, max_residues]")
        if self.pooling != POOLING or self.dtype != DTYPE:
            raise EmbeddingError("only mean pooling in float32 is implemented")

    def as_dict(self) -> dict[str, object]:
        """The spec as plain JSON-able values."""
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "dtype": self.dtype,
            "max_residues": self.max_residues,
            "window_stride": self.window_stride,
            "long_sequence_policy": self.long_sequence_policy,
        }

    def key(self) -> str:
        """SHA-256 of the canonical spec: checkpoint, revision, pooling and policy."""
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class UniqueSequence:
    """One distinct sequence of the registry and how many accessions share it."""

    sha256: str
    sequence: str
    accessions: int


@dataclass(frozen=True, slots=True)
class ExtractOptions:
    """How :func:`extract` runs; none of it changes the bytes it writes."""

    shard_size: int = SHARD_SIZE
    stop_after: int | None = None
    on_shard: Callable[[int, int, float], None] | None = None


@dataclass(frozen=True, slots=True)
class ExtractReport:
    """What one :func:`extract` call did."""

    shards_total: int
    shards_computed: int
    shards_skipped: int
    sequences_computed: int
    seconds: float
    complete: bool


def windows(
    length: int, size: int = MAX_RESIDUES, stride: int = WINDOW_STRIDE
) -> list[tuple[int, int]]:
    """Half-open residue windows covering ``[0, length)``.

    A sequence that fits is one window. A longer one starts a window every
    ``stride`` residues and the last window is moved back to end exactly at
    ``length``, so every residue is covered and no window is shorter than needed.
    """
    if length <= 0 or size <= 0 or stride <= 0:
        raise EmbeddingError("length, size and stride must be positive")
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size, stride))
    starts.append(length - size)
    return [(start, start + size) for start in starts]


def residue_weights(length: int, wins: Sequence[tuple[int, int]]) -> list[int]:
    """How many windows cover each residue. Every entry must be at least one."""
    counts = [0] * length
    for start, end in wins:
        for position in range(start, end):
            counts[position] += 1
    if min(counts) < 1:
        raise EmbeddingError("windows leave a residue uncovered")
    return counts


def select_unique(corpus: Path, registry: Path) -> list[UniqueSequence]:
    """Distinct sequences of the registry, sorted by sequence hash.

    Identical sequences under different accessions share one vector, matching
    Phase 6's collapse. The corpus supplies the sequence, the registry the set.
    """
    lines = registry.read_text(encoding="ascii").splitlines()
    if not lines or lines[0].split("\t")[:2] != ["split_version", "accession"]:
        raise EmbeddingError(f"{registry} is not a split registry")
    wanted = {line.split("\t")[1] for line in lines[1:]}
    found: dict[str, tuple[str, int]] = {}
    seen = 0
    with corpus.open("rb") as handle:
        for raw in handle:
            # Canonical lines start with {"accession":"X": skip the others unparsed.
            if not raw.startswith(_ACCESSION_PREFIX):
                raise EmbeddingError("corpus line does not start with the accession field")
            accession = raw[len(_ACCESSION_PREFIX) : raw.index(b'"', len(_ACCESSION_PREFIX))]
            if accession.decode("ascii") not in wanted:
                continue
            record = json.loads(raw)
            seen += 1
            sha = str(record["sequence_sha256"])
            sequence = str(record["sequence_normalized"])
            if hashlib.sha256(sequence.encode("ascii")).hexdigest() != sha:
                raise EmbeddingError(f"{record['accession']}: sequence does not match its hash")
            previous = found.get(sha)
            found[sha] = (sequence, 1 if previous is None else previous[1] + 1)
    if seen != len(wanted):
        raise EmbeddingError(f"registry names {len(wanted)} accessions, corpus has {seen}")
    return [UniqueSequence(sha, found[sha][0], found[sha][1]) for sha in sorted(found)]


def stratified_sample(items: Sequence[UniqueSequence], n: int) -> list[UniqueSequence]:
    """``n`` sequences spread evenly over the length distribution, deterministically.

    Items are ordered by ``(length, sha256)`` and taken at evenly spaced ranks, so
    the sample spans the shortest to the longest sequence with no random draw. The
    result is returned in sha order, the order shards use.
    """
    if not 0 < n <= len(items):
        raise EmbeddingError(f"cannot sample {n} of {len(items)} sequences")
    ranked = sorted(items, key=lambda item: (len(item.sequence), item.sha256))
    if n == 1:
        picked = [ranked[len(ranked) // 2]]
    else:
        picked = [ranked[i * (len(ranked) - 1) // (n - 1)] for i in range(n)]
    return sorted(picked, key=lambda item: item.sha256)


def plan_shards(items: Sequence[UniqueSequence], shard_size: int) -> list[list[UniqueSequence]]:
    """Consecutive runs of the hash-sorted sequences. Shard ``i`` is fixed by the input."""
    if shard_size <= 0:
        raise EmbeddingError("shard_size must be positive")
    ordered = sorted(items, key=lambda item: item.sha256)
    return [ordered[i : i + shard_size] for i in range(0, len(ordered), shard_size)]


def shard_name(number: int) -> str:
    """Stem of the files of shard ``number``."""
    return f"shard_{number:05d}"


def pack_vector(vector: Sequence[float]) -> bytes:
    """Little-endian float32 bytes of one vector."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(data: bytes) -> tuple[float, ...]:
    """Inverse of :func:`pack_vector`."""
    return struct.unpack(f"<{len(data) // _FLOAT_BYTES}f", data)


def max_abs_diff(first: Sequence[float], second: Sequence[float]) -> float:
    """Largest absolute element-wise difference of two equal-length vectors."""
    if len(first) != len(second):
        raise EmbeddingError("vectors differ in length")
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _shard_rows(shard: Sequence[UniqueSequence]) -> bytes:
    rows = ["sha256\tlength\twindows"]
    for item in shard:
        count = len(windows(len(item.sequence)))
        rows.append(f"{item.sha256}\t{len(item.sequence)}\t{count}")
    return ("\n".join(rows) + "\n").encode("ascii")


def _shard_done(directory: Path, number: int, shard: Sequence[UniqueSequence], dim: int) -> bool:
    stem = directory / shard_name(number)
    table = stem.with_suffix(".tsv")
    data = stem.with_suffix(".f32")
    if not (table.exists() and data.exists()):
        return False
    if table.read_bytes() != _shard_rows(shard):
        return False
    return data.stat().st_size == len(shard) * dim * _FLOAT_BYTES


def _prepare(directory: Path, spec: CacheSpec) -> None:
    spec.validate()
    directory.mkdir(parents=True, exist_ok=True)
    spec_file = directory / "spec.json"
    document = {"cache_key": spec.key(), "spec": spec.as_dict()}
    if spec_file.exists():
        existing = json.loads(spec_file.read_bytes())
        if existing.get("cache_key") != spec.key():
            raise EmbeddingError(
                f"{directory} holds a cache with a different key; use a new directory"
            )
    else:
        _write_atomic(spec_file, canonical_json_bytes(document))


def extract(
    items: Sequence[UniqueSequence],
    directory: Path,
    spec: CacheSpec,
    encoder: Encoder,
    options: ExtractOptions | None = None,
) -> ExtractReport:
    """Embed every item, one committed shard at a time; skip shards already done.

    ``options.stop_after`` commits at most that many new shards and returns, which is
    how a test interrupts a run at a shard boundary. ``index.json`` exists only after
    every shard is committed.
    """
    chosen = options or ExtractOptions()
    _prepare(directory, spec)
    plan = plan_shards(items, chosen.shard_size)
    computed = skipped = sequences = 0
    started = time.perf_counter()
    for number, shard in enumerate(plan):
        if _shard_done(directory, number, shard, spec.embedding_dim):
            skipped += 1
            continue
        if chosen.stop_after is not None and computed >= chosen.stop_after:
            break
        shard_started = time.perf_counter()
        stem = directory / shard_name(number)
        stem.with_suffix(".tsv").unlink(missing_ok=True)
        vectors = encoder.embed_batch([item.sequence for item in shard])
        if len(vectors) != len(shard) or any(len(v) != spec.embedding_dim for v in vectors):
            raise EmbeddingError("encoder returned the wrong number or size of vectors")
        _write_atomic(stem.with_suffix(".f32"), b"".join(pack_vector(v) for v in vectors))
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
    directory: Path, spec: CacheSpec, plan: Sequence[Sequence[UniqueSequence]]
) -> None:
    shards: list[dict[str, object]] = []
    for number, shard in enumerate(plan):
        stem = directory / shard_name(number)
        data_sha, data_bytes = sha256_file(stem.with_suffix(".f32"))
        table_sha, _ = sha256_file(stem.with_suffix(".tsv"))
        shards.append(
            {
                "name": shard_name(number),
                "sequences": len(shard),
                "f32_sha256": data_sha,
                "f32_bytes": data_bytes,
                "tsv_sha256": table_sha,
            }
        )
    index = {
        "cache_key": spec.key(),
        "spec": spec.as_dict(),
        "sequence_count": sum(len(shard) for shard in plan),
        "shard_count": len(shards),
        "shard_index_sha256": hashlib.sha256(canonical_json_bytes(shards)).hexdigest(),
        "shards": shards,
    }
    _write_atomic(directory / "index.json", canonical_json_bytes(index))


def read_index(directory: Path) -> dict[str, object]:
    """The cache index, or an error if the cache is incomplete."""
    path = directory / "index.json"
    if not path.exists():
        raise EmbeddingError(f"{directory} has no index.json: extraction is not complete")
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise EmbeddingError("index.json must be a JSON object")
    return {str(k): v for k, v in document.items()}


def cache_size_bytes(directory: Path) -> int:
    """Total bytes of the committed shard and table files."""
    return sum(
        path.stat().st_size for path in directory.iterdir() if path.suffix in {".f32", ".tsv"}
    )


def read_vector(directory: Path, sha256: str) -> tuple[float, ...]:
    """The cached vector of the sequence with this hash."""
    if not _SHA256.fullmatch(sha256):
        raise EmbeddingError("sha256 must be 64 lowercase hex characters")
    index = read_index(directory)
    dim = int(str(_mapping(index["spec"])["embedding_dim"]))
    for table in sorted(directory.glob("shard_*.tsv")):
        rows = table.read_text(encoding="ascii").splitlines()[1:]
        for row_number, row in enumerate(rows):
            if row.split("\t")[0] == sha256:
                offset = row_number * dim * _FLOAT_BYTES
                with table.with_suffix(".f32").open("rb") as handle:
                    handle.seek(offset)
                    return unpack_vector(handle.read(dim * _FLOAT_BYTES))
    raise EmbeddingError(f"no vector for {sha256}")


def verify_cache(directory: Path) -> list[str]:
    """Problems found by re-hashing every shard against ``index.json``; empty if sound."""
    index = read_index(directory)
    problems: list[str] = []
    shards = index["shards"]
    if not isinstance(shards, list):
        return ["index.json has no shard list"]
    if hashlib.sha256(canonical_json_bytes(shards)).hexdigest() != index["shard_index_sha256"]:
        problems.append("shard_index_sha256 does not match the shard list")
    for entry in shards:
        stem = directory / str(entry["name"])
        for suffix, field in ((".f32", "f32_sha256"), (".tsv", "tsv_sha256")):
            path = stem.with_suffix(suffix)
            if not path.exists():
                problems.append(f"{path.name} is missing")
            elif sha256_file(path)[0] != entry[field]:
                problems.append(f"{path.name} does not match its recorded hash")
    return problems


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EmbeddingError("expected a JSON object")
    return {str(k): v for k, v in value.items()}


__all__ = [
    "DTYPE",
    "LONG_SEQUENCE_POLICY",
    "MAX_RESIDUES",
    "POOLING",
    "SHARD_SIZE",
    "WINDOW_STRIDE",
    "CacheSpec",
    "EmbeddingError",
    "Encoder",
    "ExtractOptions",
    "ExtractReport",
    "UniqueSequence",
    "cache_size_bytes",
    "extract",
    "max_abs_diff",
    "pack_vector",
    "plan_shards",
    "read_index",
    "read_vector",
    "residue_weights",
    "select_unique",
    "shard_name",
    "stratified_sample",
    "unpack_vector",
    "verify_cache",
    "windows",
]
