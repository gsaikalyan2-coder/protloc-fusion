"""The embedding cache: windows, keys, shards, resume and lookup, with a fake encoder.

No torch and no numpy: the cache logic is tested with a deterministic stand-in whose
vector depends only on the sequence, so every property below is about the cache.
Sequences here are toy strings, as in the leakage proofs; the real working set is
embedded only by ``scripts/embed.py``.
"""

from collections.abc import Sequence
import hashlib
from itertools import pairwise
import json
from pathlib import Path

import pytest

from protloc.data.embeddings import (
    MAX_RESIDUES,
    WINDOW_STRIDE,
    CacheSpec,
    EmbeddingError,
    ExtractOptions,
    UniqueSequence,
    cache_size_bytes,
    extract,
    max_abs_diff,
    pack_vector,
    plan_shards,
    read_index,
    read_vector,
    residue_weights,
    select_unique,
    stratified_sample,
    unpack_vector,
    verify_cache,
    windows,
)

REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
DIM = 8


def _spec(**changes: object) -> CacheSpec:
    fields: dict[str, object] = {
        "checkpoint": "facebook/esm2_t30_150M_UR50D",
        "revision": REVISION,
        "embedding_dim": DIM,
    }
    fields.update(changes)
    return CacheSpec(**fields)  # type: ignore[arg-type]


class FakeEncoder:
    """Vector depends only on the sequence, like a deterministic real encoder."""

    def __init__(self) -> None:
        """Start with no calls counted."""
        self.calls = 0

    def embed_batch(self, sequences: Sequence[str]) -> list[list[float]]:
        """One digest-derived vector per sequence."""
        self.calls += 1
        out: list[list[float]] = []
        for sequence in sequences:
            digest = hashlib.sha256(sequence.encode("ascii")).digest()
            out.append([digest[i] / 255.0 for i in range(DIM)])
        return out


def _item(sequence: str, accessions: int = 1) -> UniqueSequence:
    return UniqueSequence(
        hashlib.sha256(sequence.encode("ascii")).hexdigest(), sequence, accessions
    )


def _items(count: int) -> list[UniqueSequence]:
    return [_item("M" + "ACDEFGHIK"[i % 9] * (i + 3) + "L" * i) for i in range(count)]


# ── windows ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("length", [1, 100, MAX_RESIDUES])
def test_a_sequence_that_fits_is_one_window(length: int) -> None:
    assert windows(length) == [(0, length)]


@pytest.mark.parametrize("length", [1023, 1100, 1496, 2044, 3000])
def test_long_sequences_are_covered_by_overlapping_windows(length: int) -> None:
    wins = windows(length)
    assert wins[0][0] == 0
    assert wins[-1][1] == length
    assert all(end - start <= MAX_RESIDUES for start, end in wins)
    assert all(b[0] - a[0] <= WINDOW_STRIDE for a, b in pairwise(wins))
    assert min(residue_weights(length, wins)) >= 1


def test_residue_weights_count_the_windows_over_each_residue() -> None:
    wins = windows(1496)
    weights = residue_weights(1496, wins)
    assert sum(weights) == sum(end - start for start, end in wins)
    assert weights[0] == 1
    assert weights[-1] == 1


def test_windows_reject_nonsense() -> None:
    with pytest.raises(EmbeddingError):
        windows(0)
    with pytest.raises(EmbeddingError):
        residue_weights(10, [(0, 5)])


# ── spec and key ─────────────────────────────────────────────────────────────


def test_the_key_depends_on_every_field() -> None:
    base = _spec().key()
    assert _spec().key() == base
    for change in (
        {"checkpoint": "facebook/esm2_t33_650M_UR50D"},
        {"revision": "0" * 40},
        {"embedding_dim": DIM + 1},
        {"max_residues": 1000},
        {"window_stride": 100},
        {"long_sequence_policy": "truncate"},
    ):
        assert _spec(**change).key() != base, change


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39, ""])
def test_a_revision_must_be_a_forty_hex_commit_hash(revision: str) -> None:
    with pytest.raises(EmbeddingError):
        _spec(revision=revision).validate()


def test_only_mean_float32_is_supported() -> None:
    with pytest.raises(EmbeddingError):
        _spec(pooling="cls").validate()


# ── selection ────────────────────────────────────────────────────────────────


def _corpus(tmp_path: Path, sequences: dict[str, str]) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    lines = []
    for accession, sequence in sequences.items():
        sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        record = {"accession": accession, "sequence_normalized": sequence, "sequence_sha256": sha}
        lines.append(json.dumps(record, separators=(",", ":")))
    corpus.write_text("\n".join(lines) + "\n", encoding="ascii")
    registry = tmp_path / "registry.tsv"
    rows = ["split_version\taccession\tpartition\tcluster_id\tpartitioner\tidentity_threshold"]
    rows += [f"homology_v1\t{a}\t1\tC0\tgraphpart\t0.30" for a in list(sequences)[:-1]]
    registry.write_text("\n".join(rows) + "\n", encoding="ascii")
    return corpus, registry


def test_identical_sequences_share_one_entry_and_the_registry_decides_the_set(
    tmp_path: Path,
) -> None:
    corpus, registry = _corpus(
        tmp_path, {"P1": "MKT", "P2": "MKT", "P3": "MAAA", "P4": "MNOT_IN_REGISTRY"}
    )
    chosen = select_unique(corpus, registry)
    assert [item.sequence for item in chosen] == sorted(
        ["MKT", "MAAA"], key=lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    assert {item.sequence: item.accessions for item in chosen} == {"MKT": 2, "MAAA": 1}


def test_a_registry_accession_missing_from_the_corpus_is_an_error(tmp_path: Path) -> None:
    corpus, registry = _corpus(tmp_path, {"P1": "MKT", "P2": "MAAA"})
    registry.write_text(registry.read_text(encoding="ascii") + "homology_v1\tP9\t1\tC0\tg\t0.3\n")
    with pytest.raises(EmbeddingError):
        select_unique(corpus, registry)


def test_the_sample_is_deterministic_and_spans_the_lengths() -> None:
    items = _items(60)
    first = stratified_sample(items, 7)
    assert first == stratified_sample(list(reversed(items)), 7)
    lengths = sorted(len(item.sequence) for item in items)
    assert min(len(i.sequence) for i in first) == lengths[0]
    assert max(len(i.sequence) for i in first) == lengths[-1]
    assert [i.sha256 for i in first] == sorted(i.sha256 for i in first)


def test_shards_are_runs_of_the_sorted_hashes() -> None:
    items = _items(10)
    shards = plan_shards(list(reversed(items)), 4)
    assert [len(s) for s in shards] == [4, 4, 2]
    flat = [item.sha256 for shard in shards for item in shard]
    assert flat == sorted(item.sha256 for item in items)


# ── vectors ──────────────────────────────────────────────────────────────────


def test_a_vector_round_trips_through_its_bytes() -> None:
    vector = [0.1, -2.5, 3.0e-7, 1.0]
    assert unpack_vector(pack_vector(vector)) == unpack_vector(
        pack_vector(unpack_vector(pack_vector(vector)))
    )
    assert max_abs_diff(unpack_vector(pack_vector(vector)), vector) < 1e-6


# ── extraction, resume, lookup ───────────────────────────────────────────────


def test_extraction_writes_every_vector_and_reads_it_back(tmp_path: Path) -> None:
    items = _items(11)
    encoder = FakeEncoder()
    report = extract(items, tmp_path, _spec(), encoder, ExtractOptions(shard_size=4))
    assert report.complete
    assert (report.shards_total, report.shards_computed, report.shards_skipped) == (3, 3, 0)
    index = read_index(tmp_path)
    assert index["sequence_count"] == 11
    assert index["cache_key"] == _spec().key()
    assert cache_size_bytes(tmp_path) > 11 * DIM * 4
    for item in items:
        expected = unpack_vector(pack_vector(encoder.embed_batch([item.sequence])[0]))
        assert read_vector(tmp_path, item.sha256) == expected
    assert verify_cache(tmp_path) == []


def test_a_resumed_run_has_the_same_bytes_as_an_uninterrupted_one(tmp_path: Path) -> None:
    items = _items(11)
    clean, resumed = tmp_path / "clean", tmp_path / "resumed"
    extract(items, clean, _spec(), FakeEncoder(), ExtractOptions(shard_size=4))
    partial = extract(items, resumed, _spec(), FakeEncoder(), ExtractOptions(4, stop_after=1))
    assert not partial.complete
    assert not (resumed / "index.json").exists()
    # a kill during the next shard leaves a temporary file and no commit mark
    (resumed / "shard_00001.f32.tmp").write_bytes(b"torn")
    final = extract(items, resumed, _spec(), FakeEncoder(), ExtractOptions(shard_size=4))
    assert (final.shards_skipped, final.shards_computed) == (1, 2)
    assert read_index(resumed)["shard_index_sha256"] == read_index(clean)["shard_index_sha256"]
    for name in ("shard_00000", "shard_00001", "shard_00002"):
        for suffix in (".f32", ".tsv"):
            assert (clean / (name + suffix)).read_bytes() == (
                resumed / (name + suffix)
            ).read_bytes()


def test_a_finished_cache_is_not_recomputed(tmp_path: Path) -> None:
    items = _items(6)
    extract(items, tmp_path, _spec(), FakeEncoder(), ExtractOptions(shard_size=3))
    encoder = FakeEncoder()
    report = extract(items, tmp_path, _spec(), encoder, ExtractOptions(shard_size=3))
    assert (report.shards_computed, report.shards_skipped, encoder.calls) == (0, 2, 0)


def test_a_directory_with_another_key_is_refused(tmp_path: Path) -> None:
    items = _items(3)
    extract(items, tmp_path, _spec(), FakeEncoder())
    with pytest.raises(EmbeddingError):
        extract(items, tmp_path, _spec(revision="1" * 40), FakeEncoder())


def test_an_encoder_returning_the_wrong_size_is_an_error(tmp_path: Path) -> None:
    class Wrong:
        def embed_batch(self, sequences: Sequence[str]) -> list[list[float]]:
            return [[0.0] for _ in sequences]

    with pytest.raises(EmbeddingError):
        extract(_items(2), tmp_path, _spec(), Wrong())


def test_a_corrupted_shard_is_detected(tmp_path: Path) -> None:
    extract(_items(5), tmp_path, _spec(), FakeEncoder(), ExtractOptions(shard_size=2))
    shard = tmp_path / "shard_00001.f32"
    data = bytearray(shard.read_bytes())
    data[0] ^= 1
    shard.write_bytes(bytes(data))
    assert any("shard_00001.f32" in problem for problem in verify_cache(tmp_path))


def test_an_incomplete_cache_has_no_index(tmp_path: Path) -> None:
    extract(_items(5), tmp_path, _spec(), FakeEncoder(), ExtractOptions(2, stop_after=1))
    with pytest.raises(EmbeddingError):
        read_index(tmp_path)


def test_a_missing_hash_is_an_error(tmp_path: Path) -> None:
    extract(_items(3), tmp_path, _spec(), FakeEncoder())
    with pytest.raises(EmbeddingError):
        read_vector(tmp_path, "f" * 64)
