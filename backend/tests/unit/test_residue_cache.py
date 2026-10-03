"""The per-residue cache: layout, resume, verification, reading and the pooled comparison.

A fake encoder stands in for ESM-2 (no torch, no weights): its states are a pure function of
the sequence, so every expected value here is known without running a model.
"""

import hashlib
from pathlib import Path
from typing import Any
import zlib

import pytest

from protloc.data import embeddings, residue_cache

DIM = 4
SEQUENCES = ["MKTAYIAK", "MKV", "GGGGGGGGGGGG", "ACDEFGHIK", "LLLLL", "PQRSTVWY", "MM"]


def _np() -> Any:
    return pytest.importorskip("numpy")


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _items(sequences: list[str]) -> list[embeddings.UniqueSequence]:
    return [embeddings.UniqueSequence(_sha(s), s, 1) for s in sequences]


def _states(sequence: str) -> Any:
    np = _np()
    rng = np.random.default_rng(zlib.crc32(sequence.encode("ascii")))
    return rng.uniform(-1.0, 1.0, size=(len(sequence), DIM)).astype(np.float32)


class FakeEncoder:
    """Per-residue states and pooled vectors from the same pure function of the sequence."""

    def __init__(self) -> None:
        """Start with no calls counted."""
        self.calls = 0

    def embed_residues(self, sequences: Any) -> list[bytes]:
        """Float16 states of each sequence."""
        self.calls += 1
        return [_states(s).astype("<f2").tobytes() for s in sequences]

    def embed_batch(self, sequences: Any) -> list[list[float]]:
        """Pooled vector of each sequence: the mean of its states."""
        return [_states(s).mean(axis=0).tolist() for s in sequences]


@pytest.fixture
def spec() -> residue_cache.ResidueSpec:
    base = embeddings.CacheSpec(
        checkpoint="fake/esm", revision="a" * 40, embedding_dim=DIM, max_residues=8, window_stride=4
    )
    return residue_cache.ResidueSpec(base)


def _options(size: int = 3, stop_after: int | None = None) -> embeddings.ExtractOptions:
    return embeddings.ExtractOptions(shard_size=size, stop_after=stop_after)


def test_states_are_stored_and_read_back_exactly(tmp_path: Path, spec: Any) -> None:
    np = _np()
    items = _items(SEQUENCES)
    report = residue_cache.extract_residues(items, tmp_path, spec, FakeEncoder(), _options())
    assert report.complete
    assert report.shards_total == 3  # 7 sequences in shards of 3 -> 3 + 3 + 1
    store = residue_cache.ResidueStore(tmp_path)
    assert store.dim == DIM
    assert store.lengths == {_sha(s): len(s) for s in SEQUENCES}
    for sequence in SEQUENCES:
        stored = store.states(_sha(sequence))
        assert stored.shape == (len(sequence), DIM)
        assert (stored == _states(sequence).astype("<f2")).all()
    assert np.isfinite(store.states(_sha("MKV")).astype(np.float32)).all()


def test_the_index_counts_sequences_and_residues(tmp_path: Path, spec: Any) -> None:
    residue_cache.extract_residues(_items(SEQUENCES), tmp_path, spec, FakeEncoder(), _options())
    index = residue_cache.read_residue_index(tmp_path)
    assert index["sequence_count"] == len(SEQUENCES)
    assert index["residue_count"] == sum(len(s) for s in SEQUENCES)
    assert index["cache_key"] == spec.key()
    assert residue_cache.verify_residue_cache(tmp_path) == []
    total = sum(entry["f16_bytes"] for entry in index["shards"])
    assert total == sum(len(s) for s in SEQUENCES) * DIM * residue_cache.HALF_BYTES


def test_a_killed_run_resumes_and_gives_the_same_bytes(tmp_path: Path, spec: Any) -> None:
    items = _items(SEQUENCES)
    whole = tmp_path / "whole"
    residue_cache.extract_residues(items, whole, spec, FakeEncoder(), _options())
    resumed = tmp_path / "resumed"
    first = residue_cache.extract_residues(
        items, resumed, spec, FakeEncoder(), _options(stop_after=1)
    )
    assert not first.complete
    assert not (resumed / "index.json").exists()
    encoder = FakeEncoder()
    second = residue_cache.extract_residues(items, resumed, spec, encoder, _options())
    assert second.complete
    assert (second.shards_skipped, second.shards_computed) == (1, 2)
    assert encoder.calls == 2
    assert (
        residue_cache.read_residue_index(whole)["shard_index_sha256"]
        == residue_cache.read_residue_index(resumed)["shard_index_sha256"]
    )


def test_a_directory_with_another_key_is_refused(tmp_path: Path, spec: Any) -> None:
    residue_cache.extract_residues(_items(SEQUENCES), tmp_path, spec, FakeEncoder(), _options())
    other = residue_cache.ResidueSpec(
        embeddings.CacheSpec(
            checkpoint="fake/esm",
            revision="b" * 40,
            embedding_dim=DIM,
            max_residues=8,
            window_stride=4,
        )
    )
    with pytest.raises(embeddings.EmbeddingError, match="different key"):
        residue_cache.extract_residues(
            _items(SEQUENCES), tmp_path, other, FakeEncoder(), _options()
        )


def test_an_encoder_that_returns_the_wrong_size_is_refused(tmp_path: Path, spec: Any) -> None:
    class Short(FakeEncoder):
        def embed_residues(self, sequences: Any) -> list[bytes]:
            return [blob[:-2] for blob in super().embed_residues(sequences)]

    with pytest.raises(embeddings.EmbeddingError, match="wrong number or size"):
        residue_cache.extract_residues(_items(SEQUENCES), tmp_path, spec, Short(), _options())
    assert not list(tmp_path.glob("shard_*.tsv"))


def test_a_flipped_byte_is_found_by_verification(tmp_path: Path, spec: Any) -> None:
    residue_cache.extract_residues(_items(SEQUENCES), tmp_path, spec, FakeEncoder(), _options())
    target = tmp_path / "shard_00001.f16"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))
    problems = residue_cache.verify_residue_cache(tmp_path)
    assert problems == ["shard_00001.f16 does not match its recorded hash"]


def test_an_incomplete_cache_cannot_be_opened(tmp_path: Path, spec: Any) -> None:
    residue_cache.extract_residues(
        _items(SEQUENCES), tmp_path, spec, FakeEncoder(), _options(stop_after=1)
    )
    with pytest.raises(embeddings.EmbeddingError, match="not complete"):
        residue_cache.ResidueStore(tmp_path)


def test_an_unknown_hash_is_an_error(tmp_path: Path, spec: Any) -> None:
    residue_cache.extract_residues(_items(SEQUENCES), tmp_path, spec, FakeEncoder(), _options())
    with pytest.raises(embeddings.EmbeddingError, match="no residue states"):
        residue_cache.ResidueStore(tmp_path).states("0" * 64)


def test_a_pooled_cache_is_not_a_residue_cache(tmp_path: Path, spec: Any) -> None:
    embeddings.extract(_items(SEQUENCES), tmp_path, spec.base, FakeEncoder(), _options())
    with pytest.raises(embeddings.EmbeddingError, match="not a per-residue cache index"):
        residue_cache.read_residue_index(tmp_path)


def test_the_residue_key_depends_on_the_pooled_key_and_the_storage(spec: Any) -> None:
    assert spec.as_dict()["pooled_cache_key"] == spec.base.key()
    assert spec.key() != spec.base.key()
    with pytest.raises(embeddings.EmbeddingError, match="float16"):
        residue_cache.ResidueSpec(spec.base, dtype="float32").validate()


def test_the_residue_directory_sits_beside_the_pooled_one() -> None:
    pooled = Path("data/embeddings/0e528d4a46831b05_subset500")
    assert residue_cache.residue_directory(pooled).name == "0e528d4a46831b05_subset500_residues"
    assert residue_cache.residue_directory(pooled).parent == pooled.parent


def _pooled(tmp_path: Path, spec: Any, shift: float = 0.0) -> Path:
    class Shifted(FakeEncoder):
        def embed_batch(self, sequences: Any) -> list[list[float]]:
            return [[v + shift for v in vec] for vec in super().embed_batch(sequences)]

    directory = tmp_path / "pooled"
    embeddings.extract(_items(SEQUENCES), directory, spec.base, Shifted(), _options())
    return directory


def test_pooled_lengths_come_from_the_shard_tables(tmp_path: Path, spec: Any) -> None:
    directory = _pooled(tmp_path, spec)
    assert residue_cache.pooled_lengths(directory) == {_sha(s): len(s) for s in SEQUENCES}
    with pytest.raises(embeddings.EmbeddingError, match="no committed shard tables"):
        residue_cache.pooled_lengths(tmp_path / "empty")


def test_the_mean_of_the_states_matches_the_pooled_vector(tmp_path: Path, spec: Any) -> None:
    pooled = _pooled(tmp_path, spec)
    residues = tmp_path / "residues"
    residue_cache.extract_residues(_items(SEQUENCES), residues, spec, FakeEncoder(), _options())
    report = residue_cache.compare_with_pooled(residue_cache.ResidueStore(residues), pooled)
    assert report["sequences"] == len(SEQUENCES)
    assert report["within_tolerance"] is True
    assert 0.0 < float(report["max_abs_diff"]) < residue_cache.POOLED_TOLERANCE


def test_a_pooled_vector_that_disagrees_is_reported(tmp_path: Path, spec: Any) -> None:
    pooled = _pooled(tmp_path, spec, shift=0.1)
    residues = tmp_path / "residues"
    residue_cache.extract_residues(_items(SEQUENCES), residues, spec, FakeEncoder(), _options())
    report = residue_cache.compare_with_pooled(residue_cache.ResidueStore(residues), pooled)
    assert report["within_tolerance"] is False
    assert float(report["max_abs_diff"]) == pytest.approx(0.1, abs=1e-3)


def _manifest(spec: Any) -> dict[str, Any]:
    return {
        "schema_version": residue_cache.SCHEMA_VERSION,
        "scope": "subset",
        "cache_key": spec.key(),
        "pooled_cache_key": spec.base.key(),
        "pooled_comparison": {"within_tolerance": True},
    }


def test_an_absent_or_sound_manifest_has_no_problems(spec: Any) -> None:
    assert residue_cache.check_residue_manifest(None, spec, spec.base.key()) == []
    assert residue_cache.check_residue_manifest(_manifest(spec), spec, spec.base.key()) == []


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m.update(schema_version=99), "schema_version"),
        (lambda m: m.update(cache_key="x"), "cache_key differs"),
        (lambda m: m.update(pooled_cache_key="x"), "pooled_cache_key"),
        (lambda m: m.update(pooled_comparison={"within_tolerance": False}), "not shown"),
        (lambda m: m.update(scope="full"), "subset cache"),
    ],
)
def test_each_manifest_disagreement_is_reported(spec: Any, edit: Any, message: str) -> None:
    manifest = _manifest(spec)
    edit(manifest)
    problems = residue_cache.check_residue_manifest(manifest, spec, spec.base.key())
    assert any(message in problem for problem in problems)
