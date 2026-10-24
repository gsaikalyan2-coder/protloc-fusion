"""The opt-in live encoder (feature 13, ADR-0019).

The encoder module is tested with fake ESM-2 implementations, so no torch is needed. The engine
tests use the real registry and residue cache (they skip without them, like ``test_engine.py``)
and a fake encoder that returns the cached states for a cached sequence and deterministic
stand-in states for any other: what is proven is the wiring (self-check, labelling, refusal,
provenance), not ESM-2 itself. The real encoder is exercised by ``scripts/fetch_esm.py``.
"""

import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from protloc.inference.live_encoder import (  # noqa: E402
    SELF_CHECK_TOLERANCE,
    LiveEncoder,
    LiveEncoderError,
    load_live_encoder,
    torch_factory,
)

SPEC = {
    "checkpoint": "facebook/esm2_t30_150M_UR50D",
    "revision": "a" * 40,
    "embedding_dim": 4,
    "pooling": "mean",
    "dtype": "float32",
    "max_residues": 1022,
    "window_stride": 511,
    "long_sequence_policy": "sliding_window",
}


def _states(sequence: str, dim: int = 4, shift: float = 0.0) -> Any:
    base = np.array([[(ord(c) % 7) / 7 + j / 10 for j in range(dim)] for c in sequence])
    return (base + shift).astype("<f2")


class _Fake:
    def __init__(self, shift: float = 0.0, wrong_size: bool = False) -> None:
        self.shift = shift
        self.wrong_size = wrong_size
        self.calls: list[str] = []

    def embed_residues(self, sequences: list[str]) -> list[bytes]:
        self.calls.extend(sequences)
        out = []
        for s in sequences:
            arr = _states(s, shift=self.shift)
            out.append((arr[:-1] if self.wrong_size else arr).tobytes())
        return out


def _factory(fake: _Fake) -> Any:
    return lambda spec, cache_dir, device: fake


# -- the module -----------------------------------------------------------------------------


def test_a_matching_encoder_passes_the_self_check_and_returns_cache_shaped_states() -> None:
    fake = _Fake()
    live = load_live_encoder(SPEC, None, "cpu", ("MKTAY", _states("MKTAY")), factory=_factory(fake))
    assert live.self_check_max_abs_diff == 0.0
    out = live.states("ACDEFG")
    assert out.shape == (6, 4)
    assert out.dtype == np.dtype("<f2")
    assert live.describe()["sequences_encoded"] == 1
    assert fake.calls == ["MKTAY", "ACDEFG"]


def test_a_drifting_encoder_stops_startup() -> None:
    drift = _Fake(shift=SELF_CHECK_TOLERANCE * 5)
    with pytest.raises(LiveEncoderError, match="differ from the cached ones"):
        load_live_encoder(SPEC, None, "cpu", ("MKTAY", _states("MKTAY")), factory=_factory(drift))


def test_a_small_difference_is_tolerated_and_recorded() -> None:
    near = _Fake(shift=SELF_CHECK_TOLERANCE / 4)
    live = load_live_encoder(SPEC, None, "cpu", ("MKTAY", _states("MKTAY")), factory=_factory(near))
    assert 0 < (live.self_check_max_abs_diff or 0) <= SELF_CHECK_TOLERANCE


def test_a_wrong_shape_is_refused() -> None:
    with pytest.raises(LiveEncoderError, match="values for"):
        load_live_encoder(
            SPEC, None, "cpu", ("MKTAY", _states("MKTAY")), factory=_factory(_Fake(wrong_size=True))
        )


def test_a_broken_factory_is_a_reason_not_a_crash() -> None:
    def broken(spec: Any, cache_dir: Any, device: str) -> Any:
        raise RuntimeError("CUDA driver missing")

    with pytest.raises(LiveEncoderError, match="CUDA driver missing"):
        load_live_encoder(SPEC, None, "cpu", ("MKTAY", _states("MKTAY")), factory=broken)


def test_without_torch_the_reason_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed here; this case is for a numpy-only host")
    for name in ("HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LiveEncoderError, match="live` extra"):
        torch_factory(SPEC, Path("/nowhere"), "cpu")


def test_states_reports_a_mean_time() -> None:
    live = LiveEncoder(inner=_Fake(), dim=4, checkpoint="c", revision="r")
    live.states("MKT")
    live.states("MKTA")
    described = live.describe()
    assert described["sequences_encoded"] == 2
    assert described["mean_seconds"] is not None


# -- the engine, against the real registry and cache ---------------------------------------------

from test_engine import _NEEDED, GALLERY, REPO, _settings  # noqa: E402

from protloc.inference.engine import EngineRefusal, EngineStartError, load_engine  # noqa: E402

needs_data = pytest.mark.skipif(
    not (all(p.exists() for p in _NEEDED) and GALLERY.is_file()),
    reason="needs the Phase 7-14 data under data/ and the gallery",
)


class _CacheBacked:
    """Cached states for a cached sequence; stand-in states for anything else."""

    def __init__(self, shift: float = 0.0) -> None:
        from test_engine import CACHE  # noqa: PLC0415

        from protloc.data.records import normalize_sequence, sequence_sha256  # noqa: PLC0415
        from protloc.data.residue_cache import ResidueStore  # noqa: PLC0415

        self.store = ResidueStore(CACHE)
        self.sha = lambda s: sequence_sha256(normalize_sequence(s))
        self.shift = shift
        self.donor = next(iter(self.store.lengths))

    def embed_residues(self, sequences: list[str]) -> list[bytes]:
        out = []
        for s in sequences:
            sha = self.sha(s)
            if sha in self.store.lengths:
                arr = np.asarray(self.store.states(sha), dtype=np.float32) + self.shift
            else:
                donor = np.asarray(self.store.states(self.donor), dtype=np.float32)
                rows = np.resize(donor, (len(s), donor.shape[1]))
                arr = rows
            out.append(arr.astype("<f2").tobytes())
        return out


# Synthetic, so certainly not one of the 500 cached sequences.
UNCACHED = "M" + "ACDEFGHIKLMNPQRSTVWY" * 8


@pytest.fixture(scope="module")
def live_engine(tmp_path_factory: pytest.TempPathFactory) -> Any:
    if not (all(p.exists() for p in _NEEDED) and GALLERY.is_file()):
        pytest.skip("needs the Phase 7-14 data under data/ and the gallery")
    fake = _CacheBacked()
    settings = _settings(tmp_path_factory.mktemp("live"), ENCODER_MODE="live")
    return load_engine(settings, REPO, live_factory=lambda spec, cache_dir, device: fake)


@needs_data
def test_live_mode_scores_an_uncached_sequence_and_labels_it(live_engine: Any) -> None:
    assert live_engine.has_states(UNCACHED)
    r = live_engine.predict(UNCACHED, "eukaryote", "novel")
    assert (r["encoder"], r["outside_evaluated_subset"]) == ("live", True)
    assert r["localization"][0]["source"] == "model"
    assert len(r["attribution"]) == len(UNCACHED)
    assert live_engine.live_encoder["sequences_encoded"] >= 1


@needs_data
def test_live_mode_still_serves_cached_sequences_from_the_cache(live_engine: Any) -> None:
    gallery = json.loads(GALLERY.read_text())["examples"][0]
    before = live_engine.live_encoder["sequences_encoded"]
    r = live_engine.predict(gallery["sequence"], gallery["organism_group"], "g")
    assert (r["encoder"], r["outside_evaluated_subset"]) == ("cache", False)
    assert live_engine.live_encoder["sequences_encoded"] == before


@needs_data
def test_live_provenance_names_the_encoder_and_its_self_check(live_engine: Any) -> None:
    enc = live_engine.provenance["encoder"]
    assert enc["mode"] == "live"
    assert enc["self_check_max_abs_diff"] == 0.0
    index = json.loads(
        (
            REPO / "data" / "embeddings" / "0e528d4a46831b05_subset500_residues" / "index.json"
        ).read_text()
    )
    assert enc["revision"] == index["spec"]["states"]["revision"]
    assert "ADR-0019" in live_engine.provenance["predictor"]


@needs_data
def test_a_drifting_live_encoder_stops_the_service(tmp_path: Path) -> None:
    drift = _CacheBacked(shift=SELF_CHECK_TOLERANCE * 10)
    settings = _settings(tmp_path, ENCODER_MODE="live")
    with pytest.raises(EngineStartError, match="differ from the cached ones"):
        load_engine(settings, REPO, live_factory=lambda spec, cache_dir, device: drift)


@needs_data
def test_cache_only_mode_is_unchanged(tmp_path: Path) -> None:
    engine = load_engine(_settings(tmp_path), REPO)
    assert engine.live_encoder is None
    assert engine.provenance["encoder"]["mode"] == "cache_only"
    assert not engine.has_states(UNCACHED)
    with pytest.raises(EngineRefusal) as refused:
        engine.predict(UNCACHED, "eukaryote", "novel")
    assert refused.value.code == "not_in_embedding_cache"


# -- settings ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"), [("", None), ("   ", None), ("data/esm2", "data/esm2")]
)
def test_a_blank_esm_cache_dir_means_unset(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str | None
) -> None:
    from protloc.config.serving import ServingSettings  # noqa: PLC0415

    base = {
        "MODEL_REGISTRY": "r",
        "MODEL_VERSION": "v",
        "MODEL_SERVING_MODE": "queue",
        "MAX_BATCH_SEQUENCES": "1",
        "INLINE_RESULT_WINDOW_SECONDS": "0",
        "JOB_RETENTION_HOURS": "1",
        "DATA_ROOT": "d",
        "EMBED_CACHE": "e",
        "EVIDENCE_ROOT": "x",
        "SWISSPROT_DB_PATH": "s",
        "EVIDENCE_TIMEOUT_SECONDS": "1",
        "EVIDENCE_TOP_K": "1",
        "ESM_CACHE_DIR": raw,
    }
    for key, value in base.items():
        monkeypatch.setenv(key, value)
    settings = ServingSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.ENCODER_MODE == "cache_only"
    got = settings.ESM_CACHE_DIR
    assert (None if got is None else got.as_posix()) == expected
