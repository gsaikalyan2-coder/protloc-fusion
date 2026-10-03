"""The embedding pins and manifest agree, and a manifest cannot be built from a mismatch."""

from pathlib import Path

import pytest

from protloc.data.embedding_manifest import (
    EmbeddingManifestError,
    Measured,
    build_manifest,
    check_documents,
    read_pins,
)
from protloc.data.embeddings import CacheSpec

ROOT = Path(__file__).resolve().parents[3]
REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
CHECKPOINT = "facebook/esm2_t30_150M_UR50D"
CONFIG = f"""trunk:
  checkpoint: ${{oc.env:ESM_CHECKPOINT,{CHECKPOINT}}}
  revision: {REVISION}
  embedding_dim: 640
"""
ENV = f"ESM_CHECKPOINT={CHECKPOINT}\nESM_REVISION={REVISION}\n"


def test_the_repository_pins_are_consistent_and_real() -> None:
    spec = read_pins(
        (ROOT / "configs/model/trunk_esm2_150m.yaml").read_text(encoding="utf-8"),
        (ROOT / ".env.example").read_text(encoding="utf-8"),
    )
    assert spec.checkpoint == "facebook/esm2_t30_150M_UR50D"
    assert spec.embedding_dim == 640
    assert len(spec.revision) == 40


def test_pins_are_read_from_config_and_env() -> None:
    assert read_pins(CONFIG, ENV) == CacheSpec(CHECKPOINT, REVISION, 640)


@pytest.mark.parametrize(
    ("config", "env"),
    [
        (CONFIG.replace(REVISION, "???"), ENV),
        (CONFIG.replace("640", "null"), ENV),
        (CONFIG, ENV.replace(REVISION, "0" * 40)),
        (CONFIG, ENV.replace("t30_150M", "t33_650M")),
        (CONFIG, "ESM_CHECKPOINT=x\n"),
    ],
)
def test_inconsistent_pins_are_rejected(config: str, env: str) -> None:
    with pytest.raises(EmbeddingManifestError):
        read_pins(config, env)


def _measured(key: str, sequences: int = 3) -> Measured:
    keyed = {"cache_key": key, "input": {"source": "pack", "registry_sha256": "r" * 64}}
    return Measured(
        facts={
            "registry_sha256": "r" * 64,
            "accessions": 4,
            "distinct_sequences": sequences,
            "windowed_sequences": 1,
            "per_residue_float16_bytes": 10,
        },
        measure=dict(keyed),
        extract_run={
            **keyed,
            "cache_bytes": 99,
            "seconds_per_sequence": 0.5,
            "hardware": {"device": "cpu"},
        },
        index={**keyed, "sequence_count": 3, "shard_count": 1, "shard_index_sha256": "s" * 64},
        verify={
            **keyed,
            "byte_identical_across_processes": True,
            "byte_identical_after_kill_and_resume": True,
        },
    )


def test_a_manifest_is_built_from_agreeing_inputs() -> None:
    spec = read_pins(CONFIG, ENV)
    manifest = build_manifest(spec, "r" * 64, _measured(spec.key()))
    assert manifest["cache_key"] == spec.key()
    assert check_documents(manifest, CONFIG, ENV) == []


def test_a_manifest_is_refused_when_inputs_disagree() -> None:
    spec = read_pins(CONFIG, ENV)
    with pytest.raises(EmbeddingManifestError):
        build_manifest(spec, "r" * 64, _measured("0" * 64))
    with pytest.raises(EmbeddingManifestError):
        build_manifest(spec, "x" * 64, _measured(spec.key()))
    with pytest.raises(EmbeddingManifestError):
        build_manifest(spec, "r" * 64, _measured(spec.key(), sequences=4))


def test_a_run_on_another_registry_is_refused() -> None:
    """A pack built from a different registry must not be frozen into this manifest."""
    spec = read_pins(CONFIG, ENV)
    measured = _measured(spec.key())
    measured.extract_run["input"] = {"source": "pack", "registry_sha256": "x" * 64}
    with pytest.raises(EmbeddingManifestError):
        build_manifest(spec, "r" * 64, measured)


def test_a_manifest_for_another_revision_fails_the_check() -> None:
    spec = read_pins(CONFIG, ENV)
    manifest = build_manifest(spec, "r" * 64, _measured(spec.key()))
    other = CONFIG.replace(REVISION, "1" * 40)
    problems = check_documents(manifest, other, ENV.replace(REVISION, "1" * 40))
    assert problems
    assert check_documents(None, CONFIG, ENV) == []
