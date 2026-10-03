"""The embedding manifest: what the Phase 7 cache is, and the pins that identify it.

``data/manifests/embedding_manifest.json`` is written once by
``scripts/embed.py --freeze`` from measured inputs only. ``--check`` needs no cache
bytes, so CI can run it: the pins in ``configs/model/trunk_esm2_150m.yaml`` and
``.env.example`` must agree with each other, and with the manifest once it exists.
"""

from dataclasses import dataclass
import re
from typing import Any

from protloc.data.embeddings import (
    DTYPE,
    LONG_SEQUENCE_POLICY,
    MAX_RESIDUES,
    POOLING,
    WINDOW_STRIDE,
    CacheSpec,
    EmbeddingError,
)

_REVISION = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_LINE = re.compile(r"^\s*checkpoint:\s*\$\{oc\.env:ESM_CHECKPOINT,([^}]+)\}\s*$", re.M)
_REVISION_LINE = re.compile(r"^\s*revision:\s*(\S+)\s*$", re.M)
_DIM_LINE = re.compile(r"^\s*embedding_dim:\s*(\S+)\s*$", re.M)
_ENV_CHECKPOINT = re.compile(r"^ESM_CHECKPOINT=(\S+)\s*$", re.M)
_ENV_REVISION = re.compile(r"^ESM_REVISION=(\S+)\s*$", re.M)


class EmbeddingManifestError(EmbeddingError):
    """The embedding manifest or the pins are inconsistent."""


def _one(pattern: re.Pattern[str], text: str, what: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise EmbeddingManifestError(f"{what} not found")
    return match.group(1)


def read_pins(config_text: str, env_text: str) -> CacheSpec:
    """The spec stated by the model config, after checking ``.env.example`` agrees.

    Raises if the revision is not 40 hex characters, the dimension is not an integer,
    or the config and ``.env.example`` name different checkpoints or revisions.
    """
    checkpoint = _one(_CHECKPOINT_LINE, config_text, "trunk.checkpoint")
    revision = _one(_REVISION_LINE, config_text, "trunk.revision")
    dim_text = _one(_DIM_LINE, config_text, "trunk.embedding_dim")
    if not _REVISION.fullmatch(revision):
        raise EmbeddingManifestError(f"trunk.revision is not a commit hash: {revision!r}")
    if not dim_text.isdigit():
        raise EmbeddingManifestError(f"trunk.embedding_dim is not an integer: {dim_text!r}")
    if _one(_ENV_CHECKPOINT, env_text, "ESM_CHECKPOINT") != checkpoint:
        raise EmbeddingManifestError(".env.example ESM_CHECKPOINT differs from the config")
    if _one(_ENV_REVISION, env_text, "ESM_REVISION") != revision:
        raise EmbeddingManifestError(".env.example ESM_REVISION differs from the config")
    spec = CacheSpec(checkpoint, revision, int(dim_text))
    spec.validate()
    return spec


@dataclass(frozen=True, slots=True)
class Measured:
    """The five measured documents a manifest is built from."""

    facts: dict[str, Any]
    measure: dict[str, Any]
    extract_run: dict[str, Any]
    index: dict[str, Any]
    verify: dict[str, Any]


def build_manifest(spec: CacheSpec, registry_sha256: str, measured: Measured) -> dict[str, object]:
    """The manifest, from measured inputs only. Raises if they disagree."""
    facts, measure = measured.facts, measured.measure
    extract_run, index, verify = measured.extract_run, measured.index, measured.verify
    key = spec.key()
    for name, document in (("index", index), ("measure", measure), ("extract_run", extract_run)):
        if document.get("cache_key") != key:
            raise EmbeddingManifestError(f"{name} was produced under a different cache key")
    if facts.get("registry_sha256") != registry_sha256:
        raise EmbeddingManifestError("facts were computed on a different registry")
    if index["sequence_count"] != facts["distinct_sequences"]:
        raise EmbeddingManifestError(
            "cache holds a different number of sequences than the registry"
        )
    if verify.get("byte_identical_across_processes") is not True:
        raise EmbeddingManifestError("byte identity across processes is not proven")
    if verify.get("byte_identical_after_kill_and_resume") is not True:
        raise EmbeddingManifestError("byte identity after kill and resume is not proven")
    return {
        "schema_version": 1,
        "cache_key": key,
        "spec": spec.as_dict(),
        "registry_sha256": registry_sha256,
        "sequences": {
            "accessions": facts["accessions"],
            "distinct": facts["distinct_sequences"],
            "windowed": facts["windowed_sequences"],
            "excluded": 0,
            "long_sequence_policy": LONG_SEQUENCE_POLICY,
        },
        "cache": {
            "bytes": extract_run["cache_bytes"],
            "shard_count": index["shard_count"],
            "shard_index_sha256": index["shard_index_sha256"],
            "seconds_per_sequence_full_run": extract_run["seconds_per_sequence"],
        },
        "measure_200": measure,
        "verify": verify,
        "hardware": extract_run["hardware"],
        "per_residue_storage_estimate_bytes": facts["per_residue_float16_bytes"],
    }


def check_documents(manifest: dict[str, Any] | None, config_text: str, env_text: str) -> list[str]:
    """Problems found without any cache bytes; empty when everything agrees."""
    problems: list[str] = []
    try:
        spec = read_pins(config_text, env_text)
    except EmbeddingError as error:
        return [str(error)]
    fixed = (
        ("pooling", POOLING),
        ("dtype", DTYPE),
        ("max_residues", MAX_RESIDUES),
        ("window_stride", WINDOW_STRIDE),
        ("long_sequence_policy", LONG_SEQUENCE_POLICY),
    )
    if manifest is None:
        return problems
    if manifest.get("cache_key") != spec.key():
        problems.append("manifest cache_key differs from the key of the configured spec")
    recorded = manifest.get("spec")
    if recorded != spec.as_dict():
        problems.append("manifest spec differs from the configured spec")
    if isinstance(recorded, dict):
        problems.extend(
            f"manifest {name} is {recorded.get(name)!r}, expected {value!r}"
            for name, value in fixed
            if recorded.get(name) != value
        )
    return problems
