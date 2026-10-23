"""Serving settings (Phase 14): where the registry, the caches and the evidence index are.

Separate from :mod:`protloc.config.settings` so the Phase 1 block stays exactly what it was.
Like it, this fails loudly and no field has a default: a service that guessed where its model
lives could serve a different model than the one it reports. Values come from the process
environment first and ``.env`` second; ``configs/serving/engine.json`` holds the data that is
not an environment choice (bundle versions, tag set, forbidden transitions).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    """The ``Phase 14: serving`` and ``Phase 13: evidence`` blocks of ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8-sig", case_sensitive=True, extra="ignore"
    )

    MODEL_REGISTRY: Path = Field(description="Directory holding registry/<task>/<version>/.")
    MODEL_VERSION: str = Field(
        min_length=1,
        description=(
            "The composite version this service claims, e.g. "
            "localization-0.10.0+signal_peptide-0.11.0+go_function-0.12.0"
            "+evidence_homology-0.13.0. "
            "It must equal the version derived from the bundles actually loaded, or the "
            "service refuses to start."
        ),
    )
    MODEL_SERVING_MODE: Literal["queue"] = Field(
        description="ADR-0001: 'queue' is the only supported mode.",
    )
    MAX_BATCH_SEQUENCES: int = Field(ge=1, description="Sequences accepted in one batch job.")
    INLINE_RESULT_WINDOW_SECONDS: int = Field(
        ge=0,
        description="Reserved size-one fast path (ADR-0001); the demo API always answers 202.",
    )
    JOB_RETENTION_HOURS: int = Field(ge=1, description="How long a finished job stays readable.")
    DATA_ROOT: Path = Field(description="Root of the data tree (manifests, GO files, gallery).")
    EMBED_CACHE: Path = Field(description="Directory holding the embedding caches.")
    EVIDENCE_ROOT: Path = Field(description="Root of the evidence tree (index, binaries, terms).")
    SWISSPROT_DB_PATH: Path = Field(description="Directory of the MMseqs2 Swiss-Prot index.")
    EVIDENCE_TIMEOUT_SECONDS: int = Field(ge=1, description="Wall-clock budget of one search.")
    EVIDENCE_TOP_K: int = Field(ge=1, description="Hits shown per sequence.")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description=(
            "Level of the JSON logs (protloc.access, protloc.jobs). The one field with a "
            "default: it changes what is logged, never which model is served. Job polls "
            "(GET /demo/jobs/{job_id} answering 200) log at DEBUG."
        ),
    )


class ServingSettingsError(RuntimeError):
    """The environment cannot produce valid serving settings."""


@lru_cache(maxsize=1)
def get_serving_settings() -> ServingSettings:
    """Load the process-wide serving settings; raise one error naming every bad variable."""
    try:
        return ServingSettings()
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(part) for part in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        message = (
            "Serving configuration is invalid; the service will not start.\n"
            f"{problems}\n"
            "Fill the Phase 13 and Phase 14 blocks of .env.example in .env."
        )
        raise ServingSettingsError(message) from exc
