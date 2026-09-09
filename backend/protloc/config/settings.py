"""Runtime settings, read once from the environment and validated on first use.

Two rules govern this module.

**It fails loudly.** A required variable that is absent or malformed raises at
startup with a message naming the variable and the file it belongs in. A service
that starts with a wrong ``MODEL_VERSION`` and serves predictions is worse than
one that refuses to start, because the second failure is visible.

**It holds only what the project uses today.** The variables below are the
Phase 1 block of ``CLAUDE.md``. Later phases add their own fields in the phase
that first needs them; ``.env.example`` documents the whole schedule, so a
variable named there but absent here belongs to a phase that has not run yet and
is ignored rather than rejected.
"""

from functools import lru_cache
import sys
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = ".env"
_EXAMPLE_FILE = ".env.example"


class Settings(BaseSettings):
    """Configuration for the ProtLoc Fusion backend.

    Values come from the process environment first and from ``.env`` second, so
    a container's injected variables always win over a developer's local file.
    None of these fields carries a default: a value that matters enough to be
    configurable matters enough to be stated.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        # Variables belonging to phases that have not run yet are present in a
        # developer's .env and are not errors. Unknown does not mean invalid.
        extra="ignore",
    )

    APP_ENV: Literal["development", "staging", "production"] = Field(
        description="Deployment environment. Decides log format and error verbosity.",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        description="Root log level for the service.",
    )
    API_HOST: str = Field(
        min_length=1,
        description="Interface the API binds to.",
    )
    API_PORT: int = Field(
        ge=1,
        le=65535,
        description="Port the API listens on.",
    )
    CORS_ALLOWED_ORIGINS: str = Field(
        min_length=1,
        description=(
            "Comma-separated list of origins permitted to call the API. "
            "Read through the cors_allowed_origins property, never split by hand."
        ),
    )

    @field_validator("CORS_ALLOWED_ORIGINS")
    @classmethod
    def _reject_wildcard_origin(cls, value: str) -> str:
        """Reject ``*``.

        The API is unauthenticated and rate-limited by origin; a wildcard turns
        every browser on the internet into a client of this instance. Phase 16
        pins this to the frontend's Space URL. If a wildcard is ever genuinely
        wanted, it is a decision, and decisions become ADRs.
        """
        if "*" in value:
            message = (
                "CORS_ALLOWED_ORIGINS must list explicit origins; '*' is not accepted. "
                "Set it to the frontend origin, e.g. http://localhost:5173"
            )
            raise ValueError(message)
        return value

    @property
    def cors_allowed_origins(self) -> tuple[str, ...]:
        """The allowed origins, split and stripped."""
        return tuple(
            origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()
        )

    @property
    def is_production(self) -> bool:
        """Whether this process is serving production traffic."""
        return self.APP_ENV == "production"


class SettingsError(RuntimeError):
    """Raised when the environment cannot produce a valid :class:`Settings`.

    Carries the full pydantic report rather than the first failing field, so one
    restart surfaces every problem instead of one per attempt.
    """


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call.

    Raises:
        SettingsError: If any required variable is missing or invalid. The
            message names each offending variable and points at ``.env.example``.
    """
    try:
        # Fields are populated from the environment and .env; the pydantic mypy
        # plugin (configured in pyproject.toml) understands this call site.
        return Settings()
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        message = (
            f"Configuration is invalid; the service will not start.\n"
            f"{problems}\n"
            f"Copy {_EXAMPLE_FILE} to {_ENV_FILE} and fill the variables for the "
            f"current phase, or set them in the process environment."
        )
        raise SettingsError(message) from exc


def main() -> int:
    """Validate the environment and print the resolved non-secret settings.

    Run as ``python -m protloc.config.settings``. Exits non-zero with the
    validation report when the environment is incomplete, which makes it usable
    as a container health gate and as a pre-deploy check.

    Returns:
        ``0`` when the configuration is valid, ``1`` otherwise.
    """
    try:
        settings = get_settings()
    except SettingsError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(
        f"APP_ENV={settings.APP_ENV}\n"
        f"LOG_LEVEL={settings.LOG_LEVEL}\n"
        f"API_HOST={settings.API_HOST}\n"
        f"API_PORT={settings.API_PORT}\n"
        f"CORS_ALLOWED_ORIGINS={','.join(settings.cors_allowed_origins)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
