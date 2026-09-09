"""Settings refuse to load rather than defaulting silently.

The requirement in ``PHASE_01_scaffold.md`` §4 is that a missing required
variable stops the service. These tests assert the refusal, not just the happy
path, because a configuration layer is only useful for the case it rejects.
"""

from pathlib import Path

import pytest

from protloc.config.settings import Settings, SettingsError, get_settings

VALID_ENV = {
    "APP_ENV": "development",
    "LOG_LEVEL": "INFO",
    "API_HOST": "0.0.0.0",  # noqa: S104 — binding all interfaces is the container default
    "API_PORT": "8000",
    "CORS_ALLOWED_ORIGINS": "http://localhost:5173",
}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test against an empty environment and no .env file on disk."""
    for key in VALID_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def _set(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {**VALID_ENV, **overrides}
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_a_complete_environment_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch)
    settings = Settings()
    assert settings.APP_ENV == "development"
    assert settings.API_PORT == 8000
    assert settings.cors_allowed_origins == ("http://localhost:5173",)
    assert settings.is_production is False


def test_a_missing_variable_raises_and_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch)
    monkeypatch.delenv("API_PORT")
    with pytest.raises(SettingsError) as excinfo:
        get_settings()
    assert "API_PORT" in str(excinfo.value)
    assert ".env.example" in str(excinfo.value)


def test_every_required_variable_is_actually_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """No field quietly acquired a default."""
    for missing in VALID_ENV:
        _set(monkeypatch)
        monkeypatch.delenv(missing)
        get_settings.cache_clear()
        with pytest.raises(SettingsError, match=missing):
            get_settings()


def test_an_out_of_range_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, API_PORT="70000")
    with pytest.raises(SettingsError, match="API_PORT"):
        get_settings()


def test_an_unknown_app_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, APP_ENV="prod")
    with pytest.raises(SettingsError, match="APP_ENV"):
        get_settings()


def test_a_wildcard_cors_origin_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, CORS_ALLOWED_ORIGINS="*")
    with pytest.raises(SettingsError, match="CORS_ALLOWED_ORIGINS"):
        get_settings()


def test_multiple_origins_are_split_and_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, CORS_ALLOWED_ORIGINS="http://localhost:5173, https://example.org")
    assert Settings().cors_allowed_origins == ("http://localhost:5173", "https://example.org")


def test_variables_from_later_phases_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A .env carrying the full schedule is valid at Phase 1."""
    _set(monkeypatch)
    monkeypatch.setenv("GOA_RELEASE", "not-yet-used")
    monkeypatch.setenv("WANDB_PROJECT", "protloc-fusion")
    assert Settings().APP_ENV == "development"
