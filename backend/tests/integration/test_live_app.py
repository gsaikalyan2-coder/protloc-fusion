"""The live app's startup and its 202-then-poll round trip (Phase 14).

The startup test needs no data: pointed at an empty registry the app must not come up. The
round-trip test needs the Phase 7-13 data and a gallery (``scripts/build_gallery.py --write``)
and skips without them.
"""

import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("numpy")

from protloc.api.app_v2 import app
from protloc.config.serving import get_serving_settings
from protloc.inference.engine import EngineStartError, composite_version

REPO = Path(__file__).resolve().parents[3]
ENGINE_JSON = json.loads((REPO / "configs" / "serving" / "engine.json").read_text())
GALLERY = REPO / "data" / "gallery" / "gallery.json"
POLL_SECONDS = 0.1
POLL_LIMIT = 300


def _environment(monkeypatch: pytest.MonkeyPatch, registry: Path, tmp: Path) -> None:
    values = {
        "MODEL_REGISTRY": str(registry),
        "MODEL_VERSION": composite_version(ENGINE_JSON["bundles"]),
        "MODEL_SERVING_MODE": "queue",
        "MAX_BATCH_SEQUENCES": "10",
        "INLINE_RESULT_WINDOW_SECONDS": "2",
        "JOB_RETENTION_HOURS": "1",
        "DATA_ROOT": str(REPO / "data"),
        "EMBED_CACHE": str(REPO / "data" / "embeddings"),
        "EVIDENCE_ROOT": str(tmp / "evidence"),
        "SWISSPROT_DB_PATH": str(tmp / "evidence" / "swissprot_mmseqs"),
        "EVIDENCE_TIMEOUT_SECONDS": "5",
        "EVIDENCE_TOP_K": "5",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_serving_settings.cache_clear()


def test_the_app_does_not_start_without_a_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch, tmp_path / "empty_registry", tmp_path)
    with pytest.raises(EngineStartError), TestClient(app):
        pass
    get_serving_settings.cache_clear()


@pytest.mark.skipif(not GALLERY.is_file(), reason="needs data/gallery/gallery.json")
def test_a_gallery_sequence_round_trips_through_the_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch, REPO / "registry", tmp_path)
    example = json.loads(GALLERY.read_text())["examples"][0]
    with TestClient(app) as client:
        accepted = client.post(
            "/demo/predict",
            json={"sequence": example["sequence"], "organism_group": example["organism_group"]},
        )
        assert accepted.status_code == 202
        body = accepted.json()
        assert body["status"] == "queued"
        assert body["model_version"] == composite_version(ENGINE_JSON["bundles"])
        for _ in range(POLL_LIMIT):
            job = client.get(body["result_url"]).json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(POLL_SECONDS)
        assert job["status"] == "completed", job.get("error")
        assert job["started_at"] >= job["created_at"]
        assert job["finished_at"] >= job["started_at"]
        result = job["results"][0]
        assert result["task_sources"]["localization"]["source"] == "model"
        assert job["model_version"] == body["model_version"]
        assert client.get("/demo/version").json()["tau"] is None
    get_serving_settings.cache_clear()
