"""``GET /demo/structure/{sha256}`` serves a verified backbone, or says why not."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from fastapi.testclient import TestClient

from protloc.api import app_v2
from protloc.inference import structures

DIGEST = "ab" * 32
DOC = {
    "accession": "P12345",
    "source": "AlphaFold DB prediction, not an experimental structure",
    "license": "CC-BY-4.0",
    "length": 2,
    "ca": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
    "plddt": [90.0, 80.0],
    "sequence_sha256": DIGEST,
}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    root = tmp_path / "structures"
    root.mkdir()
    raw = (json.dumps(DOC) + "\n").encode()
    (root / f"{DIGEST}.json").write_bytes(raw)
    manifest = {"files_sha256": {DIGEST: hashlib.sha256(raw).hexdigest()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(app_v2, "get_serving_settings", lambda: SimpleNamespace(DATA_ROOT=tmp_path))
    structures.store_for.cache_clear()
    return TestClient(app_v2.app)


def test_serves_the_backbone_with_its_provenance(client: TestClient) -> None:
    response = client.get(f"/demo/structure/{DIGEST}")
    assert response.status_code == 200
    body = response.json()
    assert body["accession"] == "P12345"
    assert body["ca"] == DOC["ca"]
    assert "not an experimental structure" in body["note"]


def test_unknown_sequence_is_404_with_a_reason(client: TestClient) -> None:
    response = client.get(f"/demo/structure/{'cd' * 32}")
    assert response.status_code == 404
    assert "No structure for this sequence" in response.json()["detail"]


def test_malformed_hash_is_400(client: TestClient) -> None:
    assert client.get("/demo/structure/not-a-hash").status_code == 400
