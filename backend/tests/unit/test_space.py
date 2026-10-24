"""The protein-space map: served only when verified, and built deterministically."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from protloc.api import app_v2
from protloc.inference import service, space

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = {"schema_version": 1, "method": "pca", "points": [{"sha256": "ab" * 32, "xyz": [0, 0, 0]}]}


def _write(root: Path, body: bytes, digest: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / space.SPACE_FILE).write_bytes(body)
    recorded = digest or hashlib.sha256(body).hexdigest()
    (root / space.MANIFEST_FILE).write_text(json.dumps({"files_sha256": {"space.json": recorded}}))


def test_store_returns_none_when_not_built(tmp_path: Path) -> None:
    assert space.SpaceStore(tmp_path / "space").get() is None


def test_store_serves_a_matching_file(tmp_path: Path) -> None:
    _write(tmp_path, json.dumps(DOC).encode())
    assert space.SpaceStore(tmp_path).get() == DOC


def test_store_refuses_a_tampered_file(tmp_path: Path) -> None:
    _write(tmp_path, json.dumps(DOC).encode(), digest="0" * 64)
    with pytest.raises(space.SpaceError):
        space.SpaceStore(tmp_path).get()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_v2, "get_serving_settings", lambda: SimpleNamespace(DATA_ROOT=tmp_path))
    monkeypatch.setattr(service, "provenance", lambda: {"model_version": "test"})
    space.store_for.cache_clear()
    return TestClient(app_v2.app)


def test_route_serves_the_map_with_provenance(client: TestClient, tmp_path: Path) -> None:
    _write(tmp_path / "space", json.dumps(DOC).encode())
    response = client.get("/demo/space")
    assert response.status_code == 200
    assert response.json()["points"] == DOC["points"]
    assert response.json()["provenance"] == {"model_version": "test"}


def test_route_says_how_to_build_a_missing_map(client: TestClient) -> None:
    response = client.get("/demo/space")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "space_not_built"
    assert "build_space.py" in response.json()["detail"]


def test_route_refuses_a_tampered_map(client: TestClient, tmp_path: Path) -> None:
    _write(tmp_path / "space", json.dumps(DOC).encode(), digest="0" * 64)
    response = client.get("/demo/space")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "space_refused"


def _builder():  # noqa: ANN202 - a module loaded from scripts/
    pytest.importorskip("numpy")
    path = REPO_ROOT / "scripts" / "build_space.py"
    spec = importlib.util.spec_from_file_location("build_space", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pca_is_deterministic_and_sign_fixed() -> None:
    build = _builder()
    np = build.np
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 12))
    first, ratio = build.pca(matrix, 3)
    flipped, _ = build.pca(-matrix, 3)
    assert np.allclose(np.abs(first), np.abs(flipped))
    assert ratio.shape == (3,)
    assert np.all(ratio[:-1] >= ratio[1:])
    again, _ = build.pca(matrix, 3)
    assert np.array_equal(first, again)


def test_neighbours_exclude_self_and_are_sorted() -> None:
    build = _builder()
    np = build.np
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.2]])
    order, similarity = build.neighbours(matrix, 2)
    assert all(i not in row for i, row in enumerate(order))
    assert np.all(similarity[:, 0] >= similarity[:, 1])
