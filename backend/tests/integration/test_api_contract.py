"""The application, exercised over HTTP.

``test_contracts.py`` checks the shapes. This file checks that the running
service actually produces them: that the envelope survives serialisation, that a
refusal is a refusal and not a 500, and that the endpoints the contract names are
the endpoints the app mounts.

Every test builds the app with an explicit ``Settings`` rather than reading the
environment, so the suite passes on a machine with no ``.env`` — which is what CI
is, and what a fresh clone is.
"""

from collections.abc import Iterator
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from protloc.api.app import create_app
from protloc.config.settings import Settings
from protloc.domain.contracts import API_VERSION
from protloc.domain.ports import ExampleGallery, JobService, ProvenanceSource
from protloc.inference.provenance import UnavailableJobService, UnloadedProvenance

ENVELOPE_FIELDS = ("model_version", "config_hash", "rules_version", "experiment_id")

# Every path the API map in CLAUDE.md names as part of the contract. /docs is
# absent deliberately: Swagger UI is a rendering of the contract, not part of it.
CONTRACT_PATHS = {
    ("GET", "/health"),
    ("GET", "/healthz"),
    ("GET", "/version"),
    ("POST", "/predict"),
    ("POST", "/batch"),
    ("GET", "/jobs/{job_id}"),
    ("POST", "/jobs/{job_id}/cancel"),
    ("GET", "/jobs/{job_id}/results"),
    ("GET", "/examples"),
}


@pytest.fixture(name="app")
def app_fixture() -> FastAPI:
    """An application built against explicit settings, not the environment."""
    return create_app(
        Settings(
            APP_ENV="development",
            LOG_LEVEL="INFO",
            API_HOST="127.0.0.1",
            API_PORT=8000,
            CORS_ALLOWED_ORIGINS="http://localhost:5173",
        )
    )


@pytest.fixture(name="client")
def client_fixture(app: FastAPI) -> Iterator[TestClient]:
    """A client that does not raise on 4xx and 5xx, since those are under test."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ── The implementations satisfy the ports ───────────────────────────────────


def test_unloaded_provenance_satisfies_its_port():
    """Structural conformance, asserted rather than assumed.

    Nothing imports the protocol to implement it, so a method renamed on one side
    would otherwise be found by the first caller in Phase 7 rather than here.
    """
    assert isinstance(UnloadedProvenance(), ProvenanceSource)


def test_the_job_service_satisfies_both_of_its_ports():
    service = UnavailableJobService()
    assert isinstance(service, JobService)
    assert isinstance(service, ExampleGallery)


# ── The endpoints the contract names ────────────────────────────────────────


def _documented(app: FastAPI) -> set[tuple[str, str]]:
    """Every (METHOD, path) the generated document declares.

    Read from app.openapi() rather than app.routes: the document is the frozen
    artifact a client reads, and FastAPI's internal route objects have changed
    shape between versions. Testing the document tests the deliverable.
    """
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }


def test_every_contract_endpoint_is_documented(app: FastAPI):
    """The document declares exactly the contract, with nothing missing."""
    missing = CONTRACT_PATHS - _documented(app)
    assert not missing, f"endpoints in the API map but not in the document: {sorted(missing)}"


def test_nothing_undocumented_is_served(app: FastAPI):
    """And nothing beyond it. An endpoint outside the API map is a contract change."""
    extra = _documented(app) - CONTRACT_PATHS
    assert not extra, (
        f"endpoints served but not in CLAUDE.md's API map: {sorted(extra)}. "
        f"Adding one means amending the API map in the same commit."
    )


def test_there_is_no_synchronous_prediction_path(app: FastAPI):
    """ADR-0001, asserted against the document rather than trusted."""
    paths = {path for _, path in _documented(app)}
    assert "/predict/sync" not in paths
    assert "/predictions" not in paths


# ── Health and version are live ─────────────────────────────────────────────


def test_health_is_200_and_degraded(client: TestClient):
    """Serving and unable to predict is a 200 with status degraded.

    Not a 503: a readiness probe that fails while the service correctly serves
    its contract would restart a container that is working.
    """
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["bundle_loaded"] is False
    assert body["queue_depth"] == 0
    assert body["detail"]


def test_healthz_reports_the_registry_state(client: TestClient):
    """/healthz is a 200 that names the registry's state in the body."""
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["registry_state"] == "empty"
    assert body["detail"]
    for field in ENVELOPE_FIELDS:
        assert body[field] is None


def test_version_reports_the_contract_and_the_package(client: TestClient):
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == API_VERSION
    assert body["service_version"]
    assert body["checksums"] == {}


# ── The refusals are real, typed and honest ─────────────────────────────────


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/predict", {"json": {"sequence": "MKTFF", "organism_group": "eukarya"}}),
        ("get", f"/jobs/{uuid4()}", {}),
        ("post", f"/jobs/{uuid4()}/cancel", {}),
        ("get", f"/jobs/{uuid4()}/results", {}),
        ("get", "/examples", {}),
    ],
)
def test_endpoints_needing_a_bundle_refuse_with_503(
    client: TestClient, method: str, path: str, kwargs: dict[str, object]
):
    """A missing dependency is a 503 naming itself, never a 500 and never a mock.

    This test asserts the *reason*, not only the code. When Phase 7 packages a
    bundle these assertions fail loudly and are rewritten, rather than passing on
    a service that has quietly stopped being honest about what it can do.
    """
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "no_registry_bundle"
    assert "Phase 7" in body["detail"]


def test_batch_refuses_with_503(client: TestClient):
    """The multipart path refuses for the same reason as the JSON one."""
    response = client.post(
        "/batch",
        data={"organism_group": "eukarya"},
        files={"file": ("p.fasta", b">p\nMKTFF\n", "text/plain")},
    )
    assert response.status_code == 503
    assert response.json()["error"] == "no_registry_bundle"


# ── The envelope is on everything ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/health", {}),
        ("get", "/healthz", {}),
        ("get", "/version", {}),
        ("get", "/examples", {}),
        ("post", "/predict", {"json": {"sequence": "MKTFF", "organism_group": "eukarya"}}),
        ("post", "/predict", {"json": {"nonsense": True}}),
        ("get", "/nothing-here", {}),
    ],
)
def test_every_response_carries_the_envelope(
    client: TestClient, method: str, path: str, kwargs: dict[str, object]
):
    """Contract rule 6, over the wire.

    The list deliberately mixes successes, a refusal, a validation failure and a
    404 — the guarantee is worth nothing if it holds only on the happy path.
    """
    body = getattr(client, method)(path, **kwargs).json()
    for field in ENVELOPE_FIELDS:
        assert field in body, f"{method.upper()} {path} is missing {field}"
        assert body[field] is None, "no bundle is loaded, so every field must be null"


def test_a_404_is_typed_not_fastapis_default(client: TestClient):
    """FastAPI's default {"detail": ...} would be the one shape carrying no provenance."""
    body = client.get("/nothing-here").json()
    assert body["error"] == "not_found"
    assert "detail" in body


def test_validation_failure_names_the_field(client: TestClient):
    response = client.post("/predict", json={"sequence": "MKTFF"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_failed"
    assert "organism_group" in body["detail"]


def test_an_unknown_field_is_rejected(client: TestClient):
    """extra=forbid over the wire: a client's typo is an error, not a silent drop."""
    response = client.post(
        "/predict",
        json={"sequence": "MKTFF", "organism_group": "eukarya", "organism": "human"},
    )
    assert response.status_code == 422


def test_a_malformed_job_id_is_a_422_not_a_500(client: TestClient):
    """job_id is a UUIDv4; anything else fails validation before reaching the service."""
    assert client.get("/jobs/not-a-uuid").status_code == 422


# ── The document the app produces ───────────────────────────────────────────


def test_the_app_emits_openapi_31(app: FastAPI):
    """3.1, not 3.0: the phase's deliverable names the version."""
    assert app.openapi()["openapi"].startswith("3.1")


def test_predict_is_asynchronous_in_the_document(app: FastAPI):
    """202 is declared and no 200 prediction body is.

    This is the assertion that stops someone reintroducing a synchronous path by
    accident — the thing ADR-0001 exists to prevent, checked in the artifact a
    client actually reads.
    """
    responses = app.openapi()["paths"]["/predict"]["post"]["responses"]
    assert "202" in responses
    assert "200" not in responses


def test_every_documented_response_has_a_schema(app: FastAPI):
    """A bare {} in the document is a shape a reviewer cannot predict."""
    bare: list[str] = []
    for path, operations in app.openapi()["paths"].items():
        for method, operation in operations.items():
            for code, response in operation.get("responses", {}).items():
                content = response.get("content")
                if content is None:
                    continue
                for media, media_type in content.items():
                    # /jobs/{id}/results declares two media types whose bodies are
                    # produced by the download path; they are described in prose.
                    if media_type.get("schema") is None and "results" not in path:
                        bare.append(f"{method.upper()} {path} {code} {media}")
    assert not bare, f"responses with no schema: {sorted(bare)}"
