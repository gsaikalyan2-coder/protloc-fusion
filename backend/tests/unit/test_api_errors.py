"""Every failure path of the live app answers with a stable code (Phase 17).

The routes under test refuse before anything is queued, so a stub engine is enough: no
registry, cache or index is read.
"""

from collections.abc import Iterator
import time
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("numpy")

from protloc.api import app_v2
from protloc.domain.errors import MAX_REQUEST_CHARS, MAX_SEQUENCE_LENGTH
from protloc.inference import service
from protloc.inference.engine import EngineRefusal
from protloc.jobs import queue as jobs


class StubEngine:
    """Just enough of the engine for the refusal paths."""

    def __init__(self) -> None:
        """Set the attributes the routes read."""
        self.groups = ("eukaryote",)
        self.cached_sequences = 500
        self.provenance = {"model_version": "test", "config_hash": "c", "rules_version": "r"}
        self.evidence_state = {
            "available": False,
            "reason": "index not loaded",
            "swissprot_release": "x",
        }

    def has_states(self, _sequence: str) -> bool:
        """Nothing is cached: every sequence is cold."""
        return False


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    service.configure(StubEngine())  # type: ignore[arg-type]
    monkeypatch.setattr(
        app_v2,
        "get_serving_settings",
        lambda: SimpleNamespace(MAX_BATCH_SEQUENCES=2, DATA_ROOT="."),
    )
    yield TestClient(app_v2.app)
    service.configure(None)


def _code(response: Any) -> str:
    return str(response.json()["error"]["code"])


def test_an_oversized_sequence_is_refused_before_queueing(client: TestClient) -> None:
    r = client.post("/demo/predict", json={"sequence": "A" * (MAX_SEQUENCE_LENGTH + 1)})
    assert (r.status_code, _code(r)) == (422, "sequence_too_long")
    assert str(MAX_SEQUENCE_LENGTH) in r.json()["detail"]
    assert r.json()["error"]["retryable"] is False


def test_a_payload_beyond_the_backstop_is_sequence_too_long(client: TestClient) -> None:
    r = client.post("/demo/predict", json={"sequence": "A" * (MAX_REQUEST_CHARS + 1)})
    assert (r.status_code, _code(r)) == (422, "sequence_too_long")


def test_an_oversized_fasta_body_is_413(client: TestClient) -> None:
    r = client.post("/demo/batch", json={"fasta": ">a\n" + "A" * MAX_REQUEST_CHARS})
    assert (r.status_code, _code(r)) == (413, "request_too_large")


def test_the_scanner_has_the_same_length_limit(client: TestClient) -> None:
    r = client.post("/demo/scan", json={"sequence": "A" * (MAX_SEQUENCE_LENGTH + 1)})
    assert _code(r) in {"sequence_too_long", "invalid_input"}


def test_an_empty_sequence_is_invalid_input_and_keeps_the_detail_list(client: TestClient) -> None:
    r = client.post("/demo/predict", json={"sequence": ""})
    assert (r.status_code, _code(r)) == (422, "invalid_input")
    assert isinstance(r.json()["detail"], list)


def test_a_cold_cache_sequence_is_not_served_and_never_guessed(client: TestClient) -> None:
    r = client.post("/demo/predict", json={"sequence": "M" * 60, "organism_group": "eukaryote"})
    assert (r.status_code, _code(r)) == (422, "not_cached")
    assert "ESM-2 states are cached" in r.json()["detail"]
    assert "job_id" not in r.json()


def test_an_unserved_group_is_group_not_served(client: TestClient) -> None:
    other = next(g for g in service.ORGANISM_GROUPS if g not in StubEngine().groups)
    r = client.post("/demo/predict", json={"sequence": "M" * 60, "organism_group": other})
    assert (r.status_code, _code(r)) == (422, "group_not_served")


def test_an_unknown_group_is_400(client: TestClient) -> None:
    r = client.post("/demo/predict", json={"sequence": "M" * 60, "organism_group": "nope"})
    assert (r.status_code, _code(r)) == (400, "unknown_group")


@pytest.mark.parametrize(
    ("fasta", "code", "line"),
    [
        ("MKT\n>a\nMKT\n", "fasta_no_header", 1),
        (">a\nMK1T\n", "fasta_invalid_character", 2),
        (">a\n>b\nMKT\n", "fasta_empty_record", 1),
        (">a\nM\n>b\nM\n>c\nM\n", "batch_too_large", 5),
        ("\n\n", "fasta_empty", None),
    ],
)
def test_malformed_fasta_reports_a_code_and_a_line(
    client: TestClient, fasta: str, code: str, line: int | None
) -> None:
    r = client.post("/demo/batch", json={"fasta": fasta, "organism_group": "eukaryote"})
    assert r.status_code in (400, 422)
    assert _code(r) == code
    assert r.json()["error"]["line"] == line


def test_unknown_jobs_are_404_with_a_code(client: TestClient) -> None:
    for r in (client.get("/demo/jobs/nope"), client.post("/demo/jobs/nope/cancel")):
        assert (r.status_code, _code(r)) == (404, "unknown_job")


def test_a_bad_structure_key_is_400(client: TestClient) -> None:
    r = client.get("/demo/structure/not-a-hash")
    assert (r.status_code, _code(r)) == (400, "structure_bad_hash")


def test_health_flags_degraded_when_evidence_is_unavailable(client: TestClient) -> None:
    body = client.get("/demo/health").json()
    assert body["degraded"] is True
    assert body["evidence"]["reason"] == "index not loaded"


def test_the_kill_switch_answers_with_a_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_v2, "ENABLED", False)
    r = client.get("/demo/health")
    assert (r.status_code, _code(r)) == (503, "demo_disabled")


def _evidence(available: bool, reason: str | None) -> dict[str, Any]:
    return {"evidence_status": {"available": available, "unavailable_reason": reason}}


def test_degraded_flags_a_missing_evidence_track_and_lists_reasons() -> None:
    out = app_v2._degraded([_evidence(True, None), _evidence(False, "ran out of time")])
    assert out == {"degraded": True, "degraded_reasons": ["ran out of time"]}


def test_degraded_is_false_when_evidence_worked_or_nothing_was_scored() -> None:
    assert app_v2._degraded([_evidence(True, None)])["degraded"] is False
    assert app_v2._degraded([])["degraded"] is False


def test_a_refused_batch_record_carries_its_stable_code() -> None:
    f = app_v2._failure("q1", EngineRefusal("not_in_embedding_cache", "not among them"))
    assert (f["id"], f["code"], f["retryable"]) == ("q1", "not_cached", False)


def test_a_failed_job_carries_the_code_of_the_refusal_that_failed_it() -> None:
    def boom(_progress: Any, _cancelled: Any) -> Any:
        raise EngineRefusal("not_in_embedding_cache", "cold")

    sub = jobs.submit("predict", 1, boom)
    for _ in range(100):
        job = jobs.get(sub["job_id"])
        if job is not None and job["status"] == "failed":
            break
        time.sleep(0.05)
    assert job is not None
    assert (job["status"], job["error_code"]) == ("failed", "not_cached")
