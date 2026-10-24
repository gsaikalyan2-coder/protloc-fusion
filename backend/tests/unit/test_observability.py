"""Request IDs, JSON logs and metrics (features 11 + 12).

The utilities are standard library only and tested directly. The middleware and the
``/demo/metrics`` route are tested against a stub engine, as in ``test_api_errors.py``; the
job counters are tested through the real in-memory queue with tiny jobs.
"""

from collections.abc import Iterator
import json
import logging
import sys
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from protloc.jobs import queue as jobs
from protloc.utils.observability import (
    METRICS,
    REQUEST_ID_HEADER,
    JsonFormatter,
    LatencyWindow,
    Metrics,
    configure_logging,
    new_request_id,
    percentile,
    request_id_var,
    status_class,
)

# -- request ids ------------------------------------------------------------------------


@pytest.mark.parametrize("given", ["abc", "A-b_c.9", "x" * 64, "0f" * 16])
def test_a_safe_caller_id_is_kept(given: str) -> None:
    assert new_request_id(given) == given


@pytest.mark.parametrize(
    "given", [None, "", "x" * 65, "has space", "semi;colon", "ünïcode", "a\nb"]
)
def test_an_unsafe_or_missing_id_is_replaced(given: str | None) -> None:
    made = new_request_id(given)
    assert made != given
    assert len(made) == 32
    int(made, 16)


# -- percentiles and windows -------------------------------------------------------------


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile([7.0], 0.95) == 7.0
    with pytest.raises(ValueError, match="no values"):
        percentile([], 0.5)


def test_a_window_keeps_only_the_most_recent_values_but_counts_all() -> None:
    window = LatencyWindow(size=3)
    for seconds in (10.0, 0.001, 0.002, 0.003):
        window.add(seconds)
    summary = window.summary()
    assert (summary["count"], summary["window"]) == (4, 3)
    assert summary["max_ms"] == 3.0
    assert summary["p50_ms"] == 2.0


def test_an_empty_window_reports_none() -> None:
    assert LatencyWindow().summary() == {
        "count": 0,
        "window": 0,
        "p50_ms": None,
        "p95_ms": None,
        "max_ms": None,
    }


def test_status_classes() -> None:
    assert [status_class(s) for s in (200, 202, 404, 503)] == ["2xx", "2xx", "4xx", "5xx"]


# -- metrics -------------------------------------------------------------------------------


def test_metrics_group_requests_by_route_and_status_class() -> None:
    m = Metrics()
    m.record_request("GET", "/demo/jobs/{job_id}", 200, 0.010)
    m.record_request("GET", "/demo/jobs/{job_id}", 404, 0.002)
    m.record_request("POST", "/demo/predict", 202, 0.030)
    snap = m.snapshot()
    assert snap["requests_total"] == 3
    poll = snap["routes"]["GET /demo/jobs/{job_id}"]
    assert poll["statuses"] == {"2xx": 1, "4xx": 1}
    assert poll["latency"]["count"] == 2
    assert snap["routes"]["POST /demo/predict"]["latency"]["p95_ms"] == 30.0


def test_metrics_count_job_outcomes_timings_and_degraded() -> None:
    m = Metrics()
    m.record_job("predict", "completed", wait_seconds=0.1, run_seconds=0.4, degraded=True)
    m.record_job("predict", "failed", wait_seconds=0.0, run_seconds=0.2, degraded=False)
    m.record_job("batch", "cancelled", wait_seconds=None, run_seconds=None, degraded=False)
    snap = m.snapshot()
    assert snap["jobs_finished_total"] == 3
    assert snap["jobs_degraded_total"] == 1
    assert snap["jobs"]["predict"]["outcomes"] == {"completed": 1, "failed": 1}
    assert snap["jobs"]["predict"]["run"]["max_ms"] == 400.0
    assert snap["jobs"]["batch"]["run"]["count"] == 0
    m.reset()
    assert m.snapshot()["requests_total"] == 0


def test_a_snapshot_is_json_ready() -> None:
    m = Metrics()
    m.record_request("GET", "/demo/health", 200, 0.001)
    json.dumps(m.snapshot())


# -- JSON logs -----------------------------------------------------------------------------


def _record(msg: str, fields: dict[str, Any] | None = None, **kw: Any) -> logging.LogRecord:
    record = logging.makeLogRecord(
        {"name": "protloc.access", "levelname": "INFO", "msg": msg, **kw}
    )
    if fields is not None:
        record.fields = fields
    return record


def test_a_line_is_one_json_object_with_its_fields() -> None:
    line = JsonFormatter().format(_record("request", {"status": 202, "route": "/demo/predict"}))
    assert "\n" not in line
    body = json.loads(line)
    assert body["msg"] == "request"
    assert body["level"] == "INFO"
    assert body["logger"] == "protloc.access"
    assert (body["status"], body["route"]) == (202, "/demo/predict")
    assert body["ts"].endswith("+00:00")


def test_the_current_request_id_is_added_unless_given() -> None:
    token = request_id_var.set("rid-1")
    try:
        assert json.loads(JsonFormatter().format(_record("x")))["request_id"] == "rid-1"
        given = JsonFormatter().format(_record("x", {"request_id": "job-rid"}))
        assert json.loads(given)["request_id"] == "job-rid"
    finally:
        request_id_var.reset(token)
    assert "request_id" not in json.loads(JsonFormatter().format(_record("x")))


def _boom() -> None:
    raise RuntimeError("boom")


def test_an_exception_becomes_one_error_string() -> None:
    try:
        _boom()
    except RuntimeError:
        record = _record("unhandled error", exc_info=sys.exc_info())
    body = json.loads(JsonFormatter().format(record))
    assert body["error"] == "RuntimeError: boom"


def test_configure_logging_is_idempotent_and_quiets_uvicorn_access() -> None:
    root = logging.getLogger("protloc")
    before = (root.level, root.propagate, list(root.handlers))
    try:
        configure_logging("WARNING")
        configure_logging("DEBUG")
        assert [h.get_name() for h in root.handlers].count("protloc-json") == 1
        assert root.level == logging.DEBUG
        assert logging.getLogger("uvicorn.access").disabled
    finally:
        root.setLevel(before[0])
        root.propagate = before[1]
        root.handlers[:] = before[2]
        logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def captured(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture ``protloc.*`` records even after ``configure_logging`` stopped propagation."""
    root = logging.getLogger("protloc")
    level = root.level
    root.addHandler(caplog.handler)
    root.setLevel(logging.DEBUG)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    root.removeHandler(caplog.handler)
    root.setLevel(level)


# -- the queue records jobs ----------------------------------------------------------------


def _wait(job_id: str) -> dict[str, Any]:
    deadline = time.time() + 10
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_a_job_records_its_request_and_is_counted() -> None:
    METRICS.reset()
    token = request_id_var.set("req-42")
    try:
        sub = jobs.submit("obs_test", 2, lambda progress, cancelled: {"degraded": True})
    finally:
        request_id_var.reset(token)
    job = _wait(sub["job_id"])
    assert job["request_id"] == "req-42"
    snap = METRICS.snapshot()
    assert snap["jobs"]["obs_test"]["outcomes"] == {"completed": 1}
    assert snap["jobs_degraded_total"] == 1
    assert snap["jobs"]["obs_test"]["run"]["count"] == 1


def test_a_failed_job_is_counted_and_logged_as_a_warning(
    captured: pytest.LogCaptureFixture,
) -> None:
    METRICS.reset()

    def boom(progress: Any, cancelled: Any) -> None:
        raise RuntimeError("model blew up")

    sub = jobs.submit("obs_fail", 1, boom)
    _wait(sub["job_id"])
    deadline = time.time() + 5
    while time.time() < deadline and not METRICS.snapshot()["jobs"].get("obs_fail"):
        time.sleep(0.02)
    assert METRICS.snapshot()["jobs"]["obs_fail"]["outcomes"] == {"failed": 1}
    deadline = time.time() + 5
    finished: list[logging.LogRecord] = []
    while time.time() < deadline and not finished:
        finished = [
            r
            for r in captured.records
            if r.getMessage() == "job finished" and r.fields["kind"] == "obs_fail"  # type: ignore[attr-defined]
        ]
        time.sleep(0.02)
    assert finished
    assert finished[-1].levelno == logging.WARNING
    assert finished[-1].fields["error_code"] == "job_failed"  # type: ignore[attr-defined]
    assert "sequence" not in finished[-1].fields  # type: ignore[attr-defined]


# -- the middleware and /demo/metrics -------------------------------------------------------


class _Stub:
    groups: ClassVar[tuple[str, ...]] = ("eukaryote",)
    cached_sequences = 500
    provenance: ClassVar[dict[str, str]] = {
        "model_version": "test",
        "config_hash": "c",
        "rules_version": "r",
    }
    evidence_state: ClassVar[dict[str, Any]] = {
        "available": True,
        "reason": None,
        "swissprot_release": "x",
    }
    live_encoder: ClassVar[dict[str, Any] | None] = None

    def has_states(self, _sequence: str) -> bool:
        return False


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    pytest.importorskip("numpy")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from protloc.api import app_v2  # noqa: PLC0415
    from protloc.inference import service  # noqa: PLC0415

    service.configure(_Stub())  # type: ignore[arg-type]
    monkeypatch.setattr(
        app_v2,
        "get_serving_settings",
        lambda: SimpleNamespace(MAX_BATCH_SEQUENCES=2, DATA_ROOT="."),
    )
    METRICS.reset()
    yield TestClient(app_v2.app)
    service.configure(None)


def test_every_response_carries_a_request_id(client: Any) -> None:
    made = client.get("/demo/health").headers[REQUEST_ID_HEADER]
    assert len(made) == 32
    echoed = client.get("/demo/health", headers={REQUEST_ID_HEADER: "trace-7"})
    assert echoed.headers[REQUEST_ID_HEADER] == "trace-7"
    unsafe = client.get("/demo/health", headers={REQUEST_ID_HEADER: "bad id;"})
    assert unsafe.headers[REQUEST_ID_HEADER] != "bad id;"


def test_error_responses_carry_a_request_id_too(client: Any) -> None:
    r = client.get("/demo/jobs/not-a-job", headers={REQUEST_ID_HEADER: "err-1"})
    assert r.status_code == 404
    assert r.headers[REQUEST_ID_HEADER] == "err-1"
    r = client.post("/demo/predict", json={"sequence": "MKT"})
    assert (r.status_code, r.json()["error"]["code"]) == (422, "not_cached")
    assert REQUEST_ID_HEADER in r.headers


def test_metrics_report_route_templates_not_raw_ids(client: Any) -> None:
    for job_id in ("a", "b", "c"):
        client.get(f"/demo/jobs/{job_id}")
    client.get("/demo/health")
    client.get("/no/such/route")
    body = client.get("/demo/metrics").json()
    routes = body["routes"]
    assert routes["GET /demo/jobs/{job_id}"]["statuses"] == {"4xx": 3}
    assert routes["GET /demo/health"]["statuses"] == {"2xx": 1}
    assert "GET <unmatched>" in routes
    assert not any("/demo/jobs/a" in key for key in routes)
    assert body["queue"]["depth"] >= 0
    assert body["model_version"] == "test"
    assert body["evidence"]["available"] is True


def test_one_access_line_per_request_and_polls_are_debug(
    client: Any, captured: pytest.LogCaptureFixture
) -> None:
    token_job = jobs.submit("obs_poll", 1, lambda progress, cancelled: {})["job_id"]
    _wait(token_job)
    captured.clear()
    client.get(f"/demo/jobs/{token_job}", headers={REQUEST_ID_HEADER: "poll-1"})
    client.get("/demo/health", headers={REQUEST_ID_HEADER: "health-1"})
    lines = [r for r in captured.records if r.name == "protloc.access"]
    assert len(lines) == 2
    poll, health = lines
    assert poll.levelno == logging.DEBUG
    assert poll.fields["route"] == "/demo/jobs/{job_id}"  # type: ignore[attr-defined]
    assert poll.fields["job_id"] == token_job  # type: ignore[attr-defined]
    assert health.levelno == logging.INFO
    assert health.fields["status"] == 200  # type: ignore[attr-defined]
    rendered = json.loads(JsonFormatter().format(health))
    assert rendered["duration_ms"] >= 0
    assert "client" not in rendered


def test_the_kill_switch_503_is_counted(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from protloc.api import app_v2  # noqa: PLC0415

    monkeypatch.setattr(app_v2, "ENABLED", False)
    r = client.get("/demo/health")
    assert r.status_code == 503
    assert REQUEST_ID_HEADER in r.headers
    assert METRICS.snapshot()["requests_total"] == 1
