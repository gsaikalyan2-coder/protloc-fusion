"""In-memory async job queue — mirrors ADR-0001 (every prediction is a job).

Single worker thread, FIFO. Jobs expire after the retention window -> 410 Gone. State is lost on
restart (ADR-0008 says so to the user). Every job records when it was created, started and
finished, so queue wait and compute time are separable (ADR-0001 section 4.3), and the queue
learns its own seconds-per-sequence from the jobs it has finished: the wait estimate is that
measurement, never a constant, and it is ``None`` until one job has run.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import logging
import queue
import threading
import time
from typing import Any
import uuid

from protloc.domain.errors import api_code_for_refusal
from protloc.utils.observability import JOBS_LOG, METRICS, current_request_id

SECONDS_PER_HOUR = 3600
RETENTION_SECONDS = SECONDS_PER_HOUR  # replaced by configure(); the Phase 1 default
RATE_WINDOW = 50  # finished jobs the wait estimate averages over

_jobs: dict[str, dict[str, Any]] = {}
_q: queue.Queue[str] = queue.Queue()
_lock = threading.Lock()
_rates: deque[float] = deque(maxlen=RATE_WINDOW)  # compute seconds per sequence, finished jobs


def configure(retention_hours: float) -> None:
    """Set the retention window (``JOB_RETENTION_HOURS``)."""
    global RETENTION_SECONDS  # noqa: PLW0603 - one queue per process
    RETENTION_SECONDS = round(retention_hours * SECONDS_PER_HOUR)


def estimated_wait_seconds(sequences_ahead: int, count: int) -> float | None:
    """Measured seconds until a new job of ``count`` sequences would finish; None before data."""
    with _lock:
        if not _rates:
            return None
        return round(sum(_rates) / len(_rates) * (sequences_ahead + count), 1)


def pending_sequences() -> int:
    """Sequences in jobs that are queued or running."""
    with _lock:
        return sum(j["total"] for j in _jobs.values() if j["status"] in ("queued", "running"))


def submit(
    kind: str, total: int, fn: Callable[[Callable[[int], None], Callable[[], bool]], Any]
) -> dict[str, Any]:
    """Queue a job; returns its id, status and queue position (counting itself)."""
    jid = str(uuid.uuid4())
    now = time.time()
    with _lock:
        _jobs[jid] = {
            "job_id": jid,
            "kind": kind,
            "status": "queued",
            "total": total,
            "completed_count": 0,
            "failed_count": 0,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
            "result": None,
            "error": None,
            "error_code": None,
            "request_id": current_request_id(),
            "_fn": fn,
        }
        pos = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
    _q.put(jid)
    JOBS_LOG.info(
        "job queued",
        extra={"fields": {"job_id": jid, "kind": kind, "total": total, "queue_position": pos}},
    )
    return {"job_id": jid, "status": "queued", "queue_position": pos}


def get(jid: str) -> dict[str, Any] | None:
    """The public view of a job (no private keys), or None."""
    with _lock:
        j = _jobs.get(jid)
        return None if j is None else {k: v for k, v in j.items() if not k.startswith("_")}


def expired(j: dict[str, Any]) -> bool:
    """Whether the job is past the retention window."""
    return bool(time.time() - j["created_at"] > RETENTION_SECONDS)


def cancel(jid: str) -> dict[str, Any] | None:
    """Cancel a queued or running job; returns its public view."""
    with _lock:
        j = _jobs.get(jid)
        if j and j["status"] in ("queued", "running"):
            j["status"] = "cancelled"
            j["updated_at"] = time.time()
    return get(jid)


def depth() -> int:
    """Jobs queued or running."""
    with _lock:
        return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))


def _worker() -> None:
    while True:
        jid = _q.get()
        with _lock:
            j = _jobs.get(jid)
            if not j or j["status"] != "queued":
                continue
            j["status"] = "running"
            j["started_at"] = time.time()

        def progress(n: int, _j: dict[str, Any] = j) -> None:
            with _lock:
                _j["completed_count"] = n
                _j["updated_at"] = time.time()

        def cancelled(_j: dict[str, Any] = j) -> bool:
            return bool(_j["status"] == "cancelled")

        try:
            res = j["_fn"](progress, cancelled)
            with _lock:
                if j["status"] != "cancelled":
                    j["result"], j["status"] = res, "completed"
        except Exception as exc:  # surfaced to the client, never swallowed
            with _lock:
                j["status"], j["error"] = "failed", str(exc)
                j["error_code"] = api_code_for_refusal(str(getattr(exc, "code", "")))
        with _lock:
            j["finished_at"] = j["updated_at"] = time.time()
            if j["status"] == "completed" and j["total"] > 0:
                _rates.append((j["finished_at"] - j["started_at"]) / j["total"])
            finished = {k: v for k, v in j.items() if not k.startswith("_") and k != "result"}
            result = j["result"]
        _record_finish(finished, result)


def _record_finish(j: dict[str, Any], result: Any) -> None:
    """Count the finished job and write its one log line (no sequence, no result body)."""
    wait = j["started_at"] - j["created_at"] if j["started_at"] else None
    run = j["finished_at"] - j["started_at"] if j["started_at"] else None
    degraded = isinstance(result, dict) and bool(result.get("degraded"))
    METRICS.record_job(
        str(j["kind"]), str(j["status"]), wait_seconds=wait, run_seconds=run, degraded=degraded
    )
    fields = {
        "job_id": j["job_id"],
        "kind": j["kind"],
        "status": j["status"],
        "total": j["total"],
        "wait_ms": None if wait is None else round(wait * 1000, 1),
        "run_ms": None if run is None else round(run * 1000, 1),
        "degraded": degraded,
        "error_code": j["error_code"],
        "request_id": j["request_id"],
    }
    level = logging.WARNING if j["status"] == "failed" else logging.INFO
    JOBS_LOG.log(level, "job finished", extra={"fields": fields})


threading.Thread(target=_worker, daemon=True, name="job-worker").start()
