"""In-memory async job queue — mirrors ADR-0001 (every prediction is a job).
Single worker thread, FIFO. Jobs expire after RETENTION_SECONDS -> 410 Gone.
Demo only: state is lost on restart.
"""

from __future__ import annotations

from collections.abc import Callable
import queue
import threading
import time
from typing import Any
import uuid

RETENTION_SECONDS = 3600

_jobs: dict[str, dict[str, Any]] = {}
_q: queue.Queue[str] = queue.Queue()
_lock = threading.Lock()


def submit(
    kind: str, total: int, fn: Callable[[Callable[[int], None], Callable[[], bool]], Any]
) -> dict:
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
            "updated_at": now,
            "result": None,
            "error": None,
            "_fn": fn,
        }
        pos = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
    _q.put(jid)
    return {"job_id": jid, "status": "queued", "queue_position": pos}


def get(jid: str) -> dict | None:
    with _lock:
        j = _jobs.get(jid)
        return None if j is None else {k: v for k, v in j.items() if not k.startswith("_")}


def expired(j: dict) -> bool:
    return time.time() - j["created_at"] > RETENTION_SECONDS


def cancel(jid: str) -> dict | None:
    with _lock:
        j = _jobs.get(jid)
        if j and j["status"] in ("queued", "running"):
            j["status"] = "cancelled"
            j["updated_at"] = time.time()
    return get(jid)


def depth() -> int:
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

        def progress(n: int, _j=j) -> None:
            with _lock:
                _j["completed_count"] = n
                _j["updated_at"] = time.time()

        def cancelled(_j=j) -> bool:
            return _j["status"] == "cancelled"

        try:
            res = j["_fn"](progress, cancelled)
            with _lock:
                if j["status"] != "cancelled":
                    j["result"], j["status"] = res, "completed"
        except Exception as exc:  # surfaced to the client, never swallowed
            with _lock:
                j["status"], j["error"] = "failed", str(exc)
        with _lock:
            j["updated_at"] = time.time()


threading.Thread(target=_worker, daemon=True, name="demo-job-worker").start()
