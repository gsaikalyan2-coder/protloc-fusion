"""Request IDs, structured JSON logs and in-process metrics for the live app.

Infrastructure, standard library only: importable from ``api`` and ``jobs`` and importing
nothing from ``protloc``. Three things live here:

* **Request IDs.** Every HTTP request gets an id: the caller's ``X-Request-ID`` when it is a
  safe token (letters, digits, ``.``, ``_``, ``-``; at most 64 characters), otherwise a new
  UUID4 hex. It is echoed on the response and carried in a :class:`~contextvars.ContextVar`,
  so a job submitted while handling the request records which request created it.
* **JSON logs.** :class:`JsonFormatter` writes one JSON object per line. ``protloc.access``
  logs one line per request; ``protloc.jobs`` logs one line per job transition. No sequence,
  no request body and no client address is ever logged.
* **Metrics.** :data:`METRICS` counts requests and jobs and keeps a bounded window of recent
  latencies, from which ``/demo/metrics`` reports p50, p95 and max. In-memory and
  per-process, like the job queue (ADR-0008): a restart starts them from zero.
"""

from __future__ import annotations

from collections import Counter, deque
from contextvars import ContextVar
from datetime import UTC, datetime
import json
import logging
import math
import re
import sys
import threading
import time
from typing import Any, Final
import uuid

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
#: Latencies kept per route (and per job kind) for the percentiles; older ones drop off.
LATENCY_WINDOW: Final[int] = 500
#: The route label for a request that matched no route (a 404 from the router).
UNMATCHED: Final[str] = "<unmatched>"

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

request_id_var: ContextVar[str | None] = ContextVar("protloc_request_id", default=None)


def new_request_id(incoming: str | None) -> str:
    """The caller's id when it is a safe token, otherwise a fresh UUID4 hex."""
    if incoming is not None and _SAFE_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    """The id of the request being handled in this context, if any."""
    return request_id_var.get()


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an ascending list (``fraction`` in (0, 1])."""
    if not sorted_values:
        raise ValueError("no values")
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


class LatencyWindow:
    """The most recent ``LATENCY_WINDOW`` durations, in seconds."""

    def __init__(self, size: int = LATENCY_WINDOW) -> None:
        """An empty window holding at most ``size`` values."""
        self._values: deque[float] = deque(maxlen=size)
        self.count = 0

    def add(self, seconds: float) -> None:
        """Record one duration."""
        self._values.append(seconds)
        self.count += 1

    def summary(self) -> dict[str, Any]:
        """Count plus p50, p95 and max over the window, in milliseconds (``None`` when empty)."""
        ordered = sorted(self._values)
        if not ordered:
            return {
                "count": self.count,
                "window": 0,
                "p50_ms": None,
                "p95_ms": None,
                "max_ms": None,
            }
        return {
            "count": self.count,
            "window": len(ordered),
            "p50_ms": round(percentile(ordered, 0.50) * 1000, 1),
            "p95_ms": round(percentile(ordered, 0.95) * 1000, 1),
            "max_ms": round(ordered[-1] * 1000, 1),
        }


def status_class(status: int) -> str:
    """``2xx``, ``4xx``, ``5xx`` and so on."""
    return f"{status // 100}xx"


class Metrics:
    """Thread-safe request and job counters for one process."""

    def __init__(self) -> None:
        """Start every counter at zero, now."""
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Zero everything (startup and tests)."""
        with self._lock:
            self.started_at = time.time()
            self._requests: Counter[tuple[str, str, str]] = Counter()
            self._latency: dict[tuple[str, str], LatencyWindow] = {}
            self._jobs: Counter[tuple[str, str]] = Counter()
            self._job_run: dict[str, LatencyWindow] = {}
            self._job_wait: dict[str, LatencyWindow] = {}
            self._degraded = 0

    def record_request(self, method: str, route: str, status: int, seconds: float) -> None:
        """Count one finished request and keep its latency."""
        with self._lock:
            self._requests[(method, route, status_class(status))] += 1
            self._latency.setdefault((method, route), LatencyWindow()).add(seconds)

    def record_job(
        self,
        kind: str,
        outcome: str,
        *,
        wait_seconds: float | None,
        run_seconds: float | None,
        degraded: bool,
    ) -> None:
        """Count one finished job (completed, failed or cancelled) and keep its timings."""
        with self._lock:
            self._jobs[(kind, outcome)] += 1
            if wait_seconds is not None:
                self._job_wait.setdefault(kind, LatencyWindow()).add(wait_seconds)
            if run_seconds is not None:
                self._job_run.setdefault(kind, LatencyWindow()).add(run_seconds)
            if degraded:
                self._degraded += 1

    def snapshot(self) -> dict[str, Any]:
        """Everything, as plain JSON-ready values."""
        with self._lock:
            routes: dict[str, dict[str, Any]] = {}
            for (method, route), window in sorted(self._latency.items()):
                key = f"{method} {route}"
                statuses = {
                    cls: n
                    for (m, r, cls), n in sorted(self._requests.items())
                    if (m, r) == (method, route)
                }
                routes[key] = {"statuses": statuses, "latency": window.summary()}
            kinds = sorted({k for k, _ in self._jobs} | set(self._job_run) | set(self._job_wait))
            jobs = {
                kind: {
                    "outcomes": {o: n for (k, o), n in sorted(self._jobs.items()) if k == kind},
                    "wait": self._job_wait.get(kind, LatencyWindow()).summary(),
                    "run": self._job_run.get(kind, LatencyWindow()).summary(),
                }
                for kind in kinds
            }
            return {
                "started_at": datetime.fromtimestamp(self.started_at, UTC).isoformat(),
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests_total": sum(self._requests.values()),
                "routes": routes,
                "jobs_finished_total": sum(self._jobs.values()),
                "jobs": jobs,
                "jobs_degraded_total": self._degraded,
                "latency_window": LATENCY_WINDOW,
            }


METRICS: Final[Metrics] = Metrics()


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message and the record's ``fields``."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record``; an exception becomes an ``error`` string, never a traceback block."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        rid = current_request_id()
        if rid is not None and "request_id" not in payload:
            payload["request_id"] = rid
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, default=str, separators=(",", ":"))


_HANDLER_NAME = "protloc-json"


def configure_logging(level: str = "INFO", *, quiet_uvicorn_access: bool = True) -> None:
    """Send every ``protloc.*`` logger to stderr as JSON at ``level`` (idempotent).

    With ``quiet_uvicorn_access`` the plain uvicorn access line is switched off, because
    ``protloc.access`` writes a richer one per request.
    """
    root = logging.getLogger("protloc")
    root.setLevel(level.upper())
    if not any(h.get_name() == _HANDLER_NAME for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.propagate = False
    logging.getLogger("uvicorn.access").disabled = quiet_uvicorn_access


ACCESS_LOG: Final[logging.Logger] = logging.getLogger("protloc.access")
JOBS_LOG: Final[logging.Logger] = logging.getLogger("protloc.jobs")

__all__ = [
    "ACCESS_LOG",
    "JOBS_LOG",
    "LATENCY_WINDOW",
    "METRICS",
    "REQUEST_ID_HEADER",
    "UNMATCHED",
    "JsonFormatter",
    "LatencyWindow",
    "Metrics",
    "configure_logging",
    "current_request_id",
    "new_request_id",
    "percentile",
    "request_id_var",
    "status_class",
]
