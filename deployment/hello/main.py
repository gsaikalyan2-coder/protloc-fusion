"""Hello-world service for the Phase 2 hosting acceptance test.

It exists to prove one thing: a containerised FastAPI process on the chosen host
answers ``GET /healthz`` with 200 from the public internet, and to let a client
tell a freshly started instance from a warm one. It is not the ProtLoc service and
imports nothing from it.

``instance_uptime_seconds`` is the field that makes a cold-start measurement
honest: a response whose uptime is a few seconds came from an instance that was
started to answer it, and one whose uptime is minutes did not.
"""

import time

from fastapi import FastAPI

_STARTED = time.monotonic()

app = FastAPI(title="protloc-hello", docs_url=None, redoc_url=None, openapi_url=None)


def _body() -> dict[str, object]:
    return {
        "status": "degraded",
        "registry_state": "empty",
        "instance_uptime_seconds": round(time.monotonic() - _STARTED, 3),
    }


@app.get("/healthz")
def healthz() -> dict[str, object]:
    """Mirror the contract path and its registry-state field."""
    return _body()


@app.get("/health")
def health() -> dict[str, object]:
    """Second path, so a platform that reserves ``/healthz`` can be told apart."""
    return _body()
