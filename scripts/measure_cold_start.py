"""Measure client-observed latency of a public endpoint, cold and warm.

Used by the Phase 2 hosting acceptance test. It requests a URL repeatedly, waiting
between requests long enough for the host to scale the service to zero, and
records for each request the wall-clock time from sending it to reading the whole
body.

A request counts as cold only when the response proves it: the hello service
reports ``instance_uptime_seconds``, and an instance that has been up for less than
``--cold-uptime`` seconds was started to answer that request. Waiting is how a
cold start is *provoked*; the uptime field is how it is *verified*. A run whose
requests all hit warm instances reports no cold figure rather than a wrong one.

Standard library only, so it runs from any terminal with Python 3.11::

    python scripts/measure_cold_start.py https://HOST/healthz --runs 5 --idle-seconds 1200
"""

import argparse
from http import HTTPStatus
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
import urllib.error
import urllib.request

_TIMEOUT_SECONDS = 120


def _request(url: str) -> dict[str, Any]:
    """Issue one GET and return what the client observed."""
    started = time.perf_counter()
    status = 0
    body: dict[str, Any] = {}
    error: str | None = None
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            status = response.status
            raw = response.read()
        elapsed = time.perf_counter() - started
        try:
            parsed = json.loads(raw)
            body = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            error = "response body is not JSON"
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        status = exc.code
        error = f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        elapsed = time.perf_counter() - started
        error = str(exc)
    return {
        "status": status,
        "seconds": round(elapsed, 3),
        "uptime_seconds": body.get("instance_uptime_seconds"),
        "registry_state": body.get("registry_state"),
        "error": error,
    }


def _summarise(rows: list[dict[str, Any]], cold_uptime: float) -> dict[str, Any]:
    """Split the successful rows into cold and warm by instance uptime."""
    ok = [row for row in rows if row["status"] == HTTPStatus.OK]
    cold: list[float] = []
    warm: list[float] = []
    for row in ok:
        uptime = row["uptime_seconds"]
        is_cold = isinstance(uptime, int | float) and uptime < cold_uptime
        (cold if is_cold else warm).append(row["seconds"])
    return {
        "requests": len(rows),
        "ok": len(ok),
        "cold_seconds": cold,
        "cold_median": statistics.median(cold) if cold else None,
        "warm_seconds": warm,
        "warm_median": statistics.median(warm) if warm else None,
    }


def main() -> int:
    """Run the measurement; exit 1 when any request was not a 200."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="Full URL, e.g. https://HOST/healthz")
    parser.add_argument("--runs", type=int, default=5, help="Cold attempts (default 5)")
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=1200,
        help="Wait before each cold attempt (default 1200 = 20 min)",
    )
    parser.add_argument(
        "--cold-uptime",
        type=float,
        default=10.0,
        help="Instance uptime below this proves a fresh start (default 10 s)",
    )
    parser.add_argument("--out", type=Path, help="Write the full results to this JSON file")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for attempt in range(1, args.runs + 1):
        sys.stdout.write(f"[{attempt}/{args.runs}] idle {args.idle_seconds}s ...\n")
        sys.stdout.flush()
        time.sleep(args.idle_seconds)
        for label in ("first", "second"):
            row = {"attempt": attempt, "request": label, **_request(args.url)}
            rows.append(row)
            sys.stdout.write(f"  {label}: {json.dumps(row)}\n")
            sys.stdout.flush()

    summary = _summarise(rows, args.cold_uptime)
    sys.stdout.write(f"summary: {json.dumps(summary)}\n")
    if args.out:
        args.out.write_text(
            json.dumps({"url": args.url, "rows": rows, "summary": summary}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if summary["ok"] == summary["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
