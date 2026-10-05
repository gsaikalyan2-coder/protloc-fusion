"""Local rehearsal of the MODEL_VERSION pin: stale pin, correct pin, restart outage.

Starts the real app (``protloc.api.app_v2``) under uvicorn as a subprocess:
  1. with a deliberately wrong MODEL_VERSION: records whether it refuses to serve
     and keeps the log so the refusal reason can be read, not assumed;
  2. with the correct composite pin (read from serving_measurements.json, or
     ``--pin``): records time to ready and the first /demo/examples call;
  3. restarts it: records the outage window (kill to ready again).

Standard library only. Run from the repo root. This is a LOCAL machine, not the
deployment host, and not a Docker rehearsal. The registry holds one version per
bundle, so this does not swap bundles; it shows only fail-closed pin behaviour and
restart time.

    python scripts/rehearse_rollback.py --app-dir backend --port 8765
"""

import argparse
from http import HTTPStatus
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

_MEASUREMENTS = Path("data/manifests/serving_measurements.json")
_LOG_DIR = Path("data/manifests/rollback_logs")


def _get(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    """GET a URL; return (status, body), or (0, b"") when nothing answers."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - http://127.0.0.1 only
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


def _start(app_dir: str, port: int, pin: str, log: Path) -> subprocess.Popen[bytes]:
    """Start uvicorn with MODEL_VERSION set to ``pin``; stdout and stderr go to ``log``."""
    env = dict(os.environ)
    env["MODEL_VERSION"] = pin
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("wb")
    return subprocess.Popen(  # noqa: S603 - fixed argv, no untrusted input
        [
            sys.executable,
            "-m",
            "uvicorn",
            "protloc.api.app_v2:app",
            "--app-dir",
            app_dir,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def _wait_ready(proc: subprocess.Popen[bytes], url: str, limit: float) -> float | None:
    """Seconds until ``url`` answers 200, or None if the process exits or time runs out."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if proc.poll() is not None:
            return None
        if _get(url)[0] == HTTPStatus.OK:
            return time.perf_counter() - t0
        time.sleep(0.25)
    return None


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the process, escalating to kill."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)


def _tail(path: Path, lines: int = 8) -> list[str]:
    """Last lines of a log, so a failure is explained by the app's own words."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.strip().splitlines()[-lines:]


def main() -> int:
    """Run the three rehearsal phases and write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app-dir", default="backend")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--limit", type=float, default=120.0)
    ap.add_argument("--pin", default=None, help="correct MODEL_VERSION; default from manifest")
    ap.add_argument("--out", type=Path, default=Path("data/manifests/rollback_rehearsal.json"))
    a = ap.parse_args()

    pin = a.pin or json.loads(_MEASUREMENTS.read_text(encoding="utf-8"))["model_version_expected"]
    base = f"http://127.0.0.1:{a.port}"
    ver = f"{base}/demo/version"
    rep: dict[str, object] = {
        "environment": "local machine, not the deployment host; not Docker",
        "scope": "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers",
        "correct_pin": pin,
    }

    stale_log = _LOG_DIR / "stale.log"
    p = _start(a.app_dir, a.port, "stale-0.0.0", stale_log)
    ready = _wait_ready(p, ver, a.limit)
    exited = p.poll() is not None
    rep["stale_pin"] = {
        "served_requests": ready is not None,
        "process_exited": exited,
        "exit_code": p.poll(),
        "log_tail": _tail(stale_log),
        "result": "refused to serve (read log_tail for the reason)"
        if ready is None
        else "SERVED WITH WRONG PIN",
    }
    _stop(p)

    good_log = _LOG_DIR / "correct.log"
    p = _start(a.app_dir, a.port, pin, good_log)
    ready = _wait_ready(p, ver, a.limit)
    rep["correct_pin_time_to_ready_s"] = None if ready is None else round(ready, 2)
    if ready is None:
        rep["correct_pin_failure"] = {
            "exit_code": p.poll(),
            "log_tail": _tail(good_log),
        }
    else:
        s, body = _get(ver)
        rep["reported_model_version"] = (
            json.loads(body).get("model_version") if s == HTTPStatus.OK else None
        )
        t1 = time.perf_counter()
        s, _ = _get(f"{base}/demo/examples", 30)
        rep["examples_status"] = s
        rep["examples_s"] = round(time.perf_counter() - t1, 3)
        t_kill = time.perf_counter()
        _stop(p)
        p = _start(a.app_dir, a.port, pin, _LOG_DIR / "restart.log")
        again = _wait_ready(p, ver, a.limit)
        rep["restart_outage_s"] = None if again is None else round(time.perf_counter() - t_kill, 2)
    _stop(p)

    a.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
