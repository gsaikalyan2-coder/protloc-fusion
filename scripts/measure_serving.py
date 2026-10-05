"""Measure the served path through the real queue API (Phase 14, ADR-0001 section 4.3).

Starts the live app in-process (the real engine, the real MMseqs2 index, the real queue and
worker thread), then drives it only through ``/demo/predict``, ``/demo/batch`` and
``/demo/jobs/{id}``. Every figure comes from the timestamps the queue records on the job:

* queue wait  = started_at - created_at
* compute     = finished_at - started_at
* time to result = finished_at - created_at

Four sections: one cold request, sequential single-sequence jobs, a batch-size sweep, and a
burst of simultaneous jobs (where queue wait appears). Nothing is estimated; a section that
does not run is absent from the manifest, and the manifest says what the machine was.

Usage (repo root, after ``scripts/build_gallery.py --write``)::

    python scripts/measure_serving.py --write
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from protloc.api.app_v2 import app  # noqa: E402
from protloc.inference.engine import composite_version  # noqa: E402

GALLERY = REPO / "data" / "gallery" / "gallery.json"
OUT = REPO / "data" / "manifests" / "serving_measurements.json"
POLL_SECONDS = 0.05
JOB_DEADLINE_SECONDS = 600.0
SWEEP_SIZES = (1, 2, 5, 10, 20, 40)
SWEEP_REPEATS = 3
SINGLE_ROUNDS = 3
BURST_JOBS = 10
SWEEP_CAP = 64  # the sweep's own ceiling, never the served MAX_BATCH_SEQUENCES
P95 = 0.95
ACCEPTED = 202


def load_dotenv(path: Path) -> None:
    """Read ``KEY=VALUE`` lines into the environment without overriding what is set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if text and not text.startswith("#") and "=" in text:
            key, value = text.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..1) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def describe(values: list[float]) -> dict[str, float | int]:
    """n, min, median, mean, p95 and max, rounded to milliseconds."""
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "p95": round(percentile(values, P95), 3),
        "max": round(max(values), 3),
    }


def wait_for(client: Any, url: str) -> dict[str, Any]:
    """Poll a job until it completes or fails; return its body."""
    deadline = time.monotonic() + JOB_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(url).json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(POLL_SECONDS)
    msg = f"job {url} did not finish in {JOB_DEADLINE_SECONDS} s"
    raise TimeoutError(msg)


def timings(job: dict[str, Any]) -> dict[str, float]:
    """Queue wait, compute and time to result of a finished job, in seconds."""
    return {
        "queue_wait": job["started_at"] - job["created_at"],
        "compute": job["finished_at"] - job["started_at"],
        "time_to_result": job["finished_at"] - job["created_at"],
    }


def degraded(job: dict[str, Any]) -> int:
    """How many results in a job came back without homology evidence."""
    return sum(
        1 for r in job.get("results", []) if not r.get("evidence_status", {}).get("available")
    )


def fasta(examples: list[dict[str, Any]], count: int) -> str:
    """``count`` records cycling through the examples (ids made unique)."""
    return "\n".join(f">m{i}\n{examples[i % len(examples)]['sequence']}" for i in range(count))


def summarise(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Describe every timing column of a list of job timings."""
    return {key: describe([r[key] for r in rows]) for key in rows[0]}


def cold(client: Any, first: dict[str, Any]) -> dict[str, Any]:
    """The first request after startup."""
    sub = client.post(
        "/demo/predict",
        json={"sequence": first["sequence"], "organism_group": first["organism_group"]},
    )
    job = wait_for(client, sub.json()["result_url"])
    return {
        **{k: round(v, 3) for k, v in timings(job).items()},
        "status": job["status"],
        "evidence_degraded": degraded(job),
    }


def singles(client: Any, gallery: list[dict[str, Any]]) -> dict[str, Any]:
    """Sequential single-sequence jobs: one job in the queue at a time."""
    rows: list[dict[str, float]] = []
    bad = 0
    for _ in range(SINGLE_ROUNDS):
        for ex in gallery:
            sub = client.post(
                "/demo/predict",
                json={"sequence": ex["sequence"], "organism_group": ex["organism_group"]},
            )
            job = wait_for(client, sub.json()["result_url"])
            if job["status"] != "completed":
                bad += 1
                continue
            rows.append(timings(job))
            bad += degraded(job)
    return {"jobs": len(rows), "failed_or_degraded": bad, **summarise(rows)}


def sweep_one(client: Any, euk: list[dict[str, Any]], size: int) -> tuple[dict[str, Any], str]:
    """Three repeats of one batch size; returns (row or {}, error text or '')."""
    reps: list[dict[str, float]] = []
    degraded_count = 0
    for _ in range(SWEEP_REPEATS):
        sub = client.post(
            "/demo/batch", json={"fasta": fasta(euk, size), "organism_group": "eukaryote"}
        )
        if sub.status_code != ACCEPTED:
            return {}, f"size {size}: HTTP {sub.status_code} {sub.text[:200]}"
        job = wait_for(client, sub.json()["result_url"])
        if job["status"] != "completed":
            return {}, f"size {size}: {job['status']} {job.get('error')}"
        reps.append(timings(job))
        degraded_count += degraded(job)
    compute = [r["compute"] for r in reps]
    return {
        "sequences": size,
        "repeats": len(reps),
        "evidence_degraded": degraded_count,
        "compute_s": describe(compute),
        "compute_s_per_sequence_median": round(statistics.median(compute) / size, 3),
    }, ""


def burst(client: Any, gallery: list[dict[str, Any]]) -> dict[str, Any]:
    """Jobs submitted together, so the later ones wait."""
    subs = [
        client.post(
            "/demo/predict",
            json={"sequence": g["sequence"], "organism_group": g["organism_group"]},
        ).json()
        for g in (gallery * 2)[:BURST_JOBS]
    ]
    rows = []
    for s in subs:
        job = wait_for(client, s["result_url"])
        if job["status"] == "completed":
            rows.append(timings(job))
    out: dict[str, Any] = {"jobs_submitted": len(subs), "jobs_completed": len(rows)}
    if rows:
        out.update(summarise(rows))
    return out


def say(text: str) -> None:
    """Write a line to stdout."""
    sys.stdout.write(text + "\n")


def main() -> int:
    """Run the four sections and print (and optionally write) the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(REPO)}")
    args = parser.parse_args()

    os.chdir(REPO)
    load_dotenv(REPO / ".env")
    served_limit = os.environ.get("MAX_BATCH_SEQUENCES", "")
    os.environ["MAX_BATCH_SEQUENCES"] = str(SWEEP_CAP)  # the sweep, not the served limit

    if not GALLERY.is_file():
        sys.stderr.write("run scripts/build_gallery.py --write first\n")
        return 1
    gallery: list[dict[str, Any]] = json.loads(GALLERY.read_text(encoding="utf-8"))["examples"]
    euk = [e for e in gallery if e["organism_group"] == "eukaryote"]

    manifest: dict[str, Any] = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "scope": "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers",
        "served_max_batch_sequences_when_measured": served_limit,
        "sweep_cap": SWEEP_CAP,
        "examples": len(gallery),
        "eukaryote_examples": len(euk),
    }

    started = time.monotonic()
    with TestClient(app) as client:
        manifest["startup_seconds"] = round(time.monotonic() - started, 3)
        health = client.get("/demo/health").json()
        manifest["evidence_state"] = health["evidence"]
        manifest["model_version"] = health["model_version"]
        manifest["model_version_expected"] = composite_version(
            json.loads((REPO / "configs" / "serving" / "engine.json").read_text())["bundles"]
        )
        manifest["cold_first_request"] = cold(client, gallery[0])
        manifest["sequential_singles"] = singles(client, gallery)
        rows, errors = [], []
        for size in SWEEP_SIZES:
            row, error = sweep_one(client, euk, size)
            if row:
                rows.append(row)
            else:
                errors.append(error)
        manifest["batch_sweep"] = rows
        if errors:
            manifest["sweep_errors"] = errors
        manifest["burst"] = burst(client, gallery)

    text = json.dumps(manifest, indent=2, sort_keys=True)
    say(text)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        say(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
