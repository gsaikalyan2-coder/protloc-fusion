"""ProtLoc Fusion — PRESENTATION DEMO API (standalone).

Run:  uvicorn app.main:app --port 8010       (from demo/backend)
Kill switch: PROTLOC_DEMO_ENABLED=0 makes every /demo route return 503.
Nothing in backend/protloc imports this file, and this file imports nothing
from backend/protloc. See demo/OVERRIDE.md.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from protloc.data import reference_demo as data
from protloc.features.classical import LOC_NAMES, ORGANISM_GROUPS
from protloc.inference import service
from protloc.jobs import queue as jobs

ENABLED = os.getenv("PROTLOC_DEMO_ENABLED", "1") != "0"
MAX_BATCH = int(os.getenv("DEMO_MAX_BATCH", "200"))

app = FastAPI(title="ProtLoc Fusion - presentation demo", version=service.MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DEMO_CORS", "http://localhost:5174,http://127.0.0.1:5174").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def kill_switch(request: Request, call_next):
    if not ENABLED and request.url.path.startswith("/demo"):
        return JSONResponse(
            {"detail": "Demo disabled (PROTLOC_DEMO_ENABLED=0)", **service.provenance()}, 503
        )
    return await call_next(request)


class PredictIn(BaseModel):
    sequence: str = Field(min_length=1, max_length=20000)
    organism_group: str = "eukaryote"
    id: str = "query"


class BatchIn(BaseModel):
    fasta: str = Field(min_length=1, max_length=2_000_000)
    organism_group: str = "eukaryote"


class ScanIn(BaseModel):
    sequence: str = Field(min_length=1, max_length=5000)
    organism_group: str = "eukaryote"
    target: str | None = None
    mutations: str = ""
    substitute: str = "A"


def _accepted(sub: dict, count: int) -> JSONResponse:
    body = {
        **sub,
        "result_url": f"/demo/jobs/{sub['job_id']}",
        "sequence_count": count,
        "estimated_wait_seconds": max(1, round(0.03 * count * sub["queue_position"])),
        **service.provenance(),
    }
    return JSONResponse(body, status_code=202)


def _check_group(g: str) -> None:
    if g not in ORGANISM_GROUPS:
        raise HTTPException(400, f"Unknown organism_group '{g}'")


@app.get("/demo/health")
def health() -> dict:
    return {
        "status": "ok",
        "enabled": ENABLED,
        "queue_depth": jobs.depth(),
        "job_retention_seconds": jobs.RETENTION_SECONDS,
        **service.provenance(),
    }


@app.get("/demo/version")
def version() -> dict:
    return {
        **service.provenance(),
        "organism_groups": {
            k: {"label": v["label"], "locations": v["locs"]} for k, v in ORGANISM_GROUPS.items()
        },
        "location_names": LOC_NAMES,
        "tau": service.TRUST["tau"],
        "temperature": service.TRUST["temperature"],
        "threshold_policy": service.THRESHOLD_POLICY,
        "calibration_version": service.CALIBRATION_VERSION,
        "max_batch": MAX_BATCH,
        "rules": service.RULES["rules"],
        "disclaimer": "Demo predictor: interpretable motif engine, not the trained ProtLoc Fusion model. "
        "Reference and benchmark sequences are synthetic.",
    }


@app.get("/demo/examples")
def examples() -> dict:
    return {"examples": data.EXAMPLES, "provenance": service.provenance()}


@app.get("/demo/reference")
def reference() -> dict:
    return {
        "count": len(data.REFERENCE),
        "entries": [{k: v for k, v in r.items() if k != "sequence"} for r in data.REFERENCE],
        "provenance": service.provenance(),
    }


@app.get("/demo/trust")
def trust() -> dict:
    return {**service.TRUST, "provenance": service.provenance()}


@app.post("/demo/predict")
def post_predict(body: PredictIn):
    _check_group(body.organism_group)

    def run(progress, cancelled):
        r = service.predict(body.sequence, body.organism_group, body.id)
        progress(1)
        return {
            "mode": "predict",
            "results": [r],
            "summary": [service.summarize(r)],
            "landscape": service.landscape([r]),
        }

    return _accepted(jobs.submit("predict", 1, run), 1)


@app.post("/demo/batch")
def post_batch(body: BatchIn):
    _check_group(body.organism_group)
    try:
        recs = service.parse_fasta(body.fasta, MAX_BATCH)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    def run(progress, cancelled):
        results, failures = [], []
        for i, (sid, seq) in enumerate(recs, start=1):
            if cancelled():
                break
            try:
                results.append(service.predict(seq, body.organism_group, sid))
            except ValueError as exc:
                failures.append({"id": sid, "error": str(exc)})
            progress(i)
        summary = sorted((service.summarize(r) for r in results), key=lambda s: -s["attention"])
        return {
            "mode": "batch",
            "results": results,
            "summary": summary,
            "landscape": service.landscape(results),
            "failures": failures,
        }

    return _accepted(jobs.submit("batch", len(recs), run), len(recs))


@app.post("/demo/scan")
def post_scan(body: ScanIn):
    _check_group(body.organism_group)

    def run(progress, cancelled):
        out = service.scan(
            body.sequence, body.organism_group, body.target, body.mutations, body.substitute
        )
        progress(1)
        return {"mode": "scan", "scan": out}

    return _accepted(jobs.submit("scan", 1, run), 1)


@app.get("/demo/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "Unknown job_id")
    if jobs.expired(j):
        raise HTTPException(410, "Result expired")
    body = {k: v for k, v in j.items() if k != "result"}
    body["age_seconds"] = round(time.time() - j["created_at"], 1)
    if j["status"] == "completed":
        body.update(j["result"])
    body.update(service.provenance())
    return body


@app.post("/demo/jobs/{job_id}/cancel")
def post_cancel(job_id: str):
    j = jobs.cancel(job_id)
    if j is None:
        raise HTTPException(404, "Unknown job_id")
    return {"job_id": job_id, "status": j["status"], **service.provenance()}
