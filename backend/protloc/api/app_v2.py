"""ProtLoc Fusion — the live app (``/demo/*`` routes the frontend uses).

Run (from ``backend/``):  ``uvicorn protloc.api.app_v2:app --port 8010``

The routes keep the names and the 202-then-poll shape the five screens already speak; since
Phase 14 the engine behind them is the trained one. Startup loads the model registry through
:func:`protloc.inference.engine.load_engine` and refuses to start, listing every problem, if a
bundle, weight file, manifest, cache, ontology, hit-term table or index byte is not the one the
bundles record. Routers parse, validate and delegate; no scoring happens here.

Kill switch: ``PROTLOC_DEMO_ENABLED=0`` makes every ``/demo`` route return 503.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from protloc.config.serving import get_serving_settings
from protloc.inference import service, structures
from protloc.inference.engine import EngineRefusal, load_engine
from protloc.inference.trust import trust_payload
from protloc.jobs import queue as jobs

ENABLED = os.getenv("PROTLOC_DEMO_ENABLED", "1") != "0"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load and verify the model before the first request; a failure stops the process."""
    settings = get_serving_settings()
    service.configure(load_engine(settings))
    jobs.configure(settings.JOB_RETENTION_HOURS)
    yield
    service.configure(None)


app = FastAPI(title="ProtLoc Fusion", version="1.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DEMO_CORS", "http://localhost:5174,http://127.0.0.1:5174").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def kill_switch(request: Request, call_next: Any) -> Any:
    """Return 503 on every /demo route when the kill switch is set."""
    if not ENABLED and request.url.path.startswith("/demo"):
        return JSONResponse(
            {"detail": "Disabled (PROTLOC_DEMO_ENABLED=0)", **service.provenance()}, 503
        )
    return await call_next(request)


class PredictIn(BaseModel):
    """One sequence."""

    sequence: str = Field(min_length=1, max_length=20000)
    organism_group: str = "eukaryote"
    id: str = "query"


class StructureOut(BaseModel):
    """An AlphaFold backbone for one cached sequence: C-alpha points and per-residue pLDDT."""

    sequence_sha256: str
    accession: str
    source: str
    license: str
    entry_id: str | None = None
    model_version: int | str | None = None
    model_created: str | None = None
    length: int
    ca: list[list[float]]
    plddt: list[float]
    note: str


class BatchIn(BaseModel):
    """A FASTA text."""

    fasta: str = Field(min_length=1, max_length=2_000_000)
    organism_group: str = "eukaryote"


class ScanIn(BaseModel):
    """An in-silico scan (motif engine)."""

    sequence: str = Field(min_length=1, max_length=5000)
    organism_group: str = "eukaryote"
    target: str | None = None
    mutations: str = ""
    substitute: str = "A"


def _max_batch() -> int:
    return get_serving_settings().MAX_BATCH_SEQUENCES


def _accepted(sub: dict[str, Any], count: int, ahead: int) -> JSONResponse:
    body = {
        **sub,
        "result_url": f"/demo/jobs/{sub['job_id']}",
        "sequence_count": count,
        "estimated_wait_seconds": jobs.estimated_wait_seconds(ahead, count),
        **service.provenance(),
    }
    return JSONResponse(body, status_code=202)


def _check_group(group: str) -> None:
    if group not in service.ORGANISM_GROUPS:
        raise HTTPException(400, f"Unknown organism_group '{group}'")


def _check_served(group: str) -> None:
    _check_group(group)
    if group not in service.engine().groups:
        covered = ", ".join(service.engine().groups)
        raise HTTPException(
            422,
            f"No trained head covers '{group}': the training corpus is eukaryotic "
            f"(served groups: {covered}).",
        )


def _check_cached(sequence: str) -> None:
    if not service.engine().has_states(sequence):
        raise HTTPException(
            422,
            "This build scores only sequences whose ESM-2 states are cached "
            f"({service.engine().cached_sequences} of them, ADR-0016). Pick a gallery example.",
        )


@app.get("/demo/health")
def health() -> dict[str, Any]:
    """Liveness, queue depth and the evidence state."""
    return {
        "status": "ok",
        "enabled": ENABLED,
        "queue_depth": jobs.depth(),
        "job_retention_seconds": jobs.RETENTION_SECONDS,
        "evidence": service.engine().evidence_state,
        **service.provenance(),
    }


def _banner(provenance: dict[str, Any]) -> str:
    return (
        f"Model {provenance['model_version']} · trained heads on frozen ESM-2 150M · "
        f"{provenance['scope_label']}"
    )


@app.get("/demo/version")
def version() -> dict[str, Any]:
    """Model identity, the groups served, limits and the banner text."""
    engine = service.engine()
    p = service.provenance()
    return {
        **p,
        "banner": _banner(p),
        "organism_groups": {
            k: {"label": v["label"], "locations": v["locs"], "served": k in engine.groups}
            for k, v in service.ORGANISM_GROUPS.items()
        },
        "location_names": service.LOC_NAMES,
        "tau": None,
        "temperature": None,
        "threshold_policy": "per_label_mcc",
        "calibration_version": engine.manifests["localization"]["frozen"]["bundle"][
            "calibration_version"
        ],
        "max_batch": _max_batch(),
        "evidence": engine.evidence_state,
        "rules": service.engine().rules,
        "limits": [
            f"Only the {engine.cached_sequences} sequences whose ESM-2 states are cached can be "
            "scored; the gallery lists examples (ADR-0016).",
            "Numbers are from a 500-sequence subset, not comparable to DeepLoc or to the "
            "Phase 8 full-corpus numbers.",
            "Membrane type and the scanner are motif-engine outputs and say so. The residue "
            "attribution is computed from the trained localization head (method named in the "
            "result); the hydropathy track is a motif output.",
        ],
    }


@app.get("/demo/examples")
def examples() -> dict[str, Any]:
    """Gallery sequences: real Swiss-Prot entries whose states are cached."""
    return {"examples": service.engine().gallery, "provenance": service.provenance()}


@app.get("/demo/trust")
def trust() -> dict[str, Any]:
    """The frozen calibration of the localization head, read from its manifest."""
    engine = service.engine()
    return trust_payload(
        engine.manifests["localization"], engine.localization_codes, service.provenance()
    )


_STRUCTURE_NOTE = (
    "AlphaFold DB prediction, shown for orientation. It is not an experimental structure, not "
    "this model's output, and it never feeds a score."
)


@app.get("/demo/structure/{sequence_sha256}", response_model=StructureOut)
def get_structure(sequence_sha256: str) -> dict[str, Any]:
    """The AlphaFold backbone of a cached sequence, served only if its bytes match the manifest."""
    if not structures.is_sha256(sequence_sha256):
        raise HTTPException(400, "sequence_sha256 must be 64 lowercase hex characters")
    store = structures.store_for(get_serving_settings().DATA_ROOT / "structures")
    try:
        document = store.get(sequence_sha256)
    except structures.StructureError as exc:
        raise HTTPException(500, f"Structure refused: {exc}") from exc
    if document is None:
        raise HTTPException(
            404,
            "No structure for this sequence: AlphaFold DB has no model for it, or its sequence "
            "differs from the one scored here.",
        )
    return {**document, "note": _STRUCTURE_NOTE}


@app.post("/demo/predict")
def post_predict(body: PredictIn) -> JSONResponse:
    """Queue one sequence (a job of size one)."""
    _check_served(body.organism_group)
    _check_cached(body.sequence)
    ahead = jobs.pending_sequences()

    def run(progress: Any, cancelled: Any) -> dict[str, Any]:
        r = service.predict(body.sequence, body.organism_group, body.id)
        progress(1)
        return {
            "mode": "predict",
            "results": [r],
            "summary": [service.summarize(r)],
            "landscape": service.landscape([r]),
        }

    return _accepted(jobs.submit("predict", 1, run), 1, ahead)


@app.post("/demo/batch")
def post_batch(body: BatchIn) -> JSONResponse:
    """Queue a FASTA of up to MAX_BATCH_SEQUENCES sequences (client-side chunking, ADR-0008)."""
    _check_served(body.organism_group)
    try:
        recs = service.parse_fasta(body.fasta, _max_batch())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ahead = jobs.pending_sequences()

    def run(progress: Any, cancelled: Any) -> dict[str, Any]:
        if cancelled():
            return {"mode": "batch", "results": [], "summary": [], "landscape": {}, "failures": []}
        outcome = service.predict_many(recs, body.organism_group)
        results = [o for o in outcome if not isinstance(o, EngineRefusal)]
        failures = [
            {"id": recs[i][0], "error": str(o)}
            for i, o in enumerate(outcome)
            if isinstance(o, EngineRefusal)
        ]
        progress(len(recs))
        summary = sorted((service.summarize(r) for r in results), key=lambda s: -s["attention"])
        return {
            "mode": "batch",
            "results": results,
            "summary": summary,
            "landscape": service.landscape(results),
            "failures": failures,
        }

    return _accepted(jobs.submit("batch", len(recs), run), len(recs), ahead)


@app.post("/demo/scan")
def post_scan(body: ScanIn) -> JSONResponse:
    """Queue an in-silico scan. The scanner is the motif engine and says so."""
    _check_group(body.organism_group)
    ahead = jobs.pending_sequences()

    def run(progress: Any, cancelled: Any) -> dict[str, Any]:
        out = service.scan(
            body.sequence, body.organism_group, body.target, body.mutations, body.substitute
        )
        progress(1)
        return {"mode": "scan", "scan": out}

    return _accepted(jobs.submit("scan", 1, run), 1, ahead)


@app.get("/demo/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Status, timing and (when completed) the results."""
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
def post_cancel(job_id: str) -> dict[str, Any]:
    """Cancel a queued or running job."""
    j = jobs.cancel(job_id)
    if j is None:
        raise HTTPException(404, "Unknown job_id")
    return {"job_id": job_id, "status": j["status"], **service.provenance()}
