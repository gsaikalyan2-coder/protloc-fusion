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
import contextlib
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from protloc.config.serving import get_serving_settings
from protloc.domain.errors import (
    ERRORS,
    MAX_REQUEST_CHARS,
    MAX_SEQUENCE_LENGTH,
    ApiError,
    api_code_for_refusal,
    error_object,
)
from protloc.inference import catalog, export, service, structures
from protloc.inference.engine import CONFIG_FILE, EngineRefusal, load_engine
from protloc.inference.fasta import parse_fasta_strict
from protloc.inference.trust import trust_payload
from protloc.jobs import queue as jobs
from protloc.utils.observability import (
    ACCESS_LOG,
    METRICS,
    REQUEST_ID_HEADER,
    UNMATCHED,
    configure_logging,
    new_request_id,
    request_id_var,
)

ENABLED = os.getenv("PROTLOC_DEMO_ENABLED", "1") != "0"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load and verify the model before the first request; a failure stops the process."""
    settings = get_serving_settings()
    configure_logging(settings.LOG_LEVEL)
    METRICS.reset()
    service.configure(load_engine(settings))
    service.configure_cache(settings.RESULT_CACHE_ENTRIES)
    catalog.configure(
        catalog.load_for(settings.DATA_ROOT, CONFIG_FILE, service.engine().has_states)
    )
    jobs.configure(settings.JOB_RETENTION_HOURS)
    yield
    service.configure(None)
    catalog.configure(catalog.unavailable("the service has stopped"))


app = FastAPI(title="ProtLoc Fusion", version="1.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DEMO_CORS", "http://localhost:5174,http://127.0.0.1:5174").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(error: ApiError) -> JSONResponse:
    """The structured error body; provenance is added only when an engine is loaded."""
    body = error.body()
    with contextlib.suppress(Exception):  # no engine yet (startup) must not hide the error
        body.update(service.provenance())
    return JSONResponse(body, status_code=error.spec.status)


@app.exception_handler(ApiError)
async def api_error_handler(_request: Request, error: ApiError) -> JSONResponse:
    """Every catalogued failure answers with its status and a stable code."""
    return _error_response(error)


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, error: RequestValidationError) -> JSONResponse:
    """Schema failures keep FastAPI's ``detail`` list and gain a stable code."""
    items = error.errors()
    kinds = {str(i.get("type")) for i in items}
    last = [str(i.get("loc", ("",))[-1]) for i in items]
    if "string_too_long" in kinds and "sequence" in last:
        code = "sequence_too_long"
    elif "string_too_long" in kinds:
        code = "request_too_large"
    else:
        code = "invalid_input"
    spec = ERRORS[code]
    text = "; ".join(
        f"{'.'.join(str(p) for p in i.get('loc', ()) if p != 'body')}: {i.get('msg')}"
        for i in items
    )
    body: dict[str, Any] = {
        "detail": [
            {"loc": list(i.get("loc", ())), "msg": str(i.get("msg")), "type": str(i.get("type"))}
            for i in items
        ],
        "error": error_object(spec, text),
    }
    return JSONResponse(body, status_code=spec.status)


@app.middleware("http")
async def kill_switch(request: Request, call_next: Any) -> Any:
    """Return 503 on every /demo route when the kill switch is set."""
    if not ENABLED and request.url.path.startswith("/demo"):
        return _error_response(ApiError("demo_disabled", "Disabled (PROTLOC_DEMO_ENABLED=0)"))
    return await call_next(request)


_POLL_ROUTE = "/demo/jobs/{job_id}"


def _route_of(request: Request) -> str:
    """The matched route template (``/demo/jobs/{job_id}``), never the raw path with its ids."""
    route = request.scope.get("route")
    return str(getattr(route, "path", None) or UNMATCHED)


@app.middleware("http")
async def observe(request: Request, call_next: Any) -> Any:
    """Give every request an id, time it, count it and write one JSON access line.

    Registered after the kill switch, so it is the outermost middleware and also sees the 503
    the kill switch returns. Job polls that answer 200 log at DEBUG: a page polling every
    350 ms would otherwise drown every other line.
    """
    request_id = new_request_id(request.headers.get(REQUEST_ID_HEADER))
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
    except Exception:
        ACCESS_LOG.exception(
            "unhandled error",
            extra={"fields": {"method": request.method, "route": _route_of(request)}},
        )
        raise
    else:
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - started
        route = _route_of(request)
        METRICS.record_request(request.method, route, status, elapsed)
        quiet = request.method == "GET" and route == _POLL_ROUTE and status == 200  # noqa: PLR2004
        ACCESS_LOG.log(
            logging.DEBUG if quiet else logging.INFO,
            "request",
            extra={
                "fields": {
                    "method": request.method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 1),
                    "job_id": request.path_params.get("job_id"),
                }
            },
        )
        request_id_var.reset(token)


class PredictIn(BaseModel):
    """One sequence."""

    sequence: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
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

    fasta: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    organism_group: str = "eukaryote"


class ScanIn(BaseModel):
    """An in-silico scan (motif engine)."""

    sequence: str = Field(min_length=1, max_length=5000)
    organism_group: str = "eukaryote"
    target: str | None = None
    mutations: str = ""
    substitute: str = "A"


def _failure(record_id: str, refusal: EngineRefusal) -> dict[str, Any]:
    """One refused record of a batch, with its stable code."""
    spec = ERRORS[api_code_for_refusal(refusal.code)]
    return {
        "id": record_id,
        "error": str(refusal),
        "code": spec.code,
        "retryable": spec.retryable,
    }


def _degraded(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag a job whose evidence search was unavailable: the predictions are unchanged."""
    reasons = sorted(
        {
            str(r["evidence_status"]["unavailable_reason"])
            for r in results
            if not r["evidence_status"]["available"]
        }
    )
    return {"degraded": bool(reasons), "degraded_reasons": reasons}


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
        raise ApiError("unknown_group", f"Unknown organism_group '{group}'")


def _check_length(sequence: str) -> None:
    """Refuse an over-long sequence before it is queued (whitespace is not counted)."""
    length = len("".join(sequence.split()))
    if length > MAX_SEQUENCE_LENGTH:
        raise ApiError(
            "sequence_too_long",
            f"The sequence has {length} residues; the limit is {MAX_SEQUENCE_LENGTH}.",
        )


def _check_served(group: str) -> None:
    _check_group(group)
    if group not in service.engine().groups:
        covered = ", ".join(service.engine().groups)
        raise ApiError(
            "group_not_served",
            f"No trained head covers '{group}': the training corpus is eukaryotic "
            f"(served groups: {covered}).",
        )


def _check_cached(sequence: str) -> None:
    if not service.engine().has_states(sequence):
        raise ApiError(
            "not_cached",
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
        "degraded": not service.engine().evidence_state["available"],
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
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "error_codes": {
            code: {"status": s.status, "retryable": s.retryable} for code, s in ERRORS.items()
        },
        "evidence": engine.evidence_state,
        "rules": service.engine().rules,
        "encoder": engine.provenance.get("encoder"),
        "limits": [
            (
                f"Sequences outside the {engine.cached_sequences} cached ones are encoded live by "
                "frozen ESM-2 (ADR-0019) and marked outside the evaluated subset: no reported "
                "number was measured on sequences like them."
                if engine.live_encoder is not None
                else f"Only the {engine.cached_sequences} sequences whose ESM-2 states are cached "
                "can be scored; the gallery lists examples (ADR-0016)."
            ),
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


@app.get("/demo/catalog")
def list_catalog(  # noqa: PLR0913, PLR0917 - one query parameter per filter the screen offers
    q: str = Query("", max_length=100),
    location: str | None = Query(None, max_length=64),
    group: str | None = Query(None, max_length=64),
    partition: int | None = Query(None, ge=1, le=5),
    limit: int = Query(25, ge=1, le=catalog.MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Every cached sequence the engine can score, filtered and paged (no sequences)."""
    if group is not None:
        _check_group(group)
    current = catalog.current()
    page = current.search(
        q, location=location, group=group, partition=partition, limit=limit, offset=offset
    )
    return {
        **page,
        "catalog": current.state(),
        "locations": current.locations(),
        "provenance": service.provenance(),
    }


@app.get("/demo/catalog/{key}")
def get_catalog_entry(key: str) -> dict[str, Any]:
    """One cached sequence, with its sequence, by accession or entry name."""
    current = catalog.current()
    if not current.available:
        raise ApiError("catalog_unavailable", f"The catalog is not loaded: {current.reason}")
    entry = current.lookup(key[:64])
    if entry is None:
        raise ApiError(
            "accession_not_cached",
            f"No cached sequence has the accession or entry name '{key[:64]}'. This build scores "
            f"only the {service.engine().cached_sequences} cached sequences (ADR-0016).",
        )
    return {**entry, "provenance": service.provenance()}


@app.get("/demo/metrics")
def metrics() -> dict[str, Any]:
    """Request and job counters, recent latency percentiles and the queue, for this process.

    In-memory and per-process like the queue (ADR-0008): a restart starts every counter at
    zero. Latency percentiles are over the most recent requests per route, not all time.
    """
    return {
        **METRICS.snapshot(),
        "queue": {"depth": jobs.depth(), "pending_sequences": jobs.pending_sequences()},
        "result_cache": service.cache_stats(),
        "live_encoder": service.engine().live_encoder,
        "evidence": service.engine().evidence_state,
        "catalog": catalog.current().state(),
        **service.provenance(),
    }


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
        raise ApiError("structure_bad_hash", "sequence_sha256 must be 64 lowercase hex characters")
    store = structures.store_for(get_serving_settings().DATA_ROOT / "structures")
    try:
        document = store.get(sequence_sha256)
    except structures.StructureError as exc:
        raise ApiError("structure_refused", f"Structure refused: {exc}") from exc
    if document is None:
        raise ApiError(
            "structure_not_found",
            "No structure for this sequence: AlphaFold DB has no model for it, or its sequence "
            "differs from the one scored here.",
        )
    return {**document, "note": _STRUCTURE_NOTE}


@app.post("/demo/predict")
def post_predict(body: PredictIn) -> JSONResponse:
    """Queue one sequence (a job of size one)."""
    _check_served(body.organism_group)
    _check_length(body.sequence)
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
            **_degraded([r]),
        }

    return _accepted(jobs.submit("predict", 1, run), 1, ahead)


@app.post("/demo/batch")
def post_batch(body: BatchIn) -> JSONResponse:
    """Queue a FASTA of up to MAX_BATCH_SEQUENCES sequences (client-side chunking, ADR-0008)."""
    _check_served(body.organism_group)
    recs = parse_fasta_strict(body.fasta, _max_batch(), MAX_SEQUENCE_LENGTH)
    ahead = jobs.pending_sequences()

    def run(progress: Any, cancelled: Any) -> dict[str, Any]:
        if cancelled():
            return {
                "mode": "batch",
                "results": [],
                "summary": [],
                "landscape": {},
                "failures": [],
                **_degraded([]),
            }
        outcome = service.predict_many(recs, body.organism_group)
        results = [o for o in outcome if not isinstance(o, EngineRefusal)]
        failures = [
            _failure(recs[i][0], o) for i, o in enumerate(outcome) if isinstance(o, EngineRefusal)
        ]
        progress(len(recs))
        summary = sorted((service.summarize(r) for r in results), key=lambda s: -s["attention"])
        return {
            "mode": "batch",
            "results": results,
            "summary": summary,
            "landscape": service.landscape(results),
            "failures": failures,
            **_degraded(results),
        }

    return _accepted(jobs.submit("batch", len(recs), run), len(recs), ahead)


@app.post("/demo/scan")
def post_scan(body: ScanIn) -> JSONResponse:
    """Queue an in-silico scan. The scanner is the motif engine and says so."""
    _check_group(body.organism_group)
    _check_length(body.sequence)
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
        raise ApiError("unknown_job", "Unknown job_id")
    if jobs.expired(j):
        raise ApiError("result_expired", "Result expired")
    body = {k: v for k, v in j.items() if k != "result"}
    body["age_seconds"] = round(time.time() - j["created_at"], 1)
    if j["status"] == "failed":
        spec = ERRORS[str(j.get("error_code") or "job_failed")]
        body["error_info"] = error_object(spec, str(j["error"]))
    if j["status"] == "completed":
        body.update(j["result"])
    body.update(service.provenance())
    return body


@app.get("/demo/jobs/{job_id}/results")
def download_results(
    job_id: str,
    format: str = Query("json", pattern="^(json|csv|fasta)$"),  # noqa: A002
) -> Response:
    """The whole finished job as a file: JSON, one-row-per-sequence CSV, or FASTA."""
    j = jobs.get(job_id)
    if j is None:
        raise ApiError("unknown_job", "Unknown job_id")
    if jobs.expired(j):
        raise ApiError("result_expired", "Result expired")
    if j["status"] in ("queued", "running"):
        raise ApiError("job_not_finished", f"The job is {j['status']}; download it once completed.")
    if j["status"] != "completed" or not isinstance(j.get("result"), dict):
        raise ApiError("no_results", f"The job is {j['status']}; it has no results to download.")
    merged = {**{k: v for k, v in j.items() if k != "result"}, **j["result"]}
    mode = str(merged.get("mode", ""))
    if not export.supports(mode, format):
        raise ApiError(
            "export_unsupported", f"A {mode} job exports as JSON only, not {format.upper()}."
        )
    media_type, _ = export.FORMATS[format]
    return Response(
        export.build(merged, service.provenance(), format),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{export.filename(job_id, format)}"'
        },
    )


@app.post("/demo/jobs/{job_id}/cancel")
def post_cancel(job_id: str) -> dict[str, Any]:
    """Cancel a queued or running job."""
    j = jobs.cancel(job_id)
    if j is None:
        raise ApiError("unknown_job", "Unknown job_id")
    return {"job_id": job_id, "status": j["status"], **service.provenance()}
