"""The FastAPI application.

This module assembles the service: it reads configuration, mounts the routers,
and installs the exception handlers that make every error body carry the
reproducibility envelope. It contains no business logic, and it is the only place
that knows the application exists as a whole.

``scripts/export_openapi.py`` imports :func:`create_app` and serialises
``app.openapi()`` to ``docs/api/openapi.yaml``. The document is therefore a
projection of this code, never a parallel description of it — which is why CI
regenerates and diffs it rather than reviewing it.

Run it with::

    uvicorn protloc.api.app:create_app --factory --host "$API_HOST" --port "$API_PORT"

Factory mode, not a module-level ``app``: building the application reads and
validates the settings, and a module-level instance would do that at *import*
time. That turns ``import protloc.api.app`` — which the OpenAPI export and every
test do — into an operation that fails on an incomplete environment. The factory
keeps configuration a startup concern, which is where ``settings.py`` already
puts it.
"""

from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from protloc.api import deps
from protloc.api.routers import examples, health, jobs, predict
from protloc.config.settings import Settings, get_settings
from protloc.domain.contracts import API_VERSION, ErrorResponse
from protloc.domain.ports import (
    JobExpiredError,
    JobNotFoundError,
    NoRegistryBundleError,
    ResultsNotReadyError,
)

_DESCRIPTION = """\
Unified protein annotation from one shared representation: one sequence and one
organism group in, seven annotations out, with calibrated probabilities,
abstention, biological-consistency flags, per-residue explanation and independent
database evidence.

**The contract is asynchronous.** `POST /predict` returns `202 Accepted` with a
job identifier and a result URL, never a prediction. A single sequence is a job
of size one; there is exactly one inference path, so the answer a user gets is
the answer the paper measures (ADR-0001).

**Every response carries provenance.** `model_version`, `config_hash`,
`rules_version` and `experiment_id` appear on every body, including acceptances
and errors. They are `null` when no registry bundle is loaded.

**Result URLs are unguessable but not private.** A `job_id` is a UUIDv4; anyone
holding the URL can read the result. Results expire after a retention window.
"""


def _error_body(
    request: Request,
    *,
    error: str,
    message: str,
    detail: str | None,
) -> dict[str, object]:
    """Build an error body carrying the reproducibility envelope.

    The envelope is read from the application's provenance source, honouring a
    test's dependency override, so an error raised under test reports the same
    provenance as the responses around it.
    """
    override = request.app.dependency_overrides.get(deps.provenance_source)
    source = override() if override is not None else deps.provenance_source()
    body = ErrorResponse(
        error=error,
        message=message,
        detail=detail,
        **source.envelope.model_dump(),
    )
    return body.model_dump()


def _service_version() -> str:
    """Return the installed package version, or a marker when not installed.

    An editable install always resolves; the fallback covers running from a
    source tree with no metadata, where reporting nothing is better than
    reporting a number that is not the one on disk.
    """
    try:
        return version("protloc")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "unknown"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration to build against. Read from the environment when
            omitted, which is what the server does. ``export_openapi.py`` passes
            an explicit instance instead, because the OpenAPI document must not
            depend on whose machine generated it — CORS origins and ports do not
            appear in the document, and requiring a filled ``.env`` to produce it
            would make the CI diff environment-sensitive.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="ProtLoc Fusion",
        summary="Seven protein annotations from one shared representation.",
        description=_DESCRIPTION,
        version=API_VERSION,
        license_info={"name": "Apache-2.0", "identifier": "Apache-2.0"},
        contact={
            "name": "ProtLoc Fusion",
            "url": "https://github.com/gsaikalyan2-coder/protloc-fusion",
        },
        openapi_tags=[
            {"name": "service", "description": "Liveness and what is loaded."},
            {"name": "prediction", "description": "Submitting work. Always asynchronous."},
            {"name": "jobs", "description": "Following work and reading results."},
            {"name": "examples", "description": "The gallery, with pre-computed results."},
        ],
    )

    # The API is unauthenticated and rate-limited by origin; the settings
    # validator already refuses a wildcard, so this list is always explicit.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(NoRegistryBundleError)
    async def _no_bundle(request: Request, exc: NoRegistryBundleError) -> JSONResponse:
        """Report that prediction is unavailable, and why.

        503 rather than 501: the endpoint exists and its contract is frozen; the
        dependency it needs is not loaded. That is a temporary condition by
        definition, and it resolves when Phase 7 packages a bundle.
        """
        return JSONResponse(
            status_code=503,
            content=_error_body(
                request,
                error="no_registry_bundle",
                message="The service cannot run predictions right now.",
                detail=str(exc),
            ),
        )

    @app.exception_handler(JobNotFoundError)
    async def _not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        """Report that no such job exists."""
        return JSONResponse(
            status_code=404,
            content=_error_body(
                request,
                error="not_found",
                message="No job with that identifier exists.",
                detail=str(exc) or None,
            ),
        )

    @app.exception_handler(JobExpiredError)
    async def _expired(request: Request, exc: JobExpiredError) -> JSONResponse:
        """Report that the job existed and its retention window closed.

        Separate from 404 on purpose. A user holding a URL from last month should
        be told the result expired, not that it never existed — the second
        reading invites them to doubt whether they ever ran the job.
        """
        return JSONResponse(
            status_code=410,
            content=_error_body(
                request,
                error="job_expired",
                message="The results for this job are past their retention window.",
                detail=str(exc) or None,
            ),
        )

    @app.exception_handler(ResultsNotReadyError)
    async def _not_ready(request: Request, exc: ResultsNotReadyError) -> JSONResponse:
        """Report that the job has not completed, so has no results to download."""
        return JSONResponse(
            status_code=409,
            content=_error_body(
                request,
                error="results_not_ready",
                message="This job has not completed; it has no results to download.",
                detail=str(exc) or "Poll GET /jobs/{job_id} for its status.",
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Report a malformed request, naming the fields that failed."""
        fields = ", ".join(
            ".".join(str(part) for part in error["loc"][1:]) or "<body>" for error in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content=_error_body(
                request,
                error="validation_failed",
                message="The request does not match the contract.",
                detail=f"Invalid or missing: {fields}",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Give every HTTP error the same body shape as every success.

        Without this, FastAPI's default ``{"detail": ...}`` would be the one
        response shape in the document that carries no provenance — and the
        guarantee would be a guarantee with an exception in it.
        """
        codes = {
            404: "not_found",
            405: "method_not_allowed",
            409: "results_not_ready",
            410: "job_expired",
            413: "payload_too_large",
            429: "rate_limited",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                request,
                error=codes.get(exc.status_code, "error"),
                message=str(exc.detail),
                detail=None,
            ),
            headers=getattr(exc, "headers", None),
        )

    app.state.service_version = _service_version()

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(jobs.router)
    app.include_router(examples.router)

    return app


__all__ = ["create_app"]
