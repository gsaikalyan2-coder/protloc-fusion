"""Service endpoints: what is running, and what it has loaded.

These two are fully live. They are the endpoints that do not need a model, and
answering them truthfully today is more useful than answering them impressively
later: ``/health`` is what the container's HEALTHCHECK and the deployment probe
call, and ``degraded`` is a state this service genuinely occupies.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from protloc.api.deps import provenance_source
from protloc.domain.contracts import (
    API_VERSION,
    HealthResponse,
    HealthzResponse,
    VersionResponse,
)
from protloc.domain.ports import ProvenanceSource

router = APIRouter(tags=["service"])

ProvenanceDep = Annotated[ProvenanceSource, Depends(provenance_source)]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness, what is loaded, and how deep the queue is",
    response_description="The service is answering. Read `status` for whether it can predict.",
)
def health(provenance: ProvenanceDep) -> HealthResponse:
    """Report liveness.

    Answers ``200`` whenever the process is up, including when it cannot
    predict — a probe that fails while the service is correctly serving its
    contract would restart a container that is working. Inability to predict is
    reported in ``status``, not in the status code.
    """
    loaded = provenance.bundle_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        detail=None if loaded else "No registry bundle is loaded; prediction is unavailable.",
        bundle_loaded=loaded,
        # No workers run until Phase 11. Zero is the queue's true depth, not a
        # placeholder for a number that will exist later.
        queue_depth=0,
        **provenance.envelope.model_dump(),
    )


@router.get(
    "/healthz",
    response_model=HealthzResponse,
    summary="Model-registry state, for platform and phone probes",
    response_description="The service is answering. Read `registry_state` for the bundle.",
)
def healthz(provenance: ProvenanceDep) -> HealthzResponse:
    """Report the model-registry state.

    Answers ``200`` whenever the process is up, including with no bundle loaded;
    the registry's state is in the body, not the status code.
    """
    loaded = provenance.bundle_loaded
    return HealthzResponse(
        status="ok" if loaded else "degraded",
        registry_state="loaded" if loaded else "empty",
        detail=None if loaded else "No registry bundle is loaded; prediction is unavailable.",
        **provenance.envelope.model_dump(),
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Contract version, service version and bundle checksums",
    response_description="Everything needed to identify exactly what answered.",
)
def service_version(request: Request, provenance: ProvenanceDep) -> VersionResponse:
    """Report what is running.

    ``api_version`` is the contract's version and ``service_version`` the
    package's; they move independently, because a bug fix that changes no shape
    must not look like a contract change to a client pinning one.
    """
    return VersionResponse(
        api_version=API_VERSION,
        service_version=str(request.app.state.service_version),
        checksums=dict(provenance.checksums),
        **provenance.envelope.model_dump(),
    )


__all__ = ["router"]
