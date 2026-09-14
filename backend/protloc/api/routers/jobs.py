"""Job endpoints: following work, stopping it, and reading what it produced.

Two of these return results and they are not interchangeable — the distinction is
stated here because the frozen contract must let a reviewer predict which one to
call without reading any code.

``GET /jobs/{job_id}`` is the job *resource*. It answers ``200`` at every status,
carries the results inline once ``completed``, and is what ``result_url`` points
at and what the frontend polls.

``GET /jobs/{job_id}/results`` is the result *representation*, for download. It
takes a format, sets ``Content-Disposition``, refuses a job that has not completed
rather than returning a partial one, and is never polled.
"""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, status
from fastapi.responses import JSONResponse

from protloc.api.deps import job_service
from protloc.domain.contracts import CancelResponse, ErrorResponse, JobResource
from protloc.domain.ports import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])

JobServiceDep = Annotated[JobService, Depends(job_service)]
JobIdPath = Annotated[UUID, Path(description="The job's UUIDv4, as returned at acceptance.")]

_UNAVAILABLE: dict[str, object] = {
    "model": ErrorResponse,
    "description": "No registry bundle is loaded. `error` is `no_registry_bundle`.",
}
_NOT_FOUND: dict[str, object] = {
    "model": ErrorResponse,
    "description": "No such job. `error` is `not_found`.",
}
_EXPIRED: dict[str, object] = {
    "model": ErrorResponse,
    "description": (
        "The job existed and its retention window closed. `error` is `job_expired`. "
        "Distinct from 404 deliberately: a user holding an old URL is told it "
        "expired, never that it never existed."
    ),
}

JOB_ERRORS: dict[int | str, dict[str, object]] = {
    status.HTTP_404_NOT_FOUND: _NOT_FOUND,
    status.HTTP_410_GONE: _EXPIRED,
    status.HTTP_503_SERVICE_UNAVAILABLE: _UNAVAILABLE,
}


@router.get(
    "/{job_id}",
    response_model=JobResource,
    responses=JOB_ERRORS,
    summary="The job, in whatever state it is in",
    response_description=(
        "The job resource. `results` is populated only when `status` is `completed`."
    ),
)
def read_job(job_id: JobIdPath, jobs: JobServiceDep) -> JobResource:
    """Return the job.

    This is the URL in ``result_url``. It is safe to poll: it answers ``200`` for
    a queued, running, completed, failed or cancelled job alike, and the client
    reads ``status`` rather than branching on a status code.
    """
    return jobs.read(job_id)


@router.post(
    "/{job_id}/cancel",
    response_model=CancelResponse,
    responses=JOB_ERRORS,
    summary="Stop a queued or running job",
    response_description="`cancelled` says whether this call changed anything.",
)
def cancel_job(job_id: JobIdPath, jobs: JobServiceDep) -> CancelResponse:
    """Cancel the job.

    Cancelling a job that already reached a terminal state is not an error: the
    caller's intent — that this job not run — is already satisfied, and returning
    ``409`` for it would make a retry after a network failure look like a bug.
    """
    return jobs.cancel(job_id)


@router.get(
    "/{job_id}/results",
    response_model=None,
    responses={
        **JOB_ERRORS,
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": (
                "The job has not completed, so it has no results. "
                "`error` is `results_not_ready`. Poll `GET /jobs/{job_id}` instead."
            ),
        },
        status.HTTP_200_OK: {
            "description": "The results, as an attachment.",
            "content": {
                "application/json": {},
                "text/csv": {},
            },
        },
    },
    summary="Download a completed job's results",
    response_description="An attachment in the requested format.",
)
def download_results(
    job_id: JobIdPath,
    jobs: JobServiceDep,
    fmt: Annotated[
        Literal["json", "csv"],
        Query(
            alias="format",
            description=(
                "json preserves the full nested result; csv flattens one row per "
                "sequence and task, and necessarily omits per-residue attribution."
            ),
        ),
    ] = "json",
) -> JSONResponse:
    """Return the results as a download.

    Never returns a partial result: a job that has not completed is a ``409``,
    because a truncated download that looks complete is worse than a refusal.
    """
    results = jobs.results(job_id, fmt)
    return JSONResponse(
        content=[result.model_dump(mode="json") for result in results],
        headers={"Content-Disposition": f'attachment; filename="protloc-{job_id}.{fmt}"'},
    )


__all__ = ["router"]
