"""Submission endpoints. Both are asynchronous; neither returns a prediction.

``POST /predict`` and ``POST /batch`` differ only in how the sequences arrive.
One sequence is a job of size one (ADR-0001), which is why there is one inference
path behind both and one set of numbers for the paper to report.

Both delegate to the job service and return what it returns. They decide nothing:
today the service refuses, because no registry bundle is loaded, and the refusal
surfaces as ``503``. Phase 11 supplies a service that accepts, and neither
function here changes.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from protloc.api.deps import job_service
from protloc.domain.contracts import ErrorResponse, JobAccepted, PredictRequest
from protloc.domain.ports import JobService
from protloc.domain.types import OrganismGroup

router = APIRouter(tags=["prediction"])

JobServiceDep = Annotated[JobService, Depends(job_service)]

SUBMISSION_ERRORS: dict[int | str, dict[str, object]] = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The request does not match the contract. `error` is `validation_failed`.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "No registry bundle is loaded, so no work can be accepted. "
            "`error` is `no_registry_bundle`."
        ),
    },
}


@router.post(
    "/predict",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    responses=SUBMISSION_ERRORS,
    summary="Submit one sequence. Returns a job, never a prediction",
    response_description=(
        "The job was accepted. Poll `result_url` for status and, once completed, results."
    ),
)
def predict(request: PredictRequest, jobs: JobServiceDep) -> JobAccepted:
    """Accept one sequence as a job of size one.

    This endpoint does not return a prediction, and no future version of it will:
    that is the contract, not a property of the current implementation. A
    size-one job that finishes inside the inline window may be answered with
    ``200`` and a job resource as a convenience, but a client that assumes ``200``
    is broken against any deployment that happens to be busy.
    """
    return jobs.submit_sequence(request)


@router.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    responses=SUBMISSION_ERRORS,
    summary="Submit a FASTA file. Returns a job",
    response_description="The job was accepted. `sequence_count` reports how many were parsed.",
)
async def batch(
    jobs: JobServiceDep,
    organism_group: Annotated[
        OrganismGroup,
        Form(description="Applied to every sequence in the uploaded file."),
    ],
    file: Annotated[
        UploadFile,
        File(description="FASTA. Multipart rather than JSON: a proteome is tens of megabytes."),
    ],
) -> JobAccepted:
    """Accept a FASTA upload as one job.

    Async because reading the upload is I/O. The size cap is enforced by
    ``MAX_UPLOAD_BYTES`` in Phase 15, not by a constant here; parsing the FASTA
    belongs to the job service, which is why the bytes are handed over whole.
    """
    payload = await file.read()
    return jobs.submit_batch(organism_group, payload)


__all__ = ["router"]
