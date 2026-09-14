"""Ports: what the inner layers promise the outer ones.

A port is an interface declared where it is *needed*, not where it is
implemented. ``api/`` needs to know a model version without knowing that versions
come from a directory on disk; ``inference/`` supplies that knowledge. Declaring
the protocol here, at depth 100, lets both depend on the shape and neither depend
on the other's internals — which is what keeps the routers thin and the layering
test green.

This is also the mechanism the Phase 1 handover flagged for Phase 10:
``interpretability`` may not import ``models``, so per-residue attribution receives
a scored model through a port declared in this module. The pattern starts here.
"""

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from protloc.domain.contracts import (
    CancelResponse,
    ExamplesResponse,
    JobAccepted,
    JobResource,
    PredictRequest,
    ReproducibilityEnvelope,
    SequenceResult,
)
from protloc.domain.types import OrganismGroup


class NoRegistryBundleError(RuntimeError):
    """Raised when an operation needs a registry bundle and none is loaded.

    Not an error condition in the usual sense — it is the ordinary state of the
    service between Phase 2, which freezes the contract, and Phase 7, which
    packages the first bundle. The API surfaces it as ``503`` with the error code
    ``no_registry_bundle``, which is a true statement about the service rather
    than a placeholder standing in for one.
    """


@runtime_checkable
class ProvenanceSource(Protocol):
    """Where the four reproducibility fields come from.

    One implementation exists today: :class:`protloc.inference.provenance.UnloadedProvenance`,
    which reports that nothing is loaded. Phase 7 adds a registry-backed one. The
    routers change in neither case, because they depend on this protocol and not
    on either class.
    """

    @property
    def bundle_loaded(self) -> bool:
        """Whether a registry bundle is loaded and usable for prediction."""
        ...

    @property
    def envelope(self) -> ReproducibilityEnvelope:
        """The four fields to attach to this response.

        Every field is ``None`` while no bundle is loaded. The envelope is always
        returned — never omitted — because its presence is the API guarantee and
        its nullity is the honest value.
        """
        ...

    @property
    def checksums(self) -> Mapping[str, str]:
        """SHA-256 of each artifact in the loaded bundle, by filename.

        Empty when no bundle is loaded.
        """
        ...

    def require_bundle(self) -> None:
        """Raise if no bundle is loaded.

        Raises:
            NoRegistryBundleError: When prediction is impossible because nothing
                is loaded. The message names the phase that supplies the bundle,
                so the failure is self-explaining rather than merely truthful.
        """
        ...


@runtime_checkable
class JobService(Protocol):
    """Submitting work and following it.

    Declared as one protocol rather than three because a job's lifecycle is one
    thing: whatever can accept a job can report on it and cancel it. Splitting it
    would let an implementation exist that accepts work it cannot describe.

    Every method may raise :class:`NoRegistryBundleError`, which the API renders
    as ``503``. Phase 11 supplies the queue-backed implementation; the routers do
    not change when it arrives, because they already depend only on this shape.
    """

    def submit_sequence(self, request: PredictRequest) -> JobAccepted:
        """Accept one sequence as a job of size one.

        Returns the acceptance, never a prediction (ADR-0001).
        """
        ...

    def submit_batch(self, organism_group: OrganismGroup, fasta: bytes) -> JobAccepted:
        """Accept a FASTA payload as one job."""
        ...

    def read(self, job_id: UUID) -> JobResource:
        """Return the job in whatever state it is in.

        Raises:
            JobNotFoundError: When no such job exists.
            JobExpiredError: When the job existed and its retention window closed.
                Distinct from not-found on purpose: "this expired" and "this never
                existed" are different things to tell a user holding a URL.
        """
        ...

    def cancel(self, job_id: UUID) -> CancelResponse:
        """Stop a queued or running job. Terminal jobs are not an error."""
        ...

    def results(self, job_id: UUID, fmt: Literal["json", "csv"]) -> list[SequenceResult]:
        """Return a completed job's results for download.

        Raises:
            ResultsNotReadyError: When the job has not completed. A partial
                result is never returned in its place.
        """
        ...


@runtime_checkable
class ExampleGallery(Protocol):
    """The pre-computed gallery.

    Exists because the asynchronous contract has no instant demo (ADR-0001 §5).
    Its results are cached and are labelled as cached, because a fast answer
    presented as freshly computed would misrepresent how the system behaves.
    """

    def examples(self) -> ExamplesResponse:
        """Return the gallery in display order."""
        ...


class JobNotFoundError(LookupError):
    """No job with that identifier exists. Rendered as ``404``."""


class JobExpiredError(LookupError):
    """The job existed and its retention window closed. Rendered as ``410``."""


class ResultsNotReadyError(RuntimeError):
    """The job has not completed, so it has no results. Rendered as ``409``."""


__all__ = [
    "ExampleGallery",
    "JobExpiredError",
    "JobNotFoundError",
    "JobService",
    "NoRegistryBundleError",
    "ProvenanceSource",
    "ResultsNotReadyError",
]
