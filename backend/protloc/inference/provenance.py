"""The provenance source the service runs with today.

``inference/`` is the conductor: it owns the order of the pipeline and none of the
algorithms. Knowing which model bundle is loaded is exactly that kind of
knowledge — it belongs to the layer that would run the pipeline, not to the layer
that answers HTTP.

There is one implementation here, and it reports that nothing is loaded. That is
not a stand-in for a real one; it is a complete and correct description of a
service with no registry bundle, which is what this service is until Phase 7
packages bundle v0. Every method is total: there is no branch that has not been
written, no value that has been guessed, and nothing that will need deleting
later. Phase 7 adds ``RegistryProvenance`` beside it and changes which one is
constructed — one line, in one place, in ``api/deps.py``.

The alternative was to read ``MODEL_REGISTRY`` now and report whatever a directory
happened to contain. That variable belongs to Phase 11 in ``CLAUDE.md``'s fill
schedule, and pulling it forward would mean a settings field, an ``.env.example``
regrouping and a filesystem dependency, all to discover the thing this class
already knows for certain.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, NoReturn
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
from protloc.domain.ports import NoRegistryBundleError
from protloc.domain.types import OrganismGroup

_NO_CHECKSUMS: Mapping[str, str] = MappingProxyType({})

_UNAVAILABLE = (
    "No registry bundle is loaded, so the service cannot run predictions. "
    "The first bundle is packaged in Phase 7 (vertical slice); the HTTP "
    "contract is frozen ahead of it in Phase 2 so that the frontend and the "
    "evaluation layer can be built against a stable shape."
)


class UnloadedProvenance:
    """Reports the absence of a registry bundle, accurately and without guessing.

    Satisfies :class:`protloc.domain.ports.ProvenanceSource`. Structural typing
    means this class does not import the protocol to implement it; the
    integration test asserts the conformance so that a drift in either direction
    fails a check rather than a deployment.
    """

    @property
    def bundle_loaded(self) -> bool:
        """Always ``False``: no bundle exists to load yet."""
        return False

    @property
    def envelope(self) -> ReproducibilityEnvelope:
        """The envelope, with every field ``None``.

        ``None`` rather than ``"unloaded"`` or ``"0.0.0"``: a client can test for
        ``None``, but cannot tell an invented version string from a real one, and
        a plausible-looking default is the failure mode ``PROJECT_PLAN.md`` §7.2
        exists to prevent.
        """
        return ReproducibilityEnvelope(
            model_version=None,
            config_hash=None,
            rules_version=None,
            experiment_id=None,
        )

    @property
    def checksums(self) -> Mapping[str, str]:
        """No artifacts, so no checksums."""
        return _NO_CHECKSUMS

    @property
    def unavailable_reason(self) -> str:
        """Why prediction is unavailable, in words a caller can act on."""
        return _UNAVAILABLE

    def require_bundle(self) -> None:
        """Always raise.

        Raises:
            NoRegistryBundleError: Always, because no bundle is loaded.
        """
        raise NoRegistryBundleError(_UNAVAILABLE)


class UnavailableJobService:
    """The job service of a system with no model: it accepts nothing, honestly.

    Satisfies both :class:`protloc.domain.ports.JobService` and
    :class:`protloc.domain.ports.ExampleGallery`. Every method raises the same
    error, and that is the complete and correct behaviour of a service with no
    registry bundle — there is no job to find because none can be created, and
    the gallery's entries are pre-computed *results*, which require the model
    that produced them.

    It is not a stub standing in for logic that should be here. The logic that is
    missing is the queue, and the queue is Phase 11's deliverable, listed as such
    in ``PROJECT_PLAN.md`` §4. What this phase owes is a frozen contract and a
    service that tells the truth about its own capability, which is what this is.
    """

    def _refuse(self) -> NoReturn:
        """Raise the one error this service can produce.

        Raises:
            NoRegistryBundleError: Always.
        """
        raise NoRegistryBundleError(_UNAVAILABLE)

    def submit_sequence(self, request: PredictRequest) -> JobAccepted:
        """Refuse the submission; no bundle is loaded."""
        self._refuse()

    def submit_batch(self, organism_group: OrganismGroup, fasta: bytes) -> JobAccepted:
        """Refuse the submission; no bundle is loaded."""
        self._refuse()

    def read(self, job_id: UUID) -> JobResource:
        """Refuse; no job can exist because none can be created."""
        self._refuse()

    def cancel(self, job_id: UUID) -> CancelResponse:
        """Refuse; no job can exist because none can be created."""
        self._refuse()

    def results(self, job_id: UUID, fmt: Literal["json", "csv"]) -> list[SequenceResult]:
        """Refuse; no job can exist because none can be created."""
        self._refuse()

    def examples(self) -> ExamplesResponse:
        """Refuse; the gallery's cached results require the model that made them."""
        self._refuse()


__all__ = ["UnavailableJobService", "UnloadedProvenance"]
