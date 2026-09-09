"""Vocabulary shared by every layer of ProtLoc Fusion.

These enumerations are the words the rest of the system speaks: the organism
groups a user may select, the seven task heads, the states a job passes through,
and the severity of a consistency violation. They are values, not behaviour, and
this module imports nothing from ``protloc``.

Each member's ``value`` is the wire form. It appears in API payloads, in dataset
manifests and in registry metadata, so a value is not renamed without a
migration: renaming one silently invalidates every stored job and manifest that
used the old string.
"""

from enum import StrEnum, unique


@unique
class OrganismGroup(StrEnum):
    """The organism group a user selects alongside the sequence.

    Sorting signals and localization compartments differ enough between these
    groups that the model is conditioned on the choice rather than asked to infer
    it. The four groups follow the convention established by the signal-peptide
    literature the project benchmarks against.
    """

    EUKARYA = "eukarya"
    GRAM_POSITIVE = "gram_positive"
    GRAM_NEGATIVE = "gram_negative"
    ARCHAEA = "archaea"

    @property
    def is_prokaryotic(self) -> bool:
        """Whether this group is prokaryotic.

        The prokaryotic localization head applies only to these groups; for
        :attr:`EUKARYA` it is masked rather than scored, which is a different
        thing from scoring it and abstaining.
        """
        return self is not OrganismGroup.EUKARYA


@unique
class TaskName(StrEnum):
    """The seven task heads sharing one trunk representation.

    Scope is fixed at seven by ``PROJECT_PLAN.md`` §3. A task is added by
    amending that section first, because each addition changes the multi-task
    loss, the label schema and the consistency rule set together.
    """

    LOCALIZATION = "localization"
    MEMBRANE_TYPE = "membrane_type"
    PROKARYOTIC_LOCALIZATION = "prokaryotic_localization"
    SIGNAL_PEPTIDE = "signal_peptide"
    MITOCHONDRIAL_TRANSIT_PEPTIDE = "mitochondrial_transit_peptide"
    CHLOROPLAST_TRANSIT_PEPTIDE = "chloroplast_transit_peptide"
    GO_FUNCTION = "go_function"


@unique
class JobStatus(StrEnum):
    """The lifecycle of a prediction job.

    Every prediction is a job; a single sequence is a job of size one
    (ADR-0001). A job leaves a terminal state only by being resubmitted as a new
    job — statuses are never rewound.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether no further status change is possible for this job."""
        return self in _TERMINAL_JOB_STATUSES

    @property
    def has_results(self) -> bool:
        """Whether results are available to read.

        Only :attr:`COMPLETED` carries results. A failed or cancelled job has an
        error or a cancellation reason, which is not a partial result and is not
        served as one.
        """
        return self is JobStatus.COMPLETED


_TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


@unique
class Severity(StrEnum):
    """How a biological-consistency violation is reported.

    Severity decides presentation and nothing else: a violation never silently
    edits a prediction. The user sees what the model said and, beside it, what
    the rule set thinks of it. The rules themselves and their severities live in
    ``backend/rules/consistency_rules.yaml``, not in Python.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


__all__ = ["JobStatus", "OrganismGroup", "Severity", "TaskName"]
