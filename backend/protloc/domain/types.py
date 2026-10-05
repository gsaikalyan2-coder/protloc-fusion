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

    The three levels come from ``docs/DESIGN.md`` §3.3 and mean different
    things: ``note`` and ``warning`` flag a prediction without altering it,
    while ``blocking`` forces abstention on the affected task. A violation never
    silently edits a prediction — the user sees what the model said and, beside
    it, what the rule set thinks of it. The rules themselves and their
    severities live in ``backend/rules/consistency_rules.yaml``, not in Python.
    """

    NOTE = "note"
    WARNING = "warning"
    BLOCKING = "blocking"


@unique
class LabelState(StrEnum):
    """What the corpus knows about one task for one protein.

    Four states, not two, because three different kinds of "no positive label"
    mean three different things to the multi-task loss (ADR-0006 §3.1):

    * :attr:`NEGATIVE` — curated absence, under the task's rule in ADR-0006 §3.3.
      It is a training signal.
    * :attr:`UNKNOWN` — nobody annotated this. It is not a training signal, and
      it must never be read as a negative.
    * :attr:`NOT_APPLICABLE` — the task is not defined for this organism group.
      It is masked by design, and counted separately from unknown in every
      coverage table.
    """

    ANNOTATED = "annotated"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@unique
class EngineSource(StrEnum):
    """Which engine produced a prediction-bearing field.

    The five screens show this rather than hide it: a model probability and a
    motif score are not the same kind of number (``docs/phases/03.md`` §1). Phases
    9-14 move each field from :attr:`MOTIF` to :attr:`MODEL` without a schema change.
    """

    MOTIF = "motif"
    MODEL = "model"


@unique
class AttributionMethod(StrEnum):
    """How a per-residue attribution track was computed (Phase 15, ADR-0017).

    Each method makes a different claim, so the wire carries the method beside the source and
    the claim is in :data:`ATTRIBUTION_CLAIMS`, the one place it is written.
    """

    POOLING_ATTENTION = "pooling_attention"
    OCCLUSION = "occlusion"
    CRF_MARGINAL = "crf_marginal"
    MOTIF = "motif"


ATTRIBUTION_CLAIMS: dict[AttributionMethod, str] = {
    AttributionMethod.POOLING_ATTENTION: (
        "Where the head's pooled vector looked, as a share of the total (sums to 1). The same "
        "for every label: it shows what the head attended to, not why a given label won."
    ),
    AttributionMethod.OCCLUSION: (
        "How much this label's raw score fell when a window of residues was removed from the "
        "head's pooling input (signed: positive supports the label, negative argues against "
        "it). Computed on the head's input only: the frozen encoder is not re-run, so context "
        "already encoded in neighbouring residues remains. Not causal importance, not a "
        "binding site."
    ),
    AttributionMethod.CRF_MARGINAL: (
        "Probability, from the CRF forward-backward marginals, that the residue is part of the "
        "N-terminal signal peptide (0 to 1). Only the residues up to the CRF's length cap are "
        "scored; later residues read 0.0 because the model did not look at them."
    ),
    AttributionMethod.MOTIF: (
        "A heuristic weight painted from detected motifs (hydrophobic region, nuclear "
        "localization signal, transmembrane segments). Not computed by a trained model."
    ),
}


@unique
class ConfidenceBand(StrEnum):
    """A coarse reading of how far a probability clears its threshold.

    :attr:`NONE` means below the threshold: the system is abstaining on that
    label. The band styles a row; the probability stays the datum
    (``docs/DESIGN.md``), because a band reads as more certain than the number
    behind it.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@unique
class ThresholdPolicy(StrEnum):
    """How the decision thresholds in a result were chosen.

    :attr:`FIXED` is one threshold for every label. :attr:`PER_LABEL_MCC` is one
    threshold per label, tuned on a calibration split and frozen in the registry
    bundle (Phase 10).
    """

    FIXED = "fixed"
    PER_LABEL_MCC = "per_label_mcc"


__all__ = [
    "ConfidenceBand",
    "EngineSource",
    "JobStatus",
    "LabelState",
    "OrganismGroup",
    "Severity",
    "TaskName",
    "ThresholdPolicy",
]
