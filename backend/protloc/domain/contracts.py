"""The HTTP contract of ProtLoc Fusion, as Pydantic models.

This module is the single source of the API's shape. ``scripts/export_openapi.py``
serialises the application built from these models into
``docs/api/openapi.yaml``, and ``frontend/src/types/api.generated.ts`` is generated
from that document, so a field added here reaches the frontend's type checker
without anyone retyping it.

It lives in ``domain/`` rather than in ``api/`` deliberately. A schema is
vocabulary, not delivery: the job layer, the evaluation layer and the routers all
need to speak the same shapes, and ``domain/`` is the only module every one of them
may import. Putting the contract in ``api/`` would mean either duplicating it or
importing a delivery module from a deeper layer, and both are violations.

Three rules govern what may appear below.

**No biology.** Compartment names, membrane categories and GO identifiers are
values the registry bundle supplies, not literals in a Python file. Every label is
typed ``str``. A ``Literal["cytoplasm", "nucleus", ...]`` here would be a
biological constant in code, which ``CLAUDE.md`` calls a bug.

**No thresholds.** ``abstention_threshold`` is reported as whatever the calibrator
metadata held at prediction time. There is no default, because a default would be a
hardcoded tau. Phase 3's ``threshold`` follows the same rule: it has no literal
default either. When a payload from before Phase 3 omits it, it is read from that
payload's own ``abstention_threshold`` — the value the decision actually used.

**No invented values.** The four reproducibility fields are ``str | None`` and are
``None`` exactly when no registry bundle is loaded. A sentinel string would be
indistinguishable from a real version to any client reading it.
"""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from protloc.domain.types import (
    ATTRIBUTION_CLAIMS,
    AttributionMethod,
    ConfidenceBand,
    EngineSource,
    JobStatus,
    OrganismGroup,
    Severity,
    TaskName,
    ThresholdPolicy,
)

#: SemVer of the HTTP contract, independent of the service package's own version.
#: It changes when a client would have to change — a removed field, a renamed one,
#: a narrowed type. Adding an optional field does not change it. It lives here
#: rather than in ``api/`` because it versions the contract, and the contract is
#: what this module is.
API_VERSION = "1.3.0"

# Every model forbids unknown fields. A response a client cannot fully account for
# is a response whose shape the OpenAPI document does not actually describe, and
# this phase's exit criterion is that the document describes it exactly.
_STRICT = ConfigDict(extra="forbid")


class ReproducibilityEnvelope(BaseModel):
    """The four fields every response carries, without exception.

    ``CLAUDE.md`` contract rule 6 makes this an API guarantee rather than a
    frontend decoration: a prediction a user cannot trace back to a model version,
    a resolved config and an experiment is a prediction they cannot reproduce, and
    the paper's claim is reproducibility.

    Each field is ``None`` when, and only when, no registry bundle is loaded — the
    state of the service between this phase and Phase 7. ``None`` is used rather
    than a placeholder string because a client can test for ``None``; it cannot
    tell ``"unloaded"`` from a version someone actually shipped.
    """

    model_config = _STRICT

    model_version: str | None = Field(
        description=(
            "SemVer of the loaded registry bundle. null when no bundle is loaded, "
            "which is the case until Phase 7 packages bundle v0."
        ),
    )
    config_hash: str | None = Field(
        description=(
            "Hash of the resolved Hydra configuration the bundle was produced "
            "under. null when no bundle is loaded."
        ),
    )
    rules_version: str | None = Field(
        description=(
            "SemVer of the biological-consistency rule set that scored this "
            "response. null when no rule set is loaded."
        ),
    )
    experiment_id: str | None = Field(
        description=(
            "Identifier of the run manifest the bundle came from, e.g. EXP-0031. "
            "null when no bundle is loaded."
        ),
    )


# ── Requests ────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """One sequence and the organism group to interpret it under.

    The organism group is required and has no default. Sorting signals and
    localization compartments differ enough between groups that guessing on the
    user's behalf would silently change the answer, so the API asks rather than
    infers.

    Length and alphabet policy are deliberately absent. ``MAX_SEQUENCE_LENGTH``
    and the accepted residue alphabet are frozen in Phase 3 from the corpus's own
    distribution; encoding a guess here would put an unmeasured constant into the
    contract, where changing it later is a breaking change.
    """

    model_config = _STRICT

    sequence: str = Field(
        min_length=1,
        description=(
            "The amino-acid sequence, raw. Surrounding whitespace and FASTA "
            "headers are handled by the server; the length and alphabet policy is "
            "frozen in Phase 3 and enforced from configuration, not from this "
            "schema."
        ),
    )
    organism_group: OrganismGroup = Field(
        description=(
            "The organism group the sequence belongs to. Required: the model is "
            "conditioned on this choice rather than asked to infer it."
        ),
    )
    accession: str | None = Field(
        default=None,
        description=(
            "An optional identifier echoed back in the result, so a caller "
            "submitting many single sequences can match responses to inputs. Not "
            "used by the model."
        ),
    )


class BatchSubmission(BaseModel):
    """The non-file half of a ``POST /batch`` request.

    The FASTA itself is an uploaded file, not a JSON field: a proteome is tens of
    megabytes and base64 in a JSON body would inflate it by a third for no gain.
    This model documents the fields that accompany the upload.
    """

    model_config = _STRICT

    organism_group: OrganismGroup = Field(
        description="Applied to every sequence in the uploaded file.",
    )


# ── Acceptance ──────────────────────────────────────────────────────────────


class JobAccepted(ReproducibilityEnvelope):
    """The body of a ``202 Accepted`` from ``POST /predict`` or ``POST /batch``.

    ``POST /predict`` does not return a prediction (ADR-0001). It returns this.
    A single sequence is a job of size one, which is why one model serves both
    endpoints.

    For a size-one job that completes inside ``INLINE_RESULT_WINDOW_SECONDS`` the
    server may answer ``200`` with a :class:`JobResource` instead. That fast path
    is a convenience and never a contract: a client that assumes ``200`` is broken.
    """

    model_config = _STRICT

    job_id: UUID = Field(
        description=(
            "UUIDv4. Unguessable, but the result URL is not access-controlled — "
            "anyone holding it can read the result."
        ),
    )
    result_url: str = Field(
        description="Path of the job resource to poll, e.g. /jobs/{job_id}.",
    )
    status: JobStatus = Field(
        description="The job's status at acceptance. Always queued in practice.",
    )
    sequence_count: int = Field(
        ge=1,
        description="Number of sequences in this job. 1 for POST /predict.",
    )
    queue_position: int = Field(
        ge=0,
        description="Jobs ahead of this one at acceptance. 0 means it starts now.",
    )
    estimated_wait_seconds: int = Field(
        ge=0,
        description=(
            "Best-effort estimate from measured throughput and queue depth. "
            "Displayed as approximate; never a guarantee."
        ),
    )


# ── Result payload ──────────────────────────────────────────────────────────


class TaskPrediction(BaseModel):
    """One task head's answer for one sequence.

    Abstention and a low-confidence answer are different outcomes and are reported
    as such. When ``abstained`` is true the model declined because no label cleared
    the calibrated threshold, or because a blocking consistency violation forced
    it; ``labels`` is then empty rather than holding a best guess.

    A task that does not apply to the submitted organism group — prokaryotic
    localization for a eukaryotic sequence — is *masked*, not abstained, and is
    absent from the results list entirely. Scoring a task and declining is not the
    same as never having scored it, and conflating the two would misreport
    coverage.
    """

    model_config = _STRICT

    task: TaskName = Field(description="Which of the seven heads produced this.")
    labels: list[str] = Field(
        description=(
            "Predicted labels, empty when abstained. Multi-label for "
            "localization and GO function, single-element otherwise. The label "
            "vocabulary comes from the registry bundle, not from this schema."
        ),
    )
    probabilities: dict[str, float] = Field(
        description=(
            "Calibrated probability per candidate label. Present even when "
            "abstained, so a user can see how close the call was."
        ),
    )
    abstained: bool = Field(
        description=(
            "True when no label cleared the threshold, or when a blocking "
            "consistency violation forced abstention on this task."
        ),
    )
    abstention_threshold: float = Field(
        description=(
            "The tau this head was scored against, read from calibrator metadata "
            "at prediction time. Reported so a user can reproduce the decision; "
            "it has no default because there is no default tau."
        ),
    )
    cleavage_site: int | None = Field(
        default=None,
        description=(
            "1-based residue position of the predicted cleavage site, for the "
            "signal- and transit-peptide heads. null for every other task and "
            "when no site is predicted."
        ),
    )
    source: EngineSource = Field(
        default=EngineSource.MOTIF,
        description=(
            "Which engine produced this task's numbers: motif (the interpretable "
            "rule engine) or model (a trained head). The UI shows it. Defaults to "
            "motif because every pre-Phase-3 result came from the motif engine."
        ),
    )
    threshold: float = Field(
        description=(
            "The decision threshold actually applied to this task's labels. Equal "
            "to abstention_threshold under the fixed policy; the per-label "
            "thresholds a trained head uses arrive with Phase 10. When a "
            "pre-Phase-3 payload omits it, it is read from abstention_threshold."
        ),
    )
    calibrated: bool = Field(
        default=False,
        description=(
            "Whether a fitted calibrator was applied. A raw score or a sigmoid is "
            "not calibrated and is not displayed as if it were."
        ),
    )
    confidence_band: ConfidenceBand = Field(
        default=ConfidenceBand.NONE,
        description=(
            "Coarse band for the UI. none means below the threshold (abstaining) "
            "or, for a pre-Phase-3 payload, not banded: abstention is always read "
            "from abstained. The probability stays the datum; the band only "
            "styles it."
        ),
    )
    bundle_version: str | None = Field(
        default=None,
        description=(
            "SemVer of the registry bundle that produced this task. null for a field "
            "no bundle produced (a motif-engine task)."
        ),
    )
    bundle_sha256: str | None = Field(
        default=None,
        description="SHA-256 of that bundle (the loader refuses a mismatch). null with it.",
    )

    @model_validator(mode="before")
    @classmethod
    def _read_threshold_from_abstention_threshold(cls, data: Any) -> Any:
        """Back-compat: a payload from before Phase 3 has no ``threshold``.

        Its decision threshold was ``abstention_threshold``, so that is what it
        reads as. No literal is introduced, so no tau is hardcoded.
        """
        if isinstance(data, dict) and "threshold" not in data and "abstention_threshold" in data:
            return {**data, "threshold": data["abstention_threshold"]}
        return data


class ConsistencyViolation(BaseModel):
    """A rule the prediction set broke.

    A violation never silently edits a prediction. The user sees what the model
    said and, beside it, what the rule set thinks of it. Only ``blocking`` changes
    an outcome, and it does so by forcing abstention on the tasks it names — which
    is then visible in those tasks' ``abstained`` field.
    """

    model_config = _STRICT

    rule_id: str = Field(
        description=(
            "Identifier from backend/rules/consistency_rules.yaml, which is where "
            "the rules and their severities live. Not a Python constant."
        ),
    )
    severity: Severity = Field(description="note, warning or blocking.")
    message: str = Field(description="What the rule found, in the rule set's own words.")
    tasks: list[TaskName] = Field(
        description="The heads this violation concerns. A blocking violation abstains each.",
    )


class ResidueAttribution(BaseModel):
    """One residue's contribution to one task's prediction."""

    model_config = _STRICT

    position: int = Field(
        ge=1,
        description="1-based position in the normalized sequence.",
    )
    residue: str = Field(
        min_length=1,
        max_length=1,
        description="The single-letter amino-acid code at this position.",
    )
    score: float = Field(
        description=(
            "Per-residue weight. What it means depends on the task's attribution method; each "
            "method's claim is: "
            + " ".join(f"[{m.value}] {text}" for m, text in ATTRIBUTION_CLAIMS.items())
            + " Comparable within one task and sequence only — never across tasks, and never "
            "as a probability."
        ),
    )


class TaskAttribution(BaseModel):
    """Per-residue attribution for one task."""

    model_config = _STRICT

    task: TaskName = Field(description="The head being explained.")
    residues: list[ResidueAttribution] = Field(
        description="One entry per residue, in sequence order.",
    )
    source: EngineSource | None = Field(
        default=None,
        description="model when a trained head produced the weights, motif otherwise. Absent in "
        "a payload from before Phase 15.",
    )
    method: AttributionMethod | None = Field(
        default=None,
        description="How the weights were computed; the claim each method makes is in the "
        "description of ResidueAttribution.score. Absent in a payload from before Phase 15.",
    )


class HomologyHit(BaseModel):
    """One Swiss-Prot match, from the local MMseqs2 snapshot (ADR-0003).

    Evidence is independent of the model: it is what the databases already say,
    offered beside the prediction rather than folded into it. The frontend renders
    it as visually subordinate for that reason.
    """

    model_config = _STRICT

    accession: str = Field(description="Swiss-Prot accession of the match.")
    protein_name: str = Field(description="The matched entry's protein name.")
    identity: float = Field(
        ge=0.0,
        le=1.0,
        description="Sequence identity as a fraction, not a percentage.",
    )
    e_value: float = Field(ge=0.0, description="MMseqs2 E-value for the alignment.")
    annotated_locations: list[str] = Field(
        description="Subcellular locations annotated on the matched entry.",
    )


class GoAnnotation(BaseModel):
    """One GO term transferred from a homolog (ADR-0003, ADR-0015). ``source`` is ``homology``.

    A term the trained GO head predicts is a :class:`ModelGoAnnotation`; the two never share a
    field, so a reader can always tell a prediction from a transfer.
    """

    model_config = _STRICT

    source: Literal["homology"] = Field(
        default="homology",
        description="Always homology: the term is carried by a database hit, not predicted.",
    )
    go_id: str = Field(description="GO identifier, e.g. GO:0005739.")
    name: str = Field(description="The term's label.")
    aspect: Literal["cellular_component", "molecular_function", "biological_process"] = Field(
        description=(
            "The GO aspect. These three are the Gene Ontology's own top-level "
            "structure, not a biological choice this project made."
        ),
    )
    evidence_code: str | None = Field(
        default=None,
        description=(
            "GO evidence code of the source annotation, e.g. IDA. null when the hit-term "
            "table does not carry one (the Phase 13 table does not)."
        ),
    )
    source_accession: str = Field(
        description="The homolog the annotation was transferred from.",
    )


class ModelGoAnnotation(BaseModel):
    """One GO term the trained GO head scored (Phase 12). ``source`` is ``model``.

    It has no ``source_accession`` because nothing was transferred. The probability is the
    isotonic-calibrated, ancestor-propagated score; ``threshold`` is the head's single global
    threshold read from the bundle.
    """

    model_config = _STRICT

    source: Literal["model"] = Field(
        default="model",
        description="Always model: the term is predicted from the sequence by the GO head.",
    )
    go_id: str = Field(description="GO identifier, e.g. GO:0003677.")
    name: str = Field(description="The term's label.")
    aspect: Literal["cellular_component", "molecular_function", "biological_process"] = Field(
        description="The GO aspect (the GO head covers molecular_function).",
    )
    probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated, propagated probability of the term."
    )
    predicted: bool = Field(description="Whether the probability reaches the threshold.")
    threshold: float = Field(description="The decision threshold applied, from the bundle.")
    calibrated: bool = Field(description="Whether a fitted calibrator was applied.")


class Evidence(BaseModel):
    """Independent evidence, and an honest statement when there is none.

    The evidence service degrades gracefully: killing it still returns a complete
    prediction. ``available`` false with a populated ``unavailable_reason`` is that
    degradation made visible, rather than an empty list that reads like "no
    homologs found".
    """

    model_config = _STRICT

    available: bool = Field(
        description="False when the evidence service did not answer in time or is not configured.",
    )
    unavailable_reason: str | None = Field(
        default=None,
        description="Why evidence is missing. null when available is true.",
    )
    homology: list[HomologyHit] = Field(
        description="Top hits, capped by EVIDENCE_TOP_K. Empty when unavailable.",
    )
    go_annotations: list[GoAnnotation] = Field(
        description="Transferred GO terms. Empty when unavailable.",
    )
    swissprot_release: str | None = Field(
        default=None,
        description="Release string of the Swiss-Prot snapshot searched.",
    )
    goa_release: str | None = Field(
        default=None,
        description="Release string of the UniProt-GOA snapshot searched.",
    )


class SequenceResult(BaseModel):
    """Everything the system concluded about one sequence.

    This is the shape ``CLAUDE.md`` describes as "Blueprint 21.3's response
    schema": the payload of a completed job, never the body of a POST.
    """

    model_config = _STRICT

    index: int = Field(
        ge=0,
        description="0-based position in the submitted job. 0 for a single sequence.",
    )
    accession: str | None = Field(
        default=None,
        description="The caller's identifier, echoed. FASTA header id in batch mode.",
    )
    sequence_sha256: str = Field(
        description="SHA-256 of the normalized sequence. The cache and dedup key.",
    )
    length: int = Field(ge=1, description="Residue count after normalization.")
    organism_group: OrganismGroup = Field(
        description="The group this sequence was scored under.",
    )
    predictions: list[TaskPrediction] = Field(
        description=(
            "One entry per task that applies to this organism group. A task that "
            "does not apply is absent, not abstained."
        ),
    )
    consistency: list[ConsistencyViolation] = Field(
        description="Violations found. Empty means the rule set checked and found none.",
    )
    rules_checked: int = Field(
        ge=0,
        description=(
            "How many rules ran. Reported so that an empty violations list is "
            "readable as 'checked and clean' rather than 'not checked'."
        ),
    )
    attributions: list[TaskAttribution] = Field(
        description="Per-residue explanation, per task.",
    )
    evidence: Evidence = Field(description="Independent database evidence.")
    go_terms: list[ModelGoAnnotation] = Field(
        default_factory=list,
        description=(
            "GO molecular-function terms the trained head scored, source model. Homology-"
            "transferred terms are in evidence.go_annotations, never here."
        ),
    )
    error: str | None = Field(
        default=None,
        description=(
            "Why this sequence failed, when a batch job completed but this member "
            "did not. null on success. A failed member never carries a partial "
            "prediction."
        ),
    )
    no_confident_call: bool = Field(
        default=False,
        description=(
            "True when no label of any task clears its threshold. The screen "
            "then says so instead of showing the top-scoring label. An explicit "
            "boolean, not the absence of labels, so a reader can tell a declined "
            "call from a missing one."
        ),
    )
    calibration_version: str | None = Field(
        default=None,
        description=(
            "Identifies the fitted calibrator that scored this result. null while "
            "no calibrator has been fitted."
        ),
    )
    threshold_policy: ThresholdPolicy = Field(
        default=ThresholdPolicy.FIXED,
        description=(
            "How the thresholds in this result were chosen, so the Trust Console "
            "can state it: fixed, or per_label_mcc once Phase 10 lands."
        ),
    )


# ── Job resource ────────────────────────────────────────────────────────────


class JobProgress(BaseModel):
    """Counts for a job in flight. Batch jobs report these; size-one jobs do too."""

    model_config = _STRICT

    total: int = Field(ge=1, description="Sequences in the job.")
    completed_count: int = Field(ge=0, description="Finished successfully so far.")
    failed_count: int = Field(ge=0, description="Failed so far. Does not stop the job.")


class JobResource(ReproducibilityEnvelope):
    """The body of ``GET /jobs/{job_id}`` — the job, in whatever state it is in.

    This is the URL in ``result_url`` and the one the frontend polls. It answers
    ``200`` for a live job at every status, and carries ``results`` only when
    ``status`` is ``completed``. Past the retention window it answers ``410 Gone``
    rather than a silent empty result.

    ``GET /jobs/{job_id}/results`` is a different thing: the same results as a
    downloadable representation, with a format parameter. It is never polled, and
    it refuses a job that has not completed rather than returning a partial one.
    """

    model_config = _STRICT

    job_id: UUID = Field(description="The job's identifier.")
    status: JobStatus = Field(description="queued, running, completed, failed or cancelled.")
    sequence_count: int = Field(ge=1, description="Sequences submitted.")
    created_at: AwareDatetime = Field(
        description="When the job was accepted. Timezone-aware, always UTC.",
    )
    updated_at: AwareDatetime = Field(
        description="When the status last changed. Timezone-aware, always UTC.",
    )
    progress: JobProgress = Field(description="Counts, valid at every status.")
    queue_position: int | None = Field(
        default=None,
        description="Jobs ahead of this one. null once running or terminal.",
    )
    estimated_wait_seconds: int | None = Field(
        default=None,
        description="Best-effort estimate. null once running or terminal.",
    )
    results: list[SequenceResult] | None = Field(
        default=None,
        description=(
            "Present only when status is completed. null otherwise — a running "
            "job has no partial results to offer, and a failed one has an error."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Why the job failed or was cancelled. null otherwise.",
    )


# ── Service endpoints ───────────────────────────────────────────────────────


class HealthResponse(ReproducibilityEnvelope):
    """The body of ``GET /health``.

    ``ok`` means the process is serving and a registry bundle is loaded.
    ``degraded`` means it is serving without one, which is the truthful state of
    the service until Phase 7 — it answers requests, and it cannot predict.
    """

    model_config = _STRICT

    status: Literal["ok", "degraded"] = Field(
        description="degraded means serving but unable to predict; see detail.",
    )
    detail: str | None = Field(
        default=None,
        description="What is degraded. null when status is ok.",
    )
    bundle_loaded: bool = Field(description="Whether a registry bundle is loaded.")
    queue_depth: int = Field(
        ge=0,
        description="Jobs waiting. 0 with no workers running, which is a true 0.",
    )


class HealthzResponse(ReproducibilityEnvelope):
    """The body of ``GET /healthz`` — the model-registry state, and nothing else.

    A probe for platforms and phones: it says whether a registry bundle is
    loaded, and carries none of ``/health``'s queue accounting. It answers
    ``200`` whenever the process is serving, for the same reason ``/health`` does:
    a probe that failed while the service was correctly serving its contract would
    restart a container that is working.
    """

    model_config = _STRICT

    status: Literal["ok", "degraded"] = Field(
        description="ok when a registry bundle is loaded; degraded when serving without one.",
    )
    registry_state: Literal["loaded", "empty"] = Field(
        description="loaded when a registry bundle is loaded; empty when none is.",
    )
    detail: str | None = Field(
        default=None,
        description="What is missing. null when registry_state is loaded.",
    )


class VersionResponse(ReproducibilityEnvelope):
    """The body of ``GET /version`` — what is loaded, and what it hashes to."""

    model_config = _STRICT

    api_version: str = Field(description="SemVer of the HTTP contract itself.")
    service_version: str = Field(description="SemVer of the running service package.")
    checksums: dict[str, str] = Field(
        description=(
            "SHA-256 of each artifact in the loaded bundle, by filename. Empty "
            "when no bundle is loaded."
        ),
    )


class ExampleSequence(BaseModel):
    """One gallery entry, with its result already computed.

    The gallery exists because the asynchronous contract has no instant demo
    (ADR-0001 section 5). Its results are cached, and the frontend says so — a fast
    answer presented as if it had just been computed would be a lie about the
    system's behaviour.
    """

    model_config = _STRICT

    example_id: str = Field(description="Stable identifier for this gallery entry.")
    name: str = Field(description="Display name, e.g. Human cytochrome c oxidase subunit 4.")
    description: str = Field(description="Why this example is interesting.")
    organism_group: OrganismGroup = Field(description="The group it is scored under.")
    sequence: str = Field(min_length=1, description="The sequence, so a user can resubmit it.")
    result: SequenceResult = Field(description="The pre-computed, cached result.")
    computed_at: AwareDatetime = Field(
        description="When this cached result was produced. Shown to the user as cached.",
    )


class ExamplesResponse(ReproducibilityEnvelope):
    """The body of ``GET /examples``."""

    model_config = _STRICT

    examples: list[ExampleSequence] = Field(description="The gallery, in display order.")


class CancelResponse(ReproducibilityEnvelope):
    """The body of ``POST /jobs/{job_id}/cancel``.

    Cancelling a job that already reached a terminal state is not an error: the
    caller's intent — that this job not run — is already satisfied. ``cancelled``
    says whether this call changed anything.
    """

    model_config = _STRICT

    job_id: UUID = Field(description="The job addressed.")
    status: JobStatus = Field(description="The job's status after the call.")
    cancelled: bool = Field(
        description="True if this call stopped the job; false if it was already terminal.",
    )


# ── Errors ──────────────────────────────────────────────────────────────────


class ErrorResponse(ReproducibilityEnvelope):
    """Every non-2xx body the service produces.

    Errors carry the reproducibility envelope too. A user reporting a failure
    should be able to say which model version produced it without being asked to
    reproduce it first.
    """

    model_config = _STRICT

    error: str = Field(
        description=(
            "Stable machine-readable code, e.g. no_registry_bundle, "
            "job_not_found, job_expired, results_not_ready. Clients branch on "
            "this, never on the message."
        ),
    )
    message: str = Field(description="Human-readable explanation, safe to display.")
    detail: str | None = Field(
        default=None,
        description=(
            "Additional context, such as the field that failed validation or the "
            "phase that supplies a missing dependency. null when there is none."
        ),
    )


def utc_now() -> datetime:
    """Return the current time, timezone-aware in UTC.

    Present so that no module constructs a naive datetime for a contract field.
    Every timestamp this API emits is aware, which ``CLAUDE.md``'s data-schema
    rules require and ruff's DTZ rules enforce.
    """
    return datetime.now(UTC)


__all__ = [
    "API_VERSION",
    "BatchSubmission",
    "CancelResponse",
    "ConsistencyViolation",
    "ErrorResponse",
    "Evidence",
    "ExampleSequence",
    "ExamplesResponse",
    "GoAnnotation",
    "HealthResponse",
    "HealthzResponse",
    "HomologyHit",
    "JobAccepted",
    "JobProgress",
    "JobResource",
    "ModelGoAnnotation",
    "PredictRequest",
    "ReproducibilityEnvelope",
    "ResidueAttribution",
    "SequenceResult",
    "TaskAttribution",
    "TaskPrediction",
    "VersionResponse",
    "utc_now",
]
