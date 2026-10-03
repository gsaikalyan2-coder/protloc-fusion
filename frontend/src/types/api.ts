/**
 * The ProtLoc Fusion HTTP contract, in TypeScript.
 *
 * The types come from `api.generated.ts`, which `npm run generate:types`
 * produces from `docs/api/openapi.yaml` — itself generated from the Pydantic
 * models in `backend/protloc/domain/contracts.py`. So a field the backend adds,
 * renames or narrows reaches this project's type checker without anyone
 * retyping it, and CI fails the build if the two have drifted. Before Phase 2
 * these unions were maintained by hand and nothing checked that they agreed with
 * the backend; that is the gap this file closes.
 *
 * This module is hand-written and re-exports from the generated one. It exists
 * for two reasons.
 *
 * First, names. The generated file addresses everything through
 * `components["schemas"][...]`, which is accurate and unreadable at a call site.
 * The aliases below give the shapes the names the rest of the app already uses.
 *
 * Second, the helpers. `openapi-typescript` emits types and nothing else, so the
 * three runtime values at the bottom — which are code *about* the contract, not
 * part of it — have no home in the generated file and would be deleted by the
 * next regeneration if they were put there.
 *
 * Adding a hand-written *type* here is almost always a mistake: if the backend
 * does not describe it, the server does not send it.
 */

import type { components } from "@/types/api.generated";

type Schemas = components["schemas"];

// ── Vocabulary ──────────────────────────────────────────────────────────────

/** The organism group a user selects alongside the sequence. */
export type OrganismGroup = Schemas["OrganismGroup"];

/** The seven task heads sharing one trunk representation. */
export type TaskName = Schemas["TaskName"];

/** The lifecycle of a prediction job. Every prediction is a job (ADR-0001). */
export type JobStatus = Schemas["JobStatus"];

/** How a biological-consistency violation is reported. */
export type Severity = Schemas["Severity"];

/** Which engine produced a prediction-bearing field: the motif engine or a trained model. */
export type EngineSource = Schemas["EngineSource"];

/** Coarse reading of how far a probability clears its threshold; `none` is below it. */
export type ConfidenceBand = Schemas["ConfidenceBand"];

/** How the decision thresholds in a result were chosen. */
export type ThresholdPolicy = Schemas["ThresholdPolicy"];

// ── Envelope ────────────────────────────────────────────────────────────────

/**
 * Carried by every response, including 202 and errors.
 *
 * Every field is `null` when no registry bundle is loaded — the state of the
 * service until Phase 7. A client that renders these must handle null rather
 * than assume a string, which is why the generated type says so.
 */
export type Provenance = Schemas["ReproducibilityEnvelope"];

// ── Requests ────────────────────────────────────────────────────────────────

/** One sequence and the organism group to interpret it under. */
export type PredictRequest = Schemas["PredictRequest"];

/** The non-file half of a `POST /batch` request. */
export type BatchSubmission = Schemas["BatchSubmission"];

// ── Responses ───────────────────────────────────────────────────────────────

/** The body of a 202 from `POST /predict` or `POST /batch`. Never a prediction. */
export type JobAccepted = Schemas["JobAccepted"];

/** The body of `GET /jobs/{job_id}` — the job, in whatever state it is in. */
export type JobResource = Schemas["JobResource"];

/** Counts for a job in flight. */
export type JobProgress = Schemas["JobProgress"];

/** Everything the system concluded about one sequence. */
export type SequenceResult = Schemas["SequenceResult"];

/** One task head's answer for one sequence. */
export type TaskPrediction = Schemas["TaskPrediction"];

/** A rule the prediction set broke. */
export type ConsistencyViolation = Schemas["ConsistencyViolation"];

/** Per-residue attribution for one task. */
export type TaskAttribution = Schemas["TaskAttribution"];

/** One residue's contribution to one task's prediction. */
export type ResidueAttribution = Schemas["ResidueAttribution"];

/** Independent database evidence, and an honest statement when there is none. */
export type Evidence = Schemas["Evidence"];

/** One Swiss-Prot match from the local snapshot. */
export type HomologyHit = Schemas["HomologyHit"];

/** One GO term from the local UniProt-GOA snapshot. */
export type GoAnnotation = Schemas["GoAnnotation"];

/** The body of `GET /health`. */
export type HealthResponse = Schemas["HealthResponse"];

/** The body of `GET /version`. */
export type VersionResponse = Schemas["VersionResponse"];

/** One gallery entry, with its result already computed. */
export type ExampleSequence = Schemas["ExampleSequence"];

/** The body of `GET /examples`. */
export type ExamplesResponse = Schemas["ExamplesResponse"];

/** The body of `POST /jobs/{job_id}/cancel`. */
export type CancelResponse = Schemas["CancelResponse"];

/** Every non-2xx body the service produces. */
export type ErrorResponse = Schemas["ErrorResponse"];

// ── Helpers ─────────────────────────────────────────────────────────────────
//
// Runtime values, not part of the contract. They live here because
// openapi-typescript emits types only and would delete them on regeneration.

/** The statuses from which no further transition is possible. */
export const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed", "cancelled"];

/** Whether no further status change is possible for this job. */
export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

/**
 * Whether results are available to read.
 *
 * Only a completed job carries results. A failed or cancelled job has an error
 * or a cancellation reason, which is not a partial result and is not served as
 * one.
 */
export function hasResults(status: JobStatus): boolean {
  return status === "completed";
}

/**
 * Whether the service that produced this response could identify what produced
 * it.
 *
 * False while no registry bundle is loaded. A reproducibility footer rendered
 * from an unidentified response should say so rather than show four blanks.
 */
export function isIdentified(provenance: Provenance): boolean {
  return (
    provenance.model_version !== null &&
    provenance.config_hash !== null &&
    provenance.rules_version !== null &&
    provenance.experiment_id !== null
  );
}
