/**
 * Types for the ProtLoc Fusion HTTP contract.
 *
 * Phase 2 freezes the contract as an OpenAPI document and generates this file
 * from it. Until then it holds only what is already frozen in CLAUDE.md — the
 * shared vocabulary, the reproducibility fields every response carries, and the
 * 202 acceptance shape. Nothing here is guessed: a response body that CLAUDE.md
 * does not specify is absent rather than invented, because a type that turns out
 * to be wrong is worse than a type that does not exist yet.
 *
 * The string unions mirror the enums in backend/protloc/domain/types.py. When
 * this file is generated in Phase 2, that mirroring stops being maintained by
 * hand and starts being checked by the build.
 */

/** The organism group a user selects alongside the sequence. */
export type OrganismGroup = "eukarya" | "gram_positive" | "gram_negative" | "archaea";

/** The seven task heads sharing one trunk representation. */
export type TaskName =
  | "localization"
  | "membrane_type"
  | "prokaryotic_localization"
  | "signal_peptide"
  | "mitochondrial_transit_peptide"
  | "chloroplast_transit_peptide"
  | "go_function";

/** The lifecycle of a prediction job. Every prediction is a job (ADR-0001). */
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

/** How a biological-consistency violation is reported. */
export type Severity = "note" | "warning" | "blocking";

/** The statuses from which no further transition is possible. */
export const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed", "cancelled"];

/**
 * Carried by every response, including 202 and errors.
 *
 * This is an API guarantee rather than a frontend decoration: a result the user
 * cannot trace back to a model version, a config hash and an experiment is a
 * result they cannot reproduce.
 */
export interface Provenance {
  model_version: string;
  config_hash: string;
  rules_version: string;
  experiment_id: string;
}

/**
 * The body of a 202 Accepted from POST /predict or POST /batch.
 *
 * `POST /predict` does not return a prediction. For a size-one job that finishes
 * inside the inline window the server may answer 200 with a result instead, but
 * that fast path is a convenience and never a contract — a client that assumes
 * 200 is a broken client.
 */
export interface JobAccepted extends Provenance {
  job_id: string;
  result_url: string;
  status: JobStatus;
  sequence_count: number;
  queue_position: number;
  /** Best effort, from measured throughput and queue depth. Never a guarantee. */
  estimated_wait_seconds: number;
}

/** The body of GET /health: liveness, the loaded model version, queue depth. */
export interface HealthResponse {
  status: string;
  model_version: string;
  queue_depth: number;
}

/** Narrowing helper for the statuses that carry no further transition. */
export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

/** Narrowing helper: only a completed job carries results. */
export function hasResults(status: JobStatus): boolean {
  return status === "completed";
}
