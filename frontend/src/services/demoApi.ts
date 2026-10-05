/* Single place that talks to the demo backend. Every screen goes through here. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { ConfidenceBand, EngineSource, ThresholdPolicy } from "@/types/api";

export interface Provenance {
  model_version: string;
  config_hash: string;
  rules_version: string;
  experiment_id: string;
  predictor: string;
  /** Every registry bundle behind this response, with the hash the loader verified. */
  bundles?: Record<string, { version: string; sha256: string }>;
  /** What the numbers on screen are measured on (the subset, not a benchmark). */
  scope_label?: string;
}
export interface LocRow {
  code: string;
  name: string;
  class_name: string;
  probability: number;
  raw: number;
  predicted: boolean;
  /** Which engine produced this number. The UI shows it (docs/phases/03.md). */
  source: EngineSource;
  /** The decision threshold actually applied to this label. */
  threshold: number;
  /** Whether a fitted calibrator was applied. */
  calibrated: boolean;
  calibration_method: string;
  confidence_band: ConfidenceBand;
}
export interface TaskSource {
  source: EngineSource;
  calibrated: boolean;
  /** The registry bundle that produced the task; null for a motif-engine task. */
  bundle_version: string | null;
  bundle_sha256: string | null;
  calibration_version: string | null;
}
export interface Flag {
  rule_id: string;
  severity: "blocked" | "warning" | "uncertain" | "note";
  message: string;
  /** Which engine fed each input of the rule. */
  basis?: Record<string, EngineSource>;
}
export interface GoTermRef {
  go_id: string;
  name: string;
  aspect: string;
}
/** One Swiss-Prot hit from the local MMseqs2 index (source: homology). */
export interface Hit {
  accession: string;
  identity: number;
  evalue: number;
  bits: number;
  query_coverage: number;
  target_coverage: number;
  alignment_length: number;
  exact_match: boolean;
  go_term_count: number;
  go_terms: GoTermRef[];
  database: string;
  source: "homology";
}
/** A GO term the trained head scored (source: model) - never carries an accession. */
export interface GoPrediction {
  go_id: string;
  name: string;
  aspect: string;
  probability: number;
  raw: number;
  predicted: boolean;
  threshold: number;
  source: "model";
  calibrated: boolean;
}
/** A GO term carried by homologous hits (source: homology) - always names its source hit. */
export interface GoTransfer {
  go_id: string;
  name: string | null;
  aspect: string | null;
  score: number;
  source: "homology";
  source_accession: string;
  identity: number;
  evalue: number;
  database: string;
  in_model_vocabulary: boolean;
}
export interface EvidenceStatus {
  available: boolean;
  unavailable_reason: string | null;
  swissprot_release: string;
  bundle_version: string;
  bundle_sha256: string;
  source: "homology";
}
export interface EvidenceAgreement {
  top_hit_accession: string;
  identity: number;
  exact_match: boolean;
  voting_hits: number;
  basis: string;
  shared: string[];
  model_only: string[];
  homology_only: string[];
  agrees: boolean;
}
export type AttributionMethod = "pooling_attention" | "occlusion" | "crf_marginal" | "motif";
export interface AttributionTrack {
  method: AttributionMethod;
  signed: boolean;
  target: string;
  values: number[];
  window: number | null;
  stride: number | null;
  forward_passes: number;
  scored_length: number;
  /** What this method does and does not show; shown as the tooltip. */
  claim: string;
}
export interface Result {
  id: string;
  length: number;
  sequence: string;
  sequence_sha256: string;
  organism_group: string;
  localization: LocRow[];
  top_location: string;
  abstained: boolean;
  /** True when no label clears its threshold: say so instead of showing the argmax. */
  no_confident_call: boolean;
  calibration_version: string | null;
  threshold_policy: ThresholdPolicy;
  task_sources: Record<string, TaskSource>;
  /** null since Phase 14: per-label thresholds replaced the single tau. */
  tau: number | null;
  temperature: number | null;
  membrane_type: Record<string, number>;
  sorting_signal: {
    type: string;
    probability: number;
    region: [number, number] | null;
    cleavage_site: number | null;
    candidates: Record<string, number>;
    threshold?: number;
    source?: EngineSource;
    calibrated?: boolean;
    untrained?: string[];
  };
  /** The trained GO head's terms (source: model). */
  go_function: GoPrediction[];
  features: Record<string, unknown>;
  hydropathy: number[];
  /** The default attribution track for the top label (see attribution_method). */
  attribution: number[];
  /** Which engine produced the residue attribution: model since Phase 15. */
  attribution_source: EngineSource;
  /** How the default track was computed; each method makes a different claim. */
  attribution_method: AttributionMethod;
  /** True when scores can be negative (occlusion): positive supports the label. */
  attribution_signed: boolean;
  /** The localization code the attribution explains. */
  attribution_target: string;
  /** Every track computed for the top label, keyed by method. */
  attribution_tracks: Record<string, AttributionTrack>;
  /** The CRF signal-peptide posterior per residue (method crf_marginal). */
  signal_track: AttributionTrack;
  consistency_flags: Flag[];
  /** Swiss-Prot hits (source: homology). Empty when the index is unavailable. */
  evidence: Hit[];
  /** GO terms transferred from those hits (source: homology). Never merged with go_function. */
  evidence_go: GoTransfer[];
  evidence_status: EvidenceStatus;
  evidence_agreement: EvidenceAgreement | null;
  provenance: Provenance;
}
export interface Summary {
  id: string;
  length: number;
  top_location: string;
  top_probability: number;
  predicted: string[];
  abstained: boolean;
  signal: string;
  membrane: string;
  n_flags: number;
  worst_flag: string | null;
  evidence_agrees: boolean | null;
  attention: number;
  reasons: string[];
}
export interface Landscape {
  codes: string[];
  names: Record<string, string>;
  matrix: Record<string, Record<string, number>>;
  counts: Record<string, number>;
  abstained: number;
  total: number;
}
export interface Scan {
  length: number;
  sequence: string;
  organism_group: string;
  target: string;
  target_name: string;
  baseline: Record<string, number>;
  substitute: string;
  position_deltas: number[];
  most_sensitive: { position: number; residue: string; delta: number }[];
  truncation: Record<string, number>[];
  variant: null | {
    mutations: string[];
    relocalized: boolean;
    wild_type: Pick<
      Result,
      | "localization"
      | "top_location"
      | "abstained"
      | "sorting_signal"
      | "membrane_type"
      | "consistency_flags"
    >;
    variant: Pick<
      Result,
      | "localization"
      | "top_location"
      | "abstained"
      | "sorting_signal"
      | "membrane_type"
      | "consistency_flags"
    >;
  };
  tau: number | null;
  /** The scanner runs the motif engine, not the trained heads. */
  scanner_source: EngineSource;
  provenance: Provenance;
}
export type JobBody = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  total: number;
  completed_count: number;
  /** Seconds since the epoch: queue wait is started_at - created_at, compute is finished_at - started_at. */
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  results?: Result[];
  summary?: Summary[];
  landscape?: Landscape;
  scan?: Scan;
  failures?: { id: string; error: string }[];
} & Provenance;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    ...(init?.body ? { headers: { "content-type": "application/json" } } : {}),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 202)
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
  return body as T;
}

export const getVersion = () => call<Record<string, any>>("/demo/version");
export const getExamples = () =>
  call<{
    examples: {
      id: string;
      name: string;
      organism_group: string;
      sequence: string;
      note: string;
      in_training_set?: boolean;
      length?: number;
    }[];
  }>("/demo/examples");
export const getTrust = () => call<Record<string, any>>("/demo/trust");
interface Accepted {
  job_id: string;
  result_url: string;
  queue_position: number;
  /** Measured from finished jobs; null until one job has run. */
  estimated_wait_seconds: number | null;
}
const post = (p: string, b: unknown) =>
  call<Accepted>(p, { method: "POST", body: JSON.stringify(b) });

/**
 * useJob — ADR-0001 in the client: POST returns 202, we poll the result URL.
 * `run` submits; `job` is the latest poll; `done` is the completed body.
 */
export function useJob() {
  const [job, setJob] = useState<JobBody | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<number | null>(null);

  const stop = () => {
    if (timer.current) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  };
  useEffect(() => stop, []);

  const run = useCallback(async (path: string, body: unknown) => {
    stop();
    setError(null);
    setBusy(true);
    setJob(null);
    try {
      const acc = await post(path, body);
      timer.current = window.setInterval(async () => {
        try {
          const b = await call<JobBody>(acc.result_url);
          setJob(b);
          if (b.status === "completed" || b.status === "failed" || b.status === "cancelled") {
            stop();
            setBusy(false);
            if (b.status === "failed") setError(b.error ?? "Job failed");
          }
        } catch (e) {
          stop();
          setBusy(false);
          setError(String(e));
        }
      }, 350);
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const done = job?.status === "completed" ? job : null;
  return { run, job, done, error, busy };
}

export const predict = (sequence: string, organism_group: string, id = "query") =>
  ["/demo/predict", { sequence, organism_group, id }] as const;
export const batch = (fasta: string, organism_group: string) =>
  ["/demo/batch", { fasta, organism_group }] as const;
export const scanReq = (
  sequence: string,
  organism_group: string,
  target: string | null,
  mutations: string,
  substitute: string,
) => ["/demo/scan", { sequence, organism_group, target, mutations, substitute }] as const;
