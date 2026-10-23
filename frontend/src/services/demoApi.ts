/* Single place that talks to the demo backend. Every screen goes through here. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { ConfidenceBand, EngineSource, ThresholdPolicy } from "@/types/api";
import {
  DemoApiError,
  jobProblem,
  problemOf,
  type ApiErrorInfo,
  type BatchFailure,
  type Problem,
} from "./problems";

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
  /** "hit" when this result was served from the server's result cache (feature 3). */
  result_cache?: "hit" | "miss";
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
  failures?: BatchFailure[];
  /** Set on a failed job: the stable code and whether a retry can help. */
  error_info?: ApiErrorInfo;
  /** True when the evidence search was unavailable; the predictions are unchanged. */
  degraded?: boolean;
  degraded_reasons?: string[];
} & Provenance;

/** The structured `error` member of a response body, or null when there is none. */
function errorInfo(body: unknown): ApiErrorInfo | null {
  if (typeof body !== "object" || body === null) return null;
  const e = (body as { error?: unknown }).error;
  if (typeof e !== "object" || e === null) return null;
  const { code } = e as { code?: unknown };
  return typeof code === "string" ? (e as ApiErrorInfo) : null;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      ...(init?.body ? { headers: { "content-type": "application/json" } } : {}),
    });
  } catch (e) {
    throw new DemoApiError(0, e instanceof Error ? e.message : String(e));
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 202) {
    throw new DemoApiError(res.status, body?.detail ?? res.statusText, errorInfo(body));
  }
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
/** Formats GET /demo/jobs/{id}/results builds on the server for a whole job. */
export type JobExportFormat = "json" | "csv" | "fasta";
/** The server-side download URL of a whole finished job. */
export const jobResultsUrl = (jobId: string, format: JobExportFormat): string =>
  `/demo/jobs/${encodeURIComponent(jobId)}/results?format=${format}`;

/** One cached sequence the engine can score (GET /demo/catalog); the listing omits the sequence. */
export interface CatalogEntry {
  id: string;
  entry_name: string;
  name: string;
  organism: string | null;
  organism_group: string;
  length: number;
  partition: number;
  /** Partitions 1-4: the heads were fitted on it. Partition 5: held-out test. */
  in_training_set: boolean;
  sequence_sha256: string;
  database_labels: {
    localization: string[];
    signal_peptide: string[];
    signal_peptide_cleavage_site: number | null;
  };
}
export interface CatalogDetail extends CatalogEntry {
  sequence: string;
  note: string;
}
export interface CatalogPage {
  total: number;
  offset: number;
  limit: number;
  entries: CatalogEntry[];
  catalog: {
    available: boolean;
    reason: string | null;
    entries: number;
    dropped_not_cached: number;
  };
  locations: string[];
}
export interface CatalogFilter {
  q?: string | undefined;
  location?: string | undefined;
  group?: string | undefined;
  partition?: number | undefined;
  limit?: number | undefined;
  offset?: number | undefined;
}
export function catalogPath(f: CatalogFilter): string {
  const p = new URLSearchParams();
  if (f.q?.trim()) p.set("q", f.q.trim());
  if (f.location) p.set("location", f.location);
  if (f.group) p.set("group", f.group);
  if (f.partition !== undefined) p.set("partition", String(f.partition));
  if (f.limit !== undefined) p.set("limit", String(f.limit));
  if (f.offset) p.set("offset", String(f.offset));
  const qs = p.toString();
  return qs ? `/demo/catalog?${qs}` : "/demo/catalog";
}
export const getCatalog = (f: CatalogFilter = {}) => call<CatalogPage>(catalogPath(f));
/** Look one cached sequence up by accession or entry name; a miss is `accession_not_cached`. */
export const getCatalogEntry = (key: string) =>
  call<CatalogDetail>(`/demo/catalog/${encodeURIComponent(key.trim())}`);
export interface Accepted {
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
  const [error, setError] = useState<Problem | null>(null);
  const [accepted, setAccepted] = useState<Accepted | null>(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<number | null>(null);
  const last = useRef<{ path: string; body: unknown } | null>(null);
  /* Each run (and the unmount) bumps the generation. A run whose generation is no longer the
     current one is stale: it must not start a poller, and a poller it started stops itself.
     Without this, a run superseded while its POST was in flight, or one whose screen unmounted,
     left an interval nobody held a handle to, polling a finished job forever. */
  const generation = useRef(0);

  const stop = (): void => {
    generation.current += 1;
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  };
  useEffect(() => stop, []);

  const run = useCallback(async (path: string, body: unknown) => {
    stop();
    const mine = generation.current;
    last.current = { path, body };
    setError(null);
    setBusy(true);
    setJob(null);
    setAccepted(null);
    try {
      const acc = await post(path, body);
      if (generation.current !== mine) return;
      setAccepted(acc);
      const id = window.setInterval(async () => {
        if (generation.current !== mine) {
          window.clearInterval(id);
          return;
        }
        const finish = (): void => {
          window.clearInterval(id);
          if (timer.current === id) timer.current = null;
          setBusy(false);
        };
        try {
          const b = await call<JobBody>(acc.result_url);
          if (generation.current !== mine) {
            window.clearInterval(id);
            return;
          }
          setJob(b);
          if (b.status === "completed" || b.status === "failed" || b.status === "cancelled") {
            finish();
            const problem = jobProblem(b);
            if (problem) setError(problem);
          }
        } catch (e) {
          if (generation.current !== mine) {
            window.clearInterval(id);
            return;
          }
          finish();
          setError(problemOf(e));
        }
      }, 350);
      timer.current = id;
    } catch (e) {
      if (generation.current !== mine) return;
      setBusy(false);
      setError(problemOf(e));
    }
  }, []);

  /** Submit the last request again; the same function a retry button calls. */
  const retry = useCallback(() => {
    const req = last.current;
    if (req) void run(req.path, req.body);
  }, [run]);

  const done = job?.status === "completed" ? job : null;
  return { run, retry, job, done, error, accepted, busy };
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

/** An AlphaFold DB backbone for one cached sequence (GET /demo/structure/{sha256}). */
export interface StructureDoc {
  sequence_sha256: string;
  accession: string;
  source: string;
  license: string;
  entry_id?: string | null;
  model_version?: number | string | null;
  model_created?: string | null;
  length: number;
  /** C-alpha positions in Angstrom, one [x, y, z] per residue. */
  ca: number[][];
  /** Per-residue pLDDT from AlphaFold, 0 to 100. */
  plddt: number[];
  note: string;
}

export const getStructure = (sha256: string): Promise<StructureDoc> =>
  call<StructureDoc>(`/demo/structure/${sha256}`);

export type StructureState =
  | { status: "loading" }
  | { status: "ready"; data: StructureDoc }
  | { status: "missing"; message: string }
  | { status: "error"; problem: Problem };

/** Loads the backbone for a sequence hash; a 404 is "missing", not an error. */
export function useStructure(sha256: string): StructureState {
  const [state, setState] = useState<StructureState>({ status: "loading" });
  useEffect(() => {
    let live = true;
    setState({ status: "loading" });
    getStructure(sha256)
      .then((data) => {
        if (live) setState({ status: "ready", data });
      })
      .catch((e: unknown) => {
        if (!live) return;
        if (e instanceof DemoApiError && e.status === 404) {
          setState({ status: "missing", message: e.problem.message });
        } else {
          setState({ status: "error", problem: problemOf(e) });
        }
      });
    return () => {
      live = false;
    };
  }, [sha256]);
  return state;
}
