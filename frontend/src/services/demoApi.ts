/* Single place that talks to the demo backend. Every screen goes through here. */
import { useCallback, useEffect, useRef, useState } from "react";

export interface Provenance {
  model_version: string;
  config_hash: string;
  rules_version: string;
  experiment_id: string;
  predictor: string;
}
export interface LocRow {
  code: string;
  name: string;
  probability: number;
  raw: number;
  predicted: boolean;
}
export interface Flag {
  rule_id: string;
  severity: "blocked" | "warning" | "uncertain" | "note";
  message: string;
}
export interface Hit {
  accession: string;
  family: string;
  organism_group: string;
  location: string;
  location_name: string;
  evidence_code: string;
  identity: number;
  kmer_jaccard: number;
  coverage: number;
  same_group: boolean;
  aligned_blocks: [number, number][];
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
  tau: number;
  temperature: number;
  membrane_type: Record<string, number>;
  sorting_signal: {
    type: string;
    probability: number;
    region: [number, number] | null;
    cleavage_site: number | null;
    candidates: Record<string, number>;
  };
  go_terms: { go_id: string; term: string; span: [number, number] }[];
  features: Record<string, unknown>;
  hydropathy: number[];
  attribution: number[];
  consistency_flags: Flag[];
  evidence: Hit[];
  evidence_agreement: { top_hit_location: string; agrees: boolean; identity: number } | null;
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
  tau: number;
  provenance: Provenance;
}
export type JobBody = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  total: number;
  completed_count: number;
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
    }[];
  }>("/demo/examples");
export const getTrust = () => call<Record<string, any>>("/demo/trust");
export const getReference = () =>
  call<{
    count: number;
    entries: Omit<
      Hit,
      "identity" | "kmer_jaccard" | "coverage" | "same_group" | "aligned_blocks" | "location_name"
    >[];
  }>("/demo/reference");

interface Accepted {
  job_id: string;
  result_url: string;
  queue_position: number;
  estimated_wait_seconds: number;
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
