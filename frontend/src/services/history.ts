/*
 * Recent jobs (feature 7): the jobs this browser submitted, kept in localStorage only.
 * Nothing here is sent anywhere. Storage can be unavailable (a private window, blocked site
 * data), so every read and write is guarded and the app works the same without it: the list
 * is simply empty. A predict entry keeps its sequence so it can be reopened in the Inspector;
 * a re-run is served from the server's result cache when the result was cached.
 */

export type JobKind = "predict" | "batch" | "scan";

export interface HistoryEntry {
  job_id: string;
  kind: JobKind;
  /** The sequence id, or the first FASTA record id and the record count. */
  label: string;
  organism_group: string;
  /** Milliseconds since the epoch, in this browser. */
  submitted_at: number;
  /** Predict and scan only: the sequence, so the entry can be reopened. */
  sequence?: string;
  sequence_count: number;
}

export const HISTORY_KEY = "protloc.recentJobs.v1";
export const HISTORY_LIMIT = 20;
export const HISTORY_EVENT = "protloc:history";

const KIND_BY_PATH: Readonly<Record<string, JobKind>> = {
  "/demo/predict": "predict",
  "/demo/batch": "batch",
  "/demo/scan": "scan",
};

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

function fastaSummary(fasta: string): { first: string; count: number } {
  const headers = fasta.split(/\r?\n/).filter((line) => line.startsWith(">"));
  const first = (headers[0] ?? ">").slice(1).trim().split(/\s+/)[0] ?? "";
  return { first: first || "records", count: headers.length };
}

/** The history entry for one accepted request, or null for a path that is not a job. */
export function entryFor(
  path: string,
  body: unknown,
  jobId: string,
  now: number,
): HistoryEntry | null {
  const kind = KIND_BY_PATH[path];
  if (kind === undefined || typeof body !== "object" || body === null) return null;
  const b = body as Record<string, unknown>;
  const group = str(b.organism_group, "eukaryote");
  if (kind === "batch") {
    const { first, count } = fastaSummary(str(b.fasta));
    const label = count > 1 ? `${first} + ${String(count - 1)} more` : first;
    return {
      job_id: jobId,
      kind,
      label,
      organism_group: group,
      submitted_at: now,
      sequence_count: count,
    };
  }
  const sequence = str(b.sequence).replace(/\s+/g, "");
  return {
    job_id: jobId,
    kind,
    label: kind === "scan" ? "scan" : str(b.id, "query"),
    organism_group: group,
    submitted_at: now,
    sequence,
    sequence_count: 1,
  };
}

function isEntry(v: unknown): v is HistoryEntry {
  if (typeof v !== "object" || v === null) return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.job_id === "string" &&
    typeof e.kind === "string" &&
    e.kind in { predict: 1, batch: 1, scan: 1 } &&
    typeof e.label === "string" &&
    typeof e.organism_group === "string" &&
    typeof e.submitted_at === "number" &&
    typeof e.sequence_count === "number"
  );
}

/** The stored entries, newest first; empty when storage is unavailable or corrupt. */
export function loadHistory(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEntry).slice(0, HISTORY_LIMIT) : [];
  } catch {
    return [];
  }
}

function save(entries: HistoryEntry[]): void {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, HISTORY_LIMIT)));
  } catch {
    return;
  }
  window.dispatchEvent(new Event(HISTORY_EVENT));
}

/** Put an entry first, dropping any older entry for the same job and anything past the limit. */
export function addHistory(entry: HistoryEntry): void {
  save([entry, ...loadHistory().filter((e) => e.job_id !== entry.job_id)]);
}

/** Record an accepted request (called by useJob); a failure to store is ignored. */
export function recordJob(path: string, body: unknown, jobId: string): void {
  const entry = entryFor(path, body, jobId, Date.now());
  if (entry) addHistory(entry);
}

export function removeHistory(jobId: string): void {
  save(loadHistory().filter((e) => e.job_id !== jobId));
}

export function clearHistory(): void {
  try {
    window.localStorage.removeItem(HISTORY_KEY);
  } catch {
    return;
  }
  window.dispatchEvent(new Event(HISTORY_EVENT));
}
