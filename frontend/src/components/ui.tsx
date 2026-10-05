/* Shared building blocks. Chip = compartment (color + code); ConfidenceRow = DESIGN.md 3.1. */
import { useEffect, useState, type ReactElement, type ReactNode } from "react";
import {
  getExamples,
  getVersion,
  type Flag,
  type JobBody,
  type LocRow,
  type Accepted,
  type Provenance,
  type TaskSource,
} from "../services/demoApi";
import { degradedProblem, describeError, type Problem } from "../services/problems";

/* Compartment colour tokens are generated as --loc-<code>-fill / -ink (docs/DESIGN.md section 2).
   A code with no token (PCM, not served today) falls back to neutral instead of to nothing. */
export const locVar = (code: string) => `var(--loc-${code.toLowerCase()}-fill, var(--surface-2))`;
export const locText = (code: string) =>
  `var(--loc-${code.toLowerCase()}-ink, var(--text-secondary))`;

export function Chip({
  code,
  name,
  membrane,
}: {
  code: string;
  name?: string | undefined;
  membrane?: string | undefined;
}) {
  const cls =
    membrane === "transmembrane"
      ? "tm"
      : membrane === "lipid_anchored"
        ? "lipid"
        : membrane === "peripheral"
          ? "peripheral"
          : "";
  return (
    <span
      className={`chip ${cls}`}
      style={{ background: locVar(code), color: locText(code) }}
      title={name ?? code}
    >
      {code}
    </span>
  );
}

/** Which engine answered, and whether a fitted calibrator was applied. Never hidden. */
export function ProvenanceBadge({ src }: { src: TaskSource | undefined }) {
  if (!src) return null;
  return (
    <span
      className={`badge ${src.source}`}
      title={
        src.source === "motif"
          ? "Produced by the interpretable motif engine, not a trained model."
          : `Produced by registry bundle ${src.bundle_version ?? "?"} (sha256 ${(src.bundle_sha256 ?? "").slice(0, 12)}…).`
      }
    >
      {src.source} · {src.calibrated ? "calibrated" : "uncalibrated"}
    </span>
  );
}

/** One label's row. The threshold marker is this label's own threshold, not a constant. */
export function ConfidenceRow({ row }: { row: LocRow }) {
  return (
    <div className="conf" data-band={row.confidence_band}>
      <span className="row" style={{ gap: 6 }}>
        <Chip code={row.code} name={row.name} />
        <span style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)" }}>
          {row.name}
        </span>
      </span>
      <span className="track" style={{ position: "relative" }}>
        <span
          className="fill"
          style={{
            width: `${row.probability * 100}%`,
            background: locText(row.code),
            opacity: row.predicted ? 1 : 0.45,
            display: "block",
          }}
        />
        <span
          className="tau"
          style={{ position: "absolute", top: -2, bottom: -2, left: `${row.threshold * 100}%` }}
        />
      </span>
      <span className="num" title={`band: ${row.confidence_band}`}>
        {row.probability.toFixed(2)}
      </span>
    </div>
  );
}

export function Flags({ flags }: { flags: Flag[] }) {
  if (!flags.length)
    return (
      <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
        No consistency rule fired.
      </p>
    );
  return (
    <>
      {flags.map((f) => (
        <div key={f.rule_id} className={`flag ${f.severity}`}>
          <strong className="mono">
            {f.rule_id} · {f.severity}
          </strong>{" "}
          — {f.message}
        </div>
      ))}
    </>
  );
}

export function ProvenanceBar({ p }: { p: Provenance | null }) {
  if (!p) return null;
  return (
    <div className="prov">
      <span>model {p.model_version}</span>
      <span>config {p.config_hash}</span>
      <span>rules {p.rules_version}</span>
      <span>{p.experiment_id}</span>
      {p.scope_label && <span>{p.scope_label}</span>}
    </div>
  );
}

const BLOCKING: readonly string[] = ["unreachable", "server_error", "job_failed", "disabled"];

/**
 * A failure stated in words: what happened, why, and what to do. Never a stack trace. The retry
 * button appears only when the server said a retry can help (`problem.retryable`) AND the
 * screen can actually resubmit (`onRetry`); a button that cannot change the outcome is not drawn.
 */
export function ProblemNotice({
  problem,
  onRetry,
}: {
  problem: Problem;
  onRetry?: (() => void) | undefined;
}): ReactElement {
  const blocking = BLOCKING.includes(problem.kind);
  return (
    <div
      className={`flag ${blocking ? "blocked" : "warning"}`}
      role={problem.kind === "evidence_degraded" ? "status" : "alert"}
      data-problem-kind={problem.kind}
    >
      <strong>{problem.title}.</strong> {problem.message}
      <div style={{ fontSize: "var(--text-sm)", marginTop: "var(--space-1)" }}>{problem.hint}</div>
      {problem.code && (
        <div className="muted mono" style={{ fontSize: "var(--text-xs)" }}>
          code: {problem.code}
        </div>
      )}
      {problem.retryable && onRetry && (
        <button
          type="button"
          className="btn"
          style={{ marginTop: "var(--space-2)" }}
          onClick={onRetry}
        >
          Try again
        </button>
      )}
    </div>
  );
}

/** Queue and run state for the job in flight (ADR-0001: 202, then poll), or the failure. */
export function JobState({
  job,
  error,
  accepted,
  onRetry,
}: {
  job: JobBody | null;
  error: Problem | null;
  accepted?: Accepted | null;
  onRetry?: (() => void) | undefined;
}): ReactElement | null {
  if (error) return <ProblemNotice problem={error} onRetry={onRetry} />;
  if (job?.status === "completed") {
    return job.degraded ? (
      <ProblemNotice problem={degradedProblem(job.degraded_reasons ?? [])} />
    ) : null;
  }
  if (!job && !accepted) return null;
  const wait = accepted?.estimated_wait_seconds;
  const queued = (job?.status ?? "queued") === "queued";
  return (
    <p className="muted mono" role="status" style={{ fontSize: "var(--text-sm)" }}>
      job {(job?.job_id ?? accepted?.job_id ?? "").slice(0, 8)} · {job?.status ?? "queued"}
      {job ? ` · ${String(job.completed_count)}/${String(job.total)}` : ""}
      {queued && accepted ? ` · position ${String(accepted.queue_position)} in the queue` : ""}
      {queued && typeof wait === "number"
        ? ` · about ${String(Math.max(1, Math.round(wait)))} s until done (estimate from measured speed)`
        : ""}
    </p>
  );
}

export function Card({
  title,
  aside,
  children,
}: {
  title?: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      {(title || aside) && (
        <div className="head" style={{ marginBottom: "var(--space-3)" }}>
          {title ? <h2>{title}</h2> : <span />}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export interface Example {
  id: string;
  name: string;
  organism_group: string;
  sequence: string;
  note: string;
  in_training_set?: boolean;
  organism?: string;
  sequence_sha256?: string;
  length?: number;
  partition?: number;
  database_labels?: {
    localization?: string[];
    signal_peptide_cleavage_site?: number | null;
  };
}

/** Examples come from GET /demo/examples — the pickers are never hardcoded. */
export function useExamples() {
  const [examples, setExamples] = useState<Example[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getExamples()
      .then((b) => {
        setExamples(b.examples);
      })
      .catch((e) => {
        setError(describeError(e));
      });
  }, []);
  return { examples, error };
}

export function ExamplePicker({
  examples,
  value,
  onPick,
}: {
  examples: Example[];
  value: string;
  onPick: (e: Example) => void;
}) {
  return (
    <select
      value={value}
      onChange={(ev) => {
        const ex = examples.find((x) => x.id === ev.target.value);
        if (ex) onPick(ex);
      }}
    >
      <option value="">Load an example…</option>
      {examples.map((e) => (
        <option key={e.id} value={e.id}>
          {e.name}
        </option>
      ))}
    </select>
  );
}

export const GROUPS: [string, string][] = [
  ["eukaryote", "Animal / fungi"],
  ["plant", "Plant"],
  ["gram_neg", "Gram-negative bacteria"],
  ["gram_pos", "Gram-positive bacteria"],
];

/** Which organism groups a trained head covers, from GET /demo/version. Null until loaded. */
export function useServedGroups(): Set<string> | null {
  const [served, setServed] = useState<Set<string> | null>(null);
  useEffect(() => {
    getVersion()
      .then((v) => {
        const groups = (v.organism_groups ?? {}) as Record<string, { served?: boolean }>;
        setServed(new Set(Object.keys(groups).filter((k) => groups[k]?.served)));
      })
      .catch(() => {
        setServed(null);
      });
  }, []);
  return served;
}

/**
 * The organism-group picker. With `servedOnly` a group no trained head covers is shown but
 * disabled and says why (the training corpus is eukaryotic); the Scanner runs the motif
 * engine, which covers all four, so it does not pass `servedOnly`.
 */
export function GroupSelect({
  value,
  onChange,
  servedOnly = false,
}: {
  value: string;
  onChange: (group: string) => void;
  servedOnly?: boolean;
}) {
  const served = useServedGroups();
  return (
    <select
      value={value}
      onChange={(e) => {
        onChange(e.target.value);
      }}
    >
      {GROUPS.map(([v, l]) => {
        const blocked = servedOnly && served !== null && !served.has(v);
        return (
          <option key={v} value={v} disabled={blocked}>
            {blocked ? `${l} (no trained head)` : l}
          </option>
        );
      })}
    </select>
  );
}
