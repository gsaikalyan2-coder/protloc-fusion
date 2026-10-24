/*
 * Shared building blocks (design D3, ADR-0020). Chip = compartment (colour + code);
 * ConfidenceRow = DESIGN.md §3.1. Styling lives in styles/app.css; dynamic values reach CSS as
 * custom properties (--chip-fill, --p, --tau), never as inline colours.
 */
import {
  useEffect,
  useState,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import { ChevronDown, type LucideIcon } from "lucide-react";
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
      style={{ "--chip-fill": locVar(code), "--chip-ink": locText(code) } as CSSProperties}
      title={name ?? code}
    >
      {code}
    </span>
  );
}

/** Which engine answered, and whether a fitted calibrator was applied. Never hidden. */
export function ProvenanceBadge({ src }: { src: TaskSource | undefined }) {
  if (!src) return null;
  const why =
    src.source === "motif"
      ? "Produced by the interpretable motif engine, not a trained model."
      : `Produced by registry bundle ${src.bundle_version ?? "?"} (sha256 ${(src.bundle_sha256 ?? "").slice(0, 12)}…).`;
  return (
    <Tip content={why}>
      <button
        type="button"
        className={`badge ${src.source}`}
        aria-label={`${src.source}, ${src.calibrated ? "calibrated" : "uncalibrated"}. ${why}`}
      >
        {src.source} · {src.calibrated ? "calibrated" : "uncalibrated"}
      </button>
    </Tip>
  );
}

/** One label's row. The threshold marker is this label's own threshold, not a constant. */
export function ConfidenceRow({ row }: { row: LocRow }) {
  return (
    <div
      className="conf"
      data-band={row.confidence_band}
      data-predicted={row.predicted}
      style={
        {
          "--p": `${String(row.probability * 100)}%`,
          "--tau": `${String(row.threshold * 100)}%`,
          "--ink": locText(row.code),
        } as CSSProperties
      }
    >
      <span className="conf-label">
        <Chip code={row.code} name={row.name} />
        <span>{row.name}</span>
      </span>
      <span className="track">
        <span className="fill" />
        <span className="tau" title={`threshold ${row.threshold.toFixed(2)}`} />
      </span>
      <span className="num" title={`band: ${row.confidence_band}`}>
        {row.probability.toFixed(2)}
      </span>
    </div>
  );
}

export function Flags({ flags }: { flags: Flag[] }) {
  if (!flags.length) return <p className="hint">No consistency rule fired.</p>;
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
      <div className="flag-detail">{problem.hint}</div>
      {problem.code && <div className="note mono">code: {problem.code}</div>}
      {problem.retryable && onRetry && (
        <button type="button" className="btn sm" onClick={onRetry}>
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
    <p className="jobline" role="status">
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
  className,
  children,
}: {
  title?: string;
  aside?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={className ? `card ${className}` : "card"}>
      {(title || aside) && (
        <div className="card-head">
          {title ? <h2>{title}</h2> : <span />}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

/** A native select with a drawn chevron: keeps the platform picker, keyboard and form semantics. */
export function Select({
  children,
  className,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>): ReactElement {
  return (
    <span className={className ? `select ${className}` : "select"}>
      <select {...props}>{children}</select>
      <ChevronDown size={14} aria-hidden="true" />
    </span>
  );
}

/** Hover-and-focus tooltip for supplementary text. The trigger must be focusable. */
export function Tip({ content, children }: { content: ReactNode; children: ReactElement }) {
  return (
    <Tooltip.Provider delayDuration={300}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="tooltip" sideOffset={6} collisionPadding={8}>
            {content}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

/** Placeholder lines while a request is in flight. Announced once as busy, not read line by line. */
export function Skeleton({ lines = 3, label = "Loading" }: { lines?: number; label?: string }) {
  return (
    <div className="skeleton" role="status" aria-label={label} aria-busy="true">
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} />
      ))}
    </div>
  );
}

/** Nothing to show yet: what this place is for, and the one thing to do next. */
export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <span className="empty-icon" aria-hidden="true">
        <Icon size={18} strokeWidth={1.75} />
      </span>
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {action}
    </div>
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
    <Select
      aria-label="Load an example"
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
    </Select>
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
    <Select
      aria-label="Organism group"
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
    </Select>
  );
}

/**
 * A segmented control: one choice from a few, all visible. A group of toggle buttons
 * (`aria-pressed`), so each option is a real button in the tab order with its own name.
 */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string; disabled?: boolean; reason?: string }[];
  onChange: (value: T) => void;
}): ReactElement {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className="btn"
          aria-pressed={o.value === value}
          disabled={o.disabled}
          title={o.disabled ? o.reason : undefined}
          onClick={() => {
            onChange(o.value);
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** The organism group as a segmented control; groups no trained head covers are disabled. */
export function GroupSegmented({
  value,
  onChange,
  allGroups = false,
}: {
  value: string;
  onChange: (group: string) => void;
  /** The motif engine covers every group, so the Scanner enables all four. */
  allGroups?: boolean;
}): ReactElement {
  const served = useServedGroups();
  return (
    <Segmented
      label="Organism group"
      value={value}
      onChange={onChange}
      options={GROUPS.map(([v, l]) => ({
        value: v,
        label: l,
        disabled: !allGroups && served !== null && !served.has(v),
        reason: "No trained head covers this group; it is refused, not guessed.",
      }))}
    />
  );
}

/**
 * The sequence input shared by Predict, Evidence and the Scanner: one labelled textarea, the
 * organism group, an example picker and the screen's one action. `children` adds controls
 * between the picker and the action (the Scanner's target and mutation fields).
 */
export function SequenceForm({
  id,
  label = "Sequence",
  value,
  onChange,
  group,
  onGroup,
  examples,
  onPick,
  action,
  busyAction,
  busy,
  onSubmit,
  compact = false,
  allGroups = false,
  children,
}: {
  id: string;
  label?: string;
  value: string;
  onChange: (value: string) => void;
  group: string;
  onGroup: (group: string) => void;
  examples: Example[];
  onPick: (e: Example) => void;
  action: string;
  busyAction: string;
  busy: boolean;
  onSubmit: () => void;
  compact?: boolean;
  allGroups?: boolean;
  children?: ReactNode;
}): ReactElement {
  const residues = value.replace(/\s+/g, "").length;
  return (
    <form
      className={compact ? "card input-card compact" : "card input-card"}
      onSubmit={(ev) => {
        ev.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor={id} className="launch-label">
        {label}
        <span className="note">{residues > 0 ? `${String(residues)} residues` : "empty"}</span>
      </label>
      <textarea
        id={id}
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
        }}
        spellCheck={false}
      />
      <div className="launch-controls">
        <GroupSegmented value={group} onChange={onGroup} allGroups={allGroups} />
        <div className="row tight">
          <ExamplePicker examples={examples} value="" onPick={onPick} />
          {children}
          <button className="btn primary" type="submit" disabled={busy || !value.trim()}>
            {busy ? busyAction : action}
          </button>
        </div>
      </div>
    </form>
  );
}
