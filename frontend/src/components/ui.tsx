/* Shared building blocks. Chip = compartment (color + code); ConfidenceRow = DESIGN.md 3.1. */
import { useEffect, useState, type ReactNode } from "react";
import {
  getExamples,
  type Flag,
  type JobBody,
  type LocRow,
  type Provenance,
} from "../services/demoApi";

export const locVar = (code: string) => `var(--loc-${code.toLowerCase()})`;
export const locText = (code: string) => `var(--loc-${code.toLowerCase()}-text)`;

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

export function ConfidenceRow({ row, tau }: { row: LocRow; tau: number }) {
  return (
    <div className="conf">
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
          style={{ position: "absolute", top: -2, bottom: -2, left: `${tau * 100}%` }}
        />
      </span>
      <span className="num">{row.probability.toFixed(2)}</span>
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
    </div>
  );
}

export function JobState({ job, error }: { job: JobBody | null; error: string | null }) {
  if (error) return <p className="err">Backend error: {error}</p>;
  if (!job) return null;
  if (job.status === "completed") return null;
  return (
    <p className="muted mono" style={{ fontSize: "var(--text-sm)" }}>
      job {job.job_id.slice(0, 8)} · {job.status} · {job.completed_count}/{job.total}
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
        setError(String(e));
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
