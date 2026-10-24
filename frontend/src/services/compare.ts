/*
 * Side-by-side comparison of two results (feature 8). Pure: it reshapes two results the server
 * returned and computes nothing new about the proteins. Probabilities are each result's own
 * calibrated values; a difference between them is a difference between two model outputs, not
 * a measured biological difference, and the page says so.
 */
import type { Result } from "./demoApi";

export interface LocationRow {
  code: string;
  name: string;
  a: number | null;
  b: number | null;
  /** b - a, rounded to 3 places; null when either side lacks the label. */
  delta: number | null;
  aPredicted: boolean;
  bPredicted: boolean;
}

export interface SummaryRow {
  label: string;
  a: string;
  b: string;
  differs: boolean;
}

export interface GoTerm {
  go_id: string;
  name: string;
}

export interface Comparison {
  sameSequence: boolean;
  summary: SummaryRow[];
  locations: LocationRow[];
  go: { shared: GoTerm[]; onlyA: GoTerm[]; onlyB: GoTerm[] };
}

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

function call(r: Result): string {
  if (r.no_confident_call) return "no confident call";
  const predicted = r.localization.filter((l) => l.predicted).map((l) => l.code);
  return predicted.length > 0 ? predicted.join(", ") : r.top_location;
}

function membrane(r: Result): string {
  const entries = Object.entries(r.membrane_type);
  if (entries.length === 0) return "—";
  const [best] = entries.reduce((m, e) => (e[1] > m[1] ? e : m));
  return best.replaceAll("_", " ");
}

function signal(r: Result): string {
  const s = r.sorting_signal;
  return s.type === "none" ? "none" : `${s.type} (cleaves after ${String(s.cleavage_site ?? "—")})`;
}

function homolog(r: Result): string {
  if (!r.evidence_status.available) return "evidence unavailable";
  const top = r.evidence[0];
  return top ? `${top.accession} (${String(Math.round(top.identity * 100))}% id)` : "no hit";
}

function flags(r: Result): string {
  return r.consistency_flags.length > 0
    ? r.consistency_flags.map((f) => `${f.rule_id}:${f.severity}`).join(", ")
    : "none";
}

function row(label: string, a: string, b: string): SummaryRow {
  return { label, a, b, differs: a !== b };
}

function predictedGo(r: Result): Map<string, string> {
  return new Map(r.go_function.filter((g) => g.predicted).map((g) => [g.go_id, g.name]));
}

export function compareResults(a: Result, b: Result): Comparison {
  const summary = [
    row("Length", String(a.length), String(b.length)),
    row("Organism group", a.organism_group, b.organism_group),
    row("Predicted location", call(a), call(b)),
    row("Abstained", a.abstained ? "yes" : "no", b.abstained ? "yes" : "no"),
    row("Sorting signal", signal(a), signal(b)),
    row("Membrane type (motif engine)", membrane(a), membrane(b)),
    row("Consistency flags", flags(a), flags(b)),
    row("Nearest homolog", homolog(a), homolog(b)),
  ];

  const byCode = new Map<string, LocationRow>();
  for (const [side, r] of [
    ["a", a],
    ["b", b],
  ] as const) {
    for (const l of r.localization) {
      const cur = byCode.get(l.code) ?? {
        code: l.code,
        name: l.name,
        a: null,
        b: null,
        delta: null,
        aPredicted: false,
        bPredicted: false,
      };
      if (side === "a") {
        cur.a = l.probability;
        cur.aPredicted = l.predicted;
      } else {
        cur.b = l.probability;
        cur.bPredicted = l.predicted;
      }
      byCode.set(l.code, cur);
    }
  }
  const locations = [...byCode.values()]
    .map((l) => ({ ...l, delta: l.a !== null && l.b !== null ? round3(l.b - l.a) : null }))
    .sort((x, y) => Math.max(y.a ?? 0, y.b ?? 0) - Math.max(x.a ?? 0, x.b ?? 0));

  const goA = predictedGo(a);
  const goB = predictedGo(b);
  const term = (id: string, names: Map<string, string>): GoTerm => ({
    go_id: id,
    name: names.get(id) ?? "",
  });
  return {
    sameSequence: a.sequence_sha256 === b.sequence_sha256,
    summary,
    locations,
    go: {
      shared: [...goA.keys()].filter((id) => goB.has(id)).map((id) => term(id, goA)),
      onlyA: [...goA.keys()].filter((id) => !goB.has(id)).map((id) => term(id, goA)),
      onlyB: [...goB.keys()].filter((id) => !goA.has(id)).map((id) => term(id, goB)),
    },
  };
}
