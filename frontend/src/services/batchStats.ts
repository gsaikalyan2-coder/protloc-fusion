/*
 * Batch dashboard numbers (feature 9). Pure: it counts what the server already returned for a
 * batch and computes nothing new about the proteins. Every count is over the scored records;
 * refused records are counted separately and never folded into a distribution.
 */
import type { Result } from "./demoApi";

export interface Bar {
  key: string;
  label: string;
  count: number;
}

export interface BatchStats {
  scored: number;
  refused: number;
  /** Proteins with each location predicted (a protein can count under several). */
  locations: Bar[];
  noConfidentCall: number;
  /** The highest calibrated location probability per protein, in five bins. */
  confidence: Bar[];
  signal: Bar[];
  /** Predicted GO terms, most frequent first, at most `goLimit`. */
  goTerms: Bar[];
  evidence: Bar[];
  cacheHits: number;
  /** Records encoded live, outside the evaluated subset (ADR-0019). */
  liveEncoded: number;
}

export const CONFIDENCE_BINS: readonly [number, number, string][] = [
  [0, 0.2, "0.0–0.2"],
  [0.2, 0.4, "0.2–0.4"],
  [0.4, 0.6, "0.4–0.6"],
  [0.6, 0.8, "0.6–0.8"],
  [0.8, 1.000001, "0.8–1.0"],
];

function sorted(counts: Map<string, Bar>): Bar[] {
  return [...counts.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

function bump(counts: Map<string, Bar>, key: string, label: string): void {
  const cur = counts.get(key);
  if (cur) cur.count += 1;
  else counts.set(key, { key, label, count: 1 });
}

function signalLabel(type: string): string {
  if (type === "none") return "no signal";
  if (type === "SP" || type === "signal_peptide") return "signal peptide";
  return type.replaceAll("_", " ");
}

export function batchStats(results: readonly Result[], refused = 0, goLimit = 8): BatchStats {
  const locations = new Map<string, Bar>();
  const signal = new Map<string, Bar>();
  const go = new Map<string, Bar>();
  const evidence = new Map<string, Bar>();
  const confidence = CONFIDENCE_BINS.map(([, , label]) => ({ key: label, label, count: 0 }));
  let noConfidentCall = 0;
  let cacheHits = 0;
  let liveEncoded = 0;

  for (const r of results) {
    if (r.no_confident_call) noConfidentCall += 1;
    for (const l of r.localization) if (l.predicted) bump(locations, l.code, l.name);
    const top = Math.max(0, ...r.localization.map((l) => l.probability));
    const bin = CONFIDENCE_BINS.findIndex(([lo, hi]) => top >= lo && top < hi);
    const target = confidence[bin === -1 ? confidence.length - 1 : bin];
    if (target) target.count += 1;
    bump(signal, r.sorting_signal.type, signalLabel(r.sorting_signal.type));
    for (const g of r.go_function) if (g.predicted) bump(go, g.go_id, g.name || g.go_id);
    if (!r.evidence_status.available) bump(evidence, "unavailable", "evidence unavailable");
    else if (r.evidence.length === 0) bump(evidence, "no_hit", "no homolog found");
    else if (r.evidence_agreement?.agrees === false)
      bump(evidence, "disagrees", "homolog disagrees");
    else bump(evidence, "hit", "homolog found");
    if (r.result_cache === "hit") cacheHits += 1;
    if (r.outside_evaluated_subset === true) liveEncoded += 1;
  }

  return {
    scored: results.length,
    refused,
    locations: sorted(locations),
    noConfidentCall,
    confidence,
    signal: sorted(signal),
    goTerms: sorted(go).slice(0, goLimit),
    evidence: sorted(evidence),
    cacheHits,
    liveEncoded,
  };
}
