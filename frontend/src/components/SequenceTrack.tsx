/*
 * SequenceTrack — residue-level view used by the Inspector and Evidence screens.
 * Three overlays over the same coordinates:
 *   mode 'attribution' — per-residue importance for the top call (from the backend)
 *   mode 'hydropathy'  — smoothed Kyte-Doolittle value per residue
 *   mode 'alignment'   — residues covered by aligned blocks of a reference hit
 * Signal-peptide region is underlined, the cleavage site is a 2px achromatic rule
 * (DESIGN.md: a coordinate is not a category).
 */
import type { Result } from "../services/demoApi";
import { locText, locVar } from "./ui";

type Mode = "attribution" | "hydropathy" | "alignment";

function shade(mode: Mode, r: Result, i: number, blocks: [number, number][]): string {
  if (mode === "attribution") {
    const a = r.attribution[i] ?? 0;
    return a > 0
      ? `color-mix(in srgb, ${locText(r.top_location)} ${Math.round(a * 70)}%, transparent)`
      : "transparent";
  }
  if (mode === "hydropathy") {
    const h = r.hydropathy[i] ?? 0;
    return h > 0
      ? `color-mix(in srgb, var(--state-warn) ${Math.round(Math.min(h / 4.5, 1) * 70)}%, transparent)`
      : `color-mix(in srgb, var(--accent) ${Math.round(Math.min(-h / 4.5, 1) * 55)}%, transparent)`;
  }
  return blocks.some(([a, b]) => i >= a && i < b) ? "var(--accent-subtle)" : "transparent";
}

export function SequenceTrack({
  result,
  mode,
  blocks = [],
  limit = 600,
}: {
  result: Result;
  mode: Mode;
  blocks?: [number, number][];
  limit?: number;
}) {
  const seq = result.sequence.slice(0, limit);
  const sp = result.sorting_signal.region;
  const cs = result.sorting_signal.cleavage_site;
  const tms: [number, number][] = (result.features.tm_segments as [number, number][]) ?? [];
  return (
    <div>
      <p className="seq">
        {seq.split("").map((aa, i) => {
          const inSp = sp && i >= sp[0] && i < sp[1];
          const inTm = tms.some(([a, b]) => i >= a && i < b);
          return (
            <span
              key={i}
              title={`${i + 1} ${aa}${mode === "hydropathy" ? ` · KD ${result.hydropathy[i]?.toFixed(2)}` : mode === "attribution" ? ` · w ${result.attribution[i]?.toFixed(2)}` : ""}`}
              style={{
                background: shade(mode, result, i, blocks),
                borderBottom: inSp
                  ? "2px solid var(--state-warn)"
                  : inTm
                    ? `2px solid ${locVar("mem")}`
                    : "2px solid transparent",
                borderLeft: cs !== null && i === cs ? "2px solid var(--text-primary)" : undefined,
              }}
            >
              {aa}
            </span>
          );
        })}
      </p>
      {result.length > limit && (
        <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
          Showing first {limit} of {result.length} residues.
        </p>
      )}
      <div className="row" style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
        <span>underline: signal region</span>
        <span>vertical rule: cleavage site {cs !== null ? `(${cs})` : "(none)"}</span>
        <span>shade: {mode}</span>
      </div>
    </div>
  );
}
