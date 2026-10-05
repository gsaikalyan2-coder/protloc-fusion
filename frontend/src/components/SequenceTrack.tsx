/*
 * SequenceTrack — residue-level view used by the Inspector and Evidence screens.
 * Three overlays over the same coordinates:
 *   mode 'attribution' — the trained head's default track for the top call (occlusion; signed)
 *   mode 'attention'   — where the head's pooled vector looked (class-agnostic)
 *   mode 'signal'      — CRF signal-peptide posterior
 *   mode 'hydropathy'  — smoothed Kyte-Doolittle value per residue
 *   mode 'alignment'   — residues covered by aligned blocks of a reference hit
 * Signal-peptide region is underlined, the cleavage site is a 2px achromatic rule
 * (DESIGN.md: a coordinate is not a category).
 */
import type { Result } from "../services/demoApi";
import { locText, locVar } from "./ui";

type Mode = "attribution" | "attention" | "signal" | "hydropathy" | "alignment";

interface Numeric {
  values: number[];
  signed: boolean;
  method: string;
}

/** The numeric track a mode draws, or null for hydropathy and alignment. */
function numericTrack(mode: Mode, r: Result): Numeric | null {
  if (mode === "attribution") {
    return { values: r.attribution, signed: r.attribution_signed, method: r.attribution_method };
  }
  if (mode === "attention") {
    const t = r.attribution_tracks.pooling_attention;
    return t ? { values: t.values, signed: false, method: t.method } : null;
  }
  if (mode === "signal") {
    return { values: r.signal_track.values, signed: false, method: r.signal_track.method };
  }
  return null;
}

/** Shade for one residue: |value| scaled by the track's own largest magnitude. */
function trackShade(r: Result, t: Numeric, scale: number, i: number): string {
  const v = t.values[i] ?? 0;
  const pct = scale > 0 ? Math.round(Math.min(Math.abs(v) / scale, 1) * 70) : 0;
  if (pct === 0) return "transparent";
  if (t.signed) {
    return v > 0
      ? `color-mix(in srgb, var(--accent) ${pct}%, transparent)`
      : `color-mix(in srgb, var(--state-block) ${pct}%, transparent)`;
  }
  return `color-mix(in srgb, ${locText(r.top_location)} ${pct}%, transparent)`;
}

function shade(
  mode: Mode,
  r: Result,
  i: number,
  blocks: [number, number][],
  t: Numeric | null,
  scale: number,
): string {
  if (t) return trackShade(r, t, scale, i);
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
  const track = numericTrack(mode, result);
  const scale = track ? Math.max(0, ...track.values.map((v) => Math.abs(v))) : 0;
  return (
    <div>
      <p className="seq">
        {seq.split("").map((aa, i) => {
          const inSp = sp && i >= sp[0] && i < sp[1];
          const inTm = tms.some(([a, b]) => i >= a && i < b);
          return (
            <span
              key={i}
              title={`${i + 1} ${aa}${mode === "hydropathy" ? ` · KD ${result.hydropathy[i]?.toFixed(2)}` : track ? ` · ${track.method} ${(track.values[i] ?? 0).toFixed(3)}` : ""}`}
              style={{
                background: shade(mode, result, i, blocks, track, scale),
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
        <span>shade: {track ? track.method : mode}</span>
        {track?.signed && <span>accent: supports the label · alert: argues against</span>}
      </div>
    </div>
  );
}
