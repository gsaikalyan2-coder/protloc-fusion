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
import {
  useEffect,
  useId,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
} from "react";
import type { Result } from "../services/demoApi";
import { locText } from "./ui";

export type Mode = "attribution" | "attention" | "signal" | "hydropathy" | "alignment";

export interface Numeric {
  values: number[];
  signed: boolean;
  method: string;
}

/** The numeric track a mode draws, or null for hydropathy and alignment. */
export function numericTrack(mode: Mode, r: Result): Numeric | null {
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

/**
 * Shade strength is a fraction of --shade-max, which app.css sets per theme so that the residue
 * letter keeps 4.5:1 contrast on the strongest shade (about 60% in light, 40% in dark).
 */
const amount = (ratio: number): string => `calc(var(--shade-max) * ${ratio.toFixed(3)})`;

/** Shade for one residue: |value| scaled by the track's own largest magnitude. */
function trackShade(r: Result, t: Numeric, scale: number, i: number): string {
  const v = t.values[i] ?? 0;
  const ratio = scale > 0 ? Math.min(Math.abs(v) / scale, 1) : 0;
  if (ratio === 0) return "transparent";
  if (t.signed) {
    return v > 0
      ? `color-mix(in srgb, var(--accent) ${amount(ratio)}, transparent)`
      : `color-mix(in srgb, var(--state-block) ${amount(ratio)}, transparent)`;
  }
  return `color-mix(in srgb, ${locText(r.top_location)} ${amount(ratio)}, transparent)`;
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
      ? `color-mix(in srgb, var(--state-warn) ${amount(Math.min(h / 4.5, 1))}, transparent)`
      : `color-mix(in srgb, var(--accent) ${amount(Math.min(-h / 4.5, 1))}, transparent)`;
  }
  return blocks.some(([a, b]) => i >= a && i < b) ? "var(--accent-subtle)" : "transparent";
}

function tmSegments(r: Result): [number, number][] {
  return (r.features.tm_segments as [number, number][] | undefined) ?? [];
}

/** What a screen reader says for one residue. Also shown as text under the viewer. */
export function describeResidue(
  result: Result,
  mode: Mode,
  i: number,
  blocks: [number, number][] = [],
): string {
  const parts = [`Residue ${String(i + 1)}, ${result.sequence.charAt(i)}`];
  const track = numericTrack(mode, result);
  if (mode === "hydropathy") {
    parts.push(`hydropathy ${(result.hydropathy[i] ?? 0).toFixed(2)}`);
  } else if (track) {
    const v = track.values[i] ?? 0;
    let side = "";
    if (track.signed && v > 0) side = ", supports the label";
    if (track.signed && v < 0) side = ", argues against the label";
    parts.push(`${track.method} ${v.toFixed(3)}${side}`);
  } else if (mode === "alignment") {
    parts.push(
      blocks.some(([a, b]) => i >= a && i < b)
        ? "inside an aligned block"
        : "outside aligned blocks",
    );
  }
  const sp = result.sorting_signal.region;
  if (sp && i >= sp[0] && i < sp[1]) parts.push("in the signal peptide region");
  const cs = result.sorting_signal.cleavage_site;
  if (cs !== null && i === cs - 1) parts.push("last residue before the cleavage site");
  if (tmSegments(result).some(([a, b]) => i >= a && i < b)) {
    parts.push("in a predicted transmembrane segment");
  }
  return parts.join(", ");
}

/** Residues per block. Each block carries its first residue number above it, like a ruler. */
export const BLOCK = 10;

/** Block start indices (0-based) for a sequence of `length` residues. */
export function blockStarts(length: number): number[] {
  return Array.from({ length: Math.ceil(length / BLOCK) }, (_, k) => k * BLOCK);
}

/** One legend entry: a swatch drawn from tokens by app.css, and what it means. */
interface LegendItem {
  swatch:
    | "support"
    | "against"
    | "shade"
    | "hydrophobic"
    | "hydrophilic"
    | "aligned"
    | "signal"
    | "cleavage"
    | "tm";
  text: string;
}

/** The legend for a mode: only what the viewer can actually be showing. */
export function legendItems(
  mode: Mode,
  track: Numeric | null,
  cleavage: number | null,
): LegendItem[] {
  const items: LegendItem[] = [];
  if (track?.signed) {
    items.push({ swatch: "support", text: "supports the label" });
    items.push({ swatch: "against", text: "argues against it (dotted top rule)" });
  } else if (track) {
    items.push({ swatch: "shade", text: `${track.method}: stronger shade, larger value` });
  } else if (mode === "hydropathy") {
    items.push({ swatch: "hydrophobic", text: "hydrophobic" });
    items.push({ swatch: "hydrophilic", text: "hydrophilic" });
  } else {
    items.push({ swatch: "aligned", text: "inside an aligned block" });
  }
  items.push({ swatch: "signal", text: "signal-peptide region" });
  items.push({
    swatch: "cleavage",
    text: cleavage !== null ? `cleavage site (after ${String(cleavage)})` : "cleavage site (none)",
  });
  items.push({ swatch: "tm", text: "transmembrane segment" });
  return items;
}

/* Keys: Left/Right move one residue, Up/Down ten, PageUp/PageDown fifty, Home/End the ends. */
const STEP: Record<string, number> = {
  ArrowRight: 1,
  ArrowLeft: -1,
  ArrowDown: 10,
  ArrowUp: -10,
  PageDown: 50,
  PageUp: -50,
};

export function SequenceTrack({
  result,
  mode,
  blocks = [],
  limit = 600,
  onCursor,
}: {
  result: Result;
  mode: Mode;
  blocks?: [number, number][];
  limit?: number;
  /** Told the cursor residue (0-based) whenever it changes, so a 3D view can follow it. */
  onCursor?: (index: number) => void;
}): ReactElement {
  const seq = result.sequence.slice(0, limit);
  const sp = result.sorting_signal.region;
  const cs = result.sorting_signal.cleavage_site;
  const tms = tmSegments(result);
  const track = numericTrack(mode, result);
  const scale = track ? Math.max(0, ...track.values.map((v) => Math.abs(v))) : 0;

  const uid = useId();
  const [cursor, setCursor] = useState(0);
  const last = seq.length - 1;
  const at = Math.max(0, Math.min(cursor, last));
  const rid = (i: number): string => `${uid}-r${String(i)}`;

  useEffect(() => {
    onCursor?.(at);
  }, [at, onCursor]);

  // Keep the cursor residue visible while the viewer has focus.
  useEffect(() => {
    const el = document.getElementById(rid(at));
    if (el && el.closest('[role="listbox"]') === document.activeElement && "scrollIntoView" in el) {
      el.scrollIntoView({ block: "nearest" });
    }
  });

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>): void {
    if (e.ctrlKey || e.metaKey || e.altKey || last < 0) return;
    let next: number | null = null;
    const d = STEP[e.key];
    if (d !== undefined) next = at + d;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = last;
    if (next === null) return;
    e.preventDefault();
    setCursor(Math.max(0, Math.min(last, next)));
  }

  return (
    <div>
      <div
        className="seq"
        role="listbox"
        tabIndex={0}
        aria-label={`Residues of ${result.id}, ${String(result.length)} in total. Left and right arrows move one residue, up and down ten, Home and End jump to the ends.`}
        aria-activedescendant={last >= 0 ? rid(at) : undefined}
        onKeyDown={onKeyDown}
      >
        {blockStarts(seq.length).map((start) => (
          <span key={start} className="seq-block" role="presentation">
            <span className="seq-pos mono" aria-hidden="true">
              {start + 1}
            </span>
            <span className="seq-res" role="presentation">
              {seq
                .slice(start, start + BLOCK)
                .split("")
                .map((aa, k) => {
                  const i = start + k;
                  const inSp = sp && i >= sp[0] && i < sp[1];
                  const inTm = tms.some(([a, b]) => i >= a && i < b);
                  return (
                    <span
                      key={i}
                      id={rid(i)}
                      role="option"
                      aria-selected={i === at}
                      aria-label={describeResidue(result, mode, i, blocks)}
                      className={i === at ? "cursor" : undefined}
                      onClick={() => {
                        setCursor(i);
                      }}
                      style={{
                        background: shade(mode, result, i, blocks, track, scale),
                        borderBottom: inSp
                          ? "2px solid var(--state-warn)"
                          : inTm
                            ? `2px solid ${locText("mem")}`
                            : "2px solid transparent",
                        borderTop:
                          track?.signed && (track.values[i] ?? 0) < 0
                            ? "2px dotted var(--state-block)"
                            : "2px solid transparent",
                        borderLeft:
                          cs !== null && i === cs ? "2px solid var(--text-primary)" : undefined,
                      }}
                    >
                      {aa}
                    </span>
                  );
                })}
            </span>
          </span>
        ))}
      </div>
      {last >= 0 && (
        <p className="mono" style={{ fontSize: "var(--text-xs)", margin: "var(--space-2) 0" }}>
          {describeResidue(result, mode, at, blocks)}
        </p>
      )}
      {result.length > limit && (
        <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
          Showing first {limit} of {result.length} residues.
        </p>
      )}
      <ul
        className="seq-legend"
        aria-label="Legend"
        style={{ "--swatch": locText(result.top_location) } as CSSProperties}
      >
        {legendItems(mode, track, cs).map((item) => (
          <li key={item.swatch}>
            <span className={`seq-swatch sw-${item.swatch}`} aria-hidden="true" />
            {item.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
