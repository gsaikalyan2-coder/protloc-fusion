/*
 * StructureViewer - a dependency-free canvas view of an AlphaFold C-alpha trace.
 *
 * The backbone is drawn as a depth-sorted tube: one segment per residue pair, thicker and more
 * opaque toward the viewer, each with a thin outline in the card colour so crossing strands read
 * as over and under. It is coloured by AlphaFold's pLDDT or by whichever per-residue explanation
 * track Predict is showing. The predicted signal peptide is a halo in the extracellular colour
 * (DESIGN.md §2.4) and the cleavage site is a ring with its position. Drag or arrow keys rotate,
 * wheel or +/- zoom, 0 resets; the view turns slowly until the first interaction, and not at all
 * under reduced motion.
 *
 * The structure is orientation only: it is an AlphaFold DB prediction, not an experimental
 * structure and not this model's output, and no score uses it. Colours are read from the design
 * tokens at draw time, never hard-coded.
 */
import { useCallback, useMemo, useRef, useState, type ReactElement } from "react";
import { Pause, Play, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import type { Result, StructureDoc } from "../services/demoApi";
import {
  PLDDT_BANDS,
  bandLabel,
  centered,
  clamp,
  nearest,
  plddtBand,
  project,
  toPoints,
  type PlddtBand,
  type Projected,
  type View,
} from "../services/structureMath";
import { describeResidue, numericTrack, type Mode } from "./SequenceTrack";
import { Segmented } from "./ui";
import { useOrbit } from "./useOrbit";

type ColourBy = "plddt" | "explanation";

const BAND_TOKEN: Record<PlddtBand, string> = {
  very_high: "--accent",
  confident: "--state-note",
  low: "--state-warn",
  very_low: "--state-block",
};

const START_VIEW: View = { yaw: 0.6, pitch: -0.3, zoom: 1 };
const HIT_RADIUS = 12;
/** Residues per depth-sorted run of the tube. */
const RUN = 4;

interface Explanation {
  values: number[];
  signed: boolean;
  /** Token for a positive value (or the only colour of an unsigned track). */
  positive: string;
  /** Token for a negative value; unused for an unsigned track. */
  negative: string;
}

/** The per-residue values behind a mode, or null when the mode has no numeric track. */
function explanationFor(mode: Mode, r: Result): Explanation | null {
  const t = numericTrack(mode, r);
  if (t) {
    return {
      values: t.values,
      signed: t.signed,
      positive: t.signed ? "--accent" : `--loc-${r.top_location.toLowerCase()}-ink`,
      negative: "--state-block",
    };
  }
  if (mode === "hydropathy") {
    return { values: r.hydropathy, signed: true, positive: "--state-warn", negative: "--accent" };
  }
  return null;
}

interface Scene {
  canvas: HTMLCanvasElement;
  points: ReturnType<typeof centered>;
  plddt: number[];
  colourBy: ColourBy;
  explanation: Explanation | null;
  /** 0-based inclusive start and exclusive end of the signal peptide, or null. */
  signal: [number, number] | null;
  /** 1-based position the signal peptide is cleaved after, or null. */
  cleavage: number | null;
  cursor: number;
  hover: number | null;
  view: View;
}

/** Paints the scene and returns the projected points for hit-testing. */
function paint(scene: Scene): Projected[] {
  const { canvas, points, plddt, colourBy, explanation, signal, cleavage, cursor, hover, view } =
    scene;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const css = getComputedStyle(canvas);
  const token = (name: string): string => css.getPropertyValue(name).trim() || css.color;
  const projected = project(points.points, points.radius, view, width, height);
  if (projected.length === 0) return projected;

  const scale = explanation ? explanation.values.reduce((m, v) => Math.max(m, Math.abs(v)), 0) : 0;
  const colours = new Map<string, string>();
  const tokenColour = (name: string): string => {
    let c = colours.get(name);
    if (c === undefined) {
      c = token(name);
      colours.set(name, c);
    }
    return c;
  };
  /** Colour and strength (0 to 1) of one residue; strength becomes opacity. */
  const colourOf = (i: number): { colour: string; strength: number } => {
    if (colourBy === "explanation" && explanation) {
      const v = explanation.values[i] ?? 0;
      const strength = scale > 0 ? Math.min(Math.abs(v) / scale, 1) : 0;
      const name = explanation.signed && v < 0 ? explanation.negative : explanation.positive;
      return { colour: tokenColour(name), strength: 0.25 + 0.75 * strength };
    }
    return { colour: tokenColour(BAND_TOKEN[plddtBand(plddt[i] ?? 0)]), strength: 1 };
  };

  const tube = clamp(0.011 * Math.min(width, height), 3, 6.5) * view.zoom ** 0.5;
  const outline = tokenColour("--surface-1");
  const halo = tokenColour("--loc-ext-ink");
  const inSignal = (i: number): boolean => signal !== null && i >= signal[0] && i < signal[1];

  // Short runs of residues, back to front. Each run gets one outline (so a strand passing in
  // front of another reads as over it) and then its coloured segments, so joints stay seamless.
  const runs: { start: number; end: number; depth: number }[] = [];
  for (let start = 0; start + 1 < projected.length; start += RUN) {
    const end = Math.min(start + RUN, projected.length - 1);
    let depth = 0;
    for (let i = start; i <= end; i++) depth += projected[i]?.depth ?? 0;
    runs.push({ start, end, depth: depth / (end - start + 1) });
  }
  runs.sort((s, t) => s.depth - t.depth);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const { start, end, depth } of runs) {
    const first = projected[start];
    if (!first) continue;
    const nearness = clamp((depth + points.radius) / (2 * points.radius), 0, 1);
    const width = tube * first.scale;
    const trace = (): void => {
      ctx.beginPath();
      ctx.moveTo(first.x, first.y);
      for (let i = start + 1; i <= end; i++) {
        const p = projected[i];
        if (p) ctx.lineTo(p.x, p.y);
      }
    };
    if (inSignal(start) || inSignal(end - 1)) {
      ctx.globalAlpha = 0.28;
      ctx.strokeStyle = halo;
      ctx.lineWidth = width + 9;
      trace();
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = outline;
    ctx.lineWidth = width + 3;
    trace();
    ctx.stroke();
    for (let i = start; i < end; i++) {
      const a = projected[i];
      const b = projected[i + 1];
      if (!a || !b) continue;
      const { colour, strength } = colourOf(i);
      ctx.globalAlpha = (0.5 + 0.5 * nearness) * strength;
      ctx.strokeStyle = colour;
      ctx.lineWidth = tube * ((a.scale + b.scale) / 2);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  const ringAt = (i: number, extra: number, colour: string, lineWidth: number): void => {
    const p = projected[i];
    if (!p) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.arc(p.x, p.y, tube * p.scale * 0.5 + extra, 0, 2 * Math.PI);
    ctx.stroke();
  };

  if (cleavage !== null && cleavage > 0 && cleavage <= projected.length) {
    const p = projected[cleavage - 1];
    if (p) {
      ringAt(cleavage - 1, 6, tokenColour("--text-primary"), 1.5);
      const label = `cleaves after ${String(cleavage)}`;
      ctx.font = `500 11px ${token("--font-mono")}`;
      ctx.textAlign = "left";
      ctx.lineWidth = 4;
      ctx.strokeStyle = tokenColour("--surface-1");
      ctx.strokeText(label, p.x + 12, p.y - 10);
      ctx.fillStyle = tokenColour("--text-primary");
      ctx.fillText(label, p.x + 12, p.y - 10);
    }
  }
  ringAt(cursor, 5, tokenColour("--text-primary"), 2.5);
  if (hover !== null && hover !== cursor) ringAt(hover, 4, tokenColour("--text-secondary"), 1.5);
  return projected;
}

export function StructureViewer({
  result,
  mode,
  structure,
  cursor,
}: {
  result: Result;
  mode: Mode;
  structure: StructureDoc;
  /** 0-based residue the sequence view is on; ringed in the 3D view. */
  cursor: number;
}): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewRef = useRef<View>({ ...START_VIEW });
  const projectedRef = useRef<Projected[]>([]);
  const points = useMemo(() => centered(toPoints(structure.ca)), [structure]);
  const explanation = useMemo(() => explanationFor(mode, result), [mode, result]);
  const [colourBy, setColourBy] = useState<ColourBy>("plddt");
  const [hover, setHover] = useState<number | null>(null);

  const effective: ColourBy = colourBy === "explanation" && explanation ? "explanation" : "plddt";
  const last = Math.max(0, structure.ca.length - 1);
  const shown = hover ?? Math.min(cursor, last);
  const region = result.sorting_signal.region;
  const signal: [number, number] | null =
    result.sorting_signal.type === "signal_peptide" && region ? [region[0], region[1]] : null;
  const cleavage = signal ? result.sorting_signal.cleavage_site : null;

  const redraw = useCallback((): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    projectedRef.current = paint({
      canvas,
      points,
      plddt: structure.plddt,
      colourBy: effective,
      explanation,
      signal,
      cleavage,
      cursor: Math.min(cursor, last),
      hover,
      view: viewRef.current,
    });
    // signal is rebuilt each render from result; its parts are the real dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, structure.plddt, effective, explanation, region, cleavage, cursor, last, hover]);

  const orbit = useOrbit(canvasRef, viewRef, redraw, START_VIEW, (x, y) => {
    setHover(nearest(projectedRef.current, x, y, HIT_RADIUS));
  });

  const value = structure.plddt[shown] ?? 0;
  const band = plddtBand(value);
  const hp = hover !== null ? projectedRef.current[hover] : undefined;

  return (
    <div className="structure">
      <div className="viz-toolbar">
        <Segmented
          label="Colour the backbone by"
          value={effective}
          onChange={setColourBy}
          options={[
            { value: "plddt", label: "AlphaFold pLDDT" },
            {
              value: "explanation",
              label: "explanation track",
              disabled: !explanation,
              reason: "This view has no per-residue track to draw.",
            },
          ]}
        />
        <div className="row tight" role="group" aria-label="View controls">
          <button
            type="button"
            className="icon-btn"
            aria-pressed={orbit.spin}
            aria-label={orbit.spin ? "Stop rotating" : "Rotate"}
            title={orbit.spin ? "Stop rotating" : "Rotate"}
            onClick={() => {
              orbit.setSpin(!orbit.spin);
            }}
          >
            {orbit.spin ? <Pause size={15} /> : <Play size={15} />}
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label="Zoom in"
            title="Zoom in"
            onClick={() => {
              orbit.zoomBy(1.15);
            }}
          >
            <ZoomIn size={15} />
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label="Zoom out"
            title="Zoom out"
            onClick={() => {
              orbit.zoomBy(1 / 1.15);
            }}
          >
            <ZoomOut size={15} />
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label="Reset view"
            title="Reset view"
            onClick={orbit.reset}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      </div>
      <div className="viz-stage">
        <canvas
          ref={canvasRef}
          className="structure-canvas"
          tabIndex={0}
          role="img"
          aria-label={`3D backbone of ${result.id}, ${String(structure.length)} residues, from AlphaFold DB. Drag or use the arrow keys to rotate, plus and minus to zoom, 0 to reset.`}
          {...orbit.handlers}
          onPointerEnter={() => {
            orbit.setSpin(false);
          }}
          onPointerLeave={() => {
            setHover(null);
          }}
        />
        {hover !== null && hp && (
          <div className="viz-tip" style={{ left: hp.x, top: hp.y }} aria-hidden="true">
            <strong className="mono">
              {result.sequence.charAt(hover)}
              {hover + 1}
            </strong>
            <span className="note">
              pLDDT {(structure.plddt[hover] ?? 0).toFixed(1)}
              {signal && hover >= signal[0] && hover < signal[1] ? " · signal peptide" : ""}
            </span>
          </div>
        )}
      </div>
      <p className="mono structure-status" role="status">
        {describeResidue(result, mode, shown)} · AlphaFold pLDDT {value.toFixed(1)} (
        {bandLabel(band)})
      </p>
      <div className="structure-legend">
        {effective === "plddt" ? (
          PLDDT_BANDS.map((b) => (
            <span key={b.band} className={`legend-band band-${b.band}`}>
              <i />
              {b.label}
            </span>
          ))
        ) : (
          <span>
            colour: {explanation?.signed ? "signed track, stronger is larger" : "track strength"}
          </span>
        )}
        {signal && (
          <span className="legend-band band-signal">
            <i />
            signal peptide {signal[0] + 1}–{signal[1]}
          </span>
        )}
        <span>ring: residue selected in the sequence view</span>
      </div>
    </div>
  );
}
