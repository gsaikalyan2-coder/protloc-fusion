/*
 * StructureViewer - a dependency-free canvas view of an AlphaFold C-alpha trace.
 *
 * It draws one point per residue, joined in sequence order, and lets the reader rotate (drag or
 * arrow keys), zoom (wheel or + and -), spin, and colour by AlphaFold's pLDDT or by whichever
 * per-residue explanation track the Inspector is showing. The structure is orientation only: it
 * is an AlphaFold DB prediction, not an experimental structure and not this model's output, and
 * no score uses it. Colours are read from the design tokens at draw time, never hard-coded.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
} from "react";
import type { Result, StructureDoc } from "../services/demoApi";
import {
  PLDDT_BANDS,
  PITCH_LIMIT,
  ZOOM_MAX,
  ZOOM_MIN,
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

type ColourBy = "plddt" | "explanation";

const BAND_TOKEN: Record<PlddtBand, string> = {
  very_high: "--accent",
  confident: "--state-note",
  low: "--state-warn",
  very_low: "--state-block",
};

const START_VIEW: View = { yaw: 0.6, pitch: -0.3, zoom: 1 };
const HIT_RADIUS = 12;

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
      positive: t.signed ? "--accent" : `--loc-${r.top_location}-ink`,
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
  cursor: number;
  view: View;
}

/** Paints the scene and returns the projected points for hit-testing. */
function paint(scene: Scene): Projected[] {
  const { canvas, points, plddt, colourBy, explanation, cursor, view } = scene;
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

  ctx.strokeStyle = token("--border-stronger");
  ctx.lineWidth = 1;
  ctx.beginPath();
  projected.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();

  const scale = explanation ? explanation.values.reduce((m, v) => Math.max(m, Math.abs(v)), 0) : 0;
  /** Fill colour and strength (0 to 1) of one residue; strength becomes transparency. */
  const colourOf = (i: number): { colour: string; strength: number } => {
    if (colourBy === "explanation" && explanation) {
      const v = explanation.values[i] ?? 0;
      const strength = scale > 0 ? Math.min(Math.abs(v) / scale, 1) : 0;
      const name = explanation.signed && v < 0 ? explanation.negative : explanation.positive;
      return { colour: token(name), strength: 0.2 + 0.8 * strength };
    }
    return { colour: token(BAND_TOKEN[plddtBand(plddt[i] ?? 0)]), strength: 1 };
  };

  const base = Math.max(2, 0.012 * Math.min(width, height) * view.zoom ** 0.5);
  const ordered = [...projected].sort((a, b) => a.depth - b.depth);
  for (const p of ordered) {
    const nearness = clamp((p.depth + points.radius) / (2 * points.radius), 0, 1);
    const { colour, strength } = colourOf(p.index);
    ctx.globalAlpha = (0.5 + 0.5 * nearness) * strength;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(p.x, p.y, base * p.scale, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  const at = projected[cursor];
  if (at) {
    ctx.strokeStyle = token("--text-primary");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(at.x, at.y, base * at.scale * 2.4, 0, 2 * Math.PI);
    ctx.stroke();
  }
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
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const points = useMemo(() => centered(toPoints(structure.ca)), [structure]);
  const explanation = useMemo(() => explanationFor(mode, result), [mode, result]);
  const [colourBy, setColourBy] = useState<ColourBy>("plddt");
  const [spin, setSpin] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  const effective: ColourBy = colourBy === "explanation" && explanation ? "explanation" : "plddt";
  const last = Math.max(0, structure.ca.length - 1);
  const shown = hover ?? Math.min(cursor, last);

  const redraw = useCallback((): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    projectedRef.current = paint({
      canvas,
      points,
      plddt: structure.plddt,
      colourBy: effective,
      explanation,
      cursor: Math.min(cursor, last),
      view: viewRef.current,
    });
  }, [points, structure.plddt, effective, explanation, cursor, last]);

  // Match the backing store to the displayed size and the device pixel ratio.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const fit = (): void => {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(canvas.clientWidth * ratio);
      canvas.height = Math.round(canvas.clientHeight * ratio);
      redraw();
    };
    fit();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(fit);
    observer.observe(canvas);
    return (): void => {
      observer.disconnect();
    };
  }, [redraw]);

  useEffect(() => {
    if (!spin) return undefined;
    let frame = 0;
    const tick = (): void => {
      viewRef.current.yaw += 0.006;
      redraw();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return (): void => {
      cancelAnimationFrame(frame);
    };
  }, [spin, redraw]);

  // Wheel zoom needs a non-passive listener so the page does not scroll with it.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const view = viewRef.current;
      view.zoom = clamp(view.zoom * Math.exp(-e.deltaY * 0.001), ZOOM_MIN, ZOOM_MAX);
      redraw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return (): void => {
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [redraw]);

  function rotateBy(dyaw: number, dpitch: number): void {
    const view = viewRef.current;
    view.yaw += dyaw;
    view.pitch = clamp(view.pitch + dpitch, -PITCH_LIMIT, PITCH_LIMIT);
    redraw();
  }

  function zoomBy(factor: number): void {
    const view = viewRef.current;
    view.zoom = clamp(view.zoom * factor, ZOOM_MIN, ZOOM_MAX);
    redraw();
  }

  function reset(): void {
    viewRef.current = { ...START_VIEW };
    redraw();
  }

  function onKeyDown(e: KeyboardEvent<HTMLCanvasElement>): void {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const keys: Record<string, () => void> = {
      ArrowLeft: () => {
        rotateBy(-0.12, 0);
      },
      ArrowRight: () => {
        rotateBy(0.12, 0);
      },
      ArrowUp: () => {
        rotateBy(0, -0.12);
      },
      ArrowDown: () => {
        rotateBy(0, 0.12);
      },
      "+": () => {
        zoomBy(1.15);
      },
      "=": () => {
        zoomBy(1.15);
      },
      "-": () => {
        zoomBy(1 / 1.15);
      },
      "0": reset,
    };
    const action = keys[e.key];
    if (!action) return;
    e.preventDefault();
    action();
  }

  function onPointerDown(e: PointerEvent<HTMLCanvasElement>): void {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY };
  }

  function onPointerMove(e: PointerEvent<HTMLCanvasElement>): void {
    const drag = dragRef.current;
    if (drag) {
      rotateBy((e.clientX - drag.x) * 0.01, (e.clientY - drag.y) * 0.01);
      dragRef.current = { x: e.clientX, y: e.clientY };
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    setHover(
      nearest(projectedRef.current, e.clientX - rect.left, e.clientY - rect.top, HIT_RADIUS),
    );
  }

  function onPointerUp(e: PointerEvent<HTMLCanvasElement>): void {
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }

  const value = structure.plddt[shown] ?? 0;
  const band = plddtBand(value);

  return (
    <div className="structure">
      <div className="row" role="group" aria-label="Structure view controls">
        <button
          className="btn"
          aria-pressed={effective === "plddt"}
          onClick={() => {
            setColourBy("plddt");
          }}
        >
          colour: AlphaFold pLDDT
        </button>
        <button
          className="btn"
          aria-pressed={effective === "explanation"}
          disabled={!explanation}
          title={
            explanation
              ? "Draw the track shown above on the AlphaFold backbone."
              : "This view has no per-residue track to draw."
          }
          onClick={() => {
            setColourBy("explanation");
          }}
        >
          colour: explanation track
        </button>
        <button
          className="btn"
          aria-pressed={spin}
          onClick={() => {
            setSpin((s) => !s);
          }}
        >
          spin
        </button>
        <button
          className="btn"
          onClick={() => {
            zoomBy(1.15);
          }}
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          className="btn"
          onClick={() => {
            zoomBy(1 / 1.15);
          }}
          aria-label="Zoom out"
        >
          -
        </button>
        <button className="btn" onClick={reset}>
          reset view
        </button>
      </div>
      <canvas
        ref={canvasRef}
        className="structure-canvas"
        tabIndex={0}
        role="img"
        aria-label={`3D backbone of ${result.id}, ${String(structure.length)} residues, from AlphaFold DB. Drag or use the arrow keys to rotate, plus and minus to zoom, 0 to reset.`}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => {
          setHover(null);
        }}
      />
      <p className="mono structure-status" role="status">
        {describeResidue(result, mode, shown)} · AlphaFold pLDDT {value.toFixed(1)} (
        {bandLabel(band)})
      </p>
      <div className="structure-legend">
        {effective === "plddt" ? (
          PLDDT_BANDS.map((b) => (
            <span key={b.band}>
              <i style={{ background: `var(${BAND_TOKEN[b.band]})` }} />
              {b.label}
            </span>
          ))
        ) : (
          <span>
            colour: {explanation?.signed ? "signed track, stronger is larger" : "track strength"} ·
            ring: residue selected in the sequence view
          </span>
        )}
      </div>
    </div>
  );
}
