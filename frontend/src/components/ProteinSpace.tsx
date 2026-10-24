/*
 * ProteinSpace — the cached sequences as a rotatable 3D point cloud (GET /demo/space).
 *
 * Each point is one cached sequence, placed by the first three principal components of its
 * mean-pooled ESM-2 150M vector (the representation every head reads) and coloured by its
 * Swiss-Prot location (database annotation, not the model's call). A highlighted sequence is drawn
 * larger, with lines to its nearest cached neighbours in all 640 dimensions. Orientation only:
 * nothing on this map is a prediction and nothing here feeds a score.
 *
 * Dependency-free canvas, like the structure viewer: drag or arrow keys rotate, wheel or +/- zoom,
 * 0 resets. Colours are read from the design tokens at draw time.
 */
import {
  useCallback,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from "react";
import { Pause, Play, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import type { SpaceDoc, SpacePoint } from "../services/demoApi";
import { normalised } from "../services/spaceMath";
import { clamp, nearest, project, type Projected, type View } from "../services/structureMath";
import { Chip, locText } from "./ui";
import { useOrbit } from "./useOrbit";

/** DESIGN.md §2.1 order: legends never reorder. */
const ORDER = ["CYT", "NUC", "EXT", "MEM", "MIT", "PLA", "ER", "LYS", "GOL", "PER"];
const START: View = { yaw: 0.7, pitch: -0.35, zoom: 1 };
const RADIUS = 1.85;
const HIT = 10;

const tokenOf = (code: string | undefined): string =>
  code ? `--loc-${code.toLowerCase()}-ink` : "--text-disabled";

interface PaintArgs {
  canvas: HTMLCanvasElement;
  points: SpacePoint[];
  coords: [number, number, number][];
  view: View;
  focus: number | null;
  hover: number | null;
  filter: Set<string>;
}

const CUBE: [number, number, number][] = [
  [-1, -1, -1],
  [1, -1, -1],
  [1, 1, -1],
  [-1, 1, -1],
  [-1, -1, 1],
  [1, -1, 1],
  [1, 1, 1],
  [-1, 1, 1],
];
const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
]; // prettier-ignore

function paint({ canvas, points, coords, view, focus, hover, filter }: PaintArgs): Projected[] {
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const css = getComputedStyle(canvas);
  const token = (name: string): string => css.getPropertyValue(name).trim() || css.color;

  // Reference box and axes: faint, so the cloud reads as a volume rather than a smear.
  const box = project(
    CUBE.map(([x, y, z]) => ({ x, y, z })),
    RADIUS,
    view,
    width,
    height,
  );
  ctx.strokeStyle = token("--border");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [a, b] of EDGES) {
    const p = box[a];
    const q = box[b];
    if (!p || !q) continue;
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(q.x, q.y);
  }
  ctx.stroke();
  const axes = project(
    [
      { x: 1.18, y: -1, z: -1 },
      { x: -1, y: 1.18, z: -1 },
      { x: -1, y: -1, z: 1.18 },
    ],
    RADIUS,
    view,
    width,
    height,
  );
  ctx.fillStyle = token("--text-muted");
  ctx.font = `500 11px ${token("--font-mono")}`;
  ctx.textAlign = "center";
  ["PC1", "PC2", "PC3"].forEach((label, i) => {
    const a = axes[i];
    if (a) ctx.fillText(label, a.x, a.y);
  });

  const projected = project(
    coords.map(([x, y, z]) => ({ x, y, z })),
    RADIUS,
    view,
    width,
    height,
  );
  const base = clamp(0.0068 * Math.min(width, height), 2.4, 4) * Math.sqrt(view.zoom);
  const focused = focus !== null ? points[focus] : undefined;
  const neighbourSet = new Set(focused?.neighbours.map(([j]) => j) ?? []);

  // Lines from the highlighted sequence to its nearest neighbours, under the points.
  const fp = focus !== null ? projected[focus] : undefined;
  if (focused && fp) {
    ctx.strokeStyle = token("--text-secondary");
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (const [j] of focused.neighbours) {
      const q = projected[j];
      if (!q) continue;
      ctx.moveTo(fp.x, fp.y);
      ctx.lineTo(q.x, q.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  const colours = new Map<string, string>();
  const colour = (code: string | undefined): string => {
    const key = tokenOf(code);
    let c = colours.get(key);
    if (c === undefined) {
      c = token(key);
      colours.set(key, c);
    }
    return c;
  };
  const ring = token("--surface-1");

  const ordered = [...projected].sort((a, b) => a.depth - b.depth);
  for (const p of ordered) {
    const point = points[p.index];
    if (!point) continue;
    const nearness = clamp((p.depth + RADIUS) / (2 * RADIUS), 0, 1);
    const shown = filter.size === 0 || point.locations.some((c) => filter.has(c));
    const isNeighbour = neighbourSet.has(p.index);
    const r = base * p.scale * (isNeighbour ? 1.35 : 1);
    ctx.globalAlpha = shown ? 0.35 + 0.65 * nearness : 0.08;
    ctx.fillStyle = colour(point.locations[0]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
    ctx.fill();
    if (shown && (point.locations.length > 1 || isNeighbour)) {
      // A second location, when there is one, is the ring; a neighbour gets a contrasting ring.
      ctx.lineWidth = isNeighbour ? 1.5 : 1.25;
      ctx.strokeStyle = isNeighbour ? token("--text-primary") : colour(point.locations[1]);
      ctx.stroke();
    } else if (shown) {
      ctx.lineWidth = 0.75;
      ctx.strokeStyle = ring;
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  if (focused && fp) {
    const r = base * fp.scale * 2.1;
    ctx.fillStyle = colour(focused.locations[0]);
    ctx.beginPath();
    ctx.arc(fp.x, fp.y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = token("--text-primary");
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(fp.x, fp.y, r + 5, 0, 2 * Math.PI);
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  const hp = hover !== null ? projected[hover] : undefined;
  if (hp) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = token("--text-primary");
    ctx.beginPath();
    ctx.arc(hp.x, hp.y, base * hp.scale + 4, 0, 2 * Math.PI);
    ctx.stroke();
  }
  return projected;
}

export function ProteinSpace({
  space,
  focusSha,
  onOpen,
  height = 460,
}: {
  space: SpaceDoc;
  /** The sequence to highlight with its neighbours, if it is on the map. */
  focusSha?: string | undefined;
  /** Open a cached sequence by accession (click a point). */
  onOpen?: ((accession: string) => void) | undefined;
  height?: number;
}): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const projectedRef = useRef<Projected[]>([]);
  const [hover, setHover] = useState<number | null>(null);
  const [filter, setFilter] = useState<Set<string>>(new Set());
  const points = space.points;
  const coords = useMemo(() => normalised(points), [points]);
  const focus = useMemo(() => {
    const i = focusSha ? points.findIndex((p) => p.sha256 === focusSha) : -1;
    return i >= 0 ? i : null;
  }, [points, focusSha]);

  const counts = useMemo(() => {
    const c = new Map<string, number>();
    for (const p of points) for (const code of p.locations) c.set(code, (c.get(code) ?? 0) + 1);
    return ORDER.filter((code) => c.has(code)).map((code) => [code, c.get(code) ?? 0] as const);
  }, [points]);

  const viewRef = useRef<View>({ ...START });
  const redraw = useCallback((): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    projectedRef.current = paint({
      canvas,
      points,
      coords,
      view: viewRef.current,
      focus,
      hover,
      filter,
    });
  }, [points, coords, focus, hover, filter]);

  const orbit = useOrbit(canvasRef, viewRef, redraw, START, (x, y) => {
    setHover(nearest(projectedRef.current, x, y, HIT));
  });

  const hovered = hover !== null ? points[hover] : undefined;
  const hp = hover !== null ? projectedRef.current[hover] : undefined;
  const v = space.explained_variance_ratio;

  const toggle = (code: string): void => {
    setFilter((f) => {
      const next = new Set(f);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
  };

  return (
    <div className="space">
      <div className="viz-toolbar">
        <div className="row tight space-legend" role="group" aria-label="Show compartments">
          {counts.map(([code, n]) => (
            <button
              key={code}
              type="button"
              className="legend-chip"
              aria-pressed={filter.has(code)}
              style={{ "--chip-ink": locText(code) } as CSSProperties}
              onClick={() => {
                toggle(code);
              }}
            >
              <span className="legend-dot" aria-hidden="true" />
              {code}
              <span className="note">{n}</span>
            </button>
          ))}
          {filter.size > 0 && (
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => {
                setFilter(new Set());
              }}
            >
              Show all
            </button>
          )}
        </div>
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
          className="viz-canvas"
          style={{ height }}
          tabIndex={0}
          role="img"
          aria-label={`Protein space: ${String(points.length)} cached sequences placed by the first three principal components of their ESM-2 vectors, coloured by Swiss-Prot location.${focus !== null ? " The current sequence is highlighted with lines to its nearest neighbours." : ""} Drag or use the arrow keys to rotate, plus and minus to zoom, 0 to reset.`}
          {...orbit.handlers}
          onPointerEnter={() => {
            orbit.setSpin(false);
          }}
          onPointerLeave={() => {
            setHover(null);
          }}
          onClick={() => {
            if (orbit.wasDrag() || !hovered || !onOpen) return;
            const id = hovered.ids[0];
            if (id) onOpen(id);
          }}
        />
        {hovered && hp && (
          <div
            className="viz-tip"
            style={{ left: hp.x, top: hp.y } as CSSProperties}
            aria-hidden="true"
          >
            <strong className="mono">{hovered.ids[0]}</strong>
            <span>{hovered.entry_name}</span>
            <span className="row tight">
              {hovered.locations.length > 0 ? (
                hovered.locations.map((c) => <Chip key={c} code={c} />)
              ) : (
                <span className="note">no location</span>
              )}
            </span>
            <span className="note">
              {hovered.in_training_set ? "in-sample" : "held-out"} · p{hovered.partition}
              {onOpen ? " · click to open" : ""}
            </span>
          </div>
        )}
      </div>

      <p className="note viz-caption">
        PCA of mean-pooled ESM-2 vectors · PC1 {(100 * (v[0] ?? 0)).toFixed(1)}%, PC2{" "}
        {(100 * (v[1] ?? 0)).toFixed(1)}%, PC3 {(100 * (v[2] ?? 0)).toFixed(1)}% of variance · each
        axis scaled to its own spread · colour = Swiss-Prot location, ring = a second location ·
        position is similarity, not a prediction.
      </p>
    </div>
  );
}

/** The highlighted sequence's nearest cached neighbours, as a list a keyboard can use. */
export function Neighbours({
  space,
  focusSha,
  onOpen,
}: {
  space: SpaceDoc;
  focusSha: string;
  onOpen?: ((accession: string) => void) | undefined;
}): ReactElement {
  const point = space.points.find((p) => p.sha256 === focusSha);
  if (!point) {
    return (
      <p className="hint">
        This sequence is not one of the cached sequences, so it has no place on the map.
      </p>
    );
  }
  return (
    <div className="neighbours">
      <div className="overline">Nearest cached sequences</div>
      <ol>
        {point.neighbours.map(([j, similarity]) => {
          const n = space.points[j];
          const id = n?.ids[0];
          if (!n || !id) return null;
          return (
            <li key={n.sha256}>
              <button
                type="button"
                className="neighbour"
                disabled={!onOpen}
                onClick={() => {
                  onOpen?.(id);
                }}
              >
                <span className="neighbour-main">
                  <span className="mono">{id}</span>
                  <span className="note">{n.entry_name}</span>
                </span>
                <span className="row tight">
                  {n.locations.map((c) => (
                    <Chip key={c} code={c} />
                  ))}
                </span>
                <span className="num" title="cosine similarity of centred ESM-2 vectors">
                  {similarity.toFixed(2)}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      <p className="note">
        Cosine similarity in all 640 dimensions. Neighbours that share a Swiss-Prot location are
        context for the model&apos;s call, not evidence for it.
      </p>
    </div>
  );
}
