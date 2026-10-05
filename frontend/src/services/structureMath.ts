/* Geometry for the structure viewer. Pure functions with no DOM, so they are unit-tested. */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface View {
  yaw: number;
  pitch: number;
  zoom: number;
}

export interface Projected {
  index: number;
  x: number;
  y: number;
  /** Distance toward the viewer after rotation; larger is nearer. */
  depth: number;
  /** Perspective factor: above 1 is nearer than the centre, below 1 is farther. */
  scale: number;
}

export type PlddtBand = "very_high" | "confident" | "low" | "very_low";

/** AlphaFold's own pLDDT bands, in the order the legend lists them. */
export const PLDDT_BANDS: readonly { band: PlddtBand; label: string }[] = [
  { band: "very_high", label: "very high (above 90)" },
  { band: "confident", label: "confident (70 to 90)" },
  { band: "low", label: "low (50 to 70)" },
  { band: "very_low", label: "very low (50 or below)" },
];

export const PITCH_LIMIT = 1.4;
export const ZOOM_MIN = 0.4;
export const ZOOM_MAX = 4;

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

export function plddtBand(value: number): PlddtBand {
  if (value > 90) return "very_high";
  if (value > 70) return "confident";
  if (value > 50) return "low";
  return "very_low";
}

export function bandLabel(band: PlddtBand): string {
  return PLDDT_BANDS.find((b) => b.band === band)?.label ?? band;
}

/** Rows of [x, y, z] as points; a short row reads as 0 rather than throwing. */
export function toPoints(ca: number[][]): Vec3[] {
  return ca.map((row) => ({ x: row[0] ?? 0, y: row[1] ?? 0, z: row[2] ?? 0 }));
}

/** The points moved so their mean is the origin, and the largest distance from it (min 1). */
export function centered(points: Vec3[]): { points: Vec3[]; radius: number } {
  const n = points.length;
  if (n === 0) return { points: [], radius: 1 };
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
    sz += p.z;
  }
  const moved = points.map((p) => ({ x: p.x - sx / n, y: p.y - sy / n, z: p.z - sz / n }));
  const radius = moved.reduce((m, p) => Math.max(m, Math.hypot(p.x, p.y, p.z)), 1);
  return { points: moved, radius };
}

/** Yaw about the vertical axis, then pitch about the horizontal one. */
export function rotate(p: Vec3, yaw: number, pitch: number): Vec3 {
  const x1 = p.x * Math.cos(yaw) + p.z * Math.sin(yaw);
  const z1 = -p.x * Math.sin(yaw) + p.z * Math.cos(yaw);
  return {
    x: x1,
    y: p.y * Math.cos(pitch) - z1 * Math.sin(pitch),
    z: p.y * Math.sin(pitch) + z1 * Math.cos(pitch),
  };
}

/** Screen positions of centred points with weak perspective, fitted to the canvas. */
export function project(
  points: Vec3[],
  radius: number,
  view: View,
  width: number,
  height: number,
): Projected[] {
  const fit = (0.46 * Math.min(width, height)) / radius;
  const camera = 3 * radius;
  return points.map((p, index) => {
    const q = rotate(p, view.yaw, view.pitch);
    const scale = camera / (camera - q.z);
    return {
      index,
      x: width / 2 + q.x * fit * view.zoom * scale,
      y: height / 2 - q.y * fit * view.zoom * scale,
      depth: q.z,
      scale,
    };
  });
}

/** Index of the projected point closest to (x, y) within maxDistance, or null. */
export function nearest(
  projected: Projected[],
  x: number,
  y: number,
  maxDistance: number,
): number | null {
  let best: number | null = null;
  let bestDistance = maxDistance;
  for (const p of projected) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d <= bestDistance) {
      bestDistance = d;
      best = p.index;
    }
  }
  return best;
}
