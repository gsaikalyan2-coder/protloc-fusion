/* Geometry for the protein-space map. Pure functions with no DOM, so they are unit-tested. */
import type { SpacePoint } from "./demoApi";
import { clamp } from "./structureMath";

/** Coordinates beyond this many robust spreads are drawn at the edge of the box. */
export const EDGE = 1.15;

/**
 * Each axis divided by its 97th-percentile magnitude, so a handful of outliers do not squash the
 * cloud into the middle of the box. Axes are drawn at equal length; the caption gives each
 * component's share of the variance.
 */
export function normalised(points: SpacePoint[]): [number, number, number][] {
  const spread = [0, 1, 2].map((axis) => {
    const magnitudes = points.map((p) => Math.abs(p.xyz[axis] ?? 0)).sort((a, b) => a - b);
    const at = magnitudes[Math.floor(0.97 * (magnitudes.length - 1))] ?? 1;
    return at > 0 ? at : 1;
  });
  return points.map((p) => [
    clamp(p.xyz[0] / (spread[0] ?? 1), -EDGE, EDGE),
    clamp(p.xyz[1] / (spread[1] ?? 1), -EDGE, EDGE),
    clamp(p.xyz[2] / (spread[2] ?? 1), -EDGE, EDGE),
  ]);
}
