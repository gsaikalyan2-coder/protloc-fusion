import { describe, expect, it } from "vitest";
import { centered, clamp, nearest, plddtBand, project, rotate, toPoints } from "./structureMath";

describe("plddtBand", () => {
  it("uses AlphaFold's own thresholds, with the boundary in the lower band", () => {
    expect(plddtBand(95)).toBe("very_high");
    expect(plddtBand(90)).toBe("confident");
    expect(plddtBand(70.1)).toBe("confident");
    expect(plddtBand(70)).toBe("low");
    expect(plddtBand(50.1)).toBe("low");
    expect(plddtBand(50)).toBe("very_low");
    expect(plddtBand(0)).toBe("very_low");
  });
});

describe("toPoints and centered", () => {
  it("reads rows as points and treats a short row as zeros", () => {
    expect(toPoints([[1, 2, 3], [4]])).toEqual([
      { x: 1, y: 2, z: 3 },
      { x: 4, y: 0, z: 0 },
    ]);
  });

  it("moves the mean to the origin and reports the largest distance", () => {
    const c = centered(
      toPoints([
        [0, 0, 0],
        [10, 0, 0],
      ]),
    );
    expect(c.points).toEqual([
      { x: -5, y: 0, z: 0 },
      { x: 5, y: 0, z: 0 },
    ]);
    expect(c.radius).toBe(5);
  });

  it("has a radius of at least 1, and an empty input is safe", () => {
    expect(centered(toPoints([[1, 1, 1]])).radius).toBe(1);
    expect(centered([])).toEqual({ points: [], radius: 1 });
  });
});

describe("rotate", () => {
  it("leaves a point alone at zero angles and keeps its length otherwise", () => {
    const p = { x: 1, y: 2, z: 3 };
    expect(rotate(p, 0, 0)).toEqual(p);
    const q = rotate(p, 0.7, -0.4);
    expect(Math.hypot(q.x, q.y, q.z)).toBeCloseTo(Math.hypot(1, 2, 3), 10);
  });

  it("a quarter turn of yaw sends +z to +x", () => {
    const q = rotate({ x: 0, y: 0, z: 1 }, Math.PI / 2, 0);
    expect(q.x).toBeCloseTo(1, 10);
    expect(q.z).toBeCloseTo(0, 10);
  });
});

describe("project and nearest", () => {
  const view = { yaw: 0, pitch: 0, zoom: 1 };

  it("puts the centre in the middle of the canvas", () => {
    const [p] = project([{ x: 0, y: 0, z: 0 }], 10, view, 400, 200);
    expect(p?.x).toBe(200);
    expect(p?.y).toBe(100);
    expect(p?.scale).toBe(1);
  });

  it("draws nearer points larger and higher y as up the screen", () => {
    const [near, far, up] = project(
      [
        { x: 0, y: 0, z: 5 },
        { x: 0, y: 0, z: -5 },
        { x: 0, y: 5, z: 0 },
      ],
      10,
      view,
      400,
      400,
    );
    expect(near?.scale).toBeGreaterThan(1);
    expect(far?.scale).toBeLessThan(1);
    expect(up?.y).toBeLessThan(200);
  });

  it("zoom scales distance from the centre", () => {
    const at = (zoom: number): number =>
      project([{ x: 5, y: 0, z: 0 }], 10, { yaw: 0, pitch: 0, zoom }, 400, 400)[0]?.x ?? 0;
    expect(at(2) - 200).toBeCloseTo(2 * (at(1) - 200), 10);
  });

  it("finds the closest point within the radius, and null outside it", () => {
    const projected = project(
      [
        { x: -5, y: 0, z: 0 },
        { x: 5, y: 0, z: 0 },
      ],
      10,
      view,
      400,
      400,
    );
    const right = projected[1];
    expect(nearest(projected, (right?.x ?? 0) + 2, right?.y ?? 0, 12)).toBe(1);
    expect(nearest(projected, 0, 0, 12)).toBeNull();
    expect(nearest([], 0, 0, 12)).toBeNull();
  });
});

describe("clamp", () => {
  it("bounds a value", () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-1, 0, 3)).toBe(0);
    expect(clamp(2, 0, 3)).toBe(2);
  });
});
