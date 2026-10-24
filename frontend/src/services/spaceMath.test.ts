import { describe, expect, it } from "vitest";
import type { SpacePoint } from "./demoApi";
import { EDGE, normalised } from "./spaceMath";

const at = (xyz: [number, number, number]): SpacePoint => ({
  sha256: "",
  ids: [],
  entry_name: "",
  organism: null,
  organism_group: "eukaryote",
  length: 1,
  partition: 1,
  in_training_set: true,
  locations: [],
  xyz,
  neighbours: [],
});

describe("normalised", () => {
  it("scales each axis to its own spread", () => {
    const points = Array.from({ length: 101 }, (_, i) => at([i / 100, i / 1000, -i / 10]));
    const out = normalised(points);
    expect(out[97]?.[0]).toBeCloseTo(1);
    expect(out[97]?.[1]).toBeCloseTo(1);
    expect(out[97]?.[2]).toBeCloseTo(-1);
  });

  it("pins outliers to the edge of the box", () => {
    const points = [...Array.from({ length: 99 }, () => at([0.1, 0.1, 0.1])), at([50, 0.1, 0.1])];
    expect(normalised(points)[99]?.[0]).toBe(EDGE);
  });

  it("survives an axis with no spread", () => {
    expect(normalised([at([0, 0, 0])])[0]).toEqual([0, 0, 0]);
  });
});
