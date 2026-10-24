import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import type { LocRow } from "./demoApi";
import { compareResults } from "./compare";

function loc(code: string, probability: number, predicted: boolean): LocRow {
  return {
    code,
    name: code.toLowerCase(),
    class_name: code.toLowerCase(),
    probability,
    raw: probability,
    predicted,
    source: "model",
    threshold: 0.5,
    calibrated: true,
    calibration_method: "isotonic",
    confidence_band: "high",
  };
}

describe("compareResults", () => {
  const a = makeResult({
    id: "A",
    localization: [loc("CYT", 0.8, true), loc("NUC", 0.3, false)],
  });
  const b = makeResult({
    id: "B",
    sequence_sha256: "other",
    length: 300,
    localization: [loc("CYT", 0.2, false), loc("NUC", 0.7, true), loc("MIT", 0.05, false)],
    sorting_signal: {
      type: "none",
      probability: 1,
      region: null,
      cleavage_site: null,
      candidates: {},
    },
    go_function: [],
  });

  it("lines the location labels up with the difference B - A", () => {
    const c = compareResults(a, b);
    const cyt = c.locations.find((l) => l.code === "CYT");
    expect(cyt).toMatchObject({ a: 0.8, b: 0.2, delta: -0.6, aPredicted: true, bPredicted: false });
    const mit = c.locations.find((l) => l.code === "MIT");
    expect(mit).toMatchObject({ a: null, b: 0.05, delta: null });
    expect(c.locations.map((l) => l.code)).toEqual(["CYT", "NUC", "MIT"]);
  });

  it("marks the summary rows that differ", () => {
    const c = compareResults(a, b);
    const byLabel = Object.fromEntries(c.summary.map((r) => [r.label, r]));
    expect(byLabel["Predicted location"]).toMatchObject({ a: "CYT", b: "NUC", differs: true });
    expect(byLabel["Sorting signal"]).toMatchObject({
      a: "signal_peptide (cleaves after 5)",
      b: "none",
      differs: true,
    });
    expect(byLabel["Organism group"]?.differs).toBe(false);
    expect(byLabel.Length).toMatchObject({ a: "12", b: "300" });
  });

  it("splits the predicted GO terms into shared and one-sided", () => {
    const c = compareResults(a, b);
    expect(c.go.onlyA.map((t) => t.go_id)).toEqual(["GO:0003824"]);
    expect(c.go.onlyB).toEqual([]);
    expect(c.go.shared).toEqual([]);
    expect(compareResults(a, a).go.shared.map((t) => t.name)).toEqual(["catalytic activity"]);
  });

  it("says when both sides are the same sequence", () => {
    expect(compareResults(a, a).sameSequence).toBe(true);
    expect(compareResults(a, b).sameSequence).toBe(false);
  });

  it("names a no-confident-call result and unavailable evidence plainly", () => {
    const quiet = makeResult({
      no_confident_call: true,
      evidence_status: { ...a.evidence_status, available: false, unavailable_reason: "down" },
    });
    const byLabel = Object.fromEntries(compareResults(quiet, a).summary.map((r) => [r.label, r]));
    expect(byLabel["Predicted location"]?.a).toBe("no confident call");
    expect(byLabel["Nearest homolog"]?.a).toBe("evidence unavailable");
    expect(byLabel["Nearest homolog"]?.b).toBe("no hit");
  });
});
