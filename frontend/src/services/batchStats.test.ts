import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import type { LocRow, Result } from "./demoApi";
import { batchStats } from "./batchStats";

function loc(code: string, probability: number, predicted: boolean): LocRow {
  return {
    code,
    name: code === "CYT" ? "Cytoplasm" : code === "NUC" ? "Nucleus" : code,
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

function result(over: Partial<Result>): Result {
  return makeResult(over);
}

describe("batchStats", () => {
  const base = makeResult();
  const rs = [
    result({ id: "a", localization: [loc("CYT", 0.9, true), loc("NUC", 0.7, true)] }),
    result({ id: "b", localization: [loc("CYT", 0.55, true)], result_cache: "hit" }),
    result({
      id: "c",
      no_confident_call: true,
      localization: [loc("CYT", 0.1, false)],
      sorting_signal: {
        type: "none",
        probability: 1,
        region: null,
        cleavage_site: null,
        candidates: {},
      },
      go_function: [],
      evidence_status: { ...base.evidence_status, available: false, unavailable_reason: "down" },
    }),
  ];

  it("counts each predicted location, a protein under several when dual-targeted", () => {
    const s = batchStats(rs, 2);
    expect(s.locations).toEqual([
      { key: "CYT", label: "Cytoplasm", count: 2 },
      { key: "NUC", label: "Nucleus", count: 1 },
    ]);
    expect(s.refused).toBe(2);
    expect(s.scored).toBe(3);
    expect(s.noConfidentCall).toBe(1);
  });

  it("bins the top calibrated probability", () => {
    const counts = batchStats(rs).confidence.map((b) => b.count);
    expect(counts).toEqual([1, 0, 1, 0, 1]);
  });

  it("counts signals, GO terms, evidence states and cache hits", () => {
    const s = batchStats(rs);
    expect(s.signal).toEqual([
      { key: "signal_peptide", label: "signal peptide", count: 2 },
      { key: "none", label: "no signal", count: 1 },
    ]);
    expect(s.goTerms).toEqual([{ key: "GO:0003824", label: "catalytic activity", count: 2 }]);
    expect(Object.fromEntries(s.evidence.map((b) => [b.key, b.count]))).toEqual({
      no_hit: 2,
      unavailable: 1,
    });
    expect(s.cacheHits).toBe(1);
    expect(s.liveEncoded).toBe(0);
    expect(batchStats([makeResult({ outside_evaluated_subset: true })]).liveEncoded).toBe(1);
  });

  it("caps the GO list and handles an empty batch", () => {
    expect(batchStats(rs, 0, 0).goTerms).toEqual([]);
    const empty = batchStats([]);
    expect(empty.scored).toBe(0);
    expect(empty.locations).toEqual([]);
    expect(empty.confidence.every((b) => b.count === 0)).toBe(true);
  });
});
