import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  HISTORY_EVENT,
  HISTORY_KEY,
  HISTORY_LIMIT,
  addHistory,
  clearHistory,
  entryFor,
  loadHistory,
  recordJob,
  removeHistory,
  type HistoryEntry,
} from "./history";

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("entryFor", () => {
  it("records a predict request with its sequence, without whitespace", () => {
    const e = entryFor(
      "/demo/predict",
      { sequence: "MKT AY\n", organism_group: "plant", id: "P1" },
      "j1",
      5,
    );
    expect(e).toEqual({
      job_id: "j1",
      kind: "predict",
      label: "P1",
      organism_group: "plant",
      submitted_at: 5,
      sequence: "MKTAY",
      sequence_count: 1,
    });
  });

  it("records a batch by its first record and count, without the FASTA", () => {
    const e = entryFor(
      "/demo/batch",
      { fasta: ">a x\nMKT\n>b\nMK\n>c\nM", organism_group: "eukaryote" },
      "j2",
      1,
    );
    expect(e).toMatchObject({ kind: "batch", label: "a + 2 more", sequence_count: 3 });
    expect(e?.sequence).toBeUndefined();
  });

  it("ignores anything that is not a job", () => {
    expect(entryFor("/demo/catalog", {}, "x", 1)).toBeNull();
    expect(entryFor("/demo/predict", null, "x", 1)).toBeNull();
  });
});

describe("the stored list", () => {
  const entry = (id: string): HistoryEntry => ({
    job_id: id,
    kind: "predict",
    label: id,
    organism_group: "eukaryote",
    submitted_at: 1,
    sequence: "MKT",
    sequence_count: 1,
  });

  it("puts the newest first, de-duplicates and caps the length", () => {
    for (let i = 0; i < HISTORY_LIMIT + 5; i += 1) addHistory(entry(`j${String(i)}`));
    addHistory(entry("j3"));
    const list = loadHistory();
    expect(list).toHaveLength(HISTORY_LIMIT);
    expect(list[0]?.job_id).toBe("j3");
    expect(list.filter((e) => e.job_id === "j3")).toHaveLength(1);
  });

  it("announces every change", () => {
    const heard = vi.fn();
    window.addEventListener(HISTORY_EVENT, heard);
    recordJob("/demo/predict", { sequence: "MKT", organism_group: "eukaryote", id: "q" }, "j9");
    removeHistory("j9");
    clearHistory();
    window.removeEventListener(HISTORY_EVENT, heard);
    expect(heard).toHaveBeenCalledTimes(3);
    expect(loadHistory()).toEqual([]);
  });

  it("drops corrupt storage instead of crashing", () => {
    window.localStorage.setItem(HISTORY_KEY, "{not json");
    expect(loadHistory()).toEqual([]);
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify([{ job_id: 1 }, entry("ok")]));
    expect(loadHistory().map((e) => e.job_id)).toEqual(["ok"]);
  });

  it("works when storage throws (a private window)", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(loadHistory()).toEqual([]);
    expect(() => {
      addHistory(entry("x"));
    }).not.toThrow();
  });
});
