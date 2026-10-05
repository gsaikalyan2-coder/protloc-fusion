import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { makeResult } from "../test/fixtures";
import type * as DemoApi from "../services/demoApi";
import type { StructureDoc, StructureState } from "../services/demoApi";
import { StructurePanel } from "./StructurePanel";

let state: StructureState = { status: "loading" };

vi.mock("../services/demoApi", async (importOriginal) => {
  const original = await importOriginal<typeof DemoApi>();
  return { ...original, useStructure: (): StructureState => state };
});

function doc(length: number): StructureDoc {
  return {
    sequence_sha256: "sha-test",
    accession: "P12345",
    source: "AlphaFold DB prediction, not an experimental structure",
    license: "CC-BY-4.0",
    model_version: 4,
    length,
    ca: Array.from({ length }, (_, i) => [i * 3.8, 0, 0]),
    plddt: Array.from({ length }, () => 80),
    note: "AlphaFold DB prediction, shown for orientation.",
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("StructurePanel", () => {
  it("says it is loading", () => {
    state = { status: "loading" };
    render(<StructurePanel result={makeResult()} mode="attribution" cursor={0} />);
    expect(screen.getByText(/Loading the AlphaFold backbone/)).toBeTruthy();
  });

  it("states the reason when there is no structure and draws nothing", () => {
    state = { status: "missing", message: "No structure for this sequence: no AlphaFold model." };
    render(<StructurePanel result={makeResult()} mode="attribution" cursor={0} />);
    expect(screen.getByText(/No structure for this sequence/)).toBeTruthy();
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("refuses to draw a backbone whose length differs from the sequence", () => {
    const result = makeResult();
    state = { status: "ready", data: doc(result.sequence.length + 3) };
    render(<StructurePanel result={result} mode="attribution" cursor={0} />);
    expect(screen.getByText(/so nothing is drawn/)).toBeTruthy();
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("draws an accessible canvas with its provenance and legend when lengths match", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const result = makeResult();
    state = { status: "ready", data: doc(result.sequence.length) };
    render(<StructurePanel result={result} mode="attribution" cursor={0} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain(
      "Drag or use the arrow keys",
    );
    expect(screen.getByText(/not an experimental structure/)).toBeTruthy();
    expect(screen.getByText(/the model did not use the structure/)).toBeTruthy();
    expect(screen.getByText(/very high \(above 90\)/)).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("AlphaFold pLDDT 80.0");
  });
});
