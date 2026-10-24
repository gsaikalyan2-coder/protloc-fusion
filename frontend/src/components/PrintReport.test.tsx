import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import { sequenceLines, topResidues } from "../services/report";
import { PrintReport } from "./PrintReport";

const PRINTED = new Date("2026-10-24T09:30:00Z");

describe("PrintReport", () => {
  it("carries the result, its engines and its provenance", () => {
    render(
      <PrintReport result={makeResult()} jobId="job-123" inSample={false} printedAt={PRINTED} />,
    );
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "ProtLoc Fusion report · TEST1",
    );
    expect(screen.getByText(/printed 2026-10-24 09:30 UTC/)).toBeTruthy();
    expect(screen.getByText(/Cytoplasm \(80\.0%\)/)).toBeTruthy();
    expect(screen.getByText(/signal peptide, cleavage after residue 5/)).toBeTruthy();
    expect(screen.getByText(/GO:0003824 catalytic activity/)).toBeTruthy();
    expect(screen.getAllByText(/motif, not calibrated/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Model 1\.0\.0-test · config cfg-test/)).toBeTruthy();
    expect(screen.getByText(/job job-123/)).toBeTruthy();
    expect(screen.getByText(/not for clinical or diagnostic use/)).toBeTruthy();
  });

  it("states in-sample results, missing evidence and abstention", () => {
    const r = makeResult({
      no_confident_call: true,
      evidence_status: {
        ...makeResult().evidence_status,
        available: false,
        unavailable_reason: "timed out",
      },
    });
    render(<PrintReport result={r} jobId="j" inSample printedAt={PRINTED} />);
    expect(screen.getByText(/In-sample sequence/)).toBeTruthy();
    expect(screen.getByText(/Homology evidence unavailable \(timed out\)/)).toBeTruthy();
    expect(screen.getByText(/no confident call/)).toBeTruthy();
    expect(screen.getByText("Not searched.")).toBeTruthy();
  });

  it("lays the sequence out in numbered blocks of ten", () => {
    const lines = sequenceLines("A".repeat(65));
    expect(lines).toHaveLength(2);
    expect(lines[0]).toBe(`    1  ${Array.from({ length: 6 }, () => "A".repeat(10)).join(" ")}`);
    expect(lines[1]).toBe("   61  AAAAA");
  });

  it("lists the residues with the largest positive attribution", () => {
    const top = topResidues(makeResult(), 3);
    expect(top.map((x) => `${x.residue}${String(x.position)}`)).toEqual(["T3", "R10", "A7"]);
    expect(top.every((x) => x.score > 0)).toBe(true);
  });
});
