import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import { SourceSummary } from "./SourceSummary";

function rowFor(label: string): HTMLElement {
  const cell = screen.getByText(label);
  return cell.closest("tr") as HTMLElement;
}

describe("SourceSummary", () => {
  it("names the engine behind every output, motif included", () => {
    render(<SourceSummary result={makeResult()} />);
    expect(rowFor("Membrane type").textContent).toContain("motif");
    expect(rowFor("Membrane type").textContent).toContain("not calibrated");
    expect(rowFor("Subcellular location").textContent).toContain("model");
    expect(rowFor("Subcellular location").textContent).toContain("calibrated (0.10.0)");
    expect(rowFor("Hydropathy track").textContent).toContain("motif");
    expect(rowFor("Per-residue attribution").textContent).toContain("method occlusion");
    expect(rowFor("Homology evidence").textContent).toContain("never feeds a score");
  });

  it("says so when the evidence index is unavailable", () => {
    const r = makeResult();
    render(
      <SourceSummary
        result={{
          ...r,
          evidence_status: {
            ...r.evidence_status,
            available: false,
            unavailable_reason: "index not loaded",
          },
        }}
      />,
    );
    expect(rowFor("Homology evidence").textContent).toContain("unavailable: index not loaded");
  });
});
