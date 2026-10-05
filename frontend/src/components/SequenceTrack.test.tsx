import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import { SequenceTrack, describeResidue } from "./SequenceTrack";

function active(listbox: HTMLElement): HTMLElement {
  const id = listbox.getAttribute("aria-activedescendant");
  expect(id).not.toBeNull();
  const el = document.getElementById(id ?? "");
  expect(el).not.toBeNull();
  if (!el) throw new Error("aria-activedescendant element not found");
  return el;
}

describe("SequenceTrack keyboard operation", () => {
  it("is one focusable listbox with a named option per residue", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const listbox = screen.getByRole("listbox");
    expect(listbox.tabIndex).toBe(0);
    expect(listbox.getAttribute("aria-label")).toContain("TEST1");
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(12);
    for (const o of options) expect(o.getAttribute("aria-label")).toMatch(/^Residue \d+, [A-Z]/);
  });

  it("starts on the first residue and moves with the arrow keys", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const listbox = screen.getByRole("listbox");
    expect(active(listbox).textContent).toBe("M");
    fireEvent.keyDown(listbox, { key: "ArrowRight" });
    expect(active(listbox).textContent).toBe("K");
    expect(active(listbox).getAttribute("aria-selected")).toBe("true");
    fireEvent.keyDown(listbox, { key: "ArrowLeft" });
    expect(active(listbox).textContent).toBe("M");
  });

  it("stays inside the sequence at both ends", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const listbox = screen.getByRole("listbox");
    fireEvent.keyDown(listbox, { key: "ArrowLeft" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 1,");
    fireEvent.keyDown(listbox, { key: "End" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 12,");
    fireEvent.keyDown(listbox, { key: "ArrowRight" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 12,");
    fireEvent.keyDown(listbox, { key: "Home" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 1,");
  });

  it("moves ten with ArrowDown and clamps", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const listbox = screen.getByRole("listbox");
    fireEvent.keyDown(listbox, { key: "ArrowDown" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 11,");
    fireEvent.keyDown(listbox, { key: "ArrowDown" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 12,");
    fireEvent.keyDown(listbox, { key: "ArrowUp" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 2,");
  });

  it("leaves browser shortcuts alone", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const listbox = screen.getByRole("listbox");
    fireEvent.keyDown(listbox, { key: "ArrowRight", ctrlKey: true });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 1,");
  });

  it("moves the cursor on click and shows the same description as text", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const options = screen.getAllByRole("option");
    const opt5 = options[5];
    if (!opt5) throw new Error("expected at least 6 options");
    fireEvent.click(opt5);
    const listbox = screen.getByRole("listbox");
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 6,");
    expect(screen.getByText(describeResidue(makeResult(), "attribution", 5))).toBeTruthy();
  });
});

describe("describeResidue announces what the viewer shows", () => {
  const r = makeResult();

  it("says whether a signed score supports or argues against the label", () => {
    expect(describeResidue(r, "attribution", 2)).toContain("supports the label");
    expect(describeResidue(r, "attribution", 5)).toContain("argues against the label");
    expect(describeResidue(r, "attribution", 3)).not.toContain("label");
  });

  it("names the signal region, cleavage site and transmembrane segment", () => {
    expect(describeResidue(r, "attribution", 1)).toContain("signal peptide region");
    expect(describeResidue(r, "attribution", 4)).toContain("last residue before the cleavage site");
    expect(describeResidue(r, "attribution", 9)).toContain("transmembrane");
  });

  it("names the method and the track for each mode", () => {
    expect(describeResidue(r, "attribution", 0)).toContain("occlusion");
    expect(describeResidue(r, "attention", 0)).toContain("pooling_attention");
    expect(describeResidue(r, "signal", 0)).toContain("crf_marginal");
    expect(describeResidue(r, "hydropathy", 0)).toContain("hydropathy 1.80");
    expect(describeResidue(r, "alignment", 0, [[0, 3]])).toContain("inside an aligned block");
    expect(describeResidue(r, "alignment", 6, [[0, 3]])).toContain("outside aligned blocks");
  });
});
