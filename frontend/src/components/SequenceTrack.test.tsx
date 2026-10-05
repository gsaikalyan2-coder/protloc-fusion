import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import { BLOCK, SequenceTrack, blockStarts, describeResidue, legendItems } from "./SequenceTrack";

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

describe("SequenceTrack layout", () => {
  it("groups residues in blocks of ten, each labelled with its first residue number", () => {
    const { container } = render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const labels = [...container.querySelectorAll(".seq-pos")].map((e) => e.textContent);
    expect(labels).toEqual(["1", "11"]);
    expect(container.querySelectorAll(".seq-block")).toHaveLength(2);
    expect(screen.getAllByRole("option")).toHaveLength(12);
  });

  it("keeps the ruler out of the accessibility tree", () => {
    const { container } = render(<SequenceTrack result={makeResult()} mode="attribution" />);
    for (const e of container.querySelectorAll(".seq-pos")) {
      expect(e.getAttribute("aria-hidden")).toBe("true");
    }
    expect(
      screen
        .getAllByRole("option")
        .every((o) => /^Residue \d+,/.test(o.getAttribute("aria-label") ?? "")),
    ).toBe(true);
  });

  it("blockStarts covers every residue exactly once", () => {
    expect(blockStarts(0)).toEqual([]);
    expect(blockStarts(BLOCK)).toEqual([0]);
    expect(blockStarts(BLOCK + 1)).toEqual([0, BLOCK]);
    expect(blockStarts(25)).toEqual([0, 10, 20]);
  });

  it("keyboard navigation crosses block boundaries", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const listbox = screen.getByRole("listbox");
    for (let n = 0; n < 10; n += 1) fireEvent.keyDown(listbox, { key: "ArrowRight" });
    expect(active(listbox).getAttribute("aria-label")).toContain("Residue 11,");
  });

  it("tells the 3D view where the cursor is", () => {
    const seen: number[] = [];
    render(
      <SequenceTrack result={makeResult()} mode="attribution" onCursor={(i) => seen.push(i)} />,
    );
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "ArrowDown" });
    expect(seen.at(-1)).toBe(10);
  });
});

describe("SequenceTrack legend", () => {
  it("names both sides of a signed track and the fixed marks", () => {
    render(<SequenceTrack result={makeResult()} mode="attribution" />);
    const legend = screen.getByRole("list", { name: "Legend" });
    expect(legend.textContent).toContain("supports the label");
    expect(legend.textContent).toContain("argues against it");
    expect(legend.textContent).toContain("signal-peptide region");
    expect(legend.textContent).toContain("transmembrane segment");
  });

  it("shows only what the mode can show", () => {
    const r = makeResult();
    const kinds = (m: Parameters<typeof legendItems>[0]): string[] =>
      legendItems(m, null, null).map((i) => i.swatch);
    expect(kinds("hydropathy")).toContain("hydrophobic");
    expect(kinds("hydropathy")).not.toContain("support");
    expect(kinds("alignment")).toContain("aligned");
    expect(
      legendItems("signal", { values: r.attribution, signed: false, method: "m" }, 5)[0]?.swatch,
    ).toBe("shade");
  });

  it("says when there is no cleavage site", () => {
    expect(legendItems("attribution", null, null).some((i) => i.text.includes("(none)"))).toBe(
      true,
    );
  });
});

describe("SequenceTrack themes and tokens", () => {
  for (const theme of ["light", "dark"]) {
    it(`renders in the ${theme} theme with token colours only`, () => {
      document.documentElement.setAttribute("data-theme", theme);
      const { container } = render(<SequenceTrack result={makeResult()} mode="attribution" />);
      expect(screen.getAllByRole("option")).toHaveLength(12);
      expect(container.innerHTML).not.toMatch(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/);
      document.documentElement.removeAttribute("data-theme");
    });
  }
});
