import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { SpaceDoc, SpacePoint } from "../services/demoApi";
import { Neighbours, ProteinSpace } from "./ProteinSpace";

function point(i: number, locations: string[], neighbours: [number, number][]): SpacePoint {
  return {
    sha256: String(i).repeat(64).slice(0, 64),
    ids: [`P0000${String(i)}`],
    entry_name: `E${String(i)}_HUMAN`,
    organism: "Homo sapiens",
    organism_group: "eukaryote",
    length: 100,
    partition: 1,
    in_training_set: true,
    locations,
    xyz: [i / 10, -i / 10, 0],
    neighbours,
  };
}

const SPACE: SpaceDoc = {
  method: "pca",
  components: 3,
  explained_variance_ratio: [0.24, 0.12, 0.07],
  neighbour_metric: "cosine of centred vectors, 640 dimensions",
  embedding: { checkpoint: "esm2", revision: "a695f60", pooling: "mean", cache_key: "k" },
  scope_label: "subset (500)",
  note: "note",
  points: [
    point(
      1,
      ["EXT"],
      [
        [1, 0.8],
        [2, 0.5],
      ],
    ),
    point(
      2,
      ["EXT", "MEM"],
      [
        [0, 0.8],
        [2, 0.6],
      ],
    ),
    point(
      3,
      ["NUC"],
      [
        [1, 0.6],
        [0, 0.5],
      ],
    ),
  ],
};

describe("ProteinSpace", () => {
  it("draws an accessible canvas and a legend with counts in the design order", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    render(<ProteinSpace space={SPACE} focusSha={SPACE.points[0]?.sha256} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("3 cached sequences");
    const chips = screen.getAllByRole("button", { pressed: false }).map((b) => b.textContent);
    expect(chips.slice(0, 3)).toEqual(["NUC1", "EXT2", "MEM1"]);
    expect(screen.getByText(/PC1 24.0%/)).toBeTruthy();
  });

  it("filters by compartment and offers to show all again", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    render(<ProteinSpace space={SPACE} />);
    fireEvent.click(screen.getByRole("button", { name: /EXT/ }));
    expect(screen.getByRole("button", { name: /EXT/ }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Show all" }));
    expect(screen.queryByRole("button", { name: "Show all" })).toBeNull();
  });
});

describe("Neighbours", () => {
  it("lists the nearest cached sequences and opens one", () => {
    const onOpen = vi.fn();
    render(<Neighbours space={SPACE} focusSha={SPACE.points[0]?.sha256 ?? ""} onOpen={onOpen} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]?.textContent).toContain("P00002");
    expect(items[0]?.textContent).toContain("0.80");
    fireEvent.click(screen.getByRole("button", { name: /P00002/ }));
    expect(onOpen).toHaveBeenCalledWith("P00002");
  });

  it("says when the sequence is not on the map", () => {
    render(<Neighbours space={SPACE} focusSha={"f".repeat(64)} />);
    expect(screen.getByText(/no place on the map/)).toBeTruthy();
  });
});
