import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi, type Mock } from "vitest";
import type * as DemoApi from "../services/demoApi";
import type { CatalogDetail } from "../services/demoApi";
import { DemoApiError } from "../services/problems";
import { CommandPalette } from "./CommandPalette";
import { paletteOptions } from "./nav";

let lookup: (key: string) => Promise<CatalogDetail>;

vi.mock("../services/demoApi", async (importOriginal) => {
  const original = await importOriginal<typeof DemoApi>();
  return {
    ...original,
    getCatalogEntry: (key: string): Promise<CatalogDetail> => lookup(key),
  };
});

function detail(id: string): CatalogDetail {
  return {
    id,
    entry_name: `${id}_HUMAN`,
    name: `${id}_HUMAN (${id})`,
    organism: "Homo sapiens (Human)",
    organism_group: "eukaryote",
    length: 4,
    partition: 1,
    in_training_set: true,
    sequence_sha256: "0".repeat(64),
    database_labels: {
      localization: [],
      signal_peptide: [],
      signal_peptide_cleavage_site: null,
    },
    sequence: "MKLV",
    note: "",
  };
}

function setup(): {
  onNavigate: Mock;
  onOpenEntry: Mock;
  onOpenChange: Mock;
  input: HTMLElement;
} {
  const onNavigate = vi.fn();
  const onOpenEntry = vi.fn();
  const onOpenChange = vi.fn();
  render(
    <CommandPalette
      open
      onOpenChange={onOpenChange}
      onNavigate={onNavigate}
      onOpenEntry={onOpenEntry}
    />,
  );
  return { onNavigate, onOpenEntry, onOpenChange, input: screen.getByRole("combobox") };
}

describe("paletteOptions", () => {
  it("lists every screen for an empty query", () => {
    expect(paletteOptions("").filter((o) => o.kind === "screen")).toHaveLength(10);
  });

  it("offers a lookup first for an accession-like query", () => {
    expect(paletteOptions("P12345")[0]).toMatchObject({ kind: "lookup", query: "P12345" });
  });

  it("offers a lookup for an entry name", () => {
    expect(paletteOptions("INS_HUMAN")[0]).toMatchObject({ kind: "lookup" });
  });

  it("offers no lookup for a word or a phrase", () => {
    expect(paletteOptions("calibration").some((o) => o.kind === "lookup")).toBe(false);
    expect(paletteOptions("mutation scanner").some((o) => o.kind === "lookup")).toBe(false);
  });
});

describe("CommandPalette", () => {
  it("jumps to the screen chosen with the keyboard", () => {
    const { onNavigate, onOpenChange, input } = setup();
    fireEvent.change(input, { target: { value: "calibration" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onNavigate).toHaveBeenCalledWith("trust");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("moves the active option with the arrow keys", () => {
    const { input } = setup();
    fireEvent.keyDown(input, { key: "ArrowDown" });
    const active = screen
      .getAllByRole("option")
      .find((o) => o.getAttribute("aria-selected") === "true");
    expect(input.getAttribute("aria-activedescendant")).toBe(active?.id);
    expect(active?.textContent).toContain("Predict");
  });

  it("opens a looked-up accession in Predict", async () => {
    lookup = (key): Promise<CatalogDetail> => Promise.resolve(detail(key));
    const { onOpenEntry, input } = setup();
    fireEvent.change(input, { target: { value: "P12345" } });
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter" });
      await Promise.resolve();
    });
    expect(onOpenEntry).toHaveBeenCalledWith({
      id: "P12345",
      sequence: "MKLV",
      group: "eukaryote",
    });
  });

  it("shows the backend's reason when a lookup misses", async () => {
    lookup = (): Promise<CatalogDetail> =>
      Promise.reject(
        new DemoApiError(404, "not cached", {
          code: "accession_not_cached",
          message: "Q00000 is not one of the cached sequences.",
          retryable: false,
        }),
      );
    const { onOpenEntry, input } = setup();
    fireEvent.change(input, { target: { value: "Q00000" } });
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter" });
      await Promise.resolve();
    });
    expect(onOpenEntry).not.toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).not.toBe("");
  });
});
