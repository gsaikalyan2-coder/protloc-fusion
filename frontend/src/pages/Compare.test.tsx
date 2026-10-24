import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type * as DemoApi from "../services/demoApi";
import Compare from "./Compare";

const examples = [
  { id: "P1", name: "ONE_HUMAN (P1)", organism_group: "eukaryote", sequence: "MKTAY", note: "n" },
  { id: "P2", name: "TWO_ARATH (P2)", organism_group: "plant", sequence: "MKKLL", note: "n" },
];

vi.mock("../services/demoApi", async (importOriginal) => {
  const original = await importOriginal<typeof DemoApi>();
  return {
    ...original,
    getExamples: (): Promise<{ examples: typeof examples }> => Promise.resolve({ examples }),
    getVersion: (): Promise<Record<string, unknown>> =>
      Promise.resolve({
        organism_groups: { eukaryote: { served: true }, plant: { served: true } },
      }),
  };
});

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("Compare page", () => {
  it("enables Compare only once both sides have a sequence", async () => {
    render(<Compare />);
    await flush();
    const button = screen.getByRole("button", { name: "Compare" });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    const pickers = screen
      .getAllByRole("combobox")
      .filter((el) => el.textContent.includes("Load an example"));
    const [first, second] = pickers;
    if (!first || !second) throw new Error("expected two example pickers");
    fireEvent.change(first, { target: { value: "P1" } });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(second, { target: { value: "P2" } });
    expect((button as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByText("P1")).toBeTruthy();
    expect(screen.getAllByText("5 residues", { exact: false })).toHaveLength(2);
  });

  it("refuses an empty accession without calling the server", async () => {
    render(<Compare />);
    await flush();
    const [load] = screen.getAllByRole("button", { name: "Load" });
    if (!load) throw new Error("expected a Load button");
    fireEvent.click(load);
    expect(screen.getByRole("alert").textContent).toContain("Enter an accession");
  });
});
