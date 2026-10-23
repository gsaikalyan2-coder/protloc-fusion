import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type * as DemoApi from "../services/demoApi";
import type { CatalogDetail, CatalogEntry, CatalogFilter, CatalogPage } from "../services/demoApi";
import { DemoApiError } from "../services/problems";
import { CatalogBrowser } from "./CatalogBrowser";

const calls: CatalogFilter[] = [];
let page: CatalogPage;
let lookup: (key: string) => Promise<CatalogDetail>;

vi.mock("../services/demoApi", async (importOriginal) => {
  const original = await importOriginal<typeof DemoApi>();
  return {
    ...original,
    getCatalog: (f: CatalogFilter): Promise<CatalogPage> => {
      calls.push(f);
      return Promise.resolve(page);
    },
    getCatalogEntry: (key: string): Promise<CatalogDetail> => lookup(key),
  };
});

function entry(id: string, over: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id,
    entry_name: `${id}_HUMAN`,
    name: `${id}_HUMAN (${id})`,
    organism: "Homo sapiens (Human)",
    organism_group: "eukaryote",
    length: 120,
    partition: 1,
    in_training_set: true,
    sequence_sha256: "0".repeat(64),
    database_labels: {
      localization: ["cell_membrane"],
      signal_peptide: [],
      signal_peptide_cleavage_site: null,
    },
    ...over,
  };
}

function pageOf(entries: CatalogEntry[], total = entries.length): CatalogPage {
  return {
    total,
    offset: 0,
    limit: 25,
    entries,
    catalog: { available: true, reason: null, entries: 507, dropped_not_cached: 0 },
    locations: ["cell_membrane", "plastid"],
  };
}

async function settle(): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  calls.length = 0;
  page = pageOf([entry("P11111"), entry("Q22222", { partition: 5, in_training_set: false })]);
  lookup = (key): Promise<CatalogDetail> =>
    Promise.resolve({ ...entry(key.toUpperCase()), sequence: "MKTAYIAKQR", note: "n" });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("CatalogBrowser", () => {
  it("lists entries and says which split each one is in", async () => {
    render(<CatalogBrowser onOpen={vi.fn()} />);
    await settle();
    expect(screen.getByText("P11111")).toBeTruthy();
    expect(screen.getByText("in-sample")).toBeTruthy();
    expect(screen.getByText("held-out")).toBeTruthy();
    expect(screen.getByText("507 scorable")).toBeTruthy();
    expect(screen.getAllByText("cell membrane").length).toBeGreaterThan(0);
  });

  it("sends the filters to the server and resets to the first page", async () => {
    render(<CatalogBrowser onOpen={vi.fn()} />);
    await settle();
    fireEvent.change(screen.getByLabelText("Search accession, entry name or organism"), {
      target: { value: "arabidopsis" },
    });
    fireEvent.change(screen.getByLabelText("Partition"), { target: { value: "5" } });
    await settle();
    const last = calls.at(-1);
    expect(last).toMatchObject({ q: "arabidopsis", partition: 5, offset: 0, limit: 25 });
  });

  it("opens a row in the Inspector with its sequence and group", async () => {
    const onOpen = vi.fn();
    render(<CatalogBrowser onOpen={onOpen} />);
    await settle();
    fireEvent.click(screen.getByLabelText("Open result for P11111"));
    await settle();
    expect(onOpen).toHaveBeenCalledWith({
      id: "P11111",
      sequence: "MKTAYIAKQR",
      group: "eukaryote",
    });
  });

  it("looks up an accession typed by hand", async () => {
    const onOpen = vi.fn();
    render(<CatalogBrowser onOpen={onOpen} />);
    await settle();
    fireEvent.change(screen.getByLabelText("Accession or entry name"), {
      target: { value: "q22222" },
    });
    fireEvent.click(screen.getByText("Open result"));
    await settle();
    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ id: "Q22222" }));
  });

  it("refuses an empty lookup without calling the server", async () => {
    const spy = vi.fn(lookup);
    lookup = spy;
    render(<CatalogBrowser onOpen={vi.fn()} />);
    await settle();
    fireEvent.click(screen.getByText("Open result"));
    expect(screen.getByRole("alert").textContent).toContain("Enter an accession");
    expect(spy).not.toHaveBeenCalled();
  });

  it("explains an accession that is not cached, by its code", async () => {
    lookup = (): Promise<CatalogDetail> =>
      Promise.reject(
        new DemoApiError(404, "No cached sequence has the accession 'P99999'.", {
          code: "accession_not_cached",
          retryable: false,
        }),
      );
    const onOpen = vi.fn();
    render(<CatalogBrowser onOpen={onOpen} />);
    await settle();
    fireEvent.change(screen.getByLabelText("Accession or entry name"), {
      target: { value: "P99999" },
    });
    fireEvent.click(screen.getByText("Open result"));
    await settle();
    expect(screen.getByRole("alert").textContent).toContain(
      "No cached sequence has that accession",
    );
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("pages when there are more matches than fit", async () => {
    page = pageOf([entry("P11111")], 60);
    render(<CatalogBrowser onOpen={vi.fn()} />);
    await settle();
    expect(screen.getByText("1–25 of 60")).toBeTruthy();
    fireEvent.click(screen.getByText("Next"));
    await settle();
    expect(calls.at(-1)?.offset).toBe(25);
  });

  it("says why when the catalog is not loaded", async () => {
    page = {
      ...pageOf([]),
      catalog: { available: false, reason: "no catalog", entries: 0, dropped_not_cached: 0 },
    };
    render(<CatalogBrowser onOpen={vi.fn()} />);
    await settle();
    expect(screen.getByText(/The catalog is not loaded: no catalog/)).toBeTruthy();
  });
});
