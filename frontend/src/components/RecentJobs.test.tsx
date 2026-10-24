import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type * as DemoApi from "../services/demoApi";
import type { JobBody } from "../services/demoApi";
import { addHistory, loadHistory } from "../services/history";
import { DemoApiError } from "../services/problems";
import { liveStateOf } from "../services/liveState";
import { RecentJobs } from "./RecentJobs";

const answers: Record<string, () => Promise<JobBody>> = {};

vi.mock("../services/demoApi", async (importOriginal) => {
  const original = await importOriginal<typeof DemoApi>();
  return {
    ...original,
    getJob: (id: string): Promise<JobBody> =>
      (answers[id] ?? ((): Promise<JobBody> => Promise.reject(new Error("x"))))(),
  };
});

const done = (id: string): Promise<JobBody> =>
  Promise.resolve({ job_id: id, status: "completed" } as JobBody);

beforeEach(() => {
  window.localStorage.clear();
  addHistory({
    job_id: "batch-1",
    kind: "batch",
    label: "a + 1 more",
    organism_group: "eukaryote",
    submitted_at: 1,
    sequence_count: 2,
  });
  addHistory({
    job_id: "old-1",
    kind: "predict",
    label: "P1",
    organism_group: "plant",
    submitted_at: 2,
    sequence: "MKTAY",
    sequence_count: 1,
  });
  answers["batch-1"] = (): Promise<JobBody> => done("batch-1");
  answers["old-1"] = (): Promise<JobBody> =>
    Promise.reject(new DemoApiError(410, "Result expired"));
});

afterEach(() => {
  vi.restoreAllMocks();
});

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("RecentJobs", () => {
  it("shows each job with its live state from the server", async () => {
    render(<RecentJobs onOpen={vi.fn()} />);
    await flush();
    expect(screen.getByText("P1")).toBeTruthy();
    expect(screen.getByText("expired on the server")).toBeTruthy();
    expect(screen.getByText("completed")).toBeTruthy();
  });

  it("offers downloads only for a completed job, and reopening for a single sequence", async () => {
    const onOpen = vi.fn();
    render(<RecentJobs onOpen={onOpen} />);
    await flush();
    expect(screen.getAllByText("CSV")).toHaveLength(1);
    fireEvent.click(screen.getByLabelText("Open P1 again"));
    expect(onOpen).toHaveBeenCalledWith({
      id: "P1",
      sequence: "MKTAY",
      group: "plant",
      kind: "predict",
    });
    expect(screen.queryByLabelText("Open a + 1 more again")).toBeNull();
  });

  it("forgets one job and clears them all", async () => {
    render(<RecentJobs onOpen={vi.fn()} />);
    await flush();
    fireEvent.click(screen.getByLabelText("Forget P1"));
    await flush();
    expect(loadHistory().map((e) => e.job_id)).toEqual(["batch-1"]);
    fireEvent.click(screen.getByText("Clear"));
    await flush();
    expect(screen.getByText(/No jobs yet/)).toBeTruthy();
  });

  it("maps server answers to states", () => {
    expect(liveStateOf({ status: "running" })).toBe("running");
    expect(liveStateOf(new DemoApiError(404, "Unknown job_id"))).toBe("gone");
    expect(liveStateOf(new DemoApiError(410, "Result expired"))).toBe("expired");
    expect(liveStateOf(new DemoApiError(0, "Failed to fetch"))).toBe("unreachable");
  });
});
