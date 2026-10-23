import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJob } from "./demoApi";

/* A fake backend: POST answers 202 with a job id; GET answers the job's status. Every poll is
   recorded, so a poller that outlives its job (or its screen) shows up as extra GETs. */
const polls: string[] = [];
const status = new Map<string, string>();
let nextId = 0;
let releasePost: (() => void) | null = null;
let holdPosts = false;

function respond(body: unknown, code = 200): Response {
  return new Response(JSON.stringify(body), {
    status: code,
    headers: { "content-type": "application/json" },
  });
}

async function fakeFetch(input: string, init?: RequestInit): Promise<Response> {
  const path = input;
  if (init?.method === "POST") {
    if (holdPosts) {
      await new Promise<void>((resolve) => {
        releasePost = resolve;
      });
    }
    nextId += 1;
    const id = `job-${String(nextId)}`;
    status.set(id, "running");
    return respond({ job_id: id, result_url: `/demo/jobs/${id}`, status: "queued" }, 202);
  }
  const id = path.split("/").at(-1) ?? "";
  polls.push(id);
  return respond({ job_id: id, status: status.get(id) ?? "running" });
}

async function tick(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("fetch", vi.fn(fakeFetch));
  polls.length = 0;
  status.clear();
  nextId = 0;
  releasePost = null;
  holdPosts = false;
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useJob polling", () => {
  it("stops polling once the job completes", async () => {
    const { result } = renderHook(() => useJob());
    await act(async () => {
      await result.current.run("/demo/predict", {});
    });
    await tick(1000);
    status.set("job-1", "completed");
    await tick(1000);
    const seen = polls.length;
    await tick(5000);
    expect(polls.length).toBe(seen);
    expect(result.current.busy).toBe(false);
  });

  it("never polls after the screen unmounts while the POST is in flight", async () => {
    holdPosts = true;
    const { result, unmount } = renderHook(() => useJob());
    let pending: Promise<void> = Promise.resolve();
    act(() => {
      pending = result.current.run("/demo/predict", {});
    });
    unmount();
    await act(async () => {
      releasePost?.();
      await pending;
    });
    await tick(5000);
    expect(polls).toEqual([]);
  });

  it("keeps only the newest of two overlapping runs, and stops it when done", async () => {
    holdPosts = true;
    const { result } = renderHook(() => useJob());
    let first: Promise<void> = Promise.resolve();
    let second: Promise<void> = Promise.resolve();
    act(() => {
      first = result.current.run("/demo/predict", { n: 1 });
    });
    const releaseFirst = releasePost;
    act(() => {
      second = result.current.run("/demo/predict", { n: 2 });
    });
    const releaseSecond = releasePost;
    await act(async () => {
      releaseFirst?.();
      releaseSecond?.();
      await Promise.all([first, second]);
    });
    await tick(1000);
    status.set("job-1", "completed");
    status.set("job-2", "completed");
    await tick(1000);
    const seen = polls.length;
    await tick(5000);
    expect(polls.length).toBe(seen);
    expect(new Set(polls.filter((id) => id === "job-1")).size).toBeLessThanOrEqual(1);
  });
});
