import { describe, expect, it } from "vitest";
import type { JobBody } from "./demoApi";
import {
  DemoApiError,
  describeError,
  detailText,
  jobProblem,
  problemOf,
  toProblem,
} from "./problems";

describe("toProblem", () => {
  it("410 is a legible expiry, never an empty result", () => {
    const p = toProblem(410, "Result expired");
    expect(p.kind).toBe("expired");
    expect(p.title).toBe("This result has expired");
    expect(p.message).toContain("never shown as an empty result");
    expect(p.hint).toContain("Run the prediction again");
  });

  it("404 says the job is unknown and why that can happen", () => {
    const p = toProblem(404, "Unknown job_id");
    expect(p.kind).toBe("unknown_job");
    expect(p.message).toContain("server restarted");
  });

  it("422 for an uncached sequence points at the gallery and keeps the server's words", () => {
    const text =
      "This build scores only sequences whose ESM-2 states are cached (500 of them, ADR-0016). Pick a gallery example.";
    const p = toProblem(422, text);
    expect(p.kind).toBe("not_cached");
    expect(p.message).toBe(text);
    expect(p.hint).toContain("Gallery");
  });

  it("422 for an unserved group says which groups are served", () => {
    const p = toProblem(
      422,
      "No trained head covers 'gram_neg': the training corpus is eukaryotic (served groups: eukaryote, plant).",
    );
    expect(p.kind).toBe("group_not_served");
    expect(p.message).toContain("served groups: eukaryote, plant");
    expect(p.hint).toContain("Plant");
  });

  it("422 validation lists each field message instead of [object Object]", () => {
    const p = toProblem(422, [
      { loc: ["body", "sequence"], msg: "String should have at least 30 characters", type: "x" },
    ]);
    expect(p.kind).toBe("invalid_input");
    expect(p.message).toBe("sequence: String should have at least 30 characters");
  });

  it("400 is an input problem", () => {
    expect(toProblem(400, "Unknown organism_group 'x'").kind).toBe("invalid_input");
  });

  it("status 0 is an unreachable backend with the start command", () => {
    const p = toProblem(0, "Failed to fetch");
    expect(p.kind).toBe("unreachable");
    expect(p.hint).toContain("uvicorn protloc.api.app_v2:app --port 8010");
  });

  it("5xx is a server error that asks for the model version and job id", () => {
    const p = toProblem(500, "");
    expect(p.kind).toBe("server_error");
    expect(p.message).toContain("500");
    expect(p.hint).toContain("job id");
  });
});

describe("detailText", () => {
  it("handles strings, lists and junk", () => {
    expect(detailText("a")).toBe("a");
    expect(detailText([{ msg: "bad" }, 3, null])).toBe("bad");
    expect(detailText({ nope: 1 })).toBe("");
    expect(detailText(undefined)).toBe("");
  });
});

const job = (over: Partial<JobBody>): JobBody =>
  ({
    job_id: "j",
    status: "completed",
    total: 1,
    completed_count: 1,
    created_at: 0,
    started_at: 0,
    finished_at: 1,
    error: null,
    model_version: "v",
    config_hash: "c",
    rules_version: "r",
    experiment_id: "e",
    predictor: "p",
    ...over,
  }) satisfies JobBody;

describe("jobProblem", () => {
  it("is null for jobs that are fine or still running", () => {
    expect(jobProblem(job({ status: "completed" }))).toBeNull();
    expect(jobProblem(job({ status: "running" }))).toBeNull();
    expect(jobProblem(job({ status: "cancelled" }))).toBeNull();
  });

  it("states a failure and never an empty message", () => {
    const p = jobProblem(job({ status: "failed", error: null }));
    expect(p?.kind).toBe("job_failed");
    expect(p?.message.length).toBeGreaterThan(10);
  });

  it("recognises a refused sequence inside a failed job", () => {
    const p = jobProblem(
      job({ status: "failed", error: "only sequences whose ESM-2 states are cached" }),
    );
    expect(p?.kind).toBe("not_cached");
  });
});

describe("problemOf and describeError", () => {
  it("unwraps a DemoApiError", () => {
    expect(problemOf(new DemoApiError(410, "x")).kind).toBe("expired");
  });

  it("never prints a stack trace or [object Object]", () => {
    const text = describeError(new TypeError("boom"));
    expect(text).toBe("Something went wrong. boom");
    expect(describeError({ weird: true })).not.toContain("Error:");
    expect(describeError(new DemoApiError(410, "x"))).toContain("This result has expired.");
  });
});
