import { describe, expect, it } from "vitest";
import type { JobBody } from "./demoApi";
import {
  CODE_TEXT,
  DemoApiError,
  degradedProblem,
  failureProblems,
  jobProblem,
  toProblem,
} from "./problems";

describe("oversized input", () => {
  it("sequence_too_long keeps the server's numbers and cannot be retried", () => {
    const detail = "The sequence has 5001 residues; the limit is 5000.";
    const p = toProblem(422, detail, { code: "sequence_too_long", retryable: false });
    expect(p.kind).toBe("too_large");
    expect(p.code).toBe("sequence_too_long");
    expect(p.message).toBe(detail);
    expect(p.retryable).toBe(false);
  });

  it("request_too_large and batch_too_large point at chunking", () => {
    for (const code of ["request_too_large", "batch_too_large"]) {
      const p = toProblem(code === "request_too_large" ? 413 : 422, "x", { code });
      expect(p.kind).toBe("too_large");
      expect(p.hint).toContain("smaller pieces");
    }
  });
});

describe("malformed FASTA", () => {
  it("keeps the line number the server put in the detail", () => {
    const p = toProblem(400, "line 3: '1' is not a letter. Residues are letters only.", {
      code: "fasta_invalid_character",
      retryable: false,
      line: 3,
    });
    expect(p.kind).toBe("fasta_error");
    expect(p.message).toContain("line 3");
    expect(p.hint).toContain("Fix the line named above");
    expect(p.retryable).toBe(false);
  });

  it("every fasta_* code is a fasta_error", () => {
    for (const code of Object.keys(CODE_TEXT).filter((c) => c.startsWith("fasta_"))) {
      expect(CODE_TEXT[code]?.kind).toBe("fasta_error");
    }
  });
});

describe("evidence service down", () => {
  it("is a flag, not a failure: scores unchanged, no retry", () => {
    const p = degradedProblem(["the evidence search ran out of time"]);
    expect(p.kind).toBe("evidence_degraded");
    expect(p.message).toContain("ran out of time");
    expect(p.message).toContain("scores are unchanged");
    expect(p.retryable).toBe(false);
  });

  it("still reads when the server gave no reason", () => {
    expect(degradedProblem([]).message).toContain("could not run.");
  });
});

describe("cold cache", () => {
  it("not_cached says the sequence is not served and points at the Catalog", () => {
    const p = toProblem(422, "This build scores only sequences whose ESM-2 states are cached.", {
      code: "not_cached",
      retryable: false,
    });
    expect(p.kind).toBe("not_cached");
    expect(p.hint).toContain("Catalog");
    expect(p.hint).toContain("rather than guessed");
    expect(p.retryable).toBe(false);
  });

  it("batch failures are grouped per code and name their records", () => {
    const notices = failureProblems([
      { id: "a", error: "cold", code: "not_cached", retryable: false },
      { id: "b", error: "cold", code: "not_cached", retryable: false },
      { id: "c", error: "bad", code: "invalid_sequence", retryable: false },
    ]);
    expect(notices).toHaveLength(2);
    expect(notices[0]?.message).toContain("2 sequence(s): a, b.");
    expect(notices[1]?.kind).toBe("invalid_input");
  });
});

describe("codes drive the problem", () => {
  it("retryable comes from the server, not from the status", () => {
    expect(toProblem(500, "x", { code: "job_failed", retryable: true }).retryable).toBe(true);
    expect(toProblem(500, "x", { code: "model_error", retryable: false }).retryable).toBe(false);
  });

  it("an unknown code falls back to the status", () => {
    expect(toProblem(410, "x", { code: "from_the_future" }).kind).toBe("expired");
  });

  it("legacy responses: unreachable and 5xx are retryable, 4xx are not", () => {
    expect(toProblem(0, "x").retryable).toBe(true);
    expect(toProblem(503, "x").retryable).toBe(true);
    expect(toProblem(422, "x").retryable).toBe(false);
    expect(toProblem(410, "x").retryable).toBe(false);
  });

  it("DemoApiError carries the coded problem", () => {
    const e = new DemoApiError(404, "No structure", { code: "structure_not_found" });
    expect(e.problem.kind).toBe("no_structure");
    expect(e.problem.message).toBe("No structure");
  });

  it("a failed job with error_info uses its code", () => {
    const job = {
      job_id: "j",
      status: "failed",
      total: 1,
      completed_count: 0,
      created_at: 0,
      started_at: 0,
      finished_at: 1,
      error: "boom",
      error_info: { code: "job_failed", retryable: true },
      model_version: "v",
      config_hash: "c",
      rules_version: "r",
      experiment_id: "e",
      predictor: "p",
    } satisfies JobBody;
    const p = jobProblem(job);
    expect(p?.code).toBe("job_failed");
    expect(p?.retryable).toBe(true);
  });
});
