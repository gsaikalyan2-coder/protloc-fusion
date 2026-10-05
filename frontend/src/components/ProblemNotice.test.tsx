import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { JobBody } from "../services/demoApi";
import { degradedProblem, toProblem } from "../services/problems";
import { JobState, ProblemNotice } from "./ui";

const retryable = toProblem(500, "boom", { code: "job_failed", retryable: true });
const final = toProblem(422, "cold", { code: "not_cached", retryable: false });

describe("ProblemNotice retry", () => {
  it("shows Try again only when retryable and the screen can resubmit", () => {
    const onRetry = vi.fn();
    render(<ProblemNotice problem={retryable} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("draws no button for a problem a retry cannot fix", () => {
    render(<ProblemNotice problem={final} onRetry={vi.fn()} />);
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("draws no button when the screen offers no retry", () => {
    render(<ProblemNotice problem={retryable} />);
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows the stable code so a report can quote it", () => {
    render(<ProblemNotice problem={final} />);
    expect(screen.getByRole("alert").textContent).toContain("code: not_cached");
  });

  it("uses the tokenised classes, never a colour of its own", () => {
    render(<ProblemNotice problem={retryable} />);
    const alert = screen.getByRole("alert");
    expect(alert.className).toContain("blocked");
    expect(alert.getAttribute("style") ?? "").not.toMatch(/#|rgb|hsl/);
  });
});

describe("JobState situations", () => {
  const done = (over: Partial<JobBody>): JobBody =>
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

  it("evidence down: a completed job shows the degraded notice as a status", () => {
    render(
      <JobState
        job={done({ degraded: true, degraded_reasons: ["index not loaded"] })}
        error={null}
      />,
    );
    const note = screen.getByRole("status");
    expect(note.textContent).toContain("The evidence track is unavailable");
    expect(note.textContent).toContain("index not loaded");
    expect(note.textContent).toContain("scores are unchanged");
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("evidence working: nothing is drawn", () => {
    const { container } = render(<JobState job={done({ degraded: false })} error={null} />);
    expect(container.textContent).toBe("");
  });

  it("oversized input: the refusal is an alert with no retry", () => {
    const p = toProblem(422, "The sequence has 5001 residues; the limit is 5000.", {
      code: "sequence_too_long",
      retryable: false,
    });
    render(<JobState job={null} error={p} onRetry={vi.fn()} />);
    expect(screen.getByRole("alert").textContent).toContain("5001 residues");
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("malformed FASTA: names the line and offers no retry", () => {
    const p = toProblem(400, "line 2: 'x' is not a letter.", {
      code: "fasta_invalid_character",
      retryable: false,
      line: 2,
    });
    render(<JobState job={null} error={p} onRetry={vi.fn()} />);
    expect(screen.getByRole("alert").textContent).toContain("line 2");
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("cold cache: states it is not served, no retry", () => {
    render(<JobState job={null} error={final} onRetry={vi.fn()} />);
    expect(screen.getByRole("alert").textContent).toContain("Not one of the sequences");
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("an unreachable backend is retryable and the button calls retry", () => {
    const onRetry = vi.fn();
    render(<JobState job={null} error={toProblem(0, "Failed to fetch")} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("the degraded notice is built from the same helper the page uses", () => {
    expect(degradedProblem(["x"]).kind).toBe("evidence_degraded");
  });
});
