import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Accepted } from "../services/demoApi";
import { toProblem } from "../services/problems";
import { JobState } from "./ui";

const accepted: Accepted = {
  job_id: "9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04",
  result_url: "/demo/jobs/9f2c1e84",
  queue_position: 3,
  estimated_wait_seconds: 24.4,
};

describe("JobState", () => {
  it("shows the 202 state: queued, position and a labelled estimate", () => {
    render(<JobState job={null} error={null} accepted={accepted} />);
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("job 9f2c1e84");
    expect(status.textContent).toContain("queued");
    expect(status.textContent).toContain("position 3 in the queue");
    expect(status.textContent).toContain("about 24 s until done (estimate from measured speed)");
  });

  it("omits the estimate before any job has been timed", () => {
    render(
      <JobState job={null} error={null} accepted={{ ...accepted, estimated_wait_seconds: null }} />,
    );
    expect(screen.getByRole("status").textContent).not.toContain("about");
  });

  it("renders an expired result as an alert with a title and what to do", () => {
    render(<JobState job={null} error={toProblem(410, "Result expired")} accepted={null} />);
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("This result has expired");
    expect(alert.textContent).toContain("Run the prediction again");
    expect(alert.className).toContain("warning");
  });

  it("renders an unreachable backend as a blocking alert", () => {
    render(<JobState job={null} error={toProblem(0, "Failed to fetch")} />);
    const alert = screen.getByRole("alert");
    expect(alert.className).toContain("blocked");
    expect(alert.textContent).toContain("Cannot reach the backend");
  });

  it("renders nothing when there is nothing to say", () => {
    const { container } = render(<JobState job={null} error={null} />);
    expect(container.textContent).toBe("");
  });
});
