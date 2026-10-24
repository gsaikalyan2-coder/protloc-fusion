import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { jobResultsUrl } from "../services/demoApi";
import { toProblem } from "../services/problems";
import { ServerDownloads } from "./ServerDownloads";

describe("ServerDownloads", () => {
  it("links each format to the job's server export", () => {
    render(<ServerDownloads jobId="9f2c1e84-3b7a" />);
    const csv = screen.getByText("CSV").closest("a");
    expect(csv?.getAttribute("href")).toBe("/demo/jobs/9f2c1e84-3b7a/results?format=csv");
    expect(csv?.hasAttribute("download")).toBe(true);
    expect(screen.getByText("JSON")).toBeTruthy();
    expect(screen.getByText("FASTA")).toBeTruthy();
  });

  it("offers only the formats it is given", () => {
    render(<ServerDownloads jobId="j" formats={["json"]} />);
    expect(screen.getByText("JSON")).toBeTruthy();
    expect(screen.queryByText("CSV")).toBeNull();
  });

  it("escapes the job id in the URL", () => {
    expect(jobResultsUrl("a/b c", "fasta")).toBe("/demo/jobs/a%2Fb%20c/results?format=fasta");
  });

  it("explains the export refusals by code", () => {
    const waiting = toProblem(409, "The job is running", {
      code: "job_not_finished",
      retryable: true,
    });
    expect(waiting.title).toBe("The job has not finished yet");
    expect(waiting.retryable).toBe(true);
    const none = toProblem(409, "failed", { code: "no_results", retryable: false });
    expect(none.title).toBe("This job has no results to download");
    const scan = toProblem(422, "scan", { code: "export_unsupported", retryable: false });
    expect(scan.hint).toContain("JSON only");
  });
});
