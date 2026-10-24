/*
 * Whole-job downloads built by the server (GET /demo/jobs/{id}/results). The browser-built
 * files in DownloadButtons cover one result; these cover the whole job, so a batch becomes one
 * CSV row per sequence. They are plain links: the server sends them as attachments, and they
 * stop working when the job expires (the browser-built files do not).
 */
import type { ReactElement } from "react";
import { jobResultsUrl, type JobExportFormat } from "../services/demoApi";

const FORMATS: [JobExportFormat, string, string][] = [
  ["csv", "CSV", "one row per sequence: calls, signal, GO terms, flags, nearest homolog"],
  ["json", "JSON", "every result and refusal in the job, as the server holds it"],
  ["fasta", "FASTA", "the input sequences, with the job id and model version in each header"],
];

export function ServerDownloads({
  jobId,
  formats = ["csv", "json", "fasta"],
}: {
  jobId: string;
  formats?: readonly JobExportFormat[];
}): ReactElement {
  return (
    <div className="row" role="group" aria-label="Download the whole job from the server">
      <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
        Whole job, from the server:
      </span>
      {FORMATS.filter(([f]) => formats.includes(f)).map(([format, label, what]) => (
        <a
          key={format}
          className="btn"
          href={jobResultsUrl(jobId, format)}
          download
          title={what}
          aria-label={`Download the whole job as ${label}: ${what}`}
        >
          {label}
        </a>
      ))}
    </div>
  );
}
