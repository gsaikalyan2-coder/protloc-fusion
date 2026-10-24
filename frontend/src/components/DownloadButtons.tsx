/* Three client-side downloads of the result on screen. Built in the browser, never uploaded. */
import type { ReactElement } from "react";
import { buildExport, saveFile, type ExportFormat } from "../services/download";
import type { Result } from "../services/demoApi";
import { ServerDownloads } from "./ServerDownloads";
import { Card } from "./ui";

const FORMATS: [ExportFormat, string, string][] = [
  ["json", "JSON", "the full result, as the server returned it"],
  ["csv", "CSV", "per-residue tracks and per-label scores"],
  ["fasta", "FASTA", "the input sequence, with the job id in the header"],
];

export function DownloadButtons({
  result,
  jobId,
}: {
  result: Result;
  jobId: string;
}): ReactElement {
  return (
    <Card title="Download">
      <div className="row" role="group" aria-label="Download this result">
        {FORMATS.map(([format, label, what]) => (
          <button
            key={format}
            className="btn sm"
            title={what}
            aria-label={`Download ${label}: ${what}`}
            onClick={() => {
              saveFile(buildExport(format, result, jobId));
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="note">
        Files are built in your browser from this result, so they still work after the result
        expires. Each states its model version, the engine and method behind every track, and its
        scope.
      </p>
      {jobId !== "unknown" && <ServerDownloads jobId={jobId} />}
    </Card>
  );
}
