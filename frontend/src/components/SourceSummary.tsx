/*
 * SourceSummary — one table naming the engine behind every output on the result page, and
 * whether a fitted calibrator was applied. Rows come from the result itself (task_sources,
 * attribution_source, evidence_status), so a task that falls back to the motif engine shows up
 * here as motif instead of being implied to be a trained model.
 */
import type { ReactElement } from "react";
import type { Result } from "../services/demoApi";
import { Card } from "./ui";

const TASKS: [string, string][] = [
  ["localization", "Subcellular location"],
  ["membrane_type", "Membrane type"],
  ["signal_peptide", "Signal peptide and cleavage"],
  ["go_function", "GO molecular function"],
];

interface Row {
  key: string;
  label: string;
  engine: string;
  calibration: string;
  detail: string;
}

function rowsOf(r: Result): Row[] {
  const rows: Row[] = [];
  for (const [key, label] of TASKS) {
    const s = r.task_sources[key];
    if (!s) continue;
    rows.push({
      key,
      label,
      engine: s.source,
      calibration: s.calibrated
        ? `calibrated${s.calibration_version ? ` (${s.calibration_version})` : ""}`
        : "not calibrated",
      detail: s.bundle_version ? `bundle ${s.bundle_version}` : "rule-based, no registry bundle",
    });
  }
  rows.push({
    key: "attribution",
    label: "Per-residue attribution",
    engine: r.attribution_source,
    calibration: "not applicable",
    detail: `method ${r.attribution_method}, target ${r.attribution_target}`,
  });
  rows.push({
    key: "hydropathy",
    label: "Hydropathy track",
    engine: "motif",
    calibration: "not applicable",
    detail: "Kyte-Doolittle, smoothed",
  });
  const ev = r.evidence_status;
  rows.push({
    key: "evidence",
    label: "Homology evidence",
    engine: "homology",
    calibration: "never feeds a score",
    detail: ev.available
      ? `Swiss-Prot ${ev.swissprot_release}, bundle ${ev.bundle_version}`
      : `unavailable: ${ev.unavailable_reason ?? "no reason given"}`,
  });
  return rows;
}

export function SourceSummary({ result }: { result: Result }): ReactElement {
  return (
    <Card title="What produced each output">
      <table>
        <thead>
          <tr>
            <th>output</th>
            <th>engine</th>
            <th>calibration</th>
            <th>detail</th>
          </tr>
        </thead>
        <tbody>
          {rowsOf(result).map((row) => (
            <tr key={row.key}>
              <td>{row.label}</td>
              <td>
                <span className={`badge ${row.engine}`}>{row.engine}</span>
              </td>
              <td>{row.calibration}</td>
              <td className="mono muted">{row.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
