/*
 * Client-side exports of one result (gate B: no new API endpoint, no contract change).
 * Every file states, inside itself, the model version, the engine behind each track and the
 * method that computed it, so a file read months later without the page is still honest.
 * The builders are pure (they take `now`) so they can be tested without a browser.
 */
import type { Result } from "./demoApi";

export type ExportFormat = "json" | "csv" | "fasta";

export interface ExportFile {
  filename: string;
  mime: string;
  content: string;
}

const NOT_REPORTED = "not reported by the server";

function cell(v: string | number | boolean | null | undefined): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function stem(r: Result, jobId: string): string {
  const safe = r.id.replace(/[^A-Za-z0-9._-]/g, "_");
  return `protloc_${safe}_${jobId.slice(0, 8)}`;
}

function scope(r: Result): string {
  return r.provenance.scope_label ?? NOT_REPORTED;
}

function buildJson(r: Result, jobId: string, now: Date): ExportFile {
  const p = r.provenance;
  const doc = {
    export: {
      format: "json",
      exported_at: now.toISOString(),
      job_id: jobId,
      model_version: p.model_version,
      config_hash: p.config_hash,
      rules_version: p.rules_version,
      experiment_id: p.experiment_id,
      scope: scope(r),
      note: "The full result exactly as the server returned it. The engine behind each task is in result.task_sources; each residue track names its method in result.attribution_tracks and result.signal_track. Result URLs are not access-controlled and expire.",
    },
    result: r,
  };
  return {
    filename: `${stem(r, jobId)}.json`,
    mime: "application/json",
    content: `${JSON.stringify(doc, null, 2)}\n`,
  };
}

interface TrackColumn {
  name: string;
  source: string;
  method: string;
  values: readonly number[];
}

function trackColumns(r: Result): TrackColumn[] {
  const cols: TrackColumn[] = [
    {
      name: "attribution",
      source: r.attribution_source,
      method: r.attribution_method,
      values: r.attribution,
    },
  ];
  for (const [key, t] of Object.entries(r.attribution_tracks)) {
    if (key === r.attribution_method) continue;
    cols.push({ name: key, source: r.attribution_source, method: t.method, values: t.values });
  }
  cols.push({
    name: "signal_crf",
    source: r.task_sources.signal_peptide?.source ?? NOT_REPORTED,
    method: r.signal_track.method,
    values: r.signal_track.values,
  });
  cols.push({
    name: "hydropathy",
    source: "motif",
    method: "kyte_doolittle",
    values: r.hydropathy,
  });
  return cols;
}

function buildCsv(r: Result, jobId: string, now: Date): ExportFile {
  const p = r.provenance;
  const lines: string[] = [
    "# ProtLoc Fusion export: per-residue tracks and per-label scores",
    `# exported_at: ${now.toISOString()}`,
    `# job_id: ${jobId}`,
    `# model_version: ${p.model_version}`,
    `# config_hash: ${p.config_hash}`,
    `# rules_version: ${p.rules_version}`,
    `# experiment_id: ${p.experiment_id}`,
    `# scope: ${scope(r)}`,
    `# organism_group: ${r.organism_group}`,
    `# sequence_sha256: ${r.sequence_sha256}`,
    `# attribution_target: ${r.attribution_target}; signed: ${String(r.attribution_signed)} (signed: positive supports the label)`,
    "# Attribution describes the trained head, not causal importance and not a binding site. Comparable within this sequence and label only.",
    "# record_type: residue rows carry one track value per position; location and go_term rows carry one score per label.",
    "record_type,name,position,residue,value,raw,threshold,predicted,source,method,calibration",
  ];
  for (const col of trackColumns(r)) {
    for (let i = 0; i < r.sequence.length; i += 1) {
      lines.push(
        [
          "residue",
          col.name,
          i + 1,
          r.sequence.charAt(i),
          col.values[i],
          "",
          "",
          "",
          col.source,
          col.method,
          "",
        ]
          .map(cell)
          .join(","),
      );
    }
  }
  for (const row of r.localization) {
    lines.push(
      [
        "location",
        row.code,
        "",
        "",
        row.probability,
        row.raw,
        row.threshold,
        row.predicted,
        row.source,
        "",
        row.calibrated ? row.calibration_method : "uncalibrated",
      ]
        .map(cell)
        .join(","),
    );
  }
  for (const g of r.go_function) {
    lines.push(
      [
        "go_term",
        `${g.go_id} ${g.name}`,
        "",
        "",
        g.probability,
        g.raw,
        g.threshold,
        g.predicted,
        g.source,
        "",
        g.calibrated ? "calibrated" : "uncalibrated",
      ]
        .map(cell)
        .join(","),
    );
  }
  return {
    filename: `${stem(r, jobId)}.csv`,
    mime: "text/csv",
    content: `${lines.join("\n")}\n`,
  };
}

function buildFasta(r: Result, jobId: string): ExportFile {
  const id = r.id.replace(/\s+/g, "_");
  const header = `>${id}|job_id=${jobId}|source=input_sequence_not_a_prediction|group=${r.organism_group}|model_version=${r.provenance.model_version}|sha256=${r.sequence_sha256}`;
  const body: string[] = [];
  for (let i = 0; i < r.sequence.length; i += 60) body.push(r.sequence.slice(i, i + 60));
  return {
    filename: `${stem(r, jobId)}.fasta`,
    mime: "text/plain",
    content: `${header}\n${body.join("\n")}\n`,
  };
}

export function buildExport(
  format: ExportFormat,
  r: Result,
  jobId: string,
  now: Date = new Date(),
): ExportFile {
  if (format === "json") return buildJson(r, jobId, now);
  if (format === "csv") return buildCsv(r, jobId, now);
  return buildFasta(r, jobId);
}

/** Hand a built file to the browser. Nothing is uploaded anywhere. */
export function saveFile(f: ExportFile): void {
  const url = URL.createObjectURL(new Blob([f.content], { type: `${f.mime};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = f.filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 1000);
}
