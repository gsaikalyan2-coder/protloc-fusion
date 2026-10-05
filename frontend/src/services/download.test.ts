import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import { buildExport } from "./download";

const NOW = new Date("2026-10-05T12:00:00.000Z");
const JOB = "9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04";

describe("JSON export", () => {
  const f = buildExport("json", makeResult(), JOB, NOW);
  const doc = JSON.parse(f.content) as {
    export: Record<string, unknown>;
    result: { id: string; task_sources: Record<string, { source: string }> };
  };

  it("states model version, job id and scope inside the file", () => {
    expect(doc.export.model_version).toBe("1.0.0-test");
    expect(doc.export.job_id).toBe(JOB);
    expect(doc.export.exported_at).toBe("2026-10-05T12:00:00.000Z");
    expect(String(doc.export.scope)).toContain("not reported");
  });

  it("carries the full result, including the engine per task", () => {
    expect(doc.result.id).toBe("TEST1");
    expect(doc.result.task_sources.membrane_type?.source).toBe("motif");
    expect(f.filename).toBe("protloc_TEST1_9f2c1e84.json");
  });

  it("uses the server's scope label when there is one", () => {
    const r = makeResult();
    const withScope = buildExport(
      "json",
      { ...r, provenance: { ...r.provenance, scope_label: "subset (500)" } },
      JOB,
      NOW,
    );
    expect(withScope.content).toContain('"scope": "subset (500)"');
  });
});

describe("CSV export", () => {
  const f = buildExport("csv", makeResult(), JOB, NOW);
  const lines = f.content.trimEnd().split("\n");

  it("starts with comment lines naming version, method and caveat", () => {
    expect(lines[0]).toMatch(/^# /);
    expect(f.content).toContain("# model_version: 1.0.0-test");
    expect(f.content).toContain("# attribution_target: CYT; signed: true");
    expect(f.content).toContain("not causal importance");
  });

  it("has one row per residue per track, with source and method on each row", () => {
    const residue = lines.filter((l) => l.startsWith("residue,"));
    expect(residue).toHaveLength(4 * 12);
    expect(residue).toContain("residue,attribution,1,M,0.1,,,,model,occlusion,");
    expect(
      residue.some((l) => l.includes("hydropathy") && l.includes(",motif,kyte_doolittle,")),
    ).toBe(true);
    expect(residue.some((l) => l.includes("signal_crf") && l.includes(",crf_marginal,"))).toBe(
      true,
    );
  });

  it("has per-label score rows for locations and GO terms", () => {
    expect(lines).toContain("location,CYT,,,0.8,1.2,0.5,true,model,,isotonic");
    expect(lines).toContain(
      "go_term,GO:0003824 catalytic activity,,,0.7,0.9,0.4,true,model,,calibrated",
    );
  });

  it("quotes cells that contain a comma", () => {
    const g = buildExport(
      "csv",
      makeResult({
        localization: [],
        go_function: [
          {
            go_id: "GO:1",
            name: "a, b",
            aspect: "molecular_function",
            probability: 0.5,
            raw: 0.5,
            predicted: true,
            threshold: 0.4,
            source: "model",
            calibrated: true,
          },
        ],
      }),
      JOB,
      NOW,
    );
    expect(g.content).toContain('"GO:1 a, b"');
  });
});

describe("FASTA export", () => {
  it("puts the job id in the header and says it is not a prediction", () => {
    const f = buildExport("fasta", makeResult(), JOB, NOW);
    const [header, ...body] = f.content.trimEnd().split("\n");
    expect(header).toContain(`job_id=${JOB}`);
    expect(header).toContain("source=input_sequence_not_a_prediction");
    expect(header).toContain("model_version=1.0.0-test");
    expect(body.join("")).toBe("MKTAYIAKQRQI");
  });

  it("wraps long sequences at 60 residues", () => {
    const seq = "A".repeat(130);
    const f = buildExport("fasta", makeResult({ sequence: seq, length: 130 }), JOB, NOW);
    const body = f.content.trimEnd().split("\n").slice(1);
    expect(body.map((l) => l.length)).toEqual([60, 60, 10]);
  });
});
