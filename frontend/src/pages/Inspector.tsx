/*
 * SCREEN 2 — Evidence-linked Inspector
 * One sequence, the annotation outputs on one surface: calibrated multi-label localization
 * with each label's own threshold marker (Phase 10), membrane type as a chip modifier, the
 * signal peptide and its cleavage coordinate (Phase 11 CRF), GO function (Phase 12 head),
 * per-residue explanation, consistency flags, and the homology verdict. Every card says which
 * engine produced it: model or motif. POST /demo/predict, then poll (ADR-0001).
 */
import { useEffect, useState } from "react";
import { predict, useJob } from "../services/demoApi";
import { SequenceTrack } from "../components/SequenceTrack";
import {
  Card,
  Chip,
  ConfidenceRow,
  ExamplePicker,
  Flags,
  GroupSelect,
  JobState,
  ProvenanceBadge,
  ProvenanceBar,
  useExamples,
} from "../components/ui";

export default function Inspector({
  seed,
  onScan,
}: {
  seed: { sequence: string; group: string } | null;
  onScan: (seq: string, group: string) => void;
}) {
  const { examples } = useExamples();
  const [seq, setSeq] = useState(seed?.sequence ?? "");
  const [group, setGroup] = useState(seed?.group ?? "eukaryote");
  const [mode, setMode] = useState<"attribution" | "hydropathy">("attribution");
  const { run, job, done, error, busy } = useJob();

  useEffect(() => {
    if (seed) {
      setSeq(seed.sequence);
      setGroup(seed.group);
    }
  }, [seed]);

  useEffect(() => {
    const ex = examples[0];
    if (!seq && ex) {
      setSeq(ex.sequence);
      setGroup(ex.organism_group);
    }
  }, [examples, seq]);

  const r = done?.results?.[0] ?? null;

  return (
    <>
      <div className="head">
        <div>
          <h1>Inspector</h1>
          <p className="sub">
            Annotations for one sequence, each labelled with the engine that produced it.
          </p>
        </div>
        <div className="row">
          <GroupSelect value={group} onChange={setGroup} servedOnly />
          <ExamplePicker
            examples={examples}
            value=""
            onPick={(e) => {
              setSeq(e.sequence);
              setGroup(e.organism_group);
            }}
          />
          <button
            className="btn primary"
            onClick={() => run(...predict(seq, group))}
            disabled={busy || !seq.trim()}
          >
            {busy ? "Running…" : "Predict"}
          </button>
        </div>
      </div>

      <Card title="Sequence">
        <textarea
          value={seq}
          onChange={(e) => {
            setSeq(e.target.value);
          }}
          spellCheck={false}
        />
        <JobState job={job} error={error} />
      </Card>

      {r && (
        <>
          <div className="grid two">
            <Card
              title="Localization"
              aside={
                <span className="row" style={{ gap: 6 }}>
                  <ProvenanceBadge src={r.task_sources.localization} />
                  <span className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
                    {r.threshold_policy} · calibrator {r.calibration_version ?? "none"}
                  </span>
                </span>
              }
            >
              {r.localization.map((row) => (
                <ConfidenceRow key={row.code} row={row} />
              ))}
              {r.no_confident_call && (
                <div className="flag uncertain">
                  <strong>No confident call.</strong> No compartment clears its own threshold. The
                  highest score, {r.top_location} at{" "}
                  {r.localization.find((l) => l.code === r.top_location)?.probability.toFixed(2)},
                  is shown for transparency and is not asserted.
                </div>
              )}
            </Card>

            <div>
              <Card
                title="Membrane type"
                aside={<ProvenanceBadge src={r.task_sources.membrane_type} />}
              >
                <div className="row">
                  {Object.entries(r.membrane_type)
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => (
                      <span key={k} className="row" style={{ gap: 4 }}>
                        <Chip code={r.top_location} membrane={k} />
                        <span style={{ fontSize: "var(--text-sm)" }}>{k.replace("_", "-")}</span>
                        <span className="mono num">{v.toFixed(2)}</span>
                      </span>
                    ))}
                </div>
              </Card>
              <Card
                title="Sorting signal"
                aside={<ProvenanceBadge src={r.task_sources.signal_peptide} />}
              >
                <p className="mono">
                  {r.sorting_signal.type} · p={r.sorting_signal.probability.toFixed(2)}
                  {r.sorting_signal.region
                    ? ` · region ${r.sorting_signal.region[0] + 1}–${r.sorting_signal.region[1]}`
                    : ""}
                  {r.sorting_signal.cleavage_site !== null
                    ? ` · cleavage after ${r.sorting_signal.cleavage_site}`
                    : ""}
                </p>
                <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
                  candidates:{" "}
                  {Object.entries(r.sorting_signal.candidates)
                    .map(([k, v]) => `${k} ${v.toFixed(2)}`)
                    .join(" · ")}
                </p>
              </Card>
              <Card
                title="GO function (molecular function)"
                aside={<ProvenanceBadge src={r.task_sources.go_function} />}
              >
                {r.go_function.filter((g) => g.predicted).length === 0 && (
                  <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
                    No term reaches its threshold.
                  </p>
                )}
                {r.go_function
                  .filter((g) => g.predicted)
                  .map((g) => (
                    <p key={g.go_id} style={{ fontSize: "var(--text-sm)", margin: "0 0 4px" }}>
                      <span className="mono">{g.go_id}</span> {g.name}{" "}
                      <span className="muted mono">p={g.probability.toFixed(2)}</span>
                    </p>
                  ))}
                <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
                  Predicted by the trained head, calibrated, propagated to ancestors; threshold{" "}
                  {r.go_function[0]?.threshold.toFixed(2)}. Terms carried by homologs are on the
                  Evidence Ledger, not here.
                </p>
              </Card>
            </div>
          </div>

          <Card
            title="Per-residue explanation"
            aside={
              <div className="row">
                <span
                  className={`badge ${r.attribution_source}`}
                  title="Which engine produced the per-residue weights. A trained-model attribution arrives with Phase 15."
                >
                  attribution · {r.attribution_source}
                </span>
                <button
                  className="btn"
                  onClick={() => {
                    setMode("attribution");
                  }}
                  disabled={mode === "attribution"}
                >
                  attribution
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    setMode("hydropathy");
                  }}
                  disabled={mode === "hydropathy"}
                >
                  hydropathy
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    onScan(r.sequence, r.organism_group);
                  }}
                >
                  send to Scanner →
                </button>
              </div>
            }
          >
            <SequenceTrack result={r} mode={mode} />
          </Card>

          <Card title="Consistency check">
            <Flags flags={r.consistency_flags} />
            {r.evidence_agreement && (
              <p style={{ fontSize: "var(--text-sm)" }}>
                Closest Swiss-Prot hit{" "}
                <span className="mono">{r.evidence_agreement.top_hit_accession}</span> at{" "}
                <span className="mono">{(r.evidence_agreement.identity * 100).toFixed(1)}%</span>{" "}
                identity — its GO terms{" "}
                {r.evidence_agreement.agrees
                  ? "agree with the model's."
                  : "differ from the model's (see the Evidence Ledger)."}
              </p>
            )}
          </Card>
          <ProvenanceBar p={r.provenance} />
        </>
      )}
    </>
  );
}
