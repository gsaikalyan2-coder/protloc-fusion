/*
 * SCREEN 2 — Evidence-linked Inspector
 * One sequence, all seven outputs on one surface, each traceable to residues:
 * calibrated multi-label localization with the tau marker, membrane type as a chip
 * modifier, sorting signal + cleavage coordinate, GO motif hits, per-residue
 * attribution, consistency flags, and the homology verdict. POST /demo/predict,
 * then poll — same async contract as the batch path (ADR-0001).
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
  GROUPS,
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
            Seven annotations for one sequence, each anchored to the residues that produced it.
          </p>
        </div>
        <div className="row">
          <select
            value={group}
            onChange={(e) => {
              setGroup(e.target.value);
            }}
          >
            {GROUPS.map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
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
                    T={r.temperature} · τ={r.tau} · {r.threshold_policy}
                  </span>
                </span>
              }
            >
              {r.localization.map((row) => (
                <ConfidenceRow key={row.code} row={row} />
              ))}
              {r.no_confident_call && (
                <div className="flag uncertain">
                  <strong>No confident call.</strong> No compartment clears its threshold (τ ={" "}
                  {r.tau}). The highest score, {r.top_location} at{" "}
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
                title="GO function (motif-derived)"
                aside={<ProvenanceBadge src={r.task_sources.go_function} />}
              >
                {r.go_terms.length === 0 && (
                  <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
                    No scoring motif matched.
                  </p>
                )}
                {r.go_terms.map((g) => (
                  <p key={g.go_id} style={{ fontSize: "var(--text-sm)", margin: "0 0 4px" }}>
                    <span className="mono">{g.go_id}</span> {g.term}{" "}
                    <span className="muted mono">
                      [{g.span[0] + 1}–{g.span[1]}]
                    </span>
                  </p>
                ))}
              </Card>
            </div>
          </div>

          <Card
            title="Per-residue explanation"
            aside={
              <div className="row">
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
                Closest reference hit is annotated{" "}
                <Chip code={r.evidence_agreement.top_hit_location} /> at{" "}
                <span className="mono">{(r.evidence_agreement.identity * 100).toFixed(1)}%</span>{" "}
                identity —{" "}
                {r.evidence_agreement.agrees
                  ? "agrees with the model."
                  : "disagrees with the model."}
              </p>
            )}
          </Card>
          <ProvenanceBar p={r.provenance} />
        </>
      )}
    </>
  );
}
