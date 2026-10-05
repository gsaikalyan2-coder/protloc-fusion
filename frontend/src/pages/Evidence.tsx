/*
 * SCREEN 3 — Independent Evidence Ledger  (NOVEL)
 * The trained model's call and the homology evidence are computed on separate paths and
 * shown side by side, so the screen can say "these two disagree" instead of quietly
 * averaging them. Hits come from the local Swiss-Prot MMseqs2 index (ADR-0003, ADR-0015),
 * searched inside /demo/predict. Model GO terms (source: model) and transferred GO terms
 * (source: homology, with the hit they came from) are two tables, never one.
 */
import { useEffect, useState } from "react";
import { predict, useJob } from "../services/demoApi";
import {
  Card,
  Chip,
  ExamplePicker,
  GroupSelect,
  JobState,
  ProvenanceBar,
  useExamples,
} from "../components/ui";

function termList(ids: string[], empty: string) {
  if (ids.length === 0) return <span className="muted">{empty}</span>;
  return (
    <span className="mono" style={{ fontSize: "var(--text-xs)" }}>
      {ids.join(" · ")}
    </span>
  );
}

export default function Evidence() {
  const { examples } = useExamples();
  const [seq, setSeq] = useState("");
  const [group, setGroup] = useState("eukaryote");
  const { run, job, done, error, accepted, busy } = useJob();

  useEffect(() => {
    const ex = examples[2] ?? examples[0];
    if (!seq && ex) {
      setSeq(ex.sequence);
      setGroup(ex.organism_group);
    }
  }, [examples, seq]);

  const r = done?.results?.[0] ?? null;
  const hits = r?.evidence ?? [];
  const status = r?.evidence_status;
  const top = hits[0];
  const agreement = r?.evidence_agreement ?? null;
  const modelTerms = (r?.go_function ?? []).filter((g) => g.predicted);
  const topRow = r?.localization.find((l) => l.code === r.top_location);

  return (
    <>
      <div className="head">
        <div>
          <h1>Evidence Ledger</h1>
          <p className="sub">
            Model call vs. independent homology evidence, never merged into one number. Evidence is
            the nearest Swiss-Prot entries by MMseqs2 on this server, and what those entries already
            say.
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
            {busy ? "Searching…" : "Search evidence"}
          </button>
        </div>
      </div>

      <Card title="Query">
        <textarea
          value={seq}
          onChange={(e) => {
            setSeq(e.target.value);
          }}
          spellCheck={false}
        />
        <JobState job={job} error={error} accepted={accepted} />
      </Card>

      {r && status && !status.available && (
        <div className="flag warning">
          <strong>Evidence unavailable.</strong> {status.unavailable_reason}. The model result is
          complete without it; nothing below was searched.
        </div>
      )}

      {r && (
        <>
          <div className="grid two">
            <Card title="Model says">
              <p className="row">
                <Chip code={r.top_location} name={topRow?.name} />
                <span>{topRow?.name}</span>
                <span className="mono">{topRow?.probability.toFixed(2)}</span>
              </p>
              <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
                {r.no_confident_call
                  ? "No compartment clears its own threshold: not asserted."
                  : "Asserted: at least one compartment clears its own threshold."}{" "}
                Sequence hash <span className="mono">{r.sequence_sha256.slice(0, 12)}</span>
              </p>
              <p style={{ fontSize: "var(--text-sm)", margin: 0 }}>
                GO function (model):{" "}
                {termList(
                  modelTerms.map((g) => g.name),
                  "no term reaches its threshold",
                )}
              </p>
            </Card>
            <Card title="Homology says">
              {!status?.available && <p className="muted">Not searched.</p>}
              {status?.available && hits.length === 0 && (
                <p className="muted">No Swiss-Prot hit passes the frozen filter.</p>
              )}
              {top && (
                <>
                  <p className="row">
                    <span className="mono">{top.accession}</span>
                    <span className="mono">{(top.identity * 100).toFixed(1)}% id</span>
                    <span className="mono">e {top.evalue.toExponential(1)}</span>
                    {top.exact_match && (
                      <span className="badge homology">this sequence is in Swiss-Prot</span>
                    )}
                  </p>
                  {agreement && (
                    <div className={`flag ${agreement.agrees ? "" : "warning"}`}>
                      {agreement.agrees
                        ? "The model's GO terms and the voting homologs' terms are the same set."
                        : "The model's GO terms and the voting homologs' terms differ — surfaced, not resolved."}
                    </div>
                  )}
                  {agreement && (
                    <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
                      {agreement.voting_hits} voting hit(s). {agreement.basis}.
                    </p>
                  )}
                </>
              )}
            </Card>
          </div>

          {agreement && (
            <Card title="Where the two paths differ (model GO vocabulary)">
              <table>
                <tbody>
                  <tr>
                    <td>both say</td>
                    <td>{termList(agreement.shared, "none")}</td>
                  </tr>
                  <tr>
                    <td>model only</td>
                    <td>{termList(agreement.model_only, "none")}</td>
                  </tr>
                  <tr>
                    <td>homology only</td>
                    <td>{termList(agreement.homology_only, "none")}</td>
                  </tr>
                </tbody>
              </table>
            </Card>
          )}

          <Card title="Top hits" aside={<span className="badge homology">homology</span>}>
            <table>
              <thead>
                <tr>
                  <th>accession</th>
                  <th>identity</th>
                  <th>e-value</th>
                  <th>bits</th>
                  <th>query cov.</th>
                  <th>target cov.</th>
                  <th>aln. length</th>
                  <th>GO terms carried</th>
                </tr>
              </thead>
              <tbody>
                {hits.map((h) => (
                  <tr key={h.accession}>
                    <td className="mono">{h.accession}</td>
                    <td className="mono">{(h.identity * 100).toFixed(1)}%</td>
                    <td className="mono">{h.evalue.toExponential(1)}</td>
                    <td className="mono">{h.bits.toFixed(0)}</td>
                    <td className="mono">{(h.query_coverage * 100).toFixed(0)}%</td>
                    <td className="mono">{(h.target_coverage * 100).toFixed(0)}%</td>
                    <td className="mono">{h.alignment_length}</td>
                    <td style={{ fontSize: "var(--text-xs)" }}>
                      {h.go_term_count === 0
                        ? "—"
                        : `${h.go_term_count}: ${h.go_terms
                            .slice(0, 3)
                            .map((t) => t.name)
                            .join(", ")}${h.go_term_count > 3 ? "…" : ""}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {hits.length === 0 && (
              <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
                No hits to show.
              </p>
            )}
          </Card>

          <Card
            title="GO terms carried by the voting homologs"
            aside={<span className="badge homology">homology · transferred</span>}
          >
            <table>
              <thead>
                <tr>
                  <th>term</th>
                  <th>name</th>
                  <th>share of voting hits</th>
                  <th>from</th>
                  <th>identity</th>
                  <th>in the model's vocabulary</th>
                </tr>
              </thead>
              <tbody>
                {r.evidence_go.map((t) => (
                  <tr key={t.go_id}>
                    <td className="mono">{t.go_id}</td>
                    <td>{t.name ?? "—"}</td>
                    <td className="mono">{(t.score * 100).toFixed(0)}%</td>
                    <td className="mono">{t.source_accession}</td>
                    <td className="mono">{(t.identity * 100).toFixed(1)}%</td>
                    <td>{t.in_model_vocabulary ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {r.evidence_go.length === 0 && (
              <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
                No term transferred.
              </p>
            )}
            <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
              Swiss-Prot {status?.swissprot_release}, evidence bundle {status?.bundle_version}.
              These terms are read from the matched entries' own annotation; they never enter a
              model score, a calibrator or a threshold. The hit's source accession is always shown.
            </p>
          </Card>
          <ProvenanceBar p={r.provenance} />
        </>
      )}
    </>
  );
}
