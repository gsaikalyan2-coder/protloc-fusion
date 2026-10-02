/*
 * SCREEN 3 — Independent Evidence Ledger  (NOVEL)
 * The model's call and the homology evidence are computed on separate paths and
 * shown side by side, so the screen can say "these two disagree" instead of
 * quietly averaging them. Hits come from a local reference set (GET /demo/reference,
 * searched inside /demo/predict): k-mer prefilter, then per-hit identity and the
 * aligned blocks, which are painted onto the query sequence.
 */
import { useEffect, useState } from "react";
import { getReference, predict, useJob } from "../services/demoApi";
import { SequenceTrack } from "../components/SequenceTrack";
import {
  Card,
  Chip,
  ExamplePicker,
  GROUPS,
  JobState,
  ProvenanceBar,
  useExamples,
} from "../components/ui";

export default function Evidence() {
  const { examples } = useExamples();
  const [seq, setSeq] = useState("");
  const [group, setGroup] = useState("eukaryote");
  const [pickHit, setPickHit] = useState(0);
  const [refCount, setRefCount] = useState<number | null>(null);
  const { run, job, done, error, busy } = useJob();

  useEffect(() => {
    getReference()
      .then((b) => {
        setRefCount(b.count);
      })
      .catch(() => {
        setRefCount(null);
      });
  }, []);
  useEffect(() => {
    const ex = examples[2];
    if (!seq && ex) {
      setSeq(ex.sequence);
      setGroup(ex.organism_group);
    }
  }, [examples, seq]);

  const r = done?.results?.[0] ?? null;
  const hits = r?.evidence ?? [];
  const hit = hits[pickHit];
  const top = hits[0];

  return (
    <>
      <div className="head">
        <div>
          <h1>Evidence Ledger</h1>
          <p className="sub">
            Model call vs. independent homology evidence, never merged into one number. Reference
            set: {refCount ?? "…"} annotated entries (synthetic, demo).
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
              setPickHit(0);
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
        <JobState job={job} error={error} />
      </Card>

      {r && (
        <>
          <div className="grid two">
            <Card title="Model says">
              <p className="row">
                <Chip code={r.top_location} name={r.top_location} />
                <span>{r.localization.find((l) => l.code === r.top_location)?.name}</span>
                <span className="mono">
                  {r.localization.find((l) => l.code === r.top_location)?.probability.toFixed(2)}
                </span>
              </p>
              <p className="muted" style={{ fontSize: "var(--text-sm)" }}>
                {r.abstained ? `Abstained (below τ = ${r.tau}).` : "Asserted above τ."} Sequence
                hash <span className="mono">{r.sequence_sha256.slice(0, 12)}</span>
              </p>
            </Card>
            <Card title="Evidence says">
              {hits.length === 0 && <p className="muted">No reference hit.</p>}
              {top && (
                <>
                  <p className="row">
                    <Chip code={top.location} name={top.location_name} />
                    <span>{top.location_name}</span>
                    <span className="mono">{(top.identity * 100).toFixed(1)}% id</span>
                  </p>
                  <div className={`flag ${r.evidence_agreement?.agrees ? "" : "warning"}`}>
                    {r.evidence_agreement?.agrees
                      ? "Independent evidence agrees with the model call."
                      : "Independent evidence disagrees with the model call — this is surfaced, not resolved."}
                  </div>
                </>
              )}
            </Card>
          </div>

          <Card
            title="Top hits"
            aside={
              <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
                click a hit to paint its aligned blocks
              </span>
            }
          >
            <table>
              <thead>
                <tr>
                  <th>accession</th>
                  <th>family</th>
                  <th>group</th>
                  <th>annotation</th>
                  <th>evidence code</th>
                  <th>identity</th>
                  <th>k-mer</th>
                  <th>coverage</th>
                </tr>
              </thead>
              <tbody>
                {hits.map((h, i) => (
                  <tr
                    key={h.accession}
                    className={`pick ${i === pickHit ? "on" : ""}`}
                    onClick={() => {
                      setPickHit(i);
                    }}
                  >
                    <td className="mono">{h.accession}</td>
                    <td className="mono">{h.family}</td>
                    <td className="mono">
                      {h.organism_group}
                      {h.same_group ? "" : " ⚠"}
                    </td>
                    <td>
                      <Chip code={h.location} name={h.location_name} /> {h.location_name}
                    </td>
                    <td className="mono">{h.evidence_code}</td>
                    <td className="mono">{(h.identity * 100).toFixed(1)}%</td>
                    <td className="mono">{h.kmer_jaccard.toFixed(3)}</td>
                    <td className="mono">{(h.coverage * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {hit && (
            <Card title={`Alignment coverage — ${hit.accession}`}>
              <SequenceTrack result={r} mode="alignment" blocks={hit.aligned_blocks} />
              <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
                {hit.aligned_blocks.length} aligned block(s) ≥ 8 residues. Annotation transfer is
                only as good as this coverage — which is why the ledger shows it rather than a
                single "homology supports" badge.
              </p>
            </Card>
          )}
          <ProvenanceBar p={r.provenance} />
        </>
      )}
    </>
  );
}
