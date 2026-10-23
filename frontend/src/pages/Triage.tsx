/*
 * SCREEN 1 — Proteome Triage Board  (NOVEL)
 * Paste or generate a FASTA batch, POST /demo/batch, poll the job, and rank the
 * batch by ATTENTION SCORE: abstentions, consistency-rule severity, missing
 * homology support and multi-compartment calls push a protein to the top. The
 * biologist's question is "which 12 of these 200 do I have to look at myself" —
 * not "what is protein 57". Also renders the dual-targeting co-occurrence matrix.
 */
import { useEffect, useMemo, useState } from "react";
import { ServerDownloads } from "../components/ServerDownloads";
import { batch, useJob } from "../services/demoApi";
import { failureProblems } from "../services/problems";
import {
  Card,
  Chip,
  ExamplePicker,
  GroupSelect,
  JobState,
  ProblemNotice,
  ProvenanceBar,
  useExamples,
} from "../components/ui";

export default function Triage({ onOpen }: { onOpen: (seq: string, group: string) => void }) {
  const { examples } = useExamples();
  const [group, setGroup] = useState("eukaryote");
  const [fasta, setFasta] = useState("");
  const { run, retry, job, done, error, accepted, busy } = useJob();

  // Seed the textarea from the live /demo/examples payload, then run once.
  useEffect(() => {
    if (!examples.length || fasta) return;
    const pick = examples.filter((e) => e.organism_group === group);
    setFasta(pick.map((e) => `>${e.id}\n${e.sequence}`).join("\n"));
  }, [examples, group, fasta]);

  const submit = () => run(...batch(fasta, group));
  const rows = done?.summary ?? [];
  const land = done?.landscape;
  const kpi = useMemo(
    () => ({
      total: rows.length,
      abstained: rows.filter((r) => r.abstained).length,
      flagged: rows.filter((r) => r.n_flags > 0).length,
      disagree: rows.filter((r) => r.evidence_agrees === false).length,
    }),
    [rows],
  );

  return (
    <>
      <div className="head">
        <div>
          <h1>Proteome Triage Board</h1>
          <p className="sub">
            Rank a batch by how much human attention it needs. Attention score combines abstention,
            consistency-rule severity, absence of a close homolog and multi-compartment calls.
          </p>
        </div>
        <div className="row">
          <GroupSelect
            value={group}
            onChange={(g) => {
              setGroup(g);
              setFasta("");
            }}
            servedOnly
          />
          <ExamplePicker
            examples={examples}
            value=""
            onPick={(e) => {
              setFasta((f) => `${f}\n>${e.id}\n${e.sequence}`.trim());
            }}
          />
          <button className="btn primary" onClick={submit} disabled={busy || !fasta.trim()}>
            {busy ? "Running…" : "Run batch"}
          </button>
        </div>
      </div>

      <Card
        title="Input (FASTA)"
        aside={
          <span className="muted mono" style={{ fontSize: "var(--text-xs)" }}>
            {fasta.split(">").length - 1} sequences
          </span>
        }
      >
        <textarea
          value={fasta}
          onChange={(e) => {
            setFasta(e.target.value);
          }}
          spellCheck={false}
        />
        <JobState job={job} error={error} accepted={accepted} onRetry={retry} />
        {failureProblems(done?.failures ?? []).map((p) => (
          <ProblemNotice key={p.code ?? p.title} problem={p} />
        ))}
      </Card>

      {done && (
        <>
          <div className="grid kpi">
            {(
              [
                ["sequences", kpi.total],
                ["abstained", kpi.abstained],
                ["rule flags", kpi.flagged],
                ["evidence conflicts", kpi.disagree],
              ] as const
            ).map(([l, v]) => (
              <div className="card" key={l} style={{ marginBottom: 0 }}>
                <div className="kpi-v">{v}</div>
                <div className="kpi-l">{l}</div>
              </div>
            ))}
          </div>

          <Card
            title="Triage queue"
            aside={
              <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
                click a row to open it in the Inspector
              </span>
            }
          >
            <ServerDownloads jobId={done.job_id} />
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>id</th>
                  <th>len</th>
                  <th>call</th>
                  <th>p</th>
                  <th>signal</th>
                  <th>flags</th>
                  <th>evidence</th>
                  <th>attention</th>
                  <th>why</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const full = done.results?.find((x) => x.id === r.id);
                  return (
                    <tr
                      key={r.id}
                      className="pick"
                      onClick={() => full && onOpen(full.sequence, full.organism_group)}
                    >
                      <td className="mono">{i + 1}</td>
                      <td className="mono">{r.id}</td>
                      <td className="mono">{r.length}</td>
                      <td>
                        <Chip code={r.top_location} membrane={r.membrane} />
                        {r.abstained && <span className="muted"> abstained</span>}
                      </td>
                      <td className="mono">{r.top_probability.toFixed(2)}</td>
                      <td className="mono">{r.signal}</td>
                      <td className="mono">{r.worst_flag ?? "—"}</td>
                      <td className="mono">
                        {r.evidence_agrees === null
                          ? "—"
                          : r.evidence_agrees
                            ? "agrees"
                            : "conflicts"}
                      </td>
                      <td className="mono">{r.attention.toFixed(2)}</td>
                      <td className="muted" style={{ fontSize: "var(--text-xs)" }}>
                        {r.reasons.join(", ") || "clean"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          {land && (
            <Card
              title="Dual-targeting landscape"
              aside={
                <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
                  counts of proteins predicted to both compartments
                </span>
              }
            >
              <table className="hm">
                <thead>
                  <tr>
                    <th />
                    {land.codes.map((c) => (
                      <th key={c} style={{ textAlign: "center" }}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {land.codes.map((a) => (
                    <tr key={a}>
                      <td>
                        <Chip code={a} name={land.names[a]} />
                      </td>
                      {land.codes.map((b) => {
                        const n = land.matrix[a]?.[b] ?? 0;
                        const max = Math.max(1, ...land.codes.map((x) => land.matrix[x]?.[x] ?? 0));
                        return (
                          <td
                            key={b}
                            title={`${a} + ${b}: ${n}`}
                            style={{
                              background: n
                                ? `color-mix(in srgb, var(--accent) ${Math.round((n / max) * 70)}%, transparent)`
                                : "transparent",
                            }}
                          >
                            {n || ""}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
                Diagonal = proteins assigned to that compartment. Off-diagonal = co-assignment, i.e.
                candidate dual-targeting. {land.abstained} of {land.total} abstained.
              </p>
            </Card>
          )}
          <ProvenanceBar
            p={
              done
                ? {
                    model_version: done.model_version,
                    config_hash: done.config_hash,
                    rules_version: done.rules_version,
                    experiment_id: done.experiment_id,
                    predictor: done.predictor,
                  }
                : null
            }
          />
        </>
      )}
    </>
  );
}
