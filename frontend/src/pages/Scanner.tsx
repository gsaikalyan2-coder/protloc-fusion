/*
 * SCREEN 4 — Counterfactual Mutation Scanner  (NOVEL)
 * Answers "what would have to change for this call to change" instead of just
 * reporting the call. Three live views from POST /demo/scan:
 *   1. N-terminal truncation series — probability of every compartment as the
 *      N-terminus is removed 3 residues at a time (the signal-peptide cliff)
 *   2. Substitution scan — every position mutated to a chosen residue, coloured
 *      by the delta it causes in the target compartment's probability
 *   3. Variant diff — a user's mutation list (e.g. K83A,L20P) run as wild-type vs variant
 */
import { useEffect, useState } from "react";
import { scanReq, useJob } from "../services/demoApi";
import {
  Card,
  Chip,
  ExamplePicker,
  GroupSelect,
  JobState,
  ProvenanceBar,
  locText,
  useExamples,
} from "../components/ui";

export default function Scanner({ seed }: { seed: { sequence: string; group: string } | null }) {
  const { examples } = useExamples();
  const [seq, setSeq] = useState(seed?.sequence ?? "");
  const [group, setGroup] = useState(seed?.group ?? "eukaryote");
  const [target, setTarget] = useState<string>("");
  const [muts, setMuts] = useState("");
  const [sub, setSub] = useState("A");
  const { run, job, done, error, accepted, busy } = useJob();

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

  const s = done?.scan ?? null;
  const codes = s ? Object.keys(s.baseline) : [];
  const maxAbs = s ? Math.max(0.01, ...s.position_deltas.map((d) => Math.abs(d))) : 1;

  return (
    <>
      <div className="head">
        <div>
          <h1>Mutation Scanner</h1>
          <p className="sub">
            Counterfactuals, not just predictions: which residues hold this localization up, and
            what breaks it. The scanner runs the interpretable motif engine (source: motif), not the
            trained heads, because it needs hundreds of re-scorings of sequences that have no cached
            ESM-2 states.
          </p>
        </div>
        <div className="row">
          <GroupSelect value={group} onChange={setGroup} />
          <ExamplePicker
            examples={examples}
            value=""
            onPick={(e) => {
              setSeq(e.sequence);
              setGroup(e.organism_group);
              setTarget("");
            }}
          />
        </div>
      </div>

      <Card title="Input">
        <textarea
          value={seq}
          onChange={(e) => {
            setSeq(e.target.value);
          }}
          spellCheck={false}
        />
        <div className="row" style={{ marginTop: "var(--space-3)" }}>
          <label className="muted" style={{ fontSize: "var(--text-sm)" }}>
            target
            <select
              value={target}
              onChange={(e) => {
                setTarget(e.target.value);
              }}
              style={{ marginLeft: 8 }}
            >
              <option value="">(top call)</option>
              {(codes.length ? codes : []).map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="muted" style={{ fontSize: "var(--text-sm)" }}>
            substitute to
            <input
              value={sub}
              maxLength={1}
              onChange={(e) => {
                setSub(e.target.value.toUpperCase());
              }}
              style={{ width: 52, marginLeft: 8 }}
            />
          </label>
          <input
            placeholder="variant, e.g. K83A,L20P"
            value={muts}
            onChange={(e) => {
              setMuts(e.target.value);
            }}
            style={{ minWidth: 220 }}
          />
          <button
            className="btn primary"
            onClick={() => run(...scanReq(seq, group, target || null, muts, sub))}
            disabled={busy || !seq.trim()}
          >
            {busy ? "Scanning…" : "Run scan"}
          </button>
        </div>
        <JobState job={job} error={error} accepted={accepted} />
      </Card>

      {s && (
        <>
          <Card title={`N-terminal truncation series — target ${s.target} (${s.target_name})`}>
            <table>
              <thead>
                <tr>
                  <th>removed</th>
                  {codes.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {s.truncation.map((t) => (
                  <tr key={t.removed}>
                    <td className="mono">{t.removed}</td>
                    {codes.map((c) => (
                      <td
                        key={c}
                        className="mono"
                        title={`${c} ${(t[c] ?? 0).toFixed(3)}`}
                        style={{
                          background: `color-mix(in srgb, ${locText(c)} ${Math.round((t[c] ?? 0) * 60)}%, transparent)`,
                        }}
                      >
                        {(t[c] ?? 0).toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
              Each row re-runs the full predictor on a shorter sequence. A cliff between two rows
              locates the functional signal.
            </p>
          </Card>

          <Card
            title={`Substitution scan — every position → ${s.substitute}`}
            aside={
              <span className="muted mono" style={{ fontSize: "var(--text-xs)" }}>
                baseline {s.target} = {(s.baseline[s.target] ?? 0).toFixed(2)}
              </span>
            }
          >
            <p className="seq">
              {s.sequence.split("").map((aa, i) => {
                const d = s.position_deltas[i] ?? 0;
                const mag = Math.round((Math.abs(d) / maxAbs) * 75);
                return (
                  <span
                    key={i}
                    title={`${aa}${i + 1}${s.substitute}: Δ${d.toFixed(3)}`}
                    style={{
                      background:
                        d < 0
                          ? `color-mix(in srgb, var(--state-block) ${mag}%, transparent)`
                          : d > 0
                            ? `color-mix(in srgb, var(--accent) ${mag}%, transparent)`
                            : "transparent",
                    }}
                  >
                    {aa}
                  </span>
                );
              })}
            </p>
            <div className="row" style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
              <span>red: mutation lowers {s.target}</span>
              <span>green: raises it</span>
              <span>hover for Δ</span>
            </div>
            <table style={{ marginTop: "var(--space-3)" }}>
              <thead>
                <tr>
                  <th>rank</th>
                  <th>mutation</th>
                  <th>Δ probability</th>
                </tr>
              </thead>
              <tbody>
                {s.most_sensitive.map((m, i) => (
                  <tr key={m.position}>
                    <td className="mono">{i + 1}</td>
                    <td className="mono">
                      {m.residue}
                      {m.position}
                      {s.substitute}
                    </td>
                    <td className="mono">{m.delta.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {s.variant && (
            <Card title={`Variant diff — ${s.variant.mutations.join(", ")}`}>
              <div className={`flag ${s.variant.relocalized ? "warning" : ""}`}>
                {s.variant.relocalized
                  ? `Relocalized: ${s.variant.wild_type.top_location} → ${s.variant.variant.top_location}`
                  : `No change of top call (${s.variant.wild_type.top_location}).`}
              </div>
              <table>
                <thead>
                  <tr>
                    <th>compartment</th>
                    <th>wild type</th>
                    <th>variant</th>
                    <th>Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {s.variant.wild_type.localization.map((w) => {
                    const v = s.variant!.variant.localization.find((x) => x.code === w.code)!;
                    const d = v.probability - w.probability;
                    return (
                      <tr key={w.code}>
                        <td>
                          <Chip code={w.code} name={w.name} /> {w.name}
                        </td>
                        <td className="mono">{w.probability.toFixed(2)}</td>
                        <td className="mono">{v.probability.toFixed(2)}</td>
                        <td
                          className="mono"
                          style={{
                            color:
                              d < -0.01
                                ? "var(--state-block)"
                                : d > 0.01
                                  ? "var(--accent)"
                                  : undefined,
                          }}
                        >
                          {d >= 0 ? "+" : ""}
                          {d.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
                signal: {s.variant.wild_type.sorting_signal.type} →{" "}
                {s.variant.variant.sorting_signal.type}
                {" · "}flags: {s.variant.wild_type.consistency_flags.length} →{" "}
                {s.variant.variant.consistency_flags.length}
              </p>
            </Card>
          )}
          <ProvenanceBar p={s.provenance} />
        </>
      )}
    </>
  );
}
