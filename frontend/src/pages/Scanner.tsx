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
import { useEffect, useState, type CSSProperties } from "react";
import { scanReq, useJob } from "../services/demoApi";
import { Dna } from "lucide-react";
import {
  Card,
  Chip,
  EmptyState,
  JobState,
  ProvenanceBar,
  Select,
  SequenceForm,
  Skeleton,
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
  const { run, retry, job, done, error, accepted, busy } = useJob();

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
          <h1>Mutation scanner</h1>
          <p className="sub">
            Counterfactuals, not just predictions: which residues hold this localization up, and
            what breaks it. The scanner runs the interpretable motif engine (source: motif), not the
            trained heads, because it needs hundreds of re-scorings of sequences that have no cached
            ESM-2 states.
          </p>
        </div>
      </div>

      <SequenceForm
        id="scanner-input"
        label="Input"
        value={seq}
        onChange={setSeq}
        group={group}
        onGroup={setGroup}
        allGroups
        examples={examples}
        onPick={(e) => {
          setSeq(e.sequence);
          setGroup(e.organism_group);
          setTarget("");
        }}
        action="Run scan"
        busyAction="Scanning…"
        busy={busy}
        onSubmit={() => void run(...scanReq(seq, group, target || null, muts, sub))}
        compact={s !== null}
      >
        <Select
          aria-label="Target compartment"
          value={target}
          onChange={(e) => {
            setTarget(e.target.value);
          }}
        >
          <option value="">Target: top call</option>
          {codes.map((c) => (
            <option key={c} value={c}>
              Target: {c}
            </option>
          ))}
        </Select>
        <input
          aria-label="Substitute every position to"
          title="Substitute every position to this residue"
          className="mono input-xs"
          value={sub}
          maxLength={1}
          onChange={(e) => {
            setSub(e.target.value.toUpperCase());
          }}
        />
        <input
          aria-label="Variant to diff"
          className="mono"
          placeholder="variant, e.g. K83A,L20P"
          value={muts}
          onChange={(e) => {
            setMuts(e.target.value);
          }}
        />
      </SequenceForm>
      <JobState job={job} error={error} accepted={accepted} onRetry={retry} />
      {busy && !s && (
        <Card>
          <Skeleton lines={5} label="Running the scan" />
        </Card>
      )}

      {!busy && !s && !error && (
        <Card>
          <EmptyState icon={Dna} title="No scan yet">
            Run a scan to see the truncation series, a substitution scan of every position and, if
            you enter one, the variant diff.
          </EmptyState>
        </Card>
      )}

      {s && (
        <>
          <Card title={`N-terminal truncation series — target ${s.target} (${s.target_name})`}>
            <div className="table-wrap">
              <table className="hm">
                <thead>
                  <tr>
                    <th>removed</th>
                    {codes.map((c) => (
                      <th key={c} className="center">
                        {c}
                      </th>
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
                          className="heat-loc"
                          title={`${c} ${(t[c] ?? 0).toFixed(3)}`}
                          style={
                            {
                              "--heat": `${String(Math.round((t[c] ?? 0) * 60))}%`,
                              "--heat-ink": locText(c),
                            } as CSSProperties
                          }
                        >
                          {(t[c] ?? 0).toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note mt-3">
              Each row re-runs the full predictor on a shorter sequence. A cliff between two rows
              locates the functional signal.
            </p>
          </Card>

          <Card
            title={`Substitution scan — every position → ${s.substitute}`}
            aside={
              <span className="note mono">
                baseline {s.target} = {(s.baseline[s.target] ?? 0).toFixed(2)}
              </span>
            }
          >
            <p className="seq scan-seq">
              {s.sequence.split("").map((aa, i) => {
                const d = s.position_deltas[i] ?? 0;
                const mag = Math.round((Math.abs(d) / maxAbs) * 75);
                return (
                  <span
                    key={i}
                    title={`${aa}${i + 1}${s.substitute}: Δ${d.toFixed(3)}`}
                    data-dir={d < 0 ? "down" : d > 0 ? "up" : undefined}
                    style={{ "--mag": `${String(mag)}%` } as CSSProperties}
                  >
                    {aa}
                  </span>
                );
              })}
            </p>
            <ul className="seq-legend">
              <li>
                <span className="seq-swatch sw-against" /> mutation lowers {s.target}
              </li>
              <li>
                <span className="seq-swatch sw-support" /> raises it
              </li>
              <li>hover a residue for Δ</li>
            </ul>
            <div className="table-wrap mt-3">
              <table>
                <thead>
                  <tr>
                    <th>rank</th>
                    <th>mutation</th>
                    <th className="num">Δ probability</th>
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
                      <td className="num">{m.delta.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {s.variant && (
            <Card title={`Variant diff — ${s.variant.mutations.join(", ")}`}>
              <div className={`flag ${s.variant.relocalized ? "warning" : ""}`}>
                {s.variant.relocalized
                  ? `Relocalized: ${s.variant.wild_type.top_location} → ${s.variant.variant.top_location}`
                  : `No change of top call (${s.variant.wild_type.top_location}).`}
              </div>
              <div className="table-wrap">
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
                            className={`num ${d < -0.01 ? "delta-down" : d > 0.01 ? "delta-up" : ""}`}
                          >
                            {d >= 0 ? "+" : ""}
                            {d.toFixed(2)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="note mt-3">
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
