/*
 * Printable report (feature 10): one result laid out for paper or "Save as PDF". Hidden on
 * screen; the print stylesheet hides everything else and shows only this. It repeats what the
 * Inspector shows, with the same labels about which engine produced each output, and it carries
 * its own provenance and limits so a printed page read later is still honest about its scope.
 * Printing never sends anything anywhere.
 */
import type { ReactElement } from "react";
import type { Result } from "../services/demoApi";
import { sequenceLines, topResidues } from "../services/report";

const pct = (p: number): string => `${(p * 100).toFixed(1)}%`;
const pretty = (s: string): string => s.replaceAll("_", " ");

function source(r: Result, task: string): string {
  const t = r.task_sources[task];
  if (!t) return "not reported";
  return `${t.source}${t.calibrated ? ", calibrated" : ", not calibrated"}${t.bundle_version ? ` · bundle ${t.bundle_version}` : ""}`;
}

export function PrintReport({
  result: r,
  jobId,
  inSample,
  printedAt,
}: {
  result: Result;
  jobId: string;
  inSample: boolean;
  printedAt: Date;
}): ReactElement {
  const p = r.provenance;
  const predicted = r.localization.filter((l) => l.predicted);
  const goPredicted = r.go_function.filter((g) => g.predicted);
  const signal = r.sorting_signal;
  const residues = topResidues(r);
  return (
    <article className="print-report" aria-label={`Printable report for ${r.id}`}>
      <header>
        <h1>ProtLoc Fusion report · {r.id}</h1>
        <p>
          {r.length} residues · organism group {r.organism_group} · printed{" "}
          {printedAt.toISOString().replace("T", " ").slice(0, 16)} UTC
        </p>
      </header>

      {inSample && (
        <p className="pr-warn">
          In-sample sequence: the heads were fitted on it, so this result is not evidence of
          accuracy.
        </p>
      )}
      {!r.evidence_status.available && (
        <p className="pr-warn">
          Homology evidence unavailable ({r.evidence_status.unavailable_reason ?? "no reason"}). The
          model outputs are unaffected.
        </p>
      )}

      <section>
        <h2>Subcellular location</h2>
        <p className="pr-meta">
          {source(r, "localization")} · thresholds {r.threshold_policy}
        </p>
        <p>
          <strong>Call: </strong>
          {r.no_confident_call
            ? "no confident call (no label cleared its own threshold)"
            : predicted.map((l) => `${l.name} (${pct(l.probability)})`).join(", ") ||
              r.top_location}
        </p>
        <table>
          <thead>
            <tr>
              <th>label</th>
              <th>probability</th>
              <th>threshold</th>
              <th>predicted</th>
              <th>calibration</th>
            </tr>
          </thead>
          <tbody>
            {[...r.localization]
              .sort((a, b) => b.probability - a.probability)
              .map((l) => (
                <tr key={l.code}>
                  <td>
                    {l.name} ({l.code})
                  </td>
                  <td>{pct(l.probability)}</td>
                  <td>{pct(l.threshold)}</td>
                  <td>{l.predicted ? "yes" : "no"}</td>
                  <td>{l.calibrated ? l.calibration_method : "none"}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>

      <section className="pr-two">
        <div>
          <h2>Sorting signal</h2>
          <p className="pr-meta">{source(r, "signal_peptide")}</p>
          <p>
            {signal.type === "none"
              ? "No N-terminal signal detected."
              : `${pretty(signal.type)}, cleavage after residue ${String(signal.cleavage_site ?? "—")}.`}
            {signal.untrained?.length
              ? ` Not predicted: ${signal.untrained.join(", ")} (no trained head).`
              : ""}
          </p>
        </div>
        <div>
          <h2>Membrane type</h2>
          <p className="pr-meta">{source(r, "membrane_type")}</p>
          <p>
            {Object.entries(r.membrane_type)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => `${pretty(k)} ${pct(v)}`)
              .join(" · ")}
          </p>
        </div>
      </section>

      <section>
        <h2>GO molecular function</h2>
        <p className="pr-meta">{source(r, "go_function")}</p>
        {goPredicted.length === 0 ? (
          <p>No term cleared its threshold ({r.go_function.length} terms scored).</p>
        ) : (
          <ul>
            {goPredicted.map((g) => (
              <li key={g.go_id}>
                {g.go_id} {g.name} · {pct(g.probability)} (threshold {pct(g.threshold)})
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2>Consistency flags</h2>
        {r.consistency_flags.length === 0 ? (
          <p>None fired.</p>
        ) : (
          <ul>
            {r.consistency_flags.map((f) => (
              <li key={f.rule_id}>
                {f.rule_id} · {f.severity} · {f.message}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2>Homology evidence</h2>
        <p className="pr-meta">
          MMseqs2 against Swiss-Prot {r.evidence_status.swissprot_release} · never feeds a score
        </p>
        {r.evidence.length === 0 ? (
          <p>{r.evidence_status.available ? "No hit passed the filter." : "Not searched."}</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>accession</th>
                <th>identity</th>
                <th>e-value</th>
                <th>query coverage</th>
              </tr>
            </thead>
            <tbody>
              {r.evidence.slice(0, 5).map((h) => (
                <tr key={h.accession}>
                  <td>{h.accession}</td>
                  <td>{pct(h.identity)}</td>
                  <td>{h.evalue.toExponential(1)}</td>
                  <td>{pct(h.query_coverage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {r.evidence_agreement && (
          <p>
            Agreement with the model: {r.evidence_agreement.agrees ? "agrees" : "disagrees"} (
            {r.evidence_agreement.basis}).
          </p>
        )}
      </section>

      <section>
        <h2>Per-residue explanation</h2>
        <p className="pr-meta">
          {r.attribution_source} · method {r.attribution_method} · explains {r.attribution_target}
        </p>
        <p>
          {residues.length === 0
            ? "No residue has a positive attribution."
            : `Residues most supporting ${r.attribution_target}: ${residues.map((x) => `${x.residue}${String(x.position)}`).join(", ")}.`}{" "}
          Faithfulness to the trained head, not a claim about biology.
        </p>
      </section>

      <section>
        <h2>Sequence</h2>
        <pre>{sequenceLines(r.sequence).join("\n")}</pre>
      </section>

      <footer>
        <p>
          Model {p.model_version} · config {p.config_hash} · rules {p.rules_version} · experiment{" "}
          {p.experiment_id} · job {jobId} · sha256 {r.sequence_sha256.slice(0, 16)}…
        </p>
        <p>
          Scope: {p.scope_label ?? "not reported"}. Results from a research prototype; not for
          clinical or diagnostic use.
        </p>
      </footer>
    </article>
  );
}
