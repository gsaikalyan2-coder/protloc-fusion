/*
 * SCREEN 5 — Trust & Abstention Console  (NOVEL)
 * What a probability from this system is worth. GET /demo/trust returns the frozen Phase 10
 * calibration of the localization head, read from its manifest: per-label calibrator, the
 * label's own MCC-tuned threshold, and reliability measured on validation partitions 1-4.
 * Nothing is refitted here and nothing is computed in the browser. There is no global tau:
 * a call is asserted when a label clears its own threshold, and "no confident call" is the
 * answer when none does.
 */
import { describeError } from "../services/problems";
import { useEffect, useState } from "react";
import { getTrust, getVersion } from "../services/demoApi";
import { Card, Chip, ProvenanceBar } from "../components/ui";

interface Interval {
  value: number;
  ci_low: number;
  ci_high: number;
}
interface Bin {
  fraction_positive: number | null;
  mean_score: number | null;
  rows: number;
}
interface LabelTrust {
  label: string;
  code: string;
  name: string;
  method: string;
  calibrated: boolean;
  threshold: number;
  positives: number;
  negatives: number;
  ece_raw: Interval;
  ece_calibrated: Interval;
  ece_delta: Interval;
  ece_change_significant: boolean;
  reliability: { calibrated: Bin[]; raw: Bin[] };
}
interface TrustBody {
  threshold_policy: string;
  calibration_version: string;
  bundle_version: string;
  bundle_sha256: string;
  scope_label: string;
  n_sequences: number;
  n_labelled_rows: number;
  validation_partitions: number[];
  bins: number;
  macro: {
    ece_raw: Interval;
    ece_calibrated: Interval;
    ece_delta: Interval;
    brier_raw: Interval;
    brier_calibrated: Interval;
  };
  significance_rule: string;
  labels: LabelTrust[];
  provenance: Parameters<typeof ProvenanceBar>[0]["p"];
}

const fmt = (x: number) => x.toFixed(3);
const span = (i: Interval) => `${fmt(i.value)} [${fmt(i.ci_low)}, ${fmt(i.ci_high)}]`;

export default function Trust() {
  const [t, setT] = useState<TrustBody | null>(null);
  const [rules, setRules] = useState<
    { id: string; severity: string; message: string; active?: boolean }[]
  >([]);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<string>("");

  useEffect(() => {
    getTrust()
      .then((b) => {
        const body = b as unknown as TrustBody;
        setT(body);
        setPick(body.labels[0]?.label ?? "");
      })
      .catch((e) => {
        setError(describeError(e));
      });
    getVersion()
      .then((v) => {
        setRules(v.rules);
      })
      .catch(() => undefined);
  }, []);

  if (error) return <p className="err">Backend error: {error}</p>;
  if (!t) return <p className="muted">Loading calibration…</p>;

  const chosen = t.labels.find((l) => l.label === pick) ?? t.labels[0];
  const macro = t.macro;
  const eceRaw = macro.ece_raw;
  const eceCal = macro.ece_calibrated;
  const brierRaw = macro.brier_raw;
  const brierCal = macro.brier_calibrated;
  const eceDelta = macro.ece_delta;

  return (
    <>
      <div className="head">
        <div>
          <h1>Trust &amp; Abstention Console</h1>
          <p className="sub">
            What a probability from this system is worth. Calibrators and per-label thresholds were
            fitted cross-fold on validation partitions {t.validation_partitions.join(", ")}; the
            numbers below are the frozen Phase 10 measurements, not recomputed here.
          </p>
          <p className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
            {t.scope_label}
          </p>
          <p className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
            threshold policy: {t.threshold_policy} · calibrator: {t.calibration_version} · source:
            model · bundle {t.bundle_version} ({t.bundle_sha256.slice(0, 12)}…)
          </p>
        </div>
      </div>

      <div className="grid kpi">
        {(
          [
            ["macro ECE before", span(eceRaw)],
            ["macro ECE after", span(eceCal)],
            ["macro Brier before → after", `${fmt(brierRaw.value)} → ${fmt(brierCal.value)}`],
            ["ECE change", span(eceDelta)],
          ] as const
        ).map(([l, v]) => (
          <div className="card" key={l} style={{ marginBottom: 0 }}>
            <div className="kpi-v" style={{ fontSize: "var(--text-lg)" }}>
              {v}
            </div>
            <div className="kpi-l">{l}</div>
          </div>
        ))}
      </div>

      <Card
        title="Per-label calibrator and threshold"
        aside={
          <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
            {t.n_labelled_rows} labelled rows · 95% bootstrap intervals by cluster
          </span>
        }
      >
        <table>
          <thead>
            <tr>
              <th>label</th>
              <th>calibrator</th>
              <th>threshold</th>
              <th>positives</th>
              <th>ECE before</th>
              <th>ECE after</th>
              <th>change</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {t.labels.map((l) => (
              <tr
                key={l.label}
                className={`pick ${l.label === chosen?.label ? "on" : ""}`}
                onClick={() => {
                  setPick(l.label);
                }}
              >
                <td>
                  <Chip code={l.code} name={l.name} /> {l.name}
                </td>
                <td className="mono">{l.calibrated ? l.method : "identity"}</td>
                <td className="mono">{l.threshold.toFixed(3)}</td>
                <td className="mono">{l.positives}</td>
                <td className="mono">{fmt(l.ece_raw.value)}</td>
                <td className="mono">{fmt(l.ece_calibrated.value)}</td>
                <td className="mono">{span(l.ece_delta)}</td>
                <td className="muted" style={{ fontSize: "var(--text-xs)" }}>
                  {l.ece_change_significant ? "interval excludes 0" : "no significant change"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
          {t.significance_rule}. Labels with few positives (peroxisome, plastid) carry wide
          intervals; read their rows as indications, not findings.
        </p>
      </Card>

      {chosen && (
        <Card
          title={`Reliability — ${chosen.name}`}
          aside={
            <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
              {t.bins} equal-width score bins · click a label above to change
            </span>
          }
        >
          <table>
            <thead>
              <tr>
                <th>score bin</th>
                <th>n</th>
                <th>mean score (after)</th>
                <th>observed positive rate</th>
                <th>gap</th>
                <th>uncalibrated: mean score → positive rate</th>
              </tr>
            </thead>
            <tbody>
              {chosen.reliability.calibrated.map((b, i) => {
                const raw = chosen.reliability.raw[i];
                if (b.rows === 0 || b.mean_score === null || b.fraction_positive === null)
                  return null;
                const gap = b.fraction_positive - b.mean_score;
                return (
                  <tr key={i}>
                    <td className="mono">
                      {(i / t.bins).toFixed(1)}–{((i + 1) / t.bins).toFixed(1)}
                    </td>
                    <td className="mono">{b.rows}</td>
                    <td className="mono">{fmt(b.mean_score)}</td>
                    <td className="mono">{fmt(b.fraction_positive)}</td>
                    <td
                      className="mono"
                      style={{ color: Math.abs(gap) > 0.1 ? "var(--state-warn)" : undefined }}
                    >
                      {gap >= 0 ? "+" : ""}
                      {gap.toFixed(3)}
                    </td>
                    <td className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
                      {raw &&
                      raw.rows > 0 &&
                      raw.mean_score !== null &&
                      raw.fraction_positive !== null
                        ? `${raw.mean_score.toFixed(2)} → ${raw.fraction_positive.toFixed(2)} (n ${raw.rows})`
                        : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}

      <Card title="Abstention">
        <p style={{ fontSize: "var(--text-sm)" }}>
          There is no single τ. Each label asserts when its calibrated probability reaches its own
          threshold (table above, chosen to maximise MCC for that label). When no label reaches its
          threshold the result says <em>no confident call</em> and shows the highest score only for
          transparency.
        </p>
      </Card>

      <Card
        title="Consistency rule set"
        aside={
          <span className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
            rules {t.provenance?.rules_version}
          </span>
        }
      >
        <table>
          <thead>
            <tr>
              <th>id</th>
              <th>severity</th>
              <th>rule</th>
              <th>status</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((r) => (
              <tr key={r.id}>
                <td className="mono">{r.id}</td>
                <td className="mono">{r.severity}</td>
                <td>{r.message}</td>
                <td className="mono muted">{r.active === false ? "inactive" : "active"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
          An inactive rule is not evaluated: R06 compares the model with the annotated location of
          the nearest hit, and Phase 13's hits carry GO terms but no location.
        </p>
      </Card>
      <ProvenanceBar p={t.provenance} />
    </>
  );
}
