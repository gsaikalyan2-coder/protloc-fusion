/*
 * SCREEN 5 — Trust & Abstention Console  (NOVEL)
 * The thing reviewers ask for and tools never show: what the confidence numbers
 * are worth. GET /demo/trust returns the live fit — temperature fitted on the
 * calibration split, tau chosen there, reliability and coverage/accuracy
 * measured on the untouched test split (PROJECT_PLAN.md rule: freeze before you
 * measure). Moving the tau slider re-reads the measured curve; it never refits.
 */
import { useEffect, useState } from "react";
import { getTrust, getVersion } from "../services/demoApi";
import { Card, ProvenanceBar } from "../components/ui";

interface Bin {
  bin: number;
  lo: number;
  hi: number;
  count: number;
  confidence: number | null;
  accuracy: number | null;
}

export default function Trust() {
  const [t, setT] = useState<Record<string, any> | null>(null);
  const [rules, setRules] = useState<{ id: string; severity: string; message: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tau, setTau] = useState<number | null>(null);

  useEffect(() => {
    getTrust()
      .then((b) => {
        setT(b);
        setTau(b.tau);
      })
      .catch((e) => {
        setError(String(e));
      });
    getVersion()
      .then((v) => {
        setRules(v.rules);
      })
      .catch(() => undefined);
  }, []);

  if (error) return <p className="err">Backend error: {error}</p>;
  if (!t || tau === null) return <p className="muted">Loading calibration…</p>;

  const curve = t.coverage_curve as { tau: number; coverage: number; selective_accuracy: number }[];
  const first = curve[0];
  if (!first) return <p className="muted">No coverage curve.</p>;
  const at = curve.reduce((a, b) => (Math.abs(b.tau - tau) < Math.abs(a.tau - tau) ? b : a), first);
  const bins = (t.reliability_after as Bin[]).filter((b) => b.count > 0);
  const binsBefore = (t.reliability_before as Bin[]).filter((b) => b.count > 0);

  return (
    <>
      <div className="head">
        <div>
          <h1>Trust &amp; Abstention Console</h1>
          <p className="sub">
            What a probability from this system is worth. Temperature and τ were fitted on the
            calibration split ({t.n_calibration} sequences); everything below is measured on the
            held-out test split ({t.n_test}).
          </p>
        </div>
      </div>

      <div className="grid kpi">
        {(
          [
            ["temperature", t.temperature],
            ["frozen τ", t.tau],
            ["ECE before", t.ece_before],
            ["ECE after", t.ece_after],
          ] as const
        ).map(([l, v]) => (
          <div className="card" key={l} style={{ marginBottom: 0 }}>
            <div className="kpi-v">{v}</div>
            <div className="kpi-l">{l}</div>
          </div>
        ))}
      </div>

      <Card
        title="Reliability — predicted vs. observed"
        aside={
          <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
            after calibration; grey = before
          </span>
        }
      >
        <table>
          <thead>
            <tr>
              <th>confidence bin</th>
              <th>n</th>
              <th>mean confidence</th>
              <th>observed accuracy</th>
              <th>gap</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {bins.map((b) => {
              const before = binsBefore.find((x) => x.bin === b.bin);
              const gap = (b.accuracy ?? 0) - (b.confidence ?? 0);
              return (
                <tr key={b.bin}>
                  <td className="mono">
                    {b.lo.toFixed(1)}–{b.hi.toFixed(1)}
                  </td>
                  <td className="mono">{b.count}</td>
                  <td className="mono">{b.confidence?.toFixed(3)}</td>
                  <td className="mono">{b.accuracy?.toFixed(3)}</td>
                  <td
                    className="mono"
                    style={{ color: Math.abs(gap) > 0.1 ? "var(--state-warn)" : undefined }}
                  >
                    {gap >= 0 ? "+" : ""}
                    {gap.toFixed(3)}
                  </td>
                  <td>
                    <span className="track" style={{ display: "block", width: 180 }}>
                      <span
                        className="fill"
                        style={{
                          width: `${(b.accuracy ?? 0) * 100}%`,
                          background: "var(--accent)",
                          display: "block",
                        }}
                      />
                    </span>
                    {before && (
                      <span className="muted mono" style={{ fontSize: "var(--text-xs)" }}>
                        uncal. {before.accuracy?.toFixed(2)} @ {before.confidence?.toFixed(2)}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <Card
        title="Abstention trade-off"
        aside={
          <span className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
            {t.tau_rule}
          </span>
        }
      >
        <div className="row">
          <label className="muted" style={{ fontSize: "var(--text-sm)" }}>
            explore τ
            <input
              type="range"
              min={first.tau}
              max={(curve[curve.length - 1] ?? first).tau}
              step={0.03}
              value={tau}
              onChange={(e) => {
                setTau(Number(e.target.value));
              }}
              style={{ marginLeft: 8, width: 260 }}
            />
          </label>
          <span className="mono">τ = {at.tau.toFixed(2)}</span>
          <span className="mono">coverage {(at.coverage * 100).toFixed(1)}%</span>
          <span className="mono">
            selective accuracy {(at.selective_accuracy * 100).toFixed(1)}%
          </span>
          {Math.abs(at.tau - t.tau) > 0.001 && (
            <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
              exploratory — the served τ stays {t.tau}
            </span>
          )}
        </div>
        <table style={{ marginTop: "var(--space-3)" }}>
          <thead>
            <tr>
              <th>τ</th>
              <th>coverage</th>
              <th>selective accuracy</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {curve.map((c) => (
              <tr key={c.tau} className={Math.abs(c.tau - at.tau) < 0.001 ? "on" : ""}>
                <td className="mono">{c.tau.toFixed(2)}</td>
                <td className="mono">{(c.coverage * 100).toFixed(1)}%</td>
                <td className="mono">{(c.selective_accuracy * 100).toFixed(1)}%</td>
                <td>
                  <span className="track" style={{ display: "block", width: 220 }}>
                    <span
                      className="fill"
                      style={{
                        width: `${c.coverage * 100}%`,
                        background: "var(--border-stronger)",
                        display: "block",
                      }}
                    />
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
          Benchmark carries {(t.label_noise * 100).toFixed(0)}% injected label noise, so selective
          accuracy cannot reach 1.0 — by design.
        </p>
      </Card>

      <Card
        title="Consistency rule set"
        aside={
          <span className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
            rules {t.provenance.rules_version}
          </span>
        }
      >
        <table>
          <thead>
            <tr>
              <th>id</th>
              <th>severity</th>
              <th>rule</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((r) => (
              <tr key={r.id}>
                <td className="mono">{r.id}</td>
                <td className="mono">{r.severity}</td>
                <td>{r.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <ProvenanceBar p={t.provenance} />
    </>
  );
}
