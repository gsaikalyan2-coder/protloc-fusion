/*
 * Compare — two cached sequences side by side (feature 8). Each side is chosen by accession or
 * entry name (GET /demo/catalog/{key}) or from the gallery, runs as its own job (the result
 * cache answers a repeat at once), and the two results are laid out row by row with the
 * per-label probability difference. The page compares two model outputs; it does not claim the
 * proteins differ biologically by those amounts.
 */
import { useState, type FormEvent, type ReactElement } from "react";
import { getCatalogEntry, predict, useJob, type Result } from "../services/demoApi";
import { compareResults } from "../services/compare";
import { describeError } from "../services/problems";
import { Card, ExamplePicker, GroupSelect, JobState, useExamples } from "../components/ui";

interface Choice {
  id: string;
  sequence: string;
  group: string;
}

function Side({
  title,
  choice,
  onChoose,
}: {
  title: string;
  choice: Choice | null;
  onChoose: (c: Choice | null) => void;
}): ReactElement {
  const { examples } = useExamples();
  const [key, setKey] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = (ev: FormEvent): void => {
    ev.preventDefault();
    if (!key.trim()) {
      setProblem("Enter an accession or entry name first.");
      return;
    }
    setProblem(null);
    setLoading(true);
    getCatalogEntry(key)
      .then((e) => {
        onChoose({ id: e.id, sequence: e.sequence, group: e.organism_group });
      })
      .catch((e: unknown) => {
        setProblem(describeError(e));
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <Card title={title}>
      <form className="row" onSubmit={load} style={{ flexWrap: "wrap" }}>
        <input
          aria-label={`${title}: accession or entry name`}
          className="mono"
          placeholder="P12345 or ABC1_HUMAN"
          maxLength={64}
          value={key}
          onChange={(ev) => {
            setKey(ev.target.value);
            setProblem(null);
          }}
        />
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Loading…" : "Load"}
        </button>
        <ExamplePicker
          examples={examples}
          value=""
          onPick={(e) => {
            setProblem(null);
            onChoose({ id: e.id, sequence: e.sequence, group: e.organism_group });
          }}
        />
      </form>
      {problem && (
        <p className="flag blocked" role="alert">
          {problem}
        </p>
      )}
      {choice ? (
        <div className="row" style={{ marginTop: "var(--space-3)" }}>
          <span className="mono">{choice.id}</span>
          <span className="muted">{choice.sequence.length} residues</span>
          <GroupSelect
            value={choice.group}
            onChange={(g) => {
              onChoose({ ...choice, group: g });
            }}
            servedOnly
          />
        </div>
      ) : (
        <p className="sub">No sequence chosen.</p>
      )}
    </Card>
  );
}

const fmt = (p: number | null): string => (p === null ? "—" : p.toFixed(3));

function Delta({ value }: { value: number | null }): ReactElement {
  if (value === null) return <span className="muted">—</span>;
  const sign = value > 0 ? "+" : "";
  return <span className="mono">{`${sign}${value.toFixed(3)}`}</span>;
}

function Results({ a, b }: { a: Result; b: Result }): ReactElement {
  const c = compareResults(a, b);
  return (
    <>
      {c.sameSequence && (
        <p className="flag warning">Both sides are the same sequence; only the group differs.</p>
      )}
      <Card title="Side by side">
        <table>
          <thead>
            <tr>
              <th />
              <th>{a.id}</th>
              <th>{b.id}</th>
            </tr>
          </thead>
          <tbody>
            {c.summary.map((r) => (
              <tr key={r.label} className={r.differs ? "differs" : undefined}>
                <td>{r.label}</td>
                <td>{r.a}</td>
                <td>{r.b}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Card title="Location probabilities (calibrated)">
        <table>
          <thead>
            <tr>
              <th>label</th>
              <th>{a.id}</th>
              <th>{b.id}</th>
              <th>difference (B − A)</th>
            </tr>
          </thead>
          <tbody>
            {c.locations.map((l) => (
              <tr key={l.code}>
                <td>
                  {l.name} <span className="muted mono">{l.code}</span>
                </td>
                <td className="num">
                  {fmt(l.a)}
                  {l.aPredicted ? " ✓" : ""}
                </td>
                <td className="num">
                  {fmt(l.b)}
                  {l.bPredicted ? " ✓" : ""}
                </td>
                <td className="num">
                  <Delta value={l.delta} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
          ✓ marks a label above its own threshold. A difference is between two model outputs, not a
          measured difference between the proteins.
        </p>
      </Card>
      <Card title="GO molecular function (predicted terms)">
        <div className="grid two">
          {(
            [
              ["Both", c.go.shared],
              [`Only ${a.id}`, c.go.onlyA],
              [`Only ${b.id}`, c.go.onlyB],
            ] as const
          ).map(([label, terms]) => (
            <div key={label}>
              <strong>{label}</strong>
              {terms.length === 0 ? (
                <p className="sub">none</p>
              ) : (
                <ul className="plain">
                  {terms.map((t) => (
                    <li key={t.go_id}>
                      <span className="mono">{t.go_id}</span> {t.name}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}

export default function Compare(): ReactElement {
  const [left, setLeft] = useState<Choice | null>(null);
  const [right, setRight] = useState<Choice | null>(null);
  const jobA = useJob();
  const jobB = useJob();
  const busy = jobA.busy || jobB.busy;
  const a = jobA.done?.results?.[0] ?? null;
  const b = jobB.done?.results?.[0] ?? null;

  const compare = (): void => {
    if (!left || !right) return;
    void jobA.run(...predict(left.sequence, left.group, left.id));
    void jobB.run(...predict(right.sequence, right.group, right.id));
  };

  return (
    <div>
      <div className="head">
        <div>
          <h1>Compare</h1>
          <p className="sub">
            Two cached sequences side by side: location calls and calibrated probabilities, sorting
            signal, membrane type, GO terms, consistency flags and the nearest homolog.
          </p>
        </div>
        <button className="btn primary" onClick={compare} disabled={busy || !left || !right}>
          {busy ? "Running…" : "Compare"}
        </button>
      </div>
      <div className="grid two">
        <Side title="Sequence A" choice={left} onChoose={setLeft} />
        <Side title="Sequence B" choice={right} onChoose={setRight} />
      </div>
      <div className="grid two">
        <JobState job={jobA.job} error={jobA.error} accepted={jobA.accepted} onRetry={jobA.retry} />
        <JobState job={jobB.job} error={jobB.error} accepted={jobB.accepted} onRetry={jobB.retry} />
      </div>
      {a && b && <Results a={a} b={b} />}
    </div>
  );
}
