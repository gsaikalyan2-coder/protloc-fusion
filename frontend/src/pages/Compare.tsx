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
import { Columns2 } from "lucide-react";
import {
  Card,
  EmptyState,
  ExamplePicker,
  GroupSelect,
  JobState,
  useExamples,
} from "../components/ui";

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
      <form className="row tight" onSubmit={load}>
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
        <div className="chosen">
          <span className="mono chosen-id">{choice.id}</span>
          <span className="note">{choice.sequence.length} residues</span>
          <GroupSelect
            value={choice.group}
            onChange={(g) => {
              onChoose({ ...choice, group: g });
            }}
            servedOnly
          />
        </div>
      ) : (
        <p className="chosen empty-line note">No sequence chosen.</p>
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
        <div className="table-wrap">
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
        </div>
      </Card>
      <Card title="Location probabilities (calibrated)">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>label</th>
                <th className="num">{a.id}</th>
                <th className="num">{b.id}</th>
                <th className="num">difference (B − A)</th>
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
        </div>
        <p className="note mt-3">
          ✓ marks a label above its own threshold. A difference is between two model outputs, not a
          measured difference between the proteins.
        </p>
      </Card>
      <Card title="GO molecular function (predicted terms)">
        <div className="go-compare">
          {(
            [
              ["Both", c.go.shared],
              [`Only ${a.id}`, c.go.onlyA],
              [`Only ${b.id}`, c.go.onlyB],
            ] as const
          ).map(([label, terms]) => (
            <div key={label}>
              <div className="overline">{label}</div>
              {terms.length === 0 ? (
                <p className="note">none</p>
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
    <>
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
      <div className="grid two job-pair">
        <JobState job={jobA.job} error={jobA.error} accepted={jobA.accepted} onRetry={jobA.retry} />
        <JobState job={jobB.job} error={jobB.error} accepted={jobB.accepted} onRetry={jobB.retry} />
      </div>
      {a && b && <Results a={a} b={b} />}
      {!a && !b && !busy && (
        <Card>
          <EmptyState icon={Columns2} title="Choose two sequences">
            Load each side by accession or from the examples, then press Compare. Both run as
            ordinary jobs; a repeat is answered from the result cache.
          </EmptyState>
        </Card>
      )}
    </>
  );
}
