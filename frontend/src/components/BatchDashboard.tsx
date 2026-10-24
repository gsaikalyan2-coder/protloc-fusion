/*
 * Batch dashboard (feature 9): distributions over one finished batch, drawn as plain CSS bars
 * (React only: no chart library, docs/DESIGN.md tokens only). Location bars use each
 * compartment's own colour token, as everywhere else in the app. Every bar states its count in
 * text, so nothing depends on reading a length or a colour.
 */
import type { ReactElement } from "react";
import { batchStats, type Bar } from "../services/batchStats";
import type { Result } from "../services/demoApi";
import { Card, locText, locVar } from "./ui";

function Bars({
  title,
  bars,
  total,
  colour,
  empty = "none",
}: {
  title: string;
  bars: readonly Bar[];
  total: number;
  colour?: (key: string) => { fill: string; ink: string };
  empty?: string;
}): ReactElement {
  const max = Math.max(1, ...bars.map((b) => b.count));
  return (
    <figure className="bars" aria-label={title}>
      <figcaption>{title}</figcaption>
      {bars.length === 0 ? (
        <p className="sub">{empty}</p>
      ) : (
        <ul>
          {bars.map((b) => {
            const c = colour?.(b.key);
            const share = total > 0 ? Math.round((b.count / total) * 100) : 0;
            return (
              <li key={b.key} title={`${b.label}: ${String(b.count)} of ${String(total)}`}>
                <span className="bar-label">{b.label}</span>
                <span className="bar-track">
                  <span
                    className="bar-fill"
                    style={{
                      width: `${String((b.count / max) * 100)}%`,
                      ...(c ? { background: c.fill, borderColor: c.ink } : {}),
                    }}
                  />
                </span>
                <span className="bar-count mono">
                  {b.count} <span className="muted">({share}%)</span>
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </figure>
  );
}

export function BatchDashboard({
  results,
  refused = 0,
}: {
  results: readonly Result[];
  refused?: number;
}): ReactElement {
  const s = batchStats(results, refused);
  const n = s.scored;
  return (
    <Card
      title="Batch dashboard"
      aside={
        <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
          {n} scored{s.refused ? ` · ${String(s.refused)} refused` : ""}
          {s.cacheHits ? ` · ${String(s.cacheHits)} from cache` : ""}
        </span>
      }
    >
      <div className="grid two">
        <Bars
          title="Predicted locations (a protein can count twice)"
          bars={s.locations}
          total={n}
          colour={(code) => ({ fill: locVar(code), ink: locText(code) })}
          empty="No location cleared its threshold."
        />
        <Bars
          title={`Top calibrated probability${s.noConfidentCall ? ` · ${String(s.noConfidentCall)} no confident call` : ""}`}
          bars={s.confidence}
          total={n}
        />
        <Bars title="Sorting signal" bars={s.signal} total={n} />
        <Bars title="Homology evidence" bars={s.evidence} total={n} />
      </div>
      <Bars
        title="Most frequent predicted GO terms"
        bars={s.goTerms}
        total={n}
        empty="No GO term cleared its threshold."
      />
      <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
        Counts of model outputs over this batch only. Probabilities are calibrated per label on the
        500-sequence subset; a distribution here is not a prevalence estimate.
      </p>
    </Card>
  );
}
