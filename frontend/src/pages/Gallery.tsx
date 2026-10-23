/*
 * Gallery — the cached example sequences from GET /demo/examples. Only these (and the rest of
 * the 500-sequence residue cache) can be scored. Each card states that the result is in-sample,
 * because the heads were fitted on these sequences. "Open result" runs the prediction and
 * shows it in the Inspector. Below the cards, CatalogBrowser lists every cached sequence.
 */
import type { ReactElement } from "react";
import { CatalogBrowser } from "../components/CatalogBrowser";
import { Card, useExamples } from "../components/ui";

export default function Gallery({
  onOpen,
}: {
  onOpen: (e: { id: string; sequence: string; group: string }) => void;
}): ReactElement {
  const { examples, error } = useExamples();

  return (
    <div>
      <div className="head">
        <div>
          <h1>Example gallery</h1>
          <p className="sub">
            Real Swiss-Prot entries whose ESM-2 states are cached. The database annotation is shown
            beside each so you can compare it with the prediction. Every result here is in-sample:
            the heads were fitted on these sequences, so agreement is not evidence of accuracy.
          </p>
        </div>
      </div>

      {error && <p className="flag blocked">Could not load the gallery: {error}</p>}
      {!error && examples.length === 0 && <p className="sub">Loading examples…</p>}

      <div className="grid two">
        {examples.map((e) => {
          const labels = e.database_labels?.localization ?? [];
          const cleavage = e.database_labels?.signal_peptide_cleavage_site;
          return (
            <Card
              key={e.id}
              title={e.name}
              aside={
                e.in_training_set ? (
                  <span className="badge insample" title={e.note}>
                    in-sample
                  </span>
                ) : undefined
              }
            >
              <p className="sub" style={{ margin: 0 }}>
                {e.organism ?? "—"}
              </p>
              <p className="mono muted" style={{ fontSize: "var(--text-xs)" }}>
                {e.organism_group} · {String(e.length ?? e.sequence.length)} residues
                {e.partition !== undefined ? ` · partition ${String(e.partition)}` : ""}
              </p>
              <p style={{ fontSize: "var(--text-sm)" }}>
                Database location:{" "}
                <strong>{labels.length > 0 ? labels.join(", ").replaceAll("_", " ") : "—"}</strong>
                {typeof cleavage === "number" &&
                  ` · signal peptide, cleavage after ${String(cleavage)}`}
              </p>
              <button
                className="btn primary"
                onClick={() => {
                  onOpen({ id: e.id, sequence: e.sequence, group: e.organism_group });
                }}
              >
                Open result
              </button>
            </Card>
          );
        })}
      </div>

      <div style={{ marginTop: "var(--space-5)" }}>
        <CatalogBrowser onOpen={onOpen} />
      </div>
    </div>
  );
}
