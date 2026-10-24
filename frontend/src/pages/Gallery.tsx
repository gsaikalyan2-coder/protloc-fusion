/*
 * Gallery — the cached example sequences from GET /demo/examples. Only these (and the rest of
 * the 500-sequence residue cache) can be scored. Each card states that the result is in-sample,
 * because the heads were fitted on these sequences. "Open result" runs the prediction and
 * shows it in the Inspector. Below the cards, CatalogBrowser lists every cached sequence.
 */
import type { ReactElement } from "react";
import { CatalogBrowser } from "../components/CatalogBrowser";
import { ArrowRight } from "lucide-react";
import { Skeleton, useExamples } from "../components/ui";

export default function Gallery({
  onOpen,
}: {
  onOpen: (e: { id: string; sequence: string; group: string }) => void;
}): ReactElement {
  const { examples, error } = useExamples();

  return (
    <>
      <div className="head">
        <div>
          <h1>Catalog</h1>
          <p className="sub">
            Real Swiss-Prot entries whose ESM-2 states are cached. The database annotation is shown
            beside each so you can compare it with the prediction. Examples and partitions 1–4 are
            in-sample: the heads were fitted on them, so agreement there is not evidence of
            accuracy. Partition 5 is the held-out test set.
          </p>
        </div>
      </div>

      <section className="section first" aria-labelledby="examples-title">
        <div className="section-head">
          <h2 id="examples-title">Examples</h2>
          <span className="note">
            {examples.length > 0 ? `${String(examples.length)} sequences` : ""}
          </span>
        </div>
        {error && <p className="flag blocked">Could not load the examples: {error}</p>}
        {!error && examples.length === 0 && <Skeleton lines={3} label="Loading examples" />}
        <ul className="example-grid">
          {examples.map((e) => {
            const labels = e.database_labels?.localization ?? [];
            const cleavage = e.database_labels?.signal_peptide_cleavage_site;
            return (
              <li key={e.id}>
                <button
                  type="button"
                  className="example"
                  onClick={() => {
                    onOpen({ id: e.id, sequence: e.sequence, group: e.organism_group });
                  }}
                >
                  <span className="example-top">
                    <span className="mono">{e.id}</span>
                    {e.in_training_set && <span className="badge insample">in-sample</span>}
                  </span>
                  <strong>{e.name}</strong>
                  <span className="note">
                    {e.organism ?? e.organism_group} · {String(e.length ?? e.sequence.length)}{" "}
                    residues
                    {e.partition !== undefined ? ` · partition ${String(e.partition)}` : ""}
                  </span>
                  <span className="hint">
                    Database:{" "}
                    {labels.length > 0 ? labels.join(", ").replaceAll("_", " ") : "no location"}
                    {typeof cleavage === "number" && ` · SP cleaves after ${String(cleavage)}`}
                  </span>
                  <span className="example-go">
                    Open result <ArrowRight size={14} aria-hidden="true" />
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      <CatalogBrowser onOpen={onOpen} />
    </>
  );
}
