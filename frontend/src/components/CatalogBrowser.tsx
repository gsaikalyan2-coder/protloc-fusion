/*
 * Every cached sequence, not just the gallery's eleven. The engine scores any of the 500
 * sequences in the residue cache (ADR-0016); this lists them from GET /demo/catalog with search,
 * filters and paging, and looks one up by accession through GET /demo/catalog/{accession}.
 * "Open result" hands the sequence to the Inspector exactly as a gallery card does. Each row
 * says whether the heads were fitted on it (partitions 1-4) or never saw it (partition 5).
 */
import { useEffect, useState, type FormEvent, type ReactElement } from "react";
import {
  getCatalog,
  getCatalogEntry,
  type CatalogEntry,
  type CatalogPage,
} from "../services/demoApi";
import { describeError } from "../services/problems";
import { Card } from "./ui";

export const PAGE_SIZE = 25;
const SEARCH_DELAY_MS = 250;

export interface OpenRequest {
  id: string;
  sequence: string;
  group: string;
}

const pretty = (s: string): string => s.replaceAll("_", " ");

function SplitBadge({ entry }: { entry: CatalogEntry }): ReactElement {
  return entry.in_training_set ? (
    <span className="badge insample" title="The heads were fitted on this sequence">
      in-sample
    </span>
  ) : (
    <span className="badge model" title="Held-out test partition: never seen during fitting">
      held-out
    </span>
  );
}

export function CatalogBrowser({ onOpen }: { onOpen: (e: OpenRequest) => void }): ReactElement {
  const [query, setQuery] = useState("");
  const [location, setLocation] = useState("");
  const [partition, setPartition] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<CatalogPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lookup, setLookup] = useState("");
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [opening, setOpening] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const timer = setTimeout(() => {
      getCatalog({
        q: query,
        location: location || undefined,
        partition: partition ? Number(partition) : undefined,
        limit: PAGE_SIZE,
        offset,
      })
        .then((p) => {
          if (!live) return;
          setPage(p);
          setError(null);
        })
        .catch((e: unknown) => {
          if (live) setError(describeError(e));
        });
    }, SEARCH_DELAY_MS);
    return (): void => {
      live = false;
      clearTimeout(timer);
    };
  }, [query, location, partition, offset]);

  const open = (key: string, onFail: (message: string) => void): void => {
    setOpening(key);
    getCatalogEntry(key)
      .then((e) => {
        onOpen({ id: e.id, sequence: e.sequence, group: e.organism_group });
      })
      .catch((e: unknown) => {
        onFail(describeError(e));
      })
      .finally(() => {
        setOpening(null);
      });
  };

  const submitLookup = (ev: FormEvent): void => {
    ev.preventDefault();
    if (!lookup.trim()) {
      setLookupError("Enter an accession or entry name first.");
      return;
    }
    setLookupError(null);
    open(lookup, setLookupError);
  };

  const total = page?.total ?? 0;
  const unavailable = page && !page.catalog.available ? page.catalog.reason : null;

  return (
    <Card
      title="All cached sequences"
      aside={
        <span className="muted" style={{ fontSize: "var(--text-xs)" }}>
          {page ? `${String(page.catalog.entries)} scorable` : "loading…"}
        </span>
      }
    >
      <form className="row" onSubmit={submitLookup} aria-label="Look up by accession">
        <label htmlFor="catalog-lookup">Accession or entry name</label>
        <input
          id="catalog-lookup"
          className="mono"
          placeholder="P12345 or ABC1_HUMAN"
          value={lookup}
          maxLength={64}
          onChange={(ev) => {
            setLookup(ev.target.value);
            setLookupError(null);
          }}
        />
        <button className="btn primary" type="submit" disabled={opening !== null}>
          Open result
        </button>
      </form>
      {lookupError && (
        <p className="flag blocked" role="alert">
          {lookupError}
        </p>
      )}

      <div className="row" style={{ marginTop: "var(--space-3)", flexWrap: "wrap" }}>
        <input
          aria-label="Search accession, entry name or organism"
          placeholder="Search accession, entry name, organism"
          value={query}
          maxLength={100}
          onChange={(ev) => {
            setQuery(ev.target.value);
            setOffset(0);
          }}
        />
        <select
          aria-label="Database location"
          value={location}
          onChange={(ev) => {
            setLocation(ev.target.value);
            setOffset(0);
          }}
        >
          <option value="">Any database location</option>
          {(page?.locations ?? []).map((l) => (
            <option key={l} value={l}>
              {pretty(l)}
            </option>
          ))}
        </select>
        <select
          aria-label="Partition"
          value={partition}
          onChange={(ev) => {
            setPartition(ev.target.value);
            setOffset(0);
          }}
        >
          <option value="">Any partition</option>
          {[1, 2, 3, 4].map((n) => (
            <option key={n} value={String(n)}>
              Partition {n} (training)
            </option>
          ))}
          <option value="5">Partition 5 (held-out test)</option>
        </select>
      </div>

      {error && <p className="flag blocked">Could not load the catalog: {error}</p>}
      {unavailable && <p className="flag warning">The catalog is not loaded: {unavailable}</p>}

      {page && !unavailable && (
        <>
          <p className="muted" style={{ fontSize: "var(--text-xs)" }}>
            {total === 0
              ? "No cached sequence matches."
              : `${String(offset + 1)}–${String(Math.min(offset + PAGE_SIZE, total))} of ${String(total)}`}
          </p>
          {page.entries.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>accession</th>
                  <th>entry</th>
                  <th>organism</th>
                  <th>len</th>
                  <th>database location</th>
                  <th>split</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {page.entries.map((e) => (
                  <tr key={e.id}>
                    <td className="mono">{e.id}</td>
                    <td className="mono">{e.entry_name}</td>
                    <td>{e.organism ?? "—"}</td>
                    <td className="num">{e.length}</td>
                    <td>
                      {e.database_labels.localization.length > 0
                        ? e.database_labels.localization.map(pretty).join(", ")
                        : "—"}
                    </td>
                    <td>
                      <SplitBadge entry={e} /> <span className="muted">p{e.partition}</span>
                    </td>
                    <td>
                      <button
                        className="btn"
                        disabled={opening !== null}
                        aria-label={`Open result for ${e.id}`}
                        onClick={() => {
                          open(e.id, setError);
                        }}
                      >
                        {opening === e.id ? "Opening…" : "Open"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {total > PAGE_SIZE && (
            <div className="row" style={{ marginTop: "var(--space-3)" }}>
              <button
                className="btn"
                disabled={offset === 0}
                onClick={() => {
                  setOffset(Math.max(0, offset - PAGE_SIZE));
                }}
              >
                Previous
              </button>
              <button
                className="btn"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => {
                  setOffset(offset + PAGE_SIZE);
                }}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
