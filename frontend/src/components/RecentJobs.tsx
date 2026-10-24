/*
 * Recent jobs (feature 7): the jobs this browser submitted, newest first, from localStorage.
 * Each row asks the server for the job's current state once: completed, failed, still running,
 * expired (410, past the retention window) or gone (404: the queue is in memory, so a server
 * restart forgets every job, ADR-0008). A predict or scan entry can be reopened; it re-runs the
 * sequence, which the result cache answers at once when it holds it. A finished predict or batch
 * job can be downloaded whole while the server still has it.
 */
import { useEffect, useState, type ReactElement } from "react";
import { getJob } from "../services/demoApi";
import { STATE_TEXT, liveStateOf, type LiveState } from "../services/liveState";
import {
  HISTORY_EVENT,
  clearHistory,
  loadHistory,
  removeHistory,
  type HistoryEntry,
} from "../services/history";
import { ServerDownloads } from "./ServerDownloads";
import { History } from "lucide-react";
import { Card, EmptyState } from "./ui";

function when(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
}

export function RecentJobs({
  onOpen,
  limit = 10,
}: {
  onOpen: (e: { id: string; sequence: string; group: string; kind: HistoryEntry["kind"] }) => void;
  limit?: number;
}): ReactElement {
  const [entries, setEntries] = useState<HistoryEntry[]>(() => loadHistory());
  const [live, setLive] = useState<Record<string, LiveState>>({});

  useEffect(() => {
    const reload = (): void => {
      setEntries(loadHistory());
    };
    window.addEventListener(HISTORY_EVENT, reload);
    return (): void => {
      window.removeEventListener(HISTORY_EVENT, reload);
    };
  }, []);

  const shown = entries.slice(0, limit);
  const ids = shown.map((e) => e.job_id).join(",");

  useEffect(() => {
    let alive = true;
    for (const id of ids ? ids.split(",") : []) {
      setLive((m) => ({ ...m, [id]: "checking" }));
      getJob(id)
        .then((b) => {
          if (alive) setLive((m) => ({ ...m, [id]: liveStateOf(b) }));
        })
        .catch((e: unknown) => {
          if (alive) setLive((m) => ({ ...m, [id]: liveStateOf(e) }));
        });
    }
    return (): void => {
      alive = false;
    };
  }, [ids]);

  return (
    <Card
      title="Recent jobs"
      aside={
        shown.length > 0 ? (
          <button
            className="btn ghost sm"
            onClick={() => {
              clearHistory();
            }}
          >
            Clear
          </button>
        ) : undefined
      }
    >
      {shown.length === 0 ? (
        <EmptyState icon={History} title="No jobs yet">
          Jobs you run here are listed in this browser only; nothing is stored on a server beyond
          the result&apos;s retention window.
        </EmptyState>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>when</th>
                <th>kind</th>
                <th>what</th>
                <th>group</th>
                <th>state</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {shown.map((e) => {
                const state = live[e.job_id] ?? "checking";
                const reopen = e.kind !== "batch" && e.sequence;
                const downloadable = state === "completed" && e.kind !== "scan";
                return (
                  <tr key={e.job_id}>
                    <td className="mono">{when(e.submitted_at)}</td>
                    <td>{e.kind}</td>
                    <td className="mono" title={e.job_id}>
                      {e.label}
                      {e.kind === "batch" ? ` (${String(e.sequence_count)})` : ""}
                    </td>
                    <td>{e.organism_group}</td>
                    <td>
                      <span className="state" data-state={state}>
                        {STATE_TEXT[state]}
                      </span>
                    </td>
                    <td>
                      <div className="row tight cell-action">
                        {reopen && (
                          <button
                            className="btn sm"
                            aria-label={`Open ${e.label} again`}
                            onClick={() => {
                              onOpen({
                                id: e.label,
                                sequence: e.sequence ?? "",
                                group: e.organism_group,
                                kind: e.kind,
                              });
                            }}
                          >
                            Open
                          </button>
                        )}
                        {downloadable && (
                          <ServerDownloads jobId={e.job_id} formats={["csv", "json"]} />
                        )}
                        <button
                          className="btn ghost sm"
                          aria-label={`Forget ${e.label}`}
                          onClick={() => {
                            removeHistory(e.job_id);
                          }}
                        >
                          Forget
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="note">
        Kept in this browser's local storage, newest first (at most 20). Clearing it or using a
        private window removes it. Anyone who has a job link can read its result until it expires.
      </p>
    </Card>
  );
}
