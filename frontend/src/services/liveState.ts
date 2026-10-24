/* The state of a remembered job, from GET /demo/jobs/{id}: its status, or why it is not there. */
import { DemoApiError, problemOf } from "./problems";

export type LiveState =
  | "checking"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "expired"
  | "gone"
  | "unreachable";

export const STATE_TEXT: Readonly<Record<LiveState, string>> = {
  checking: "checking…",
  queued: "queued",
  running: "running",
  completed: "completed",
  failed: "failed",
  cancelled: "cancelled",
  expired: "expired on the server",
  gone: "not on the server (restarted)",
  unreachable: "server unreachable",
};

/** The live state for a job-status response or error. */
export function liveStateOf(outcome: unknown): LiveState {
  if (outcome instanceof DemoApiError) {
    if (outcome.status === 410) return "expired";
    if (outcome.status === 404) return "gone";
    return "unreachable";
  }
  if (typeof outcome === "object" && outcome !== null && "status" in outcome) {
    const s = String((outcome as { status: unknown }).status);
    if (s in STATE_TEXT) return s as LiveState;
  }
  return problemOf(outcome).kind === "unreachable" ? "unreachable" : "gone";
}
