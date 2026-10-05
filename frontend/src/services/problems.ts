/*
 * Every way a request can go wrong, turned into text a person can act on. The server does not
 * send machine-readable error codes yet (Phase 17), so the two 422 refusals are told apart by
 * the wording of the server's own message; that matching lives here and nowhere else.
 */
import type { JobBody } from "./demoApi";

export type ProblemKind =
  | "expired"
  | "unknown_job"
  | "not_cached"
  | "group_not_served"
  | "invalid_input"
  | "unreachable"
  | "job_failed"
  | "server_error";

export interface Problem {
  kind: ProblemKind;
  title: string;
  message: string;
  hint: string;
}

/** FastAPI sends detail as a string, or as a list of {loc, msg} for a validation failure. */
export function detailText(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts: string[] = [];
    for (const item of detail as unknown[]) {
      if (typeof item !== "object" || item === null) continue;
      const { loc, msg } = item as { loc?: unknown; msg?: unknown };
      if (typeof msg !== "string") continue;
      const where = Array.isArray(loc)
        ? (loc as unknown[])
            .filter((p) => p !== "body")
            .map((p) => String(p))
            .join(".")
        : "";
      parts.push(where ? `${where}: ${msg}` : msg);
    }
    return parts.join("; ");
  }
  return "";
}

const START_HINT =
  "Start it from backend/ with: uvicorn protloc.api.app_v2:app --port 8010. It refuses to start, and says why, if the model registry does not verify.";

export function toProblem(status: number, detail: unknown): Problem {
  const text = detailText(detail);
  if (status === 0) {
    return {
      kind: "unreachable",
      title: "Cannot reach the backend",
      message: text || "The request got no answer.",
      hint: START_HINT,
    };
  }
  if (status === 410) {
    return {
      kind: "expired",
      title: "This result has expired",
      message:
        "Results are kept for a short time and then removed; an expired job is never shown as an empty result.",
      hint: "Run the prediction again. Files you downloaded earlier are unaffected.",
    };
  }
  if (status === 404) {
    return {
      kind: "unknown_job",
      title: "The server does not know this job",
      message:
        "The job id is unknown. It may have expired and been cleaned up, or the server restarted (jobs are held in memory).",
      hint: "Run the prediction again.",
    };
  }
  if (status === 422 && text.includes("ESM-2 states are cached")) {
    return {
      kind: "not_cached",
      title: "Not one of the sequences this build can score",
      message: text,
      hint: "Open one from the Gallery. Anything else is refused rather than guessed.",
    };
  }
  if (status === 422 && text.includes("No trained head covers")) {
    return {
      kind: "group_not_served",
      title: "That organism group is not served",
      message: text,
      hint: "Choose Animal / fungi or Plant.",
    };
  }
  if (status === 422 || status === 400) {
    return {
      kind: "invalid_input",
      title: "The input was rejected",
      message: text || "The server could not use this input.",
      hint: "Check the sequence (standard amino-acid letters) and the organism group, then try again.",
    };
  }
  return {
    kind: "server_error",
    title: "The server hit an error",
    message: text || `The server answered with status ${String(status)}.`,
    hint: "Try again. If it repeats, report the model version and job id shown on the page.",
  };
}

/** A finished-but-unsuccessful job, as a Problem. Completed and in-flight jobs return null. */
export function jobProblem(job: JobBody): Problem | null {
  if (job.status !== "failed") return null;
  const text = job.error ?? "";
  if (text.includes("ESM-2 states are cached")) return toProblem(422, text);
  return {
    kind: "job_failed",
    title: "The job failed",
    message: text || "The job ended without a result and gave no reason.",
    hint: "Run it again. If it repeats, report the model version and job id shown on the page.",
  };
}

export class DemoApiError extends Error {
  readonly status: number;
  readonly problem: Problem;

  constructor(status: number, detail: unknown) {
    const problem = toProblem(status, detail);
    super(problem.message);
    this.name = "DemoApiError";
    this.status = status;
    this.problem = problem;
  }
}

/** Any thrown value as a Problem, so no screen ever prints a stack trace or "[object Object]". */
export function problemOf(e: unknown): Problem {
  if (e instanceof DemoApiError) return e.problem;
  return {
    kind: "server_error",
    title: "Something went wrong",
    message: e instanceof Error ? e.message : String(e),
    hint: "Reload the page and try again.",
  };
}

/** One line, for places that show a string (Trust, the gallery). */
export function describeError(e: unknown): string {
  const p = problemOf(e);
  return `${p.title}. ${p.message}`;
}
