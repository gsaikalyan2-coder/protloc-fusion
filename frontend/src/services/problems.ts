/*
 * Every way a request can go wrong, turned into text a person can act on. Since Phase 17 the
 * server sends a stable machine-readable `error.code` with every failure, and that code (never
 * the wording of the message) picks the title, the hint and whether a retry can help. A
 * response without a code (an older server, a proxy error, a dead backend) falls back to the
 * HTTP status. The text for each code lives here and nowhere else.
 */
import type { JobBody } from "./demoApi";

export type ProblemKind =
  | "expired"
  | "unknown_job"
  | "not_cached"
  | "group_not_served"
  | "invalid_input"
  | "too_large"
  | "fasta_error"
  | "no_structure"
  | "disabled"
  | "evidence_degraded"
  | "unreachable"
  | "job_failed"
  | "server_error";

export interface Problem {
  kind: ProblemKind;
  /** The server's stable error code, or null when the failure did not come from a code. */
  code: string | null;
  title: string;
  message: string;
  hint: string;
  /** True only where running the same request again can change the outcome. */
  retryable: boolean;
}

/** The structured `error` member of a failed response or job (backend domain/errors.py). */
export interface ApiErrorInfo {
  code: string;
  message?: string;
  detail?: string;
  retryable?: boolean;
  line?: number | null;
}

interface CodeText {
  kind: ProblemKind;
  title: string;
  hint: string;
  /** Used instead of the server's detail when the server's wording adds nothing. */
  message?: string;
}

const FASTA_HINT =
  "Fix the line named above and submit again. Each record starts with '>' and holds letters only.";
const CHUNK_HINT =
  "Split the input into smaller pieces and submit them one at a time (batches are chunked on the client, ADR-0008).";

export const CODE_TEXT: Readonly<Record<string, CodeText>> = {
  invalid_input: {
    kind: "invalid_input",
    title: "The input was rejected",
    hint: "Check the sequence (standard amino-acid letters) and the organism group, then try again.",
  },
  request_too_large: { kind: "too_large", title: "The request is too large", hint: CHUNK_HINT },
  sequence_too_long: {
    kind: "too_large",
    title: "The sequence is too long",
    hint: "Use a sequence within the limit stated in the message.",
  },
  batch_too_large: {
    kind: "too_large",
    title: "Too many sequences for one batch",
    hint: CHUNK_HINT,
  },
  fasta_empty: { kind: "fasta_error", title: "No FASTA records found", hint: FASTA_HINT },
  fasta_no_header: { kind: "fasta_error", title: "The FASTA has no header", hint: FASTA_HINT },
  fasta_empty_record: {
    kind: "fasta_error",
    title: "A FASTA record has no residues",
    hint: FASTA_HINT,
  },
  fasta_invalid_character: {
    kind: "fasta_error",
    title: "The FASTA holds a character that is not a letter",
    hint: FASTA_HINT,
  },
  unknown_group: {
    kind: "invalid_input",
    title: "Unknown organism group",
    hint: "Choose Animal / fungi or Plant.",
  },
  group_not_served: {
    kind: "group_not_served",
    title: "That organism group is not served",
    hint: "Choose Animal / fungi or Plant.",
  },
  not_cached: {
    kind: "not_cached",
    title: "Not one of the sequences this build can score",
    hint: "Open one from the Gallery. Anything else is refused rather than guessed.",
  },
  invalid_sequence: {
    kind: "invalid_input",
    title: "The sequence holds characters the model cannot use",
    hint: "Use the standard amino-acid letters only.",
  },
  unknown_job: {
    kind: "unknown_job",
    title: "The server does not know this job",
    message:
      "The job id is unknown. It may have expired and been cleaned up, or the server restarted (jobs are held in memory).",
    hint: "Run the prediction again.",
  },
  result_expired: {
    kind: "expired",
    title: "This result has expired",
    message:
      "Results are kept for a short time and then removed; an expired job is never shown as an empty result.",
    hint: "Run the prediction again. Files you downloaded earlier are unaffected.",
  },
  structure_bad_hash: {
    kind: "invalid_input",
    title: "The structure key is not valid",
    hint: "Run the prediction again to get a fresh key.",
  },
  accession_not_cached: {
    kind: "not_cached",
    title: "No cached sequence has that accession",
    hint: "Search the list below by accession, entry name or organism; only cached sequences can be scored.",
  },
  catalog_unavailable: {
    kind: "server_error",
    title: "The sequence catalog is not loaded",
    hint: "Run python scripts/build_catalog.py --write, then restart the backend. The gallery still works.",
  },
  structure_not_found: {
    kind: "no_structure",
    title: "No structure for this sequence",
    hint: "The prediction is unaffected; the 3D view is orientation only.",
  },
  structure_refused: {
    kind: "server_error",
    title: "The stored structure failed its integrity check",
    hint: "Try again. If it repeats, report the sequence and the model version.",
  },
  demo_disabled: {
    kind: "disabled",
    title: "The service is switched off",
    hint: "The operator has set PROTLOC_DEMO_ENABLED=0. Try again once it is switched back on.",
  },
  model_error: {
    kind: "server_error",
    title: "The model could not score this sequence",
    hint: "Report the model version and the sequence id shown on the page.",
  },
  job_failed: {
    kind: "job_failed",
    title: "The job failed",
    hint: "Run it again. If it repeats, report the model version and job id shown on the page.",
  },
};

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

/** The first non-empty string, else the fallback (an empty string is not a message). */
function firstText(options: readonly (string | undefined)[], fallback: string): string {
  return options.find((o) => o !== undefined && o !== "") ?? fallback;
}

function fromCode(info: ApiErrorInfo, text: string): Problem | null {
  const entry = CODE_TEXT[info.code];
  if (!entry) return null;
  return {
    kind: entry.kind,
    code: info.code,
    title: entry.title,
    message: entry.message ?? firstText([text, info.detail, info.message], entry.title),
    hint: entry.hint,
    retryable: info.retryable === true,
  };
}

export function toProblem(status: number, detail: unknown, info?: ApiErrorInfo | null): Problem {
  const text = detailText(detail);
  if (info) {
    const coded = fromCode(info, text);
    if (coded) return coded;
  }
  const base = { code: null, retryable: false } as const;
  if (status === 0) {
    return {
      ...base,
      kind: "unreachable",
      title: "Cannot reach the backend",
      message: text || "The request got no answer.",
      hint: START_HINT,
      retryable: true,
    };
  }
  if (status === 410) return { ...base, ...legacy("result_expired") };
  if (status === 404) return { ...base, ...legacy("unknown_job") };
  if (status === 422 && text.includes("ESM-2 states are cached")) {
    return { ...base, ...legacy("not_cached"), message: text };
  }
  if (status === 422 && text.includes("No trained head covers")) {
    return { ...base, ...legacy("group_not_served"), message: text };
  }
  if (status === 422 || status === 400) {
    return {
      ...base,
      ...legacy("invalid_input"),
      message: text || "The server could not use this input.",
    };
  }
  return {
    ...base,
    kind: "server_error",
    title: "The server hit an error",
    message: text || `The server answered with status ${String(status)}.`,
    hint: "Try again. If it repeats, report the model version and job id shown on the page.",
    retryable: status >= 500,
  };
}

/** The fixed text of a code, for responses that arrived without one. */
function legacy(code: string): Pick<Problem, "kind" | "title" | "message" | "hint"> {
  const entry = CODE_TEXT[code];
  if (!entry) throw new Error(`unknown error code ${code}`);
  return {
    kind: entry.kind,
    title: entry.title,
    message: entry.message ?? entry.title,
    hint: entry.hint,
  };
}

/** The evidence search could not run. The prediction is complete; this is a flag, not a failure. */
export function degradedProblem(reasons: readonly string[]): Problem {
  const why = reasons.length ? ` (${reasons.join("; ")})` : "";
  return {
    kind: "evidence_degraded",
    code: null,
    title: "The evidence track is unavailable",
    message: `The homology search could not run${why}. The prediction is complete and its scores are unchanged; only the evidence is missing.`,
    hint: "Evidence never feeds a score. Run it again later to include the evidence.",
    retryable: false,
  };
}

/** Refused records of a batch, one notice per code, each naming its records. */
export function failureProblems(failures: readonly BatchFailure[]): Problem[] {
  const byCode = new Map<string, BatchFailure[]>();
  for (const f of failures) {
    const code = f.code ?? "job_failed";
    byCode.set(code, [...(byCode.get(code) ?? []), f]);
  }
  return [...byCode.entries()].map(([code, rows]) => {
    const base = toProblem(422, rows[0]?.error ?? "", {
      code,
      retryable: rows.some((r) => r.retryable === true),
    });
    return {
      ...base,
      message: `${String(rows.length)} sequence(s): ${rows.map((r) => r.id).join(", ")}. ${base.message}`,
    };
  });
}

export interface BatchFailure {
  id: string;
  error: string;
  code?: string;
  retryable?: boolean;
}

/** A finished-but-unsuccessful job, as a Problem. Completed and in-flight jobs return null. */
export function jobProblem(job: JobBody): Problem | null {
  if (job.status !== "failed") return null;
  const text = job.error ?? "";
  if (job.error_info) {
    const coded = fromCode(job.error_info, text);
    if (coded) return coded;
  }
  if (text.includes("ESM-2 states are cached")) return toProblem(422, text);
  return {
    kind: "job_failed",
    code: null,
    title: "The job failed",
    message: text || "The job ended without a result and gave no reason.",
    hint: "Run it again. If it repeats, report the model version and job id shown on the page.",
    retryable: true,
  };
}

export class DemoApiError extends Error {
  readonly status: number;
  readonly problem: Problem;

  constructor(status: number, detail: unknown, info?: ApiErrorInfo | null) {
    const problem = toProblem(status, detail, info);
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
    code: null,
    title: "Something went wrong",
    message: e instanceof Error ? e.message : String(e),
    hint: "Reload the page and try again.",
    retryable: false,
  };
}

/** One line, for places that show a string (Trust, the gallery). */
export function describeError(e: unknown): string {
  const p = problemOf(e);
  return `${p.title}. ${p.message}`;
}
