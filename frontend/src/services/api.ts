/**
 * The ProtLoc Fusion API client — the only module permitted to call fetch().
 *
 * ESLint enforces that everywhere else; the reason is that a contract change
 * should be a change to one file rather than a search across components. What
 * lives here is the transport: base URL resolution, timeouts, and turning a
 * non-2xx response into a typed error that carries the server's provenance
 * fields. Endpoint methods arrive with the contract they belong to — /health is
 * here because CLAUDE.md already specifies it; the prediction endpoints are
 * added in Phase 11, against the OpenAPI document frozen in Phase 2.
 */

import type { HealthResponse, Provenance } from "@/types/api";

/** Requests that outlive this are abandoned rather than left hanging. */
const DEFAULT_TIMEOUT_MS = 10_000;

/**
 * Where the API lives.
 *
 * In development Vite proxies same-origin `/api` to the backend. In production
 * the frontend and backend are separate deployables (ADR-0005) and the origin
 * is injected at build time. A trailing slash is removed so that path joining
 * never produces a double slash.
 */
export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(
  /\/+$/,
  "",
);

/** A request that reached the server and came back as an error. */
export class ApiError extends Error {
  readonly status: number;
  readonly url: string;
  readonly provenance: Partial<Provenance>;

  constructor(status: number, url: string, message: string, provenance: Partial<Provenance>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
    this.provenance = provenance;
  }
}

/** A request that never got an answer: offline, DNS, TLS, timeout, abort. */
export class NetworkError extends Error {
  readonly url: string;

  constructor(url: string, cause: unknown) {
    super(`Could not reach the ProtLoc Fusion API at ${url}`);
    this.name = "NetworkError";
    this.url = url;
    this.cause = cause;
  }
}

function readProvenance(body: unknown): Partial<Provenance> {
  if (typeof body !== "object" || body === null) {
    return {};
  }
  const record = body as Record<string, unknown>;
  const provenance: Partial<Provenance> = {};
  for (const key of ["model_version", "config_hash", "rules_version", "experiment_id"] as const) {
    const value = record[key];
    if (typeof value === "string") {
      provenance[key] = value;
    }
  }
  return provenance;
}

function readMessage(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null) {
    const { detail } = body as { detail?: unknown };
    if (typeof detail === "string") {
      return detail;
    }
  }
  return `Request failed with status ${String(status)}`;
}

export interface RequestOptions {
  /** Abort and raise NetworkError after this many milliseconds. */
  timeoutMs?: number;
  /** Caller-owned signal; aborting it aborts the request. */
  signal?: AbortSignal;
}

/**
 * Perform one JSON request against the API.
 *
 * Throws {@link ApiError} for a response the server produced and
 * {@link NetworkError} for one it did not. Callers decide how to present
 * either; this module never logs and never renders.
 */
export async function request<T>(
  path: string,
  init: RequestInit = {},
  options: RequestOptions = {},
): Promise<T> {
  const url = `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
  const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const signal =
    options.signal === undefined ? timeout : AbortSignal.any([timeout, options.signal]);

  let response: Response;
  try {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    response = await fetch(url, { ...init, signal, headers });
  } catch (cause) {
    throw new NetworkError(url, cause);
  }

  const text = await response.text();
  const body: unknown = text === "" ? null : JSON.parse(text);

  if (!response.ok) {
    throw new ApiError(
      response.status,
      url,
      readMessage(body, response.status),
      readProvenance(body),
    );
  }

  return body as T;
}

/** Liveness, the loaded model version and the current queue depth. */
export async function getHealth(options: RequestOptions = {}): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { method: "GET" }, options);
}
