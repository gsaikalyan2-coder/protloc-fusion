# Observability: `/demo/metrics`, request IDs and JSON logs (features 11 + 12)

Branch `phase/19b-observability`, from `main` (`8a6fc6c`, PR #17). Done 2026-10-23. Additive: no
registry bundle, calibrator, threshold or `engine.json` pin changed, so `config_hash` and
`model_version` are unchanged.

## 1. What was built

- `protloc/utils/observability.py` (stdlib only, infrastructure layer): request-id rules, a
  `ContextVar` for the current request, `JsonFormatter`, `configure_logging`, and a thread-safe
  `Metrics` with bounded latency windows (500 most recent per route and per job kind).
- `api/app_v2.py`: one `observe` middleware (outermost, so it also sees the kill-switch 503) and
  `GET /demo/metrics`. Startup calls `configure_logging(LOG_LEVEL)` and zeroes the counters.
- `jobs/queue.py`: a job records the `request_id` that created it (a new field in
  `GET /demo/jobs/{id}`), logs `job queued` / `job finished`, and is counted by kind and outcome
  with its queue wait and run time, plus a `degraded` count.
- `config/serving.py`: optional `LOG_LEVEL` (default `INFO`; already in `.env`). The one serving
  field with a default: it changes what is logged, never which model is served.

## 2. Behaviour

- **Request IDs.** `X-Request-ID` is echoed when it is 1-64 characters of `[A-Za-z0-9._-]`;
  otherwise a UUID4 hex is issued. Every response carries it, errors included.
- **Access log** (`protloc.access`, one JSON line per request): `method`, `route` (the template,
  e.g. `/demo/jobs/{job_id}`, never the raw path), `status`, `duration_ms`, `job_id`,
  `request_id`. Job polls answering 200 log at `DEBUG`, so `INFO` is no longer one line per poll.
  The plain uvicorn access line is switched off because this replaces it.
- **Job log** (`protloc.jobs`): `job queued`, then `job finished` with `status`, `wait_ms`,
  `run_ms`, `degraded`, `error_code` and the creating `request_id`; a failed job is `WARNING`.
- **Never logged:** sequences, request bodies, results, client addresses.
- **`/demo/metrics`:** `uptime_seconds`, `requests_total`, per route `statuses` (2xx/4xx/5xx) and
  `latency` (count, p50, p95, max in ms over the recent window), per job kind `outcomes`, `wait`
  and `run`, `jobs_degraded_total`, the queue depth, evidence and catalog state, and provenance.
  In-memory and per-process like the queue (ADR-0008): a restart starts from zero. It is
  unauthenticated, like every other `/demo` route.

## 3. Checks (Linux VM over the Windows checkout)

ruff, format, scripts: clean (122 files). mypy strict on `inference models jobs api registry
evidence utils`: clean (33 files). `pytest tests`: 1,064 passed, 0 failed, 45 skipped (44 torch,
1 empty parameter set); `test_observability.py` adds 29. Live, real engine: request id echoed and
carried into the job; 4 polls counted, none logged at INFO; metrics grouped by route template.

## 4. UNVERIFIED

- The checks above on Windows (yours, below), and the log lines in a real uvicorn window.
- Overhead of the middleware under load (not measured; it is one timer, one lock, one log call).

## 5. Run, check, push (yours)

See the chat for the PowerShell commands.
