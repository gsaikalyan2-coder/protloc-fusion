# Result cache and whole-job downloads (features 3 + 6)

Branch `phase/19c-cache-download`, from `phase/19b-observability`. Done 2026-10-23. Additive: no
registry bundle, calibrator, threshold or `engine.json` pin changed; `config_hash` and
`model_version` are unchanged.

## 1. Feature 3: result cache

- `inference/result_cache.py` (stdlib): thread-safe LRU, key `(sha256 of the normalized
  sequence, organism group, with_evidence)`, each entry tagged with the model version and
  dropped if it differs. Copies go in and out; the caller's `id` replaces the stored one.
- `inference/service.py`: `predict` and `predict_many` serve hits from the cache and send only
  the misses to the engine, still in one call (one evidence search), order preserved.
- Never cached: refusals, and a **degraded** result (evidence requested but unavailable or timed
  out), so the next request retries the search.
- Every result carries `result_cache: "hit" | "miss"`. `/demo/metrics` reports `result_cache`
  (entries, capacity, hits, misses, hit rate, skipped degraded).
- `RESULT_CACHE_ENTRIES` (optional, default 1024; 0 turns it off). In-memory, per process.

Why it is safe: the served engine is deterministic for a sequence, group and loaded model (numpy
heads over cached states, frozen calibrators and thresholds, fixed index).

## 2. Feature 6: whole-job downloads

- `GET /demo/jobs/{job_id}/results?format=json|csv|fasta` (default `json`), sent as an attachment
  `protloc_job_<id8>.<ext>`. Built by `inference/export.py` from the finished job.
- **CSV**: one row per scored sequence (top location, predicted locations with calibrated
  probabilities, sorting signal, cleavage site, predicted GO terms, consistency flags, evidence
  availability, nearest homolog, cache hit), plus one row per refused record with its code; `#`
  lines state model version, config hash, rules, scope and degraded reasons. Cells a spreadsheet
  would run as a formula are prefixed with `'`.
- **JSON**: every result and refusal plus a provenance header. **FASTA**: inputs, 60 per line,
  job id and model version in each header. Scanner jobs export as JSON only.
- New codes: `job_not_finished` (409, retryable), `no_results` (409), `export_unsupported` (422).
  `test_errors.py` now lists `job_not_finished` as retryable, deliberately.
- Frontend: `ServerDownloads` links on the Inspector's download card and on the Triage batch
  table. The browser-built per-result files are unchanged.

## 3. Checks (Linux VM over the Windows checkout)

ruff, format: clean. mypy strict (35 files): clean. `pytest tests`: 1,088 passed, 0 failed, 45
skipped. New: `test_result_cache.py` 13, `test_export.py` 11. Frontend: eslint, prettier, tsc
clean; Vitest 104 passed (12 files); build 219.30 kB. Real engine: a 6-record batch (5 cached, 1
cold) completed; CSV, JSON and FASTA downloads correct, the cold record a `refused` row; the cache
skipped all 5 results because evidence is degraded on Linux (the Windows binary cannot run there).

## 4. UNVERIFIED

- A cache **hit** against the real engine (needs a working search; Windows) and its speed-up.
- The download links clicked in a real browser.
