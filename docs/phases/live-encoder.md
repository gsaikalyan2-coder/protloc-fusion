# Live encoder (feature 13, ADR-0019)

Branch `phase/19f-docker-live-encoder`, with feature 15 (`docs/phases/docker-live.md`). Done
2026-10-24. Decision: `docs/adr/ADR-0019-live-encoder.md` (amends ADR-0016).

**Default unchanged:** `ENCODER_MODE=cache_only` serves exactly as before, numpy only, and
`config_hash` / `model_version` are unchanged in either mode. No registry bundle, calibrator,
threshold or `engine.json` pin changed.

## 1. What was built

- `inference/live_encoder.py`: loads the Phase 7 `TorchEncoder` with local files only (hub cache
  `ESM_CACHE_DIR`, offline flags set), returns per-residue float16 states in the cache's layout,
  self-checks against one cached sequence at startup (tolerance 0.02), and reports its identity
  and counters. torch and transformers are imported only in live mode.
- `inference/engine.py`: `load_engine(..., live_factory=...)` attaches the encoder when
  `ENCODER_MODE=live` (startup refuses, listing why, on a missing extra, missing weights, missing
  gallery for the self-check, or a failed self-check); `has_states` accepts any sequence in live
  mode; a cached sequence is still read from the cache; every result carries `encoder` and
  `outside_evaluated_subset`; provenance `encoder` names mode, checkpoint, revision, self-check
  sequence and difference.
- Settings: `ENCODER_MODE` (`cache_only` | `live`), `ESM_CACHE_DIR` (blank = hub default),
  `ESM_DEVICE`. `.env.example` lists them with `RESULT_CACHE_ENTRIES`.
- `pyproject.toml`: `live` extra = numpy 2.4.6, torch 2.14.1, transformers 5.18.0.
- `scripts/fetch_esm.py`: downloads the pinned revision (read from the residue cache's
  `index.json`) into `data/esm2` (or `--cache-dir`), then runs the same self-check the server
  runs and prints the `.env` lines.
- API: `/demo/version` reports `encoder` and states the live limit; `/demo/metrics` reports
  `live_encoder`; the whole-job CSV gains `encoder` and `outside_evaluated_subset` columns.
- Frontend: Inspector notice "Encoded live, outside the evaluated subset", the same warning on the
  printed report, "N encoded live" on the batch dashboard.
- Docker: `Dockerfile.live` takes `EXTRAS` (default `serve`, no torch; `serve,live` adds the CPU
  torch build and transformers); `docker-compose.encoder.yml` layers live mode over the live
  stack. CI validates the merged compose file; the torch image is not built in CI.

## 2. Checks (Linux VM over the Windows checkout; no torch there)

ruff, format, scripts: clean (128 files). mypy strict: clean (36 files). `pytest tests`: 1,103
passed, 0 failed, 45 skipped. `test_live_encoder.py` adds 15, five of them against the **real**
registry and residue cache with a fake encoder: an uncached sequence is scored and labelled live,
cached sequences still come from the cache, provenance names the encoder and a zero self-check
difference, a drifting encoder stops the service, and cache-only mode refuses the same sequence
as before. On the serving-only dependency set: cache-only starts and never imports torch; live
mode refuses to start and names the `live` extra. Frontend: Vitest 133 passed, build clean.

## 3. Measured on Windows (maintainer run, 2026-10-24)

`scripts/fetch_esm.py`: `facebook/esm2_t30_150M_UR50D` @ `a695f604...` downloaded to
`data/esm2` (402 MB transferred, 595 MB on disk, 150.6 s). Self-check on `A0A172M4N1`
(130 residues): max |live - cached| = **0.000244** (tolerance 0.02), load + encode 11.5 s.
The transformers load report lists the five `lm_head.*` weights as unexpected: the encoder loads
ESM-2 without its masked-language head, which it never uses.

## 4. UNVERIFIED

- A live prediction through the server, and the steady-state seconds per sequence.
- The torch image build (`EXTRAS=serve,live`) under Docker.
- Any accuracy claim for live sequences: none is made; they are out-of-sample for every metric.
