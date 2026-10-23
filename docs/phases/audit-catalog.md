# Audit + catalog: code audit, evidence fix, browse all cached sequences, accession lookup

Branch `phase/19a-catalog`, from `origin/main` (PR #16, v0.18.0). Done 2026-10-23. Additive only:
no registry bundle, calibrator, threshold, `engine.json` pin or contract route changed, so
`config_hash` and `model_version` are unchanged.

**Scope: "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers".**

## 1. Audit (run on a Linux VM over the Windows checkout, Python 3.11.17)

| Check | Result |
|---|---|
| `ruff check` / `ruff format --check` backend, `ruff check` scripts | clean (120 files) |
| mypy `domain` + `config`; strict on `inference models jobs api registry evidence` | clean (8 / 31 files) |
| `pytest tests` | 1,035 passed, 0 failed, 45 skipped (44 need torch, 1 empty parameter set) |
| Real engine start (`app_v2` lifespan, `.env`, real bundles, cache, ontology, index) | starts in ~30 s, every integrity check passes |
| End-to-end through the queue (TestClient, real engine) | predict 0.21-0.43 s, 11-sequence batch 0.84-1.10 s, scan, trust, structure 200; refusals `not_cached`, `unknown_group`, `fasta_no_header` |
| Frontend (clean `npm ci` copy) | eslint, prettier, tsc clean; Vitest 97 passed (10 files); build 217.72 kB (gzip 68.75 kB) |
| `export_openapi.py` | no content drift (CRLF vs LF only) |

## 2. Fixed

**F1. An unstartable search binary failed the whole job.** `evidence/search.py` mapped a missing
binary, a non-zero exit and a timeout to `EvidenceUnavailable`, but not an `OSError` from
`Popen` (wrong platform, no execute bit, blocked by antivirus, missing DLL) or an unusable work
directory. The job failed `job_failed` instead of completing `degraded`, as Phase 17 promises.
Both now raise `EvidenceUnavailable`. Tests: `test_a_binary_that_cannot_start_is_a_reason_not_a_crash`,
`test_an_unusable_work_directory_is_a_reason_not_a_crash`. Seen for real: the Windows
`mmseqs.exe` under Linux now gives a completed job with `degraded: true`.

**F12. The job poller could outlive its job and its screen.** `useJob` stored one interval handle.
A run superseded while its POST was in flight, or one whose screen unmounted before the POST
answered, started an interval nobody held; its completion branch cleared the *current* handle,
not its own, so it polled a finished job forever (seen on Windows: two completed jobs polled for
20+ minutes). Each run now carries a generation; a stale run starts no poller and a poller clears
its own interval. Test: `frontend/src/services/useJob.test.ts` (3 cases; 2 failed before the fix).

**F13. The frontend dependency gate failed on `main` since Phase 16.** `a9ae234` added four
test-only dev dependencies (`@testing-library/dom`, `@testing-library/react`, `jsdom`, `vitest`)
without allowlisting them in `scripts/check_frontend_deps.py`, so CI's dependency job exited 1 on
every push since PR #15. They are now on the dev allowlist with their reasons. None ships to the
browser or renders product UI, so the React-only rule holds and no ADR is needed (the allowlist is
the decision record, as for `openapi-typescript`).

## 3. Found, not changed

F2 `/demo/health` reports evidence `available` when the binary exists but cannot run (startup checks
its hash, never runs it) · F3 the CI pytest job installs `[dev,data]` without numpy, so all 10
engine/API test files are skipped in CI · F4 local `main` was behind `origin/main` · F5 README is stale
(Phase 1 status, deleted files, `api.app` run command) · F6 docs cite `consistency_rules.yaml`; the file is
`consistency.json` · F7 `app_v2` reads `DEMO_CORS`; `CORS_ALLOWED_ORIGINS` is unused by it · F8 no
rate limiting · F9 `rules_version` is `demo-0.1.0` · F10 stale `backend/build/` copy of old code ·
F11 untracked: pptx, `docs/report/`, `data/manifests/rollback_logs/`, `.claude/`.
F2, F3 + F5-F7 and F8 were scoped Out of this round (expand-and-contract).

## 4. Features 1 + 2: the catalog

- `scripts/build_catalog.py --write` -> `data/gallery/catalog.json` (gitignored): every registry
  accession whose sequence is in the residue cache. **507 accessions, 500 distinct sequences**;
  partitions 1: 146, 2: 77, 3: 93, 4: 92, 5: 99. Built in 18 s.
- `inference/catalog.py` (stdlib only): reads it once at startup, keeps entries the engine can
  score (`has_states`), refuses a different residue cache or schema. A missing file is a reported
  state, not a startup failure.
- `GET /demo/catalog?q&location&group&partition&limit&offset` (no sequences, max 100 per page) and
  `GET /demo/catalog/{accession or entry name}` (with sequence). New codes `accession_not_cached`
  (404) and `catalog_unavailable` (503), both not retryable.
- Frontend: `CatalogBrowser` on the Gallery page — search, location and partition filters,
  paging, an accession box; "Open" hands the sequence to the Inspector like a gallery card. Each
  row is badged `in-sample` (partitions 1-4) or `held-out` (partition 5).
- Real engine: 507 listed, 0 dropped; a lookup then predict completes; an unknown accession is 404.

Partition 5 database labels are displayed beside predictions; nothing is tuned or frozen from them.

## 5. Verified on Windows (maintainer run, 2026-10-23)

- `pytest tests`: 0 failed; the torch parity, head, CRF and GO tests pass (6 POSIX-only skips).
- Frontend lint, 97 Vitest tests and the build pass (before F12; 100 after, VM run).
- Live: catalog 507 entries, 0 dropped; lookup -> predict -> `ER` (database label ER); unknown
  accession 404 `accession_not_cached`.
- Live MMseqs2 search: a cold first query ran 5.9 s and hit the 5 s budget (`degraded`, prediction
  unchanged); the next ran 3.0 s, 5 hits, not degraded. `EVIDENCE_TIMEOUT_SECONDS=5` is tight for a
  cold index; raising it is a local `.env` choice, not changed here.

## 6. UNVERIFIED

- The F12 fix in a real browser (unit-tested only).
- The catalog panel by eye in light and dark against the live backend.
- CI: the new engine-side tests are skipped there too (F3).

## 7. Push (yours)

```powershell
cd C:\Users\saika\protloc-fusion
Remove-Item .git\stale-index-lock-from-claude
git fetch
git switch -c phase/19a-catalog origin/main
python scripts/build_catalog.py --write
cd backend; python -m pytest tests -q; cd ..
cd frontend; npm run lint; npm run test; npm run build; cd ..
git add backend/protloc/evidence/search.py backend/tests/unit/test_evidence_search.py
git commit -m "fix(evidence): an unstartable search binary degrades instead of failing the job"
git add backend/protloc backend/tests scripts/build_catalog.py frontend/src docs/phases/audit-catalog.md  # includes useJob.test.ts
git commit -m "feat(catalog): browse all cached sequences and look one up by accession"
# demoApi.ts carries both the catalog client and the F12 poller fix, so they share this commit
git push -u origin phase/19a-catalog
```
