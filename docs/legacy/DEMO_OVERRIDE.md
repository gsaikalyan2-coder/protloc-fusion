# OVERRIDE / REMOVAL — read this first

This whole folder is a **temporary, standalone presentation artifact**. It does not
change the 17-phase plan, any phase status, `CLAUDE.md`, `PROJECT_PLAN.md`,
`docs/DESIGN.md`, or anything under `backend/`, `frontend/`, `configs/`, `registry/`.

## Isolation guarantees

| Guarantee | How it holds |
|---|---|
| No import in either direction | `demo/backend/app/*` imports only stdlib + FastAPI/Pydantic. Nothing in `backend/protloc` imports `demo`. |
| No port collision | demo API on **8010** (real one: 8000). Demo SPA on **5174** (real one: 5173). |
| No route collision | every demo route is under `/demo/*`. |
| No config/registry writes | the demo reads nothing from `configs/`, `registry/`, `data/`. |
| No design-system edit | `docs/DESIGN.md` untouched. `demo/frontend/src/styles/tokens.css` re-declares the same token *names* with dark-green values, in the demo tree only. |
| No document edits | this folder is additive; `## Current Phase` in `CLAUDE.md` is unchanged (still Phase 1 open). |

## Three levels of override

1. **Kill switch (no file changes).** Start the API with `PROTLOC_DEMO_ENABLED=0`
   and every `/demo/*` route returns `503` with the reason. The SPA then shows the
   "backend unreachable" banner and nothing else.
2. **Stop using it.** Don't run the two dev servers. Nothing in the real project
   references them, so there is nothing else to disable.
3. **Delete it.** `rm -rf demo/` (Windows: delete the `demo` folder). That is the
   complete uninstall — no leftovers anywhere else in the repo.

## If you later want to keep an idea from here

Do it the plan's way, not by promoting this code: write an ADR in `docs/adr/`,
then implement it inside the phase that owns it (Triage/Scanner/Trust screens
belong to Phase 12, the evidence ledger to Phase 10, calibration to Phase 9).
The code here is a demo-grade heuristic and should not be merged as-is.
