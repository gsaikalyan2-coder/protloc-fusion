# PHASE 02 — Contracts & wireframes

> The phase that decides what the system promises, before anything is built that
> could quietly promise something else. Nothing here predicts anything. What this
> phase produces is a contract a reviewer can read and, from it alone, predict the
> shape of any response the service will ever return.

Read `PHASE_PROTOCOL.md` once before starting. Read this file at the start of the
phase and again before the exit gate.

---

## 1. Prerequisites — the entry gate

Each statement must be **verified true**, not assumed. The check is named beside it.

| # | Statement | How to verify |
|---|---|---|
| 1 | Phase 1 is closed and its handover exists | `test -f docs/handover/handover_phase_01.txt` → exit 0 |
| 2 | `CLAUDE.md` marks Phase 1 `[x]` and Phase 2 `[ ]` | read `## Current phase` |
| 3 | The working tree is the repository, not the archive | `git -C . rev-parse --show-toplevel` → `…/protloc-fusion` |
| 4 | The working tree is clean | `git status --porcelain` → no output |
| 5 | CI is green on `main` | `gh run list --branch main --limit 1` → `completed  success` |
| 6 | The backend installs and its tests pass | `pip install -e "backend[dev]"`; `cd backend && python -m pytest tests` → `51 passed` |
| 7 | The frontend installs, lints and builds | `cd frontend && npm ci && npm run lint && npx tsc --noEmit && npm run build` → exit 0 each |
| 8 | ADR-0001 is Accepted | `grep -n 'Status' docs/adr/ADR-0001-serving-strategy.md` → `**Accepted**` |
| 9 | `protloc/api` is empty of application code | `ls backend/protloc/api` → `__init__.py` only |
| 10 | `frontend/src/types/api.ts` is still hand-written | it carries the "Phase 2 generates this" header |

Statements 9 and 10 are the two gaps this phase closes; if either is already false,
someone started Phase 2 and the work must be reconciled before continuing.

---

## 2. The prompt

> You are executing Phase 2 of ProtLoc Fusion. All work happens in
> `C:\Users\saika\protloc-fusion`, which is the git repository.
> `C:\Users\saika\ProtLoc` is an archive — never edit it.
>
> Read, in order: `CLAUDE.md`, `docs/handover/handover_phase_01.txt`,
> `PHASE_PROTOCOL.md`, `PROJECT_PLAN.md` §4, `FOLDER_STRUCTURE.md`,
> `docs/adr/ADR-0001-serving-strategy.md`, and this file in full.
>
> Phase 2 turns the blueprint into interface contracts. It freezes the HTTP API as
> an OpenAPI 3.1 document, generates the frontend's types from that document, makes
> the generation a CI gate, and replaces `Dockerfile.serve`'s CMD — currently the
> config validator — with the real FastAPI application.
>
> **The contract is asynchronous.** `POST /predict` returns `202` with
> `{job_id, result_url, status}` and never a prediction (ADR-0001). Every response,
> including the `202` and every error, carries `model_version`, `config_hash`,
> `rules_version` and `experiment_id`.
>
> **Nothing in this phase predicts anything.** There is no model, no registry bundle
> and no job queue until Phases 7 and 11. Do not write a stub that returns a
> plausible prediction so that "the shape is visible" — the shape lives in the
> OpenAPI document, which is the deliverable. An endpoint whose dependency does not
> exist yet reports that truthfully (§4.2), and that reporting is real behaviour,
> not a placeholder.
>
> Work one subtask at a time, in the order given in §3. After each, report in the
> `PHASE_PROTOCOL.md` §5 confirmation format and stop for a go-ahead. Commit after
> each subtask using Conventional Commits and push, so CI runs.
>
> Layer depths are enforced by `backend/tests/test_layering.py`. `api/` may import
> only `{domain, inference, config, utils}`. No business logic in `api/`: routers
> parse, validate, delegate, return. mypy is invoked **from `backend/`**. Run
> pytest and mypy as `python -m pytest` / `python -m mypy`. Edit
> `.github/workflows/ci.yml` in an editor, never with a PowerShell `-replace`.

---

## 3. Deliverables

Seven subtasks. Each one is a commit.

### 3.1 — This phase file
```
docs/phases/PHASE_02_contracts.md
```
`docs(phase-2): write the phase file`

### 3.2 — The contract schemas
```
backend/protloc/domain/contracts.py       Pydantic v2 models for every request and response
backend/tests/unit/test_contracts.py      the envelope rule and the enum mirroring, as assertions
```
Every model in one module in `domain/`, not in `api/`. A schema is vocabulary, not
delivery: `domain/` is depth 100 and imports nothing internal, so the contract is
readable by the evaluation layer, the job layer and the router layer alike without
any of them importing each other. Pydantic is an external dependency, which
`domain/` is permitted.

### 3.3 — The application
```
backend/protloc/domain/ports.py           ProvenanceSource protocol
backend/protloc/inference/provenance.py   the registry-backed implementation
backend/protloc/api/app.py                the FastAPI application factory
backend/protloc/api/routers/health.py     GET /health, GET /version
backend/protloc/api/routers/predict.py    POST /predict, POST /batch
backend/protloc/api/routers/jobs.py       GET/POST /jobs/*
backend/protloc/api/routers/examples.py   GET /examples
backend/tests/integration/test_api_contract.py
```
`feat(api): serve the frozen contract`

### 3.4 — The frozen OpenAPI document
```
scripts/export_openapi.py                 app → docs/api/openapi.yaml
docs/api/openapi.yaml                     GENERATED. Never hand-edited
```
plus a CI job `openapi-current` that regenerates and fails on a diff.
`feat(contracts): freeze the API as OpenAPI 3.1`

### 3.5 — Generated frontend types
```
frontend/src/types/api.generated.ts       GENERATED from docs/api/openapi.yaml
frontend/src/types/api.ts                 re-exports the generated types; keeps the runtime helpers
frontend/package.json                     + openapi-typescript, pinned
scripts/check_frontend_deps.py            + its allowlist entry and reason
```
plus a CI job `types-current` that regenerates and fails on a diff.
`feat(frontend): generate the API types from the OpenAPI document`

### 3.6 — The serving entry point
```
deployment/Dockerfile.serve               CMD → uvicorn; HEALTHCHECK → GET /health
.github/workflows/ci.yml                  the docker-build assertions, corrected
```
`feat(deploy): serve the FastAPI application`

### 3.7 — Architecture, wireframes, schema freeze, handover
```
docs/ARCHITECTURE.md
docs/diagrams/request_sequence.md         submission → queue → poll → result
docs/diagrams/wireframes/landing.md
docs/diagrams/wireframes/predict.md
docs/diagrams/wireframes/result.md
docs/diagrams/wireframes/batch.md
docs/diagrams/wireframes/job.md
CLAUDE.md                                 data schema section marked frozen; Current Phase updated
FOLDER_STRUCTURE.md                       docs/api/ and docs/diagrams/wireframes/ declared
docs/adr/ADR-0001-serving-strategy.md     §7 citation corrected
docs/handover/handover_phase_02.txt
```
`docs(phase-2): architecture, wireframes and handover`

---

## 4. Execution notes

Rules specific to this phase. Each of §4.1–§4.7 is a decision taken at the start of
the phase rather than discovered halfway through it.

### 4.1 The OpenAPI document is generated, never written

FastAPI 0.141 emits OpenAPI 3.1 from the Pydantic models. Hand-writing the YAML
would create a second source of truth that drifts from the code within one phase.
`scripts/export_openapi.py` serialises `app.openapi()` to `docs/api/openapi.yaml`
with sorted keys, and CI regenerates and diffs it — the same pattern that already
protects `tokens.css`. The committed YAML is the frozen artifact a reviewer reads;
the code is what produces it.

`docs/api/` is a new directory. `FOLDER_STRUCTURE.md` does not currently name a
home for the OpenAPI document, so this phase declares one.

### 4.2 What an endpoint with no implementation returns — and why it is not a stub

There is no registry bundle until Phase 7 and no job queue until Phase 11. Five of
the eight endpoints therefore have no dependency to delegate to.

They return **`503 Service Unavailable`** with a typed `ServiceUnavailableError`
body naming the missing dependency and the phase that supplies it.

This is not the placeholder that `PHASE_PROTOCOL.md` §3.3 forbids. A placeholder
asserts something false — a mock prediction claims a model ran. A `503` asserts
something true: the service cannot accept jobs because no model bundle is loaded.
The condition is real, it is checked at request time rather than hardcoded, and it
stops being true the moment Phase 7 puts a bundle in the registry — at which point
the routers do not change, only what `ProvenanceSource` finds on disk. The
behaviour is tested, and the test is the assertion that the service is honest about
what it cannot do.

`GET /health` and `GET /version` are live now and answer truthfully: health reports
the process, the loaded bundle (or its absence) and a queue depth of zero, because
zero is the true depth of a queue with no workers.

### 4.3 The reproducibility envelope before there is anything to reproduce

`CLAUDE.md` contract rule 6 makes `model_version`, `config_hash`, `rules_version`
and `experiment_id` an API guarantee on every response, including errors. None of
those four values exists in Phase 2.

**Decision:** the four fields are **present on every response and typed
`string | null`**, and they are `null` exactly when no registry bundle is loaded.
The OpenAPI schema says so in each field's description.

The alternative — a sentinel like `"unloaded"` or `"0.0.0"` — was rejected. A
client cannot distinguish an invented version string from a real one, and
`PROJECT_PLAN.md` §7.2 exists to stop plausible defaults from entering the system
where a missing value belongs. `null` is unambiguous, survives into Phase 7
unchanged, and keeps the field present as rule 6 requires.

The values come from a `ProvenanceSource` port declared in `domain/ports.py` and
implemented in `inference/provenance.py`, which reads the registry directory named
by `MODEL_REGISTRY`. `api/` depends on the port, never on the filesystem: that is
what keeps the routers thin.

### 4.4 Two endpoints return results, and they are not the same endpoint

The API map lists both `GET /jobs/{job_id}` and `GET /jobs/{job_id}/results`. The
exit criterion requires a reviewer to predict any response shape from the spec
alone, so the spec must state the difference rather than leave it to be inferred:

- **`GET /jobs/{job_id}`** — the job resource. Always `200` for a live job,
  whatever its status. Carries status, progress and, when `status == "completed"`,
  the full result payload inline. This is the URL in `result_url` and the one the
  frontend polls. `410 Gone` past the retention window.
- **`GET /jobs/{job_id}/results`** — the result *representation*, for download.
  `?format=json|csv`, `Content-Disposition: attachment`, and `409 Conflict` when
  the job has not completed. It never returns a partial result and it is never
  polled.

`GET /docs` is FastAPI's built-in Swagger UI. It is documented as a note in
`ARCHITECTURE.md`, not as a path in the frozen contract: it is a rendering of the
contract, not part of it.

### 4.5 The generated TypeScript file is not the file the app imports

`openapi-typescript` emits types only. The existing `frontend/src/types/api.ts`
also exports runtime values — `TERMINAL_JOB_STATUSES`, `isTerminal`, `hasResults` —
which generation would delete.

**Decision:** generation writes `src/types/api.generated.ts`, which carries a
"GENERATED — do not edit" header and is regenerated in CI. `src/types/api.ts` stays
hand-written, re-exports the generated types under the names the app already uses,
and keeps the three runtime helpers. `FOLDER_STRUCTURE.md`'s note that `types/` is
generated from the OpenAPI spec remains true of the types; the helpers are code
about the types, not types.

`openapi-typescript` is a devDependency that emits a `.ts` file at build time and
ships nothing to the browser. It renders no UI, so the React-only rule permits it —
but `scripts/check_frontend_deps.py` works from an allowlist, so it must be added
there, in the same commit, with its reason. That allowlist entry is the decision
record; this does not need an ADR, because nothing about it is irreversible.

### 4.6 The docker-build CI job breaks when the CMD changes, by design

`ci.yml`'s step *"Serving image validates its configuration and refuses an empty
one"* runs the container to completion and asserts its exit code. Once the CMD is
`uvicorn`, the container no longer exits and that step hangs until the job times
out. It must be changed in the same commit as the Dockerfile, to:

1. run `python -m protloc.config.settings` as an explicit command override, keeping
   both the refusal-on-empty-environment assertion and the success case; and
2. start the container detached with a valid environment, poll `GET /health` until
   it answers `200`, assert the body's `status`, and stop the container.

The second check is what proves the image now serves the application rather than
merely containing it.

`.github/workflows/ci.yml` must be edited in an editor. The desktop file bridge
refuses to write under `.github/`; the workaround recorded in the Phase 1 handover
is to write the file elsewhere in the repository and `Move-Item` it into place,
then diff it against a known-good copy before committing. A PowerShell `-replace`
already put a backtick where a backslash belonged once and broke CI.

`HEALTHCHECK` becomes a request to `/health`. The slim image has no `curl`, so the
check is `python -c` with `urllib.request`, which is already present.

### 4.7 Wireframes are committed markdown, not image files

The five screens are Landing, Predict, Result, Batch and Job. Each wireframe is a
markdown file with a monospace layout sketch and a numbered list of the states the
screen must handle — including the **queued/running** state that ADR-0001 §4.2
introduces and `docs/DESIGN.md` §3.4 defines.

They are markdown because `docs/DESIGN.md` is the visual authority and a binary
mockup would compete with it: a mockup pins colours and spacing that the tokens
already own, and cannot be diffed in review. The wireframe's job is to fix layout,
hierarchy and state coverage, and to name which token each region draws from.

### 4.8 A correction to ADR-0001

ADR-0001 §7 checks off *"`docs/DESIGN.md` §3.5 defines the queued/running visual
state"*. §3.4 is the pending state; §3.5 is evidence. §4.2 of the same ADR also
says the addition is "to be made in Phase 12" while §7 marks it already done.

Both are documentation errors in an Accepted ADR, not decision changes. Fix the
section number and reword §4.2 to past tense, in the `docs:` commit of §3.7. The
decision itself is untouched, so the ADR's status does not change.

---

## 5. Verification — the exit gate

Every check is binary. Run from a clean tree at the phase's final commit.

| # | Command | Expected |
|---|---|---|
| 1 | `cd backend && python -m pytest tests` | `N passed`, N > 51, 0 failed |
| 2 | `cd backend && python -m mypy protloc/domain protloc/config` | `Success: no issues found` |
| 3 | `ruff check backend && ruff format --check backend` | `All checks passed!` |
| 4 | `ruff check --config backend/ruff.toml scripts` | `All checks passed!` |
| 5 | `python scripts/export_openapi.py && git diff --exit-code -- docs/api/openapi.yaml` | exit 0 |
| 6 | `python -c "import yaml,sys; d=yaml.safe_load(open('docs/api/openapi.yaml')); print(d['openapi'])"` | `3.1.0` |
| 7 | every endpoint in `CLAUDE.md`'s API map appears in the document | `test_every_documented_endpoint_is_in_the_contract` passes |
| 8 | every response in the document has a schema — no bare `{}` | `test_every_response_has_a_schema` passes |
| 9 | every response schema carries the four envelope fields | `test_every_response_carries_provenance` passes |
| 10 | `POST /predict` declares `202` and does **not** declare a prediction body on `200` | `test_predict_is_asynchronous` passes |
| 11 | `cd frontend && npm run generate:types && git diff --exit-code -- src/types/api.generated.ts` | exit 0 |
| 12 | `cd frontend && npx tsc --noEmit` | exit 0 |
| 13 | `cd frontend && npm run lint && npm run build` | exit 0, `dist/index.html` exists |
| 14 | `python scripts/check_frontend_deps.py` | `React only; 15 dependencies, all pinned` |
| 15 | `docker build -f deployment/Dockerfile.serve -t protloc-serve:local .` | succeeds |
| 16 | run that image with a valid environment; `curl -fsS localhost:8000/health` | `200`, body has `status`, `model_version: null` |
| 17 | `docker run --rm protloc-serve:local python -c "import torch"` | non-zero — still separate |
| 18 | the five wireframes exist and each names its queued/running state | `ls docs/diagrams/wireframes/*.md` → 5 files; grep each |
| 19 | `gh run list --branch main --limit 1` | `completed  success`, all jobs including the two new ones |
| 20 | `docs/handover/handover_phase_02.txt` exists and follows the §6 template | read it |

Checks 7–10 are how the phase's exit criterion in `PROJECT_PLAN.md` §4 — *"a
reviewer can predict any response shape from the spec alone"* — becomes binary.
That sentence is not machine-checkable as written; these four tests are the
mechanical form of it, and they run against the generated document rather than
against the code that produced it.

---

## 6. Handover

Write `docs/handover/handover_phase_02.txt` to the `PHASE_PROTOCOL.md` §6 template
before closing. It must be self-contained: a cold session resumes from it alone.

It must record, at minimum:

- that the contract is frozen, and that changing it after Phase 2 means a version
  bump and a regenerated types file, not an edit;
- the `null` envelope decision (§4.3) and the phase that makes the fields non-null;
- the `503` decision (§4.2) and which endpoints carry it;
- that `docs/api/openapi.yaml` and `src/types/api.generated.ts` are generated and
  both are CI-diffed;
- the corrected `docker-build` steps (§4.6), because the next person to touch the
  serving CMD will hit the same trap;
- the exact `openapi-typescript` version and its allowlist reason.

---

## 7. Failure modes

What goes wrong in this phase, and the signal that it is happening.

**The spec drifts from the code.** Someone edits `docs/api/openapi.yaml` to fix a
description. *Signal:* the `openapi-current` CI job fails with a diff. *Prevention:*
the file's header says GENERATED and names the script; fix descriptions in the
Pydantic models' `Field(description=...)`.

**A stub becomes permanent.** The `503` handlers are convenient, and Phase 11
arrives to find something that looks finished. *Signal:* `test_api_contract.py`
still asserts `503` after a registry bundle exists. *Prevention:* the test asserts
the *reason* — no bundle loaded — not the status code alone, so it fails loudly
once a bundle is present rather than passing on a lie.

**The envelope is added to the routers.** Four fields on every response invites a
helper in `api/` that assembles them, which is business logic in a router.
*Signal:* `api/` grows a module that is not a router. *Prevention:* the envelope is
a field group on the base response model in `domain/contracts.py`, populated from
the injected `ProvenanceSource`.

**The frontend types are generated but nothing uses them.** `api.generated.ts`
exists, `services/api.ts` still refers to the hand-written interfaces, and the
build passes. *Signal:* grep `services/api.ts` for a type name that only the
hand-written file defines. *Prevention:* check 12 is run after deleting the
hand-written duplicates, not alongside them.

**The CMD change silently disables the config gate.** The Dockerfile starts
`uvicorn`, the old CI step is deleted rather than rewritten, and the image loses
its refusal-on-empty-environment behaviour without anyone noticing. *Signal:* the
`docker-build` job gets shorter. *Prevention:* §4.6's step 1 keeps both halves of
the original assertion as an explicit command override.

**Wireframes omit the pending state.** Every screen is drawn in its happy state and
the queued/running case surfaces in Phase 12 as a missing screen. *Signal:* a
wireframe with no state list. *Prevention:* exit-gate check 18 greps for it.
