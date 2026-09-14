# PHASE 01 — Scaffold & source of truth

- **Objective:** Lock the repository skeleton, conventions and CI so that every later
  phase adds code to a structure that already enforces the rules.
- **Exit criterion:** `git clone` → install → lint → CI green on an empty scaffold, with
  frontend and backend both building.
- **Governing ADRs:** 0001 (async contract), 0005 (hosting shapes the Dockerfiles)
- **Estimated size:** 6 subtasks

---

## 1. Prerequisites — entry gate

All must be **verified true** before starting.

| # | Statement | How to verify |
|---|---|---|
| 1 | Phase 0 is closed | `docs/` contains `DESIGN.md`, `THIRD_PARTY.md`, `EXTERNAL_WORKFLOW.md`, `CONTEXT_MEMORY.md` |
| 2 | The plan and contract exist and agree | `PROJECT_PLAN.md`, `CLAUDE.md`, `FOLDER_STRUCTURE.md` present; §4 phase list matches `CLAUDE.md` Current Phase list |
| 3 | All five ADRs are Accepted | `ls docs/adr/` shows ADR-0001…0005; each header reads `Status: Accepted` |
| 4 | No decision is outstanding | `PROJECT_PLAN.md` §7.1 shows every row resolved |
| 5 | Toolchain present | `python --version` → 3.11.x · `node --version` → ≥20 · `git --version` · `docker --version` |
| 6 | A GitHub remote exists | `git remote -v` resolves, or the repo is created this phase as subtask 0 |

If #3 or #4 fails, stop — that is a decision, and decisions become ADRs before code.

---

## 2. The prompt

> Paste this to start the phase. It is self-contained.

```
You are executing Phase 1 of ProtLoc Fusion.

Read first, in this order:
  1. CLAUDE.md                     — the coding contract and layer rules
  2. FOLDER_STRUCTURE.md           — the exact tree to create and why
  3. PHASE_PROTOCOL.md             — how phases are run
  4. docs/phases/PHASE_01_scaffold.md — this file

Build the complete repository skeleton for ProtLoc Fusion: a React + Vite +
TypeScript frontend and a FastAPI backend whose 17 modules follow a strict
L-1 (domain) → L5 (evidence) dependency layering.

Phase 1 creates STRUCTURE, not behaviour. Every package, config file, linter,
CI job and Dockerfile is real and working. No prediction logic, no model code,
no API business logic — those belong to Phases 3-12.

The one thing that must genuinely work is the enforcement: linting, type
checking, the import-layer rule, and CI must all run and pass on the empty
scaffold. If the rules are not enforced from the first commit, they will not
be enforced later.

Constraints that apply from the first line of code:
  - No placeholder code. A file that exists is complete for its current scope.
    Do not write `# TODO: implement` or a function body of `pass`.
  - No business logic in backend/protloc/api/. Routers parse, validate,
    delegate, return.
  - backend/protloc/domain/ imports nothing internal.
  - Nothing in the serving path imports backend/protloc/training/.
  - No colour literal anywhere in the frontend — colours come from tokens
    generated from docs/DESIGN.md.
  - No secret values. .env.example carries ENTER_YOUR_VALUE_HERE only.

Work through the six subtasks in §3 of this file, in order. After each one,
report using the confirmation format in PHASE_PROTOCOL.md §5 and stop for a
go-ahead. Commit after each subtask using Conventional Commits.

Begin with subtask 1.
```

---

## 3. Subtasks and deliverables

### Subtask 1 — Repository skeleton and git hygiene

Create the full tree from `FOLDER_STRUCTURE.md`. Every Python package gets a real
`__init__.py`; every directory that must exist but is empty for now gets a `.gitkeep`.

```
.gitignore          # .env, __pycache__, node_modules, data/raw, data/processed,
                    # registry/*/model.pt, .venv, dist, *.pt, *.pkl
.gitattributes      # LF normalisation — you are on Windows, CI is on Linux
README.md           # what the project is, how to run it, links to the four key docs
LICENSE             # choose deliberately; note the restricted-dependency constraint
```

**Windows note:** `.gitattributes` with `* text=auto eol=lf` prevents the CRLF/LF churn
that otherwise makes every CI diff unreadable.

### Subtask 2 — Backend package and layer enforcement

```
backend/pyproject.toml       # deps, ruff + mypy config, package metadata
backend/ruff.toml            # or inline in pyproject
backend/protloc/__init__.py
backend/protloc/<17 modules>/__init__.py
backend/protloc/domain/types.py       # the enums and dataclasses that exist today:
                                      # OrganismGroup, TaskName, JobStatus, Severity
backend/protloc/config/settings.py    # pydantic-settings reading .env
backend/tests/test_layering.py        # THE IMPORTANT ONE
```

`test_layering.py` is the phase's most valuable artifact. It walks the AST of every module
under `backend/protloc/` and asserts the dependency rules:

- `domain/` imports nothing from `protloc.*`
- no module in the serving path imports `protloc.training`
- `api/`, `web/`, `jobs/` import only `domain`, `inference`, `config`
- each layer imports only from strictly lower layers

This is what makes the architecture real rather than aspirational. Written now, it fails
loudly the first time someone takes a shortcut in Phase 8.

### Subtask 3 — Frontend scaffold and design tokens

```
frontend/package.json, vite.config.ts, tsconfig.json (strict: true)
frontend/eslint.config.js         # includes a no-hex-colour rule
frontend/index.html
frontend/src/main.tsx, App.tsx
frontend/src/styles/tokens.css    # GENERATED — see below
frontend/src/services/api.ts      # the only file allowed to call fetch()
frontend/src/types/api.ts         # placeholder until Phase 2 generates from OpenAPI
scripts/generate_tokens.py        # docs/DESIGN.md tables → tokens.css
```

`tokens.css` is **generated, never hand-edited**. `scripts/generate_tokens.py` parses the
token and palette tables in `docs/DESIGN.md` and emits the CSS custom properties for both
themes. A header comment marks the file as generated. This closes the gap where a design
system exists in a document and drifts from the code within two weeks.

The eslint rule banning raw hex in `src/**` is what enforces it.

### Subtask 4 — Environment and configuration

```
.env.example        # every variable from CLAUDE.md, ENTER_YOUR_VALUE_HERE only
configs/            # Hydra tree: data/ model/ training/ evaluation/ deployment/
configs/config.yaml # root composition
```

`settings.py` must **fail loudly at startup** on a missing required variable rather than
defaulting silently. A service that starts with a wrong `MODEL_VERSION` and serves
predictions is worse than one that refuses to start.

### Subtask 5 — Containers

```
deployment/Dockerfile.serve   # CPU. NO training dependencies installed.
deployment/Dockerfile.train   # GPU base + EMBOSS (needleall) for GraphPart
deployment/docker-compose.yml # backend + frontend for local development
```

The two Dockerfiles are separate on purpose. `Dockerfile.serve` omitting training
dependencies is what turns the "training never imports into serving" rule from a
convention into a build failure. EMBOSS goes into `Dockerfile.train` now, not in Phase 4
when GraphPart fails confusingly (`EXTERNAL_WORKFLOW.md` §8).

### Subtask 6 — CI

```
.github/workflows/ci.yml
```

Jobs, all required to pass:

| Job | Command |
|---|---|
| backend-lint | `ruff check` + `ruff format --check` |
| backend-types | `mypy backend/protloc/domain backend/protloc/config` (strict) |
| backend-test | `pytest backend/tests` — includes `test_layering.py` |
| frontend-lint | `eslint` + `prettier --check` |
| frontend-build | `tsc --noEmit` + `vite build` |
| tokens-current | regenerate `tokens.css`, `git diff --exit-code` |
| docker-build | both images build |

`tokens-current` catches a hand-edited token file. `test_layering` catches an architecture
violation. Those two jobs are the reason this phase is worth doing properly.

---

## 4. Execution notes

- **Resolve the `<PIN>` SHAs** in `docs/THIRD_PARTY.md` for anything installed this phase
  (`PROJECT_PLAN.md` §7.3, item M2). Pin versions in `pyproject.toml` and `package.json`
  exactly — no `^` or `~` ranges on anything that affects results.
- **Do not create files for phases that have not started.** An empty `models/trunk.py`
  invites someone to fill it before Phase 7. Directories with `.gitkeep`, yes; speculative
  modules, no.
- **`test_layering.py` is written before the modules it polices**, so the first violation
  is caught by a test that already existed.
- **Do not install ESM, torch or netcal yet.** They arrive in the phase that uses them.
  Phase 1's backend dependency set is FastAPI, pydantic, pydantic-settings, pytest, ruff,
  mypy — nothing heavier.

---

## 5. Verification — exit gate

Every check binary. Run from a **fresh clone** — verifying in your working directory hides
files that were never committed.

```bash
git clone <repo> /tmp/protloc-verify && cd /tmp/protloc-verify
```

| # | Check | Command | Expected |
|---|---|---|---|
| 1 | Backend installs | `pip install -e backend[dev]` | exit 0 |
| 2 | Backend lints | `ruff check backend` | `All checks passed!` |
| 3 | Formatting | `ruff format --check backend` | exit 0, no files reformatted |
| 4 | Types | `mypy backend/protloc/domain backend/protloc/config` | `Success: no issues found` |
| 5 | **Layer rules hold** | `pytest backend/tests/test_layering.py -v` | all pass |
| 6 | Frontend installs | `cd frontend && npm ci` | exit 0 |
| 7 | Frontend lints | `npm run lint` | 0 errors, 0 warnings |
| 8 | Frontend type-checks | `npx tsc --noEmit` | exit 0 |
| 9 | Frontend builds | `npm run build` | `dist/` produced |
| 10 | **Tokens are generated** | `python scripts/generate_tokens.py && git diff --exit-code` | no diff |
| 11 | Serving image builds | `docker build -f deployment/Dockerfile.serve .` | success |
| 12 | Training image has EMBOSS | `docker run --rm <train-img> which needleall` | a path |
| 13 | **Serving image lacks torch** | `docker run --rm <serve-img> python -c "import torch"` | **ModuleNotFoundError** |
| 14 | No secrets committed | `git log -p \| grep -iE "(api_key\|secret\|token)=[^E]"` | no matches |
| 15 | `.env.example` is placeholders only | `grep -v ENTER_YOUR_VALUE_HERE .env.example \| grep "="` | only non-secret defaults |
| 16 | CI green | GitHub Actions on the push | all jobs pass |

Check 13 is deliberately inverted: the serving image **must fail** to import torch at this
stage. It proves the two images are genuinely separate rather than sharing a base by
accident.

---

## 6. Handover

Fill `docs/handover/handover_phase_01.txt` using the `PHASE_PROTOCOL.md` §6 template.
Phase-specific fields to record:

- Pinned versions of every dependency installed, and the `<PIN>` SHAs resolved.
- The exact layer rules `test_layering.py` enforces (so Phase 2 does not have to read it).
- Anything in `FOLDER_STRUCTURE.md` that was changed during scaffolding, and why — the
  tree is a plan until it meets a real toolchain.

---

## 7. Failure modes for this phase

**Scaffolding beyond scope.** The commonest failure: writing a `predict()` that returns a
fake response "so the shape is visible". It becomes load-bearing, and by Phase 11 nobody
remembers it was a mock. Phase 1 creates structure and enforcement, nothing else.

**Deferring the enforcement.** Adding `test_layering.py` and `tokens-current` "once there
is something to check" means adding them after the first violations already exist, when
they fail on legacy code and get disabled. They are cheapest on an empty repo.

**CRLF churn.** Windows author, Linux CI. Without `.gitattributes`, every diff is noise
and formatting checks fail for reasons unrelated to the code.

**Dependency ranges.** A `^` in `package.json` or an unpinned `pyproject.toml` means the
scaffold that passes CI today fails in three months on a transitive bump — and worse, a
training run becomes irreproducible. Pin exactly.

**Verifying in the working directory.** A scaffold that "works" locally but omits a file
from git passes every local check and fails on a fresh clone. Always verify from `/tmp`.
