# CLAUDE.md — ProtLoc Fusion Master Instructions

> **Single source of truth for whoever executes this project — human or agent.**
> Read fully before starting any phase. Update `## Current Phase` when a phase completes.
> If this file disagrees with anything said in chat, this file wins.

---

## Project overview

- **What:** A unified protein-analysis web server and API. One sequence plus one
  organism-group selection in; seven annotation outputs out, from a shared
  protein-language-model representation, with calibrated probabilities, abstention,
  biological-consistency flags, per-residue explanation and independent evidence.
- **Who:** Saikalyan. Solo, AI-assisted execution.
- **Goal:** A public research web server whose served path produces the paper's numbers.
- **Full plan:** `PROJECT_PLAN.md` · **Repo blueprint:** `FOLDER_STRUCTURE.md` ·
  **Visual system:** `docs/DESIGN.md` · **Dependencies:** `docs/THIRD_PARTY.md`

---

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Frontend | React 18 + Vite + TypeScript | SPA. Types generated from the backend OpenAPI spec |
| Frontend styling | CSS custom properties from `docs/DESIGN.md` | `tokens.css` is generated, never hand-edited |
| Backend | FastAPI + Uvicorn (Python 3.11) | Thin routers; all logic in the module layers |
| Validation | Pydantic v2 | Every request and response has a schema |
| ML | PyTorch 2.x | Pin the exact version in the run manifest |
| Representation | ESM-2 (candidate: 650M) | Cache key includes checkpoint **and** revision |
| Adaptation | HuggingFace PEFT (LoRA) | Apache-2.0 |
| Calibration | netcal | Apache-2.0. Supplies scaling + ECE; abstention policy is ours |
| Splits | GraphPart + EMBOSS `needleall` | BSD-3-Clause. EMBOSS must be in `Dockerfile.train` |
| Config | Hydra / OmegaConf | Composable overrides, resolved config hashed per run |
| Experiment tracking | Run manifest (authoritative) + W&B (viewer) | ADR-0004. W&B is never the source of truth |
| Evidence | MMseqs2 vs Swiss-Prot snapshot + UniProt-GOA GAF | ADR-0003. Local snapshots; no network on the request path |
| Hosting | HF Spaces — Docker (backend) + Static (frontend) | ADR-0005. 2 vCPU / 16 GB / 50 GB non-persistent |
| Artifact storage | HF Hub model + dataset repos | ADR-0005. Spaces disk is non-persistent; pull at start |
| Data versioning | DVC | Pointers in git; bytes never committed |
| Container | Docker | Two images: `Dockerfile.serve` (CPU) and `Dockerfile.train` (GPU) |
| CI | GitHub Actions | lint · type-check · unit · integration · eval gate |
| Dev memory | mem0 (self-hosted) | Development tooling only. See `docs/CONTEXT_MEMORY.md` |

---

## Repository structure

See `FOLDER_STRUCTURE.md` for the full tree and the reasoning. The short form:

```
frontend/         React + Vite + TS SPA — separate deployable
backend/protloc/  17 modules, L-1 (domain) → L5 (evidence)
configs/          Hydra tree
data/             DVC-tracked; raw bytes never committed
registry/         Versioned bundles: checkpoint + calibrator + rules + config
experiments/      EXP-NNNN run manifests
benchmarks/       External tool outputs + label_mapping.yaml
scripts/          One job per script
docs/             DESIGN, ARCHITECTURE, THIRD_PARTY, adr/, handover/
deployment/       Dockerfiles, compose, IaC
```

---

## Environment variables + fill schedule

> Every secret uses `ENTER_YOUR_VALUE_HERE` in `.env.example`. **Never commit a real
> value.** Fill each one in the phase that first needs it, in a local `.env` only.

```env
# ── Phase 1: scaffold ──
APP_ENV=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
CORS_ALLOWED_ORIGINS=http://localhost:5173

# ── Phase 3: data ──
DATA_ROOT=./data
UNIPROT_RELEASE=ENTER_YOUR_VALUE_HERE
MAX_SEQUENCE_LENGTH=ENTER_YOUR_VALUE_HERE

# ── Phase 5: features ──
EMBED_CACHE=./data/embeddings
ESM_CHECKPOINT=facebook/esm2_t33_650M_UR50D
ESM_REVISION=ENTER_YOUR_VALUE_HERE
HF_TOKEN=ENTER_YOUR_VALUE_HERE

# ── Phase 6: experiment tracking ──
WANDB_API_KEY=ENTER_YOUR_VALUE_HERE
WANDB_PROJECT=protloc-fusion

# ── Phase 11: serving ──
MODEL_REGISTRY=./registry
MODEL_VERSION=1.0.0
MODEL_SERVING_MODE=queue          # ADR-0001. 'queue' is the only supported mode
MAX_BATCH_SEQUENCES=ENTER_YOUR_VALUE_HERE
INLINE_RESULT_WINDOW_SECONDS=2    # size-1 fast path; a convenience, never a contract
JOB_RETENTION_HOURS=ENTER_YOUR_VALUE_HERE

# ── Phase 10: evidence (ADR-0003 — local snapshots, no request-path network) ──
EVIDENCE_ROOT=./data/evidence
SWISSPROT_DB_PATH=./data/evidence/swissprot_mmseqs
GOA_STORE_PATH=./data/evidence/goa.db
GOA_RELEASE=ENTER_YOUR_VALUE_HERE
EVIDENCE_TIMEOUT_SECONDS=5
EVIDENCE_TOP_K=5

# ── Phase 16: hosting (ADR-0005) ──
HF_MODEL_REPO=sk8069/protloc-fusion
HF_EVIDENCE_REPO=sk8069/protloc-fusion-evidence
HF_CHECKPOINT_REPO=sk8069/protloc-fusion-checkpoints
HF_DATA_REPO=sk8069/protloc-fusion-data
ARTIFACT_CACHE_DIR=/data/artifacts
FRONTEND_ORIGIN=ENTER_YOUR_VALUE_HERE

# ── Phase 15: observability + security ──
OTEL_EXPORTER_OTLP_ENDPOINT=ENTER_YOUR_VALUE_HERE
RATE_LIMIT_PER_MINUTE=ENTER_YOUR_VALUE_HERE
MAX_UPLOAD_BYTES=ENTER_YOUR_VALUE_HERE

# ── Phase 16: deploy ──
SENTRY_DSN=ENTER_YOUR_VALUE_HERE
```

| Variable | Fill in phase | Where to get it |
|---|---|---|
| `UNIPROT_RELEASE` | 3 | The release string on the UniProt download page the day you snapshot |
| `MAX_SEQUENCE_LENGTH` | 3 | Decided from the corpus length distribution; record in the dataset card |
| `ESM_REVISION` | 5 | HuggingFace model page → commit hash of the revision you pull |
| `HF_TOKEN` | 5 | huggingface.co → Settings → Access Tokens (read scope only) |
| `WANDB_API_KEY` | 6 | wandb.ai → Settings → API keys |
| `MAX_BATCH_SEQUENCES` | 11 | Chosen from measured throughput in Phase 13; start conservative |
| `JOB_RETENTION_HOURS` | 16 | Set from storage budget; result URLs return `410` past it |
| `GOA_RELEASE` | 10 | The GAF release string on the day you snapshot UniProt-GOA |
| `FRONTEND_ORIGIN` | 16 | The frontend Static Space URL; CORS is restricted to it |
| `HF_TOKEN` (write scope) | 16 | Re-issue with write access for artifact pushes |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 15 | Your chosen tracing backend |
| `RATE_LIMIT_PER_MINUTE`, `MAX_UPLOAD_BYTES` | 15 | Set from Phase 13 load numbers |
| `SENTRY_DSN` | 16 | sentry.io → project → Client Keys |

---

## API map

Frozen in Phase 2 as OpenAPI. **The contract is asynchronous** — see
`docs/adr/ADR-0001-serving-strategy.md` (Accepted). Every prediction is a job; a single
sequence is a job of size one. There is no synchronous prediction endpoint.

| Method | Endpoint | Returns | Description | Auth |
|---|---|---|---|---|
| `GET` | `/health` | `200` | Liveness, loaded model version, queue depth | none |
| `GET` | `/version` | `200` | Registry bundle version, checksums, rule-set version | none |
| `POST` | `/predict` | **`202`** | One sequence → `{job_id, result_url, status}` | none, rate-limited |
| `POST` | `/batch` | **`202`** | FASTA upload → `{job_id, result_url, status}` | none, rate-limited |
| `GET` | `/jobs/{job_id}` | `200` | Status + progress; **results when `completed`** | none |
| `POST` | `/jobs/{job_id}/cancel` | `200` | Cancel a queued or running job | none |
| `GET` | `/jobs/{job_id}/results` | `200` | Download results (JSON or CSV) | none |
| `GET` | `/examples` | `200` | Gallery sequences with **pre-computed cached results** | none |
| `GET` | `/docs` | `200` | Swagger UI | none |

### Contract rules

1. **`POST /predict` does not return a prediction.** It returns `202 Accepted` with
   `{job_id, result_url, status: "queued"}`. Blueprint §21.3's response schema is the
   shape of `GET /jobs/{job_id}` **when `status == "completed"`** — not of any `POST`.
2. **Job states:** `queued` · `running` · `completed` · `failed` · `cancelled`.
   Batch jobs additionally report `completed_count` / `failed_count` / `total`.
3. **The fast path is a convenience, never a contract.** For a size-one job that finishes
   inside the inline window, the server may return `200` with the result directly. The
   frontend must handle `202` correctly regardless and must never assume `200`.
4. **`job_id` is a UUIDv4** — unguessable, but result URLs are not access-controlled.
   Anyone with the URL can read the result. Say this in the landing-page privacy note;
   do not imply confidentiality the system does not provide.
5. **Results expire.** Retention window set in Phase 16; `GET /jobs/{id}` returns `410
   Gone` past it, never a silent empty result.
6. **Every response** — including `202` and errors — carries `model_version`,
   `config_hash`, `rules_version` and `experiment_id`. The reproducibility footer is an
   API guarantee, not a frontend decoration.

### `202` response shape

```json
{
  "job_id": "9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04",
  "result_url": "/jobs/9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04",
  "status": "queued",
  "sequence_count": 1,
  "queue_position": 3,
  "estimated_wait_seconds": 24,
  "model_version": "1.0.0",
  "config_hash": "…",
  "rules_version": "1.2.0",
  "experiment_id": "EXP-0031"
}
```

`estimated_wait_seconds` is a best-effort estimate from measured throughput and queue
depth. It is displayed as approximate and never presented as a guarantee.

---

## Data schema (draft — frozen in Phase 2)

**Sequence record** — `accession` (PK) · `sequence_raw` · `sequence_normalized` ·
`sequence_sha256` (indexed, unique) · `length` · `organism` · `taxonomy_id` ·
`organism_group` · `source_dataset` · `source_release` · `cluster_id` (indexed) ·
`annotation_evidence_code` · seven task label columns, each nullable with an explicit
`unknown` state distinct from `negative`.

**Split registry** — `split_version` · `accession` (FK, indexed) · `partition` ·
`partitioner` · `identity_threshold`.

**Job** — `job_id` (PK) · `input_sha256` · `sequence_count` · `model_version` ·
`config_hash` · `status` · `completed` · `failed` · `created_at` (tz-aware) ·
`updated_at`.

Rules: an index on every foreign key. Times are timezone-aware. Nullable label columns
must distinguish *not annotated* from *annotated negative* — conflating them silently
corrupts the multi-task loss.

---

## Coding conventions

- **Python:** ruff (lint + format), mypy strict on `domain/` and `inference/`.
- **TypeScript:** eslint + prettier, `strict: true`.
- **Boundary rule:** no business logic in routers. `api/` parses, validates, delegates,
  returns. If deleting `api/` would break your ability to run a prediction from Python,
  the boundary is wrong.
- **Layer rule:** a module may import from a strictly lower layer only. `training/` is
  never imported by anything in the serving path. Violations fail CI.
- **No inline biology.** Consistency rules live in `rules/consistency_rules.yaml`.
  Thresholds live in the registry bundle. A biological constant hardcoded in a `.py` file
  is a bug.
- **No hardcoded τ.** The abstention threshold is read from calibrator metadata at
  runtime. A literal `0.5` in code is a bug, not a default.
- **Colors:** every color comes from `docs/DESIGN.md` tokens. A raw hex in a component is
  a review rejection.
- **Queries:** parameterized only. No `SELECT *` in production paths.
- **Secrets:** never in code, never in logs, never in a run manifest.
- **Restricted licenses:** repos marked restricted in `docs/THIRD_PARTY.md` are read,
  never copied. Never vendored, never bundled, never served.
- **Commits:** Conventional Commits (`feat`/`fix`/`chore`/`docs`/`refactor`/`test`).
  **Versioning:** SemVer, applied to the registry bundle as well as the code.

---

## Phase execution rules

1. **One phase at a time.** Do not start Phase N+1 until Phase N's exit criterion in
   `PROJECT_PLAN.md` §4 is verified — not "mostly working", verified.
2. **Confirm before proceeding.** List what was built and ask for a go-ahead.
3. **No placeholders in code.** Everything runnable. No `# TODO: implement`.
4. **Test as you build.** Each feature gets at least one verification step.
5. **Update `## Current Phase`** below after each phase.
6. **Incremental commits** after each meaningful subtask.
7. **Re-read the relevant `PROJECT_PLAN.md` section** before starting a phase.
8. **Write the handover** — `docs/handover/handover_phase_X.txt`, self-contained, before
   the phase is closed.
9. **Freeze before you measure.** Any value the paper reports — split threshold, τ, loss
   weights — is frozen on validation *before* the test set is touched. Touching test to
   tune a value invalidates the result.
10. **An ADR for every irreversible decision.** `docs/adr/`. If undoing it later would
    mean retraining or re-splitting, it needs an ADR first.

---

## Current phase

```
[x] Phase 0  — Research & prior art
[ ] Phase 1  — Scaffold & source of truth
[ ] Phase 2  — Contracts & wireframes
[ ] Phase 3  — Data corpus & manifest
[ ] Phase 4  — Homology splits & leakage gates
[ ] Phase 5  — Feature layer
[ ] Phase 6  — Baselines
[ ] Phase 7  — Vertical slice (one head, end-to-end, served)
[ ] Phase 8  — Full multi-task + LoRA
[ ] Phase 9  — Calibration & abstention
[ ] Phase 10 — Trust & evidence layers
[ ] Phase 11 — Backend service & batch
[ ] Phase 12 — Frontend
[ ] Phase 13 — Integration & performance
[ ] Phase 14 — Evaluation & external benchmarking
[ ] Phase 15 — Observability & security
[ ] Phase 16 — Governance, cost & release readiness
[ ] Phase 17 — Release & post-launch
```

### Decision status — no phase is blocked

| ADR | Decision | Status | Unblocks |
|---|---|---|---|
| 0001 | Queue everything; no synchronous prediction | Accepted | 2, 11, 12 |
| 0002 | 30% max global identity · `needleall` · 5 partitions | Accepted | 4 |
| 0003 | Self-hosted Swiss-Prot MMseqs2 + UniProt-GOA snapshots | Accepted | 10 |
| 0004 | Run manifest authoritative; W&B as viewer | Accepted | 6, 8 |
| 0005 | HF Spaces (Docker + Static) + HF Hub artifact repos | Accepted | 16, 17 |

Everything still open is a **value produced by running a phase**, not a decision waiting
on a person — see `PROJECT_PLAN.md` §7.2 for the freeze schedule and §7.3 for the three
tracked measurements.

**Phase 1 may begin immediately.**

**Open at Phase 1:** resolve every `<PIN>` commit SHA in `docs/THIRD_PARTY.md`.
