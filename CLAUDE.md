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
- **How we got here:** The 17-phase from-scratch plan was abandoned on 2026-09-28 at
  Phase 3 for lack of calendar time. The working five-screen presentation app was
  promoted into this repository instead, and the project now follows the 20-phase
  **MVP-upgrade** plan: keep the app, replace the motif engine with trained models,
  deploy. See `PROJECT_PLAN.md` §4.
- **Plan of record:** `PROJECT_PLAN.md` · **Per-phase notes:** `docs/phases/NN.md` ·
  **Visual system:** `docs/DESIGN.md` · **Dependencies:** `docs/THIRD_PARTY.md` ·
  **Data licensing:** `docs/DATA_LICENSING.md`

> ## Working folder — read this before touching a file
>
> **All work happens in `C:\Users\saika\protloc-fusion`.** It is the git repository:
> every commit, every check and every push starts there. It is the only ProtLoc folder
> that exists — `C:\Users\saika\ProtLoc` was emptied on 2026-09-28 and its two
> survivors are now `docs/ProtLoc_Documentation.pdf` and `docs/legacy/`.
>
> If a path in any document or chat says `ProtLoc\...`, read it as `protloc-fusion\...`.

---

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Frontend | React 18 + Vite + TypeScript | SPA. Types generated from the backend OpenAPI spec |
| Frontend styling | CSS custom properties from `docs/DESIGN.md` | `tokens.css` is generated, never hand-edited |
| Backend | FastAPI + Uvicorn (Python 3.11) | Thin routers; all logic in the module layers |
| Validation | Pydantic v2 | Every request and response has a schema |
| ML | PyTorch 2.x for training only; **numpy for serving** (ADR-0016) | Pin the exact version in the run manifest. The served heads are numpy forward passes with a torch parity test; serving never imports `training/` |
| Representation | **ESM-2 150M** (`facebook/esm2_t30_150M_UR50D`) — 30 layers, hidden dim 640, MIT | Revised 2026-10-02. 650M was the old pick; DeepLoc's authors tested 650M **and** 15B and reported no improvement on this task, and 150M matched 3B across 36–40 DMS datasets. `esm2_t12_35M_UR50D` is the dev tier. ESM-C 300M (MIT) is the one swap worth a same-day test once caching is encoder-generic |
| Adaptation | **None — frozen encoder, trained heads** | Revised 2026-10-02. LoRA and full fine-tuning are both out of budget for one week. Attention pooling + MLP heads per DeepLoc's published recipe |
| Loss | Per-label weighted focal loss, averaged over labels | DeepLoc 2.x recipe |
| Calibration | Per-label isotonic (primary), temperature scaling (fallback); `netcal` for ECE | Apache-2.0. Multi-label calibration is marginal, not multi-class — one global temperature cannot fix per-label miscalibration when base rates differ |
| Splits | GraphPart, 5 partitions, 30% max global identity | BSD-3-Clause. `mmseqs2needle` backend, not plain `needle` — 25k sequences is ~3.1e8 pairs |
| Config | Hydra / OmegaConf | Composable overrides, resolved config hashed per run |
| Experiment tracking | Run manifest (authoritative) + W&B (viewer) | ADR-0004. W&B is never the source of truth |
| Evidence | MMseqs2 vs Swiss-Prot snapshot + UniProt-GOA GAF | ADR-0003. Local snapshots; no network on the request path. **ENH** — k-mer fallback if the index does not fit the host |
| Hosting | **Decided in Phase 2 — ADR-0007** | ADR-0005 is superseded: HF **Docker** Spaces went paid (~8–9 July 2026, PRO $9/mo); only Static Spaces are free. Candidates: PRO + CPU Basic (2 vCPU / 16 GB / 50 GB ephemeral), or Cloud Run at 2 GiB on the always-free allowance. Render Free and Koyeb are rejected on 512 MB / 0.1 vCPU |
| Artifact storage | HF Hub model repo under `sk8069`, pinned by commit SHA | Never git: GitHub LFS Free is 10 GiB storage + 10 GiB bandwidth per cycle |
| GPU | Colab / Kaggle, for embedding extraction only | HF ZeroGPU is Gradio-only and 5 min/day free — architecturally unusable here |
| Container | Docker | `Dockerfile.serve` (CPU) and `Dockerfile.train` (GPU) |
| CI | GitHub Actions | lint · type-check · unit · integration · spec-drift · eval gate |

---

## Repository structure

`FOLDER_STRUCTURE.md` was deleted on 2026-09-28 — this section replaces it. The tree as
it actually stands:

```
frontend/src/        React + Vite + TS SPA
  App.tsx            sidebar shell, 5 screens, no router dependency
  pages/             Triage · Inspector · Evidence · Scanner · Trust
  components/        ui.tsx (Chip, ConfidenceRow, Flags, ProvenanceBar, Card)
                     SequenceTrack.tsx (attribution / hydropathy / alignment)
  services/          api.ts (scaffold client) · demoApi.ts (202-then-poll, in use)
  styles/            tokens.css (generated) · app.css · tokens.demo.css
  types/             api.ts · api.generated.ts (generated from the OpenAPI spec)
backend/protloc/
  domain/            contracts.py (frozen HTTP contract) · ports.py · types.py
  api/               app.py · app_v2.py (the live app) · deps.py · routers/
  features/          classical.py — the motif engine; becomes the hand-crafted
                     feature branch beside the PLM embeddings, not deleted
  inference/         engine.py (the trained pipeline: features -> model -> calibration ->
                     consistency -> evidence; the only place that order is written) · trust.py ·
                     motif.py + service.py (the Phase 0 motif engine, kept for the scanner,
                     membrane type, attribution and the frozen golden set)
  models/            attention_head.py · crf_tagger.py — numpy forward passes of the frozen heads
  evidence/          ontology.py · search.py (MMseqs2 subprocess, deadline, degraded state)
  jobs/              queue.py — in-memory 202 + poll + cancel + retention
  data/              build.py · parse.py · validate.py · manifest.py · rules.py
                     reference_demo.py (synthetic; deleted at Phase 6)
  config/            settings.py
backend/rules/       consistency.json · organism_groups.yaml ·
                     residue_policy.yaml · uniprot_label_map.yaml
backend/tests/       unit/ · integration/ · golden/ · test_layering.py
configs/             Hydra tree — data · model · training · evaluation · deployment
deployment/          Dockerfile.serve · Dockerfile.train · docker-compose.yml
docs/                DESIGN · ARCHITECTURE · THIRD_PARTY · DATA_LICENSING ·
                     adr/ · api/openapi.yaml · diagrams/ · phases/ · legacy/ ·
                     ProtLoc_Documentation.pdf
scripts/             one job per script
```

Directories the old plan created and this one does not need are gone: `baselines/`,
`benchmarks/`, `experiments/`, `notebooks/`, `registry/`, `data/`, `_phase3_hold/`,
`docs/handover/`, `docs/phases/` (old content). `data/`, `registry/` and
`experiments/` are recreated by the phase that first writes to them. `data/` and
`experiments/` stay gitignored; `registry/localization/<version>/*.json` (bundle and
calibrator, ADR-0012) is committed, weights and pickles are not.

---

## Environment variables + fill schedule

> Every secret uses `ENTER_YOUR_VALUE_HERE` in `.env.example`. **Never commit a real
> value.** Fill each one in the phase that first needs it, in a local `.env` only.
> Phase numbers below are the 20-phase plan in `PROJECT_PLAN.md` §4.

```env
# ── Phase 1: repository ──
APP_ENV=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8010
CORS_ALLOWED_ORIGINS=http://localhost:5174

# ── Phase 5: data corpus ──
DATA_ROOT=./data
UNIPROT_RELEASE=2026_03
MAX_SEQUENCE_LENGTH=ENTER_YOUR_VALUE_HERE

# ── Phase 7: embeddings ──
EMBED_CACHE=./data/embeddings
ESM_CHECKPOINT=facebook/esm2_t30_150M_UR50D
ESM_REVISION=a695f6045e2e32885fa60af20c13cb35398ce30c
HF_TOKEN=ENTER_YOUR_VALUE_HERE

# ── Phase 8: experiment tracking (ENH) ──
WANDB_API_KEY=ENTER_YOUR_VALUE_HERE
WANDB_PROJECT=protloc-fusion

# ── Phase 14: serving ──
MODEL_REGISTRY=./registry
MODEL_VERSION=ENTER_YOUR_VALUE_HERE
MODEL_SERVING_MODE=queue          # ADR-0001. 'queue' is the only supported mode
MAX_BATCH_SEQUENCES=ENTER_YOUR_VALUE_HERE
INLINE_RESULT_WINDOW_SECONDS=2    # size-1 fast path; a convenience, never a contract
JOB_RETENTION_HOURS=1

# ── Phase 13: evidence (ADR-0003 — local snapshots, no request-path network · ENH) ──
EVIDENCE_ROOT=./data/evidence
SWISSPROT_DB_PATH=./data/evidence/swissprot_mmseqs
GOA_STORE_PATH=./data/evidence/goa.db
GOA_RELEASE=ENTER_YOUR_VALUE_HERE
EVIDENCE_TIMEOUT_SECONDS=5
EVIDENCE_TOP_K=5

# ── Phase 2 + 19: hosting (ADR-0007) ──
HF_MODEL_REPO=sk8069/protloc-fusion
ARTIFACT_CACHE_DIR=ENTER_YOUR_VALUE_HERE
ARTIFACT_REVISION=ENTER_YOUR_VALUE_HERE
FRONTEND_ORIGIN=ENTER_YOUR_VALUE_HERE
BACKEND_URL=ENTER_YOUR_VALUE_HERE

# ── Phase 17: observability + limits (ENH) ──
RATE_LIMIT_PER_MINUTE=ENTER_YOUR_VALUE_HERE
MAX_UPLOAD_BYTES=ENTER_YOUR_VALUE_HERE
```

| Variable | Fill in phase | Where to get it |
|---|---|---|
| `MAX_SEQUENCE_LENGTH` | 5 | Decided from the corpus length distribution; record in the dataset card |
| `ESM_REVISION` | 7 | Filled 2026-10-03: `a695f604…` (huggingface.co/api/models); do not float it |
| `HF_TOKEN` | 7 (read) · 19 (write) | huggingface.co → Settings → Access Tokens |
| `MODEL_VERSION` | 14 | Filled 2026-10-05: the composite of the four registry bundle versions (`localization-0.10.0+signal_peptide-0.11.0+go_function-0.12.0+evidence_homology-0.13.0`); the service refuses to start if it differs from what is loaded |
| `ARTIFACT_REVISION` | 19 | The HF model-repo commit SHA the server pins (left for Phase 19, Gate D) |
| `ARTIFACT_CACHE_DIR` | 2 | Host-dependent; ephemeral on both candidates, so weights are re-pulled on cold start |
| `MAX_BATCH_SEQUENCES` | 14 | Filled 2026-10-05: **20**, half the largest batch measured (40) on one 12-core machine (`data/manifests/serving_measurements.json`); re-measure in Phase 19 |
| `GOA_RELEASE` | 13 | Unused by the Phase 13 design: GO terms for hits come from the corpus `DR GO` lines, recorded as "embedded in Swiss-Prot 2026_03" (hash = corpus SHA-256); Phase 14 dropped it from the serving contract; it is never read |
| `FRONTEND_ORIGIN`, `BACKEND_URL` | 19 | The two deployed URLs; CORS is restricted to the first |
| `RATE_LIMIT_PER_MINUTE`, `MAX_UPLOAD_BYTES` | 17 | From Phase 16 load numbers |

`UNIPROT_RELEASE` is pinned to `2026_03` (02-Sep-2026, 575,748 entries) and is not a
blank to fill: that release revised annotations on 403,255 entries while adding only
247 sequences, so a floating release makes "reproducible" false.

`SENTRY_DSN`, `OTEL_EXPORTER_OTLP_ENDPOINT` and `DVC` are dropped. Tracing a single-
instance free-tier service is not worth the week.

---

## API map

**Frozen.** The authoritative document is `docs/api/openapi.yaml`, generated from the
FastAPI application by `scripts/export_openapi.py` and diffed by CI. The table below is
the human-readable index of it; where the two disagree, the document is right and the
table is a bug. **The contract is asynchronous** — see
`docs/adr/ADR-0001-serving-strategy.md` (Accepted). Every prediction is a job; a single
sequence is a job of size one. There is no synchronous prediction endpoint.

| Method | Endpoint | Returns | Description | Auth |
|---|---|---|---|---|
| `GET` | `/health` | `200` | Liveness, loaded model version, queue depth | none |
| `GET` | `/healthz` | `200` | Model-registry state (`loaded` / `empty`) for platform probes | none |
| `GET` | `/version` | `200` | Registry bundle version, checksums, rule-set version | none |
| `POST` | `/predict` | **`202`** | One sequence → `{job_id, result_url, status}` | none, rate-limited |
| `POST` | `/batch` | **`202`** | FASTA upload → `{job_id, result_url, status}` | none, rate-limited |
| `GET` | `/jobs/{job_id}` | `200` | Status + progress; **results when `completed`** | none |
| `POST` | `/jobs/{job_id}/cancel` | `200` | Cancel a queued or running job | none |
| `GET` | `/jobs/{job_id}/results` | `200` | Download results (JSON or CSV) | none |
| `GET` | `/examples` | `200` | Gallery sequences with **pre-computed cached results** | none |

`GET /docs` serves Swagger UI. It is a rendering of the contract, not part of it, and it
is deliberately absent from `docs/api/openapi.yaml`.

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

## Data schema (frozen 2026-09-14, Phase 2 · amended 2026-09-15, Phase 3, ADR-0006)

**Sequence record** — `accession` (PK) · `entry_name` · `sequence_raw` ·
`sequence_normalized` · `sequence_sha256` (indexed, **not unique** — identical
sequences under different accessions are real) · `length` · `is_fragment` ·
`organism` · `taxonomy_id` · `organism_group` · `source_dataset` · `source_release` ·
`labels`: one entry per task (all seven, always present), each
`{state, values, cleavage_site, evidence}`.

- `state` ∈ `annotated` · `negative` · `unknown` · `not_applicable`
  (`protloc.domain.types.LabelState`). `values` is non-empty exactly when
  `annotated`. Where a `negative` may come from is fixed per task by ADR-0006 §3.3.
- `evidence` is per label value: ECO codes for locations and features, GO evidence
  codes for GO terms. The old single `annotation_evidence_code` column is removed.
- `cluster_id` moved to the split registry (it is produced in Phase 4).
- The corpus is canonical JSON Lines — `data/processed/corpus.jsonl` — and its
  SHA-256 is the dataset hash (`docs/phases/05.md`).

**Split registry** — `split_version` · `accession` (FK, indexed) · `partition` ·
`cluster_id` (indexed) · `partitioner` · `identity_threshold`.

**Job** — `job_id` (PK) · `input_sha256` · `sequence_count` · `model_version` ·
`config_hash` · `status` · `completed` · `failed` · `created_at` (tz-aware) ·
`updated_at`.

Rules: an index on every foreign key. Times are timezone-aware. A label's state must
distinguish *not annotated* (`unknown`) from *annotated negative* (`negative`) and
from *does not apply* (`not_applicable`) — conflating them silently corrupts the
multi-task loss.

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
- **React only.** The frontend is React + Vite + TypeScript with plain CSS driven by the
  generated tokens. No CSS framework, no component library, no styling or UI runtime
  beyond React and its type packages. A dependency that renders UI is a decision, and
  decisions become ADRs.
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
5. **Update `## Current phase`** below after each phase — tick the box and record the result.
6. **Incremental commits** after each meaningful subtask.
7. **Re-read the relevant `PROJECT_PLAN.md` section** before starting a phase.
8. **Write the phase note** — `docs/phases/NN.md`, self-contained, before the phase is
   closed. `docs/handover/` was deleted on 2026-09-28; the phase note replaces it and
   doubles as the GitHub Release body.
9. **Freeze before you measure.** Any value the paper reports — split threshold, τ, loss
   weights — is frozen on validation *before* the test set is touched. Touching test to
   tune a value invalidates the result.
10. **An ADR for every irreversible decision.** `docs/adr/`. If undoing it later would
    mean retraining or re-splitting, it needs an ADR first.

---

## Current phase

**Plan of record: the 20-phase MVP-upgrade plan.** Full table with deliverables, exit
criteria, hours and git commands in `PROJECT_PLAN.md` §4; per-phase detail in
`docs/phases/NN.md`. `MVP` is the minimum viable path, `ENH` is droppable.

```
[x] Phase 0   — Cleanup & restructure (one repo, MVP promoted)      done 2026-09-28
[x] Phase 1   — Repository init, licensing, CI green          MVP   2.0 h   done 2026-10-02
[x] Phase 2   — Host decision, supersede ADR-0005             MVP   3.0 h   done 2026-10-02 (v0.2.0)
[x] Phase 3   — Widen contract: provenance + calibration      MVP   2.5 h   done 2026-10-02 (v0.3.0)
[x] Phase 4   — Motif-engine test harness & golden set        MVP   3.0 h   done 2026-10-02 (v0.4.0)
[x] Phase 5   — Swiss-Prot corpus, labels, manifest           MVP   8.0 h   done 2026-10-02 (branch; you tag v0.5.0)
[x] Phase 6   — GraphPart splits & leakage gate               MVP   5.0 h   done 2026-10-03 (branch; you tag v0.6.0)
[x] Phase 7   — ESM-2 embedding cache                         MVP   6.0 h   done 2026-10-03 (branch; 500-sequence subset only; you tag v0.7.0)
[x] Phase 8   — Classical baselines (metric floor)            ENH   3.0 h   done 2026-10-03 (branch; you tag v0.8.0)
[x] Phase 9   — Localization head (attention pooling + MLP)   MVP   7.0 h   done 2026-10-03 (branch; 500-sequence subset only; you tag v0.9.0)
[x] Phase 10  — Calibration & per-label thresholds            MVP   5.0 h   done 2026-10-03 (branch; 500-sequence subset only; you tag v0.10.0)
[x] Phase 11  — Signal-peptide CRF head                       ENH   8.0 h   done 2026-10-04 (branch; 500-sequence subset only; you tag v0.11.0)
[x] Phase 12  — GO head                                       ENH   4.0 h   done 2026-10-04 (branch; 500-sequence subset only; you tag v0.12.0)
[x] Phase 13  — Homology evidence (MMseqs2 index)             ENH   5.0 h   done 2026-10-04 (branch; 500-sequence subset queries, full Swiss-Prot database; you tag v0.13.0)
[x] Phase 14  — Swap the engine: model behind the same API     MVP   6.0 h   done 2026-10-05 (branch; 500-sequence subset only, cache-only encoder; you tag v0.14.0)
[x] Phase 15  — Attribution from the real model                ENH   4.0 h   done 2026-10-05 (branch; 500-sequence subset only; you tag v0.15.0)
[ ] Phase 16  — Frontend upgrade & landing page                MVP   7.0 h   <-- NEXT
[ ] Phase 17  — Limits, errors, degraded states                ENH   3.0 h
[ ] Phase 18  — Nightingale visual rebuild                     ENH   5.0 h
[ ] Phase 19  — Deploy backend + frontend                      MVP   5.0 h
[ ] Phase 20  — Eval report, docs, v1.0.0 release              MVP   4.0 h
```

93.5 h planned against roughly 84 h achievable in the week. The MVP path —
1-7, 9, 10, 14, 16, 19, 20 — is 57.5 h. Drop from the bottom of the ENH list first.

**What cannot be done in a week, stated plainly.** Durable whole-proteome batch mode. Both
hosting candidates have ephemeral disk and the queue is in-memory, so a restart loses every
job. ADR-0008 scopes batch mode to client-side chunking into many short requests and the
docs say so rather than promising durability the host cannot deliver.

### Decision status

| ADR | Decision | Status | Unblocks |
|---|---|---|---|
| 0001 | Queue everything; no synchronous prediction | Accepted · amended by 0008 | 3, 14 |
| 0002 | 30% max global identity · 5 partitions | Accepted · backend changed to `mmseqs2needle` | 6 |
| 0003 | Self-hosted Swiss-Prot MMseqs2 + UniProt-GOA snapshots | Accepted · ENH, k-mer fallback allowed | 13 |
| 0004 | Run manifest authoritative; W&B as viewer | Accepted | 8, 9 |
| 0005 | HF Spaces (Docker + Static) + HF Hub artifact repos | **Superseded 2026-10-02** — Docker Spaces are paid | — |
| 0006 | Label semantics: four states, per-label evidence, per-task negative rule | Accepted 2026-10-02 | 5, 9, 20 |
| 0007 | Hosting, post-paywall | Accepted for the Phase 2 acceptance host; **host for the real service undecided** | 19, 20 |
| 0008 | Batch durability scoped to client-side chunking | Accepted | 14, 16 |
| 0009 | Frozen encoder + trained heads; ESM-2 150M; sliding window for sequences over 1022 residues | Accepted 2026-10-03 (measured cost filled at the Phase 7 close) | 7, 9 |
| 0010 | Baseline protocol: leave-one-partition-out on registry partitions 1-4, per-class MCC-optimal thresholds, test partition 5 scored once after freeze | Accepted 2026-10-03 | 8, 9 |
| 0011 | Localization head: attention pooling over per-residue ESM-2 states + MLP; per-residue cache for the 500-sequence subset; inner whole-cluster early stopping; subset-only numbers | Accepted 2026-10-03 | 9, 10 |
| 0012 | Per-label calibration and thresholds frozen in the registry bundle; loader refuses a hash mismatch | Accepted 2026-10-03 | 10, 14 |
| 0013 | Signal-peptide CRF head: CNN encoder + linear-chain CRF, tags O/S, cap 100 | Accepted 2026-10-04 | 11, 14 |
| 0014 | GO head: multi-label molecular function, evidence policy experimental + curated, term floor 20 (12 terms), open-world labels, three floors, subset-only numbers | Accepted 2026-10-04 | 12, 13, 14 |
| 0015 | Homology evidence: local MMseqs2 index, GO transfer from hits, never feeds a score | Accepted 2026-10-04 | 13, 14 |
| 0016 | Serving engine: numpy heads over cached ESM-2 states, served groups, one engine config, refuse to start | Accepted 2026-10-05 | 14, 15, 19 |

### Phase 0 close — 2026-09-28

`C:\Users\saika\ProtLoc` emptied; its PDF and notes preserved as
`docs/ProtLoc_Documentation.pdf` and `docs/legacy/`. ~200 MB of caches and both
`node_modules` removed. 13 empty `backend/protloc/*` packages and the empty frontend
scaffold deleted. The five-screen app promoted: motif engine to
`features/classical.py`, service to `inference/service.py`, queue to `jobs/queue.py`,
app to `api/app_v2.py`, rules to `rules/consistency.json`, and all five pages plus
`ui.tsx` and `SequenceTrack.tsx` into `frontend/src/`.

### Phase 3 close — 2026-10-02

The contract carries provenance: per task `source`, `threshold`, `calibrated`,
`confidence_band`; per result `no_confident_call`, `calibration_version`,
`threshold_policy`. The motif engine reports `motif`, its real tau (0.67), uncalibrated.
`threshold` has no literal default (a legacy payload reads it from `abstention_threshold`).
Deviations and UNVERIFIED items: `docs/phases/03.md` §8.

### Phase 4 close — 2026-10-02

`backend/tests/golden/golden_set.json` locks the motif engine: 67 entries (10 gallery,
23 synthetic, 34 engineered, 0 real), full recorded output plus the engine identity and
TRUST fit. Test-only, never served. A changed engine fails `pytest tests/golden` with a
per-sequence, per-field diff; `python scripts/regen_golden.py --write` is the only way to
rewrite it. No real protein is in it (UniProt was unreachable): `docs/phases/04.md` §7.

### Phase 5 close — 2026-10-02

Swiss-Prot release `2026_03` (02-Sep-2026) is snapshotted and built: 548,702 records from
575,748 entries, corpus SHA-256 `d493478d38630651490adfe91c7869d1a6ac8ab759effdc3847ce84e295d41a8`,
identical across three builds; `MAX_SEQUENCE_LENGTH` frozen at 1496 (quantile 0.99; 5,478
records longer, kept in the corpus). Manifest, dataset card, build report and class counts are in
`data/manifests/`; the bytes are gitignored. ADR-0006 is Accepted. UNVERIFIED: the ESM-2 context
(1022) and cross-platform hash reproducibility. Detail: `docs/phases/05.md`.

### Phase 6 close — 2026-10-03

Working set: 35,443 accessions (34,965 distinct sequences), the eukaryotic localization records with
ECO:0000269 evidence (narrower than the plan; deviation, `docs/phases/06.md` §3). GraphPart 1.0.2
(`mmseqs2needle` backend, 30% identity, 5 partitions) removed 1,557 accessions; the independent leakage
check then found 7 cross-partition pairs above 30% global identity, so 7 more were removed by a recorded
rule: 1,564 removed in all (4.41%). Partitions 9,373 / 6,060 / 6,008 / 5,885 / 6,553 (33,879 accessions).
Registry SHA-256 `37ca5e0583b783a6aa4945f582fa2342b9fdf5b1008f68f71e9eb87869de276a` (gitignored); the
leakage report passes with 0 duplicates and 0 cross-partition pairs. Raw GraphPart runs are not
deterministic; the frozen split uses sorted edges (§4). The calibration split must be made from whole
`cluster_id` clusters. UNVERIFIED: the leakage zero is bounded by a heuristic search; cross-machine
reproducibility. Detail: `docs/phases/06.md`.

### Phase 7 close — 2026-10-03

ESM-2 150M pinned at `a695f6045e2e32885fa60af20c13cb35398ce30c` (ADR-0009, hidden size 640, 1022-residue context,
sliding window for the 2,060 longer sequences). **Only a 500-sequence subset is embedded**, spread over the length
distribution (partitions 144/77/89/92/98); the full 33,461-sequence cache is outstanding (about 19 h on the CPU,
an estimate; `scripts/embed.py --pack`/`--bundle` prepare a free-GPU run). Measured on a Windows CPU, torch
2.14.1+cpu: 2.011 s/sequence; subset cache 1,315,501 bytes; one sequence byte-identical across two processes and
after a real kill and resume; padding changes a vector by at most 9.5e-07. Byte identity is claimed for that machine
and device only. Manifest `data/manifests/embedding_manifest.json` says `subset`. Detail: `docs/phases/07.md`.

### Phase 8 close — 2026-10-03

LogReg, LinearSVC and RandomForest on a 29-column motif vector, plus the untrained motif rules as a second row (ADR-0010).
Protocol: leave-one-partition-out over registry partitions 1-4, per-class MCC-optimal thresholds tuned on out-of-fold scores,
test = partition 5 scored once after `--freeze` (`data/manifests/baseline_metrics.json`, frozen_sha256 `60bdfcd2f2b7173e5e8e21a66a106db91dab126698eeebeecd919908c7a664f4`).
Localization macro F1, validation / test: RandomForest 0.479 / 0.472 (the floor Phase 9 must beat), LogReg 0.440 / 0.458,
LinearSVC 0.437 / 0.456, motif rules 0.364 / 0.365. RF Jaccard 0.492 / 0.467. Membrane type RF 0.525 / 0.561; signal peptide RF 0.936 / 0.931.
DeepLoc (about 0.46 macro F1, about 0.53 Jaccard) is indicative only: different benchmark and split. `configs/evaluation/default.yaml` is not
edited; it numbers partitions 0-4 where the registry says 1-5 (n = index + 1), so Phase 9 reconciles it. `data/baselines/` is gitignored.
UNVERIFIED: Docker, GitHub Actions and the frontend gate (not runnable in the sandbox). Detail: `docs/phases/08.md`.

### Phase 9 close — 2026-10-03

Attention pooling over per-residue ESM-2 states plus an MLP, 10 sigmoids (ADR-0011), trained and scored on the **500-sequence subset only**
(every number: "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). A float16 per-residue cache for the subset was added
(286 MB, mean of states within 8.6e-4 of the Phase 7 pooled vectors). Chosen: lr 1e-3, dropout 0.5, hidden 128, gamma 0, 125,707 parameters.
Same protocol as ADR-0010; manifest `data/manifests/head_metrics.json`, frozen_sha256 `719c9866603640a781bf13193e38ee5d27c7fd13426aa44c5221a1e4099c9dae`.
Localization macro F1 validation / test: head 0.476 [0.428, 0.516] / 0.366 [0.302, 0.427]; subset floor LogReg 0.415 / 0.363, RandomForest 0.408 / 0.347,
LinearSVC 0.368 / 0.343, motif rules 0.357 / 0.230. Test Jaccard: head 0.330, LogReg 0.404, RF 0.458. **The head does not beat the floor on test** (macro F1 a tie
inside wide intervals, Jaccard lower); 99 test accessions, one seed. `configs/evaluation/default.yaml` now uses registry numbering ([1,2,3,4] / [5]) with a test.
UNVERIFIED: any claim beyond 500 sequences, seed variance, DeepLoc parity, Docker, GitHub Actions and the frontend gate (not runnable in the sandbox).
Detail: `docs/phases/09.md`.

### Phase 10 close — 2026-10-03

Per-label calibration (isotonic for 6 labels, Platt for 4, else identity) cross-fitted on the Phase 9 head's out-of-fold scores, plus MCC-tuned thresholds frozen in
`registry/localization/0.10.0/{bundle.json,calibrator.json}` (ADR-0012; `bundle_sha256` `c341da61936383928ff07765a28577b40a1e66860ede8629455f687830beb62c`; loader refuses any hash mismatch).
**500-sequence subset only** ("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). Manifest `data/manifests/calibration_metrics.json`,
frozen_sha256 `ee61452da32acde9d94cbf1c8a2f5b8689e85e128711ca49efb910329c779877`. Validation, pooled over partitions 1-4: macro ECE 0.1171 -> 0.0421 (delta CI [-0.0899, -0.0583]),
macro Brier 0.0928 -> 0.0688, macro F1 0.4763 -> 0.4578 (CIs overlap), Jaccard 0.4696 -> 0.5088. Cytoplasm and nucleus show no significant ECE change; peroxisome (6 positives) is
Platt in the frozen fit but uncalibrated in two folds. Partition 5 was not read (the Phase 9 head touched it once). `data/calibration/` is gitignored.
UNVERIFIED: any claim beyond 500 sequences, per-label claims on 6-18 positives, seed variance, DeepLoc parity, Docker, GitHub Actions and the frontend gate (not runnable in the sandbox).
Detail: `docs/phases/10.md`.

### Phase 11 close — 2026-10-04

Signal-peptide CRF head: a CNN encoder and a linear-chain CRF (plain torch) over frozen per-residue ESM-2 states, tags O/S, length cap 100, selected from an 8-point grid (ADR-0013).
Transit-peptide tags are undefined and not trained (18 mitochondrial and 9 chloroplast validation positives on the subset, floor 20). **500-sequence subset only**
("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). Frozen in `registry/signal_peptide/0.11.0/{bundle.json,calibrator.json}`
(`bundle_sha256` `91830aa7c14d36a389dc8327d7bd283a7b3f0c5ab66d5b78038a1c3d9ac3b169`; `head.frozen_sha256` `1398bdd1de380d694422de88aaab407ad41f47012e00c33623c53cb290064891` is the training record's hash).
Manifest `data/manifests/crf_metrics.json`, frozen_sha256 `f5f2728cfbe7027e4c861ec4bb712914a1d2f86ac3fb8b30eb7c944034bbd3be`; weights `5d08a194b97789799a443c4f1c214285a373fc7ae3dce5e374a13c8994d7285c`.
Chosen: CNN, lr 1e-3, dropout 0.5 (cleavage +-1 selector 0.8542). Validation pooled over partitions 1-4 (83 positives), CRF vs the motif floor: cleavage +-1 0.843 [0.762, 0.918] vs 0.542 [0.429, 0.645] (intervals disjoint);
detection MCC 0.923 [0.838, 0.965] vs 0.815 [0.672, 0.884], AUC 0.985 vs 0.964, per-residue F1 0.913 vs 0.788 (intervals overlap). Partition 5 scored once (`touches: 1`; 95 rows, 12 positives): CRF MCC 0.915, cleavage +-1 0.833, floor 0.704 and 0.667, intervals very wide;
the localization head's count is still 1. Training took 17,612 s on CPU. Keys: registry `37ca5e0583b783a6aa4945f582fa2342b9fdf5b1008f68f71e9eb87869de276a`, corpus `d493478d38630651490adfe91c7869d1a6ac8ab759effdc3847ce84e295d41a8`,
embedding cache `0e528d4a46831b052b6232333be803d2fd9f11f3902d2e743d53c5dfc70be0d1`, residue cache `05c9bb393be0e8b25a1af2eddcea811bc3d19edafe18580c34eb547b444a3349`. `data/crf/` is gitignored; the registry bundle has a per-task head contract (`TASK_HEAD_KEYS`).
TargetP's 83% / 46-49% are context only (`comparable: false`). UNVERIFIED: any claim beyond 500 sequences or one organism group, partition 5 (12 positives), seed variance, TargetP parity,
10 sandbox golden-test hydropathy failures (unattributed), Docker, GitHub Actions and the frontend gate (not runnable in the sandbox). Detail: `docs/phases/11.md`.

### Phase 12 close — 2026-10-04

GO molecular-function head: attention pooling over per-residue ESM-2 states plus an MLP, 12 sigmoids, evidence policy experimental + curated, term floor 20 (ADR-0014), trained and scored on the **500-sequence subset only**
("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). Labels are open-world (`annotated`/`unknown`, never `negative`); counts and the vocabulary come from subset rows only (12 trainable terms, 705 under the floor).
Ontology `releases/2026-07-26`, SHA-256 `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc` (CC BY 4.0, pointer and hash only). Frozen in `registry/go_function/0.12.0/{bundle.json,calibrator.json,terms.json}`
(`bundle_sha256` `c34a1befad3ba5e01d91998197cdb418acd5263fa13a5ecc5541821a3b627d07`; `head.frozen_sha256` `6b3ee6f7bc835da9d7867fcdfc518d58a9327634aa6c5bd00c0334f7930affb8` is the training record's hash; vocabulary `ee8e9de5dd1f0a60ef8a31eb77b666b09e5ba6dd071e8641a15f511e603a91ba`).
Manifest `data/manifests/go_metrics.json`, frozen_sha256 `9fc6057c291e04bd66b137844cda8810f2abda12a4292c90a19e6360ed470e70`; weights `1d9d77f93861444b551a058317d9c0c9175818c02bf7475bfca87bab4d0eac6f`; calibrator `956e55da98c4749806f7c9cbabd7f23821e24e38c0599295ff427cb5da59a691`.
Chosen: lr 1e-3, dropout 0.5, hidden 256, gamma 0 (one-value axis, 8 grid points), 209,549 parameters; training took 773.7 s on CPU. Validation pooled over partitions 1-4 (209 labelled rows), head vs floors:
p@3 0.592 [0.551, 0.645] vs frequency prior 0.523 [0.487, 0.552], logreg 0.460 [0.417, 0.506], motif 0.493 [0.456, 0.525]; Fmax (F at a nested global threshold) 0.609 [0.576, 0.654] vs 0.531, 0.500, 0.529 (intervals disjoint for all three).
Calibration (isotonic, 12 terms): macro ECE 0.1240 -> 0.0980 (delta CI [-0.0429, -0.0096]), Brier 0.1593 -> 0.1485; ranking unchanged. Partition 5 scored once (`touches: 1`; 64 labelled rows, 52 evaluable): head p@3 0.583 [0.484, 0.667], calibrated 0.635 [0.540, 0.712],
Fmax 0.622 / 0.632, frequency prior 0.571 / 0.568; the head overlaps the prior and the motif on test. The CRF's and the localization head's counts are still 1 each. Keys: registry `37ca5e0583b783a6aa4945f582fa2342b9fdf5b1008f68f71e9eb87869de276a`,
corpus `d493478d38630651490adfe91c7869d1a6ac8ab759effdc3847ce84e295d41a8`, embedding cache `0e528d4a46831b052b6232333be803d2fd9f11f3902d2e743d53c5dfc70be0d1`, residue cache `05c9bb393be0e8b25a1af2eddcea811bc3d19edafe18580c34eb547b444a3349`.
`data/go/` is gitignored; `TASK_HEAD_KEYS["go_function"]` adds the vocabulary and ontology checks. Phase 14 needs a model-shaped GO output with `source: "model"` (`GoAnnotation` requires `source_accession`).
UNVERIFIED: any claim beyond 500 sequences, per-term claims on 22-111 positives, seed variance, published GO parity (none fetched), Docker, GitHub Actions and the frontend gate (not runnable in the sandbox). Detail: `docs/phases/12.md`.

### Phase 13 close — 2026-10-04

Homology evidence (ADR-0015): an MMseqs2 `18-8cc5c` index over the full Swiss-Prot 2026_03 snapshot (548,702 records) and GO transfer from the nearest hits, scored on the **500-sequence subset queries**
("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers") under the Phase 12 protocol (12-term vocabulary, same rows, folds, nested thresholds, 1,000-sample cluster bootstrap, seed 20260909). Native Windows binary (MIT, licence read 2026-10-04);
it needs `sh.exe` = a byte copy of the shipped `busybox.exe` (Gate D fix, hash recorded). GO terms for hits are the corpus `DR GO` lines mapped by the Phase 12 code (ontology `releases/2026-07-26`, `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`); no GAF.
Measured: build 72.0 s; index 2.29 GiB (2,463,105,431 bytes); peak working set 2.16 GiB, peak commit 4.03 GiB; batch search of 507 queries 19.8 s; single-query p50 2.31 s, p95 2.80 s, 1.655 CPU-s (per-process, index loaded each time).
Four leakage masks (self accession, self sha256, same cluster, same partition) with per-mask counts in the manifest; database sequences outside the split registry (514,823 of 548,702) cannot be masked by cluster or partition, so a registry-only arm is reported beside the primary one.
Evidence never feeds a model score, calibrator or threshold (stdlib-only module, import-graph tests, manifest flags); transferred terms are `source: "homology"` with `source_accession`.
Selected filter: e-value 1e-3, identity 0, coverage 0, top-k 5 (108-point grid on partitions 1-4). Validation (209 rows): homology_vote p@3 0.710 [0.669, 0.755], Fmax 0.736 [0.695, 0.772] vs model_raw 0.592 [0.551, 0.645] / 0.609 [0.576, 0.654] (disjoint), frequency prior 0.523 / 0.531;
registry-only p@3 0.635 [0.598, 0.679], Fmax 0.631 [0.604, 0.668] (overlaps the model). Partition 5 scored once (`touches: 1`; 64 rows, 52 evaluable): vote p@3 0.763 [0.674, 0.836] vs model_raw 0.583 [0.484, 0.667] and the prior 0.571 [0.496, 0.639] (disjoint), Fmax 0.726 [0.614, 0.806] (overlaps the models).
GO, CRF and localization counts are still 1 each. Hit coverage 0.982 (validation), 0.633 registry-only.
Hashes: index `58d46319c5c0a6514592363ab76a491c06facb49d92d84e0efadcecc21ce5979`; `mmseqs.exe` `ad319fc9fc67500ca2da9b9247d9611d2113e66cb800c15b603949f5652e2a0b`; archive `9752087ca4e25e93b830775832a75044dc94ae348d66d4681f9ff0355eecfcbf`; corpus (the GOA source) `d493478d38630651490adfe91c7869d1a6ac8ab759effdc3847ce84e295d41a8`;
registry `37ca5e0583b783a6aa4945f582fa2342b9fdf5b1008f68f71e9eb87869de276a`; bundle `registry/evidence_homology/0.13.0` `bundle_sha256` `3f6167b7b19a3c15dcbc9d3f008939952151105cdb57615c4a086f07f5231cec`; manifest `data/manifests/homology_metrics.json` `frozen_sha256` `93354fe2f4028d15fff90db8fbac4b20703659a86c0e329c06681b35d61440c2`.
`data/evidence/` is gitignored. API wiring and the refuse-to-start check are Phase 14 (Gate F); Phase 14 also adds the model-shaped GO output (`source: "model"`).
UNVERIFIED: any claim beyond 500 sequences, how much advantage comes from unmasked unregistered sequences, resident-server latency, `busybox.exe` licence, published comparisons (none fetched), Docker, GitHub Actions and the frontend gate. Detail: `docs/phases/13.md`.

### Phase 14 close — 2026-10-05

The served path (`/demo/*`, `api/app_v2.py`, API 1.2.0) returns trained-model output (ADR-0016): localization (Phase 10 calibrated, per-label thresholds), signal peptide with cleavage site (Phase 11 CRF) and GO function (Phase 12, `source: "model"`, no accession), all from registry bundles; homology terms from the Phase 13 bundle are `source: "homology"` with `source_accession`, in their own field, never mixed with model terms.
**Only the 500-sequence subset is answerable** ("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"): the encoder is the verified residue cache, a sequence not in it is refused with `not_in_embedding_cache`. Served groups: `eukaryote` (9 classes) and `plant` (10); bacterial groups are refused (no head was trained on bacteria).
Serving is numpy (`numpy==2.4.6`, `serve` extra) with a parity test against the torch forward; `test_layering` is green. `load_engine` refuses to start on a missing or tampered bundle, a calibrator or rule mismatch, a changed index byte or a changed GO source or ontology hash, listing every problem at once.
`MODEL_VERSION` is the composite of the four bundle versions; `ARTIFACT_REVISION` is left for Phase 19. Measured on one Windows 10, 12-logical-CPU machine through the real queue API (`data/manifests/serving_measurements.json`): single-sequence compute p95 2.59 s, 33 sequential jobs, queue wait 0; burst of 10, time-to-result p95 21.1 s of which queue wait p95 19.1 s and compute p95 2.6 s; a 40-sequence batch 6.0 s (0.149 s per sequence). `MAX_BATCH_SEQUENCES=20`. Host timing is Phase 19.
Golden set: `golden_set_motif.json` is a byte copy of `golden_set.json` (Phase 4 record, prefix `424e61c2e450`) and guards the motif module; the served engine refuses all 67 golden inputs (48 not cached, 17 unsupported group, 1 invalid, 1 unknown group). On the 408 cached real sequences the served top location equals the motif engine's for 54.2% of 397 comparable (11 outside the motif engine's 30-3000 range). Membrane type, the scanner, the residue attribution and the hydropathy track are still `source: "motif"`; consistency rule R06 is inactive (Phase 13 hits carry no location); `rules_version` is still "demo-0.1.0". `data/reference_demo.py` was **not** deleted (Gate E). The UI banner shows the model version instead of "synthetic".
UNVERIFIED: any claim beyond 500 sequences, gallery sequences are in-sample, Docker, GitHub Actions, host timing, the search-executable tamper test, seed variance. Detail: `docs/phases/14.md`.

### Phase 15 close — 2026-10-05

The served per-residue track now comes from the trained localization head (ADR-0017, API 1.3.0): `attribution_source: "model"`, default method `occlusion` (signed `p_full - p_removed` for the predicted class; window 5, stride 1, at most 64 forward passes; the head is re-run, the frozen encoder is not). Also returned: `pooling_attention` (class-agnostic, unsigned) and the CRF signal posterior as `signal_track`. Each method was proven by a frozen-protocol deletion test (402 sequences, 313 clusters, cluster bootstrap) before being served; all lower bounds of drop minus random are positive. GO attribution passes but is not served.
Mean localization deletion drop at k=5/10/20/40: occlusion 0.058/0.085/0.112/0.136, pooling attention 0.045/0.077/0.114/0.149, motif 0.022/0.047/0.074/0.098. Signal-peptide AUROC: CRF 0.9990, motif hydropathy 0.863, position-only prior 0.9936 (so much of the CRF signal is N-terminal position).
Cost: 40-sequence batch 5.97 s to 11.40 s (0.149 to 0.285 s per sequence); single p95 2.59 s to 3.52 s (possibly noise, n=33). `MAX_BATCH_SEQUENCES` stays 20. Hashes: protocol `67b09db9f9720325703e1e8d40da4540b3f542a062dfa3e77cc9a8d6a4652750`, metrics `frozen_sha256` `2dfae0a6bb77782abc9a680034e6aacd512ea9a08e7a011e48a3b77d57f37e2e`. No new dependency.
UNVERIFIED: biological meaning of any track, anything beyond 500 sequences, host timing, whether the single-job p95 shift is real. Detail: `docs/phases/15.md`.

### Two things a cold session must know

1. **The served path is trained models for 500 sequences, and a motif engine for the rest of the screen.**
   Localization, signal peptide, cleavage site and GO function come from the frozen registry
   bundles (numpy, ADR-0016) but only for the 500 sequences in the residue cache; anything else
   is refused with a reason, not guessed. The residue attribution is model-derived since Phase 15
   (ADR-0017). Membrane type, the in-silico scanner and the hydropathy track are still the
   Phase 0 motif engine (`source: "motif"`), and the scanner still uses a synthetic-fit tau. The banner shows
   the model version.
2. **`docs/api/openapi.yaml` is generated.** Edit
   `backend/protloc/domain/contracts.py` and the route decorators, then run
   `python scripts/export_openapi.py`. A hand edit is reverted by the next build and
   fails CI.
