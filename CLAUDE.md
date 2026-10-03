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
| ML | PyTorch 2.x, CPU wheel for serving | Pin the exact version in the run manifest |
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
  inference/         service.py — predict / scan / triage / landscape / evidence
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
`experiments/` are recreated by the phase that first writes to them and stay
gitignored.

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
| `MODEL_VERSION`, `ARTIFACT_REVISION` | 14 | The HF model-repo commit SHA the server pins |
| `ARTIFACT_CACHE_DIR` | 2 | Host-dependent; ephemeral on both candidates, so weights are re-pulled on cold start |
| `MAX_BATCH_SEQUENCES` | 14 | From measured throughput; start conservative |
| `GOA_RELEASE` | 13 | The GAF release string on the day you snapshot UniProt-GOA |
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
[ ] Phase 9   — Localization head (attention pooling + MLP)   MVP   7.0 h   <-- NEXT
[ ] Phase 10  — Calibration & per-label thresholds            MVP   5.0 h
[ ] Phase 11  — Signal-peptide CRF head                       ENH   8.0 h
[ ] Phase 12  — GO head                                       ENH   4.0 h
[ ] Phase 13  — Homology evidence (MMseqs2 index)             ENH   5.0 h
[ ] Phase 14  — Swap the engine: model behind the same API     MVP   6.0 h
[ ] Phase 15  — Attribution from the real model                ENH   4.0 h
[ ] Phase 16  — Frontend upgrade & landing page                MVP   7.0 h
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

### Two things a cold session must know

1. **The predictor is a motif engine and the reference data is synthetic.** Real
   Kyte-Doolittle hydropathy, real h-region detection with an AxA cleavage search, real
   TM windows and NLS/KDEL/PTS1/LPXTG patterns — every number on screen is computed live
   from the sequence, which is why mutating a residue changes the answer. But no model is
   trained and the reference set and benchmark are constructed with known labels. The app
   says so in a banner, and the banner stays until Phase 14.
2. **`docs/api/openapi.yaml` is generated.** Edit
   `backend/protloc/domain/contracts.py` and the route decorators, then run
   `python scripts/export_openapi.py`. A hand edit is reverted by the next build and
   fails CI.
