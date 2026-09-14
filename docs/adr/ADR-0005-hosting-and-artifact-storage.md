# ADR-0005 — Hosting and artifact storage

- **Status:** **Accepted** (2026-09-09), pending the measurements in §7
- **Deciders:** Saikalyan
- **Blocks:** Phase 16 (Governance, cost & release readiness), Phase 17 (Release) —
  and closes ADR-0001's outstanding checkbox
- **Related:** ADR-0001 §1.1, §7; ADR-0003 §4; ADR-0004 §3.3

---

## 1. Context

`PROJECT_PLAN.md` §7 carried hosting as `[CLIENT-TBD]`, and ADR-0001 was accepted with
its provider measurements outstanding. Both are closed here.

ProtLoc Fusion needs three things hosted:

1. **A queue worker** holding ESM-2 650M and running predictions (ADR-0001).
2. **A frontend** — a static React bundle.
3. **Durable artifact storage** — model registry bundles, training checkpoints
   (ADR-0004 §3.3), and the evidence snapshots (ADR-0003).

The operator is a student with no funding line, so a recurring bill is a project risk,
not merely a cost.

---

## 2. Decision drivers

1. **Zero recurring cost, or as close as possible.** A service that dies when a card
   expires leaves a dead URL in a published paper.
2. **Enough memory for ESM-2 650M.** ≈2.6 GB fp32 / ≈1.3 GB fp16 for weights, plus
   activations and runtime.
3. **Durability across Colab session death** (ADR-0004 §3.3).
4. **Credibility.** A research tool hosted somewhere biologists already trust is easier
   to cite and to find.
5. **Existing accounts.** A HuggingFace account (`sk8069`) already exists.

---

## 3. Options considered

| Option | Memory | Cost | Notes |
|---|---|---|---|
| **HF Spaces CPU Basic (chosen)** | 2 vCPU, 16 GB RAM, 50 GB **non-persistent** disk | Free | Docker Space; sleeps when idle; ML-native audience |
| Render / Railway / Fly free tiers | Typically ≤512 MB–1 GB | Free | Too little memory for the 650M checkpoint |
| Oracle Cloud always-free VM | Up to 24 GB (ARM) | Free | Full control, but full sysadmin burden and ARM wheel friction |
| Small paid GPU/VM | Ample | Recurring | Best UX; unfunded (ADR-0001 Option C) |

---

## 4. Decision

**Host on Hugging Face: a Docker Space for the backend worker, a Static Space for the
frontend, and HF Dataset/Model repositories for durable artifacts.**

### 4.1 Backend — HF Spaces, Docker SDK, CPU Basic

Per HuggingFace's Spaces documentation, the default CPU Basic environment provides
**2 vCPU, 16 GB RAM and 50 GB of non-persistent disk, at no hourly cost.**

16 GB comfortably holds ESM-2 650M. **This does not reopen ADR-0001**, and it is worth
being precise about why: the binding constraint was never only memory. Two vCPU means
650M-parameter CPU inference takes tens of seconds per sequence, and an idle Space
sleeps — so a cold request additionally pays a container start and a model load. An
asynchronous queue is exactly the right architecture for *slow but sufficient* compute.
Memory was the constraint that made synchronous serving impossible; throughput and cold
starts are the constraints that keep it inadvisable.

ADR-0001 revisit trigger 4 is therefore evaluated and **not triggered**.

### 4.2 Frontend — HF Static Space

The Vite build output deploys as a Static Space, free and always-on, calling the backend
Space over HTTPS. CORS is configured to the frontend Space's origin only.

### 4.3 Durable artifacts — HF Hub repositories

**Spaces disk is non-persistent.** Anything that must survive a restart lives in a Hub
repository, pulled at container start:

| Artifact | Location | Pulled at |
|---|---|---|
| Registry bundles (`model.pt`, calibrator, rules, config, checksums) | HF **Model** repo `sk8069/protloc-fusion` | container start |
| Evidence snapshots (Swiss-Prot MMseqs2 index, GOA store) | HF **Dataset** repo `sk8069/protloc-fusion-evidence` | container start |
| Training checkpoints (ADR-0004 §3.3) | HF Dataset repo `sk8069/protloc-fusion-checkpoints` | written each epoch from Colab |
| Frozen corpus + manifests | HF Dataset repo `sk8069/protloc-fusion-data` | on demand |

This resolves ADR-0004 §3.3's "durable location" and gives the DVC remote a home.

**Job state and the embedding cache are ephemeral by design.** A restart loses queued
jobs and cached embeddings. Rather than engineering around that on free hosting, it is
accepted and made visible: `JOB_RETENTION_HOURS` is short, the UI states that results
expire, and a job lost to a restart returns `410 Gone` with a resubmit action — never a
silent failure or a wrong answer.

### 4.4 Cold start

An idle Space sleeps; the first request afterwards pays container start plus model load.
Mitigations, in order of preference:

1. **The gallery is pre-computed** (`PROJECT_PLAN.md` §1, ≤500 ms) — served from static
   JSON, so a sleeping worker never blocks a first impression.
2. `/health` reports `warming` distinctly from `ready`; the frontend shows the §3.4
   `queued` state rather than an error.
3. Custom sleep time configured as generously as the tier allows.

No keep-alive pinger. Defeating a free tier's sleep behaviour with synthetic traffic is
abuse of a service being given for nothing.

### 4.5 Environment additions

```env
# ── Phase 16: hosting ──
HF_MODEL_REPO=sk8069/protloc-fusion
HF_EVIDENCE_REPO=sk8069/protloc-fusion-evidence
HF_CHECKPOINT_REPO=sk8069/protloc-fusion-checkpoints
HF_DATA_REPO=sk8069/protloc-fusion-data
ARTIFACT_CACHE_DIR=/data/artifacts
FRONTEND_ORIGIN=ENTER_YOUR_VALUE_HERE
```

`HF_TOKEN` (already Phase 5) gains **write** scope at Phase 16 for artifact pushes.

---

## 5. Consequences

**Positive.** Zero recurring cost, so the URL outlives the project's funding. 16 GB removes
the memory ceiling entirely. Artifacts, model and demo sit on one platform biologists and
ML reviewers already use, and the Hub gives versioning and checksums for free. Rollback
becomes repointing `MODEL_VERSION` at a previous Hub revision.

**Negative.** Two vCPU makes throughput modest — Phase 13 must measure it honestly and the
≥500 seq/hour/worker target may need revising downward with evidence. Cold starts are real.
Ephemeral job state is a genuine limitation, disclosed rather than hidden. Free tiers change
terms, which is a project risk with no mitigation beyond portability: everything is
Dockerised, so moving is a redeploy.

**Neutral.** Hub repos under `sk8069` tie the project to a personal account; transfer to an
organisation later if the project outgrows it.

---

## 6. Revisit triggers

1. Phase 13 measures single-sequence compute above **120 s** — ADR-0001 revisit trigger 2
   fires and the interactive tier needs reconsidering.
2. HF Spaces CPU Basic terms change materially.
3. Evidence snapshots plus registry exceed the 50 GB disk after pull — reduce the
   Swiss-Prot subset (ADR-0003 revisit trigger 1).
4. Sustained traffic makes a single worker's queue wait exceed the §4.3 target in
   `PROJECT_PLAN.md` §1.

---

## 7. Verification

- [x] Provider chosen; `[CLIENT-TBD]` on hosting closed.
- [x] ADR-0001 revisit trigger 4 evaluated — not triggered (§4.1).
- [x] ADR-0004 §3.3 durable checkpoint location resolved (§4.3).
- [ ] **Measure and record here, before Phase 17:** actual available RAM inside a running
      Space, observed cold-start time to first prediction, and the Space request timeout.
      The 16 GB / 2 vCPU / 50 GB figures come from HuggingFace's published Spaces
      documentation and at least one community thread quotes different numbers — treat
      the docs as authoritative but confirm empirically before release.
- [ ] Frontend Space origin recorded in `FRONTEND_ORIGIN`; CORS restricted to it.
- [ ] Artifact pull-at-start verified on a cold container.
- [ ] Rollback rehearsed by repointing `MODEL_VERSION` at a previous Hub revision
      (Phase 16 exit criterion).
