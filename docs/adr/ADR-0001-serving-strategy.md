# ADR-0001 — Serving strategy for the public prediction path

- **Status:** **Accepted** (2026-09-09)
- **Date proposed:** 2026-09-09 · **Date accepted:** 2026-09-09
- **Deciders:** Saikalyan
- **Blocks:** Phase 11 (Backend service & batch), Phase 12 (Frontend), Phase 13
  (Integration & performance)
- **Related:** `PROJECT_PLAN.md` §1 (non-functional targets), §6 (infrastructure
  conflict); Blueprint §3.3 (product scope), §20 (batch processing), §21 (API)

---

## 1. Context

The product definition commits to a **public, unauthenticated research web server**
with whole-proteome batch mode. The available infrastructure is **Colab or Kaggle GPU
sessions for training and a free tier for serving**.

These two commitments cannot both be satisfied by the obvious architecture — a
synchronous `POST /predict` that loads ESM-2 650M and returns a result inside the HTTP
request. The conflict is not a tuning problem; it is a resource ceiling.

### 1.1 The constraint, stated concretely

Weight memory is arithmetic from parameter count and dtype:

| Checkpoint | Params | fp32 weights | fp16 weights |
|---|---|---|---|
| ESM-2 650M (`esm2_t33_650M_UR50D`) | 650 M | ≈ 2.6 GB | ≈ 1.3 GB |
| ESM-2 150M (`esm2_t30_150M_UR50D`) | 150 M | ≈ 0.6 GB | ≈ 0.3 GB |
| ESM-2 35M (`esm2_t12_35M_UR50D`) | 35 M | ≈ 0.14 GB | ≈ 0.07 GB |

These are **weights only**. Attention activations scale with sequence length and add
materially on top for long proteins, and the Python process, PyTorch runtime and web
server carry their own overhead before a single residue is embedded.

Free serving tiers commonly cap container memory at or below 512 MB and suspend idle
containers, so a cold request pays a full model load. The 650M row does not fit, and the
150M row fits only after the runtime overhead is accounted for — which it is not.

**Note:** the memory figures above are derived, not measured. Actual tier limits and
cold-start behaviour must be confirmed against the chosen provider before this ADR moves
from Proposed to Accepted. The *shape* of the conflict is certain; the exact headroom is
not.

### 1.2 Why this cannot be deferred

Phase 11 builds the backend service and Phase 12 the frontend against it. If the answer
turns out to be "everything is a job," the frontend has no synchronous result page — it
has a submission form and a result URL, which is a different set of screens. Deciding
this after Phase 12 means rebuilding Phase 12.

---

## 2. Decision drivers

1. **The paper must not be blocked by hosting.** The contribution is the model and its
   trust layers. An architecture that makes deployment contingent on funding puts the
   paper's demo instance at risk.
2. **Served-path evaluation is non-negotiable.** Blueprint §3.3 requires the evaluation
   layer to call the real served path. Whatever is served must be the thing that is
   measured, so a serving compromise silently becomes a *results* compromise unless it is
   handled deliberately.
3. **Honest latency expectations.** No biologist expects proteome annotation to be
   instant. The perceived cost of waiting is far lower here than in a consumer product.
4. **Solo operator.** Every additional moving part is maintenance that competes with the
   science layer's 24-item Core list.
5. **Reversibility.** The decision should be cheap to revisit if funding or a lab GPU
   appears later.

---

## 3. Options considered

### Option A — Two-tier model

Serve a small checkpoint (35M/150M) for interactive single-sequence requests; run the
650M model only in queued batch.

- **Good:** interactive latency stays low; free tier becomes plausible.
- **Bad:** **two models means two sets of results.** Every head needs calibrating twice,
  every consistency rule validating twice, and the paper must state which tier produced
  which number. A user's interactive answer and their batch answer for the same sequence
  could disagree — which is precisely the incoherence this project exists to eliminate.
- **Cost:** roughly doubles Phases 8–10.

### Option B — Queue everything

No synchronous prediction. Every submission — one sequence or a whole proteome — creates
a job and returns a `job_id` and result URL. A single worker with the 650M model
processes the queue.

- **Good:** the latency constraint disappears, because there is no request holding a
  connection open. Memory is a *worker* concern, not a per-request one, so one modest
  always-on worker serves everyone. One model, one set of results, one calibration.
  Batch mode already required this machinery (Blueprint §20), so the single-sequence path
  becomes a batch of size one rather than a second code path.
- **Bad:** a user waits — seconds to a minute for one sequence. The frontend needs
  polling or SSE and a result-URL flow. There is no "instant demo" moment.
- **Cost:** low. It reuses the job system Phase 11 builds anyway.

### Option C — Pay for serving

A small GPU or high-memory VM running the 650M model synchronously.

- **Good:** simplest architecture; best user experience; no compromise on the model.
- **Bad:** a recurring cost with no funding line, borne by a student indefinitely. When
  the card expires the service dies — and a dead URL in a published paper is worse than a
  slow one.
- **Cost:** money, monthly, forever.

### Option D — Rejected without further analysis

Synchronous 650M on a free tier. This is the default that the constraint rules out; it is
recorded only so a future reader knows it was considered and why it is not viable.

---

## 4. Decision

**Adopt Option B — queue everything.**

Every prediction request, regardless of size, becomes a job:

```
POST /predict   → 202 Accepted { job_id, result_url }   # one sequence = a job of size 1
POST /batch     → 202 Accepted { job_id, result_url }   # FASTA upload
GET  /jobs/{id} → status, progress, and results when complete
```

The single-sequence path is a batch of size one. There is exactly one inference code
path, one model, one calibrator set, one consistency rule set, and one set of numbers
that the evaluation layer measures.

### 4.1 Consequences for the API contract (Phase 2)

`POST /predict` no longer returns a prediction. It returns `202` with a `job_id`. The
response schema in Blueprint §21.3 becomes the shape of `GET /jobs/{id}` once the job
completes, not the shape of the `POST` response. **This must be reflected in the OpenAPI
spec before Phase 2 exits.**

For a job of size one that completes quickly, the server may hold the connection briefly
and return the result inline as a convenience — but the contract is asynchronous, and the
frontend must never assume the fast path.

### 4.2 Consequences for the frontend (Phase 12)

The result page becomes a **result URL**, reachable and shareable — which is arguably
better for a research tool than an ephemeral response. The screens gain a pending state
that is neither an error nor an empty state, and `docs/DESIGN.md` needs a fourth state in
its visual language: *queued/running*, visually distinct from *abstained* and from
*failed*. This was written as §3.4 of `DESIGN.md`; the Phase 2 wireframes carry it into
each of the five screens, and Phase 12 implements it. (This paragraph said "to be made in
Phase 12" while §7 below recorded it as done; corrected in Phase 2.)

Result URLs are unguessable (UUIDv4) and results expire after a retention window set in
Phase 16. They are not access-controlled — anyone with the URL can read the result — which
is acceptable because the input is a sequence the user already had, and the service holds
no account data. State this in the privacy note on the landing page rather than implying
confidentiality the system does not provide.

### 4.3 Consequences for evaluation (Phase 14)

`scripts/benchmark_api.py` submits jobs and polls. Latency is reported as
**time-to-result**, split into queue wait and compute time, rather than as request
latency. The §1 non-functional targets in `PROJECT_PLAN.md` are revised accordingly:

| Old target | Revised target |
|---|---|
| Single-sequence p95 latency (warm) ≤ 3 s | Time-to-result p95, single sequence, warm cache ≤ 10 s |
| Single-sequence p95 latency (cold) ≤ 20 s | Time-to-result p95, single sequence, cold ≤ 60 s |
| Batch throughput ≥ 500 seq/hour/worker | Unchanged |
| — | Queue wait p95 under nominal load ≤ 30 s (new) |

These revised numbers are targets to validate in Phase 13, not measurements.

### 4.4 Consequences for training

Unaffected by this decision, but the Colab/Kaggle session limit remains a separate
constraint: every training run must checkpoint to durable storage each epoch and support
`--resume`, or a session termination loses the run. This is a **Phase 8 requirement**,
tracked separately from this ADR.

---

## 5. Consequences summary

**Positive.** One model, one code path, one set of results — the served path and the
measured path are the same thing, which is what Blueprint §3.3 demands. Free-tier serving
becomes genuinely feasible. The job machinery was already required for proteome mode, so
this consolidates rather than adds. Result URLs are shareable and citable. Memory pressure
moves from per-request to per-worker, where it can be reasoned about.

**Negative.** No instant demo, which costs something when showing the tool to a reviewer
or a supervisor — mitigate with the example-sequence gallery serving pre-computed,
cached results so the demo path is fast and honest about being cached. Frontend gains
polling and a pending state. A cold single-sequence request may take a minute, and users
must be told that up front rather than left watching a spinner.

**Neutral.** Horizontal scaling later means adding workers, which is a well-understood
path if traffic ever justifies it.

---

## 6. Revisit triggers

Reopen this ADR if any of these become true:

1. A lab GPU or funded VM becomes available for **at least 12 months** — Option C becomes
   viable and simplifies the frontend.
2. Measured Phase 13 compute time for a single sequence on the target hardware exceeds
   **120 s**, making even the asynchronous experience unacceptable — reconsider Option A
   for the interactive tier.
3. Queue wait under real traffic exceeds the §4.3 target and adding workers is not
   affordable.
4. Measured provider limits turn out to be materially higher than §1.1 assumes, making
   synchronous 650M serving genuinely possible.

---

## 7. Verification

- [x] `MODEL_SERVING_MODE=queue` set in `.env.example` and read by `config/`
      (`CLAUDE.md` env schedule, Phase 1).
- [x] The API map reflects the `202 Accepted` contract (`CLAUDE.md`; OpenAPI spec
      formalised at Phase 2 exit).
- [x] `PROJECT_PLAN.md` §1 carries the revised targets from §4.3.
- [x] `docs/DESIGN.md` **§3.4** defines the queued/running visual state. (Corrected in
      Phase 2: this line read §3.5, which is the evidence section.)
- [ ] **Open — does not block acceptance.** The chosen hosting provider's measured
      memory limit, idle-suspension behaviour and request timeout, replacing the derived
      estimates in §1.1. Provider is `[CLIENT-TBD]`; record the three numbers here on
      selection, before Phase 16.

### 7.1 Note on accepting with §1.1 unmeasured

This ADR is accepted while its provider measurements are still outstanding, which is a
deliberate call rather than an oversight.

The decision does not depend on them. Option B is chosen because it yields **one model,
one code path, one set of results** — an argument from coherence, not from a memory
ceiling. Better hosting than assumed would not make a two-model architecture desirable;
it would only make Option C affordable, which is already captured as revisit trigger 1.

The measurements matter for exactly one thing: revisit trigger 4, which asks whether
synchronous 650M serving was possible after all. That question can be answered at Phase
16 without any rework, because an asynchronous contract can always be served faster —
the reverse is not true. Accepting now unblocks Phases 2, 11 and 12; the open checkbox
above keeps the gap visible rather than closing it with a guess.
