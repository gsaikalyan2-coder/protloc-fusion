# ARCHITECTURE.md — ProtLoc Fusion

How the system is put together, and why it is put together that way. `CLAUDE.md` is the
execution contract, `FOLDER_STRUCTURE.md` says where a file goes, `docs/adr/` records the
irreversible choices. This document is the map that makes those three legible together.

Written in Phase 2, when the interfaces were frozen. Nothing here predicts anything yet.

---

## 1. The problem the shape solves

Five published tools answer five related questions about the same protein sequence. A
researcher runs each, in a different input format, and reconciles the answers by hand —
which is not merely tedious. The tools disagree, and nothing in the workflow surfaces the
disagreement. A protein predicted to carry a mitochondrial transit peptide by one tool and
localised to the cell wall by another is a contradiction that a human notices only if they
are looking for it.

ProtLoc Fusion answers all of them from **one shared representation**, which is what makes
the contradiction detectable at all: seven heads reading the same embedding can be checked
against each other by a rule set. That is the contribution, and the architecture exists to
protect it.

Three properties follow, and each one shows up as a structural constraint rather than a
convention:

**One inference path.** The served path and the measured path are the same code. If the
evaluation layer could call the model directly, the paper's numbers would describe
something the user never touches. So `evaluation/` reaches the system over HTTP, and the
import test enforces it.

**No hidden constants.** Every threshold, label vocabulary and biological rule lives in the
registry bundle or in `rules/consistency_rules.yaml`, never in a `.py` file. A constant in
code is a value nobody can audit and nobody remembers to change.

**Honest uncertainty.** Abstention, masking and pending are three different things and the
contract keeps them distinct at every layer, because collapsing them is how a system
reports a non-answer as an answer.

---

## 2. Layers

Seventeen modules under `backend/protloc/`, each assigned a depth. **A module may import
only a strictly deeper one.** Equal depth is a violation too, which is what stops two
pipeline stages importing each other and leaves `inference/` as the only module that knows
the pipeline's order.

```
depth   modules                              may import
─────   ──────────────────────────────────   ────────────────────────────────────
  0     api, web, jobs                       domain, inference, config, utils
  0     evaluation                           domain, config, utils  (HTTP otherwise)
  0     training                             anything deeper; imported by nothing
 10     inference                            everything deeper
 20     features, data                       models, registry, … , domain
 30     models, registry                     calibration, consistency, … , domain
 40     calibration, consistency             interpretability, evidence, … , domain
 50     interpretability, evidence           config, utils, domain
 90     config, utils                        domain
100     domain                               nothing internal
```

`backend/tests/test_layering.py` is this table as assertions. Adding an eighteenth module
fails a test until its depth is declared, which is deliberate: a new module is a decision.

Three depths deviate from a naive reading of `FOLDER_STRUCTURE.md`, each for a reason
recorded in the Phase 1 handover: `training` is an entry point rather than a peer of
`models`; `config` and `utils` sit at depth 90 so that every layer can read its own paths;
`evaluation` is an allowlisted entry point rather than a support module.

### 2.1 The consequence that bites later

`interpretability` (50) may **not** import `models` (30). Per-residue attribution receives
the scored model through a port protocol declared in `domain/`. This is the documented
hexagonal design and it is also a real constraint on Phase 10 — cheaper to know now.

---

## 3. Ports

A port is an interface declared where it is *needed*, not where it is implemented. They
live in `backend/protloc/domain/ports.py`, at depth 100, so both sides can depend on the
shape and neither on the other's internals.

| Port | Declared for | Implementation today | Implementation later |
|---|---|---|---|
| `ProvenanceSource` | `api/` needs a model version without knowing versions come from disk | `UnloadedProvenance` — reports that nothing is loaded | registry-backed, Phase 7 |
| `JobService` | `api/` needs to submit and follow work without owning a queue | `UnavailableJobService` — refuses, naming why | queue-backed, Phase 11 |
| `ExampleGallery` | `api/` needs cached results without knowing where they are cached | `UnavailableJobService` | Phase 12 |

Swapping an implementation is three lines in `backend/protloc/api/deps.py`. No router
changes, and no router test changes. That is the whole reason the indirection exists.

---

## 4. The request path

```
  browser                     API (depth 0)            inference (10)        worker
     │                             │                        │                  │
     │  POST /predict              │                        │                  │
     ├────────────────────────────►│                        │                  │
     │                             │  submit_sequence()     │                  │
     │                             ├───────────────────────►│                  │
     │                             │                        │  enqueue         │
     │                             │                        ├─────────────────►│
     │  202 {job_id, result_url}   │◄───────────────────────┤                  │
     │◄────────────────────────────┤     JobAccepted        │                  │
     │                             │                        │   embed          │
     │  GET /jobs/{id}   (poll)    │                        │   score 7 heads  │
     ├────────────────────────────►│                        │   calibrate      │
     │  200 {status: "running"}    │                        │   check rules    │
     │◄────────────────────────────┤                        │   attribute      │
     │                             │                        │   fetch evidence │
     │  GET /jobs/{id}             │                        │◄─────────────────┤
     ├────────────────────────────►│                        │                  │
     │  200 {status: "completed",  │                        │                  │
     │       results: [...]}       │                        │                  │
     │◄────────────────────────────┤                        │                  │
```

The worker column is Phase 11's. `docs/diagrams/request_sequence.md` carries the same flow
with the failure and cancellation branches.

**Why there is no synchronous branch.** ADR-0001. Serving ESM-2 650M inside an HTTP request
does not fit the available infrastructure, and the two architectures that would fit — a
second smaller model for interactive requests, or paid always-on hosting — cost either two
sets of results or a recurring bill with no funding line. Queueing everything yields one
model, one code path, one calibration and one set of numbers. A single sequence is a job of
size one.

The server *may* answer a size-one job inline with `200` if it completes inside
`INLINE_RESULT_WINDOW_SECONDS`. That is a convenience and is deliberately **not** in the
OpenAPI document: a client that assumes it is broken against any busy deployment.

---

## 5. The contract

`docs/api/openapi.yaml` — OpenAPI 3.1, 8 paths, 26 schemas. Generated from the FastAPI
application by `scripts/export_openapi.py`; CI regenerates it and fails on a diff.

Two rules shape every response.

**Rule 6 — the reproducibility envelope.** `model_version`, `config_hash`, `rules_version`
and `experiment_id` appear on every body, including the `202` and every error. They are
`null` while no registry bundle is loaded, rather than a sentinel string: a client can test
for `null`, but cannot tell `"unloaded"` from a version somebody shipped. A result a user
cannot trace back to a model, a config and an experiment is a result they cannot reproduce,
and reproducibility is the paper's claim.

**Two endpoints return results, and they are not interchangeable.**

- `GET /jobs/{job_id}` is the job *resource*. `200` at every status, results inline once
  `completed`, `410 Gone` past the retention window. This is `result_url`, and what the
  frontend polls.
- `GET /jobs/{job_id}/results` is the result *representation*, for download. Takes
  `?format=json|csv`, sets `Content-Disposition`, and answers `409` for a job that has not
  completed. Never polled, never partial.

`GET /docs` serves Swagger UI. It renders the contract; it is not part of it, and it is
absent from the document.

### 5.1 What the service does today

`/health` and `/version` are live. The other five endpoints answer `503` with
`no_registry_bundle`, naming Phase 7 as the phase that supplies one.

This is not a placeholder. A placeholder asserts something false — a mock prediction claims
a model ran. A `503` asserts something true: the service cannot accept jobs because no
model bundle is loaded. The condition is checked at request time, not hardcoded, and it
stops being true when Phase 7 puts a bundle in the registry without any router changing.

`/health` answers `200` while reporting `degraded`. A readiness probe that failed here
would restart a container that is correctly serving its frozen contract.

---

## 6. The frontend

React 18 + Vite + TypeScript. No CSS framework, no component library, no UI runtime beyond
React — enforced by an allowlist in `scripts/check_frontend_deps.py`, because a rule of that
shape is broken by `npm install` and noticed at review, if at all.

Two generated files, both regenerated and diffed by CI:

```
docs/DESIGN.md        ──generate_tokens.py──►  src/styles/tokens.css
docs/api/openapi.yaml ──openapi-typescript──►  src/types/api.generated.ts
```

`src/types/api.ts` is hand-written: it gives the generated shapes readable names and holds
the three runtime helpers (`isTerminal`, `hasResults`, `isIdentified`) that a types-only
generator has nowhere to put. Adding a hand-written *type* there is almost always a
mistake — if the backend does not describe it, the server does not send it.

`src/services/api.ts` is the only module permitted to call `fetch()`, enforced by ESLint.
One place knows the API exists, so a contract change is a change to one file.

Colour literals are forbidden anywhere under `src/`. Every colour is a custom property from
`tokens.css`, which is generated from `docs/DESIGN.md` — a hex in a component is how a
design system starts drifting from its own document.

### 6.1 Five screens

Landing, Predict, Result, Batch, Job. Wireframes in `docs/diagrams/wireframes/`. Each one
must handle the **pending** state that ADR-0001 introduced and `docs/DESIGN.md` §3.4
defines: `queued`/`running` is a fourth trust state, and it must never borrow abstention's
visual vocabulary. "The model declined to assert" and "the model has not run yet" are
opposite claims, and a user who confuses them reads an empty page as a scientific result.

---

## 7. Where the numbers come from

```
configs/ (Hydra)  ──resolve──►  config_hash  ─┐
                                              ├──►  experiments/EXP-NNNN  ──►  registry/
data/ (DVC)       ──build───►  dataset hash  ─┘         run manifest            bundle
                                                             │                     │
                                                       W&B (viewer only)      served path
                                                                                   │
                                          scripts/evaluate.py ──over HTTP──────────┘
```

The run manifest is authoritative and W&B is a viewer (ADR-0004). The evaluation layer
reaches the system over HTTP so that the paper's numbers are the user's numbers. Every
metric in a document names the command, config hash and experiment id that produced it; a
number with no provenance is removed, not footnoted.

---

## 8. Deployment

Two images, and their separation is asserted rather than assumed:

| | `Dockerfile.serve` | `Dockerfile.train` |
|---|---|---|
| Base | `python:3.11-slim-bookworm` | CUDA 12.4 runtime |
| Contains | FastAPI, uvicorn, pydantic | + torch, EMBOSS `needleall`, hydra |
| Entry point | `uvicorn protloc.api.app:create_app --factory` | training scripts |
| CI asserts | `import torch` **fails**; `/health` answers `200` | `which needleall` succeeds, and a real alignment runs |

The torch check is deliberately inverted: if the serving image *can* import torch, the two
images are sharing a base and the train/serve separation is decorative.

Hosting is HF Spaces — Docker for the backend, Static for the frontend, with artifacts in
HF Hub repos because the Spaces disk is non-persistent (ADR-0005).

`create_app()` is a factory and there is no module-level `app`. Building the application
reads and validates the settings; a module-level instance would do that at *import* time,
which would make `import protloc.api.app` fail on an incomplete environment — and the
OpenAPI export and every test perform exactly that import.

---

## 9. What is not built yet

Honest inventory, by the phase that owns each item.

| Not built | Owned by |
|---|---|
| The dataset, its manifest and the length/alphabet policy | 3 |
| Homology splits and the leakage report | 4 |
| Embeddings and the content-addressed cache | 5 |
| Any model, any head, any trained weight | 6–8 |
| Calibration, and therefore every real τ | 9 |
| `rules/consistency_rules.yaml`, attribution, the evidence service | 10 |
| The job queue, the worker, resumable batch | 11 |
| Every screen | 12 |
| Every measured number in this document's targets | 13–14 |

The contract above describes all of it. That is the point of freezing it first: the
frontend and the evaluation layer are built against a stable shape rather than against
whatever the backend happened to return the week they were written.
