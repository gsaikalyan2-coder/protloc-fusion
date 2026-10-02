# CONTEXT_MEMORY.md — ProtLoc Fusion

A running memory service that holds project context across coding sessions, so
an assistant picking up the project on day 60 knows what was frozen on day 12.

**Scope boundary, stated first:** this is development-time tooling. It stores
decisions, conventions and frozen values about *the project*. It never touches
protein data, never runs in the serving container, never appears in a
reproducibility manifest, and is not a dependency of the paper. If this service
is down, ProtLoc Fusion is entirely unaffected.

---

## 1. Choice: mem0, self-hosted

Three credible open-source options were compared (all Apache-2.0):

| | mem0 | Letta | Graphiti (Zep) |
|---|---|---|---|
| Shape | Memory layer — add/search/update/delete | Full stateful agent server | Temporal knowledge graph |
| Infra to run | Python lib, or `docker compose` | Agent server + DB | Graph DB (FalkorDB / Neo4j / Kuzu) + an LLM provider per ingest |
| Fits "hold project context" | Directly | Overshoots — you get an agent framework | Overshoots — you get a graph you must model |
| Ongoing cost | Embedding calls only | Server + LLM | LLM call per ingested episode |

**mem0 self-hosted is the choice.** The requirement is "remember decisions about
this project," not "run an agent" or "model a temporal graph." Letta and
Graphiti are stronger systems for problems ProtLoc Fusion does not have, and
Graphiti in particular bills an LLM call per ingestion — a real recurring cost
for what is, in the end, a notes store with semantic search.

- Repository: `mem0ai/mem0` · Apache-2.0
- Library: `pip install mem0ai`
- Self-hosted server: `docker compose up`
- Optional hybrid search: `pip install mem0ai[nlp]` then
  `python -m spacy download en_core_web_sm`

Verified from the repository on 2026-09-09. Pin the version in
`THIRD_PARTY.md` before relying on it.

---

## 2. What goes in — and what does not

The value of a memory store is inversely proportional to how much noise is in
it. A store full of "fixed a typo" retrieves nothing useful.

**Store (durable, decision-shaped):**

- Frozen values and *why* they were frozen — identity threshold, τ, loss
  weights, split version
- Architectural rulings and their rationale — "GraphPart adopted over
  hand-rolled clustering, because it is citable"
- Licensing rulings — "Prot2Token: methodology reference only, AGPL network
  clause"
- Conventions — naming, module boundaries, what must never import what
- Experiment outcomes at the claim level — "H4: LoRA reached within 0.4 macro-F1
  of full fine-tune at 1.2% trainable params, EXP-0031"
- Open decisions still unfrozen, with what would close them

**Do not store:**

- Anything reproducibility depends on. The run manifest, dataset hashes and
  registry bundle are the source of truth. Memory is a convenience index over
  decisions, never the record itself. **If it matters for the paper, it lives
  in a versioned file in the repo.**
- Protein sequences, dataset contents, model outputs
- Anything transient — a branch name, a current bug, today's failing test
- Credentials of any kind

**The test:** would this still be true and useful in three months? If not, it
belongs in a commit message, not here.

---

## 3. Layout

Namespace memories so retrieval stays sharp:

| Namespace | Holds |
|---|---|
| `protloc/decisions` | Frozen values and rulings, each with a date and rationale |
| `protloc/conventions` | Code, naming and architecture rules |
| `protloc/experiments` | Claim-level outcomes with their EXP id |
| `protloc/licensing` | Redistribution rulings per dependency |
| `protloc/open` | Unfrozen decisions and what would close them |

Every entry carries the artifact it points at — an ADR number, an EXP id, a file
path. A memory that asserts a fact without naming where the fact is recorded is
worse than no memory, because it cannot be checked.

---

## 4. Write discipline

Write at the moment a decision is made, not in a weekly cleanup. The rationale
is only in your head on the day.

Three moments that always produce a memory:

1. **An ADR is accepted** → one entry in `protloc/decisions` naming the ADR.
2. **A value is frozen** → one entry with the value, the date, and the
   experiment that justified it.
3. **A licensing ruling is made** → one entry in `protloc/licensing`.

And one rule for reading: **memory is a pointer, not an authority.** When a
retrieved memory disagrees with a file in the repo, the file wins and the memory
gets corrected. Memory drifts; version-controlled artifacts do not.

---

## 5. Why this is not `CLAUDE.md`

A repo context file and a memory service solve different halves of the problem,
and running the service does not remove the need for the file:

- `CLAUDE.md` is **deterministic and versioned** — it is read in full, every
  session, by anyone who clones the repo. It carries the rules that must never
  be missed: the architecture layering, the dependency prohibitions, the
  no-copying rule for restricted licenses.
- mem0 is **searchable and accumulating** — it answers "why did we set the
  threshold there?" three months later, when nobody remembers.

Rules go in the file. History goes in the service. If a memory turns out to be
something every session needs, that is the signal to promote it into
`CLAUDE.md` and delete it from the store.

---

## 6. Honest limitations

- **This is unproven for this project.** It is a reasonable choice on the
  evidence, not a validated one. Give it a month; if the store is not being read,
  delete it rather than maintaining it out of obligation.
- **Retrieval quality claims in this space are vendor-run and mutually
  contradictory.** Benchmark numbers you find comparing memory layers are mostly
  published by the vendors being compared. Choose on shape and operational cost,
  which is what was done above — not on a leaderboard.
- **It adds a service to keep alive.** That is a real cost against a project
  whose Core list is already 24 items. If it starts competing with the science
  layer for attention, it loses.
