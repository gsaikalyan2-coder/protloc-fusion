# ADR-0003 — Evidence layer: homology and GO annotation sources

- **Status:** **Accepted** (2026-09-09)
- **Deciders:** Saikalyan
- **Blocks:** Phase 10 (Trust & evidence layers)
- **Related:** Blueprint §16; `docs/DESIGN.md` §3.5; ADR-0001

---

## 1. Context

Blueprint §16 requires the response to carry nearest homologs (accession, identity,
coverage, e-value, database) and GO terms as **independent corroboration** — evidence
that contextualises the prediction without feeding into the score.

Two sources must be chosen: what to search for homologs against, and where GO
annotations come from. `HOMOLOGY_DB_URL` and `GO_SOURCE_URL` are currently
`[CLIENT-TBD]` and block Phase 10.

The obvious approach — call UniProt/EBI/QuickGO REST APIs at request time — is wrong
here, for reasons that are architectural rather than practical.

---

## 2. Decision drivers

1. **Reproducibility.** A result carrying `experiment_id` and `config_hash` claims to be
   reproducible. If part of the response came from a live third-party API whose contents
   changed yesterday, that claim is false and nothing in the manifest reveals it.
2. **Graceful degradation is already required.** Blueprint §23.2 and `DESIGN.md` §3.5
   demand that evidence-service failure never blocks a prediction. Fewer external
   dependencies means fewer paths to that degraded state.
3. **Latency.** Under ADR-0001 the worker processes a queue. A per-sequence outbound API
   call with retries becomes the throughput bottleneck in whole-proteome mode — 20,000
   sequences means 20,000 network round-trips.
4. **Politeness and rate limits.** Hammering EBI with a proteome-scale batch is both
   rate-limited and rude.
5. **Cost.** Self-hosting costs disk; API calls cost latency and reliability.

---

## 3. Options considered

| Option | For | Against |
|---|---|---|
| **Self-hosted snapshots (chosen)** | Versioned, hashed, reproducible; no network at request time; batch-friendly; no rate limits | Disk cost; index build step; snapshots go stale and must be refreshed deliberately |
| Live UniProt / QuickGO REST | Always current; no storage | Breaks reproducibility; rate-limited; a network hop per sequence; an outage degrades every response |
| Hybrid — snapshot with live fallback | Best coverage | Two code paths, and the response can no longer state *which one* produced it without extra plumbing. Complexity that buys little |

---

## 4. Decision

**Both evidence sources are version-pinned local snapshots, built at Phase 10 and
refreshed deliberately. No external network call occurs on the request path.**

### 4.1 Homology — MMseqs2 against a Swiss-Prot snapshot

```
Swiss-Prot snapshot (same release as the training corpus)
        │
        ├─► mmseqs createdb  → target DB
        ├─► mmseqs createindex
        ▼
query sequence ──► mmseqs search ──► top-k hits
                   (identity, coverage, e-value, accession)
```

- **Database:** UniProtKB/Swiss-Prot, the **same release** frozen in Phase 3. Using the
  identical release for training and evidence means the evidence layer never references
  a protein the model has never seen, and the whole system pins to one release string.
- **Engine:** MMseqs2, already in the register (`THIRD_PARTY.md` §2.2) as an external
  binary — never linked, so its copyleft terms stay off our image.
- **Returns:** top-k hits with identity, coverage, e-value, accession, database name.

**A necessary exclusion.** Homologs are searched against Swiss-Prot, which *includes the
training partitions*. A nearest neighbour that is literally a training sequence is not
independent corroboration — it is the model's own training data reflected back as
though it were evidence. The homology response therefore **flags every hit that appears
in the training partitions** using the split registry, and the UI marks them. This costs
one lookup and prevents a genuinely misleading display.

### 4.2 GO — UniProt-GOA GAF snapshot

- **Source:** UniProt-GOA annotations, distributed by EBI in GAF format and freely
  downloadable.
- **Storage:** parsed into a local key-value store keyed by accession; GO term id, name,
  aspect (BP/MF/CC) and **evidence code** retained.
- **Evidence codes are shown, not hidden.** An `IEA` (electronically inferred, unreviewed)
  annotation and an `EXP` (experimental) annotation are not the same claim. The UI
  displays the code, and the GO panel states that unreviewed annotations are included.
  Silently mixing them would make the evidence panel look stronger than it is.

### 4.3 Provenance in every response

```json
"evidence": {
  "homology": {
    "source": "swissprot",
    "release": "<same as dataset manifest>",
    "engine": "mmseqs2",
    "engine_version": "<PIN>",
    "hits": [ { "accession": "...", "identity": 0.78, "coverage": 0.91,
                "evalue": 1e-40, "in_training_set": true } ]
  },
  "go": {
    "source": "uniprot-goa",
    "release": "<GAF release>",
    "sha256": "...",
    "terms": [ { "id": "GO:0005739", "name": "mitochondrion",
                 "aspect": "CC", "evidence_code": "IEA" } ]
  }
}
```

Both releases and the GAF hash go into the dataset manifest, so a stored result is fully
reconstructable.

### 4.4 Environment variables

`HOMOLOGY_DB_URL` and `GO_SOURCE_URL` are replaced — they described an architecture this
ADR rejects:

```env
EVIDENCE_ROOT=./data/evidence
SWISSPROT_DB_PATH=./data/evidence/swissprot_mmseqs
GOA_STORE_PATH=./data/evidence/goa.db
GOA_RELEASE=ENTER_YOUR_VALUE_HERE        # Phase 10
EVIDENCE_TIMEOUT_SECONDS=5
EVIDENCE_TOP_K=5
```

---

## 5. Consequences

**Positive.** A stored result is fully reproducible — every component names its source and
release. No request-path network dependency, so the degraded state becomes rare and the
throughput ceiling stays under our control. Proteome batches do not hit anyone's rate
limiter. Training-set homology hits are labelled rather than passed off as independent
evidence.

**Negative.** Disk: a Swiss-Prot MMseqs2 index plus a GOA store is a real footprint and
interacts with the hosting decision (ADR-0005 §4.3). Evidence goes stale between refreshes
— acceptable, because it is pinned to the training release by design, but it must be
stated in the UI so nobody reads a two-year-old GO term as current. Index build is a Phase
10 step with its own runtime.

**Neutral.** Refreshing evidence becomes a deliberate, versioned operation — the same
discipline as a model release.

---

## 6. Revisit triggers

1. Evidence storage exceeds the hosting disk budget (ADR-0005) — consider a reduced
   Swiss-Prot subset restricted to the organism groups actually served.
2. Users ask for current annotations rather than release-pinned ones — that is a *feature
   request for a second, clearly-labelled live lookup*, not a reason to change the default.
3. MMseqs2 search latency dominates per-sequence compute in Phase 13 profiling.

---

## 7. Verification

- [x] Sources decided: Swiss-Prot via MMseqs2, and UniProt-GOA GAF, both snapshotted.
- [x] `[CLIENT-TBD]` on `HOMOLOGY_DB_URL` / `GO_SOURCE_URL` closed; variables replaced.
- [ ] `GOA_RELEASE` and its SHA-256 recorded in the dataset manifest (Phase 10).
- [ ] `in_training_set` flag implemented and rendered in the evidence drawer.
- [ ] Evidence-failure degraded path tested (Blueprint §23.2).
