# EXTERNAL_WORKFLOW.md — ProtLoc Fusion

How each external repository slots into the pipeline, and — more importantly —
where the boundary sits between *their* workflow and *ours*.

Companion to `THIRD_PARTY.md` (the legal/provenance register). This document is
the engineering view: what each tool does, at which stage, with what inputs and
outputs, and what we still own.

---

## 0. The integration principle

> **Adopt workflows for anything a reviewer would recognise. Own everything
> that constitutes the contribution.**

Reimplementing homology partitioning or ECE binning gains nothing and costs
credibility — a reviewer has to take your bespoke script on faith. Conversely,
importing someone's multi-task architecture gains a shortcut and costs the
paper its claim.

So the line falls here:

| Adopt (external) | Own (ours) |
|---|---|
| Homology partitioning (GraphPart) | The shared trunk + seven heads |
| Clustering backend (MMseqs2) | Task masking and loss weighting |
| PLM embeddings (ESM-2) | Feature fusion of PLM + classical branch |
| Calibration mappings and ECE (netcal) | Abstention policy and coverage–risk analysis |
| LoRA machinery (PEFT) | Which modules to adapt, and the trade-off study |
| Config/experiment scaffolding pattern (Hydra) | The biological consistency engine |
| Benchmark baselines (DTU tools, PSORTb) | The evidence layer and served-path evaluation |

The right-hand column is the paper. Nothing in it is imported.

---

## 1. Stage-by-stage integration

```
RAW UNIPROT SNAPSHOT
        │
        ├─ parsing + normalization ....................... ours
        ├─ duplicate detection (sequence hashing) ........ ours
        │
        ▼
HOMOLOGY PARTITIONING ............................ GraphPart (+ MMseqs2)
        │   graphpart needle --fasta-file <in> --threshold <T>
        │                    --partitions 5 --out-file splits.csv
        ▼
SPLIT REGISTRY + LEAKAGE ASSERTIONS .............. ours
        │
        ▼
EMBEDDING GENERATION ............................. ESM-2 (facebookresearch/esm)
        │   cache key = sha256(sequence) + checkpoint + revision
        ▼
CLASSICAL FEATURE BRANCH ......................... ours (biopython-assisted)
        │
        ▼
SHARED TRUNK + 7 TASK HEADS ...................... ours
        │   adaptation layer ......................... PEFT (LoRA)
        ▼
CALIBRATION FIT .................................. netcal
        │   temperature / vector scaling, ECE, reliability
        ▼
ABSTENTION POLICY ................................ ours
        │
        ▼
BIOLOGICAL CONSISTENCY ENGINE .................... ours (YAML rules)
        │
        ▼
INTERPRETABILITY + EVIDENCE ...................... ours
        │
        ▼
SERVED PATH (FastAPI)
        │
        ▼
EVALUATION ....................................... ours, via the served path
        └─ external comparison ...... DTU tools + PSORTb, run offline
```

---

## 2. GraphPart — the split step

**Replaces:** Blueprint §7.3's hand-rolled clustering split.

**Workflow adopted.** GraphPart builds a similarity graph over the sequences and
partitions it so that closely related sequences always land in the same
partition, then removes the minimum number of sequences needed to satisfy the
constraint. This is stronger than cluster-then-assign: it makes an explicit,
reportable trade-off between partition balance and how many sequences had to be
discarded.

**Our wrappers.**

- `scripts/cluster_sequences.py` → invokes `graphpart`, captures its stdout,
  stderr, exit code and version into the dataset manifest.
- `scripts/build_splits.py` → converts GraphPart's partition assignment into
  our split registry format, then runs the leakage assertions from Blueprint
  §7.4 **independently**. We verify GraphPart's output rather than trusting it;
  the assertions are the contract, GraphPart is the implementation.

**What must be recorded in the dataset manifest:**

```yaml
split_version: homology_v1
partitioner: graphpart
partitioner_version: <VERSION>
alignment_backend: needleall        # or mmseqs2
identity_threshold: <FREEZE>
partitions: 5
sequences_removed: <N>              # GraphPart's own trade-off number
leakage_assertions: PASS
```

`sequences_removed` matters. It is the honest cost of the split and belongs in
the paper's dataset table, not hidden in a log.

**Open decision (needs an ADR before freezing):** identity threshold, and
needleall vs MMseqs2 as the backend. DeepLoc 2.1's threshold is a useful
reference point, not a value to copy — Blueprint §7.3 is explicit about this.

**Dependency note:** GraphPart needs EMBOSS `needleall` on PATH. This must be
in the training image, and it is a common cause of a pipeline that runs on one
machine and not another.

---

## 3. netcal — the calibration step

**Replaces:** hand-written temperature scaling and ECE.

**Workflow adopted.** Fit a calibration mapping on the *validation* split only,
then apply it, frozen, to the test split. netcal supplies the mappings
(temperature scaling, vector scaling, binning methods) and the measurement
tools (ECE, reliability diagrams).

**Sequence, per task head:**

```
head logits (validation) ──► netcal fit ──► calibrator object
                                                 │
                                                 ├─► ECE / Brier / NLL
                                                 ├─► reliability diagram
                                                 └─► serialized into the registry
```

**Non-negotiables:**

1. **One calibrator per head.** Seven heads means seven calibrator objects.
   A single global calibrator would average away exactly the per-task
   miscalibration you are trying to measure.
2. **Fit on validation, never on test.** A calibrator fitted on test makes ECE
   meaningless and is the most common way this analysis goes wrong.
3. **Record the binning strategy and bin count** alongside every ECE value.
   ECE is not a single well-defined number; an unqualified ECE invites the
   reviewer question you cannot answer afterwards.
4. **The calibrator ships in the registry bundle** with the checkpoint and the
   consistency rules, as one atomic version (Blueprint §24.3). A calibrator
   paired with the wrong checkpoint produces confident nonsense.

**What netcal does not do:** decide τ, decide when to abstain, or produce the
coverage–risk curve. Those are ours, and they are what H6 tests.

---

## 4. PEFT — the LoRA experiment

**Workflow adopted:** wrap the ESM-2 trunk in a LoRA configuration, train, and
compare against both a frozen-trunk baseline and full fine-tuning.

The comparison is a **trade-off curve, not a winner** (Blueprint §12.2). Three
runs, identical in every other respect — same split, same seed, same loss
weights — differing only in adaptation mode:

| Run | Adaptation | Trainable params | Peak GPU mem | Epoch time | Test macro-F1 |
|-----|-----------|------------------|--------------|------------|---------------|
| A | frozen trunk | | | | |
| B | LoRA (r=?, α=?) | | | | |
| C | full fine-tune | | | | |

"Comparable quality at a fraction of the trainable parameters" is a stronger and
more defensible result than "LoRA won."

---

## 5. External baselines — the benchmark step

**Tools:** DeepLoc 2.1, SignalP 6.0, TargetP 2.0, PSORTb.

**Workflow: strictly offline, strictly one-directional.**

```
frozen test split (FASTA)
        │
        ├──► DeepLoc 2.1   ──┐
        ├──► SignalP 6.0   ──┤
        ├──► TargetP 2.0   ──┼──► raw outputs (versioned, in benchmarks/)
        └──► PSORTb        ──┘         │
                                       ▼
                    scripts/benchmark_external.py
                                       │
                                       ▼
                    normalized metrics, our label schema
```

**Rules:**

1. Run locally, in their own environments or containers. Never in our image,
   never in the served product, never called at request time.
2. **Label schema mapping is a documented artifact, not an inline dict.**
   Their class vocabularies do not align with ours. Every mapping decision —
   and every class that has no counterpart — goes in
   `benchmarks/label_mapping.yaml` with a rationale. This is where quiet
   score inflation happens, so it gets its own reviewable file.
3. Store the raw outputs, not just the parsed metrics. A reviewer asking
   "how did you score their abstentions?" needs an answer that is checkable.
4. Record each tool's exact version and the date it was run. These are moving
   services; a number from 2026 is not reproducible in 2028 without that.

**The contamination caveat.** These tools were trained on UniProt releases
overlapping your test split. Say so in the paper, in the results section, not
buried in limitations.

---

## 6. Hydra scaffolding

**Adopted as a pattern, not a template.** `lightning-hydra-template` demonstrates
the config-composition discipline the blueprint already calls for: a single
config tree, composable experiment overrides, and every run reconstructable from
its resolved config.

Take the pattern:

- `configs/` split into `data/ model/ training/ evaluation/ deployment/`
- experiments as thin override files, not forked scripts
- the fully resolved config hashed into the run manifest (`config_sha256`)

Do not take the template wholesale. Its directory layout assumes a
Lightning-centric project; ProtLoc Fusion's hexagonal architecture (17 modules,
L0→L5) has a different dependency shape, and bending one to the other would
cost more than it saves.

---

## 7. What this changes in the existing documents

| Document | Change |
|---|---|
| `ARCHITECTURE.md` | `data` module gains an explicit external-tool adapter boundary — GraphPart is invoked through a port, so the split step is swappable and testable |
| Dataset manifest schema | Add `partitioner`, `partitioner_version`, `alignment_backend`, `sequences_removed` |
| `calibration` module | Wraps netcal rather than implementing scaling; abstention stays ours |
| Registry bundle | Calibrator objects are netcal-serialized; record the netcal version in `metadata.json` |
| Definition of Done §30.1 | Add: every external tool version and run date recorded in `benchmarks/` |
| New file | `benchmarks/label_mapping.yaml` — external-to-ours label schema, with rationale |
| New ADRs required | GraphPart threshold + backend · calibration method choice · Prot2Token AGPL ruling |

---

## 8. Three risks worth naming now

**EMBOSS is a hidden dependency.** GraphPart shells out to `needleall`. It is
not a pip package, it will not be in a bare Python image, and the failure
surfaces as a confusing subprocess error at split time. Put it in the training
Dockerfile the same day you adopt GraphPart.

**Prot2Token is a licensing trap, not a shortcut.** It is the nearest published
work to your thesis and the most tempting to borrow from — and it is AGPL with a
research-only restriction, aimed squarely at a project whose stated deliverable
is a public network service. Cite it, engage with it in related work, and keep
its code out of `src/`.

**GraphPart will delete some of your data.** That is the algorithm working as
designed, not a bug — but it means your final sequence count will not match your
post-filtering count, and if you notice that during paper writing rather than
now, you will have a table that does not reconcile. Track it from the first run.
