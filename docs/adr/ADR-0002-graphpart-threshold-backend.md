# ADR-0002 — Homology partitioning: threshold, backend and partition count

- **Status:** **Accepted** (2026-09-09)
- **Deciders:** Saikalyan
- **Blocks:** Phase 4 (Homology splits & leakage gates), and transitively every phase
  that reports a metric
- **Related:** Blueprint §7.3–7.5; `docs/EXTERNAL_WORKFLOW.md` §2; `docs/THIRD_PARTY.md` §2.1

---

## 1. Context

Phase 4 builds the split that every reported number in the paper depends on. Three
parameters must be frozen before it runs, and none can be changed afterwards without
invalidating every downstream result:

1. **Identity threshold** — the maximum sequence identity permitted *between* partitions.
2. **Alignment backend** — GraphPart's `needle`/`needleall` global alignment, or MMseqs2
   pre-clustering.
3. **Partition count** — how many folds.

Blueprint §7.3 is explicit that ProtLoc Fusion must document its own threshold rather
than copy one without validation. This ADR documents the choice and the reasoning; it
does not claim the value was derived experimentally, because it was not — it was chosen
for comparability, which is a different and weaker justification that must be stated
plainly.

---

## 2. Decision drivers

1. **Comparability with the tools being replaced.** ProtLoc Fusion's central claim is
   that it improves on DeepLoc, SignalP, TargetP and PSORTb. A comparison across
   differently-constructed splits is not a comparison.
2. **Citability.** A reviewer should recognise the protocol without having to trust a
   bespoke justification.
3. **Honesty about difficulty.** A threshold chosen to make the test set easy is
   scientific self-harm. Lower identity between partitions means a harder, more
   informative benchmark.
4. **Cost.** Global pairwise alignment over the full corpus is expensive; the backend
   choice is partly a compute decision.

---

## 3. Options considered

### Threshold

| Option | For | Against |
|---|---|---|
| **30% max global identity** | Directly matches DeepLoc 2.1, which partitioned its dataset into five partitions with GraphPart at a maximum 30% global identity (Ødum et al., *NAR* 2024). Comparisons become like-for-like. Widely used in the localization literature. | 30% is generous by modern standards — remote homology persists below it, so some leakage-like advantage remains |
| 25% or 20% | Harder, more honest generalization test | No comparability with the baseline tools; more sequences discarded; no precedent to cite |
| 40%+ | Retains more data | Test set becomes unrealistically easy; the exact failure §7.3 warns against |

### Backend

| Option | For | Against |
|---|---|---|
| **GraphPart `needleall`** | True global alignment identity — the quantity the threshold actually names. Matches the DeepLoc 2.1 protocol | Slower; requires EMBOSS installed |
| MMseqs2 `easy-cluster` pre-clustering | Much faster on large corpora | Clusters at an approximate identity; "30%" then means something subtly different, breaking the comparability that motivated the threshold |

### Partition count

Five partitions, matching DeepLoc 2.1 — usable as five-fold cross-validation during
development and collapsible into train/validation/test for the final protocol.

---

## 4. Decision

**Freeze the split as: maximum 30% global sequence identity between partitions,
GraphPart with the `needleall` alignment backend, five partitions.**

```bash
graphpart needle \
  --fasta-file data/processed/protloc_corpus_v1.fasta \
  --threshold 0.3 \
  --partitions 5 \
  --labels-name localization \
  --out-file data/manifests/splits_homology_v1.csv
```

Exact flags are verified against the pinned GraphPart version at Phase 4; the
*parameters* — 0.3, needleall, 5 — are what this ADR freezes.

### 4.1 Final protocol

Partitions 1–3 → train · partition 4 → validation · partition 5 → test. The test
partition is **not opened** until Phases 8–9 have frozen every tunable value on
validation (`CLAUDE.md` phase rule 9).

### 4.2 Required manifest fields

```yaml
split_version: homology_v1
partitioner: graphpart
partitioner_version: <PIN at Phase 4>
alignment_backend: needleall
identity_threshold: 0.30
identity_type: global
partitions: 5
sequences_before: <N>
sequences_removed: <N>
removal_rate: <PCT>
leakage_assertions: PASS
```

`sequences_removed` and `removal_rate` go in the paper's dataset table. They are the
honest cost of the split and hiding them is how a dataset section becomes unreconcilable.

### 4.3 Independent verification is still required

**GraphPart's output is not taken on trust.** Phase 4's leakage assertions run
separately, as Blueprint §7.4 specifies: zero exact-duplicate pairs across partitions,
zero shared homology clusters, and the maximum observed cross-partition identity
reported.

This matters because published splits have failed before. MULocDeep (Jiang et al., 2021)
reported that a DeepLoc dataset described as split at 30% identity with PSI-CD-HIT
contained substantial cross-split similarity in practice. The lesson is not that DeepLoc
was careless — it is that *asserting* a threshold and *verifying* it are different acts,
and only the second belongs in a methods section.

### 4.4 Secondary split — no extra cost, real extra evidence

Also produce a **random split** at the same partition sizes. It is not a serious
evaluation protocol; it is a diagnostic. The gap between random-split and homology-split
scores measures how much of the model's apparent performance is homology memorisation.
Reporting both makes the homology number credible in a way that reporting it alone
cannot.

A temporal split (Blueprint §7.5) remains optional and is deferred to post-launch.

---

## 5. Consequences

**Positive.** Head-to-head comparison with DeepLoc 2.1 is like-for-like. The protocol is
citable in one sentence. Five partitions give cross-validation during development at no
extra cost. The random-split diagnostic adds a defensible figure for free.

**Negative.** `needleall` global alignment over the full corpus is compute-heavy and will
be one of Phase 4's slower steps; budget for it and run it once. 30% is a permissive
threshold, so remote-homology advantage is not eliminated — this must be stated as a
limitation, not glossed. Some sequences are discarded.

**Neutral.** EMBOSS becomes a hard dependency of `Dockerfile.train` (already recorded in
`docs/EXTERNAL_WORKFLOW.md` §8).

---

## 6. Revisit triggers

1. Phase 4's `removal_rate` exceeds **15%** — the threshold is destroying the corpus and
   the trade-off needs re-examining.
2. `needleall` runtime exceeds a working day on the final corpus — evaluate MMseqs2
   pre-filtering *before* needleall (a hybrid), never as a silent replacement.
3. A reviewer challenges 30% as too permissive — the honest response is the random-split
   diagnostic from §4.4, not a re-split.

---

## 7. Verification

- [x] Threshold, backend and partition count frozen: 0.30 · needleall · 5.
- [ ] `partitioner_version` pinned at Phase 4.
- [ ] `leakage_report.json` shows 0 cross-partition duplicates and 0 shared clusters.
- [ ] `removal_rate` recorded in the dataset card and the paper's dataset table.
- [ ] Random-split diagnostic produced alongside the homology split.

---

## 8. Sources

- Ødum MT, et al. **DeepLoc 2.1: multi-label membrane protein type prediction using
  protein language models.** *Nucleic Acids Research*, 2024 — five GraphPart partitions,
  maximum 30% global sequence identity.
- Teufel F, et al. **GraphPart: homology partitioning for biological sequence analysis.**
  *NAR Genomics and Bioinformatics*, 2023.
- Jiang Y, et al. **MULocDeep.** *Computational and Structural Biotechnology Journal*,
  2021 — reports cross-split similarity in a dataset described as 30%-partitioned.
- Florensa AF, et al. **SpanSeq: similarity-based sequence data splitting.** 2024 —
  demonstrates that random and high-similarity partitions produce rapid overfitting.
