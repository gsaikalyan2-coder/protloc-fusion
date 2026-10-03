# ADR-0011 — Localization head: attention pooling over per-residue ESM-2 states, trained and scored on the 500-sequence subset

- **Status:** **Accepted** (2026-10-03)
- **Deciders:** Saikalyan (architecture, loss, optimiser and protocol confirmed at the Phase 9 gates)
- **Related:** ADR-0006 (label semantics), ADR-0009 (frozen ESM-2 150M, prototype subset),
  ADR-0010 (evaluation protocol), CLAUDE.md rules 9 and 10, `docs/phases/09.md`

---

## 1. Context

Phase 8 fixed the floor: three classical models and the untrained motif rules on 29 motif
features, scored by leave-one-partition-out (LOPO) on registry partitions 1-4 and once on
partition 5. Phase 9 trains the first real head on the shared representation, the
localization head (10 classes, multi-label), and must be comparable with that floor.

Four facts forced decisions:

- **Prototype scope.** ADR-0009 scoped the prototype to a 500-sequence ESM-2 subset. The
  full 33,461-sequence extraction is not run in this phase. Every Phase 9 number is therefore
  a subset number.
- **No per-residue states.** Phase 7 cached one mean-pooled vector per sequence and ADR-0009
  section 2.7 chose not to store per-residue states (19.1 GB in float16 for the full corpus).
  An attention head pools over residues, so it needs them. A mean-pooled vector cannot give
  attention weights, and Phase 10 needs weights to explain.
- **Two numberings of the partitions.** `configs/evaluation/default.yaml` said validation
  `[0, 1, 2, 3]` and test `[4]`; the registry says 1-4 and 5. ADR-0010 section 2.1 recorded the
  mapping and left the file to this phase.
- **Tiny supervision.** 500 sequences give about 507 annotated localization rows, so a class can
  have a handful of positives in a fold and the confidence intervals are wide.

## 2. Decision

1. **Scope.** The head is trained and evaluated on the 500 sequences of the Phase 7 subset
   and nothing else. The subset is resolved from the Phase 7 pooled cache's committed shard
   tables (`protloc.training.subset`), proven against the feature table by the
   `log1p_length` column, and recorded in the manifest (`subset.sequences_sha256`). Every
   number carries the label *subset (500), not comparable to DeepLoc or to the Phase 8
   full-corpus numbers*. The code reads the cache through its API and runs unchanged on the
   full cache.
2. **Architecture** (`configs/training/head.yaml`, `protloc.training.head_net`). For a sequence
   of per-residue states `x` (L x 640):
   - attention weights `a = softmax over residues of Linear(64 -> heads)(tanh(Linear(640 -> 64)(x)))`,
     one head, padding masked to exactly 0;
   - pooled vector `sum_l a_l x_l`;
   - `LayerNorm`, `Linear(640 -> hidden)`, GELU, dropout, `Linear(hidden -> 10)`;
   - 10 independent sigmoids (multi-label). Class order is the column order of
     `backend/rules/baseline_class_map.yaml`.
   `forward` returns the logits and the attention weights (B, heads, L); Phase 10 wires them
   into the explanation output. With hidden 256 the head has 209,035 trainable parameters.
   The model is a plain class around `torch.nn.ModuleDict`, not an `nn.Module` subclass, so strict
   mypy holds without torch stubs and torch stays a lazy import.
3. **Per-residue cache (amends ADR-0009 section 2.7).** Per-residue states are stored
   for the 500-sequence subset only: float16, 50 sequences per shard, its own spec and key,
   beside the pooled cache and never mixed into it (`protloc.data.residue_cache`,
   `scripts/embed.py --residues --subset 500`, about 286 MB). A residue's state is the last hidden
   state averaged over the sliding windows that saw it, which is exactly what Phase 7's pooled vector
   averages, so the mean of a sequence's stored states must reproduce its pooled vector. The
   extraction run measures the largest element difference and the manifest records it; the check
   fails above 5e-3. The full corpus stays at 19.1 GB and is not stored in this phase.
4. **Labels (ADR-0006).** Only `annotated` rows are labelled. `unknown` and `not_applicable` are
   dropped, never negatives. A class with fewer than 2 positives in the rows being scored has no
   defined MCC: it is reported as MCC 0.0, marked `"undefined": true`, kept in the table and in
   macro F1, never dropped. The same rule is applied to the floor (`baseline_models.metrics_report`),
   so the head and the floor are scored alike.
5. **Loss.** Per-class `pos_weight = negatives / positives` computed from the training rows only
   (1.0 for a class with no positive there), times the focal factor `(1 - p_t) ** gamma`, averaged
   over rows and classes. `gamma = 0` is exactly binary cross-entropy with `pos_weight`;
   `gamma = 2` is the focal form CLAUDE.md names. Both are in the grid. The weights are computed
   from the rows the loss sees (the fit rows of a fold), never from the validation or test rows
   (a unit test intercepts the call and checks its argument).
6. **Optimiser and grid** (`configs/training/head.yaml`, nothing repeated in code). AdamW,
   weight decay 0.01, batch size 32 (length-bucketed), seed 20260909, deterministic torch flags.
   Grid: learning rate {1e-3, 3e-4} x dropout {0.3, 0.5} x hidden {256, 128} x gamma {0, 2} =
   16 points. The chosen point is the one with the highest mean per-fold macro F1 under nested
   thresholds (ADR-0010 section 2.3); ties go to the earlier point.
7. **Protocol (ADR-0010, unchanged).** LOPO over registry partitions 1-4; fold *k* thresholds are
   tuned on the other validation partitions' out-of-fold scores; frozen thresholds on all
   validation out-of-fold rows; no literal 0.5. Test is partition 5, scored once, after a
   freeze that writes `frozen_sha256`. Macro F1, per-class F1 and MCC, sample-wise Jaccard and
   support are reported with a cluster bootstrap (1,000 samples, seed 20260909, resampling
   `cluster_id`). Before every fit the cluster sets of the fit rows, the early-stopping holdout,
   the validation rows and the test partition are asserted pairwise disjoint.
8. **Early stopping on an inner whole-cluster holdout.** A head needs a stopping signal and
   the outer folds are the evaluation, so inside each fold the training partitions are split
   by whole clusters (about 20 percent of the clusters, `Shuffler(seed)`) into fit and holdout. After each
   epoch the holdout macro F1 (MCC-optimal thresholds tuned on the holdout) is computed;
   training stops after 10 epochs without improvement (at most 80) and the best epoch's weights are kept.
   The final head is refit on all of partitions 1-4 for the median best epoch of the chosen
   point's four folds, with no holdout.
9. **Freeze and test.** `scripts/head.py --freeze` writes `data/manifests/head_metrics.json`
   (selected hyperparameters, thresholds, SHA-256 of the saved weights, validation metrics, every
   input hash, software and hardware, `frozen_sha256`, `test: null`) and refuses if it exists.
   `--evaluate --test` refuses unless `check_head_manifest` finds nothing wrong, the test block is
   absent and the weights hash matches; it then writes the test block (`touches: 1`). `--check`
   needs no data bytes. Smoke output goes to `data/head/smoke` and can never be frozen.
10. **Config reconciliation (supersedes the note in ADR-0010 section 2.1).**
    `configs/evaluation/default.yaml` now says validation `[1, 2, 3, 4]` and test `[5]`, the
    registry's numbering, and `test_evaluation_partitions.py` fails if the config and the registry
    disagree.

## 3. Consequences

- The head and the floor share rows, folds, thresholds rule, bootstrap and undefined-class
  rule, so their rows in one table are comparable with each other (and only with each other).
- Phase 10 gets attention weights from `forward` and `predict(..., attention=True)`.
- Moving to the full corpus means running the full per-residue extraction (19.1 GB) and
  `head.py` unchanged against it; the subset manifest is not reused.
- `training/` is an entry-point layer imported by nothing; torch is the optional `head` extra and
  never reaches the served image.

## 4. Known limits (stated, not hidden)

- **Early-stopping F1 is optimistic.** Its thresholds are tuned on the same holdout rows it scores.
  It only picks an epoch and is never reported; the reported numbers use thresholds from other folds.
- **Selection and reporting share the outer folds,** as in the floor: the grid point with the best
  mean fold macro F1 is also the one whose validation numbers are reported, so the validation
  figure is mildly optimistic. The test partition is the unbiased estimate.
- **The test partition is small** (99 accessions in the subset), so test intervals are wide, and a
  class with fewer than 2 test positives is marked undefined. A single test number cannot separate
  the head from the floor unless the gap is large.
- **Subset only.** 500 sequences cannot support claims about the full task or about DeepLoc.
- **One training seed.** The seed is fixed, so seed-to-seed variance is not measured.
- **float16 storage.** States are rounded to float16; the measured difference from the pooled
  vector is recorded, not assumed.

## 5. Alternatives rejected

- **Mean pooling plus an MLP.** Reuses the Phase 7 cache, but gives no attention weights, so
  Phase 10's per-residue explanation would need another model.
- **Plain BCE with `pos_weight` only.** The prompt named it; CLAUDE.md names focal. Because
  `gamma = 0` is exactly that loss, the grid holds both and the validation folds choose.
- **Early stopping on the validation fold.** Would make the fold's score a function of the data it
  is scored on. The inner whole-cluster holdout costs a fifth of the training rows to avoid it.
- **Early stopping on a random row holdout.** Homologous rows in one cluster would sit on both sides
  and stop training too late.
- **A literal 0.5 threshold.** With 10 imbalanced classes it is wrong for most of them (ADR-0010).
- **Extracting the full corpus now.** Out of scope for the prototype; the code is written for it.
