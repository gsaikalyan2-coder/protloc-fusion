# ADR-0010 — Baseline evaluation protocol: leave-one-partition-out on 1-4, test on 5, scored once

- **Status:** **Accepted** (2026-10-03)
- **Deciders:** Saikalyan
- **Related:** ADR-0002 (GraphPart split), ADR-0006 (label semantics), CLAUDE.md rule 9
  (freeze before you measure), `docs/phases/08.md`

---

## 1. Context

Phase 8 trains LogReg, a linear SVM and a random forest on the motif features and scores the
untrained motif rules beside them. Phase 9 will train the real localization head against the
same partitions. Whatever protocol Phase 8 uses becomes the protocol every later number is
compared with, and the test partition can be scored only once before it stops being a test.
Undoing the choice would mean re-running every model, so it needs an ADR (CLAUDE.md rule 10).

Two facts forced a decision:

- **Numbering.** The registry numbers partitions 1 to 5 (9,373 / 6,060 / 6,008 / 5,885 /
  6,553 accessions). `configs/evaluation/default.yaml` says validation `[0, 1, 2, 3]` and
  test `[4]`, counting from zero.
- **No separate model-selection set.** Phase 6 made five partitions and no calibration split
  (its note says a calibration split must be made from whole `cluster_id` clusters later).

## 2. Decision

1. **Mapping.** Registry partition *n* is the zero-based index *n - 1*. Validation is registry
   partitions **1, 2, 3, 4**; test is registry partition **5**. This ADR records the mapping;
   `configs/evaluation/default.yaml` is **not edited here**, because Phase 9 owns it. Until
   Phase 9 reconciles the file, the two numberings disagree on paper and agree in meaning.
2. **Model selection: leave-one-partition-out (LOPO).** Four folds. Fold *k* trains on the other
   three of partitions 1-4 and predicts partition *k*. Partition 5 is never in a training set,
   a fold, a threshold or a hyperparameter choice.
3. **Hyperparameters** come from a small fixed grid (`protloc.training.baselines.GRIDS`). The
   chosen point is the one with the highest mean over the four folds of per-fold macro F1.
   Ties go to the earlier grid point. The chosen point is refit on partitions 1-4.
4. **Thresholds.** One per class, the score that maximises MCC (a positive call is
   `score >= threshold`). No literal is used anywhere; thresholds are stored in
   `data/manifests/baseline_metrics.json`. Fold *k* is scored with thresholds tuned on the
   out-of-fold scores of the **other** validation partitions, so a number is never measured
   with a threshold that saw its own labels. The **frozen** thresholds are tuned on the
   out-of-fold scores of all four validation partitions and are what the test uses.
5. **Freeze, then test.** `--freeze` writes the manifest with validation numbers, parameters,
   thresholds, input hashes and software versions, and a SHA-256 of that whole block. Only then
   does `--evaluate --test` score partition 5, once, adding a `test` block that repeats the
   block hash. The command refuses to run if the manifest is missing, unsound, or already has
   a `test` block. `--check` fails if the hash, the touch count (must be exactly 1) or the
   coverage of the test block is wrong.
6. **Metrics.** Macro F1, per-class MCC and F1, sample-wise Jaccard, per-class support.
   F1 and MCC are 0.0 where their denominator is 0. Sample-wise Jaccard is 1.0 for a row whose
   predicted and true sets are both empty. Intervals are percentile 2.5 to 97.5 over 1,000
   bootstrap samples (`reporting.bootstrap_samples` in the evaluation config) that **resample
   whole `cluster_id` clusters**, because homologous rows in one cluster are not independent.
   The seed is `seed` in `configs/config.yaml`.
7. **Labels.** Only `annotated` rows are positives. `unknown` and `not_applicable` rows are
   dropped from a task, never treated as negatives (ADR-0006). Two informative negatives are
   read, both made informative by ADR-0006 §3.3: `soluble` (membrane type) and
   `no_signal_peptide`. Localization has no per-class negatives in the corpus, so each class
   is a one-vs-rest problem over `annotated` rows with a closed-world reading at fit time.
8. **Scope.** Three tasks: localization (10 classes, multi-label), membrane type (4 collapsed
   classes) and signal-peptide presence (2 mutually exclusive columns). Class tables are in
   `backend/rules/baseline_class_map.yaml`, not in code.
9. **Rows.** (1) models trained on the 29-column numeric motif vector only (scaler fit on each
   training fold only); (2) the untrained motif rules (`raw_scores`, engine group `plant`),
   scored on the same rows with thresholds tuned the same way. `raw_scores` is never a model
   input.

## 3. Consequences

- Every Phase 8 number is a LOPO out-of-fold number on partitions 1-4 or a single test number
  on partition 5. Neither is the DeepLoc benchmark, so comparison with DeepLoc's published
  figures is indicative only.
- Phase 9 inherits the mapping and must reconcile `configs/evaluation/default.yaml`
  (`validation: [1,2,3,4]`, `test: [5]`, or an explicit offset) before it reports anything.
- The cost of the safeguards is time: LOPO refits each grid point four times.

## 4. Known limits (stated, not hidden)

- The thresholds used to score fold *k* were tuned on out-of-fold scores that came from models
  which had fold *k* in their training data. Fold *k*'s labels never entered the threshold
  fit, but they shaped the models that produced the scores. A fully nested scheme would
  remove this at about four times the cost. It is not used here.
- The frozen thresholds come from models trained on three partitions and are applied to a model
  trained on four. The score scale can shift slightly between the two.
- Row counts differ per task because `unknown` rows are dropped from membrane type and signal
  peptide; the per-task counts per partition are in the manifest.
- Clusters are GraphPart clusters at 30% identity. Bootstrapping them is a better unit than
  rows, not a proof of independence.

## 5. Alternatives rejected

- **Follow the config literally** (0-3 / 4): ambiguous against a registry numbered 1-5;
  one off-by-one would put the wrong partition under the test label.
- **Random k-fold:** homologous sequences would cross folds and inflate every number
  (ADR-0002).
- **Tune thresholds on test, or choose the best model by test score:** invalidates the test
  (CLAUDE.md rule 9).
- **Fixed 0.5 threshold:** the abstention and class-imbalance story of this project is that a
  threshold is a measured quantity (CLAUDE.md, "no hardcoded tau").
