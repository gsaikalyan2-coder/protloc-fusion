# ADR-0012 — Calibration and per-label thresholds for the localization head: cross-fitted on the out-of-fold scores, frozen in the registry bundle

- **Status:** **Accepted** (2026-10-03)
- **Deciders:** Saikalyan (calibration split, partition-5 policy, eligibility rule, ECE binning and registry location confirmed at the Phase 10 gates)
- **Related:** ADR-0006 (label semantics), ADR-0010 (evaluation protocol), ADR-0011 (localization head),
  CLAUDE.md rules 9 and 10 and "No hardcoded tau", `PROJECT_PLAN.md` section 4 row 10 and section 6
  ("Unversioned artifacts"), `docs/phases/10.md`

---

## 1. Context

Phase 9 froze a head (`data/manifests/head_metrics.json`, `frozen_sha256` `719c9866603640a781bf13193e38ee5d27c7fd13426aa44c5221a1e4099c9dae`)
whose scores are sigmoid outputs of a loss that re-weights positives, so they are not probabilities. Its ten
thresholds were tuned on those raw scores. Phase 10 makes the scores mean something and fixes the decision
thresholds on the calibrated scale.

Facts that forced decisions:

- **No calibration set exists.** Phase 6 made five GraphPart partitions and no calibration split; its note says
  one must be made from whole `cluster_id` clusters. The subset has 408 annotated validation rows (partitions 1-4)
  and 99 test rows (partition 5).
- **Partition 5 was scored once.** `head_metrics.json` has `test.touches: 1`. A second look is a second touch.
- **Tiny positive counts.** Validation positives per label: cytoplasm 136, nucleus 125, extracellular 86,
  mitochondrion 46, cell membrane 45, ER 41, Golgi 18, lysosome/vacuole 17, plastid 13, peroxisome 6. Isotonic
  regression on a handful of positives memorises them.
- **The registry was an empty package** (`backend/protloc/registry/__init__.py`, a docstring). There was no
  loader, no `registry/` directory and no atomic bundle, though `PROJECT_PLAN.md` section 6 requires that a
  checkpoint is never served with another checkpoint's calibrator.

## 2. Decision

1. **Phase 10 supersedes the Phase 9 raw thresholds.** The ten thresholds in `head_metrics.json` were tuned on
   uncalibrated scores and stay there as the record of Phase 9; they are not used after Phase 10. The new
   thresholds are tuned on calibrated out-of-fold scores, nested the way ADR-0010 nests them: a fold is scored
   with thresholds that never saw it. `calibration_metrics.json` records the Phase 9 numbers next to them
   (`phase9_reference`) and the evaluation re-computes the raw decisions and fails unless they reproduce the Phase 9
   validation macro F1 to 1e-9.
2. **Calibration split: cross-fitting on the 408 out-of-fold validation scores.** Leave-one-partition-out over
   registry partitions 1-4, the folds of ADR-0010. Fold *k*'s calibrator is fitted on the out-of-fold scores of the
   other three validation partitions and applied to partition *k*. The frozen calibrator is fitted on all 408.
   Partitions are whole `cluster_id` clusters, so no homologous pair straddles a fit and a scored set. Before every
   fit the cluster sets of the calibrator rows, the scored rows and the test partition are asserted pairwise
   disjoint (`head.assert_no_overlap`).
3. **Doubly nested thresholds.** Fold *k*'s thresholds must not see fold *k*'s labels, directly or through a
   calibrator. They are therefore tuned on the scores of each other partition *m* calibrated by a calibrator fitted
   on the two partitions that are neither *k* nor *m*. An inner calibrator reuses the method of its outer
   calibrator (fold *k*'s), so the thresholds sit on the same score scale as the scores they are applied to; if
   the inner rows hold no positive or no negative it falls back to the identity and the report says so. The frozen
   thresholds are tuned on the cross-fitted scores of all four partitions, never on in-sample calibrated scores
   (a calibrator scores its own training rows too well). Per label, the threshold is the calibrated score that
   maximises MCC (`baseline_models.best_threshold`); there is no literal 0.5.
4. **Calibrator and eligibility rule (`configs/training/calibration.yaml`).** Counted on the rows each
   calibrator is fitted on: isotonic regression if the label has at least 20 positives and 20 negatives; otherwise
   Platt scaling (a sigmoid of the score's logit, two parameters, Newton's method on Platt's smoothed targets) if it
   has at least 5 of each; otherwise **`uncalibrated`**: the identity, `calibrated: false`, threshold still tuned,
   never dropped (mirroring the Phase 9 "undefined" rule). Platt, not a single temperature, is the fallback: a
   temperature cannot move a base rate, and the labels that fall back are the rare ones. Isotonic is stored as its
   fitted knots and applied by linear interpolation clipped to the end values; nothing is pickled.
5. **Labels (ADR-0006).** Only `annotated` rows. `unknown` and `not_applicable` are dropped, never negatives. A
   label with fewer than 2 positives in the rows being scored has MCC 0.0, is marked `undefined` and is kept
   in every table and in macro F1 (`baseline_models.metrics_report`, unchanged).
6. **ECE: 10 equal-mass bins, lowered so each holds at least 20 rows, ties never split.** With 408 rows and rare
   labels, equal-width bins are mostly empty. Reported per label and macro with Brier, reliability bin data and a
   cluster bootstrap (1,000 samples, seed 20260909, resampling `cluster_id`) for ECE, Brier and their change
   (calibrated minus raw). The bootstrap resamples the fixed cross-fitted scores; it does not refit the calibrators.
7. **Partition 5 is not touched in Phase 10.** It stays at `touches: 1` from Phase 9. Everything is reported on
   out-of-fold validation. There is no test mode in `scripts/calibrate.py`; `check_manifest` fails a manifest that
   carries a test block or a policy other than `untouched`. With 99 rows and 2-7 positives on several labels, a
   second scoring would add almost no evidence and would spend the partition.
8. **The calibrator and thresholds live in the registry bundle, not in code.** `registry/localization/<version>/`
   holds `calibrator.json` (per-label method, knots or Platt parameters, counts, threshold) and `bundle.json`
   (head weights SHA-256, head `frozen_sha256`, head config hash, class order, calibrator SHA-256 and
   `calibration_version`, SHA-256 of `baseline_class_map.yaml` and `consistency.json`, and `bundle_sha256` over the
   rest). `protloc.registry.bundle.load_bundle` (standard library only) recomputes every hash and raises
   `BundleError` on any mismatch, so the service can refuse a bundle whose calibrator was fitted on other weights;
   pass the weights file and it hashes that too. The abstention threshold is read from this metadata (CLAUDE.md "No
   hardcoded tau"). The runtime types and `apply` live in `protloc.calibration.calibrator` (layer L4), importable
   without numpy or scikit-learn.
   The bundle JSON files are **committed** (about 11 KB): `.gitignore` already versions `registry/` and excludes only
   `model.pt` and `calibrator.pkl`, and `scripts/calibrate.py --check` in CI needs them. This differs from the CLAUDE.md
   sentence that `registry/` stays gitignored; CLAUDE.md is corrected in the same commit.
9. **Settings are data.** Minimum counts, fallback, Platt numerics, ECE bins and the bundle version are in
   `configs/training/calibration.yaml`, which `calibrate.py` hashes into the manifest. A unit test fails if a
   setting reappears as a literal in the module, or if the config and the root or evaluation configs disagree.
10. **No new dependency.** ECE, Brier, Platt and the eligibility rule are pure Python; isotonic uses
    scikit-learn `IsotonicRegression` from the existing `baselines` extra. No `calibration` extra, no `netcal`.
11. **Freeze, then check.** `scripts/calibrate.py --freeze` writes the bundle and `data/manifests/calibration_metrics.json`
    (knots and thresholds, every input hash, ECE and Brier before and after with intervals, decisions at the new
    thresholds, `frozen_sha256`), and refuses if either exists. `--check` needs no data bytes. Smoke output goes to
    `data/calibration/smoke` and can never be frozen.

## 3. Consequences

- The head's probabilities can be shown as probabilities for labels with `calibrated: true`; labels marked
  `uncalibrated` must be displayed as raw scores (`calibrated: false` in the contract).
- Calibration is monotone, so it cannot change the ranking of scores and cannot raise AUC. It can change the
  decisions only through the new thresholds, and an isotonic plateau coarsens the threshold grid. F1 may go up or
  down; the report states which, measured.
- A calibrated head and its calibrator are one deployable unit. Phase 14 loads the bundle, not loose files.
- The cross-fitted ECE is an out-of-fold estimate with a wide interval, not a guarantee for new data.

## 4. Known limits (stated, not hidden)

- **Tiny positives.** Peroxisome has 6 validation positives, plastid 13, lysosome/vacuole 17, Golgi 18. Intervals on
  2-7 positives (and on most per-label MCC and ECE values) are wide; per-label claims are not supported, only the
  pooled and macro figures are meaningful, and those are wide too.
- **Cross-fitting optimism.** The out-of-fold scores come from four different fold models, while the served head is
  a fifth model refit on all four partitions; the calibrator is fitted on the former and applied to the latter.
  Selection of the head's hyperparameters used the same folds (ADR-0011 section 4).
- **The bootstrap does not refit** the calibrators, so it understates the uncertainty of the calibrators themselves.
- **Borderline labels change method across folds** (a label near the 20-positive line gets isotonic in one fold and
  Platt in another); `methods.consistent` in the manifest says which labels do, and the pooled scores of such a label
  mix two score maps.
- **Inner calibrators are fitted on two partitions** (about 170 rows), so a rare label's inner calibrator is
  forced to the outer method on very few positives.
- **Isotonic plateaus** map many scores to the same value, including exactly 0 and 1; thresholds then sit on a coarse grid.
- **One seed, subset only.** 500 sequences; the full 33,461-sequence cache is not extracted, so nothing here is
  evidence about the full task or about DeepLoc.
- **Platt on 5 positives** is two parameters on five events: allowed by the rule, not trustworthy.

## 5. Alternatives rejected

- **A whole-cluster calibration holdout and a retrain.** Cleaner separation, but removes about a fifth of an
  already tiny training set and costs a rerun of the 29-minute head training for little more validity than
  cross-fitting.
- **Temperature scaling as the only calibrator.** One parameter per label is safe on few positives but cannot fix
  base-rate shifts and drops the plan's isotonic primary.
- **A lower isotonic minimum (10 positives).** Would calibrate Golgi, lysosome/vacuole and plastid on 13-18
  positives, where isotonic memorises.
- **Equal-width ECE bins (15).** Mostly empty for rare labels, noisier than 10 equal-mass bins.
- **Scoring partition 5 once more, after a freeze.** Possible and honest as `touches: 2`, but it adds almost no
  evidence at 99 rows and spends the partition for Phases 11-14.
- **Single nesting of the thresholds** (thresholds for fold *k* from the cross-fitted scores of the others). The
  calibrators that produced those scores saw fold *k*'s labels, so fold *k*'s labels would reach its own thresholds
  second-hand.
- **`netcal`.** A new pinned dependency for an ECE a few lines of Python compute and test by hand.
- **A pickled calibrator.** Not auditable, not diffable, unsafe to load, and the plan wants the calibrator inside a
  checksummed bundle.
