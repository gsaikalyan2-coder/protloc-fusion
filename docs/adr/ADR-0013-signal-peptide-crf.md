# ADR-0013 — Signal-peptide head: a linear-chain CRF over frozen per-residue ESM-2 states, scored against the motif floor, shipped in the registry

- **Status:** **Accepted** (2026-10-04)
- **Deciders:** Saikalyan (dependencies, tag scope, grid, protocol, registry and serving scope and base branch confirmed at the Phase 11 gates)
- **Related:** ADR-0006 (label semantics, `cleavage_site` is the 1-based last residue of the peptide), ADR-0010 (evaluation protocol),
  ADR-0011 (localization head), ADR-0012 (calibration and the registry bundle), CLAUDE.md rules 9 and 10 and "No hardcoded tau",
  `PROJECT_PLAN.md` section 4 row 11, `docs/phases/11.md`

---

## 1. Context

The motif h-region detector is a floor: an untrained rule set that scores a signal peptide and proposes a cleavage site. Phase 11 asks
whether a head trained on frozen per-residue ESM-2 states can beat it, and gives the answer in the same protocol as every earlier phase.

Facts that forced decisions (measured on the 500-sequence subset, 507 corpus rows, before any design; label states from `corpus.jsonl`):

| Task | Rows | Positives with a cleavage site, partitions 1 / 2 / 3 / 4 / 5 | Validation (1-4) | Annotated without a site | Not applicable |
|---|---|---|---|---|---|
| signal_peptide | 507 (400 negative, 12 unknown) | 45 / 7 / 16 / 15 / 12 = 95 | **83** | 0 | 0 |
| mitochondrial_transit_peptide | 507 (458 negative, 30 unknown) | 4 / 4 / 6 / 4 / 1 = 19 | **18** | 0 | 0 |
| chloroplast_transit_peptide | 507 (458 negative, 40 unknown) | 1 / 5 / 1 / 2 / 0 = 9 | **9** | 0 | 0 |

Every subset row is in organism group `eukarya`. Partition 2 holds 7 signal-peptide positives, which makes fold 2 noisy.

- **Per-residue states exist** for the subset only (`data/embeddings/0e528d4a46831b05_subset500_residues`, 640-d float16). The full corpus is not extracted.
- **No GPU, no new dependency** is the working constraint (gate A): the CRF is written in torch.
- **Partition 5 was scored once** by the Phase 9 head. A CRF test is a second, separate head scored once; the localization head stays at `touches: 1`.

## 2. Decision

1. **Labels to tags (gate B).** Residues 1..`cleavage_site` carry the signal tag S, the rest O; `cleavage_site` is the 1-based last peptide residue (ADR-0006).
   `unknown` and `not_applicable` rows are dropped, never negatives. A positive without a usable cleavage site (missing, or beyond the length cap) is
   **masked in the tag loss** and pays only the detection loss `-log(1 - P(every residue outside))`; the subset has none missing, and the run prints how many positives exceed the cap.
2. **Scope: signal peptide only, 2 tags (O, S).** A transit-peptide tag is trained only with at least 20 positives that carry a cleavage site
   (`tags.min_cleavage_positives`). Mitochondrial transit has 18 and chloroplast transit 9 in the validation partitions, so both are **undefined**: not
   trained, listed in the manifest under `tag_scope.undefined` with their counts and `trained: false`, and `check_crf_manifest` fails a manifest that claims
   otherwise. A TargetP-style four-tag model on 18 and 9 examples would be noise presented as a result.
3. **Network.** Frozen per-residue states (640-d) -> LayerNorm -> dropout -> {1-D CNN: 2 convolutions, kernel 5, 64 channels, GELU, padding masked to zero before
   each layer | BiLSTM: 1 layer, 64 per direction, packed sequences} -> linear emissions (2 tags) -> linear-chain CRF. Padding never reaches a loss, a score or a decode
   (tested: emissions and every CRF quantity are invariant to padding). Length cap **100**: only the first 100 residues are read; residues beyond the cap are not in
   the path and are reported as outside. A labelled site beyond the cap makes that positive detection-only; the run reports how many there are.
4. **The CRF is plain torch, not an `nn.Module` subclass** (the strict-mypy pattern of `head_net.py`): log-partition by the forward algorithm in log space, Viterbi with
   identity back-pointers on padding, forward-backward marginals, path score. Transition constraints are a boolean `allowed` matrix from `configs/training/crf.yaml`;
   a forbidden transition (O to S) is replaced by the forbidden score (-10000.0), which underflows to exactly 0 under `exp`. Valid paths are `S^k O^(L-k)`: the signal
   is at the N-terminus, contiguous, and never resumes after O. Tests compare log Z, the arg-max path and the marginals with brute-force enumeration (length <= 6,
   several tag counts and constraint sets) before any training.
5. **Outputs per sequence:** the tag path, the per-residue signal marginal (Phase 15 attribution), `cleavage_site` (1-based, the `TaskPrediction.cleavage_site`
   convention), the detection score and `source: "model"`.
   - **Detection score = `P(any residue is S)` = `1 - P(every residue is O)`.** Under the contiguity constraint this equals the signal marginal at residue 1, which
     is also the maximum over positions of the signal marginal; the sum of marginals is rejected because it counts residues (a long peptide would outscore a short one).
     A test asserts the equalities and `max <= sum`. `log(1 - exp(x))` is computed stably.
   - **Cleavage site = the last S of the best path given that residue 1 is S** (Viterbi with O forbidden at the start). Reported only when detection fires. The
     unconstrained best path (`path_cleavage`) is also kept; it is 0 whenever the best single path is all outside, which is common for a weak positive, and would
     make "cleavage accuracy among positives" an artefact of the detection threshold.
6. **Fallback arm = the floor, scored in the same protocol.** The untrained motif detector (`source: "motif"`): score `R_signal_peptide[:, 0]`, cleavage
   `cleavage_position` (0 means none found; same 1-based convention). It is never trained, but its detection threshold is tuned by the same nested MCC rule, so
   the two arms differ only in the model. If the CRF does not beat the floor the report says so, headline by headline, with the intervals.
7. **Protocol (ADR-0010 and ADR-0011, gate C).** Leave-one-partition-out over registry partitions 1-4; partition 5 is not read until one scoring after the freeze.
   In each fold the training partitions are split into fit and early-stopping holdout by whole `cluster_id` clusters (fraction 0.2); training stops after 10 epochs
   without gain in the holdout negative log-likelihood (at most 80); the weights of the best epoch score the validation partition; the final head is refit on
   partitions 1-4 for the median best epoch. The cluster sets of fit, holdout, validation and test are asserted pairwise disjoint before every fit. Shuffling is the
   deterministic SHA-256 shuffler (no `random`); seed 20260909.
8. **Grid and selection (gates B and C).** 8 points: encoder {CNN, BiLSTM} x learning rate {1e-3, 3e-4} x dropout {0.3, 0.5}. Selected by the mean over folds of
   **cleavage accuracy within +-1 residue** among true positives with a usable site; tie-break the mean fold detection MCC under nested thresholds; then the earlier
   grid point. Cleavage is the property the head exists for; detection alone is something the floor already does.
9. **Thresholds are nested** as in ADR-0010: fold k is scored with the MCC-optimal threshold tuned on the other partitions' out-of-fold scores; the frozen threshold
   is tuned on all validation rows. No literal 0.5 anywhere.
10. **Calibration reuses Phase 10 (ADR-0012).** The detection score is calibrated with `crossfit`, `fit_label`, `calibration_report` and `CalibratorSet` (one label,
    `signal_peptide`), cross-fitted on the out-of-fold scores; eligibility 20/20 isotonic, 5/5 Platt, else identity (`configs/training/calibration.yaml`, not
    duplicated). The cleavage position is **not** calibrated; only accuracy is reported.
11. **Metrics, pooled over folds, with cluster-bootstrap CIs (1,000 samples, resampling `cluster_id`, seed 20260909):** detection F1, MCC and rank-based AUC (tied
    scores share their mean rank); cleavage exact-match and within +-1, +-2, +-3 among true positives with a usable site; mean absolute cleavage error; the same cleavage
    metrics restricted to positives the decision also catches (`cleavage_detected_*`); per-residue F1 of the signal tag; per organism group only where positives >= 20.
    A slice with fewer than 2 positives has MCC, AUC and cleavage metrics 0.0, is flagged `undefined` and is kept. TargetP's published 83% (signal peptide) and 46-49%
    (transit peptides) are context only, recorded in the manifest as `comparable: false`: other data, other protocol.
12. **Freeze (`data/manifests/crf_metrics.json`).** `frozen_sha256` is over the frozen block (inputs, protocol, selection, thresholds, tag scope, validation
    including the floor, bundle). It is written before the test run; the test block repeats the hash and records `touches: 1`. `crf.py --check` needs no data bytes
    and rejects a test block that lacks the hash, a second touch, an undefined tag that has enough positives, a missing metric, or a reproduction failure.
13. **Reproduce.** `--evaluate` recomputes every fold's detection scores and cleavage sites from the saved fold weights; the maximum absolute difference must be at
    most 1e-9 and the nested decisions identical, or the stage fails (it is not recorded as a pass).
14. **Registry bundle (gate F1).** `registry/signal_peptide/<version>/{bundle.json,calibrator.json}` through the Phase 10 loader, extended rather than forked:
    `protloc.registry.bundle.TASK_HEAD_KEYS` lists the head keys a task must record (none for localization, which is unchanged; for signal_peptide the tag names,
    length cap, encoder, detection-score definition, cleavage rule and `source`), and `load_bundle` refuses a bundle without them.
    **Hash order.** The manifest's frozen block contains the bundle hash, so the bundle cannot contain the manifest's hash. `head.frozen_sha256` in this bundle is
    therefore the SHA-256 of the training record (`data/crf/train/final.json`: selection, grid, epochs, fingerprint), also pinned in the manifest as
    `inputs.training_record_sha256`; the manifest and the bundle are cross-checked by `--check`.
15. **Serving scope (gate F2).** The service's refuse-to-start check on a bad bundle and the `MAX_BATCH_SEQUENCES` measurement are **deferred to Phase 14**, where the
    service is built. `PROJECT_PLAN.md` lines 239-246 and 284 ("designed out in Phase 11") are met as far as design goes: the bundle, its loader that raises on any
    mismatch, and the task contract exist; no service code and no contract change in this phase (`docs/api/openapi.yaml`, `frontend/` and `tokens.css` are unchanged).
16. **Labels are read from the corpus, once.** `data/crf/labels.tsv` is built by streaming `corpus.jsonl` for the subset's accessions (the pinned corpus hash is
    verified first) and must agree with the feature table's `Y_signal_peptide` and `M_signal_peptide` on every row, or the run fails. The feature table has no
    cleavage sites; the corpus does.
17. **Settings are data.** Grid, caps, tag set, forbidden transitions, tolerances, schedule and bundle version are in `configs/training/crf.yaml` (hashed into the
    manifest). A unit test fails if a setting reappears as a literal in either module or the config disagrees with the root, model or evaluation configs.
18. **No new dependency** (gate A). `--smoke` (every 4th row, 2 epochs, 2 grid points, `data/crf/smoke`) is a code check and can never be frozen.

## 3. Consequences

- The head is trained on 83 positives with a cleavage site (95 including partition 5). Fold 2 has 7. Every interval is wide; only pooled figures are meaningful.
- The floor is evaluated beside the head on the same rows, thresholds and bootstrap, so a gap is attributable to the model.
- A signal-peptide head and its calibrator are one deployable unit, loaded from the bundle in Phase 14.
- Transit-peptide prediction is **not** delivered in Phase 11; it needs the full-corpus residue extraction first.
- Phase 15 can read the per-residue marginals without a change to this head.

## 4. Known limits (stated, not hidden)

- **83 validation positives, all eukarya.** No per-group claim is possible; there is one group.
- **Cleavage is scored among positives with a usable site only**, so it says nothing about the negatives; the detection metrics cover them.
- **The length cap (100) truncates** a sequence's read-out; a labelled site beyond it is detection-only.
- **Out-of-fold scores come from four fold models**; the served head is a fifth refit on all four partitions. Hyperparameters were selected on the same folds (ADR-0011).
- **One seed, subset only.** No variance over seeds; nothing here is evidence about the full corpus or about TargetP.
- **The detection score of a CRF is a path probability**, not a calibrated probability, until the Phase 10 calibrator is applied; even then it is calibrated on 408 rows.
- **The floor is a hand-written rule set**, not a trained baseline; beating it says the head adds something to the rules, not that it is the best possible model.

## 5. Alternatives rejected

- **Four tags (O, S, M, C), TargetP-style.** Mitochondrial (18) and chloroplast (9) positives are under the 20-example floor; the ADR-0010 rule is to mark a tag undefined, not to train it.
- **Detection score as the maximum Viterbi score, or the sum of marginals.** The first is unbounded and not a probability; the second counts residues. The all-outside probability is exact and bounded.
- **Cleavage from the unconstrained best path.** It is 0 whenever the best single path is all outside, so cleavage accuracy would partly measure detection.
- **`pytorch-crf` or another CRF package.** A new pinned dependency for forward, Viterbi and forward-backward routines short enough to test against brute force.
- **A CRF without constraints** (3-tag BIO or free O/S). It can decode `S O S`; the constraint makes the cleavage site unique and the marginal at residue 1 the detection probability.
- **Selecting by detection MCC or loss.** Detection is solved well by the floor; the head is justified only by cleavage.
- **Spending partition 5 on a second look at the localization head.** Not needed; it stays at `touches: 1`.
- **Starting the serving checks now.** Phase 14 builds the service; checks written before it would be tested against nothing.
