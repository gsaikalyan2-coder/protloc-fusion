# ADR-0014 — GO head: multi-label molecular-function prediction from frozen per-residue ESM-2 states, scored against three floors, shipped in the registry

- **Status:** **Accepted** (2026-10-04)
- **Deciders:** Saikalyan (subset scope, evidence policy and term floor, gates A, B3-B5, E, F, and the pooling and head choice confirmed at the Phase 12 gates)
- **Related:** ADR-0006 (label semantics; GO is open-world), ADR-0010 (evaluation protocol), ADR-0011 (attention head), ADR-0012 (calibration and the registry bundle),
  ADR-0013 (per-task head contract, `TASK_HEAD_KEYS`), CLAUDE.md rules 9 and 10, `docs/phases/12.md`

**Scope: every number here is on the 500-sequence ESM-2 subset ("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). No published GO figure is compared with anything below.**

---

## 1. Context

GO function is the open-world task of ADR-0006: a missing annotation is `unknown`, never `negative`. The question is whether a head trained on frozen per-residue ESM-2 states ranks GO molecular-function
terms better than three floors, under the protocol every earlier phase used.

Facts measured before design (`scripts/go.py --labels`, subset mask only, never the whole-corpus label table):

- 507 subset accessions: 405 `annotated`, 102 `unknown`. 273 are **labelled** (annotated with at least one policy term); 132 are annotated without a policy term; 175 have a motif hit.
  Labelled rows by partition: 75 / 37 / 45 / 52 / 64.
- Ontology: `go-basic.obo`, `data-version: releases/2026-07-26`, SHA-256 `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`, 48,340 `[Term]` stanzas, 10,248 obsolete, CC BY 4.0.
  Propagation over `is_a` and `part_of` stays inside molecular_function (0 cross-aspect edges). The root GO:0003674 is on every row and is not a label.
- Mapping counts, whole corpus (occurrences / distinct ids): primary 1,045,052 / 8,251; replaced_by 278 / 12; obsolete_dropped 90 / 3; alt_id 0; absent 0; other_aspect 0.
  Subset: primary 1,218 / 620; replaced_by 3 / 2; nothing else.
- With evidence policy {experimental, curated} and a floor of 20 positives in partitions 1-4 (after propagation, subset rows), **12 terms are trainable and 705 are under the floor** (largest 19).

| Term | Name | Pos. 1-4 | Pos. 5 | Val. AUC [95% CI] | Val. AUPR head | Val. AUPR prior | Ancestors in vocabulary |
|---|---|---|---|---|---|---|---|
| GO:0003676 | nucleic acid binding | 41 | 10 | 0.730 [0.648, 0.816] | 0.405 | 0.184 | GO:0005488 |
| GO:0003677 | DNA binding | 22 | 4 | 0.753 [0.633, 0.848] | 0.267 | 0.094 | GO:0003676, GO:0005488 |
| GO:0003723 | RNA binding | 22 | 7 | 0.607 [0.507, 0.719] | 0.172 | 0.091 | GO:0003676, GO:0005488 |
| GO:0003824 | catalytic activity | 97 | 26 | 0.899 [0.856, 0.941] | 0.856 | 0.410 | - |
| GO:0005488 | binding | 111 | 38 | 0.614 [0.543, 0.685] | 0.637 | 0.510 | - |
| GO:0005515 | protein binding | 57 | 25 | 0.661 [0.586, 0.732] | 0.388 | 0.247 | GO:0005488 |
| GO:0016740 | transferase activity | 37 | 11 | 0.832 [0.772, 0.899] | 0.588 | 0.156 | GO:0003824 |
| GO:0016787 | hydrolase activity | 37 | 13 | 0.745 [0.667, 0.859] | 0.421 | 0.153 | GO:0003824 |
| GO:0036094 | small molecule binding | 36 | 12 | 0.596 [0.485, 0.694] | 0.232 | 0.153 | GO:0005488 |
| GO:0042802 | identical protein binding | 23 | 11 | 0.708 [0.521, 0.826] | 0.234 | 0.103 | GO:0005488, GO:0005515 |
| GO:0043167 | ion binding | 32 | 12 | 0.648 [0.528, 0.744] | 0.234 | 0.136 | GO:0005488, GO:0036094 |
| GO:0140096 | catalytic activity, acting on a protein | 38 | 9 | 0.806 [0.748, 0.876] | 0.504 | 0.165 | GO:0003824 |

The twelve are generic, high-level terms. The specific terms a researcher cares about are under the floor and are not predicted by this head.

## 2. Decision

1. **Labels.** Evidence policy: a term occurrence is a label only if one of its codes is `experimental` (EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP) or `curated` (TAS, IC). Electronic (IEA), similarity, author (NAS) and ND are not labels.
   Terms are mapped through `alt_id` and `replaced_by`; an obsolete term with no live replacement is dropped and counted. A true-path rule propagates every label to its ancestors over `is_a` and `part_of`.
2. **Open world, closed only at loss and metric time.** A row is *labelled* when it is annotated and has at least one policy term; only then is a vocabulary term absent from its propagated annotation a negative.
   `unknown` rows and annotated rows with no policy term are neither positives nor negatives and never enter a loss or a metric. The rule is written to the manifest (`label_policy.closed_world`).
3. **Vocabulary from subset rows only.** Counts and the term floor come from the feature-table mask (500 sequences), never from the whole-corpus label table (the Phase 11 `--freeze` scope bug). `counts_scope: "subset_mask"` is in the manifest and a test fails if the whole-corpus table is read.
4. **Head.** `head_net.AttentionHead` (attention pooling over per-residue states, 640-d, 1 head, attention hidden 64) with an MLP and one sigmoid per term. Loss: BCE with `pos_weight` = negatives / positives capped at 10 (the weighted focal loss at gamma 0), masked to labelled rows.
   Early stopping watches negative mean BCE on an inner whole-cluster holdout (fraction 0.2, patience 10, at most 80 epochs). All settings are in `configs/training/go.yaml`, hashed into the manifest; a test forbids repeating them in code.
5. **Grid.** learning rate {1e-3, 3e-4} x dropout {0.3, 0.5} x hidden {256, 128} x gamma {0.0} = 8 points. Gamma is an axis with one value (see §4).
6. **Protocol, identical for every arm.** Same rows, leave-one-partition-out folds over partitions 1-4, nested thresholds, cluster bootstrap (1,000 samples, seed 20260909, unit `cluster_id`), same selection rule:
   mean fold precision@3 over rows with a vocabulary positive, then mean fold Fmax, then the earlier grid point. Partition 5 is scored once, after `--freeze`, by `--evaluate --test` (`touches: 1`; a second run and a second freeze are refused).
7. **Scores and metrics.** A term's score is the maximum over itself and its descendants in the vocabulary (true-path consistent). Metrics are computed over *evaluable* rows (at least one vocabulary positive): precision@k and recall@k for k = 1, 3, 5; CAFA-style F
   (precision averaged over rows with at least one term at or above the threshold, recall over all evaluable rows) and coverage. **"Fmax" here is F at a global threshold chosen on the other partitions (nested), not an oracle maximum.**
   Per-term AUC and AUPR only for terms with at least 20 positives and 20 negatives in the rows scored.
8. **Three floors, same protocol** (`ARMS`: model_raw, model_calibrated, frequency_prior, logreg, motif): (a) training-set term frequency; (b) the Phase 8 logistic regression extended to multi-label (C chosen by the shared rule from the Phase 8 grid);
   (c) the PROSITE-style motif hits (`source: "motif"`; a term scores 1 + 0.5 x prior when a motif maps to it or a descendant, 0.5 x prior otherwise).
9. **Calibration.** The Phase 10 machinery per term: isotonic when a term has at least the isotonic minimum of positives and negatives, identity otherwise; the Platt fallback is switched off. One global threshold from nested Fmax, carried by every frozen calibrator.
   The cross-fit is doubly nested as in Phase 10; it lives in `go.py` because `cal.crossfit` tunes a per-label threshold.
10. **Registry.** `registry/go_function/0.12.0/{bundle.json,calibrator.json,terms.json}`. `terms.json` is the committed vocabulary, bound by hash. `TASK_HEAD_KEYS["go_function"]` lists the head contract; the loader refuses a missing key,
    a changed vocabulary file, a changed ontology hash and a tampered file. `source` is `"model"`. The threshold is read from the calibrator metadata at runtime, never a literal.

## 3. Results (measured; detail in `docs/phases/12.md` and `data/manifests/go_metrics.json`)

Chosen: grid point 2 (lr 1e-3, dropout 0.5, hidden 256, gamma 0), 209,549 trainable parameters, refit for 15 epochs on partitions 1-4. Selector 0.5929, mean fold Fmax 0.5964.
The selectors of the top four points are 0.5929 / 0.5923 / 0.5892 / 0.5890: the choice is within noise and is recorded as such.

Validation, pooled over partitions 1-4 (209 labelled rows; 95% cluster-bootstrap CIs):

| Arm | p@1 | p@3 | p@5 | Fmax | coverage |
|---|---|---|---|---|---|
| model_raw | 0.828 [0.775, 0.891] | 0.592 [0.551, 0.645] | 0.463 [0.431, 0.505] | 0.609 [0.576, 0.654] | 0.970 |
| model_calibrated | 0.805 [0.750, 0.873] | 0.582 [0.534, 0.649] | 0.470 [0.436, 0.515] | 0.605 [0.571, 0.648] | 0.994 |
| frequency_prior | 0.657 [0.590, 0.741] | 0.523 [0.487, 0.552] | 0.399 [0.374, 0.423] | 0.531 [0.501, 0.555] | 1.000 |
| logreg | 0.562 [0.495, 0.645] | 0.460 [0.417, 0.506] | 0.411 [0.379, 0.449] | 0.500 [0.468, 0.541] | 0.970 |
| motif | 0.657 [0.587, 0.735] | 0.493 [0.456, 0.525] | 0.396 [0.367, 0.421] | 0.529 [0.499, 0.552] | 1.000 |

The head is above the frequency prior on Fmax with disjoint intervals; on p@3 it is higher with overlapping intervals. It is above logreg and motif on both p@3 and Fmax with disjoint intervals.

Partition 5 (scored once; 64 labelled rows, 52 evaluable): model_raw p@3 0.583 [0.484, 0.667], Fmax 0.622 [0.546, 0.691]; model_calibrated p@3 0.635 [0.540, 0.712], Fmax 0.632 [0.563, 0.695];
frequency_prior 0.571 [0.496, 0.639] / 0.568 [0.510, 0.625]; logreg 0.378 [0.277, 0.473] / 0.479 [0.407, 0.541]; motif 0.519 [0.414, 0.616] / 0.568 [0.496, 0.639].
On test the head and the frequency prior overlap on both metrics.

Calibration (validation, macro over 12 terms): ECE 0.1240 [0.1168, 0.1535] -> 0.0980 [0.0942, 0.1242], delta -0.0260 [-0.0429, -0.0096]; Brier 0.1593 -> 0.1485, delta -0.0108 [-0.0188, -0.0044].
Calibration lowers ECE and Brier; it does not change the ranking (Fmax 0.609 -> 0.605, p@3 0.592 -> 0.582, intervals overlap). All 12 frozen calibrators are isotonic; in individual folds some terms stayed uncalibrated. Frozen global threshold 0.25.

## 4. Known limits (stated, not hidden)

- Only 12 generic terms clear the floor; on the others the head says nothing. Several of the 12 are near the frequency prior in AUC (binding 0.614, small molecule binding 0.596, RNA binding 0.607).
- 273 labelled rows in all (209 in partitions 1-4), 52 evaluable in test: intervals are wide; one seed; one subset.
- Gamma is a one-value axis; the grid is 8 points, not 16.
- Nested thresholds use out-of-fold scores from models that saw the held fold (the same as the ADR-0013 protocol); the bias is not quantified.
- Grid point 7, fold 4 stopped at epoch 1 (inner holdout NLL 0.7013): a degenerate fit, not a result.
- The ontology release is a point-in-time pin; labels and the ontology must stay on the same GOA snapshot and release.
- The full-corpus ESM-2 extraction (about 19 h on CPU) is still outstanding; with it the floor of 20 would admit far more terms.

## 5. Phase 14 item: a model-shaped GO output

The contract's `GoAnnotation` requires `source_accession`, which is true of homology transfer and false of a model prediction. Phase 14 adds a model-shaped GO output (term, score, calibrated, `source: "model"`, no accession)
rather than faking an accession. Phase 13 (homology evidence) keeps `source: "homology"` for transferred terms so the two are never confused on screen.

## 6. Alternatives rejected

- **Per-term label counts from the whole-corpus table**: wrong scope; the term set would not match what the subset can train.
- **Treating absent terms as negatives for every row**: contradicts ADR-0006; an unannotated protein is not a protein without the function.
- **Including IEA/similarity evidence**: the labels would partly be the output of a homology tool, which Phase 13 uses as the evidence layer. Circular.
- **A lower floor (10)**: more terms, but 10 positives split over four folds gives folds with 0-3 positives.
- **Platt fallback for small terms**: identity is more honest than a two-parameter fit on a handful of positives.
- **Oracle Fmax**: tunes the threshold on the rows it scores. Nested thresholds are used.
- **Aspect-wide training (BP, CC)**: out of scope; one aspect, one head.
