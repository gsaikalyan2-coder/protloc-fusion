# ADR-0015 — Homology evidence: an MMseqs2 index over Swiss-Prot, GO transfer from nearest hits, independent of every model score

- **Status:** **Accepted** (2026-10-04)
- **Deciders:** Saikalyan (gates A-F at Phase 13 start; Gate D answered again when `createindex` failed: copy `busybox.exe` as `sh.exe`)
- **Related:** ADR-0003 (evidence layer, local snapshots, no request-path network), ADR-0006 (open-world labels), ADR-0010 (protocol), ADR-0012 (registry bundle), ADR-0014 (GO head), CLAUDE.md rules 9 and 10, `docs/phases/13.md`

**Scope: every query-side number here is on the 500-sequence subset queries ("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). The database is the full Swiss-Prot 2026_03 snapshot (548,702 records). No external figure is compared with anything below.**

---

## 1. Context

Phase 12 left a GO head that ranks 12 generic terms. ADR-0003 asks for a second, independent verdict: the nearest Swiss-Prot homologs of a query (accession, identity, coverage, e-value, database) and the GO terms those homologs carry, from local pinned snapshots, off the request path.
The questions are whether that verdict is worth showing, how it compares with the Phase 12 model under the same protocol, and how to keep it from being flattered by leakage.

Facts measured (`scripts/homology.py`, Windows 10, AMD64, 12 logical CPUs, Python 3.11.9):

- MMseqs2 `18-8cc5c` (commit `8cc5ce367b5638c4306c2d7cfc652dd099a4643f`), MIT, native Windows (Cygwin) build, archive SHA-256 `9752087ca4e25e93b830775832a75044dc94ae348d66d4681f9ff0355eecfcbf`, `mmseqs.exe` SHA-256 `ad319fc9fc67500ca2da9b9247d9611d2113e66cb800c15b603949f5652e2a0b`.
- Index over all 548,702 Swiss-Prot sequences: build 72.0 s (createdb 5.4 s, createindex 66.6 s); index files 2,463,105,431 bytes (2.29 GiB), database 232,414,098 bytes, total 2,695,519,529 bytes; peak working set 2,322,669,568 bytes (2.16 GiB), peak commit 4,327,165,952 bytes (4.03 GiB); 8 threads. `index_sha256` `58d46319c5c0a6514592363ab76a491c06facb49d92d84e0efadcecc21ce5979`.
- Search: 507 subset queries in one batch, 81,173 hit lines, 19.8 s, peak working set 2,127,187,968 bytes. Single-query runs (50 queries spread over the length distribution; one `createdb + search + convertalis` process set each, index loading included): p50 2.31 s, p95 2.80 s, 1.655 CPU-s per query. A resident server would be faster; that was not measured.
- GO annotations of the database sequences are the Phase 12 labels, mapped by the Phase 12 code: 50,821 of 548,702 database sequences carry at least one policy term (experimental + curated, propagated).

## 2. Decision

1. **Engine and index.** An external MMseqs2 binary, never linked; `createdb` over `swissprot.fasta` (accession headers, written from the corpus), `createindex`, `search -s 5.7 -e 0.001 --max-seqs 1000 --alignment-mode 3`, `convertalis` with fixed columns. All settings are in `configs/training/homology.yaml`, hashed into the manifest; a test forbids repeating them as literals.
2. **GO source (Gate B).** The corpus `DR GO` lines, the same records Phase 12 mapped, with the Phase 12 ontology (`releases/2026-07-26`, SHA-256 `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`), evidence policy {experimental, curated} and `is_a`/`part_of` propagation. No GAF was downloaded. The GOA release is recorded as "embedded in Swiss-Prot 2026_03"; the hash that stands for it is the corpus SHA-256 `d493478d38630651490adfe91c7869d1a6ac8ab759effdc3847ce84e295d41a8`. The code is `protloc.training.go` (`scan_corpus`, `map_record`, `load_ontology`), reused and not copied.
3. **Transfer.** A hit transfers terms only if its own annotation is non-empty. A vote is the fraction of the top-k annotated hits that carry the term; the training frequency prior is added with weight 0.01 and only breaks ties (a test checks the weight is below 1 / max top_k). A transferred term is `source: "homology"` with `source_accession`; a model term is `source: "model"`; the two are never in one field.
4. **Leakage masks (exactly four, in this order, refused if one is dropped).** Every database sequence is searched; hits are then removed for (1) the query's own accession, (2) any database sequence with the query's `sequence_sha256`, (3) any sequence in the query's `cluster_id`, (4) any sequence in the query's partition. Each removed hit is attributed to the first mask that matched, so the counts add up; the hits each mask alone would remove are also recorded. Tests plant a self-hit (by accession and by hash), a same-cluster hit and a same-partition hit.
   **Known limit:** only 33,879 of the 548,702 database sequences are in the split registry. The others have no cluster or partition, so masks 3 and 4 cannot remove them. A registry-only diagnostic arm (hits restricted to registered sequences of other partitions) is reported beside the primary arm and is not tuned.
5. **Independence.** Evidence never feeds a model score, the calibrator or a threshold. `homology.py` is stdlib only and imports nothing from `training`, `calibration` or `models`; no model, calibrator or head module imports the evidence layer; the homology arms are built by functions that never see a model array; the manifest records `feeds_model_score`, `feeds_calibrator`, `feeds_threshold` and `homology_arms_read_model_scores` as `false` and `--check` refuses anything else. Tests fail if any of that changes.
6. **Protocol, identical to Phase 12.** The Phase 12 vocabulary (12 terms), labelled rows, leave-one-partition-out folds over partitions 1-4, nested global thresholds, cluster bootstrap (1,000 samples, seed 20260909, unit `cluster_id`). The hit filter is tuned on partitions 1-4 only over a 108-point grid (e-value {1e-20, 1e-10, 1e-5, 1e-3} x min identity {0, 0.3, 0.5} x min query coverage {0, 0.5, 0.8} x top-k {1, 3, 5}), selected by mean fold precision@3, then mean fold Fmax, then the earlier point. The best-hit arm is tuned over the top-k = 1 points.
   Arms: `homology_vote`, `homology_best`, `homology_registry_only`, `frequency_prior`, and the Phase 12 `model_raw` and `model_calibrated` rebuilt from their saved out-of-fold scores (they reproduce the frozen Phase 12 numbers to 0.0). Metrics: precision@k, recall@k, Fmax (F at a nested global threshold, not an oracle), coverage, per-identity bins, and hit coverage (fraction of evaluable rows with any passing hit).
7. **Registry.** `registry/evidence_homology/0.13.0/bundle.json` binds the database, engine, index file list and hash, GO source, ontology, vocabulary, hit filter, masks, transfer rule and the independence flags. `load_evidence_bundle` refuses a changed index byte, a different GO source hash, a different ontology hash, a tampered file, a changed mask list or an independence flag that is not `false`.
8. **Partition 5.** Scored once, after the freeze, by `--evaluate --test` (`touches: 1`; a second run and a second freeze are refused). The GO, CRF and localization counts still read 1 each; homology has its own count. The model arms on test are copied from `go_metrics.json`; the GO head was not re-run.

## 3. Results

Selected hit filter: e-value 1e-3, min identity 0, min query coverage 0, top-k 5 (grid point 83; selector 0.7034, mean fold Fmax 0.7328). Best-hit arm: top-k 1 (grid point 81; 0.6602 / 0.6781). Frozen thresholds: vote 0.4012, best 0.00506, registry-only 0.00425, prior 0.25.

Masking (hits removed per mask, exclusive / independent):

| Set | Raw | Kept | self accession | self sha256 | same cluster | same partition |
|---|---|---|---|---|---|---|
| Validation (209 rows) | 30,846 | 28,498 | 209 / 209 | 20 / 229 | 1,631 / 1,848 | 488 / 2,336 |
| Test (64 rows) | 19,663 | 17,167 | 64 / 64 | 8 / 72 | 1,939 / 2,005 | 485 / 2,490 |

Validation, pooled over partitions 1-4 (209 labelled rows, 169 evaluable; 95% cluster-bootstrap CIs):

| Arm | p@1 | p@3 | p@5 | r@3 | Fmax | coverage |
|---|---|---|---|---|---|---|
| homology_vote | 0.923 [0.886, 0.958] | 0.710 [0.669, 0.755] | 0.524 [0.492, 0.559] | 0.739 [0.694, 0.776] | 0.736 [0.695, 0.772] | 0.864 |
| homology_best | 0.864 [0.821, 0.914] | 0.667 [0.631, 0.710] | 0.501 [0.471, 0.538] | 0.701 [0.655, 0.739] | 0.692 [0.657, 0.729] | 0.959 |
| homology_registry_only | 0.805 [0.750, 0.875] | 0.635 [0.598, 0.679] | 0.469 [0.438, 0.505] | 0.665 [0.625, 0.708] | 0.631 [0.604, 0.668] | 1.000 |
| model_raw | 0.828 [0.775, 0.891] | 0.592 [0.551, 0.645] | 0.463 [0.431, 0.505] | 0.614 [0.572, 0.658] | 0.609 [0.576, 0.654] | 0.970 |
| model_calibrated | 0.805 [0.750, 0.873] | 0.582 [0.534, 0.649] | 0.470 [0.436, 0.515] | 0.594 [0.549, 0.649] | 0.605 [0.571, 0.648] | 0.994 |
| frequency_prior | 0.657 [0.590, 0.741] | 0.523 [0.487, 0.552] | 0.399 [0.374, 0.423] | 0.557 [0.514, 0.595] | 0.531 [0.501, 0.555] | 1.000 |

Partition 5 (scored once; 64 labelled rows, 52 evaluable):

| Arm | p@1 | p@3 | p@5 | r@3 | Fmax | coverage |
|---|---|---|---|---|---|---|
| homology_vote | 0.885 [0.783, 0.964] | 0.763 [0.674, 0.836] | 0.569 [0.478, 0.656] | 0.758 [0.688, 0.835] | 0.726 [0.614, 0.806] | 0.846 |
| homology_best | 0.865 [0.755, 0.955] | 0.750 [0.651, 0.822] | 0.554 [0.478, 0.618] | 0.736 [0.663, 0.811] | 0.756 [0.680, 0.809] | 1.000 |
| homology_registry_only | 0.788 [0.667, 0.882] | 0.712 [0.619, 0.785] | 0.512 [0.447, 0.565] | 0.694 [0.613, 0.786] | 0.655 [0.604, 0.695] | 1.000 |
| frequency_prior | 0.731 [0.588, 0.860] | 0.571 [0.496, 0.639] | 0.415 [0.374, 0.449] | 0.566 [0.497, 0.657] | 0.568 [0.510, 0.625] | 1.000 |
| model_raw (copied) | | 0.583 [0.484, 0.667] | | | 0.622 [0.546, 0.691] | |
| model_calibrated (copied) | | 0.635 [0.540, 0.712] | | | 0.632 [0.563, 0.695] | |

Hit coverage (fraction of evaluable rows with any passing hit): validation 0.982 (204 of 209 rows have a hit; 0.876 have a hit that can vote), registry-only 0.633; test 0.962, registry-only 0.654.

What the intervals say. On validation the vote arm is above `model_raw`, `model_calibrated` and the prior on p@3 and Fmax with disjoint intervals. The best-hit arm is above the model on Fmax with disjoint intervals and overlaps it on p@3. The registry-only arm overlaps the model on p@3 and Fmax. On test the vote arm's p@3 is above `model_raw` and the prior with disjoint intervals and overlaps `model_calibrated`; Fmax overlaps both models. With 52 evaluable test rows, nothing on test is shown to differ except those two p@3 comparisons.

Per-identity bins (validation, best passing hit of the selected filter; p@3 / Fmax point values, CIs in the manifest; bins under 10 evaluable rows are listed and not scored):

| Bin | Rows (evaluable) | vote | best | model_raw | model_calibrated | prior |
|---|---|---|---|---|---|---|
| [0, 0.3) | 11 (6) | not scored | | | | |
| [0.3, 0.5) | 40 (26) | 0.744 / 0.800 | 0.679 / 0.673 | 0.615 / 0.633 | 0.577 / 0.618 | 0.449 / 0.469 |
| [0.5, 0.7) | 35 (33) | 0.707 / 0.784 | 0.687 / 0.710 | 0.556 / 0.651 | 0.576 / 0.608 | 0.475 / 0.501 |
| [0.7, 0.9) | 53 (44) | 0.644 / 0.685 | 0.606 / 0.628 | 0.545 / 0.563 | 0.545 / 0.587 | 0.538 / 0.542 |
| [0.9, 1] | 65 (57) | 0.760 / 0.742 | 0.713 / 0.735 | 0.655 / 0.616 | 0.626 / 0.606 | 0.585 / 0.572 |
| no hit | 5 (3) | not scored | | | | |

Test bins: only [0.7, 0.9) (19 rows, 18 evaluable) and [0.9, 1] (21 rows, 16 evaluable) reach 10 evaluable rows; they are in the manifest.

## 4. Known limits (stated, not hidden)

- **Unmasked unregistered sequences.** 94% of the database is outside the split registry and is never masked by cluster or partition. The primary arms can therefore find close relatives the model's split would have excluded. The registry-only arm is the diagnostic: it is lower on every metric (validation Fmax 0.631 vs 0.736) and overlaps the model. The size of the true leakage is not measured.
- Few rows: 209 labelled rows in validation, 52 evaluable in test; 12 generic terms; one subset; one run. Intervals are wide.
- The best-hit and registry-only arms' frozen thresholds (0.0051, 0.0043) sit at the tie-break prior's scale, so those arms also call terms through the prior; their Fmax is not a pure homology figure.
- Latency is per-process on this machine and includes index loading; no resident-server figure and no number from another machine.
- The GO annotation of a hit is the corpus `DR GO` annotation, not a fresh GOA release; it can lag UniProt-GOA.
- Hit annotations use experimental and curated evidence only, so a hit with only electronic annotations has nothing to transfer.
- The `sh.exe` shell shim is a copy of the shipped `busybox.exe` (see §5); MMseqs2 workflows on Windows depend on it.

## 5. Gate D: the index failed once, and the fix

`createindex` exited 1: the Windows build runs its workflow scripts with `#!/bin/sh`, and the release ships `busybox.exe` but no `sh.exe` (`Failed to execute .../createindex.sh with error 2`). Nothing was deleted. You chose: copy `busybox.exe` as `sh.exe` beside `mmseqs.exe`. `scripts/homology.py` does this once, refuses an `sh.exe` that is not a byte copy of `busybox.exe`, and records `busybox_sha256` with the index. `mmseqs.exe` and its pin are untouched.

## 6. Alternatives rejected

- **A GAF download for the hit terms**: another snapshot to pin and a risk that hit terms and Phase 12 labels differ; the corpus lines guarantee the same snapshot.
- **A Linux MMseqs2 under WSL (Gate A)**: more moving parts; the native binary worked once the shell shim was in place.
- **A k-mer search over a reference slice**: the spec's fallback, not needed.
- **Letting evidence adjust a model score or threshold**: breaks the "second, independent verdict".
- **Scoring only registered hits as the primary arm**: removes most hits (hit coverage 0.63) and hides the question; reported as a diagnostic instead.
- **Tuning the filter on all five partitions**: partition 5 was not read before `--freeze`.

## 7. Phase 14 item

API wiring and the refuse-to-start check for the evidence bundle are deferred to Phase 14 (Gate F). Serving must read the frozen bundle, refuse a changed index or GO hash, and keep `source: "homology"` and `source_accession` on every transferred term.
