# ADR-0016 — The serving engine: numpy forward passes over cached ESM-2 states, served groups, one engine config, refuse to start

- **Status:** **Accepted** (2026-10-05)
- **Deciders:** Saikalyan (Phase 14 gates A-E: A, the label on the metric floor is the Phase 8 baseline; B, cache-only encoder; C, local-only homology with a degraded state; D, local bundle versions for `MODEL_VERSION` and `ARTIFACT_REVISION` left for Phase 19; E, no deletions, and the motif baseline frozen as a test-only fixture)
- **Related:** ADR-0001 (queue everything), ADR-0003 (local evidence), ADR-0006 (label states), ADR-0008 (batch scope), ADR-0009 (frozen encoder), ADR-0011 to ADR-0015 (the heads, calibration, evidence), CLAUDE.md rules 9 and 10, `docs/phases/14.md`

**Scope: every number here is on the 500-sequence subset ("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"). Timings are from one Windows 10 machine with 12 logical CPUs; the real host (Phase 19) has not been measured.**

---

## 1. Context

Phases 9-13 produced three trained heads (localization, signal peptide with cleavage site, GO molecular function), per-label calibrators and thresholds, and a homology evidence index, each frozen in `registry/<task>/<version>/` with a hash. The served app was still the Phase 0 motif engine over a synthetic reference set. Phase 14 puts the trained models behind the same asynchronous API and the same five screens.

Four things were not decided by earlier ADRs: how the heads run in the serving process (the training code needs PyTorch), what to do with a sequence the encoder has not seen, which organism groups the heads may answer for, and where the engine's own constants live.

## 2. Decision

1. **One pipeline, one owner.** `protloc.inference.engine` owns the order **features -> model -> calibration -> consistency -> evidence**, and nothing else knows it. `predict_many` is the single place the order is written; a test fails if a scoring stage mentions evidence or if the order changes. Evidence never feeds a score, a calibrator or a threshold (ADR-0015 stays in force; its independence tests still pass).
2. **numpy serving, torch training.** The served heads are `protloc.models.attention_head.AttentionHeadNet` and `protloc.models.crf_tagger.CrfTaggerNet`: plain numpy forward passes over the frozen weights (`data/{head,go,crf}/final_weights.npz`, verified by hash against the registry bundles). The CRF runs in float64. Serving never imports `training/` (`test_layering` stays green). `tests/unit/test_models_parity.py` is the one place both sides meet: numpy against the torch training forward on 40 cached sequences spread over the length distribution. Stated tolerances: localization and GO sigmoids 1e-5 absolute; CRF detection 5e-4 (the torch side computes `1 - P(all outside)` in float32, which is off by up to 1e-4 near 0; the test records 1.2e-4 measured over all 500 subset sequences); cleavage site exact; signal posterior 1e-4. `numpy==2.4.6` is the only new serving dependency (`serve` extra).
3. **Cache-only encoder (Gate B).** The engine does not run ESM-2. It reads per-residue states from the verified residue cache (`data/embeddings/0e528d4a46831b05_subset500_residues`, 500 sequences) keyed by the corpus SHA-256 of the sequence (`normalize_sequence`: uppercase, whitespace removed, no substitution). A sequence whose states are not cached is refused with the code `not_in_embedding_cache` and a message; it is never scored from something else. Reasons: the full-corpus extraction is outstanding (about 19 h on CPU, Phase 7), a live 150M encoder costs about 2 s per sequence on the measured CPU and 600 MB of weights in the image, and neither fits Phase 14's six hours. The cost is that the live app answers only for the 500 cached sequences (the gallery lists 11 of them). This is the largest limit of the release and the banner says it.
4. **Served groups.** The corpus is eukaryotic (`eukarya`; ADR-0006, Phase 6). The UI groups `eukaryote` and `plant` are served; `gram_neg` and `gram_pos` are refused with HTTP 422 and a message, because no head was trained on bacteria and a score would be an extrapolation presented as a prediction. `eukaryote` scores 9 classes (no plastid), `plant` scores 10. The set is data (`organism_groups_with_trained_heads`), and a test checks it is only eukaryotic.
5. **One engine config.** `configs/serving/engine.json` holds what is not an environment choice: bundle versions, weight and manifest paths, the encoder pin, the class codes in head order, the CRF tag set and forbidden transition, the served groups, the ontology pin, the GO annotation source hash and the evidence executable and settings. `engine.py` reads it and repeats none of it; `test_engine_config.py` fails if it disagrees with `configs/training/*.yaml` or `backend/rules/baseline_class_map.yaml`. `ServingSettings` (`MODEL_REGISTRY`, `MODEL_VERSION`, `MODEL_SERVING_MODE`, `MAX_BATCH_SEQUENCES`, ...) has no defaults: a service that guessed where its model lives could serve a different model than it reports.
6. **`MODEL_VERSION` is composite and verified.** `localization-0.10.0+signal_peptide-0.11.0+go_function-0.12.0+evidence_homology-0.13.0`, derived from the bundles actually loaded. The service refuses to start if the configured string differs (Gate D). `ARTIFACT_REVISION` is the Hugging Face commit SHA and is filled in Phase 19.
7. **Refuse to start.** `load_engine` collects every problem and raises one `EngineStartError` naming all of them: a missing bundle; any bundle, weight, calibrator, manifest or rule-file hash that differs from the recorded one; an ontology file that is not the pinned release; a hit-term table, search executable or index byte that differs from the evidence bundle; a GO annotation source hash that is not the pinned one; a residue cache that fails verification or has another dimension. Tests exercise each of these except the executable hash (§5).
8. **Sources stay honest.** Every task carries `source`, `calibrated`, `bundle_version`, `bundle_sha256`. Localization, signal peptide and GO function are `source: "model"` (GO rows have no accession: `ModelGoAnnotation`). Homology terms are `source: "homology"` with `source_accession` in their own field (`GoAnnotation`). Membrane type, the scanner, the residue attribution and the hydropathy track are the motif engine's and say `source: "motif"` (§4). The wire contract is `contracts.py` `API_VERSION` 1.2.0; `docs/api/openapi.yaml` and `api.generated.ts` are regenerated from it.
9. **Evidence off the request path's critical failure modes.** `protloc.evidence.search` runs the MMseqs2 binary as a subprocess under a deadline (base 5 s plus 0.25 s per extra query), kills the whole process tree on timeout and returns a reason; the engine then serves the prediction with `evidence_status.available: false` and the reason (the degraded state, Gate C). No network is used.
10. **Limits come from measurement.** `MAX_BATCH_SEQUENCES=20`: half the largest batch measured (40), because the measurement reuses 10 distinct eukaryote sequences cyclically and the real host has fewer cores. The queue estimates `estimated_wait_seconds` from the compute time it has measured per sequence and reports `null` before one job has run.
11. **The motif baseline is frozen, not served for trained fields.** `backend/tests/golden/golden_set.json` is byte-identical to the new `golden_set_motif.json` (Phase 4 record, SHA-256 prefix `424e61c2e450`); `pytest tests/golden` guards it through `protloc.inference.motif`, a verbatim copy of the old service. Nothing was deleted (Gate E): `data/reference_demo.py` stays, is no longer reachable from the served routes except through the motif module the scanner uses, and is still imported by that module at import time.

## 3. Consequences

- The app answers real trained-model output for 500 sequences in two organism groups, and refuses everything else with a reason. That is narrower than the plan wanted and more honest than a motif fallback.
- Serving needs numpy and no PyTorch; the training code can change without touching the served path, and the parity test is what tells us when the two have drifted.
- The golden set cannot test the served engine (67 inputs: 48 not cached, 17 unsupported group, 1 invalid sequence, 1 unknown group). Served behaviour is tested against real cached sequences in `test_engine.py` and `test_live_app.py`, which skip where the data is absent (CI).
- Membrane type and the in-silico scanner are still rule-based. Phase 15 replaces the attribution.

## 4. Known limits (stated, not hidden)

1. **500 cached sequences only**, all of which were in the training or validation folds of the heads: the gallery examples are in-sample. A gallery result is a demonstration of the pipeline, not a measure of accuracy.
2. **Scanner, membrane type, attribution and hydropathy are motif outputs.** The scanner still uses the legacy synthetic-fit temperature and tau of the motif engine; they are never shown beside a model number.
3. **Consistency rule R06 is inactive:** Phase 13 hits carry no location, so the rule that compares a hit's location with the call has nothing to compare. The rule file (`rules_version` "demo-0.1.0") is hash-bound but not yet renamed.
4. **The formal contract app `api/app.py` still reports Unavailable;** the live app is `api/app_v2.py` (`/demo/*`).
5. **Batch progress is reported once** (completed count jumps from 0 to the total): one `predict_many` call scores the batch.
6. **Latency is dominated by the search**, which loads the index on every call (about 2 s). A resident search server would remove most of it and was not built.
7. **No tamper test for the search executable hash** (the bundle check exists in code); the other refusal paths each have a test.
8. **Window of validity:** the timings are one machine, one run; Phase 19 re-measures on the host.

## 5. Alternatives rejected

- **Serve with PyTorch.** Adds a 200+ MB dependency and the training package to the image for a forward pass that is a few matrix products. Rejected; the parity test removes the fidelity worry.
- **Run ESM-2 live on CPU.** About 2 s per sequence and the weights in the image, for a deployment target of 2 vCPU. Out of the week's budget; revisit after the full-corpus extraction and the host decision.
- **Fall back to the motif engine for an uncached sequence.** It would put a number from a different method under the same banner. Rejected: a refusal with a reason is the honest answer.
- **Serve bacterial groups from the eukaryotic head.** An extrapolation with no evidence; refused.
- **Delete `reference_demo.py` now.** Gate E answered "no deletions"; deferred, not forgotten (Phase 16 or 20).

## 6. Phase 15 item

Replace the motif residue attribution with the attention weights the numpy head already returns (`AttentionHeadNet.forward(...).attention`), define what the weights are claimed to show, test them against a perturbation baseline on the subset, and render them with a `model` source badge.
