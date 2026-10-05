# ADR-0017 — Residue attribution from the trained heads: pooling attention, occlusion and the CRF posterior, each served only after a frozen deletion test

- **Status:** **Accepted** (2026-10-05)
- **Deciders:** Saikalyan (Phase 15 gates: A, occlusion is the default method; B, window 5, stride 1, at most 64 forward passes; C, extend the existing wire; D, a method that fails its test is dropped)
- **Related:** ADR-0009 (frozen encoder), ADR-0011 (localization head), ADR-0013 (CRF), ADR-0014 (GO head), ADR-0016 (serving engine), `docs/phases/15.md`

**Scope: every number here is on the 500-sequence subset ("subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"): 402 localization/GO sequences in 313 clusters (partitions 1-4) and 83 signal-peptide rows in 49 clusters. Partition 5 is untouched.**

---

## 1. Context

Phase 14 left the per-residue track as the Phase 0 motif output. A track drawn over the sequence looks like an explanation of the model's answer; a motif track was not. The head's own attention is class-agnostic: it is computed before the final layer, so one set of weights serves every class. Showing it as "why this location" would be wrong.

## 2. Decision

1. **Three methods, each with one claim** (`ATTRIBUTION_CLAIMS` in `domain/types.py` is the only source of the wording):
   - `pooling_attention`: where the head's pooling put its mass (mean over heads, sums to 1, unsigned). Not class-specific.
   - `occlusion` (default): `p_full - p_removed` for the predicted class when a window of residues is removed from the pooling input. Signed: positive supports the class, negative argues against it. Window 5, stride 1, capped at 64 forward passes (stride raised and window widened to cover all residues).
   - `crf_marginal`: the CRF's posterior that a residue is signal, 0..1.
   `motif` stays as the legacy label for the hydropathy track and the old attribution; it is not served as attribution.
2. **Occlusion re-runs the head, not the encoder.** The frozen encoder's states are the input; a window is removed from the pooling input only. The track therefore measures what the head does with the states, not what the sequence does to the encoder.
3. **Deletion test before serving, protocol frozen first.** `data/manifests/attribution_protocol.json` (sha256 `67b09db9...` in the metrics manifest) fixes k = 5/10/20/40 %, 20 random draws with fixed seeds, and a cluster bootstrap (1000 samples, seed 20261005). Pass rule: the lower 2.5th percentile of (method drop minus random drop) is above 0 at every k. Only a passing method is served as `source: "model"`.
4. **Serving stays numpy.** `inference/attribution.py` imports only numpy, the standard library and the two numpy head modules; a test checks the imports. Attribution never feeds a score, calibrator or threshold.
5. **Wire (Gate C):** `API_VERSION` 1.3.0; `attribution_source`, new `attribution_method`, `attribution_signed`, `attribution_target`, `attribution_tracks` and `signal_track` on the result. `engine.json` holds the settings, and a test checks they equal the frozen protocol.
6. **GO attribution is validated, not served.** It passes the same test but the UI has no place to show a per-term track yet.

## 3. Consequences

- The default track is signed and tied to the predicted class; the UI shows the method and its claim and says it is not causal importance and not a binding site.
- Per-sequence cost rose by about 0.14 s in batches (40-sequence batch 5.97 s to 11.40 s, 0.149 to 0.285 s per sequence). `MAX_BATCH_SEQUENCES` stays 20 under the half-of-largest-measured rule.

## 4. Known limits

1. **Faithfulness is to this head, not to the biology.** The deletion test says removing the highlighted residues lowers the head's probability more than random removal; it says nothing about which residues matter in the cell.
2. **Occlusion does not re-run the encoder**, so context already mixed into each state is not removed with the window.
3. **CRF versus a position prior:** signal-peptide AUROC CRF 0.9990 [0.9965, 0.9999], motif hydropathy 0.863 [0.805, 0.889], a position-only prior 0.9936 [0.9896, 0.9964]. The CRF beats the motif track, but the position-only prior is close, so much of the signal is N-terminal position.
4. Mean deletion drops at k = 5/10/20/40: localization occlusion 0.058/0.085/0.112/0.136, pooling attention 0.045/0.077/0.114/0.149, motif 0.022/0.047/0.074/0.098; GO occlusion 0.130/0.184/0.227/0.253, GO pooling attention 0.165/0.217/0.261/0.286; random about 0 to 0.013. At k = 20 and 40 pooling attention drops the probability more than occlusion does, so "occlusion is the best deleter" is not claimed.
5. Gallery sequences are in-sample for the heads.
6. Timings come from one machine; the single-job p95 shift (2.587 to 3.523 s, n = 33) may be noise.

## 5. Alternatives rejected

- **Gradient x input:** needs torch autograd in serving.
- **Integrated gradients:** tens of forward and backward passes per sequence; too costly for the queue.
- **Attention only:** class-agnostic, so it cannot say why one class beat another.
- **Keep the motif track:** it is not derived from the model and had the weakest deletion drop.
