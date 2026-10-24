# ADR-0019: Opt-in live encoder for sequences outside the residue cache

**Status:** Accepted 2026-10-24 · **Amends:** ADR-0016 (serving engine) · **Feature:** 13

## Context

ADR-0016 serves the trained heads over cached ESM-2 states only: 500 sequences are answerable,
everything else is refused `not_cached`, and serving is numpy-only. That keeps the served path
small and makes every served number one that was measured, but it means the tool cannot score a
protein a user actually brings. The heads themselves take any (length, 640) float16 state matrix;
only the encoder is missing at serve time.

## Decision

1. **Opt-in, default off.** `ENCODER_MODE=cache_only` (default) is ADR-0016 unchanged: numpy-only,
   no torch import anywhere on the path, the same images and CI checks. `ENCODER_MODE=live` loads
   frozen ESM-2 and encodes any sequence the cache does not hold.
2. **Same encoder, same numbers.** Live states come from the Phase 7 code that built the cache
   (`TorchEncoder.embed_residues`) with the checkpoint, revision, context and sliding-window
   policy read from the cache's own `index.json`, rounded to float16 as the cache stores them.
   Cached sequences are still read from the cache in live mode.
3. **No network on the request path or at startup.** Weights load with local files only;
   `scripts/fetch_esm.py` fetches the pinned revision once into a hub cache (`ESM_CACHE_DIR`,
   default `data/esm2`). A missing file stops startup with that command in the message.
4. **Self-check or refuse.** At startup the shortest cached gallery sequence is encoded live and
   compared with its cached states; a maximum absolute difference above 0.02 stops the service.
   The measured difference is reported in provenance.
5. **Labelled, not hidden.** Every result says `encoder: "cache" | "live"`; a live one also says
   `outside_evaluated_subset: true`, the Inspector, printed report, batch dashboard and CSV say so,
   and `/demo/version` replaces the "only the 500 cached" limit with the live one. Calibrators and
   thresholds are unchanged; they were fitted on the subset, and no metric was measured on live
   sequences.
6. **Dependencies.** A separate `live` extra pins torch 2.14.1 and transformers 5.18.0 (the cache's
   own versions). `Dockerfile.live` takes `EXTRAS=serve,live`; `docker-compose.encoder.yml`
   layers it over the live stack. The default image keeps its no-torch check.

## Consequences

- Any eukaryote or plant sequence up to 5,000 residues can be scored when the operator opts in,
  at about 2 s per sequence on CPU (Phase 7 measurement) plus head time; the result cache makes a
  repeat free.
- `config_hash` is unchanged between modes (same bundles and pins); `provenance.encoder.mode`
  distinguishes them.
- Live results inherit every limit of the subset numbers, more strongly: they are out-of-sample
  for calibration too. The label is the mitigation; abstention is not widened (operator choice).
- The image with torch is about 1 GB larger; it is not built in CI.

## Rejected

- *Always on:* makes torch a hard serving dependency and breaks ADR-0016's guarantee.
- *Separate encoder service:* cleanest isolation, more moving parts than this week allows; the
  `ResidueEncoder` protocol leaves room for it.
- *Download on first start:* puts the network on the startup path and makes startup depend on a
  third-party host.
