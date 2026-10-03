# ADR-0009 — Frozen encoder, trained heads; ESM-2 150M

- **Status:** **Accepted** (2026-10-03). Measured figures in §4 are from a 500-sequence subset
  (`docs/phases/07.md`); the full-cache numbers are still outstanding.
- **Deciders:** Saikalyan
- **Related:** ADR-0001 (queue everything), ADR-0007 (hosting), ADR-0008 (batch mode)

---

## 1. Context

The shared representation is a protein language model. Three things constrain the choice:

- **One week of calendar time.** LoRA and full fine-tuning need GPU hours the plan does not
  have, and each fine-tuned variant must be re-evaluated on all seven heads.
- **Serving on a free or near-free CPU host** (ADR-0007). fp32 weights are about 600 MB for
  150M parameters against about 2.6 GB for 650M (ADR-0001 §1.1).
- **Reproducibility.** A vector must be traceable to the exact weights that made it.

The earlier plan named `esm2_t33_650M_UR50D`. CLAUDE.md revised this to 150M on 2026-10-02
on published evidence: DeepLoc's authors tested 650M and 15B and reported no improvement on
this task, and 150M matched 3B across 36 to 40 DMS datasets. Those are the cited sources'
statements, not measurements made in this project.

## 2. Decision

1. **Encoder:** `facebook/esm2_t30_150M_UR50D` (30 layers, hidden size 640, MIT licence),
   pinned at commit `a695f6045e2e32885fa60af20c13cb35398ce30c`. `config.json` at that
   revision gives `hidden_size` 640 and `max_position_embeddings` 1026.
2. **Adaptation:** none. The encoder is frozen and run in eval mode with no gradients; only
   the heads are trained (Phase 9 onward).
3. **Representation stored:** the mean over residues of the last hidden state, float32, one
   vector per distinct sequence. `<cls>`, `<eos>` and padding never enter the mean.
4. **Long sequences:** ESM-2 holds 1022 residues (1026 positions, less the 2-position offset,
   less `<cls>` and `<eos>`). Longer sequences are cut into 1022-residue windows that start
   every 511 residues, the last moved back to end at the sequence end. A residue covered by
   several windows contributes the average of its states; the vector is the mean over
   residues. Nothing is truncated or dropped. In the working set this affects 2,060 of
   33,461 distinct sequences (6.2%), the longest being 1,496 residues.
5. **Cache identity:** `sha256` of the canonical spec (checkpoint, revision, dimension,
   pooling, dtype, window size, stride, policy). A cache directory holding another key is
   refused. A vector produced under another key is a different vector.
6. **Determinism claim:** byte-identical output on the **same machine, same device and same
   torch build**, across separate processes and after a real kill and resume. No claim is made
   across CPU and GPU or across torch builds; the CPU-vs-GPU maximum difference is reported
   separately.
7. **Per-residue states are not cached in Phase 7.** The registry's 14,935,037 distinct
   residues at 640 dimensions would take 19.1 GB in float16 (38.2 GB in float32), computed from
   the corpus, against a repository volume with about 3.4 GB free. Phase 11 (signal-peptide
   CRF, ENH) needs per-residue states and will recompute them for the subset it uses, or
   reopen this decision. This is recorded, not silently dropped.

## 3. Alternatives

| Option | Why not |
|---|---|
| ESM-2 650M | About 4.3x the parameters and 2.6 GB of weights for no reported gain on this task; does not fit the CPU host comfortably. |
| LoRA or full fine-tuning | Out of budget for one week; each variant multiplies evaluation. Reopen only if frozen heads miss the metric floor. |
| Truncate to the first 1022 residues | Loses C-terminal signals (PTS1, KDEL, transmembrane segments) on 6.2% of sequences, the ones a localisation model most needs. |
| Exclude long sequences | Biases the benchmark against long proteins. |
| ESM-C 300M | Worth one same-day test once caching is encoder-generic; not adopted without a measurement. |

## 4. Measured cost

Computed from the corpus (`scripts/embed.py --facts`, registry SHA-256 `37ca5e05…276a`):

| | Value |
|---|---|
| Accessions / distinct sequences | 33,879 / 33,461 |
| Pooled cache, expected (distinct × 640 × 4 bytes) | 85,660,160 bytes (an expectation, replaced by the measured size) |
| Per-residue float16 / float32, if ever stored | 19.1 GB / 38.2 GB |
| Sequences containing B, O, U, X or Z | 166 (all letters in the working set are in the ESM-2 vocabulary) |

Measured on a Windows CPU (12 logical cores, torch 2.14.1+cpu, transformers 5.18.0), 500 real
registry sequences spread over the length distribution:

| | Value |
|---|---|
| Extraction time | 2.011 s per sequence (1,005.3 s for 500). Full run would be about 18.7 h: an estimate |
| Cache size, 500 sequences | 1,315,501 bytes |
| Padding effect (alone vs batched) | at most 9.54e-07 absolute |
| Byte identity | same output across two processes (`797b5c9a…cab`) and after a real kill and resume (`1b46ee27…f06`), on this machine only |

The full 33,461-sequence cache is not built. GPU timing and CPU-vs-GPU difference are not measured.

## 5. Consequences

**Positive.** One cheap, reproducible representation shared by every head; a cache of about
86 MB that fits in the HF model repo and in memory; heads train in minutes, so the metric
floor and all seven heads are reachable in the week.

**Negative.** A frozen encoder caps accuracy below what fine-tuning could reach. Window edges
carry less context than window centres; averaging mitigates but does not remove this. Per-residue
explanation (Phase 15) cannot read the cache.

**Revisit if** frozen heads fall below the Phase 8 floor on validation, or Phase 11 needs
per-residue states for more than a subset.
