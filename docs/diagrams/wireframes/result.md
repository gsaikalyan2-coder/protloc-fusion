# Wireframe — Result

**Route** `/jobs/{job_id}` with `status: "completed"` · **Data** `GET /jobs/{job_id}`

The screen the whole project exists to produce. Its hierarchy encodes a claim: the
prediction is the answer, the rule set audits it, and the evidence is independent of both.

```
┌──────────────────────────────────────────────────────────────────────┐
│  my_protein_1 · 1 842 aa · Eukarya            [ Download ▾ ] [ Copy ]│
├──────────────────────────────────────────────────────────────────────┤
│  ⚠  2 consistency notes — 47 rules checked                           │  BEFORE results
│     ▸ Signal peptide predicted with nuclear localization  [warning]  │  --state-warn
│     ▸ Transit peptide cleavage site beyond residue 90     [note]     │  --state-note
│                                                                      │
│  Localization                                                        │
│  ● Mitochondrion  ● Cytoplasm                                        │  --loc-mit --loc-cyt
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  0.87    τ 0.61                         │  --bar-fill
│  ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  0.64    τ 0.61                         │
│                                                                      │
│  Membrane type                                                       │
│  ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱  Abstained · top 0.44 below τ 0.58       │  HATCH = abstain
│                                                                      │
│  Prokaryotic localization                                            │
│  Not applicable for Eukarya — masked, not scored                     │  MASKED ≠ abstained
│                                                                      │
│  Signal peptide            None · 0.03                               │
│  Mito. transit peptide     Present · 0.91 · cleavage after residue 34│
│                                                                      │
│  ─── Per-residue explanation ──────────────────────────────────────  │
│  [ Localization ▾ ]                                                  │
│   1 M K T F F V L L L A C V S V L A Q S E E K L V L H N G Q L V S   │  --font-mono
│     ▁▃▅█████▇▅▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                             │  signed attribution
│     └─ transit peptide region ─┘                                     │  region underlay
│  Comparable within this task and sequence only. Not a probability.   │
│                                                                      │
│  ─── Independent evidence ─────────────────────────────────────────  │  SUBORDINATE
│  Swiss-Prot 2026_03 · GOA 2026-02-14                                 │  --text-secondary
│  P12345  Cytochrome c oxidase subunit 4   61% id  1e-88  Mitochondrion│
│  Q67890  COX4 homolog                     44% id  3e-51  Mitochondrion│
│  GO:0005739 mitochondrion · IDA · from P12345                        │
│                                                                      │
│  model 1.0.0 · rules 1.2.0 · EXP-0031 · config a3f9c1…  [ Cite ]     │
└──────────────────────────────────────────────────────────────────────┘
```

## States

1. **Clean** — no violations. Still show *"47 rules checked, none triggered"*. An empty
   list must read as *checked and clean*, never as *not checked*; that is what
   `rules_checked` is in the contract for.
2. **Blocking violation** — the affected task shows as abstained, and the violation card
   says the rule forced it. The model's own output is still shown beside it: a violation
   never silently edits a prediction.
3. **Abstained task** — hatched track, the top probability and the τ it failed to clear.
   Showing how close the call was is the difference between an abstention a user can
   reason about and one that looks like a bug.
4. **Masked task** — a sentence, no bar, no chip. Never scored, so never abstained.
5. **Evidence unavailable** — `available: false` with the `unavailable_reason` rendered.
   **Never two empty tables**, which would read as *"no homologs found"* — a different and
   much stronger claim.
6. **Member failed** (batch) — the sequence's `error`, and no partial prediction.
7. **Long sequence** — the residue grid virtualises; keyboard navigation still reaches
   every residue.

## Rules

- **The consistency summary is above the predictions.** A self-audit placed below the
  answer is a footnote, and this project's contribution is that the audit happens at all.
- **Evidence is visually subordinate** — smaller type, `--text-secondary`, below the fold.
  It is what the databases already say, offered beside the prediction, never folded into
  it. Presenting it as confirmation would launder homology into model confidence.
- **τ is shown, never assumed.** It comes from calibrator metadata at prediction time.
  There is no default τ, and a literal `0.5` anywhere is a bug.
- **Attribution is signed and scoped.** The caption is part of the design, not a nicety:
  comparable within one task and sequence, never across tasks, never as a probability.
- Every compartment colour is a `--loc-*` token from `docs/DESIGN.md`. A hex in a component
  is a review rejection.
