# Wireframe — Predict

**Route** `/predict` · **Data** `POST /predict` → `202` · **Phase** 12.

The submission screen. Its entire job is to be honest about what happens next.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ProtLoc Fusion                                    [ ☀/☾ ]  [ Docs ] │
├──────────────────────────────────────────────────────────────────────┤
│  Predict                                                             │
│                                                                      │
│  Sequence                                            1 842 residues  │  live count
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ MKTFFVLLLACVSVLAQSEEKLVLHNGQLVSGSAEQVGQ…                       │  │  --font-mono
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  FASTA headers are stripped. Whitespace and case are normalised.     │  --text-secondary
│                                                                      │
│  Organism group                                                      │
│  ( ) Eukarya   ( ) Gram-positive   ( ) Gram-negative   ( ) Archaea    │  no default
│  Conditions the model. Prokaryotic localization is masked for        │
│  Eukarya — masked, which is not the same as abstained.               │
│                                                                      │
│  Identifier (optional)  ┌──────────────┐                             │
│                         │ my_protein_1 │  echoed back, not modelled  │
│                         └──────────────┘                             │
│                                                                      │
│                                      [ Cancel ]   [  Submit  ]       │
│                                                                      │
│  Submitting queues a job. You'll get a link you can share or come    │  SET EXPECTATION
│  back to. Nothing runs in this tab.                                  │  BEFORE the click
└──────────────────────────────────────────────────────────────────────┘
```

## States

1. **Empty** — Submit disabled; the reason is its accessible name, not a tooltip.
2. **Invalid** — inline, on blur, never on every keystroke: empty sequence, a character
   outside the accepted alphabet (policy frozen in Phase 3, read from config), no organism
   group chosen. The offending residue is highlighted in the textarea.
3. **Submitting** — button shows a spinner and keeps its label. The form is disabled but
   still readable, so a slow connection does not erase what the user typed.
4. **Accepted (202)** — navigate immediately to `/jobs/{job_id}`. Do not render a result
   here; there is none.
5. **Accepted inline (200)** — the fast path may happen. Navigate to `/jobs/{job_id}`
   anyway and render the result there. **One code path.** Special-casing `200` here is how
   a frontend acquires two result renderers that drift.
6. **Refused (503)** — *"Predictions are unavailable: no model bundle is loaded."* Show the
   `detail`, which names the phase. The typed sequence is preserved.
7. **Too large (413)** — the cap, and what to do: use Batch.
8. **Rate limited (429)** — when to retry, from the response.
9. **Network error** — *"Could not reach the service."* Nothing was submitted; say so, so
   the user does not fear a duplicate job.

## Rules

- **The wait is disclosed before the click, not after.** A user who reads "queued" only
  once the spinner is already turning has been surprised by the architecture.
- **Never assume `200`.** The client handles `202` correctly regardless — the contract is
  asynchronous and the inline window is a convenience.
- `error` codes drive branching; `message` is for the human and may be reworded without a
  contract change.
