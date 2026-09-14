# Wireframe — Job

**Route** `/jobs/{job_id}` · **Data** `GET /jobs/{job_id}`, `POST /jobs/{job_id}/cancel`

This is `result_url`. It is the screen ADR-0001 created and the one most likely to be
designed badly, because it renders a state that is neither an answer nor a failure.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ProtLoc Fusion                                    [ ☀/☾ ]  [ Docs ] │
├──────────────────────────────────────────────────────────────────────┤
│  Job 9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04          [ Copy link ]     │  --font-mono
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ● Running                                                     │  │  --accent
│  │  ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░  shimmer, not a percentage     │  │  DESIGN §3.4
│  │  Position 3 · about 24 s remaining                             │  │  hedged, always
│  │                                              [ Cancel job ]    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Localization                 [ no chip — nothing predicted yet ]    │  ABSENT, not grey
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░  Queued                                 │  empty track
│                                                                      │
│  Membrane type                                                       │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░  Queued                                 │
│                                                                      │
│  … five more heads …                                                 │
│                                                                      │
│  This page updates itself. You can close it and come back — the      │
│  link works until the result expires.                                │
│                                                                      │
│  model — · rules — · experiment — · no bundle loaded                 │  null renders as —
└──────────────────────────────────────────────────────────────────────┘
```

## States — all five job statuses, plus the two that are not statuses

1. **queued** — empty track, the word `Queued`, position and hedged estimate. No chip.
2. **running** — shimmering track, the word `Running`. Shimmer is decoration: progress is
   also text, because the animation is off under `prefers-reduced-motion` and invisible to
   a screen reader. Batch jobs announce `completed / total` via `aria-live="polite"`,
   throttled to at most one announcement per 10 s so a 20 000-sequence job does not flood
   a screen reader.
3. **completed** — replace this view with the Result screen. Same route.
4. **failed** — the track is replaced by an error rule; the job's `error` is shown. No
   partial results, because there are none.
5. **cancelled** — dimmed track, the reason, and a "Submit again" action.
6. **410 Gone** — *"This result has expired."* Not a 404 page: the job existed. Offer to
   resubmit; the sequence is gone, so say that too.
7. **404** — *"No job with that identifier."* Distinct copy from the above.

## Rules — the ones that matter more than the layout

- **Pending must never look like abstention.** `docs/DESIGN.md` §3.4 is explicit: no
  hatching, because diagonal hatch means abstention and nothing else. *"The model declined
  to assert"* and *"the model has not run yet"* are opposite claims, and a user who
  confuses them reads an empty page as a scientific result.
- **The compartment chip is absent, not greyed.** A desaturated chip means abstention; an
  absent chip means no answer yet.
- **The estimate always carries a hedging word** — `about`, `~`. It is computed from queue
  depth and measured throughput and is never presented as a promise.
- **Polling backs off** — ~1 s for the first 10 s, then 2 s, then 5 s, capped. Stop on any
  terminal status, on `410`, and when the tab is hidden.
- **The reproducibility footer renders `null` as `—`** plus a note, never four blanks.
