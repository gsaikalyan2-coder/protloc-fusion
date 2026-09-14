# Wireframe — Batch

**Route** `/batch` · **Data** `POST /batch` → `202`, then `/jobs/{job_id}`

Whole-proteome mode. The screen where a user commits twenty thousand sequences and needs
to believe they can walk away.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Batch                                                               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │         Drop a FASTA file, or [ choose a file ]                │  │  --border dashed
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ecoli_k12.fasta · 4 401 sequences · 2.1 MB          [ remove ]      │  parsed locally
│                                                                      │
│  Organism group                                                      │
│  ( ) Eukarya  ( ) Gram-positive  (•) Gram-negative  ( ) Archaea       │
│  Applied to every sequence in the file.                              │  SAY IT
│                                                                      │
│                                                    [  Submit  ]      │
│                                                                      │
│  Large jobs take hours. You'll get a link — close the tab and come   │
│  back to it. Results expire after the retention window.              │
└──────────────────────────────────────────────────────────────────────┘

        ── after submission, at /jobs/{job_id} ──

┌──────────────────────────────────────────────────────────────────────┐
│  Job 3e91… · ecoli_k12.fasta · 4 401 sequences   [ Copy link ]       │
│                                                                      │
│  ● Running                                                           │
│  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  1 204 / 4 401 · 11 failed             │  REAL progress
│  about 38 minutes remaining                          [ Cancel job ]  │  hedged
│                                                                      │
│  Completed sequences are downloadable as they finish? NO —           │
│  the contract has no partial download. Wait, or cancel.              │  409 until complete
│                                                                      │
│  ─── when completed ──────────────────────────────────────────────   │
│  4 390 succeeded · 11 failed                                         │
│  [ Download JSON ]  [ Download CSV ]                                 │
│  CSV is one row per sequence and task. It omits per-residue          │  SAY IT UP FRONT
│  attribution, which does not fit a flat table.                       │
│                                                                      │
│  Failed sequences (11)                                    [ expand ] │
│  ▸ sp|P0A7B8 — sequence contains a character outside the alphabet    │
└──────────────────────────────────────────────────────────────────────┘
```

## States

1. **Empty** — drop zone only; Submit disabled with a reason.
2. **File chosen** — name, parsed sequence count and size. Parsing the count client-side
   is worth it: a user who uploaded the wrong file finds out before committing hours.
3. **Too large (413)** — the cap, in the same units as the file just shown.
4. **Malformed FASTA** — reject before submitting, naming the first bad line.
5. **Accepted (202)** — go to `/jobs/{job_id}`.
6. **Running** — `completed_count / total` and `failed_count`, all from `JobProgress`.
   Unlike a single sequence this is a real fraction, so the bar is determinate here and a
   shimmer elsewhere.
7. **Partly failed** — a completed job with failures is a **success**, not an error state.
   Failures are listed with their per-sequence `error`.
8. **Cancelled mid-run** — say how many completed before the stop.
9. **Refused (503)** — as everywhere else, with the reason.

## Rules

- **No partial download.** `GET /jobs/{id}/results` answers `409` until the job completes.
  A truncated download that looks complete is worse than a refusal, and a user who
  analyses 1 204 of 4 401 sequences without noticing has been actively misled.
- **A failed member never fails the job**, and never carries a partial prediction.
- **The CSV caveat appears next to the CSV button**, not in documentation. Someone will
  diff the two downloads and should already know why they differ.
- **The job survives the tab.** Say so on the submit screen, before the commitment, not
  after.
- The pending rules from `job.md` apply here too: hedged estimates, no hatching, progress
  legible as text for screen readers, throttled `aria-live`.
