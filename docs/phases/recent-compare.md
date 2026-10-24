# Recent jobs and Compare (features 7 + 8)

Branch `phase/19d-recent-compare`, from `phase/19c-cache-download`. Done 2026-10-24.
Frontend only: no backend, contract, registry or dependency change.

## 1. Feature 7: Recent jobs

- `services/history.ts`: every accepted job (predict, batch, scan) is recorded by `useJob` in
  `localStorage` (`protloc.recentJobs.v1`), newest first, at most 20, de-duplicated by job id.
  A predict or scan entry keeps its sequence so it can be reopened; a batch keeps only its first
  record id and count, never the FASTA. Every storage read and write is guarded: in a private
  window or with site data blocked the list is just empty.
- `components/RecentJobs.tsx` on Home: each row asks `GET /demo/jobs/{id}` once and shows
  completed, failed, running, `expired on the server` (410) or `not on the server (restarted)`
  (404; the queue is in memory, ADR-0008). Open re-runs a predict entry in the Inspector (the
  result cache answers it at once when it holds it) or loads a scan entry in the Scanner;
  a completed predict or batch job offers the whole-job CSV and JSON downloads; Forget and Clear.
- Home's Privacy card states what the list stores and where.

## 2. Feature 8: Compare

- New screen `Compare` (sidebar, after Inspector). Each side is chosen by accession or entry name
  (`GET /demo/catalog/{key}`) or from the gallery, with its own organism group; Compare runs both
  as jobs.
- `services/compare.ts` (pure): a summary table (length, group, predicted locations or "no
  confident call", abstention, sorting signal and cleavage site, membrane type, consistency
  flags, nearest homolog) with differing rows highlighted; every location label with both
  calibrated probabilities, the threshold marks and the difference B - A; predicted GO terms split
  into both / only A / only B. The page states that a difference is between two model outputs,
  not a measured difference between the proteins.
- One CSS rule (`tr.differs`), built from an existing token.

## 3. Checks (Linux VM over the Windows checkout)

eslint, prettier, `tsc --noEmit`: clean. Vitest 122 passed in 16 files (new: `history` 7,
`RecentJobs` 4, `compare` 5, `Compare` page 2). `vite build` 230.79 kB (gzip 72.48 kB).
Backend unchanged since `cache-download` (1,088 passed).

## 4. UNVERIFIED

- Both screens by eye in light and dark against the live backend, and at narrow width.
- Recent jobs across a real backend restart (the 404 path is unit-tested only).
