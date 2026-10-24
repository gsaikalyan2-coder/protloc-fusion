# Batch dashboard and printable report (features 9 + 10)

Branch `phase/19e-dashboard-report`, from `phase/19d-recent-compare`. Done 2026-10-24.
Frontend only: no backend, contract, registry or dependency change (React only; no chart or
PDF library).

## 1. Feature 9: batch dashboard

- `services/batchStats.ts` (pure): over the scored records of one batch, counts predicted
  locations (a dual-targeted protein counts under each), the top calibrated probability in five
  bins plus the no-confident-call count, sorting-signal types, the eight most frequent predicted
  GO terms, evidence states (homolog found / disagrees / none found / unavailable) and result-cache
  hits. Refused records are counted apart and never enter a distribution.
- `components/BatchDashboard.tsx` on the Triage page, below the KPI tiles: plain CSS bars
  (`.bars`, tokens only), location bars in each compartment's own colour token, and every bar
  carries its count and share in text. The card states that these are counts of model outputs
  over this batch, not prevalence estimates.

## 2. Feature 10: printable report

- `components/PrintReport.tsx`, rendered by the Inspector and hidden on screen. "Print report"
  calls `window.print()`; the print stylesheet (`@media print`, A4, 14 mm margins) hides the
  sidebar and every other element of the page and shows only the report, in system colours
  (`Canvas`, `CanvasText`, `GrayText`) so it prints dark-on-light in either theme.
- Contents: the location call (or "no confident call") and every label's calibrated probability,
  threshold and calibration; sorting signal and cleavage site with untrained types named;
  membrane type; predicted GO terms; consistency flags; up to five homologs and the model-homology
  agreement; the residues with the largest positive attribution and the claim that goes with it;
  the sequence in numbered blocks of ten; the engine behind each task; and a footer with model
  version, config hash, rules, experiment, job id, sequence hash, scope and "not for clinical or
  diagnostic use". In-sample and evidence-unavailable warnings print when they apply.
- `services/report.ts` (pure): `sequenceLines`, `topResidues`.

## 3. Checks (Linux VM over the Windows checkout)

eslint, prettier, `tsc --noEmit`: clean. Vitest 132 passed in 19 files (new: `batchStats` 4,
`BatchDashboard` 2, `PrintReport` 4). `vite build` 240.30 kB (gzip 75.07 kB), CSS 14.28 kB.
Backend unchanged (1,088 passed at `cache-download`).

## 4. UNVERIFIED

- The print preview in a real browser (page breaks, one vs two pages for long sequences), in
  light and dark themes.
- The dashboard by eye against a real batch, light and dark, and at narrow width.
