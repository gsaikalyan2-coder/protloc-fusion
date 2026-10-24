# Design D3–D7 — components, screens, responsive and accessibility

ADR-0020 · 2026-10-24 · builds on D1 (tokens) and D2 (shell)

## D3 — shared components

- `styles/app.css` rewritten on the v0.2 tokens: buttons (primary = ink, secondary, ghost;
  28 / 32 / 40 px), inputs, native selects with a drawn chevron, segmented controls, tables in a
  bordered wrapper with tabular numerals, badges, trust-state flags, KPI tiles, skeletons, empty
  states, tooltips, job status line. The print stylesheet now targets the D2 shell classes (it
  still targeted `.shell`/`.side`/`.main`, which D2 had removed).
- `components/ui.tsx`: new `Select`, `Tip` (Radix Tooltip), `Skeleton`, `EmptyState`,
  `Segmented`, `GroupSegmented`, `SequenceForm`; `Card` takes a class; `Chip` and
  `ConfidenceRow` pass colour and geometry as CSS custom properties instead of inline styles;
  `ProvenanceBadge` is a focusable button with its explanation as a tooltip.
- Static inline styles removed from every page and component. What remains passes data
  (`--p`, `--tau`, `--heat`, `--mag`, residue shading), never a colour literal.
- Catalog browser and Recent jobs restyled (toolbar, pager, empty state, job-state dots).

## D4 — Home and About

- Home is a launchpad: the product promise, one sequence box (raw residues or FASTA; several
  records run as a batch), organism group as a segmented control, example picker, Predict.
  Below: the six outputs in one line each, then Recent jobs. `services/fasta.ts` parses input.
- New screen **About & limits** (`#/about`) holds what Home used to: the full statement of each
  output, the served limits from `/demo/version`, and the privacy notes. `pages/content.ts` is
  the one description of the outputs both screens read.
- Wording fix: "Catalog sequences are in-sample" (D2 rename) was wrong for partition 5; About
  now says examples and partitions 1–4 are in-sample and partition 5 is held out.

## D5 — Predict

- Order: input (compact once a result exists) · result header (id, length, group, hash, in-sample
  and live badges, Print, Send to scanner) · notices · **five-cell summary** (location, membrane,
  sorting signal, GO, homology) · per-residue track with the method as a segmented control ·
  localization beside membrane / signal / GO · **tabs** (Consistency, 3D structure, Engines,
  Export; Radix Tabs) · provenance footer. Skeleton while running, empty state before.

## D6 — the other screens

- Batch, Evidence and Scanner share `SequenceForm`. Batch runs a FASTA handed over from Home.
- Batch: KPI tiles, keyboard-operable triage rows, heatmap through `--heat`.
- Catalog: example tiles (buttons) above the full table. Compare: chosen-sequence strip, three-
  column GO comparison, empty state. Evidence and Scanner: empty states; the Scanner legend now
  says blue (it said green). Trust: KPI tiles, keyboard-operable label rows, skeleton and error
  states. Headings in sentence case.

## D7 — responsive, accessibility, print

- Below 768 px the sidebar is a drawer (menu button, scrim, Escape, focus moves in); below
  600 px the summary stacks, the search trigger is icon-only and cards tighten.
- No horizontal scroll at 390 px on Home, Predict, Catalog or Batch (measured
  `scrollWidth` = 390).

## Verification

| Check | Result |
|---|---|
| `tsc --noEmit` · `eslint --max-warnings 0` · `prettier --check` | clean |
| `vitest run` | 22 files, 149 tests pass (new: `fasta.test.ts`; palette count now 9 screens) |
| `vite build` | pass |
| axe-core 4.10.3 (WCAG 2.0 A/AA, 2.1 AA), 8 screens × light and dark, mocked backend | 0 violations; colour contrast of residue shading reported "incomplete" (not computable by axe; `--shade-max` keeps it at 4.5:1, docs/phases/16.md) |
| Print to PDF (Chromium, A4) from Predict | only the report prints |
| `generate_tokens.py --check` | current |

New pinned dependencies: `@radix-ui/react-tooltip` 1.4.1, `@radix-ui/react-tabs` 1.2.1
(allowlisted with reasons; THIRD_PARTY row 19).

## UNVERIFIED

- Every screen against the live backend (all visual checks used a mocked `/demo` API with
  invented values; the Trust screen and the scan results were not rendered with data).
- `npm ci` on Windows; GitHub Actions; screen readers (NVDA / VoiceOver); Safari and Firefox.
