# Design D2 — app shell, routing, status and command palette

ADR-0020 · 2026-10-24 · builds on D1

## What was built

- **Shell (`App.tsx`, `components/Sidebar.tsx`, `components/TopBar.tsx`, `styles/shell.css`).**
  Grouped, collapsible sidebar (240 / 64 px; the choice is kept in `localStorage`
  `protloc.sidebar.collapsed`, read and written in `try`/`catch`): Home · Analyze (Predict,
  Batch, Compare) · Explore (Catalog, Evidence, Mutation scanner) · Model (Trust). Sticky top bar
  with a breadcrumb, the palette trigger, backend status and the theme switch. Content column is
  fluid to 1440 px and centred. Skip link; focus, scroll and `document.title` follow the route.
  Below 900 px the sidebar shows icons only (the drawer is D7).
- **Routing (`services/router.ts`).** Hash routes, no dependency: `#/`, `#/predict`, `#/batch`,
  `#/compare`, `#/catalog`, `#/evidence`, `#/scanner`, `#/trust`. Refresh keeps the screen, Back
  works, and `#/predict?acc=<accession>` reopens and runs a cached sequence. Pasted sequences are
  never put in the URL.
- **Backend status (`components/StatusPill.tsx`).** The permanent banner is gone. A pill says
  Connecting / Online / Offline; its popover shows model, config, rules and threshold policy,
  the backend's own banner sentence and limits. Offline: every screen shows one callout with the
  start command and a copy button, and the app re-checks every 15 s.
- **Command palette (`components/CommandPalette.tsx`).** ⌘K / Ctrl+K. Jumps to any screen, and
  for a UniProt accession or entry name (`P12345`, `INS_HUMAN`) opens it in Predict through
  `GET /demo/catalog/{key}`; a miss shows the backend's reason inline. Combobox + listbox with
  `aria-activedescendant`; Radix Dialog supplies focus trap, Escape and scroll lock.
- **Renames to match the navigation:** Inspector → Predict, Proteome Triage Board → Batch triage,
  Example gallery → Catalog (headings, one hint in `problems.ts`, and the tests that assert them).
- **Dependencies (allowlisted with reasons, THIRD_PARTY rows 19–20):** `@radix-ui/react-dialog`
  1.3.1, `@radix-ui/react-popover` 1.3.1, `lucide-react` 1.56.0. New token `--scrim` (DESIGN.md
  §1.3).

## Verification

| Check | Result |
|---|---|
| `tsc --noEmit` · `eslint --max-warnings 0` · `prettier --check` | clean |
| `vitest run` | 21 files, 145 tests pass (new: `router.test.ts`, `CommandPalette.test.tsx`) |
| `vite build` | pass |
| `generate_tokens.py --check` · `check_frontend_deps.py` | current · pass |
| Built app in Chromium, mocked `/demo/version` | online, offline, palette, popover (dark), collapsed and 800 px layouts rendered and reviewed |

## UNVERIFIED

- Every screen against the live backend; the mocked run only exercised the shell.
- `npm ci` on Windows with the updated lockfile; GitHub Actions.
- Screen-reader pass (NVDA / VoiceOver) on the palette and popover.
