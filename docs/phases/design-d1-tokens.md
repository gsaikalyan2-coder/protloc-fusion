# Design D1 — tokens, type and the design-system ADR

Branch `phase/d1-design-tokens` · ADR-0020 · 2026-10-24

## What was built

- **ADR-0020** (Accepted): Radix Colors values, unstyled Radix Primitives (from D3), IBM Plex via
  `@fontsource`, `lucide-react` (from D2). Amends the "React only" line in `CLAUDE.md`.
- **`docs/DESIGN.md` v0.2.** §1 rewritten on Radix Sand / Indigo, §3.3 on Amber / Red, every value
  annotated with scale and step. New token names: `--surface-hover`, `--surface-active`,
  `--text-disabled`, `--accent-subtle-hover`, `--accent-solid`, `--action-bg`, `--action-bg-hover`,
  `--action-text`, `--state-{note,warn,block}-ink`, `--text-md`, `--tracking-tight`,
  `--tracking-caps`, `--radius-full`, `--z-*`, and §1.5 layout tokens. No v0.1 name removed.
  Compartment dictionary (§2) unchanged.
- **Type:** IBM Plex Sans / Mono 400 and 500, Latin subset, self-hosted (`main.tsx`). Base size
  15 → 14 px; radii 4/8/12 → 4/6/8.
- **Generator fix (`scripts/generate_tokens.py`).** CSS comments and at-rules are stripped before
  parsing. Before the fix the `:root` inside `@media (prefers-reduced-motion)` was read as a plain
  `:root` rule, so `--duration-fast` and `--duration-base` were `0ms` for every viewer; they are now
  120 ms and 180 ms, and 0 ms only under reduced motion.
- `app.css`: trust-state *text* now uses `--state-*-ink` (7 lines). No layout change in D1.
- `THIRD_PARTY.md` rows 17 (Radix Colors) and 18 (IBM Plex); `check_frontend_deps.py` allowlist
  gains the two `@fontsource` packages with reasons; `package-lock.json` updated.

## Verification

| Check | Result |
|---|---|
| `python scripts/generate_tokens.py --check` | current, 144 declarations |
| `python scripts/check_frontend_deps.py` | pass, 22 dependencies, all pinned |
| `tsc --noEmit` · `eslint --max-warnings 0` · `prettier --check` | clean |
| `vitest run` | 19 files, 133 tests pass |
| `vite build` | pass; fonts emitted as 4 woff2 + 4 woff |
| Built app in Chromium, light and dark | `--duration-fast` = 120 ms; `document.fonts.check("IBM Plex Sans")` true |
| WCAG 2 contrast (computed) | text-primary/secondary ≥ 5.3:1 on every surface, both themes; accent 6.0:1 light, 8.5:1 dark; focus ring 4.9:1 light, 4.4:1 dark; warning ink 10.9:1; blocking ink 4.5:1 light, 7.8:1 dark; all 13 compartment inks ≥ 5.6:1 on the new page colour |

Checks ran on Linux (Node 22, npm 10) against a copy of `frontend/`; the lockfile was regenerated
there and is platform-neutral.

## UNVERIFIED

- Visual review of every screen against the live backend, both themes (pages still use the old
  layout until D2–D6).
- GitHub Actions run on the branch.
- `npm ci` on Windows with the regenerated lockfile.
- Inline `style` colours in `Scanner.tsx` and `Trust.tsx` still use `--state-*` for text on
  `--surface-1` (4.6:1 for amber); they are removed in D3.
