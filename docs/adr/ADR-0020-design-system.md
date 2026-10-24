# ADR-0020: Design system — Radix Colors, Radix Primitives, IBM Plex

**Status:** Accepted 2026-10-24 · **Amends:** the "React only" convention in `CLAUDE.md` · **Supersedes:** the v0.1 token values in `docs/DESIGN.md`

## Context

The frontend grew from five screens to eight plus recent jobs, compare, a batch dashboard and a
printable report. It still uses browser-default controls (native `select`, unstyled `textarea`),
a flat eight-item navigation, a permanent banner and hand-picked neutrals with no step logic. A
generator bug (fixed here) also read the `prefers-reduced-motion` rule as a plain `:root` rule, so
every duration was `0ms` for everybody. The result reads as a document, not an instrument.

Building accessible menus, selects, tabs, tooltips and dialogs by hand is the expensive and
error-prone part (focus trapping, roving tab index, typeahead, collision-aware positioning). The
look is the cheap part and is already governed by `docs/DESIGN.md` ("chrome is achromatic, colour
means biology").

## Decision

1. **Colour values: Radix Colors** (`radix-ui/colors`, MIT, pinned `dbdb854`, package 3.0.0). Sand
   for neutrals, Indigo for the single accent, Amber and Red for the warning and blocking states.
   Values are **copied into `docs/DESIGN.md`** with their scale and step annotated; no Radix Colors
   package is installed or shipped. The compartment dictionary (§2) is unchanged.
2. **Interaction primitives: Radix Primitives** (`radix-ui/primitives`, MIT), unstyled. Each
   primitive package is added in the design phase that first uses it (Dialog and Popover in D2,
   Tooltip in D3, Tabs in D5), pinned exactly and
   listed in `scripts/check_frontend_deps.py` with its reason. Styling stays plain CSS on the
   generated tokens. No styled kit (Radix Themes, shadcn/ui, MUI, Mantine), no CSS framework.
3. **Type: IBM Plex Sans and IBM Plex Mono** (SIL OFL 1.1), self-hosted with `@fontsource` 5.3.0,
   Latin subset, weights 400 and 500 only. No font CDN.
4. **Icons: `lucide-react`** (ISC), added in D2 where the navigation first uses it, 16 px, in
   navigation and action buttons only.
5. **Primary action is achromatic** (`--action-bg` = Sand 12). The accent is for links, focus and
   selection. Text uses Radix steps 11 and 12 only, so every text token clears 4.5:1.

   **Not adopted:** Radix Select (a styled native `select` keeps the platform picker, the
   keyboard model and form semantics), Toggle Group (a segmented control is a `role="group"` of
   `aria-pressed` buttons), Toast (nothing in the product needs a transient message).

## Consequences

- `CLAUDE.md`'s "React only" rule now reads "React plus what ADR-0020 admits"; the allowlist in
  `scripts/check_frontend_deps.py` remains the enforcement, one reason per package.
- Token *names* from v0.1 are unchanged; v0.2 adds names (`--surface-hover`, `--action-*`,
  `--state-*-ink`, layout and layer tokens). Existing components render with the new values
  without code changes.
- Amber 11 on Amber 3 is 4.3:1, under the AA floor, so warning text uses Amber 12
  (`--state-warn-ink`) in light mode; the rule and icon keep Amber 11.
- Bundle: four woff2 files, about 77 kB in total, loaded with `font-display: swap`.
- Reversible: the token names insulate components from the scale, and the primitives are wrapped in
  `components/ui.tsx`, so a later swap touches one file per primitive.

## Alternatives rejected

- **shadcn/ui** — requires Tailwind (a CSS framework, against the convention) and is the most
  recognisable generated-app look.
- **IBM Carbon** — strong for dense data, but heavy, and the product would look like an IBM product.
- **Material 3, Mantine, Primer** — full styled kits; they replace the design system instead of
  serving it.
