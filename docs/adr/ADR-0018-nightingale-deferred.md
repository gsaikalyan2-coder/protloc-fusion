# ADR-0018 — Nightingale deferred: the sequence viewer stays hand-built and is polished instead

- **Status:** **Accepted** (2026-10-05)
- **Deciders:** Saikalyan (Phase 18 step 7: do the fallback)
- **Related:** ADR-0017 (attribution claims), `docs/phases/17-18.md`, `PROJECT_PLAN.md` §4 row 18

## 1. Context

Row 18 asked for Nightingale tracks to replace the hand-rolled `SequenceTrack`. The viewer is a
keyboard-operable listbox (one named option per residue, `aria-activedescendant`, arrow, Page and
Home/End keys), drives the 3D cursor link, draws three explanation tracks (signed occlusion, pooling
attention, CRF signal) from tokens with a per-theme `--shade-max`, and works in light and dark.

## 2. Decision

Do not adopt Nightingale now. Keep `SequenceTrack` and polish it (blocks of ten with a residue
ruler, a swatch legend, a bordered scrolling panel). No dependency is added.

## 3. Reasons

1. Nightingale draws in canvas/SVG web components. The listbox, per-residue ARIA names and
   keyboard model would have to be rebuilt on top of it: the behaviour the work was told not to break.
2. The ADR-0017 claims depend on exactly what is drawn (signed shade, dotted "argues against" rule).
   Re-expressing them through a library API risks the claim drifting from the picture.
3. The repo bans colour literals and generates tokens; Nightingale styles through its own attributes,
   so an adapter would be needed to keep the contrast guarantee behind `--shade-max`.
4. It adds dependencies (ISC and MIT licences seen on npm, not reviewed), bundle size and notices work.
   This repo has no `molstar` or web-component dependency today.

## 3b. Not verified

Nightingale's keyboard and screen-reader behaviour was not tested; the reasons above rest on it being
a canvas/SVG component library. A prototype would settle that.

## 4. Consequences

- Row 18's "Nightingale tracks replace `SequenceTrack`" is **not met**; the row is marked deferred.
- Revisit if a prototype shows Nightingale can carry the listbox semantics, the three tracks and both
  themes, or if a feature the hand-built viewer cannot offer is needed (for example standard
  UniProt feature tracks).
