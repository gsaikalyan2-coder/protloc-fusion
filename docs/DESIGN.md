# DESIGN.md — ProtLoc Fusion

Single source of truth for the visual system. Every UI surface, every exported
paper figure, and every generated diagram reads its colors, spacing and states
from this document. Nothing hardcodes a hex value that is not defined here.

Status: **v0.2 — accepted (ADR-0020).** Neutrals, accent and state colours are
Radix Colors steps; type is IBM Plex. Token *names* from v0.1 are unchanged and
may be depended on; v0.2 only adds names.

---

## 0. The governing rule

> **Chrome is achromatic. Color means biology.**

The application shell — navigation, buttons, cards, borders, form controls — is
built entirely from a neutral gray ramp plus one restrained accent. Saturated
color is reserved exclusively for subcellular compartments and for the three
trust states (uncertain / warning / blocked).

This exists because the result page shows up to seven task outputs at once. If
buttons, links and headers compete for hue, a reader cannot tell at a glance
that "the teal thing" means extracellular. Spending color on chrome would spend
the one channel that carries scientific meaning.

**Corollary — color is never the only encoding.** Every compartment carries a
color *and* a three-letter code *and* a fixed ordinal position *and* a fill
pattern. Any one of those alone is sufficient to identify it. This is what makes
the accessibility contract achievable rather than aspirational.

---

## 1. Token layer

All tokens are CSS custom properties on `:root`, redefined under
`:root[data-theme="dark"]` and under `@media (prefers-color-scheme: dark)` for
viewers who have set no explicit preference.

**Source of the values (v0.2, ADR-0020).** Every neutral, accent and state colour
below is a step of a [Radix Colors](https://github.com/radix-ui/colors) 12-step
scale (MIT, `radix-ui/colors` at `dbdb854`, package 3.0.0), copied here as values:
no Radix Colors package ships to the browser. The scales are **Sand** (neutrals),
**Indigo** (the one UI accent), **Amber** (warning) and **Red** (blocking). Each
value is annotated with its scale and step so a change can be made by moving a
step, not by inventing a hex. The compartment palette in §2 is unchanged and is
not a Radix scale: it is a fixed dictionary of its own.

Radix's step semantics are kept: 1–2 backgrounds, 3–5 component states, 6–8
borders, 9–10 solid fills, 11 low-contrast text, 12 high-contrast text.

### 1.1 Neutrals and surfaces

```css
:root {
  /* Surfaces — page, card, inset, and the two interaction states */
  --surface-0:           #F9F9F8;  /* sand 2  — page */
  --surface-1:           #FFFFFF;  /* white   — card, panel, popover */
  --surface-2:           #F1F0EF;  /* sand 3  — inset, input, table header */
  --surface-hover:       #E9E8E6;  /* sand 4 */
  --surface-active:      #E2E1DE;  /* sand 5 */

  /* Text. Radix uses step 12 and step 11 for text and nothing lighter, so
     --text-muted equals --text-secondary: both clear 4.5:1 on every surface.
     --text-disabled is for disabled controls and placeholders only. */
  --text-primary:        #21201C;  /* sand 12 */
  --text-secondary:      #63635E;  /* sand 11 */
  --text-muted:          #63635E;  /* sand 11 */
  --text-disabled:       #8D8D86;  /* sand 9  */
  --text-inverse:        #FDFDFC;  /* sand 1  */

  /* Borders */
  --border:              #DAD9D6;  /* sand 6 — dividers, card edges */
  --border-strong:       #CFCECA;  /* sand 7 — control edges */
  --border-stronger:     #BCBBB5;  /* sand 8 — hovered control edges */

  /* The single UI accent — links, focus, the selected nav item.
     Indigo is a blue; the nucleus compartment ink (#3D3C8C) is a violet,
     and the two are never placed adjacent at the same weight. */
  --accent:              #3A5BC7;  /* indigo 11 */
  --accent-hover:        #1F2D5C;  /* indigo 12 */
  --accent-subtle:       #EDF2FE;  /* indigo 3  */
  --accent-subtle-hover: #E1E9FF;  /* indigo 4  */
  --accent-solid:        #3E63DD;  /* indigo 9  */
  --focus-ring:          #3E63DD;  /* indigo 9  — 4.9:1 on the page */

  /* Primary action. Achromatic on purpose: the strongest button on a page
     is ink, not a hue, so it never competes with a compartment. */
  --action-bg:           #21201C;  /* sand 12 */
  --action-bg-hover:     #3B3A37;  /* sand dark 6 */
  --action-text:         #FDFDFC;  /* sand 1  */
}

:root[data-theme="dark"] {
  --surface-0:           #111110;  /* sand dark 1 */
  --surface-1:           #191918;  /* sand dark 2 */
  --surface-2:           #222221;  /* sand dark 3 */
  --surface-hover:       #2A2A28;  /* sand dark 4 */
  --surface-active:      #31312E;  /* sand dark 5 */

  --text-primary:        #EEEEEC;  /* sand dark 12 */
  --text-secondary:      #B5B3AD;  /* sand dark 11 */
  --text-muted:          #B5B3AD;  /* sand dark 11 */
  --text-disabled:       #6F6D66;  /* sand dark 9  */
  --text-inverse:        #111110;  /* sand dark 1  */

  --border:              #3B3A37;  /* sand dark 6 */
  --border-strong:       #494844;  /* sand dark 7 */
  --border-stronger:     #62605B;  /* sand dark 8 */

  --accent:              #9EB1FF;  /* indigo dark 11 */
  --accent-hover:        #D6E1FF;  /* indigo dark 12 */
  --accent-subtle:       #182449;  /* indigo dark 3  */
  --accent-subtle-hover: #1D2E62;  /* indigo dark 4  */
  --accent-solid:        #3E63DD;  /* indigo dark 9  */
  --focus-ring:          #5472E4;  /* indigo dark 10 — 4.4:1 on the page */

  --action-bg:           #EEEEEC;  /* sand dark 12 */
  --action-bg-hover:     #B5B3AD;  /* sand dark 11 */
  --action-text:         #111110;  /* sand dark 1  */
}
```

### 1.2 Type

Two families from **IBM Plex** (SIL OFL 1.1), self-hosted through
`@fontsource` so no request leaves the page. Plex Sans for the interface; Plex
Mono for anything that is a residue, an accession, a hash, a probability or a
coordinate.

```css
:root {
  --font-sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;

  --text-xs:   12px;   /* metadata, hashes, axis ticks, overlines */
  --text-sm:   13px;   /* table cells, captions, badges, nav */
  --text-base: 14px;   /* body and controls */
  --text-md:   16px;   /* card titles, the launch input */
  --text-lg:   18px;   /* section headings */
  --text-xl:   24px;   /* page title */
  --text-2xl:  32px;   /* home launchpad only */

  --leading-tight: 1.25;
  --leading-body:  1.55;

  --tracking-tight: -0.01em;  /* --text-xl and up */
  --tracking-caps:   0.06em;  /* uppercase overlines */

  --weight-regular: 400;
  --weight-medium:  500;
}
```

Two weights only — 400 and 500. Bold-heavy interfaces read as marketing; this
is an instrument. Numerals that are compared down a column use
`font-variant-numeric: tabular-nums`.

### 1.3 Space, radius, elevation, layers

```css
:root {
  --space-1: 4px;   --space-2: 8px;   --space-3: 12px;
  --space-4: 16px;  --space-5: 24px;  --space-6: 32px;
  --space-7: 48px;  --space-8: 64px;

  --radius-sm:   4px;     /* badges, chips, checkboxes */
  --radius-md:   6px;     /* buttons, inputs, menus */
  --radius-lg:   8px;     /* cards, panels, dialogs */
  --radius-full: 9999px;  /* status dots, pills */

  --border-width:      1px;
  --border-width-emph: 3px;   /* left rule on trust callouts */

  /* Elevation is expressed with borders, not shadows, except for
     genuinely floating layers (menu, popover, tooltip, dialog, toast). */
  --shadow-overlay: 0 10px 38px -10px rgb(22 23 24 / 0.35), 0 10px 20px -15px rgb(22 23 24 / 0.2);
  /* Behind a modal layer (the command palette, dialogs). */
  --scrim:          rgb(22 23 24 / 0.32);

  --z-sticky:  10;
  --z-topbar:  20;
  --z-overlay: 50;
  --z-toast:   60;
}

:root[data-theme="dark"] {
  --shadow-overlay: 0 10px 38px -10px rgb(0 0 0 / 0.7), 0 10px 20px -15px rgb(0 0 0 / 0.5);
  --scrim:          rgb(0 0 0 / 0.55);
}
```

### 1.4 Motion

```css
:root {
  --duration-fast: 120ms;   /* hover, focus, press */
  --duration-base: 180ms;   /* panels, menus, the collapsing sidebar */
  --ease: cubic-bezier(0.2, 0, 0, 1);
}
@media (prefers-reduced-motion: reduce) {
  :root { --duration-fast: 0ms; --duration-base: 0ms; }
}
```

No animated transitions on anything that displays a probability. A number that
counts up is a number a reader cannot copy mid-animation.

### 1.5 Layout

```css
:root {
  --sidebar-width:           240px;
  --sidebar-width-collapsed: 64px;
  --topbar-height:           52px;
  --content-max:             1440px;
  --content-gutter:          32px;
}
```

The content column is fluid up to `--content-max` and centred; it never
leaves a dead gutter on one side.

---

## 2. Compartment palette

The palette is a **fixed dictionary**, not a cycling scale. `Mitochondrion` is
`#B23A48` on the result page, in the heatmap legend, in the batch CSV's color
column, and in every paper figure. A compartment never changes color because it
appeared in a different position in a different prediction.

### 2.1 Eukaryotic compartments

| # | Code | Compartment | Token | Light fill | Light stroke/text | Dark fill | Dark text | Pattern |
|---|------|-------------|-------|-----------|-------------------|-----------|-----------|---------|
| 1 | CYT | Cytoplasm | `--loc-cyt` | `#E7EAEE` | `#4A5766` | `#333C46` | `#C4CCD6` | solid |
| 2 | NUC | Nucleus | `--loc-nuc` | `#E6E6F5` | `#3D3C8C` | `#2E2D5E` | `#BFBEE8` | dots |
| 3 | EXT | Extracellular | `--loc-ext` | `#DDF1EB` | `#0B6E5A` | `#0E4438` | `#8FD9C6` | diag-45 |
| 4 | MEM | Cell membrane | `--loc-mem` | `#F7EBD6` | `#8A5A11` | `#4A340C` | `#E7BE79` | horiz |
| 5 | MIT | Mitochondrion | `--loc-mit` | `#F7E3E5` | `#8E2C38` | `#4E1A20` | `#E9A7AE` | crosshatch |
| 6 | PLA | Plastid / chloroplast | `--loc-pla` | `#E6F0DC` | `#3D6E1A` | `#22400E` | `#AFD08C` | diag-135 |
| 7 | ER | Endoplasmic reticulum | `--loc-er` | `#EFE4F7` | `#6B3B95` | `#3A2152` | `#CFAEE6` | vert |
| 8 | LYS | Lysosome / vacuole | `--loc-lys` | `#F8E7DB` | `#96491F` | `#502713` | `#EBB68F` | dense-dots |
| 9 | GOL | Golgi apparatus | `--loc-gol` | `#F7E4EE` | `#8F3960` | `#4E1F34` | `#E6A8C6` | dashes |
| 10 | PER | Peroxisome | `--loc-per` | `#EFE8DF` | `#5E4630` | `#33261A` | `#C7B096` | checker |

### 2.2 Prokaryotic compartments

Shared compartments reuse the eukaryotic token — cytoplasm is cytoplasm.

| # | Code | Compartment | Token | Light fill | Light stroke/text | Dark fill | Dark text | Pattern |
|---|------|-------------|-------|-----------|-------------------|-----------|-----------|---------|
| 1 | CYT | Cytoplasm | `--loc-cyt` | *(as above)* | | | | solid |
| 2 | PCM | Cytoplasmic membrane | `--loc-mem` | *(as above)* | | | | horiz |
| 3 | PPL | Periplasm | `--loc-ppl` | `#E0EEF6` | `#1C6086` | `#0F3346` | `#9BCBE4` | diag-45 |
| 4 | POM | Outer membrane | `--loc-pom` | `#F2E9D5` | `#6E4C0C` | `#3C2A07` | `#D6BC84` | horiz-thick |
| 5 | PCW | Cell wall / surface | `--loc-pcw` | `#EDEFD9` | `#565F16` | `#2F340C` | `#C3CB8B` | vert |
| 6 | EXT | Extracellular | `--loc-ext` | *(as above)* | | | | diag-45 |

### 2.3 Membrane type

Membrane type is an attribute of a localization, not a location. It is encoded
as a **shape modifier on the compartment chip**, never as a new hue:

| Type | Chip treatment |
|------|----------------|
| Soluble | plain rounded chip |
| Peripheral | chip with a 1px inner offset ring |
| Transmembrane | chip with a 3px solid left cap |
| Lipid-anchored | chip with a 3px dotted left cap |

### 2.4 Signal and transit peptides

These are *regions of the sequence*, so they are rendered as underlays on the
sequence viewer rather than as chips. Each uses the compartment color of its
destination, at 35% alpha, with its pattern:

| Signal | Destination color | Region label |
|--------|-------------------|--------------|
| Signal peptide (SP) | `--loc-ext` | `SP 1–22` |
| Mitochondrial transit (mTP) | `--loc-mit` | `mTP 1–34` |
| Chloroplast transit (cTP) | `--loc-pla` | `cTP 1–58` |
| Thylakoid luminal (lTP) | `--loc-pla` | `lTP 59–81` |

The **cleavage site** is a single 2px vertical rule in `--text-primary` spanning
the full residue-row height, with the position number in mono beneath it. It is
achromatic on purpose: it is a coordinate, not a category.

### 2.5 Rules for using the palette

1. Never introduce a hue that is not in this table. A new compartment requires a
   new row here first.
2. Never reorder. The `#` column is the canonical sort order for legends,
   stacked bars, tables and figure keys.
3. In any figure with more than three compartments, the pattern fill is
   **mandatory**, not decorative.
4. Grayscale test: print the figure in grayscale. If two compartments become
   indistinguishable, the pattern is doing the work — verify it is present.

---

## 3. Confidence, abstention and consistency — the trust language

This is the part of the design system that carries the paper's actual claims. A
reader must be able to distinguish, without reading prose: *this is a confident
call*, *this is a hedge*, *the model declined*, and *the system caught itself
contradicting itself*.

### 3.1 What a probability looks like

Every predicted label renders as a **confidence row**:

```
[CHIP]  Mitochondrion            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░   0.87
 MIT                             calibrated · τ = 0.55
```

- The chip carries the compartment color, code and pattern.
- The bar **track** is `--surface-2`; the bar **fill** is the compartment's
  light-stroke color. Bar geometry is identical for every task — one 100%-width
  track, so bars are visually comparable down a column.
- The numeral is mono, two decimals, always shown. The bar is an aid; the number
  is the datum.
- The caption states that the value is calibrated and names the frozen
  abstention threshold τ for this registry version.

**Confidence bands do not use hue.** Hue is compartment. Bands are encoded by
fill opacity and an explicit word:

| Band | Condition | Bar fill | Caption word |
|------|-----------|----------|--------------|
| High | p ≥ 0.80 | 100% opacity | `high` |
| Moderate | 0.80 > p ≥ τ | 70% opacity | `moderate` |
| Below threshold | p < τ | 35% opacity, hatched | `below threshold` |

τ is read from the registry's calibrator metadata at render time. It is never a
CSS constant and never a literal in component code. If a build hardcodes 0.5,
that is a bug.

### 3.2 What abstention looks like

Abstention is a **first-class answer**, not an empty state, and must never look
like a loading skeleton or an error.

```css
.confidence-row[data-abstained="true"] {
  --bar-fill: var(--surface-2);
  background-image: repeating-linear-gradient(
    45deg, transparent 0 4px, var(--border-strong) 4px 5px
  );
}
.chip[data-abstained="true"] {
  border-style: dashed;
  filter: saturate(0.3);
}
```

Rendered:

```
[chip]  Uncertain                ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱   —
 ---    top candidate: Mitochondrion (0.41)
        below abstention threshold τ = 0.55 · coverage mode: strict
```

Three requirements:

1. The word **Uncertain** appears. Not "N/A", not "—" alone, not a blank cell.
2. The top candidate and its calibrated probability are still shown, in
   `--text-secondary`. Abstaining is refusing to *assert*, not refusing to
   *inform*.
3. The threshold and coverage mode are named, so the reader can reproduce the
   decision.

In batch exports, an abstention is the literal string `ABSTAIN` in the label
column, with the probability preserved in the score column. Never an empty cell
— empty cells are indistinguishable from pipeline failures.

### 3.3 What a consistency violation looks like

Three severities, and only here does the system use conventional semantic color.
These are the only three saturated non-compartment colors in the entire product.

```css
:root {
  /* *-ink is the text colour on the matching *-bg; the bare token is the
     3px rule and icon. Amber 11 is 4.3:1 on amber 3, under the AA floor, so
     warning text uses amber 12. */
  --state-note:      #63635E;  --state-note-ink:  #63635E;  --state-note-bg:  #F1F0EF;  /* sand 11 / 11 / 3 */
  --state-warn:      #AB6400;  --state-warn-ink:  #4F3422;  --state-warn-bg:  #FFF7C2;  /* amber 11 / 12 / 3 */
  --state-block:     #CE2C31;  --state-block-ink: #CE2C31;  --state-block-bg: #FEEBEC;  /* red 11 / 11 / 3 */
}
:root[data-theme="dark"] {
  --state-note:      #B5B3AD;  --state-note-ink:  #B5B3AD;  --state-note-bg:  #222221;  /* sand dark 11 / 11 / 3 */
  --state-warn:      #FFCA16;  --state-warn-ink:  #FFCA16;  --state-warn-bg:  #302008;  /* amber dark 11 / 11 / 3 */
  --state-block:     #FF9592;  --state-block-ink: #FF9592;  --state-block-bg: #3B1219;  /* red dark 11 / 11 / 3 */
}
```

| Severity | Meaning | Effect on the prediction | Treatment |
|----------|---------|--------------------------|-----------|
| `note` | Rule fired, biologically tolerable | none | 3px left rule, `--state-note` |
| `warning` | Rule fired, prediction is suspect | none — flagged only | 3px left rule, `--state-warn`, warn background |
| `blocking` | Rule fired, joint prediction is incoherent | forced abstention on the affected task | 3px left rule, `--state-block`, block background |

Every violation card is required to render five fields — this is a contract, not
a style suggestion:

```
[ warning ]  secreted_requires_signal                     rules v1.2.0
             Localization = Extracellular (0.81) but
             signal peptide = absent (0.12).
             Affected: localization, signal_peptide
             Action taken: flagged — prediction unchanged
```

`rule id` · `plain-language explanation` · `affected tasks` · `action taken` ·
`rule set version`. A violation that does not say what the system *did* is worse
than no violation display at all, because the reader cannot tell whether the
number above it was modified.

When **zero** rules fire, the panel is not hidden. It states:
`12 rules checked · 0 violations · rules v1.2.0`. A silent panel is
indistinguishable from a panel that failed to run.

### 3.4 What a pending prediction looks like

ADR-0001 makes the API asynchronous: every prediction is a job. So the result page has a
state that is **neither an answer nor a failure** — the work is queued or running.

This is a fourth trust state and it must not borrow the vocabulary of the other three. In
particular it must never look like abstention: *"the model declined to assert"* and
*"the model has not run yet"* are opposite claims, and a user who confuses them will
read an empty page as a scientific result.

| State | Meaning | Bar treatment | Word shown |
|---|---|---|---|
| `queued` | Accepted, not started | empty track, no fill, no hatch | `Queued` |
| `running` | In progress | indeterminate shimmer along the track | `Running` |
| `completed` | Result available | normal confidence row (§3.1) | the prediction |
| `failed` | Job errored | track replaced by error rule | `Failed` |
| `cancelled` | User cancelled | track dimmed | `Cancelled` |

```css
.confidence-row[data-state="queued"] .bar-track { background: var(--surface-2); }

.confidence-row[data-state="running"] .bar-track {
  background: linear-gradient(90deg,
    var(--surface-2) 0%, var(--accent-subtle) 50%, var(--surface-2) 100%);
  background-size: 200% 100%;
  animation: shimmer 1.4s linear infinite;
}
@keyframes shimmer { to { background-position: -200% 0; } }

@media (prefers-reduced-motion: reduce) {
  .confidence-row[data-state="running"] .bar-track { animation: none; }
}
```

Four requirements:

1. **No hatching.** Diagonal hatch means abstention (§3.2) and nothing else. A pending
   row uses an empty or shimmering track, never the hatch pattern.
2. **The compartment chip is absent, not greyed.** In a pending state no label has been
   predicted, so no chip is shown at all. A desaturated chip means abstention; an absent
   chip means no answer yet.
3. **Show position and estimate, marked as approximate:**
   `Queued · position 3 · about 24 s` — in `--text-secondary`, with a hedging word
   (`about`, `~`) always present. `estimated_wait_seconds` is an estimate from queue
   depth and measured throughput, never a promise.
4. **The shimmer is decoration, not information.** Progress must also be legible as text,
   because the animation is disabled under `prefers-reduced-motion` and is invisible to a
   screen reader. Batch jobs announce `completed / total` via `aria-live="polite"`,
   throttled to at most one announcement every 10 seconds so a 20,000-sequence job does
   not flood a screen reader.

**Failure is not silence.** A `failed` job states which stage failed and whether it can be
resubmitted. A `410 Gone` past the retention window says the result expired and offers
resubmission — it never renders as an empty or zero-confidence result.

### 3.5 What evidence looks like

Homology hits and GO terms are **visually subordinate by design**, because the
architecture forbids them from influencing the score and the UI must not imply
otherwise.

- No fills. No bars. No colored chips. Hairline `--border` only.
- Body text at `--text-secondary`, identifiers in mono.
- A persistent caption on the evidence panel:
  `Independent evidence — not an input to the predictions above.`
- If the evidence service is unavailable, the panel renders a degraded state
  (`Evidence unavailable — predictions above are unaffected`) and the rest of
  the page is unchanged. Evidence failure must never block a result.

---

## 3.6 Application shell and shared components (v0.2, ADR-0020)

Implemented in `frontend/src/styles/{shell,app,pages}.css` and `components/ui.tsx`.

- **Shell.** Grouped sidebar (Home · Analyze · Explore · Model), collapsible to icons; a top bar
  with breadcrumb, command palette (⌘K / Ctrl+K), backend status and theme. Below 768 px the
  sidebar is a drawer. Routes are URL hashes.
- **One primary action per screen**, drawn in `--action-bg` (ink, not a hue). Secondary
  buttons are bordered surface; ghost buttons have no border. Heights 28 / 32 / 40 px.
- **Inputs** are 32 px, `--radius-md`, surface-1 with a `--border-strong` edge; focus adds the
  focus ring. Selects are native with a drawn chevron.
- **Segmented controls** choose one of a few visible options (organism group, explanation
  method). Groups no trained head serves are disabled and say why.
- **Tables** sit in a bordered wrapper with a surface-2 header row, tabular numerals and
  right-aligned numbers. A clickable row is keyboard-operable.
- **Empty states** name what the place is for and the one thing to do next; **skeletons**
  stand in while a request is in flight and never look like abstention (§3.2, §3.4).
- **Result page order:** input · result header · five-cell summary · per-residue track beside the
  3D backbone · detail cards · protein space · tabs (consistency, engines, export) · provenance.
- **3D views** are hand-drawn canvases with one control set (drag, arrows, wheel, +/−, 0) and an
  idle spin that never runs under reduced motion. 3D is used only where position carries meaning
  (a backbone, a representation space); a number to be compared is never drawn in 3D.

---

## 4. Focus, contrast and the accessibility floor

- All body text meets WCAG AA (4.5:1); text ≥18px meets 3:1. Every
  compartment's light-stroke color has been chosen to clear 4.5:1 on its own
  light fill and on `--surface-1`.
- Focus is visible on every interactive element: `outline: 2px solid
  var(--focus-ring); outline-offset: 2px`. Never removed, never replaced by a
  color change alone.
- The sequence viewer is keyboard navigable: arrow keys move residue focus,
  `Home`/`End` jump to termini, and each focused residue announces
  `position, residue, task, attribution score`.
- Attribution heatmaps pair intensity with a numeric readout on focus — a
  heatmap alone is not accessible to a screen reader.
- `prefers-reduced-motion` disables all transitions.

---

## 5. Compliance

A component is considered conformant when:

1. It references no hex value that is not defined in §1 or §2.
2. Every compartment it renders carries color **and** code **and** pattern.
3. Every probability it renders is accompanied by the numeral and the frozen τ.
4. Every abstention says the word "Uncertain" and names the threshold, and no pending,
   failed or expired state is ever rendered using abstention's hatch or a chip.
5. Every violation names its rule id, affected tasks, action taken and rule set
   version.
6. It renders correctly in both themes and passes the grayscale test.

Deviations require an ADR.
