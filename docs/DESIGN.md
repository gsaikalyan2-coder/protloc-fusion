# DESIGN.md — ProtLoc Fusion

Single source of truth for the visual system. Every UI surface, every exported
paper figure, and every generated diagram reads its colors, spacing and states
from this document. Nothing hardcodes a hex value that is not defined here.

Status: **v0.1 — proposed.** Token *names* are stable and may be depended on.
Token *values* may still change until v1.0 is tagged alongside the first model
registry release.

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

### 1.1 Neutrals and surfaces

```css
:root {
  /* Surfaces — page, card, inset */
  --surface-0:        #FBFAF8;
  --surface-1:        #FFFFFF;
  --surface-2:        #F4F2EE;

  /* Text */
  --text-primary:     #15130F;
  --text-secondary:   #5B564E;
  --text-muted:       #8A8479;
  --text-inverse:     #FBFAF8;

  /* Borders */
  --border:           #E4E0D8;
  --border-strong:    #CFC9BE;
  --border-stronger:  #A9A296;

  /* The single UI accent — links, focus rings, primary buttons.
     Deliberately desaturated so it never competes with a compartment. */
  --accent:           #3A3A6E;
  --accent-hover:     #2C2C57;
  --accent-subtle:    #ECECF4;
  --focus-ring:       #3A3A6E;
}

:root[data-theme="dark"] {
  --surface-0:        #121110;
  --surface-1:        #1B1A18;
  --surface-2:        #242220;

  --text-primary:     #F2EFE9;
  --text-secondary:   #B3AEA4;
  --text-muted:       #7E7970;
  --text-inverse:     #121110;

  --border:           #2E2C29;
  --border-strong:    #423F3A;
  --border-stronger:  #5E5A53;

  --accent:           #A9A9E0;
  --accent-hover:     #C0C0EC;
  --accent-subtle:    #24243A;
  --focus-ring:       #A9A9E0;
}
```

### 1.2 Type scale

Two families. One sans for interface, one mono for anything that is a residue,
an accession, a hash, or a coordinate.

```css
:root {
  --font-sans: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;

  --text-xs:   12px;   /* metadata, hashes, axis ticks */
  --text-sm:   13px;   /* table cells, captions, badge text */
  --text-base: 15px;   /* body */
  --text-lg:   18px;   /* section headings */
  --text-xl:   22px;   /* page title */
  --text-2xl:  28px;   /* landing hero only */

  --leading-tight: 1.25;
  --leading-body:  1.6;

  --weight-regular: 400;
  --weight-medium:  500;
}
```

Two weights only — 400 and 500. Bold-heavy interfaces read as marketing; this
is an instrument.

### 1.3 Space, radius, elevation

```css
:root {
  --space-1: 4px;   --space-2: 8px;   --space-3: 12px;
  --space-4: 16px;  --space-5: 24px;  --space-6: 32px;
  --space-7: 48px;  --space-8: 64px;

  --radius-sm: 4px;   /* badges, chips, inputs */
  --radius-md: 8px;   /* buttons, panels */
  --radius-lg: 12px;  /* cards */

  --border-width:      1px;
  --border-width-emph: 3px;   /* left rule on trust callouts */

  /* Elevation is expressed with borders, not shadows, except for
     genuinely floating layers (dropdown, drawer, modal). */
  --shadow-overlay: 0 8px 24px rgb(0 0 0 / 0.12);
}
```

### 1.4 Motion

```css
:root {
  --duration-fast: 120ms;
  --duration-base: 200ms;
  --ease: cubic-bezier(0.2, 0, 0.2, 1);
}
@media (prefers-reduced-motion: reduce) {
  :root { --duration-fast: 0ms; --duration-base: 0ms; }
}
```

No animated transitions on anything that displays a probability. A number that
counts up is a number a reader cannot copy mid-animation.

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
  --state-note:   #5B564E;  --state-note-bg:  #F4F2EE;
  --state-warn:   #8A5A11;  --state-warn-bg:  #F9F0DF;
  --state-block:  #A32D2D;  --state-block-bg: #FAE9E9;
}
:root[data-theme="dark"] {
  --state-note:   #B3AEA4;  --state-note-bg:  #242220;
  --state-warn:   #E7BE79;  --state-warn-bg:  #3B2C10;
  --state-block:  #F09595;  --state-block-bg: #3E1A1A;
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
