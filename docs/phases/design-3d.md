# Design — 3D views: protein-space map and backbone upgrade

ADR-0020 (no new dependency) · 2026-10-24 · after D1–D7

## Protein space (new)

- **Data.** `scripts/build_space.py` (numpy, offline) reads the 500-sequence embedding cache
  (mean-pooled ESM-2 150M, 640-d), `data/gallery/catalog.json` and the label-to-code map in
  `configs/serving/engine.json`. It writes `data/space/space.json` (PCA to three components by
  SVD, signs fixed, rounded; eight nearest neighbours per sequence by cosine of the centred
  vectors in all 640 dimensions) and `data/space/manifest.json` (its SHA-256). `--check` exits 1
  when the file is stale. Measured on this cache: 500 points; PC1–3 explain 23.9 / 11.6 / 6.8% of
  variance; the nearest neighbour shares a Swiss-Prot location for 64% of sequences; median
  nearest-neighbour cosine 0.71.
- **Serving.** `protloc/inference/space.py` (stdlib only) serves the document only when its bytes
  match the manifest. `GET /demo/space`; new error codes `space_not_built` (404) and
  `space_refused` (500), both non-retryable. Orientation only: never an input to a score.
- **Frontend.** `components/ProteinSpace.tsx`: hand-drawn canvas point cloud in a reference box,
  colour = first Swiss-Prot location (ring = second), compartment filter chips with counts in
  DESIGN.md order, hover card, click to open in Predict, idle rotation off under reduced motion.
  Each axis is scaled to its own 97th-percentile spread (`services/spaceMath.ts`) so outliers do
  not squash the cloud; the caption says so and gives each component's variance share.
  - **Predict:** a "Protein space" card highlights the result with dashed lines to its eight
    neighbours, and lists them (accession, locations, similarity) as keyboard buttons.
  - **New screen** `#/space` (Explore → Protein space): the whole map, 600 px.

## Backbone viewer (upgraded)

- Drawn as a depth-sorted tube: runs of four residues, each with an outline in the card colour
  so crossings read as over and under; thicker and more opaque toward the viewer.
- Signal peptide shown as a halo in the extracellular colour (DESIGN.md §2.4) and the cleavage
  site as a labelled ring; hover card with residue and pLDDT.
- Moved out of the tab into a two-column row beside the per-residue track, so the cursor link
  between the two is visible.
- `components/useOrbit.ts` holds the shared controls (drag, keys, wheel, reset, idle spin that
  stops at first interaction).
- **Fix:** explanation colouring for an unsigned track used an upper-case token name
  (`--loc-EXT-ink`), which does not exist, so it silently drew in the text colour.

## Verification

| Check | Result |
|---|---|
| Backend: `pytest tests/unit/test_space.py test_errors.py test_structure_route.py test_api_errors.py test_structures.py` (Python 3.11, cloud copy) | pass; 8 new tests |
| `ruff check`, `ruff format --check`, `mypy --strict protloc/inference/space.py` | clean |
| `python scripts/build_space.py --check` on this machine | current, `ff58725c9cb4` |
| Frontend `tsc`, `eslint --max-warnings 0`, `prettier --check`, `vitest` | clean; 24 files, 156 tests |
| Visual review in Chromium | the real `space.json`; a synthetic helix stood in for an AlphaFold trace |

## UNVERIFIED

- The backbone view against real AlphaFold traces, and `/demo/space` against the live backend.
- The full backend suite and `test_layering.py` on Windows (only the five files above were run).
- `data/` is gitignored: run `python scripts/build_space.py` on any machine that serves the app.
