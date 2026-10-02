# ProtLoc Fusion — presentation demo (standalone)

A working, minimal five-screen app for tomorrow's presentation. It is **not** the
Phase-12 frontend and **not** the paper model. See `OVERRIDE.md` for isolation and
removal — the short version: `/demo/*` routes, ports 8010/5174, delete the folder to undo.

## Run it (two terminals)

> **Install into a venv, not your global Python.** The demo's deps are declared with
> permissive lower bounds (they are satisfied by `protloc`'s own pins, so nothing gets
> downgraded), but a venv keeps this folder from touching the real project's environment
> at all. If a global install already downgraded things, see "Repair" below.

```bash
# terminal 1 — API on :8010
cd demo/backend
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --port 8010 --reload

# terminal 2 — SPA on :5174  (proxies /demo -> 8010)
cd demo/frontend
npm install
npm run dev          # open http://localhost:5174
```

Checks: `curl http://localhost:8010/demo/health` · `cd demo/backend && python -m pytest -q tests`
(6 tests) · `cd demo/frontend && npm run typecheck`.

Kill switch: `PROTLOC_DEMO_ENABLED=0 python -m uvicorn app.main:app --port 8010`.

### Repair (if the demo was pip-installed globally)

Restore the real project's pins in the global environment, then use the venv above:

```bash
python -m pip install "fastapi==0.141.1" "pydantic==2.13.5" "uvicorn[standard]==0.52.4"
```

The demo is verified against exactly those versions (all 6 tests pass), so this repair
costs the demo nothing. Note: `crewai` wants `pydantic<2.13` while `protloc` pins
`2.13.5` — that conflict is pre-existing between those two projects and unrelated to
this folder; keep them in separate environments.

## What the predictor actually is — say this out loud

Phases 5–9 have not run, so there is no trained model and no ESM-2 checkpoint here.
The demo backend is an **interpretable motif engine**: real Kyte-Doolittle hydropathy,
real signal-peptide h-region detection with an AxA cleavage search, TM-helix windows,
NLS / KDEL / PTS1 / LPXTG motifs, and transit-peptide composition — mapped to the
same seven outputs the real system will produce. Every number on screen is computed
live from the sequence, which is why mutating a residue changes the answer sensibly.
Reference and benchmark sequences are **synthetic and constructed with known labels**,
so nothing is passed off as UniProt data. The app says this in a banner on every screen.

## Novel vs. baseline

| Feature | Novel? | Why |
|---|---|---|
| Proteome Triage Board (attention ranking) | **novel** | Ranks *which proteins a human must review* — abstention, rule severity, missing homology, multi-compartment — instead of listing predictions |
| Dual-targeting landscape matrix | **novel** | Co-assignment counts across a batch: candidate dual-targeted proteins as a first-class view |
| Counterfactual Mutation Scanner | **novel** | Truncation series + full substitution scan + variant diff: "what would change the call", not just the call |
| Independent Evidence Ledger | **novel** | Model and homology shown as two separate verdicts with alignment coverage; disagreement is surfaced, never averaged away |
| Trust & Abstention Console | **novel** | Reliability table, ECE before/after, coverage vs. selective accuracy at every τ, with τ frozen on a calibration split and reported on a held-out split |
| Sequence → localization form | baseline | Exists conceptually; deliberately **not** the centre of this app. It lives inside the Inspector as one card among seven outputs |

## Screens

1. **Triage Board** — paste/generate FASTA → `POST /demo/batch` → job polling → ranked queue + KPIs + landscape matrix. Click a row to open it in the Inspector.
2. **Inspector** — one sequence, all seven outputs: calibrated multi-label localization with the τ marker, membrane type as chip modifiers, sorting signal + cleavage coordinate, GO motif hits, per-residue attribution/hydropathy track, consistency flags, homology verdict.
3. **Evidence Ledger** — model verdict vs. reference-hit verdict side by side; hit table (identity, k-mer score, coverage, evidence code); clicking a hit paints its aligned blocks onto the query.
4. **Mutation Scanner** — N-terminal truncation series (the signal-peptide cliff), every-position substitution scan coloured by Δ probability, top-10 most sensitive residues, and a wild-type vs variant diff from a mutation list like `K83A,L20P`.
5. **Trust Console** — `GET /demo/trust`: temperature, frozen τ, ECE before/after, reliability bins, and the coverage/selective-accuracy curve, plus the live consistency rule set.

## API (all under `/demo`, all async per ADR-0001)

`GET /demo/health` · `GET /demo/version` · `GET /demo/examples` · `GET /demo/reference` ·
`GET /demo/trust` · `POST /demo/predict` → **202** · `POST /demo/batch` → **202** ·
`POST /demo/scan` → **202** · `GET /demo/jobs/{id}` (results when `completed`, `410` after 1 h) ·
`POST /demo/jobs/{id}/cancel`. Every response carries `model_version`, `config_hash`,
`rules_version`, `experiment_id`.

## Files — what is what

```
demo/
├── README.md                  this file
├── OVERRIDE.md                isolation guarantees + how to switch off / delete
├── backend/
│   ├── requirements.txt       fastapi, uvicorn, pydantic, pytest
│   ├── .env.example           PROTLOC_DEMO_ENABLED, DEMO_MAX_BATCH, DEMO_CORS
│   ├── app/engine.py          sequence features + per-compartment scores + attribution
│   ├── app/data.py            synthetic reference set, gallery examples, labelled benchmark
│   ├── app/service.py         calibration fit (T, τ), predict(), scan(), triage, landscape, evidence
│   ├── app/rules.json         the six consistency rules (no biology hardcoded in .py)
│   ├── app/jobs.py            in-memory job queue: 202 + polling + cancel + retention
│   ├── app/main.py            FastAPI routes, kill switch, CORS
│   └── tests/test_smoke.py    6 tests incl. "cutting the SP kills the secreted call"
└── frontend/
    ├── vite.config.ts         port 5174, proxy /demo -> 8010
    ├── src/api.ts             the only place that fetches; useJob() implements 202-then-poll
    ├── src/App.tsx            sidebar shell, 5 screens, no router dependency
    ├── src/styles/tokens.css  DESIGN.md token names, dark-green surface values
    ├── src/styles/app.css     layout + components; no raw hex, tokens only
    ├── src/components/ui.tsx  Chip, ConfidenceRow, Flags, ProvenanceBar, Card, example picker
    ├── src/components/SequenceTrack.tsx  residue view: attribution / hydropathy / alignment
    └── src/pages/{Triage,Inspector,Evidence,Scanner,Trust}.tsx
```

## Presentation script (≈5 min)

- **Open on Triage.** "Five tools today answer five questions about one sequence, and a
  human reconciles them by hand. Start from the real workflow: I have 200 proteins,
  which ones do I have to look at myself?" Run the batch. Point at the attention column:
  top row is flagged *and* contradicted by homology; bottom rows are clean and can be
  trusted without opening them. "This ranking is the product — the per-protein page is a detail."
- **Landscape matrix.** "Off-diagonal cells are candidate dual-targeted proteins, which is
  a question you cannot even ask when a tool returns one label per protein."
- **Click the top row → Inspector.** "Seven annotations, one representation, one screen.
  Colour means compartment, never chrome. The bar is an aid; the number is the datum.
  The dashed line is τ — below it the system abstains instead of guessing."
  Toggle attribution → hydropathy: "the same call, explained at the residue."
  Point at the R01 flag: "the system caught itself: a nuclear call with a secretory
  signal peptide is biologically incoherent, and it says so rather than shipping it."
- **Evidence Ledger.** "The model and the homology evidence are computed on separate paths.
  Here they disagree, and the app shows both verdicts and the alignment coverage behind
  the hit. Averaging them would hide exactly the case a reviewer cares about."
- **Scanner.** Run it. "Truncating the N-terminus three residues at a time: the secreted
  probability collapses right where the cleavage site is — the model is using the signal,
  not the length. The substitution scan ranks the residues holding the call up; the top
  hits are the NLS lysines. Then a specific variant, wild type vs mutant, side by side."
- **Trust Console.** "What is a 0.9 worth here? Temperature and τ were fitted on the
  calibration split; everything shown is measured on a split that was never touched.
  ECE drops after calibration, and the curve is the honest trade: at τ = 0.67 we answer
  90% of cases at 83% selective accuracy. The benchmark carries 8% injected label noise,
  so 100% is unreachable by construction."
- **Close.** "Predictor is a demo-grade motif engine — Phases 5–9 replace it with the
  trained model behind the same API and the same screens. The plan is unchanged; this
  folder deletes in one command."
