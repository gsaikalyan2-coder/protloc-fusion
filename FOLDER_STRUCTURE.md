# FOLDER_STRUCTURE.md — ProtLoc Fusion

The canonical repository tree. Two top-level application folders — `frontend/` and
`backend/` — plus the artifact, documentation and infrastructure directories that make
the project reproducible.

This tree reconciles two things: the blueprint's 17-module hexagonal architecture
(L0→L5) and a conventional full-stack split. The 17 modules live inside
`backend/protloc/`; the layering is enforced by import rules, not by directory distance.

---

## The tree

```
ProtLoc-Fusion/
│
├── frontend/                      # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── components/            # Presentational components; no data fetching
│   │   │   ├── sequence/          # Residue grid, attribution heatmap, region underlays
│   │   │   ├── confidence/        # Confidence row, chip, abstention state
│   │   │   ├── consistency/       # Violation cards, rules-checked summary
│   │   │   ├── evidence/          # Homology table, GO list (visually subordinate)
│   │   │   └── ui/                # Buttons, inputs, panels — the design-system primitives
│   │   ├── pages/                 # Route-level screens: Landing, Predict, Result, Batch, Job
│   │   ├── hooks/                 # useePrediction, useJob, useTheme — stateful logic
│   │   ├── services/              # API client; the ONLY place fetch() appears
│   │   ├── types/                 # api.generated.ts is GENERATED from docs/api/openapi.yaml;
│   │   │                          # api.ts names those types and holds the runtime helpers
│   │   ├── styles/                # tokens.css — generated from DESIGN.md, never hand-edited
│   │   └── assets/
│   ├── public/
│   ├── tests/                     # Component + e2e (Playwright)
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── eslint.config.js
│   └── package.json
│
├── backend/                       # FastAPI service + the ML core
│   ├── protloc/
│   │   │                          # ── L-1: dependency root ──
│   │   ├── domain/                # Dataclasses, enums, port protocols. Imports NOTHING internal
│   │   │
│   │   │                          # ── L0: delivery ──
│   │   ├── api/                   # HTTP routers. Thin: parse, validate, delegate, return
│   │   ├── web/                   # Server-rendered fallbacks / static mounting
│   │   ├── jobs/                  # Proteome-scale runs: chunking, progress, cancel, resume
│   │   │
│   │   │                          # ── L1: application ──
│   │   ├── inference/             # The conductor. Owns the pipeline ORDER, owns no algorithms
│   │   ├── config/                # Single source of truth for paths, hyperparams, settings
│   │   │
│   │   │                          # ── L2: features ──
│   │   ├── features/              # ESM-2 adapter, classical features, k-mers, embedding cache
│   │   │
│   │   │                          # ── L3: model core ──
│   │   ├── models/                # Shared trunk, 7 task heads, classical baselines
│   │   ├── training/              # Train loops, LoRA config. NEVER imported by inference/
│   │   │
│   │   │                          # ── L4: trust ──
│   │   ├── calibration/           # netcal wrappers; abstention policy (ours)
│   │   ├── consistency/           # YAML rule engine, severity handling
│   │   │
│   │   │                          # ── L5: evidence ──
│   │   ├── interpretability/      # Per-residue attribution
│   │   ├── evidence/              # Homology + GO lookup, circuit-broken
│   │   │
│   │   │                          # ── support ──
│   │   ├── data/                  # Corpus building, splits, leakage assertions
│   │   ├── registry/              # Atomic bind: checkpoint + calibrator + rules + config
│   │   ├── evaluation/            # Benchmarks — calls the SERVED path, not the model
│   │   └── utils/                 # Logging, seeding, hashing. No business logic
│   │
│   ├── rules/                     # consistency_rules.yaml — versioned biological constraints
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── golden/                # The §23.1 golden protein set + expected behavior
│   ├── pyproject.toml
│   └── ruff.toml
│
├── configs/                       # Hydra config tree — composable, never forked scripts
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── deployment/
│
├── data/                          # DVC-tracked pointers. Raw data is NEVER committed
│   ├── raw/                       # Immutable download snapshots
│   ├── interim/
│   ├── processed/                 # The frozen corpus
│   └── manifests/                 # dataset manifest + SHA-256 + dataset card
│
├── registry/                      # Versioned model bundles — the deployment unit
│   └── protloc-fusion/v1.0.0/
│       ├── model.pt
│       ├── calibrator.pkl
│       ├── consistency.yaml
│       ├── config.yaml
│       ├── metadata.json
│       └── checksums.txt
│
├── experiments/                   # Tracked runs: EXP-0001/ with manifest + metrics
├── benchmarks/                    # External tool outputs + label_mapping.yaml
├── baselines/                     # Classical + frozen-PLM baseline definitions
├── notebooks/                     # Exploration ONLY — nothing production depends on these
│
├── scripts/                       # One job per script, all CLI-invocable
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── validate_sequences.py
│   ├── cluster_sequences.py       # GraphPart wrapper
│   ├── build_splits.py
│   ├── generate_embeddings.py
│   ├── train.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── benchmark_api.py
│   ├── benchmark_external.py
│   ├── package_registry.py
│   ├── export_openapi.py          # app -> docs/api/openapi.yaml; --check exits 1 if stale
│   └── run_batch.py
│
├── docs/
│   ├── DESIGN.md                  # ✓ written
│   ├── ARCHITECTURE.md            # ✓ written (Phase 2)
│   ├── api/
│   │   └── openapi.yaml           # ✓ GENERATED by scripts/export_openapi.py. The frozen
│   │                              #   contract; CI regenerates it and fails on a diff
│   ├── THIRD_PARTY.md             # ✓ written
│   ├── EXTERNAL_WORKFLOW.md       # ✓ written
│   ├── CONTEXT_MEMORY.md          # ✓ written
│   ├── CITATIONS.md
│   ├── adr/                       # ADR-0001.md … one per irreversible decision
│   ├── diagrams/                  # Request sequence, data lineage
│   │   └── wireframes/            # ✓ The five screens, as markdown (Phase 2)
│   └── handover/                  # handover_phase_X.txt
│
├── deployment/
│   ├── Dockerfile.serve           # CPU serving image — no training deps
│   ├── Dockerfile.train           # GPU training image — includes EMBOSS for GraphPart
│   ├── docker-compose.yml
│   └── infra/                     # IaC — provider [CLIENT-TBD]
│
├── .github/workflows/
│   ├── ci.yml                     # lint · type-check · unit · integration · eval gate
│   └── deploy.yml
│
├── .env.example                   # Placeholders ONLY
├── CLAUDE.md                      # Execution contract — read before every phase
├── PROJECT_PLAN.md
├── FOLDER_STRUCTURE.md
└── README.md
```

---

## Why the boundaries fall where they do

**`frontend/` and `backend/` are separate deployables, not a cosmetic split.** They build
independently, ship as different artifacts, and communicate only over the documented HTTP
contract. The frontend's TypeScript types are *generated* from the backend's OpenAPI spec,
so a backend schema change that breaks the frontend fails at build time rather than in a
user's browser.

**`domain/` imports nothing.** It holds dataclasses, enums and port protocols and sits
below every other module. This is what makes the rest swappable: `evidence/` can be
replaced with a stub in tests because everything above it depends on the protocol in
`domain/`, not on the implementation.

**Routers are thin — this is the rule most often broken.** `api/` parses the request,
validates its shape, calls `inference/`, and returns. No feature extraction, no
thresholding, no rule evaluation. If you can delete `api/` and still run a full prediction
from a Python function, the boundary is correct.

**`inference/` owns order, not algorithms.** It is the one place that knows the pipeline
runs features → model → calibration → consistency → interpretability → evidence. Each of
those modules knows only its own job. This is why the order is auditable and why the
sequence diagram in Phase 2 is worth drawing.

**`training/` and `inference/` never touch.** They share `models/` and `features/`, which
is exactly what prevents train/serve skew — one feature code path, two lifecycles. A
training import appearing in the serving path is a CI failure, not a style opinion, and
`Dockerfile.serve` does not install training dependencies, so the mistake breaks the build.

**`evaluation/` calls the served path.** It is placed inside `backend/protloc/` but it
speaks to the API, not to model objects. This is deliberate and it is the blueprint's §3.3
requirement: the numbers in the paper must be the numbers a user receives.

**`registry/` is the deployment unit.** A checkpoint alone is not deployable. A checkpoint
plus the calibrator fitted for it plus the consistency rules it was validated against, all
checksummed together, is. Rollback is repointing `MODEL_VERSION` at the previous directory.

**`data/` is versioned, never committed.** Raw UniProt snapshots are large and immutable;
DVC pointers go in git, bytes do not. `manifests/` holds the hashes that make a run
reproducible.

**`notebooks/` is a dead end by design.** Exploration lives there and nothing imports from
it. The moment production logic appears in a notebook, reproducibility is gone.

**`rules/consistency_rules.yaml` sits outside the Python package.** Biological constraints
change when biology says so, not when code ships. Keeping them as data — versioned,
reviewable by someone who does not read Python — is the entire point of the consistency
engine.

---

## Naming conventions

| Kind | Convention | Example |
|---|---|---|
| Python modules/packages | `snake_case` | `backend/protloc/interpretability/` |
| Python classes | `PascalCase` | `ConsistencyEngine` |
| React components | `PascalCase.tsx` | `ConfidenceRow.tsx` |
| React hooks | `useCamelCase.ts` | `usePrediction.ts` |
| CSS custom properties | `--kebab-case` | `--loc-mit` |
| Config files | `snake_case.yaml` | `homology_v1.yaml` |
| Experiments | `EXP-NNNN` | `EXP-0031` |
| Registry versions | SemVer | `v1.0.0` |
| ADRs | `ADR-NNNN-kebab-title.md` | `ADR-0002-graphpart-threshold.md` |
| Branches | `type/short-description` | `feat/localization-head` |
| Commits | Conventional Commits | `feat(models): add shared trunk` |
