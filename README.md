# ProtLoc Fusion

One protein sequence in, one organism-group selection in — seven annotations out, from a
single shared protein-language-model representation.

Today a researcher answers five related questions about the same sequence by running five
separately published tools, each with its own input format, and reconciling the answers by
hand. ProtLoc Fusion answers them together, from one representation, with calibrated
probabilities, explicit abstention, biological-consistency checks, per-residue explanation
and independent homology evidence.

Delivered as a **web server and HTTP API with whole-proteome batch mode** — not a library,
not a CLI.

**Status:** Phase 1 — scaffold and source of truth. The repository structure, linting,
type checking, architectural import rules and CI are in place. No prediction logic exists
yet; it arrives in Phases 3–12. See `## Current phase` in `CLAUDE.md`.

---

## Outputs

| # | Output | Form |
|---|---|---|
| 1 | Subcellular location | Multi-label, calibrated probabilities |
| 2 | Membrane type | Categorical |
| 3 | N-terminal sorting signal | Presence + type |
| 4 | Cleavage site | Residue position |
| 5 | GO molecular function | Ranked terms |
| 6 | Homology evidence | Top hits against a local Swiss-Prot snapshot |
| 7 | Per-residue explanation | Attribution over the sequence |

Every response carries `model_version`, `config_hash`, `rules_version` and
`experiment_id`. Reproducibility is an API guarantee, not a footer decoration.

---

## Repository layout

```
frontend/         React 18 + Vite + TypeScript SPA — a separate deployable
backend/protloc/  17 modules, layered L-1 (domain) → L5 (evidence)
configs/          Hydra config tree
data/             DVC-tracked; raw bytes are never committed
registry/         Versioned bundles: checkpoint + calibrator + rules + config
experiments/      EXP-NNNN run manifests
benchmarks/       External tool outputs + label_mapping.yaml
scripts/          One job per script, all CLI-invocable
docs/             DESIGN, THIRD_PARTY, EXTERNAL_WORKFLOW, adr/, phases/, handover/
deployment/       Dockerfile.serve (CPU) · Dockerfile.train (GPU) · compose
```

`FOLDER_STRUCTURE.md` gives the full tree and the reasoning behind each boundary.

### The layering, in one paragraph

`domain/` imports nothing internal and holds the dataclasses, enums and port protocols
everything else depends on. `api/` parses, validates, delegates and returns — no business
logic. `inference/` owns the pipeline *order* and none of the algorithms. `training/` is
never imported by anything in the serving path, and `Dockerfile.serve` does not install
training dependencies, so a violation breaks the build rather than merely the convention.
`backend/tests/test_layering.py` asserts all of this by walking the import AST.

---

## Requirements

| Tool | Version |
|---|---|
| Python | 3.11 |
| Node.js | ≥ 20 |
| Docker | any recent release |

---

## Running it

```bash
# Backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e "backend[dev]"
cp .env.example .env                                 # then fill the Phase 1 block
uvicorn protloc.api.app:app --reload                 # available from Phase 2

# Frontend
cd frontend && npm ci && npm run dev

# Both, containerised
docker compose -f deployment/docker-compose.yml up --build
```

## Checks

```bash
ruff check backend && ruff format --check backend
cd backend && mypy protloc/domain protloc/config && python -m pytest tests && cd ..
cd frontend && npm run lint && npx tsc --noEmit && npm run build
python scripts/generate_tokens.py && git diff --exit-code   # tokens.css is generated
```

CI runs every one of these on each push. `tokens-current` catches a hand-edited token
file; `test_layering` catches an architecture violation.

---

## Conventions worth knowing before your first commit

- **No placeholder code.** A file that exists is complete for its current scope.
- **`frontend/src/styles/tokens.css` is generated** from the tables in `docs/DESIGN.md` by
  `scripts/generate_tokens.py`. Never hand-edit it. No raw colour literal appears anywhere
  under `frontend/src/` — ESLint rejects it.
- **React only.** No CSS framework, no component library, no UI runtime beyond React.
  Styling is plain CSS reading the generated tokens.
- **No inline biology.** Consistency rules live in `backend/rules/consistency_rules.yaml`;
  thresholds live in the registry bundle. A biological constant in a `.py` file is a bug.
- **No hardcoded τ.** The abstention threshold is read from calibrator metadata at runtime.
- **Secrets never leave `.env`.** `.env.example` carries `ENTER_YOUR_VALUE_HERE` only.
- **Conventional Commits**, SemVer, and an ADR before any irreversible decision.

---

## Documentation

| Question | Document |
|---|---|
| What is this project and why | `PROJECT_PLAN.md` |
| The execution contract and coding rules | `CLAUDE.md` |
| Where does a file go | `FOLDER_STRUCTURE.md` |
| How is a phase run | `PHASE_PROTOCOL.md` |
| What does it look like | `docs/DESIGN.md` |
| Why was X decided | `docs/adr/` |
| Can we use this dependency | `docs/THIRD_PARTY.md` |

---

## Licence

Apache License 2.0 — see `LICENSE`.

Some tools ProtLoc Fusion is benchmarked against carry non-commercial or copyleft terms
(`docs/THIRD_PARTY.md` marks them **restricted**). Those repositories are **read, never
copied**: never vendored, never bundled into an image, never served. Their weights are not
redistributed. Nothing under a restricted licence contributes source to this repository,
which is what keeps the Apache-2.0 grant above accurate.
