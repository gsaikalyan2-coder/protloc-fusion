# Contributing

ProtLoc Fusion is built one phase at a time. Read `CLAUDE.md` first; if it disagrees
with anything said elsewhere, it wins.

## Phase workflow

1. **Branch** `phase/NN-slug`, for example `phase/04-swissprot-download`.
2. **Commit** with a conventional-commit subject scoped to the phase:
   `type(phase-NN): subject` — for example `feat(phase-04): pin Swiss-Prot 2026_03`.
   `git log --oneline --grep "phase-04"` then reconstructs the phase. Types: `feat`,
   `fix`, `docs`, `chore`, `build`, `test`, `refactor`.
3. **Check** before committing (all must pass; CI runs the same):
   ```bash
   ruff check backend && ruff format --check backend
   (cd backend && python -m mypy protloc/domain protloc/config && python -m pytest tests)
   (cd frontend && npx eslint . --max-warnings 0 && npx prettier --check . && npx tsc --noEmit && npm run build)
   ```
4. **Release notes** go in `docs/phases/NN.md`: what was built, what was decided, what
   broke, what is deliberately left open.
5. **Tag** the phase with an annotated tag `v0.NN.0`:
   `git tag -a v0.NN.0 -m "Phase NN: <one line>"`.
6. **Push** is optional and is the only step that leaves the machine:
   `git push origin main --follow-tags`.

Do not start phase N+1 until phase N's acceptance test has been run and recorded.

## Ground rules

- Every prediction the backend returns comes from a trained model. The motif engine in
  `features/classical.py` is a feature branch and a fallback label, never the answer.
- No file under a restricted licence is copied in (`THIRD_PARTY_NOTICES.md`). No data or
  weights are committed (`docs/DATA_LICENSING.md`).
- Layering is enforced by `backend/tests/test_layering.py`. A new module is a layering
  decision and must be added to `LAYER_DEPTH`.
- Temporary lint and layering exemptions for the promoted MVP code are listed in
  `backend/ruff.toml`, `frontend/eslint.config.js` and `backend/tests/test_layering.py`.
  Each is deleted when its file is replaced. New code gets no exemptions.
- Line endings are LF everywhere (`.gitattributes`). On Windows, set
  `git config core.autocrlf true` and `git config --system core.longpaths true`.
