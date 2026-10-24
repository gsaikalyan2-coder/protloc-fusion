"""Assert the frontend dependency set against the React-only rule.

``CLAUDE.md`` states that the frontend is React + Vite + TypeScript with plain
CSS driven by generated tokens, and that no CSS framework, component library or
other UI runtime is added without an ADR. A rule of that shape is broken by
`npm install` and noticed at review, if at all, so it is checked here and in CI.

Two things are asserted:

* every declared dependency is on the allowlist below, and
* every version is pinned exactly — no ``^``, ``~``, ``*`` or range. An
  unpinned frontend is a scaffold that passes CI today and fails in three months
  on a transitive bump.

Adding a dependency means adding it here, in the same commit, with the reason it
belongs. That is the point: the list is a decision record, not a mirror of
package.json.

Run it from the repository root::

    python scripts/check_frontend_deps.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"

# An exact version: 1.2.3, optionally with a prerelease or build suffix.
_EXACT_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

# Runtime dependencies. React and its DOM renderer, plus what ADR-0020 admits:
# no CSS framework, no styled component library, no state, routing or styling runtime.
ALLOWED_DEPENDENCIES: dict[str, str] = {
    "react": "the UI runtime named in CLAUDE.md",
    "react-dom": "React's DOM renderer",
    "@fontsource/ibm-plex-sans": (
        "IBM Plex Sans font files, self-hosted (ADR-0020, docs/DESIGN.md §1.2). "
        "CSS @font-face and woff2 only; no script, renders no UI. SIL OFL 1.1."
    ),
    "@fontsource/ibm-plex-mono": (
        "IBM Plex Mono font files, self-hosted (ADR-0020, docs/DESIGN.md §1.2). "
        "CSS @font-face and woff2 only; no script, renders no UI. SIL OFL 1.1."
    ),
    "@radix-ui/react-dialog": (
        "unstyled, accessible modal dialog (focus trap, scroll lock, Escape) for the "
        "command palette (ADR-0020, design D2). Styled with our tokens. MIT."
    ),
    "@radix-ui/react-popover": (
        "unstyled, collision-aware popover for the backend status details (ADR-0020, "
        "design D2). Styled with our tokens. MIT."
    ),
    "@radix-ui/react-tabs": (
        "unstyled, accessible tabs (roving focus, arrow keys) for the secondary panels of "
        "a result (ADR-0020, design D5). Styled with our tokens. MIT."
    ),
    "@radix-ui/react-tooltip": (
        "unstyled hover-and-focus tooltip for provenance badges (ADR-0020, design D3). "
        "Styled with our tokens. MIT."
    ),
    "lucide-react": (
        "16 px stroke icons for navigation and actions only (ADR-0020, design D2). "
        "Tree-shaken SVG components; ISC."
    ),
}

# Build and check tooling. None of it ships to the browser.
ALLOWED_DEV_DEPENDENCIES: dict[str, str] = {
    "@eslint/js": "ESLint's own recommended rule set",
    "@testing-library/dom": (
        "peer of @testing-library/react: DOM queries for the Vitest suite. Test-only; "
        "it is never bundled and renders no UI in the product (Phase 16)."
    ),
    "@testing-library/react": (
        "renders components inside the Vitest suite so tests assert what a user "
        "sees. Test-only; never bundled (Phase 16)."
    ),
    "@types/node": "types for vite.config.ts, which runs in Node",
    "@types/react": "React type definitions",
    "@types/react-dom": "React DOM type definitions",
    "@vitejs/plugin-react": "React support for the Vite build",
    "eslint": "the linter that enforces the colour and fetch rules",
    "eslint-plugin-react-hooks": "rules of hooks",
    "eslint-plugin-react-refresh": "fast-refresh correctness in development",
    "globals": "environment globals for the ESLint config",
    "jsdom": (
        "the simulated browser DOM Vitest runs component tests in. Test-only; "
        "never bundled (Phase 16)."
    ),
    "openapi-typescript": (
        "generates src/types/api.generated.ts from docs/api/openapi.yaml. A "
        "build-time code generator: it emits type declarations, ships nothing "
        "to the browser and renders no UI, so the React-only rule permits it. "
        "It is what closes the gap where the frontend's types mirrored the "
        "backend's schemas by hand and nothing checked that they agreed."
    ),
    "prettier": "formatting, checked in CI",
    "typescript": "the compiler and the strict type gate",
    "typescript-eslint": "type-aware linting",
    "vite": "the bundler and dev server",
    "vitest": (
        "the test runner behind `npm run test` (Phase 16). Runs in Node on the Vite "
        "pipeline; ships nothing to the browser."
    ),
}


def _problems(manifest: dict[str, object]) -> list[str]:
    """Return every rule violation in the parsed manifest."""
    found: list[str] = []

    for field, allowed in (
        ("dependencies", ALLOWED_DEPENDENCIES),
        ("devDependencies", ALLOWED_DEV_DEPENDENCIES),
    ):
        declared = manifest.get(field, {})
        if not isinstance(declared, dict):
            found.append(f"{field} is not an object")
            continue

        for name, version in sorted(declared.items()):
            if name not in allowed:
                found.append(
                    f"{field}: '{name}' is not on the allowlist. The frontend is React "
                    f"only; adding a UI dependency is a decision and needs an ADR."
                )
            if not isinstance(version, str) or not _EXACT_VERSION.match(version):
                found.append(
                    f"{field}: '{name}' is pinned as '{version!s}'. "
                    f"Pin an exact version — no ranges."
                )

    return found


def main() -> int:
    """Check ``frontend/package.json``.

    Returns:
        ``0`` when the dependency set obeys the React-only rule, ``1`` otherwise.
    """
    if not PACKAGE_JSON.is_file():
        sys.stderr.write(f"not found: {PACKAGE_JSON}\n")
        return 1

    manifest = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    problems = _problems(manifest)

    if problems:
        sys.stderr.write("frontend/package.json violates the React-only rule:\n")
        for problem in problems:
            sys.stderr.write(f"  - {problem}\n")
        return 1

    total = len(ALLOWED_DEPENDENCIES) + len(ALLOWED_DEV_DEPENDENCIES)
    sys.stdout.write(f"frontend/package.json is React only; {total} dependencies, all pinned\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
