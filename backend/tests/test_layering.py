"""The architecture, stated as assertions.

``FOLDER_STRUCTURE.md`` describes a dependency layering and ``CLAUDE.md`` calls a
violation of it a CI failure rather than a style opinion. This module is what
makes that true: it walks the abstract syntax tree of every file under
``backend/protloc/`` and checks each internal import against the rules below.

It is written before the modules it polices, so the first violation is caught by
a test that already existed rather than added after the fact and disabled for
failing on legacy code.

The rules
---------

1. ``domain`` imports nothing from ``protloc``. It is the dependency root, which
   is what lets everything above it be swapped for a stub in tests.
2. Nothing imports ``training`` — it is an entry point, not a dependency. This is
   the rule that prevents train/serve skew, and ``Dockerfile.serve`` omits the
   training dependencies so that breaking it also breaks the build.
3. ``api``, ``web`` and ``jobs`` import only ``domain``, ``inference``, ``config``
   and ``utils``. A router that reaches past ``inference`` is doing business
   logic, and business logic in a router is the boundary failing.
4. ``evaluation`` imports only ``domain``, ``config`` and ``utils``. It reaches
   the system over HTTP, so that the numbers in the paper are the numbers a user
   receives.
5. Otherwise a module may import only from a strictly deeper layer. Siblings at
   the same depth may not import each other: that is what keeps each pipeline
   stage ignorant of the others and leaves ``inference`` the only module that
   knows the order.

Depth is the layer's distance from delivery. ``domain`` is deepest; the delivery
modules are shallowest. A module may import a module of strictly greater depth.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "protloc"
PACKAGE_NAME = "protloc"

# Layer depth per module. Larger is deeper; a module may import strictly deeper
# modules only. Adding a module to protloc/ without adding it here fails
# test_every_module_has_a_declared_depth, which is deliberate: a new module is a
# layering decision and is made explicitly.
LAYER_DEPTH: dict[str, int] = {
    # L0 — delivery and entry points
    "api": 0,
    "web": 0,
    "jobs": 0,
    "evaluation": 0,
    "training": 0,
    # L1 — the conductor
    "inference": 10,
    # L2 — data and representation
    "features": 20,
    "data": 20,
    # L3 — model core
    "models": 30,
    "registry": 30,
    # L4 — trust
    "calibration": 40,
    "consistency": 40,
    # L5 — evidence
    "interpretability": 50,
    "evidence": 50,
    # infrastructure — importable from anywhere, imports only domain
    "config": 90,
    "utils": 90,
    # L-1 — the dependency root
    "domain": 100,
}

# Modules whose imports are restricted beyond the depth rule, and to what.
EXPLICIT_ALLOWLIST: dict[str, frozenset[str]] = {
    "api": frozenset({"domain", "inference", "config", "utils"}),
    "web": frozenset({"domain", "inference", "config", "utils"}),
    "jobs": frozenset({"domain", "inference", "config", "utils"}),
    "evaluation": frozenset({"domain", "config", "utils"}),
    "domain": frozenset(),
}

# Imported by nothing inside the package. Entry points, not dependencies.
NEVER_IMPORTED: frozenset[str] = frozenset({"training"})


def _module_of(path: Path) -> str:
    """Return the top-level ``protloc`` sub-package a source file belongs to."""
    return path.relative_to(PACKAGE_ROOT).parts[0]


def _source_files() -> list[Path]:
    """Every Python source file inside the package, sorted for stable reporting."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _internal_imports(path: Path) -> set[tuple[str, int]]:
    """Return ``(sub-package, line number)`` for each internal import in a file.

    Both absolute imports of ``protloc.x`` and relative imports that resolve to
    another sub-package are reported; a relative import inside one sub-package is
    not an internal dependency and is ignored.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    own_module = _module_of(path)
    found: set[tuple[str, int]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= _from_plain_import(node)
        elif isinstance(node, ast.ImportFrom):
            found |= _from_import_from(node, path)

    return {(module, line) for module, line in found if module != own_module}


def _from_plain_import(node: ast.Import) -> set[tuple[str, int]]:
    """Sub-packages named by ``import protloc.x`` / ``import protloc.x.y``."""
    found: set[tuple[str, int]] = set()
    for alias in node.names:
        parts = alias.name.split(".")
        if parts[0] == PACKAGE_NAME and len(parts) > 1:
            found.add((parts[1], node.lineno))
    return found


def _from_import_from(node: ast.ImportFrom, path: Path) -> set[tuple[str, int]]:
    """Sub-packages named by a ``from ... import ...`` statement.

    Covers the three spellings that reach another sub-package: an absolute
    ``from protloc.x import y``, an absolute ``from protloc import x`` where the
    sub-package is the imported name rather than part of the module path, and a
    relative import that climbs out of the current sub-package.
    """
    found: set[tuple[str, int]] = set()

    if node.level == 0:
        parts = node.module.split(".") if node.module else []
        if parts[:1] == [PACKAGE_NAME] and len(parts) > 1:
            found.add((parts[1], node.lineno))
        elif parts == [PACKAGE_NAME]:
            found |= {(alias.name, node.lineno) for alias in node.names}
        return found

    # Relative import: resolve it against this file's package path.
    package_parts = list(path.relative_to(PACKAGE_ROOT).parent.parts)
    climbed = package_parts[: len(package_parts) - (node.level - 1)]
    target = [*climbed, *(node.module.split(".") if node.module else [])]
    if target:
        found.add((target[0], node.lineno))
    else:
        # `from ... import x` reaching the package root: x is the sub-package.
        found |= {(alias.name, node.lineno) for alias in node.names}
    return found


def _all_internal_edges() -> list[tuple[Path, str, str, int]]:
    """Every internal import in the package as ``(file, importer, imported, line)``."""
    return [
        (path, _module_of(path), imported, line)
        for path in _source_files()
        for imported, line in _internal_imports(path)
    ]


def _location(path: Path, line: int) -> str:
    """A short, clickable file:line reference for failure messages."""
    return f"{path.relative_to(PACKAGE_ROOT.parent)}:{line}"


def test_package_root_exists() -> None:
    """The package this test polices is where it is expected to be."""
    assert PACKAGE_ROOT.is_dir(), f"package not found at {PACKAGE_ROOT}"


def test_every_module_has_a_declared_depth() -> None:
    """Every sub-package of ``protloc`` has an explicit layer depth.

    A new module is a layering decision. Adding one to the tree without adding it
    to LAYER_DEPTH leaves its dependencies unchecked, so that omission fails here
    rather than passing silently.
    """
    on_disk = {
        entry.name
        for entry in PACKAGE_ROOT.iterdir()
        if entry.is_dir() and not entry.name.startswith((".", "__"))
    }
    undeclared = sorted(on_disk - LAYER_DEPTH.keys())
    stale = sorted(LAYER_DEPTH.keys() - on_disk)
    assert not undeclared, f"modules with no declared layer depth: {undeclared}"
    assert not stale, f"LAYER_DEPTH names modules that do not exist: {stale}"


def test_module_count_is_seventeen() -> None:
    """The architecture is seventeen modules.

    Not a style rule: the count is stated in ``CLAUDE.md`` and
    ``FOLDER_STRUCTURE.md``, and an eighteenth module appearing without those
    documents changing means the tree and the plan have diverged.
    """
    assert len(LAYER_DEPTH) == 17, f"expected 17 modules, found {len(LAYER_DEPTH)}"


def test_domain_imports_nothing_internal() -> None:
    """``domain`` is the dependency root and depends on no other module."""
    violations = [
        f"{_location(path, line)} imports protloc.{imported}"
        for path, importer, imported, line in _all_internal_edges()
        if importer == "domain"
    ]
    assert not violations, "domain/ must import nothing from protloc:\n" + "\n".join(violations)


def test_training_is_imported_by_nothing() -> None:
    """Nothing in the package imports ``training``.

    One feature code path, two lifecycles. A training import inside the serving
    path is how train/serve skew starts, and it is a build failure here and in
    ``Dockerfile.serve``, which does not install the training dependencies.
    """
    violations = [
        f"{_location(path, line)} ({importer}) imports protloc.{imported}"
        for path, importer, imported, line in _all_internal_edges()
        if imported in NEVER_IMPORTED
    ]
    assert not violations, "training/ is an entry point and is imported by nothing:\n" + "\n".join(
        violations
    )


def test_explicit_allowlists_hold() -> None:
    """Delivery and evaluation modules import only what they are permitted to."""
    violations = []
    for path, importer, imported, line in _all_internal_edges():
        allowed = EXPLICIT_ALLOWLIST.get(importer)
        if allowed is not None and imported not in allowed:
            permitted = ", ".join(sorted(allowed)) or "nothing"
            violations.append(
                f"{_location(path, line)}: {importer} imports {imported}; "
                f"{importer} may import only {permitted}"
            )
    assert not violations, "restricted module imported something it may not:\n" + "\n".join(
        violations
    )


def test_imports_only_go_deeper() -> None:
    """A module imports only from a strictly deeper layer.

    Equal depth is a violation too: two pipeline stages importing each other is
    exactly the coupling the layering exists to prevent, and it leaves
    ``inference`` no longer the only module that knows the pipeline order.
    """
    violations = []
    for path, importer, imported, line in _all_internal_edges():
        importer_depth = LAYER_DEPTH[importer]
        imported_depth = LAYER_DEPTH[imported]
        if imported_depth <= importer_depth:
            relation = "the same layer as" if imported_depth == importer_depth else "shallower than"
            violations.append(
                f"{_location(path, line)}: {importer} (depth {importer_depth}) imports "
                f"{imported} (depth {imported_depth}), which is {relation} it"
            )
    assert not violations, "import crosses a layer boundary the wrong way:\n" + "\n".join(
        violations
    )


def test_no_module_imports_itself_circularly() -> None:
    """The internal import graph is acyclic.

    Implied by the depth rule while that rule holds, and asserted separately so
    that a future exception to the depth rule cannot quietly introduce a cycle.
    """
    edges: dict[str, set[str]] = {name: set() for name in LAYER_DEPTH}
    for _path, importer, imported, _line in _all_internal_edges():
        edges[importer].add(imported)

    visiting: set[str] = set()
    done: set[str] = set()
    cycles: list[str] = []

    def walk(node: str, trail: list[str]) -> None:
        if node in done:
            return
        if node in visiting:
            start = trail.index(node)
            cycles.append(" -> ".join([*trail[start:], node]))
            return
        visiting.add(node)
        for neighbour in sorted(edges[node]):
            walk(neighbour, [*trail, neighbour])
        visiting.discard(node)
        done.add(node)

    for name in sorted(edges):
        walk(name, [name])

    assert not cycles, "circular dependency between modules:\n" + "\n".join(cycles)


@pytest.mark.parametrize("module", sorted(LAYER_DEPTH))
def test_module_is_a_real_package(module: str) -> None:
    """Every declared module is an importable package with a docstring.

    A directory with no ``__init__.py`` is not a package, and a package with no
    docstring is one whose layer nobody wrote down.
    """
    init = PACKAGE_ROOT / module / "__init__.py"
    assert init.is_file(), f"{module}/ has no __init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    assert ast.get_docstring(tree), f"{module}/__init__.py has no module docstring"
