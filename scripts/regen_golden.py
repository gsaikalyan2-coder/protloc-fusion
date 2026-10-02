"""Regenerate the motif-engine golden set, deliberately.

``backend/tests/golden/golden_set.json`` records what the motif engine returns for
every golden input. It must never change silently, so this script is the only way to
rewrite it, and it never runs automatically (no test, hook or CI step calls it).

It always prints a summary of what would change first. It refuses to overwrite
anything unless ``--write`` is given::

    python scripts/regen_golden.py            # compare only; exit 1 if the file is stale
    python scripts/regen_golden.py --write    # print the summary, then overwrite

Run it from the repository root with ``backend`` on ``PYTHONPATH`` (or ``protloc``
installed). Review the summary and the git diff before committing the new file.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "backend" / "tests" / "golden"
_SHOWN_PER_ENTRY = 3


def _load(name: str) -> ModuleType:
    """Import a golden-set support module by path; it is test support, not a package."""
    spec = importlib.util.spec_from_file_location(name, GOLDEN_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        msg = f"cannot load {name} from {GOLDEN_DIR}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _say(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _summarise(lib: ModuleType, old: dict | None, new: dict) -> int:
    """Print what differs between the committed golden and a fresh recording; return the count."""
    if old is None:
        count = len(new["entries"])
        _say(f"No golden file at {lib.GOLDEN_PATH}: {count} entries would be created.")
        return count
    ignore = lib.INFORMATIONAL_HEADER_KEYS
    strip = lambda h: {k: v for k, v in h.items() if k not in ignore}  # noqa: E731
    header = lib.diff(strip(old["header"]), strip(new["header"]))
    old_by = {e["id"]: e for e in old["entries"]}
    new_by = {e["id"]: e for e in new["entries"]}
    added = [i for i in new_by if i not in old_by]
    removed = [i for i in old_by if i not in new_by]
    changed = {}
    for entry_id, entry in new_by.items():
        if entry_id in old_by:
            deltas = lib.diff(old_by[entry_id], entry)
            if deltas:
                changed[entry_id] = deltas
    _say(f"Engine header / TRUST: {len(header)} difference(s)")
    if header:
        _say(lib.format_deltas(header, limit=10))
    _say(f"Entries: {len(new_by)} now, {len(old_by)} recorded")
    _say(f"  added:     {len(added)} {added}")
    _say(f"  removed:   {len(removed)} {removed}")
    _say(f"  changed:   {len(changed)}")
    _say(f"  unchanged: {len(new_by) - len(added) - len(changed)}")
    for entry_id, deltas in changed.items():
        _say(f"  - {entry_id}: {len(deltas)} field difference(s)")
        _say(lib.format_deltas(deltas, limit=_SHOWN_PER_ENTRY).replace("    ", "      ", 1))
    return len(header) + len(added) + len(removed) + len(changed)


def main() -> int:
    """Compare, then write only when asked to."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite golden_set.json after printing the summary (otherwise refuse)",
    )
    args = parser.parse_args()
    lib = _load("golden_lib")
    inputs = _load("golden_inputs")
    new = lib.build_golden(inputs.build_entries())
    old = lib.load_golden() if lib.GOLDEN_PATH.exists() else None
    changes = _summarise(lib, old, new)
    if changes == 0:
        _say("Golden set is current. Nothing to write.")
        return 0
    if not args.write:
        _say("REFUSING to overwrite: the golden set differs. Review the above, then pass --write.")
        return 1
    lib.GOLDEN_PATH.write_text(lib.dumps(new), encoding="utf-8", newline="\n")
    _say(f"Wrote {lib.GOLDEN_PATH} ({len(new['entries'])} entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
