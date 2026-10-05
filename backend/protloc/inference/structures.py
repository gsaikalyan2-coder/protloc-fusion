"""Read-only AlphaFold backbones for the cached sequences (Phase 16, the 3D viewer).

``scripts/fetch_structures.py`` writes ``data/structures/<sequence sha256>.json`` and a
``manifest.json`` of each file's SHA-256, once, offline. This module only reads them. A file is
served only if its bytes match the manifest, so a corrupted or edited structure is refused
rather than drawn. Nothing here scores anything, and a structure never feeds a prediction.

The coordinates are AlphaFold DB predictions (CC-BY 4.0, EMBL-EBI), not experimental structures
and not this project's output; the document says so and the API repeats it.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
MANIFEST = "manifest.json"


class StructureError(RuntimeError):
    """A structure file does not match the manifest, so it must not be served."""


def is_sha256(value: str) -> bool:
    """Whether ``value`` is 64 lowercase hex characters (also what keeps paths inside root)."""
    return _SHA256.fullmatch(value) is not None


class StructureStore:
    """The structures under one directory, checked against that directory's manifest."""

    def __init__(self, root: Path) -> None:
        """Read structures from ``root`` (nothing is read until the first ``get``)."""
        self._root = root
        self._expected: dict[str, str] | None = None

    def _manifest(self) -> dict[str, str]:
        if self._expected is None:
            path = self._root / MANIFEST
            if not path.is_file():
                self._expected = {}
            else:
                files = json.loads(path.read_text(encoding="utf-8")).get("files_sha256", {})
                self._expected = {str(k): str(v) for k, v in files.items()}
        return self._expected

    @property
    def count(self) -> int:
        """How many structures the manifest lists."""
        return len(self._manifest())

    def get(self, sequence_sha256: str) -> dict[str, Any] | None:
        """The structure document for this sequence hash, or None if there is none.

        Raises ``ValueError`` for a malformed hash and ``StructureError`` if the file's bytes
        differ from the manifest.
        """
        if not is_sha256(sequence_sha256):
            raise ValueError("sequence_sha256 must be 64 lowercase hex characters")
        expected = self._manifest().get(sequence_sha256)
        path = self._root / f"{sequence_sha256}.json"
        if expected is None or not path.is_file():
            return None
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise StructureError(f"{path.name} does not match the structures manifest")
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise StructureError(f"{path.name} is not a JSON object")
        return {str(k): v for k, v in document.items()}


@lru_cache(maxsize=4)
def store_for(root: Path) -> StructureStore:
    """One store per directory, so the manifest is read once per process."""
    return StructureStore(root)
