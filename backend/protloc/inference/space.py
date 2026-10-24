"""Read-only protein-space map: the cached ESM-2 vectors projected to three dimensions.

``scripts/build_space.py`` writes ``data/space/space.json`` and ``manifest.json`` (the file's
SHA-256) once, offline. This module only reads them, and serves the document only if its bytes
match the manifest, so an edited or truncated file is refused rather than drawn. Nothing here
scores anything and the map never feeds a prediction: it places each cached sequence by a
principal-component projection of its mean-pooled ESM-2 vector, coloured by its Swiss-Prot
location.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

SPACE_FILE = "space.json"
MANIFEST_FILE = "manifest.json"


class SpaceError(RuntimeError):
    """The space file does not match its manifest, so it must not be served."""


class SpaceStore:
    """The space document under one directory, checked against that directory's manifest."""

    def __init__(self, root: Path) -> None:
        """Read from ``root``; nothing is read until the first ``get``."""
        self._root = root
        self._document: dict[str, Any] | None = None

    def get(self) -> dict[str, Any] | None:
        """The verified document, or None when the map has not been built.

        Raises ``SpaceError`` when the file exists but its bytes differ from the manifest.
        """
        if self._document is not None:
            return self._document
        path = self._root / SPACE_FILE
        manifest = self._root / MANIFEST_FILE
        if not path.is_file() or not manifest.is_file():
            return None
        expected = json.loads(manifest.read_text(encoding="utf-8")).get("files_sha256", {})
        body = path.read_bytes()
        actual = hashlib.sha256(body).hexdigest()
        if expected.get(SPACE_FILE) != actual:
            message = f"{SPACE_FILE} has sha256 {actual[:12]}, the manifest records another"
            raise SpaceError(message)
        document: dict[str, Any] = json.loads(body)
        self._document = document
        return document


@lru_cache(maxsize=4)
def store_for(root: Path) -> SpaceStore:
    """One store per directory, so the file is read and hashed once per process."""
    return SpaceStore(root)
