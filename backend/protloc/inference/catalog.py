"""The cached-sequence catalog: browse and look up every sequence the engine can score.

Layer L1 (inference), standard library only. ``scripts/build_catalog.py`` writes
``data/gallery/catalog.json``; this module reads it once at startup, keeps only the entries whose
sequence the loaded engine can actually score (so the list never offers a sequence that would be
refused ``not_cached``), and answers two questions: *which cached sequences match this filter*
and *which cached sequence has this accession*.

A missing or malformed file is not a startup failure. The catalog is a convenience over the
cache, not part of the model, so the service starts without it and reports it as unavailable
with the reason. Nothing here reads a score, a calibrator or a threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
#: Where the catalog lives, relative to ``DATA_ROOT`` (written by ``scripts/build_catalog.py``).
CATALOG_FILE = Path("gallery") / "catalog.json"
#: Largest page a listing returns; a client pages with ``offset``.
MAX_PAGE = 100
#: Fields a listing returns per entry (the sequence is fetched by lookup, one at a time).
LIST_FIELDS = (
    "id",
    "entry_name",
    "name",
    "organism",
    "organism_group",
    "length",
    "partition",
    "in_training_set",
    "database_labels",
    "sequence_sha256",
)


@dataclass(frozen=True, slots=True)
class Catalog:
    """The usable entries, by accession, and why the catalog is unavailable if it is."""

    entries: tuple[dict[str, Any], ...] = ()
    reason: str | None = "the catalog has not been loaded"
    dropped: int = 0
    _by_id: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def available(self) -> bool:
        """True when the catalog was read and verified."""
        return self.reason is None

    def state(self) -> dict[str, Any]:
        """What ``/demo/catalog`` reports about itself."""
        return {
            "available": self.available,
            "reason": self.reason,
            "entries": len(self.entries),
            "dropped_not_cached": self.dropped,
        }

    def lookup(self, key: str) -> dict[str, Any] | None:
        """The entry for an accession or entry name (case-insensitive), or ``None``."""
        return self._by_id.get(key.strip().upper())

    def search(  # noqa: PLR0913 - one argument per filter the screen offers
        self,
        query: str = "",
        *,
        location: str | None = None,
        group: str | None = None,
        partition: int | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """One page of matching entries (without sequences) and the total count."""
        needle = query.strip().casefold()
        matched = [
            e
            for e in self.entries
            if _matches(e, needle)
            and (location is None or location in e["database_labels"]["localization"])
            and (group is None or e["organism_group"] == group)
            and (partition is None or e["partition"] == partition)
        ]
        size = max(1, min(limit, MAX_PAGE))
        start = max(0, offset)
        return {
            "total": len(matched),
            "offset": start,
            "limit": size,
            "entries": [{k: e[k] for k in LIST_FIELDS} for e in matched[start : start + size]],
        }

    def locations(self) -> list[str]:
        """Every database location that occurs, for the filter control."""
        return sorted({v for e in self.entries for v in e["database_labels"]["localization"]})


def _matches(entry: Mapping[str, Any], needle: str) -> bool:
    if not needle:
        return True
    haystack = (entry["id"], entry["entry_name"], entry.get("organism") or "")
    return any(needle in str(h).casefold() for h in haystack)


def unavailable(reason: str) -> Catalog:
    """A catalog that lists nothing and says why."""
    return Catalog(reason=reason)


def load_catalog(path: Path, residue_cache: str, is_scorable: Callable[[str], bool]) -> Catalog:
    """Read ``path``; keep the entries ``is_scorable`` accepts; never raise for a bad file."""
    if not path.is_file():
        return unavailable(f"no catalog at {path.name}; run scripts/build_catalog.py --write")
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        return unavailable(f"the catalog is unreadable: {error}")
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        return unavailable("the catalog has an unknown schema version")
    if document.get("residue_cache") != residue_cache:
        return unavailable("the catalog was built for a different residue cache")
    raw = document.get("entries")
    if not isinstance(raw, list):
        return unavailable("the catalog holds no entry list")
    kept: list[dict[str, Any]] = []
    for item in raw:
        if _well_formed(item) and is_scorable(str(item["sequence"])):
            kept.append(dict(item))
    by_id: dict[str, dict[str, Any]] = {}
    for entry in kept:
        by_id.setdefault(str(entry["id"]).upper(), entry)
        by_id.setdefault(str(entry["entry_name"]).upper(), entry)
    return Catalog(tuple(kept), None, len(raw) - len(kept), by_id)


def load_for(data_root: Path, engine_config: Path, is_scorable: Callable[[str], bool]) -> Catalog:
    """The catalog under ``data_root``, checked against the residue cache ``engine_config`` pins."""
    try:
        pins = json.loads(engine_config.read_bytes())
        residue_cache = str(pins["encoder"]["residue_cache"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return unavailable(f"the engine configuration names no residue cache: {error}")
    return load_catalog(data_root / CATALOG_FILE, residue_cache, is_scorable)


_current: Catalog = unavailable("the catalog has not been loaded")


def configure(catalog: Catalog) -> None:
    """Install the process-wide catalog (startup)."""
    global _current  # noqa: PLW0603 - one process-wide catalog, set once at startup
    _current = catalog


def current() -> Catalog:
    """The process-wide catalog."""
    return _current


_REQUIRED: Sequence[str] = (*LIST_FIELDS, "sequence", "note")


def _well_formed(item: object) -> bool:
    if not isinstance(item, dict) or any(k not in item for k in _REQUIRED):
        return False
    labels = item["database_labels"]
    return isinstance(labels, dict) and isinstance(labels.get("localization"), list)


__all__ = [
    "CATALOG_FILE",
    "MAX_PAGE",
    "Catalog",
    "configure",
    "current",
    "load_catalog",
    "load_for",
    "unavailable",
]
