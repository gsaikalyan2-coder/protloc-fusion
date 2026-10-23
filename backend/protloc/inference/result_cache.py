"""A bounded in-memory cache of finished results (feature 3).

Layer L1 (inference), standard library only. The engine is deterministic for a given sequence,
organism group and loaded model: the heads are numpy forward passes over cached states, the
calibrators and thresholds are frozen, and the evidence search reads a fixed index. So a result
computed once can be served again without recomputing it. :mod:`protloc.inference.service`
consults this cache; the engine knows nothing about it.

The rules:

* The key is ``(sha256 of the normalized sequence, organism group, with_evidence)``. The cache
  lives in one process with one loaded model, so the model version is implied; it is still
  recorded per entry and checked, so an entry can never outlive the model that produced it.
* A **degraded** result (evidence was requested but the search was unavailable or timed out) is
  never stored: the next request retries the search instead of repeating the failure.
* Refusals are never stored.
* Entries are deep-copied in and out, so a caller that edits a result cannot change the cache,
  and the caller's own ``id`` replaces the stored one.
* Least recently used entries are evicted beyond ``capacity``; ``capacity == 0`` disables it.

Every result says whether it was served from the cache (``result_cache: "hit"`` or ``"miss"``).
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import threading
from typing import Any, Final

#: Default number of entries: 500 cached sequences x 2 served groups, with headroom.
DEFAULT_CAPACITY: Final[int] = 1024

Key = tuple[str, str, bool]


def is_cacheable(result: dict[str, Any], with_evidence: bool) -> bool:
    """False for a result whose requested evidence search did not run (it is retried later)."""
    if not with_evidence:
        return True
    status = result.get("evidence_status")
    return isinstance(status, dict) and bool(status.get("available"))


class ResultCache:
    """A thread-safe least-recently-used map from :data:`Key` to a result."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """An empty cache holding at most ``capacity`` results (0 disables it)."""
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._items: OrderedDict[Key, tuple[str, dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.skipped_degraded = 0

    @property
    def enabled(self) -> bool:
        """Whether anything is ever stored."""
        return self.capacity > 0

    def get(self, key: Key, model_version: str, seq_id: str) -> dict[str, Any] | None:
        """A private copy of the stored result relabelled ``seq_id``, or ``None`` (a miss)."""
        with self._lock:
            found = self._items.get(key)
            if found is None or found[0] != model_version:
                if found is not None:
                    del self._items[key]
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            stored = found[1]
        result = copy.deepcopy(stored)
        result["id"] = seq_id
        result["result_cache"] = "hit"
        return result

    def put(
        self, key: Key, model_version: str, result: dict[str, Any], with_evidence: bool
    ) -> None:
        """Store a copy of ``result`` unless the cache is off or the result is degraded."""
        if not self.enabled:
            return
        if not is_cacheable(result, with_evidence):
            with self._lock:
                self.skipped_degraded += 1
            return
        stored = copy.deepcopy(result)
        stored.pop("result_cache", None)
        with self._lock:
            self._items[key] = (model_version, stored)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry and zero the counters."""
        with self._lock:
            self._items.clear()
            self.hits = self.misses = self.skipped_degraded = 0

    def stats(self) -> dict[str, Any]:
        """What ``/demo/metrics`` reports."""
        with self._lock:
            looked = self.hits + self.misses
            return {
                "enabled": self.enabled,
                "capacity": self.capacity,
                "entries": len(self._items),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / looked, 3) if looked else None,
                "skipped_degraded": self.skipped_degraded,
            }


__all__ = ["DEFAULT_CAPACITY", "Key", "ResultCache", "is_cacheable"]
