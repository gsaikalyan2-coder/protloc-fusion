"""The result cache (feature 3): rules of the cache, then its use by the service facade.

The service is exercised with a fake engine that counts how many sequences reach it, so a hit
is proven by the engine *not* being called, not by timing.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from protloc.inference import service
from protloc.inference.engine import EngineRefusal
from protloc.inference.result_cache import ResultCache, is_cacheable


def _result(seq_id: str = "a", available: bool = True, **over: Any) -> dict[str, Any]:
    base = {
        "id": seq_id,
        "sequence": "MKT",
        "top_location": "CYT",
        "evidence_status": {"available": available, "unavailable_reason": None},
        "localization": [{"code": "CYT", "probability": 0.9}],
    }
    base.update(over)
    return base


KEY = ("sha", "eukaryote", True)

# -- the cache itself -----------------------------------------------------------------------


def test_a_stored_result_comes_back_as_a_private_relabelled_copy() -> None:
    cache = ResultCache(4)
    original = _result("first")
    cache.put(KEY, "m1", original, with_evidence=True)
    original["top_location"] = "changed after put"
    got = cache.get(KEY, "m1", "second")
    assert got is not None
    assert (got["id"], got["top_location"], got["result_cache"]) == ("second", "CYT", "hit")
    got["localization"][0]["probability"] = -1
    again = cache.get(KEY, "m1", "third")
    assert again is not None
    assert again["localization"][0]["probability"] == 0.9


def test_a_degraded_result_is_never_stored() -> None:
    cache = ResultCache(4)
    cache.put(KEY, "m1", _result(available=False), with_evidence=True)
    assert cache.get(KEY, "m1", "x") is None
    assert cache.stats()["skipped_degraded"] == 1


def test_without_evidence_the_evidence_state_does_not_matter() -> None:
    assert is_cacheable(_result(available=False), with_evidence=False)
    assert not is_cacheable(_result(available=False), with_evidence=True)
    assert not is_cacheable({"id": "no status"}, with_evidence=True)


def test_an_entry_from_another_model_is_a_miss_and_is_dropped() -> None:
    cache = ResultCache(4)
    cache.put(KEY, "m1", _result(), with_evidence=True)
    assert cache.get(KEY, "m2", "x") is None
    assert cache.stats()["entries"] == 0


def test_the_least_recently_used_entry_is_evicted() -> None:
    cache = ResultCache(2)
    a, b, c = (("a", "g", True), ("b", "g", True), ("c", "g", True))
    cache.put(a, "m", _result(), True)
    cache.put(b, "m", _result(), True)
    assert cache.get(a, "m", "x") is not None
    cache.put(c, "m", _result(), True)
    assert cache.get(b, "m", "x") is None
    assert cache.get(a, "m", "x") is not None
    assert cache.stats()["entries"] == 2


def test_capacity_zero_stores_nothing_and_negative_is_refused() -> None:
    cache = ResultCache(0)
    cache.put(KEY, "m", _result(), True)
    assert not cache.enabled
    assert cache.stats()["entries"] == 0
    with pytest.raises(ValueError, match=">= 0"):
        ResultCache(-1)


def test_stats_report_the_hit_rate() -> None:
    cache = ResultCache(4)
    assert cache.stats()["hit_rate"] is None
    cache.put(KEY, "m", _result(), True)
    cache.get(KEY, "m", "x")
    cache.get(("other", "g", True), "m", "x")
    stats = cache.stats()
    assert (stats["hits"], stats["misses"], stats["hit_rate"]) == (1, 1, 0.5)
    cache.clear()
    assert cache.stats()["entries"] == 0


# -- the service uses it ------------------------------------------------------------------------


class _CountingEngine:
    """Scores every sequence it is given and remembers how many it was given."""

    provenance: dict[str, Any]

    def __init__(self, degraded: bool = False) -> None:
        self.provenance = {"model_version": "m1"}
        self.seen: list[str] = []
        self.calls = 0
        self.degraded = degraded

    def predict_many(
        self, records: list[tuple[str, str]], group: str, with_evidence: bool = True
    ) -> list[dict[str, Any] | EngineRefusal]:
        self.calls += 1
        out: list[dict[str, Any] | EngineRefusal] = []
        for seq_id, raw in records:
            self.seen.append(raw)
            if raw == "BAD":
                out.append(EngineRefusal("not_in_embedding_cache", "not cached"))
            else:
                out.append(_result(seq_id, available=not self.degraded, sequence=raw.upper()))
        return out


@pytest.fixture
def counting() -> Iterator[_CountingEngine]:
    fake = _CountingEngine()
    service.configure(fake)  # type: ignore[arg-type]
    service.configure_cache(8)
    yield fake
    service.configure(None)
    service.configure_cache(8)


def test_a_repeat_is_served_without_the_engine(counting: _CountingEngine) -> None:
    first = service.predict("MKTAY", "eukaryote", "one")
    second = service.predict("mkt ay", "eukaryote", "two")
    assert counting.seen == ["MKTAY"]
    assert (first["result_cache"], second["result_cache"]) == ("miss", "hit")
    assert second["id"] == "two"
    assert service.cache_stats()["hits"] == 1


def test_the_group_and_the_evidence_flag_are_part_of_the_key(counting: _CountingEngine) -> None:
    service.predict("MKTAY", "eukaryote")
    service.predict("MKTAY", "plant")
    service.predict("MKTAY", "eukaryote", with_evidence=False)
    assert len(counting.seen) == 3


def test_a_batch_sends_only_the_misses_in_one_call_and_keeps_order(
    counting: _CountingEngine,
) -> None:
    service.predict("AAAA", "eukaryote", "warm")
    counting.seen.clear()
    calls_before = counting.calls
    out = service.predict_many([("x", "CCCC"), ("y", "AAAA"), ("z", "BAD")], "eukaryote")
    assert counting.seen == ["CCCC", "BAD"]
    assert counting.calls == calls_before + 1
    assert [o["id"] if isinstance(o, dict) else o.code for o in out] == [
        "x",
        "y",
        "not_in_embedding_cache",
    ]
    assert [o["result_cache"] for o in out if isinstance(o, dict)] == ["miss", "hit"]


def test_a_refusal_is_raised_by_predict_and_never_cached(counting: _CountingEngine) -> None:
    with pytest.raises(EngineRefusal):
        service.predict("BAD", "eukaryote")
    with pytest.raises(EngineRefusal):
        service.predict("BAD", "eukaryote")
    assert counting.seen == ["BAD", "BAD"]


def test_a_degraded_result_is_recomputed_next_time() -> None:
    fake = _CountingEngine(degraded=True)
    service.configure(fake)  # type: ignore[arg-type]
    service.configure_cache(8)
    try:
        service.predict("MKTAY", "eukaryote")
        service.predict("MKTAY", "eukaryote")
        assert fake.seen == ["MKTAY", "MKTAY"]
        assert service.cache_stats()["skipped_degraded"] == 2
    finally:
        service.configure(None)


def test_a_disabled_cache_always_reaches_the_engine(counting: _CountingEngine) -> None:
    service.configure_cache(0)
    service.predict("MKTAY", "eukaryote")
    service.predict("MKTAY", "eukaryote")
    assert len(counting.seen) == 2
