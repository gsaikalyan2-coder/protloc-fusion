"""The served inference facade (Phase 14).

Layer L6 (inference). The live app imports this module and nothing else from ``inference/``. It
holds the one verified :class:`~protloc.inference.engine.Engine` for the process, and it keeps
the same names the pre-Phase-14 service had (``predict``, ``scan``, ``summarize``,
``landscape``, ``parse_fasta``, ``provenance``) so the routers did not move. The motif engine
survives as :mod:`protloc.inference.motif`: it still drives the Scanner and the golden baseline,
and every field it produces is labelled ``source: "motif"``.

Order of work lives in ``engine.py`` only: features → model → calibration → consistency →
evidence. Nothing here scores anything.
"""

from typing import Any

from protloc.data.records import normalize_sequence, sequence_sha256
from protloc.features.classical import LOC_NAMES, ORGANISM_GROUPS
from protloc.inference import motif
from protloc.inference.engine import Engine, EngineRefusal
from protloc.inference.result_cache import DEFAULT_CAPACITY, Key, ResultCache

parse_fasta = motif.parse_fasta
parse_mutations = motif.parse_mutations

_SEVERITY_ORDER = ("blocked", "warning", "uncertain")
_SEVERITY_WEIGHT = {"blocked": 4.0, "warning": 2.0, "uncertain": 1.0}
_ABSTAIN_WEIGHT = 3.0
_NO_HOMOLOG_WEIGHT = 1.0
_MULTI_COMPARTMENT_WEIGHT = 1.5
_MULTI_COMPARTMENT_MIN = 2
_CLOSE_HOMOLOG_IDENTITY = 0.3

_engine: Engine | None = None
_cache = ResultCache(DEFAULT_CAPACITY)


class EngineNotConfigured(RuntimeError):  # noqa: N818
    """The service was asked to predict before an engine was loaded."""


def configure(engine: Engine | None) -> None:
    """Install (or, with ``None``, remove) the process-wide engine."""
    global _engine  # noqa: PLW0603 - one engine per process by design
    _engine = engine


def engine() -> Engine:
    """The loaded engine, or :class:`EngineNotConfigured`."""
    if _engine is None:
        raise EngineNotConfigured("no model is loaded; the service has not started correctly")
    return _engine


def provenance() -> dict[str, Any]:
    """The four reproducibility fields plus the bundles behind them."""
    return dict(engine().provenance)


def configure_cache(capacity: int) -> None:
    """Replace the result cache with an empty one of ``capacity`` entries (0 turns it off)."""
    global _cache  # noqa: PLW0603 - one cache per process, beside the one engine
    _cache = ResultCache(capacity)


def cache_stats() -> dict[str, Any]:
    """Size, hits, misses and hit rate of the result cache."""
    return _cache.stats()


def _key(raw: str, group: str, with_evidence: bool) -> Key | None:
    """The cache key, or ``None`` when the sequence does not normalize (the engine refuses it)."""
    try:
        return (sequence_sha256(normalize_sequence(raw)), group, with_evidence)
    except ValueError:
        return None


def predict(
    seq_raw: str, group: str, seq_id: str = "query", with_evidence: bool = True
) -> dict[str, Any]:
    """One sequence through the engine; an :class:`EngineRefusal` is a ``ValueError``."""
    only = predict_many([(seq_id, seq_raw)], group, with_evidence)[0]
    if isinstance(only, EngineRefusal):
        raise only
    return only


def predict_many(
    records: list[tuple[str, str]], group: str, with_evidence: bool = True
) -> list[dict[str, Any] | EngineRefusal]:
    """Many sequences with one evidence search; refusals are returned in place.

    Results already in the cache are served from it; only the rest reach the engine, still in
    one call (one evidence search). Order is preserved.
    """
    model = str(engine().provenance["model_version"])
    keys = [_key(raw, group, with_evidence) for _, raw in records]
    outcome: list[dict[str, Any] | EngineRefusal | None] = []
    todo: list[int] = []
    for index, ((seq_id, _), key) in enumerate(zip(records, keys, strict=True)):
        hit = _cache.get(key, model, seq_id) if key is not None and _cache.enabled else None
        outcome.append(hit)
        if hit is None:
            todo.append(index)
    if todo:
        fresh = engine().predict_many([records[i] for i in todo], group, with_evidence)
        for index, result in zip(todo, fresh, strict=True):
            key = keys[index]
            if isinstance(result, dict):
                result["result_cache"] = "miss"
                if key is not None:
                    _cache.put(key, model, result, with_evidence)
            outcome[index] = result
    return [r for r in outcome if r is not None]


def scan(
    seq_raw: str,
    group: str,
    target: str | None = None,
    mutations: str = "",
    substitute: str = "A",
) -> dict[str, Any]:
    """The in-silico scanner. It runs the motif engine and says so in its provenance."""
    out = motif.scan(seq_raw, group, target, mutations, substitute)
    out["scanner_source"] = "motif"
    out["provenance"] = provenance()
    return out


def attention_score(r: dict[str, Any]) -> tuple[float, list[str]]:
    """How much a result deserves a person's eyes (triage). Higher is more."""
    reasons: list[str] = []
    score = 0.0
    if r["abstained"]:
        score += _ABSTAIN_WEIGHT
        reasons.append("abstained")
    for flag in r["consistency_flags"]:
        score += _SEVERITY_WEIGHT[flag["severity"]]
        reasons.append(f"{flag['rule_id']} {flag['severity']}")
    status = r["evidence_status"]
    agreement = r["evidence_agreement"]
    if status["available"] and (
        agreement is None or agreement["identity"] < _CLOSE_HOMOLOG_IDENTITY
    ):
        score += _NO_HOMOLOG_WEIGHT
        reasons.append("no close homolog")
    predicted = [row for row in r["localization"] if row["predicted"]]
    if len(predicted) >= _MULTI_COMPARTMENT_MIN:
        score += _MULTI_COMPARTMENT_WEIGHT
        reasons.append("multi-compartment")
    score += round(1 - max(row["probability"] for row in r["localization"]), 3)
    return round(score, 3), reasons


def summarize(r: dict[str, Any]) -> dict[str, Any]:
    """One row of the triage table."""
    score, reasons = attention_score(r)
    membrane = r["membrane_type"]
    agreement = r["evidence_agreement"]
    severities = [f["severity"] for f in r["consistency_flags"]]
    return {
        "id": r["id"],
        "length": r["length"],
        "top_location": r["top_location"],
        "top_probability": max(row["probability"] for row in r["localization"]),
        "predicted": [row["code"] for row in r["localization"] if row["predicted"]],
        "abstained": r["abstained"],
        "signal": r["sorting_signal"]["type"],
        "membrane": max(membrane, key=lambda k: membrane[k]),
        "n_flags": len(severities),
        "worst_flag": min(severities, key=_SEVERITY_ORDER.index, default=None),
        "evidence_agrees": agreement["agrees"] if agreement else None,
        "attention": score,
        "reasons": reasons,
    }


def landscape(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Co-occurrence of predicted compartments over a batch."""
    codes = [row["code"] for row in results[0]["localization"]] if results else []
    matrix = {a: dict.fromkeys(codes, 0) for a in codes}
    for r in results:
        predicted = [row["code"] for row in r["localization"] if row["predicted"]]
        for a in predicted:
            for b in predicted:
                matrix[a][b] += 1
    return {
        "codes": codes,
        "names": {c: LOC_NAMES[c] for c in codes},
        "matrix": matrix,
        "counts": {c: matrix[c][c] for c in codes},
        "abstained": sum(1 for r in results if r["abstained"]),
        "total": len(results),
    }


__all__ = [
    "LOC_NAMES",
    "ORGANISM_GROUPS",
    "EngineNotConfigured",
    "attention_score",
    "configure",
    "engine",
    "landscape",
    "parse_fasta",
    "parse_mutations",
    "predict",
    "predict_many",
    "provenance",
    "scan",
    "summarize",
]
