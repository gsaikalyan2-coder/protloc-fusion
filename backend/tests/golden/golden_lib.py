"""Record, load and diff the motif engine's golden outputs.

The golden set locks the *motif engine's* current behaviour so that Phase 14 can swap
the engine and show the difference as an explicit diff. It is a regression harness,
test-only, and is never served: every served prediction comes from a trained model.

Float tolerance
---------------
``ABS_TOL = 1e-6``, absolute. Every float the engine reports is rounded before it
leaves the engine (probabilities, raw scores and evidence to 4 decimals; hydropathy,
attribution and features to 3-4), so the smallest step a real change can produce is
one reporting quantum, 1e-4 or 1e-3 -- 100x to 1000x the tolerance. Platform libm
differences are around 1e-16 relative and vanish under that rounding except when a
value sits on a rounding boundary. A tolerance far below the quantum therefore
absorbs representation noise and nothing else; a looser one could hide a real
one-quantum drift. A change to ``ABS_TOL`` is itself recorded in the file header, so
it needs an explicit regeneration.

Structure
---------
Integers, strings, booleans and ``None`` compare exactly. A field added to or removed
from the engine's output is a difference, not something to ignore.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
from typing import Any

from protloc.inference import motif as service

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
GOLDEN_FORMAT = 1
ABS_TOL = 1e-6
#: Header keys that describe where a recording was made, not what the engine did.
INFORMATIONAL_HEADER_KEYS = ("recorded_with",)
#: Keys that identify an element of a list of records, so the diff can say which one.
_IDENTITY_KEYS = ("code", "rule_id", "accession", "go_id")
#: The categories PLAN §5 lists for the golden set, in its order.
SECTION5_CATEGORIES = (
    "secreted",
    "membrane",
    "cytoplasmic",
    "mitochondrial",
    "chloroplast",
    "gram_pos",
    "gram_neg",
    "multi_label",
    "ambiguous",
    "edge_case",
)


@dataclass(frozen=True)
class Delta:
    """One difference between a recorded value and the value now."""

    path: str
    detail: str


# ───────────────────────── recording ─────────────────────────
def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so tuples become lists, exactly as the file stores them."""
    return json.loads(json.dumps(value))


def build_header() -> dict[str, Any]:
    """The engine identity and the TRUST fit (tau, temperature) a recording was made under."""
    prov = service.provenance()
    return _jsonable(
        {
            "golden_format": GOLDEN_FORMAT,
            "purpose": (
                "Regression lock for the MOTIF engine. Test-only, never served. Phase 14 swaps "
                "the engine; this file then shows the difference."
            ),
            "engine": {
                "predictor": prov["predictor"],
                "model_version": prov["model_version"],
                "config_hash": prov["config_hash"],
                "rules_version": prov["rules_version"],
                "experiment_id": prov["experiment_id"],
                "source": service.ENGINE_SOURCE,
                "calibrated": service.CALIBRATED,
                "calibration_version": service.CALIBRATION_VERSION,
                "threshold_policy": service.THRESHOLD_POLICY,
            },
            "trust": service.TRUST,
            "tolerance": {"abs": ABS_TOL, "applies_to": "floats; everything else is exact"},
            "recorded_with": {"python": platform.python_version()},
        }
    )


def record(entry: dict[str, Any]) -> dict[str, Any]:
    """Run the engine on one entry's input and return its outcome as JSON-shaped data."""
    try:
        out = service.predict(entry["sequence"], entry["organism_group"], entry["id"])
    except ValueError as exc:
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
    return {"status": "ok", "output": _jsonable(out)}


def derive_categories(entry: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """Declared categories plus those the organism group and the recorded output define.

    ``gram_pos`` / ``gram_neg``: the organism group. ``multi_label``: two or more
    localization labels predicted. ``ambiguous``: no confident call, or at least one
    consistency flag fired.
    """
    cats = set(entry["declared_categories"])
    if entry["organism_group"] == "gram_pos":
        cats.add("gram_pos")
    if entry["organism_group"] == "gram_neg":
        cats.add("gram_neg")
    if recorded["status"] == "ok":
        out = recorded["output"]
        if sum(1 for row in out["localization"] if row["predicted"]) >= 2:
            cats.add("multi_label")
        if out["no_confident_call"] or out["consistency_flags"]:
            cats.add("ambiguous")
    return sorted(cats)


def build_golden(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Record every input under the current engine."""
    entries = []
    for item in inputs:
        recorded = record(item)
        entries.append(
            {**item, "categories": derive_categories(item, recorded), "recorded": recorded}
        )
    return {"header": build_header(), "entries": entries}


def dumps(golden: dict[str, Any]) -> str:
    """Stable text form: header on one line, one entry per line, so git diffs are per entry."""
    lines = ["{", f'"header": {json.dumps(golden["header"], indent=None)},', '"entries": [']
    body = [json.dumps(e, indent=None) for e in golden["entries"]]
    lines.append(",\n".join(body))
    lines += ["]", "}"]
    return "\n".join(lines) + "\n"


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, Any]:
    """Read the committed golden file."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# ───────────────────────── diffing ─────────────────────────
def _short(value: Any, width: int = 70) -> str:
    text = json.dumps(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _string_delta(rec: str, act: str, context: int = 20) -> str:
    """Long strings: show the first differing position with surrounding text."""
    i = next(
        (k for k, (r, a) in enumerate(zip(rec, act, strict=False)) if r != a),
        min(len(rec), len(act)),
    )
    lo = max(0, i - context)
    return (
        f"strings differ at char {i} (lengths {len(rec)} -> {len(act)}): "
        f"{rec[lo : i + context]!r} -> {act[lo : i + context]!r}"
    )


def _is_num(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _diff_numeric_lists(path: str, rec: list[Any], act: list[Any], tol: float) -> list[Delta]:
    if len(rec) != len(act):
        return [Delta(path, f"length {len(rec)} -> {len(act)}")]
    bad = [i for i, (r, a) in enumerate(zip(rec, act, strict=True)) if abs(a - r) > tol]
    if not bad:
        return []
    worst = max(bad, key=lambda i: abs(act[i] - rec[i]))
    delta = act[worst] - rec[worst]
    return [
        Delta(
            path,
            f"{len(bad)} of {len(rec)} positions differ (first [{bad[0]}]); "
            f"largest at [{worst}]: {rec[worst]} -> {act[worst]} (delta {delta:+.6g})",
        )
    ]


def _identity_key(items: list[Any]) -> str | None:
    if not items or not all(isinstance(i, dict) for i in items):
        return None
    for key in _IDENTITY_KEYS:
        if all(key in i for i in items):
            return key
    return None


def _diff_lists(path: str, rec: list[Any], act: list[Any], tol: float) -> list[Delta]:
    if all(_is_num(x) for x in rec) and all(_is_num(x) for x in act):
        return _diff_numeric_lists(path, rec, act, tol)
    key = _identity_key(rec)
    if key is not None and _identity_key(act) == key:
        by_rec = {i[key]: i for i in rec}
        by_act = {i[key]: i for i in act}
        out = [Delta(f"{path}[{k}]", "element removed") for k in by_rec.keys() - by_act.keys()]
        out += [Delta(f"{path}[{k}]", "element added") for k in by_act.keys() - by_rec.keys()]
        for k in by_rec.keys() & by_act.keys():
            out += diff(by_rec[k], by_act[k], f"{path}[{k}]", tol)
        return out
    out = []
    if len(rec) != len(act):
        out.append(Delta(path, f"length {len(rec)} -> {len(act)}"))
    for i, (r, a) in enumerate(zip(rec, act, strict=False)):
        out += diff(r, a, f"{path}[{i}]", tol)
    return out


def _diff_scalars(recorded: Any, actual: Any, path: str, tol: float) -> list[Delta]:
    """Numbers within ``tol``; everything else (str, bool, None, mixed types) exactly."""
    if _is_num(recorded) and _is_num(actual):
        if abs(actual - recorded) <= tol:
            return []
        return [Delta(path, f"{recorded} -> {actual} (delta {actual - recorded:+.6g})")]
    if type(recorded) is type(actual) and recorded == actual:
        return []
    if (
        isinstance(recorded, str)
        and isinstance(actual, str)
        and max(len(recorded), len(actual)) > 70
    ):
        return [Delta(path, _string_delta(recorded, actual))]
    return [Delta(path, f"{_short(recorded)} -> {_short(actual)}")]


def diff(recorded: Any, actual: Any, path: str = "", tol: float = ABS_TOL) -> list[Delta]:
    """Every difference between ``recorded`` and ``actual``, as readable path/detail pairs."""
    if isinstance(recorded, dict) and isinstance(actual, dict):
        out = [
            Delta(_join(path, k), f"field removed (was {_short(recorded[k])})")
            for k in recorded.keys() - actual.keys()
        ]
        out += [
            Delta(_join(path, k), f"field added ({_short(actual[k])})")
            for k in actual.keys() - recorded.keys()
        ]
        for k in recorded:
            if k in actual:
                out += diff(recorded[k], actual[k], _join(path, k), tol)
        return out
    if isinstance(recorded, list) and isinstance(actual, list):
        return _diff_lists(path, recorded, actual, tol)
    return _diff_scalars(recorded, actual, path, tol)


def format_deltas(deltas: list[Delta], limit: int = 25) -> str:
    """Indented, one line per difference, capped so a wholesale change stays readable."""
    lines = [f"    {d.path}: {d.detail}" for d in deltas[:limit]]
    if len(deltas) > limit:
        lines.append(f"    ... and {len(deltas) - limit} more")
    return "\n".join(lines)
