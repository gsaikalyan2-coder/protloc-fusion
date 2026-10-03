"""scripts/class_counts.py: states stay separate and the two evidence views differ."""

import importlib.util
from pathlib import Path
import sys

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "class_counts.py"
_SPEC = importlib.util.spec_from_file_location("class_counts", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
class_counts = importlib.util.module_from_spec(_SPEC)
sys.modules["class_counts"] = class_counts
_SPEC.loader.exec_module(class_counts)

_EXP = "ECO:0000269"
_IEA = "ECO:0000501"


def _label(state: str, values: dict[str, list[str]] | None = None) -> dict[str, object]:
    values = values or {}
    return {
        "state": state,
        "values": sorted(values),
        "cleavage_site": None,
        "evidence": {k: v for k, v in values.items() if v},
    }


def _record(group: str, labels: dict[str, dict[str, object]]) -> dict[str, object]:
    return {"organism_group": group, "labels": labels}


def _tally() -> "class_counts.Tally":
    tally = class_counts.Tally()
    tally.add(
        _record(
            "eukarya",
            {
                "localization": _label("annotated", {"nucleus": [_EXP], "cytoplasm": [_IEA]}),
                "go_function": _label("annotated", {"GO:0003677": ["IDA"]}),
            },
        )
    )
    tally.add(
        _record(
            "eukarya",
            {
                "localization": _label("annotated", {"nucleus": [_IEA]}),
                "go_function": _label("unknown"),
            },
        )
    )
    tally.add(
        _record(
            "archaea",
            {
                "localization": _label("not_applicable"),
                "go_function": _label("negative"),
            },
        )
    )
    return tally


def test_states_are_counted_separately() -> None:
    tally = _tally()
    assert tally.state("localization", "eukarya", "annotated") == 2
    assert tally.state("localization", "archaea", "not_applicable") == 1
    assert tally.state("localization", None, "unknown") == 0
    assert tally.state("go_function", None, "negative") == 1
    assert tally.state("go_function", None, "unknown") == 1


def test_record_level_experimental_view_needs_one_experimental_value() -> None:
    tally = _tally()
    assert tally.experimental("localization", "eukarya") == 1
    assert tally.experimental("localization", None) == 1


def test_class_level_experimental_view_counts_only_the_classes_own_evidence() -> None:
    tally = _tally()
    assert tally.class_count(tally.classes, "localization", "nucleus") == 2
    assert tally.class_count(tally.classes_experimental, "localization", "nucleus") == 1
    assert tally.class_count(tally.classes, "localization", "cytoplasm") == 1
    assert tally.class_count(tally.classes_experimental, "localization", "cytoplasm") == 0


def test_go_function_has_no_class_table_and_no_experimental_count() -> None:
    tally = _tally()
    assert tally.class_names("go_function") == []
    assert tally.experimental("go_function", None) == 0
