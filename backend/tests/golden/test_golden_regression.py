"""The motif engine still produces exactly what the golden file recorded.

A failure here is not necessarily a bug: it says the engine's behaviour changed, for
which sequence, in which field, from what to what. If the change is intended, review
the diff and regenerate deliberately with ``python scripts/regen_golden.py --write``.
"""

from typing import Any

from golden_lib import (
    INFORMATIONAL_HEADER_KEYS,
    build_header,
    diff,
    format_deltas,
    load_golden,
    record,
)
import pytest

GOLDEN = load_golden()


@pytest.mark.golden
def test_engine_identity_and_trust_fit_are_unchanged():
    recorded = {k: v for k, v in GOLDEN["header"].items() if k not in INFORMATIONAL_HEADER_KEYS}
    now = {k: v for k, v in build_header().items() if k not in INFORMATIONAL_HEADER_KEYS}
    deltas = diff(recorded, now)
    assert not deltas, (
        "engine identity / TRUST (tau, temperature) differs from the recording:\n"
        + format_deltas(deltas)
    )


@pytest.mark.golden
@pytest.mark.parametrize("entry", GOLDEN["entries"], ids=lambda e: e["id"])
def test_recorded_output_is_reproduced(entry: dict[str, Any]):
    deltas = diff(entry["recorded"], record(entry))
    assert not deltas, (
        f"golden mismatch for {entry['id']} (origin={entry['origin']}, "
        f"group={entry['organism_group']}): {len(deltas)} difference(s)\n" + format_deltas(deltas)
    )
