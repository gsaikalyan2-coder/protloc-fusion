"""The golden set is big enough, covers every PLAN section 5 category, and is honest.

Exit criterion (PROJECT_PLAN section 4, row 4): at least 40 sequences. Section 5 lists
the coverage: secreted, membrane, cytoplasmic, mitochondrial, chloroplast, Gram+,
Gram-, multi-label, ambiguous and edge cases. This file reads only the committed
golden file, never the live engine, so it keeps passing after Phase 14 swaps the
engine; the regression test is the one that moves.
"""

from collections import Counter

from golden_inputs import build_entries
from golden_lib import SECTION5_CATEGORIES, derive_categories, diff, format_deltas, load_golden
import pytest

from protloc.data import reference_demo

GOLDEN = load_golden()
ENTRIES = GOLDEN["entries"]
MIN_TOTAL = 40
MIN_PER_CATEGORY = 3
ORIGINS = {"gallery", "synthetic", "engineered", "real"}
COUNTS = Counter(c for e in ENTRIES for c in e["categories"])


def _table() -> str:
    return ", ".join(f"{c}={COUNTS[c]}" for c in SECTION5_CATEGORIES)


@pytest.mark.golden
def test_at_least_forty_sequences():
    assert len(ENTRIES) >= MIN_TOTAL, f"{len(ENTRIES)} entries; need >= {MIN_TOTAL}"


@pytest.mark.golden
@pytest.mark.parametrize("category", SECTION5_CATEGORIES)
def test_every_section5_category_has_enough_entries(category: str):
    assert COUNTS[category] >= MIN_PER_CATEGORY, (
        f"category {category!r} has {COUNTS[category]} entries; need >= {MIN_PER_CATEGORY}. "
        f"All counts: {_table()}"
    )


@pytest.mark.golden
def test_only_known_categories_are_used():
    unknown = set(COUNTS) - set(SECTION5_CATEGORIES) - {"other_compartment"}
    assert not unknown, f"unrecognised categories: {sorted(unknown)}"


@pytest.mark.golden
def test_stored_categories_match_the_recorded_outputs():
    for e in ENTRIES:
        assert e["categories"] == derive_categories(e, e["recorded"]), e["id"]


@pytest.mark.golden
def test_ids_are_unique_and_origins_are_labelled():
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids))
    for e in ENTRIES:
        assert e["origin"] in ORIGINS, e["id"]
        assert e["derivation"], f"{e['id']}: derivation is empty"


@pytest.mark.golden
def test_no_sequence_is_called_real_without_a_source():
    for e in ENTRIES:
        if e["origin"] == "real":
            assert e["real_protein"] is True, e["id"]
            assert e["source"], f"{e['id']}: real needs a source"
            assert e["source"].get("accession"), f"{e['id']}: real source needs an accession"
        else:
            assert not e["real_protein"], f"{e['id']}: only origin=real may be a real protein"
            assert e["source"] is None, e["id"]


@pytest.mark.golden
def test_stored_inputs_match_the_deterministic_builder():
    """The sequences are frozen: if the generator changes, that is a visible diff."""
    built = build_entries()
    assert [e["id"] for e in ENTRIES] == [b["id"] for b in built]
    for stored, fresh in zip(ENTRIES, built, strict=True):
        deltas = diff({k: stored[k] for k in fresh}, fresh)
        assert not deltas, (
            f"{fresh['id']}: stored input differs from golden_inputs:\n" + format_deltas(deltas)
        )


@pytest.mark.golden
def test_all_ten_gallery_sequences_are_present():
    gallery = {e["id"] for e in ENTRIES if e["origin"] == "gallery"}
    assert gallery == {e["id"] for e in reference_demo.EXAMPLES}


@pytest.mark.golden
def test_header_records_engine_and_trust():
    header = GOLDEN["header"]
    assert header["engine"]["source"] == "motif"
    assert {"tau", "temperature"} <= set(header["trust"])
    assert header["tolerance"]["abs"] > 0
