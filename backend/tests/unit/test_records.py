"""The corpus record keeps unknown, negative and not-applicable apart.

ADR-0006 §3.1. A multi-task loss that reads an unannotated protein as a curated
negative is wrong in a way no single metric shows, so these tests pin the
distinction at the byte level: each state survives a round trip through the
canonical form, and the states never serialise to the same bytes.
"""

import pytest

from protloc.data.records import (
    TASK_ORDER,
    RecordError,
    SequenceRecord,
    TaskLabel,
    from_canonical_line,
    normalize_sequence,
    sequence_sha256,
    to_canonical_line,
)
from protloc.domain.types import LabelState, OrganismGroup, TaskName

RAW = "mkt ayiak\nqr"


def _record(labels: dict[TaskName, TaskLabel] | None = None) -> SequenceRecord:
    chosen = labels or {}
    normalized = normalize_sequence(RAW)
    return SequenceRecord(
        accession="P00001",
        entry_name="TEST_HUMAN",
        sequence_raw=RAW,
        sequence_normalized=normalized,
        sequence_sha256=sequence_sha256(normalized),
        length=len(normalized),
        is_fragment=False,
        organism="Homo sapiens (Human)",
        taxonomy_id=9606,
        organism_group=OrganismGroup.EUKARYA,
        source_dataset="uniprotkb_swissprot",
        source_release="2099_01",
        labels=tuple(
            (task, chosen.get(task, TaskLabel(LabelState.UNKNOWN))) for task in TASK_ORDER
        ),
    )


def _label_for(state: LabelState) -> TaskLabel:
    if state is LabelState.ANNOTATED:
        return TaskLabel.build(
            state, {"signal_peptide"}, cleavage_site=4, evidence={"signal_peptide": {"ECO:1"}}
        )
    return TaskLabel(state)


@pytest.mark.parametrize("state", list(LabelState))
def test_every_label_state_round_trips(state: LabelState) -> None:
    record = _record({TaskName.SIGNAL_PEPTIDE: _label_for(state)})
    line = to_canonical_line(record)
    back = from_canonical_line(line)
    assert back == record
    assert back.label(TaskName.SIGNAL_PEPTIDE).state is state


def test_unknown_and_negative_are_different_bytes() -> None:
    unknown = to_canonical_line(_record({TaskName.GO_FUNCTION: TaskLabel(LabelState.UNKNOWN)}))
    negative = to_canonical_line(_record({TaskName.GO_FUNCTION: TaskLabel(LabelState.NEGATIVE)}))
    masked = to_canonical_line(
        _record({TaskName.GO_FUNCTION: TaskLabel(LabelState.NOT_APPLICABLE)})
    )
    assert len({unknown, negative, masked}) == 3
    assert b'"state":"unknown"' in unknown
    assert b'"state":"negative"' in negative


def test_the_four_states_are_exactly_these() -> None:
    assert [s.value for s in LabelState] == ["annotated", "negative", "unknown", "not_applicable"]


def test_canonical_bytes_do_not_depend_on_input_order() -> None:
    a = TaskLabel.build(
        LabelState.ANNOTATED,
        {"nucleus", "cytoplasm"},
        evidence={"nucleus": {"ECO:2", "ECO:1"}, "cytoplasm": {"ECO:3"}},
    )
    b = TaskLabel.build(
        LabelState.ANNOTATED,
        {"cytoplasm", "nucleus"},
        evidence={"cytoplasm": {"ECO:3"}, "nucleus": {"ECO:1", "ECO:2"}},
    )
    assert to_canonical_line(_record({TaskName.LOCALIZATION: a})) == to_canonical_line(
        _record({TaskName.LOCALIZATION: b})
    )


def test_a_line_is_ascii_and_ends_in_one_lf() -> None:
    line = to_canonical_line(_record())
    assert line.endswith(b"\n")
    assert not line.endswith(b"\r\n")
    assert line.count(b"\n") == 1
    line.decode("ascii")


def test_a_non_canonical_line_is_refused() -> None:
    line = to_canonical_line(_record())
    spaced = line.replace(b'"length":', b'"length": ')
    with pytest.raises(RecordError, match="canonical"):
        from_canonical_line(spaced)


def test_reordered_keys_are_refused() -> None:
    line = to_canonical_line(_record())
    swapped = line.replace(b'{"accession":"P00001","entry_name":"TEST_HUMAN",', b"{", 1)
    swapped = swapped.replace(b"{", b'{"entry_name":"TEST_HUMAN","accession":"P00001",', 1)
    with pytest.raises(RecordError, match="keys"):
        from_canonical_line(swapped)


@pytest.mark.parametrize(
    "state", [LabelState.NEGATIVE, LabelState.UNKNOWN, LabelState.NOT_APPLICABLE]
)
def test_only_annotated_may_carry_values(state: LabelState) -> None:
    with pytest.raises(RecordError, match="values are non-empty exactly"):
        TaskLabel.build(state, {"nucleus"})


def test_annotated_needs_values() -> None:
    with pytest.raises(RecordError):
        TaskLabel(LabelState.ANNOTATED)


def test_cleavage_site_requires_annotation() -> None:
    with pytest.raises(RecordError, match="cleavage"):
        TaskLabel(LabelState.NEGATIVE, cleavage_site=20)


def test_evidence_must_name_a_label_value() -> None:
    with pytest.raises(RecordError, match="evidence keys"):
        TaskLabel.build(LabelState.ANNOTATED, {"nucleus"}, evidence={"cytoplasm": {"ECO:1"}})


def test_labels_must_cover_all_seven_tasks() -> None:
    record = _record()
    with pytest.raises(RecordError, match="seven tasks"):
        SequenceRecord(**{**_fields(record), "labels": record.labels[:-1]})


def test_hash_must_match_the_sequence() -> None:
    record = _record()
    with pytest.raises(RecordError, match="sha256"):
        SequenceRecord(**{**_fields(record), "sequence_sha256": "0" * 64})


def test_normalisation_substitutes_nothing() -> None:
    assert normalize_sequence(" mxu\tob z\r\n") == "MXUOBZ"


def test_an_unknown_state_string_is_refused() -> None:
    line = to_canonical_line(_record()).replace(b'"state":"unknown"', b'"state":"missing"', 1)
    with pytest.raises(RecordError, match="unknown label state"):
        from_canonical_line(line)


def _fields(record: SequenceRecord) -> dict[str, object]:
    return {name: getattr(record, name) for name in SequenceRecord.__slots__}
