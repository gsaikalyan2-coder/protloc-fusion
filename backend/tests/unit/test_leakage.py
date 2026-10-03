"""The leakage checks catch a leak and stay quiet on a clean split (Phase 6).

Pure functions only: the MMseqs2 search itself is exercised by
``scripts/leakage_check.py --toy leaked|clean`` (docs/phases/06.md), because CI has
no MMseqs2. Records are real-format corpus records, built through ``SequenceRecord``.
"""

import json
from pathlib import Path

import pytest

from protloc.data.leakage import (
    confirm_pairs,
    cross_partition_pairs,
    exact_duplicate_crossings,
    parse_needle_identity,
    partition_stats,
    read_hits,
    registry_cluster_crossings,
    spanning_components,
)
from protloc.data.records import (
    TASK_ORDER,
    SequenceRecord,
    TaskLabel,
    normalize_sequence,
    sequence_sha256,
    to_canonical_line,
)
from protloc.domain.types import LabelState, OrganismGroup, TaskName


def _record(accession: str, raw: str, group: OrganismGroup = OrganismGroup.EUKARYA) -> dict:
    normalized = normalize_sequence(raw)
    labels = {task: TaskLabel(LabelState.UNKNOWN) for task in TASK_ORDER}
    labels[TaskName.LOCALIZATION] = TaskLabel.build(
        LabelState.ANNOTATED, {"nucleus"}, evidence={"nucleus": {"ECO:0000269"}}
    )
    record = SequenceRecord(
        accession=accession,
        entry_name=f"{accession}_TEST",
        sequence_raw=raw,
        sequence_normalized=normalized,
        sequence_sha256=sequence_sha256(normalized),
        length=len(normalized),
        is_fragment=False,
        organism="Homo sapiens (Human)",
        taxonomy_id=9606,
        organism_group=group,
        source_dataset="uniprotkb_swissprot",
        source_release="2099_01",
        labels=tuple((task, labels[task]) for task in TASK_ORDER),
    )
    return json.loads(to_canonical_line(record))


def test_exact_duplicate_in_two_partitions_is_a_leak() -> None:
    sha_of = {"A": "h1", "B": "h1", "C": "h2"}
    assert exact_duplicate_crossings({"A": 0, "B": 1, "C": 1}, sha_of) == ["h1"]


def test_exact_duplicate_in_one_partition_is_not() -> None:
    sha_of = {"A": "h1", "B": "h1", "C": "h2"}
    assert exact_duplicate_crossings({"A": 2, "B": 2, "C": 1}, sha_of) == []


def test_registry_cluster_spanning_partitions_is_reported() -> None:
    rows = [
        {"cluster_id": "C1", "partition": "1"},
        {"cluster_id": "C1", "partition": "2"},
        {"cluster_id": "C2", "partition": "3"},
        {"cluster_id": "C2", "partition": "3"},
    ]
    assert registry_cluster_crossings(rows) == ["C1"]


def test_pairs_are_strictly_above_threshold_and_cross_partition_only() -> None:
    partition_of = {"A": 0, "B": 1, "C": 1, "D": 0}
    hits = [
        ("A", "B", 0.300),  # exactly at the threshold: not above
        ("A", "C", 0.301),  # above, across partitions: a leak
        ("C", "A", 0.350),  # the same pair from the other side: counted once, best identity
        ("A", "D", 0.900),  # above but inside one partition: fine
        ("A", "A", 1.000),  # self hit
        ("A", "Z", 0.900),  # not in the registry
    ]
    assert cross_partition_pairs(hits, partition_of, 0.30) == [("A", "C", 0.35)]


def test_a_chain_through_one_partition_still_spans_partitions() -> None:
    partition_of = {"A": 0, "B": 0, "C": 1, "D": 2}
    hits = [("A", "B", 0.8), ("B", "C", 0.5)]
    assert spanning_components(hits, partition_of, 0.30) == 1


def test_clusters_inside_one_partition_do_not_span() -> None:
    partition_of = {"A": 0, "B": 0, "C": 1, "D": 1}
    hits = [("A", "B", 0.9), ("C", "D", 0.9), ("B", "C", 0.29)]
    assert spanning_components(hits, partition_of, 0.30) == 0


def test_read_hits_parses_mmseqs_rows(tmp_path: Path) -> None:
    path = tmp_path / "hits.tsv"
    path.write_text("A\tB\t0.859\nB\tA\t0.859\n", encoding="ascii")
    assert list(read_hits(path)) == [("A", "B", 0.859), ("B", "A", 0.859)]


def test_partition_stats_counts_real_format_records() -> None:
    records = [
        _record("P1", "MKTAYIAKQR"),
        _record("P2", "MKVLAAGIVG"),
        _record("P3", "MSEQNNTEMT"),
    ]
    stats = partition_stats(records, {"P1": 0, "P2": 0, "P3": 1})
    assert stats["partition_sizes"] == {"1": 2, "2": 1}
    assert stats["per_group"]["1"] == {"eukarya": 2}
    assert stats["task_states"]["1"]["localization"] == {"annotated": 2}
    assert stats["task_states"]["2"]["go_function"] == {"unknown": 1}


NEEDLE_REPORT = """#=======================================
# Length: 120
# Identity:      36/120 (30.0%)
# Similarity:    60/120 (50.0%)
#=======================================
"""


def test_needle_identity_is_identical_residues_over_alignment_length() -> None:
    """The ADR's definition: gaps count in the denominator."""
    assert parse_needle_identity(NEEDLE_REPORT) == 0.3
    with pytest.raises(ValueError, match="Identity"):
        parse_needle_identity("no identity here")


def test_a_high_local_identity_over_a_short_match_is_not_a_global_leak() -> None:
    """Candidates from the local search are re-scored; only global identity above 0.30 counts."""
    sequences = {"A": "MKV", "B": "MKVLLL", "C": "MKVAAA", "D": "WWW"}
    candidates = [("A", "B", 0.9), ("A", "C", 0.9), ("A", "D", 0.8)]
    scores = {("MKV", "MKVLLL"): 0.5, ("MKV", "MKVAAA"): 0.3, ("MKV", "WWW"): 0.1}

    def align_many(pairs: list[tuple[str, str]]) -> list[float]:
        return [scores[pair] for pair in pairs]

    confirmed = confirm_pairs(candidates, sequences, align_many, 0.30)
    assert confirmed == [("A", "B", 0.5)]
