"""Split plumbing: identical sequences stay together, clusters are components, removal is total."""

from pathlib import Path

import pytest

from protloc.data.records import (
    TASK_ORDER,
    SequenceRecord,
    TaskLabel,
    normalize_sequence,
    sequence_sha256,
    to_canonical_line,
)
from protloc.data.splits import (
    SplitError,
    build_registry,
    distinct_sequences,
    fasta_text,
    homology_clusters,
    leakage_removals,
    read_graphpart,
    read_registry,
    registry_bytes,
)
from protloc.domain.types import LabelState, OrganismGroup


def _line(accession: str, raw: str) -> bytes:
    normalized = normalize_sequence(raw)
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
        organism_group=OrganismGroup.EUKARYA,
        source_dataset="uniprotkb_swissprot",
        source_release="2099_01",
        labels=tuple((task, TaskLabel(LabelState.UNKNOWN)) for task in TASK_ORDER),
    )
    return to_canonical_line(record)


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_bytes(
        b"".join(
            [
                _line("P1", "MKTAYIAKQR"),
                _line("P2", "MKVLAAGIVG"),
                _line("P3", "MKTAYIAKQR"),  # same sequence as P1
                _line("P9", "MSEQNNTEMT"),  # not in the working set
            ]
        )
    )
    return path


def test_identical_sequences_collapse_to_the_smallest_accession(tmp_path: Path) -> None:
    rep_of, sequence_of = distinct_sequences(_corpus(tmp_path), ["P1", "P2", "P3"])
    assert rep_of == {"P1": "P1", "P2": "P2", "P3": "P1"}
    assert sorted(sequence_of) == ["P1", "P2"]
    assert fasta_text(sequence_of) == ">P1\nMKTAYIAKQR\n>P2\nMKVLAAGIVG\n"
    assert fasta_text(sequence_of, every=2) == ">P1\nMKTAYIAKQR\n"


def test_a_working_set_accession_missing_from_the_corpus_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SplitError):
        distinct_sequences(_corpus(tmp_path), ["P1", "P404"])


def test_clusters_are_components_above_the_threshold() -> None:
    edges = [("A", "B", 0.5), ("B", "C", 0.31), ("C", "D", 0.30), ("E", "A", 0.25)]
    clusters = homology_clusters(["A", "B", "C", "D", "E"], edges)
    assert clusters["A"] == clusters["B"] == clusters["C"]
    assert len({clusters["A"], clusters["D"], clusters["E"]}) == 3
    # numbered by the smallest member, so the numbering depends only on the data
    assert clusters["A"] == 0


def test_registry_keeps_identical_sequences_together_and_removes_them_together() -> None:
    rep_of = {"P1": "P1", "P3": "P1", "P2": "P2", "P5": "P5", "P6": "P5"}
    rows, removed = build_registry(rep_of, {"P1": 0, "P2": 4}, {"P1": 0, "P2": 1})
    by_accession = {r["accession"]: r for r in rows}
    assert by_accession["P1"]["partition"] == by_accession["P3"]["partition"] == 1
    assert by_accession["P1"]["cluster_id"] == by_accession["P3"]["cluster_id"]
    assert by_accession["P2"]["partition"] == 5
    assert removed == ["P5", "P6"]
    assert {r["partitioner"] for r in rows} == {"graphpart"}
    assert {r["identity_threshold"] for r in rows} == {"0.30"}


def test_registry_round_trips_through_bytes(tmp_path: Path) -> None:
    rows, _ = build_registry({"P1": "P1"}, {"P1": 2}, {"P1": 7})
    path = tmp_path / "registry.tsv"
    path.write_bytes(registry_bytes(rows))
    assert read_registry(path)[0]["cluster_id"] == "C0000007"
    assert registry_bytes(rows).endswith(b"\n")
    assert b"\r" not in registry_bytes(rows)


def test_graphpart_cluster_column_is_the_partition(tmp_path: Path) -> None:
    path = tmp_path / "graphpart.csv"
    path.write_text("AC,priority,label-val,cluster\nP1,False,0,3.0\nP2,False,0,0.0\n")
    assert read_graphpart(path) == {"P1": 3, "P2": 0}


def test_a_leaking_pair_loses_one_member_and_its_identical_twins() -> None:
    """Drop from the larger partition, ties to the later accession; twins go together."""
    partition_of = {"A": 1, "B": 2, "B2": 2, "C": 2, "D": 1, "E": 1}
    rep_of = {"A": "A", "B": "B", "B2": "B", "C": "C", "D": "D", "E": "E"}
    # Partition 2 has three accessions, partition 1 has three: tie, so the later accession goes.
    assert leakage_removals([("A", "B", 0.5)], partition_of, rep_of) == ["B", "B2"]
    # Partition 2 is larger than partition 1 after removing nothing: its member goes.
    partition_of = {"A": 1, "B": 2, "C": 2, "D": 2}
    rep_of = {k: k for k in partition_of}
    assert leakage_removals([("A", "B", 0.5)], partition_of, rep_of) == ["B"]
    # A pair with a member already dropped is skipped.
    assert leakage_removals([("A", "B", 0.5), ("B", "C", 0.4)], partition_of, rep_of) == ["B"]
