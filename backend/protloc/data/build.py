"""Snapshot → corpus. Deterministic by construction, and tested for it.

The build reads one ``uniprot_sprot.dat.gz``, parses every entry, and writes:

* ``corpus.jsonl`` holds the records in canonical form, sorted by accession.
  Its SHA-256 is the dataset hash.
* ``build_report.json`` holds the counts: records, exclusions by reason, label
  states per task, classes per task, and vocabulary the rules did not cover.

Nothing time-, host- or environment-dependent reaches either file. Every input
that can change the output is hashed into the manifest: the snapshot, the three
rule files and the parser version.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import IO, cast

import Bio
from Bio import SwissProt
from Bio.SwissProt import Record as SwissProtRecord

from protloc.data.manifest import canonical_json_bytes, sha256_file
from protloc.data.parse import SOURCE_DATASET, parse_entry
from protloc.data.records import TASK_ORDER, SequenceRecord, to_canonical_line
from protloc.data.rules import CorpusRules
from protloc.domain.types import LabelState, TaskName

#: Biopython's parser is not annotated; this is the signature it has.
_parse_swissprot = cast(
    "Callable[[IO[str]], Iterator[SwissProtRecord]]",
    SwissProt.parse,
)

CORPUS_FILE = "corpus.jsonl"
REPORT_FILE = "build_report.json"

#: Tasks whose class vocabulary is small enough to count per class in the report.
_COUNTED_CLASS_TASKS: frozenset[TaskName] = frozenset(TASK_ORDER) - {TaskName.GO_FUNCTION}


class BuildError(RuntimeError):
    """The snapshot cannot be turned into a corpus as it stands."""


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Everything a build produced, before any of it is written."""

    lines: tuple[bytes, ...]
    report: dict[str, object]


def iter_entries(snapshot: Path) -> Iterator[SwissProtRecord]:
    """Stream parsed entries from a gzipped Swiss-Prot flat file."""
    with gzip.open(snapshot, mode="rt", encoding="utf-8") as handle:
        yield from _parse_swissprot(handle)


def build_corpus(
    entries: Iterator[SwissProtRecord], release: str, rules: CorpusRules
) -> BuildResult:
    """Parse every entry and assemble the sorted corpus and its report."""
    by_accession: dict[str, bytes] = {}
    excluded: Counter[str] = Counter()
    unmapped_terms: Counter[str] = Counter()
    unmapped_notes: Counter[str] = Counter()
    states: dict[TaskName, Counter[str]] = {task: Counter() for task in TASK_ORDER}
    classes: dict[TaskName, Counter[str]] = {task: Counter() for task in _COUNTED_CLASS_TASKS}
    go_terms: set[str] = set()
    groups: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    fragments = 0
    seen = 0

    for entry in entries:
        seen += 1
        outcome = parse_entry(entry, release, rules)
        unmapped_terms.update(outcome.diagnostics.unmapped_location_terms)
        unmapped_notes.update(outcome.diagnostics.unmapped_transit_notes)
        if outcome.record is None:
            excluded[outcome.excluded or "unspecified"] += 1
            continue
        record = outcome.record
        if record.accession in by_accession:
            raise BuildError(f"accession {record.accession} appears twice in the snapshot")
        by_accession[record.accession] = to_canonical_line(record)
        _tally(record, states, classes, go_terms)
        groups[record.organism_group.value] += 1
        hashes[record.sequence_sha256] += 1
        fragments += int(record.is_fragment)

    if not by_accession:
        raise BuildError("the snapshot produced no records")

    lines = tuple(by_accession[accession] for accession in sorted(by_accession))
    report: dict[str, object] = {
        "source_dataset": SOURCE_DATASET,
        "source_release": release,
        "entries_read": seen,
        "record_count": len(lines),
        "excluded": dict(sorted(excluded.items())),
        "excluded_total": sum(excluded.values()),
        "organism_groups": dict(sorted(groups.items())),
        "fragment_count": fragments,
        "distinct_sequences": len(hashes),
        "sequences_shared_by_several_accessions": sum(1 for n in hashes.values() if n > 1),
        "label_states": {
            task.value: {state.value: states[task][state.value] for state in LabelState}
            for task in TASK_ORDER
        },
        "label_classes": {
            task.value: dict(sorted(classes[task].items()))
            for task in TASK_ORDER
            if task in _COUNTED_CLASS_TASKS
        },
        "go_function_distinct_terms": len(go_terms),
        "unmapped_location_terms": dict(sorted(unmapped_terms.items())),
        "unmapped_transit_notes": dict(sorted(unmapped_notes.items())),
        "rules_sha256": dict(rules.sha256),
        "rules_versions": {
            "organism_groups": rules.groups.version,
            "uniprot_label_map": rules.labels.version,
            "residue_policy": rules.residues.version,
        },
        "parser": {"biopython": Bio.__version__},
    }
    return BuildResult(lines=lines, report=report)


def _tally(
    record: SequenceRecord,
    states: dict[TaskName, Counter[str]],
    classes: dict[TaskName, Counter[str]],
    go_terms: set[str],
) -> None:
    for task, label in record.labels:
        states[task][label.state.value] += 1
        if task in classes:
            classes[task].update(label.values)
        elif task is TaskName.GO_FUNCTION:
            go_terms.update(label.values)


def write_corpus(result: BuildResult, out_dir: Path) -> dict[str, object]:
    """Write the corpus and its report; return the facts the manifest records.

    The corpus is written to a temporary name and renamed into place, so an
    interrupted build never leaves a truncated file that looks complete.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / CORPUS_FILE
    partial = out_dir / (CORPUS_FILE + ".partial")
    with partial.open("wb") as handle:
        for line in result.lines:
            handle.write(line)
    partial.replace(corpus)
    report = out_dir / REPORT_FILE
    report.write_bytes(canonical_json_bytes(result.report))
    corpus_sha, corpus_bytes = sha256_file(corpus)
    report_sha, report_bytes = sha256_file(report)
    return {
        "corpus_sha256": corpus_sha,
        "corpus_bytes": corpus_bytes,
        "report_sha256": report_sha,
        "report_bytes": report_bytes,
    }


__all__ = [
    "CORPUS_FILE",
    "REPORT_FILE",
    "BuildError",
    "BuildResult",
    "build_corpus",
    "iter_entries",
    "write_corpus",
]
