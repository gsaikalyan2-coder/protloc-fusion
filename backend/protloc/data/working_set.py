"""Which corpus records enter the Phase 6 split, and how that is counted.

The decision is the user's (Phase 6 brief, decision 2). A record is **necessary** when
it is in the corpus, is no longer than ``MAX_SEQUENCE_LENGTH``, and at least one
task defined for its organism group is ``annotated`` or ``negative`` (ADR-0006
§3.1). A record that is ``unknown`` or ``not_applicable`` on every task carries no
loss signal and is left out. It stays in the corpus and is counted here by reason.

Two other readings exist only so they can be measured side by side before the
choice is made. No evidence filter is applied: ECO codes are a loss-time view
(ADR-0006 §3.5) and must not change partition membership.

The functions read plain mappings (``json.loads`` of a corpus line), not
``SequenceRecord``: the corpus is 1 GB and the counting pass must stay fast. The
corpus is canonical, so this is the same data (``records.from_canonical_line``
proves the round trip).
"""

from collections import Counter
from collections.abc import Iterator, Mapping
import json
from pathlib import Path
from typing import Any

READING_NECESSARY = "necessary"
READING_EUKARYA_LOCALIZATION = "eukarya_localization"
READING_ALL = "all_length_filtered"
READING_EUKARYA_LOCALIZATION_ECO = "eukarya_localization_eco"
READINGS: tuple[str, ...] = (
    READING_NECESSARY,
    READING_EUKARYA_LOCALIZATION,
    READING_ALL,
    READING_EUKARYA_LOCALIZATION_ECO,
)

#: Experimental evidence (ADR-0006 §3.5). Used by one reading only, and only because the
#: full-size readings do not fit the compute budget (docs/phases/06.md).
EXPERIMENTAL = "ECO:0000269"

#: States that carry a loss signal (ADR-0006 §3.1).
LOSS_STATES = frozenset({"annotated", "negative"})
STATES: tuple[str, ...] = ("annotated", "negative", "unknown", "not_applicable")

REASON_TOO_LONG = "length_over_max"
REASON_NO_SIGNAL = "no_loss_signal"
REASON_NOT_EUKARYA_LOCALIZATION = "not_eukarya_with_annotated_localization"
REASON_NOT_EXPERIMENTAL = "localization_not_experimental"


class WorkingSetError(ValueError):
    """The working-set inputs are unusable."""


def iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each corpus line as a dict, in file order (accession order)."""
    with path.open("rb") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise WorkingSetError(f"{path}: a line is not a JSON object")
            yield record


def _carries_loss_signal(labels: Mapping[str, Any]) -> bool:
    return any(label["state"] in LOSS_STATES for label in labels.values())


def _eukarya_localization_reason(record: Mapping[str, Any], experimental_only: bool) -> str | None:
    localization = record["labels"]["localization"]
    if record["organism_group"] != "eukarya" or localization["state"] != "annotated":
        return REASON_NOT_EUKARYA_LOCALIZATION
    if experimental_only and not any(
        EXPERIMENTAL in codes for codes in localization["evidence"].values()
    ):
        return REASON_NOT_EXPERIMENTAL
    return None


def exclusion_reason(record: Mapping[str, Any], reading: str, max_length: int) -> str | None:
    """Why ``record`` is left out under ``reading``, or ``None`` when it is kept.

    The length check comes first for every reading, so each excluded record has
    exactly one reason.
    """
    if reading not in READINGS:
        raise WorkingSetError(f"unknown reading {reading!r}; expected one of {READINGS}")
    if record["length"] > max_length:
        return REASON_TOO_LONG
    if reading == READING_ALL:
        return None
    if reading == READING_NECESSARY:
        return None if _carries_loss_signal(record["labels"]) else REASON_NO_SIGNAL
    return _eukarya_localization_reason(record, reading == READING_EUKARYA_LOCALIZATION_ECO)


class Tally:
    """Counts for one reading, filled one record at a time."""

    def __init__(self, reading: str, max_length: int) -> None:
        """Start an empty tally for ``reading`` at ``max_length``."""
        self.reading = reading
        self.max_length = max_length
        self.corpus_records = 0
        self.excluded: Counter[str] = Counter()
        self.groups: Counter[str] = Counter()
        self.fragments = 0
        self.fragments_by_group: Counter[str] = Counter()
        self.sequence_counts: Counter[str] = Counter()
        #: task -> group -> state -> records, over kept records only.
        self.task_states: dict[str, dict[str, Counter[str]]] = {}
        self.kept: list[str] = []

    def add(self, record: Mapping[str, Any], keep_accessions: bool = False) -> bool:
        """Count one record. Returns True when it is kept."""
        self.corpus_records += 1
        reason = exclusion_reason(record, self.reading, self.max_length)
        if reason is not None:
            self.excluded[reason] += 1
            return False
        group = record["organism_group"]
        self.groups[group] += 1
        if record["is_fragment"]:
            self.fragments += 1
            self.fragments_by_group[group] += 1
        self.sequence_counts[record["sequence_sha256"]] += 1
        for task, label in record["labels"].items():
            self.task_states.setdefault(task, {}).setdefault(group, Counter())[label["state"]] += 1
        if keep_accessions:
            self.kept.append(record["accession"])
        return True

    def summary(self) -> dict[str, Any]:
        """A JSON-ready summary. Every value is counted, none is typed in."""
        in_set = sum(self.groups.values())
        shared = [n for n in self.sequence_counts.values() if n > 1]
        return {
            "reading": self.reading,
            "max_sequence_length": self.max_length,
            "corpus_records": self.corpus_records,
            "in_set": in_set,
            "excluded_total": sum(self.excluded.values()),
            "excluded_by_reason": dict(sorted(self.excluded.items())),
            "per_group": dict(sorted(self.groups.items())),
            "fragments": self.fragments,
            "fragments_by_group": dict(sorted(self.fragments_by_group.items())),
            "distinct_sequences": len(self.sequence_counts),
            "sequences_shared_by_several_accessions": len(shared),
            "records_in_shared_sequences": sum(shared),
            "task_states": {
                task: {
                    group: {state: counts.get(state, 0) for state in STATES}
                    for group, counts in sorted(groups.items())
                }
                for task, groups in sorted(self.task_states.items())
            },
        }


def tally_corpus(
    path: Path, max_length: int, readings: tuple[str, ...], keep_for: str | None = None
) -> dict[str, Tally]:
    """One pass over the corpus, filling one :class:`Tally` per reading."""
    tallies = {reading: Tally(reading, max_length) for reading in readings}
    for record in iter_corpus(path):
        for reading, tally in tallies.items():
            tally.add(record, keep_accessions=reading == keep_for)
    return tallies


__all__ = [
    "LOSS_STATES",
    "READINGS",
    "READING_ALL",
    "READING_EUKARYA_LOCALIZATION",
    "READING_EUKARYA_LOCALIZATION_ECO",
    "READING_NECESSARY",
    "REASON_NOT_EUKARYA_LOCALIZATION",
    "REASON_NOT_EXPERIMENTAL",
    "REASON_NO_SIGNAL",
    "REASON_TOO_LONG",
    "STATES",
    "Tally",
    "WorkingSetError",
    "exclusion_reason",
    "iter_corpus",
    "tally_corpus",
]
