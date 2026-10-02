"""The corpus record and its canonical byte form.

One :class:`SequenceRecord` per Swiss-Prot accession, carrying the seven task
labels of ADR-0006. The record is also the unit of the dataset hash, so this
module fixes exactly how a record becomes bytes:

* the fields are written in declaration order, and the labels in
  :class:`~protloc.domain.types.TaskName` order;
* every list is sorted and de-duplicated before it is written;
* ``json.dumps`` is called with ``ensure_ascii=True`` and the most compact
  separators;
* there are no floats anywhere, so float formatting can't vary between
  platforms;
* each line ends with exactly one newline byte.

:func:`from_canonical_line` re-serialises what it parsed and refuses a line
that doesn't round-trip byte for byte. The corpus can therefore only be read
back as it was written, never "nearly".
"""

from dataclasses import dataclass
import hashlib
import json

from protloc.domain.types import LabelState, OrganismGroup, TaskName

#: The labels of every record, in this order. Fixed by the enum.
TASK_ORDER: tuple[TaskName, ...] = tuple(TaskName)


class RecordError(ValueError):
    """A record, or a line claiming to be one, violates the corpus schema."""


def normalize_sequence(raw: str) -> str:
    """Return the normalised sequence: uppercase, with all whitespace removed.

    No residue is substituted (``backend/rules/residue_policy.yaml``).
    """
    return "".join(raw.split()).upper()


def sequence_sha256(normalized: str) -> str:
    """SHA-256 of a normalised sequence, over its ASCII bytes."""
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _sorted_unique(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(items)))


@dataclass(frozen=True, slots=True)
class TaskLabel:
    """One task's label for one record.

    Attributes:
        state: which of the four states applies.
        values: class strings; non-empty exactly when ``state`` is annotated.
        cleavage_site: 1-based position of the last residue of the peptide, for
            an annotated signal or transit peptide; otherwise ``None``.
        evidence: ``(value, codes)`` pairs, one per value that has evidence.
    """

    state: LabelState
    values: tuple[str, ...] = ()
    cleavage_site: int | None = None
    evidence: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        """Enforce the invariants that make the four states distinguishable."""
        if self.values != _sorted_unique(self.values):
            raise RecordError(f"label values must be sorted and unique: {self.values}")
        if (self.state is LabelState.ANNOTATED) != bool(self.values):
            raise RecordError(
                f"state {self.state.value!r} is incompatible with values {self.values}: "
                "values are non-empty exactly when the state is annotated"
            )
        if self.cleavage_site is not None and (
            self.state is not LabelState.ANNOTATED or self.cleavage_site < 1
        ):
            raise RecordError("a cleavage site needs an annotated state and a position >= 1")
        keys = tuple(value for value, _ in self.evidence)
        if keys != _sorted_unique(keys) or not set(keys) <= set(self.values):
            raise RecordError(f"evidence keys must be sorted, unique label values: {keys}")
        for value, codes in self.evidence:
            if codes != _sorted_unique(codes):
                raise RecordError(f"evidence codes for {value!r} must be sorted and unique")

    @classmethod
    def build(
        cls,
        state: LabelState,
        values: set[str] | None = None,
        cleavage_site: int | None = None,
        evidence: dict[str, set[str]] | None = None,
    ) -> "TaskLabel":
        """Construct a label from unordered collections, ordering them canonically."""
        ordered_evidence = tuple(
            (value, tuple(sorted(codes))) for value, codes in sorted((evidence or {}).items())
        )
        return cls(
            state=state,
            values=tuple(sorted(values or set())),
            cleavage_site=cleavage_site,
            evidence=ordered_evidence,
        )

    def to_json_obj(self) -> dict[str, object]:
        """The label as a JSON-ready mapping in canonical key order."""
        return {
            "state": self.state.value,
            "values": list(self.values),
            "cleavage_site": self.cleavage_site,
            "evidence": {value: list(codes) for value, codes in self.evidence},
        }

    @classmethod
    def from_json_obj(cls, obj: object) -> "TaskLabel":
        """Parse a label mapping produced by :meth:`to_json_obj`."""
        mapping = _as_mapping(obj, "label")
        _require_keys(mapping, ("state", "values", "cleavage_site", "evidence"), "label")
        cleavage = mapping["cleavage_site"]
        if cleavage is not None and (not isinstance(cleavage, int) or isinstance(cleavage, bool)):
            raise RecordError("cleavage_site must be an integer or null")
        evidence = _as_mapping(mapping["evidence"], "evidence")
        try:
            state = LabelState(_as_str(mapping["state"], "state"))
        except ValueError as exc:
            raise RecordError(f"unknown label state {mapping['state']!r}") from exc
        return cls(
            state=state,
            values=_as_str_tuple(mapping["values"], "values"),
            cleavage_site=cleavage,
            evidence=tuple(
                (value, _as_str_tuple(codes, "evidence codes")) for value, codes in evidence.items()
            ),
        )


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One Swiss-Prot entry, as the corpus stores it (CLAUDE.md § Data schema)."""

    accession: str
    entry_name: str
    sequence_raw: str
    sequence_normalized: str
    sequence_sha256: str
    length: int
    is_fragment: bool
    organism: str
    taxonomy_id: int
    organism_group: OrganismGroup
    source_dataset: str
    source_release: str
    labels: tuple[tuple[TaskName, TaskLabel], ...]

    def __post_init__(self) -> None:
        """Enforce the record-level invariants."""
        if not self.accession:
            raise RecordError("accession is empty")
        if self.sequence_normalized != normalize_sequence(self.sequence_raw):
            raise RecordError(f"{self.accession}: sequence_normalized is not the normalised raw")
        if self.sequence_sha256 != sequence_sha256(self.sequence_normalized):
            raise RecordError(f"{self.accession}: sequence_sha256 does not match the sequence")
        if self.length != len(self.sequence_normalized) or self.length < 1:
            raise RecordError(f"{self.accession}: length does not match the sequence")
        if tuple(task for task, _ in self.labels) != TASK_ORDER:
            raise RecordError(f"{self.accession}: labels must cover all seven tasks, in order")

    def label(self, task: TaskName) -> TaskLabel:
        """The label for one task."""
        return dict(self.labels)[task]

    def to_json_obj(self) -> dict[str, object]:
        """The record as a JSON-ready mapping in canonical key order."""
        return {
            "accession": self.accession,
            "entry_name": self.entry_name,
            "sequence_raw": self.sequence_raw,
            "sequence_normalized": self.sequence_normalized,
            "sequence_sha256": self.sequence_sha256,
            "length": self.length,
            "is_fragment": self.is_fragment,
            "organism": self.organism,
            "taxonomy_id": self.taxonomy_id,
            "organism_group": self.organism_group.value,
            "source_dataset": self.source_dataset,
            "source_release": self.source_release,
            "labels": {task.value: label.to_json_obj() for task, label in self.labels},
        }


_RECORD_KEYS: tuple[str, ...] = (
    "accession",
    "entry_name",
    "sequence_raw",
    "sequence_normalized",
    "sequence_sha256",
    "length",
    "is_fragment",
    "organism",
    "taxonomy_id",
    "organism_group",
    "source_dataset",
    "source_release",
    "labels",
)


def to_canonical_line(record: SequenceRecord) -> bytes:
    """Serialise a record to its one canonical line, newline included."""
    text = json.dumps(record.to_json_obj(), ensure_ascii=True, separators=(",", ":"))
    return text.encode("ascii") + b"\n"


def from_canonical_line(line: bytes) -> SequenceRecord:
    """Parse one corpus line, refusing anything that is not canonical."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RecordError(f"not JSON: {exc}") from exc
    mapping = _as_mapping(obj, "record")
    _require_keys(mapping, _RECORD_KEYS, "record")
    label_map = _as_mapping(mapping["labels"], "labels")
    try:
        labels = tuple((TaskName(key), TaskLabel.from_json_obj(v)) for key, v in label_map.items())
        group = OrganismGroup(_as_str(mapping["organism_group"], "organism_group"))
    except ValueError as exc:
        if isinstance(exc, RecordError):
            raise
        raise RecordError(f"unknown enum value: {exc}") from exc
    record = SequenceRecord(
        accession=_as_str(mapping["accession"], "accession"),
        entry_name=_as_str(mapping["entry_name"], "entry_name"),
        sequence_raw=_as_str(mapping["sequence_raw"], "sequence_raw"),
        sequence_normalized=_as_str(mapping["sequence_normalized"], "sequence_normalized"),
        sequence_sha256=_as_str(mapping["sequence_sha256"], "sequence_sha256"),
        length=_as_int(mapping["length"], "length"),
        is_fragment=_as_bool(mapping["is_fragment"], "is_fragment"),
        organism=_as_str(mapping["organism"], "organism"),
        taxonomy_id=_as_int(mapping["taxonomy_id"], "taxonomy_id"),
        organism_group=group,
        source_dataset=_as_str(mapping["source_dataset"], "source_dataset"),
        source_release=_as_str(mapping["source_release"], "source_release"),
        labels=labels,
    )
    if to_canonical_line(record) != line:
        raise RecordError(f"{record.accession}: line is valid but not in canonical form")
    return record


# ── Narrowing helpers: JSON is untyped, the record is not. ──────────────────


def _as_mapping(obj: object, what: str) -> dict[str, object]:
    if not isinstance(obj, dict) or not all(isinstance(k, str) for k in obj):
        raise RecordError(f"{what} must be a JSON object")
    return {str(k): v for k, v in obj.items()}


def _require_keys(mapping: dict[str, object], keys: tuple[str, ...], what: str) -> None:
    if tuple(mapping) != keys:
        raise RecordError(f"{what} keys must be exactly {keys}, got {tuple(mapping)}")


def _as_str(obj: object, what: str) -> str:
    if not isinstance(obj, str):
        raise RecordError(f"{what} must be a string")
    return obj


def _as_int(obj: object, what: str) -> int:
    if not isinstance(obj, int) or isinstance(obj, bool):
        raise RecordError(f"{what} must be an integer")
    return obj


def _as_bool(obj: object, what: str) -> bool:
    if not isinstance(obj, bool):
        raise RecordError(f"{what} must be a boolean")
    return obj


def _as_str_tuple(obj: object, what: str) -> tuple[str, ...]:
    if not isinstance(obj, list) or not all(isinstance(item, str) for item in obj):
        raise RecordError(f"{what} must be a list of strings")
    return tuple(str(item) for item in obj)


__all__ = [
    "TASK_ORDER",
    "RecordError",
    "SequenceRecord",
    "TaskLabel",
    "from_canonical_line",
    "normalize_sequence",
    "sequence_sha256",
    "to_canonical_line",
]
