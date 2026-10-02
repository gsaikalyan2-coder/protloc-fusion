"""One Swiss-Prot flat-file entry → one corpus record, under ADR-0006.

Pure: no file or network I/O, and no state kept between entries. Biopython
(``Bio.SwissProt``) does the flat-file parsing. This module maps the parsed
entry to ProtLoc labels, and every term, token and code it compares against
comes from :class:`~protloc.data.rules.CorpusRules`.

An entry ends up one of three ways:

* **a record**: every one of the seven labels is set, each in one of the four
  states;
* **excluded**, with a reason the build counts (no organism group, a residue
  code the policy excludes);
* **an error**: the input is not what the parser assumes (an unreviewed entry,
  a character outside the residue policy). The build stops, because a corpus
  built from misread input has a hash that means nothing.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import re

from Bio.SeqFeature import ExactPosition, SeqFeature
from Bio.SwissProt import Record as SwissProtRecord

from protloc.data.records import (
    TASK_ORDER,
    SequenceRecord,
    TaskLabel,
    normalize_sequence,
    sequence_sha256,
)
from protloc.data.rules import CorpusRules, LabelRules, TermMap
from protloc.domain.types import LabelState, OrganismGroup, TaskName

#: The value written to ``source_dataset`` for every record (decision D1).
SOURCE_DATASET = "uniprotkb_swissprot"

_REVIEWED = "Reviewed"
_SUBCELLULAR = "SUBCELLULAR LOCATION: "
_ISOFORM_PREFIX = "["
_ECO = re.compile(r"ECO:\d{7}")
_BRACES = re.compile(r"\{[^}]*\}")
_NOTE = re.compile(r"\s*Note=")
_STATEMENT_END = re.compile(r"\.(?:\s+|$)")
_FLAGS = "Flags:"
#: A ``DR   GO`` cross-reference is (database, id, aspect:name, evidence:source).
_GO_FIELDS = 4


class EntryError(ValueError):
    """The entry is not what a Swiss-Prot release should contain."""


@dataclass(frozen=True, slots=True)
class Statement:
    """One sentence of a subcellular-location block: a term and its qualifiers."""

    term: str
    term_evidence: frozenset[str]
    qualifiers: tuple[tuple[str, frozenset[str]], ...]


@dataclass(frozen=True, slots=True)
class _EntryFacts:
    """What the label rules read from one entry."""

    group: OrganismGroup
    is_fragment: bool
    statements: list[Statement]
    features: list[SeqFeature]
    cross_references: list[tuple[str, ...]]


@dataclass(slots=True)
class Diagnostics:
    """Vocabulary the rule files did not cover. Counted, never silently dropped."""

    unmapped_location_terms: list[str] = field(default_factory=list)
    unmapped_transit_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EntryOutcome:
    """The result of parsing one entry."""

    accession: str
    record: SequenceRecord | None
    excluded: str | None
    diagnostics: Diagnostics


def parse_entry(entry: SwissProtRecord, release: str, rules: CorpusRules) -> EntryOutcome:
    """Map one parsed Swiss-Prot entry to a record or an exclusion."""
    accessions: list[str] = list(entry.accessions)
    if not accessions:
        raise EntryError(f"entry {entry.entry_name!r} has no accession")
    accession = accessions[0]
    if entry.data_class != _REVIEWED:
        raise EntryError(f"{accession}: data class {entry.data_class!r}, expected Reviewed")
    diagnostics = Diagnostics()

    group = rules.groups.classify(tuple(entry.organism_classification))
    if not isinstance(group, OrganismGroup):
        return EntryOutcome(accession, None, f"organism_group:{group}", diagnostics)

    raw: str = entry.sequence
    normalized = normalize_sequence(raw)
    observed = set(normalized)
    unknown = sorted(observed - rules.residues.keep - rules.residues.exclude_record)
    if unknown:
        raise EntryError(f"{accession}: residue codes outside the residue policy: {unknown}")
    excluded_codes = sorted(observed & rules.residues.exclude_record)
    if excluded_codes:
        return EntryOutcome(accession, None, f"residue:{''.join(excluded_codes)}", diagnostics)

    features: list[SeqFeature] = list(entry.features)
    lab = rules.labels
    is_fragment = _is_fragment(_text(entry.description), features, lab)
    statements = _location_statements(entry.comments)
    facts = _EntryFacts(
        group=group,
        is_fragment=is_fragment,
        statements=statements,
        features=features,
        cross_references=[tuple(ref) for ref in entry.cross_references],
    )
    labels = _labels(facts, lab, diagnostics)

    taxonomy_ids: list[str] = list(entry.taxonomy_id)
    if len(taxonomy_ids) != 1 or not taxonomy_ids[0].isdigit():
        raise EntryError(f"{accession}: expected one numeric NCBI_TaxID, got {taxonomy_ids}")

    record = SequenceRecord(
        accession=accession,
        entry_name=_text(entry.entry_name),
        sequence_raw=raw,
        sequence_normalized=normalized,
        sequence_sha256=sequence_sha256(normalized),
        length=len(normalized),
        is_fragment=is_fragment,
        organism=_text(entry.organism).strip().rstrip("."),
        taxonomy_id=int(taxonomy_ids[0]),
        organism_group=group,
        source_dataset=SOURCE_DATASET,
        source_release=release,
        labels=tuple((task, labels[task]) for task in TASK_ORDER),
    )
    return EntryOutcome(accession, record, None, diagnostics)


def _text(value: object) -> str:
    """A flat-file text field as one string.

    Biopython declares some ``Record`` text fields loosely (its own annotations
    say list or ``None``) while filling them with strings; this narrows either
    shape without guessing.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    raise EntryError(f"unexpected flat-file field value {value!r}")


# ── Record-level facts ──────────────────────────────────────────────────────


def _is_fragment(description: str, features: list[SeqFeature], lab: LabelRules) -> bool:
    flags: set[str] = set()
    if _FLAGS in description:
        tail = description.split(_FLAGS, 1)[1]
        flags = {part.strip() for part in tail.split(";") if part.strip()}
    if flags & lab.fragment_flags:
        return True
    return any(feature.type in lab.fragment_feature_types for feature in features)


def _eco_codes(text: str) -> frozenset[str]:
    return frozenset(_ECO.findall(text))


def _location_statements(comments: Iterable[str]) -> list[Statement]:
    """Statements from every non-isoform subcellular-location block (ADR-0006 §3.4)."""
    statements: list[Statement] = []
    for comment in comments:
        if not comment.startswith(_SUBCELLULAR):
            continue
        body = comment[len(_SUBCELLULAR) :].strip()
        if body.startswith(_ISOFORM_PREFIX):
            continue
        body = _NOTE.split(body, maxsplit=1)[0]
        for sentence in _STATEMENT_END.split(body):
            if not sentence.strip():
                continue
            parts = [_split_component(component) for component in sentence.split(";")]
            parts = [part for part in parts if part[0]]
            if not parts:
                continue
            (term, evidence), *qualifiers = parts
            statements.append(Statement(term, evidence, tuple(qualifiers)))
    return statements


def _split_component(component: str) -> tuple[str, frozenset[str]]:
    return _BRACES.sub("", component).strip(), _eco_codes(component)


# ── Labels ──────────────────────────────────────────────────────────────────


def _labels(
    facts: _EntryFacts, lab: LabelRules, diagnostics: Diagnostics
) -> dict[TaskName, TaskLabel]:
    group, is_fragment = facts.group, facts.is_fragment
    statements, features = facts.statements, facts.features
    applicable = dict(lab.applicability)
    labels: dict[TaskName, TaskLabel] = {}

    location_map = (
        lab.localization
        if TaskName.LOCALIZATION in _tasks_for(group, applicable)
        else lab.prokaryotic_localization
    )
    location = _location_label(statements, location_map, diagnostics)
    has_mapped_location = location.state is LabelState.ANNOTATED
    for task in (TaskName.LOCALIZATION, TaskName.PROKARYOTIC_LOCALIZATION):
        labels[task] = location if group in applicable[task] else _masked()

    labels[TaskName.MEMBRANE_TYPE] = _membrane_label(
        statements, features, is_fragment, has_mapped_location, lab
    )
    labels[TaskName.SIGNAL_PEPTIDE] = _signal_label(features, is_fragment, lab)
    transit = _transit_labels(features, is_fragment, lab, diagnostics)
    for task, label in transit.items():
        labels[task] = label
    labels[TaskName.GO_FUNCTION] = _go_label(facts.cross_references, lab)

    return {
        task: label if group in applicable[task] else _masked() for task, label in labels.items()
    }


def _tasks_for(
    group: OrganismGroup, applicable: dict[TaskName, frozenset[OrganismGroup]]
) -> set[TaskName]:
    return {task for task, groups in applicable.items() if group in groups}


def _masked() -> TaskLabel:
    return TaskLabel(LabelState.NOT_APPLICABLE)


def _location_label(
    statements: list[Statement], term_map: TermMap, diagnostics: Diagnostics
) -> TaskLabel:
    evidence: dict[str, set[str]] = {}
    for statement in statements:
        cls = term_map.lookup(statement.term)
        if cls is None:
            diagnostics.unmapped_location_terms.append(statement.term)
            continue
        evidence.setdefault(cls, set()).update(statement.term_evidence)
    if not evidence:
        return TaskLabel(LabelState.UNKNOWN)
    return TaskLabel.build(
        LabelState.ANNOTATED,
        set(evidence),
        evidence={cls: codes for cls, codes in evidence.items() if codes},
    )


def _membrane_label(
    statements: list[Statement],
    features: list[SeqFeature],
    is_fragment: bool,
    has_mapped_location: bool,
    lab: LabelRules,
) -> TaskLabel:
    topology = dict(lab.topology)
    evidence: dict[str, set[str]] = {}
    for statement in statements:
        for qualifier, codes in statement.qualifiers:
            cls = topology.get(qualifier)
            if cls is not None:
                evidence.setdefault(cls, set()).update(codes)
    if evidence:
        return TaskLabel.build(
            LabelState.ANNOTATED,
            set(evidence),
            evidence={cls: codes for cls, codes in evidence.items() if codes},
        )
    marker = re.compile(rf"\b{re.escape(lab.membrane_marker_word)}\b", re.IGNORECASE)
    membrane_term = any(marker.search(statement.term) for statement in statements)
    membrane_feature = any(f.type in lab.membrane_feature_types for f in features)
    if has_mapped_location and not is_fragment and not membrane_term and not membrane_feature:
        return TaskLabel(LabelState.NEGATIVE)
    return TaskLabel(LabelState.UNKNOWN)


def _exact_n_terminal_end(feature: SeqFeature) -> int | None:
    """The 1-based end of a feature that starts at residue 1 with exact ends."""
    location = feature.location
    if location is None:
        return None
    start, end = location.start, location.end
    if type(start) is not ExactPosition or type(end) is not ExactPosition:
        return None
    if int(start) != 0:
        return None
    return int(end)


def _feature_codes(feature: SeqFeature) -> set[str]:
    raw = feature.qualifiers.get("evidence", "")
    text = " ".join(raw) if isinstance(raw, list) else str(raw)
    return set(_eco_codes(text))


def _signal_label(features: list[SeqFeature], is_fragment: bool, lab: LabelRules) -> TaskLabel:
    signals = [f for f in features if f.type == lab.signal_feature_type]
    return _peptide_label(signals, is_fragment, lab.signal_value, no_feature_of_kind=not signals)


def _peptide_label(
    candidates: list[SeqFeature], is_fragment: bool, value: str, *, no_feature_of_kind: bool
) -> TaskLabel:
    if no_feature_of_kind:
        return TaskLabel(LabelState.UNKNOWN if is_fragment else LabelState.NEGATIVE)
    ends = {_exact_n_terminal_end(f) for f in candidates}
    if len(ends) != 1 or None in ends:
        return TaskLabel(LabelState.UNKNOWN)
    (end,) = ends
    codes: set[str] = set()
    for feature in candidates:
        codes |= _feature_codes(feature)
    return TaskLabel.build(
        LabelState.ANNOTATED,
        {value},
        cleavage_site=end,
        evidence={value: codes} if codes else None,
    )


def _transit_labels(
    features: list[SeqFeature], is_fragment: bool, lab: LabelRules, diagnostics: Diagnostics
) -> dict[TaskName, TaskLabel]:
    notes = dict(lab.transit_notes)
    transits = [f for f in features if f.type == lab.transit_feature_type]
    per_task: dict[TaskName, list[SeqFeature]] = {task: [] for task, _ in lab.transit_values}
    for feature in transits:
        note = _note(feature)
        tasks = notes.get(note)
        if tasks is None:
            diagnostics.unmapped_transit_notes.append(note)
            continue
        for task in tasks:
            per_task[task].append(feature)
    result: dict[TaskName, TaskLabel] = {}
    for task, value in lab.transit_values:
        if per_task[task]:
            result[task] = _peptide_label(
                per_task[task], is_fragment, value, no_feature_of_kind=False
            )
        elif transits:
            # A transit peptide exists but not for this task: ADR-0006 §3.3 grants
            # a negative only when there is no TRANSIT feature at all.
            result[task] = TaskLabel(LabelState.UNKNOWN)
        else:
            result[task] = _peptide_label([], is_fragment, value, no_feature_of_kind=True)
    return result


def _note(feature: SeqFeature) -> str:
    raw = feature.qualifiers.get("note", "")
    text = " ".join(raw) if isinstance(raw, list) else str(raw)
    return _BRACES.sub("", text).strip()


def _go_label(cross_references: Iterable[tuple[str, ...]], lab: LabelRules) -> TaskLabel:
    evidence: dict[str, set[str]] = {}
    for ref in cross_references:
        if (
            len(ref) < _GO_FIELDS
            or ref[0] != lab.go_database
            or not ref[2].startswith(lab.go_aspect_prefix)
        ):
            continue
        code = ref[3].split(":", 1)[0].strip()
        evidence.setdefault(ref[1], set())
        if code:
            evidence[ref[1]].add(code)
    if not evidence:
        return TaskLabel(LabelState.UNKNOWN)
    return TaskLabel.build(
        LabelState.ANNOTATED,
        set(evidence),
        evidence={term: codes for term, codes in evidence.items() if codes},
    )


__all__ = ["SOURCE_DATASET", "Diagnostics", "EntryError", "EntryOutcome", "parse_entry"]
