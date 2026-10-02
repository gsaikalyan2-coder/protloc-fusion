"""Loading the corpus rule files in ``backend/rules/``.

All three files are data (ADR-0006 §2, driver 2). This module parses them into
typed, immutable objects and rejects any malformed file when it is loaded,
before a single entry is parsed. It contains no biology of its own: every term,
lineage token and residue code comes from the YAML.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import yaml

from protloc.domain.types import OrganismGroup, TaskName


class RuleFileError(ValueError):
    """A rule file is missing a key or holds a value of the wrong shape."""


@dataclass(frozen=True, slots=True)
class GroupRule:
    """One first-match-wins rule of ``organism_groups.yaml``."""

    any_of: frozenset[str]
    group: OrganismGroup | None
    exclude: str | None


@dataclass(frozen=True, slots=True)
class GroupRules:
    """The ordered lineage-to-group rules."""

    version: str
    rules: tuple[GroupRule, ...]

    def classify(self, lineage: tuple[str, ...]) -> OrganismGroup | str:
        """Return the group for a lineage, or the exclusion reason as a string."""
        tokens = set(lineage)
        for rule in self.rules:
            if rule.any_of & tokens:
                if rule.group is not None:
                    return rule.group
                if rule.exclude is not None:
                    return rule.exclude
        return "no_rule_matched"


@dataclass(frozen=True, slots=True)
class TermMap:
    """UniProt location terms mapped to classes, with longest-prefix matching."""

    classes: frozenset[str]
    terms: tuple[tuple[str, str], ...]

    def lookup(self, term: str) -> str | None:
        """The class for a term, or ``None`` when no key matches.

        A key matches when the term equals it or continues it with a space or a
        comma; the longest matching key wins.
        """
        best: tuple[int, str] | None = None
        for key, cls in self.terms:
            matches = term == key or term.startswith((key + " ", key + ","))
            if matches and (best is None or len(key) > best[0]):
                best = (len(key), cls)
        return None if best is None else best[1]


@dataclass(frozen=True, slots=True)
class LabelRules:
    """Everything ``uniprot_label_map.yaml`` says."""

    version: str
    applicability: tuple[tuple[TaskName, frozenset[OrganismGroup]], ...]
    localization: TermMap
    prokaryotic_localization: TermMap
    topology: tuple[tuple[str, str], ...]
    membrane_classes: frozenset[str]
    membrane_marker_word: str
    membrane_feature_types: frozenset[str]
    signal_feature_type: str
    signal_value: str
    transit_feature_type: str
    transit_notes: tuple[tuple[str, tuple[TaskName, ...]], ...]
    transit_values: tuple[tuple[TaskName, str], ...]
    fragment_flags: frozenset[str]
    fragment_feature_types: frozenset[str]
    go_database: str
    go_aspect_prefix: str


@dataclass(frozen=True, slots=True)
class ResiduePolicy:
    """What ``residue_policy.yaml`` does with each residue code."""

    version: str
    keep: frozenset[str]
    keep_groups: tuple[tuple[str, frozenset[str]], ...]
    exclude_record: frozenset[str]


@dataclass(frozen=True, slots=True)
class CorpusRules:
    """The three rule files, parsed, with the SHA-256 of each as read."""

    groups: GroupRules
    labels: LabelRules
    residues: ResiduePolicy
    sha256: tuple[tuple[str, str], ...]


GROUPS_FILE = "organism_groups.yaml"
LABELS_FILE = "uniprot_label_map.yaml"
RESIDUES_FILE = "residue_policy.yaml"


def load_corpus_rules(rules_dir: Path) -> CorpusRules:
    """Load and validate every rule file in ``rules_dir``."""
    raw: dict[str, bytes] = {
        name: (rules_dir / name).read_bytes() for name in (GROUPS_FILE, LABELS_FILE, RESIDUES_FILE)
    }
    return CorpusRules(
        groups=_parse_groups(_load(raw[GROUPS_FILE], GROUPS_FILE)),
        labels=_parse_labels(_load(raw[LABELS_FILE], LABELS_FILE)),
        residues=_parse_residues(_load(raw[RESIDUES_FILE], RESIDUES_FILE)),
        sha256=tuple(
            (f"backend/rules/{name}", hashlib.sha256(data).hexdigest())
            for name, data in sorted(raw.items())
        ),
    )


def _load(data: bytes, name: str) -> dict[str, object]:
    obj = yaml.safe_load(data)
    return _mapping(obj, name)


def _parse_groups(doc: dict[str, object]) -> GroupRules:
    rules: list[GroupRule] = []
    for i, item in enumerate(_list(doc, "rules", GROUPS_FILE)):
        where = f"{GROUPS_FILE} rules[{i}]"
        entry = _mapping(item, where)
        group = entry.get("group")
        exclude = entry.get("exclude")
        if (group is None) == (exclude is None):
            raise RuleFileError(f"{where}: exactly one of 'group' or 'exclude' is required")
        try:
            parsed_group = None if group is None else OrganismGroup(_str_value(group, where))
        except ValueError as exc:
            raise RuleFileError(f"{where}: unknown organism group {group!r}") from exc
        rules.append(
            GroupRule(
                any_of=frozenset(_str_list(entry, "any_of", where)),
                group=parsed_group,
                exclude=None if exclude is None else _str_value(exclude, where),
            )
        )
    return GroupRules(version=_str(doc, "version", GROUPS_FILE), rules=tuple(rules))


def _term_map(doc: dict[str, object], section: str) -> TermMap:
    where = f"{LABELS_FILE} {section}"
    body = _section(doc, section)
    classes = frozenset(_str_list(body, "classes", where))
    terms = _str_mapping(body, "terms", where)
    unknown = sorted(set(terms.values()) - classes)
    if unknown:
        raise RuleFileError(f"{where}: terms map to undeclared classes {unknown}")
    return TermMap(classes=classes, terms=tuple(sorted(terms.items())))


def _parse_labels(doc: dict[str, object]) -> LabelRules:
    membrane = _section(doc, "membrane_type")
    membrane_classes = frozenset(_str_list(membrane, "classes", "membrane_type"))
    topology = _str_mapping(membrane, "topology", "membrane_type")
    if not set(topology.values()) <= membrane_classes:
        raise RuleFileError("membrane_type: topology maps to an undeclared class")

    transit = _section(doc, "transit_peptide")
    notes_raw = _mapping(transit.get("notes"), "transit_peptide.notes")
    notes: list[tuple[str, tuple[TaskName, ...]]] = []
    for note, tasks in sorted(notes_raw.items()):
        names = _str_list({"t": tasks}, "t", f"transit_peptide.notes[{note}]")
        notes.append((note, tuple(_transit_task(name) for name in names)))
    values = _str_mapping(transit, "values", "transit_peptide")
    transit_values = tuple((_transit_task(task), value) for task, value in sorted(values.items()))

    signal = _section(doc, "signal_peptide")
    fragment = _section(doc, "fragment")
    go = _section(doc, "go_function")
    return LabelRules(
        version=_str(doc, "version", LABELS_FILE),
        applicability=_parse_applicability(_section(doc, "applicability")),
        localization=_term_map(doc, "localization"),
        prokaryotic_localization=_term_map(doc, "prokaryotic_localization"),
        topology=tuple(sorted(topology.items())),
        membrane_classes=membrane_classes,
        membrane_marker_word=_str(membrane, "membrane_marker_word", "membrane_type"),
        membrane_feature_types=frozenset(
            _str_list(membrane, "membrane_feature_types", "membrane_type")
        ),
        signal_feature_type=_str(signal, "feature_type", "signal_peptide"),
        signal_value=_str(signal, "value", "signal_peptide"),
        transit_feature_type=_str(transit, "feature_type", "transit_peptide"),
        transit_notes=tuple(notes),
        transit_values=transit_values,
        fragment_flags=frozenset(_str_list(fragment, "description_flags", "fragment")),
        fragment_feature_types=frozenset(_str_list(fragment, "feature_types", "fragment")),
        go_database=_str(go, "database", "go_function"),
        go_aspect_prefix=_str(go, "aspect_prefix", "go_function"),
    )


def _parse_applicability(
    body: dict[str, object],
) -> tuple[tuple[TaskName, frozenset[OrganismGroup]], ...]:
    try:
        parsed = {
            TaskName(task): frozenset(
                OrganismGroup(group) for group in _str_list(body, task, "applicability")
            )
            for task in body
        }
    except ValueError as exc:
        if isinstance(exc, RuleFileError):
            raise
        raise RuleFileError(f"applicability: {exc}") from exc
    if set(parsed) != set(TaskName):
        missing = sorted(t.value for t in set(TaskName) - set(parsed))
        raise RuleFileError(f"applicability must name all seven tasks; missing {missing}")
    return tuple((task, parsed[task]) for task in TaskName)


_TRANSIT_TASKS: frozenset[TaskName] = frozenset(
    {TaskName.MITOCHONDRIAL_TRANSIT_PEPTIDE, TaskName.CHLOROPLAST_TRANSIT_PEPTIDE}
)


def _transit_task(name: str) -> TaskName:
    try:
        task = TaskName(name)
    except ValueError as exc:
        raise RuleFileError(f"transit_peptide: unknown task {name!r}") from exc
    if task not in _TRANSIT_TASKS:
        raise RuleFileError(f"transit_peptide: {name!r} is not a transit-peptide task")
    return task


def _parse_residues(doc: dict[str, object]) -> ResiduePolicy:
    keep_section = _section(doc, "keep")
    keep_groups = tuple(
        (group, frozenset(_str_list(keep_section, group, f"{RESIDUES_FILE} keep")))
        for group in sorted(keep_section)
    )
    keep: set[str] = set()
    for _, codes_in_group in keep_groups:
        keep |= codes_in_group
    exclude = set(_str_list(doc, "exclude_record", RESIDUES_FILE))
    codes = keep | exclude
    if any(len(code) != 1 or not code.isupper() for code in codes):
        raise RuleFileError(f"{RESIDUES_FILE}: every code must be one uppercase letter")
    if keep & exclude:
        both = sorted(keep & exclude)
        raise RuleFileError(f"{RESIDUES_FILE}: codes both kept and excluded: {both}")
    return ResiduePolicy(
        version=_str(doc, "version", RESIDUES_FILE),
        keep=frozenset(keep),
        keep_groups=keep_groups,
        exclude_record=frozenset(exclude),
    )


# ── Narrowing helpers ───────────────────────────────────────────────────────


def _mapping(obj: object, where: str) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise RuleFileError(f"{where}: expected a mapping")
    return {str(k): v for k, v in obj.items()}


def _section(doc: dict[str, object], key: str) -> dict[str, object]:
    return _mapping(doc.get(key), key)


def _str_value(obj: object, where: str) -> str:
    if not isinstance(obj, str) or not obj:
        raise RuleFileError(f"{where}: expected a non-empty string")
    return obj


def _str(doc: dict[str, object], key: str, where: str) -> str:
    return _str_value(doc.get(key), f"{where}.{key}")


def _list(doc: dict[str, object], key: str, where: str) -> list[object]:
    value = doc.get(key)
    if not isinstance(value, list) or not value:
        raise RuleFileError(f"{where}.{key}: expected a non-empty list")
    return list(value)


def _str_list(doc: dict[str, object], key: str, where: str) -> list[str]:
    return [_str_value(item, f"{where}.{key}") for item in _list(doc, key, where)]


def _str_mapping(doc: dict[str, object], key: str, where: str) -> dict[str, str]:
    body = _mapping(doc.get(key), f"{where}.{key}")
    return {k: _str_value(v, f"{where}.{key}.{k}") for k, v in body.items()}


__all__ = [
    "CorpusRules",
    "GroupRules",
    "LabelRules",
    "ResiduePolicy",
    "RuleFileError",
    "TermMap",
    "load_corpus_rules",
]
