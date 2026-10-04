"""Phase 12 GO head: configuration, ontology, labels, rows, metrics, protocol and manifest.

Everything here runs without torch. ``protloc.training.go_net`` holds the network and the
training loop and imports torch lazily; this module holds what CI can import and test on a
runner with neither torch nor a GPU:

* the configuration (``configs/training/go.yaml``) parsed and checked, and the grid,
* the Gene Ontology file parsed in plain Python, corpus term ids resolved against it
  (``alt_id``, ``replaced_by``, obsolete, absent, other aspect) and ancestors propagated,
* the label table read from the corpus, the vocabulary counted on SUBSET rows only and the
  rows the head is trained and scored on (ADR-0006: GO is open-world),
* the metrics: precision@k, recall@k, Fmax with propagation, per-term AUC and AUPR, coverage,
  with cluster-bootstrap intervals,
* the floors (frequency prior, Phase 8 logistic regression, motif hits), the protocol (nested
  global threshold, the selection rule) and the frozen manifest checks (ADR-0014).

``training/`` is an entry-point layer: nothing imports it.
"""

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from protloc.calibration.calibrator import LabelCalibrator
from protloc.training import baseline_models as bm
from protloc.training import calibration as cal
from protloc.training import head
from protloc.training.baselines import (
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    Fold,
    check_subset_block,
    digest,
)
from protloc.training.head import Architecture, Schedule, SmokeSettings

Array = Any
SCHEMA_VERSION = 1
PHASE = 12
TASK = "go_function"
SOURCE_MODEL = "model"
SOURCE_MOTIF = "motif"
STATE_ANNOTATED = "annotated"
STATE_UNKNOWN = "unknown"
#: Order in which the grid expands: the first key varies slowest.
GRID_KEYS: tuple[str, ...] = ("learning_rate", "dropout", "hidden", "gamma")
RELATIONS: tuple[str, ...] = ("is_a", "part_of")
#: What happens to a corpus term id when it is looked up in the ontology.
CASE_PRIMARY = "primary"
CASE_ALT = "alt_id"
CASE_REPLACED = "replaced_by"
CASE_OBSOLETE = "obsolete_dropped"
CASE_ABSENT = "absent"
CASE_ASPECT = "other_aspect"
CASES: tuple[str, ...] = (
    CASE_PRIMARY,
    CASE_ALT,
    CASE_REPLACED,
    CASE_OBSOLETE,
    CASE_ABSENT,
    CASE_ASPECT,
)
#: The scored arms, in report order.
ARMS: tuple[str, ...] = ("model_raw", "model_calibrated", "frequency_prior", "logreg", "motif")
LABEL_COLUMNS: tuple[str, ...] = (
    "accession",
    "state",
    "direct",
    "dropped",
    "motif",
    "propagated",
    "motif_propagated",
)
LIST_SEPARATOR = ";"
CODE_SEPARATOR = ","
PAIR_SEPARATOR = "="
#: Divisor for parity and for the mean-rank AUC terms.
EVEN_DIVISOR = 2
#: Cells of one label-table row.
LABEL_WIDTH = len(LABEL_COLUMNS)
EPSILON = 1e-12
MOTIF_CELL = 4
_SHA = re.compile(r"[0-9a-f]{64}")
_ACCESSION_KEY = b'"accession":"'
_GO_KEY = b'"go_function":'


# -- configuration ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OntologyPin:
    """The ontology release the labels are mapped onto, and how it is propagated."""

    file: str
    data_version: str
    sha256: str
    size: int
    term_stanzas: int
    obsolete_stanzas: int
    namespace: str
    root: str
    relations: tuple[str, ...]
    license: str
    license_url: str


@dataclass(frozen=True, slots=True)
class EvidenceRules:
    """GO evidence codes grouped into classes, and which classes are labels."""

    classes: Mapping[str, tuple[str, ...]]
    policy: tuple[str, ...]

    @property
    def policy_codes(self) -> frozenset[str]:
        """Every code in a policy class."""
        return frozenset(code for name in self.policy for code in self.classes[name])

    def class_of(self, code: str) -> str:
        """The class a code belongs to; an unknown code is an error, never a guess."""
        for name, codes in self.classes.items():
            if code in codes:
                return name
        raise BaselineError(f"evidence code {code!r} is in no evidence class")


@dataclass(frozen=True, slots=True)
class MetricSettings:
    """How the head is scored."""

    k_values: tuple[int, ...]
    primary_k: int
    per_term_min_positives: int
    max_threshold_candidates: int


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One hyperparameter setting."""

    learning_rate: float
    dropout: float
    hidden: int
    gamma: float

    def as_dict(self) -> dict[str, float | int]:
        """The setting as plain JSON-able values, in grid order."""
        return {
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "hidden": self.hidden,
            "gamma": self.gamma,
        }


@dataclass(frozen=True, slots=True)
class GoConfig:
    """``configs/training/go.yaml``, parsed and checked."""

    seed: int
    ontology: OntologyPin
    evidence: EvidenceRules
    term_floor: int
    closed_world: str
    architecture: Architecture
    pos_weight_cap: float
    weight_decay: float
    schedule: Schedule
    grid: Mapping[str, tuple[float, ...]]
    metrics: MetricSettings
    reproduction_tolerance: float
    motif_tiebreak_weight: float
    bundle_version: str
    bundle_directory: str
    vocabulary_file: str
    calibration_config: str
    platt_fallback: bool
    smoke: SmokeSettings
    sha256: str
    labels_sha256: str


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{where} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise BaselineError(f"{where} must be a positive number")
    return float(value)


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BaselineError(f"{where} must be a positive integer")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineError(f"{where} must be a non-empty string")
    return value


def _names(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise BaselineError(f"{where} must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise BaselineError(f"{where} must not repeat a name")
    return tuple(value)


def _sha(value: object, where: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise BaselineError(f"{where} must be a SHA-256")
    return value


def _ontology(doc: Mapping[str, Any]) -> OntologyPin:
    relations = _names(doc.get("propagate_over"), "ontology.propagate_over")
    if not set(relations) <= set(RELATIONS):
        raise BaselineError(f"ontology.propagate_over must be among {list(RELATIONS)}")
    return OntologyPin(
        file=_text(doc.get("file"), "ontology.file"),
        data_version=_text(doc.get("data_version"), "ontology.data_version"),
        sha256=_sha(doc.get("sha256"), "ontology.sha256"),
        size=_count(doc.get("bytes"), "ontology.bytes"),
        term_stanzas=_count(doc.get("term_stanzas"), "ontology.term_stanzas"),
        obsolete_stanzas=_count(doc.get("obsolete_stanzas"), "ontology.obsolete_stanzas"),
        namespace=_text(doc.get("namespace"), "ontology.namespace"),
        root=_text(doc.get("root"), "ontology.root"),
        relations=relations,
        license=_text(doc.get("license"), "ontology.license"),
        license_url=_text(doc.get("license_url"), "ontology.license_url"),
    )


def _evidence(doc: Mapping[str, Any]) -> EvidenceRules:
    raw = _mapping(doc.get("evidence_classes"), "labels.evidence_classes")
    classes = {
        name: _names(codes, f"labels.evidence_classes.{name}") for name, codes in raw.items()
    }
    every = [code for codes in classes.values() for code in codes]
    if len(set(every)) != len(every):
        raise BaselineError("an evidence code is in more than one class")
    policy = _names(doc.get("evidence_policy"), "labels.evidence_policy")
    if not set(policy) <= set(classes):
        raise BaselineError("labels.evidence_policy names an unknown evidence class")
    return EvidenceRules(classes=classes, policy=policy)


def _grid(doc: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    if set(doc) != set(GRID_KEYS):
        raise BaselineError(f"grid must have exactly the keys {list(GRID_KEYS)}")
    grid: dict[str, tuple[float, ...]] = {}
    for key in GRID_KEYS:
        values = doc[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise BaselineError(f"grid.{key} must be a non-empty list of distinct values")
        grid[key] = tuple(float(v) for v in values)
    if any(not 0.0 <= d < 1.0 for d in grid["dropout"]):
        raise BaselineError("grid.dropout values must be in [0, 1)")
    if any(g < 0.0 for g in grid["gamma"]):
        raise BaselineError("grid.gamma values must be >= 0")
    if any(h < 1 or h != int(h) for h in grid["hidden"]):
        raise BaselineError("grid.hidden values must be positive integers")
    if any(lr <= 0.0 for lr in grid["learning_rate"]):
        raise BaselineError("grid.learning_rate values must be positive")
    return grid


def _metrics(doc: Mapping[str, Any]) -> MetricSettings:
    ks = doc.get("k_values")
    if (
        not isinstance(ks, list)
        or not ks
        or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in ks)
        or sorted(set(ks)) != ks
    ):
        raise BaselineError("metrics.k_values must be increasing positive integers")
    primary = _count(doc.get("primary_k"), "metrics.primary_k")
    if primary not in ks:
        raise BaselineError("metrics.primary_k must be one of metrics.k_values")
    return MetricSettings(
        k_values=tuple(ks),
        primary_k=primary,
        per_term_min_positives=_count(
            doc.get("per_term_min_positives"), "metrics.per_term_min_positives"
        ),
        max_threshold_candidates=_count(
            doc.get("max_threshold_candidates"), "metrics.max_threshold_candidates"
        ),
    )


def load_config(path: Path) -> GoConfig:
    """Parse the GO configuration and reject anything malformed before a fit starts."""
    raw = path.read_bytes()
    doc = _mapping(yaml.safe_load(raw), str(path))
    if doc.get("name") != "go" or doc.get("task") != TASK:
        raise BaselineError("go.yaml must be name 'go' for task 'go_function'")
    if _mapping(doc.get("optimiser"), "optimiser").get("name") != "adamw":
        raise BaselineError("optimiser.name must be 'adamw'")
    labels = _mapping(doc.get("labels"), "labels")
    arch = _mapping(doc.get("architecture"), "architecture")
    schedule = _mapping(doc.get("schedule"), "schedule")
    fraction = _positive(schedule.get("inner_holdout_fraction"), "schedule.inner_holdout_fraction")
    if fraction >= 1.0:
        raise BaselineError("schedule.inner_holdout_fraction must be below 1")
    bundle = _mapping(doc.get("bundle"), "bundle")
    calibration = _mapping(doc.get("calibration"), "calibration")
    smoke = _mapping(doc.get("smoke"), "smoke")
    ontology = _ontology(_mapping(doc.get("ontology"), "ontology"))
    return GoConfig(
        seed=_count(doc.get("seed"), "seed"),
        ontology=ontology,
        evidence=_evidence(labels),
        term_floor=_count(labels.get("term_floor"), "labels.term_floor"),
        closed_world=_text(labels.get("closed_world"), "labels.closed_world"),
        architecture=Architecture(
            heads=_count(arch.get("heads"), "architecture.heads"),
            attention_hidden=_count(arch.get("attention_hidden"), "architecture.attention_hidden"),
            embedding_dim=_count(arch.get("embedding_dim"), "architecture.embedding_dim"),
        ),
        pos_weight_cap=_positive(
            _mapping(doc.get("loss"), "loss").get("pos_weight_cap"), "loss.pos_weight_cap"
        ),
        weight_decay=_positive(
            _mapping(doc.get("optimiser"), "optimiser").get("weight_decay"), "weight_decay"
        ),
        schedule=Schedule(
            batch_size=_count(schedule.get("batch_size"), "schedule.batch_size"),
            max_epochs=_count(schedule.get("max_epochs"), "schedule.max_epochs"),
            patience=_count(schedule.get("patience"), "schedule.patience"),
            inner_holdout_fraction=fraction,
        ),
        grid=_grid(_mapping(doc.get("grid"), "grid")),
        metrics=_metrics(_mapping(doc.get("metrics"), "metrics")),
        reproduction_tolerance=_positive(
            _mapping(doc.get("evaluation"), "evaluation").get("reproduction_tolerance"),
            "evaluation.reproduction_tolerance",
        ),
        motif_tiebreak_weight=_positive(
            _mapping(doc.get("floors"), "floors").get("motif_tiebreak_weight"),
            "floors.motif_tiebreak_weight",
        ),
        bundle_version=_text(bundle.get("version"), "bundle.version"),
        bundle_directory=_text(bundle.get("directory"), "bundle.directory"),
        vocabulary_file=_text(bundle.get("vocabulary_file"), "bundle.vocabulary_file"),
        calibration_config=_text(calibration.get("config"), "calibration.config"),
        platt_fallback=bool(calibration.get("platt_fallback")),
        smoke=SmokeSettings(
            stride=_count(smoke.get("stride"), "smoke.stride"),
            max_epochs=_count(smoke.get("max_epochs"), "smoke.max_epochs"),
            grid_points=_count(smoke.get("grid_points"), "smoke.grid_points"),
        ),
        sha256=hashlib.sha256(raw).hexdigest(),
        labels_sha256=digest({"ontology": doc.get("ontology"), "labels": doc.get("labels")}),
    )


def grid_points(config: GoConfig) -> list[GridPoint]:
    """Every grid point in expansion order (the first key varies slowest)."""
    return [
        GridPoint(learning_rate=lr, dropout=dropout, hidden=int(hidden), gamma=gamma)
        for lr, dropout, hidden, gamma in itertools.product(*(config.grid[k] for k in GRID_KEYS))
    ]


def _np() -> Any:
    return importlib.import_module("numpy")


# -- the ontology ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Term:
    """One ``[Term]`` stanza: only the fields the labels need."""

    id: str
    name: str
    namespace: str
    obsolete: bool
    replaced_by: tuple[str, ...]
    is_a: tuple[str, ...]
    part_of: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ontology:
    """The parsed ontology: terms by id, ``alt_id`` to primary id, and the stanza counts."""

    data_version: str
    terms: Mapping[str, Term]
    alt: Mapping[str, str]
    stanzas: int
    obsolete: int


@dataclass(slots=True)
class _Stanza:
    """A ``[Term]`` stanza while it is being read."""

    id: str = ""
    name: str = ""
    namespace: str = ""
    obsolete: bool = False
    alt: list[str] = field(default_factory=list)
    replaced_by: list[str] = field(default_factory=list)
    is_a: list[str] = field(default_factory=list)
    part_of: list[str] = field(default_factory=list)

    def add(self, key: str, value: str) -> None:
        """Record one ``key: value`` line; keys the labels do not use are ignored."""
        if key == "id":
            self.id = value
        elif key == "name":
            self.name = value
        elif key == "namespace":
            self.namespace = value
        elif key == "is_obsolete":
            self.obsolete = value == "true"
        elif key == "alt_id":
            self.alt.append(value.split(maxsplit=1)[0])
        elif key == "replaced_by":
            self.replaced_by.append(value.split(maxsplit=1)[0])
        elif key == "is_a":
            self.is_a.append(value.split(maxsplit=1)[0])
        elif key == "relationship":
            relation = value.split()
            if relation[0] == "part_of":
                self.part_of.append(relation[1])


def parse_obo(text: str) -> Ontology:
    """Read an OBO file in plain Python (no goatools, no obonet).

    Only ``[Term]`` stanzas are kept. ``is_a`` and ``relationship: part_of`` are the edges the
    true-path rule follows; ``alt_id`` maps a secondary id to the term that now carries it.
    """
    data_version = ""
    stanzas: list[_Stanza] = []
    current: _Stanza | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            current = _Stanza() if line == "[Term]" else None
            if current is not None:
                stanzas.append(current)
            continue
        key, separator, value = line.partition(": ")
        if not separator:
            continue
        if current is None:
            if key == "data-version":
                data_version = value
            continue
        current.add(key, value)
    terms: dict[str, Term] = {}
    alt: dict[str, str] = {}
    for stanza in stanzas:
        if not stanza.id or stanza.id in terms:
            raise BaselineError(
                f"the ontology has a [Term] without an id or with a repeated id: {stanza.id!r}"
            )
        terms[stanza.id] = Term(
            id=stanza.id,
            name=stanza.name,
            namespace=stanza.namespace,
            obsolete=stanza.obsolete,
            replaced_by=tuple(stanza.replaced_by),
            is_a=tuple(stanza.is_a),
            part_of=tuple(stanza.part_of),
        )
        if not stanza.obsolete:
            for secondary in stanza.alt:
                alt[secondary] = stanza.id
    return Ontology(
        data_version=data_version,
        terms=terms,
        alt=alt,
        stanzas=len(stanzas),
        obsolete=sum(1 for t in terms.values() if t.obsolete),
    )


def ontology_problems(onto: Ontology, pin: OntologyPin) -> list[str]:
    """Every way the parsed ontology differs from the release the configuration pins."""
    found: list[str] = []
    if onto.data_version != pin.data_version:
        found.append(f"data-version is {onto.data_version!r}, not {pin.data_version!r}")
    if onto.stanzas != pin.term_stanzas:
        found.append(f"{onto.stanzas} [Term] stanzas, not {pin.term_stanzas}")
    if onto.obsolete != pin.obsolete_stanzas:
        found.append(f"{onto.obsolete} obsolete stanzas, not {pin.obsolete_stanzas}")
    root = onto.terms.get(pin.root)
    if root is None or root.obsolete or root.namespace != pin.namespace:
        found.append(f"the root {pin.root} is not a live {pin.namespace} term")
    return found


class Ancestry:
    """Ancestors over the configured relations, inside one aspect, with a memo.

    The true-path rule: whatever is annotated with a term is annotated with every ancestor.
    Obsolete terms and parents in another aspect are never followed.
    """

    def __init__(self, onto: Ontology, pin: OntologyPin) -> None:
        """Bind the ontology and the relations to follow."""
        self._onto = onto
        self._relations = pin.relations
        self._namespace = pin.namespace
        self._root = pin.root
        self._memo: dict[str, frozenset[str]] = {}

    @property
    def root(self) -> str:
        """The aspect root, which is never a label."""
        return self._root

    def of(self, term: str) -> frozenset[str]:
        """Every proper ancestor of a term (the root included), or none for an unknown term."""
        known = self._memo.get(term)
        if known is not None:
            return known
        node = self._onto.terms.get(term)
        found: set[str] = set()
        if node is not None and not node.obsolete:
            for relation in self._relations:
                for parent in getattr(node, relation):
                    target = self._onto.terms.get(parent)
                    if target is None or target.obsolete or target.namespace != self._namespace:
                        continue
                    found.add(parent)
                    found |= self.of(parent)
        result = frozenset(found)
        self._memo[term] = result
        return result

    def propagate(self, terms: Collection[str]) -> frozenset[str]:
        """The terms and all their ancestors, without the root."""
        closure: set[str] = set(terms)
        for term in terms:
            closure |= self.of(term)
        closure.discard(self._root)
        return frozenset(closure)


def cross_aspect_edges(onto: Ontology, pin: OntologyPin) -> int:
    """Edges of live terms of the aspect that lead to a term of another aspect (measured: 0)."""
    crossing = 0
    for term in onto.terms.values():
        if term.obsolete or term.namespace != pin.namespace:
            continue
        for relation in pin.relations:
            for parent in getattr(term, relation):
                target = onto.terms.get(parent)
                if target is not None and target.namespace != pin.namespace:
                    crossing += 1
    return crossing


@dataclass(frozen=True, slots=True)
class Resolution:
    """What a corpus term id became: a case and, for a kept term, the primary id."""

    case: str
    term: str | None


def resolve_term(onto: Ontology, term_id: str, namespace: str) -> Resolution:
    """Map a corpus GO id onto the pinned release. Nothing is guessed.

    A live id is itself. An ``alt_id`` maps to its primary id. An obsolete id follows
    ``replaced_by`` only when there is exactly one target and that target is live; otherwise it is
    dropped. An id the release does not contain is dropped, and so is a kept term outside the
    aspect.
    """
    node = onto.terms.get(term_id)
    if node is not None and not node.obsolete:
        case, target = CASE_PRIMARY, term_id
    elif term_id in onto.alt:
        case, target = CASE_ALT, onto.alt[term_id]
    elif node is not None:
        replaced = node.replaced_by
        replacement = onto.terms.get(replaced[0]) if len(replaced) == 1 else None
        if replacement is None or replacement.obsolete:
            return Resolution(CASE_OBSOLETE, None)
        case, target = CASE_REPLACED, replaced[0]
    else:
        return Resolution(CASE_ABSENT, None)
    if onto.terms[target].namespace != namespace:
        return Resolution(CASE_ASPECT, None)
    return Resolution(case, target)


def mapping_report(
    occurrences: Mapping[str, int], onto: Ontology, namespace: str
) -> dict[str, Any]:
    """How many term occurrences, and how many distinct ids, fell into each resolution case."""
    by_case: Counter[str] = Counter()
    distinct: Counter[str] = Counter()
    for term_id, count in occurrences.items():
        case = resolve_term(onto, term_id, namespace).case
        by_case[case] += count
        distinct[case] += 1
    return {
        "occurrences": {case: by_case[case] for case in CASES},
        "distinct_ids": {case: distinct[case] for case in CASES},
        "total_occurrences": sum(by_case.values()),
        "total_distinct_ids": sum(distinct.values()),
    }


# -- labels from the corpus ------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RawRecord:
    """What the corpus says about one wanted accession, before the ontology is applied."""

    accession: str
    group: str
    sequence: str
    state: str
    values: tuple[str, ...]
    evidence: Mapping[str, tuple[str, ...]]


@dataclass(slots=True)
class CorpusScan:
    """One streaming pass over the corpus: whole-corpus tallies and the wanted records."""

    lines: int = 0
    states: Counter[str] = field(default_factory=Counter)
    term_occurrences: Counter[str] = field(default_factory=Counter)
    code_occurrences: Counter[str] = field(default_factory=Counter)
    records: dict[str, RawRecord] = field(default_factory=dict)


def _go_label(line: bytes) -> dict[str, Any]:
    start = line.find(_GO_KEY)
    if start < 0:
        raise BaselineError("a corpus line has no go_function label")
    text = line[start + len(_GO_KEY) :].decode("utf-8").lstrip()
    value, _end = json.JSONDecoder().raw_decode(text)
    if not isinstance(value, dict):
        raise BaselineError("go_function label is not an object")
    return {str(k): v for k, v in value.items()}


def _raw_record(line: bytes, label: Mapping[str, Any]) -> RawRecord:
    record = json.loads(line)
    evidence = label.get("evidence") or {}
    return RawRecord(
        accession=str(record["accession"]),
        group=str(record["organism_group"]),
        sequence=str(record["sequence_normalized"]),
        state=str(label["state"]),
        values=tuple(str(v) for v in label.get("values") or ()),
        evidence={str(t): tuple(str(c) for c in codes) for t, codes in evidence.items()},
    )


def scan_corpus(path: Path, wanted: Collection[str]) -> CorpusScan:
    """Stream the canonical corpus once (pure Python).

    Every line's ``go_function`` label is tallied (states, term ids, evidence codes) for the
    whole-corpus mapping report. A line is parsed in full only when its accession is wanted. Every
    wanted accession must be found or the call raises.
    """
    keep = set(wanted)
    scan = CorpusScan()
    with path.open("rb") as handle:
        for line in handle:
            start = line.find(_ACCESSION_KEY)
            if start < 0:
                continue
            start += len(_ACCESSION_KEY)
            accession = line[start : line.find(b'"', start)].decode("ascii")
            label = _go_label(line)
            scan.lines += 1
            scan.states[str(label["state"])] += 1
            for term in label.get("values") or ():
                scan.term_occurrences[str(term)] += 1
            for codes in (label.get("evidence") or {}).values():
                scan.code_occurrences.update(str(c) for c in codes)
            if accession in keep:
                scan.records[accession] = _raw_record(line, label)
    missing = sorted(keep - set(scan.records))
    if missing:
        raise BaselineError(f"{len(missing)} accessions are not in the corpus, e.g. {missing[0]}")
    return scan


@dataclass(frozen=True, slots=True)
class LabelRow:
    """One accession after the ontology is applied.

    ``direct`` maps a kept primary term to its evidence codes (the union over the corpus ids that
    resolved to it). ``propagated`` is the policy terms plus their ancestors, without the root:
    empty means the row is not labelled. ``motif_propagated`` is the same for the motif engine's
    GO hits.
    """

    accession: str
    state: str
    direct: Mapping[str, tuple[str, ...]]
    dropped: Mapping[str, int]
    motif: tuple[str, ...]
    propagated: tuple[str, ...]
    motif_propagated: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelContext:
    """What mapping a record needs: the ontology, its ancestry and the configuration."""

    onto: Ontology
    ancestry: Ancestry
    config: GoConfig


def _kept_terms(
    ctx: LabelContext, record: RawRecord
) -> tuple[dict[str, tuple[str, ...]], Counter[str]]:
    codes: dict[str, set[str]] = {}
    dropped: Counter[str] = Counter()
    for term_id in record.values:
        resolution = resolve_term(ctx.onto, term_id, ctx.config.ontology.namespace)
        if resolution.term is None:
            dropped[resolution.case] += 1
            continue
        found = record.evidence.get(term_id, ())
        for code in found:
            ctx.config.evidence.class_of(code)
        codes.setdefault(resolution.term, set()).update(found)
    return {term: tuple(sorted(c)) for term, c in sorted(codes.items())}, dropped


def _motif_terms(ctx: LabelContext, motif_ids: Sequence[str]) -> tuple[str, ...]:
    kept: set[str] = set()
    for term_id in motif_ids:
        resolution = resolve_term(ctx.onto, term_id, ctx.config.ontology.namespace)
        if resolution.term is not None:
            kept.add(resolution.term)
    return tuple(sorted(ctx.ancestry.propagate(kept)))


def map_record(ctx: LabelContext, record: RawRecord, motif_ids: Sequence[str]) -> LabelRow:
    """Apply the ontology and the evidence policy to one record (ADR-0014)."""
    if record.state not in (STATE_ANNOTATED, STATE_UNKNOWN):
        raise BaselineError(f"{record.accession}: GO is open-world, state {record.state!r}")
    direct, dropped = _kept_terms(ctx, record)
    policy = ctx.config.evidence.policy_codes
    chosen = {t for t, codes in direct.items() if policy.intersection(codes)}
    chosen.discard(ctx.config.ontology.root)
    return LabelRow(
        accession=record.accession,
        state=record.state,
        direct=direct,
        dropped={case: dropped[case] for case in CASES if dropped[case]},
        motif=tuple(sorted(motif_ids)),
        propagated=tuple(sorted(ctx.ancestry.propagate(chosen))),
        motif_propagated=_motif_terms(ctx, motif_ids),
    )


def label_table_text(labels: Mapping[str, LabelRow]) -> str:
    """The label table as TSV text, sorted by accession (deterministic bytes)."""
    lines = ["\t".join(LABEL_COLUMNS)]
    for accession in sorted(labels):
        row = labels[accession]
        direct = LIST_SEPARATOR.join(
            f"{t}{PAIR_SEPARATOR}{CODE_SEPARATOR.join(codes)}" for t, codes in row.direct.items()
        )
        dropped = LIST_SEPARATOR.join(f"{c}{PAIR_SEPARATOR}{n}" for c, n in row.dropped.items())
        cells = [
            row.accession,
            row.state,
            direct,
            dropped,
            LIST_SEPARATOR.join(row.motif),
            LIST_SEPARATOR.join(row.propagated),
            LIST_SEPARATOR.join(row.motif_propagated),
        ]
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def _split(cell: str) -> list[str]:
    return cell.split(LIST_SEPARATOR) if cell else []


def parse_label_table(text: str) -> dict[str, LabelRow]:
    """Read a table written by :func:`label_table_text`; anything else is refused."""
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != LABEL_COLUMNS:
        raise BaselineError("the GO label table has an unexpected header")
    labels: dict[str, LabelRow] = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != LABEL_WIDTH:
            raise BaselineError("a GO label table row has the wrong number of columns")
        direct = {
            pair.partition(PAIR_SEPARATOR)[0]: tuple(
                c for c in pair.partition(PAIR_SEPARATOR)[2].split(CODE_SEPARATOR) if c
            )
            for pair in _split(cells[2])
        }
        dropped = {
            pair.partition(PAIR_SEPARATOR)[0]: int(pair.partition(PAIR_SEPARATOR)[2])
            for pair in _split(cells[3])
        }
        labels[cells[0]] = LabelRow(
            accession=cells[0],
            state=cells[1],
            direct=direct,
            dropped=dropped,
            motif=tuple(_split(cells[MOTIF_CELL])),
            propagated=tuple(_split(cells[5])),
            motif_propagated=tuple(_split(cells[6])),
        )
    return labels


# -- the vocabulary: counted on SUBSET rows only ---------------------------------------
@dataclass(frozen=True, slots=True)
class VocabTerm:
    """One trained term: its positives in partitions 1-4 and its ancestors in the vocabulary."""

    id: str
    name: str
    positives: int
    by_partition: Mapping[str, int]
    ancestors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The trained terms (sorted by id) and the terms present in the subset but under the floor."""

    terms: tuple[VocabTerm, ...]
    undefined: Mapping[str, int]
    floor: int

    @property
    def ids(self) -> tuple[str, ...]:
        """Term ids in column order."""
        return tuple(t.id for t in self.terms)

    def __len__(self) -> int:
        """Number of trained terms."""
        return len(self.terms)


def subset_labels(labels: Mapping[str, LabelRow], subset: Collection[str]) -> dict[str, LabelRow]:
    """Only the labels of accessions in the subset mask; any other label is invisible downstream."""
    keep = set(subset)
    return {a: row for a, row in labels.items() if a in keep}


def term_counts(
    labels: Mapping[str, LabelRow], partition_of: Mapping[str, int]
) -> dict[str, dict[int, int]]:
    """Positives per term per partition over the labels given (propagated policy terms)."""
    counts: dict[str, dict[int, int]] = {}
    for accession, row in labels.items():
        partition = partition_of[accession]
        for term in row.propagated:
            per = counts.setdefault(term, {})
            per[partition] = per.get(partition, 0) + 1
    return counts


def build_vocabulary(
    labels: Mapping[str, LabelRow],
    subset: Collection[str],
    partition_of: Mapping[str, int],
    ctx: LabelContext,
) -> Vocabulary:
    """Terms with at least the floor of positives in partitions 1-4 of SUBSET rows (ADR-0014).

    ``labels`` may hold more accessions than the subset (the whole working set); only the subset
    mask decides what is counted, so a label outside it can change neither the vocabulary nor a
    count.
    """
    kept = subset_labels(labels, subset)
    counts = term_counts(kept, partition_of)
    validation = {t: sum(per.get(p, 0) for p in VALIDATION_PARTITIONS) for t, per in counts.items()}
    chosen = sorted(t for t, n in validation.items() if n >= ctx.config.term_floor)
    chosen_set = set(chosen)
    terms = tuple(
        VocabTerm(
            id=t,
            name=ctx.onto.terms[t].name,
            positives=validation[t],
            by_partition={
                str(p): counts[t].get(p, 0) for p in (*VALIDATION_PARTITIONS, TEST_PARTITION)
            },
            ancestors=tuple(sorted(a for a in ctx.ancestry.of(t) if a in chosen_set)),
        )
        for t in chosen
    )
    undefined = {t: n for t, n in sorted(validation.items()) if n < ctx.config.term_floor}
    return Vocabulary(terms=terms, undefined=undefined, floor=ctx.config.term_floor)


def vocabulary_document(vocab: Vocabulary, config: GoConfig) -> dict[str, Any]:
    """The committed vocabulary file: terms in column order, bound to the release and the policy."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "ontology": {
            "data_version": config.ontology.data_version,
            "sha256": config.ontology.sha256,
            "namespace": config.ontology.namespace,
            "root": config.ontology.root,
        },
        "evidence_policy": list(config.evidence.policy),
        "evidence_codes": sorted(config.evidence.policy_codes),
        "propagation": list(config.ontology.relations),
        "term_floor": vocab.floor,
        "terms": [
            {
                "id": t.id,
                "name": t.name,
                "positives_validation": t.positives,
                "ancestors": list(t.ancestors),
            }
            for t in vocab.terms
        ],
    }


def vocabulary_from_document(doc: Mapping[str, Any]) -> Vocabulary:
    """Rebuild the trained terms from a vocabulary file (per-partition counts are not in it)."""
    entries = doc.get("terms")
    if not isinstance(entries, list) or not entries:
        raise BaselineError("the vocabulary file has no terms")
    terms = tuple(
        VocabTerm(
            id=str(e["id"]),
            name=str(e["name"]),
            positives=int(e["positives_validation"]),
            by_partition={},
            ancestors=tuple(str(a) for a in e["ancestors"]),
        )
        for e in entries
    )
    return Vocabulary(terms=terms, undefined={}, floor=int(doc["term_floor"]))


# -- rows ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Rows:
    """The labelled accessions used: sequence hash, 0/1 term matrix, partition, cluster."""

    accession: Sequence[str]
    shas: Sequence[str]
    y: Array
    partition: Array
    cluster_code: Array
    seq_index: Array

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.shas)


def build_rows(
    table: Mapping[str, Array],
    order: Sequence[str],
    member: Array,
    labels: Mapping[str, LabelRow],
    spec: tuple[Vocabulary, Sequence[int], int],
) -> Rows:
    """The labelled subset rows of the given partitions, every stride-th.

    ``member`` is the feature table's subset mask. A row is labelled when its record is annotated
    and has a policy term; the closed-world rule (ADR-0006, applied here at loss and metric time
    over labelled rows only) makes every vocabulary term absent from it a negative.
    """
    np = _np()
    vocab, partitions, stride = spec
    column = {t: j for j, t in enumerate(vocab.ids)}
    candidates = np.flatnonzero(member & np.isin(table["partition"], list(partitions)))
    chosen: list[int] = []
    for i in candidates.tolist():
        accession = str(table["accession"][i])
        row = labels.get(accession)
        if row is None:
            raise BaselineError(f"no GO label row for {accession}")
        if row.state == STATE_ANNOTATED and row.propagated:
            chosen.append(i)
    index = np.array(chosen[::stride], dtype=np.int64)
    y = np.zeros((len(index), len(vocab)), dtype=np.int64)
    for r, i in enumerate(index.tolist()):
        for term in labels[str(table["accession"][i])].propagated:
            if term in column:
                y[r, column[term]] = 1
    seq_index = table["seq_index"][index]
    return Rows(
        accession=[str(table["accession"][i]) for i in index],
        shas=[order[int(s)] for s in seq_index],
        y=y,
        partition=table["partition"][index],
        cluster_code=table["cluster_code"][index],
        seq_index=seq_index,
    )


def fold_rows(rows: Rows, fold: Fold) -> tuple[list[int], list[int]]:
    """Row indices of the training partitions and of the validation partition of a fold."""
    train = [i for i, p in enumerate(rows.partition) if int(p) in fold.train]
    held = [i for i, p in enumerate(rows.partition) if int(p) == fold.validate]
    return train, held


def capped_pos_weights(y: Sequence[Sequence[int]], cap: float) -> list[float]:
    """Per-term ``negatives / positives`` over the TRAINING rows passed in, capped at ``cap``.

    A term with no positive in them gets 1.0 rather than an infinite weight.
    """
    if not len(y):
        raise BaselineError("pos_weights needs at least one training row")
    total = len(y)
    width = len(y[0])
    positives = [sum(int(row[j]) for row in y) for j in range(width)]
    return [min((total - p) / p, cap) if p > 0 else 1.0 for p in positives]


def holdout_nll(score: Array, y: Array) -> float:
    """Mean binary cross-entropy of sigmoid scores (the early-stopping signal; lower is better)."""
    np = _np()
    clipped = np.clip(score, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped)))


# -- scores: propagation, top-k, thresholds --------------------------------------------
@dataclass(frozen=True, slots=True)
class Scoring:
    """How a set of scores is judged: the metric settings and the cluster bootstrap."""

    metrics: MetricSettings
    boot: bm.Bootstrap


def descendant_mask(vocab: Vocabulary) -> Array:
    """``mask[a, t]`` is True when term ``t`` is term ``a`` or one of its descendants."""
    np = _np()
    column = {t: j for j, t in enumerate(vocab.ids)}
    mask = np.eye(len(vocab), dtype=bool)
    for j, term in enumerate(vocab.terms):
        for ancestor in term.ancestors:
            mask[column[ancestor], j] = True
    return mask


def propagate_scores(score: Array, vocab: Vocabulary) -> Array:
    """Make scores obey the true-path rule: a term scores at least as high as any descendant."""
    np = _np()
    mask = descendant_mask(vocab)
    out = np.empty(score.shape, dtype=float)
    for a in range(len(vocab)):
        out[:, a] = score[:, mask[a]].max(axis=1)
    return out


def evaluable(y: Array) -> Array:
    """Rows with at least one vocabulary positive: the rows precision, recall and Fmax can use."""
    return y.sum(axis=1) > 0


def top_k_hits(score: Array, y: Array, k: int) -> Array:
    """Per row, how many of the ``min(k, T)`` top-scoring terms are true (ties: lower column)."""
    np = _np()
    width = score.shape[1]
    order = np.argsort(-score, axis=1, kind="stable")[:, : min(k, width)]
    return np.take_along_axis(y, order, axis=1).sum(axis=1)


def base_names(ks: Sequence[int]) -> list[str]:
    """The ratio estimators every metric is built from, in column order."""
    names: list[str] = []
    for k in ks:
        names += [f"precision_at_{k}", f"recall_at_{k}"]
    return [*names, "fmax_precision", "fmax_recall", "coverage"]


def metric_names(ks: Sequence[int]) -> list[str]:
    """Every reported pooled metric name for the given k values."""
    names: list[str] = []
    for k in ks:
        names += [f"precision_at_{k}", f"recall_at_{k}"]
    return [*names, "fmax", "fmax_precision", "fmax_recall", "coverage"]


def row_terms(score: Array, y: Array, decision: Array, ks: Sequence[int]) -> tuple[Array, Array]:
    """Per-row numerators and denominators of :func:`base_names` (rows must be evaluable)."""
    np = _np()
    width = score.shape[1]
    positives = y.sum(axis=1).astype(float)
    ones = np.ones(len(y))
    nums: list[Array] = []
    dens: list[Array] = []
    for k in ks:
        hits = top_k_hits(score, y, k).astype(float)
        nums += [hits / min(k, width), hits / positives]
        dens += [ones, ones]
    called = decision.astype(bool)
    true_called = (called & (y == 1)).sum(axis=1).astype(float)
    predicted = called.sum(axis=1).astype(float)
    has = predicted > 0
    nums += [np.where(has, true_called / np.maximum(predicted, 1.0), 0.0)]
    dens += [has.astype(float)]
    nums += [true_called / positives, has.astype(float)]
    dens += [ones, ones]
    return np.stack(nums, axis=1), np.stack(dens, axis=1)


def summarise(num_sum: Array, den_sum: Array, ks: Sequence[int]) -> dict[str, float]:
    """Pooled metric values from summed numerators and denominators (ratio estimators)."""
    np = _np()
    safe = np.where(den_sum > 0, den_sum, 1.0)
    ratio = np.where(den_sum > 0, num_sum / safe, 0.0)
    values = {name: float(v) for name, v in zip(base_names(ks), ratio, strict=True)}
    p, r = values["fmax_precision"], values["fmax_recall"]
    values["fmax"] = 2.0 * p * r / (p + r) if p + r > 0.0 else 0.0
    return {name: values[name] for name in metric_names(ks)}


def _interval(value: float, draws: Sequence[float]) -> dict[str, float | None]:
    np = _np()
    if not len(draws):
        return {"value": float(value), "ci_low": None, "ci_high": None}
    low, high = (float(v) for v in np.percentile(np.array(draws), bm.CI_PERCENTILES))
    return {"value": float(value), "ci_low": low, "ci_high": high}


def metric_report(
    score: Array, y: Array, decision: Array, cluster_code: Array, scoring: Scoring
) -> dict[str, Any]:
    """Pooled metrics with cluster-bootstrap intervals, over the evaluable rows only.

    ``score`` is the propagated score, ``decision`` the boolean call at each row's own threshold.
    """
    np = _np()
    keep = evaluable(y)
    if not keep.any():
        raise BaselineError("no evaluable row: every row has an empty vocabulary annotation")
    ks = scoring.metrics.k_values
    num, den = row_terms(score[keep], y[keep], decision[keep], ks)
    point = summarise(num.sum(axis=0), den.sum(axis=0), ks)
    draws: dict[str, list[float]] = {name: [] for name in point}
    samples = scoring.boot.samples
    if samples > 0:
        _, inverse = np.unique(cluster_code[keep], return_inverse=True)
        count = int(inverse.max()) + 1
        cluster_num = np.zeros((count, num.shape[1]))
        cluster_den = np.zeros((count, den.shape[1]))
        np.add.at(cluster_num, inverse, num)
        np.add.at(cluster_den, inverse, den)
        rng = np.random.default_rng(scoring.boot.seed)
        for _ in range(samples):
            picks = rng.integers(0, count, size=count)
            sums = summarise(cluster_num[picks].sum(axis=0), cluster_den[picks].sum(axis=0), ks)
            for name, value in sums.items():
                draws[name].append(value)
    return {
        "rows": len(y),
        "evaluable_rows": int(keep.sum()),
        "metrics": {name: _interval(value, draws[name]) for name, value in point.items()},
        "bootstrap": {"samples": samples, "seed": scoring.boot.seed, "unit": "cluster_id"},
    }


def _average_ranks(values: Array) -> Array:
    np = _np()
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(values)]))
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def auc_of(score: Array, y: Array) -> float:
    """Area under the ROC curve by the rank statistic (ties share their average rank)."""
    positive = y == 1
    p = int(positive.sum())
    n = len(y) - p
    if p == 0 or n == 0:
        raise BaselineError("AUC needs both classes")
    ranks = _average_ranks(score)
    return float((ranks[positive].sum() - p * (p + 1) / 2.0) / (p * n))


def aupr_of(score: Array, y: Array) -> float:
    """Average precision over the distinct score thresholds."""
    np = _np()
    p = int((y == 1).sum())
    if p == 0:
        raise BaselineError("average precision needs a positive")
    order = np.argsort(-score, kind="mergesort")
    ordered = score[order]
    hits = np.cumsum((y[order] == 1).astype(float))
    last = np.flatnonzero(np.append(ordered[1:] != ordered[:-1], True))
    precision = hits[last] / (last + 1.0)
    recall = hits[last] / p
    steps = recall - np.concatenate(([0.0], recall[:-1]))
    return float((steps * precision).sum())


def per_term_report(
    score: Array, y: Array, cluster_code: Array, vocab: Vocabulary, scoring: Scoring
) -> dict[str, Any]:
    """Per-term AUC and AUPR for terms with enough positives and negatives; the rest are listed.

    Rows are every labelled row of the set (the closed-world negatives); a bootstrap draw with a
    single class for a term is skipped for that term.
    """
    np = _np()
    minimum = scoring.metrics.per_term_min_positives
    draws: list[Array] = []
    if scoring.boot.samples > 0:
        _, inverse = np.unique(cluster_code, return_inverse=True)
        members = [np.flatnonzero(inverse == c) for c in range(int(inverse.max()) + 1)]
        rng = np.random.default_rng(scoring.boot.seed)
        for _ in range(scoring.boot.samples):
            picks = rng.integers(0, len(members), size=len(members))
            draws.append(np.concatenate([members[int(p)] for p in picks]))
    out: dict[str, Any] = {}
    for j, term in enumerate(vocab.terms):
        column = y[:, j]
        positives = int(column.sum())
        negatives = len(column) - positives
        entry: dict[str, Any] = {
            "name": term.name,
            "positives": positives,
            "negatives": negatives,
            "scored": positives >= minimum and negatives >= minimum,
        }
        if entry["scored"]:
            auc, aupr = auc_of(score[:, j], column), aupr_of(score[:, j], column)
            auc_draws: list[float] = []
            aupr_draws: list[float] = []
            for chosen in draws:
                sample = column[chosen]
                if 0 < int(sample.sum()) < len(sample):
                    auc_draws.append(auc_of(score[chosen, j], sample))
                    aupr_draws.append(aupr_of(score[chosen, j], sample))
            entry["auc"] = _interval(auc, auc_draws)
            entry["aupr"] = _interval(aupr, aupr_draws)
            entry["positive_rate"] = positives / len(column)
        out[term.id] = entry
    return out


def f_measure(score: Array, y: Array, threshold: float) -> float:
    """CAFA-style F at one global threshold over the evaluable rows of ``y`` (0.0 if undefined)."""
    np = _np()
    keep = evaluable(y)
    if not keep.any():
        return 0.0
    truth = y[keep] == 1
    called = score[keep] >= threshold
    true_called = (called & truth).sum(axis=1).astype(float)
    predicted = called.sum(axis=1).astype(float)
    has = predicted > 0
    precision = float(np.mean(true_called[has] / predicted[has])) if has.any() else 0.0
    recall = float(np.mean(true_called / truth.sum(axis=1)))
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0


def threshold_candidates(score: Array, limit: int) -> Array:
    """The distinct scores, ascending; above ``limit`` of them, evenly spaced order statistics."""
    np = _np()
    values = np.unique(score)
    if len(values) <= limit:
        return values
    picks = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(np.int64))
    return values[picks]


def best_f_threshold(score: Array, y: Array, limit: int) -> float:
    """The global threshold that maximises F over the evaluable rows (ties: the higher one)."""
    keep = evaluable(y)
    if not keep.any():
        raise BaselineError("a threshold needs at least one evaluable row")
    best_value, best = -1.0, 0.0
    for candidate in threshold_candidates(score[keep], limit).tolist():
        value = f_measure(score, y, float(candidate))
        if value >= best_value:
            best_value, best = value, float(candidate)
    return best


def nested_thresholds(
    score: Array, y: Array, partition: Array, limit: int
) -> tuple[dict[int, float], float]:
    """Fold k's threshold is tuned on the other partitions only; the frozen one on all of them."""
    per_fold = {
        int(p): best_f_threshold(score[partition != p], y[partition != p], limit)
        for p in sorted({int(v) for v in partition})
    }
    return per_fold, best_f_threshold(score, y, limit)


def decide(score: Array, partition: Array, thresholds: Mapping[int, float]) -> Array:
    """Boolean calls in which every row uses the threshold of its own partition's fold."""
    np = _np()
    cut = np.array([thresholds[int(p)] for p in partition], dtype=float)
    return score >= cut[:, None]


def fold_means(
    score: Array, y: Array, partition: Array, metrics: MetricSettings
) -> tuple[float, float]:
    """Mean over folds of precision@primary_k and of F at the fold's nested threshold.

    ``score`` is propagated. A fold with no evaluable validation row is skipped, not zeroed.
    """
    np = _np()
    thresholds, _ = nested_thresholds(score, y, partition, metrics.max_threshold_candidates)
    precisions: list[float] = []
    fs: list[float] = []
    for p in sorted(thresholds):
        held = partition == p
        keep = held & evaluable(y)
        if not keep.any():
            continue
        hits = top_k_hits(score[keep], y[keep], metrics.primary_k)
        precisions.append(float(np.mean(hits / min(metrics.primary_k, score.shape[1]))))
        fs.append(f_measure(score[held], y[held], thresholds[p]))
    if not precisions:
        raise BaselineError("no fold has an evaluable validation row")
    return math.fsum(precisions) / len(precisions), math.fsum(fs) / len(fs)


def select_point(values: Sequence[tuple[float, float]]) -> int:
    """Best grid point: higher mean precision@k, then higher mean Fmax, then the earlier point."""
    return max(range(len(values)), key=lambda i: (values[i][0], values[i][1], -i))


def fold_summaries(
    score: Array, y: Array, decision: Array, partition: Array, ks: Sequence[int]
) -> list[dict[str, Any]]:
    """Per-fold metric values (no intervals) at each fold's own decisions."""
    out: list[dict[str, Any]] = []
    for p in sorted({int(v) for v in partition}):
        held = partition == p
        keep = held & evaluable(y)
        values: dict[str, float] | None = None
        if keep.any():
            num, den = row_terms(score[keep], y[keep], decision[keep], ks)
            values = summarise(num.sum(axis=0), den.sum(axis=0), ks)
        out.append(
            {
                "validate": p,
                "rows": int(held.sum()),
                "evaluable_rows": int(keep.sum()),
                "metrics": values,
            }
        )
    return out


# -- the floors: identical rows, folds, thresholds, bootstrap and selection ------------
def fold_prior_scores(y: Array, partition: Array, folds: Sequence[Fold]) -> Array:
    """Each validation row scores every term with its fold's training-partition frequency."""
    np = _np()
    out = np.zeros(y.shape, dtype=float)
    for fold in folds:
        train = np.isin(partition, list(fold.train))
        out[partition == fold.validate] = y[train].mean(axis=0)
    return out


def feature_matrix(table: Mapping[str, Array], rows: Rows) -> Array:
    """The Phase 8 motif/composition features of each row's sequence."""
    return table["X"][rows.seq_index]


def logreg_fit(x: Array, y: Array, c: float, runtime: bm.Runtime) -> dict[str, Any]:
    """The Phase 8 logistic regression (balanced, one-vs-rest) fitted on multi-label targets."""
    return bm.fit_model("logreg", {"C": c}, x, y, runtime)


def logreg_apply(fitted: Mapping[str, Any], x: Array) -> Array:
    """Per-term sigmoid of the margins, so every arm scores on (0, 1)."""
    np = _np()
    return 1.0 / (1.0 + np.exp(-bm.model_scores(fitted, x)))


def logreg_scores(
    x: Array, y: Array, partition: Array, spec: tuple[Sequence[Fold], float, bm.Runtime]
) -> Array:
    """Out-of-fold logistic-regression scores: fold k is scored by a model fitted without k."""
    np = _np()
    folds, c, runtime = spec
    out = np.zeros(y.shape, dtype=float)
    for fold in folds:
        train = np.isin(partition, list(fold.train))
        held = partition == fold.validate
        out[held] = logreg_apply(logreg_fit(x[train], y[train], c, runtime), x[held])
    return out


def motif_matrix(rows: Rows, labels: Mapping[str, LabelRow], vocab: Vocabulary) -> Array:
    """1 where the PROSITE-style engine's GO hits (with their ancestors) name a vocabulary term."""
    np = _np()
    column = {t: j for j, t in enumerate(vocab.ids)}
    hits = np.zeros((len(rows), len(vocab)), dtype=float)
    for r, accession in enumerate(rows.accession):
        for term in labels[accession].motif_propagated:
            if term in column:
                hits[r, column[term]] = 1.0
    return hits


def motif_scores(hits: Array, prior: Array, weight: float) -> Array:
    """``1 + weight * prior`` for a hit term, ``weight * prior`` otherwise (prior breaks ties)."""
    return hits + weight * prior


# -- calibration: the Phase 10 machinery, one calibrator per term ----------------------
def calibration_settings(base: cal.CalibrationConfig, config: GoConfig) -> cal.CalibrationConfig:
    """The Phase 10 settings, with the Platt fallback switched off unless the config keeps it."""
    if config.platt_fallback:
        return base
    e = base.eligibility
    return replace(
        base,
        eligibility=cal.Eligibility(
            e.isotonic_min_positives,
            e.isotonic_min_negatives,
            e.isotonic_min_positives,
            e.isotonic_min_negatives,
        ),
    )


@dataclass(frozen=True, slots=True)
class CalibData:
    """Out-of-fold raw per-term scores with labels, partitions and clusters."""

    raw: Array
    y: Array
    partition: Array
    cluster_code: Array
    vocab: Vocabulary


@dataclass(frozen=True, slots=True)
class CrossFit:
    """Cross-fitted calibrated scores and the thresholds tuned on them (none saw its own row)."""

    calibrated: Array
    propagated: Array
    fold_thresholds: dict[int, float]
    frozen_threshold: float
    frozen: list[LabelCalibrator]
    methods: dict[str, Any]


def _fit_terms(
    data: CalibData, fit: Array, cfg: cal.CalibrationConfig, forced: Sequence[str] | None
) -> list[LabelCalibrator]:
    return [
        cal.fit_label(
            term,
            [float(v) for v in data.raw[fit][:, j]],
            [int(v) for v in data.y[fit][:, j]],
            cfg,
            None if forced is None else forced[j],
        )
        for j, term in enumerate(data.vocab.ids)
    ]


def apply_calibrators(calibrators: Sequence[LabelCalibrator], raw: Array) -> Array:
    """Calibrated per-term probabilities of raw scores (identity for an uncalibrated term)."""
    np = _np()
    out = np.zeros(raw.shape, dtype=float)
    for j, calibrator in enumerate(calibrators):
        out[:, j] = [calibrator.apply(float(v)) for v in raw[:, j]]
    return out


def _clusters_of(data: CalibData, mask: Array) -> set[int]:
    return {int(c) for c in data.cluster_code[mask]}


def _disjoint(data: CalibData, masks: Mapping[str, Array]) -> None:
    head.assert_no_overlap({name: _clusters_of(data, m) for name, m in masks.items()})


def crossfit(
    data: CalibData, folds: Sequence[Fold], cfg: cal.CalibrationConfig, limit: int
) -> CrossFit:
    """Cross-fit the per-term calibrators and one global threshold over the folds.

    Fold k is scored with calibrators fitted on the other partitions. Its threshold is tuned on
    calibrated scores of the other partitions whose calibrators also never saw fold k. The frozen
    threshold is tuned on the cross-fitted scores of all four partitions.
    """
    np = _np()
    calibrated = np.zeros(data.raw.shape, dtype=float)
    outer: dict[int, list[LabelCalibrator]] = {}
    methods: dict[str, Any] = {"fold": {}, "inner": {}}
    for fold in folds:
        scored = data.partition == fold.validate
        _disjoint(data, {"calibration": ~scored, "scored": scored})
        outer[fold.validate] = _fit_terms(data, ~scored, cfg, None)
        calibrated[scored] = apply_calibrators(outer[fold.validate], data.raw[scored])
        methods["fold"][str(fold.validate)] = [c.method for c in outer[fold.validate]]
    fold_thresholds: dict[int, float] = {}
    for fold in folds:
        forced = [c.method for c in outer[fold.validate]]
        pooled: list[Array] = []
        pooled_y: list[Array] = []
        inner_methods: dict[str, list[str]] = {}
        for other in folds:
            if other.validate == fold.validate:
                continue
            scored = data.partition == other.validate
            fit = (data.partition != other.validate) & (data.partition != fold.validate)
            _disjoint(
                data,
                {
                    "calibration": fit,
                    "scored": scored,
                    "outer_fold": data.partition == fold.validate,
                },
            )
            cals = _fit_terms(data, fit, cfg, forced)
            inner_methods[str(other.validate)] = [c.method for c in cals]
            pooled.append(apply_calibrators(cals, data.raw[scored]))
            pooled_y.append(data.y[scored])
        methods["inner"][str(fold.validate)] = inner_methods
        fold_thresholds[fold.validate] = best_f_threshold(
            propagate_scores(np.concatenate(pooled), data.vocab), np.concatenate(pooled_y), limit
        )
    propagated = propagate_scores(calibrated, data.vocab)
    frozen_threshold = best_f_threshold(propagated, data.y, limit)
    everything = np.ones(len(data.partition), dtype=bool)
    frozen = [
        replace(c, threshold=frozen_threshold) for c in _fit_terms(data, everything, cfg, None)
    ]
    methods["frozen"] = [c.method for c in frozen]
    methods["consistent"] = {
        term: all(methods["fold"][k][j] == frozen[j].method for k in methods["fold"])
        for j, term in enumerate(data.vocab.ids)
    }
    return CrossFit(calibrated, propagated, fold_thresholds, frozen_threshold, frozen, methods)


def ece_report(
    data: CalibData, fit: CrossFit, cfg: cal.CalibrationConfig, boot: bm.Bootstrap
) -> dict[str, Any]:
    """ECE and Brier of the raw and cross-fitted per-term scores (Phase 10 report, per term)."""
    scored = cal.ScoredRows(
        raw=data.raw,
        calibrated=fit.calibrated,
        y=data.y,
        cluster_code=data.cluster_code,
        classes=data.vocab.ids,
    )
    return cal.calibration_report(scored, cfg.ece, boot)


# -- the manifest ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Pins:
    """What a GO manifest is checked against; read from the repository, not from the manifest."""

    registry_sha256: str
    corpus_sha256: str
    ontology_sha256: str
    embedding_cache_key: str
    residue_cache_key: str
    config_sha256: str
    labels_sha256: str
    calibration_config_sha256: str
    vocabulary_sha256: str
    vocabulary_ids: tuple[str, ...]
    term_floor: int
    closed_world: str
    k_values: tuple[int, ...]
    reproduction_tolerance: float
    torch: str


def _object(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise BaselineError(f"manifest is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def _check_inputs(frozen: Mapping[str, Any], inputs: Mapping[str, Any], pins: Pins) -> list[str]:
    problems = check_subset_block(frozen, cache_key=pins.embedding_cache_key)
    expected = {
        "registry_sha256": pins.registry_sha256,
        "corpus_sha256": pins.corpus_sha256,
        "ontology_sha256": pins.ontology_sha256,
        "residue_cache_key": pins.residue_cache_key,
        "go_config_sha256": pins.config_sha256,
        "label_policy_sha256": pins.labels_sha256,
        "calibration_config_sha256": pins.calibration_config_sha256,
        "vocabulary_sha256": pins.vocabulary_sha256,
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            problems.append(f"inputs.{key} differs from the repository")
    for key in (
        "feature_table_sha256",
        "label_table_sha256",
        "weights_sha256",
        "training_record_sha256",
        "calibrator_sha256",
    ):
        if not _SHA.fullmatch(str(inputs.get(key, ""))):
            problems.append(f"inputs.{key} is not a SHA-256")
    return problems


def _check_vocabulary(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    block = frozen.get("vocabulary")
    if not isinstance(block, dict):
        return ["manifest is missing the object 'vocabulary'"]
    problems: list[str] = []
    if block.get("counts_scope") != "subset_mask":
        problems.append("vocabulary counts must come from the subset mask")
    if block.get("term_floor") != pins.term_floor:
        problems.append("vocabulary.term_floor differs from go.yaml")
    if block.get("ids") != list(pins.vocabulary_ids):
        problems.append("vocabulary.ids differs from the committed vocabulary file")
    trained = block.get("positives_validation")
    undefined = block.get("undefined")
    if not isinstance(trained, dict) or not isinstance(undefined, dict):
        return [*problems, "vocabulary needs positives_validation and undefined counts"]
    if list(trained) != list(pins.vocabulary_ids):
        problems.append("positives_validation does not cover exactly the vocabulary")
    for term, count in trained.items():
        if not isinstance(count, int) or count < pins.term_floor:
            problems.append(f"{term} is trained with fewer than the floor of positives")
    for term, count in undefined.items():
        if not isinstance(count, int) or count >= pins.term_floor or term in trained:
            problems.append(f"{term} is marked undefined but clears the floor or is also trained")
    policy = frozen.get("label_policy")
    if not isinstance(policy, dict) or policy.get("closed_world") != pins.closed_world:
        problems.append("label_policy.closed_world differs from go.yaml")
    elif policy.get("states") != [STATE_ANNOTATED, STATE_UNKNOWN]:
        problems.append("label_policy.states must be exactly annotated and unknown")
    return problems


def _check_report(report: object, name: str, pins: Pins) -> list[str]:
    if not isinstance(report, dict) or not isinstance(report.get("pooled"), dict):
        return [f"validation.{name} has no pooled report"]
    metrics = report["pooled"].get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(metric_names(pins.k_values)):
        return [f"validation.{name} does not report every metric"]
    if not isinstance(report.get("per_term"), dict):
        return [f"validation.{name} has no per-term report"]
    return []


def _check_validation(frozen: Mapping[str, Any], pins: Pins) -> list[str]:
    validation = frozen.get("validation")
    if not isinstance(validation, dict):
        return ["manifest is missing the object 'validation'"]
    problems: list[str] = []
    for arm in ARMS:
        problems += _check_report(validation.get(arm), arm, pins)
    reproduction = validation.get("reproduction")
    if not isinstance(reproduction, dict):
        return [*problems, "validation.reproduction is missing"]
    difference = reproduction.get("max_abs_difference")
    if not isinstance(difference, float) or not difference <= pins.reproduction_tolerance:
        problems.append("the raw scores did not reproduce from the saved weights")
    if reproduction.get("decisions_identical") is not True:
        problems.append("the raw decisions did not reproduce from the saved weights")
    if not isinstance(validation.get("calibration"), dict):
        problems.append("validation.calibration (ECE before and after) is missing")
    return problems


def check_go_manifest(manifest: Mapping[str, Any], pins: Pins) -> list[str]:
    """Every way the GO manifest disagrees with itself or the repository; empty if sound."""
    try:
        frozen = _object(manifest, "frozen")
        protocol = _object(frozen, "protocol")
        inputs = _object(frozen, "inputs")
        selected = _object(frozen, "selected")
        software = _object(frozen, "software")
        thresholds = _object(frozen, "thresholds")
    except BaselineError as error:
        return [str(error)]
    problems = _check_inputs(frozen, inputs, pins) + _check_vocabulary(frozen, pins)
    problems += _check_validation(frozen, pins)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version is not the one this code writes")
    if protocol.get("validation_partitions") != list(VALIDATION_PARTITIONS):
        problems.append("validation partitions are not [1, 2, 3, 4]")
    if protocol.get("test_partition") != TEST_PARTITION:
        problems.append("test partition is not 5")
    if protocol.get("k_values") != list(pins.k_values):
        problems.append("protocol.k_values differs from go.yaml")
    if not isinstance(selected.get("trainable_parameters"), int):
        problems.append("selected.trainable_parameters must be an integer")
    if not isinstance(selected.get("final_epochs"), int) or selected["final_epochs"] < 1:
        problems.append("selected.final_epochs must be a positive integer")
    for arm in ARMS:
        value = thresholds.get(arm)
        if not isinstance(value, float) or not math.isfinite(value):
            problems.append(f"thresholds.{arm} must be a finite number")
    if software.get("torch") != pins.torch:
        problems.append("torch version differs from the pyproject pin")
    if manifest.get("frozen_sha256") != digest(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
    test = manifest.get("test")
    if test is not None:
        problems += _check_test(test, str(manifest.get("frozen_sha256")), pins)
    return problems


def _check_test(test: object, frozen_sha256: str, pins: Pins) -> list[str]:
    if not isinstance(test, dict):
        return ["test must be an object or null"]
    problems: list[str] = []
    if test.get("frozen_sha256") != frozen_sha256:
        problems.append("test was scored against a different frozen block")
    if test.get("touches") != 1:
        problems.append("the test partition must be touched exactly once")
    metrics = test.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(ARMS):
        return [*problems, "test metrics do not cover every arm"]
    for name, report in metrics.items():
        body = report.get("pooled", {}).get("metrics") if isinstance(report, dict) else None
        if not isinstance(body, dict) or set(body) != set(metric_names(pins.k_values)):
            problems.append(f"test.{name} does not report every metric")
    return problems


# -- building the label table and the blocks the manifest records ----------------------
def load_ontology(path: Path, pin: OntologyPin) -> Ontology:
    """Read the pinned release; any difference in bytes, hash or counts is an error."""
    raw = path.read_bytes()
    if len(raw) != pin.size or hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise BaselineError(f"{path.name} is not the ontology release go.yaml pins")
    onto = parse_obo(raw.decode("utf-8"))
    problems = ontology_problems(onto, pin)
    if problems:
        raise BaselineError(f"{path.name}: {problems[0]}")
    return onto


def motif_ids(sequence: str) -> tuple[str, ...]:
    """The GO ids the PROSITE-style engine reports for one sequence."""
    classical = importlib.import_module("protloc.features.classical")
    return tuple(sorted({str(hit["go_id"]) for hit in classical.extract(sequence).go_hits}))


def build_labels(scan: CorpusScan, ctx: LabelContext) -> dict[str, LabelRow]:
    """Apply the ontology and the evidence policy to every record the scan kept."""
    return {
        accession: map_record(ctx, record, motif_ids(record.sequence))
        for accession, record in sorted(scan.records.items())
    }


def label_summary(
    labels: Mapping[str, LabelRow], partition_of: Mapping[str, int]
) -> dict[str, Any]:
    """States, labelled rows and motif hits of the given labels, in total and per partition."""
    out: dict[str, Any] = {
        "accessions": len(labels),
        "states": dict.fromkeys((STATE_ANNOTATED, STATE_UNKNOWN), 0),
        "labelled": 0,
        "annotated_without_policy_term": 0,
        "with_motif_hit": 0,
        "labelled_by_partition": {},
    }
    for accession, row in labels.items():
        out["states"][row.state] += 1
        if row.state == STATE_ANNOTATED and row.propagated:
            out["labelled"] += 1
            key = str(partition_of[accession])
            out["labelled_by_partition"][key] = out["labelled_by_partition"].get(key, 0) + 1
        elif row.state == STATE_ANNOTATED:
            out["annotated_without_policy_term"] += 1
        if row.motif_propagated:
            out["with_motif_hit"] += 1
    out["labelled_by_partition"] = dict(sorted(out["labelled_by_partition"].items()))
    return out


def vocabulary_block(vocab: Vocabulary) -> dict[str, Any]:
    """The manifest's record of what was trained, counted on the subset mask."""
    return {
        "counts_scope": "subset_mask",
        "term_floor": vocab.floor,
        "ids": list(vocab.ids),
        "positives_validation": {t.id: t.positives for t in vocab.terms},
        "undefined": dict(vocab.undefined),
    }


def label_policy_block(config: GoConfig) -> dict[str, Any]:
    """The manifest's record of the label policy (ADR-0006: open world, closed over labelled)."""
    return {
        "states": [STATE_ANNOTATED, STATE_UNKNOWN],
        "evidence_classes": {name: list(codes) for name, codes in config.evidence.classes.items()},
        "evidence_policy": list(config.evidence.policy),
        "evidence_codes": sorted(config.evidence.policy_codes),
        "term_floor": config.term_floor,
        "closed_world": config.closed_world,
        "namespace": config.ontology.namespace,
        "root": config.ontology.root,
        "propagation": list(config.ontology.relations),
        "ontology_data_version": config.ontology.data_version,
    }
