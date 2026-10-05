"""The serving engine (Phase 14, ADR-0016): trained heads behind the unchanged API.

This module owns the order of the pipeline — features -> model -> calibration -> consistency ->
evidence — and nothing else knows it. :func:`load_engine` verifies everything the service will
rely on and raises :class:`EngineStartError` naming every problem; a service that cannot verify
its bundles does not start. The pieces:

* the four registry bundles (``load_bundle`` / ``load_evidence_bundle``): every hash, the
  calibrators, the rule files, the ontology release, the GO annotation source and every byte of
  the MMseqs2 index;
* the numpy forward passes of ``protloc.models`` over frozen ESM-2 per-residue states read from
  the verified residue cache (cache-only encoder, ADR-0016): a sequence whose states are not
  cached is refused with a reason, never scored from something else;
* the calibrators and the per-label thresholds the bundles carry; nothing here holds a tau;
* homology evidence, searched off the request path under a deadline and kept in its own fields.
  Evidence never feeds a score, a calibrator or a threshold.

Membrane type and the scanner stay on the motif engine and say so (``source: "motif"``): there
is no trained head for them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from protloc.calibration.calibrator import CalibratorSet, canonical_json_bytes
from protloc.config.serving import ServingSettings
from protloc.data.embeddings import EmbeddingError
from protloc.data.records import normalize_sequence, sequence_sha256
from protloc.data.residue_cache import ResidueStore, read_residue_index, verify_residue_cache
from protloc.evidence import homology
from protloc.evidence.homology import Hit, HitFilter, TransferredTerm
from protloc.evidence.ontology import OboTerm, TermStore, read_obo
from protloc.evidence.search import EvidenceUnavailable, SearchPlan, search
from protloc.features.classical import (
    LOC_NAMES,
    ORGANISM_GROUPS,
    extract,
    membrane_type,
    normalize,
    residue_attribution,
)
from protloc.inference.motif import RULES, confidence_band
from protloc.models.attention_head import AttentionHeadNet, ModelError
from protloc.models.crf_tagger import CrfTaggerNet, TagSpec
from protloc.registry.bundle import Bundle, BundleError, file_sha256, load_bundle
from protloc.registry.evidence import EvidenceBundleError, load_evidence_bundle

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = REPO_ROOT / "configs" / "serving" / "engine.json"
RULES_DIRECTORY = REPO_ROOT / "backend" / "rules"
TASKS_WITH_WEIGHTS = ("localization", "signal_peptide", "go_function")
EVIDENCE_TASK = "evidence_homology"
SOURCE_MODEL = "model"
SOURCE_MOTIF = "motif"
POLICY_PER_LABEL = "per_label_mcc"
SPAN_SEQUENCE = re.compile(r"[A-Z]+")
_SHA_PREFIX = 12
_MAJORITY_NUMERATOR = 2


class EngineStartError(RuntimeError):
    """A bundle, weight, cache, manifest or index did not verify; the service must not start."""


class EngineRefusal(ValueError):  # noqa: N818
    """The engine will not score this input; ``code`` is stable, ``str(error)`` is for people."""

    def __init__(self, code: str, message: str) -> None:
        """Keep the stable code beside the message."""
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence for one sequence: either hits and transferred terms, or why there are none."""

    available: bool
    reason: str | None
    hits: tuple[Hit, ...]
    terms: tuple[TransferredTerm, ...]
    chosen: tuple[Hit, ...]


@dataclass(frozen=True, slots=True)
class _Pins:
    """Values the engine reads from ``engine.json``."""

    document: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.document.get(name)
        if not isinstance(value, dict):
            raise EngineStartError(f"{CONFIG_FILE.name} has no section {name!r}")
        return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise EngineStartError(f"{path} cannot be read as JSON: {error}") from error
    if not isinstance(value, dict):
        raise EngineStartError(f"{path}: top level must be a JSON object")
    return {str(k): v for k, v in value.items()}


def composite_version(bundles: Mapping[str, str]) -> str:
    """The version string the service reports: every bundle, in a fixed order."""
    order = (*TASKS_WITH_WEIGHTS, EVIDENCE_TASK)
    return "+".join(f"{task}-{bundles[task]}" for task in order)


def _manifest_problems(task: str, path: Path, bundle_sha256: str) -> list[str]:
    """A frozen manifest must hash to what it claims and name the bundle that is loaded."""
    try:
        manifest = _read_json(path)
    except EngineStartError as error:
        return [str(error)]
    frozen = manifest.get("frozen")
    if not isinstance(frozen, dict):
        return [f"{path.name}: no frozen block"]
    found = []
    if hashlib.sha256(canonical_json_bytes(frozen)).hexdigest() != manifest.get("frozen_sha256"):
        found.append(f"{path.name}: frozen_sha256 does not match the frozen block")
    named = (frozen.get("bundle") or {}).get("bundle_sha256")
    if named != bundle_sha256:
        found.append(f"{path.name}: the manifest froze another {task} bundle than the one loaded")
    return found


class Engine:
    """Everything loaded and verified, and the pipeline that uses it."""

    def __init__(self, parts: "_Parts") -> None:
        """Hold the verified parts; use :func:`load_engine` to build one."""
        self._p = parts

    # -- identity --------------------------------------------------------------------
    @property
    def provenance(self) -> dict[str, Any]:
        """The reproducibility fields every response carries, plus the bundles behind them."""
        return self._p.provenance

    @property
    def evidence_state(self) -> dict[str, Any]:
        """Whether evidence can be searched on this host, and if not why."""
        return {
            "available": self._p.search_plan is not None,
            "reason": self._p.evidence_reason,
            "swissprot_release": self._p.pins.section("evidence")["swissprot_release"],
        }

    @property
    def groups(self) -> tuple[str, ...]:
        """The organism groups the trained heads cover."""
        return self._p.groups

    @property
    def cached_sequences(self) -> int:
        """How many sequences have cached states."""
        return len(self._p.store.lengths)

    @property
    def manifests(self) -> dict[str, dict[str, Any]]:
        """The verified frozen manifest of each task."""
        return self._p.manifests

    @property
    def rules(self) -> list[dict[str, Any]]:
        """The consistency rules, each marked active or (Phase 14) inactive."""
        inactive = set(self._p.provenance["rules_inactive"])
        return [{**r, "active": r["id"] not in inactive} for r in RULES["rules"]]

    @property
    def localization_codes(self) -> dict[str, str]:
        """Class name to the three-letter code the screens use."""
        return dict(self._p.codes)

    @property
    def gallery(self) -> list[dict[str, Any]]:
        """Gallery examples (cached sequences with their database annotation)."""
        return self._p.gallery

    def has_states(self, sequence: str) -> bool:
        """Whether this normalized sequence is one the cache can score."""
        return sequence_sha256(normalize_sequence(sequence)) in self._p.store.lengths

    # -- the pipeline ----------------------------------------------------------------
    def predict_many(
        self, records: Sequence[tuple[str, str]], group: str, with_evidence: bool = True
    ) -> list[dict[str, Any] | EngineRefusal]:
        """Results for ``(id, raw sequence)`` records, one search for all of them."""
        if group not in ORGANISM_GROUPS:
            raise EngineRefusal("unknown_group", f"Unknown organism_group '{group}'")
        if group not in self._p.groups:
            covered = ", ".join(self._p.groups)
            raise EngineRefusal(
                "unsupported_group",
                f"No trained head covers '{group}': the training corpus is eukaryotic "
                f"(served groups: {covered}).",
            )
        scored: dict[int, dict[str, Any]] = {}
        outcome: list[dict[str, Any] | EngineRefusal] = []
        for index, (seq_id, raw) in enumerate(records):
            try:
                scored[index] = self._score(seq_id, raw, group)
                outcome.append(scored[index])
            except EngineRefusal as refusal:
                outcome.append(refusal)
        for result in scored.values():
            result["consistency_flags"] = self._consistency(result)
        if with_evidence and scored:
            found = self._evidence({i: r["sequence"] for i, r in scored.items()})
            for index, result in scored.items():
                self._attach_evidence(result, found[index])
        else:
            for result in scored.values():
                self._attach_evidence(
                    result, _no_evidence("evidence was not requested for this call")
                )
        return outcome

    def predict(
        self, raw: str, group: str, seq_id: str, with_evidence: bool = True
    ) -> dict[str, Any]:
        """Result of one sequence; raises :class:`EngineRefusal` when it cannot be scored."""
        only = self.predict_many([(seq_id, raw)], group, with_evidence)[0]
        if isinstance(only, EngineRefusal):
            raise only
        return only

    # -- stage 1-3: features, model, calibration -------------------------------------
    def _score(self, seq_id: str, raw: str, group: str) -> dict[str, Any]:
        try:
            scored_form = normalize(raw)  # the motif engine's form: U, B, Z are substituted
        except ValueError as error:
            raise EngineRefusal("invalid_sequence", str(error)) from error
        sequence = normalize_sequence(raw)  # the corpus form the cache is keyed by
        sha = sequence_sha256(sequence)
        if sha not in self._p.store.lengths:
            raise EngineRefusal(
                "not_in_embedding_cache",
                "This build scores only the sequences whose ESM-2 states are cached "
                f"({self.cached_sequences} of them, ADR-0016); this one is not among them. "
                "Pick a gallery example.",
            )
        states = self._p.store.states(sha)
        try:
            loc = self._p.loc_net.forward(states)
            go = self._p.go_net.forward(states)
            sp = self._p.sp_net.forward(states)
        except ModelError as error:
            raise EngineRefusal("model_error", str(error)) from error
        motif = extract(scored_form)
        localization = self._localization(loc.scores, group)
        signal = self._signal(sp.detection, sp.cleavage)
        go_rows = self._go(go.scores)
        top = max(localization["rows"], key=lambda r: r["probability"])["code"]
        calibrations = self._p.calibration_versions
        sources = {
            "localization": self._task_source("localization", True),
            "membrane_type": {
                "source": SOURCE_MOTIF,
                "calibrated": False,
                "bundle_version": None,
                "bundle_sha256": None,
                "calibration_version": None,
            },
            "signal_peptide": self._task_source("signal_peptide", True),
            "go_function": self._task_source("go_function", True),
        }
        features = {
            "signal_peptide_score": round(signal["probability_detected"], 4),
            "h_region": motif.h_region,
            "tm_segments": motif.tm_segments,
            "nls": motif.nls,
            "kdel": motif.kdel,
            "pts1": motif.pts1,
            "lpxtg": motif.lpxtg,
            "mtp_score": round(motif.mtp_score, 4),
            "ctp_score": round(motif.ctp_score, 4),
            "source": SOURCE_MOTIF,
        }
        return {
            "id": seq_id,
            "length": len(sequence),
            "sequence": sequence,
            "sequence_sha256": sha,
            "organism_group": group,
            "localization": localization["rows"],
            "top_location": top,
            "abstained": localization["none_cleared"],
            "no_confident_call": localization["none_cleared"],
            "calibration_version": calibrations["localization"],
            "threshold_policy": POLICY_PER_LABEL,
            "task_sources": sources,
            "tau": None,
            "temperature": None,
            "membrane_type": membrane_type(motif),
            "sorting_signal": signal["block"],
            "go_function": go_rows,
            "features": features,
            "hydropathy": motif.hydropathy,
            "attribution": residue_attribution(scored_form, motif, top),
            "attribution_source": SOURCE_MOTIF,
            "consistency_flags": [],
            "evidence": [],
            "evidence_go": [],
            "evidence_status": {},
            "evidence_agreement": None,
            "provenance": self.provenance,
            "_motif": motif,
        }

    def _task_source(self, task: str, calibrated: bool) -> dict[str, Any]:
        bundle = self._p.bundles[task]
        return {
            "source": SOURCE_MODEL,
            "calibrated": calibrated,
            "bundle_version": bundle.version,
            "bundle_sha256": str(bundle.document["bundle_sha256"]),
            "calibration_version": self._p.calibration_versions[task],
        }

    def _localization(self, raw: Any, group: str) -> dict[str, Any]:
        calibrators = self._p.bundles["localization"].calibrators
        values = [float(v) for v in raw]
        calibrated = calibrators.calibrate(values)
        shown = ORGANISM_GROUPS[group]["locs"]
        rows = []
        for entry, score, probability in zip(calibrators.labels, values, calibrated, strict=True):
            code = self._p.codes[entry.label]
            if code not in shown:
                continue
            rows.append(
                {
                    "code": code,
                    "name": LOC_NAMES[code],
                    "class_name": entry.label,
                    "probability": round(probability, 4),
                    "raw": round(score, 4),
                    "predicted": probability >= entry.threshold,
                    "source": SOURCE_MODEL,
                    "threshold": round(entry.threshold, 4),
                    "calibrated": entry.calibrated,
                    "calibration_method": entry.method,
                    "confidence_band": confidence_band(probability, entry.threshold),
                }
            )
        order = {code: i for i, code in enumerate(shown)}
        rows.sort(key=lambda r: order[str(r["code"])])
        return {"rows": rows, "none_cleared": not any(r["predicted"] for r in rows)}

    def _signal(self, detection: float, cleavage: int) -> dict[str, Any]:
        entry = self._p.bundles["signal_peptide"].calibrators.labels[0]
        probability = entry.apply(detection)
        detected = probability >= entry.threshold
        block = {
            "type": "SP" if detected else "none",
            "probability": round(probability if detected else 1.0 - probability, 4),
            "probability_detected": round(probability, 4),
            "raw": round(detection, 4),
            "region": [0, cleavage] if detected else None,
            "cleavage_site": cleavage if detected else None,
            "candidates": {"SP": round(probability, 4)},
            "untrained": ["mTP", "cTP"],
            "threshold": round(entry.threshold, 4),
            "source": SOURCE_MODEL,
            "calibrated": entry.calibrated,
        }
        return {"block": block, "probability_detected": probability}

    def _go(self, raw: Any) -> list[dict[str, Any]]:
        calibrators: CalibratorSet = self._p.bundles["go_function"].calibrators
        values = [float(v) for v in raw]
        calibrated = calibrators.calibrate(values)
        propagated = [
            max(calibrated[j] for j in self._p.descendants[a]) for a in range(len(values))
        ]
        rows = []
        for entry, score, own, probability in zip(
            calibrators.labels, values, calibrated, propagated, strict=True
        ):
            term = self._p.vocabulary[entry.label]
            rows.append(
                {
                    "go_id": entry.label,
                    "name": term["name"],
                    "aspect": self._p.vocabulary_aspect,
                    "probability": round(probability, 4),
                    "calibrated_before_propagation": round(own, 4),
                    "raw": round(score, 4),
                    "predicted": probability >= entry.threshold,
                    "threshold": round(entry.threshold, 4),
                    "source": SOURCE_MODEL,
                    "calibrated": entry.calibrated,
                }
            )
        rows.sort(key=lambda r: (-r["probability"], r["go_id"]))
        return rows

    # -- stage 4: consistency --------------------------------------------------------
    def _consistency(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate the hash-bound rule set. Each rule names the engine that fed it."""
        motif = result.pop("_motif")
        signal = result["sorting_signal"]
        sp_detected = signal["type"] == "SP"
        by_code = {row["code"]: row for row in result["localization"]}
        membrane = result["membrane_type"]
        fired = []
        for rule in RULES["rules"]:
            when, basis, hit = rule["when"], {}, False
            if when == "loc_and_signal":
                row = by_code.get(rule["loc"])
                hit = bool(row and row["predicted"]) and sp_detected
                basis = {"localization": SOURCE_MODEL, "signal_peptide": SOURCE_MODEL}
            elif when == "retention_without_entry":
                hit = motif.kdel is not None and not sp_detected
                basis = {"retention_motif": SOURCE_MOTIF, "signal_peptide": SOURCE_MODEL}
            elif when == "dual_nterminal":
                # The mitochondrial transit peptide has no trained head (ADR-0013): its score
                # is the motif engine's, and the flag says so.
                hit = sp_detected and motif.mtp_score > _unit_midpoint()
                basis = {"signal_peptide": SOURCE_MODEL, "mitochondrial_transit": SOURCE_MOTIF}
            elif when == "tm_but_secreted":
                hit = (
                    membrane["transmembrane"] > _unit_midpoint()
                    and result["top_location"] in self._p.secreted_codes
                )
                basis = {"membrane_type": SOURCE_MOTIF, "localization": SOURCE_MODEL}
            elif when == "pts1_with_sp":
                hit = motif.pts1 is not None and sp_detected
                basis = {"pts1_motif": SOURCE_MOTIF, "signal_peptide": SOURCE_MODEL}
            elif when == "evidence_disagrees":
                continue  # inactive in Phase 14: see ``rules_inactive`` in the provenance
            if hit:
                fired.append(
                    {
                        "rule_id": rule["id"],
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "basis": basis,
                    }
                )
        return fired

    # -- stage 5: evidence -----------------------------------------------------------
    def _evidence(self, queries: Mapping[int, str]) -> dict[int, Evidence]:
        plan = self._p.search_plan
        if plan is None:
            return {i: _no_evidence(self._p.evidence_reason or "no local index") for i in queries}
        try:
            found = search(plan, {str(i): s for i, s in queries.items()})
        except EvidenceUnavailable as error:
            return {i: _no_evidence(str(error)) for i in queries}
        out: dict[int, Evidence] = {}
        for index in queries:
            out[index] = self._summarise_hits(found.get(str(index), []))
        return out

    def _summarise_hits(self, hits: Sequence[Hit]) -> Evidence:
        rule = self._p.vote_rule
        terms_of = self._p.terms
        ranked = homology.passing(hits, rule)
        chosen = homology.candidates(hits, rule, terms_of)
        transferred = homology.transfer_terms(chosen, terms_of)
        return Evidence(
            True, None, tuple(ranked[: self._p.top_k]), tuple(transferred), tuple(chosen)
        )

    def _attach_evidence(self, result: dict[str, Any], evidence: Evidence) -> None:
        result["evidence_status"] = {
            "available": evidence.available,
            "unavailable_reason": evidence.reason,
            "swissprot_release": self._p.pins.section("evidence")["swissprot_release"],
            "bundle_version": self._p.provenance["bundles"][EVIDENCE_TASK]["version"],
            "bundle_sha256": self._p.provenance["bundles"][EVIDENCE_TASK]["sha256"],
            "source": "homology",
        }
        if not evidence.available:
            return
        shown = self._p.pins.section("evidence")["hit_terms_shown"]
        result["evidence"] = [self._hit_row(h, shown) for h in evidence.hits]
        result["evidence_go"] = [self._term_row(t) for t in evidence.terms]
        result["evidence_agreement"] = self._agreement(result, evidence)

    def _hit_row(self, hit: Hit, shown: int) -> dict[str, Any]:
        terms = sorted(self._p.terms.get(hit.target, frozenset()))
        return {
            "accession": hit.target,
            "identity": round(hit.identity, 4),
            "evalue": hit.evalue,
            "bits": hit.bits,
            "query_coverage": round(hit.query_coverage, 4),
            "target_coverage": round(hit.target_coverage, 4),
            "alignment_length": hit.alignment_length,
            "exact_match": hit.identity >= 1.0
            and hit.query_coverage >= 1.0
            and hit.target_coverage >= 1.0,
            "go_term_count": len(terms),
            "go_terms": [self._describe(t) for t in terms[:shown] if t in self._p.ontology],
            "database": homology.DATABASE_NAME,
            "source": "homology",
        }

    def _describe(self, go_id: str) -> dict[str, str]:
        term: OboTerm = self._p.ontology[go_id]
        return {"go_id": go_id, "name": term.name, "aspect": term.aspect}

    def _term_row(self, term: TransferredTerm) -> dict[str, Any]:
        onto = self._p.ontology.get(term.term)
        return {
            "go_id": term.term,
            "name": onto.name if onto else None,
            "aspect": onto.aspect if onto else None,
            "score": round(term.score, 4),
            "source": term.source,
            "source_accession": term.source_accession,
            "identity": round(term.identity, 4),
            "evalue": term.evalue,
            "database": term.database,
            "in_model_vocabulary": term.term in self._p.vocabulary,
        }

    def _agreement(self, result: dict[str, Any], evidence: Evidence) -> dict[str, Any] | None:
        """Model and homology compared on the model's own vocabulary; they are never merged."""
        if not evidence.hits:
            return None
        votes = len(evidence.chosen)
        majority = {
            t.term
            for t in evidence.terms
            if t.term in self._p.vocabulary and _MAJORITY_NUMERATOR * round(t.score * votes) > votes
        }
        model = {row["go_id"] for row in result["go_function"] if row["predicted"]}
        shared, model_only, homology_only = model & majority, model - majority, majority - model
        top = evidence.hits[0]
        return {
            "top_hit_accession": top.target,
            "identity": round(top.identity, 4),
            "exact_match": result["evidence"][0]["exact_match"],
            "voting_hits": votes,
            "basis": "go_function vocabulary: model-predicted terms vs terms carried by a strict "
            "majority of the voting homologs",
            "shared": sorted(shared),
            "model_only": sorted(model_only),
            "homology_only": sorted(homology_only),
            "agrees": bool(shared) and not model_only and not homology_only,
        }


def _unit_midpoint() -> float:
    """The motif engine's own 'scored above even odds' cut-off, used only by rules R03 and R04.

    Those two rules are written as '> 0.5' in the rule text (``backend/rules/consistency.json``,
    hash-bound by every bundle); this is the rule's constant, not a decision threshold.
    """
    return 1.0 / _MAJORITY_NUMERATOR


def _no_evidence(reason: str) -> Evidence:
    return Evidence(False, reason, (), (), ())


# -- loading ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Parts:
    """Everything :class:`Engine` needs, built and verified by :func:`load_engine`."""

    pins: _Pins
    provenance: dict[str, Any]
    bundles: dict[str, Bundle]
    calibration_versions: dict[str, str | None]
    loc_net: AttentionHeadNet
    go_net: AttentionHeadNet
    sp_net: CrfTaggerNet
    store: ResidueStore
    codes: dict[str, str]
    groups: tuple[str, ...]
    secreted_codes: frozenset[str]
    vocabulary: dict[str, dict[str, Any]]
    vocabulary_aspect: str
    descendants: list[list[int]]
    ontology: dict[str, OboTerm]
    terms: Mapping[str, frozenset[str]]
    vote_rule: HitFilter
    search_plan: SearchPlan | None
    evidence_reason: str | None
    top_k: int
    manifests: dict[str, dict[str, Any]]
    gallery: list[dict[str, Any]]


def _tensors(path: Path) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415 - serving dependency, imported where it is used

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _descendants(vocabulary: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """For each term, the columns of itself and of every term that lists it as an ancestor."""
    column = {t["id"]: j for j, t in enumerate(vocabulary)}
    out: list[list[int]] = [[a] for a in range(len(vocabulary))]
    for j, term in enumerate(vocabulary):
        for ancestor in term["ancestors"]:
            out[column[ancestor]].append(j)
    return out


def load_engine(  # noqa: PLR0912, PLR0915
    settings: ServingSettings, repo_root: Path = REPO_ROOT
) -> Engine:
    """Verify and load everything; raise :class:`EngineStartError` listing every problem."""
    pins = _Pins(_read_json(repo_root / "configs" / "serving" / "engine.json"))
    versions = {str(k): str(v) for k, v in pins.section("bundles").items()}
    problems: list[str] = []
    expected = composite_version(versions)
    if expected != settings.MODEL_VERSION:
        problems.append(
            f"MODEL_VERSION is {settings.MODEL_VERSION!r}, the pinned bundles are {expected!r}"
        )
    data_root = settings.DATA_ROOT
    registry = settings.MODEL_REGISTRY
    rules = {
        "baseline_class_map": repo_root / "backend" / "rules" / "baseline_class_map.yaml",
        "consistency": repo_root / "backend" / "rules" / "consistency.json",
    }

    ontology_pin = pins.section("ontology")
    ontology_file = data_root / str(ontology_pin["file"])
    ontology_sha = ""
    ontology: dict[str, OboTerm] = {}
    if not ontology_file.is_file():
        problems.append(f"the ontology release {ontology_file} is missing")
    else:
        ontology_sha = file_sha256(ontology_file)
        if ontology_sha != ontology_pin["sha256"]:
            problems.append("the ontology file is not the release engine.json pins")
        else:
            ontology = read_obo(ontology_file)

    bundles: dict[str, Bundle] = {}
    for task in TASKS_WITH_WEIGHTS:
        weights = data_root / str(pins.section("weights")[task])
        try:
            bundles[task] = load_bundle(
                registry / task / versions[task],
                rules,
                weights=weights,
                ontology=ontology_sha if task == "go_function" else None,
            )
        except (BundleError, OSError) as error:
            problems.append(f"{task} bundle: {error}")

    evidence_pin = pins.section("evidence")
    evidence_root = settings.EVIDENCE_ROOT
    hit_meta_path = evidence_root / str(evidence_pin["hit_terms_meta"])
    hit_terms_path = evidence_root / str(evidence_pin["hit_terms"])
    index_directory = settings.SWISSPROT_DB_PATH
    evidence_reason: str | None = None
    evidence_doc: dict[str, Any] = {}
    exe = evidence_root / str(evidence_pin["executable"])
    if not index_directory.is_dir() or not hit_terms_path.is_file():
        evidence_reason = "no local Swiss-Prot index on this host"
    try:
        evidence_doc = load_evidence_bundle(
            registry / EVIDENCE_TASK / versions[EVIDENCE_TASK],
            index_directory=index_directory if evidence_reason is None else None,
            goa_sha256=str(pins.document["go_annotation_source_sha256"]),
            ontology_sha256=ontology_sha or None,
        )
    except (EvidenceBundleError, OSError) as error:
        problems.append(f"evidence bundle: {error}")
    if evidence_doc and evidence_reason is None:
        problems += _evidence_file_problems(evidence_doc, hit_meta_path, hit_terms_path, exe, pins)

    manifests: dict[str, dict[str, Any]] = {}
    for task, relative in pins.section("manifests").items():
        bundle_sha = (
            str(bundles[task].document["bundle_sha256"])
            if task in bundles
            else str(evidence_doc.get("bundle_sha256", ""))
        )
        path = data_root / str(relative)
        found = _manifest_problems(str(task), path, bundle_sha)
        problems += found
        if not found:
            manifests[str(task)] = _read_json(path)

    encoder = pins.section("encoder")
    cache_directory = settings.EMBED_CACHE / str(encoder["residue_cache"])
    store: ResidueStore | None = None
    try:
        problems += [f"residue cache: {p}" for p in verify_residue_cache(cache_directory)]
        index = read_residue_index(cache_directory)
        if int(index["spec"]["states"]["embedding_dim"]) != int(encoder["dimension"]):
            problems.append("the residue cache dimension is not the one engine.json pins")
        store = ResidueStore(cache_directory)
    except (EmbeddingError, OSError, KeyError, ValueError) as error:
        problems.append(f"residue cache: {error}")

    if problems or store is None:
        raise EngineStartError(
            "the service will not start:\n"
            + "\n".join(f"  - {p}" for p in problems or ["no cache"])
        )

    try:
        parts = _build(
            settings, pins, bundles, evidence_doc, ontology, manifests, store, evidence_reason, exe
        )
    except (ModelError, KeyError, ValueError, OSError) as error:
        raise EngineStartError(f"the service will not start:\n  - {error}") from error
    return Engine(parts)


def _evidence_file_problems(
    document: Mapping[str, Any], meta: Path, table: Path, exe: Path, pins: _Pins
) -> list[str]:
    found: list[str] = []
    go = document.get("go") or {}
    if not meta.is_file():
        return [f"{meta.name} is missing"]
    info = _read_json(meta)
    if file_sha256(table) != info.get("sha256") or info.get("sha256") != go.get("hit_terms_sha256"):
        found.append("the hit-term table is not the one the evidence bundle records")
    if info.get("corpus_sha256") != pins.document["go_annotation_source_sha256"]:
        found.append("the GO annotation source is not the one engine.json pins")
    engine = document.get("engine") or {}
    if exe.is_file() and file_sha256(exe) != engine.get("executable_sha256"):
        found.append("the search binary is not the executable the evidence bundle records")
    return found


def _build(  # noqa: PLR0913, PLR0917
    settings: ServingSettings,
    pins: _Pins,
    bundles: dict[str, Bundle],
    evidence_doc: dict[str, Any],
    ontology: dict[str, OboTerm],
    manifests: dict[str, dict[str, Any]],
    store: ResidueStore,
    evidence_reason: str | None,
    exe: Path,
) -> _Parts:
    eps = float(pins.section("network")["layer_norm_eps"])
    weights_root = settings.DATA_ROOT
    weights = pins.section("weights")
    loc_net = AttentionHeadNet(_tensors(weights_root / str(weights["localization"])), eps)
    go_net = AttentionHeadNet(_tensors(weights_root / str(weights["go_function"])), eps)
    crf = pins.section("crf")
    spec = TagSpec(
        names=tuple(crf["tags"]),
        outside=str(crf["outside"]),
        signal=str(crf["signal"]),
        forbidden=tuple((str(a), str(b)) for a, b in crf["forbidden_transitions"]),
        forbidden_score=float(crf["forbidden_score"]),
    )
    sp_head = bundles["signal_peptide"].document["head"]
    sp_net = CrfTaggerNet(
        _tensors(weights_root / str(weights["signal_peptide"])),
        spec,
        int(sp_head["length_cap"]),
        eps,
    )
    for net_dim in (loc_net.dim, go_net.dim, sp_net.dim):
        if net_dim != store.dim:
            raise ModelError("a head's input width differs from the cached state width")
    codes = {str(k): str(v) for k, v in pins.section("localization")["codes"].items()}
    loc_classes = tuple(bundles["localization"].calibrators.classes)
    if set(codes) != set(loc_classes) or loc_net.classes != len(loc_classes):
        raise ModelError("the localization classes differ from engine.json")
    go_doc = json.loads((bundles["go_function"].directory / "terms.json").read_bytes())
    vocabulary = {str(t["id"]): t for t in go_doc["terms"]}
    ordered = [vocabulary[c] for c in bundles["go_function"].calibrators.classes]
    if go_net.classes != len(ordered):
        raise ModelError("the GO head width differs from the vocabulary")
    thresholds = {e.threshold for e in bundles["go_function"].calibrators.labels}
    if len(thresholds) != 1:
        raise ModelError("the GO bundle must carry one global threshold")
    hit_filter = evidence_doc["hit_filter"]["vote"]
    rule = HitFilter(
        e_value=float(hit_filter["e_value"]),
        min_identity=float(hit_filter["min_identity"]),
        min_query_coverage=float(hit_filter["min_query_coverage"]),
        top_k=int(hit_filter["top_k"]),
    )
    plan: SearchPlan | None = None
    terms: Mapping[str, frozenset[str]] = {}
    reason = evidence_reason
    evidence_pin = pins.section("evidence")
    if reason is None and not exe.is_file():
        reason = "the search binary is not installed on this host"
    if reason is None:
        search_doc = evidence_doc["hit_filter"]["search"]
        evidence_root = settings.EVIDENCE_ROOT
        plan = SearchPlan(
            executable=exe,
            database=settings.SWISSPROT_DB_PATH / str(evidence_doc["index"]["database_name"]),
            work_directory=evidence_root / str(evidence_pin["work_directory"]),
            settings=homology.SearchSettings(
                sensitivity=float(search_doc["sensitivity"]),
                e_value=float(search_doc["e_value"]),
                max_seqs=int(search_doc["max_seqs"]),
                alignment_mode=int(search_doc["alignment_mode"]),
                threads=int(evidence_pin["threads"]),
                verbosity=int(evidence_pin["verbosity"]),
            ),
            threads=int(evidence_pin["threads"]),
            verbosity=int(evidence_pin["verbosity"]),
            base_timeout_seconds=float(settings.EVIDENCE_TIMEOUT_SECONDS),
            per_query_seconds=float(evidence_pin["per_query_timeout_seconds"]),
        )
        terms = TermStore(evidence_root / str(evidence_pin["hit_terms"]))
    shas = {t: str(b.document["bundle_sha256"]) for t, b in bundles.items()}
    shas[EVIDENCE_TASK] = str(evidence_doc["bundle_sha256"])
    frozen = sorted(str(m["frozen_sha256"]) for m in manifests.values())
    pin_bytes = canonical_json_bytes(pins.document)
    config_hash = hashlib.sha256(
        canonical_json_bytes({"bundles": shas, "pins": hashlib.sha256(pin_bytes).hexdigest()})
    ).hexdigest()[:_SHA_PREFIX]
    provenance = {
        "model_version": settings.MODEL_VERSION,
        "config_hash": config_hash,
        "rules_version": RULES["rules_version"],
        "experiment_id": "frozen-"
        + hashlib.sha256("".join(frozen).encode()).hexdigest()[:_SHA_PREFIX],
        "predictor": "trained heads over frozen ESM-2 150M states (cache-only encoder); "
        "membrane type and the scanner are the motif engine",
        "bundles": {
            t: {"version": versions_of(bundles, evidence_doc, t), "sha256": shas[t]} for t in shas
        },
        "rules_inactive": ["R06"],
        "scope_label": str(bundles["localization"].document["scope_label"]),
        "encoder": {"mode": "cache_only", "cached_sequences": len(store.lengths)},
    }
    secreted = frozenset({codes["extracellular"]})
    gallery = _read_gallery(settings.DATA_ROOT / str(pins.section("gallery")["file"]))
    return _Parts(
        pins=pins,
        provenance=provenance,
        bundles=bundles,
        calibration_versions={
            t: bundles[t].calibrators.calibration_version for t in TASKS_WITH_WEIGHTS
        },
        loc_net=loc_net,
        go_net=go_net,
        sp_net=sp_net,
        store=store,
        codes=codes,
        groups=tuple(str(g) for g in pins.document["organism_groups_with_trained_heads"]),
        secreted_codes=secreted,
        vocabulary=vocabulary,
        vocabulary_aspect=str(go_doc["ontology"]["namespace"]),
        descendants=_descendants(ordered),
        ontology=ontology,
        terms=terms,
        vote_rule=rule,
        search_plan=plan,
        evidence_reason=reason,
        top_k=settings.EVIDENCE_TOP_K,
        manifests=manifests,
        gallery=gallery,
    )


def versions_of(bundles: Mapping[str, Bundle], evidence_doc: Mapping[str, Any], task: str) -> str:
    """The SemVer of one loaded bundle."""
    return bundles[task].version if task in bundles else str(evidence_doc["bundle_version"])


def _read_gallery(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    document = _read_json(path)
    entries = document.get("examples")
    return [dict(e) for e in entries] if isinstance(entries, list) else []


__all__ = [
    "Engine",
    "EngineRefusal",
    "EngineStartError",
    "composite_version",
    "load_engine",
]
