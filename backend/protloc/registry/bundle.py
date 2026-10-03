"""The registry bundle of a head: weights hash + calibrator + rules, verified atomically.

Layer L3. Standard library only. A head's weights, the calibrator fitted on them and the rule
files that fix its class order are one deployable unit (``PROJECT_PLAN.md`` section 6,
"Unversioned artifacts"): a checkpoint served with another checkpoint's calibrator produces
confidently wrong probabilities and is nearly undetectable. ``load_bundle`` therefore recomputes
every hash the bundle records and raises :class:`BundleError` on any mismatch, so the service
refuses to start on a bundle whose checksums do not verify.

A bundle is a directory ``registry/<task>/<version>/`` holding ``bundle.json`` and
``calibrator.json``. The weights are not inside it; the bundle records their SHA-256 and, when
the file is supplied, the loader hashes it too.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from protloc.calibration.calibrator import (
    CalibratorError,
    CalibratorSet,
    canonical_json_bytes,
    is_sha256,
    set_from_dict,
    sha256_hex,
)

SCHEMA_VERSION = 1
_CHUNK = 1 << 20
BUNDLE_FILE = "bundle.json"
CALIBRATOR_FILE = "calibrator.json"
RULE_KEYS: tuple[str, ...] = ("baseline_class_map", "consistency")


class BundleError(ValueError):
    """A registry bundle does not verify; the service must not use it."""


@dataclass(frozen=True, slots=True)
class Bundle:
    """A verified bundle: its directory, its document and its calibrator set."""

    directory: Path
    document: Mapping[str, Any]
    calibrators: CalibratorSet

    @property
    def version(self) -> str:
        """The bundle's SemVer."""
        return str(self.document["bundle_version"])


def bundle_digest(document: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical bundle content, excluding the ``bundle_sha256`` field itself."""
    body = {key: value for key, value in document.items() if key != "bundle_sha256"}
    return sha256_hex(canonical_json_bytes(body))


def build_document(
    *,
    version: str,
    head: Mapping[str, Any],
    calibrator: CalibratorSet,
    rules: Mapping[str, str],
    scope: Mapping[str, str],
) -> dict[str, Any]:
    """The ``bundle.json`` content for a head and the calibrator fitted on it."""
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": version,
        "task": calibrator.task,
        "scope": scope["scope"],
        "scope_label": scope["scope_label"],
        "head": dict(head),
        "calibrator": {
            "file": CALIBRATOR_FILE,
            "sha256": sha256_hex(calibrator.to_bytes()),
            "calibration_version": calibrator.calibration_version,
        },
        "rules": dict(rules),
    }
    document["bundle_sha256"] = bundle_digest(document)
    return document


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large checkpoint is never held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BundleError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise BundleError(f"{name}: top level must be a JSON object")
    return {str(k): v for k, v in data.items()}


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise BundleError(f"{path.name} cannot be read: {error}") from error


def _section(document: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise BundleError(f"bundle.json is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def _document_problems(document: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        found.append("schema_version is not the one this code reads")
    if document.get("bundle_sha256") != bundle_digest(document):
        found.append("bundle_sha256 does not match the bundle content")
    head = _section(document, "head")
    for key in ("weights_sha256", "frozen_sha256", "head_config_sha256"):
        if not is_sha256(head.get(key)):
            found.append(f"head.{key} is not a SHA-256")
    rules = _section(document, "rules")
    if set(rules) != {f"{key}_sha256" for key in RULE_KEYS}:
        found.append(f"rules must hold exactly {[f'{key}_sha256' for key in RULE_KEYS]}")
    return found


def _calibrator_problems(
    document: Mapping[str, Any], raw: bytes, calibrators: CalibratorSet
) -> list[str]:
    head = _section(document, "head")
    entry = _section(document, "calibrator")
    found: list[str] = []
    if entry.get("sha256") != sha256_hex(raw):
        found.append("calibrator.json does not match the SHA-256 the bundle records")
    if entry.get("calibration_version") != calibrators.calibration_version:
        found.append("calibration_version differs between bundle.json and calibrator.json")
    if calibrators.head_weights_sha256 != head.get("weights_sha256"):
        found.append("the calibrator was fitted on different head weights than the bundle names")
    if calibrators.head_frozen_sha256 != head.get("frozen_sha256"):
        found.append("the calibrator was fitted on a different frozen head manifest")
    if list(calibrators.classes) != list(head.get("classes", [])):
        found.append("the calibrator and the head disagree on the class order")
    if calibrators.task != document.get("task"):
        found.append("the calibrator is for another task")
    return found


def _rule_problems(document: Mapping[str, Any], rules: Mapping[str, Path]) -> list[str]:
    recorded = _section(document, "rules")
    found: list[str] = []
    for key in RULE_KEYS:
        path = rules.get(key)
        if path is None:
            found.append(f"no path was given for the rule file {key!r}")
        elif not path.is_file() or recorded.get(f"{key}_sha256") != file_sha256(path):
            found.append(f"rule file {key!r} does not match the SHA-256 the bundle records")
    return found


def load_bundle(directory: Path, rules: Mapping[str, Path], weights: Path | None = None) -> Bundle:
    """Load and verify a bundle; raise :class:`BundleError` unless every hash agrees.

    ``rules`` maps ``baseline_class_map`` and ``consistency`` to the files to hash. ``weights``
    is optional: CI has no weights, the service passes them and gets the weights hash checked.
    """
    document = _parse_object(_read_bytes(directory / BUNDLE_FILE), BUNDLE_FILE)
    problems = _document_problems(document)
    raw = _read_bytes(directory / CALIBRATOR_FILE)
    try:
        calibrators = set_from_dict(_parse_object(raw, CALIBRATOR_FILE))
    except CalibratorError as error:
        raise BundleError(f"{CALIBRATOR_FILE} is unsound: {error}") from error
    problems += _calibrator_problems(document, raw, calibrators)
    problems += _rule_problems(document, rules)
    if weights is not None:
        expected = _section(document, "head").get("weights_sha256")
        if not weights.is_file() or file_sha256(weights) != expected:
            problems.append("the weights file is not the one the bundle records")
    if problems:
        raise BundleError("; ".join(problems))
    return Bundle(directory=directory, document=document, calibrators=calibrators)
