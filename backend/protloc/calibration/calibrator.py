"""Fitted per-label calibrators and their thresholds, applied in pure Python.

Layer L4. Standard library only, so the serving image can load a calibrator without numpy or
scikit-learn. A calibrator is plain data: isotonic as its fitted knots, Platt as two numbers,
an uncalibrated label as the identity. It serialises to JSON without pickle, which is what lets
it live in the registry bundle (``protloc.registry.bundle``) and be checksummed there.

The threshold tau of each label is read from this metadata at runtime; it is never a literal
in code (CLAUDE.md "No hardcoded tau").
"""

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
import math
from typing import Any

SCHEMA_VERSION = 1
ISOTONIC = "isotonic"
PLATT = "platt"
UNCALIBRATED = "uncalibrated"
METHODS: tuple[str, ...] = (ISOTONIC, PLATT, UNCALIBRATED)
_SHA_LENGTH = 64
_HEX = frozenset("0123456789abcdef")


class CalibratorError(ValueError):
    """A calibrator document is malformed or inconsistent."""


def canonical_json_bytes(obj: object) -> bytes:
    """Deterministic JSON bytes; identical to ``protloc.data.manifest.canonical_json_bytes``."""
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)
    return text.encode("ascii") + b"\n"


def sha256_hex(data: bytes) -> str:
    """SHA-256 of some bytes as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def is_sha256(value: object) -> bool:
    """True for a 64-character lowercase hex string."""
    return isinstance(value, str) and len(value) == _SHA_LENGTH and set(value) <= _HEX


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(score: float, eps: float) -> float:
    """``log(s / (1 - s))`` with ``s`` clipped to ``[eps, 1 - eps]`` so it stays finite."""
    clipped = min(max(score, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _interpolate(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation through the knots, clipped to the end values."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    right = bisect_right(xs, x)
    x0, x1 = xs[right - 1], xs[right]
    y0, y1 = ys[right - 1], ys[right]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


@dataclass(frozen=True, slots=True)
class LabelCalibrator:
    """The calibrator and decision threshold of one label.

    ``positives`` and ``negatives`` are the counts of the rows it was fitted on (the eligibility
    rule's input). ``threshold`` is the MCC-optimal score on the CALIBRATED scale and is NaN until
    thresholds are tuned; a set with a NaN threshold fails ``problems``.
    """

    label: str
    method: str
    positives: int
    negatives: int
    threshold: float
    knots_x: tuple[float, ...] = ()
    knots_y: tuple[float, ...] = ()
    slope: float = 0.0
    intercept: float = 0.0
    logit_eps: float = 0.0

    @property
    def calibrated(self) -> bool:
        """False only for an uncalibrated label, whose scores pass through unchanged."""
        return self.method != UNCALIBRATED

    def apply(self, score: float) -> float:
        """The calibrated probability for one raw score."""
        if self.method == ISOTONIC:
            return _interpolate(self.knots_x, self.knots_y, score)
        if self.method == PLATT:
            return _sigmoid(self.slope * logit(score, self.logit_eps) + self.intercept)
        return score

    def problems(self) -> list[str]:
        """Every way this calibrator is unusable; empty if sound."""
        found: list[str] = []
        if self.method not in METHODS:
            return [f"{self.label}: unknown method {self.method!r}"]
        if not math.isfinite(self.threshold):
            found.append(f"{self.label}: threshold is not a finite number")
        if self.positives < 0 or self.negatives < 0:
            found.append(f"{self.label}: negative row counts")
        if self.method == ISOTONIC:
            found += self._knot_problems()
        elif self.knots_x or self.knots_y:
            found.append(f"{self.label}: only isotonic carries knots")
        if self.method == PLATT:
            values = (self.slope, self.intercept, self.logit_eps)
            if not all(math.isfinite(v) for v in values) or not 0.0 < self.logit_eps < 1.0:
                found.append(f"{self.label}: Platt parameters are not usable")
        return found

    def _knot_problems(self) -> list[str]:
        xs, ys = self.knots_x, self.knots_y
        found: list[str] = []
        if not xs or len(xs) != len(ys):
            return [f"{self.label}: isotonic needs equal, non-empty knot lists"]
        if not all(math.isfinite(v) for v in (*xs, *ys)):
            found.append(f"{self.label}: knots must be finite")
        if any(b <= a for a, b in pairwise(xs)):
            found.append(f"{self.label}: knot scores must strictly increase")
        if any(b < a for a, b in pairwise(ys)):
            found.append(f"{self.label}: knot values must not decrease (monotone)")
        if any(not 0.0 <= v <= 1.0 for v in ys):
            found.append(f"{self.label}: knot values must be probabilities")
        return found

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form; the inverse of :func:`label_from_dict`."""
        return {
            "label": self.label,
            "method": self.method,
            "calibrated": self.calibrated,
            "positives": self.positives,
            "negatives": self.negatives,
            "threshold": self.threshold,
            "knots_x": list(self.knots_x),
            "knots_y": list(self.knots_y),
            "slope": self.slope,
            "intercept": self.intercept,
            "logit_eps": self.logit_eps,
        }


def _floats(doc: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = doc.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
    ):
        raise CalibratorError(f"{key} must be a list of numbers")
    return tuple(float(v) for v in value)


def _number(doc: Mapping[str, Any], key: str) -> float:
    value = doc.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibratorError(f"{key} must be a number")
    return float(value)


def _count(doc: Mapping[str, Any], key: str) -> int:
    value = doc.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibratorError(f"{key} must be an integer")
    return value


def label_from_dict(doc: Mapping[str, Any]) -> LabelCalibrator:
    """Rebuild one label's calibrator from its JSON form, rejecting malformed fields."""
    return LabelCalibrator(
        label=str(doc.get("label", "")),
        method=str(doc.get("method", "")),
        positives=_count(doc, "positives"),
        negatives=_count(doc, "negatives"),
        threshold=_number(doc, "threshold"),
        knots_x=_floats(doc, "knots_x"),
        knots_y=_floats(doc, "knots_y"),
        slope=_number(doc, "slope"),
        intercept=_number(doc, "intercept"),
        logit_eps=_number(doc, "logit_eps"),
    )


@dataclass(frozen=True, slots=True)
class CalibratorSet:
    """Every label's calibrator for one head, bound to the weights it was fitted on."""

    task: str
    calibration_version: str
    classes: tuple[str, ...]
    head_weights_sha256: str
    head_frozen_sha256: str
    threshold_rule: str
    labels: tuple[LabelCalibrator, ...]

    def problems(self) -> list[str]:
        """Every way this set is unusable; empty if sound."""
        found: list[str] = []
        if tuple(c.label for c in self.labels) != self.classes:
            found.append("labels do not match the class order")
        if not is_sha256(self.head_weights_sha256) or not is_sha256(self.head_frozen_sha256):
            found.append("head hashes are not SHA-256")
        for entry in self.labels:
            found += entry.problems()
        return found

    def calibrate(self, scores: Sequence[float]) -> list[float]:
        """Calibrated probabilities for one row of raw scores (class order)."""
        if len(scores) != len(self.labels):
            raise CalibratorError("one score per label is required")
        return [entry.apply(s) for entry, s in zip(self.labels, scores, strict=True)]

    def decide(self, scores: Sequence[float]) -> list[bool]:
        """Which labels clear their own threshold, on the calibrated scale."""
        calibrated = self.calibrate(scores)
        return [p >= entry.threshold for entry, p in zip(self.labels, calibrated, strict=True)]

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form; the inverse of :func:`set_from_dict`."""
        return {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "calibration_version": self.calibration_version,
            "classes": list(self.classes),
            "head": {
                "weights_sha256": self.head_weights_sha256,
                "frozen_sha256": self.head_frozen_sha256,
            },
            "threshold_rule": self.threshold_rule,
            "labels": [entry.to_dict() for entry in self.labels],
        }

    def to_bytes(self) -> bytes:
        """The canonical bytes whose SHA-256 the bundle records."""
        return canonical_json_bytes(self.to_dict())


def set_from_dict(doc: Mapping[str, Any]) -> CalibratorSet:
    """Rebuild a calibrator set from its JSON form and refuse an unsound one."""
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise CalibratorError("schema_version is not the one this code reads")
    head = doc.get("head")
    labels = doc.get("labels")
    classes = doc.get("classes")
    if not isinstance(head, dict) or not isinstance(labels, list) or not isinstance(classes, list):
        raise CalibratorError("head, labels and classes are required")
    result = CalibratorSet(
        task=str(doc.get("task", "")),
        calibration_version=str(doc.get("calibration_version", "")),
        classes=tuple(str(c) for c in classes),
        head_weights_sha256=str(head.get("weights_sha256", "")),
        head_frozen_sha256=str(head.get("frozen_sha256", "")),
        threshold_rule=str(doc.get("threshold_rule", "")),
        labels=tuple(label_from_dict(entry) for entry in labels if isinstance(entry, dict)),
    )
    problems = result.problems()
    if problems:
        raise CalibratorError("; ".join(problems))
    return result
