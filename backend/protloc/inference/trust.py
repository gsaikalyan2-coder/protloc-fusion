"""The Trust screen's payload, read from the frozen Phase 10 calibration manifest (Phase 14).

Layer L6 (inference). Every number here was measured in Phase 10 on validation partitions 1-4 of
the 500-sequence subset; this module reads them and computes nothing new. Partition 5 is not
read. The pre-Phase-14 payload (a temperature, one global tau and a synthetic test split) no
longer exists, because the served path has neither.
"""

from collections.abc import Mapping
from typing import Any

from protloc.features.classical import LOC_NAMES

_SIGNIFICANCE = "the 95% bootstrap interval of the ECE change excludes zero"


def _interval(block: Mapping[str, Any]) -> dict[str, float]:
    return {
        "value": float(block["value"]),
        "ci_low": float(block["ci_low"]),
        "ci_high": float(block["ci_high"]),
    }


def trust_payload(
    manifest: Mapping[str, Any], codes: Mapping[str, str], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-label calibration, reliability bins and macro numbers of the served localization head."""
    frozen = manifest["frozen"]
    calibration = frozen["validation"]["calibration"]
    macro = calibration["macro"]
    fits = {m["label"]: m for m in frozen["labels"]}
    reliability = calibration["reliability"]
    labels: list[dict[str, Any]] = []
    for label in frozen["classes"]:
        fit = fits[label]
        per = calibration["per_label"][label]
        delta = _interval(per["ece_delta"])
        code = codes[label]
        labels.append(
            {
                "label": label,
                "code": code,
                "name": LOC_NAMES[code],
                "method": fit["method"],
                "calibrated": bool(fit["calibrated"]),
                "threshold": float(fit["threshold"]),
                "positives": int(fit["positives"]),
                "negatives": int(fit["negatives"]),
                "ece_raw": _interval(per["ece_raw"]),
                "ece_calibrated": _interval(per["ece_calibrated"]),
                "ece_delta": delta,
                "ece_change_significant": delta["ci_high"] < 0 or delta["ci_low"] > 0,
                "reliability": reliability[label],
            }
        )
    return {
        "threshold_policy": "per_label_mcc",
        "calibration_version": frozen["bundle"]["calibration_version"],
        "bundle_version": frozen["bundle"]["version"],
        "bundle_sha256": frozen["bundle"]["bundle_sha256"],
        "scope_label": frozen["scope_label"],
        "n_sequences": int(frozen["subset"]["sequences"]),
        "n_labelled_rows": int(frozen["subset"]["labelled_rows_per_task"]["localization"]),
        "validation_partitions": list(frozen["protocol"]["validation_partitions"]),
        "bins": int(frozen["protocol"]["ece"]["bins"]),
        "macro": {k: _interval(v) for k, v in macro.items()},
        "significance_rule": _SIGNIFICANCE,
        "labels": labels,
        "tau": None,
        "temperature": None,
        "provenance": dict(provenance),
    }
