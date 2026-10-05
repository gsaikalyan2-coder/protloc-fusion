"""Contract-shaped fixtures built from the live motif engine's real outputs.

The golden set arrives in Phase 4, so Phase 3 builds its fixtures from what the
engine actually returns for the gallery sequences in ``reference_demo.EXAMPLES``
rather than from hand-written literals. ``legacy_result_dict`` produces the
pre-Phase-3 ``SequenceResult`` shape (the one frozen in v0.2.0); the committed
``fixtures/pre_phase3_results.json`` is that function's output, captured before the
contract was widened. ``widened_result_dict`` adds the Phase 3 fields.

This is test support, not serving code: the demo-to-contract mapping is the seed of
Phase 14's engine swap, and it is deliberately not part of ``protloc/``.

Fields the engine cannot honestly supply are left out instead of invented: the GO
head (motif hits carry no probability), and homology E-values (the reference
search is k-mer based and reports none), so evidence is ``available=False`` with
the reason stated.
"""

from __future__ import annotations

from typing import Any

from protloc.data import reference_demo
from protloc.inference import motif as service

_GROUP = {
    "eukaryote": "eukarya",
    "plant": "eukarya",
    "gram_neg": "gram_negative",
    "gram_pos": "gram_positive",
}
_SEVERITY = {"blocked": "blocking", "warning": "warning", "uncertain": "note", "note": "note"}
_BAND_ORDER = ["none", "low", "medium", "high"]


def demo_results() -> list[dict[str, Any]]:
    """The engine's real output for every gallery example, in gallery order."""
    return [
        service.predict(e["sequence"], e["organism_group"], e["id"])
        for e in reference_demo.EXAMPLES
    ]


def legacy_result_dict(r: dict[str, Any], index: int) -> dict[str, Any]:
    """One engine result in the pre-Phase-3 ``SequenceResult`` shape."""
    prokaryote = _GROUP[r["organism_group"]] != "eukarya"
    loc_task = "prokaryotic_localization" if prokaryote else "localization"
    tau = r["tau"]
    mem = r["membrane_type"]
    sig = r["sorting_signal"]
    predictions: list[dict[str, Any]] = [
        {
            "task": loc_task,
            "labels": [row["code"] for row in r["localization"] if row["predicted"]],
            "probabilities": {row["code"]: row["probability"] for row in r["localization"]},
            "abstained": r["abstained"],
            "abstention_threshold": tau,
            "cleavage_site": None,
        },
        {
            "task": "membrane_type",
            "labels": [max(mem, key=lambda k: mem[k])],
            "probabilities": mem,
            "abstained": False,
            "abstention_threshold": tau,
            "cleavage_site": None,
        },
        {
            "task": "signal_peptide",
            "labels": [] if sig["type"] == "none" else [sig["type"]],
            "probabilities": sig["candidates"],
            "abstained": False,
            "abstention_threshold": tau,
            "cleavage_site": sig["cleavage_site"],
        },
    ]
    return {
        "index": index,
        "accession": r["id"],
        "sequence_sha256": r["sequence_sha256"],
        "length": r["length"],
        "organism_group": _GROUP[r["organism_group"]],
        "predictions": predictions,
        "consistency": [
            {
                "rule_id": f["rule_id"],
                "severity": _SEVERITY[f["severity"]],
                "message": f["message"],
                "tasks": [loc_task],
            }
            for f in r["consistency_flags"]
        ],
        "rules_checked": len(service.RULES["rules"]),
        "attributions": [
            {
                "task": loc_task,
                "residues": [
                    {"position": i + 1, "residue": r["sequence"][i], "score": s}
                    for i, s in enumerate(r["attribution"])
                ],
            }
        ],
        "evidence": {
            "available": False,
            "unavailable_reason": (
                "k-mer reference search reports no E-value; not a Swiss-Prot MMseqs2 hit"
            ),
            "homology": [],
            "go_annotations": [],
            "swissprot_release": None,
            "goa_release": None,
        },
        "error": None,
    }


def widened_result_dict(r: dict[str, Any], index: int) -> dict[str, Any]:
    """The same result with the Phase 3 fields as the engine actually reports them."""
    out = legacy_result_dict(r, index)
    rows = r["localization"]
    for pred in out["predictions"]:
        if pred["task"] in ("localization", "prokaryotic_localization"):
            pred["threshold"] = rows[0]["threshold"]
            pred["source"] = rows[0]["source"]
            pred["calibrated"] = rows[0]["calibrated"]
            pred["confidence_band"] = max(
                (row["confidence_band"] for row in rows), key=_BAND_ORDER.index
            )
        else:
            src = r["task_sources"][pred["task"]]
            pred["threshold"] = pred["abstention_threshold"]
            pred["source"] = src["source"]
            pred["calibrated"] = src["calibrated"]
            pred["confidence_band"] = "none"
    out["no_confident_call"] = r["no_confident_call"]
    out["calibration_version"] = r["calibration_version"]
    out["threshold_policy"] = r["threshold_policy"]
    return out
