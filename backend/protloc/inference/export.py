"""Whole-job downloads built on the server (feature 6).

Layer L1 (inference), standard library only. The browser already builds per-result files
(``frontend/src/services/download.ts``); this builds files for a **whole job**, so a batch of
twenty sequences is one CSV row per sequence, and so a script can fetch results with one GET
instead of rebuilding them. Three formats:

* ``json`` - every result exactly as the job holds it, plus the job's failures and a header
  stating the model version, scope and export time.
* ``csv`` - one row per scored sequence: the top location and every predicted location with
  its calibrated probability, the sorting signal and cleavage site, predicted GO terms, the
  consistency flags, the nearest homolog and whether evidence was available. ``#`` lines state
  the provenance. Refused records are listed as rows with their error code.
* ``fasta`` - the input sequences, with the job id and model version in each header.

A cell that a spreadsheet would read as a formula (it starts with ``=``, ``+``, ``-``, ``@``,
a tab or a carriage return) is prefixed with ``'``: a FASTA header is user input.
Nothing here scores anything; it only reshapes a finished job.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import io
import json
from typing import Any, Final

FORMATS: Final[dict[str, tuple[str, str]]] = {
    "json": ("application/json", "json"),
    "csv": ("text/csv; charset=utf-8", "csv"),
    "fasta": ("text/plain; charset=utf-8", "fasta"),
}
#: Which job modes carry per-sequence results a CSV or FASTA can be built from.
ROW_MODES: Final[frozenset[str]] = frozenset({"predict", "batch"})

CSV_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "status",
    "error_code",
    "length",
    "organism_group",
    "top_location",
    "predicted_locations",
    "abstained",
    "no_confident_call",
    "sorting_signal",
    "signal_probability",
    "cleavage_site",
    "predicted_go_terms",
    "consistency_flags",
    "evidence_available",
    "top_hit",
    "top_hit_identity",
    "result_cache",
    "sequence_sha256",
)

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def filename(job_id: str, fmt: str) -> str:
    """``protloc_job_<first 8 of the id>.<ext>``."""
    safe = "".join(ch for ch in job_id[:8] if ch.isalnum() or ch == "-")
    return f"protloc_job_{safe}.{FORMATS[fmt][1]}"


def supports(mode: str, fmt: str) -> bool:
    """JSON exports any job; CSV and FASTA need per-sequence results."""
    return fmt == "json" or mode in ROW_MODES


def _safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def _header(job: dict[str, Any], provenance: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        "format_note": (
            "ProtLoc Fusion whole-job export. Results are scoped as stated; result URLs are not "
            "access-controlled and expire."
        ),
        "exported_at": now.isoformat(timespec="seconds"),
        "job_id": job["job_id"],
        "mode": job.get("mode"),
        "model_version": provenance.get("model_version"),
        "config_hash": provenance.get("config_hash"),
        "rules_version": provenance.get("rules_version"),
        "experiment_id": provenance.get("experiment_id"),
        "scope": provenance.get("scope_label"),
        "degraded": job.get("degraded", False),
        "degraded_reasons": job.get("degraded_reasons", []),
    }


def _row(r: dict[str, Any]) -> dict[str, Any]:
    predicted = [loc for loc in r.get("localization", []) if loc.get("predicted")]
    signal = r.get("sorting_signal") or {}
    go = [g for g in r.get("go_function", []) if g.get("predicted")]
    flags = r.get("consistency_flags") or []
    hits = r.get("evidence") or []
    status = r.get("evidence_status") or {}
    top = hits[0] if hits else {}
    return {
        "id": r.get("id"),
        "status": "scored",
        "error_code": "",
        "length": r.get("length"),
        "organism_group": r.get("organism_group"),
        "top_location": r.get("top_location"),
        "predicted_locations": "; ".join(
            f"{loc['code']}:{loc['probability']}" for loc in predicted
        ),
        "abstained": r.get("abstained"),
        "no_confident_call": r.get("no_confident_call"),
        "sorting_signal": signal.get("type"),
        "signal_probability": signal.get("probability_detected"),
        "cleavage_site": signal.get("cleavage_site"),
        "predicted_go_terms": "; ".join(f"{g['go_id']}:{g['probability']}" for g in go),
        "consistency_flags": "; ".join(f"{f['rule_id']}:{f['severity']}" for f in flags),
        "evidence_available": status.get("available"),
        "top_hit": top.get("accession", ""),
        "top_hit_identity": top.get("identity", ""),
        "result_cache": r.get("result_cache", ""),
        "sequence_sha256": r.get("sequence_sha256"),
    }


def _failure_row(f: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS, "")
    row.update({"id": f.get("id"), "status": "refused", "error_code": f.get("code")})
    return row


def build(
    job: dict[str, Any], provenance: dict[str, Any], fmt: str, now: datetime | None = None
) -> str:
    """The file body for a completed ``job`` (its public view with results merged in)."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    stamp = now or datetime.now(UTC)
    header = _header(job, provenance, stamp)
    results: list[dict[str, Any]] = list(job.get("results") or [])
    if fmt == "json":
        body = {
            "export": header,
            "results": results,
            "failures": job.get("failures", []),
            "scan": job.get("scan"),
        }
        return json.dumps(body, indent=1, default=str) + "\n"
    if fmt == "fasta":
        lines: list[str] = []
        for r in results:
            ident = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(r["id"]))
            lines.append(
                f">{ident}|job_id={job['job_id']}|group={r.get('organism_group')}"
                f"|model_version={provenance.get('model_version')}|sha256={r.get('sequence_sha256')}"
            )
            seq = str(r.get("sequence", ""))
            lines.extend(seq[i : i + 60] for i in range(0, len(seq), 60))
        return "\n".join(lines) + "\n"
    out = io.StringIO()
    for key in ("exported_at", "job_id", "model_version", "config_hash", "rules_version", "scope"):
        out.write(f"# {key}: {header[key]}\n")
    out.write(
        f"# degraded: {header['degraded']} {'; '.join(header['degraded_reasons'])}\n".rstrip()
        + "\n"
    )
    out.write(
        "# Probabilities are calibrated; locations and GO terms list predicted labels only.\n"
    )
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in results:
        writer.writerow({k: _safe(v) for k, v in _row(r).items()})
    for f in job.get("failures") or []:
        writer.writerow({k: _safe(v) for k, v in _failure_row(f).items()})
    return out.getvalue()


__all__ = ["CSV_COLUMNS", "FORMATS", "ROW_MODES", "build", "filename", "supports"]
