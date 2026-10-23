"""Whole-job downloads (feature 6): the builders, then the route against a stub engine."""

from collections.abc import Iterator
import csv
from datetime import UTC, datetime
import io
import json
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from protloc.inference import export
from protloc.jobs import queue as jobs

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
PROV = {
    "model_version": "loc-0.10.0+x",
    "config_hash": "abc123",
    "rules_version": "r1",
    "experiment_id": "frozen-1",
    "scope_label": "subset (500)",
}


def _result(seq_id: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": seq_id,
        "length": 12,
        "organism_group": "eukaryote",
        "sequence": "MKTAYIAKQRQI",
        "sequence_sha256": "f" * 64,
        "top_location": "ER",
        "abstained": False,
        "no_confident_call": False,
        "localization": [
            {"code": "ER", "probability": 0.81, "predicted": True},
            {"code": "CYT", "probability": 0.10, "predicted": False},
            {"code": "GOL", "probability": 0.44, "predicted": True},
        ],
        "sorting_signal": {"type": "SP", "probability_detected": 0.97, "cleavage_site": 22},
        "go_function": [
            {"go_id": "GO:0005488", "probability": 0.7, "predicted": True},
            {"go_id": "GO:0003824", "probability": 0.1, "predicted": False},
        ],
        "consistency_flags": [{"rule_id": "R02", "severity": "warning"}],
        "evidence": [{"accession": "P12345", "identity": 0.93}],
        "evidence_status": {"available": True},
        "result_cache": "miss",
    }
    base.update(over)
    return base


def _job(**over: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "job_id": "9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04",
        "mode": "batch",
        "results": [_result("seq1"), _result("=cmd|' /C calc'!A0", top_location="CYT")],
        "failures": [{"id": "cold1", "code": "not_cached", "error": "not cached"}],
        "degraded": False,
        "degraded_reasons": [],
    }
    job.update(over)
    return job


def _csv_rows(text: str) -> list[dict[str, str]]:
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return list(csv.DictReader(io.StringIO(body)))


def test_csv_has_one_row_per_sequence_and_one_per_refusal() -> None:
    text = export.build(_job(), PROV, "csv", NOW)
    assert "# model_version: loc-0.10.0+x" in text
    assert "# exported_at: 2026-10-23T12:00:00+00:00" in text
    rows = _csv_rows(text)
    assert [r["status"] for r in rows] == ["scored", "scored", "refused"]
    first = rows[0]
    assert first["predicted_locations"] == "ER:0.81; GOL:0.44"
    assert first["predicted_go_terms"] == "GO:0005488:0.7"
    assert (first["sorting_signal"], first["cleavage_site"]) == ("SP", "22")
    assert (first["top_hit"], first["top_hit_identity"]) == ("P12345", "0.93")
    assert first["consistency_flags"] == "R02:warning"
    assert rows[2]["error_code"] == "not_cached"
    assert list(rows[0]) == list(export.CSV_COLUMNS)


def test_csv_neutralises_a_formula_in_a_user_id() -> None:
    rows = _csv_rows(export.build(_job(), PROV, "csv", NOW))
    assert rows[1]["id"].startswith("'=")


def test_csv_states_a_degraded_job() -> None:
    text = export.build(_job(degraded=True, degraded_reasons=["timed out"]), PROV, "csv", NOW)
    assert "# degraded: True timed out" in text


def test_json_keeps_every_result_and_states_provenance() -> None:
    body = json.loads(export.build(_job(), PROV, "json", NOW))
    assert body["export"]["model_version"] == "loc-0.10.0+x"
    assert body["export"]["scope"] == "subset (500)"
    assert body["results"][0]["id"] == "seq1"
    assert body["failures"][0]["code"] == "not_cached"


def test_fasta_wraps_at_60_and_cleans_ids() -> None:
    long = _result("a b/c", sequence="M" * 130)
    text = export.build(_job(results=[long]), PROV, "fasta", NOW)
    lines = text.splitlines()
    assert lines[0].startswith(">a_b_c|job_id=9f2c1e84")
    assert "model_version=loc-0.10.0+x" in lines[0]
    assert [len(x) for x in lines[1:]] == [60, 60, 10]


def test_formats_and_filenames() -> None:
    assert export.supports("batch", "csv")
    assert export.supports("scan", "json")
    assert not export.supports("scan", "csv")
    assert export.filename("9f2c1e84-3b7a", "csv") == "protloc_job_9f2c1e84.csv"
    with pytest.raises(ValueError, match="unknown format"):
        export.build(_job(), PROV, "xlsx", NOW)


# -- the route ---------------------------------------------------------------------------------


class _Stub:
    groups: ClassVar[tuple[str, ...]] = ("eukaryote",)
    cached_sequences = 500
    provenance: ClassVar[dict[str, str]] = PROV  # type: ignore[assignment]
    evidence_state: ClassVar[dict[str, Any]] = {"available": True, "reason": None}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    pytest.importorskip("numpy")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from protloc.api import app_v2  # noqa: PLC0415
    from protloc.inference import service  # noqa: PLC0415

    service.configure(_Stub())  # type: ignore[arg-type]
    monkeypatch.setattr(
        app_v2,
        "get_serving_settings",
        lambda: SimpleNamespace(MAX_BATCH_SEQUENCES=2, DATA_ROOT="."),
    )
    yield TestClient(app_v2.app)
    service.configure(None)


def _finished(result: dict[str, Any] | None = None, fail: bool = False) -> str:
    def run(progress: Any, cancelled: Any) -> dict[str, Any]:
        if fail:
            raise RuntimeError("boom")
        return result or {}

    job_id = str(jobs.submit("predict", 1, run)["job_id"])
    deadline = time.time() + 10
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["status"] not in ("queued", "running"):
            return job_id
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_the_route_serves_a_csv_attachment(client: Any) -> None:
    job_id = _finished({"mode": "batch", "results": [_result("s1")], "failures": []})
    r = client.get(f"/demo/jobs/{job_id}/results", params={"format": "csv"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert f'filename="protloc_job_{job_id[:8]}.csv"' in r.headers["content-disposition"]
    assert _csv_rows(r.text)[0]["id"] == "s1"


def test_the_route_defaults_to_json(client: Any) -> None:
    job_id = _finished({"mode": "predict", "results": [_result("s1")]})
    body = client.get(f"/demo/jobs/{job_id}/results").json()
    assert body["export"]["job_id"] == job_id


def test_a_scan_job_exports_as_json_only(client: Any) -> None:
    job_id = _finished({"mode": "scan", "scan": {"positions": []}})
    r = client.get(f"/demo/jobs/{job_id}/results", params={"format": "fasta"})
    assert (r.status_code, r.json()["error"]["code"]) == (422, "export_unsupported")
    assert client.get(f"/demo/jobs/{job_id}/results").status_code == 200


def test_route_refusals_have_stable_codes(client: Any) -> None:
    r = client.get("/demo/jobs/nope/results")
    assert (r.status_code, r.json()["error"]["code"]) == (404, "unknown_job")
    failed = _finished(fail=True)
    r = client.get(f"/demo/jobs/{failed}/results")
    assert (r.status_code, r.json()["error"]["code"]) == (409, "no_results")
    ok = _finished({"mode": "batch", "results": []})
    assert client.get(f"/demo/jobs/{ok}/results", params={"format": "xlsx"}).status_code == 422


def test_an_unfinished_job_is_409_and_retryable(client: Any) -> None:
    import threading  # noqa: PLC0415

    gate = threading.Event()

    def slow(progress: Any, cancelled: Any) -> dict[str, Any]:
        gate.wait(5)
        return {"mode": "predict", "results": []}

    job_id = str(jobs.submit("predict", 1, slow)["job_id"])
    try:
        r = client.get(f"/demo/jobs/{job_id}/results")
        assert (r.status_code, r.json()["error"]["code"]) == (409, "job_not_finished")
        assert r.json()["error"]["retryable"] is True
    finally:
        gate.set()
