"""Phase 3 acceptance tests 1 and 2: round trip, and backward compatibility.

Fixtures are the live motif engine's real outputs (``engine_fixtures``), not
hand-written literals; the golden set that will replace them lands in Phase 4.
``fixtures/pre_phase3_results.json`` was captured from the v0.2.0 contract before
this phase widened it, so it is a genuine pre-Phase-3 payload.
"""

import json
from pathlib import Path
from typing import Any

from engine_fixtures import demo_results, widened_result_dict
import pytest

from protloc.domain.contracts import (
    JobProgress,
    JobResource,
    SequenceResult,
    TaskPrediction,
    utc_now,
)
from protloc.domain.types import ConfidenceBand, EngineSource, JobStatus, ThresholdPolicy

LEGACY = json.loads((Path(__file__).parent / "fixtures" / "pre_phase3_results.json").read_text())

NEW_TASK_FIELDS = ("source", "threshold", "calibrated", "confidence_band")
NEW_RESULT_FIELDS = ("no_confident_call", "calibration_version", "threshold_policy")


def _widened() -> list[dict[str, Any]]:
    return [widened_result_dict(r, i) for i, r in enumerate(demo_results())]


@pytest.mark.parametrize("payload", _widened(), ids=lambda p: p["accession"])
def test_sequence_result_round_trips(payload: dict[str, Any]):
    """Model.model_validate(Model(...).model_dump(mode="json")) is the identity."""
    model = SequenceResult.model_validate(payload)
    assert SequenceResult.model_validate(model.model_dump(mode="json")) == model


def test_job_resource_with_results_round_trips():
    results = [SequenceResult.model_validate(p) for p in _widened()]
    now = utc_now()
    job = JobResource(
        job_id="9f2c1e84-3b7a-4d15-8e60-2a5c9d3f1b04",  # type: ignore[arg-type]
        status=JobStatus.COMPLETED,
        sequence_count=len(results),
        created_at=now,
        updated_at=now,
        progress=JobProgress(total=len(results), completed_count=len(results), failed_count=0),
        results=results,
        model_version=None,
        config_hash=None,
        rules_version=None,
        experiment_id=None,
    )
    assert JobResource.model_validate(job.model_dump(mode="json")) == job


def test_legacy_fixtures_really_are_pre_phase3():
    """The captured payloads carry none of the new fields."""
    for payload in LEGACY:
        assert not set(NEW_RESULT_FIELDS) & set(payload)
        for pred in payload["predictions"]:
            assert not set(NEW_TASK_FIELDS) & set(pred)


@pytest.mark.parametrize("payload", LEGACY, ids=lambda p: p["accession"])
def test_a_pre_phase3_payload_validates_with_documented_defaults(payload: dict[str, Any]):
    result = SequenceResult.model_validate(payload)
    assert result.no_confident_call is False
    assert result.calibration_version is None
    assert result.threshold_policy is ThresholdPolicy.FIXED
    for pred in result.predictions:
        assert pred.source is EngineSource.MOTIF
        assert pred.calibrated is False
        assert pred.confidence_band is ConfidenceBand.NONE
        # no literal default: the decision threshold was the payload's own tau
        assert pred.threshold == pred.abstention_threshold


def test_every_new_field_but_threshold_has_a_default():
    """Additive: only `threshold` is derived rather than defaulted, and why is tested above."""
    for name in NEW_TASK_FIELDS:
        required = TaskPrediction.model_fields[name].is_required()
        assert required is (name == "threshold"), name
    for name in NEW_RESULT_FIELDS:
        assert not SequenceResult.model_fields[name].is_required(), name


def test_unknown_fields_are_still_rejected():
    """Widening added named fields; it did not loosen extra='forbid'."""
    payload = {**LEGACY[0], "surprise": 1}
    with pytest.raises(ValueError, match="surprise"):
        SequenceResult.model_validate(payload)
