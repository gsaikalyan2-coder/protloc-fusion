"""Phase 3: the motif engine reports its provenance truthfully.

Acceptance test 4's backend half. The screens read these fields; they must say what
the engine actually did — one fixed tau, no fitted calibrator, motif source — and
must not echo a placeholder (the engine's tau is not 0.5).
"""

from engine_fixtures import demo_results
import pytest

from protloc.domain.types import ConfidenceBand, ThresholdPolicy
from protloc.inference import service

RESULTS = demo_results()
ABSTAINING = {"nuclear_factor", "outer_membrane"}


def test_the_engine_reports_the_tau_it_actually_applied():
    tau = service.TRUST["tau"]
    for r in RESULTS:
        assert r["tau"] == tau
        for row in r["localization"]:
            assert row["threshold"] == tau
            assert row["predicted"] is (row["probability"] >= row["threshold"])


def test_every_localization_row_says_motif_and_uncalibrated():
    for r in RESULTS:
        for row in r["localization"]:
            assert row["source"] == "motif"
            assert row["calibrated"] is False


def test_every_reported_task_carries_a_source():
    for r in RESULTS:
        assert set(r["task_sources"]) == set(service.TASK_SOURCE_KEYS)
        assert all(
            v == {"source": "motif", "calibrated": False} for v in r["task_sources"].values()
        )


def test_top_level_policy_fields():
    for r in RESULTS:
        assert r["threshold_policy"] == ThresholdPolicy.FIXED.value
        assert r["calibration_version"] is None
    assert service.TRUST["threshold_policy"] == "fixed"
    assert service.TRUST["calibration_version"] is None
    assert service.TRUST["calibrated"] is False


def test_no_confident_call_is_set_exactly_when_no_label_clears_its_threshold():
    for r in RESULTS:
        cleared = [row for row in r["localization"] if row["predicted"]]
        assert r["no_confident_call"] is (not cleared), r["id"]
        assert r["no_confident_call"] is r["abstained"], r["id"]


def test_gallery_has_sequences_that_clear_no_threshold():
    """Acceptance 4's engineered inputs come from the real gallery, not a literal."""
    got = {r["id"] for r in RESULTS if r["no_confident_call"]}
    assert got == ABSTAINING


@pytest.mark.parametrize("tau", [0.3, 0.5, 0.67, 0.9])
def test_band_is_none_below_threshold_and_monotone_above(tau: float):
    probs = [i / 100 for i in range(101)]
    order = [ConfidenceBand.NONE, ConfidenceBand.LOW, ConfidenceBand.MEDIUM, ConfidenceBand.HIGH]
    ranks = [order.index(ConfidenceBand(service.confidence_band(p, tau))) for p in probs]
    assert ranks == sorted(ranks)
    for p, rank in zip(probs, ranks, strict=True):
        assert (rank == 0) is (p < tau)
    assert service.confidence_band(1.0, tau) == "high"


def test_band_at_threshold_one_does_not_divide_by_zero():
    assert service.confidence_band(1.0, 1.0) == "high"
    assert service.confidence_band(0.99, 1.0) == "none"


def test_rows_carry_a_band_consistent_with_the_prediction():
    for r in RESULTS:
        for row in r["localization"]:
            assert (row["confidence_band"] == "none") is (not row["predicted"])
