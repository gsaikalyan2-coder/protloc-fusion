"""Attribution methods (Phase 15): shapes, sums, a hand-computed occlusion, caps, determinism."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from protloc.inference.attribution import (
    METHOD_CRF,
    METHOD_OCCLUSION,
    METHOD_POOLING,
    OcclusionSettings,
    attention_track,
    crf_track,
    occlusion_track,
    plan_windows,
    rounded,
    score_without,
)
from protloc.models.attention_head import AttentionHeadNet, ModelError
from protloc.models.crf_tagger import TaggerOutput

DIM, ATT, HIDDEN, CLASSES = 4, 3, 5, 3
EPS = 1e-5


def _tensors(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "att_in.weight": rng.normal(size=(ATT, DIM)),
        "att_in.bias": rng.normal(size=ATT),
        "att_out.weight": rng.normal(size=(1, ATT)),
        "att_out.bias": rng.normal(size=1),
        "norm.weight": rng.normal(size=DIM) + 1.0,
        "norm.bias": rng.normal(size=DIM),
        "fc1.weight": rng.normal(size=(HIDDEN, DIM)),
        "fc1.bias": rng.normal(size=HIDDEN),
        "fc2.weight": rng.normal(size=(CLASSES, HIDDEN)),
        "fc2.bias": rng.normal(size=CLASSES),
    }


def _reference(t: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """The head written out directly in float64, independent of the module under test."""
    energy = np.tanh(x @ t["att_in.weight"].T + t["att_in.bias"]) @ t["att_out.weight"].T
    energy = (energy + t["att_out.bias"])[:, 0]
    w = np.exp(energy - energy.max())
    w = w / w.sum()
    pooled = w @ x
    mu = pooled.mean()
    var = ((pooled - mu) ** 2).mean()
    normed = (pooled - mu) / math.sqrt(var + EPS) * t["norm.weight"] + t["norm.bias"]
    pre = normed @ t["fc1.weight"].T + t["fc1.bias"]
    hidden = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in pre])
    logits = hidden @ t["fc2.weight"].T + t["fc2.bias"]
    return 1.0 / (1.0 + np.exp(-logits))


@pytest.fixture(scope="module")
def head() -> AttentionHeadNet:
    return AttentionHeadNet(_tensors(), EPS)


@pytest.fixture(scope="module")
def states() -> np.ndarray:
    return np.random.default_rng(11).normal(size=(9, DIM)).astype(np.float32)


def test_reference_matches_head(head: AttentionHeadNet, states: np.ndarray) -> None:
    ours = head.forward(states).scores
    assert np.abs(ours - _reference(_tensors(), states.astype(np.float64))).max() < 1e-5


def test_attention_track_shape_and_sum(head: AttentionHeadNet, states: np.ndarray) -> None:
    track = attention_track(head.forward(states))
    assert track.values.shape == (9,)
    assert abs(float(track.values.sum()) - 1.0) < 1e-12
    assert track.method == METHOD_POOLING
    assert not track.signed
    assert track.forward_passes == 0


def test_attention_is_the_same_for_every_class(states: np.ndarray) -> None:
    base = _tensors()
    other = {k: v.copy() for k, v in base.items()}
    other["fc2.weight"][0] *= -4.0
    other["fc2.bias"][0] += 3.0
    a = AttentionHeadNet(base, EPS).forward(states)
    b = AttentionHeadNet(other, EPS).forward(states)
    assert np.array_equal(a.attention, b.attention)
    assert not np.array_equal(a.scores, b.scores)


def test_occlusion_equals_hand_computed_recompute(
    head: AttentionHeadNet, states: np.ndarray
) -> None:
    t = _tensors()
    x = states.astype(np.float64)
    cls, window = 1, 2
    track = occlusion_track(head, states, cls, OcclusionSettings(window, 1, 100))
    full = _reference(t, x)[cls]
    total = np.zeros(9)
    count = np.zeros(9)
    for start in range(0, 9 - window + 1):
        kept = np.delete(x, slice(start, start + window), axis=0)
        drop = full - _reference(t, kept)[cls]
        total[start : start + window] += drop
        count[start : start + window] += 1
    assert np.abs(track.values - total / count).max() < 1e-5
    assert track.method == METHOD_OCCLUSION
    assert track.signed
    assert (track.window, track.stride) == (window, 1)
    assert track.forward_passes == (9 - window + 1) + 1


def test_removing_nothing_changes_nothing(head: AttentionHeadNet, states: np.ndarray) -> None:
    full = float(head.forward(states).scores[2])
    assert score_without(head, states, 2, np.ones(9, dtype=bool)) == full


def test_keep_must_select_a_residue(head: AttentionHeadNet, states: np.ndarray) -> None:
    with pytest.raises(ModelError):
        score_without(head, states, 0, np.zeros(9, dtype=bool))
    with pytest.raises(ModelError):
        score_without(head, states, 0, np.ones(8, dtype=bool))


def test_occlusion_is_deterministic(head: AttentionHeadNet, states: np.ndarray) -> None:
    a = occlusion_track(head, states, 0, OcclusionSettings(3, 2, 50)).values
    b = occlusion_track(head, states, 0, OcclusionSettings(3, 2, 50)).values
    assert np.array_equal(a, b)


def test_single_residue_sequence_has_a_zero_track(head: AttentionHeadNet) -> None:
    one = np.ones((1, DIM), dtype=np.float32)
    track = occlusion_track(head, one, 0, OcclusionSettings(5, 1, 10))
    assert track.values.tolist() == [0.0]
    assert track.forward_passes == 1


@pytest.mark.parametrize("length", [2, 3, 7, 50, 300, 1000])
@pytest.mark.parametrize(
    ("window", "stride", "cap"), [(5, 1, 64), (5, 1, 2), (1, 1, 10), (9, 4, 3)]
)
def test_plan_covers_every_residue_within_the_cap(
    length: int, window: int, stride: int, cap: int
) -> None:
    width, step, starts = plan_windows(length, window, stride, cap)
    assert len(starts) <= cap
    assert 1 <= width <= length - 1
    covered = np.zeros(length, dtype=bool)
    for s in starts:
        assert s >= 0
        assert s + width <= length
        covered[s : s + width] = True
    assert covered.all()
    assert step >= stride


def test_plan_raises_stride_and_widens_window_when_capped() -> None:
    width, step, starts = plan_windows(300, 5, 1, 32)
    assert len(starts) <= 32
    assert step > 1
    assert width >= step


def test_plan_rejects_bad_settings() -> None:
    with pytest.raises(ModelError):
        plan_windows(10, 0, 1, 10)
    with pytest.raises(ModelError):
        plan_windows(10, 3, 1, 1)


def test_crf_track_pads_beyond_the_cap() -> None:
    out = TaggerOutput(0.9, 3, 3, np.array([0.9, 0.8, 0.1, 0.0]), np.zeros((4, 2)))
    track = crf_track(out, 7)
    assert track.method == METHOD_CRF
    assert not track.signed
    assert track.values.tolist() == [0.9, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0]
    assert track.scored_length == 4


def test_rounded_is_finite_floats() -> None:
    assert rounded(np.array([0.123456, float("nan"), 1.0]), 3) == [0.123, 0.0, 1.0]


def test_module_imports_nothing_from_training_or_evidence() -> None:
    path = Path(__file__).resolve().parents[2] / "protloc" / "inference" / "attribution.py"
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    internal = {n for n in names if n.startswith("protloc")}
    assert internal == {"protloc.models.attention_head", "protloc.models.crf_tagger"}
