"""Per-residue attribution computed from the trained heads (Phase 15, ADR-0017).

Stdlib and numpy only; no ``training``, no ``evidence``. Three methods, each with its own claim
and its own method id, because they answer different questions:

``pooling_attention``
    Where the pooled vector looked. The attention weights are computed before the class layers,
    so they are the SAME for every class: this says what the head attended to, not why a given
    label won. Class-agnostic. Unsigned, sums to 1.
``occlusion``
    How much one label's raw score depended on a stretch of residues. A window of residues is
    removed from the **pooling input only** and the head is run again on the remainder (the
    attention softmax renormalises over what is left); the score is ``p_full - p_removed`` for the
    label, spread over the window's residues and averaged where windows overlap. Class-specific
    and signed: positive means removing those residues lowered the label's score. The frozen ESM-2
    encoder is NOT re-run, so context already encoded into neighbouring residue states stays: this
    explains the head's own input, not the protein.
``crf_marginal``
    ``P(residue is in the signal peptide)`` from the CRF forward-backward marginals, in 0..1,
    unsigned. Only the first ``length_cap`` residues are scored by the CRF; the rest are reported
    as 0.0 and ``scored_length`` says where the model stopped.

None of this feeds a score, a calibrator or a threshold. Everything is deterministic: no random
numbers, no clock.
"""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from protloc.models.attention_head import AttentionHeadNet, HeadOutput, ModelError
from protloc.models.crf_tagger import TaggerOutput

METHOD_POOLING = "pooling_attention"
METHOD_OCCLUSION = "occlusion"
METHOD_CRF = "crf_marginal"
METHOD_MOTIF = "motif"
METHODS = (METHOD_POOLING, METHOD_OCCLUSION, METHOD_CRF, METHOD_MOTIF)

_MIN_FORWARD_PASSES = 2


@dataclass(frozen=True, slots=True)
class Track:
    """One per-residue track and exactly how it was made."""

    values: Any  # float64 (length,)
    method: str
    signed: bool
    window: int | None
    stride: int | None
    forward_passes: int
    scored_length: int


def attention_track(output: HeadOutput) -> Track:
    """Mean over attention heads, renormalised to sum 1. Class-agnostic."""
    att = np.asarray(output.attention, dtype=np.float64)
    mean = att.mean(axis=0)
    total = float(mean.sum())
    values = mean / total if total > 0.0 else mean
    return Track(values, METHOD_POOLING, False, None, None, 0, int(values.shape[0]))


def crf_track(output: TaggerOutput, length: int) -> Track:
    """The signal posterior, padded with 0.0 beyond the residues the CRF scored."""
    posterior = np.asarray(output.signal_posterior, dtype=np.float64)
    scored = int(posterior.shape[0])
    values = np.zeros(length, dtype=np.float64)
    values[: min(scored, length)] = posterior[:length]
    return Track(values, METHOD_CRF, False, None, None, 0, min(scored, length))


def plan_windows(
    length: int, window: int, stride: int, max_forward_passes: int
) -> tuple[int, int, list[int]]:
    """``(window, stride, starts)`` that cover every residue within the forward-pass cap.

    A window never covers the whole sequence (the remainder must have at least one residue). If
    the requested stride would need more passes than the cap, the stride is raised until it fits
    and the window is widened to at least the stride so no residue is left uncovered. The values
    actually used are returned so they can be recorded.
    """
    if length < _MIN_FORWARD_PASSES:
        return 1, 1, []
    if window < 1 or stride < 1 or max_forward_passes < _MIN_FORWARD_PASSES:
        raise ModelError(
            f"window and stride must be at least 1 and max_forward_passes at least "
            f"{_MIN_FORWARD_PASSES}"
        )
    width = min(window, length - 1)
    step = max(1, stride)
    while True:
        last = length - width
        starts = list(range(0, last + 1, step))
        if starts[-1] != last:
            starts.append(last)
        if len(starts) <= max_forward_passes:
            return width, step, starts
        step += 1
        width = min(max(width, step), length - 1)


def score_without(head: AttentionHeadNet, states: Any, class_index: int, keep: Any) -> float:
    """The class's raw sigmoid when only the residues where ``keep`` is true are pooled.

    ``keep`` all true is the unmodified score; the remainder must not be empty.
    """
    mask = np.asarray(keep, dtype=bool)
    if mask.shape[0] != np.asarray(states).shape[0] or not mask.any():
        raise ModelError("keep must select at least one residue of the sequence")
    return float(head.forward(np.asarray(states)[mask]).scores[class_index])


@dataclass(frozen=True, slots=True)
class OcclusionSettings:
    """Requested window, stride and the cap on forward passes (``configs/serving/engine.json``)."""

    window: int
    stride: int
    max_forward_passes: int


def occlusion_track(
    head: AttentionHeadNet, states: Any, class_index: int, settings: OcclusionSettings
) -> Track:
    """Signed drop of the class's raw score when each window is removed from the pooled input."""
    x = np.asarray(states, dtype=np.float32)
    length = int(x.shape[0])
    full = float(head.forward(x).scores[class_index])
    width, step, starts = plan_windows(
        length, settings.window, settings.stride, settings.max_forward_passes
    )
    totals = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for start in starts:
        keep = np.ones(length, dtype=bool)
        keep[start : start + width] = False
        drop = full - score_without(head, x, class_index, keep)
        totals[start : start + width] += drop
        counts[start : start + width] += 1.0
    values = np.divide(totals, counts, out=np.zeros(length), where=counts > 0)
    return Track(values, METHOD_OCCLUSION, True, width, step, len(starts) + 1, length)


def rounded(values: Any, digits: int) -> list[float]:
    """Plain floats for the wire; non-finite numbers become 0.0."""
    return [round(float(v), digits) if math.isfinite(float(v)) else 0.0 for v in values]
