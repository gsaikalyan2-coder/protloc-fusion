"""Serving forward pass of the signal-peptide CNN + linear-chain CRF (Phase 11, ADR-0013).

Numpy only; reproduces ``training/crf_net`` in evaluation mode for ONE sequence cut to its first
``length_cap`` residues. The network is ``LayerNorm -> 2 x (Conv1d k=5 same, GELU) -> Linear`` to
per-residue tag scores; the CRF is a few functions over (length, tags) arrays: the log-partition by
the forward algorithm, Viterbi, forward-backward marginals and the score of a tag path. The CRF
algebra runs in float64 on float32 emissions. A forbidden transition is replaced by the large
negative score from ``configs/serving/engine.json``, so ``exp`` of it is exactly 0.

Detection is ``P(any signal tag) = 1 - P(every residue outside)``; the cleavage site is the last
signal residue of the best path given that residue 1 is a signal residue.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from protloc.models.attention_head import ModelError, gelu, layer_norm

_RANK = 2  # states are (residues, dimension)

CRF_TENSORS = ("crf.transitions", "crf.start", "crf.end")
CNN_TENSORS = (
    "norm.weight",
    "norm.bias",
    "conv0.weight",
    "conv0.bias",
    "conv1.weight",
    "conv1.bias",
    "emit.weight",
    "emit.bias",
)


@dataclass(frozen=True, slots=True)
class TagSpec:
    """Tag names, the outside and signal tag, and the transitions that are forbidden."""

    names: tuple[str, ...]
    outside: str
    signal: str
    forbidden: tuple[tuple[str, str], ...]
    forbidden_score: float


@dataclass(frozen=True, slots=True)
class TaggerOutput:
    """What one sequence gets: detection, cleavage and the per-residue signal posterior."""

    detection: float
    cleavage: int
    path_cleavage: int
    signal_posterior: Any
    emissions: Any


def _logsumexp(a: Any, axis: int) -> Any:
    peak = a.max(axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return (peak + np.log(np.exp(a - peak).sum(axis=axis, keepdims=True))).squeeze(axis)


def conv1d_same(x: Any, weight: Any, bias: Any) -> Any:
    """Cross-correlation of (length, in) with a torch-layout (out, in, k) kernel, zero padded."""
    out_channels, in_channels, kernel = weight.shape
    pad = kernel // 2
    padded = np.pad(x, ((pad, pad), (0, 0)))
    windows = np.stack([padded[i : i + x.shape[0]] for i in range(kernel)], axis=1)
    flat_weight = weight.transpose(0, 2, 1).reshape(out_channels, kernel * in_channels)
    return windows.reshape(x.shape[0], kernel * in_channels) @ flat_weight.T + bias


class CrfTaggerNet:
    """The tagger, loaded from numpy tensors; immutable after construction."""

    def __init__(
        self, tensors: Mapping[str, Any], spec: TagSpec, length_cap: int, layer_norm_eps: float
    ) -> None:
        """Validate shapes and build the constrained CRF scores."""
        missing = [n for n in (*CNN_TENSORS, *CRF_TENSORS) if n not in tensors]
        if missing:
            raise ModelError(f"weights lack {missing}")
        self._t = {n: np.ascontiguousarray(tensors[n], dtype=np.float32) for n in CNN_TENSORS}
        size = len(spec.names)
        if self._t["emit.weight"].shape[0] != size or tensors["crf.transitions"].shape != (
            size,
            size,
        ):
            raise ModelError("the tag set does not match the weights")
        self.spec = spec
        self.length_cap = int(length_cap)
        self.dim = int(self._t["norm.weight"].shape[0])
        self._eps = float(layer_norm_eps)
        self._outside = spec.names.index(spec.outside)
        self._signal = spec.names.index(spec.signal)
        transitions = np.asarray(tensors["crf.transitions"], dtype=np.float32).astype(np.float64)
        for source, target in spec.forbidden:
            transitions[spec.names.index(source), spec.names.index(target)] = spec.forbidden_score
        self._transitions = transitions
        self._start = np.asarray(tensors["crf.start"], dtype=np.float32).astype(np.float64)
        self._end = np.asarray(tensors["crf.end"], dtype=np.float32).astype(np.float64)

    def emissions(self, states: Any) -> Any:
        """Per-residue tag scores (length, tags) of the first ``length_cap`` residues, float32."""
        x = np.asarray(states, dtype=np.float32)[: self.length_cap]
        if x.ndim != _RANK or x.shape[1] != self.dim or x.shape[0] < 1:
            raise ModelError(f"states must be (length >= 1, {self.dim}), got {x.shape}")
        t = self._t
        hidden = layer_norm(x, t["norm.weight"], t["norm.bias"], self._eps)
        for index in (0, 1):
            hidden = gelu(conv1d_same(hidden, t[f"conv{index}.weight"], t[f"conv{index}.bias"]))
        return (hidden @ t["emit.weight"].T + t["emit.bias"]).astype(np.float32)

    def _log_partition(self, em: Any, start: Any) -> float:
        alpha = start + em[0]
        for step in range(1, em.shape[0]):
            alpha = _logsumexp(alpha[:, None] + self._transitions, axis=0) + em[step]
        return float(_logsumexp(alpha + self._end, axis=0))

    def _path_score(self, em: Any, path: Sequence[int]) -> float:
        total = self._start[path[0]] + em[0, path[0]]
        for step in range(1, len(path)):
            total += self._transitions[path[step - 1], path[step]] + em[step, path[step]]
        return float(total + self._end[path[-1]])

    def _viterbi(self, em: Any, start: Any) -> list[int]:
        delta = start + em[0]
        pointers = []
        for step in range(1, em.shape[0]):
            scores = delta[:, None] + self._transitions
            pointers.append(scores.argmax(axis=0))
            delta = scores.max(axis=0) + em[step]
        tag = int((delta + self._end).argmax())
        path = [tag]
        for pointer in reversed(pointers):
            tag = int(pointer[tag])
            path.append(tag)
        return path[::-1]

    def _marginals(self, em: Any) -> Any:
        length, size = em.shape
        alpha = np.empty((length, size))
        alpha[0] = self._start + em[0]
        for step in range(1, length):
            alpha[step] = (
                _logsumexp(alpha[step - 1][:, None] + self._transitions, axis=0) + em[step]
            )
        log_z = float(_logsumexp(alpha[-1] + self._end, axis=0))
        beta = np.empty((length, size))
        beta[-1] = self._end
        for step in range(length - 2, -1, -1):
            beta[step] = _logsumexp(
                self._transitions + (em[step + 1] + beta[step + 1])[None, :], axis=1
            )
        return np.clip(np.exp(alpha + beta - log_z), 0.0, 1.0)

    def forward(self, states: Any) -> TaggerOutput:
        """Detection, cleavage and the signal posterior for one sequence."""
        em32 = self.emissions(states)
        em = em32.astype(np.float64)
        log_z = self._log_partition(em, self._start)
        outside = self._path_score(em, [self._outside] * em.shape[0]) - log_z
        detection = float(-np.expm1(outside))
        forced = self._start.copy()
        forced[self._outside] += self.spec.forbidden_score
        given = self._viterbi(em, forced)
        free = self._viterbi(em, self._start)
        return TaggerOutput(
            detection=detection,
            cleavage=_last_tagged(given, self._signal),
            path_cleavage=_last_tagged(free, self._signal),
            signal_posterior=self._marginals(em)[:, self._signal],
            emissions=em32,
        )


def _last_tagged(path: Sequence[int], tag: int) -> int:
    """The 1-based last residue carrying ``tag`` (0 if none)."""
    return max((i + 1 for i, t in enumerate(path) if t == tag), default=0)
