"""Serving forward pass of the attention-pooling head (Phase 9 localization, Phase 12 GO).

Numpy only, no torch and nothing from ``training``: the weights are the ``final_weights.npz``
the registry bundle hashes, and the arithmetic reproduces
``training/head_net.AttentionHead.forward``
in evaluation mode (dropout off) for ONE sequence at a time::

    e = att_out(tanh(att_in(x)))                  # (L, heads)
    a = softmax over residues of e                # (L, heads); each head sums to 1
    pooled = mean over heads of  sum_l a[l] * x[l]
    logits = fc2(gelu(fc1(LayerNorm(pooled))))
    scores = sigmoid(logits)

``tests/unit/test_models_parity.py`` shows the output equals the torch forward on saved rows
within a stated tolerance; that test is the only place torch is imported.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

_RANK = 2  # states are (residues, dimension)

REQUIRED_TENSORS = (
    "att_in.weight",
    "att_in.bias",
    "att_out.weight",
    "att_out.bias",
    "norm.weight",
    "norm.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
)
_SQRT2 = math.sqrt(2.0)
_erf = np.frompyfunc(math.erf, 1, 1)


class ModelError(ValueError):
    """The weights do not describe the network this module computes."""


def gelu(x: Any) -> Any:
    """Exact (erf) GELU in the dtype of ``x``, as ``torch.nn.functional.gelu`` computes it."""
    return (0.5 * x * (1.0 + _erf(x / _SQRT2).astype(x.dtype))).astype(x.dtype)


def layer_norm(x: Any, weight: Any, bias: Any, eps: float) -> Any:
    """LayerNorm over the last axis with the biased variance, as ``torch.nn.LayerNorm``."""
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = (centred * centred).mean(axis=-1, keepdims=True)
    return (centred / np.sqrt(variance + np.float32(eps)) * weight + bias).astype(x.dtype)


def sigmoid(z: Any) -> Any:
    """Logistic function that cannot overflow."""
    out = np.empty_like(z, dtype=np.float64)
    z64 = z.astype(np.float64)
    positive = z64 >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z64[positive]))
    expz = np.exp(z64[~positive])
    out[~positive] = expz / (1.0 + expz)
    return out


@dataclass(frozen=True, slots=True)
class HeadOutput:
    """Raw sigmoid scores (classes,) and the attention weights (heads, length)."""

    scores: Any
    attention: Any


class AttentionHeadNet:
    """The head, loaded from numpy tensors; immutable after construction."""

    def __init__(self, tensors: Mapping[str, Any], layer_norm_eps: float) -> None:
        """Validate shapes; refuse tensors that are not an attention head."""
        missing = [name for name in REQUIRED_TENSORS if name not in tensors]
        if missing:
            raise ModelError(f"weights lack {missing}")
        t = {
            name: np.ascontiguousarray(tensors[name], dtype=np.float32) for name in REQUIRED_TENSORS
        }
        dim = t["norm.weight"].shape[0]
        hidden = t["att_in.weight"].shape[0]
        heads = t["att_out.weight"].shape[0]
        problems = []
        if t["att_in.weight"].shape != (hidden, dim):
            problems.append("att_in.weight")
        if t["att_out.weight"].shape != (heads, hidden):
            problems.append("att_out.weight")
        if t["fc1.weight"].shape[1] != dim:
            problems.append("fc1.weight")
        if t["fc2.weight"].shape[1] != t["fc1.weight"].shape[0]:
            problems.append("fc2.weight")
        if problems:
            raise ModelError(f"inconsistent tensor shapes: {problems}")
        self._t = t
        self.dim = int(dim)
        self.heads = int(heads)
        self.classes = int(t["fc2.weight"].shape[0])
        self._eps = float(layer_norm_eps)

    def forward(self, states: Any) -> HeadOutput:
        """Scores of one sequence from its (length, dim) per-residue states."""
        x = np.asarray(states, dtype=np.float32)
        if x.ndim != _RANK or x.shape[1] != self.dim or x.shape[0] < 1:
            raise ModelError(f"states must be (length >= 1, {self.dim}), got {x.shape}")
        t = self._t
        energy = np.tanh(x @ t["att_in.weight"].T + t["att_in.bias"]) @ t["att_out.weight"].T
        energy = energy + t["att_out.bias"]
        energy = energy - energy.max(axis=0, keepdims=True)
        weights = np.exp(energy)
        weights = weights / weights.sum(axis=0, keepdims=True)
        pooled = (weights.T @ x).mean(axis=0)
        normed = layer_norm(pooled, t["norm.weight"], t["norm.bias"], self._eps)
        hidden = gelu(normed @ t["fc1.weight"].T + t["fc1.bias"])
        logits = hidden @ t["fc2.weight"].T + t["fc2.bias"]
        return HeadOutput(scores=sigmoid(logits), attention=weights.T.astype(np.float64))
