"""The opt-in live encoder (ADR-0019): frozen ESM-2 for sequences outside the residue cache.

Layer L1 (inference). With ``ENCODER_MODE=cache_only`` (the default) nothing here runs and
nothing heavy is imported: the served path stays numpy-only (ADR-0016). With
``ENCODER_MODE=live`` the engine asks this module for the per-residue states of a sequence the
cache does not hold. They are computed by the same code that built the cache
(:class:`protloc.data.esm_encoder.TorchEncoder.embed_residues`), with the same checkpoint,
revision, context and sliding-window policy, read from the cache's own ``index.json``, and
rounded to float16 exactly as the cache stores them. So a live state is what the cache would
have held, and the heads see the input they were trained on.

Three guards:

* **No network.** Weights load with ``local_files_only``; ``scripts/fetch_esm.py`` puts the
  pinned revision in the hub cache beforehand. A missing weight file stops the service.
* **A self-check at startup.** One cached sequence is encoded live and compared with its cached
  states; above :data:`SELF_CHECK_TOLERANCE` the service refuses to start, because the live
  states would not be the ones the heads were fitted on.
* **Labelled results.** Every live result says ``encoder: "live"`` and
  ``outside_evaluated_subset: true``: its probabilities come from calibrators fitted on the
  500-sequence subset, and no number was measured on sequences like it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import time
from typing import Any, Final, Protocol

#: Largest |live - cached| allowed for the self-check sequence (float16 states). The cache was
#: byte-identical across processes on the machine that built it (Phase 7); the margin covers a
#: different CPU or thread count, and is far below the scale of the states.
SELF_CHECK_TOLERANCE: Final[float] = 0.02


class LiveEncoderError(RuntimeError):
    """The live encoder cannot be loaded or failed its self-check; the message says why."""


class ResidueEncoder(Protocol):
    """What the live encoder needs from an ESM-2 implementation."""

    def embed_residues(self, sequences: list[str]) -> list[bytes]:
        """Little-endian float16 bytes of shape (length, dim), one item per sequence."""
        ...


def _np() -> Any:
    return importlib.import_module("numpy")


@dataclass(slots=True)
class LiveEncoder:
    """Per-residue states for any sequence, in the cache's exact numeric form."""

    inner: ResidueEncoder
    dim: int
    checkpoint: str
    revision: str
    self_check_max_abs_diff: float | None = None
    encoded: int = 0
    seconds: float = 0.0

    def states(self, sequence: str) -> Any:
        """The (length, dim) float16 states of ``sequence``."""
        started = time.perf_counter()
        raw = self.inner.embed_residues([sequence])[0]
        arr = _np().frombuffer(raw, dtype="<f2")
        if arr.size != len(sequence) * self.dim:
            raise LiveEncoderError(
                f"the encoder returned {arr.size} values for {len(sequence)} residues x {self.dim}"
            )
        self.encoded += 1
        self.seconds += time.perf_counter() - started
        return arr.reshape(len(sequence), self.dim)

    def describe(self) -> dict[str, Any]:
        """What provenance and ``/demo/metrics`` report about the encoder."""
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "self_check_max_abs_diff": self.self_check_max_abs_diff,
            "self_check_tolerance": SELF_CHECK_TOLERANCE,
            "sequences_encoded": self.encoded,
            "mean_seconds": round(self.seconds / self.encoded, 3) if self.encoded else None,
        }


def torch_factory(
    states_spec: Mapping[str, Any], cache_dir: Path | None, device: str
) -> ResidueEncoder:
    """The Phase 7 TorchEncoder for ``states_spec``, loaded from local files only."""
    if cache_dir is not None:
        os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        embeddings = importlib.import_module("protloc.data.embeddings")
        esm = importlib.import_module("protloc.data.esm_encoder")
        spec = embeddings.CacheSpec(**dict(states_spec))
        encoder: ResidueEncoder = esm.TorchEncoder(spec, device=device, local_files_only=True)
    except ModuleNotFoundError as error:
        raise LiveEncoderError(
            f"ENCODER_MODE=live needs the `live` extra (torch, transformers): {error}. "
            'Install it with: pip install -e "backend[serve,live]"'
        ) from error
    except OSError as error:
        raise LiveEncoderError(
            f"the pinned ESM-2 weights ({states_spec.get('checkpoint')} @ "
            f"{str(states_spec.get('revision'))[:8]}) are not in the local hub cache: {error}. "
            "Fetch them once with: python scripts/fetch_esm.py"
        ) from error
    return encoder


Factory = Callable[[Mapping[str, Any], Path | None, str], ResidueEncoder]


def load_live_encoder(
    states_spec: Mapping[str, Any],
    cache_dir: Path | None,
    device: str,
    check: tuple[str, Any],
    *,
    factory: Factory = torch_factory,
) -> LiveEncoder:
    """Load the encoder and self-check it on ``check``: (a cached sequence, its cached states)."""
    check_sequence, cached_states = check
    try:
        inner = factory(states_spec, cache_dir, device)
    except LiveEncoderError:
        raise
    except Exception as error:  # a broken install must stop startup with its reason
        raise LiveEncoderError(f"the live encoder could not be loaded: {error}") from error
    live = LiveEncoder(
        inner=inner,
        dim=int(states_spec["embedding_dim"]),
        checkpoint=str(states_spec["checkpoint"]),
        revision=str(states_spec["revision"]),
    )
    np = _np()
    fresh = live.states(check_sequence).astype(np.float32)
    cached = np.asarray(cached_states, dtype=np.float32)
    if fresh.shape != cached.shape:
        raise LiveEncoderError(
            f"self-check: live states have shape {fresh.shape}, cached {cached.shape}"
        )
    diff = float(np.max(np.abs(fresh - cached))) if fresh.size else 0.0
    if not diff <= SELF_CHECK_TOLERANCE:
        raise LiveEncoderError(
            f"self-check: live states differ from the cached ones by up to {diff:.4g} "
            f"(tolerance {SELF_CHECK_TOLERANCE}); the heads would see different inputs"
        )
    live.self_check_max_abs_diff = round(diff, 6)
    live.encoded = 0
    live.seconds = 0.0
    return live


__all__ = [
    "SELF_CHECK_TOLERANCE",
    "Factory",
    "LiveEncoder",
    "LiveEncoderError",
    "ResidueEncoder",
    "load_live_encoder",
    "torch_factory",
]
