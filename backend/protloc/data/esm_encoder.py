"""The ESM-2 encoder behind :class:`protloc.data.embeddings.Encoder`.

torch and transformers are imported inside :class:`TorchEncoder`, never at module
level: they are not serving or CI dependencies of this package until Phase 14, and
importing this module must work without them.

One sequence becomes one vector as follows. The sequence is cut into windows by
:func:`protloc.data.embeddings.windows` (one window if it fits in the context). Each
window is tokenised with ``<cls>`` and ``<eos>``; the last hidden state of the frozen
encoder gives one state per residue. A residue seen by several windows contributes
the average of its states, and the vector is the mean over residues. Special tokens
and padding never enter the mean.
"""

from collections.abc import Sequence
import importlib
import os
from pathlib import Path
import platform
import random
from typing import Any

from protloc.data.embeddings import (
    CacheSpec,
    EmbeddingError,
    ExtractOptions,
    UniqueSequence,
    extract,
    pack_vector,
    residue_weights,
    windows,
)

SEED = 20260909
#: ESM-2 position embeddings carry a fixed offset of two positions.
_POSITION_OFFSET = 2
#: Tokens that surround every window: ``<cls>`` and ``<eos>``.
_SPECIAL_TOKENS = 2


class TorchEncoder:
    """Frozen ESM-2 at a pinned revision, run in eval mode without gradients."""

    def __init__(
        self,
        spec: CacheSpec,
        device: str = "cpu",
        batch_size: int = 8,
        local_files_only: bool = False,
    ) -> None:
        """Load the tokenizer and weights of ``spec`` and check them against it."""
        spec.validate()
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")
        if device.startswith("cuda"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self._torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        random.seed(SEED)
        self._torch.manual_seed(SEED)
        self._torch.use_deterministic_algorithms(True, warn_only=True)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec.checkpoint, revision=spec.revision, local_files_only=local_files_only
        )
        self._model = transformers.AutoModel.from_pretrained(
            spec.checkpoint,
            revision=spec.revision,
            add_pooling_layer=False,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._model.to(device)
        self._check_model()

    def _check_model(self) -> None:
        config = self._model.config
        if int(config.hidden_size) != self.spec.embedding_dim:
            raise EmbeddingError(
                f"model hidden_size {config.hidden_size} != spec embedding_dim "
                f"{self.spec.embedding_dim}"
            )
        residues = int(config.max_position_embeddings) - _POSITION_OFFSET - _SPECIAL_TOKENS
        if residues < self.spec.max_residues:
            raise EmbeddingError(
                f"model context holds {residues} residues, spec asks for {self.spec.max_residues}"
            )

    def check_alphabet(self, alphabet: str) -> dict[str, int]:
        """Token id of every letter of ``alphabet``; raises if one maps to ``<unk>``."""
        unknown = int(self._tokenizer.unk_token_id)
        ids = {letter: int(self._tokenizer.convert_tokens_to_ids(letter)) for letter in alphabet}
        bad = sorted(letter for letter, token in ids.items() if token == unknown)
        if bad:
            raise EmbeddingError(f"tokeniser maps {bad} to <unk>")
        return ids

    def describe(self) -> dict[str, object]:
        """Software and hardware facts for the manifest."""
        torch = self._torch
        transformers: Any = importlib.import_module("transformers")
        return {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "device": self.device,
            "device_name": (
                str(torch.cuda.get_device_name(self.device))
                if self.device.startswith("cuda")
                else platform.processor() or platform.machine()
            ),
            "torch_threads": int(torch.get_num_threads()),
            "cpu_count": os.cpu_count(),
            "batch_windows": self.batch_size,
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    def embed_batch(self, sequences: Sequence[str]) -> list[list[float]]:
        """One pooled vector per sequence, in order.

        Windows from all sequences are sorted by length and run in fixed-size
        batches, so the batches depend only on the input list.
        """
        torch = self._torch
        spec = self.spec
        plans = [windows(len(s), spec.max_residues, spec.window_stride) for s in sequences]
        counts = [residue_weights(len(s), plan) for s, plan in zip(sequences, plans, strict=True)]
        jobs = sorted(
            (end - start, index, start, end)
            for index, plan in enumerate(plans)
            for start, end in plan
        )
        totals = [torch.zeros(spec.embedding_dim, dtype=torch.float32) for _ in sequences]
        with torch.no_grad():
            for first in range(0, len(jobs), self.batch_size):
                chunk = jobs[first : first + self.batch_size]
                texts = [sequences[index][start:end] for _, index, start, end in chunk]
                batch = self._tokenizer(texts, return_tensors="pt", padding=True)
                batch = {name: value.to(self.device) for name, value in batch.items()}
                hidden = self._model(**batch).last_hidden_state
                for row, (size, index, start, end) in enumerate(chunk):
                    states = hidden[row, 1 : 1 + size, :].float()
                    divisor = torch.tensor(
                        counts[index][start:end], dtype=torch.float32, device=self.device
                    )
                    totals[index] += (states / divisor[:, None]).sum(dim=0).cpu()
        return [
            (total / len(sequence)).tolist()
            for total, sequence in zip(totals, sequences, strict=True)
        ]


def probe_worker(spec_fields: dict[str, Any], sequence: str, out: str, device: str) -> None:
    """Embed one sequence in a fresh process and write its raw float32 bytes to ``out``."""
    encoder = TorchEncoder(CacheSpec(**spec_fields), device=device)
    Path(out).write_bytes(pack_vector(encoder.embed_batch([sequence])[0]))


def extract_worker(
    spec_fields: dict[str, Any],
    items: list[tuple[str, str]],
    directory: str,
    shard_size: int,
    device: str,
) -> None:
    """Run :func:`protloc.data.embeddings.extract` over ``(sha256, sequence)`` pairs."""
    spec = CacheSpec(**spec_fields)
    encoder = TorchEncoder(spec, device=device)
    unique = [UniqueSequence(sha, sequence, 1) for sha, sequence in items]
    extract(unique, Path(directory), spec, encoder, ExtractOptions(shard_size=shard_size))
