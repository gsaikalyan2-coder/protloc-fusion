"""The numpy serving forward passes equal the torch training forward passes (Phase 14).

Serving never imports ``training``; this test is the one place both sides meet, so a change to
either network that breaks the equality fails here. It runs on the saved final weights and the
per-residue states of the 500-sequence subset, and is skipped when torch, numpy or the data are
not present (CI holds neither). Tolerances are stated, not implied:

* localization / GO sigmoids: absolute 1e-5 (float32 arithmetic in a different order);
* CRF detection: absolute 5e-4 (measured over all 500 subset sequences: 1.2e-4). The torch
  reference computes ``1 - P(all outside)`` in float32 as ``-expm1(path_score - log Z)``, a
  difference of two sums near 10, so near 0 it is off by up to 1e-4; the numpy side runs the CRF
  in float64 and is the more accurate of the two. Cleavage site: exact. Signal posterior:
  absolute 1e-4.
"""

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from protloc.data.residue_cache import ResidueStore  # noqa: E402
from protloc.models.attention_head import AttentionHeadNet  # noqa: E402
from protloc.models.crf_tagger import CrfTaggerNet, TagSpec  # noqa: E402
from protloc.training import crf as crf_config  # noqa: E402
from protloc.training import crf_net, head_net  # noqa: E402
from protloc.training.head import Architecture  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "embeddings" / "0e528d4a46831b05_subset500_residues"
HEAD_TOL = 1e-5
CRF_TOL = 5e-4
POSTERIOR_TOL = 1e-4
EPS = 1e-5
SAMPLE = 40

pytestmark = pytest.mark.skipif(
    not (CACHE / "index.json").is_file(), reason="the residue cache is not on this machine"
)


@pytest.fixture(scope="module")
def store() -> ResidueStore:
    return ResidueStore(CACHE)


@pytest.fixture(scope="module")
def shas(store: ResidueStore) -> list[str]:
    order = sorted(store.lengths, key=lambda s: (store.lengths[s], s))
    step = max(1, len(order) // SAMPLE)
    return order[::step][:SAMPLE]


def _tensors(task_dir: str) -> dict[str, object]:
    with np.load(ROOT / "data" / task_dir / "final_weights.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize(("task_dir", "classes"), [("head", 10), ("go", 12)])
def test_attention_head_matches_torch(
    task_dir: str, classes: int, store: Any, shas: list[str]
) -> None:
    weights = _tensors(task_dir)
    hidden = int(weights["fc1.weight"].shape[0])  # type: ignore[attr-defined]
    arch = Architecture(heads=1, attention_hidden=64, embedding_dim=640)
    net = head_net.Net(architecture=arch, hidden=hidden, dropout=0.0, classes=classes)
    reference = head_net.build_model(net, 0, weights)
    mine = AttentionHeadNet(weights, EPS)
    scores, attention = head_net.predict(reference, store, shas, 8, attention=True)
    worst = 0.0
    for i, sha in enumerate(shas):
        out = mine.forward(store.states(sha))
        worst = max(worst, float(np.abs(out.scores - scores[i]).max()))
        assert np.abs(out.attention - attention[i]).max() < HEAD_TOL
        assert abs(float(out.attention.sum(axis=1)[0]) - 1.0) < 1e-5
    assert worst < HEAD_TOL, worst


def test_crf_tagger_matches_torch(store: Any, shas: list[str]) -> None:
    weights = _tensors("crf")
    tags = crf_config.Tags(("O", "S"), "O", "S", (("O", "S"),), -10000.0, (), 20)
    arch = crf_config.Architecture(640, 100, 64, 5, 2, 64, 1)
    reference = crf_net.build_model(crf_net.Net(arch, tags, "cnn", 0.0), 0, weights)
    prediction = crf_net.predict(reference, store, shas, 8, keep_marginals=True)
    mine = CrfTaggerNet(weights, TagSpec(("O", "S"), "O", "S", (("O", "S"),), -10000.0), 100, EPS)
    for i, sha in enumerate(shas):
        out = mine.forward(store.states(sha))
        assert abs(out.detection - float(prediction.score[i])) < CRF_TOL
        assert out.cleavage == int(prediction.cleavage[i])
        assert out.path_cleavage == int(prediction.path_cleavage[i])
        posterior = prediction.marginals[i]
        assert np.abs(out.signal_posterior - posterior).max() < POSTERIOR_TOL
