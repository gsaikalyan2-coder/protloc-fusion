"""Phase 9 localization head: the network, the loss and the training loop (needs torch).

torch and numpy are imported lazily, inside the functions that need them, and typed ``Any``:
the packages are the optional ``head`` extra, not dependencies of the served image, and CI
type-checks this module without them installed.

The network (ADR-0011 section 2.2), for a batch of sequences ``x`` of per-residue ESM-2
states (B, L, D) and a mask of real residues (B, L)::

    e = Linear(attention_hidden -> heads)(tanh(Linear(D -> attention_hidden)(x)))
    a = softmax over the real residues of e          # (B, L, heads); padding is exactly 0
    pooled = mean over heads of  sum_l a[l] * x[l]   # (B, D)
    logits = Linear(hidden -> classes)(dropout(gelu(Linear(D -> hidden)(LayerNorm(pooled)))))

``forward`` returns the logits AND the attention weights (B, heads, L): Phase 10 and
Phase 15 explain with them. The output is 10 independent sigmoids, one per class (multi-label).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import time
from typing import Any, Protocol

from protloc.training.baselines import BaselineError, Fold
from protloc.training.head import (
    Architecture,
    EarlyStopper,
    GridPoint,
    HeadConfig,
    Rows,
    Schedule,
    Shuffler,
    assert_no_overlap,
    clusters_of,
    epoch_batches,
    fold_rows,
    held_macro_f1,
    inner_split,
    pos_weights,
)

Array = Any


class StateReader(Protocol):
    """Anything that returns the (length, dim) per-residue states of a sequence by hash."""

    def states(self, sha256: str) -> Any:
        """The states of one sequence."""
        ...


@dataclass(frozen=True, slots=True)
class Source:
    """The labelled rows and the reader that returns their per-residue states."""

    reader: StateReader
    rows: Rows


def _torch() -> Any:
    return importlib.import_module("torch")


def _np() -> Any:
    return importlib.import_module("numpy")


@dataclass(frozen=True, slots=True)
class Net:
    """The settings of one network."""

    architecture: Architecture
    hidden: int
    dropout: float
    classes: int


@dataclass(frozen=True, slots=True)
class FitSettings:
    """Everything a fit depends on besides the data."""

    net: Net
    gamma: float
    learning_rate: float
    weight_decay: float
    schedule: Schedule
    seed: int


def settings_for(point: GridPoint, config: HeadConfig, classes: int) -> FitSettings:
    """The fit settings of one grid point."""
    return FitSettings(
        net=Net(config.architecture, point.hidden, point.dropout, classes),
        gamma=point.gamma,
        learning_rate=point.learning_rate,
        weight_decay=config.weight_decay,
        schedule=config.schedule,
        seed=config.seed,
    )


def set_deterministic(seed: int) -> None:
    """Seed torch and demand deterministic algorithms (an op without one raises)."""
    torch = _torch()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttentionHead:
    """Attention pooling over residues followed by an MLP; a plain class around ``ModuleDict``."""

    def __init__(self, net: Net, seed: int) -> None:
        """Build the layers; initial weights depend only on ``seed`` and the shapes."""
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(seed)
        arch = net.architecture
        self.net = net
        self.layers = nn.ModuleDict(
            {
                "att_in": nn.Linear(arch.embedding_dim, arch.attention_hidden),
                "att_out": nn.Linear(arch.attention_hidden, arch.heads),
                "norm": nn.LayerNorm(arch.embedding_dim),
                "fc1": nn.Linear(arch.embedding_dim, net.hidden),
                "dropout": nn.Dropout(net.dropout),
                "fc2": nn.Linear(net.hidden, net.classes),
            }
        )

    def parameters(self) -> list[Any]:
        """The trainable tensors."""
        return list(self.layers.parameters())

    def parameter_count(self) -> int:
        """Number of trainable scalars."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True) -> None:
        """Switch dropout on (training) or off (evaluation)."""
        self.layers.train(mode)

    def forward(self, x: Any, mask: Any) -> tuple[Any, Any]:
        """Logits (B, classes) and attention weights (B, heads, L); padded weights are 0."""
        torch = _torch()
        layers = self.layers
        energy = layers["att_out"](torch.tanh(layers["att_in"](x)))
        energy = energy.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(energy, dim=1)
        pooled = torch.einsum("blh,bld->bhd", weights, x).mean(dim=1)
        hidden = layers["dropout"](torch.nn.functional.gelu(layers["fc1"](layers["norm"](pooled))))
        return layers["fc2"](hidden), weights.permute(0, 2, 1)

    def state(self) -> dict[str, Array]:
        """A copy of every tensor as a numpy array (saved without pickle)."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.layers.state_dict().items()}

    def load_state(self, state: Mapping[str, Array]) -> None:
        """Load tensors produced by :meth:`state`."""
        torch = _torch()
        self.layers.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()})


def build_model(net: Net, seed: int, state: Mapping[str, Array] | None = None) -> AttentionHead:
    """A model in evaluation mode, with ``state`` loaded when given."""
    model = AttentionHead(net, seed)
    if state is not None:
        model.load_state(state)
    model.train(False)
    return model


def weighted_focal_loss(logits: Any, target: Any, pos_weight: Any, gamma: float) -> Any:
    """Mean of ``w_c * (1 - p_t) ** gamma * BCE`` over rows and classes.

    ``w_c`` is ``pos_weight`` for a positive label and 1 for a negative one. ``gamma`` 0 is
    exactly binary cross-entropy with ``pos_weight``.
    """
    torch = _torch()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    weight = target * pos_weight + (1.0 - target)
    return (weight * (1.0 - p_t) ** gamma * bce).mean()


def gather(reader: StateReader, shas: Sequence[str], dim: int) -> tuple[Any, Any]:
    """Pad the states of ``shas`` into a float32 tensor (B, Lmax, dim) and a bool mask (B, Lmax)."""
    np, torch = _np(), _torch()
    arrays = [reader.states(sha) for sha in shas]
    widest = max(int(a.shape[0]) for a in arrays)
    x = np.zeros((len(arrays), widest, dim), dtype=np.float32)
    mask = np.zeros((len(arrays), widest), dtype=bool)
    for i, array in enumerate(arrays):
        x[i, : array.shape[0]] = array
        mask[i, : array.shape[0]] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


def predict(
    model: AttentionHead,
    reader: StateReader,
    shas: Sequence[str],
    batch_size: int,
    attention: bool = False,
) -> tuple[Array, list[Array] | None]:
    """Sigmoid scores (n, classes) in the order of ``shas``; optionally per-sequence attention.

    Attention comes back as one (heads, length) array per sequence, summing to 1 per head.
    """
    np, torch = _np(), _torch()
    dim = model.net.architecture.embedding_dim
    lengths = [int(reader.states(sha).shape[0]) for sha in shas]
    order = sorted(range(len(shas)), key=lambda i: lengths[i])
    scores = np.zeros((len(shas), model.net.classes), dtype=np.float64)
    weights: list[Array | None] = [None] * len(shas)
    model.train(False)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            x, mask = gather(reader, [shas[i] for i in chunk], dim)
            logits, att = model.forward(x, mask)
            probs = torch.sigmoid(logits).numpy()
            for k, i in enumerate(chunk):
                scores[i] = probs[k]
                if attention:
                    weights[i] = att[k, :, : lengths[i]].numpy().copy()
    if not attention:
        return scores, None
    return scores, [w for w in weights if w is not None]


@dataclass(frozen=True, slots=True)
class FitResult:
    """One fit: the weights to keep and how it went."""

    state: Mapping[str, Array]
    best_epoch: int
    epochs_run: int
    hold_macro_f1: float | None
    seconds: float


def fit_head(
    source: Source,
    fit_idx: Sequence[int],
    hold_idx: Sequence[int] | None,
    settings: FitSettings,
    epochs: int | None = None,
) -> FitResult:
    """Train one head on ``fit_idx``.

    With ``hold_idx`` the holdout macro F1 (thresholds tuned on the holdout) is computed after
    every epoch, training stops after ``patience`` epochs without improvement and the best
    epoch's weights are kept. Without it, exactly ``epochs`` epochs run and the final weights
    are kept. The loss weights come from the ``fit_idx`` rows only.
    """
    if (hold_idx is None) == (epochs is None):
        raise BaselineError("give either a holdout (early stopping) or a fixed epoch count")
    np, torch = _np(), _torch()
    reader, rows = source.reader, source.rows
    began = time.perf_counter()
    set_deterministic(settings.seed)
    net, schedule = settings.net, settings.schedule
    dim = net.architecture.embedding_dim
    model = AttentionHead(net, settings.seed)
    weights = torch.tensor(pos_weights(rows.y[list(fit_idx)].tolist()), dtype=torch.float32)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    lengths = [int(reader.states(sha).shape[0]) for sha in rows.shas]
    target_all = torch.from_numpy(rows.y.astype(np.float32))
    shuffler = Shuffler(settings.seed)
    stopper = EarlyStopper(schedule.patience)
    best: Mapping[str, Array] = model.state()
    limit = schedule.max_epochs if epochs is None else epochs
    epoch = 0
    while epoch < limit:
        epoch += 1
        model.train(True)
        for batch in epoch_batches(lengths, fit_idx, schedule.batch_size, shuffler):
            x, mask = gather(reader, [rows.shas[i] for i in batch], dim)
            logits, _att = model.forward(x, mask)
            loss = weighted_focal_loss(logits, target_all[batch], weights, settings.gamma)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if hold_idx is not None:
            held = [rows.shas[i] for i in hold_idx]
            scores, _ = predict(model, reader, held, schedule.batch_size)
            if stopper.update(held_macro_f1(scores, rows.y[list(hold_idx)])):
                best = model.state()
            if stopper.should_stop:
                break
    if hold_idx is None:
        best = model.state()
    return FitResult(
        state=best,
        best_epoch=epoch if hold_idx is None else stopper.best_epoch,
        epochs_run=epoch,
        hold_macro_f1=None if hold_idx is None else float(stopper.best_score),
        seconds=time.perf_counter() - began,
    )


@dataclass(frozen=True, slots=True)
class FoldRun:
    """How one leave-one-partition-out fit went."""

    validate: int
    fit_rows: int
    holdout_rows: int
    best_epoch: int
    epochs_run: int
    hold_macro_f1: float
    seconds: float


def run_grid_point(
    source: Source,
    folds: Sequence[Fold],
    settings: FitSettings,
    test_clusters: set[int],
    log: Callable[[str], None],
) -> tuple[Array, list[FoldRun]]:
    """Out-of-fold scores of one grid point over the validation partitions.

    For each fold: the training partitions are split into fit and holdout by whole clusters,
    leakage is asserted (fit, holdout, validation and test share no cluster) and only then is
    the head fitted; the validation partition is scored by the weights of the best holdout epoch.
    """
    np = _np()
    rows = source.rows
    oof = np.zeros(rows.y.shape, dtype=np.float64)
    runs: list[FoldRun] = []
    for fold in folds:
        train_idx, held_idx = fold_rows(rows, fold)
        fit_idx, hold_idx = inner_split(
            rows.cluster_code, train_idx, settings.schedule.inner_holdout_fraction, settings.seed
        )
        assert_no_overlap(
            {
                "fit": clusters_of(rows, fit_idx),
                "holdout": clusters_of(rows, hold_idx),
                "validation": clusters_of(rows, held_idx),
                "test": test_clusters,
            }
        )
        result = fit_head(source, fit_idx, hold_idx, settings)
        model = build_model(settings.net, settings.seed, result.state)
        scores, _ = predict(
            model, source.reader, [rows.shas[i] for i in held_idx], settings.schedule.batch_size
        )
        oof[held_idx] = scores
        f1 = float(result.hold_macro_f1 or 0.0)
        runs.append(
            FoldRun(
                validate=fold.validate,
                fit_rows=len(fit_idx),
                holdout_rows=len(hold_idx),
                best_epoch=result.best_epoch,
                epochs_run=result.epochs_run,
                hold_macro_f1=f1,
                seconds=result.seconds,
            )
        )
        log(
            f"    fold {fold.validate}: best epoch {result.best_epoch}/{result.epochs_run}, "
            f"holdout macro F1 {f1:.4f}, {result.seconds:.1f} s"
        )
    return oof, runs


def refit_final(
    source: Source, settings: FitSettings, epochs: int, test_clusters: set[int]
) -> FitResult:
    """Fit on every validation-partition row for a fixed epoch count; no test cluster is in it."""
    rows = source.rows
    everything = list(range(len(rows)))
    assert_no_overlap({"train": clusters_of(rows, everything), "test": test_clusters})
    return fit_head(source, everything, None, settings, epochs=epochs)
