"""Phase 11 signal-peptide CRF head: the network, the CRF and the training loop (needs torch).

torch and numpy are imported lazily, inside the functions that need them, and typed ``Any``: the
packages are the optional ``head`` extra, not dependencies of the served image, and CI type-checks
this module without them installed.

The network (ADR-0013), for a batch of per-residue ESM-2 states ``x`` (B, L, D) cut to the first
``length_cap`` residues and a mask of real residues (B, L)::

    h = dropout(LayerNorm(x))
    h = CNN(h) | BiLSTM(h)                 # padded positions are exactly zero / packed away
    emissions = Linear(h)                  # (B, L, tags)
    tags      = linear-chain CRF over the emissions

The CRF is plain torch, a handful of functions over tensors, not an ``nn.Module`` subclass: the
log-partition by the forward algorithm in log space, Viterbi decoding, forward-backward marginals
and the score of a given tag path. Transition constraints are a boolean matrix: a forbidden
transition is replaced by a large negative score, so ``exp`` of it is exactly 0. Detection is the
probability that the path is not all-outside, ``1 - P(all outside)``, which under the
no-signal-after-outside constraint is also the marginal of the signal tag at residue 1.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import math
import time
from typing import Any

from protloc.training import crf as crf_config
from protloc.training.baselines import BaselineError, Fold
from protloc.training.head import (
    EarlyStopper,
    Schedule,
    Shuffler,
    assert_no_overlap,
    epoch_batches,
    inner_split,
)
from protloc.training.head_net import StateReader, set_deterministic

Array = Any


def _torch() -> Any:
    return importlib.import_module("torch")


def _np() -> Any:
    return importlib.import_module("numpy")


# -- the CRF, in plain torch -----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CrfTensors:
    """The CRF's scores: transitions (K, K) already constrained, start (K) and end (K)."""

    transitions: Any
    start: Any
    end: Any


def log_partition(emissions: Any, mask: Any, crf: CrfTensors) -> Any:
    """``log Z`` per sequence (B,) by the forward algorithm; padded steps leave alpha unchanged."""
    torch = _torch()
    alpha = crf.start.unsqueeze(0) + emissions[:, 0]
    for step in range(1, emissions.shape[1]):
        spread = alpha.unsqueeze(2) + crf.transitions.unsqueeze(0)
        moved = torch.logsumexp(spread, dim=1) + emissions[:, step]
        alpha = torch.where(mask[:, step].unsqueeze(1), moved, alpha)
    return torch.logsumexp(alpha + crf.end.unsqueeze(0), dim=1)


def path_score(emissions: Any, mask: Any, tags: Any, crf: CrfTensors) -> Any:
    """Score (B,) of the given tag paths (B, L): start + emissions + transitions + end, masked."""
    keep = mask.to(emissions.dtype)
    chosen = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
    total = crf.start[tags[:, 0]] + (chosen * keep).sum(dim=1)
    if emissions.shape[1] > 1:
        moves = crf.transitions[tags[:, :-1], tags[:, 1:]]
        total = total + (moves * keep[:, 1:]).sum(dim=1)
    last = mask.long().sum(dim=1) - 1
    return total + crf.end[tags.gather(1, last.unsqueeze(1)).squeeze(1)]


def viterbi(emissions: Any, mask: Any, crf: CrfTensors) -> tuple[Any, Any]:
    """The best tag path (B, L) and its score (B,); padded positions are tagged 0.

    Padded steps carry an identity back-pointer, so the best final tag flows back through them
    unchanged and the path over the real residues is the exact optimum.
    """
    torch = _torch()
    batch, width, size = emissions.shape
    delta = crf.start.unsqueeze(0) + emissions[:, 0]
    identity = torch.arange(size).unsqueeze(0).expand(batch, size)
    pointers = []
    for step in range(1, width):
        best, arg = (delta.unsqueeze(2) + crf.transitions.unsqueeze(0)).max(dim=1)
        live = mask[:, step].unsqueeze(1)
        delta = torch.where(live, best + emissions[:, step], delta)
        pointers.append(torch.where(live, arg, identity))
    final = delta + crf.end.unsqueeze(0)
    score, tag = final.max(dim=1)
    path = torch.zeros(batch, width, dtype=torch.long)
    path[:, width - 1] = tag
    for step in range(width - 1, 0, -1):
        tag = pointers[step - 1].gather(1, tag.unsqueeze(1)).squeeze(1)
        path[:, step - 1] = tag
    return path * mask.long(), score


def marginals(emissions: Any, mask: Any, crf: CrfTensors) -> Any:
    """Posterior probability (B, L, K) of each tag at each residue by forward-backward.

    Padded positions are exactly 0; at every real position the K values sum to 1 (clamped to
    [0, 1] against float32 round-off in the log-space sums).
    """
    torch = _torch()
    batch, width, size = emissions.shape
    alphas = [crf.start.unsqueeze(0) + emissions[:, 0]]
    for step in range(1, width):
        spread = alphas[-1].unsqueeze(2) + crf.transitions.unsqueeze(0)
        moved = torch.logsumexp(spread, dim=1) + emissions[:, step]
        alphas.append(torch.where(mask[:, step].unsqueeze(1), moved, alphas[-1]))
    log_z = torch.logsumexp(alphas[-1] + crf.end.unsqueeze(0), dim=1)
    tail = crf.end.unsqueeze(0).expand(batch, size)
    betas = [tail] * width
    for step in range(width - 2, -1, -1):
        onward = (emissions[:, step + 1] + betas[step + 1]).unsqueeze(1)
        moved = torch.logsumexp(crf.transitions.unsqueeze(0) + onward, dim=2)
        betas[step] = torch.where(mask[:, step + 1].unsqueeze(1), moved, tail)
    joint = torch.stack(alphas, dim=1) + torch.stack(betas, dim=1) - log_z.unsqueeze(1).unsqueeze(2)
    return torch.exp(joint).clamp(0.0, 1.0) * mask.unsqueeze(2).to(joint.dtype)


def outside_log_prob(emissions: Any, mask: Any, crf: CrfTensors, outside: int) -> Any:
    """``log P(every residue is outside)`` per sequence (<= 0)."""
    torch = _torch()
    path = torch.full(tuple(mask.shape), outside, dtype=torch.long)
    return path_score(emissions, mask, path, crf) - log_partition(emissions, mask, crf)


def log_one_minus_exp(value: Any) -> Any:
    """``log(1 - exp(value))`` for ``value <= 0``, stable near 0 and far below it."""
    torch = _torch()
    value = value.clamp(max=-torch.finfo(value.dtype).eps)
    near = torch.log(-torch.expm1(value))
    far = torch.log1p(-torch.exp(value))
    return torch.where(value > -math.log(2.0), near, far)


def last_tagged(path: Any, mask: Any, tag: int) -> Any:
    """The 1-based last residue carrying ``tag`` in each path (0 if none), as a long tensor."""
    torch = _torch()
    positions = torch.arange(1, path.shape[1] + 1).unsqueeze(0)
    return ((path == tag) & mask).long().mul(positions).max(dim=1).values


def forced_start(crf: CrfTensors, outside: int, penalty: float) -> CrfTensors:
    """The same CRF with the outside tag forbidden at residue 1 (decode given a signal peptide)."""
    torch = _torch()
    block = torch.zeros_like(crf.start)
    block[outside] = penalty
    return CrfTensors(crf.transitions, crf.start + block, crf.end)


# -- the network -----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Net:
    """The settings of one network."""

    architecture: crf_config.Architecture
    tags: crf_config.Tags
    encoder: str
    dropout: float


@dataclass(frozen=True, slots=True)
class FitSettings:
    """Everything a fit depends on besides the data."""

    net: Net
    learning_rate: float
    weight_decay: float
    schedule: Schedule
    seed: int


def settings_for(point: crf_config.GridPoint, config: crf_config.CrfConfig) -> FitSettings:
    """The fit settings of one grid point."""
    return FitSettings(
        net=Net(config.architecture, config.tags, point.encoder, point.dropout),
        learning_rate=point.learning_rate,
        weight_decay=config.weight_decay,
        schedule=config.schedule,
        seed=config.seed,
    )


class SequenceTagger:
    """Encoder, emissions and CRF scores; a plain class around ``ModuleDict`` (no ``nn.Module``)."""

    def __init__(self, net: Net, seed: int) -> None:
        """Build the layers; initial weights depend only on ``seed`` and the shapes."""
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(seed)
        arch = net.architecture
        layers: dict[str, Any] = {
            "norm": nn.LayerNorm(arch.embedding_dim),
            "drop": nn.Dropout(net.dropout),
        }
        if net.encoder == "cnn":
            width = arch.embedding_dim
            for index in range(arch.cnn_layers):
                layers[f"conv{index}"] = nn.Conv1d(
                    width, arch.cnn_hidden, arch.cnn_kernel, padding=arch.cnn_kernel // 2
                )
                width = arch.cnn_hidden
            out = arch.cnn_hidden
        elif net.encoder == "bilstm":
            layers["lstm"] = nn.LSTM(
                arch.embedding_dim,
                arch.lstm_hidden,
                num_layers=arch.lstm_layers,
                batch_first=True,
                bidirectional=True,
            )
            out = 2 * arch.lstm_hidden
        else:
            raise BaselineError(f"unknown encoder {net.encoder!r}")
        size = len(net.tags.names)
        layers["emit"] = nn.Linear(out, size)
        layers["crf"] = nn.ParameterDict(
            {
                "transitions": nn.Parameter(torch.zeros(size, size)),
                "start": nn.Parameter(torch.zeros(size)),
                "end": nn.Parameter(torch.zeros(size)),
            }
        )
        self.net = net
        self.layers = nn.ModuleDict(layers)
        self.allowed = torch.tensor(net.tags.allowed(), dtype=torch.bool)

    def parameters(self) -> list[Any]:
        """The trainable tensors."""
        return list(self.layers.parameters())

    def parameter_count(self) -> int:
        """Number of trainable scalars."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True) -> None:
        """Switch dropout on (training) or off (evaluation)."""
        self.layers.train(mode)

    def emissions(self, x: Any, mask: Any) -> Any:
        """Per-residue tag scores (B, L, K) from states ``x`` and the real-residue mask."""
        torch = _torch()
        layers = self.layers
        hidden = layers["drop"](layers["norm"](x))
        if self.net.encoder == "cnn":
            keep = mask.unsqueeze(1).to(hidden.dtype)
            hidden = hidden.transpose(1, 2) * keep
            for index in range(self.net.architecture.cnn_layers):
                conv = layers[f"conv{index}"]
                hidden = layers["drop"](torch.nn.functional.gelu(conv(hidden))) * keep
            hidden = hidden.transpose(1, 2)
        else:
            lengths = mask.long().sum(dim=1).cpu()
            rnn = torch.nn.utils.rnn
            packed = rnn.pack_padded_sequence(
                hidden, lengths, batch_first=True, enforce_sorted=False
            )
            produced, _state = layers["lstm"](packed)
            hidden, _sizes = rnn.pad_packed_sequence(
                produced, batch_first=True, total_length=mask.shape[1]
            )
            hidden = layers["drop"](hidden)
        return layers["emit"](hidden)

    def crf(self) -> CrfTensors:
        """The CRF scores with every forbidden transition replaced by the forbidden score."""
        scores = self.layers["crf"]
        constrained = scores["transitions"].masked_fill(
            ~self.allowed, self.net.tags.forbidden_score
        )
        return CrfTensors(constrained, scores["start"], scores["end"])

    def state(self) -> dict[str, Array]:
        """A copy of every tensor as a numpy array (saved without pickle)."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.layers.state_dict().items()}

    def load_state(self, state: Mapping[str, Array]) -> None:
        """Load tensors produced by :meth:`state`."""
        torch = _torch()
        self.layers.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()})


def build_model(net: Net, seed: int, state: Mapping[str, Array] | None = None) -> SequenceTagger:
    """A model in evaluation mode, with ``state`` loaded when given."""
    model = SequenceTagger(net, seed)
    if state is not None:
        model.load_state(state)
    model.train(False)
    return model


def gold_paths(cleavage: Any, mask: Any, tags: crf_config.Tags) -> Any:
    """Tag paths (B, L): residues 1..cleavage are signal, the rest and the padding outside."""
    torch = _torch()
    positions = torch.arange(1, mask.shape[1] + 1).unsqueeze(0)
    signal = ((positions <= cleavage.unsqueeze(1)) & mask).long()
    return signal * tags.signal_index + (1 - signal) * tags.outside_index


def batch_loss(model: SequenceTagger, x: Any, mask: Any, truth: tuple[Any, Any]) -> Any:
    """Mean per-sequence loss. ``truth`` is (cleavage, tagged), both (B,).

    A tagged sequence pays the tag-path NLL ``log Z - score(gold)``. A positive without a usable
    cleavage site is masked in the tag loss and pays only the detection NLL
    ``-log(1 - P(all outside))``, the probability that the path holds a signal somewhere.
    """
    cleavage, tagged = truth
    tags = model.net.tags
    emissions = model.emissions(x, mask)
    scores = model.crf()
    log_z = log_partition(emissions, mask, scores)
    tag_nll = log_z - path_score(emissions, mask, gold_paths(cleavage, mask, tags), scores)
    detection_nll = -log_one_minus_exp(
        outside_log_prob(emissions, mask, scores, tags.outside_index)
    )
    return _torch().where(tagged, tag_nll, detection_nll).mean()


# -- data ------------------------------------------------------------------------------
def gather(reader: StateReader, shas: Sequence[str], dim: int, cap: int) -> tuple[Any, Any]:
    """Pad the first ``cap`` states of each sequence into (B, Lmax, dim) float32 and a bool mask."""
    np, torch = _np(), _torch()
    arrays = [reader.states(sha)[:cap] for sha in shas]
    widest = max(int(a.shape[0]) for a in arrays)
    x = np.zeros((len(arrays), widest, dim), dtype=np.float32)
    mask = np.zeros((len(arrays), widest), dtype=bool)
    for i, array in enumerate(arrays):
        x[i, : array.shape[0]] = array
        mask[i, : array.shape[0]] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


@dataclass(frozen=True, slots=True)
class Source:
    """The labelled rows and the reader that returns their per-residue states."""

    reader: StateReader
    rows: crf_config.Rows


@dataclass(frozen=True, slots=True)
class Prediction:
    """What the head says about a list of sequences, in the order asked.

    ``score`` is ``P(any signal tag)``, the detection score. ``cleavage`` is the last signal
    residue of the best path GIVEN that a signal exists (always at least 1), which is what is
    reported when detection fires. ``path_cleavage`` is that of the unconstrained best path (0
    means the best single path has no signal). ``marginals`` is the per-residue signal posterior.
    """

    score: Array
    cleavage: Array
    path_cleavage: Array
    marginals: list[Array] | None


def predict(
    model: SequenceTagger,
    reader: StateReader,
    shas: Sequence[str],
    batch_size: int,
    keep_marginals: bool = False,
) -> Prediction:
    """Detection score, cleavage site and (optionally) per-residue marginals for each sequence."""
    np, torch = _np(), _torch()
    arch, tags = model.net.architecture, model.net.tags
    lengths = [min(int(reader.states(sha).shape[0]), arch.length_cap) for sha in shas]
    order = sorted(range(len(shas)), key=lambda i: lengths[i])
    score = np.zeros(len(shas), dtype=np.float64)
    cleavage = np.zeros(len(shas), dtype=np.int64)
    path_cleavage = np.zeros(len(shas), dtype=np.int64)
    posterior: list[Array | None] = [None] * len(shas)
    model.train(False)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            x, mask = gather(reader, [shas[i] for i in chunk], arch.embedding_dim, arch.length_cap)
            emissions, scores = model.emissions(x, mask), model.crf()
            outside = outside_log_prob(emissions, mask, scores, tags.outside_index)
            forced = forced_start(scores, tags.outside_index, tags.forbidden_score)
            given, _ = viterbi(emissions, mask, forced)
            best, _ = viterbi(emissions, mask, scores)
            detection = (-torch.expm1(outside)).numpy().astype(np.float64)
            kept = last_tagged(given, mask, tags.signal_index).numpy()
            free = last_tagged(best, mask, tags.signal_index).numpy()
            soft = marginals(emissions, mask, scores)[:, :, tags.signal_index].numpy()
            for k, i in enumerate(chunk):
                score[i], cleavage[i], path_cleavage[i] = detection[k], kept[k], free[k]
                if keep_marginals:
                    posterior[i] = soft[k, : lengths[i]].astype(np.float64)
    full = [m for m in posterior if m is not None] if keep_marginals else None
    return Prediction(score, cleavage, path_cleavage, full)


def evaluate_loss(
    model: SequenceTagger, source: Source, indices: Sequence[int], batch_size: int
) -> float:
    """Mean per-sequence loss of ``model`` on the given rows (evaluation mode, no gradient)."""
    np, torch = _np(), _torch()
    rows, arch = source.rows, model.net.architecture
    model.train(False)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = list(indices[start : start + batch_size])
            shas = [rows.shas[i] for i in chunk]
            x, mask = gather(source.reader, shas, arch.embedding_dim, arch.length_cap)
            truth = (
                torch.from_numpy(np.asarray(rows.cleavage)[chunk]),
                torch.from_numpy(np.asarray(rows.tagged)[chunk]),
            )
            total += float(batch_loss(model, x, mask, truth)) * len(chunk)
    return total / len(indices)


@dataclass(frozen=True, slots=True)
class FitResult:
    """One fit: the weights to keep and how it went."""

    state: Mapping[str, Array]
    best_epoch: int
    epochs_run: int
    hold_nll: float | None
    seconds: float


def fit_tagger(
    source: Source,
    fit_idx: Sequence[int],
    hold_idx: Sequence[int] | None,
    settings: FitSettings,
    epochs: int | None = None,
) -> FitResult:
    """Train one head on ``fit_idx``.

    With ``hold_idx`` the holdout mean NLL is computed after every epoch, training stops after
    ``patience`` epochs without improvement and the best epoch's weights are kept. Without it,
    exactly ``epochs`` epochs run and the final weights are kept. Rows beyond ``fit_idx`` are
    never read by the loss.
    """
    if (hold_idx is None) == (epochs is None):
        raise BaselineError("give either a holdout (early stopping) or a fixed epoch count")
    np, torch = _np(), _torch()
    rows, arch, schedule = source.rows, settings.net.architecture, settings.schedule
    began = time.perf_counter()
    set_deterministic(settings.seed)
    model = SequenceTagger(settings.net, settings.seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    capped = [min(int(n), arch.length_cap) for n in rows.length]
    cleavage_all = torch.from_numpy(np.asarray(rows.cleavage))
    tagged_all = torch.from_numpy(np.asarray(rows.tagged))
    shuffler, stopper = Shuffler(settings.seed), EarlyStopper(schedule.patience)
    best: Mapping[str, Array] = model.state()
    limit = schedule.max_epochs if epochs is None else epochs
    epoch = 0
    while epoch < limit:
        epoch += 1
        model.train(True)
        for batch in epoch_batches(capped, fit_idx, schedule.batch_size, shuffler):
            x, mask = gather(
                source.reader, [rows.shas[i] for i in batch], arch.embedding_dim, arch.length_cap
            )
            loss = batch_loss(model, x, mask, (cleavage_all[batch], tagged_all[batch]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if hold_idx is not None:
            if stopper.update(-evaluate_loss(model, source, hold_idx, schedule.batch_size)):
                best = model.state()
            if stopper.should_stop:
                break
    if hold_idx is None:
        best = model.state()
    return FitResult(
        state=best,
        best_epoch=epoch if hold_idx is None else stopper.best_epoch,
        epochs_run=epoch,
        hold_nll=None if hold_idx is None else -float(stopper.best_score),
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
    hold_nll: float
    seconds: float


@dataclass(frozen=True, slots=True)
class GridResult:
    """Out-of-fold predictions of one grid point and the fold weights that produced them."""

    score: Array
    cleavage: Array
    runs: list[FoldRun]
    states: dict[int, Mapping[str, Array]]


def run_grid_point(
    source: Source,
    folds: Sequence[Fold],
    settings: FitSettings,
    test_clusters: set[int],
    log: Callable[[str], None],
) -> GridResult:
    """Out-of-fold predictions of one grid point over the validation partitions.

    For each fold: the training partitions are split into fit and holdout by whole clusters,
    leakage is asserted (fit, holdout, validation and test share no cluster) and only then is
    the head fitted; the validation partition is scored by the weights of the best holdout epoch.
    """
    np = _np()
    rows = source.rows
    score = np.zeros(len(rows), dtype=np.float64)
    cleavage = np.zeros(len(rows), dtype=np.int64)
    runs: list[FoldRun] = []
    states: dict[int, Mapping[str, Array]] = {}
    for fold in folds:
        train_idx, held_idx = crf_config.fold_rows(rows, fold)
        fit_idx, hold_idx = inner_split(
            rows.cluster_code, train_idx, settings.schedule.inner_holdout_fraction, settings.seed
        )
        assert_no_overlap(
            {
                "fit": crf_config.clusters_of(rows, fit_idx),
                "holdout": crf_config.clusters_of(rows, hold_idx),
                "validation": crf_config.clusters_of(rows, held_idx),
                "test": test_clusters,
            }
        )
        result = fit_tagger(source, fit_idx, hold_idx, settings)
        model = build_model(settings.net, settings.seed, result.state)
        shas = [rows.shas[i] for i in held_idx]
        prediction = predict(model, source.reader, shas, settings.schedule.batch_size)
        score[held_idx], cleavage[held_idx] = prediction.score, prediction.cleavage
        states[fold.validate] = result.state
        nll = float(result.hold_nll or 0.0)
        runs.append(
            FoldRun(
                fold.validate,
                len(fit_idx),
                len(hold_idx),
                result.best_epoch,
                result.epochs_run,
                nll,
                result.seconds,
            )
        )
        log(
            f"    fold {fold.validate}: best epoch {result.best_epoch}/{result.epochs_run}, "
            f"holdout NLL {nll:.4f}, {result.seconds:.1f} s"
        )
    return GridResult(score, cleavage, runs, states)


def refit_final(
    source: Source, settings: FitSettings, epochs: int, test_clusters: set[int]
) -> FitResult:
    """Fit on every validation-partition row for a fixed epoch count; no test cluster is in it."""
    rows = source.rows
    everything = list(range(len(rows)))
    assert_no_overlap({"train": crf_config.clusters_of(rows, everything), "test": test_clusters})
    return fit_tagger(source, everything, None, settings, epochs=epochs)
