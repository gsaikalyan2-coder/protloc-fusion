"""Phase 12 GO head: the training loop (needs torch).

The network is the Phase 9 one, reused unchanged from ``protloc.training.head_net`` (attention
pooling over the frozen per-residue ESM-2 states, then an MLP with one sigmoid per GO term);
this module only adds what is specific to GO (ADR-0014): the loss weights are capped
``negatives / positives`` over the labelled fit rows, early stopping watches the mean binary
cross-entropy of an inner whole-cluster holdout, and the rows, folds and leakage guard are the
ones of ``protloc.training.go``.

torch and numpy are imported lazily and typed ``Any``, as in ``head_net``: CI type-checks this
module without them installed.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import time
from typing import Any

from protloc.training import go
from protloc.training.baselines import BaselineError, Fold
from protloc.training.head import (
    EarlyStopper,
    Shuffler,
    assert_no_overlap,
    epoch_batches,
    inner_split,
)
from protloc.training.head_net import (
    AttentionHead,
    FitSettings,
    Net,
    StateReader,
    build_model,
    gather,
    predict,
    set_deterministic,
    weighted_focal_loss,
)

Array = Any


def _torch() -> Any:
    return importlib.import_module("torch")


def _np() -> Any:
    return importlib.import_module("numpy")


@dataclass(frozen=True, slots=True)
class Source:
    """The labelled rows and the reader that returns their per-residue states."""

    reader: StateReader
    rows: go.Rows


@dataclass(frozen=True, slots=True)
class Plan:
    """One grid point's fit settings and the cap on the per-term positive weight."""

    settings: FitSettings
    pos_weight_cap: float


def plan_for(point: go.GridPoint, config: go.GoConfig, classes: int) -> Plan:
    """The fit plan of one grid point."""
    return Plan(
        settings=FitSettings(
            net=Net(config.architecture, point.hidden, point.dropout, classes),
            gamma=point.gamma,
            learning_rate=point.learning_rate,
            weight_decay=config.weight_decay,
            schedule=config.schedule,
            seed=config.seed,
        ),
        pos_weight_cap=config.pos_weight_cap,
    )


def clusters_of(rows: go.Rows, indices: Sequence[int]) -> set[int]:
    """The ``cluster_id`` codes of the given rows."""
    return {int(rows.cluster_code[i]) for i in indices}


@dataclass(frozen=True, slots=True)
class FitResult:
    """One fit: the weights to keep and how it went."""

    state: Mapping[str, Array]
    best_epoch: int
    epochs_run: int
    hold_nll: float | None
    seconds: float


def fit_head(
    source: Source,
    fit_idx: Sequence[int],
    hold_idx: Sequence[int] | None,
    plan: Plan,
    epochs: int | None = None,
) -> FitResult:
    """Train one head on ``fit_idx``.

    With ``hold_idx`` the holdout mean cross-entropy is computed after every epoch, training
    stops after ``patience`` epochs without improvement and the best epoch's weights are kept.
    Without it, exactly ``epochs`` epochs run and the final weights are kept. The loss weights
    come from the ``fit_idx`` rows only.
    """
    if (hold_idx is None) == (epochs is None):
        raise BaselineError("give either a holdout (early stopping) or a fixed epoch count")
    np, torch = _np(), _torch()
    reader, rows, settings = source.reader, source.rows, plan.settings
    began = time.perf_counter()
    set_deterministic(settings.seed)
    schedule = settings.schedule
    dim = settings.net.architecture.embedding_dim
    model = AttentionHead(settings.net, settings.seed)
    weights = torch.tensor(
        go.capped_pos_weights(rows.y[list(fit_idx)].tolist(), plan.pos_weight_cap),
        dtype=torch.float32,
    )
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
            if stopper.update(-go.holdout_nll(scores, rows.y[list(hold_idx)])):
                best = model.state()
            if stopper.should_stop:
                break
    if hold_idx is None:
        best = model.state()
    return FitResult(
        state=best,
        best_epoch=epoch if hold_idx is None else stopper.best_epoch,
        epochs_run=epoch,
        hold_nll=None if hold_idx is None else float(-stopper.best_score),
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
class GridRun:
    """One grid point over every fold: out-of-fold scores, how each fit went, the fold weights."""

    oof: Array
    runs: list[FoldRun]
    states: dict[int, Mapping[str, Array]]


def run_grid_point(
    source: Source,
    folds: Sequence[Fold],
    plan: Plan,
    test_clusters: set[int],
    log: Callable[[str], None],
) -> GridRun:
    """Out-of-fold scores of one grid point over the validation partitions.

    For each fold: the training partitions are split into fit and holdout by whole clusters,
    leakage is asserted (fit, holdout, validation and test share no cluster) and only then is
    the head fitted; the validation partition is scored by the weights of the best holdout epoch.
    """
    np = _np()
    rows, settings = source.rows, plan.settings
    oof = np.zeros(rows.y.shape, dtype=np.float64)
    runs: list[FoldRun] = []
    states: dict[int, Mapping[str, Array]] = {}
    for fold in folds:
        train_idx, held_idx = go.fold_rows(rows, fold)
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
        result = fit_head(source, fit_idx, hold_idx, plan)
        model = build_model(settings.net, settings.seed, result.state)
        scores, _ = predict(
            model, source.reader, [rows.shas[i] for i in held_idx], settings.schedule.batch_size
        )
        oof[held_idx] = scores
        states[fold.validate] = result.state
        nll = float(result.hold_nll if result.hold_nll is not None else float("nan"))
        runs.append(
            FoldRun(
                validate=fold.validate,
                fit_rows=len(fit_idx),
                holdout_rows=len(hold_idx),
                best_epoch=result.best_epoch,
                epochs_run=result.epochs_run,
                hold_nll=nll,
                seconds=result.seconds,
            )
        )
        log(
            f"    fold {fold.validate}: best epoch {result.best_epoch}/{result.epochs_run}, "
            f"holdout NLL {nll:.4f}, {result.seconds:.1f} s"
        )
    return GridRun(oof, runs, states)


def refit_final(source: Source, plan: Plan, epochs: int, test_clusters: set[int]) -> FitResult:
    """Fit on every validation-partition row for a fixed epoch count; no test cluster is in it."""
    rows = source.rows
    everything = list(range(len(rows)))
    assert_no_overlap({"train": clusters_of(rows, everything), "test": test_clusters})
    return fit_head(source, everything, None, plan, epochs=epochs)
