"""Phase 9 localization head: configuration, protocol guards, network, loss and manifest.

Pure-Python tests run everywhere. Tests that need numpy or torch skip themselves when the
package is absent, so CI without the ``head`` extra still runs the rest.
"""

import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from protloc.training import baseline_models as bm
from protloc.training import head, head_net
from protloc.training.baselines import (
    SUBSET_LABEL,
    SUBSET_SCOPE,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    Fold,
    digest,
    lopo_folds,
)

net = head_net
REPO = Path(__file__).resolve().parents[3]
HEAD_YAML = REPO / "configs" / "training" / "head.yaml"
SHA = "a" * 64


@pytest.fixture
def np() -> Any:
    """numpy, or skip the test."""
    return pytest.importorskip("numpy")


@pytest.fixture
def torch() -> Any:
    """Torch (and numpy), or skip the test."""
    pytest.importorskip("numpy")
    return pytest.importorskip("torch")


# -- configuration ---------------------------------------------------------------------
def _yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_config_agrees_with_the_root_and_model_configs() -> None:
    config = head.load_config(HEAD_YAML)
    assert config.seed == _yaml(REPO / "configs" / "config.yaml")["seed"]
    trunk = _yaml(REPO / "configs" / "model" / "trunk_esm2_150m.yaml")["trunk"]
    assert config.architecture.embedding_dim == trunk["embedding_dim"]
    samples = _yaml(REPO / "configs" / "evaluation" / "default.yaml")["reporting"]
    assert samples["bootstrap_samples"] == 1000


def test_grid_expands_in_the_documented_order() -> None:
    config = head.load_config(HEAD_YAML)
    points = head.grid_points(config)
    assert len(points) == 16
    assert points[0].as_dict() == {
        "learning_rate": 0.001,
        "dropout": 0.3,
        "hidden": 256,
        "gamma": 0.0,
    }
    assert points[1].gamma == 2.0  # gamma varies fastest
    assert points[8].learning_rate == 0.0003  # learning rate varies slowest
    assert len({tuple(p.as_dict().items()) for p in points}) == 16


def _write_variant(tmp_path: Path, edit: Any) -> Path:
    doc = _yaml(HEAD_YAML)
    edit(doc)
    path = tmp_path / "head.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["loss"].update(kind="bce"),
        lambda d: d["grid"].pop("gamma"),
        lambda d: d["grid"].update(dropout=[0.3, 1.0]),
        lambda d: d["grid"].update(learning_rate=[0.001, 0.001]),
        lambda d: d["architecture"].update(pooling="mean"),
        lambda d: d["schedule"].update(inner_holdout_fraction=1.0),
        lambda d: d["optimiser"].update(name="sgd"),
    ],
)
def test_malformed_config_is_refused(tmp_path: Path, edit: Any) -> None:
    with pytest.raises(BaselineError):
        head.load_config(_write_variant(tmp_path, edit))


def test_pyproject_pins_the_head_extra_exactly() -> None:
    text = (REPO / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    assert head.pinned_version(text, "torch") == "2.14.1"
    assert head.pinned_version(text, "numpy") == "2.4.6"
    with pytest.raises(BaselineError):
        head.pinned_version(text, "not-a-package")


# -- loss weights come from the training fold only -------------------------------------
def test_pos_weights_are_negatives_over_positives() -> None:
    labels = [[1, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
    assert head.pos_weights(labels) == [3.0, 1.0, 1.0]  # 3/1, no positive (1.0), 2/2
    assert head.pos_weights([[1, 0], [0, 0], [0, 0]]) == [2.0, 1.0]
    with pytest.raises(BaselineError):
        head.pos_weights([])


def test_pos_weights_use_the_training_rows_not_the_validation_rows(np: Any) -> None:
    partition = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([[1], [0], [0], [0], [1], [1], [1], [0]])
    rows = head.Rows(
        shas=[f"s{i}" for i in range(8)],
        y=y,
        partition=partition,
        cluster_code=np.arange(8),
    )
    train, held = head.fold_rows(rows, Fold(train=(1,), validate=2))
    assert held == [4, 5, 6, 7]
    train_weight = head.pos_weights(rows.y[train].tolist())
    assert train_weight == [3.0]  # partition 1 only: 3 negatives / 1 positive
    assert head.pos_weights(rows.y.tolist()) != train_weight  # the validation rows would change it


# -- early stopping, batches, inner holdout, leakage -----------------------------------
def test_early_stopper_keeps_the_best_epoch_and_stops_after_patience() -> None:
    stopper = head.EarlyStopper(patience=3)
    improved = [stopper.update(s) for s in (0.1, 0.3, 0.3, 0.2, 0.1)]
    assert improved == [True, True, False, False, False]  # a tie is not an improvement
    assert stopper.best_epoch == 2
    assert stopper.should_stop
    assert not head.EarlyStopper(3).should_stop
    with pytest.raises(BaselineError):
        head.EarlyStopper(0)


def test_epoch_batches_cover_every_row_once_and_are_deterministic() -> None:
    lengths = [(i * 37) % 101 + 1 for i in range(100)]
    indices = list(range(0, 100, 2))
    batches = head.epoch_batches(lengths, indices, 8, head.Shuffler(5))
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == indices
    assert all(1 <= len(b) <= 8 for b in batches)
    assert batches == head.epoch_batches(lengths, indices, 8, head.Shuffler(5))
    assert batches != head.epoch_batches(lengths, indices, 8, head.Shuffler(6))
    shuffler = head.Shuffler(5)
    first = head.epoch_batches(lengths, indices, 8, shuffler)
    assert first != head.epoch_batches(lengths, indices, 8, shuffler)  # epochs differ
    with pytest.raises(BaselineError):
        head.epoch_batches(lengths, indices, 0, head.Shuffler(5))


def test_inner_split_holds_out_whole_clusters() -> None:
    clusters = [i // 3 for i in range(30)]  # ten clusters of three rows
    rows = list(range(30))
    fit, hold = head.inner_split(clusters, rows, 0.2, seed=11)
    assert sorted(fit + hold) == rows
    assert {clusters[i] for i in fit}.isdisjoint({clusters[i] for i in hold})
    assert len({clusters[i] for i in hold}) == 2  # round(0.2 * 10)
    assert (fit, hold) == head.inner_split(clusters, rows, 0.2, seed=11)
    assert hold != head.inner_split(clusters, rows, 0.2, seed=12)[1]
    one = head.inner_split([0, 0, 1, 1], [0, 1, 2, 3], 0.9, seed=1)
    assert len({0, 1} - {[0, 0, 1, 1][i] for i in one[0]}) == 1  # never every cluster
    with pytest.raises(BaselineError):
        head.inner_split([0, 0, 0], [0, 1, 2], 0.2, seed=1)


def test_assert_no_overlap_names_the_leaking_sets() -> None:
    head.assert_no_overlap({"train": {1, 2}, "validation": {3}, "test": {4, 5}})
    with pytest.raises(BaselineError, match="'train' and 'test'"):
        head.assert_no_overlap({"train": {1, 2}, "validation": {3}, "test": {2, 5}})


def test_the_protocol_uses_partitions_one_to_four_and_keeps_five_out() -> None:
    assert VALIDATION_PARTITIONS == (1, 2, 3, 4)
    assert TEST_PARTITION == 5
    assert all(TEST_PARTITION not in fold.train for fold in lopo_folds())


# -- rows and metrics (numpy) ----------------------------------------------------------
def _table(np: Any) -> dict[str, Any]:
    """Six accessions over two sequences-each pattern; accession 5 has an unlabelled state."""
    mask = np.array([True, True, True, True, True, False])
    return {
        "seq_index": np.array([0, 1, 2, 3, 4, 5]),
        "partition": np.array([1, 1, 2, 3, 4, 5]),
        "cluster_code": np.array([0, 1, 2, 3, 4, 5]),
        "M_localization": mask,
        "Y_localization": np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 0]]),
    }


def test_build_rows_keeps_annotated_rows_of_the_requested_partitions(np: Any) -> None:
    order = [f"sha{i}" for i in range(6)]
    rows = head.build_rows(_table(np), order, (1, 2), 1)
    assert list(rows.shas) == ["sha0", "sha1", "sha2"]
    assert rows.y.tolist() == [[1, 0], [0, 1], [1, 1]]
    every_other = head.build_rows(_table(np), order, (1, 2, 3, 4), 2)
    assert list(every_other.shas) == ["sha0", "sha2", "sha4"]
    # A row whose label state is unknown/not_applicable is masked, never read as all-negative.
    assert "sha5" not in head.build_rows(_table(np), order, (5,), 1).shas
    assert head.held_out_clusters(_table(np)) == {5}


def test_metrics_match_a_hand_computation(np: Any) -> None:
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    p = np.array([[1, 0], [0, 1], [0, 1], [0, 0]])
    assert bm.macro_f1(y, p) == pytest.approx((2 / 3 + 1.0) / 2)
    assert bm.sample_jaccard(y, p).tolist() == [1.0, 0.5, 1.0, 1.0]  # empty vs empty is 1
    counts = bm._counts(y, p)
    mcc = bm.mcc_from_counts(*counts)
    assert mcc[0] == pytest.approx(2 / math.sqrt(12))
    assert mcc[1] == pytest.approx(1.0)
    scores = np.array([[0.9, 0.1], [0.8, 0.9], [0.2, 0.7], [0.1, 0.2]])
    assert head.held_macro_f1(scores, y) == pytest.approx(1.0)  # perfectly ranked


def test_an_undefined_class_reports_mcc_zero_and_is_kept(np: Any) -> None:
    y = np.array([[1, 0], [0, 1], [0, 1], [0, 0]])  # class 0 has a single positive
    report = bm.metrics_report(y, y.copy(), np.arange(4), ["a", "b"], bm.Bootstrap(0, 1))
    assert report["per_class"]["a"]["undefined"] is True
    assert report["per_class"]["a"]["mcc"]["value"] == 0.0
    assert report["per_class"]["b"]["undefined"] is False
    assert report["per_class"]["b"]["mcc"]["value"] == pytest.approx(1.0)


# -- the manifest ----------------------------------------------------------------------
CLASSES = tuple(f"class{i}" for i in range(10))


def _pins() -> head.Pins:
    return head.Pins(
        registry_sha256="1" * 64,
        corpus_sha256="2" * 64,
        embedding_cache_key="3" * 64,
        residue_cache_key="4" * 64,
        config_sha256="5" * 64,
        classes=CLASSES,
        torch="2.14.1",
    )


def _manifest() -> dict[str, Any]:
    frozen: dict[str, Any] = {
        "protocol": {
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
        },
        "scope": SUBSET_SCOPE,
        "scope_label": SUBSET_LABEL,
        "subset": {
            "embedding_cache_key": "3" * 64,
            "sequences_sha256": SHA,
            "sequences": 500,
        },
        "inputs": {
            "registry_sha256": "1" * 64,
            "corpus_sha256": "2" * 64,
            "residue_cache_key": "4" * 64,
            "head_config_sha256": "5" * 64,
            "feature_table_sha256": SHA,
            "class_map_sha256": SHA,
            "weights_sha256": SHA,
        },
        "classes": list(CLASSES),
        "selected": {"trainable_parameters": 209035, "final_epochs": 12},
        "thresholds": [0.5 + 0.01 * i for i in range(10)],
        "validation": {"pooled": {}},
        "software": {"torch": "2.14.1"},
    }
    return {
        "schema_version": head.SCHEMA_VERSION,
        "phase": 9,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }


def _refreeze(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["frozen_sha256"] = digest(manifest["frozen"])
    return manifest


def test_a_sound_manifest_has_no_problems_with_or_without_a_test_block() -> None:
    manifest = _manifest()
    assert head.check_head_manifest(manifest, _pins()) == []
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "metrics": {"per_class": {name: {} for name in CLASSES}},
    }
    assert head.check_head_manifest(manifest, _pins()) == []


@pytest.mark.parametrize(
    ("edit", "fragment"),
    [
        (lambda m: m["frozen"]["inputs"].update(registry_sha256="9" * 64), "registry_sha256"),
        (lambda m: m["frozen"]["inputs"].update(weights_sha256="xyz"), "weights_sha256"),
        (lambda m: m["frozen"]["inputs"].update(head_config_sha256="9" * 64), "head_config"),
        (lambda m: m["frozen"].update(scope="full"), "scope"),
        (lambda m: m["frozen"].update(scope_label="subset"), "scope_label"),
        (lambda m: m["frozen"].update(thresholds=[0.5] * 9), "one threshold per class"),
        (lambda m: m["frozen"].update(thresholds=[0.5] * 9 + [math.inf]), "finite"),
        (lambda m: m["frozen"]["software"].update(torch="2.0.0"), "torch version"),
        (lambda m: m["frozen"]["protocol"].update(test_partition=4), "test partition"),
        (lambda m: m["frozen"]["selected"].update(final_epochs=0), "final_epochs"),
        (lambda m: m["frozen"]["subset"].update(embedding_cache_key="8" * 64), "embedding_cache"),
    ],
)
def test_every_disagreement_is_reported(edit: Any, fragment: str) -> None:
    manifest = _manifest()
    edit(manifest)
    _refreeze(manifest)
    problems = head.check_head_manifest(manifest, _pins())
    assert any(fragment in p for p in problems), problems


def test_a_hand_edited_frozen_block_is_detected() -> None:
    manifest = _manifest()
    manifest["frozen"]["thresholds"][0] = 0.123  # edited after the digest was taken
    assert any("frozen_sha256" in p for p in head.check_head_manifest(manifest, _pins()))


def test_a_test_block_must_be_single_touch_and_match_the_frozen_digest() -> None:
    manifest = _manifest()
    good = {"per_class": {name: {} for name in CLASSES}}
    manifest["test"] = {"touches": 2, "frozen_sha256": "0" * 64, "metrics": good}
    problems = head.check_head_manifest(manifest, _pins())
    assert any("exactly once" in p for p in problems)
    assert any("different frozen block" in p for p in problems)
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "metrics": {"per_class": {}},
    }
    assert any("every class" in p for p in head.check_head_manifest(manifest, _pins()))


def test_a_manifest_missing_a_section_is_refused_not_crashed() -> None:
    assert head.check_head_manifest({}, _pins()) == ["manifest is missing the object 'frozen'"]
    assert hashlib.sha256(b"").hexdigest()  # keep hashlib honest for the digests above


# -- the network (torch) ---------------------------------------------------------------
DIM, CLASSES_N = 8, 3


class FakeReader:
    """Per-residue states held in memory, read by hash like the real cache."""

    def __init__(self, states: dict[str, Any]) -> None:
        """Keep the states."""
        self._states = states

    def states(self, sha256: str) -> Any:
        """The (length, dim) states of one sequence."""
        return self._states[sha256]


def _data(np: Any, n: int = 64) -> tuple[FakeReader, head.Rows]:
    """A separable toy problem: every residue of a positive row carries its class signal."""
    rng = np.random.default_rng(0)
    shas = [f"s{i:03d}" for i in range(n)]
    y = (rng.random((n, CLASSES_N)) < 0.4).astype(int)
    states = {}
    for i, sha in enumerate(shas):
        length = int(rng.integers(5, 13))
        array = rng.normal(0.0, 0.3, (length, DIM))
        array[:, :CLASSES_N] += 2.0 * (y[i] * 2 - 1)
        states[sha] = array.astype(np.float16)
    partition = np.array([1 + (i % 4) for i in range(n)])
    rows = head.Rows(shas=shas, y=y, partition=partition, cluster_code=partition - 1)
    return FakeReader(states), rows


def _settings(net_module: Any, **overrides: Any) -> Any:
    arch = head.Architecture(heads=1, attention_hidden=4, embedding_dim=DIM)
    values: dict[str, Any] = {
        "net": net_module.Net(arch, hidden=8, dropout=0.0, classes=CLASSES_N),
        "gamma": 0.0,
        "learning_rate": 0.05,
        "weight_decay": 0.01,
        "schedule": head.Schedule(
            batch_size=8, max_epochs=40, patience=6, inner_holdout_fraction=0.25
        ),
        "seed": 7,
    }
    values.update(overrides)
    return net_module.FitSettings(**values)


def _batch(np: Any, torch: Any, lengths: list[int], seed: int = 1) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    widest = max(lengths)
    x = np.zeros((len(lengths), widest, DIM), dtype=np.float32)
    mask = np.zeros((len(lengths), widest), dtype=bool)
    for i, n in enumerate(lengths):
        x[i, :n] = rng.normal(0.0, 1.0, (n, DIM))
        mask[i, :n] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


def test_shapes_and_attention_weights_sum_to_one_with_zero_padding(np: Any, torch: Any) -> None:

    settings = _settings(net)
    model = net.build_model(settings.net, 3)
    x, mask = _batch(np, torch, [5, 9, 7])
    with torch.no_grad():
        logits, attention = model.forward(x, mask)
    assert tuple(logits.shape) == (3, CLASSES_N)
    assert tuple(attention.shape) == (3, 1, 9)
    sums = attention.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert float(attention[0, 0, 5:].abs().max()) == 0.0  # padded positions are exactly 0
    assert float(attention[2, 0, 7:].abs().max()) == 0.0


def test_padding_never_changes_the_output(np: Any, torch: Any) -> None:

    model = net.build_model(_settings(net).net, 3)
    x, mask = _batch(np, torch, [5, 9])
    alone = x[:1, :5].clone()
    with torch.no_grad():
        batched, _ = model.forward(x, mask)
        solo, _ = model.forward(alone, torch.ones(1, 5, dtype=torch.bool))
        noisy = x.clone()
        noisy[0, 5:] = 1000.0  # garbage in the padded slots
        garbage, _ = model.forward(noisy, mask)
    assert torch.allclose(batched[0], solo[0], atol=1e-5)
    assert torch.allclose(batched, garbage, atol=1e-5)


def test_forward_is_deterministic_and_depends_on_the_seed(np: Any, torch: Any) -> None:

    settings = _settings(net)
    x, mask = _batch(np, torch, [6, 8])
    outputs = []
    for seed in (3, 3, 4):
        model = net.build_model(settings.net, seed)
        with torch.no_grad():
            outputs.append(model.forward(x, mask)[0])
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])


def test_trainable_parameter_count_of_the_configured_architecture(torch: Any) -> None:
    config = head.load_config(HEAD_YAML)
    model = net.AttentionHead(
        net.Net(config.architecture, hidden=256, dropout=0.3, classes=10), seed=1
    )
    # attention 41,024 + 65, layer norm 1,280, fc1 164,096, fc2 2,570
    assert model.parameter_count() == 41024 + 65 + 1280 + 164096 + 2570 == 209035


def test_the_loss_matches_a_hand_computation(torch: Any) -> None:

    logits = torch.zeros(1, 1)
    weight = torch.tensor([3.0])
    positive = net.weighted_focal_loss(logits, torch.ones(1, 1), weight, 0.0)
    assert float(positive) == pytest.approx(3 * math.log(2), abs=1e-6)  # 2.0794
    focal = net.weighted_focal_loss(logits, torch.ones(1, 1), weight, 2.0)
    assert float(focal) == pytest.approx(0.25 * 3 * math.log(2), abs=1e-6)  # 0.5199
    negative = net.weighted_focal_loss(logits, torch.zeros(1, 1), weight, 0.0)
    assert float(negative) == pytest.approx(math.log(2), abs=1e-6)  # pos_weight is for positives


def test_gamma_zero_is_binary_cross_entropy_with_pos_weight(torch: Any) -> None:

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 3, generator=generator)
    target = (torch.rand(6, 3, generator=generator) < 0.5).float()
    weight = torch.tensor([1.5, 2.0, 0.5])
    ours = net.weighted_focal_loss(logits, target, weight, 0.0)
    reference = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=weight
    )
    assert float(ours) == pytest.approx(float(reference), abs=1e-6)


def test_predict_restores_order_and_returns_normalised_attention(np: Any, torch: Any) -> None:

    reader, rows = _data(np, 12)
    model = net.build_model(_settings(net).net, 3)
    shas = list(rows.shas)
    scores, attention = net.predict(model, reader, shas, 4, attention=True)
    single, _ = net.predict(model, reader, [shas[5]], 1)
    assert np.allclose(scores[5], single[0], atol=1e-5)
    assert attention is not None
    assert len(attention) == 12
    for sha, weights in zip(shas, attention, strict=True):
        assert weights.shape == (1, reader.states(sha).shape[0])
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert net.predict(model, reader, shas, 4)[1] is None


def test_a_fit_learns_a_separable_problem_and_is_reproducible(np: Any, torch: Any) -> None:

    reader, rows = _data(np)
    source = net.Source(reader, rows)
    train = [i for i in range(len(rows)) if int(rows.partition[i]) != 4]
    held = [i for i in range(len(rows)) if int(rows.partition[i]) == 4]
    settings = _settings(net)
    first = net.fit_head(source, train, held, settings)
    again = net.fit_head(source, train, held, settings)
    assert first.hold_macro_f1 is not None
    assert first.hold_macro_f1 > 0.9
    assert 1 <= first.best_epoch <= first.epochs_run <= settings.schedule.max_epochs
    assert first.best_epoch == again.best_epoch
    for name, array in first.state.items():
        assert np.array_equal(array, again.state[name])
    model = net.build_model(settings.net, settings.seed, first.state)
    scores, _ = net.predict(model, reader, [rows.shas[i] for i in held], 8)
    assert head.held_macro_f1(scores, rows.y[held]) > 0.9


def test_a_fit_needs_exactly_one_of_a_holdout_or_an_epoch_count(np: Any, torch: Any) -> None:

    reader, rows = _data(np, 16)
    source = net.Source(reader, rows)
    with pytest.raises(BaselineError):
        net.fit_head(source, [0, 1, 2], None, _settings(net))
    with pytest.raises(BaselineError):
        net.fit_head(source, [0, 1, 2], [3, 4], _settings(net), epochs=2)
    fixed = net.fit_head(source, list(range(12)), None, _settings(net), epochs=3)
    assert fixed.epochs_run == fixed.best_epoch == 3
    assert fixed.hold_macro_f1 is None


def test_the_loss_weights_see_only_the_fit_rows(
    monkeypatch: pytest.MonkeyPatch, np: Any, torch: Any
) -> None:

    reader, rows = _data(np, 24)
    seen: list[list[list[int]]] = []
    real = net.pos_weights

    def spy(labels: Any) -> Any:
        seen.append([list(r) for r in labels])
        return real(labels)

    monkeypatch.setattr(net, "pos_weights", spy)
    fit_idx = [0, 2, 4, 6, 8, 10]
    net.fit_head(net.Source(reader, rows), fit_idx, None, _settings(net), epochs=1)
    assert seen == [[list(map(int, rows.y[i])) for i in fit_idx]]


def test_a_cluster_shared_across_partitions_stops_the_fit_before_it_starts(
    np: Any, torch: Any
) -> None:

    reader, rows = _data(np, 32)
    leaking = rows.cluster_code.copy()
    leaking[1] = leaking[0]  # row 0 is in partition 1, row 1 in partition 2: one cluster
    source = net.Source(reader, head.Rows(rows.shas, rows.y, rows.partition, leaking))
    with pytest.raises(BaselineError, match="cluster leakage"):
        net.run_grid_point(source, lopo_folds(), _settings(net), set(), lambda _m: None)


def test_a_test_cluster_in_the_training_rows_stops_the_refit(np: Any, torch: Any) -> None:

    reader, rows = _data(np, 16)
    source = net.Source(reader, rows)
    with pytest.raises(BaselineError, match="'train' and 'test'"):
        net.refit_final(source, _settings(net), 2, {int(rows.cluster_code[0])})


def test_a_grid_point_scores_every_validation_row_out_of_fold(np: Any, torch: Any) -> None:

    reader, rows = _data(np)
    source = net.Source(reader, rows)
    oof, runs = net.run_grid_point(source, lopo_folds(), _settings(net), {999}, lambda _m: None)
    assert oof.shape == rows.y.shape
    assert (oof > 0).all()  # every row was scored by some fold
    assert (oof < 1).all()
    assert [r.validate for r in runs] == [1, 2, 3, 4]
    assert all(r.holdout_rows > 0 and r.fit_rows > 0 for r in runs)
    assert head.mean_fold_macro_f1(oof, rows, lopo_folds(), ["a", "b", "c"]) > 0.8
