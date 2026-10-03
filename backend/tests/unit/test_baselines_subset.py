"""Phase 9 additions to the baselines: undefined classes and the 500-sequence subset.

Toy data appears only here. Every expected number is worked out by hand in a comment.
"""

from pathlib import Path
from typing import Any

import pytest

from protloc.training import baseline_models
from protloc.training import baselines as bl

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.fixture(scope="module")
def models() -> Any:
    pytest.importorskip("numpy")
    return baseline_models


# -- undefined classes -----------------------------------------------------------------
def test_a_class_with_one_positive_is_marked_undefined_and_has_mcc_zero(models: Any) -> None:
    np = models._np()
    # Class a has ONE positive and it is predicted: the raw MCC would be 1.0, but a class
    # with fewer than two positives has no defined MCC, so it reports 0.0 and is flagged.
    # Class b has two positives, both predicted, no false positives: MCC 1.0, defined.
    y = np.array([[1, 1], [0, 1], [0, 0], [0, 0]])
    p = y.copy()
    report = models.metrics_report(y, p, np.arange(4), ["a", "b"], models.Bootstrap(0, 1))
    assert report["per_class"]["a"]["support"] == 1
    assert report["per_class"]["a"]["undefined"] is True
    assert report["per_class"]["a"]["mcc"]["value"] == 0.0
    assert report["per_class"]["b"]["undefined"] is False
    assert report["per_class"]["b"]["mcc"]["value"] == pytest.approx(1.0)


def test_an_undefined_class_still_counts_in_macro_f1(models: Any) -> None:
    np = models._np()
    # Class a: TP 1, FP 0, FN 0 -> F1 1.0. Class b: TP 2 -> F1 1.0. Macro F1 stays 1.0:
    # marking a class undefined changes its MCC only, never its place in macro F1.
    y = np.array([[1, 1], [0, 1], [0, 0], [0, 0]])
    report = models.metrics_report(y, y.copy(), np.arange(4), ["a", "b"], models.Bootstrap(0, 1))
    assert report["macro_f1"]["value"] == pytest.approx(1.0)


def test_every_fold_reports_every_class_and_marks_the_thin_ones(models: Any) -> None:
    np = models._np()
    # 4 partitions x 4 rows. Class a: one positive, in partition 1 only. Class b: two
    # positives in every partition. So class a is undefined in every fold (support 1, 0, 0, 0)
    # and class b is defined in every fold (support 2).
    partition = np.repeat([1, 2, 3, 4], 4)
    y = np.zeros((16, 2), dtype=int)
    y[0, 0] = 1
    for start in (0, 4, 8, 12):
        y[start : start + 2, 1] = 1
    scores = y.astype(float)
    split = models.Split(y, partition, np.arange(16))
    result = models.validate_oof(scores, split, bl.lopo_folds(), ["a", "b"], models.Bootstrap(0, 1))
    assert [fold["per_class"]["a"]["support"] for fold in result["folds"]] == [1, 0, 0, 0]
    assert all(fold["per_class"]["a"]["undefined"] for fold in result["folds"])
    assert all(fold["per_class"]["a"]["mcc"] == 0.0 for fold in result["folds"])
    assert not any(fold["per_class"]["b"]["undefined"] for fold in result["folds"])
    assert all(set(fold["per_class"]) == {"a", "b"} for fold in result["folds"])


# -- the subset -------------------------------------------------------------------------
def _table(models: Any) -> Any:
    # Six accessions over three distinct sequences (X rows 0, 1, 2). The first feature column
    # is log1p(length), which is how a sequence is checked against the cache.
    np = models._np()
    names = bl.FEATURE_NAMES
    x = np.zeros((3, len(names)))
    column = names.index("log1p_length")
    x[:, column] = np.log1p(np.array([10.0, 20.0, 30.0]))
    raw = models.RawTable(
        accessions=["A", "B", "C", "D", "E", "F"],
        partitions=[1, 2, 3, 4, 5, 1],
        clusters=["c1", "c2", "c3", "c4", "c5", "c6"],
        seq_index=[0, 0, 1, 1, 2, 2],
        x_rows=x.tolist(),
        rules={"t": [[0.0], [0.0], [0.0]]},
        labels={"t": [[1], [0], [1], None, [1], [0]]},
        widths={"t": 1},
    )
    return models.assemble(raw)


def test_restricting_masks_every_row_outside_the_subset(models: Any) -> None:
    np = models._np()
    table = _table(models)
    restricted = models.restrict_to_sequences(table, np.array([1]))
    # Only sequence 1 (accessions C and D) stays; D has no label so only C is usable.
    assert list(restricted["M_t"]) == [False, False, True, False, False, False]
    assert list(models.task_rows(restricted, "t", (1, 2, 3, 4), 1)) == [2]
    # The input is not modified, and the arrays other than the masks are the same objects.
    assert list(table["M_t"]) == [True, True, True, False, True, True]
    assert restricted["X"] is table["X"]
    assert list(models.task_rows(restricted, "t", (5,), 1)) == []


def test_the_test_partition_stays_the_test_partition_after_restricting(models: Any) -> None:
    np = models._np()
    table = _table(models)
    restricted = models.restrict_to_sequences(table, np.array([2]))
    # Sequence 2 holds E (partition 5) and F (partition 1).
    assert list(models.task_rows(restricted, "t", (5,), 1)) == [4]
    assert list(models.task_rows(restricted, "t", (1, 2, 3, 4), 1)) == [5]


def test_locating_cached_hashes_is_proved_by_length(models: Any) -> None:
    np = models._np()
    table = _table(models)
    order = [SHA_A, SHA_B, SHA_C]
    found = models.locate_sequences(table, order, {SHA_C: 30, SHA_A: 10})
    assert list(found) == [0, 2]
    assert found.dtype == np.int64


def test_a_pack_that_disagrees_with_the_table_is_refused(models: Any) -> None:
    table = _table(models)
    order = [SHA_A, SHA_B, SHA_C]
    with pytest.raises(bl.BaselineError, match="lengths differ"):
        models.locate_sequences(table, order, {SHA_A: 11})
    with pytest.raises(bl.BaselineError, match="not in the pack"):
        models.locate_sequences(table, order, {SHA_D: 10})
    with pytest.raises(bl.BaselineError, match="feature table has 3"):
        models.locate_sequences(table, [SHA_A, SHA_B], {SHA_A: 10})


def _write_shard(directory: Path, name: str, rows: list[tuple[str, int, int]]) -> None:
    lines = ["sha256\tlength\twindows"] + [f"{sha}\t{n}\t{w}" for sha, n, w in rows]
    (directory / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="ascii")


def test_cache_lengths_reads_every_shard_table(tmp_path: Path) -> None:
    _write_shard(tmp_path, "shard_00000", [(SHA_A, 10, 1), (SHA_B, 20, 1)])
    _write_shard(tmp_path, "shard_00001", [(SHA_C, 30, 2)])
    assert bl.cache_lengths(tmp_path) == {SHA_A: 10, SHA_B: 20, SHA_C: 30}


def test_cache_lengths_refuses_an_empty_directory_and_a_bad_hash(tmp_path: Path) -> None:
    with pytest.raises(bl.BaselineError, match="no committed shard tables"):
        bl.cache_lengths(tmp_path)
    _write_shard(tmp_path, "shard_00000", [("not-a-hash", 10, 1)])
    with pytest.raises(bl.BaselineError, match="not a sequence hash"):
        bl.cache_lengths(tmp_path)


def test_the_subset_digest_ignores_order_and_changes_with_content() -> None:
    assert bl.subset_digest([SHA_A, SHA_B]) == bl.subset_digest([SHA_B, SHA_A])
    assert bl.subset_digest([SHA_A, SHA_B]) != bl.subset_digest([SHA_A, SHA_C])


def _frozen(cache_key: str) -> dict[str, Any]:
    return {
        "scope": bl.SUBSET_SCOPE,
        "scope_label": bl.SUBSET_LABEL,
        "subset": {
            "sequences": 500,
            "sequences_sha256": SHA_A,
            "embedding_cache_key": cache_key,
        },
    }


def test_a_sound_subset_block_has_no_problems() -> None:
    assert bl.check_subset_block(_frozen(SHA_B), cache_key=SHA_B) == []


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda f: f.update(scope="full"), "scope 'subset'"),
        (lambda f: f.update(scope_label="subset"), "required subset label"),
        (lambda f: f["subset"].update(embedding_cache_key=SHA_C), "embedding_cache_key"),
        (lambda f: f["subset"].update(sequences_sha256="x"), "sequences_sha256"),
        (lambda f: f["subset"].update(sequences=0), "positive count"),
    ],
)
def test_each_subset_disagreement_is_reported(edit: Any, message: str) -> None:
    frozen = _frozen(SHA_B)
    edit(frozen)
    problems = bl.check_subset_block(frozen, cache_key=SHA_B)
    assert any(message in problem for problem in problems)


def test_a_missing_subset_object_is_reported() -> None:
    frozen = _frozen(SHA_B)
    del frozen["subset"]
    assert any("'subset'" in problem for problem in bl.check_subset_block(frozen, cache_key=SHA_B))


def test_the_label_names_the_caveat() -> None:
    assert bl.SUBSET_LABEL == (
        "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"
    )
