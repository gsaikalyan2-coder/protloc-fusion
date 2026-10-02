"""The corpus rule files load, agree with the model config, and fail loudly.

The rule files in ``backend/rules/`` are corpus inputs (ADR-0006), and the
model config's ``applies_to`` lists decide what is masked at training time.
If the two disagreed, a task would be labelled for a group the model masks, or
the reverse, and neither mistake would show up anywhere else.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from protloc.data.rules import RuleFileError, TermMap, load_corpus_rules
from protloc.data.validate import nearest_rank
from protloc.domain.types import OrganismGroup, TaskName

RULES_DIR = Path(__file__).resolve().parents[2] / "rules"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def test_the_rule_files_load() -> None:
    rules = load_corpus_rules(RULES_DIR)
    assert [name for name, _ in rules.sha256] == [
        "backend/rules/organism_groups.yaml",
        "backend/rules/residue_policy.yaml",
        "backend/rules/uniprot_label_map.yaml",
    ]
    assert all(len(digest) == 64 for _, digest in rules.sha256)


def test_applicability_matches_the_model_config() -> None:
    rules = load_corpus_rules(RULES_DIR)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        heads = compose(config_name="config").model.heads
    for task, groups in rules.labels.applicability:
        declared = {OrganismGroup(g) for g in heads[task.value].applies_to}
        assert declared == groups, f"{task.value}: model config {declared} != rules {groups}"


def test_every_task_has_an_applicability_row() -> None:
    rules = load_corpus_rules(RULES_DIR)
    assert [task for task, _ in rules.labels.applicability] == list(TaskName)


def test_the_twenty_standard_residues_are_kept_and_j_is_excluded() -> None:
    residues = load_corpus_rules(RULES_DIR).residues
    standard = dict(residues.keep_groups)["standard"]
    assert len(standard) == 20
    assert "J" in residues.exclude_record
    assert not residues.keep & residues.exclude_record


def test_longest_prefix_wins() -> None:
    term_map = TermMap(
        classes=frozenset({"a", "b"}),
        terms=(("Cell", "a"), ("Cell outer membrane", "b")),
    )
    assert term_map.lookup("Cell outer membrane") == "b"
    assert term_map.lookup("Cell wall") == "a"
    assert term_map.lookup("Cellular") is None
    assert term_map.lookup("Cell, x") == "a"


def test_a_malformed_rule_file_is_refused(tmp_path: Path) -> None:
    for source in RULES_DIR.glob("*.yaml"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    target = tmp_path / "organism_groups.yaml"
    target.write_text("version: 1.0.0\nrules:\n  - any_of: [Eukaryota]\n", encoding="utf-8")
    with pytest.raises(RuleFileError, match="exactly one of"):
        load_corpus_rules(tmp_path)


def test_a_term_mapping_to_an_undeclared_class_is_refused(tmp_path: Path) -> None:
    for source in RULES_DIR.glob("*.yaml"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    target = tmp_path / "uniprot_label_map.yaml"
    text = target.read_text(encoding="utf-8").replace(
        "Cytosol: cytoplasm", "Cytosol: cytosol_not_declared"
    )
    target.write_text(text, encoding="utf-8")
    with pytest.raises(RuleFileError, match="undeclared classes"):
        load_corpus_rules(tmp_path)


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [(0.5, 3), (0.99, 5), (1.0, 5), (0.2, 1), (0.21, 2)],
)
def test_nearest_rank(quantile: float, expected: int) -> None:
    assert nearest_rank((1, 2, 3, 4, 5), quantile) == expected


def test_nearest_rank_refuses_a_bad_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        nearest_rank((1, 2), 0.0)
