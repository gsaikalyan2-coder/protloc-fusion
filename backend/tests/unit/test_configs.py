"""The Hydra config tree composes, and its unfrozen values stay unfrozen.

A config tree that only gets exercised when a training run starts is a config
tree that is broken for a fortnight before anyone notices. These tests compose
it the way ``scripts/train.py`` will, and assert two things the project depends
on.

*Every group option composes.* A typo in a defaults list or an unparseable YAML
file fails here rather than four minutes into a GPU run.

*A value that Phase N produces is MISSING, not defaulted.* ``PROJECT_PLAN.md``
§7.2 schedules each frozen value; a plausible-looking placeholder in its place
is how an invented number reaches a paper. Reading one raises
``MissingMandatoryValue``, which is the behaviour being asserted.
"""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
import pytest

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# Group options that must exist and compose, by group.
GROUP_OPTIONS: dict[str, list[str]] = {
    "data": ["uniprot_swissprot"],
    "model": ["trunk_esm2_150m"],
    "training": ["calibration", "frozen_trunk", "head", "lora"],
    "evaluation": ["default"],
    "deployment": ["local", "hf_spaces"],
}

# Values produced by a later phase. Each must be MISSING, never defaulted. Phase 7 pinned
# model.trunk.revision (the HuggingFace commit hash), so none is scheduled right now.
SCHEDULED_VALUES: list[tuple[str, str]] = []

MANIFEST = CONFIG_DIR.parent / "data" / "manifests" / "dataset_manifest.json"
SPLIT_MANIFEST = CONFIG_DIR.parent / "data" / "manifests" / "split_manifest.json"


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def test_the_root_config_composes() -> None:
    cfg = _compose()
    for group in GROUP_OPTIONS:
        assert group in cfg, f"group '{group}' is missing from the composed config"


@pytest.mark.parametrize(
    ("group", "option"),
    [(group, option) for group, options in GROUP_OPTIONS.items() for option in options],
)
def test_every_group_option_composes(group: str, option: str) -> None:
    cfg = _compose(f"{group}={option}")
    assert cfg[group]["name"] == option


def test_every_yaml_file_is_reachable_from_a_group() -> None:
    """No orphan config file sits in the tree unreferenced."""
    on_disk = {
        f"{path.parent.name}={path.stem}"
        for path in CONFIG_DIR.rglob("*.yaml")
        if path.parent != CONFIG_DIR
    }
    declared = {
        f"{group}={option}" for group, options in GROUP_OPTIONS.items() for option in options
    }
    assert on_disk == declared, f"tree and test disagree: {sorted(on_disk ^ declared)}"


@pytest.mark.parametrize(("dotted_path", "owner"), SCHEDULED_VALUES)
def test_scheduled_values_are_missing_not_defaulted(dotted_path: str, owner: str) -> None:
    cfg = _compose()
    assert OmegaConf.is_missing(
        OmegaConf.select(cfg, dotted_path.rsplit(".", 1)[0]),
        dotted_path.rsplit(".", 1)[1],
    ), f"{dotted_path} has acquired a default; it is produced by {owner}"

    with pytest.raises(MissingMandatoryValue):
        OmegaConf.select(cfg, dotted_path, throw_on_missing=True)


def test_frozen_data_values_equal_the_manifest() -> None:
    """Phase 5 froze these; the manifest is the record and the config must agree."""
    manifest = json.loads(MANIFEST.read_bytes())
    cfg = _compose()
    assert cfg.data.source.release == manifest["snapshot"]["release"]
    assert cfg.data.sequence.max_length == manifest["policy"]["max_sequence_length"]
    assert cfg.data.sequence.alphabet == manifest["policy"]["alphabet"]
    assert cfg.data.labels.accepted_evidence_codes == manifest["build"]["accepted_evidence_codes"]


def test_the_trunk_is_pinned_to_a_real_revision() -> None:
    """Phase 7: the cache key needs the commit hash, and the dimension is measured."""
    trunk = _compose().model.trunk
    assert trunk.checkpoint == "facebook/esm2_t30_150M_UR50D"
    assert len(trunk.revision) == 40
    assert set(trunk.revision) <= set("0123456789abcdef")
    assert trunk.embedding_dim == 640
    assert trunk.max_residues == 1022
    assert trunk.long_sequence_policy == "sliding_window"


def test_the_seed_is_recorded() -> None:
    """A run that cannot state its seed cannot be repeated."""
    assert isinstance(_compose().seed, int)


def test_frozen_split_values_equal_the_split_manifest() -> None:
    """Phase 6 froze these; the split manifest is the record and the config must agree."""
    manifest = json.loads(SPLIT_MANIFEST.read_bytes())
    splits = _compose().data.splits
    assert splits.alignment_backend == manifest["alignment_backend"]
    assert splits.partitioner_version == manifest["partitioner_version"]
    assert splits.removal_rate == manifest["removal_rate"]
    assert splits.max_identity == manifest["identity_threshold"]
    assert splits.partitions == manifest["partitions"]


def test_the_serving_mode_is_queue_everywhere() -> None:
    """ADR-0001: 'queue' is the only supported mode, in every deployment."""
    for option in GROUP_OPTIONS["deployment"]:
        cfg = _compose(f"deployment={option}")
        assert cfg.deployment.serving.mode == "queue"


def test_the_split_settings_match_adr_0002() -> None:
    """30% maximum global identity, the mmseqs2needle backend, five partitions.

    Changing any of these means re-splitting and re-training, so the numbers are
    asserted against the ADR rather than trusted to stay put.
    """
    splits = _compose().data.splits
    assert splits.partitioner == "graphpart"
    assert splits.alignment_backend == "mmseqs2needle"
    assert splits.max_identity == 0.30
    assert splits.partitions == 5


def test_evaluation_scores_the_served_path() -> None:
    """The paper's numbers come from the API, not from a model object."""
    assert _compose().evaluation.target.mode == "api"
    assert _compose().evaluation.reporting.require_provenance is True


def test_test_partition_is_disjoint_from_validation() -> None:
    """Touching test to tune a value invalidates the result; overlap makes it easy."""
    partitions = _compose().evaluation.partitions
    assert not set(partitions.validation) & set(partitions.test)


def test_every_head_declares_the_groups_it_applies_to() -> None:
    """A head that does not apply to a group is masked, never scored and abstained."""
    heads = _compose().model.heads
    assert len(heads) == 7
    valid = set(_compose().data.organism_groups)
    for name, head in heads.items():
        assert head.applies_to, f"head '{name}' applies to no organism group"
        assert set(head.applies_to) <= valid, f"head '{name}' names an unknown group"
