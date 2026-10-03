"""The evaluation config and the split registry number the partitions the same way.

Phase 8 found ``configs/evaluation/default.yaml`` counting partitions from zero
(validation ``[0, 1, 2, 3]``, test ``[4]``) while the registry numbers them 1 to 5, and
recorded the mapping n = index + 1 in ADR-0010. Phase 9 fixed the file. One off-by-one
here would put the wrong partition under the test label, so these tests fail the moment
the config, the committed split manifest, the registry or the protocol constants disagree.
"""

import csv
import json
from pathlib import Path

import pytest
import yaml

from protloc.training.baselines import TEST_PARTITION, VALIDATION_PARTITIONS

ROOT = Path(__file__).resolve().parents[3]
EVALUATION = ROOT / "configs" / "evaluation" / "default.yaml"
SPLIT_MANIFEST = ROOT / "data" / "manifests" / "split_manifest.json"
REGISTRY = ROOT / "data" / "processed" / "splits" / "registry.tsv"


def _config_partitions() -> tuple[list[int], list[int]]:
    document = yaml.safe_load(EVALUATION.read_text(encoding="utf-8"))
    block = document["partitions"]
    return list(block["validation"]), list(block["test"])


def _registry_numbers() -> set[int]:
    count = int(json.loads(SPLIT_MANIFEST.read_bytes())["partitions"])
    return set(range(1, count + 1))


def test_the_config_uses_registry_numbering_not_zero_based_indices() -> None:
    validation, test = _config_partitions()
    assert validation == [1, 2, 3, 4]
    assert test == [5]
    assert 0 not in {*validation, *test}


def test_the_config_equals_the_protocol_constants() -> None:
    validation, test = _config_partitions()
    assert tuple(validation) == VALIDATION_PARTITIONS
    assert test == [TEST_PARTITION]


def test_the_config_covers_every_registry_partition_exactly_once() -> None:
    validation, test = _config_partitions()
    assert len(set(validation)) == len(validation)
    assert not set(validation) & set(test)
    assert {*validation, *test} == _registry_numbers()


@pytest.mark.skipif(not REGISTRY.exists(), reason="the registry bytes are gitignored")
def test_the_registry_file_numbers_partitions_as_the_config_does() -> None:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        numbers = {int(row["partition"]) for row in csv.DictReader(handle, delimiter="\t")}
    validation, test = _config_partitions()
    assert numbers == {*validation, *test}
