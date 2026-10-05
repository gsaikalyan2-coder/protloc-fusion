"""``configs/serving/engine.json`` against the files it must agree with (Phase 14).

The serving engine repeats no number the training side already decided. These tests are what
make that true: they fail if the pinned class codes, tag set or forbidden transition score drift
from the training configs, or if a pinned bundle version has no directory in the registry.
No data and no torch are needed, so they run in CI.
"""

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from protloc.inference.engine import composite_version

REPO = Path(__file__).resolve().parents[3]
ENGINE = json.loads((REPO / "configs" / "serving" / "engine.json").read_text())
CRF = yaml.safe_load((REPO / "configs" / "training" / "crf.yaml").read_text())
CLASS_MAP = yaml.safe_load((REPO / "backend" / "rules" / "baseline_class_map.yaml").read_text())
GO = yaml.safe_load((REPO / "configs" / "training" / "go.yaml").read_text())


def test_codes_cover_the_localization_classes_in_the_head_order() -> None:
    classes = CLASS_MAP["tasks"]["localization"]["classes"]
    assert list(ENGINE["localization"]["codes"]) == classes


def test_the_tag_set_and_forbidden_transitions_are_the_training_ones() -> None:
    tags = CRF["tags"]
    assert ENGINE["crf"]["tags"] == tags["names"]
    assert ENGINE["crf"]["outside"] == tags["outside"]
    assert ENGINE["crf"]["signal"] == tags["signal"]
    assert ENGINE["crf"]["forbidden_transitions"] == tags["forbidden_transitions"]
    assert ENGINE["crf"]["forbidden_score"] == tags["forbidden_score"]


def test_the_encoder_dimension_is_the_heads_dimension() -> None:
    assert ENGINE["encoder"]["dimension"] == CRF["architecture"]["embedding_dim"]


def test_the_ontology_pin_is_the_go_configs() -> None:
    assert ENGINE["ontology"]["file"] == GO["ontology"]["file"].removeprefix("data/")
    assert ENGINE["ontology"]["sha256"] == GO["ontology"]["sha256"]


@pytest.mark.parametrize("task", ["localization", "signal_peptide", "go_function"])
def test_every_pinned_bundle_has_a_registry_directory(task: str) -> None:
    version = ENGINE["bundles"][task]
    directory = REPO / "registry" / task / version
    assert (directory / "bundle.json").is_file()
    assert (directory / "calibrator.json").is_file()


def test_the_evidence_bundle_is_in_the_registry() -> None:
    version = ENGINE["bundles"]["evidence_homology"]
    assert (REPO / "registry" / "evidence_homology" / version / "bundle.json").is_file()


def test_the_composite_version_names_every_bundle_in_a_fixed_order() -> None:
    versions: dict[str, Any] = ENGINE["bundles"]
    assert composite_version(versions) == (
        "localization-0.10.0+signal_peptide-0.11.0+go_function-0.12.0+evidence_homology-0.13.0"
    )


def test_only_eukaryotic_groups_are_served() -> None:
    assert ENGINE["organism_groups_with_trained_heads"] == ["eukaryote", "plant"]


# -- evidence independence ------------------------------------------------------------------
_SCORING_STAGES = ("_score", "_localization", "_signal", "_go", "_task_source")
_EVIDENCE_NAMES = frozenset({"search", "homology", "Evidence", "_evidence", "evidence", "terms"})


def _names_used(function: ast.FunctionDef) -> set[str]:
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute)}
    return names


def test_no_scoring_stage_reads_evidence() -> None:
    """Evidence never feeds a model score, a calibrator or a threshold (ADR-0015).

    The stages that produce and calibrate scores must not mention the evidence module, the
    search or the hit-term table; evidence is attached after them, in ``_attach_evidence``.
    """
    source = (REPO / "backend" / "protloc" / "inference" / "engine.py").read_text()
    tree = ast.parse(source)
    engine = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Engine")
    stages = {n.name: n for n in engine.body if isinstance(n, ast.FunctionDef)}
    assert set(_SCORING_STAGES) <= set(stages)
    for name in _SCORING_STAGES:
        leaked = _names_used(stages[name]) & _EVIDENCE_NAMES
        assert not leaked, f"Engine.{name} mentions evidence: {sorted(leaked)}"


def test_the_pipeline_order_is_stated_once_in_predict_many() -> None:
    """Features, model and calibration (``_score``), then consistency, then evidence."""
    source = (REPO / "backend" / "protloc" / "inference" / "engine.py").read_text()
    body = source[source.index("def predict_many") : source.index("def predict(")]
    tokens = ("self._score(", "self._consistency(", "self._evidence(")
    positions = [body.index(token) for token in tokens]
    assert positions == sorted(positions)
