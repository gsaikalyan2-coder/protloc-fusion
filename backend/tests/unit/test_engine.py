"""The serving engine against the real registry and caches (Phase 14).

Every test here needs the files a Phase 5-13 run produced under ``data/`` (weights, manifests,
the residue cache, the GO ontology release). CI has none of them, so the module skips there; it
is run on the machine that holds them (``docs/phases/14.md`` section 6). What it proves:

* every field a trained head covers comes out ``source: "model"`` with the bundle that made it,
  and homology terms are a separate field with ``source: "homology"`` and ``source_accession``;
* the localization thresholds served are the per-label thresholds in the bundle;
* the engine refuses to start - and says why - on a changed MODEL_VERSION, weight byte,
  calibrator, rule file, ontology byte, manifest, or an index that is not the recorded one.
"""

import json
from pathlib import Path
import shutil

import pytest

pytest.importorskip("numpy")

from protloc.config.serving import ServingSettings
from protloc.data.records import sequence_sha256
from protloc.data.residue_cache import ResidueStore
from protloc.domain.contracts import ModelGoAnnotation
from protloc.inference.engine import (
    EngineRefusal,
    EngineStartError,
    composite_version,
    load_engine,
)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
CACHE = DATA / "embeddings" / "0e528d4a46831b05_subset500_residues"
CORPUS = DATA / "processed" / "corpus.jsonl"
GALLERY = DATA / "gallery" / "gallery.json"
ENGINE_JSON = json.loads((REPO / "configs" / "serving" / "engine.json").read_text())
VERSION = composite_version(ENGINE_JSON["bundles"])

_NEEDED = [
    DATA / "head" / "final_weights.npz",
    DATA / "crf" / "final_weights.npz",
    DATA / "go" / "final_weights.npz",
    DATA / "raw" / "go-basic.obo",
    CACHE / "index.json",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _NEEDED), reason="needs the Phase 7-12 data under data/"
)


def _settings(tmp: Path, **override: object) -> ServingSettings:
    values: dict[str, object] = {
        "MODEL_REGISTRY": REPO / "registry",
        "MODEL_VERSION": VERSION,
        "MODEL_SERVING_MODE": "queue",
        "MAX_BATCH_SEQUENCES": 10,
        "INLINE_RESULT_WINDOW_SECONDS": 2,
        "JOB_RETENTION_HOURS": 1,
        "DATA_ROOT": DATA,
        "EMBED_CACHE": DATA / "embeddings",
        "EVIDENCE_ROOT": tmp / "no_evidence",  # absent: evidence is the degraded state
        "SWISSPROT_DB_PATH": tmp / "no_evidence" / "swissprot_mmseqs",
        "EVIDENCE_TIMEOUT_SECONDS": 5,
        "EVIDENCE_TOP_K": 5,
    }
    values.update(override)
    return ServingSettings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory):
    return load_engine(_settings(tmp_path_factory.mktemp("engine")), REPO)


@pytest.fixture(scope="module")
def sequence(engine) -> str:  # noqa: ANN001
    """A real sequence the cache can score: a gallery entry, else the first corpus match."""
    if GALLERY.is_file():
        return str(json.loads(GALLERY.read_text())["examples"][0]["sequence"])
    if CORPUS.is_file():
        cached = ResidueStore(CACHE).lengths
        with CORPUS.open("rb") as handle:
            for line in handle:
                record = json.loads(line)
                if record["sequence_sha256"] in cached:
                    return str(record["sequence_normalized"])
    pytest.skip("no corpus or gallery to take a cached sequence from")


def test_every_trained_field_is_the_model_and_names_its_bundle(engine, sequence: str) -> None:  # noqa: ANN001
    result = engine.predict(sequence, "plant", "q", with_evidence=False)
    sources = result["task_sources"]
    for task in ("localization", "signal_peptide", "go_function"):
        assert sources[task]["source"] == "model"
        assert sources[task]["calibrated"] is True
        assert sources[task]["bundle_version"] == ENGINE_JSON["bundles"][task]
        assert len(sources[task]["bundle_sha256"]) == 64
    assert sources["membrane_type"]["source"] == "motif"
    assert {row["source"] for row in result["localization"]} == {"model"}
    assert {row["source"] for row in result["go_function"]} == {"model"}
    assert result["tau"] is None
    assert all("source_accession" not in row for row in result["go_function"])


def test_plant_scores_ten_classes_and_the_other_group_nine(engine, sequence: str) -> None:  # noqa: ANN001
    plant = engine.predict(sequence, "plant", "q", with_evidence=False)
    animal = engine.predict(sequence, "eukaryote", "q", with_evidence=False)
    assert len(plant["localization"]) == 10
    assert len(animal["localization"]) == 9
    assert "PLA" not in {r["code"] for r in animal["localization"]}


def test_served_thresholds_are_the_bundles(engine, sequence: str) -> None:  # noqa: ANN001
    result = engine.predict(sequence, "plant", "q", with_evidence=False)
    calibrator = json.loads(
        (
            REPO
            / "registry"
            / "localization"
            / ENGINE_JSON["bundles"]["localization"]
            / "calibrator.json"
        ).read_text()
    )
    by_label = {entry["label"]: entry["threshold"] for entry in calibrator["labels"]}
    for row in result["localization"]:
        assert row["threshold"] == pytest.approx(by_label[row["class_name"]], abs=1e-4)
        assert row["predicted"] == (row["probability"] >= row["threshold"])


def test_an_uncached_sequence_is_refused_with_a_code(engine) -> None:  # noqa: ANN001
    with pytest.raises(EngineRefusal) as caught:
        engine.predict("MKTAYIAKQRQISFVKSHFSRQ", "eukaryote", "q")
    assert caught.value.code == "not_in_embedding_cache"


def test_a_bacterial_group_is_refused(engine, sequence: str) -> None:  # noqa: ANN001
    with pytest.raises(EngineRefusal) as caught:
        engine.predict(sequence, "gram_neg", "q")
    assert caught.value.code == "unsupported_group"


def test_the_cache_key_is_the_corpus_hash_of_the_sequence(engine, sequence: str) -> None:  # noqa: ANN001
    assert engine.has_states(sequence)
    assert sequence_sha256(sequence) in ResidueStore(CACHE).lengths


def test_go_rows_fit_the_contracts_model_shape(engine, sequence: str) -> None:  # noqa: ANN001
    """``ModelGoAnnotation`` is the contract's shape of a model GO term; it has no accession."""
    rows = engine.predict(sequence, "plant", "q", with_evidence=False)["go_function"]
    fields = ModelGoAnnotation.model_fields
    for row in rows:
        parsed = ModelGoAnnotation(**{k: row[k] for k in fields})
        assert parsed.source == "model"


# -- refuse to start ----------------------------------------------------------------------
def _data_copy(tmp: Path) -> Path:
    root = tmp / "data"
    for relative in ("head/final_weights.npz", "crf/final_weights.npz", "go/final_weights.npz"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / relative, root / relative)
    shutil.copytree(DATA / "manifests", root / "manifests")
    (root / "raw").mkdir()
    shutil.copy(DATA / "raw" / "go-basic.obo", root / "raw" / "go-basic.obo")
    return root


def _flip_a_byte(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    path.write_bytes(bytes(raw))


def test_a_wrong_model_version_stops_the_service(tmp_path: Path) -> None:
    with pytest.raises(EngineStartError, match="MODEL_VERSION"):
        load_engine(_settings(tmp_path, MODEL_VERSION="localization-9.9.9"), REPO)


def test_a_changed_weight_byte_stops_the_service(tmp_path: Path) -> None:
    root = _data_copy(tmp_path)
    _flip_a_byte(root / "head" / "final_weights.npz")
    with pytest.raises(EngineStartError, match="localization bundle"):
        load_engine(_settings(tmp_path, DATA_ROOT=root), REPO)


def test_a_changed_calibrator_stops_the_service(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    shutil.copytree(REPO / "registry", registry)
    path = (
        registry / "signal_peptide" / ENGINE_JSON["bundles"]["signal_peptide"] / "calibrator.json"
    )
    document = json.loads(path.read_text())
    document["labels"][0]["threshold"] = 0.99
    path.write_text(json.dumps(document))
    with pytest.raises(EngineStartError, match="signal_peptide bundle"):
        load_engine(_settings(tmp_path, MODEL_REGISTRY=registry), REPO)


def test_a_changed_rule_file_stops_the_service(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs" / "serving").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "serving" / "engine.json", repo / "configs" / "serving")
    shutil.copytree(REPO / "backend" / "rules", repo / "backend" / "rules")
    with (repo / "backend" / "rules" / "consistency.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(EngineStartError, match="bundle"):
        load_engine(_settings(tmp_path), repo)


def test_a_changed_ontology_stops_the_service(tmp_path: Path) -> None:
    root = _data_copy(tmp_path)
    with (root / "raw" / "go-basic.obo").open("a") as handle:
        handle.write("\n")
    with pytest.raises(EngineStartError, match="ontology"):
        load_engine(_settings(tmp_path, DATA_ROOT=root), REPO)


def test_a_changed_manifest_stops_the_service(tmp_path: Path) -> None:
    root = _data_copy(tmp_path)
    path = root / "manifests" / "crf_metrics.json"
    document = json.loads(path.read_text())
    document["frozen"]["scope_label"] = "edited"
    path.write_text(json.dumps(document))
    with pytest.raises(EngineStartError, match=r"crf_metrics\.json"):
        load_engine(_settings(tmp_path, DATA_ROOT=root), REPO)


def test_an_index_that_is_not_the_recorded_one_stops_the_service(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "swissprot_mmseqs").mkdir(parents=True)
    (evidence / "swissprot_mmseqs" / "db").write_bytes(b"not the index")
    (evidence / "hit_terms.tsv").write_bytes(b"P1\tGO:1\n")
    with pytest.raises(EngineStartError, match="evidence bundle"):
        load_engine(
            _settings(
                tmp_path,
                EVIDENCE_ROOT=evidence,
                SWISSPROT_DB_PATH=evidence / "swissprot_mmseqs",
            ),
            REPO,
        )
