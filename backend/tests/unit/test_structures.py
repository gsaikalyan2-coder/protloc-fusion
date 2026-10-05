"""The structure store serves only files that match the manifest."""

import hashlib
import json
from pathlib import Path

import pytest

from protloc.inference.structures import StructureError, StructureStore, is_sha256

DIGEST = "ab" * 32
DOC = {
    "accession": "P12345",
    "length": 2,
    "ca": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
    "plddt": [90.0, 80.0],
}


def _write(root: Path, document: dict[str, object], manifest_digest: str | None = None) -> None:
    raw = (json.dumps(document) + "\n").encode()
    (root / f"{DIGEST}.json").write_bytes(raw)
    sha = manifest_digest or hashlib.sha256(raw).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"files_sha256": {DIGEST: sha}}))


def test_is_sha256() -> None:
    assert is_sha256(DIGEST)
    assert not is_sha256("../etc/passwd")
    assert not is_sha256(DIGEST.upper())
    assert not is_sha256(DIGEST[:-1])


def test_serves_a_file_that_matches_the_manifest(tmp_path: Path) -> None:
    _write(tmp_path, DOC)
    store = StructureStore(tmp_path)
    assert store.count == 1
    assert store.get(DIGEST) == DOC


def test_unknown_hash_is_none(tmp_path: Path) -> None:
    _write(tmp_path, DOC)
    assert StructureStore(tmp_path).get("cd" * 32) is None


def test_missing_manifest_serves_nothing(tmp_path: Path) -> None:
    (tmp_path / f"{DIGEST}.json").write_text("{}")
    store = StructureStore(tmp_path)
    assert store.count == 0
    assert store.get(DIGEST) is None


def test_edited_file_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, DOC, manifest_digest="00" * 32)
    with pytest.raises(StructureError):
        StructureStore(tmp_path).get(DIGEST)


def test_malformed_hash_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        StructureStore(tmp_path).get("../manifest")
