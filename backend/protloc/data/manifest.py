"""The dataset manifest: the one file that says what the corpus is.

``data/manifests/dataset_manifest.json`` is authoritative (Phase 3 decision D4).
It has three sections, each written by the script that owns it:

* ``snapshot``, by ``download_data.py``: the release, the download date, and the
  hash of every downloaded byte;
* ``build``, by ``build_dataset.py``: the corpus hash, record and exclusion
  counts, and the hashes of the rule files;
* ``policy``, by ``validate_sequences.py --freeze``: the measured
  ``MAX_SEQUENCE_LENGTH`` and residue alphabet, and how each was produced.

The file itself is written deterministically, with sorted keys, two-space
indent, LF endings and a final newline, so that the same facts always give the
same bytes, and a diff in review shows only facts that changed.
"""

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1

_CHUNK = 1 << 20


class ManifestError(ValueError):
    """The manifest is missing, malformed, or lacks a section a step needs."""


def sha256_file(path: Path) -> tuple[str, int]:
    """SHA-256 and byte size of a file, read in chunks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def canonical_json_bytes(obj: object) -> bytes:
    """Deterministic, reviewable JSON bytes for a manifest-like document."""
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)
    return text.encode("ascii") + b"\n"


def write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as canonical JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(obj))


def read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON document whose top level must be an object."""
    obj = json.loads(path.read_bytes())
    if not isinstance(obj, dict):
        raise ManifestError(f"{path}: top level must be a JSON object")
    return {str(k): v for k, v in obj.items()}


def update_section(path: Path, section: str, body: dict[str, object]) -> dict[str, object]:
    """Replace one section of the manifest, keeping the others.

    Replacing ``snapshot`` or ``build`` drops any later section, because those
    were derived from what has just been replaced. ``policy`` measured on an old
    corpus is not a policy for the new one.
    """
    manifest: dict[str, object] = read_json_object(path) if path.exists() else {}
    manifest["schema_version"] = SCHEMA_VERSION
    downstream = {"snapshot": ("build", "policy"), "build": ("policy",), "policy": ()}
    if section not in downstream:
        raise ManifestError(f"unknown manifest section {section!r}")
    for stale in downstream[section]:
        manifest.pop(stale, None)
    manifest[section] = body
    write_json(path, manifest)
    return manifest


def section(manifest: dict[str, object], name: str) -> dict[str, object]:
    """One section of a manifest, or an error naming the script that writes it."""
    owners = {
        "snapshot": "scripts/download_data.py",
        "build": "scripts/build_dataset.py",
        "policy": "scripts/validate_sequences.py --freeze",
    }
    body = manifest.get(name)
    if not isinstance(body, dict):
        raise ManifestError(f"manifest has no '{name}' section; run {owners.get(name, '?')}")
    return {str(k): v for k, v in body.items()}


__all__ = [
    "SCHEMA_VERSION",
    "ManifestError",
    "canonical_json_bytes",
    "read_json_object",
    "section",
    "sha256_file",
    "update_section",
    "write_json",
]
