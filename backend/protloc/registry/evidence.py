"""The registry entry of the homology evidence layer, bound by hash and verified atomically.

Layer L3. A homology verdict is only reproducible if everything it came from is pinned together:
the Swiss-Prot snapshot, the MMseqs2 binary, the index built from that snapshot, the GO
annotations and ontology the transferred terms come from, the hit filter and the masks. This is
unlike a head's bundle (``registry/bundle.py``): there are no weights and no calibrator, and the
layer must never feed a model score. ``load_evidence_bundle`` recomputes every hash it can and
raises :class:`EvidenceBundleError` on any mismatch, so a changed index or a changed GO annotation
source is refused rather than served.

A bundle is a directory ``registry/evidence_homology/<version>/`` holding ``bundle.json``.
"""

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from protloc.calibration.calibrator import canonical_json_bytes, is_sha256, sha256_hex
from protloc.evidence.homology import MASKS, SOURCE_HOMOLOGY

SCHEMA_VERSION = 1
TASK = "homology_evidence"
BUNDLE_FILE = "bundle.json"
_CHUNK = 1 << 20
SECTIONS: tuple[str, ...] = (
    "database",
    "engine",
    "index",
    "go",
    "hit_filter",
    "masking",
    "transfer",
    "independence",
)
#: The layer is independent corroboration: each of these must be False in every bundle.
INDEPENDENCE_KEYS: tuple[str, ...] = (
    "feeds_model_score",
    "feeds_calibrator",
    "feeds_threshold",
)
SHA_KEYS: dict[str, tuple[str, ...]] = {
    "database": ("corpus_sha256", "fasta_sha256", "meta_sha256"),
    "engine": ("archive_sha256", "executable_sha256"),
    "index": ("index_sha256",),
    "go": ("sha256", "ontology_sha256", "vocabulary_sha256"),
}


class EvidenceBundleError(ValueError):
    """The evidence bundle does not verify; the service must not use it."""


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a multi-gigabyte index is never held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def index_files(directory: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    """Name, size and SHA-256 of each named file in the index directory, sorted by name."""
    out: list[dict[str, Any]] = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file():
            raise EvidenceBundleError(f"index file {name} is missing from {directory}")
        out.append({"name": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return out


def index_digest(files: Sequence[Mapping[str, Any]]) -> str:
    """One hash for the whole index: the canonical JSON of its sorted (name, bytes, sha256) list."""
    ordered = sorted(
        ({"name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in files),
        key=lambda f: str(f["name"]),
    )
    return sha256_hex(canonical_json_bytes(ordered))


def bundle_digest(document: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical bundle content, excluding the ``bundle_sha256`` field itself."""
    body = {key: value for key, value in document.items() if key != "bundle_sha256"}
    return sha256_hex(canonical_json_bytes(body))


def build_document(
    *, version: str, scope: Mapping[str, str], sections: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The ``bundle.json`` content: the sections of :data:`SECTIONS` plus the fixed fields."""
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": version,
        "task": TASK,
        "scope": scope["scope"],
        "scope_label": scope["scope_label"],
        "source": SOURCE_HOMOLOGY,
    }
    for name in SECTIONS:
        document[name] = dict(sections[name])
    document["bundle_sha256"] = bundle_digest(document)
    return document


def _section(document: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise EvidenceBundleError(f"bundle.json is missing the object {key!r}")
    return {str(k): v for k, v in value.items()}


def _section_problems(document: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for section, keys in SHA_KEYS.items():
        body = _section(document, section)
        found += [f"{section}.{k} is not a SHA-256" for k in keys if not is_sha256(body.get(k))]
    masking = _section(document, "masking")
    if masking.get("masks") != list(MASKS):
        found.append(f"masking.masks must be exactly {list(MASKS)}")
    independence = _section(document, "independence")
    found += [
        f"independence.{k} must be false"
        for k in INDEPENDENCE_KEYS
        if independence.get(k) is not False
    ]
    if document.get("source") != SOURCE_HOMOLOGY:
        found.append(f"source must be {SOURCE_HOMOLOGY!r}")
    return found


def _index_problems(document: Mapping[str, Any], directory: Path | None) -> list[str]:
    index = _section(document, "index")
    files = index.get("files")
    if not isinstance(files, list) or not files:
        return ["index.files is missing or empty"]
    listed = [dict(f) for f in files if isinstance(f, dict)]
    found: list[str] = []
    if index.get("index_sha256") != index_digest(listed):
        found.append("index.index_sha256 does not match the file list the bundle records")
    if directory is None:
        return found
    for entry in listed:
        path = directory / str(entry.get("name"))
        if not path.is_file():
            found.append(f"index file {entry.get('name')} is missing")
        elif path.stat().st_size != entry.get("bytes") or file_sha256(path) != entry.get("sha256"):
            found.append(f"index file {entry.get('name')} is not the one the bundle records")
    return found


def load_evidence_bundle(
    directory: Path,
    *,
    index_directory: Path | None = None,
    goa_sha256: str | None = None,
    ontology_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and verify the bundle; raise :class:`EvidenceBundleError` unless every hash agrees.

    ``index_directory`` is optional (CI holds no index bytes); when given, every recorded index
    file is hashed and a changed byte is refused. ``goa_sha256`` is the SHA-256 of the GO
    annotation source in use and ``ontology_sha256`` that of the ontology release; each must equal
    what the bundle recorded.
    """
    try:
        document = _read(directory / BUNDLE_FILE)
    except OSError as error:
        raise EvidenceBundleError(f"{BUNDLE_FILE} cannot be read: {error}") from error
    problems: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION or document.get("task") != TASK:
        problems.append("schema_version or task is not the one this code reads")
    if document.get("bundle_sha256") != bundle_digest(document):
        problems.append("bundle_sha256 does not match the bundle content")
    problems += _section_problems(document)
    problems += _index_problems(document, index_directory)
    go = _section(document, "go")
    if goa_sha256 is not None and go.get("sha256") != goa_sha256:
        problems.append("the GO annotation source is not the one the bundle records")
    if ontology_sha256 is not None and go.get("ontology_sha256") != ontology_sha256:
        problems.append("the ontology release is not the one the bundle records")
    if problems:
        raise EvidenceBundleError("; ".join(problems))
    return document


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise EvidenceBundleError(f"{path.name}: top level must be a JSON object")
    missing = [s for s in SECTIONS if s not in data]
    if missing:
        raise EvidenceBundleError(f"bundle.json is missing {missing}")
    return {str(k): v for k, v in data.items()}
