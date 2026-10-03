"""Build the corpus from the recorded snapshot.

Reads the ``snapshot`` section of ``data/manifests/dataset_manifest.json``,
checks that the file on disk still has the recorded SHA-256, parses it under the
rules in ``backend/rules/``, and writes ``corpus.jsonl`` and
``build_report.json``.

With the default output directory, it also writes the manifest's ``build``
section, which drops any ``policy`` section measured on an earlier corpus.
Pointing ``--out-dir`` anywhere else leaves the manifest untouched. That is how
determinism is proven: build twice into two scratch directories and compare the
hashes (``docs/phases/05.md``, determinism proof).

Run from the repository root, inside ``.venv``::

    python scripts/build_dataset.py
    python scripts/build_dataset.py --out-dir data/interim/run_a
"""

import argparse
from pathlib import Path
import shutil
import sys

from protloc.data.build import REPORT_FILE, BuildError, build_corpus, iter_entries, write_corpus
from protloc.data.manifest import read_json_object, section, sha256_file, update_section
from protloc.data.parse import EntryError
from protloc.data.rules import load_corpus_rules

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
RULES_DIR = REPO_ROOT / "backend" / "rules"
DATA_FILE = "data/raw/uniprot_sprot.dat.gz"
REPORT_COPY = REPO_ROOT / "data" / "manifests" / REPORT_FILE


def _verified_snapshot(snapshot: dict[str, object]) -> tuple[Path, str]:
    """The snapshot path and release, after checking the bytes are unchanged."""
    release = snapshot.get("release")
    files = snapshot.get("files")
    if not isinstance(release, str) or not isinstance(files, dict):
        raise BuildError("manifest snapshot section is malformed")
    entry = files.get(DATA_FILE)
    if not isinstance(entry, dict):
        raise BuildError(f"manifest snapshot does not list {DATA_FILE}")
    path = REPO_ROOT / DATA_FILE
    if not path.is_file():
        raise BuildError(f"{DATA_FILE} is missing; run scripts/download_data.py")
    digest, size = sha256_file(path)
    if digest != entry.get("sha256") or size != entry.get("bytes"):
        raise BuildError(f"{DATA_FILE} does not match the SHA-256 recorded in the manifest")
    return path, release


def main() -> int:
    """Build, write and record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()

    try:
        manifest = read_json_object(MANIFEST)
        snapshot_path, release = _verified_snapshot(section(manifest, "snapshot"))
        rules = load_corpus_rules(RULES_DIR)
        result = build_corpus(iter_entries(snapshot_path), release, rules)
    except (BuildError, EntryError, ValueError, OSError) as exc:
        sys.stderr.write(f"build failed: {exc}\n")
        return 1

    written = write_corpus(result, out_dir)
    report = result.report
    sys.stdout.write(
        f"records {report['record_count']}  excluded {report['excluded_total']}  "
        f"corpus_sha256 {written['corpus_sha256']}\n"
    )
    if out_dir != DEFAULT_OUT.resolve():
        sys.stdout.write(f"wrote {out_dir}; manifest not updated (non-default --out-dir)\n")
        return 0

    build: dict[str, object] = {
        "corpus_file": "data/processed/corpus.jsonl",
        "report_file": "data/processed/build_report.json",
        **written,
        "record_count": report["record_count"],
        "entries_read": report["entries_read"],
        "excluded": report["excluded"],
        "label_states": report["label_states"],
        "rules_sha256": report["rules_sha256"],
        "parser": report["parser"],
        "accepted_evidence_codes": "all",
        "command": "python scripts/build_dataset.py",
    }
    update_section(MANIFEST, "build", build)
    # A committed copy, so the dataset card can be regenerated and checked in CI
    # without the corpus bytes. Its hash is build.report_sha256.
    shutil.copyfile(out_dir / REPORT_FILE, REPORT_COPY)
    sys.stdout.write(
        "manifest build section written; next: "
        "python scripts/validate_sequences.py --freeze --length-quantile <q>\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
