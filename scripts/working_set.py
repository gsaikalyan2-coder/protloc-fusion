r"""Measure the Phase 6 working-set readings, and write the chosen working set.

Run from the repository root, inside the venv, with PYTHONPATH=backend::

    python scripts/working_set.py --measure
        Count the three readings of "necessary" side by side. Writes nothing.

    python scripts/working_set.py --write --reading necessary
        Write the list of accessions in the split and its summary:
          * data/processed/splits/working_set.txt   one accession per line (gitignored)
          * data/manifests/working_set.json         counts and the list's SHA-256 (committed)

The corpus SHA-256 is checked against the manifest first. MAX_SEQUENCE_LENGTH is
read from the manifest's frozen policy, never typed here.
"""

import argparse
import hashlib
from pathlib import Path
import sys

from protloc.data.manifest import read_json_object, section, sha256_file, write_json
from protloc.data.working_set import READINGS, WorkingSetError, tally_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
LIST_FILE = REPO_ROOT / "data" / "processed" / "splits" / "working_set.txt"
SUMMARY = REPO_ROOT / "data" / "manifests" / "working_set.json"


def _inputs() -> tuple[str, int]:
    """The verified corpus SHA-256 and the frozen MAX_SEQUENCE_LENGTH."""
    manifest = read_json_object(MANIFEST)
    expected = section(manifest, "build")["corpus_sha256"]
    digest, _ = sha256_file(CORPUS)
    if digest != expected:
        raise WorkingSetError(f"corpus SHA-256 {digest} != manifest {expected}")
    max_length = section(manifest, "policy")["max_sequence_length"]
    if not isinstance(max_length, int):
        raise WorkingSetError("manifest policy.max_sequence_length is not an integer")
    return digest, max_length


def _measure() -> int:
    digest, max_length = _inputs()
    tallies = tally_corpus(CORPUS, max_length, READINGS)
    sys.stdout.write(f"corpus_sha256 {digest}\nmax_sequence_length {max_length}\n")
    for reading in READINGS:
        s = tallies[reading].summary()
        sys.stdout.write(
            f"\n== {reading}\n"
            f"in_set {s['in_set']}  excluded {s['excluded_total']} {s['excluded_by_reason']}\n"
            f"per_group {s['per_group']}\n"
            f"fragments {s['fragments']} {s['fragments_by_group']}\n"
            f"distinct_sequences {s['distinct_sequences']}  "
            f"shared_sequences {s['sequences_shared_by_several_accessions']}  "
            f"records_in_shared {s['records_in_shared_sequences']}\n"
        )
    return 0


def _write(reading: str) -> int:
    digest, max_length = _inputs()
    tally = tally_corpus(CORPUS, max_length, (reading,), keep_for=reading)[reading]
    body = "".join(f"{accession}\n" for accession in tally.kept).encode("ascii")
    LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIST_FILE.write_bytes(body)
    summary = tally.summary()
    summary["corpus_sha256"] = digest
    summary["list_file"] = "data/processed/splits/working_set.txt"
    summary["list_sha256"] = hashlib.sha256(body).hexdigest()
    summary["command"] = f"python scripts/working_set.py --write --reading {reading}"
    write_json(SUMMARY, summary)
    sys.stdout.write(f"wrote {len(tally.kept)} accessions, list sha256 {summary['list_sha256']}\n")
    return 0


def main() -> int:
    """Dispatch on mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--reading", choices=READINGS)
    args = parser.parse_args()
    if args.write and args.reading is None:
        parser.error("--write requires --reading")
    try:
        return _measure() if args.measure else _write(args.reading)
    except (WorkingSetError, ValueError, OSError, KeyError) as exc:
        sys.stderr.write(f"working set failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
