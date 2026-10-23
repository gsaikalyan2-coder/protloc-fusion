r"""The cached-sequence catalog: every sequence the served engine can score, with its labels.

    python scripts/build_catalog.py [--write]

The gallery (``scripts/build_gallery.py``) is eleven hand-picked examples. The engine can score
every sequence in the 500-sequence residue cache (ADR-0016), but nothing listed them, so the
other 489 were unreachable from the UI. This script lists all of them: for every corpus record
whose ``sequence_sha256`` is in the cache and whose accession is in the split registry, one entry
with its accession, entry name, organism, length, partition and database labels.

Every entry says where it sits relative to training: partitions 1-4 are ``in_training_set``
(the heads were fitted on them, so a result is not evidence of accuracy) and partition 5 is the
held-out test partition. Showing a test sequence's database label beside a prediction tunes
nothing; no value is frozen from this file.

Without ``--write`` it prints counts. With ``--write`` it writes ``data/gallery/catalog.json``
(gitignored, like all of ``data/``).
"""

import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys

from protloc.data.manifest import sha256_file
from protloc.data.residue_cache import ResidueStore
from protloc.data.splits import read_registry

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
REGISTRY = REPO_ROOT / "data" / "processed" / "splits" / "registry.tsv"
ENGINE_CONFIG = REPO_ROOT / "configs" / "serving" / "engine.json"
OUT = REPO_ROOT / "data" / "gallery" / "catalog.json"
TRAINING_PARTITIONS = frozenset({"1", "2", "3", "4"})
LOCALIZATION = "localization"
SIGNAL = "signal_peptide"
ANNOTATED = "annotated"
PLANT_CLASS = "plastid"
SCHEMA_VERSION = 1


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


def _values(record: dict[str, object], task: str) -> list[str]:
    label = record["labels"][task]  # type: ignore[index]
    return sorted(label["values"]) if label["state"] == ANNOTATED else []


def _entry(record: dict[str, object], partition: str) -> dict[str, object]:
    labels = record["labels"]  # type: ignore[index]
    locations = _values(record, LOCALIZATION)
    in_training = partition in TRAINING_PARTITIONS
    where = (
        "in the training set: the heads were fitted on this sequence, so a result is not "
        "evidence of accuracy"
        if in_training
        else "held-out test partition: the heads never saw this sequence during fitting"
    )
    return {
        "id": str(record["accession"]),
        "entry_name": str(record["entry_name"]),
        "name": f"{record['entry_name']} ({record['accession']})",
        # The same rule as the gallery: a plastid label suggests the plant head.
        "organism_group": "plant" if PLANT_CLASS in locations else "eukaryote",
        "organism": record["organism"],
        "sequence": record["sequence_normalized"],
        "sequence_sha256": record["sequence_sha256"],
        "length": record["length"],
        "partition": int(partition),
        "in_training_set": in_training,
        "database_labels": {
            LOCALIZATION: locations,
            SIGNAL: _values(record, SIGNAL),
            "signal_peptide_cleavage_site": labels[SIGNAL]["cleavage_site"],
        },
        "note": f"Swiss-Prot {record['source_release']}; {where}.",
    }


def build(cache: Path) -> list[dict[str, object]]:
    """Every registry record of the corpus whose states are in ``cache``, by accession."""
    cached = ResidueStore(cache).lengths
    partition = {r["accession"]: r["partition"] for r in read_registry(REGISTRY)}
    entries: list[dict[str, object]] = []
    with CORPUS.open("rb") as handle:
        for line in handle:
            record = json.loads(line)
            accession = record["accession"]
            if record["sequence_sha256"] in cached and accession in partition:
                entries.append(_entry(record, str(partition[accession])))
    entries.sort(key=lambda e: str(e["id"]))
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    cache_name = json.loads(ENGINE_CONFIG.read_text(encoding="utf-8"))["encoder"]["residue_cache"]
    cache = REPO_ROOT / "data" / "embeddings" / cache_name
    for path in (CORPUS, REGISTRY, cache / "index.json"):
        if not path.exists():
            _say(f"missing: {path}")
            return 1
    entries = build(cache)
    distinct = len({e["sequence_sha256"] for e in entries})
    by_partition = Counter(int(e["partition"]) for e in entries)  # type: ignore[call-overload]
    _say(f"{len(entries)} accessions, {distinct} distinct cached sequences")
    _say("by partition: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_partition.items())))
    document = {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": sha256_file(CORPUS),
        "registry_sha256": sha256_file(REGISTRY),
        "residue_cache": cache.name,
        "selection_rule": "every registry accession whose sequence is in the residue cache",
        "entries": entries,
    }
    document["sha256"] = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        _say(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
