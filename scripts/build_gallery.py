r"""Phase 14: the example gallery, built from the corpus and the residue cache.

    python scripts/build_gallery.py [--write] [--per-class 1] [--max-length 700]

Picks real Swiss-Prot sequences whose ESM-2 states are in the 500-sequence residue cache, so
the served path can score them (ADR-0016: the encoder is cache-only). Selection is a rule, not
a draw: registry partitions 1-4 only (partition 5 is never read), eukarya, one annotated
location, length at most --max-length; for each of the ten localization classes the lowest
accession that satisfies it, plus the lowest accession with an annotated signal-peptide
cleavage site and the lowest with two annotated locations. Every entry says it is a training-set
sequence: a gallery result shows what the model does on a sequence it was fitted on, which is
not evidence of accuracy.

Without --write it prints the selection. With --write it writes data/gallery/gallery.json
(gitignored, like all of data/).
"""

import argparse
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
CACHE = REPO_ROOT / "data" / "embeddings" / "0e528d4a46831b05_subset500_residues"
OUT = REPO_ROOT / "data" / "gallery" / "gallery.json"
ENGINE_CONFIG = REPO_ROOT / "configs" / "serving" / "engine.json"
ALLOWED_PARTITIONS = frozenset({"1", "2", "3", "4"})
LOCALIZATION = "localization"
SIGNAL = "signal_peptide"
ANNOTATED = "annotated"
CORPUS_GROUP = "eukarya"  # the corpus has one eukaryotic group (OrganismGroup.EUKARYA)
PLANT_CLASS = "plastid"


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


def _candidates(max_length: int) -> list[dict[str, object]]:
    cached = ResidueStore(CACHE).lengths
    partition = {r["accession"]: r["partition"] for r in read_registry(REGISTRY)}
    found: list[dict[str, object]] = []
    with CORPUS.open("rb") as handle:
        for line in handle:
            record = json.loads(line)
            sha = record["sequence_sha256"]
            if sha not in cached or partition.get(record["accession"]) not in ALLOWED_PARTITIONS:
                continue
            if record["organism_group"] != CORPUS_GROUP or record["length"] > max_length:
                continue
            found.append({**record, "_partition": partition[record["accession"]]})
    found.sort(key=lambda r: str(r["accession"]))
    return found


def _values(record: dict[str, object], task: str) -> list[str]:
    label = record["labels"][task]  # type: ignore[index]
    return list(label["values"]) if label["state"] == ANNOTATED else []


def _select(pool: Sequence[dict[str, object]], classes: Sequence[str], per_class: int) -> list:
    chosen: dict[str, tuple[dict[str, object], str]] = {}

    def take(record: dict[str, object], why: str) -> None:
        chosen.setdefault(str(record["accession"]), (record, why))

    for cls in classes:
        hits = [r for r in pool if _values(r, LOCALIZATION) == [cls]]
        for record in hits[:per_class]:
            take(record, f"annotated {cls.replace('_', ' ')} only")
    for record in pool:
        label = record["labels"][SIGNAL]  # type: ignore[index]
        if label["state"] == ANNOTATED and label["cleavage_site"] is not None:
            take(
                record, f"annotated signal peptide, cleavage after residue {label['cleavage_site']}"
            )
            break
    for record in pool:
        if len(_values(record, LOCALIZATION)) == 2:  # noqa: PLR2004 - two annotated locations
            take(record, "two annotated locations")
            break
    return [{"record": r, "why": why} for r, why in chosen.values()]


def _entry(item: dict[str, object]) -> dict[str, object]:
    record = item["record"]
    labels = record["labels"]  # type: ignore[index]
    return {
        "id": str(record["accession"]),  # type: ignore[index]
        "name": f"{record['entry_name']} ({record['accession']})",  # type: ignore[index]
        "organism_group": "plant" if PLANT_CLASS in _values(record, LOCALIZATION) else "eukaryote",
        "organism": record["organism"],  # type: ignore[index]
        "sequence": record["sequence_normalized"],  # type: ignore[index]
        "sequence_sha256": record["sequence_sha256"],  # type: ignore[index]
        "length": record["length"],  # type: ignore[index]
        "partition": int(record["_partition"]),  # type: ignore[index]
        "in_training_set": True,
        "database_labels": {
            LOCALIZATION: _values(record, LOCALIZATION),
            SIGNAL: _values(record, SIGNAL),
            "signal_peptide_cleavage_site": labels[SIGNAL]["cleavage_site"],
        },
        "note": (
            f"Swiss-Prot {record['source_release']}; {item['why']}. In the training set: "  # type: ignore[index]
            "the model was fitted on this sequence, so its result here is not evidence of accuracy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=700)
    args = parser.parse_args(argv)
    for path in (CORPUS, REGISTRY, CACHE / "index.json"):
        if not path.exists():
            _say(f"missing: {path}")
            return 1
    classes = list(json.loads(ENGINE_CONFIG.read_text())["localization"]["codes"])
    entries = [_entry(i) for i in _select(_candidates(args.max_length), classes, args.per_class)]
    for e in entries:
        _say(f"{e['id']:<10} len {e['length']:<4} p{e['partition']} {e['database_labels']}")
    document = {
        "schema_version": 1,
        "corpus_sha256": sha256_file(CORPUS),
        "registry_sha256": sha256_file(REGISTRY),
        "residue_cache": CACHE.name,
        "selection_rule": "partitions 1-4, eukarya, length <= "
        f"{args.max_length}, lowest accession per rule",
        "examples": entries,
    }
    document["sha256"] = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _say(f"{len(entries)} examples")
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        _say(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
