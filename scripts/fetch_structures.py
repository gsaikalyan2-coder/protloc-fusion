"""Fetch AlphaFold DB backbones for the 500 cached sequences, once, on a machine with internet.

The served app never calls AlphaFold (ADR-0003: no network on the request path). This script
runs offline from serving: it resolves each cached sequence hash to a UniProt accession through
``data/processed/corpus.jsonl``, asks the AlphaFold DB API for that accession's model, and keeps
only the C-alpha trace and the per-residue pLDDT (the PDB B-factor column) in
``data/structures/<sha256>.json``.

A model is kept only when AlphaFold's sequence equals ours exactly; otherwise residue numbers
would not line up with the attribution tracks, so the accession is recorded as a miss and
nothing is drawn for it. The structures are AlphaFold DB predictions (CC-BY 4.0, EMBL-EBI), not
this project's output, and are not experimental structures.

Standard library only. Resumable: existing files are skipped.

    python scripts/fetch_structures.py            # fetch what is missing
    python scripts/fetch_structures.py --limit 5  # smoke run
"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_INDEX = REPO_ROOT / "data" / "embeddings" / "0e528d4a46831b05_subset500" / "shard_00000.tsv"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
OUT_DIR = REPO_ROOT / "data" / "structures"
API = "https://alphafold.ebi.ac.uk/api/prediction/"
SCOPE = "subset (500), not comparable to DeepLoc or to the Phase 8 full-corpus numbers"
LICENSE = "CC-BY-4.0, AlphaFold Protein Structure Database, EMBL-EBI"
THREE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip
_TIMEOUT = 60
_PAUSE_SECONDS = 0.3


def parse_ca(pdb_text: str) -> tuple[str, list[list[float]], list[float]]:
    """Sequence, C-alpha coordinates (Angstrom) and pLDDT of the first chain's ATOM records."""
    letters: list[str] = []
    coords: list[list[float]] = []
    plddt: list[float] = []
    chain: str | None = None
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if chain is None:
            chain = line[21]
        if line[21] != chain:
            break
        letters.append(THREE.get(line[17:20].strip(), "X"))
        coords.append(
            [
                round(float(line[30:38]), 1),
                round(float(line[38:46]), 1),
                round(float(line[46:54]), 1),
            ]
        )
        plddt.append(round(float(line[60:66]), 1))
    return "".join(letters), coords, plddt


def _get(url: str) -> bytes:
    """GET an https URL and return the body."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "protloc-fusion/phase-16"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310 - https only
        body: bytes = response.read()
    return body


def wanted_hashes() -> set[str]:
    """The 500 sequence hashes the served cache holds."""
    rows = CACHE_INDEX.read_text(encoding="ascii").splitlines()[1:]
    return {row.split("\t")[0] for row in rows if row}


def resolve(hashes: set[str]) -> dict[str, tuple[str, str]]:
    """Map each hash to (accession, normalised sequence): the first corpus record that has it."""
    found: dict[str, tuple[str, str]] = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            if len(found) == len(hashes):
                break
            record = json.loads(line)
            digest = record["sequence_sha256"]
            if digest in hashes and digest not in found:
                found[digest] = (record["accession"], record["sequence_normalized"])
    return found


def fetch_one(digest: str, accession: str, sequence: str) -> dict[str, Any]:
    """One structure document, or raise ValueError saying why it cannot be used."""
    listing = json.loads(_get(API + accession))
    if not listing:
        raise ValueError("AlphaFold DB has no model for this accession")
    entry = listing[0]
    pdb = _get(str(entry["pdbUrl"]))
    seq, coords, plddt = parse_ca(pdb.decode("utf-8"))
    if seq != sequence:
        raise ValueError(
            f"AlphaFold sequence differs from ours (len {len(seq)} vs {len(sequence)})"
        )
    return {
        "schema": "protloc-structure-v1",
        "sequence_sha256": digest,
        "accession": accession,
        "source": "AlphaFold DB prediction, not an experimental structure",
        "license": LICENSE,
        "entry_id": entry.get("entryId"),
        "model_version": entry.get("latestVersion"),
        "model_created": entry.get("modelCreatedDate"),
        "pdb_sha256": hashlib.sha256(pdb).hexdigest(),
        "length": len(seq),
        "ca": coords,
        "plddt": plddt,
    }


def main() -> int:
    """Fetch every missing structure, then write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many (smoke run)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved = resolve(wanted_hashes())
    files: dict[str, str] = {}
    misses: dict[str, str] = {}
    done = 0
    for digest, (accession, sequence) in sorted(resolved.items()):
        target = OUT_DIR / f"{digest}.json"
        if not target.exists():
            if args.limit and done >= args.limit:
                break
            try:
                document = fetch_one(digest, accession, sequence)
            except (ValueError, urllib.error.URLError, TimeoutError, KeyError) as exc:
                misses[accession] = str(exc)
                print(f"miss {accession}: {exc}", file=sys.stderr)  # noqa: T201
                continue
            target.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
            done += 1
            time.sleep(_PAUSE_SECONDS)
        files[digest] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "scope": SCOPE,
        "source": "AlphaFold DB predictions: not experimental, not this model's output",
        "license": LICENSE,
        "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cached_sequences": len(resolved),
        "structures": len(files),
        "misses": misses,
        "files_sha256": files,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"structures {len(files)}/{len(resolved)}  misses {len(misses)}  wrote {OUT_DIR}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
