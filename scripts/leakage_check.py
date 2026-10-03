r"""Independent leakage gate for the Phase 6 split registry.

Recomputes cross-partition leakage from the corpus sequences and the registry,
by a different path from the one GraphPart used: it writes its own FASTA from the
corpus, runs its own MMseqs2 search (sensitivity 8.0, 5000 hits per query, against
GraphPart's prefilter at 7.5 and 300), and hashes sequences itself. It never reads
GraphPart's output, edges or report.

    python scripts/leakage_check.py --registry data/processed/splits/registry.tsv --threads 12
        Writes data/manifests/leakage_report.json. Exit 1 if any leakage count is non-zero.

    python scripts/leakage_check.py --toy leaked      # must exit 1
    python scripts/leakage_check.py --toy clean       # must exit 0
        Self-test on four real corpus sequences, assigned to partitions by hand:
        HBA_HUMAN (P69905) and HBA_MOUSE (P01942), real orthologs, and two real
        accessions that share one sequence (A0A017SP50, A0A2D1VNJ8). No sequence
        is invented. The report goes to the scratch directory, not data/manifests.

Needs `mmseqs` on PATH. Run inside the venv with PYTHONPATH=backend.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys

from protloc.data.leakage import (
    confirm_pairs,
    cross_partition_pairs,
    exact_duplicate_crossings,
    parse_needle_identity,
    partition_stats,
    read_hits,
    registry_cluster_crossings,
    spanning_components,
)
from protloc.data.manifest import read_json_object, section, sha256_file, write_json
from protloc.data.splits import (
    IDENTITY_THRESHOLD,
    PARTITIONS,
    REGISTRY_COLUMNS,
    REMOVED_REASON,
    SPLIT_VERSION,
    read_registry,
    sha256_bytes,
)
from protloc.data.working_set import iter_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
WORKING_SET_SUMMARY = REPO_ROOT / "data" / "manifests" / "working_set.json"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
DEFAULT_REGISTRY = REPO_ROOT / "data" / "processed" / "splits" / "registry.tsv"
DEFAULT_REPORT = REPO_ROOT / "data" / "manifests" / "leakage_report.json"

#: The checker's own search settings. Deliberately not GraphPart's (7.5, 300).
SENSITIVITY = "8.0"
MAX_SEQS = "5000"
MIN_SEQ_ID = "0.25"
#: Identity over the shorter sequence: never lower than over the longer one, so a pair
#: above the threshold under any denominator is above it here.
SEQ_ID_MODE = "1"

TOY_ACCESSIONS = ("P69905", "P01942", "A0A017SP50", "A0A2D1VNJ8")


def _toy_registry(kind: str, path: Path) -> None:
    """A registry over four real accessions, leaked or clean."""
    if kind == "leaked":
        assign = {
            "P69905": (1, "T1"),
            "P01942": (2, "T2"),
            "A0A017SP50": (3, "T3"),
            "A0A2D1VNJ8": (4, "T4"),
        }
    else:
        assign = {
            "P69905": (1, "T1"),
            "P01942": (1, "T1"),
            "A0A017SP50": (3, "T3"),
            "A0A2D1VNJ8": (3, "T3"),
        }
    lines = ["\t".join(REGISTRY_COLUMNS)]
    for accession in sorted(assign):
        partition, cluster = assign[accession]
        lines.append(
            "\t".join(
                [
                    SPLIT_VERSION,
                    accession,
                    str(partition),
                    cluster,
                    "toy",
                    f"{IDENTITY_THRESHOLD:.2f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _load(registry_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Corpus records of the registry's accessions (one pass) and their FASTA text."""
    wanted = {row["accession"] for row in registry_rows}
    records = [r for r in iter_corpus(CORPUS) if r["accession"] in wanted]
    if len(records) != len(wanted):
        raise ValueError(f"{len(wanted) - len(records)} registry accessions are not in the corpus")
    return records, {r["accession"]: r["sequence_normalized"] for r in records}


def _search(sequences: dict[str, str], work: Path, threads: int) -> tuple[Path, str, list[str]]:
    if shutil.which("mmseqs") is None:
        raise FileNotFoundError("mmseqs not found on PATH")
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "check.fasta"
    fasta.write_bytes("".join(f">{a}\n{s}\n" for a, s in sorted(sequences.items())).encode("ascii"))
    out = work / "check_hits.tsv"
    command = [
        "mmseqs", "easy-search", str(fasta), str(fasta), str(out), str(work / "tmp"),
        "-s", SENSITIVITY, "--max-seqs", MAX_SEQS, "-e", "inf",
        "--min-seq-id", MIN_SEQ_ID, "--seq-id-mode", SEQ_ID_MODE, "--alignment-mode", "3",
        "--threads", str(threads), "--format-output", "query,target,fident",
    ]  # fmt: skip
    log = work / "check_search.log"
    with log.open("wb") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)  # noqa: S603 - fixed argv list, no shell
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs easy-search failed ({result.returncode}); see {log}")
    version = subprocess.run(["mmseqs", "version"], capture_output=True, text=True, check=True)  # noqa: S607
    return out, version.stdout.strip(), command[:1] + command[1:2] + ["..."] + command[6:]


def _needle(pair: tuple[str, str]) -> float:
    """Global identity of one pair by EMBOSS needle, GraphPart's alignment settings."""
    command = [
        "needle", f"asis::{pair[0]}", f"asis::{pair[1]}", "-gapopen", "10", "-gapextend", "0.5",
        "-endopen", "10", "-endextend", "0.5", "-datafile", "EBLOSUM62", "-sprotein1", "-sprotein2",
        "-outfile", "stdout", "-auto",
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603 - fixed argv list, no shell
    if result.returncode != 0:
        raise RuntimeError(f"needle failed ({result.returncode}): {result.stderr.strip()[:300]}")
    return parse_needle_identity(result.stdout)


def _align_many(threads: int) -> Callable[[list[tuple[str, str]]], list[float]]:
    def run(pairs: list[tuple[str, str]]) -> list[float]:
        if shutil.which("needle") is None:
            raise FileNotFoundError("needle (EMBOSS) not found on PATH")
        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(pool.map(_needle, pairs))

    return run


def _removal(registry: Path, in_set: int | None) -> dict[str, object]:
    """Removal accounting from the files the registry step wrote beside the registry."""
    out: dict[str, object] = {}
    summary_path = registry.parent / "registry_summary.json"
    if summary_path.is_file() and in_set:
        summary = read_json_object(summary_path)
        removed = int(str(summary["accessions_removed"]))
        out["graphpart_removed_accessions"] = int(
            str(summary.get("graphpart_accessions_removed", removed))
        )
        out["leakage_repair_removed_accessions"] = int(
            str(summary.get("repair_accessions_removed", 0))
        )
        out["total_removed_accessions"] = removed
        out["graphpart_removed_distinct_sequences"] = int(str(summary["representatives_in"])) - int(
            str(summary["representatives_kept"])
        )
        out["reason"] = REMOVED_REASON
        out["removal_rate"] = round(removed / in_set, 6)
    if WORKING_SET_SUMMARY.is_file():
        ws = read_json_object(WORKING_SET_SUMMARY)
        out["working_set_reading"] = str(ws["command"]).split("--reading")[-1].strip()
        out["working_set_records"] = ws["in_set"]
        out["left_out_of_the_split_before_partitioning"] = ws["excluded_by_reason"]
    return out


def main() -> int:
    """Run the gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--work-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "leakage_work"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--toy", choices=("leaked", "clean"))
    parser.add_argument(
        "--hits", type=Path, help="reuse an earlier check_hits.tsv instead of searching again"
    )
    args = parser.parse_args()

    corpus_sha = str(section(read_json_object(MANIFEST), "build")["corpus_sha256"])
    if sha256_file(CORPUS)[0] != corpus_sha:
        sys.stderr.write("corpus does not match the manifest\n")
        return 1
    registry, out = args.registry, args.out
    if args.toy:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        registry, out = args.work_dir / f"toy_{args.toy}.tsv", args.work_dir / "toy_report.json"
        _toy_registry(args.toy, registry)

    rows = read_registry(registry)
    partition_of = {r["accession"]: int(r["partition"]) - 1 for r in rows}
    records, sequences = _load(rows)
    sha_of = {str(r["accession"]): str(r["sequence_sha256"]) for r in records}  # type: ignore[index]
    if args.hits:
        hits_path = args.hits
        mmseqs_version = subprocess.run(
            ["mmseqs", "version"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        command = ["mmseqs", "easy-search", "...", "(reused hits file)"]
    else:
        hits_path, mmseqs_version, command = _search(
            sequences, args.work_dir / "search", args.threads
        )
    hits = list(read_hits(hits_path))

    duplicates = exact_duplicate_crossings(partition_of, sha_of)
    candidates = cross_partition_pairs(hits, partition_of, IDENTITY_THRESHOLD)
    crossing = confirm_pairs(candidates, sequences, _align_many(args.threads), IDENTITY_THRESHOLD)
    clusters_search = spanning_components(crossing, partition_of, IDENTITY_THRESHOLD)
    clusters_registry = registry_cluster_crossings(rows)
    stats = partition_stats(records, partition_of)
    ws_in_set = None if args.toy else int(read_json_object(WORKING_SET_SUMMARY)["in_set"])  # type: ignore[call-overload]
    passed = not (duplicates or crossing or clusters_search or clusters_registry)
    report = {
        "split_version": SPLIT_VERSION,
        "passed": passed,
        "corpus_sha256": corpus_sha,
        "registry_sha256": sha256_bytes(registry.read_bytes()),
        "registry_rows": len(rows),
        "identity_threshold": IDENTITY_THRESHOLD,
        "partitions": PARTITIONS,
        "leakage": {
            "cross_partition_exact_duplicates": len(duplicates),
            "cross_partition_homology_clusters": clusters_search,
            "registry_clusters_spanning_partitions": len(clusters_registry),
            "cross_partition_pairs_local_candidates": len(candidates),
            "cross_partition_pairs_above_threshold": len(crossing),
            "max_cross_partition_identity_found": max((i for _, _, i in crossing), default=None),
            "examples": [list(hit) for hit in crossing[:10]],
        },
        "checker": {
            "path": "independent: own FASTA from the corpus, own MMseqs2 search, own hashing",
            "stages": (
                "1. MMseqs2 local search, identity over the shorter sequence: candidate "
                "cross-partition pairs above the threshold. 2. Each candidate re-scored by "
                "EMBOSS needle global identity (identical residues / alignment length, gap open "
                "10, extend 0.5, EBLOSUM62), the definition ADR-0002 and GraphPart use. The "
                "gate counts stage 2."
            ),
            "hits_file_sha256": sha256_file(hits_path)[0],
            "mmseqs_version": mmseqs_version,
            "search": {
                "sensitivity": SENSITIVITY,
                "max_seqs": int(MAX_SEQS),
                "min_seq_id": float(MIN_SEQ_ID),
                "seq_id_mode": "shorter sequence",
                "alignment_mode": 3,
                "command": command,
            },
            "limit": "A heuristic search cannot prove absence: zero means none found at this "
            "sensitivity and hit cap.",
        },
        "sequences_removed": _removal(registry, ws_in_set),
        **stats,
    }
    write_json(out, report)
    sys.stdout.write(
        f"{'PASS' if passed else 'FAIL'}: duplicates {len(duplicates)}, "
        f"search clusters {clusters_search}, registry clusters {len(clusters_registry)}, "
        f"cross-partition pairs {len(crossing)} -> {out}\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
