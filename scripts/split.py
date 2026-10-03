r"""Prepare GraphPart's input, run it, and turn its output into the split registry.

Run from the repository root, inside the venv, with PYTHONPATH=backend. Three steps:

    python scripts/split.py prepare --out-dir data/processed/splits/run_main
        Reads data/processed/splits/working_set.txt and the corpus, collapses
        identical sequences, writes distinct.fasta and rep_map.tsv.

    python scripts/split.py run --out-dir data/processed/splits/run_main --threads 8
        Runs GraphPart (mmseqs2needle backend, 30% identity, 5 partitions) on
        distinct.fasta. Output and the log go to files in --out-dir. Do not
        wrap this in Measure-Command: it swallows stdout; read split.log.

    python scripts/split.py registry --out-dir data/processed/splits/run_main
        Writes registry.tsv (split registry), removed.txt, registry_summary.json.

    python scripts/split.py repair --out-dir DIR --report REPORT.json
        REPORT is data/manifests/leakage_report_pre_repair.json. Applies the recorded
        repair: removes one member of each cross-partition pair the independent check
        found (rule in splits.leakage_removals), keeps registry_pre_repair.tsv and
        registry_summary_pre_repair.json, and writes repair.json. Deterministic, so it
        is run on both determinism directories.

``prepare --every K`` keeps each K-th distinct sequence, for timed slices only.
Identity threshold, partition count and backend are constants of ADR-0002, not flags.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from protloc.data.manifest import read_json_object, write_json
from protloc.data.splits import (
    IDENTITY_THRESHOLD,
    PARTITIONS,
    SplitError,
    build_registry,
    distinct_sequences,
    fasta_text,
    homology_clusters,
    leakage_removals,
    read_accessions,
    read_edges,
    read_fasta,
    read_graphpart,
    read_registry,
    registry_bytes,
    sha256_bytes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
WORKING_SET = REPO_ROOT / "data" / "processed" / "splits" / "working_set.txt"

#: GraphPart `-re`: mmseqs2 identities in [RECOMPUTE, threshold) are recomputed with needle.
RECOMPUTE_THRESHOLD = 0.2


def graphpart_command(out_dir: Path, threads: int, prefilter: bool) -> list[str]:
    """The exact GraphPart command line. The same list is recorded in the split manifest."""
    command = [
        "graphpart",
        "mmseqs2needle",
        "--fasta-file",
        str(out_dir / "distinct.fasta"),
        "--threshold",
        f"{IDENTITY_THRESHOLD:.2f}",
        "--partitions",
        str(PARTITIONS),
        "--recompute-threshold",
        f"{RECOMPUTE_THRESHOLD:.2f}",
        "--threads",
        str(threads),
        "--out-file",
        str(out_dir / "graphpart.csv"),
        "--save-checkpoint-path",
        str(out_dir / "edges.csv"),
    ]
    if prefilter:
        command.append("--prefilter")
    return command


def _prepare(args: argparse.Namespace) -> int:
    accessions = read_accessions(WORKING_SET)
    representative_of, sequence_of = distinct_sequences(CORPUS, accessions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = fasta_text(sequence_of, every=args.every)
    (args.out_dir / "distinct.fasta").write_bytes(fasta.encode("ascii"))
    rep_map = "".join(f"{a}\t{representative_of[a]}\n" for a in sorted(representative_of))
    (args.out_dir / "rep_map.tsv").write_bytes(rep_map.encode("ascii"))
    kept = fasta.count(">")
    sys.stdout.write(
        f"working set {len(accessions)} accessions, {len(sequence_of)} distinct sequences, "
        f"fasta {kept} sequences (every {args.every})\n"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    args.out_dir = args.out_dir.resolve()
    for tool in ("graphpart", "mmseqs", "needleall"):
        if shutil.which(tool) is None:
            sys.stderr.write(f"{tool} not found on PATH\n")
            return 1
    command = graphpart_command(args.out_dir, args.threads, args.prefilter)
    log = args.out_dir / "split.log"
    sys.stdout.write("running: " + " ".join(command) + f"\nlog: {log}\n")
    started = time.monotonic()
    with log.open("wb") as handle:
        result = subprocess.run(  # noqa: S603 - fixed argv list, no shell
            command, stdout=handle, stderr=subprocess.STDOUT, check=False, cwd=args.out_dir
        )
    seconds = round(time.monotonic() - started, 1)
    (args.out_dir / "run.json").write_bytes(
        json.dumps(
            {"command": command, "returncode": result.returncode, "seconds": seconds}
        ).encode("ascii")
        + b"\n"
    )
    sys.stdout.write(f"graphpart exit {result.returncode} in {seconds} s\n")
    return result.returncode


def _registry(args: argparse.Namespace) -> int:
    rep_of: dict[str, str] = {}
    for line in (args.out_dir / "rep_map.tsv").read_text(encoding="ascii").splitlines():
        accession, rep = line.split("\t")
        rep_of[accession] = rep
    sequences = read_fasta(args.out_dir / "distinct.fasta")
    rep_of = {a: r for a, r in rep_of.items() if r in sequences}
    partition_of = read_graphpart(args.out_dir / "graphpart.csv")
    clusters = homology_clusters(partition_of, read_edges(args.out_dir / "edges.csv"))
    rows, removed = build_registry(rep_of, partition_of, clusters)
    body = registry_bytes(rows)
    (args.out_dir / "registry.tsv").write_bytes(body)
    (args.out_dir / "removed.txt").write_bytes("".join(f"{a}\n" for a in removed).encode("ascii"))
    summary = {
        "registry_rows": len(rows),
        "registry_sha256": sha256_bytes(body),
        "accessions_removed": len(removed),
        "representatives_in": len(sequences),
        "representatives_kept": len(partition_of),
        "clusters": len(set(clusters.values())),
    }
    write_json(args.out_dir / "registry_summary.json", summary)
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


def _repair(args: argparse.Namespace) -> int:
    out = args.out_dir
    pre_registry, pre_summary = (
        out / "registry_pre_repair.tsv",
        out / "registry_summary_pre_repair.json",
    )
    if not pre_registry.exists():
        shutil.copyfile(out / "registry.tsv", pre_registry)
        shutil.copyfile(out / "registry_summary.json", pre_summary)
    pre_bytes = pre_registry.read_bytes()
    report = read_json_object(args.report)
    leakage = report["leakage"]
    if report["registry_sha256"] != sha256_bytes(pre_bytes):
        raise SplitError("the leakage report was made on a different registry")
    pairs = [(str(a), str(b), float(i)) for a, b, i in leakage["examples"]]  # type: ignore[index, attr-defined]
    if len(pairs) != leakage["cross_partition_pairs_above_threshold"]:  # type: ignore[index]
        raise SplitError("the report lists fewer pairs than it counted; rerun the check")
    rows = read_registry(pre_registry)
    partition_of = {r["accession"]: int(r["partition"]) for r in rows}
    rep_of: dict[str, str] = {}
    for line in (out / "rep_map.tsv").read_text(encoding="ascii").splitlines():
        accession, rep = line.split("\t")
        rep_of[accession] = rep
    removals = leakage_removals(pairs, partition_of, rep_of)
    dropped = set(removals)
    kept = [r for r in rows if r["accession"] not in dropped]
    body = registry_bytes(kept)
    (out / "registry.tsv").write_bytes(body)
    summary = read_json_object(pre_summary)
    graphpart_removed = int(str(summary["accessions_removed"]))
    removed = sorted(set((out / "removed.txt").read_text(encoding="ascii").split()) | dropped)
    (out / "removed.txt").write_bytes("".join(f"{a}\n" for a in removed).encode("ascii"))
    summary.update(
        {
            "registry_rows": len(kept),
            "registry_sha256": sha256_bytes(body),
            "pre_repair_registry_sha256": sha256_bytes(pre_bytes),
            "graphpart_accessions_removed": graphpart_removed,
            "repair_accessions_removed": len(removals),
            "accessions_removed": graphpart_removed + len(removals),
        }
    )
    write_json(out / "registry_summary.json", summary)
    write_json(
        out / "repair.json",
        {
            "rule": "from each pair drop the member in the larger partition, ties to the "
            "later accession; accessions sharing the dropped sequence go with it",
            "pairs": [list(p) for p in pairs],
            "removed_accessions": removals,
            "pre_repair_registry_sha256": sha256_bytes(pre_bytes),
            "post_repair_registry_sha256": sha256_bytes(body),
        },
    )
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


def main() -> int:
    """Dispatch on step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    for name in ("prepare", "run", "registry", "repair"):
        step = steps.add_parser(name)
        step.add_argument("--out-dir", type=Path, required=True)
        if name == "prepare":
            step.add_argument("--every", type=int, default=1)
        if name == "repair":
            step.add_argument("--report", type=Path, required=True)
        if name == "run":
            step.add_argument("--threads", type=int, default=1)
            step.add_argument("--prefilter", action="store_true")
    args = parser.parse_args()
    try:
        return {"prepare": _prepare, "run": _run, "registry": _registry, "repair": _repair}[
            args.step
        ](args)
    except (SplitError, ValueError, OSError, KeyError) as exc:
        sys.stderr.write(f"split failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
