"""Phase 6 split plumbing: distinct sequences in, split registry out.

GraphPart sees **distinct sequences**, not accessions. Sequences that are
byte-identical under several accessions (``sequence_sha256`` is not unique,
CLAUDE.md data schema) are collapsed to one representative, the smallest
accession, before partitioning. Every accession then inherits the partition and
cluster of its representative, so identical sequences cannot land in different
partitions, and a removed representative removes all of its accessions.

Nothing here aligns anything. GraphPart does that; this module prepares its
input and turns its output into the registry of CLAUDE.md § Data schema.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import csv
import hashlib
from pathlib import Path
from typing import Any

from protloc.data.working_set import iter_corpus

SPLIT_VERSION = "homology_v1"
PARTITIONER = "graphpart"
IDENTITY_THRESHOLD = 0.30
PARTITIONS = 5
REGISTRY_COLUMNS: tuple[str, ...] = (
    "split_version",
    "accession",
    "partition",
    "cluster_id",
    "partitioner",
    "identity_threshold",
)
REMOVED_REASON = "graphpart_removed"


class SplitError(ValueError):
    """Split inputs or outputs are inconsistent."""


def read_accessions(path: Path) -> list[str]:
    """The working-set accession list, in file order."""
    return path.read_text(encoding="ascii").split()


def distinct_sequences(
    corpus: Path, accessions: Iterable[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Collapse identical sequences.

    Returns ``(representative_of, sequence_of)``: every accession to its
    representative (the smallest accession sharing its sequence), and each
    representative to its normalised sequence.
    """
    wanted = set(accessions)
    by_sha: dict[str, list[str]] = defaultdict(list)
    sequence_of_sha: dict[str, str] = {}
    for record in iter_corpus(corpus):
        accession = record["accession"]
        if accession in wanted:
            sha = record["sequence_sha256"]
            by_sha[sha].append(accession)
            sequence_of_sha[sha] = record["sequence_normalized"]
    if sum(len(v) for v in by_sha.values()) != len(wanted):
        raise SplitError("some working-set accessions are not in the corpus")
    representative_of: dict[str, str] = {}
    sequence_of: dict[str, str] = {}
    for sha, members in by_sha.items():
        rep = min(members)
        sequence_of[rep] = sequence_of_sha[sha]
        for accession in members:
            representative_of[accession] = rep
    return representative_of, sequence_of


def fasta_text(sequence_of: Mapping[str, str], every: int = 1) -> str:
    """FASTA of the representatives in accession order; ``every`` keeps each k-th (a slice)."""
    reps = sorted(sequence_of)[::every]
    return "".join(f">{rep}\n{sequence_of[rep]}\n" for rep in reps)


def read_fasta(path: Path) -> dict[str, str]:
    """A FASTA file of single-line sequences as ``{id: sequence}``."""
    out: dict[str, str] = {}
    name = ""
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            name = line[1:].split("|")[0].strip()
            out[name] = ""
        elif name:
            out[name] += line.strip()
    return out


def read_graphpart(path: Path) -> dict[str, int]:
    """GraphPart's CSV as ``{representative: partition}`` (0-based).

    GraphPart's own ``cluster`` column is the **partition** it assigned, not a
    homology cluster. Only kept entities appear: a removed entity is absent,
    and :func:`build_registry` records it as removed.
    """
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SplitError(f"{path} is empty")
    for column in ("AC", "cluster"):
        if column not in rows[0]:
            raise SplitError(f"{path}: no '{column}' column; columns are {list(rows[0])}")
    return {row["AC"]: int(float(row["cluster"])) for row in rows}


def read_edges(path: Path) -> list[tuple[str, str, float]]:
    """GraphPart's saved edge list (``--save-checkpoint-path``): ``a,b,identity``."""
    edges: list[tuple[str, str, float]] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            a, b, identity = line.rstrip("\n").split(",")
            edges.append((a, b, float(identity)))
    return edges


def homology_clusters(
    members: Iterable[str], edges: Iterable[tuple[str, str, float]]
) -> dict[str, int]:
    """Connected components of the identity > threshold graph over ``members``.

    Two sequences are linked when GraphPart's recorded identity exceeds
    :data:`IDENTITY_THRESHOLD`. The result maps each member to a component
    number, numbered by the smallest member of each component so the numbering
    depends only on the data.
    """
    parent = {m: m for m in members}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, identity in edges:
        if round(identity, 6) > IDENTITY_THRESHOLD and a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({find(m) for m in parent})
    number = {root: i for i, root in enumerate(roots)}
    return {m: number[find(m)] for m in parent}


def build_registry(
    representative_of: Mapping[str, str],
    partition_of: Mapping[str, int],
    cluster_of: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The split registry rows and the removed accessions.

    Partitions are written 1-based. A representative GraphPart removed has no
    partition, and every accession sharing its sequence is removed with it.
    """
    unknown = set(partition_of) - set(representative_of.values())
    if unknown:
        raise SplitError(f"GraphPart returned {len(unknown)} ids that are not representatives")
    missing = set(partition_of) - set(cluster_of)
    if missing:
        raise SplitError(f"{len(missing)} kept representatives have no cluster")
    rows: list[dict[str, Any]] = []
    removed: list[str] = []
    for accession in sorted(representative_of):
        rep = representative_of[accession]
        if rep not in partition_of:
            removed.append(accession)
            continue
        rows.append(
            {
                "split_version": SPLIT_VERSION,
                "accession": accession,
                "partition": partition_of[rep] + 1,
                "cluster_id": f"C{cluster_of[rep]:07d}",
                "partitioner": PARTITIONER,
                "identity_threshold": f"{IDENTITY_THRESHOLD:.2f}",
            }
        )
    return rows, removed


def leakage_removals(
    pairs: Iterable[tuple[str, str, float]],
    partition_of: Mapping[str, int],
    representative_of: Mapping[str, str],
) -> list[str]:
    """Accessions to drop so that none of ``pairs`` still crosses partitions.

    The rule is deterministic: from each pair, drop the member that sits in the
    larger partition (partition size counted in registry accessions before any
    drop), ties to the later accession. Every accession sharing the dropped
    sequence goes with it, so identical sequences still stay together. A pair
    whose member is already dropped is skipped.
    """
    sizes = Counter(partition_of.values())
    group: dict[str, list[str]] = defaultdict(list)
    for accession, rep in representative_of.items():
        group[rep].append(accession)
    dropped: set[str] = set()
    for a, b, _ in pairs:
        rep_a, rep_b = representative_of[a], representative_of[b]
        if rep_a in dropped or rep_b in dropped:
            continue
        key_a, key_b = (sizes[partition_of[a]], a), (sizes[partition_of[b]], b)
        dropped.add(rep_a if key_a > key_b else rep_b)
    return sorted(acc for rep in dropped for acc in group[rep] if acc in partition_of)


def registry_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Registry as TSV bytes: LF endings, fixed columns, rows in accession order."""
    lines = ["\t".join(REGISTRY_COLUMNS)]
    lines += ["\t".join(str(row[c]) for c in REGISTRY_COLUMNS) for row in rows]
    return ("\n".join(lines) + "\n").encode("ascii")


def read_registry(path: Path) -> list[dict[str, str]]:
    """Read a registry TSV written by :func:`registry_bytes`."""
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows and tuple(rows[0]) != REGISTRY_COLUMNS:
        raise SplitError(f"{path}: columns are {tuple(rows[0])}, expected {REGISTRY_COLUMNS}")
    return rows


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "IDENTITY_THRESHOLD",
    "PARTITIONER",
    "PARTITIONS",
    "REGISTRY_COLUMNS",
    "REMOVED_REASON",
    "SPLIT_VERSION",
    "SplitError",
    "build_registry",
    "distinct_sequences",
    "fasta_text",
    "homology_clusters",
    "leakage_removals",
    "read_accessions",
    "read_edges",
    "read_fasta",
    "read_graphpart",
    "read_registry",
    "registry_bytes",
    "sha256_bytes",
]
