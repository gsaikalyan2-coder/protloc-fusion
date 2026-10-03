"""Leakage checks for the split registry (Phase 6, PROJECT_PLAN §4 row 6).

Everything here is recomputed from sequences and the registry, never read from
GraphPart's own report. Three independent checks:

* **exact duplicates**: sequences with one ``sequence_sha256`` that sit in more
  than one partition (hashing, no alignment);
* **registry clusters**: registry ``cluster_id`` values spanning partitions;
* **homology**: pairs, and connected components, of an MMseqs2 search over the
  registry's own sequences, built by the checker, with identity above the
  threshold across partitions.

A search is a heuristic and cannot prove absence. A zero from it is a bounded
claim: no pair above the threshold was found by that search, at that
sensitivity and hit cap. ``docs/phases/06.md`` says so.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import re
from typing import Any

#: A search hit: query, target, identity as a fraction in [0, 1].
Hit = tuple[str, str, float]

#: MMseqs2 prints identity with three decimals, so compare at that precision.
_PRECISION = 3


def exact_duplicate_crossings(
    partition_of: Mapping[str, int], sha_of: Mapping[str, str]
) -> list[str]:
    """Sequence hashes found in more than one partition, sorted."""
    seen: dict[str, set[int]] = defaultdict(set)
    for accession, partition in partition_of.items():
        seen[sha_of[accession]].add(partition)
    return sorted(sha for sha, parts in seen.items() if len(parts) > 1)


def registry_cluster_crossings(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """Registry ``cluster_id`` values whose members sit in more than one partition."""
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[row["cluster_id"]].add(row["partition"])
    return sorted(cluster for cluster, parts in seen.items() if len(parts) > 1)


def read_hits(path: Path) -> Iterator[Hit]:
    """Parse an MMseqs2 ``query,target,fident`` TSV."""
    with path.open(encoding="ascii") as handle:
        for line in handle:
            query, target, identity = line.rstrip("\n").split("\t")[:3]
            yield query, target, float(identity)


_NEEDLE_IDENTITY = re.compile(r"^# Identity:\s+(\d+)/(\d+)", re.MULTILINE)


def parse_needle_identity(report: str) -> float:
    """Global identity from EMBOSS ``needle`` output: identical residues / alignment length.

    This is the definition GraphPart's ``needleall`` step uses (denominator
    ``full``, gaps included), and the one ADR-0002's "30% global identity" means.
    """
    match = _NEEDLE_IDENTITY.search(report)
    if match is None:
        raise ValueError("no '# Identity:' line in the needle output")
    return int(match.group(1)) / int(match.group(2))


def confirm_pairs(
    pairs: Iterable[Hit],
    sequences: Mapping[str, str],
    align_many: Callable[[list[tuple[str, str]]], list[float]],
    threshold: float,
) -> list[Hit]:
    """Candidate pairs re-scored by global alignment, keeping those above ``threshold``.

    ``pairs`` are candidates from the local search (identity over the shorter
    sequence, which is not the ADR's definition and can be high for a short
    local match). The returned hits carry the global identity.
    """
    candidates = list(pairs)
    scores = align_many([(sequences[a], sequences[b]) for a, b, _ in candidates])
    confirmed = [
        (a, b, score)
        for (a, b, _), score in zip(candidates, scores, strict=True)
        if _above(score, threshold)
    ]
    return sorted(confirmed, key=lambda hit: (-hit[2], hit[0]))


def _above(identity: float, threshold: float) -> bool:
    return round(identity, _PRECISION) > threshold


def cross_partition_pairs(
    hits: Iterable[Hit], partition_of: Mapping[str, int], threshold: float
) -> list[Hit]:
    """Unordered pairs above ``threshold`` whose two members sit in different partitions."""
    best: dict[tuple[str, str], float] = {}
    for query, target, identity in hits:
        if query == target or query not in partition_of or target not in partition_of:
            continue
        if partition_of[query] == partition_of[target] or not _above(identity, threshold):
            continue
        key = (min(query, target), max(query, target))
        best[key] = max(best.get(key, 0.0), identity)
    return sorted(((a, b, i) for (a, b), i in best.items()), key=lambda hit: (-hit[2], hit[0]))


def spanning_components(
    hits: Iterable[Hit], partition_of: Mapping[str, int], threshold: float
) -> int:
    """Connected components of the above-threshold graph that span more than one partition.

    The graph joins every pair above the threshold, inside or across
    partitions, so a component is a homology cluster as the search sees it.
    """
    parent = {a: a for a in partition_of}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for query, target, identity in hits:
        if query == target or query not in parent or target not in parent:
            continue
        if _above(identity, threshold):
            ra, rb = find(query), find(target)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, set[int]] = defaultdict(set)
    for accession, partition in partition_of.items():
        members[find(accession)].add(partition)
    return sum(1 for parts in members.values() if len(parts) > 1)


def partition_stats(
    records: Iterable[Mapping[str, Any]], partition_of: Mapping[str, int]
) -> dict[str, Any]:
    """Sizes and label-state counts per partition, per organism group and per task."""
    sizes: Counter[int] = Counter()
    groups: dict[int, Counter[str]] = defaultdict(Counter)
    tasks: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    fragments: Counter[int] = Counter()
    for record in records:
        partition = partition_of.get(record["accession"])
        if partition is None:
            continue
        sizes[partition] += 1
        groups[partition][record["organism_group"]] += 1
        fragments[partition] += int(bool(record["is_fragment"]))
        for task, label in record["labels"].items():
            tasks[partition][task][label["state"]] += 1
    return {
        "partition_sizes": {str(p + 1): sizes[p] for p in sorted(sizes)},
        "fragments_per_partition": {str(p + 1): fragments[p] for p in sorted(sizes)},
        "per_group": {str(p + 1): dict(sorted(groups[p].items())) for p in sorted(sizes)},
        "task_states": {
            str(p + 1): {t: dict(sorted(c.items())) for t, c in sorted(tasks[p].items())}
            for p in sorted(sizes)
        },
    }


__all__ = [
    "Hit",
    "cross_partition_pairs",
    "exact_duplicate_crossings",
    "partition_stats",
    "read_hits",
    "registry_cluster_crossings",
    "spanning_components",
]
