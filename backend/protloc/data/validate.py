"""Measuring the corpus, and freezing the length and alphabet policy from it.

The policy is not chosen and then justified. :func:`measure` reads the built
corpus. :func:`derive_policy` turns that measurement into two values:

* ``max_sequence_length``: the nearest-rank quantile of the length
  distribution, at a quantile passed on the command line and recorded next to
  the result. Records longer than it stay in the corpus, so the corpus hash
  doesn't depend on the policy. They are counted, and Phase 4 filters them when
  it builds the splits.
* ``alphabet``: exactly the set of characters observed in the corpus. The
  residue policy file decides what is done with each code. The measurement
  decides which codes exist.

:func:`check_policy` recomputes both from the corpus and compares them with the
manifest. That comparison is ``validate_sequences.py --check``.
"""

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from protloc.data.records import RecordError, from_canonical_line
from protloc.data.rules import ResiduePolicy

#: Percentiles reported in the dataset card. Descriptive only: none of these
#: sets a policy value except through the quantile passed to derive_policy.
REPORTED_PERCENTILES: tuple[str, ...] = ("0.5", "0.9", "0.95", "0.99", "0.999")


class PolicyError(ValueError):
    """The corpus and the recorded policy disagree."""


@dataclass(frozen=True, slots=True)
class Measurements:
    """What the corpus contains, measured."""

    record_count: int
    residue_counts: tuple[tuple[str, int], ...]
    records_containing: tuple[tuple[str, int], ...]
    lengths: tuple[int, ...]

    def percentile(self, quantile: float) -> int:
        """Nearest-rank percentile: the smallest length covering ``quantile``."""
        return nearest_rank(self.lengths, quantile)


def nearest_rank(sorted_values: tuple[int, ...], quantile: float) -> int:
    """The nearest-rank quantile of an ascending, non-empty sequence."""
    if not sorted_values:
        raise ValueError("no values to take a quantile of")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    rank = math.ceil(quantile * len(sorted_values))
    return sorted_values[rank - 1]


def measure(corpus: Path) -> Measurements:
    """Read every record, checking canonical form, order and uniqueness."""
    residues: Counter[str] = Counter()
    containing: Counter[str] = Counter()
    lengths: list[int] = []
    previous = ""
    with corpus.open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                record = from_canonical_line(line)
            except RecordError as exc:
                raise PolicyError(f"{corpus}:{number}: {exc}") from exc
            if record.accession <= previous:
                raise PolicyError(
                    f"{corpus}:{number}: {record.accession} is out of order or duplicated"
                )
            previous = record.accession
            counts = Counter(record.sequence_normalized)
            residues.update(counts)
            containing.update(counts.keys())
            lengths.append(record.length)
    if not lengths:
        raise PolicyError(f"{corpus} holds no records")
    return Measurements(
        record_count=len(lengths),
        residue_counts=tuple(sorted(residues.items())),
        records_containing=tuple(sorted(containing.items())),
        lengths=tuple(sorted(lengths)),
    )


def derive_policy(
    measurements: Measurements, quantile: float, residues: ResiduePolicy
) -> dict[str, object]:
    """The frozen policy section of the manifest, from a measurement."""
    max_length = measurements.percentile(quantile)
    observed = "".join(code for code, _ in measurements.residue_counts)
    counts = dict(measurements.residue_counts)
    containing = dict(measurements.records_containing)
    tracked = sorted(
        set().union(*(codes for name, codes in residues.keep_groups if name != "standard"))
        | residues.exclude_record
    )
    return {
        "max_sequence_length": max_length,
        "length_quantile": quantile,
        "length_rule": "nearest-rank quantile of all record lengths",
        "records_over_max_length": sum(1 for n in measurements.lengths if n > max_length),
        "alphabet": observed,
        "alphabet_rule": "exactly the characters observed in the corpus",
        "nonstandard_codes": {
            code: {
                "residues": counts.get(code, 0),
                "records": containing.get(code, 0),
                "action": "exclude_record" if code in residues.exclude_record else "keep",
            }
            for code in tracked
        },
        "residue_policy_version": residues.version,
        "record_count": measurements.record_count,
        "length_min": measurements.lengths[0],
        "length_max": measurements.lengths[-1],
        "length_percentiles": {p: measurements.percentile(float(p)) for p in REPORTED_PERCENTILES},
    }


def check_policy(
    measurements: Measurements, recorded: dict[str, object], residues: ResiduePolicy
) -> list[str]:
    """Every disagreement between the corpus and the recorded policy."""
    quantile = recorded.get("length_quantile")
    if not isinstance(quantile, float):
        return ["policy.length_quantile is missing or not a number"]
    expected = derive_policy(measurements, quantile, residues)
    problems = [
        f"policy.{key}: recorded {recorded.get(key)!r}, measured {value!r}"
        for key, value in expected.items()
        if recorded.get(key) != value
    ]
    stray = {code for code, _ in measurements.residue_counts} & residues.exclude_record
    if stray:
        problems.append(f"corpus contains excluded codes {sorted(stray)}")
    return problems


__all__ = [
    "REPORTED_PERCENTILES",
    "Measurements",
    "PolicyError",
    "check_policy",
    "derive_policy",
    "measure",
    "nearest_rank",
]
