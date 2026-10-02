"""Count label states and classes in the built corpus, in two views, from the bytes.

Reads ``data/processed/corpus.jsonl`` and prints a Markdown report:

* label states per task and organism group (ADR-0006 §4: ``unknown``,
  ``negative`` and ``not_applicable`` are never merged);
* class counts for the localization, membrane-type and signal tasks;
* the corpus beside the scale named in ``PROJECT_PLAN.md`` §4 row 5.

Two views, neither of which changes the corpus:

* **all**: every record whose label is annotated;
* **ECO:0000269**: a record counts when at least one value of that task's label
  carries the experimental-evidence code in its recorded per-label evidence. In
  the per-class tables a class counts only when that class's own evidence does.

The GO function task is reported in the states table only: its evidence codes
are GO codes, a different system from ECO (ADR-0006 §3.5).

Every number is computed here; nothing is typed in except the plan's targets.
The report's SHA-256 of the corpus is printed so it can be tied to the manifest.

Run from the repository root::

    python scripts/class_counts.py
    python scripts/class_counts.py --out data/manifests/class_counts.md
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"

EXPERIMENTAL = "ECO:0000269"
STATES = ("annotated", "negative", "unknown", "not_applicable")
GROUPS = ("eukarya", "gram_positive", "gram_negative", "archaea")
GO_TASK = "go_function"
#: Tasks whose label values are classes worth a per-class table.
CLASS_TASKS = ("localization", "prokaryotic_localization", "membrane_type")
#: Tasks whose single value is a signal type (one row each in the signal table).
SIGNAL_TASKS = ("signal_peptide", "mitochondrial_transit_peptide", "chloroplast_transit_peptide")

#: The scale PROJECT_PLAN.md §4 row 5 names. A target, not a number to hit.
PLAN_SEQUENCES = 25_000
PLAN_LOCALIZATION_CLASSES = 10
PLAN_MEMBRANE_CLASSES = 4
PLAN_SIGNAL_CLASSES = 9


def _carries_experimental(label: dict[str, object]) -> bool:
    evidence = label["evidence"]
    return isinstance(evidence, dict) and any(EXPERIMENTAL in codes for codes in evidence.values())


class Tally:
    """Everything the report prints, accumulated in one pass over the corpus."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: Counter[str] = Counter()
        self.states: Counter[tuple[str, str, str]] = Counter()
        self.annotated_experimental: Counter[tuple[str, str]] = Counter()
        self.classes: Counter[tuple[str, str, str]] = Counter()
        self.classes_experimental: Counter[tuple[str, str, str]] = Counter()
        self.tasks: list[str] = []

    def add(self, record: dict[str, object]) -> None:
        """Fold one record into the tallies."""
        group = str(record["organism_group"])
        self.records[group] += 1
        labels = record["labels"]
        if not isinstance(labels, dict):
            raise TypeError("labels must be an object")
        if not self.tasks:
            self.tasks = list(labels)
        for task, label in labels.items():
            self.states[(task, group, label["state"])] += 1
            if label["state"] != "annotated" or task == GO_TASK:
                continue
            if _carries_experimental(label):
                self.annotated_experimental[(task, group)] += 1
            for value in label["values"]:
                self.classes[(task, value, group)] += 1
                if EXPERIMENTAL in label["evidence"].get(value, ()):
                    self.classes_experimental[(task, value, group)] += 1

    def state(self, task: str, group: str | None, state: str) -> int:
        """A state count for one group, or summed over all groups."""
        groups = GROUPS if group is None else (group,)
        return sum(self.states[(task, g, state)] for g in groups)

    def experimental(self, task: str, group: str | None) -> int:
        """Annotated records of a task whose label carries ECO:0000269."""
        groups = GROUPS if group is None else (group,)
        return sum(self.annotated_experimental[(task, g)] for g in groups)

    def class_names(self, task: str) -> list[str]:
        """Classes observed for a task, sorted."""
        return sorted({value for (t, value, _g) in self.classes if t == task})

    def class_count(self, counter: Counter[tuple[str, str, str]], task: str, value: str) -> int:
        """A class's count summed over groups."""
        return sum(counter[(task, value, g)] for g in GROUPS)


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _n(value: int) -> str:
    return f"{value:,}"


def _states_table(tally: Tally) -> str:
    rows: list[tuple[str, ...]] = []
    for task in tally.tasks:
        for group in (*GROUPS, None):
            counts = [tally.state(task, group, s) for s in STATES]
            rows.append(
                (
                    task,
                    group or "all groups",
                    *(_n(c) for c in counts),
                    _n(sum(counts)),
                    "n/a (GO codes)" if task == GO_TASK else _n(tally.experimental(task, group)),
                )
            )
    header = ("Task", "Group", *STATES, "total", f"annotated with {EXPERIMENTAL}")
    return _table(header, rows)


def _class_table(tally: Tally, task: str) -> str:
    rows = []
    for value in tally.class_names(task):
        by_group = [
            tally.classes[(task, value, g)] for g in GROUPS if g in _applicable_groups(tally, task)
        ]
        rows.append(
            (
                value,
                *(_n(c) for c in by_group),
                _n(tally.class_count(tally.classes, task, value)),
                _n(tally.class_count(tally.classes_experimental, task, value)),
            )
        )
    groups = [g for g in GROUPS if g in _applicable_groups(tally, task)]
    return _table(("Class", *groups, "all records", f"{EXPERIMENTAL} only"), rows)


def _applicable_groups(tally: Tally, task: str) -> set[str]:
    """Groups in which the task is not ``not_applicable`` for every record."""
    return {g for g in GROUPS if tally.state(task, g, "not_applicable") < tally.records[g]}


def _signal_table(tally: Tally) -> str:
    rows = [
        (
            task,
            _n(tally.state(task, None, "annotated")),
            _n(tally.experimental(task, None)),
            _n(tally.state(task, None, "negative")),
            _n(tally.state(task, None, "unknown")),
            _n(tally.state(task, None, "not_applicable")),
        )
        for task in SIGNAL_TASKS
    ]
    header = (
        "Signal task",
        "annotated (all)",
        f"annotated ({EXPERIMENTAL})",
        "negative",
        "unknown",
        "not_applicable",
    )
    return _table(header, rows)


def _ratio(measured: int, target: int) -> str:
    return f"{measured / target:.1f}x"


def _scale_table(tally: Tally) -> str:
    total = sum(tally.records.values())
    loc_all = tally.state("localization", None, "annotated")
    loc_exp = tally.experimental("localization", None)
    membrane = len(tally.class_names("membrane_type"))
    localization = len(tally.class_names("localization"))
    rows = [
        ("Sequences: all records", _n(PLAN_SEQUENCES), _n(total), _ratio(total, PLAN_SEQUENCES)),
        (
            "Sequences: localization annotated (all)",
            _n(PLAN_SEQUENCES),
            _n(loc_all),
            _ratio(loc_all, PLAN_SEQUENCES),
        ),
        (
            f"Sequences: localization annotated ({EXPERIMENTAL})",
            _n(PLAN_SEQUENCES),
            _n(loc_exp),
            _ratio(loc_exp, PLAN_SEQUENCES),
        ),
        ("Localization classes", str(PLAN_LOCALIZATION_CLASSES), str(localization), ""),
        ("Membrane-type classes", str(PLAN_MEMBRANE_CLASSES), str(membrane), ""),
        (
            "Signal classes (one per signal task here)",
            str(PLAN_SIGNAL_CLASSES),
            str(len(SIGNAL_TASKS)),
            "",
        ),
    ]
    return _table(("Quantity", "Plan target", "Corpus", "Corpus / target"), rows)


def render(tally: Tally, corpus_sha256: str) -> str:
    """The whole Markdown report."""
    parts = [
        "# Class counts",
        "",
        "> GENERATED by `python scripts/class_counts.py` from `data/processed/corpus.jsonl`.",
        f"> Corpus SHA-256: `{corpus_sha256}`.",
        "",
        "Records per organism group: "
        + ", ".join(f"{g} {_n(tally.records[g])}" for g in GROUPS)
        + f"; total {_n(sum(tally.records.values()))}.",
        "",
        "## Label states per task and group",
        "",
        "The four states are never merged (ADR-0006 §4).",
        "",
        _states_table(tally),
        "",
    ]
    for task in CLASS_TASKS:
        parts += [f"## Classes: {task}", "", _class_table(tally, task), ""]
    parts += [
        "## Signal and transit-peptide tasks",
        "",
        _signal_table(tally),
        "",
        "## Corpus beside the plan's scale",
        "",
        _scale_table(tally),
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    """Read the corpus once, tally, print, and optionally write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    tally = Tally()
    digest = hashlib.sha256()
    with args.corpus.open("rb") as handle:
        for line in handle:
            digest.update(line)
            tally.add(json.loads(line))
    report = render(tally, digest.hexdigest())
    sys.stdout.write(report)
    if args.out is not None:
        args.out.write_bytes(report.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
