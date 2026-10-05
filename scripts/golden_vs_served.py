r"""Phase 14: what changed between the motif baseline and the served engine.

    python scripts/golden_vs_served.py [--write]

Two readings, neither of which rewrites ``backend/tests/golden/golden_set.json``:

1. **Golden set.** The 67 golden inputs are synthetic or gallery sequences with no cached ESM-2
   states, so the cache-only served engine (ADR-0016) refuses every one of them. This script
   counts that, per category, instead of leaving it as a surprise. The motif baseline itself is
   unchanged and still guarded by ``pytest tests/golden`` against ``golden_set_motif.json``.
2. **Cached real sequences.** For every cached eukaryotic sequence in registry partitions 1-4
   (partition 5 is not read), the motif engine's top location is compared with the served
   model's. This is an agreement between two predictors, not an accuracy: the model was fitted
   on these sequences, and the labels are not consulted.

Needs the corpus, the registry and the cache, and the serving environment of ``.env``. Output:
the summary on stdout; with --write also ``data/serving/served_vs_motif.json`` (gitignored).
"""

import argparse
from collections import Counter
from collections.abc import Sequence
import importlib.util
import json
from pathlib import Path
import sys

from protloc.config.serving import get_serving_settings
from protloc.data.splits import read_registry
from protloc.inference import motif
from protloc.inference.engine import EngineRefusal, load_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "backend" / "tests" / "golden"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
REGISTRY = REPO_ROOT / "data" / "processed" / "splits" / "registry.tsv"
OUT = REPO_ROOT / "data" / "serving" / "served_vs_motif.json"
ALLOWED = frozenset({"1", "2", "3", "4"})
GROUP = "eukaryote"  # the app group the comparison runs under
CORPUS_GROUP = "eukarya"


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


def _golden_inputs() -> list[dict[str, object]]:
    spec = importlib.util.spec_from_file_location("golden_inputs", GOLDEN_DIR / "golden_inputs.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load golden_inputs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return [dict(entry) for entry in module.build_entries()]


def _cached_real(engine_has: object) -> list[tuple[str, str]]:
    partition = {r["accession"]: r["partition"] for r in read_registry(REGISTRY)}
    found: list[tuple[str, str]] = []
    with CORPUS.open("rb") as handle:
        for line in handle:
            record = json.loads(line)
            if partition.get(record["accession"]) not in ALLOWED:
                continue
            if record["organism_group"] == CORPUS_GROUP and engine_has(
                record["sequence_normalized"]
            ):  # type: ignore[operator]
                found.append((record["accession"], record["sequence_normalized"]))
    return sorted(found)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    engine = load_engine(get_serving_settings())

    golden = _golden_inputs()
    refused = Counter()
    for entry in golden:
        try:
            engine.predict(str(entry["sequence"]), str(entry["organism_group"]), str(entry["id"]))
            refused["scored"] += 1
        except EngineRefusal as refusal:
            refused[refusal.code] += 1
    _say(f"golden inputs: {len(golden)}; served outcome: {dict(refused)}")

    sequences = _cached_real(engine.has_states)
    agree, rows = 0, []
    motif_refused: Counter[str] = Counter()
    for accession, sequence in sequences:
        served = engine.predict(sequence, GROUP, accession, with_evidence=False)
        try:
            baseline = motif.predict(sequence, GROUP, accession, with_evidence=False)
        except ValueError:  # the motif engine only scores lengths 30-3000
            motif_refused["length_outside_30_3000"] += 1
            continue
        same = served["top_location"] == baseline["top_location"]
        agree += same
        rows.append(
            {
                "accession": accession,
                "motif_top": baseline["top_location"],
                "served_top": served["top_location"],
                "same": same,
            }
        )
    summary = {
        "golden_inputs": len(golden),
        "golden_served_outcome": dict(refused),
        "cached_real_sequences": len(sequences),
        "compared": len(rows),
        "motif_engine_refused": dict(motif_refused),
        "top_location_agreement": round(agree / len(rows), 4) if rows else None,
        "note": "agreement between two predictors on sequences the model was fitted on; "
        "not an accuracy, no label consulted, partition 5 not read",
        "model_version": engine.provenance["model_version"],
    }
    _say(json.dumps(summary, indent=1))
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
        _say(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
