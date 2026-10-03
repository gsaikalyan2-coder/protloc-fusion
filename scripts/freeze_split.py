r"""Freeze the split: write the split manifest and the data config values from it.

    python scripts/freeze_split.py --freeze --run-a DIR --run-b DIR \
        --graphpart-version V --mmseqs-version V --emboss-version V
        Needs the two determinism run directories (registry_summary.json, run.json),
        the registry copied to data/processed/splits/registry.tsv, and a passing
        data/manifests/leakage_report.json for that registry. Writes
        data/manifests/split_manifest.json and the frozen keys in
        configs/data/uniprot_swissprot.yaml.

    python scripts/freeze_split.py --check
        No data bytes needed (this is what CI runs). Exit 1 unless the manifest,
        the leakage report and the config agree.

Run from the repository root inside ``.venv``.
"""

import argparse
from pathlib import Path
import sys

from protloc.data.manifest import read_json_object, write_json
from protloc.data.split_manifest import (
    SplitManifestError,
    build_manifest,
    check_documents,
    config_values,
    render_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = REPO_ROOT / "data" / "manifests"
SPLIT_MANIFEST = MANIFESTS / "split_manifest.json"
LEAKAGE = MANIFESTS / "leakage_report.json"
WORKING_SET = MANIFESTS / "working_set.json"
DATA_CONFIG = REPO_ROOT / "configs" / "data" / "uniprot_swissprot.yaml"


def _freeze(args: argparse.Namespace) -> int:
    run_a = read_json_object(args.run_a / "registry_summary.json")
    run_b = read_json_object(args.run_b / "registry_summary.json")
    timing_a = read_json_object(args.run_a / "run.json")
    timing_b = read_json_object(args.run_b / "run.json")
    raw_command = timing_a["command"]
    command = [str(x) for x in raw_command] if isinstance(raw_command, list) else []
    versions = {
        "graphpart": args.graphpart_version,
        "mmseqs2": args.mmseqs_version,
        "emboss": args.emboss_version,
    }
    seconds = {"run_a": float(str(timing_a["seconds"])), "run_b": float(str(timing_b["seconds"]))}
    manifest = build_manifest(
        run_a=run_a,
        run_b=run_b,
        leakage=read_json_object(LEAKAGE),
        working_set=read_json_object(WORKING_SET),
        provenance={
            "versions": versions,
            "command": command,
            "edge_command": timing_a["edge_command"],
            "edge_order": timing_a["edge_order"],
            "hash_seed": "0",
            "seconds": seconds,
        },
    )
    write_json(SPLIT_MANIFEST, manifest)
    DATA_CONFIG.write_bytes(
        render_config(DATA_CONFIG.read_text(encoding="utf-8"), config_values(manifest)).encode(
            "utf-8"
        )
    )
    sys.stdout.write(f"wrote {SPLIT_MANIFEST.name} and froze {DATA_CONFIG.name}\n")
    return 0


def _check(_: argparse.Namespace) -> int:
    problems = check_documents(
        read_json_object(SPLIT_MANIFEST),
        read_json_object(LEAKAGE),
        DATA_CONFIG.read_text(encoding="utf-8"),
    )
    for problem in problems:
        sys.stderr.write(f"FAIL: {problem}\n")
    if not problems:
        sys.stdout.write("split manifest, leakage report and config agree\n")
    return 1 if problems else 0


def main() -> int:
    """Parse the mode and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--run-a", type=Path)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--graphpart-version")
    parser.add_argument("--mmseqs-version")
    parser.add_argument("--emboss-version")
    args = parser.parse_args()
    if args.freeze:
        needed = (
            args.run_a,
            args.run_b,
            args.graphpart_version,
            args.mmseqs_version,
            args.emboss_version,
        )
        if not all(needed):
            parser.error("--freeze needs --run-a --run-b and the three tool versions")
        try:
            return _freeze(args)
        except SplitManifestError as error:
            sys.stderr.write(f"FAIL: {error}\n")
            return 1
    return _check(args)


if __name__ == "__main__":
    raise SystemExit(main())
