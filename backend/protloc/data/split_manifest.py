"""The split manifest: what the frozen split is, and the config values it fixes.

``data/manifests/split_manifest.json`` is written once by
``scripts/freeze_split.py --freeze`` from the two determinism runs and the
leakage report. ``--check`` needs no data bytes, so CI can run it: it verifies
that the manifest is internally consistent, that the leakage report agrees with
it, and that the data config carries the frozen values.
"""

import re
from typing import Any

from protloc.data.manifest import ManifestError, section
from protloc.data.splits import SPLIT_VERSION

FREEZE_MARK = "# frozen from data/manifests/split_manifest.json"
ALIGNMENT_BACKEND = "mmseqs2needle"


class SplitManifestError(ManifestError):
    """The split manifest is inconsistent with the report or the config."""


def build_manifest(
    *,
    run_a: dict[str, object],
    run_b: dict[str, object],
    leakage: dict[str, object],
    working_set: dict[str, object],
    provenance: dict[str, Any],
) -> dict[str, object]:
    """The manifest, from measured inputs only. Raises if the two runs differ."""
    sha_a = str(run_a["registry_sha256"])
    sha_b = str(run_b["registry_sha256"])
    if sha_a != sha_b:
        raise SplitManifestError(f"determinism failed: {sha_a} != {sha_b}")
    pre_a = str(run_a.get("pre_repair_registry_sha256", sha_a))
    pre_b = str(run_b.get("pre_repair_registry_sha256", sha_b))
    if pre_a != pre_b:
        raise SplitManifestError(f"determinism failed before repair: {pre_a} != {pre_b}")
    if leakage.get("registry_sha256") != sha_a:
        raise SplitManifestError("leakage report was run on a different registry")
    if leakage.get("passed") is not True:
        raise SplitManifestError("leakage report did not pass")
    versions: dict[str, str] = provenance["versions"]
    in_set = int(str(working_set["in_set"]))
    removed = int(str(run_a["accessions_removed"]))
    repaired = int(str(run_a.get("repair_accessions_removed", 0)))
    return {
        "split_version": SPLIT_VERSION,
        "partitioner": "graphpart",
        "partitioner_version": versions["graphpart"],
        "alignment_backend": ALIGNMENT_BACKEND,
        "tool_versions": versions,
        "identity_threshold": leakage["identity_threshold"],
        "partitions": 5,
        "command": provenance["command"],
        "edge_command": provenance["edge_command"],
        "edge_order": provenance["edge_order"],
        "hash_seed": provenance["hash_seed"],
        "corpus_sha256": leakage["corpus_sha256"],
        "working_set": {
            "reading": working_set["reading"],
            "in_set": in_set,
            "distinct_sequences": working_set["distinct_sequences"],
            "list_sha256": working_set["list_sha256"],
        },
        "registry": {
            "file": "data/processed/splits/registry.tsv",
            "sha256": sha_a,
            "rows": run_a["registry_rows"],
            "committed": False,
            "pre_repair_sha256": pre_a,
        },
        "sequences_removed": removed,
        "sequences_removed_by_graphpart": removed - repaired,
        "sequences_removed_by_leakage_repair": repaired,
        "removal_rate": round(removed / in_set, 6),
        "determinism": {
            "run_a_registry_sha256": sha_a,
            "run_b_registry_sha256": sha_b,
            "pre_repair_run_a_sha256": pre_a,
            "pre_repair_run_b_sha256": pre_b,
            "identical": True,
            "seconds": provenance["seconds"],
        },
        "leakage_report": {
            "pre_repair_file": "data/manifests/leakage_report_pre_repair.json",
            "file": "data/manifests/leakage_report.json",
            "passed": True,
            "partition_sizes": leakage.get("partition_sizes"),
        },
        "calibration_split_rule": (
            "Any calibration or validation split drawn from the training partitions "
            "must be made by whole homology clusters (the registry's cluster_id), "
            "never by random sequences, or the calibration data leaks into training."
        ),
    }


def config_values(manifest: dict[str, object]) -> dict[str, str]:
    """The YAML scalars the data config must carry, rendered as YAML text."""
    return {
        "alignment_backend": str(manifest["alignment_backend"]),
        "partitioner_version": f'"{manifest["partitioner_version"]}"',
        "removal_rate": str(manifest["removal_rate"]),
    }


def render_config(text: str, values: dict[str, str]) -> str:
    """The data config with each frozen key's value replaced, comments kept."""
    for key, value in values.items():
        pattern = re.compile(rf"^(\s+{key}:)[^\n]*$", re.MULTILINE)
        text, count = pattern.subn(rf"\1 {value}  {FREEZE_MARK}", text)
        if count != 1:
            raise SplitManifestError(f"expected one '{key}:' line in the config, found {count}")
    return text


def check_documents(
    manifest: dict[str, object], leakage: dict[str, object], config_text: str
) -> list[str]:
    """Every disagreement between manifest, leakage report and config; empty if none."""
    problems: list[str] = []
    registry = section(manifest, "registry")
    if leakage.get("passed") is not True:
        problems.append("leakage_report.json: passed is not true")
    if leakage.get("registry_sha256") != registry["sha256"]:
        problems.append("leakage_report.json: registry_sha256 differs from the manifest")
    if leakage.get("corpus_sha256") != manifest["corpus_sha256"]:
        problems.append("leakage_report.json: corpus_sha256 differs from the manifest")
    determinism = section(manifest, "determinism")
    if not (
        determinism["identical"] is True
        and determinism["run_a_registry_sha256"]
        == registry["sha256"]
        == determinism["run_b_registry_sha256"]
    ):
        problems.append("determinism section does not show two identical registry hashes")
    rendered = render_config(config_text, config_values(manifest))
    if rendered != config_text:
        problems.append("configs/data/uniprot_swissprot.yaml does not carry the frozen values")
    return problems
