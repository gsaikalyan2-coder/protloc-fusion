"""The split manifest refuses a non-deterministic or leaking split, and pins the config."""

import pytest

from protloc.data.split_manifest import (
    SplitManifestError,
    build_manifest,
    check_documents,
    config_values,
    render_config,
)

SHA = "a" * 64
CONFIG = """splits:
  alignment_backend: needleall
  partitioner_version: null
  removal_rate: null
"""


def _inputs(sha_b: str = SHA, passed: bool = True) -> dict[str, object]:
    summary = {"registry_sha256": SHA, "registry_rows": 10, "accessions_removed": 2}
    return {
        "run_a": summary,
        "run_b": {**summary, "registry_sha256": sha_b},
        "leakage": {
            "passed": passed,
            "registry_sha256": SHA,
            "corpus_sha256": "c" * 64,
            "identity_threshold": 0.3,
            "partition_sizes": {"1": 4},
        },
        "working_set": {
            "reading": "r",
            "in_set": 10,
            "distinct_sequences": 9,
            "list_sha256": "d" * 64,
        },
        "provenance": {
            "versions": {"graphpart": "1.0.2"},
            "command": ["graphpart"],
            "edge_command": ["graphpart", "mmseqs2needle"],
            "edge_order": "sorted",
            "hash_seed": "0",
            "seconds": {"run_a": 1.0, "run_b": 2.0},
        },
    }


def test_a_manifest_is_built_from_identical_runs_and_a_passing_report() -> None:
    manifest = build_manifest(**_inputs())  # type: ignore[arg-type]
    assert manifest["removal_rate"] == 0.2
    assert manifest["determinism"]["identical"] is True  # type: ignore[index]


def test_differing_registry_hashes_refuse_to_freeze() -> None:
    with pytest.raises(SplitManifestError, match="determinism"):
        build_manifest(**_inputs(sha_b="b" * 64))  # type: ignore[arg-type]


def test_a_failing_leakage_report_refuses_to_freeze() -> None:
    with pytest.raises(SplitManifestError, match="did not pass"):
        build_manifest(**_inputs(passed=False))  # type: ignore[arg-type]


def test_check_passes_after_the_config_is_rendered_and_fails_before() -> None:
    manifest = build_manifest(**_inputs())  # type: ignore[arg-type]
    leakage = _inputs()["leakage"]
    assert check_documents(manifest, leakage, CONFIG)  # type: ignore[arg-type]
    frozen = render_config(CONFIG, config_values(manifest))
    assert "alignment_backend: mmseqs2needle" in frozen
    assert 'partitioner_version: "1.0.2"' in frozen
    assert check_documents(manifest, leakage, frozen) == []  # type: ignore[arg-type]


def test_check_reports_a_leakage_report_for_another_registry() -> None:
    manifest = build_manifest(**_inputs())  # type: ignore[arg-type]
    leakage = {**_inputs()["leakage"], "registry_sha256": "e" * 64}  # type: ignore[dict-item]
    frozen = render_config(CONFIG, config_values(manifest))
    assert any("registry_sha256" in p for p in check_documents(manifest, leakage, frozen))
