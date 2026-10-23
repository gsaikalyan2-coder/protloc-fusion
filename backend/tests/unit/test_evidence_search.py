"""The serve-time search wrapper and the ontology readers (Phase 14).

The search is exercised against a fake ``mmseqs`` shell script, so the subprocess handling, the
deadline and the degraded states are tested without the 2.3 GiB index. POSIX only: on Windows the
real binary is the one exercised, by the measurement in ``docs/phases/14.md``.
"""

from pathlib import Path
import stat
import sys
import time

import pytest

from protloc.evidence import homology
from protloc.evidence.ontology import TermStore, parse_obo
from protloc.evidence.search import EvidenceUnavailable, SearchPlan, search, timeout_for

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the fake binary is a sh script")

_SETTINGS = homology.SearchSettings(
    sensitivity=5.7, e_value=1000.0, max_seqs=1000, alignment_mode=3, threads=1, verbosity=1
)

_FAKE = """#!/bin/sh
case "$1" in
  createdb) : > "$3" ;;
  search) {search_body} ;;
  convertalis) printf '{hit_line}' > "$5" ;;
esac
"""


def _plan(
    tmp_path: Path, search_body: str = ":", hit_line: str = "", budget: float = 5.0
) -> SearchPlan:
    exe = tmp_path / "mmseqs"
    exe.write_text(_FAKE.format(search_body=search_body, hit_line=hit_line))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return SearchPlan(
        executable=exe,
        database=tmp_path / "db",
        work_directory=tmp_path / "work",
        settings=_SETTINGS,
        threads=1,
        verbosity=1,
        base_timeout_seconds=budget,
        per_query_seconds=0.0,
    )


@posix_only
def test_hits_come_back_under_the_callers_own_keys(tmp_path: Path) -> None:
    line = "q1\\tP12345\\t0.9\\t0.95\\t0.9\\t1e-30\\t200\\t100\\n"
    plan = _plan(tmp_path, hit_line=line)
    found = search(plan, {"a": "MKTAYIAKQR", "b": "MKTAYIAKQRQ"})
    assert list(found) == ["b"]
    assert found["b"][0].target == "P12345"
    assert found["b"][0].identity == pytest.approx(0.9)


@posix_only
def test_the_work_directory_is_cleaned_after_a_search(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    search(plan, {"a": "MKTAYIAKQR"})
    assert list((tmp_path / "work").iterdir()) == []


@posix_only
def test_a_search_past_its_deadline_is_killed_and_reported(tmp_path: Path) -> None:
    plan = _plan(tmp_path, search_body="sleep 30", budget=0.5)
    started = time.monotonic()
    with pytest.raises(EvidenceUnavailable, match="ran out of time"):
        search(plan, {"a": "MKTAYIAKQR"})
    assert time.monotonic() - started < 10
    assert list((tmp_path / "work").iterdir()) == []


@posix_only
def test_a_failing_step_names_the_step(tmp_path: Path) -> None:
    plan = _plan(tmp_path, search_body="echo broken >&2; exit 3")
    with pytest.raises(EvidenceUnavailable, match=r"search exited 3.*broken"):
        search(plan, {"a": "MKTAYIAKQR"})


def test_a_missing_binary_is_a_reason_not_a_crash(tmp_path: Path) -> None:
    plan = SearchPlan(
        tmp_path / "absent",
        tmp_path / "db",
        tmp_path / "w",
        _SETTINGS,
        1,
        1,
        5.0,
        0.0,
    )
    with pytest.raises(EvidenceUnavailable, match="not installed"):
        search(plan, {"a": "MKT"})


def test_a_binary_that_cannot_start_is_a_reason_not_a_crash(tmp_path: Path) -> None:
    """A binary for another platform (a Windows .exe on Linux) used to fail the whole job."""
    exe = tmp_path / "mmseqs.exe"
    exe.write_bytes(b"MZ\x90\x00 not an ELF binary")
    (tmp_path / "sh.exe").write_bytes(b"")  # the Windows shell check passes; the start fails
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    plan = SearchPlan(exe, tmp_path / "db", tmp_path / "w", _SETTINGS, 1, 1, 5.0, 0.0)
    with pytest.raises(EvidenceUnavailable, match="could not be started"):
        search(plan, {"a": "MKT"})
    assert list((tmp_path / "w").iterdir()) == []


@posix_only
def test_an_unusable_work_directory_is_a_reason_not_a_crash(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    blocked = SearchPlan(
        plan.executable, plan.database, blocker / "work", plan.settings, 1, 1, 5.0, 0.0
    )
    with pytest.raises(EvidenceUnavailable, match="work directory is unusable"):
        search(blocked, {"a": "MKTAYIAKQR"})


def test_no_queries_is_no_search(tmp_path: Path) -> None:
    assert search(_plan(tmp_path), {}) == {}


def test_the_deadline_grows_with_the_batch(tmp_path: Path) -> None:
    plan = _plan(tmp_path, budget=5.0)
    assert timeout_for(plan, 1) == 5.0
    scaled = SearchPlan(
        plan.executable,
        plan.database,
        plan.work_directory,
        plan.settings,
        1,
        1,
        5.0,
        0.25,
    )
    assert timeout_for(scaled, 1) == 5.0
    assert timeout_for(scaled, 101) == 30.0


def test_obo_keeps_names_aspects_and_drops_obsolete_terms() -> None:
    text = (
        "format-version: 1.2\n\n"
        "[Term]\nid: GO:1\nname: binding\nnamespace: molecular_function\n\n"
        "[Term]\nid: GO:2\nname: gone\nnamespace: molecular_function\nis_obsolete: true\n\n"
        "[Typedef]\nid: part_of\nname: part of\n"
    )
    terms = parse_obo(text)
    assert set(terms) == {"GO:1"}
    assert terms["GO:1"].aspect == "molecular_function"


def test_the_term_store_reads_a_line_when_asked(tmp_path: Path) -> None:
    table = tmp_path / "hit_terms.tsv"
    table.write_bytes(b"P1\tGO:1;GO:2\nP2\t\nP3\tGO:9\n")
    store = TermStore(table)
    assert len(store) == 3
    assert store["P1"] == frozenset({"GO:1", "GO:2"})
    assert store["P2"] == frozenset()
    assert "P4" not in store
