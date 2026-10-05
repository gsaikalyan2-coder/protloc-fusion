"""Serve-time MMseqs2 search with a wall-clock budget (Phase 14, ADR-0003, ADR-0015).

Layer L5 (evidence), standard library only. One call searches a set of query sequences against
the local Swiss-Prot index with the settings the evidence bundle froze: ``createdb`` ->
``search`` -> ``convertalis``, each a subprocess of the pinned binary, never a network call. The
whole call has one deadline; when it passes the process tree is killed and the caller gets
:class:`EvidenceUnavailable` with a reason, which the service turns into a visible degraded
state. Nothing here reads a model score, a calibrator or a threshold.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from protloc.evidence import homology
from protloc.evidence.homology import Hit

_KILL_GRACE_SECONDS = 5
_LOG_TAIL = 400


class EvidenceUnavailable(RuntimeError):  # noqa: N818
    """The search did not produce hits; ``str(error)`` is the reason shown to the user."""


@dataclass(frozen=True, slots=True)
class SearchPlan:
    """Everything one search needs; all of it from the bundle and ``configs/serving``."""

    executable: Path
    database: Path
    work_directory: Path
    settings: homology.SearchSettings
    threads: int
    verbosity: int
    base_timeout_seconds: float
    per_query_seconds: float


def timeout_for(plan: SearchPlan, queries: int) -> float:
    """The deadline of a search over ``queries`` sequences."""
    return plan.base_timeout_seconds + plan.per_query_seconds * max(0, queries - 1)


def _kill_tree(process: "subprocess.Popen[bytes]") -> None:
    """Kill the child and everything it started."""
    if sys.platform == "win32":
        subprocess.run(  # noqa: S603 - fixed argv, the pid is our own child
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],  # noqa: S607
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()


def _run(command: Sequence[str], env: Mapping[str, str], log: Path, deadline: float) -> None:
    """Run one command until ``deadline`` (a ``time.monotonic`` instant); raise on failure."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise EvidenceUnavailable("the evidence search ran out of time")
    with log.open("wb") as sink:
        extra: dict[str, object] = (
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(  # noqa: S603 - argv built from pinned paths, never from input
            list(command),
            env=dict(env),
            stdout=sink,
            stderr=subprocess.STDOUT,
            **extra,  # type: ignore[call-overload]
        )
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            _kill_tree(process)
            raise EvidenceUnavailable("the evidence search ran out of time") from error
    if code != 0:
        tail = log.read_bytes()[-_LOG_TAIL:].decode("utf-8", errors="replace")
        step = command[1] if len(command) > 1 else command[0]
        raise EvidenceUnavailable(f"mmseqs {step} exited {code}: {tail.strip()}")


def search(plan: SearchPlan, queries: Mapping[str, str]) -> dict[str, list[Hit]]:
    """Hits of every query, by query id; a query with no hit is absent from the result."""
    if not queries:
        return {}
    if not plan.executable.is_file():
        raise EvidenceUnavailable("the search binary is not installed on this host")
    if not (plan.executable.parent / "sh.exe").is_file() and sys.platform == "win32":
        raise EvidenceUnavailable("the search binary has no shell beside it (sh.exe)")
    deadline = time.monotonic() + timeout_for(plan, len(queries))
    env = homology.engine_env(plan.executable)
    plan.work_directory.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="search_", dir=plan.work_directory))
    try:
        fasta = work / "queries.fasta"
        names = {f"q{i}": key for i, key in enumerate(queries)}
        fasta.write_text(
            "".join(f">{alias}\n{queries[key]}\n" for alias, key in names.items()),
            encoding="ascii",
            newline="\n",
        )
        query_db, result, tmp = work / "q", work / "res", work / "tmp"
        out = work / "hits.tsv"
        tmp.mkdir()
        steps = (
            homology.createdb_command(plan.executable, fasta, query_db, plan.verbosity),
            homology.search_command(
                plan.executable, (query_db, plan.database, result, tmp), plan.settings
            ),
            homology.convertalis_command(
                plan.executable,
                (query_db, plan.database, result, out),
                plan.threads,
                plan.verbosity,
            ),
        )
        for number, command in enumerate(steps):
            _run(command, env, work / f"step{number}.log", deadline)
        try:
            grouped = homology.parse_hits(out.read_text(encoding="ascii"))
        except (OSError, homology.HomologyError) as error:
            raise EvidenceUnavailable(f"the search output is unreadable: {error}") from error
        return {names[alias]: hits for alias, hits in grouped.items() if alias in names}
    finally:
        shutil.rmtree(work, ignore_errors=True)
