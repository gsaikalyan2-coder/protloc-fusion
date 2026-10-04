"""Phase 13 homology evidence: parse hits, mask leakage, filter, transfer GO terms, run MMseqs2.

Layer L5 (evidence). Standard library only: no numpy, no torch, and nothing from ``training``,
``calibration`` or ``models``. Homology is an independent, second verdict (ADR-0003, ADR-0015):

* nothing here reads a model score, a calibrator or a threshold of any head, and no function here
  returns anything that could feed one;
* a term transferred from a hit is ``source = "homology"`` and carries the accession it came
  from; it is never mixed with a model term;
* a scored query must never be rewarded for finding itself or its own cluster: :func:`mask_hits`
  removes the self hit (by accession and by sequence hash) and every database sequence in the
  query's own ``cluster_id`` and partition, and reports how many hits each mask removed.

MMseqs2 is an external binary (MIT), never linked. :func:`run_measured` runs one command and
measures wall time, CPU seconds and peak memory of the process and every child it starts: a
Windows job object on Windows, ``wait4`` resource usage elsewhere. Nothing is estimated; a figure
the platform cannot measure is ``None`` and the ``method`` says why.
"""

from collections.abc import Collection, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

SOURCE_HOMOLOGY = "homology"
DATABASE_NAME = "swissprot"
#: The columns ``mmseqs convertalis --format-output`` is asked for, in order.
HIT_COLUMNS: tuple[str, ...] = (
    "query",
    "target",
    "fident",
    "qcov",
    "tcov",
    "evalue",
    "bits",
    "alnlen",
)
ENTRY_COLUMNS: tuple[str, ...] = ("accession", "sha256", "partition", "cluster_id")
MASK_SELF_ACCESSION = "self_accession"
MASK_SELF_SEQUENCE = "self_sequence_sha256"
MASK_SAME_CLUSTER = "same_cluster_id"
MASK_SAME_PARTITION = "same_partition"
#: The four masks, in the order a hit is attributed to the first one that matches.
MASKS: tuple[str, ...] = (
    MASK_SELF_ACCESSION,
    MASK_SELF_SEQUENCE,
    MASK_SAME_CLUSTER,
    MASK_SAME_PARTITION,
)
METHOD_JOB_OBJECT = "windows_job_object"
METHOD_WAIT4 = "wait4_rusage"
METHOD_NONE = "unavailable"
_FILETIME_TICKS_PER_SECOND = 10_000_000
_KIB = 1024
_JOB_BASIC_ACCOUNTING = 1
_JOB_PROCESS_ID_LIST = 3
_JOB_EXTENDED_LIMIT = 9
_PROCESS_QUERY_LIMITED = 0x1000
_PROCESS_VM_READ = 0x0010
_MAX_JOB_PIDS = 256
_LOG_TAIL = 2000
_TSV_WIDTH = len(HIT_COLUMNS)
_ENTRY_WIDTH = len(ENTRY_COLUMNS)
_UNIT_INTERVAL = (0.0, 1.0)


class HomologyError(ValueError):
    """A hit, an entry, a mask or an engine call is unsound."""


class EngineError(HomologyError):
    """The external MMseqs2 binary failed or could not be measured."""


# -- hits and entries ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Entry:
    """A sequence in the database or a query: who it is and where the split registry put it.

    ``partition`` and ``cluster_id`` are ``None`` for a database sequence that is not in the
    registry (it was never in a partition, so it cannot have been trained on or scored).
    """

    accession: str
    sha256: str
    partition: int | None
    cluster_id: str | None


@dataclass(frozen=True, slots=True)
class Hit:
    """One alignment of a query against a database sequence."""

    query: str
    target: str
    identity: float
    query_coverage: float
    target_coverage: float
    evalue: float
    bits: float
    alignment_length: int

    def rank_key(self) -> tuple[float, float, str]:
        """Best first: lower e-value, then higher bit score, then accession (deterministic)."""
        return (self.evalue, -self.bits, self.target)


@dataclass(frozen=True, slots=True)
class HitFilter:
    """One point of the hit-filter grid; ``top_k`` is how many hits vote."""

    e_value: float
    min_identity: float
    min_query_coverage: float
    top_k: int

    def keeps(self, hit: Hit) -> bool:
        """True when a hit passes the e-value, identity and query-coverage cut-offs."""
        return (
            hit.evalue <= self.e_value
            and hit.identity >= self.min_identity
            and hit.query_coverage >= self.min_query_coverage
        )

    def as_dict(self) -> dict[str, float | int]:
        """The setting as plain JSON-able values, in grid order."""
        return {
            "e_value": self.e_value,
            "min_identity": self.min_identity,
            "min_query_coverage": self.min_query_coverage,
            "top_k": self.top_k,
        }


def _unit(value: float, what: str) -> float:
    low, high = _UNIT_INTERVAL
    if not low <= value <= high:
        raise HomologyError(f"{what} {value} is outside [0, 1]: the engine output is not fractions")
    return value


def parse_hits(text: str) -> dict[str, list[Hit]]:
    """Read ``convertalis`` output (no header, tab separated) into hits grouped by query."""
    grouped: dict[str, list[Hit]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        cells = line.split("\t")
        if len(cells) != _TSV_WIDTH:
            raise HomologyError(
                f"hit line {number} has {len(cells)} columns, expected {_TSV_WIDTH}"
            )
        try:
            hit = Hit(
                query=cells[0],
                target=cells[1],
                identity=_unit(float(cells[2]), "identity"),
                query_coverage=_unit(float(cells[3]), "query coverage"),
                target_coverage=_unit(float(cells[4]), "target coverage"),
                evalue=float(cells[5]),
                bits=float(cells[6]),
                alignment_length=int(cells[7]),
            )
        except ValueError as error:
            raise HomologyError(f"hit line {number} is unreadable: {error}") from error
        grouped.setdefault(hit.query, []).append(hit)
    return grouped


def entries_text(entries: Sequence[Entry]) -> str:
    """The entry table as TSV text, in the order given (the caller sorts for determinism)."""
    lines = ["\t".join(ENTRY_COLUMNS)]
    for entry in entries:
        partition = "" if entry.partition is None else str(entry.partition)
        lines.append("\t".join((entry.accession, entry.sha256, partition, entry.cluster_id or "")))
    return "\n".join(lines) + "\n"


def parse_entries(text: str) -> dict[str, Entry]:
    """Read a table written by :func:`entries_text`; anything else is refused."""
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != ENTRY_COLUMNS:
        raise HomologyError("the entry table has an unexpected header")
    entries: dict[str, Entry] = {}
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != _ENTRY_WIDTH:
            raise HomologyError(f"entry line {number} has {len(cells)} columns")
        accession, sha, partition, cluster = cells
        if accession in entries:
            raise HomologyError(f"entry {accession} appears twice")
        entries[accession] = Entry(
            accession=accession,
            sha256=sha,
            partition=int(partition) if partition else None,
            cluster_id=cluster or None,
        )
    return entries


# -- masking ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MaskReport:
    """What masking did to one query's hits (or, summed, to a whole query set).

    ``exclusive`` attributes each removed hit to the first mask in :data:`MASKS` that matched, so
    the numbers add up to ``raw - kept``. ``independent`` counts the hits each mask alone would
    have removed, so overlaps between masks are visible.
    """

    raw: int
    kept: int
    exclusive: Mapping[str, int]
    independent: Mapping[str, int]


def matched_masks(query: Entry, target: Entry) -> tuple[str, ...]:
    """Every mask that removes ``target`` as a hit of ``query``, in :data:`MASKS` order."""
    found: list[str] = []
    if target.accession == query.accession:
        found.append(MASK_SELF_ACCESSION)
    if target.sha256 == query.sha256:
        found.append(MASK_SELF_SEQUENCE)
    if target.cluster_id is not None and target.cluster_id == query.cluster_id:
        found.append(MASK_SAME_CLUSTER)
    if target.partition is not None and target.partition == query.partition:
        found.append(MASK_SAME_PARTITION)
    return tuple(found)


def mask_hits(
    query: Entry, hits: Sequence[Hit], database: Mapping[str, Entry]
) -> tuple[list[Hit], MaskReport]:
    """Drop the hits a scored query must not see; return the rest and the per-mask counts.

    The query must be in the split registry (it has a partition and a cluster): an unregistered
    query cannot be scored, so masking it would silently mask nothing.
    """
    if query.partition is None or query.cluster_id is None:
        raise HomologyError(f"{query.accession} is not in the split registry; it cannot be scored")
    exclusive = dict.fromkeys(MASKS, 0)
    independent = dict.fromkeys(MASKS, 0)
    kept: list[Hit] = []
    for hit in hits:
        target = database.get(hit.target)
        if target is None:
            raise HomologyError(f"hit target {hit.target} is not in the database table")
        matched = matched_masks(query, target)
        for name in matched:
            independent[name] += 1
        if matched:
            exclusive[matched[0]] += 1
        else:
            kept.append(hit)
    return kept, MaskReport(len(hits), len(kept), exclusive, independent)


def add_reports(reports: Collection[MaskReport]) -> MaskReport:
    """Sum per-query reports into one report for the whole query set."""
    return MaskReport(
        raw=sum(r.raw for r in reports),
        kept=sum(r.kept for r in reports),
        exclusive={m: sum(r.exclusive[m] for r in reports) for m in MASKS},
        independent={m: sum(r.independent[m] for r in reports) for m in MASKS},
    )


def in_registry(hits: Sequence[Hit], database: Mapping[str, Entry]) -> list[Hit]:
    """Only the hits whose target is in the split registry (the diagnostic arm's restriction)."""
    return [h for h in hits if database[h.target].partition is not None]


# -- filtering and GO transfer ---------------------------------------------------------
def passing(hits: Sequence[Hit], rule: HitFilter) -> list[Hit]:
    """The hits that pass the filter, best first."""
    return sorted((h for h in hits if rule.keeps(h)), key=Hit.rank_key)


def choose(
    ranked: Sequence[Hit],
    top_k: int,
    terms_of: Mapping[str, frozenset[str]],
    annotated_only: bool,
) -> list[Hit]:
    """The first ``top_k`` of the ranked, filtered hits that may vote.

    With ``annotated_only`` a hit whose own annotation is empty (absent from ``terms_of`` or an
    empty set) has nothing to transfer and is skipped; otherwise it takes a place and votes for
    nothing.
    """
    pool = [h for h in ranked if terms_of.get(h.target)] if annotated_only else list(ranked)
    return pool[:top_k]


def candidates(
    hits: Sequence[Hit],
    rule: HitFilter,
    terms_of: Mapping[str, frozenset[str]],
    annotated_only: bool = True,
) -> list[Hit]:
    """The hits that vote under a filter: the top ``rule.top_k`` passing hits, best first."""
    return choose(passing(hits, rule), rule.top_k, terms_of, annotated_only)


def vote(
    chosen: Sequence[Hit],
    terms_of: Mapping[str, frozenset[str]],
    vocabulary: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per term, the fraction of the chosen hits that carry it; nothing if there are no hits.

    ``vocabulary`` restricts the terms (scoring uses the Phase 12 vocabulary); ``None`` keeps
    every term a chosen hit carries.
    """
    if not chosen:
        return {}
    counts: dict[str, int] = {}
    for hit in chosen:
        for term in terms_of.get(hit.target, frozenset()):
            counts[term] = counts.get(term, 0) + 1
    wanted = None if vocabulary is None else set(vocabulary)
    return {t: n / len(chosen) for t, n in counts.items() if wanted is None or t in wanted}


@dataclass(frozen=True, slots=True)
class TransferredTerm:
    """A GO term carried over from a homolog; ``source`` is always ``"homology"``."""

    term: str
    score: float
    source: str
    source_accession: str
    identity: float
    evalue: float
    database: str


def transfer_terms(
    chosen: Sequence[Hit],
    terms_of: Mapping[str, frozenset[str]],
    vocabulary: Sequence[str] | None = None,
) -> list[TransferredTerm]:
    """Terms voted for by the chosen hits, best supported first, each naming the hit it came from.

    ``source_accession`` is the best-ranked chosen hit that carries the term. The result is never
    merged with a model prediction: the two stay in separate fields (CLAUDE.md, ADR-0014 §5).
    """
    votes = vote(chosen, terms_of, vocabulary)
    out: list[TransferredTerm] = []
    for term, score in sorted(votes.items(), key=lambda item: (-item[1], item[0])):
        support = next(h for h in chosen if term in terms_of.get(h.target, frozenset()))
        out.append(
            TransferredTerm(
                term=term,
                score=score,
                source=SOURCE_HOMOLOGY,
                source_accession=support.target,
                identity=support.identity,
                evalue=support.evalue,
                database=DATABASE_NAME,
            )
        )
    return out


# -- small statistics ------------------------------------------------------------------
def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile (``percentile`` in (0, 100]) of a non-empty sample."""
    if not values:
        raise HomologyError("a percentile needs at least one value")
    low, high = 0.0, 100.0
    if not low < percentile <= high:
        raise HomologyError("percentile must be in (0, 100]")
    ordered = sorted(values)
    rank = -(-len(ordered) * percentile // high)
    return float(ordered[int(rank) - 1])


def identity_bin(value: float, edges: Sequence[float]) -> int | None:
    """Index of the bin ``[edge_i, edge_i+1)`` holding ``value`` (the last bin is closed)."""
    if not edges[0] <= value <= edges[-1]:
        return None
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        if edges[i] <= value < edges[i + 1] or (last and value == edges[-1]):
            return i
    return None


def bin_label(edges: Sequence[float], index: int) -> str:
    """A readable label for bin ``index``."""
    closing = "]" if index == len(edges) - 2 else ")"
    return f"[{edges[index]:g}, {edges[index + 1]:g}{closing}"


# -- the engine ------------------------------------------------------------------------
def engine_env(executable: Path) -> dict[str, str]:
    """The environment the Windows build needs: its own ``bin`` first on PATH, no DOS warnings."""
    env = dict(os.environ)
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    env["CYGWIN"] = "nodosfilewarning"
    return env


def engine_path(path: Path) -> str:
    """A path as the engine wants it: forward slashes, which Cygwin and Linux both read."""
    return path.resolve().as_posix()


def createdb_command(exe: Path, fasta: Path, database: Path, verbosity: int) -> list[str]:
    """``mmseqs createdb``."""
    return [
        engine_path(exe),
        "createdb",
        engine_path(fasta),
        engine_path(database),
        "-v",
        str(verbosity),
    ]


def createindex_command(
    exe: Path, database: Path, tmp: Path, threads: int, verbosity: int
) -> list[str]:
    """``mmseqs createindex``; the engine removes its own temporary files."""
    return [
        engine_path(exe),
        "createindex",
        engine_path(database),
        engine_path(tmp),
        "--threads",
        str(threads),
        "--remove-tmp-files",
        "1",
        "-v",
        str(verbosity),
    ]


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """The ``mmseqs search`` settings (``configs/training/homology.yaml``)."""

    sensitivity: float
    e_value: float
    max_seqs: int
    alignment_mode: int
    threads: int
    verbosity: int


def search_command(
    exe: Path, paths: tuple[Path, Path, Path, Path], settings: SearchSettings
) -> list[str]:
    """``mmseqs search`` for ``(query db, target db, result db, tmp)``."""
    query, target, result, tmp = paths
    return [
        engine_path(exe),
        "search",
        engine_path(query),
        engine_path(target),
        engine_path(result),
        engine_path(tmp),
        "-s",
        str(settings.sensitivity),
        "-e",
        str(settings.e_value),
        "--max-seqs",
        str(settings.max_seqs),
        "--alignment-mode",
        str(settings.alignment_mode),
        "--threads",
        str(settings.threads),
        "--remove-tmp-files",
        "1",
        "-v",
        str(settings.verbosity),
    ]


def convertalis_command(
    exe: Path, paths: tuple[Path, Path, Path, Path], threads: int, verbosity: int
) -> list[str]:
    """``mmseqs convertalis`` for ``(query db, target db, result db, output tsv)``."""
    query, target, result, out = paths
    return [
        engine_path(exe),
        "convertalis",
        engine_path(query),
        engine_path(target),
        engine_path(result),
        engine_path(out),
        "--format-output",
        ",".join(HIT_COLUMNS),
        "--threads",
        str(threads),
        "-v",
        str(verbosity),
    ]


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one engine command cost. ``None`` means the platform could not measure it."""

    seconds: float
    cpu_seconds: float | None
    peak_commit_bytes: int | None
    peak_working_set_bytes: int | None
    method: str
    returncode: int
    samples: int

    def as_dict(self) -> dict[str, Any]:
        """The measurement as plain JSON-able values."""
        return {
            "seconds": self.seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_commit_bytes": self.peak_commit_bytes,
            "peak_working_set_bytes": self.peak_working_set_bytes,
            "method": self.method,
            "returncode": self.returncode,
            "samples": self.samples,
        }


class _ExtendedLimit(ctypes.Structure):
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` (x64 layout)."""

    _fields_ = (
        ("per_process_user_time", ctypes.c_int64),
        ("per_job_user_time", ctypes.c_int64),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    )


class _Accounting(ctypes.Structure):
    """``JOBOBJECT_BASIC_ACCOUNTING_INFORMATION``."""

    _fields_ = (
        ("total_user_time", ctypes.c_int64),
        ("total_kernel_time", ctypes.c_int64),
        ("this_period_user_time", ctypes.c_int64),
        ("this_period_kernel_time", ctypes.c_int64),
        ("total_page_faults", ctypes.c_uint32),
        ("total_processes", ctypes.c_uint32),
        ("active_processes", ctypes.c_uint32),
        ("total_terminated", ctypes.c_uint32),
    )


class _PidList(ctypes.Structure):
    """``JOBOBJECT_BASIC_PROCESS_ID_LIST`` with room for :data:`_MAX_JOB_PIDS` ids."""

    _fields_ = (
        ("assigned", ctypes.c_uint32),
        ("listed", ctypes.c_uint32),
        ("ids", ctypes.c_size_t * _MAX_JOB_PIDS),
    )


class _MemoryCounters(ctypes.Structure):
    """``PROCESS_MEMORY_COUNTERS``."""

    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("page_faults", ctypes.c_uint32),
        ("peak_working_set", ctypes.c_size_t),
        ("working_set", ctypes.c_size_t),
        ("quota_peak_paged", ctypes.c_size_t),
        ("quota_paged", ctypes.c_size_t),
        ("quota_peak_nonpaged", ctypes.c_size_t),
        ("quota_nonpaged", ctypes.c_size_t),
        ("pagefile", ctypes.c_size_t),
        ("peak_pagefile", ctypes.c_size_t),
    )


class _WindowsJob:
    """A job object holding the engine process and every child it starts (Windows only)."""

    def __init__(self) -> None:
        """Create the job and bind the Win32 calls."""
        win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 - absent on non-Windows platforms
        self._k32 = win_dll("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.restype = ctypes.c_void_p
        self._k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        self._k32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self._k32.QueryInformationJobObject.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self._k32.OpenProcess.restype = ctypes.c_void_p
        self._k32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        self._k32.K32GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        self._k32.CloseHandle.argtypes = (ctypes.c_void_p,)
        self._handle = self._k32.CreateJobObjectW(None, None)
        if not self._handle:
            raise EngineError("CreateJobObjectW failed")
        self.peak_working_set = 0
        self.samples = 0

    def assign(self, process: "subprocess.Popen[Any]") -> None:
        """Put a just-started process (and so its future children) into the job."""
        raw = int(getattr(process, "_handle"))  # noqa: B009 - the only way to reach the handle
        if not self._k32.AssignProcessToJobObject(self._handle, raw):
            raise EngineError("AssignProcessToJobObject failed")

    def _pids(self) -> list[int]:
        listing = _PidList()
        ok = self._k32.QueryInformationJobObject(
            self._handle, _JOB_PROCESS_ID_LIST, ctypes.byref(listing), ctypes.sizeof(listing), None
        )
        return [int(listing.ids[i]) for i in range(listing.listed)] if ok else []

    def sample(self) -> None:
        """Add up the working sets of the processes in the job right now; keep the maximum."""
        total = 0
        for pid in self._pids():
            handle = self._k32.OpenProcess(_PROCESS_QUERY_LIMITED | _PROCESS_VM_READ, 0, pid)
            if not handle:
                continue
            counters = _MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if self._k32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                total += int(counters.working_set)
                self.peak_working_set = max(self.peak_working_set, int(counters.peak_working_set))
            self._k32.CloseHandle(handle)
        self.peak_working_set = max(self.peak_working_set, total)
        self.samples += 1

    def totals(self) -> tuple[float, int]:
        """CPU seconds (user + kernel) and peak commit bytes of every process the job ever held."""
        accounting = _Accounting()
        if not self._k32.QueryInformationJobObject(
            self._handle,
            _JOB_BASIC_ACCOUNTING,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            raise EngineError("QueryInformationJobObject (accounting) failed")
        limit = _ExtendedLimit()
        if not self._k32.QueryInformationJobObject(
            self._handle, _JOB_EXTENDED_LIMIT, ctypes.byref(limit), ctypes.sizeof(limit), None
        ):
            raise EngineError("QueryInformationJobObject (limits) failed")
        ticks = int(accounting.total_user_time) + int(accounting.total_kernel_time)
        return ticks / _FILETIME_TICKS_PER_SECOND, int(limit.peak_job_memory_used)

    def close(self) -> None:
        """Release the job handle."""
        self._k32.CloseHandle(self._handle)


def _measure_windows(
    command: Sequence[str], env: Mapping[str, str], log: Path, poll: float
) -> Measurement:
    began = time.perf_counter()
    job = _WindowsJob()
    try:
        with log.open("wb") as sink:
            process = subprocess.Popen(  # noqa: S603 - the command is built here, never from input
                list(command), env=dict(env), stdout=sink, stderr=subprocess.STDOUT
            )
            job.assign(process)
            while process.poll() is None:
                job.sample()
                time.sleep(poll)
        cpu, commit = job.totals()
        return Measurement(
            seconds=time.perf_counter() - began,
            cpu_seconds=cpu,
            peak_commit_bytes=commit,
            peak_working_set_bytes=job.peak_working_set or None,
            method=METHOD_JOB_OBJECT,
            returncode=int(process.returncode),
            samples=job.samples,
        )
    finally:
        job.close()


def _measure_posix(command: Sequence[str], env: Mapping[str, str], log: Path) -> Measurement:
    began = time.perf_counter()
    with log.open("wb") as sink:
        process = subprocess.Popen(  # noqa: S603 - the command is built here, never from input
            list(command), env=dict(env), stdout=sink, stderr=subprocess.STDOUT
        )
        wait4 = getattr(os, "wait4", None)
        if wait4 is None:
            code = process.wait()
            return Measurement(time.perf_counter() - began, None, None, None, METHOD_NONE, code, 0)
        _pid, status, usage = wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    peak = int(usage.ru_maxrss)
    peak_bytes = peak if sys.platform == "darwin" else peak * _KIB
    return Measurement(
        seconds=time.perf_counter() - began,
        cpu_seconds=float(usage.ru_utime) + float(usage.ru_stime),
        peak_commit_bytes=None,
        peak_working_set_bytes=peak_bytes,
        method=METHOD_WAIT4,
        returncode=int(process.returncode),
        samples=0,
    )


def run_measured(
    command: Sequence[str], env: Mapping[str, str], log: Path, poll_interval: float
) -> Measurement:
    """Run one engine command, log its output, and measure what it cost.

    Raises :class:`EngineError` on a non-zero exit, with the end of the log in the message.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        measured = _measure_windows(command, env, log, poll_interval)
    else:
        measured = _measure_posix(command, env, log)
    if measured.returncode != 0:
        tail = log.read_bytes()[-_LOG_TAIL:].decode("utf-8", errors="replace")
        raise EngineError(
            f"{command[1] if len(command) > 1 else command[0]} exited {measured.returncode}: {tail}"
        )
    return measured


def engine_version(executable: Path, env: Mapping[str, str]) -> str:
    """What ``mmseqs version`` prints (the commit hash for a release build)."""
    result = subprocess.run(  # noqa: S603 - the command is built here, never from input
        [engine_path(executable), "version"],
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EngineError(
            f"mmseqs version exited {result.returncode}: {result.stderr[-_LOG_TAIL:]}"
        )
    return result.stdout.strip()


def free_bytes(directory: Path) -> int:
    """Free bytes on the drive holding ``directory``."""
    return int(shutil.disk_usage(directory).free)
