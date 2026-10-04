#!/usr/bin/env python
"""Phase 13 homology evidence: an MMseqs2 index over Swiss-Prot and GO transfer from nearest hits.

A second, independent verdict beside the model: the nearest Swiss-Prot homologs of a query
(accession, identity, coverage, e-value) and the GO terms those homologs carry. Evidence never
feeds a model score, a calibrator or a threshold (ADR-0003, ADR-0015).

Modes (one at a time; configuration is ``configs/training/homology.yaml``):

    --index                   write the FASTA and entry table from the corpus, build and index the
                              MMseqs2 database; measure build time, index size and peak memory.
    --search [--smoke]        search the subset queries once, then time single-query runs (p50,
                              p95, CPU seconds per query).
    --evaluate [--smoke]      mask leakage, tune the hit filter on partitions 1-4 only, score the
                              homology arms against the Phase 12 model arms and the frequency
                              prior.
    --freeze                  write the registry bundle and homology_metrics.json.
    --evaluate --test         score partition 5 once; refused a second time.
    --check                   no data bytes: configuration, pins and, when present, the manifest.

Every number is for the 500-sequence subset queries and is labelled "subset (500), not comparable to
DeepLoc or to the Phase 8 full-corpus numbers"; the database is the full Swiss-Prot snapshot.
``--smoke`` (every 4th query, partitions 1-4) is a code check: it can never be frozen and no number
from it is reported.

Run from the repository root inside the project venv.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from types import ModuleType
from typing import Any

from protloc.data.manifest import read_json_object, sha256_file, write_json
from protloc.data.splits import read_registry
from protloc.evidence import homology
from protloc.evidence.homology import EngineError, Entry, Hit, HitFilter, HomologyError
from protloc.registry import evidence as registry_evidence
from protloc.training import baseline_models as bm
from protloc.training import go, homology_eval
from protloc.training.baselines import (
    SUBSET_LABEL,
    SUBSET_SCOPE,
    TEST_PARTITION,
    VALIDATION_PARTITIONS,
    BaselineError,
    digest,
    lopo_folds,
    read_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOMOLOGY_CONFIG = REPO_ROOT / "configs" / "training" / "homology.yaml"
GO_CONFIG = REPO_ROOT / "configs" / "training" / "go.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "default.yaml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "homology_metrics.json"
GO_MANIFEST = REPO_ROOT / "data" / "manifests" / "go_metrics.json"
HEAD_MANIFEST = REPO_ROOT / "data" / "manifests" / "head_metrics.json"
CRF_MANIFEST = REPO_ROOT / "data" / "manifests" / "crf_metrics.json"
REGISTRY = REPO_ROOT / "data" / "processed" / "splits" / "registry.tsv"
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
GO_DIR = REPO_ROOT / "data" / "go"
EVIDENCE = REPO_ROOT / "data" / "evidence"
INDEX_FILE = "index.json"
SEARCH_FILE = "search.json"
VALIDATION_FILE = "validation.json"
QUERIES_FASTA = "queries.fasta"
QUERIES_LIST = "queries.txt"
HITS_FILE = "hits_raw.tsv"
HIT_TERMS_FILE = "hit_terms.tsv"
HIT_TERMS_META = "hit_terms.json"
THRESHOLD_RULE = (
    "one global threshold per arm, tuned for Fmax on the other partitions (nested, ADR-0011); the "
    "frozen threshold on partitions 1-4"
)
SELECTION_RULE = (
    "hit filter: higher mean fold precision@primary_k, then higher mean fold Fmax, then the "
    "earlier grid point; partitions 1-4 only; the best-hit arm is tuned over top_k = 1 points"
)
BIN_ARMS = (
    homology_eval.ARM_VOTE,
    homology_eval.ARM_BEST,
    homology_eval.ARM_PRIOR,
    homology_eval.ARM_RAW,
    homology_eval.ARM_CALIBRATED,
)


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _np() -> Any:
    return importlib.import_module("numpy")


def _go_script() -> ModuleType:
    """``scripts/go.py`` as a module: its subset, label and manifest helpers are reused."""
    spec = importlib.util.spec_from_file_location("protloc_go_script", REPO_ROOT / "scripts/go.py")
    if spec is None or spec.loader is None:
        raise BaselineError("scripts/go.py cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> homology_eval.HomologyConfig:
    config = homology_eval.load_config(HOMOLOGY_CONFIG)
    root = importlib.import_module("yaml").safe_load(
        (REPO_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    )
    if config.seed != int(root["seed"]):
        raise BaselineError("homology.yaml seed differs from configs/config.yaml")
    return config


def _stage_dir(smoke: bool) -> Path:
    return EVIDENCE / "smoke" if smoke else EVIDENCE


def _announce() -> None:
    _say(f"scope: {SUBSET_LABEL}")


def _exe(config: homology_eval.HomologyConfig) -> Path:
    return REPO_ROOT / config.engine.executable


def _verify_engine(config: homology_eval.HomologyConfig) -> dict[str, str]:
    """The binary must be the pinned one: executable hash and the commit it prints."""
    exe = _exe(config)
    if not exe.is_file():
        raise BaselineError(f"{exe} is missing: run the MMseqs2 download block first")
    sha = sha256_file(exe)[0]
    if sha != config.engine.executable_sha256:
        raise BaselineError("mmseqs.exe is not the executable homology.yaml pins")
    printed = homology.engine_version(exe, homology.engine_env(exe))
    if printed != config.engine.commit:
        raise BaselineError(f"mmseqs prints {printed!r}, not the pinned commit")
    return {"executable_sha256": sha, "version_output": printed, **_ensure_shell(exe)}


def _ensure_shell(exe: Path) -> dict[str, str]:
    """MMseqs2 workflows run ``#!/bin/sh`` scripts; the Windows release ships only busybox.exe.

    ``sh.exe`` must be a byte-for-byte copy of the shipped ``busybox.exe`` (Gate D, ADR-0015); it
    is created once beside the pinned executable and its hash is recorded with the index.
    """
    busybox, shell = exe.parent / "busybox.exe", exe.parent / "sh.exe"
    if not busybox.is_file():
        raise BaselineError(f"{busybox} is missing: the release has no shell to run workflows")
    if not shell.exists():
        shutil.copyfile(busybox, shell)
    if sha256_file(shell)[0] != sha256_file(busybox)[0]:
        raise BaselineError("sh.exe is not a copy of the shipped busybox.exe")
    return {"busybox_sha256": sha256_file(busybox)[0]}


def _db_path(config: homology_eval.HomologyConfig) -> Path:
    return EVIDENCE / config.database.mmseqs_directory / config.database.mmseqs_name


# -- --index ---------------------------------------------------------------------------
def _registry_table() -> dict[str, tuple[int, str]]:
    return {r["accession"]: (int(r["partition"]), r["cluster_id"]) for r in read_registry(REGISTRY)}


def _write_database_files(config: homology_eval.HomologyConfig) -> int:
    """FASTA (accession headers) and the entry table, streamed from the corpus; returns records."""
    table = _registry_table()
    fasta = EVIDENCE / config.database.fasta
    entries: list[Entry] = []
    with fasta.open("w", encoding="ascii", newline="\n") as out, CORPUS.open("rb") as corpus:
        for line in corpus:
            record = json.loads(line)
            accession = str(record["accession"])
            sequence = str(record["sequence_normalized"])
            out.write(f">{accession}\n{sequence}\n")
            part, cluster = table.get(accession, (None, None))
            entries.append(Entry(accession, str(record["sequence_sha256"]), part, cluster))
    entries.sort(key=lambda e: e.accession)
    meta = EVIDENCE / config.database.meta
    meta.write_text(homology.entries_text(entries), encoding="ascii", newline="\n")
    return len(entries)


def _index_summary(directory: Path) -> dict[str, Any]:
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    sizes = {name: (directory / name).stat().st_size for name in files}
    index_bytes = sum(n for name, n in sizes.items() if ".idx" in name)
    return {
        "names": files,
        "total_bytes": sum(sizes.values()),
        "index_bytes": index_bytes,
        "database_bytes": sum(sizes.values()) - index_bytes,
    }


def run_index(args: argparse.Namespace) -> int:
    """Write the database files, build the MMseqs2 database and index; measure what it cost."""
    del args
    config = _config()
    if (EVIDENCE / INDEX_FILE).exists():
        _say(f"{INDEX_FILE} exists; an index is never rebuilt over an existing one")
        return 1
    engine = _verify_engine(config)
    if sha256_file(CORPUS)[0] != config.database.corpus_sha256:
        raise BaselineError("data/processed/corpus.jsonl is not the corpus homology.yaml pins")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    records = _write_database_files(config)
    if records != config.database.records:
        raise BaselineError(
            f"the corpus holds {records} records, homology.yaml pins {config.database.records}"
        )
    _say(f"wrote {records} sequences in {time.perf_counter() - began:.1f} s")
    exe, env = _exe(config), homology.engine_env(_exe(config))
    db = _db_path(config)
    db.parent.mkdir(parents=True, exist_ok=True)
    poll = config.poll_interval
    try:
        made = homology.run_measured(
            homology.createdb_command(
                exe, EVIDENCE / config.database.fasta, db, config.engine.verbosity
            ),
            env,
            EVIDENCE / "logs" / "createdb.log",
            poll,
        )
        free = homology.free_bytes(EVIDENCE)
        _say(f"createdb: {made.seconds:.1f} s; free disk {free / 2**30:.2f} GiB")
        if free < config.min_free_bytes:
            _say(
                f"STOP (Gate D): {free} bytes free, fewer than index.min_free_bytes "
                f"{config.min_free_bytes}. The k-mer index may not fit. Nothing was deleted; ask."
            )
            return 1
        tmp = EVIDENCE / "tmp_index"
        tmp.mkdir(exist_ok=True)
        indexed = homology.run_measured(
            homology.createindex_command(
                exe, db, tmp, config.engine.threads, config.engine.verbosity
            ),
            env,
            EVIDENCE / "logs" / "createindex.log",
            poll,
        )
    except EngineError as error:
        _say(f"STOP (Gate D): the index failed: {error}")
        return 1
    summary = _index_summary(db.parent)
    files = registry_evidence.index_files(db.parent, summary["names"])
    peaks = [m.peak_working_set_bytes for m in (made, indexed) if m.peak_working_set_bytes]
    commits = [m.peak_commit_bytes for m in (made, indexed) if m.peak_commit_bytes]
    document: dict[str, Any] = {
        "engine": {**engine, "release": config.engine.release},
        "database": {
            "records": records,
            "fasta_sha256": sha256_file(EVIDENCE / config.database.fasta)[0],
            "meta_sha256": sha256_file(EVIDENCE / config.database.meta)[0],
            "corpus_sha256": config.database.corpus_sha256,
        },
        "steps": {"createdb": made.as_dict(), "createindex": indexed.as_dict()},
        "build": {
            "seconds": made.seconds + indexed.seconds,
            "createindex_seconds": indexed.seconds,
            "index_bytes": summary["index_bytes"],
            "database_bytes": summary["database_bytes"],
            "total_bytes": summary["total_bytes"],
            "peak_working_set_bytes": max(peaks) if peaks else None,
            "peak_commit_bytes": max(commits) if commits else None,
            "method": indexed.method,
            "threads": config.engine.threads,
            "free_bytes_after": homology.free_bytes(EVIDENCE),
        },
        "files": files,
        "index_sha256": registry_evidence.index_digest(files),
        "hardware": bm.hardware(),
    }
    write_json(EVIDENCE / INDEX_FILE, document)
    build = document["build"]
    _say(
        f"index built: {build['seconds']:.1f} s; index {build['index_bytes'] / 2**30:.2f} GiB, "
        f"database {build['database_bytes'] / 2**30:.2f} GiB; peak working set "
        f"{_gib(build['peak_working_set_bytes'])}, peak commit {_gib(build['peak_commit_bytes'])} "
        f"({build['method']})"
    )
    _say(f"index_sha256 {document['index_sha256'][:12]}")
    return 0


def _gib(value: int | None) -> str:
    return "not measured" if value is None else f"{value / 2**30:.2f} GiB"


# -- the session: subset, labels, database, hit terms ----------------------------------
@dataclass(frozen=True, slots=True)
class Session:
    """Everything a data stage needs, loaded and cross-checked."""

    config: homology_eval.HomologyConfig
    go_config: go.GoConfig
    script: ModuleType
    sub: Any
    labels: dict[str, go.LabelRow]
    label_info: dict[str, Any]
    vocab: go.Vocabulary
    database: dict[str, Entry]
    terms_of: dict[str, frozenset[str]]
    smoke: bool

    @property
    def stage(self) -> Path:
        """Where this run's stage files live."""
        return _stage_dir(self.smoke)


def _hit_terms(
    config: homology_eval.HomologyConfig, go_config: go.GoConfig, accessions: Sequence[str]
) -> dict[str, frozenset[str]]:
    """GO terms of every database sequence: the Phase 12 labels, mapped by the Phase 12 code.

    Cached beside the index; the cache is valid only for this corpus, ontology and go.yaml.
    """
    wanted = {
        "corpus_sha256": config.database.corpus_sha256,
        "ontology_sha256": go_config.ontology.sha256,
        "go_config_sha256": go_config.sha256,
        "records": len(accessions),
    }
    table, meta = EVIDENCE / HIT_TERMS_FILE, EVIDENCE / HIT_TERMS_META
    if table.exists() and meta.exists():
        info = read_json_object(meta)
        if {k: info.get(k) for k in wanted} == wanted and info.get("sha256") == (
            sha256_file(table)[0]
        ):
            return _read_terms(table)
    onto = go.load_ontology(REPO_ROOT / go_config.ontology.file, go_config.ontology)
    ctx = go.LabelContext(onto, go.Ancestry(onto, go_config.ontology), go_config)
    scan = go.scan_corpus(CORPUS, accessions)
    terms: dict[str, frozenset[str]] = {}
    for accession, record in sorted(scan.records.items()):
        row = go.map_record(ctx, record, ())
        terms[accession] = frozenset(row.propagated if row.state == go.STATE_ANNOTATED else ())
    lines = [f"{a}\t{';'.join(sorted(t))}" for a, t in sorted(terms.items())]
    table.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    annotated = sum(1 for t in terms.values() if t)
    write_json(meta, {**wanted, "sha256": sha256_file(table)[0], "annotated": annotated})
    return terms


def _read_terms(path: Path) -> dict[str, frozenset[str]]:
    terms: dict[str, frozenset[str]] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        accession, _tab, cell = line.partition("\t")
        terms[accession] = frozenset(t for t in cell.split(";") if t)
    return terms


def _session(args: argparse.Namespace) -> Session:
    """Load the subset, labels, vocabulary, database table and hit terms; check them together."""
    script = _go_script()
    config = _config()
    sub = script._subset(argparse.Namespace(smoke=False))
    go_config = sub.config
    info_path, tsv = GO_DIR / "labels.json", GO_DIR / "labels.tsv"
    if not (tsv.exists() and info_path.exists() and (GO_DIR / "vocabulary.json").exists()):
        raise BaselineError("no GO label table: run `python scripts/go.py --labels` first")
    info = read_manifest(info_path)
    wanted = script._label_fingerprint(sub, go_config.ontology.sha256)
    if {k: info.get(k) for k in wanted} != wanted:
        raise BaselineError("the GO label table is stale (inputs changed): rerun go.py --labels")
    if sha256_file(tsv)[0] != info["label_table_sha256"]:
        raise BaselineError("labels.tsv does not match labels.json")
    labels = go.parse_label_table(tsv.read_text(encoding="utf-8"))
    vocab = go.vocabulary_from_document(read_json_object(GO_DIR / "vocabulary.json"))
    meta = EVIDENCE / config.database.meta
    if not meta.exists():
        raise BaselineError("no database table: run `python scripts/homology.py --index` first")
    if sha256_file(meta)[0] != read_manifest(EVIDENCE / INDEX_FILE)["database"]["meta_sha256"]:
        raise BaselineError("db_meta.tsv is not the one the index was built with")
    database = homology.parse_entries(meta.read_text(encoding="ascii"))
    terms = _hit_terms(config, go_config, sorted(database))
    return Session(
        config, go_config, script, sub, labels, info, vocab, database, terms, bool(args.smoke)
    )


def _rows(session: Session, partitions: Sequence[int]) -> go.Rows:
    stride = session.config.smoke.query_stride if session.smoke else 1
    spec = (session.vocab, partitions, stride)
    return go.build_rows(
        session.sub.table, session.sub.order, session.sub.member, session.labels, spec
    )


def _query_accessions(session: Session) -> list[str]:
    """Full run: every subset sequence. Smoke: every 4th labelled row of partitions 1-4."""
    if session.smoke:
        return sorted(_rows(session, VALIDATION_PARTITIONS).accession)
    accessions: list[str] = session.script._subset_accessions(session.sub)
    return accessions


def _read_fasta(path: Path, wanted: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with path.open(encoding="ascii") as handle:
        for header in handle:
            sequence = handle.readline().strip()
            accession = header[1:].strip()
            if accession in wanted:
                found[accession] = sequence
    if set(found) != wanted:
        raise BaselineError(f"{len(wanted - set(found))} query sequences are not in the FASTA")
    return found


# -- --search --------------------------------------------------------------------------
def _write_fasta(path: Path, sequences: Mapping[str, str]) -> None:
    body = "".join(f">{a}\n{s}\n" for a, s in sorted(sequences.items()))
    path.write_text(body, encoding="ascii", newline="\n")


def _one_search(
    session: Session, fasta: Path, work: Path, tag: str
) -> tuple[list[homology.Measurement], Path]:
    """Createdb + search + convertalis for one FASTA; returns the three measurements and the TSV."""
    config = session.config
    exe, env = _exe(config), homology.engine_env(_exe(config))
    poll, threads, verbosity = config.poll_interval, config.engine.threads, config.engine.verbosity
    work.mkdir(parents=True, exist_ok=True)
    query_db, result, out = work / f"{tag}_q", work / f"{tag}_res", work / f"{tag}_hits.tsv"
    tmp = work / f"{tag}_tmp"
    tmp.mkdir(exist_ok=True)
    target = _db_path(config)
    steps = [
        homology.createdb_command(exe, fasta, query_db, verbosity),
        homology.search_command(exe, (query_db, target, result, tmp), config.search),
        homology.convertalis_command(exe, (query_db, target, result, out), threads, verbosity),
    ]
    logs = work / "logs"
    measured = [
        homology.run_measured(step, env, logs / f"{tag}_{n}.log", poll)
        for n, step in enumerate(steps)
    ]
    return measured, out


def _spread(items: Sequence[str], count: int) -> list[str]:
    """``count`` items spread evenly over a list already sorted by length."""
    if count >= len(items):
        return list(items)
    step = (len(items) - 1) / (count - 1) if count > 1 else 0.0
    return [items[round(i * step)] for i in range(count)]


def _latency(session: Session, sequences: Mapping[str, str], work: Path) -> dict[str, Any]:
    """Single-query runs over queries spread across the length distribution."""
    count = (
        session.config.smoke.latency_samples if session.smoke else session.config.latency_samples
    )
    ordered = sorted(sequences, key=lambda a: (len(sequences[a]), a))
    samples: list[dict[str, Any]] = []
    for n, accession in enumerate(_spread(ordered, count)):
        fasta = work / f"lat_{n}.fasta"
        _write_fasta(fasta, {accession: sequences[accession]})
        steps, out = _one_search(session, fasta, work / "latency", f"s{n}")
        cpu = [m.cpu_seconds for m in steps]
        peaks = [m.peak_working_set_bytes for m in steps if m.peak_working_set_bytes]
        samples.append(
            {
                "accession": accession,
                "length": len(sequences[accession]),
                "seconds": sum(m.seconds for m in steps),
                "cpu_seconds": None if None in cpu else sum(c for c in cpu if c is not None),
                "peak_working_set_bytes": max(peaks) if peaks else None,
                "hit_lines": len(out.read_text(encoding="ascii").splitlines()),
            }
        )
    seconds = [float(s["seconds"]) for s in samples]
    cpus = [s["cpu_seconds"] for s in samples]
    return {
        "samples": samples,
        "p50_seconds": homology.nearest_rank(seconds, 50),
        "p95_seconds": homology.nearest_rank(seconds, 95),
        "cpu_seconds_per_query": (
            None if None in cpus else sum(c for c in cpus if c is not None) / len(cpus)
        ),
        "method": (
            "one mmseqs createdb + search + convertalis process set per query against the on-disk "
            "index: index loading is included, a resident server would be faster; nearest-rank "
            "percentiles; queries spread evenly over the length distribution"
        ),
    }


def run_search(args: argparse.Namespace) -> int:
    """One batch search of the subset queries, then the single-query latency runs."""
    session = _session(args)
    config = session.config
    index = read_manifest(EVIDENCE / INDEX_FILE)
    engine = _verify_engine(config)
    summary = _index_summary(_db_path(config).parent)
    files = registry_evidence.index_files(_db_path(config).parent, summary["names"])
    if registry_evidence.index_digest(files) != index["index_sha256"]:
        raise BaselineError("the index on disk is not the one --index recorded")
    _announce()
    stage = session.stage
    stage.mkdir(parents=True, exist_ok=True)
    accessions = _query_accessions(session)
    sequences = _read_fasta(EVIDENCE / config.database.fasta, set(accessions))
    queries = stage / QUERIES_FASTA
    _write_fasta(queries, sequences)
    (stage / QUERIES_LIST).write_text("\n".join(accessions) + "\n", encoding="ascii", newline="\n")
    steps, out = _one_search(session, queries, stage / "work", "batch")
    out.replace(stage / HITS_FILE)
    hits = homology.parse_hits((stage / HITS_FILE).read_text(encoding="ascii"))
    _say(
        f"batch: {len(accessions)} queries, {sum(map(len, hits.values()))} hit lines in "
        f"{sum(m.seconds for m in steps):.1f} s"
    )
    latency = _latency(session, sequences, stage / "work")
    peaks = [m.peak_working_set_bytes for m in steps if m.peak_working_set_bytes]
    document = {
        "engine": engine,
        "index_sha256": index["index_sha256"],
        "homology_config_sha256": config.sha256,
        "queries": len(accessions),
        "queries_sha256": sha256_file(stage / QUERIES_FASTA)[0],
        "hits_sha256": sha256_file(stage / HITS_FILE)[0],
        "search": asdict(config.search),
        "batch": {
            "steps": [m.as_dict() for m in steps],
            "seconds": sum(m.seconds for m in steps),
            "peak_working_set_bytes": max(peaks) if peaks else None,
        },
        "query_latency": latency,
        "scope": SUBSET_SCOPE,
        "label": SUBSET_LABEL,
        "hardware": bm.hardware(),
    }
    write_json(stage / SEARCH_FILE, document)
    _say(
        f"latency over {len(latency['samples'])} single queries: "
        f"p50 {latency['p50_seconds']:.2f} s, p95 {latency['p95_seconds']:.2f} s, "
        f"CPU {latency['cpu_seconds_per_query']} s/query"
    )
    return 0


# -- masking, scoring ------------------------------------------------------------------
def _load_hits(session: Session) -> tuple[dict[str, list[Hit]], dict[str, Any]]:
    stage = session.stage
    search = read_manifest(stage / SEARCH_FILE)
    index = read_manifest(EVIDENCE / INDEX_FILE)
    if (
        search["index_sha256"] != index["index_sha256"]
        or search["homology_config_sha256"] != session.config.sha256
        or search["hits_sha256"] != sha256_file(stage / HITS_FILE)[0]
    ):
        raise BaselineError("the search stage is stale (index, configuration or hits changed)")
    return homology.parse_hits((stage / HITS_FILE).read_text(encoding="ascii")), search


def _mask(
    session: Session, accessions: Sequence[str], raw: Mapping[str, Sequence[Hit]]
) -> tuple[dict[str, list[Hit]], dict[str, list[Hit]], homology.MaskReport]:
    """Masked hits per query, the registry-only restriction of them, and the summed report."""
    kept: dict[str, list[Hit]] = {}
    registered: dict[str, list[Hit]] = {}
    reports: list[homology.MaskReport] = []
    for accession in accessions:
        query = session.database[accession]
        masked, report = homology.mask_hits(query, raw.get(accession, ()), session.database)
        kept[accession] = masked
        registered[accession] = homology.in_registry(masked, session.database)
        reports.append(report)
    return kept, registered, homology.add_reports(reports)


def _report_dict(report: homology.MaskReport) -> dict[str, Any]:
    return {
        "raw": report.raw,
        "kept": report.kept,
        "exclusive": dict(report.exclusive),
        "independent": dict(report.independent),
    }


def _scoring(session: Session, samples: int) -> go.Scoring:
    return go.Scoring(session.go_config.metrics, bm.Bootstrap(samples, session.go_config.seed))


def _samples(session: Session) -> int:
    if session.smoke:
        return session.config.smoke.bootstrap_samples
    value = importlib.import_module("yaml").safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    return int(value["reporting"]["bootstrap_samples"])


def _arm_votes(
    session: Session,
    rows: go.Rows,
    hits: Mapping[str, Sequence[Hit]],
    rule: HitFilter,
) -> homology_eval.Votes:
    spec = (session.vocab.ids, rule, session.config.transfer.annotated_hits_only)
    return homology_eval.vote_scores(rows.accession, hits, session.terms_of, spec)


def _grid_report(grid: Sequence[homology_eval.GridRow]) -> list[dict[str, Any]]:
    return [
        {
            "index": g.index,
            "params": g.rule.as_dict(),
            "precision_at_primary_k": g.selector,
            "mean_fold_fmax": g.mean_fold_fmax,
            "hit_coverage": g.hit_coverage,
        }
        for g in grid
    ]


def _selected_block(row: homology_eval.GridRow) -> dict[str, Any]:
    return {
        "index": row.index,
        "params": row.rule.as_dict(),
        "selector": row.selector,
        "mean_fold_fmax": row.mean_fold_fmax,
    }


def _rule_of(block: Mapping[str, Any]) -> HitFilter:
    p = block["params"]
    return HitFilter(
        e_value=float(p["e_value"]),
        min_identity=float(p["min_identity"]),
        min_query_coverage=float(p["min_query_coverage"]),
        top_k=int(p["top_k"]),
    )


@dataclass(frozen=True, slots=True)
class Scored:
    """Arms scored on one row set: reports, thresholds, decisions, scores, coverage, bins."""

    reports: dict[str, Any]
    thresholds: dict[str, Any]
    arms: dict[str, tuple[Any, Any]]
    coverage: dict[str, Any]
    bins: dict[str, Any]


def _score_arms(
    session: Session,
    rows: go.Rows,
    hit_sets: tuple[Mapping[str, Sequence[Hit]], Mapping[str, Sequence[Hit]]],
    plan: tuple[HitFilter, HitFilter, Any],
) -> tuple[dict[str, Any], dict[str, homology_eval.Votes]]:
    """Propagated scores of vote, best-hit, registry-only and prior arms; and their votes."""
    primary, registered = hit_sets
    vote_rule, best_rule, prior = plan
    weight = session.config.transfer.tiebreak_weight
    votes = {
        homology_eval.ARM_VOTE: _arm_votes(session, rows, primary, vote_rule),
        homology_eval.ARM_BEST: _arm_votes(session, rows, primary, best_rule),
        homology_eval.ARM_REGISTRY: _arm_votes(session, rows, registered, vote_rule),
    }
    scores = {
        name: homology_eval.with_prior(v.score, prior, weight, session.vocab)
        for name, v in votes.items()
    }
    scores[homology_eval.ARM_PRIOR] = go.propagate_scores(prior, session.vocab)
    return scores, votes


def run_evaluate(args: argparse.Namespace) -> int:
    """Mask, tune the hit filter on partitions 1-4, and score every arm out of fold."""
    session = _session(args)
    _announce()
    began = time.perf_counter()
    config, go_config = session.config, session.go_config
    raw, search = _load_hits(session)
    rows = _rows(session, VALIDATION_PARTITIONS)
    searched = set((session.stage / QUERIES_LIST).read_text(encoding="ascii").split())
    if not set(rows.accession) <= searched:
        raise BaselineError("some scored rows were never searched: rerun --search")
    primary, registered, mask_report = _mask(session, rows.accession, raw)
    _say(f"masking: {_report_dict(mask_report)}")
    folds = lopo_folds()
    prior = go.fold_prior_scores(rows.y, rows.partition, folds)
    grid = homology_eval.run_grid(
        rows, primary, session.terms_of, (session.vocab, prior, config, go_config.metrics)
    )
    chosen = homology_eval.select_row(grid)
    best = homology_eval.select_row([g for g in grid if g.rule.top_k == 1])
    _say(f"grid {len(grid)} points in {time.perf_counter() - began:.1f} s; selected {chosen.rule}")
    plan = (chosen.rule, best.rule, prior)
    scores, votes = _score_arms(session, rows, (primary, registered), plan)
    scoring = _scoring(session, _samples(session))
    ks = go_config.metrics.k_values
    reports: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for name, score in scores.items():
        reports[name], thresholds[name], decisions[name] = homology_eval.nested_arm(
            rows, score, ks, scoring
        )
    model: dict[str, Any] = {}
    reproduction: dict[str, Any] | None = None
    if not session.smoke:
        model, reproduction = _model_arms(session, rows, scoring)
        for name, (score, decision, report, threshold) in model.items():
            scores[name], decisions[name] = score, decision
            reports[name], thresholds[name] = report, threshold
    bins = homology_eval.bin_reports(
        {n: (scores[n], decisions[n]) for n in BIN_ARMS if n in scores},
        rows,
        votes[homology_eval.ARM_VOTE].best_identity,
        (config.transfer.identity_bins, config.transfer.min_rows_per_bin, scoring),
    )
    coverage = {
        name: homology_eval.coverage_block(votes[name], rows.y)
        for name in (homology_eval.ARM_VOTE, homology_eval.ARM_BEST, homology_eval.ARM_REGISTRY)
    }
    for name, report in reports.items():
        _say(_headline(name, report, ks))
    _say(f"coverage: { {n: c['hit_coverage'] for n, c in coverage.items()} }")
    write_json(
        session.stage / VALIDATION_FILE,
        {
            "fingerprint": _fingerprint(session, search),
            "scope": SUBSET_SCOPE,
            "label": SUBSET_LABEL,
            "rows": len(rows),
            "evaluable_rows": int(go.evaluable(rows.y).sum()),
            "masking": _report_dict(mask_report),
            "selected": _selected_block(chosen),
            "selected_best": _selected_block(best),
            "grid": _grid_report(grid),
            "arms": reports,
            "thresholds": thresholds,
            "coverage": coverage,
            "identity_bins": bins,
            "model_reproduction": reproduction,
            "samples": scoring.boot.samples,
            "seconds": time.perf_counter() - began,
        },
    )
    _say(f"evaluate stage: {time.perf_counter() - began:.1f} s")
    return 0


def _fingerprint(session: Session, search: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "homology_config_sha256": session.config.sha256,
        "index_sha256": search["index_sha256"],
        "queries_sha256": search["queries_sha256"],
        "hits_sha256": search["hits_sha256"],
        "label_table_sha256": session.label_info["label_table_sha256"],
        "ontology_sha256": session.go_config.ontology.sha256,
        "hit_terms_sha256": read_manifest(EVIDENCE / HIT_TERMS_META)["sha256"],
    }


def _cell(entry: Mapping[str, Any]) -> str:
    low, high = entry["ci_low"], entry["ci_high"]
    interval = "" if low is None else f" [{low:.3f}, {high:.3f}]"
    return f"{entry['value']:.3f}{interval}"


def _headline(name: str, arm: Mapping[str, Any], ks: Sequence[int]) -> str:
    metrics = arm["pooled"]["metrics"]
    parts = [f"p@{k} {_cell(metrics[f'precision_at_{k}'])}" for k in ks]
    parts += [f"Fmax {_cell(metrics['fmax'])}", f"coverage {_cell(metrics['coverage'])}"]
    return f"{name}: " + ", ".join(parts)


def _model_arms(
    session: Session, rows: go.Rows, scoring: go.Scoring
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Phase 12 model arms from their saved out-of-fold scores; they must reproduce.

    The raw scores and the cross-fitted calibrated scores are read, never recomputed by a model,
    and are never an input to a homology score. Each arm's pooled metrics must equal the values the
    frozen Phase 12 manifest recorded.
    """
    ks = session.go_config.metrics.k_values
    oof = bm.load_array(GO_DIR / "oof_score.npy")
    crossfit = bm.load_array(GO_DIR / "calibration" / "crossfit.npy")
    if oof.shape != rows.y.shape or crossfit.shape != rows.y.shape:
        raise BaselineError("the Phase 12 out-of-fold scores do not match the rows")
    raw_score = go.propagate_scores(oof, session.vocab)
    raw_report, raw_thr, raw_dec = homology_eval.nested_arm(rows, raw_score, ks, scoring)
    fit = read_manifest(GO_DIR / "calibration" / "fit.json")
    folds = {int(k): float(v) for k, v in fit["fold_thresholds"].items()}
    cal_score = go.propagate_scores(crossfit, session.vocab)
    cal_dec = go.decide(cal_score, rows.partition, folds)
    cal_report = homology_eval.assemble(rows, (cal_score, cal_dec), ks, scoring)
    frozen = read_manifest(GO_MANIFEST)["frozen"]["validation"]
    worst = 0.0
    for name, report in (
        (homology_eval.ARM_RAW, raw_report),
        (homology_eval.ARM_CALIBRATED, cal_report),
    ):
        recorded = frozen[name]["pooled"]["metrics"]
        for metric, entry in report["pooled"]["metrics"].items():
            worst = max(worst, abs(float(entry["value"]) - float(recorded[metric]["value"])))
    tolerance = session.go_config.reproduction_tolerance
    if not worst <= tolerance:
        raise BaselineError(
            f"the Phase 12 model arms do not reproduce from their saved scores (max {worst:.3e})"
        )
    arms = {
        homology_eval.ARM_RAW: (raw_score, raw_dec, raw_report, raw_thr),
        homology_eval.ARM_CALIBRATED: (
            cal_score,
            cal_dec,
            cal_report,
            {"folds": {str(k): v for k, v in folds.items()}, "frozen": fit["frozen_threshold"]},
        ),
    }
    return arms, {"max_abs_difference": float(worst), "tolerance": tolerance}


# -- the manifest ----------------------------------------------------------------------
def _pins(session_config: homology_eval.HomologyConfig, script: ModuleType) -> homology_eval.Pins:
    go_pins = script._pins()
    go_config = go.load_config(GO_CONFIG)
    arms = homology_eval.arm_names(session_config, with_model=True)
    return homology_eval.Pins(
        registry_sha256=go_pins.registry_sha256,
        corpus_sha256=go_pins.corpus_sha256,
        ontology_sha256=go_pins.ontology_sha256,
        vocabulary_sha256=go_pins.vocabulary_sha256,
        go_frozen_sha256=str(read_manifest(GO_MANIFEST)["frozen_sha256"])
        if GO_MANIFEST.exists()
        else "",
        config_sha256=session_config.sha256,
        engine_release=session_config.engine.release,
        engine_commit=session_config.engine.commit,
        engine_executable_sha256=session_config.engine.executable_sha256,
        grid_size=len(homology_eval.grid_points(session_config)),
        k_values=go_config.metrics.k_values,
        reproduction_tolerance=go_config.reproduction_tolerance,
        arms_validation=arms,
        arms_test=tuple(a for a in arms if a not in homology_eval.MODEL_ARMS),
    )


def _earlier_touches(script: ModuleType) -> dict[str, int | None]:
    touches = script._earlier_touches()
    touches["go"] = script._touches(GO_MANIFEST) if GO_MANIFEST.exists() else None
    return {k: touches[k] for k in homology_eval.EARLIER_HEADS}


def _bundle_sections(session: Session, frozen: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    config, go_config = session.config, session.go_config
    index = read_manifest(EVIDENCE / INDEX_FILE)
    inputs = frozen["inputs"]
    return {
        "database": {
            "source": config.database.source,
            "release": config.database.release,
            "records": config.database.records,
            "license": config.database.license,
            "corpus_sha256": config.database.corpus_sha256,
            "fasta_sha256": inputs["fasta_sha256"],
            "meta_sha256": inputs["meta_sha256"],
        },
        "engine": {
            "name": config.engine.name,
            "release": config.engine.release,
            "commit": config.engine.commit,
            "platform": config.engine.platform,
            "license": config.engine.license,
            "archive_sha256": config.engine.archive_sha256,
            "executable_sha256": config.engine.executable_sha256,
        },
        "index": {
            "directory": f"{config.database.directory}/{config.database.mmseqs_directory}",
            "database_name": config.database.mmseqs_name,
            "files": index["files"],
            "index_sha256": index["index_sha256"],
        },
        "go": {
            "sha256": config.database.corpus_sha256,
            "annotation_source": config.annotation_source,
            "hit_terms_sha256": inputs["hit_terms_sha256"],
            "ontology_sha256": go_config.ontology.sha256,
            "ontology_data_version": go_config.ontology.data_version,
            "vocabulary_sha256": inputs["vocabulary_sha256"],
            "evidence_policy": list(go_config.evidence.policy),
            "propagation": list(go_config.ontology.relations),
        },
        "hit_filter": {
            "vote": frozen["selected"]["params"],
            "best": frozen["selected_best"]["params"],
            "grid_size": len(frozen["grid"]),
            "search": {
                "sensitivity": config.search.sensitivity,
                "e_value": config.search.e_value,
                "max_seqs": config.search.max_seqs,
                "alignment_mode": config.search.alignment_mode,
            },
        },
        "masking": {"masks": list(homology.MASKS)},
        "transfer": {
            "source": homology.SOURCE_HOMOLOGY,
            "annotated_hits_only": config.transfer.annotated_hits_only,
            "tiebreak_weight": config.transfer.tiebreak_weight,
            "rule": "vote = fraction of the top-k annotated hits carrying the term",
        },
        "independence": {
            "feeds_model_score": False,
            "feeds_calibrator": False,
            "feeds_threshold": False,
        },
    }


def _frozen_block(session: Session, parts: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen block of the manifest from the three stage files."""
    config, go_config = session.config, session.go_config
    index, search, valid = parts["index"], parts["search"], parts["validation"]
    go_pins = session.script._pins()
    fingerprint = valid["fingerprint"]
    reports = dict(valid["arms"])
    thresholds = {
        name: float(valid["thresholds"][name]["frozen"])
        for name in homology_eval.arm_names(config, with_model=False)
    }
    frozen: dict[str, Any] = {
        "protocol": {
            "adr": "ADR-0015 (protocol of ADR-0010, ADR-0011, ADR-0013 and ADR-0014)",
            "validation_partitions": list(VALIDATION_PARTITIONS),
            "test_partition": TEST_PARTITION,
            "folds": [{"train": list(f.train), "validate": f.validate} for f in lopo_folds()],
            "threshold_rule": THRESHOLD_RULE,
            "selection_rule": SELECTION_RULE,
            "k_values": list(go_config.metrics.k_values),
            "primary_k": go_config.metrics.primary_k,
            "bootstrap": {
                "samples": valid["samples"],
                "unit": "cluster_id",
                "seed": go_config.seed,
            },
            "masks": list(homology.MASKS),
        },
        "scope": SUBSET_SCOPE,
        "scope_label": SUBSET_LABEL,
        "inputs": {
            "registry_sha256": go_pins.registry_sha256,
            "corpus_sha256": go_pins.corpus_sha256,
            "ontology_sha256": go_pins.ontology_sha256,
            "vocabulary_sha256": go_pins.vocabulary_sha256,
            "go_manifest_frozen_sha256": str(read_manifest(GO_MANIFEST)["frozen_sha256"]),
            "homology_config_sha256": config.sha256,
            "fasta_sha256": index["database"]["fasta_sha256"],
            "meta_sha256": index["database"]["meta_sha256"],
            "index_sha256": index["index_sha256"],
            "queries_sha256": search["queries_sha256"],
            "hits_sha256": search["hits_sha256"],
            "label_table_sha256": fingerprint["label_table_sha256"],
            "feature_table_sha256": str(session.sub.meta["table_sha256"]),
            "hit_terms_sha256": fingerprint["hit_terms_sha256"],
        },
        "engine": {
            "release": config.engine.release,
            "commit": config.engine.commit,
            "executable_sha256": config.engine.executable_sha256,
            "archive_sha256": config.engine.archive_sha256,
            "license": config.engine.license,
            "version_output": index["engine"]["version_output"],
        },
        "database": {
            "source": config.database.source,
            "release": config.database.release,
            "records": config.database.records,
            "license": config.database.license,
        },
        "masking": {
            "masks": list(homology.MASKS),
            "validation": valid["masking"],
            "note": (
                "every database sequence is searched; hits are then dropped for the query's own "
                "accession, its own sequence hash, its cluster_id and its partition. Database "
                "sequences outside the split registry are never masked by cluster or partition "
                "(they have none): the registry-only diagnostic arm and the identity bins show "
                "what that costs"
            ),
        },
        "selected": valid["selected"],
        "selected_best": valid["selected_best"],
        "grid": valid["grid"],
        "thresholds": thresholds,
        "coverage": valid["coverage"],
        "validation": {
            **reports,
            "model_reproduction": valid["model_reproduction"],
            "grid": valid["grid"],
            "coverage": valid["coverage"],
            "identity_bins": valid["identity_bins"],
        },
        "measurements": {
            "index_build": index["build"],
            "search_batch": search["batch"],
            "query_latency": {
                k: search["query_latency"][k]
                for k in ("p50_seconds", "p95_seconds", "cpu_seconds_per_query", "method")
            },
            "latency_samples": search["query_latency"]["samples"],
        },
        "independence": dict(homology_eval.INDEPENDENCE),
        "software": {"python": platform.python_version()},
        "hardware": {"index": index["hardware"], "search": search["hardware"]},
    }
    return frozen


def run_freeze(args: argparse.Namespace) -> int:
    """Write the registry bundle and the manifest, fixed before the test partition is read."""
    if args.smoke:
        _say("--smoke output can never be frozen")
        return 1
    if MANIFEST.exists():
        _say(f"{MANIFEST.name} exists; a frozen manifest is never rewritten")
        return 1
    _announce()
    session = _session(args)
    config = session.config
    directory = REPO_ROOT / config.bundle_directory / config.bundle_version
    if directory.exists():
        _say(f"{directory} exists; a frozen bundle is never rewritten")
        return 1
    _raw, search = _load_hits(session)
    valid = read_manifest(session.stage / VALIDATION_FILE)
    if valid["fingerprint"] != _fingerprint(session, search):
        _say("validation.json was produced from other inputs: rerun --search and --evaluate")
        return 1
    parts = {"index": read_manifest(EVIDENCE / INDEX_FILE), "search": search, "validation": valid}
    frozen = _frozen_block(session, parts)
    document = registry_evidence.build_document(
        version=config.bundle_version,
        scope={"scope": SUBSET_SCOPE, "scope_label": SUBSET_LABEL},
        sections=_bundle_sections(session, frozen),
    )
    frozen["bundle"] = {
        "version": config.bundle_version,
        "directory": f"{config.bundle_directory}/{config.bundle_version}",
        "bundle_sha256": document["bundle_sha256"],
    }
    manifest: dict[str, Any] = {
        "schema_version": homology_eval.SCHEMA_VERSION,
        "phase": homology_eval.PHASE,
        "frozen": frozen,
        "frozen_sha256": digest(frozen),
        "test": None,
    }
    problems = homology_eval.check_homology_manifest(manifest, _pins(config, session.script))
    if problems:
        for problem in problems:
            _say(f"  {problem}")
        return 1
    directory.mkdir(parents=True)
    write_json(directory / registry_evidence.BUNDLE_FILE, document)
    registry_evidence.load_evidence_bundle(
        directory,
        index_directory=_db_path(config).parent,
        goa_sha256=config.database.corpus_sha256,
        ontology_sha256=session.go_config.ontology.sha256,
    )
    write_json(MANIFEST, manifest)
    _say(f"froze {directory} and {MANIFEST}, frozen_sha256 {manifest['frozen_sha256'][:12]}")
    return 0


# -- --evaluate --test -----------------------------------------------------------------
def run_test(args: argparse.Namespace) -> int:
    """Score partition 5 once with the hit filter, thresholds and bundle the manifest froze."""
    if args.smoke:
        _say("--smoke never touches the test partition")
        return 1
    began = time.perf_counter()
    manifest = read_manifest(MANIFEST)
    session = _session(args)
    pins = _pins(session.config, session.script)
    problems = homology_eval.check_homology_manifest(manifest, pins)
    if problems or manifest["test"] is not None:
        _say("refusing: the manifest has problems or the test partition was already scored")
        for problem in problems:
            _say(f"  {problem}")
        return 1
    earlier = _earlier_touches(session.script)
    if any(v != 1 for v in earlier.values()):
        _say(f"refusing: an earlier head's test touch count is not 1: {earlier}")
        return 1
    _announce()
    frozen = manifest["frozen"]
    registry_evidence.load_evidence_bundle(
        REPO_ROOT / frozen["bundle"]["directory"],
        index_directory=_db_path(session.config).parent,
        goa_sha256=session.config.database.corpus_sha256,
        ontology_sha256=session.go_config.ontology.sha256,
    )
    raw, _search = _load_hits(session)
    rows = _rows(session, (TEST_PARTITION,))
    train = _rows(session, VALIDATION_PARTITIONS)
    searched = set((session.stage / QUERIES_LIST).read_text(encoding="ascii").split())
    if not set(rows.accession) <= searched:
        raise BaselineError("some test rows were never searched: rerun --search")
    primary, registered, mask_report = _mask(session, rows.accession, raw)
    prior = _np().tile(train.y.mean(axis=0), (len(rows), 1))
    plan = (_rule_of(frozen["selected"]), _rule_of(frozen["selected_best"]), prior)
    scores, votes = _score_arms(session, rows, (primary, registered), plan)
    scoring = _scoring(session, int(frozen["protocol"]["bootstrap"]["samples"]))
    ks = session.go_config.metrics.k_values
    metrics: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for name, score in scores.items():
        decisions[name] = score >= float(frozen["thresholds"][name])
        metrics[name] = homology_eval.assemble(rows, (score, decisions[name]), ks, scoring)
        _say(_headline(f"test {name}", metrics[name], ks))
    bins = homology_eval.bin_reports(
        {n: (scores[n], decisions[n]) for n in BIN_ARMS if n in scores},
        rows,
        votes[homology_eval.ARM_VOTE].best_identity,
        (session.config.transfer.identity_bins, session.config.transfer.min_rows_per_bin, scoring),
    )
    reference = read_manifest(GO_MANIFEST)["test"]["metrics"]
    seconds = time.perf_counter() - began
    manifest["test"] = {
        "touches": 1,
        "frozen_sha256": manifest["frozen_sha256"],
        "earlier_touches": earlier,
        "rows": len(rows),
        "evaluable_rows": int(go.evaluable(rows.y).sum()),
        "seconds": seconds,
        "scope": SUBSET_SCOPE,
        "label": SUBSET_LABEL,
        "masking": _report_dict(mask_report),
        "metrics": metrics,
        "coverage": {
            name: homology_eval.coverage_block(votes[name], rows.y)
            for name in (homology_eval.ARM_VOTE, homology_eval.ARM_BEST, homology_eval.ARM_REGISTRY)
        },
        "identity_bins": bins,
        "model_reference": {
            "source": "go_metrics.json test block, copied; the GO head was not re-run",
            "arms": {
                name: reference[name]
                for name in (homology_eval.ARM_RAW, homology_eval.ARM_CALIBRATED)
            },
        },
    }
    write_json(MANIFEST, manifest)
    _say(f"test partition scored once in {seconds:.1f} s ({len(rows)} rows, CI is wide)")
    return 0


# -- --check ---------------------------------------------------------------------------
def _bundle_problems(
    config: homology_eval.HomologyConfig, manifest: Mapping[str, Any]
) -> list[str]:
    frozen = manifest["frozen"]
    try:
        loaded = registry_evidence.load_evidence_bundle(
            REPO_ROOT / frozen["bundle"]["directory"],
            goa_sha256=config.database.corpus_sha256,
            ontology_sha256=frozen["inputs"]["ontology_sha256"],
        )
    except (registry_evidence.EvidenceBundleError, OSError) as error:
        return [f"registry bundle: {error}"]
    found: list[str] = []
    if loaded["bundle_sha256"] != frozen["bundle"]["bundle_sha256"]:
        found.append("the bundle is not the one the manifest froze")
    if loaded["index"]["index_sha256"] != frozen["inputs"]["index_sha256"]:
        found.append("the bundle names another index than the manifest")
    return found


def run_check(args: argparse.Namespace) -> int:
    """No data bytes: configuration, pins and, when present, the manifest and the bundle."""
    del args
    problems: list[str] = []
    config = _config()
    script = _go_script()
    pins = _pins(config, script)
    if MANIFEST.exists():
        manifest = read_manifest(MANIFEST)
        problems += homology_eval.check_homology_manifest(manifest, pins)
        if not problems:
            problems += _bundle_problems(config, manifest)
        state = f"{MANIFEST.name} present"
    else:
        state = f"{MANIFEST.name} absent (configuration and pins only)"
    earlier = _earlier_touches(script)
    problems += [
        f"{name} test touches is {n}, not 1" for name, n in earlier.items() if n not in (None, 1)
    ]
    for problem in problems:
        _say(f"FAIL {problem}")
    _say(
        f"homology --check: {'FAIL' if problems else 'ok'}; mmseqs {config.engine.release}; "
        f"{pins.grid_size} grid points; masks {list(homology.MASKS)}; earlier touches {earlier}; "
        f"{state}; scope: {SUBSET_LABEL}"
    )
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 13 homology evidence.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("index", "search", "evaluate", "freeze", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--test", action="store_true", help="with --evaluate: score the test once")
    parser.add_argument("--smoke", action="store_true", help="every 4th query, never frozen")
    return parser


def _mode(args: argparse.Namespace) -> int:
    if args.index:
        return run_index(args)
    if args.search:
        return run_search(args)
    if args.evaluate:
        return run_test(args) if args.test else run_evaluate(args)
    if args.freeze:
        return run_freeze(args)
    return run_check(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one mode."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.test and not args.evaluate:
        parser.error("--test goes with --evaluate")
    try:
        return _mode(args)
    except (BaselineError, HomologyError) as error:
        _say(f"error: {error}")
        return 1
    except json.JSONDecodeError as error:
        _say(f"error: unreadable JSON: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
