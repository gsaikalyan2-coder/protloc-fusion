r"""Phase 7: the ESM-2 embedding cache. One job per mode.

    python scripts/embed.py --facts
        No torch. Counts the registry's distinct sequences, how many exceed the
        1022-residue context, letters used, and the size per-residue storage would
        need. Writes data/embeddings/reports/facts.json.

    python scripts/embed.py --pack
        Writes data/embeddings/input_pack.tsv: the 33,461 distinct sequences in one small
        file, with the registry and corpus hashes in its header. Verifies the corpus
        hash first. Lets a free GPU notebook run the steps below without the corpus.

    python scripts/embed.py --bundle
        Zips the code, model config and input pack needed on a GPU notebook into
        data/embeddings/gpu_bundle.zip.

    Every mode below also accepts --input data/embeddings/input_pack.tsv to read the
    sequences from the pack instead of the corpus.

    python scripts/embed.py --measure [--n 200] [--device cpu|cuda]
        Times the encoder on 200 real registry sequences, spread evenly over the
        length distribution (no random draw). Writes reports/measure.json.

    python scripts/embed.py --extract [--device cpu|cuda]
        Embeds every distinct registry sequence into the cache, one committed shard
        at a time. Safe to kill and rerun: finished shards are kept. The script
        records its own time, so do NOT wrap it in Measure-Command or time.

    --extract --subset N embeds only N sequences (spread evenly over the length
        distribution, as --measure samples them) into its own cache directory and
        labels the manifest "subset". The full run stays outstanding.

    python scripts/embed.py --verify [--device cpu|cuda]
        Padding invariance, tokeniser coverage, byte identity of one sequence across
        two separate processes, and byte identity after a real kill and resume.
        Writes reports/verify.json.

    python scripts/embed.py --freeze
        Writes data/manifests/embedding_manifest.json from the reports and the cache.

    python scripts/embed.py --check
        No cache bytes needed (this is what CI runs). Exit 1 unless the config pins,
        .env.example and the manifest (when present) agree.

Run from the repository root inside the project venv. Downloads of the pinned
checkpoint come from huggingface.co; HF_TOKEN is read from the environment only.
"""

import argparse
from collections.abc import Sequence
import importlib
import json
import multiprocessing
from pathlib import Path
import shutil
import statistics
import sys
import time
import zipfile

from protloc.data.embedding_manifest import (
    Measured,
    build_manifest,
    check_documents,
    read_pins,
)
from protloc.data.embeddings import (
    MAX_RESIDUES,
    CacheSpec,
    EmbeddingError,
    ExtractOptions,
    UniqueSequence,
    cache_size_bytes,
    extract,
    max_abs_diff,
    read_index,
    read_pack,
    select_unique,
    stratified_sample,
    verify_cache,
    write_pack,
)
from protloc.data.esm_encoder import (
    TorchEncoder,
    extract_worker,
    probe_worker,
)
from protloc.data.manifest import read_json_object, sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
REGISTRY = REPO_ROOT / "data" / "processed" / "splits" / "registry.tsv"
DATASET_MANIFEST = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
EMBEDDING_MANIFEST = REPO_ROOT / "data" / "manifests" / "embedding_manifest.json"
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "trunk_esm2_150m.yaml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
EMBED_ROOT = REPO_ROOT / "data" / "embeddings"
REPORTS = EMBED_ROOT / "reports"
FLOAT_BYTES = 4
HALF_BYTES = 2
PADDING_SEQUENCES = 6
RESUME_SUBSET = 64
RESUME_SHARD = 4
POLL_SECONDS = 0.05


def _spec() -> CacheSpec:
    return read_pins(
        MODEL_CONFIG.read_text(encoding="utf-8"),
        ENV_EXAMPLE.read_text(encoding="utf-8"),
    )


def _cache_dir(spec: CacheSpec, subset: int | None = None) -> Path:
    """Full cache under the key; a subset cache gets its own directory, never mixed in."""
    suffix = "" if subset is None else f"_subset{subset}"
    return EMBED_ROOT / (spec.key()[:16] + suffix)


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _load_unique(
    args: argparse.Namespace,
) -> tuple[list[UniqueSequence], dict[str, object]]:
    """Distinct sequences and where they came from, from the pack or the corpus."""
    if args.input:
        path = Path(args.input)
        items, header = read_pack(path)
        pack_sha, _size = sha256_file(path)
        registry = header["registry_sha256"]
        _say(f"input pack {path.name} sha256 {pack_sha[:12]}, registry {registry[:12]}")
        return items, {
            "source": "pack",
            "pack_sha256": pack_sha,
            "registry_sha256": header["registry_sha256"],
            "corpus_sha256": header["corpus_sha256"],
        }
    registry_sha, _size = sha256_file(REGISTRY)
    _say(f"registry sha256 {registry_sha}")
    corpus_sha, _size = sha256_file(CORPUS)
    expected = read_json_object(DATASET_MANIFEST)["build"]
    if not isinstance(expected, dict) or corpus_sha != expected["corpus_sha256"]:
        raise EmbeddingError("corpus.jsonl does not match dataset_manifest.json")
    _say(f"corpus sha256 verified {corpus_sha[:12]}…")
    return select_unique(CORPUS, REGISTRY), {
        "source": "corpus",
        "registry_sha256": registry_sha,
        "corpus_sha256": corpus_sha,
    }


def _encoder(spec: CacheSpec, args: argparse.Namespace) -> TorchEncoder:
    return TorchEncoder(
        spec,
        device=args.device,
        batch_size=args.batch_windows,
        local_files_only=args.local_files_only,
    )


def _facts(args: argparse.Namespace) -> int:
    spec = _spec()
    items, source = _load_unique(args)
    registry_sha = str(source["registry_sha256"])
    lengths = [len(item.sequence) for item in items]
    letters = sorted({letter for item in items for letter in item.sequence})
    total = sum(lengths)
    facts = {
        "cache_key": spec.key(),
        "registry_sha256": registry_sha,
        "accessions": sum(item.accessions for item in items),
        "distinct_sequences": len(items),
        "windowed_sequences": sum(1 for n in lengths if n > MAX_RESIDUES),
        "windowed_accessions": sum(
            item.accessions for item in items if len(item.sequence) > MAX_RESIDUES
        ),
        "max_length": max(lengths),
        "median_length": int(statistics.median(lengths)),
        "letters": "".join(letters),
        "sequences_with_rare_letters": sum(
            1 for item in items if set(item.sequence) & set("BOUXZ")
        ),
        "total_residues_distinct": total,
        "pooled_cache_bytes_expected": len(items) * spec.embedding_dim * FLOAT_BYTES,
        "per_residue_float16_bytes": total * spec.embedding_dim * HALF_BYTES,
        "per_residue_float32_bytes": total * spec.embedding_dim * FLOAT_BYTES,
    }
    write_json(REPORTS / "facts.json", facts)
    _say(json.dumps(facts, indent=2, sort_keys=True))
    return 0


def _measure(args: argparse.Namespace) -> int:
    spec = _spec()
    items, source = _load_unique(args)
    sample = stratified_sample(items, args.n)
    loading = time.perf_counter()
    encoder = _encoder(spec, args)
    load_seconds = time.perf_counter() - loading
    ids = encoder.check_alphabet(
        "".join(sorted({c for i in sample for c in i.sequence}))
    )
    shortest = min(sample, key=lambda item: len(item.sequence))
    encoder.embed_batch([shortest.sequence])  # warm-up, not timed
    started = time.perf_counter()
    encoder.embed_batch([item.sequence for item in sample])
    seconds = time.perf_counter() - started
    lengths = [len(item.sequence) for item in sample]
    report = {
        "cache_key": spec.key(),
        "input": source,
        "sequences": len(sample),
        "model_load_seconds": round(load_seconds, 3),
        "embed_seconds": round(seconds, 3),
        "seconds_per_sequence": seconds / len(sample),
        "residues_per_second": sum(lengths) / seconds,
        "length_min": min(lengths),
        "length_median": int(statistics.median(lengths)),
        "length_max": max(lengths),
        "windowed_in_sample": sum(1 for n in lengths if n > MAX_RESIDUES),
        "estimate_full_run_seconds": seconds / len(sample) * len(items),
        "estimate_note": "ESTIMATE: sample mean times distinct-sequence count; not measured",
        "letters_checked": len(ids),
        "hardware": encoder.describe(),
    }
    write_json(REPORTS / "measure.json", report)
    _say(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _extract(args: argparse.Namespace) -> int:
    spec = _spec()
    items, source = _load_unique(args)
    if args.subset:
        items = stratified_sample(items, args.subset)
    directory = _cache_dir(spec, args.subset)
    encoder = _encoder(spec, args)
    seconds_file = REPORTS / "shard_seconds.json"
    recorded: dict[str, object] = (
        read_json_object(seconds_file)
        if seconds_file.exists() and (directory / "spec.json").exists()
        else {}
    )

    def on_shard(number: int, total: int, seconds: float) -> None:
        recorded[str(number)] = seconds
        write_json(seconds_file, recorded)
        _say(f"shard {number + 1}/{total} committed in {seconds:.1f}s")

    options = ExtractOptions(shard_size=args.shard_size, on_shard=on_shard)
    report = extract(items, directory, spec, encoder, options)
    _say(
        f"shards: {report.shards_computed} computed, {report.shards_skipped} already done, "
        f"{report.shards_total} total; this call {report.seconds:.1f}s"
    )
    if not report.complete:
        return 1
    index = read_index(directory)
    total_seconds = sum(float(str(v)) for v in recorded.values())
    run = {
        "cache_key": spec.key(),
        "input": source,
        "cache_bytes": cache_size_bytes(directory),
        "sequences": len(items),
        "subset": args.subset,
        "windowed_in_cache": sum(
            1 for item in items if len(item.sequence) > MAX_RESIDUES
        ),
        "shard_seconds_total": total_seconds,
        "seconds_per_sequence": total_seconds / len(items),
        "shard_size": args.shard_size,
        "shard_index_sha256": index["shard_index_sha256"],
        "hardware": encoder.describe(),
    }
    write_json(REPORTS / "extract_run.json", run)
    _say(json.dumps(run, indent=2, sort_keys=True))
    return 0


def _pairs(items: Sequence[UniqueSequence]) -> list[tuple[str, str]]:
    return [(item.sha256, item.sequence) for item in items]


def _spawn(target: object, *arguments: object) -> multiprocessing.Process:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=arguments)  # type: ignore[arg-type]
    process.start()
    return process


def _padding_check(encoder: TorchEncoder, sample: Sequence[UniqueSequence]) -> float:
    picked = [
        sample[i * (len(sample) - 1) // (PADDING_SEQUENCES - 1)]
        for i in range(PADDING_SEQUENCES)
    ]
    together = encoder.embed_batch([item.sequence for item in picked])
    worst = 0.0
    for item, joint in zip(picked, together, strict=True):
        alone = encoder.embed_batch([item.sequence])[0]
        worst = max(worst, max_abs_diff(alone, joint))
    return worst


def _device_check(
    spec: CacheSpec, args: argparse.Namespace, sample: Sequence[UniqueSequence]
) -> object:
    if (
        args.device != "cuda"
        or not importlib.import_module("torch").cuda.is_available()
    ):
        return None
    cpu = TorchEncoder(spec, device="cpu", local_files_only=args.local_files_only)
    gpu = TorchEncoder(spec, device="cuda", local_files_only=args.local_files_only)
    texts = [item.sequence for item in sample]
    return max(
        max_abs_diff(a, b)
        for a, b in zip(cpu.embed_batch(texts), gpu.embed_batch(texts), strict=True)
    )


def _probe(
    spec: CacheSpec, args: argparse.Namespace, item: UniqueSequence
) -> list[str]:
    scratch = EMBED_ROOT / "_verify"
    hashes: list[str] = []
    for run in (1, 2):
        out = scratch / f"probe_{run}.f32"
        process = _spawn(
            probe_worker, spec.as_dict(), item.sequence, str(out), args.device
        )
        process.join()
        if process.exitcode != 0:
            raise EmbeddingError(
                f"probe process {run} failed with exit code {process.exitcode}"
            )
        hashes.append(sha256_file(out)[0])
    return hashes


def _kill_and_resume(
    spec: CacheSpec, args: argparse.Namespace, subset: Sequence[UniqueSequence]
) -> dict[str, object]:
    scratch = EMBED_ROOT / "_verify"
    clean, killed = scratch / "clean", scratch / "killed"
    items = _pairs(subset)
    run = (spec.as_dict(), items)
    first = _spawn(extract_worker, *run, str(clean), RESUME_SHARD, args.device)
    first.join()
    victim = _spawn(extract_worker, *run, str(killed), RESUME_SHARD, args.device)
    while not (killed / "shard_00000.tsv").exists() and victim.is_alive():
        time.sleep(POLL_SECONDS)
    victim.kill()
    victim.join()
    interrupted = not (killed / "index.json").exists()
    if not interrupted:
        raise EmbeddingError(
            "the run finished before it could be killed; raise RESUME_SUBSET"
        )
    leftovers = sorted(p.name for p in killed.glob("*.tmp"))
    resumed = _spawn(extract_worker, *run, str(killed), RESUME_SHARD, args.device)
    resumed.join()
    clean_index, killed_index = read_index(clean), read_index(killed)
    return {
        "interrupted_before_complete": interrupted,
        "tmp_files_left_by_kill": leftovers,
        "clean_shard_index_sha256": clean_index["shard_index_sha256"],
        "resumed_shard_index_sha256": killed_index["shard_index_sha256"],
        "resumed_cache_problems": verify_cache(killed),
        "identical": clean_index["shard_index_sha256"]
        == killed_index["shard_index_sha256"],
    }


def _verify(args: argparse.Namespace) -> int:
    spec = _spec()
    items, source = _load_unique(args)
    sample = stratified_sample(items, args.n)
    encoder = _encoder(spec, args)
    letters = "".join(sorted({c for item in items for c in item.sequence}))
    encoder.check_alphabet(letters)
    padding = _padding_check(encoder, sample)
    device_diff = _device_check(spec, args, sample[:PADDING_SEQUENCES])
    scratch = EMBED_ROOT / "_verify"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    probe_item = stratified_sample(items, 1)[0]
    hashes = _probe(spec, args, probe_item)
    resume = _kill_and_resume(spec, args, stratified_sample(items, RESUME_SUBSET))
    report = {
        "cache_key": spec.key(),
        "input": source,
        "alphabet_checked": letters,
        "padding_max_abs_diff_alone_vs_batch": padding,
        "cpu_vs_cuda_max_abs_diff": device_diff,
        "probe_sha256": probe_item.sha256,
        "probe_length": len(probe_item.sequence),
        "probe_output_sha256_run_1": hashes[0],
        "probe_output_sha256_run_2": hashes[1],
        "byte_identical_across_processes": hashes[0] == hashes[1],
        "kill_and_resume": resume,
        "byte_identical_after_kill_and_resume": resume["identical"],
        "scope": "same machine, same device, same torch build only",
        "hardware": encoder.describe(),
    }
    write_json(REPORTS / "verify.json", report)
    _say(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["byte_identical_across_processes"] and resume["identical"] else 1


PACK = EMBED_ROOT / "input_pack.tsv"
BUNDLE = EMBED_ROOT / "gpu_bundle.zip"
BUNDLE_FILES = (
    "backend/protloc/__init__.py",
    "backend/protloc/data/__init__.py",
    "backend/protloc/data/manifest.py",
    "backend/protloc/data/embeddings.py",
    "backend/protloc/data/esm_encoder.py",
    "backend/protloc/data/embedding_manifest.py",
    "scripts/embed.py",
    "configs/model/trunk_esm2_150m.yaml",
    ".env.example",
)


def _pack(args: argparse.Namespace) -> int:
    args.input = None
    items, source = _load_unique(args)
    pack_sha = write_pack(
        items, PACK, str(source["registry_sha256"]), str(source["corpus_sha256"])
    )
    _say(f"wrote {PACK} ({PACK.stat().st_size} bytes) sha256 {pack_sha}")
    return 0


def _bundle(_: argparse.Namespace) -> int:
    if not PACK.exists():
        raise EmbeddingError("run --pack first")
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_FILES:
            archive.write(REPO_ROOT / name, name)
        archive.write(PACK, "data/embeddings/input_pack.tsv")
    _say(f"wrote {BUNDLE} ({BUNDLE.stat().st_size} bytes)")
    return 0


def _freeze(_: argparse.Namespace) -> int:
    spec = _spec()
    run = read_json_object(REPORTS / "extract_run.json")
    subset = run.get("subset")
    directory = _cache_dir(spec, int(str(subset)) if subset else None)
    problems = verify_cache(directory)
    if problems:
        raise EmbeddingError("cache failed verification: " + "; ".join(problems))
    registry_sha, _size = sha256_file(REGISTRY)
    measured = Measured(
        facts=read_json_object(REPORTS / "facts.json"),
        measure=read_json_object(REPORTS / "measure.json"),
        extract_run=run,
        index=read_index(directory),
        verify=read_json_object(REPORTS / "verify.json"),
    )
    write_json(EMBEDDING_MANIFEST, build_manifest(spec, registry_sha, measured))
    _say(f"wrote {EMBEDDING_MANIFEST.name}")
    return 0


def _check(_: argparse.Namespace) -> int:
    manifest = (
        read_json_object(EMBEDDING_MANIFEST) if EMBEDDING_MANIFEST.exists() else None
    )
    problems = check_documents(
        manifest,
        MODEL_CONFIG.read_text(encoding="utf-8"),
        ENV_EXAMPLE.read_text(encoding="utf-8"),
    )
    for problem in problems:
        sys.stderr.write(f"FAIL: {problem}\n")
    if not problems:
        state = "manifest present" if manifest else "no manifest yet (pins only)"
        _say(f"embedding pins agree ({state})")
    return 1 if problems else 0


def main() -> int:
    """Parse the mode and run it."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else ""
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in (
        "facts",
        "pack",
        "bundle",
        "measure",
        "extract",
        "verify",
        "freeze",
        "check",
    ):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--subset",
        type=int,
        help="extract only this many sequences, spread over length",
    )
    parser.add_argument("--batch-windows", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--input", help="input pack written by --pack")
    args = parser.parse_args()
    modes = {
        "facts": _facts,
        "pack": _pack,
        "bundle": _bundle,
        "measure": _measure,
        "extract": _extract,
        "verify": _verify,
        "freeze": _freeze,
        "check": _check,
    }
    chosen = next(name for name in modes if getattr(args, name))
    return modes[chosen](args)


if __name__ == "__main__":
    sys.exit(main())
