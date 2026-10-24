r"""Fetch the pinned ESM-2 weights for the live encoder, then self-check them (ADR-0019).

    python scripts/fetch_esm.py [--cache-dir DIR] [--device cpu] [--check-only]

The checkpoint and revision are read from the residue cache's own index.json, the same pin the
500 cached sequences were embedded with, so the live encoder cannot drift from the cache by
configuration. The weights go into a Hugging Face hub cache (``--cache-dir``, else
``ESM_CACHE_DIR`` from the environment, else ``data/esm2``); the server then loads them with
local files only and never touches the network.

After the download the script loads the encoder exactly as the server will, encodes the
shortest cached gallery sequence and compares it with its cached states: the same self-check
that stops the server at startup. It prints the largest difference and the encoding time.

Needs the ``live`` extra:  pip install -e "backend[serve,live]"
Put ``ESM_CACHE_DIR`` (the directory printed at the end) and ``ENCODER_MODE=live`` in ``.env``.
"""

import argparse
from collections.abc import Sequence
import importlib
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_CONFIG = REPO_ROOT / "configs" / "serving" / "engine.json"
DEFAULT_CACHE = REPO_ROOT / "data" / "esm2"
GALLERY = REPO_ROOT / "data" / "gallery" / "gallery.json"
#: Only the files the tokenizer and the safetensors model need.
ALLOW = ["*.json", "*.txt", "*.safetensors", "*.model"]


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-only", action="store_true", help="skip the download")
    args = parser.parse_args(argv)

    cache_name = json.loads(ENGINE_CONFIG.read_text(encoding="utf-8"))["encoder"]["residue_cache"]
    cache_dir = REPO_ROOT / "data" / "embeddings" / cache_name
    states_spec = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))["spec"][
        "states"
    ]
    hub = Path(args.cache_dir or os.environ.get("ESM_CACHE_DIR") or DEFAULT_CACHE).resolve()
    checkpoint, revision = str(states_spec["checkpoint"]), str(states_spec["revision"])
    _say(f"checkpoint {checkpoint} @ {revision}")
    _say(f"hub cache  {hub}")

    if not args.check_only:
        try:
            hub_api = importlib.import_module("huggingface_hub")
        except ModuleNotFoundError:
            _say('missing the live extra: pip install -e "backend[serve,live]"')
            return 1
        hub.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        path = hub_api.snapshot_download(
            repo_id=checkpoint, revision=revision, cache_dir=str(hub), allow_patterns=ALLOW
        )
        _say(f"downloaded to {path} in {time.perf_counter() - started:.1f} s")

    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from protloc.data.records import normalize_sequence  # noqa: PLC0415
    from protloc.data.residue_cache import ResidueStore  # noqa: PLC0415
    from protloc.inference.live_encoder import (  # noqa: PLC0415
        LiveEncoderError,
        load_live_encoder,
    )

    store = ResidueStore(cache_dir)
    gallery = json.loads(GALLERY.read_text(encoding="utf-8"))["examples"]
    cached = [g for g in gallery if g["sequence_sha256"] in store.lengths]
    if not cached:
        _say("no cached gallery sequence to self-check against; run scripts/build_gallery.py")
        return 1
    check = min(cached, key=lambda g: len(g["sequence"]))
    started = time.perf_counter()
    try:
        live = load_live_encoder(
            states_spec,
            hub,
            args.device,
            (normalize_sequence(check["sequence"]), store.states(check["sequence_sha256"])),
        )
    except LiveEncoderError as error:
        _say(f"SELF-CHECK FAILED: {error}")
        return 1
    seconds = time.perf_counter() - started
    _say(
        f"self-check on {check['id']} ({len(check['sequence'])} residues): max |live - cached| = "
        f"{live.self_check_max_abs_diff} (load + encode {seconds:.1f} s) - OK"
    )
    _say("")
    _say("Add to .env:")
    _say("  ENCODER_MODE=live")
    _say(f"  ESM_CACHE_DIR={hub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
