r"""Snapshot one UniProtKB/Swiss-Prot release into ``data/raw/``.

Downloads ``uniprot_sprot.dat.gz`` and its ``reldate.txt`` from UniProt's
``current_release`` directory, and writes the ``snapshot`` section of
``data/manifests/dataset_manifest.json``. That section holds the release and its
date, both read from ``reldate.txt`` and never typed, plus the UTC download
time, the source URLs, and the SHA-256 and size of each file.

The snapshot is written once. If ``data/raw/`` already holds a snapshot, the
script refuses to run: a new release is a new decision, not a re-run
(``docs/phases/05.md``). The raw file stays out of git: ``data/raw/`` is ignored and
the manifest records its SHA-256.

If ``UNIPROT_RELEASE`` is set in the environment and differs from
``reldate.txt``, the script stops before downloading the large file. That
variable is a guard, and the manifest is the record (decision D4).

The licence can't be read from a file this script can locate with certainty.
You confirm it yourself from UniProt's licence page, and pass both the licence
you read and the page's URL. Both are recorded (``docs/THIRD_PARTY.md`` row 13).

Run from the repository root, inside ``.venv``::

    python scripts/download_data.py --license-confirmed "CC BY 4.0" \
        --license-url https://www.uniprot.org/help/license
"""

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

from protloc.data.manifest import sha256_file, update_section, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
SNAPSHOT_RECORD = REPO_ROOT / "data" / "manifests" / "snapshot.json"

BASE_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/"
DATA_FILE = "uniprot_sprot.dat.gz"
RELDATE_FILE = "reldate.txt"
PREVIOUS_RELEASES = "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"

_RELDATE = re.compile(r"UniProtKB/Swiss-Prot Release (\d{4}_\d{2}) of (\d{2}-[A-Za-z]{3}-\d{4})")
_TIMEOUT_SECONDS = 120
_CHUNK = 1 << 20


def _fetch(url: str, target: Path) -> None:
    """Stream a URL to ``target`` through a partial file."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-HTTPS URL: {url}")
    partial = target.with_name(target.name + ".partial")
    # S310: the scheme is checked to be https:// immediately above.
    request = urllib.request.Request(url, headers={"User-Agent": "protloc-fusion/phase-3"})  # noqa: S310
    with (
        urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response,  # noqa: S310
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, _CHUNK)
    partial.replace(target)


def parse_reldate(text: str) -> tuple[str, str]:
    """Release string and release date from ``reldate.txt``."""
    match = _RELDATE.search(text)
    if match is None:
        raise ValueError("reldate.txt does not name a UniProtKB/Swiss-Prot release")
    release, day = match.groups()
    iso = datetime.strptime(day, "%d-%b-%Y").replace(tzinfo=UTC).date().isoformat()
    return release, iso


def main() -> int:
    """Download, verify and record the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--license-confirmed", required=True, help='e.g. "CC BY 4.0"')
    parser.add_argument("--license-url", required=True)
    args = parser.parse_args()

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        sys.stderr.write(
            f"{RAW_DIR.relative_to(REPO_ROOT)} already holds a snapshot. A snapshot is "
            "written once; delete it deliberately if a new release was decided.\n"
        )
        return 1
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    reldate = RAW_DIR / RELDATE_FILE
    _fetch(BASE_URL + RELDATE_FILE, reldate)
    release, release_date = parse_reldate(reldate.read_text(encoding="ascii"))
    guard = os.environ.get("UNIPROT_RELEASE", "")
    if guard and guard not in {release, "ENTER_YOUR_VALUE_HERE"}:
        sys.stderr.write(
            f"UNIPROT_RELEASE={guard} but reldate.txt says {release}. Nothing large was "
            "downloaded; fix .env or re-decide the release.\n"
        )
        return 1

    downloaded_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    data = RAW_DIR / DATA_FILE
    _fetch(BASE_URL + DATA_FILE, data)

    files: dict[str, object] = {}
    for path in (data, reldate):
        digest, size = sha256_file(path)
        files[path.relative_to(REPO_ROOT).as_posix()] = {
            "sha256": digest,
            "bytes": size,
            "source_url": BASE_URL + path.name,
        }

    snapshot: dict[str, object] = {
        "source": "UniProtKB/Swiss-Prot",
        "release": release,
        "release_date": release_date,
        "downloaded_at_utc": downloaded_at,
        "files": files,
        "license": args.license_confirmed,
        "license_url": args.license_url,
        "redownload_from": f"{PREVIOUS_RELEASES}release-{release}/",
    }
    write_json(SNAPSHOT_RECORD, snapshot)
    update_section(MANIFEST, "snapshot", snapshot)
    sys.stdout.write(
        f"snapshot: Swiss-Prot {release} ({release_date}), "
        f"{files[data.relative_to(REPO_ROOT).as_posix()]}\n"
        "next: python scripts/build_dataset.py\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
