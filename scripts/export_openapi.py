"""Freeze the HTTP contract as ``docs/api/openapi.yaml``.

The document is a projection of the application, never a parallel description of
it. Hand-writing the YAML would create a second source of truth that drifts from
the code within one phase; generating it means the only way to change the
contract is to change the models, which is also the only way to change what the
server actually does.

CI runs this and fails on a diff, the same arrangement that already protects
``frontend/src/styles/tokens.css``. So a schema change that someone forgot to
regenerate is a red build rather than a frontend that compiles against a contract
the backend no longer serves.

The application is built against settings declared here rather than read from the
environment. None of them appear in the output — CORS origins and ports are
runtime concerns — and requiring a filled ``.env`` would make the generated file
depend on whose machine generated it, which is precisely what the CI diff cannot
tolerate.

Run it from the repository root::

    python scripts/export_openapi.py            # write the document
    python scripts/export_openapi.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

from protloc.api.app import create_app
from protloc.config.settings import Settings
from protloc.domain.contracts import ReproducibilityEnvelope

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "api" / "openapi.yaml"

_HEADER = """\
# GENERATED FILE — DO NOT EDIT.
#
# Produced by scripts/export_openapi.py from the FastAPI application in
# backend/protloc/api/. Edit the Pydantic models in
# backend/protloc/domain/contracts.py and the route decorators, then regenerate.
#
# CI regenerates this file and fails on a diff, so a hand edit is reverted by the
# next build rather than quietly becoming a second source of truth.
#
# This is the frozen contract. A change to it is a change a client may have to
# absorb: see API_VERSION in backend/protloc/domain/contracts.py.
"""


def _build_document() -> dict[str, Any]:
    """Return the OpenAPI document the application produces."""
    app = create_app(
        Settings(
            APP_ENV="development",
            LOG_LEVEL="INFO",
            API_HOST="127.0.0.1",
            API_PORT=8000,
            CORS_ALLOWED_ORIGINS="http://localhost:5173",
        )
    )
    document: dict[str, Any] = app.openapi()

    # Pydantic flattens inheritance, so the four envelope fields appear on every
    # response schema and the envelope itself appears nowhere. That is the right
    # wire shape — CLAUDE.md's 202 example has them at the top level — but it
    # leaves a client with no name for the group, and the frontend needs one to
    # type a reproducibility footer without restating the fields by hand.
    #
    # So the schema is added explicitly, generated from the model rather than
    # written out, and left unreferenced: every response repeats the fields
    # inline, exactly as they arrive. Adding a $ref instead would nest them and
    # change the contract.
    envelope = ReproducibilityEnvelope.model_json_schema()
    envelope["description"] = (
        "The four reproducibility fields carried by every response, including "
        "202 and errors. They appear inline on each response rather than nested "
        "under a key; this schema names the group so a client can type it once. "
        "Every field is null while no registry bundle is loaded."
    )
    document["components"]["schemas"]["ReproducibilityEnvelope"] = envelope

    return document


#: FastAPI names a multipart request body after the route function that declares
#: it, producing ``Body_batch_batch_post``. That name reaches the frozen document
#: and, through it, the generated TypeScript, where it is meaningless to a reader.
#: The shape is documented as ``BatchSubmission`` in ``contracts.py``, so the
#: schema is renamed to match. Narrow and explicit: a rename that finds no target
#: is a no-op, and nothing else in the document is rewritten.
_SCHEMA_RENAMES = {"Body_batch_batch_post": "BatchSubmission"}


def _rename_schemas(document: dict[str, Any]) -> dict[str, Any]:
    """Apply :data:`_SCHEMA_RENAMES` to the document and every ``$ref`` to it."""
    rendered = yaml.safe_dump(document, sort_keys=True)
    for old, new in _SCHEMA_RENAMES.items():
        rendered = rendered.replace(f"schemas/{old}", f"schemas/{new}")
        rendered = rendered.replace(f"{old}:", f"{new}:")
        rendered = rendered.replace(f"title: {old}", f"title: {new}")
    loaded: dict[str, Any] = yaml.safe_load(rendered)
    return loaded


def _serialise(document: dict[str, Any]) -> str:
    """Render the document deterministically.

    ``sort_keys`` is on so that two runs of the same code produce byte-identical
    output regardless of dictionary insertion order; without it the CI diff would
    fail on noise.
    """
    body = yaml.safe_dump(
        document,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=88,
    )
    return f"{_HEADER}\n{body}"


def _assert_wellformed(document: dict[str, Any]) -> None:
    """Fail generation rather than emit a document that breaks the contract's rules.

    These are post-conditions on the artifact itself, checked here because a
    document that is merely *current* can still be wrong — and the whole point of
    freezing it is that a client can rely on it without reading the code.

    Raises:
        AssertionError: If any invariant is broken. Generation then writes
            nothing, so a broken document never reaches the repository.
    """
    schemas = document["components"]["schemas"]
    paths = document["paths"]

    if not document["openapi"].startswith("3.1"):
        raise AssertionError(f"expected OpenAPI 3.1, got {document['openapi']}")

    if "ReproducibilityEnvelope" not in schemas:
        raise AssertionError("the envelope must be a named schema; see _build_document")

    # ADR-0001: there is no synchronous prediction. A 200 on POST /predict would
    # mean someone reintroduced one, and this is the artifact a client reads.
    predict = paths["/predict"]["post"]["responses"]
    if "202" not in predict:
        raise AssertionError("POST /predict must declare 202")
    if "200" in predict:
        raise AssertionError(
            "POST /predict declares 200: the contract is asynchronous (ADR-0001). "
            "The inline fast path is a runtime convenience and is never documented "
            "as a contract."
        )

    envelope = set(schemas["ReproducibilityEnvelope"]["properties"])
    for name, schema in schemas.items():
        if not name.endswith(("Response", "Accepted", "Resource")):
            continue
        missing = envelope - set(schema.get("properties", {}))
        if missing:
            raise AssertionError(
                f"response schema {name} is missing {sorted(missing)}. "
                f"Contract rule 6: every response carries the envelope."
            )


def main() -> int:
    """Write or check ``docs/api/openapi.yaml``.

    Returns:
        ``0`` on success, or when the committed file is current under
        ``--check``; ``1`` when it is stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Exit 1 if the committed document is stale.",
    )
    args = parser.parse_args()

    document = _rename_schemas(_build_document())
    _assert_wellformed(document)
    rendered = _serialise(document)

    if args.check:
        if not OUTPUT.is_file():
            sys.stderr.write(f"{OUTPUT} does not exist; run this script without --check\n")
            return 1
        if OUTPUT.read_text(encoding="utf-8") != rendered:
            sys.stderr.write(
                f"{OUTPUT.relative_to(REPO_ROOT)} is stale.\n"
                f"Run: python scripts/export_openapi.py\n"
            )
            return 1
        sys.stdout.write(f"{OUTPUT.relative_to(REPO_ROOT)} is current\n")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    paths = len(_build_document()["paths"])
    sys.stdout.write(f"wrote {OUTPUT.relative_to(REPO_ROOT)} — {paths} paths\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
