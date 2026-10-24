"""Stable API error codes (Phase 17): one code per way a request can fail.

Every failure the live app can report maps to exactly one :class:`ErrorSpec`. The code is the
contract: clients branch on ``error.code``, never on the wording of the message, and the
catalog says whether a retry can help. ``detail`` carries the specific, human-readable reason
(for a FASTA fault, the line number); ``ErrorSpec.message`` is the general sentence.

Nothing here is biology and nothing is a model threshold. The two limits are request limits
and are stated in ``/demo/version``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

#: Longest single sequence the service will look at (the scanner's cap is the same).
MAX_SEQUENCE_LENGTH: Final[int] = 5000
#: Largest request body field, in characters; a backstop against runaway payloads.
MAX_REQUEST_CHARS: Final[int] = 2_000_000


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """One stable error: its code, HTTP status, general message and whether a retry can help."""

    code: str
    status: int
    message: str
    retryable: bool


_SPECS: Final[tuple[ErrorSpec, ...]] = (
    ErrorSpec("invalid_input", 422, "The input was rejected.", False),
    ErrorSpec("request_too_large", 413, "The request is larger than the service accepts.", False),
    ErrorSpec("sequence_too_long", 422, "The sequence is longer than the service accepts.", False),
    ErrorSpec(
        "batch_too_large", 422, "The FASTA holds more sequences than one batch allows.", False
    ),
    ErrorSpec("fasta_empty", 400, "No FASTA records were found.", False),
    ErrorSpec("fasta_no_header", 400, "Sequence text appears before any '>' header line.", False),
    ErrorSpec("fasta_empty_record", 400, "A FASTA record has a header but no residues.", False),
    ErrorSpec("fasta_invalid_character", 400, "A FASTA line holds a non-letter character.", False),
    ErrorSpec("unknown_group", 400, "The organism group is not one this service knows.", False),
    ErrorSpec("group_not_served", 422, "No trained head covers that organism group.", False),
    ErrorSpec("not_cached", 422, "This build does not serve that sequence.", False),
    ErrorSpec(
        "invalid_sequence", 422, "The sequence holds characters the model cannot use.", False
    ),
    ErrorSpec(
        "accession_not_cached", 404, "No cached sequence has that accession or entry name.", False
    ),
    ErrorSpec("catalog_unavailable", 503, "The cached-sequence catalog is not loaded.", False),
    ErrorSpec("unknown_job", 404, "The server does not know this job.", False),
    ErrorSpec("result_expired", 410, "This result has expired.", False),
    ErrorSpec("job_not_finished", 409, "The job has not finished yet.", True),
    ErrorSpec("no_results", 409, "The job finished without results to download.", False),
    ErrorSpec("export_unsupported", 422, "This job cannot be exported in that format.", False),
    ErrorSpec("structure_bad_hash", 400, "The structure key is not a SHA-256 digest.", False),
    ErrorSpec("structure_not_found", 404, "No structure is stored for this sequence.", False),
    ErrorSpec("structure_refused", 500, "The stored structure failed its integrity check.", True),
    ErrorSpec("space_not_built", 404, "The protein-space map has not been built.", False),
    ErrorSpec("space_refused", 500, "The protein-space map failed its integrity check.", False),
    ErrorSpec("demo_disabled", 503, "The service is switched off.", True),
    ErrorSpec("model_error", 500, "The model could not score this sequence.", False),
    ErrorSpec("job_failed", 500, "The job failed.", True),
)

#: Every code the service can emit, keyed by code.
ERRORS: Final[dict[str, ErrorSpec]] = {spec.code: spec for spec in _SPECS}

#: Engine refusal codes (``EngineRefusal.code``) to API codes. Anything else is ``job_failed``.
REFUSAL_CODES: Final[dict[str, str]] = {
    "unknown_group": "unknown_group",
    "unsupported_group": "group_not_served",
    "invalid_sequence": "invalid_sequence",
    "not_in_embedding_cache": "not_cached",
    "model_error": "model_error",
}


def api_code_for_refusal(engine_code: str) -> str:
    """The API code for an engine refusal code."""
    return REFUSAL_CODES.get(engine_code, "job_failed")


class ApiError(Exception):
    """A request failed in a way the catalog names; carries the specific reason."""

    def __init__(self, code: str, detail: str | None = None, line: int | None = None) -> None:
        """Look the code up (an unknown code is a bug and raises ``KeyError``)."""
        self.spec = ERRORS[code]
        self.detail = detail or self.spec.message
        self.line = line
        super().__init__(self.detail)

    def body(self) -> dict[str, Any]:
        """The JSON body: ``detail`` for older clients plus the structured ``error``."""
        return {"detail": self.detail, "error": error_object(self.spec, self.detail, self.line)}


def error_object(spec: ErrorSpec, detail: str, line: int | None = None) -> dict[str, Any]:
    """The structured ``error`` member shared by HTTP errors and failed jobs."""
    return {
        "code": spec.code,
        "message": spec.message,
        "detail": detail,
        "retryable": spec.retryable,
        "line": line,
    }
