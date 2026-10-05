"""Strict FASTA parsing (Phase 17): every fault is reported with its line number.

Stdlib only. A header line starts with ``>``; the first token after it is the record id (a
bare ``>`` gets ``seq<n>``). Residue lines may hold spaces, which are dropped; any other
non-letter is a fault, as is residue text before the first header, a header with no residues,
a sequence over the length limit and more records than the batch limit. Nothing is silently
repaired.
"""

from __future__ import annotations

from protloc.domain.errors import ApiError

_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _close(
    records: list[tuple[str, str]],
    rec_id: str | None,
    header_line: int,
    parts: list[str],
    limits: tuple[int, int],
) -> None:
    """Finish the record in progress, if any, applying ``(max_records, max_length)``."""
    max_records, max_length = limits
    if rec_id is None:
        return
    residues = "".join(parts)
    if not residues:
        raise ApiError(
            "fasta_empty_record",
            f"line {header_line}: record '{rec_id}' has no residues.",
            header_line,
        )
    if len(residues) > max_length:
        raise ApiError(
            "sequence_too_long",
            f"line {header_line}: record '{rec_id}' has {len(residues)} residues; "
            f"the limit is {max_length}.",
            header_line,
        )
    if len(records) >= max_records:
        raise ApiError(
            "batch_too_large",
            f"line {header_line}: more than {max_records} sequences; the batch limit is "
            f"{max_records}.",
            header_line,
        )
    records.append((rec_id, residues))


def parse_fasta_strict(text: str, max_records: int, max_length: int) -> list[tuple[str, str]]:
    """``(id, residues)`` records, or an :class:`ApiError` naming the first fault and its line."""
    records: list[tuple[str, str]] = []
    rec_id: str | None = None
    header_line = 0
    parts: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            _close(records, rec_id, header_line, parts, (max_records, max_length))
            tokens = line[1:].split()
            rec_id = tokens[0] if tokens else f"seq{len(records) + 1}"
            header_line, parts = number, []
            continue
        if rec_id is None:
            raise ApiError(
                "fasta_no_header",
                f"line {number}: sequence text before any '>' header line.",
                number,
            )
        compact = "".join(line.split())
        bad = next((c for c in compact if c not in _LETTERS), None)
        if bad is not None:
            raise ApiError(
                "fasta_invalid_character",
                f"line {number}: {bad!r} is not a letter. Residues are letters only.",
                number,
            )
        parts.append(compact)
    _close(records, rec_id, header_line, parts, (max_records, max_length))
    if not records:
        raise ApiError("fasta_empty", "No FASTA records were found (a record starts with '>').")
    return records
