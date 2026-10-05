"""Strict FASTA parsing: each fault names its code and its line."""

import pytest

from protloc.domain.errors import ApiError
from protloc.inference.fasta import parse_fasta_strict


def _fault(text: str, max_records: int = 5, max_length: int = 50) -> ApiError:
    with pytest.raises(ApiError) as caught:
        parse_fasta_strict(text, max_records, max_length)
    return caught.value


def test_parses_records_and_joins_wrapped_lines() -> None:
    text = ">a first\nMKT\nLLV\n\n>b\nAAAA\n"
    assert parse_fasta_strict(text, 5, 50) == [("a", "MKTLLV"), ("b", "AAAA")]


def test_a_bare_header_gets_a_numbered_id() -> None:
    assert parse_fasta_strict(">\nMKT\n", 5, 50) == [("seq1", "MKT")]


def test_spaces_inside_a_residue_line_are_dropped() -> None:
    assert parse_fasta_strict(">a\nMK T LL\n", 5, 50) == [("a", "MKTLL")]


def test_residues_before_a_header_name_the_line() -> None:
    error = _fault("\nMKT\n>a\nMKT\n")
    assert (error.spec.code, error.line) == ("fasta_no_header", 2)
    assert "line 2" in error.detail


def test_a_non_letter_names_the_line_and_the_character() -> None:
    error = _fault(">a\nMKT\nMK1T\n")
    assert (error.spec.code, error.line) == ("fasta_invalid_character", 3)
    assert "'1'" in error.detail


def test_a_header_without_residues_names_its_line() -> None:
    error = _fault(">a\nMKT\n>b\n>c\nMKT\n")
    assert (error.spec.code, error.line) == ("fasta_empty_record", 3)
    assert "'b'" in error.detail


def test_a_trailing_empty_record_is_caught() -> None:
    assert _fault(">a\nMKT\n>b\n").spec.code == "fasta_empty_record"


def test_a_too_long_sequence_names_its_header_line() -> None:
    error = _fault(">a\nMKT\n>b\n" + "A" * 51 + "\n")
    assert (error.spec.code, error.line) == ("sequence_too_long", 3)


def test_the_limit_itself_is_accepted() -> None:
    assert len(parse_fasta_strict(">a\n" + "A" * 50, 5, 50)[0][1]) == 50


def test_too_many_records_is_batch_too_large() -> None:
    error = _fault(">a\nM\n>b\nM\n>c\nM\n", max_records=2)
    assert (error.spec.code, error.line) == ("batch_too_large", 5)


def test_exactly_the_record_limit_is_accepted() -> None:
    assert len(parse_fasta_strict(">a\nM\n>b\nM\n", 2, 50)) == 2


@pytest.mark.parametrize("text", ["", "   \n\n", "\r\n"])
def test_nothing_to_parse_is_fasta_empty(text: str) -> None:
    assert _fault(text).spec.code == "fasta_empty"


def test_windows_line_endings_are_fine() -> None:
    assert parse_fasta_strict(">a\r\nMKT\r\n", 5, 50) == [("a", "MKT")]
