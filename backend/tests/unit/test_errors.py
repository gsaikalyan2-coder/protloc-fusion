"""The error catalog (Phase 17): stable codes, valid statuses, no orphan mappings."""

import pytest

from protloc.domain.errors import (
    ERRORS,
    REFUSAL_CODES,
    ApiError,
    api_code_for_refusal,
)


def test_codes_are_unique_and_match_their_keys() -> None:
    assert all(code == spec.code for code, spec in ERRORS.items())
    assert len(ERRORS) == len({spec.code for spec in ERRORS.values()})


def test_every_status_is_an_http_error_status() -> None:
    assert all(400 <= spec.status <= 599 for spec in ERRORS.values())


def test_every_spec_has_a_message_that_ends_like_a_sentence() -> None:
    assert all(spec.message.endswith(".") for spec in ERRORS.values())


def test_every_refusal_mapping_points_at_a_real_code() -> None:
    assert set(REFUSAL_CODES.values()) <= set(ERRORS)


def test_an_unknown_refusal_code_falls_back_to_job_failed() -> None:
    assert api_code_for_refusal("something_new") == "job_failed"
    assert api_code_for_refusal("not_in_embedding_cache") == "not_cached"
    assert api_code_for_refusal("unsupported_group") == "group_not_served"


def test_retryable_only_where_a_retry_can_change_the_outcome() -> None:
    retryable = {code for code, spec in ERRORS.items() if spec.retryable}
    assert retryable == {"structure_refused", "demo_disabled", "job_failed"}
    for code in ("not_cached", "sequence_too_long", "invalid_input", "group_not_served"):
        assert not ERRORS[code].retryable


def test_an_unknown_code_is_a_bug_and_raises() -> None:
    with pytest.raises(KeyError):
        ApiError("no_such_code")


def test_the_body_keeps_detail_and_adds_a_structured_error() -> None:
    body = ApiError("fasta_no_header", "line 3: bad", 3).body()
    assert body["detail"] == "line 3: bad"
    assert body["error"] == {
        "code": "fasta_no_header",
        "message": ERRORS["fasta_no_header"].message,
        "detail": "line 3: bad",
        "retryable": False,
        "line": 3,
    }


def test_detail_defaults_to_the_general_message() -> None:
    assert ApiError("unknown_job").detail == ERRORS["unknown_job"].message
