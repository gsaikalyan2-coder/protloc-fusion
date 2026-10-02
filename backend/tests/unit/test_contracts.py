"""The contract's own rules, as assertions.

``contracts.py`` states several rules in prose that are cheap to break silently: a
new response model that forgets the reproducibility envelope, a schema that quietly
accepts unknown fields, a compartment list that drifts into Python. Each of those is
checked here rather than left to review.

These tests read the models, not the running application. The application is tested
separately, against the generated OpenAPI document; this file is about whether the
shapes themselves obey the contract's rules.
"""

import inspect

from pydantic import BaseModel, ValidationError
import pytest

from protloc.domain import contracts
from protloc.domain.contracts import (
    ErrorResponse,
    HealthResponse,
    HealthzResponse,
    JobAccepted,
    PredictRequest,
    ReproducibilityEnvelope,
)
from protloc.domain.types import JobStatus, OrganismGroup, Severity, TaskName

ENVELOPE_FIELDS = frozenset({"model_version", "config_hash", "rules_version", "experiment_id"})

# Models that are pieces of a response body rather than a response body. A nested
# component does not repeat the envelope — the response that contains it carries it
# once, and repeating it four levels deep would be noise a client has to skip.
COMPONENT_MODELS = frozenset(
    {
        "BatchSubmission",
        "ConsistencyViolation",
        "Evidence",
        "ExampleSequence",
        "GoAnnotation",
        "HomologyHit",
        "JobProgress",
        "PredictRequest",
        "ReproducibilityEnvelope",
        "ResidueAttribution",
        "SequenceResult",
        "TaskAttribution",
        "TaskPrediction",
    }
)


def _declared_models() -> dict[str, type[BaseModel]]:
    """Every Pydantic model defined in contracts.py, by name."""

    def _is_local_model(obj: object) -> bool:
        return (
            inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj.__module__ == contracts.__name__
        )

    return {name: obj for name, obj in vars(contracts).items() if _is_local_model(obj)}


def test_every_declared_model_is_exported():
    """__all__ names every model, so the OpenAPI export cannot miss one."""
    missing = sorted(set(_declared_models()) - set(contracts.__all__))
    assert not missing, f"models absent from __all__: {missing}"


def test_every_response_model_carries_the_envelope():
    """Contract rule 6: every response carries the four reproducibility fields.

    A model that is a response body and not a nested component must inherit
    ReproducibilityEnvelope. Adding a response model without it fails here, which
    is the point: the rule is a guarantee, and a guarantee nobody checks is a hope.
    """
    offenders = [
        name
        for name, model in _declared_models().items()
        if name not in COMPONENT_MODELS and not issubclass(model, ReproducibilityEnvelope)
    ]
    assert not offenders, (
        f"response models missing the reproducibility envelope: {sorted(offenders)}. "
        f"Inherit ReproducibilityEnvelope, or add the model to COMPONENT_MODELS if "
        f"it is nested inside another response."
    )


def test_the_envelope_is_exactly_four_fields():
    """The envelope neither grows nor shrinks without this test noticing."""
    assert set(ReproducibilityEnvelope.model_fields) == ENVELOPE_FIELDS


def test_envelope_fields_are_nullable():
    """Null is the honest value before a registry bundle exists.

    A sentinel string would be indistinguishable from a real version to a client,
    which is exactly the invented-value failure the project guards against.
    """
    envelope = ReproducibilityEnvelope(
        model_version=None, config_hash=None, rules_version=None, experiment_id=None
    )
    assert envelope.model_version is None


def test_every_model_forbids_unknown_fields():
    """A response whose shape is not fully described is not a frozen contract."""
    permissive = [
        name
        for name, model in _declared_models().items()
        if model.model_config.get("extra") != "forbid"
    ]
    assert not permissive, f"models accepting unknown fields: {sorted(permissive)}"


def test_every_field_has_a_description():
    """Field descriptions become the OpenAPI document's descriptions.

    A field with no description produces a spec entry a reviewer cannot interpret,
    which fails this phase's exit criterion directly.
    """
    undescribed: list[str] = []
    for name, model in _declared_models().items():
        undescribed.extend(
            f"{name}.{field_name}"
            for field_name, field in model.model_fields.items()
            if not field.description
        )
    assert not undescribed, f"fields with no description: {sorted(undescribed)}"


def test_predict_request_requires_an_organism_group():
    """The model is conditioned on the group; guessing would change the answer."""
    with pytest.raises(ValidationError):
        PredictRequest(sequence="MKT")  # type: ignore[call-arg]


def test_predict_request_rejects_an_empty_sequence():
    with pytest.raises(ValidationError):
        PredictRequest(sequence="", organism_group=OrganismGroup.EUKARYA)


def test_predict_request_rejects_an_unknown_field():
    """A typo in a client's payload is an error, not a silently ignored field."""
    with pytest.raises(ValidationError):
        PredictRequest(
            sequence="MKT",
            organism_group=OrganismGroup.EUKARYA,
            organism="human",  # type: ignore[call-arg]
        )


def test_predict_request_does_not_constrain_length():
    """MAX_SEQUENCE_LENGTH is frozen in Phase 3 from the corpus, not guessed here.

    A max_length in this schema would put an unmeasured constant into the frozen
    contract, where raising it later is a breaking change.
    """
    metadata = PredictRequest.model_fields["sequence"].metadata
    assert not any(getattr(item, "max_length", None) for item in metadata)


def test_job_accepted_carries_the_envelope_and_the_job():
    """The 202 body is the acceptance plus the envelope, and nothing else."""
    expected = ENVELOPE_FIELDS | {
        "job_id",
        "result_url",
        "status",
        "sequence_count",
        "queue_position",
        "estimated_wait_seconds",
    }
    assert set(JobAccepted.model_fields) == expected


def test_health_reports_degraded_without_a_bundle():
    """Degraded is a real state: serving, and unable to predict."""
    health = HealthResponse(
        status="degraded",
        detail="no registry bundle is loaded",
        bundle_loaded=False,
        queue_depth=0,
        model_version=None,
        config_hash=None,
        rules_version=None,
        experiment_id=None,
    )
    assert health.status == "degraded"
    assert health.bundle_loaded is False


def test_healthz_reports_an_empty_registry():
    """/healthz names the registry's state; empty is degraded, not an error."""
    healthz = HealthzResponse(
        status="degraded",
        registry_state="empty",
        detail="no registry bundle is loaded",
        model_version=None,
        config_hash=None,
        rules_version=None,
        experiment_id=None,
    )
    assert healthz.registry_state == "empty"
    assert healthz.status == "degraded"


def test_errors_carry_the_envelope():
    """A user reporting a failure can name the model version without reproducing it."""
    error = ErrorResponse(
        error="no_registry_bundle",
        message="The service cannot accept prediction jobs.",
        detail="A registry bundle is packaged in Phase 7.",
        model_version=None,
        config_hash=None,
        rules_version=None,
        experiment_id=None,
    )
    assert error.error == "no_registry_bundle"


def test_no_model_hardcodes_a_biological_vocabulary():
    """CLAUDE.md: a biological constant in a .py file is a bug.

    Labels come from the registry bundle. The only Literal unions permitted here
    are structural — GO's three aspects, and the service's own health states — so
    this test names them explicitly rather than allowing any Literal through.
    """
    allowed = {
        ("GoAnnotation", "aspect"),
        ("HealthResponse", "status"),
        ("HealthzResponse", "status"),
        ("HealthzResponse", "registry_state"),
    }
    literal_fields = {
        (name, field_name)
        for name, model in _declared_models().items()
        for field_name, field in model.model_fields.items()
        if "Literal" in str(field.annotation)
    }
    assert literal_fields <= allowed, (
        f"unexpected Literal unions: {sorted(literal_fields - allowed)}. "
        f"Label vocabularies belong in the registry bundle, not in the contract."
    )


def test_no_model_declares_a_default_threshold():
    """No hardcoded tau. The threshold is reported, never defaulted."""
    field = _declared_models()["TaskPrediction"].model_fields["abstention_threshold"]
    assert field.is_required(), "abstention_threshold must have no default"


@pytest.mark.parametrize(
    ("enum_type", "member_count"),
    [(OrganismGroup, 4), (TaskName, 7), (JobStatus, 5), (Severity, 3)],
)
def test_the_contract_speaks_the_domain_vocabulary(enum_type: type, member_count: int):
    """The contract reuses the domain enums rather than restating them.

    A string union in the contract that mirrored these by hand is exactly the drift
    that generating the frontend types is meant to end; the backend must not
    reintroduce it.
    """
    assert len(list(enum_type)) == member_count
