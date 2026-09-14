"""Wiring: which implementation the routers receive.

This module contains no business logic and no decisions — it names a class. It
exists because the routers must not construct their own dependencies (that would
be the boundary rule broken: deleting ``api/`` would then delete the wiring) and
must not import ``app.py`` (that would be a cycle).

Phase 7 changes one line here, swapping ``UnloadedProvenance`` for the
registry-backed implementation. No router changes, and no test of a router
changes, which is the whole point of the port.
"""

from functools import lru_cache

from protloc.domain.ports import ExampleGallery, JobService, ProvenanceSource
from protloc.inference.provenance import UnavailableJobService, UnloadedProvenance


@lru_cache(maxsize=1)
def provenance_source() -> ProvenanceSource:
    """Return the process-wide provenance source.

    Cached because it is a single immutable object for the process's lifetime;
    FastAPI calls it once per request through ``Depends``, and tests override it
    through ``app.dependency_overrides`` rather than by reaching in here.
    """
    return UnloadedProvenance()


@lru_cache(maxsize=1)
def job_service() -> JobService:
    """Return the process-wide job service.

    Phase 11 returns the queue-backed implementation from this function. Nothing
    else changes.
    """
    return UnavailableJobService()


@lru_cache(maxsize=1)
def example_gallery() -> ExampleGallery:
    """Return the example gallery."""
    return UnavailableJobService()


__all__ = ["example_gallery", "job_service", "provenance_source"]
