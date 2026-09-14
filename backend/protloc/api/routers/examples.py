"""The example gallery.

ADR-0001 §5 names the cost of an asynchronous contract: there is no instant demo
moment when showing the tool to a reviewer or a supervisor. The gallery is the
mitigation — pre-computed results, served immediately, and labelled as cached.

``computed_at`` is part of the contract for that reason. A fast answer presented
as though it had just been computed would misrepresent how the system behaves,
which is the one thing a tool about trustworthy prediction cannot afford to do.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from protloc.api.deps import example_gallery
from protloc.domain.contracts import ErrorResponse, ExamplesResponse
from protloc.domain.ports import ExampleGallery

router = APIRouter(tags=["examples"])

GalleryDep = Annotated[ExampleGallery, Depends(example_gallery)]


@router.get(
    "/examples",
    response_model=ExamplesResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": (
                "No registry bundle is loaded. The gallery's entries are cached "
                "results, which require the model that produced them. "
                "`error` is `no_registry_bundle`."
            ),
        },
    },
    summary="Gallery sequences with pre-computed, cached results",
    response_description="The gallery in display order. Each entry carries its computed_at.",
)
def examples(gallery: GalleryDep) -> ExamplesResponse:
    """Return the gallery."""
    return gallery.examples()


__all__ = ["router"]
