"""HTTP delivery: FastAPI routers.

Layer L0.

Parses, validates, delegates to ``inference`` and returns. No business logic.

The import rules for this package are asserted by
``backend/tests/test_layering.py`` and enforced in CI.
"""
