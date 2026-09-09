"""Benchmarks that call the SERVED path over HTTP.

Layer L0.

Speaks to the API, never to model objects, so the paper's numbers are the user's numbers.

The import rules for this package are asserted by
``backend/tests/test_layering.py`` and enforced in CI.
"""
