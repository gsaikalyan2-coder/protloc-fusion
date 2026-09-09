"""Proteome-scale runs: chunking, progress, cancellation, resume.

Layer L0.

Drives ``inference``; owns queue mechanics, not algorithms.

The import rules for this package are asserted by
``backend/tests/test_layering.py`` and enforced in CI.
"""
