"""Dependency root: dataclasses, enums and port protocols.

Layer L-1.

Imports nothing from ``protloc``. Every other module may depend on it.

The import rules for this package are asserted by
``backend/tests/test_layering.py`` and enforced in CI.
"""
