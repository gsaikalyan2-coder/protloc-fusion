"""The conductor. Owns the pipeline ORDER and no algorithms.

Layer L1.

The one module that knows the sequence features -> model -> calibration ->
consistency -> interpretability -> evidence.

The import rules for this package are asserted by
``backend/tests/test_layering.py`` and enforced in CI.
"""
