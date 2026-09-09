"""ProtLoc Fusion — unified protein annotation from one shared representation.

Seventeen modules arranged in dependency layers, from ``domain`` (which imports
nothing internal) up to the delivery layer. A module may import only from a
strictly deeper layer; ``training`` is an entry point and is imported by nothing.
See ``FOLDER_STRUCTURE.md`` for the reasoning and
``backend/tests/test_layering.py`` for the machine-checked statement of it.
"""

__version__ = "0.1.0"
