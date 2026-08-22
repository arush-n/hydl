"""Capture from the native server, for explicitly pinned worlds.

``capture_structures``  authored structure markers for pinned V2 worlds
``capture_traversal``   native traversal graphs for pinned V2 worlds
"""


# Not re-exported here: `capture_structures`, `capture_traversal`.
# They are CLI entrypoints that bootstrap sys.path and import `jax_port.*`
# absolutely, which only resolves when experimental/worldgen-v2 is on the
# path. Importing them from this __init__ breaks the
# `experimental.worldgen-v2.jax_port.*` import route that arena uses.
# Run them directly instead.

from __future__ import annotations



__all__ = [
]
