"""Compiled bundle artifacts and how they are loaded.

``bundle``            immutable catalog compiler for exact V2 JAX worlds
``compile_bundle``    command-line compiler over an analyzed capture corpus
``benchmark_bundle``  measure bundle materialization without running a game
``jax_loader``        selected-artifact materialization for exact JAX worlds
``traversal_pack``    native traversal sidecar packs for bundle artifacts
"""


# Not re-exported here: `benchmark_bundle`, `compile_bundle`.
# They are CLI entrypoints that bootstrap sys.path and import `jax_port.*`
# absolutely, which only resolves when experimental/worldgen-v2 is on the
# path. Importing them from this __init__ breaks the
# `experimental.worldgen-v2.jax_port.*` import route that arena uses.
# Run them directly instead.

from __future__ import annotations

from . import bundle as bundle
from . import jax_loader as jax_loader
from . import traversal_pack as traversal_pack


__all__ = [
    "bundle",
    "jax_loader",
    "traversal_pack",
]
