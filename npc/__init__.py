"""Run a JAX-trained policy as a live Hytale server NPC.

Almost all of this port already existed. ``npc/loop.py`` records what was
already there and what was actually missing.

The module is named ``loop``, not ``serve``, because ``__init__`` re-exports a
function called ``serve``: had the module shared that name, ``import npc.serve``
would silently bind the function instead of the module.
"""

from npc.loop import (
    ActionSurfaceMismatch,
    ServeReport,
    assert_action_surface_matches,
    serve,
)

__all__ = [
    "ActionSurfaceMismatch",
    "ServeReport",
    "assert_action_surface_matches",
    "serve",
]
