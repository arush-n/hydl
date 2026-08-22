"""Environments: declaring a scene, and finding what a scene can reach.

Split from `adk/scenarios/`, which holds what an agent *does* — tasks, shaped
minigames, scripted probe policies, opponent policies. This package holds what
the agent does it *in*: scene construction and terrain analysis.

    from adk.environments import SceneConfig
    scene = SceneConfig(weapon="iron_daggers", world="region").build()

    from adk.environments import water
    water.survey()      # where the captured regions actually hold water

The reason this is a package rather than a module: nearly every wrong result in
this project came from a scene that could not express the thing being measured
— `open_flat` pinning world queries true, an opponent parked outside its own
reach, a batch of byte-identical rows, oxygen that never moved because no scene
contained fluid. Scene construction and "can this scene reach that mechanic"
are the same problem, so they live together.

`adk/validation/region.py` stays where it is: it is fixture-identity and
evidence-pinning infrastructure with its own test suite, not scene declaration.
"""

from __future__ import annotations

__all__ = ["SceneConfig", "BuiltScene", "OPPONENT_CLOSES", "config", "water",
           "terrain", "blocks", "zones"]

#: Submodules that read artifacts only -- no JAX, no scene construction.
_SURVEYS = ("water", "terrain", "blocks", "zones")


def __getattr__(name: str):
    # Lazy so importing `water` does not drag in `config` and the training
    # stack with it. Note this does NOT avoid JAX: `adk/__init__.py` imports
    # `adk.architecture`, so `import adk` already loads it. Measured, after
    # claiming otherwise here and being wrong.
    import importlib

    if name == "config" or name in _SURVEYS:
        return importlib.import_module(f"{__name__}.{name}")
    if name in ("SceneConfig", "BuiltScene", "OPPONENT_CLOSES"):
        return getattr(importlib.import_module(f"{__name__}.config"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
