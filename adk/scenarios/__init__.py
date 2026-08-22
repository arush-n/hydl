"""Scenarios — scenes that each isolate one skill, with a declared optimum.

A scenario is a scene plus what it teaches and how you know it worked.  The
environment must be able to support a **perfect** agent even when no agent can
reach it yet, so every scenario declares a ``ceiling`` derived from measured
constants.  Without one, a low score cannot be told apart from an unwinnable
task -- and this project has spent real time on exactly that ambiguity.

    from adk.scenarios import SCENARIOS

    spacing = SCENARIOS["combat/spacing"]
    handle = spacing.build(batch=2)
    ...
    print(spacing.normalised(trajectory))   # 0.0 = floor, 1.0 = perfect

See [`DEV.md`](DEV.md) for what each scenario isolates and why its
ceiling is what it is.
"""

from adk.scenarios.combat import (
    COMBO,
    DISCIPLINE,
    HORIZON,
    PRESSURE,
    SCENARIOS,
    SPACING,
)
from adk.scenarios.core import Scenario, Trajectory, registry
from adk.scenarios import probes

__all__ = [
    "COMBO",
    "probes",
    "DISCIPLINE",
    "HORIZON",
    "PRESSURE",
    "SCENARIOS",
    "SPACING",
    "Scenario",
    "Trajectory",
    "registry",
]
