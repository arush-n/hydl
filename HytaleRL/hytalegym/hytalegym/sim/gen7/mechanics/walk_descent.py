"""Walk controller minimum descent rate (0.7.2 code). Switch ``walk_descent``; P2.

Engine-code change, no table input (TRAINING-0.7.2.md item 8). Differential: a
stairs/ledge/slope fixture extending ``adk/tools/native_motion_differential.py``
(pass bar 5e-4 m per step; FIDELITY row 1e).
"""

from __future__ import annotations

from typing import Any

SWITCH = "walk_descent"
IMPLEMENTED = False


def descent_velocity(state: Any, params: Any) -> Any:
    raise NotImplementedError("walk descent construct: design only (mechanics_072.md)")
