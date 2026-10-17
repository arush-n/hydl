"""Climbing. Switch ``Constructs.climbing``; after P2.

Inputs (gen-7 tables): ``blocks/block:*`` ``climbable`` (57 resolved block
types; 44 files author ``IsClimbable``), ``movement/config:Default``
``ClimbSpeed`` 0.035, ``ClimbSpeedLateral`` 0.035, ``ClimbUpSprintSpeed`` 0.07,
``ClimbDownSprintSpeed`` 0.09. Gen 5 writes the climbing movement-state bit
false. Differential: climb up/down/lateral speeds on a ladder fixture.
"""

from __future__ import annotations

from typing import Any

SWITCH = "climbing"
IMPLEMENTED = False


def step(state: Any, wish: Any, cell: Any, params: Any) -> Any:
    raise NotImplementedError("climbing construct: design only (mechanics_072.md)")
