"""Player slope and wall limits. Switch ``Constructs.slope_limits``; P2.

Inputs (gen-7 tables): ``movement/config:Default`` ``MaxSlopeAngleDegrees`` 30
and ``MaxWallAngleDegrees`` 15 (keys absent in 0.5.10). Differential: slope
fixture, position error <= 5e-4 per step (FIDELITY row 1d).
"""

from __future__ import annotations

from typing import Any

SWITCH = "slope_limits"
IMPLEMENTED = False


def limit(state: Any, contact_normal: Any, params: Any) -> Any:
    raise NotImplementedError("slope-limit construct: design only (mechanics_072.md)")
