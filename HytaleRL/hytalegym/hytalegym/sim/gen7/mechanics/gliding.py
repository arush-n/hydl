"""Gliding. Switch ``Constructs.gliding``; on demand.

Inputs (gen-7 tables): ``weapons/item:Glider_Scrap`` (0.7.2; 0.5.10 has
``Glider``), ``stats/stat:GlidingActive``, the Stamina ``Gliding`` drain rule
(-0.1 per 0.1 s) and the StaminaRegenDelay rules. Catalogue numbers (speed 10,
falls <= 2/s, stamina 1/s) are to be confirmed by the differential.
"""

from __future__ import annotations

from typing import Any

SWITCH = "gliding"
IMPLEMENTED = False


def step(state: Any, wish: Any, params: Any) -> Any:
    raise NotImplementedError("gliding construct: design only (mechanics_072.md)")
