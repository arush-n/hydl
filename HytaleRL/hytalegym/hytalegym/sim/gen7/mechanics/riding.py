"""Riding. Switch ``Constructs.riding``; on demand.

Inputs (gen-7 tables): ``movement/config:Mount`` (new in 0.7.2) and the mount
NPC roles in ``npc_roles``. Catalogue numbers (rider speed 8, sprint 13.2,
jump 14) are to be confirmed by the differential.
"""

from __future__ import annotations

from typing import Any

SWITCH = "riding"
IMPLEMENTED = False


def step(state: Any, wish: Any, params: Any) -> Any:
    raise NotImplementedError("riding construct: design only (mechanics_072.md)")
