"""Hookshot tool and chain hook (0.7.2). Switch ``Constructs.hookshot``; P3, on demand.

Inputs (gen-7 tables): ``weapons/item:Tool_Hookshot`` and its reached graph,
``weapons/projectile_config:Projectile_Config_Hookshot``; the rune
``weapons/item:Rune_ChainHook`` -> ``root:Root_Ability_ChainHook`` ->
``interaction:Ability_ChainHook_Cast``/``_Hit`` and
``projectile_config:Projectile_Config_ChainHook``. Differential: fire, attach,
pull distance and timing against live 0.7.2 (catalogue: chain hook to 1.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SWITCH = "hookshot"
IMPLEMENTED = False


@dataclass(frozen=True)
class HookshotParams:
    launch_force: float
    gravity: float
    max_range: float
    pull_speed: float


def params_from_tables(tables: Any) -> HookshotParams:
    raise NotImplementedError("hookshot construct: design only (mechanics_072.md)")


def step(state: Any, command: Any, params: HookshotParams) -> Any:
    raise NotImplementedError("hookshot construct: design only (mechanics_072.md)")
