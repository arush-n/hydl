"""Memories level (gen 7, row 11m, stage 4). Switch ``Constructs.crafting_memories``.

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (s), jar A6C310DD; receipts
``~/.claude/jobs/31947bd7/tmp/phase4i/receipts/repo_mem_*.json``): ``Furniture_Ancient_Chest_Small`` (requires level 2) is
refused at level 1 (``craft_recipe_memories_unavailable``, 0 recorded) and finished at level 2 (10 recorded, two
Ingredient_Bar_Copper and three Wood_Deadwood_Planks consumed and one chest added on the request tick).

The level is ``MemoriesPlugin.getMemoriesLevel``: 1 + the number of thresholds the universe-wide recorded count reaches
(``MemoriesAmountPerLevel``, default 10, 25, 50, 100, 200), so levels 1 to 6. The count is settable, so the level is not
fixed natively; the bridge's readout is ``native_memories_level``.

Gen 5 (``level_one``) returns the original: every recipe needs level 1 or less. Gen 7 (``recorded_count``) hooks
``compile_region_fieldcraft_assets``: ``memories_level=`` sets the scenario level (1 by default). A caller passes the bridge's
``native_memories_level`` when the bridge reports one; none does yet (the reader is in worldgen/native_world_actions.py). The overlay patches admit requirements up to level 6,
resolve each recipe's own level and carry the level in the fieldcraft and bench contexts; the gate is the engine's
``_recipe_precondition`` (``context.memories_level >= required_memories_level``).
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "crafting_memories"
IMPLEMENTED = True
LEVEL_ONE = "level_one"
RECORDED_COUNT = "recorded_count"
#: MemoriesAmountPerLevel (0.7.2 default): level i + 2 once the recorded count reaches the i-th threshold.
THRESHOLDS = (10, 25, 50, 100, 200)
MAX_LEVEL = len(THRESHOLDS) + 1


def _enabled(profile: Any) -> bool:
    value = profile.constructs.crafting_memories
    if value == LEVEL_ONE:
        return False
    if value != RECORDED_COUNT:
        raise ValueError(f"unknown crafting_memories switch {value!r}")
    return True


def checked_level(level: Any) -> int:
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= MAX_LEVEL:
        raise ValueError(f"memories level must be an integer in [1, {MAX_LEVEL}], got {level!r}")
    return level


def level_from_recorded_count(count: int) -> int:
    """The universe-wide memories level for a recorded count (``MemoriesPlugin.getMemoriesLevel``)."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("recorded count must be a nonnegative integer")
    return 1 + sum(1 for threshold in THRESHOLDS if count >= threshold)


def recorded_count_for_level(level: int) -> int:
    """The smallest recorded count at a level (level 1: 0). The bridge's ``memories_level`` reset option sets it."""

    level = checked_level(level)
    return 0 if level == 1 else THRESHOLDS[level - 2]


def construct(original: Any, profile: Any) -> Any:
    """Hook for ``compile_region_fieldcraft_assets``: the scenario memories level (``memories_level=``)."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def compile_region_fieldcraft_assets(recording_path, inventory_layout, *args, memories_level=None, **kwargs):
        assets = original(recording_path, inventory_layout, *args, **kwargs)
        level = 1 if memories_level is None else checked_level(memories_level)
        return assets._replace(memories_level=level)

    return compile_region_fieldcraft_assets
