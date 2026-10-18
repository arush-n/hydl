"""Memories level (gen 7, row 11m): the universe-wide recorded count sets the level (1 to 6); gen 5 keeps level 1.

Thresholds are MemoriesPlugin's MemoriesAmountPerLevel (0.7.2 default 10, 25, 50, 100, 200; FIDELITY-0.7.2.md section (s)).
The receipts pin level 2 at 10 recorded (repo_mem_l2_ancient_chest.json) and level 1 at 0 (repo_mem_l1_ancient_chest.json).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import crafting_memories
from hytalegym.sim.gen7.profile import PROFILE as GEN7


class _Assets(NamedTuple):
    bench_id_hash: object
    bench_tier_level: object
    memories_level: int = 1


def fake_compile(recording_path, inventory_layout, **kwargs) -> _Assets:
    return _Assets(bench_id_hash=np.zeros((1, 8), dtype=np.uint32), bench_tier_level=np.zeros((1,), dtype=np.int32))


@pytest.mark.parametrize(
    ("count", "level"),
    [(0, 1), (9, 1), (10, 2), (24, 2), (25, 3), (49, 3), (50, 4), (99, 4), (100, 5), (199, 5), (200, 6), (10**6, 6)],
)
def test_the_recorded_count_sets_the_level_at_the_thresholds(count: int, level: int) -> None:
    assert crafting_memories.level_from_recorded_count(count) == level


@pytest.mark.parametrize("level", range(1, 7))
def test_each_level_has_a_smallest_recorded_count_that_gives_it(level: int) -> None:
    count = crafting_memories.recorded_count_for_level(level)
    assert crafting_memories.level_from_recorded_count(count) == level
    if level > 1:
        assert crafting_memories.level_from_recorded_count(count - 1) == level - 1


def test_the_receipt_levels_match_the_thresholds() -> None:
    assert crafting_memories.recorded_count_for_level(2) == 10  # repo_mem_l2_ancient_chest.json
    assert crafting_memories.recorded_count_for_level(1) == 0  # repo_mem_l1_ancient_chest.json
    assert crafting_memories.MAX_LEVEL == 6


@pytest.mark.parametrize("bad", [0, 7, True, "2", 2.0])
def test_a_level_outside_one_to_six_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        crafting_memories.checked_level(bad)
    with pytest.raises(ValueError):
        crafting_memories.recorded_count_for_level(bad)


def test_gen5_returns_the_compile_function_and_gen7_sets_the_scenario_level() -> None:
    assert crafting_memories.construct(fake_compile, GEN5) is fake_compile
    compiled = crafting_memories.construct(fake_compile, GEN7)
    assert int(compiled(None, None).memories_level) == 1
    assert int(compiled(None, None, memories_level=2).memories_level) == 2
    assert int(compiled(None, None, memories_level=6).memories_level) == 6
    with pytest.raises(ValueError):
        compiled(None, None, memories_level=7)
