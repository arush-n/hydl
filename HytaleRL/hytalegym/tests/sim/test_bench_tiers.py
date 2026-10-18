"""Bench tiers (gen 7, row 11b): a bench row's tier is the scenario's (a fresh bench, 1, by default); gen 5 keeps the zero fill.

The oracle's replay (FIDELITY-0.7.2.md section (s)) is in test_crafting_oracle_replay.py; this file pins the helpers and the
compile hook's wrapper (a fake compile with the engine module's names, so no archive is read).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.crafting import identity_sha256_words
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import bench_tiers
from hytalegym.sim.gen7.profile import PROFILE as GEN7


class _Assets(NamedTuple):
    bench_id_hash: object
    bench_tier_level: object
    memories_level: int = 1


def _rows(*bench_ids: str) -> np.ndarray:
    return np.asarray([identity_sha256_words(bench_id) for bench_id in bench_ids], dtype=np.uint32)


def fake_compile(recording_path, inventory_layout, **kwargs) -> _Assets:
    """A stand-in for compile_region_fieldcraft_assets: two bench rows, zero-filled as the engine does at HEAD."""

    rows = _rows("Farmingbench", "Armor_Bench")
    return _Assets(bench_id_hash=jnp.asarray(rows), bench_tier_level=jnp.zeros((2,), dtype=jnp.int32))


def test_a_fresh_bench_is_tier_one_and_an_integer_sets_every_row() -> None:
    rows = _rows("Farmingbench", "Armor_Bench")
    assert bench_tiers.row_tiers(rows, None, identity_sha256_words) == [1, 1]
    assert bench_tiers.row_tiers(rows, 8, identity_sha256_words) == [8, 8]


def test_a_mapping_sets_only_the_named_bench() -> None:
    rows = _rows("Farmingbench", "Armor_Bench")
    assert bench_tiers.row_tiers(rows, {"Farmingbench": 2}, identity_sha256_words) == [2, 1]


@pytest.mark.parametrize("bad", [0, 9, True, "2", 1.0])
def test_a_tier_outside_one_to_eight_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        bench_tiers.row_tiers(_rows("Farmingbench"), bad, identity_sha256_words)
    with pytest.raises(ValueError):
        bench_tiers.row_tiers(_rows("Farmingbench"), {"Farmingbench": bad}, identity_sha256_words)


def test_gen5_returns_the_compile_function_and_gen7_sets_the_scenario_tiers() -> None:
    assert bench_tiers.construct(fake_compile, GEN5) is fake_compile
    compiled = bench_tiers.construct(fake_compile, GEN7)
    assert np.asarray(compiled(None, None).bench_tier_level).tolist() == [1, 1]
    assert np.asarray(compiled(None, None, bench_tiers={"Armor_Bench": 4}).bench_tier_level).tolist() == [1, 4]
    assert np.asarray(compiled(None, None, bench_tiers=6).bench_tier_level).tolist() == [6, 6]
    with pytest.raises(ValueError):
        compiled(None, None, bench_tiers=9)


def test_the_gen7_wrapper_survives_a_second_wrapper_on_the_same_function() -> None:
    """crafting_memories wraps the same compile function; the tier hook reads the engine globals through unwrap."""

    from hytalegym.sim.gen7.mechanics import crafting_memories

    twice = crafting_memories.construct(bench_tiers.construct(fake_compile, GEN7), GEN7)
    assert np.asarray(twice(None, None, bench_tiers=3, memories_level=2).bench_tier_level).tolist() == [3, 3]
    assert int(twice(None, None, memories_level=2).memories_level) == 2
