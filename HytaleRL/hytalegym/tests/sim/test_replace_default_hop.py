"""Replace default hop (gen 7, row 5f; ``sim/gen7/mechanics/replace_default_hop.py``).

Two kinds of test. The logic tests check the table of attacks against the asset (the Simple -> Replace -> Selector shape in
``sim/gen7/data/weapons.json``) and the writer wrapper on a stub. The render tests mount three gen-7 renders (the committed
overlay without the row, the row with the switch off, and the shipped overlay with the switch on) and run one request per
attack on a still iron-sword target, one request on step 1 (1-based steps, one tick is 1/30 s). Each lane is classified by the
attack it runs (a thrust or charged-mace request runs the listed Swing_Left; the signature attacks do not start).

Measured on the render (scratch lane_replace_hop, 2026-10-22, seed 570057, steps 1-based; "hit" is the first step the target's
health reads lower, "finish" the first step the attack's slot reads -1):

    attack                                  distance  committed hit  switch on hit  finish (both)
    Weapon_Sword_Primary_Swing_Left         1.5       8              9              14
    Weapon_Sword_Primary_Swing_Right        1.5       8              9              14
    Weapon_Sword_Primary_Swing_Down         1.0       18             19             26
    Weapon_Daggers_Primary_Swing_Left       1.5       7              8              10
    Weapon_Daggers_Primary_Swing_Right      1.5       7              8              10
    Weapon_Daggers_Primary_Stab_Left        1.5       7              8              10
    Weapon_Daggers_Primary_Stab_Right       1.5       7              8              10
    Weapon_Mace_Primary_Swing_Left          1.5       23             24             37
    Weapon_Mace_Primary_Swing_Right         1.5       23             24             37
    Weapon_Mace_Primary_Swing_Up_Left       1.5       23             24             37
    Weapon_Battleaxe_Primary_Swing_Down     1.5       18             19             34
    Weapon_Battleaxe_Primary_Swing_Down_Left  1.5     18             19             34
    Weapon_Battleaxe_Primary_Swing_Down_Right 1.5     18             19             34

Live 0.7.2 (sword left swing only, receipt in the module docstring): the health drop on tick 9, the chain finish on tick 14.
"""

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.arsenal.schema.contract import EVENT_FLAG_SERVER_SELECTOR
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import replace_default_hop as rdh
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import data_dir, mount_fresh, render_for_test, unmount

AGENT, TARGET = 0, 1
SEED = 570057
STEPS = 44
TICK = 1.0 / 30.0
SWORD = ("Weapon_Sword_Primary_Swing_Left", "Weapon_Sword_Primary_Swing_Right", "Weapon_Sword_Primary_Thrust",
         "Weapon_Sword_Signature_Vortexstrike")
DAGGERS = ("Weapon_Daggers_Primary_Swing_Left", "Weapon_Daggers_Primary_Swing_Right", "Weapon_Daggers_Primary_Stab_Left",
           "Weapon_Daggers_Primary_Stab_Right", "Weapon_Daggers_Signature_Razorstrike")
MACE = ("Weapon_Mace_Primary_Swing_Left", "Weapon_Mace_Primary_Swing_Right", "Weapon_Mace_Primary_Swing_Up_Left",
        "Weapon_Mace_Primary_Swing_Left_Charged")
BATTLEAXE = ("Weapon_Battleaxe_Primary_Swing_Down", "Weapon_Battleaxe_Primary_Swing_Down_Left",
             "Weapon_Battleaxe_Primary_Swing_Down_Right", "Weapon_Battleaxe_Signature_Whirlwind")
#: (weapon profile, target distance in blocks, attacks), one batch per group; each attack is one lane.
GROUPS = (
    ("iron_sword", 1.5, SWORD),
    ("iron_sword", 1.0, ("Weapon_Sword_Primary_Swing_Down",)),
    ("iron_daggers", 1.5, DAGGERS),
    ("iron_mace", 1.5, MACE),
    ("iron_battleaxe", 1.5, BATTLEAXE),
)
#: Requests that run a listed attack through the Primary root (a short press), with the attack they run.
ROUTED = {
    "Weapon_Sword_Primary_Thrust": "Weapon_Sword_Primary_Swing_Left",
    "Weapon_Mace_Primary_Swing_Left_Charged": "Weapon_Mace_Primary_Swing_Left",
}
#: Expected (committed hit, switch-on hit, finish) per listed attack, from the measurement in the module docstring.
EXPECTED = {
    "Weapon_Sword_Primary_Swing_Left": (8, 9, 14),
    "Weapon_Sword_Primary_Swing_Right": (8, 9, 14),
    "Weapon_Sword_Primary_Swing_Down": (18, 19, 26),
    "Weapon_Daggers_Primary_Swing_Left": (7, 8, 10),  # 0.7.2 dagger event and selector 0.066 (2026-10-23 pair)
    "Weapon_Daggers_Primary_Swing_Right": (7, 8, 10),  # 0.7.2 dagger event and selector 0.066 (2026-10-23 pair)
    "Weapon_Daggers_Primary_Stab_Left": (7, 8, 10),  # 0.7.2 dagger event and selector 0.066 (2026-10-23 pair)
    "Weapon_Daggers_Primary_Stab_Right": (7, 8, 10),  # 0.7.2 dagger event and selector 0.066 (2026-10-23 pair)
    "Weapon_Mace_Primary_Swing_Left": (23, 24, 37),
    "Weapon_Mace_Primary_Swing_Right": (23, 24, 37),
    "Weapon_Mace_Primary_Swing_Up_Left": (23, 24, 37),
    "Weapon_Battleaxe_Primary_Swing_Down": (18, 19, 34),
    "Weapon_Battleaxe_Primary_Swing_Down_Left": (16, 17, 34),  # 0.7.2 event 0.348 (overlay row melee_timing, 2026-10-23)
    "Weapon_Battleaxe_Primary_Swing_Down_Right": (18, 19, 34),
}


def _asset_rows() -> dict:
    return json.loads((data_dir("gen7") / "weapons.json").read_text(encoding="utf-8"))["rows"]


# ---- logic tests -----------------------------------------------------------------------------------------------------


def test_the_switch_is_off_in_gen5_and_on_in_gen7_and_gen5_returns_the_original() -> None:
    assert Constructs().replace_default_hop is False
    assert GEN5.constructs.replace_default_hop is False
    assert GEN7.constructs.replace_default_hop is True
    assert rdh.IMPLEMENTED is True
    sentinel = object()
    assert rdh.construct(sentinel, GEN5) is sentinel
    assert rdh.construct(sentinel, GEN7) is not sentinel


def test_the_listed_attacks_are_the_simple_replace_selector_rows_of_the_asset() -> None:
    rows = _asset_rows()
    assert len(rdh.REPLACE_DEFAULT_ATTACKS) == 13
    for name in sorted(rdh.REPLACE_DEFAULT_ATTACKS):
        document = rows[f"interaction:{name}"]["document"]
        assert document["Type"] == "Simple", name
        assert document["Next"]["Type"] == "Replace", name
        assert document["Next"]["DefaultValue"]["Interactions"] == [f"{name}_Selector"], name
    # The attacks the construct must leave alone are not Simple -> Replace -> Selector at their root.
    for name in ("Weapon_Sword_Primary_Thrust", "Weapon_Sword_Signature_Vortexstrike", "Weapon_Mace_Primary_Swing_Left_Charged"):
        assert name not in rdh.REPLACE_DEFAULT_ATTACKS


def _event(time: float, flags: int) -> dict:
    return {"kind": 1, "time": time, "f": {}, "i": {}, "flags": flags}


def test_the_writer_wrapper_hops_only_the_selector_events_of_the_listed_attacks() -> None:
    seen: list[dict] = []

    def original(arrays, batch, entity, authored):
        seen.append(authored)
        return "written"

    profile = SimpleNamespace(constructs=SimpleNamespace(replace_default_hop=True))
    wrapper = rdh.construct(original, profile)
    selector = EVENT_FLAG_SERVER_SELECTOR
    listed = {"asset": "Weapon_Sword_Primary_Swing_Left", "events": (_event(0.117, selector), _event(0.2, 0))}
    other = {"asset": "Weapon_Sword_Primary_Thrust", "events": (_event(0.05, selector),)}
    instant = {"asset": "Weapon_Mace_Primary_Swing_Left", "events": (_event(0.0, selector),)}
    authored = {"asset": "Weapon_Sword_Iron", "abilities": [listed, other, instant]}

    assert wrapper(None, 0, 0, authored) == "written"
    out = seen[0]
    assert out["asset"] == "Weapon_Sword_Iron" and out is not authored
    hopped, untouched, instant_out = out["abilities"]
    assert hopped["events"][0]["time"] == pytest.approx(0.117 + TICK)
    assert hopped["events"][1]["time"] == pytest.approx(0.2)  # no server-selector flag: the time stays
    assert hopped["events"][0]["flags"] == selector  # the flags are kept
    assert untouched is other  # not listed: the very same dict
    assert instant_out["events"][0]["time"] == 0.0  # the instant event of a listed attack keeps its time
    assert listed["events"][0]["time"] == 0.117  # the authored dict is not changed


def test_the_writer_wrapper_passes_through_when_the_switch_is_off() -> None:
    sentinel = object()
    assert rdh.construct(sentinel, SimpleNamespace(constructs=SimpleNamespace(replace_default_hop=False))) is sentinel


# ---- render tests ------------------------------------------------------------------------------------------------------


class _Result(NamedTuple):
    names: list[str]
    slots: list[int]
    trace: dict[str, np.ndarray]
    leaves: dict[str, np.ndarray]


class _Group:
    """One batch on the mounted render: each attack is one lane, requested on step 1 against a still iron-sword target."""

    def __init__(self, package: str, weapon: str, distance: float, attacks: tuple[str, ...]) -> None:
        self.combat = import_module(package + ".combat")
        self.arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        profile = import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")
        names = [ability["asset"] for ability in getattr(profile, "_" + weapon)()["abilities"]]
        self.names = names
        self.slots = [names.index(attack) for attack in attacks]
        self.params = self.combat.default_combat_params(microticks=1, target_active=True)
        self.batch = len(attacks)
        loadout = self.arsenal.hytale_0_5_7_loadouts([weapon] * self.batch, target_profiles=["iron_sword"] * self.batch)
        self.config = self.arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=self.params.sensor_range)
        position = np.zeros((self.batch, 2, 3), np.float32)
        position[:, TARGET, 2] = -np.float32(distance)
        health = jnp.tile(jnp.asarray([[self.params.agent_max_health, self.params.target_max_health]], jnp.float32),
                          (self.batch, 1))
        self.initial, _ = self.arsenal.reset_arsenal_batch(
            jax.random.split(jax.random.key(SEED), self.batch), self.params, self.config,
            entity_position=jnp.asarray(position), entity_health=health,
            entity_yaw=jnp.zeros((self.batch, 2), jnp.float32))
        self.world = environment.open_flat_arsenal_world_capabilities(self.initial, self.params, config=self.config)

    def run(self) -> _Result:
        slots = np.full((STEPS, self.batch), -1, np.int32)
        slots[0] = self.slots
        keys = jax.random.split(jax.random.key(161803), STEPS * self.batch).reshape(STEPS, self.batch)
        action = jnp.zeros((self.batch, self.combat.ACTION_SIZE), jnp.float32)

        def body(state, xs):
            slot, key = xs
            command = self.arsenal.empty_arsenal_commands(self.batch)
            command = command._replace(ability_slot=command.ability_slot.at[:, AGENT].set(slot), world=self.world)
            nxt = self.arsenal.step_arsenal_batch(state, action, key, self.params, command, self.config).state
            arsenal, combat = nxt.arsenal, nxt.combat
            return nxt, {
                "act": arsenal.active_ability_slot[:, AGENT], "tick": arsenal.ability_scheduler_tick[:, AGENT],
                "bits": arsenal.ability_selector_hit_bits[:, AGENT], "hp": combat.health[:, TARGET],
            }

        final, trace = jax.jit(lambda state, xs: jax.lax.scan(body, state, xs))(self.initial, (jnp.asarray(slots), keys))
        return _Result(self.names, self.slots, {name: np.asarray(value) for name, value in trace.items()}, _leaves(final))


def _leaves(tree: Any) -> dict[str, np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def _hit_and_finish(trace: dict[str, np.ndarray], lane: int, slot: int) -> tuple[int | None, int | None]:
    """The first step the target's health reads lower, and the first step after the start on which the slot reads -1."""

    hp = trace["hp"][:, lane]
    hit = next((step + 1 for step in range(1, STEPS) if hp[step] < hp[step - 1]), None)
    act = trace["act"][:, lane]
    started = next((step for step in range(STEPS) if act[step] == slot), None)
    finish = None
    if started is not None:
        finish = next((step + 1 for step in range(started + 1, STEPS) if act[step] == -1), None)
    return hit, finish


@contextmanager
def _mounted(path: Path) -> Iterator[str]:
    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def render_dirs(tmp_path_factory) -> dict[str, Path]:
    """The shipped gen-7 overlay (switch on, from its profile), the same overlay with the switch off, and the committed
    overlay without the row."""

    base = tmp_path_factory.mktemp("replace_default_hop_render")
    committed = replace(GEN7_OVERLAY, hooks=tuple(hook for hook in GEN7_OVERLAY.hooks if hook.id != "replace_default_hop"))
    name = "replace_default_hop_test_off_profile"
    module = types.ModuleType(name)
    module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, replace_default_hop=False))
    sys.modules[name] = module
    try:
        dirs = {}
        for label, overlay in (("on", GEN7_OVERLAY), ("off", replace(GEN7_OVERLAY, profile=f"{name}:PROFILE")),
                               ("committed", committed)):
            dirs[label] = render_for_test(overlay, base / label, source=HeadSource())
        yield dirs
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def runs(render_dirs) -> dict[str, list[_Result]]:
    """Per variant, per group (GROUPS order): the slots, the trace and the final state's leaves."""

    out: dict[str, list[_Result]] = {}
    for label in ("committed", "off", "on"):
        with _mounted(render_dirs[label]) as package:
            out[label] = [_Group(package, weapon, distance, attacks).run() for weapon, distance, attacks in GROUPS]
    return out


def test_the_switch_off_render_equals_the_committed_render_leaf_by_leaf(runs) -> None:
    for group, (weapon, _, _) in enumerate(GROUPS):
        off, committed = runs["off"][group], runs["committed"][group]
        for name in committed.trace:
            assert np.array_equal(off.trace[name], committed.trace[name]), (weapon, name)
        assert set(off.leaves) == set(committed.leaves)
        for key in committed.leaves:
            assert np.array_equal(off.leaves[key], committed.leaves[key]), key


def test_the_sword_left_swing_hits_on_step_9_and_finishes_on_14_with_the_switch_on(runs) -> None:
    on, off = runs["on"][0], runs["off"][0]
    assert _hit_and_finish(on.trace, 0, on.slots[0]) == (9, 14)
    assert float(on.trace["hp"][7, 0]) == 61.0 and float(on.trace["hp"][8, 0]) == 51.0  # 10 damage on step 9
    assert _hit_and_finish(off.trace, 0, off.slots[0]) == (8, 14)


def _executed(result: _Result, lane: int) -> list[int]:
    """The slots the lane's attack actually ran (the active slot read on any step); a request can run another slot."""

    return sorted({int(value) for value in result.trace["act"][:, lane] if value >= 0})


def _same_trace(on: _Result, off: _Result, lane: int, label: str) -> None:
    for name in on.trace:
        assert np.array_equal(on.trace[name][:, lane], off.trace[name][:, lane]), (label, name)


def test_each_lane_is_classified_by_the_attack_it_runs(runs) -> None:
    """A lane is classified by the attack it executes, not the one it requests. The thrust and the charged mace request
    run the listed Swing_Left and Swing_Left through the Primary root (a short press releases into slot 0); the signature
    attacks do not start on the scenario (cause not checked). A listed attack that runs hits one step later with the same finish,
    and the table gives the direct requests exactly."""

    for group, (_, _, attacks) in enumerate(GROUPS):
        on, off = runs["on"][group], runs["off"][group]
        for lane, requested in enumerate(attacks):
            executed = _executed(on, lane)
            assert executed == _executed(off, lane), requested  # the routing does not depend on the switch
            if not executed:
                assert requested not in rdh.REPLACE_DEFAULT_ATTACKS, requested
                _same_trace(on, off, lane, requested)
                continue
            assert len(executed) == 1, (requested, executed)
            slot = executed[0]
            ran = on.names[slot]
            if ran not in rdh.REPLACE_DEFAULT_ATTACKS:
                _same_trace(on, off, lane, requested)
                continue
            hit_off, finish_off = _hit_and_finish(off.trace, lane, slot)
            hit_on, finish_on = _hit_and_finish(on.trace, lane, slot)
            assert finish_on == finish_off, requested
            if requested == ran:
                assert (hit_off, hit_on, finish_on) == EXPECTED[ran], requested
            else:
                assert hit_on == hit_off + 1, requested  # a routed request lands the listed attack one step later
                assert requested in ROUTED and ROUTED[requested] == ran, (requested, ran)
