"""Gen-7 entry-timed footsteps (opt-in). Switch ``Constructs.hearing_entry``, default False; it acts only with ``hearing`` on.

The rule is the native 0.7.2 ``TickMovementAudio`` (FINDINGS in the lane folder ``lane_movein_rule``, sections 2 and 5): on a
change of the feet cell, MoveIn of the new cell's set (when it authors MoveIn) and the repeat timer is armed to a draw in
[0.5, 1.5] s; while the body moves, the timer counts down by dt and at zero MoveIn of the current cell plays again and re-arms.
MoveOut is not modelled: only Water authors it and the heard content tables carry MoveIn only.

State: one optional CombatState field, ``footstep`` (object, default None, appended last). Off, it is None, which is an empty
pytree node, so the switch-off state has no extra leaves. On, the reset hook sets it to ``FootstepState`` (below); from reset
onward the scan carry has it, so the jitted tree is fixed. Its leaves are [batch, entity]:
- ``cell`` (int32): the feet-cell block id of the previous step (0 at reset: native's initial lastInside is air).
- ``timer`` (float32): seconds until the repeat; -1 when disarmed (NO_REPEAT).
- ``emit`` (int32): the MoveIn semantic id this step played, 0 for none (read by heard_sounds.py this step only).

Draw (no key leaf): ``uniform(0.5, 1.5)`` from a 32-bit integer hash of (row, entity, tick), taken at the arming tick and stored
in ``timer``. Deterministic for a given run, so a replay is exact.

Hooks (``sim/gen7/overlay.py`` rows ``movein_reset`` and ``movein_step``), both installed only when the switch is on:
- ``construct_reset`` (on ``reset_arsenal_batch``): sets ``combat.footstep`` to the initial state.
- ``construct_step`` (on ``step_arsenal_batch``, which every engine path calls: the training environment, the multi-actor
  duel): after the step, updates the footstep state with the feet cells the step's capabilities report (``commands.world``).

Timing choice (INFERRED, not measured): the cell is the one the tick's capabilities see, the start-of-tick cell, so an entry is
heard on the tick after the body steps onto the block. Native's audio/movement order is open (FINDINGS "Not confirmed").
"""

from __future__ import annotations

from functools import wraps
from typing import Any, NamedTuple

import jax.numpy as jnp

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.sim.gen7.mechanics.heard_sounds import (
    FLOOR_BLOCK,
    MOVING_SPEED,
    _lookup,
    footstep_tables_from_env,
)

SWITCH = "hearing_entry"
IMPLEMENTED = True
#: The native NO_REPEAT sentinel (MovementAudioComponent, -1.0f).
NO_REPEAT = -1.0


class FootstepState(NamedTuple):
    """The optional CombatState field: per-entity [batch, entity] leaves (see the module docstring)."""

    cell: Any  # int32, the feet-cell block id of the previous step
    timer: Any  # float32, seconds to the repeat; NO_REPEAT when disarmed
    emit: Any  # int32, the MoveIn id played this step; 0 for none


def initial_footstep(batch: int, entities: int) -> FootstepState:
    """The footstep state at reset: air, disarmed, nothing emitted."""

    return FootstepState(
        cell=jnp.zeros((batch, entities), dtype=jnp.int32),
        timer=jnp.full((batch, entities), NO_REPEAT, dtype=jnp.float32),
        emit=jnp.zeros((batch, entities), dtype=jnp.int32),
    )
#: The native default MoveIn repeat range (BlockSoundSet.DEFAULT_MOVE_IN_REPEAT_RANGE), seconds.
REPEAT_MIN = 0.5
REPEAT_MAX = 1.5


def _active(profile: Any) -> bool:
    entry = profile.constructs.hearing_entry
    if entry is False:
        return False
    if entry is not True:
        raise ValueError(f"hearing_entry must be a bool, got {entry!r}")
    return bool(profile.constructs.hearing)


def draw_seconds(batch: int, entities: int, tick: jnp.ndarray) -> jnp.ndarray:
    """[batch, entities] float32 uniform in [REPEAT_MIN, REPEAT_MAX): a 32-bit integer hash of (row, entity, tick)."""

    row = jnp.arange(batch, dtype=jnp.uint32)[:, None]
    entity = jnp.arange(entities, dtype=jnp.uint32)[None, :]
    time = jnp.broadcast_to(tick.astype(jnp.uint32)[:, None], (batch, entities))
    x = (row * jnp.uint32(0x9E3779B1)) ^ (entity * jnp.uint32(0x85EBCA77)) ^ (time * jnp.uint32(0xC2B2AE3D))
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    unit = (x >> 8).astype(jnp.float32) * jnp.float32(1.0 / 16777216.0)
    return jnp.float32(REPEAT_MIN) + jnp.float32(REPEAT_MAX - REPEAT_MIN) * unit


def footstep_update(state: Any, world: Any, params: Any) -> Any:
    """The state after one step: the footstep leaves updated from the start-of-tick feet cells in ``world``."""

    footstep = state.combat.footstep
    batch, entities = footstep.cell.shape
    block = getattr(world, "block_under_feet", None)
    if block is None:
        block = jnp.full((batch, entities), semantic_id(FLOOR_BLOCK), dtype=jnp.int32)
    cur = block.astype(jnp.int32)

    tables = footstep_tables_from_env()
    block_in, known, _ = _lookup(tables.block_keys, cur, tables.move_in)
    authors = known & (block_in != 0)  # the new cell's set authors MoveIn (air and unknown ids do not)
    changed = cur != footstep.cell

    combat = state.combat
    velocity = combat.velocity.astype(jnp.float32)
    moving = jnp.sqrt(velocity[..., 0] ** 2 + velocity[..., 2] ** 2) > MOVING_SPEED
    dt = jnp.asarray(params.nominal_dt, dtype=jnp.float32)
    draw = draw_seconds(batch, entities, combat.tick_count)

    # Native order: the change arms (or disarms) first, then the repeat counts down on the same tick.
    timer = jnp.where(changed, jnp.where(authors, draw, NO_REPEAT), footstep.timer)
    armed = timer != NO_REPEAT
    timer = jnp.where(moving & armed, timer - dt, timer)
    fire = moving & armed & (timer <= 0.0)
    timer = jnp.where(fire, draw, timer)
    emit = jnp.where((changed | fire) & authors, block_in, 0).astype(jnp.int32)

    footstep = FootstepState(cell=cur, timer=timer.astype(jnp.float32), emit=emit)
    return state._replace(combat=combat._replace(footstep=footstep))


def construct_reset(original: Any, profile: Any) -> Any:
    """Hook for ``reset_arsenal_batch``: the footstep state starts at air, disarmed (the field is None without the switch)."""

    if not _active(profile):
        return original

    @wraps(original)
    def reset_arsenal_batch(keys: Any, params: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
        state, observation = original(keys, params, config, *args, **kwargs)
        batch, entities = state.combat.health.shape
        combat = state.combat._replace(footstep=initial_footstep(batch, entities))
        return state._replace(combat=combat), observation

    return reset_arsenal_batch


def construct_step(original: Any, profile: Any) -> Any:
    """Hook for ``step_arsenal_batch``: after the step, the footstep state follows the native rule."""

    if not _active(profile):
        return original

    @wraps(original)
    def step_arsenal_batch(state: Any, actions: Any, keys: Any, params: Any, commands: Any, config: Any, *args: Any,
                           **kwargs: Any) -> Any:
        transition = original(state, actions, keys, params, commands, config, *args, **kwargs)
        if transition.state.combat.footstep is None:  # a state not made by the reset hook: nothing to carry
            return transition
        return transition._replace(state=footstep_update(transition.state, commands.world, params))

    return step_arsenal_batch
