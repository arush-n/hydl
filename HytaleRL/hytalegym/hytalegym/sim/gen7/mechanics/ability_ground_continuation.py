"""Ability ground continuation (gen 7, row 5h): a parked ability's ground child starts once the parent has finished.

Switch ``Constructs.ability_ground_continuation``: ``False`` (gen 5: a resumed continuation writes its child slot and nothing
starts the child) or ``True`` (gen 7).

Native (0.7.2 asset): the charged battleaxe root (slot 0, ``interaction_type`` Primary) fires its threshold child, the
Downstrike (slot 4): an ApplyForce leap with WaitForGround and no GroundCheckDelay, ``ground_next`` 5. On landing the
continuation runs the Landing (slot 5): a forward 120 degree Horizontal selector (StartDistance 0.1, EndDistance 4, RunTime
0.072 s on gen 7), 29 Charged physical damage and knockback Force 7 (``profiles/hytale_0_5_7.py`` ``_iron_battleaxe``, indices 4
and 5). The same mechanism gates the mace Groundslam and the dagger Pounce (``continuation.py`` docstring).

HEAD (gen 5 engine, unchanged): ``continuation_resolution`` (``programs/continuation.py`` 150) reports ``ground_ready`` from the
previous tick's grounded flag with no finished term, so a resume can be written while the Downstrike is still starting
(``ability.py`` 1095 to 1115 resolve, 1194 writes ``ability_continuation_slot``; that is 5 on such a tick). The resume that matters
is the tick the parent finishes: ``ability.py`` 1179 sets ``active_ability_slot`` to -1 on it. Nothing reads ``ability_continuation_slot``, so
the Landing never starts (CONFIRMED on the render, scratch ``lane_119f1/cont_trace.py``; both lines).

Construct: ``tick_ability_programs`` is wrapped. A child starts on a tick where the parent was running before the tick, is
finished after it (``active_ability_slot`` -1) and the continuation resolved to a child (``ability_continuation_slot`` >= 0).
That gate is the one the engine lacks: a premature resume (parent still active) starts nothing. The child then runs from the next
tick, as the native chain does. The start sets the fields ``start_abilities`` sets for an accepted root, for the child:

* ``active_ability_slot`` = the child; ``active_ability_root_slot`` = the parent's root from before the tick (the engine keeps the
  requesting root while a private child executes, ``schema/spec.py`` ``interaction_scheduler``);
* ``ability_elapsed_seconds`` = 0, ``ability_scheduler_tick`` = minus the child's prelude ticks, ``ability_scheduler_clock_seconds``
  = 0 and ``ability_selector_hit_bits`` = 0, as ``start_abilities`` sets them;
* ``ability_continuation_slot`` = -1: the resume is consumed.

``start_abilities`` itself is not called: it is the admission of a requested root, and it spends the charge, sets the cooldown and
the stamina regen delay of the requested slot. A child is not admitted, so those stay as they are.

Gen 5 returns the original function. Measured and tests: ``tests/sim/test_ability_ground_continuation.py``; mechanics_072.md row 5h.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "ability_ground_continuation"
IMPLEMENTED = True


def _enabled(profile: Any) -> bool:
    return bool(profile.constructs.ability_ground_continuation)


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for gen 7 the wrapper that starts the resumed child on the parent's finishing tick."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _tick_ability_programs(state: Any, loadout: Any, *args: Any, **kwargs: Any) -> Any:
        candidate, fired = original(state, loadout, *args, **kwargs)
        finished = (state.active_ability_slot >= 0) & (candidate.active_ability_slot < 0)
        start = finished & (candidate.ability_continuation_slot >= 0)
        child = candidate.ability_continuation_slot
        capacity = loadout.ability_scheduler_prelude_ticks.shape[2]
        index = jnp.clip(child, 0, capacity - 1)
        prelude = jnp.take_along_axis(loadout.ability_scheduler_prelude_ticks, index[..., None], axis=2)[..., 0]
        candidate = candidate._replace(
            active_ability_slot=jnp.where(start, child, candidate.active_ability_slot),
            active_ability_root_slot=jnp.where(start, state.active_ability_root_slot, candidate.active_ability_root_slot),
            ability_elapsed_seconds=jnp.where(start, jnp.float32(0.0), candidate.ability_elapsed_seconds),
            ability_scheduler_tick=jnp.where(start, -prelude, candidate.ability_scheduler_tick),
            ability_scheduler_clock_seconds=jnp.where(
                start[..., None], jnp.float32(0.0), candidate.ability_scheduler_clock_seconds
            ),
            ability_selector_hit_bits=jnp.where(start, jnp.uint32(0), candidate.ability_selector_hit_bits),
            ability_continuation_slot=jnp.where(start, jnp.int32(-1), candidate.ability_continuation_slot),
        )
        return candidate, fired

    return _tick_ability_programs
