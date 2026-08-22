"""Shared machinery for minigames. Change a common factor in one place.

Every minigame is a shaping function `(next_state, info) -> (batch,)` added on
top of the native reward by `tasks.shaped()`. Writing four of them surfaced
exactly three things they all vary, and those are the knobs here:

  **what is measured**  a scalar per batch row read off state -- distance to a
      goal, separation from the target, displacement since last tick.
  **how it is paid**    `progress` pays for the *change* in that scalar;
      `band` pays for it being inside a range; `threshold` pays past a limit.
  **geometry**          horizontal (x,z) or full 3D.

Why `progress` is the default and matters: a reward on the *level* of a scalar
pays an agent that spawned in a good spot and never moved. Paying the delta
means a stationary agent earns exactly zero, so the signal is about behaviour
rather than about the reset.

Why horizontal is the default: vertical separation on this terrain is mostly
spawn height (measured y=126 / y=26 / y=46 across node seeds), so including y
makes a cliff read as distance the agent cannot close.

**Never remember the previous tick here.** A term that pays for a change takes
BOTH states -- `(previous_state, next_state, info)`, marked with
`shaping.wants_previous` -- rather than storing the last one in a closure. A
Python-dict memory is correct when called eagerly and silently wrong under
`jax.jit`: the store runs once at trace time, so repeated calls read a dead
trace's value, and a `lax.scan` body -- traced once, run many times -- reads the
same stale value on every iteration. Measured on `displacement`: 3.0 then 4.0
eagerly, `[0.0, 0.0, 0.0, 0.0]` inside a scan, with no error raised. Training
compiles the step, so that is the only context that counts.

Episode boundaries are handled by :func:`terminated`, read off the state
itself. The arsenal environment does not auto-reset cleanly -- health pins at 0
and `done` repeats -- so a row that was already dead pays zero rather than a
reset-sized jump, and that test is pure so it survives tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

from adk.contracts.shaping import wants_previous

__all__ = [
    "Minigame", "distance", "displacement", "separation", "goal_distance",
    "progress", "band", "threshold", "health", "resource", "accepted",
    "penalise", "terminated", "AGENT_ENTITY", "TARGET_ENTITY",
]


@dataclass(frozen=True)
class Minigame:
    """A named shaping objective.

    `teaches` is the behaviour it should produce; `tell` is the metric that
    would show it worked. A minigame whose `tell` cannot move on the scene you
    ran it on is a broken setup, not a failed agent.
    """

    name: str
    teaches: str
    tell: str
    build: Callable[..., Callable]
    #: Scenes where this is meaningless -- checked by the runner, not by hope.
    requires: tuple[str, ...] = ()


# --- what is measured -------------------------------------------------------

def distance(a, b, *, horizontal: bool = True):
    """Euclidean distance, x/z only by default.

    No epsilon inside the sqrt. `sqrt(0 + 1e-8)` is **1e-4**, not ~0, so a
    stationary agent collected 1e-4 of `sprint` reward every tick -- free
    payment for doing nothing, which is the exact failure `progress` exists to
    avoid. Guard the zero case instead; nothing differentiates through this, so
    the sqrt-at-zero gradient that an epsilon usually protects is not a concern.
    """

    delta = a - b
    squared = delta[..., 0] ** 2 + delta[..., 2] ** 2
    if not horizontal:
        squared = squared + delta[..., 1] ** 2
    return jnp.where(squared > 0.0, jnp.sqrt(jnp.maximum(squared, 1e-30)), 0.0)


def separation(*, horizontal: bool = True):
    """Agent-to-target distance."""

    def read(state):
        position = state.runtime.combat.position
        return distance(position[:, AGENT_ENTITY], position[:, TARGET_ENTITY],
                        horizontal=horizontal)

    return read


def goal_distance(goal, *, horizontal: bool = True):
    """Agent-to-fixed-point distance. `goal` is (3,) or (batch, 3)."""

    goal = jnp.asarray(goal, dtype=jnp.float32)

    def read(state):
        return distance(state.runtime.combat.position[:, AGENT_ENTITY], goal,
                        horizontal=horizontal)

    return read


def terminated(state):
    """Per-row episode-boundary flag, computed from the state. Pure.

    The jit-safe replacement for a remembered ``done``. Mirrors the
    environment's own predicate (``hytalegym/jax/combat/env.py:_terminated``)
    rather than importing it, so this keeps working for any state exposing the
    same rows instead of coupling shaping to a private Gym symbol. The two
    optional flags are read with ``getattr`` for the same reason: a synthetic
    or reduced state that lacks them is still answerable on health alone.
    """

    combat = state.runtime.combat
    dead = ((combat.health[:, AGENT_ENTITY] <= 0.0)
            | (combat.health[:, TARGET_ENTITY] <= 0.0))
    for name in ("geometry_exhausted", "target_navigation_unsupported"):
        flag = getattr(combat, name, None)
        if flag is not None:
            dead = dead | jnp.asarray(flag, dtype=bool)
    return dead


def displacement(*, entity: int = AGENT_ENTITY, horizontal: bool = True):
    """Per-tick movement, as a delta between two states. Stateless.

    Guarded at episode boundaries like :func:`progress`, and for the same
    reason. On a reset the entity is moved back to spawn, and the distance from
    its death position to its spawn position is not movement it earned -- it is
    a teleport. Unguarded, `sprint` paid that jump as its single largest reward
    of the episode, on the tick the agent had least to do with it.
    """

    @wants_previous
    def shape(previous_state, state, info):
        del info
        position = state.runtime.combat.position[:, entity]
        last = previous_state.runtime.combat.position[:, entity]
        moved = distance(position, last, horizontal=horizontal)
        return jnp.where(terminated(previous_state), jnp.float32(0.0), moved)

    return shape


# --- how it is paid ---------------------------------------------------------

def progress(read, *, sign: float = -1.0, bonus: float = 0.0,
             within: float | None = None):
    """Pay for the CHANGE in `read`, not its level.

    `sign=-1` pays for the scalar decreasing (closing a distance); `sign=+1`
    pays for it increasing (fleeing). `bonus` is added while `read < within`.

    Stateless: the change is measured between the two states it is handed, not
    against a remembered one. `read` is applied to both, so it must be a pure
    ``(state) -> (batch,)`` reader -- which every reader above already is.
    """

    @wants_previous
    def shape(previous_state, state, info):
        del info
        value = jnp.asarray(read(state), dtype=jnp.float32)
        last = jnp.asarray(read(previous_state), dtype=jnp.float32)
        moved = jnp.float32(sign) * (value - last)
        # A row that was already terminated has a previous state belonging to
        # the previous episode; pay it zero rather than the reset-sized jump.
        moved = jnp.where(terminated(previous_state), jnp.float32(0.0), moved)
        if within is not None and bonus:
            moved = moved + jnp.where(
                value < jnp.float32(within), jnp.float32(bonus), jnp.float32(0.0))
        return moved

    return shape


def band(read, low: float, high: float, *, reward: float = 1.0):
    """Pay a flat `reward` while `read` is inside [low, high]. Stateless."""

    def shape(state, info):
        del info
        value = jnp.asarray(read(state), dtype=jnp.float32)
        return jnp.where((value >= low) & (value <= high),
                         jnp.float32(reward), jnp.float32(0.0))

    return shape


def health(*, entity: int = TARGET_ENTITY):
    """Entity health. Pair with `progress(sign=-1)` to pay for damage dealt."""

    return lambda state: state.runtime.combat.health[:, entity]


def resource(index: int, *, entity: int = AGENT_ENTITY):
    """One of the 7 resources: stamina 0, mana 1, magic_charges 2,
    signature_energy 3, signature_charges 4, ammo 5, oxygen 6."""

    return lambda state: state.runtime.mechanics.resources[:, entity, index]


def damage(*, dealt: bool = True):
    """Per-tick damage, straight off `info.arsenal_info`. Takes `(state, info)`.

    **No entity axis.** `damage_dealt` and `damage_received` are already scoped
    to the agent and are `(batch,)`, unlike `ability_accepted` which is
    `(batch, entities)`. Indexing an entity here would read off the end or
    silently return the wrong column -- the per-entity split lives on the
    separate `entity_damage_dealt` / `entity_damage_received` rows.

    This is the direct counter. `accuracy` and `block` infer damage from the
    drop in health instead, which cannot tell a hit that was blocked from one
    that missed; prefer this when you want the event rather than the effect.
    """

    field = "damage_dealt" if dealt else "damage_received"

    def read(state, info):
        del state
        return jnp.asarray(getattr(info.arsenal_info, field), dtype=jnp.float32)

    return read


def landed(*, threshold_value: float = 0.0):
    """1.0 on ticks where the agent actually dealt damage, else 0.0.

    A *successful* attack, as distinct from an accepted one. Acceptance is not
    damage: measured, two opponent policies got abilities accepted 36 and 16
    times and dealt 0.0, so anything gating on `accepted` counts intent rather
    than effect.
    """

    read_damage = damage(dealt=True)

    def read(state, info):
        return (read_damage(state, info) > threshold_value).astype(jnp.float32)

    return read


def defended(*, blocked: bool = True):
    """Per-tick count of hits absorbed by guard, or evaded by i-frames.

    Measured `(batch,)` int32 with **no entity axis**, like `damage_dealt`.

    These are the *exact* counters and nothing used them. `block` and `dodge`
    both infer success from a drop in health that did not happen, which cannot
    separate "the guard absorbed it" from "the attack missed anyway" -- and on
    a scene where the opponent whiffs a lot those look identical while meaning
    opposite things. Pay these instead when you want the mechanic to have
    fired, not merely for damage to be absent.
    """

    field = "blocked_hits" if blocked else "invulnerable_hits"

    def read(state, info):
        del state
        return jnp.asarray(getattr(info.arsenal_info, field), dtype=jnp.float32)

    return read


def grounded(*, want: bool = True):
    """1.0 while the agent is on the ground (or airborne, with `want=False`).

    Worth paying for directly: airborne entities get **no horizontal
    translation**, so a policy that learns to hold jump stops moving entirely
    and reads as broken steering rather than as a movement choice.
    """

    def read(state):
        on_ground = jnp.asarray(state.runtime.combat.agent_grounded)
        hit = on_ground if want else ~on_ground.astype(bool)
        return hit.astype(jnp.float32)

    return read


def facing(*, entity: int = AGENT_ENTITY, target: int = TARGET_ENTITY):
    """Cosine alignment in [-1, 1] between the agent's yaw and the target bearing.

    1.0 is looking straight at it, -1.0 directly away. Horizontal by
    construction -- yaw carries no pitch, so adding y here would compare two
    different things.
    """

    def read(state):
        position = state.runtime.combat.position
        delta = position[:, target] - position[:, entity]
        bearing = jnp.arctan2(delta[..., 0], delta[..., 2])
        return jnp.cos(bearing - state.runtime.combat.yaw[:, entity])

    return read


def waypoints(route, *, entity: int = AGENT_ENTITY, horizontal: bool = True):
    """Distance still to travel along a route. A stateless potential.

    `route` is `(n, 3)` -- a sequence of world points. The value is::

        min over checkpoints i of [ distance(agent, route[i]) + length(i..end) ]

    i.e. reach the route at whichever checkpoint is cheapest, then follow it to
    the end. Pair with ``progress(sign=-1)``: advancing along the route lowers
    the potential, and every step back costs exactly what the matching step
    forward paid, so no cycle earns anything.

    **This replaced a per-row checkpoint index, and the ordering guarantee went
    with it.** The old version advanced an index once a row came within a
    tolerance, which enforced visiting checkpoints in order -- and kept that
    index in a Python dict, so under `jax.jit` it either froze at a dead
    trace's value or leaked the tracer outright (measured:
    ``UnexpectedTracerError`` from inside a `lax.scan`). Training compiles the
    step, so the ordered version did not work where it was meant to be used. A
    potential admits a shortcut straight to the final checkpoint; if a route
    must be walked in order, that needs state threaded through the scan carry,
    which the shaping contract does not currently carry.

    **Checkpoints must be real traversal nodes.** An invented coordinate can
    sit inside terrain, and the agent is then paid for walking into a wall.
    """

    route = jnp.asarray(route, dtype=jnp.float32)
    if route.ndim != 2 or route.shape[-1] != 3:
        raise ValueError(f"route must be (n, 3), got {tuple(route.shape)}")

    # suffix[i] = route length from checkpoint i to the end; suffix[-1] = 0.
    legs = distance(route[:-1], route[1:], horizontal=horizontal)
    suffix = jnp.concatenate([
        jnp.cumsum(legs[::-1])[::-1] if legs.size else legs,
        jnp.zeros(1, dtype=jnp.float32),
    ])

    def read(state):
        position = state.runtime.combat.position[:, entity]
        gaps = distance(position[:, None, :], route[None, :, :],
                        horizontal=horizontal)
        return jnp.min(gaps + suffix[None, :], axis=-1)

    read.route_length = int(route.shape[0])     # type: ignore[attr-defined]
    read.total_distance = float(jnp.sum(legs))  # type: ignore[attr-defined]
    return read


def event(read_event, *, reward: float = 1.0):
    """Pay `reward` per event. The positive counterpart of :func:`penalise`.

    Takes an `(state, info)` reader, like `penalise`, because the things worth
    paying per-occurrence -- a landed hit, a blocked hit, an accepted ability --
    are all reported on `info` rather than read off state.
    """

    def shape(state, info):
        return jnp.float32(reward) * jnp.asarray(
            read_event(state, info), dtype=jnp.float32)

    return shape


def accepted(*, entity: int = AGENT_ENTITY):
    """Per-tick accepted-ability count, read from `info.arsenal_info`.

    Requested is NOT the same as accepted -- a requested slot can be refused on
    cooldown or resources, and counting requests overstates activity. Anything
    measuring cost-per-action must divide by this, not by requests.
    """

    def read(state, info):
        del state
        return info.arsenal_info.ability_accepted[:, entity].astype(jnp.float32)

    return read


def penalise(read_event, *, cost: float = 1.0):
    """Charge `cost` per event. Takes an `(state, info)` reader, not `(state)`.

    The point of a per-action charge is that "kill it" and "kill it in few
    moves" are the same objective until wasted actions cost something.
    """

    def shape(state, info):
        return -jnp.float32(cost) * jnp.asarray(
            read_event(state, info), dtype=jnp.float32)

    return shape


def threshold(read, limit: float, *, reward: float = 1.0, above: bool = True):
    """Pay a flat `reward` while `read` is past `limit`. Stateless."""

    def shape(state, info):
        del info
        value = jnp.asarray(read(state), dtype=jnp.float32)
        hit = value > limit if above else value < limit
        return jnp.where(hit, jnp.float32(reward), jnp.float32(0.0))

    return shape
