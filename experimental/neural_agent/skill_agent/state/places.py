"""The long-term spatial memory (``docs/VIRTUAL_HUMAN.md`` section 22; 16.4): the places the agent has been, what
happened at them, and the routes between them. It sits beside the short-term stores of ``state/memory.py``, which keep
what was seen recently and restart at an episode boundary. This store keeps what matters, and keeps it past a death
(16.2).

A **place** is a point in the world with a strength in [0, 1] and one or more kinds (22.2): ``stay`` (the agent stayed
there), ``danger`` (it was hurt there), ``death`` (it died there), ``find`` (it gained an item there) and ``marked``
(the decision layer marked it: a live ``remember`` entry of ``state/written.py``). A **route** is a travel from one
place to another, with its best and last travel time in ticks, how often it was travelled and when last.

Each tick :func:`update` runs these steps for each agent, in this order (the module's rules, 22.2):

1. **fading:** an unmarked place loses strength as ``s <- s exp(-DT / TAU_PLACE)``, a route as ``s <- s exp(-DT /
   TAU_ROUTE)``. A mark never fades (19.7). Routes fade more slowly than places: landmark knowledge decays while route
   knowledge holds better (22.6).
2. **stays:** the anchor is the agent's position when it has none or has moved more than ``STAY_BLOCKS`` from it, and
   the clock starts there. A stay is made once the agent has stayed within ``STAY_BLOCKS`` of its anchor for
   ``STAY_TICKS``: stay-point detection (22.6), scaled to a game.
3. **revisits:** the place the agent is within (the nearest within ``MERGE_BLOCKS``) that existed before this tick, and
   that the agent was not inside last tick, restores its strength to at least ``REVISIT_STRENGTH`` and counts a visit.
   A place this tick forms is a first visit, not a revisit.
4. **candidates,** in a fixed order (the stay, the danger, the death, the find, then each live mark), each consolidated
   into the store: a candidate within ``MERGE_BLOCKS`` of a place refreshes the nearest one (its kind is added, its
   strength the larger, a mark makes it marked); else it takes the first free slot; else the weakest unmarked place
   gives way, with every route touching it; when every place is marked the candidate is refused and counted in
   ``evictions``. The strengths of the candidates:

       stay      STAY_STRENGTH                    [E]
       danger    0.5 + 0.5 alarm                  alarm is the affect rules' fear: an arousing event is consolidated
                                                  more strongly (22.6, McGaugh 2004)
       death     1.0                              the last position before a respawn (16.2)
       find      FIND_STRENGTH                    [E]
       marked    1.0

5. **routes:** leaving a place (the agent was inside it and now is not) records when it left. Arriving at a place from
   the place last left records the route between them, with the travel time since; a route already known updates its
   best time, its last time and its count, and a new one takes a free slot, else the weakest route's.

Positions are read only with a perception channel (22.1): without one (``has_position`` False) nothing is stayed,
revisited, routed or anchored. The capacities (``PLACE_SLOTS``, ``ROUTE_SLOTS``) and the strength and radius constants
are estimates or engineering choices, changeable without retraining (section 14). Routes are invalidated when a place
gives way, and an agent whose place gave way is no longer inside it.

The **reads** are :func:`relative`: each place's displacement in the body frame, its distance and whether it is held,
for the ``where`` block of the views (the same convention as ``state/frames.py``). The body reads the few relevant
places as tokens (section 11.6) in step 7, and the decision layer plans over the routes (step 8).

**Grounding** (22.6): a map of places and the routes between them, as rats learn a maze (Tolman 1948) and as robotics'
topological maps do (Kuipers 2000); significant places from a trace by stay-point detection (Li et al. 2008; the
thresholds scaled here, [E]); emotionally arousing events consolidated more strongly (McGaugh 2004); a death place as a
game's recovery compass (Minecraft); fading, and revisits that restore, as spatial knowledge does in virtual and real
environments (the differentiated forgetting rates of HAL hal-01440576; Bournemouth's landmark and route study).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.state.frames import world_to_body

PLACE_SLOTS: int = 128
"""The places per agent: a capacity (engineering, 22.2)."""
ROUTE_SLOTS: int = 256
"""The routes per agent: a capacity (engineering, 22.2)."""
PLACE_KINDS: tuple[str, ...] = ("stay", "danger", "death", "find", "marked")
"""A place's kinds (22.2); a place may hold several."""
MERGE_BLOCKS: float = 4.0
"""Two positions within this many blocks are the same place [E] (22.2)."""
STAY_BLOCKS: float = 3.0
"""A stay keeps the agent within this many blocks of its anchor [E] (22.6)."""
STAY_TICKS: int = 222
"""A stay lasts this many ticks: 10 s at 0.045 s a tick [E]."""
STAY_STRENGTH: float = 0.5
"""The strength of a stay's place [E]."""
FIND_STRENGTH: float = 0.6
"""The strength of a find's place [E]."""
REVISIT_STRENGTH: float = 0.8
"""A revisit restores a place to at least this strength [E] (22.2)."""
TAU_PLACE_SECONDS: float = 1800.0
"""The fading time constant of an unmarked place, in seconds [E]."""
TAU_ROUTE_SECONDS: float = 3600.0
"""The fading time constant of a route, in seconds [E]: routes fade more slowly than places (22.6)."""
DT_SECONDS: float = 0.045
"""One tick, in seconds."""
_KIND = {name: index for index, name in enumerate(PLACE_KINDS)}
_FADE_PLACE = math.exp(-DT_SECONDS / TAU_PLACE_SECONDS)
_FADE_ROUTE = math.exp(-DT_SECONDS / TAU_ROUTE_SECONDS)
_CANDIDATE_FIXED = 4
"""The candidates before the marks: the stay, the danger, the death and the find."""


class Places(NamedTuple):
    """The places of B agents, P = ``PLACE_SLOTS`` slots each. An empty slot is not valid."""

    world: jax.Array  # float32[B, P, 3]: the place's world position, in blocks
    kinds: jax.Array  # bool[B, P, K]: the kinds it holds, in ``PLACE_KINDS`` order (K = len(PLACE_KINDS))
    strength: jax.Array  # float32[B, P]: in [0, 1]
    marked: jax.Array  # bool[B, P]: the decision layer marked it; it never fades and is never evicted
    first: jax.Array  # int32[B, P]: the tick it was made; -1 for an empty slot
    last: jax.Array  # int32[B, P]: the tick it was last written or visited; -1 for an empty slot
    visits: jax.Array  # int32[B, P]: its revisits
    valid: jax.Array  # bool[B, P]


class Routes(NamedTuple):
    """The routes of B agents, R = ``ROUTE_SLOTS`` slots each, between place slots."""

    source: jax.Array  # int32[B, R]: the place left; -1 for an empty slot
    target: jax.Array  # int32[B, R]: the place arrived at
    best_ticks: jax.Array  # int32[B, R]: the shortest travel time, in ticks
    last_ticks: jax.Array  # int32[B, R]: the last travel time, in ticks
    count: jax.Array  # int32[B, R]: how often it was travelled
    last: jax.Array  # int32[B, R]: the tick it was last travelled; -1 for an empty slot
    strength: jax.Array  # float32[B, R]
    valid: jax.Array  # bool[B, R]


class PlaceState(NamedTuple):
    """The long-term spatial memory of B agents, with the working state of its detectors."""

    now: jax.Array  # int32[B]: ticks since each agent's episode began
    places: Places
    routes: Routes
    inside: jax.Array  # int32[B]: the place the agent is within, -1 for none
    departed: jax.Array  # int32[B]: the place it last left, -1 for none
    departed_at: jax.Array  # int32[B]: the tick it left that place
    stay_anchor: jax.Array  # float32[B, 3]: where the current stay began
    stay_since: jax.Array  # int32[B]: the tick the agent came to the anchor; -1 before any
    stay_made: jax.Array  # bool[B]: the stay at this anchor has been made a place
    last_position: jax.Array  # float32[B, 3]: the last position perceived
    had_position: jax.Array  # bool[B]: a position was perceived last tick
    evictions: jax.Array  # int32[B]: candidates refused because every place was marked


def initial_places(batch: int) -> PlaceState:
    """The store at an episode's start: no place and no route, no stay, no position, the clock at 0."""

    places = Places(
        world=jnp.zeros((batch, PLACE_SLOTS, 3), jnp.float32),
        kinds=jnp.zeros((batch, PLACE_SLOTS, len(PLACE_KINDS)), jnp.bool_),
        strength=jnp.zeros((batch, PLACE_SLOTS), jnp.float32),
        marked=jnp.zeros((batch, PLACE_SLOTS), jnp.bool_),
        first=jnp.full((batch, PLACE_SLOTS), -1, jnp.int32),
        last=jnp.full((batch, PLACE_SLOTS), -1, jnp.int32),
        visits=jnp.zeros((batch, PLACE_SLOTS), jnp.int32),
        valid=jnp.zeros((batch, PLACE_SLOTS), jnp.bool_))
    routes = Routes(
        source=jnp.full((batch, ROUTE_SLOTS), -1, jnp.int32),
        target=jnp.full((batch, ROUTE_SLOTS), -1, jnp.int32),
        best_ticks=jnp.zeros((batch, ROUTE_SLOTS), jnp.int32),
        last_ticks=jnp.zeros((batch, ROUTE_SLOTS), jnp.int32),
        count=jnp.zeros((batch, ROUTE_SLOTS), jnp.int32),
        last=jnp.full((batch, ROUTE_SLOTS), -1, jnp.int32),
        strength=jnp.zeros((batch, ROUTE_SLOTS), jnp.float32),
        valid=jnp.zeros((batch, ROUTE_SLOTS), jnp.bool_))
    return PlaceState(
        now=jnp.zeros((batch,), jnp.int32), places=places, routes=routes,
        inside=jnp.full((batch,), -1, jnp.int32), departed=jnp.full((batch,), -1, jnp.int32),
        departed_at=jnp.zeros((batch,), jnp.int32), stay_anchor=jnp.zeros((batch, 3), jnp.float32),
        stay_since=jnp.full((batch,), -1, jnp.int32), stay_made=jnp.zeros((batch,), jnp.bool_),
        last_position=jnp.zeros((batch, 3), jnp.float32), had_position=jnp.zeros((batch,), jnp.bool_),
        evictions=jnp.zeros((batch,), jnp.int32))


def _nearest(world: jax.Array, valid: jax.Array, point: jax.Array) -> jax.Array:
    """int32: the nearest valid place to ``point`` (float32[3]) within ``MERGE_BLOCKS``, or -1; ``world`` is
    float32[P, 3]."""

    distance = jnp.linalg.norm(world - point, axis=-1)
    near = valid & (distance <= MERGE_BLOCKS)
    return jnp.where(jnp.any(near), jnp.argmin(jnp.where(near, distance, jnp.inf)), -1).astype(jnp.int32)


def _tick(state: PlaceState, position: jax.Array, has_position: jax.Array, danger: jax.Array, alarm: jax.Array,
          find: jax.Array, respawned: jax.Array, marks_world: jax.Array, marks_live: jax.Array) -> PlaceState:
    """One agent's tick, in the order of the module docstring (:func:`update` vmaps it over the batch)."""

    now = state.now + 1
    places, routes = state.places, state.routes
    places = places._replace(strength=jnp.where(places.valid & ~places.marked, places.strength * _FADE_PLACE,
                                                places.strength))
    routes = routes._replace(strength=jnp.where(routes.valid, routes.strength * _FADE_ROUTE, routes.strength))

    moved = jnp.linalg.norm(position - state.stay_anchor) > STAY_BLOCKS
    reanchor = has_position & ((state.stay_since < 0) | moved)
    anchor = jnp.where(reanchor, position, state.stay_anchor)
    since = jnp.where(reanchor, now, state.stay_since)
    made = jnp.where(reanchor, False, state.stay_made)
    stay = has_position & (now - since >= STAY_TICKS) & ~made
    made = made | stay

    before = jnp.where(has_position, _nearest(places.world, places.valid, position), -1)
    revisit = (before >= 0) & (before != state.inside) & (jnp.arange(PLACE_SLOTS) == before)
    places = places._replace(strength=jnp.where(revisit, jnp.maximum(places.strength, REVISIT_STRENGTH),
                                                places.strength),
                             visits=places.visits + revisit.astype(jnp.int32),
                             last=jnp.where(revisit, now, places.last))

    marks = marks_world.shape[0]
    point = jnp.concatenate([jnp.stack([anchor, position, state.last_position, position]), marks_world])
    kind = jnp.concatenate([jnp.asarray([_KIND["stay"], _KIND["danger"], _KIND["death"], _KIND["find"]], jnp.int32),
                            jnp.full((marks,), _KIND["marked"], jnp.int32)])
    weight = jnp.concatenate([jnp.stack([jnp.float32(STAY_STRENGTH), 0.5 + 0.5 * alarm, jnp.float32(1.0),
                                         jnp.float32(FIND_STRENGTH)]), jnp.ones((marks,), jnp.float32)])
    mark = jnp.arange(_CANDIDATE_FIXED + marks) >= _CANDIDATE_FIXED
    live = jnp.concatenate([jnp.stack([stay, danger & has_position, respawned & state.had_position,
                                       find & has_position]), marks_live])

    def consolidate(carry, candidate):
        places, routes, inside, departed, evictions = carry
        where, what, strength, is_mark, valid = candidate
        here = _nearest(places.world, places.valid, where)
        free = ~places.valid
        has_free = jnp.any(free)
        evictable = places.valid & ~places.marked
        weakest = jnp.argmin(jnp.where(evictable, places.strength, jnp.inf))
        has_evictable = jnp.any(evictable)
        merge = valid & (here >= 0)
        new = valid & (here < 0) & (has_free | has_evictable)
        refused = valid & (here < 0) & ~has_free & ~has_evictable
        evict = new & ~has_free
        slot = jnp.where(here >= 0, here, jnp.where(has_free, jnp.argmax(free), weakest))
        at = jnp.arange(PLACE_SLOTS) == slot
        touched = (merge | new) & at
        fresh = new & at
        bits = jnp.arange(len(PLACE_KINDS)) == what
        touching = (routes.source == weakest) | (routes.target == weakest)
        routes = routes._replace(valid=routes.valid & ~(evict & touching))
        inside = jnp.where(evict & (inside == weakest), -1, inside)
        departed = jnp.where(evict & (departed == weakest), -1, departed)
        evictions = evictions + refused.astype(jnp.int32)
        places = places._replace(
            world=jnp.where(fresh[:, None], where, places.world),
            kinds=jnp.where(fresh[:, None], bits, places.kinds | (touched[:, None] & bits)),
            strength=jnp.where(fresh, strength, jnp.where(touched, jnp.maximum(places.strength, strength),
                                                          places.strength)),
            marked=jnp.where(fresh, is_mark, places.marked | (touched & is_mark)),
            first=jnp.where(fresh, now, places.first),
            last=jnp.where(touched, now, places.last),
            visits=jnp.where(fresh, 0, places.visits),
            valid=places.valid | fresh)
        return (places, routes, inside, departed, evictions), None

    (places, routes, inside, departed, evictions), _ = jax.lax.scan(
        consolidate, (places, routes, state.inside, state.departed, state.evictions),
        (point, kind, weight, mark, live))

    here = jnp.where(has_position, _nearest(places.world, places.valid, position), -1)
    leaving = has_position & (inside >= 0) & (here != inside)
    departed = jnp.where(leaving, inside, departed)
    departed_at = jnp.where(leaving, now, state.departed_at)
    arriving = has_position & (here >= 0) & (here != inside)
    record = arriving & (departed >= 0) & (departed != here)
    travel = now - departed_at
    match = routes.valid & (routes.source == departed) & (routes.target == here)
    known = jnp.any(match)
    free_route = ~routes.valid
    slot = jnp.where(known, jnp.argmax(match),
                     jnp.where(jnp.any(free_route), jnp.argmax(free_route), jnp.argmin(routes.strength)))
    at = (jnp.arange(ROUTE_SLOTS) == slot) & record
    fresh = at & ~known
    routes = routes._replace(
        source=jnp.where(fresh, departed, routes.source),
        target=jnp.where(fresh, here, routes.target),
        best_ticks=jnp.where(fresh, travel, jnp.where(at, jnp.minimum(routes.best_ticks, travel), routes.best_ticks)),
        last_ticks=jnp.where(at, travel, routes.last_ticks),
        count=jnp.where(fresh, 1, jnp.where(at, routes.count + 1, routes.count)),
        last=jnp.where(at, now, routes.last),
        strength=jnp.where(at, 1.0, routes.strength),
        valid=routes.valid | fresh)
    inside = jnp.where(has_position, here, inside)

    return PlaceState(
        now=now, places=places, routes=routes, inside=inside, departed=departed, departed_at=departed_at,
        stay_anchor=anchor, stay_since=since, stay_made=made,
        last_position=jnp.where(has_position, position, state.last_position), had_position=has_position,
        evictions=evictions)


def update(state: PlaceState, *, position: jax.Array, has_position: jax.Array, danger: jax.Array, alarm: jax.Array,
           find: jax.Array, respawned: jax.Array, marks_world: jax.Array, marks_live: jax.Array,
           reset: jax.Array | None = None) -> PlaceState:
    """The state after this tick (module docstring). Lanes where ``reset`` bool[B] is True start from
    :func:`initial_places` first. Every argument is B-leading: ``position`` float32[B, 3]; ``has_position`` bool[B];
    ``danger`` bool[B], a damage taken this tick; ``alarm`` float32[B], the affect rules' fear; ``find`` bool[B], an
    item gained this tick; ``respawned`` bool[B]; ``marks_world`` float32[B, M, 3] and ``marks_live`` bool[B, M], the
    decision layer's live place marks."""

    batch = state.now.shape[0]
    if reset is not None:
        fresh = initial_places(batch)
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(reset.reshape((batch,) + (1,) * (old.ndim - 1)), new, old), fresh, state)
    return jax.vmap(_tick)(state, position, has_position, danger, alarm, find, respawned, marks_world, marks_live)


def relative(state: PlaceState, position: jax.Array, body_yaw: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each place relative to the agent now (``position`` float32[B, 3], ``body_yaw`` float32[B] in radians): its
    displacement in the body frame float32[B, P, 3] (forward, right, up; zeros for an empty slot), its distance in
    blocks float32[B, P], and whether the slot holds a place bool[B, P]."""

    places = state.places
    offset = places.world - position[:, None, :]
    body = world_to_body(offset, body_yaw[:, None])
    valid = places.valid
    return (jnp.where(valid[..., None], body, 0.0), jnp.where(valid, jnp.linalg.norm(offset, axis=-1), 0.0), valid)


__all__ = ["DT_SECONDS", "FIND_STRENGTH", "MERGE_BLOCKS", "PLACE_KINDS", "PLACE_SLOTS", "REVISIT_STRENGTH",
           "ROUTE_SLOTS", "STAY_BLOCKS", "STAY_STRENGTH", "STAY_TICKS", "TAU_PLACE_SECONDS", "TAU_ROUTE_SECONDS",
           "PlaceState", "Places", "Routes", "initial_places", "relative", "update"]
