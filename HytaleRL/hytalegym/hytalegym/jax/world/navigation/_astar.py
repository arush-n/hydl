"""Fixed-budget A* helpers internal to the navigation package."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from hytalegym.jax.world.region.types import RegionTraversalAtlas


Array = jax.Array


ASTAR_PROGRESS_COMPUTING = 2


ASTAR_PROGRESS_ACCOMPLISHED = 3


ASTAR_PROGRESS_TERMINATED = 4


ASTAR_PROGRESS_TERMINATED_OPEN_NODE_LIMIT_EXCEEDED = 5


ASTAR_PROGRESS_TERMINATED_TOTAL_NODE_LIMIT_EXCEEDED = 6


_EMPTY = jnp.int32(-1)


_POSITION_OFFSET = jnp.float32(1023.25)


class RegionGraphAStarResult(NamedTuple):
    """One fixed-budget graph search; non-accomplished rows expose no path."""

    available: Array
    progress: Array
    path_node: Array
    path_mask: Array
    path_length: Array
    travel_cost: Array
    iterations: Array
    visited_count: Array
    open_count: Array
    invalid: Array


class _SearchState(NamedTuple):
    progress: Array
    iterations: Array
    visited_count: Array
    visited_key: Array
    visited_node: Array
    visited_cost: Array
    visited_total_cost: Array
    visited_parent: Array
    visited_length: Array
    visited_closed: Array
    open_count: Array
    open_slot: Array
    open_total_cost: Array
    goal_slot: Array
    open_overflow: Array
    total_overflow: Array


def _expand_once(
    state: _SearchState,
    atlas: RegionTraversalAtlas,
    graph: Array,
    goal_node: Array,
    goal_position: Array,
    start_position: Array,
    *,
    maximum_path_length: int,
    open_nodes_limit: int,
    total_nodes_limit: int,
) -> _SearchState:
    batch = graph.shape[0]
    batch_index = jnp.arange(batch, dtype=jnp.int32)
    active = (
        state.progress == ASTAR_PROGRESS_COMPUTING
    ) & (state.open_count > 0)
    open_capacity = state.open_slot.shape[1]
    visited_capacity = state.visited_node.shape[1]
    pop_index = jnp.clip(state.open_count - 1, 0, open_capacity - 1)
    pop_slot = state.open_slot[batch_index, pop_index]
    safe_pop_slot = jnp.clip(pop_slot, 0, visited_capacity - 1)
    current_node = state.visited_node[batch_index, safe_pop_slot]
    reached = active & (current_node == goal_node)
    closed = state.visited_closed.at[
        batch_index,
        safe_pop_slot,
    ].set(
        jnp.where(
            active,
            True,
            state.visited_closed[batch_index, safe_pop_slot],
        )
    )
    open_count = jnp.where(
        active & ~reached,
        state.open_count - 1,
        state.open_count,
    )
    progress = jnp.where(
        reached,
        jnp.int8(ASTAR_PROGRESS_ACCOMPLISHED),
        state.progress,
    )
    goal_slot = jnp.where(reached, safe_pop_slot, state.goal_slot)
    iterations = state.iterations + (active & ~reached).astype(jnp.int32)
    current_length = state.visited_length[
        batch_index,
        safe_pop_slot,
    ]
    expand = active & ~reached & (
        current_length < maximum_path_length
    )
    current = state._replace(
        progress=progress,
        iterations=iterations,
        visited_closed=closed,
        open_count=open_count,
        goal_slot=goal_slot,
    )

    edge_capacity = atlas.edge_mask.shape[2]

    def add_edge(edge_slot: int, value: _SearchState) -> _SearchState:
        return _add_edge(
            value,
            atlas,
            graph,
            safe_pop_slot,
            current_node,
            goal_position,
            start_position,
            edge_slot,
            expand,
        )

    current = jax.lax.fori_loop(0, edge_capacity, add_edge, current)
    open_limit_hit = expand & (
        (current.open_count >= open_nodes_limit)
        | current.open_overflow
    )
    total_limit_hit = expand & (
        (current.visited_count >= total_nodes_limit)
        | current.total_overflow
    )
    exhausted = (
        current.progress == ASTAR_PROGRESS_COMPUTING
    ) & (current.open_count == 0)
    progress = jnp.where(
        open_limit_hit,
        jnp.int8(ASTAR_PROGRESS_TERMINATED_OPEN_NODE_LIMIT_EXCEEDED),
        current.progress,
    )
    progress = jnp.where(
        ~open_limit_hit & total_limit_hit,
        jnp.int8(ASTAR_PROGRESS_TERMINATED_TOTAL_NODE_LIMIT_EXCEEDED),
        progress,
    )
    progress = jnp.where(
        exhausted & ~open_limit_hit & ~total_limit_hit,
        jnp.int8(ASTAR_PROGRESS_TERMINATED),
        progress,
    )
    return current._replace(progress=progress)


def _add_edge(
    state: _SearchState,
    atlas: RegionTraversalAtlas,
    graph: Array,
    current_slot: Array,
    current_node: Array,
    goal_position: Array,
    start_position: Array,
    edge_slot: int,
    expand: Array,
) -> _SearchState:
    batch = graph.shape[0]
    batch_index = jnp.arange(batch, dtype=jnp.int32)
    node_capacity = atlas.node_mask.shape[1]
    open_capacity = state.open_slot.shape[1]
    visited_capacity = state.visited_node.shape[1]
    safe_current_node = jnp.clip(current_node, 0, node_capacity - 1)
    destination = atlas.edge_destination[
        graph,
        safe_current_node,
        edge_slot,
    ]
    safe_destination = jnp.clip(destination, 0, node_capacity - 1)
    edge_valid = (
        expand
        & atlas.edge_mask[graph, safe_current_node, edge_slot]
        & (destination >= 0)
        & (destination < node_capacity)
        & atlas.node_mask[graph, safe_destination]
    )
    destination_position = atlas.node_position[
        graph,
        safe_destination,
    ]
    destination_key, key_valid = _native_position_key(
        destination_position,
        start_position,
    )
    edge_cost = atlas.edge_cost[
        graph,
        safe_current_node,
        edge_slot,
    ]
    edge_valid &= (
        key_valid
        & jnp.all(jnp.isfinite(destination_position), axis=1)
        & jnp.isfinite(edge_cost)
        & (edge_cost > 0.0)
    )
    occupied = (
        jnp.arange(visited_capacity)[None, :]
        < state.visited_count[:, None]
    )
    key_match = occupied & jnp.all(
        state.visited_key == destination_key[:, None, :],
        axis=2,
    )
    found = jnp.any(key_match, axis=1)
    found_slot = jnp.argmax(key_match, axis=1).astype(jnp.int32)
    new_node = edge_valid & ~found
    total_overflow = state.total_overflow | (
        new_node & (state.visited_count >= visited_capacity)
    )
    new_slot = jnp.clip(
        state.visited_count,
        0,
        visited_capacity - 1,
    )
    can_allocate = new_node & (
        state.visited_count < visited_capacity
    )
    open_overflow = state.open_overflow | (
        can_allocate & (state.open_count >= open_capacity)
    )
    can_insert = can_allocate & (
        state.open_count < open_capacity
    )
    current_cost = state.visited_cost[
        batch_index,
        current_slot,
    ]
    candidate_cost = current_cost + edge_cost
    estimate = _native_float_distance(
        destination_position,
        goal_position,
    )
    candidate_total = candidate_cost + estimate
    new_length = (
        state.visited_length[batch_index, current_slot] + 1
    )

    allocated = _allocate_visited(
        state,
        batch_index,
        new_slot,
        destination_key,
        safe_destination,
        candidate_cost,
        candidate_total,
        current_slot,
        new_length,
        can_insert,
    )
    allocated = _insert_open(
        allocated,
        new_slot,
        candidate_total,
        can_insert,
        open_capacity,
    )
    allocated = allocated._replace(
        visited_count=jnp.where(
            can_insert,
            allocated.visited_count + 1,
            allocated.visited_count,
        ),
        open_overflow=open_overflow,
        total_overflow=total_overflow,
    )
    existing_slot = jnp.clip(
        found_slot,
        0,
        visited_capacity - 1,
    )
    existing_cost = allocated.visited_cost[
        batch_index,
        existing_slot,
    ]
    improve = edge_valid & found & (
        candidate_cost < existing_cost
    )
    return _improve_visited(
        allocated,
        batch_index,
        existing_slot,
        candidate_cost,
        candidate_total,
        current_slot,
        new_length,
        improve,
    )


def _allocate_visited(
    state: _SearchState,
    batch_index: Array,
    slot: Array,
    key: Array,
    node: Array,
    cost: Array,
    total_cost: Array,
    parent: Array,
    length: Array,
    apply: Array,
) -> _SearchState:
    old_key = state.visited_key[batch_index, slot]
    old_node = state.visited_node[batch_index, slot]
    old_cost = state.visited_cost[batch_index, slot]
    old_total = state.visited_total_cost[batch_index, slot]
    old_parent = state.visited_parent[batch_index, slot]
    old_length = state.visited_length[batch_index, slot]
    return state._replace(
        visited_key=state.visited_key.at[batch_index, slot].set(
            jnp.where(apply[:, None], key, old_key)
        ),
        visited_node=state.visited_node.at[batch_index, slot].set(
            jnp.where(apply, node, old_node)
        ),
        visited_cost=state.visited_cost.at[batch_index, slot].set(
            jnp.where(apply, cost, old_cost)
        ),
        visited_total_cost=state.visited_total_cost.at[
            batch_index,
            slot,
        ].set(jnp.where(apply, total_cost, old_total)),
        visited_parent=state.visited_parent.at[batch_index, slot].set(
            jnp.where(apply, parent, old_parent)
        ),
        visited_length=state.visited_length.at[batch_index, slot].set(
            jnp.where(apply, length, old_length)
        ),
    )


def _insert_open(
    state: _SearchState,
    slot: Array,
    total_cost: Array,
    apply: Array,
    capacity: int,
) -> _SearchState:
    index = jnp.arange(capacity, dtype=jnp.int32)[None, :]
    active = index < state.open_count[:, None]
    insertion = jnp.sum(
        active & (state.open_total_cost >= total_cost[:, None]),
        axis=1,
        dtype=jnp.int32,
    )
    previous = jnp.clip(index - 1, 0, capacity - 1)
    shifted_slot = jnp.take_along_axis(
        state.open_slot,
        previous,
        axis=1,
    )
    shifted_cost = jnp.take_along_axis(
        state.open_total_cost,
        previous,
        axis=1,
    )
    next_slot = jnp.where(
        index < insertion[:, None],
        state.open_slot,
        jnp.where(
            index == insertion[:, None],
            slot[:, None],
            shifted_slot,
        ),
    )
    next_cost = jnp.where(
        index < insertion[:, None],
        state.open_total_cost,
        jnp.where(
            index == insertion[:, None],
            total_cost[:, None],
            shifted_cost,
        ),
    )
    return state._replace(
        open_slot=jnp.where(
            apply[:, None],
            next_slot,
            state.open_slot,
        ),
        open_total_cost=jnp.where(
            apply[:, None],
            next_cost,
            state.open_total_cost,
        ),
        open_count=jnp.where(
            apply,
            state.open_count + 1,
            state.open_count,
        ),
    )


def _improve_visited(
    state: _SearchState,
    batch_index: Array,
    slot: Array,
    cost: Array,
    total_cost: Array,
    parent: Array,
    length: Array,
    apply: Array,
) -> _SearchState:
    old_cost = state.visited_cost[batch_index, slot]
    old_total = state.visited_total_cost[batch_index, slot]
    old_parent = state.visited_parent[batch_index, slot]
    old_length = state.visited_length[batch_index, slot]
    visited_cost = state.visited_cost.at[batch_index, slot].set(
        jnp.where(apply, cost, old_cost)
    )
    visited_total = state.visited_total_cost.at[
        batch_index,
        slot,
    ].set(jnp.where(apply, total_cost, old_total))
    visited_parent = state.visited_parent.at[batch_index, slot].set(
        jnp.where(apply, parent, old_parent)
    )
    visited_length = state.visited_length.at[batch_index, slot].set(
        jnp.where(apply, length, old_length)
    )
    in_open = (
        jnp.arange(state.open_slot.shape[1])[None, :]
        < state.open_count[:, None]
    ) & (state.open_slot == slot[:, None])
    open_total = jnp.where(
        apply[:, None] & in_open,
        total_cost[:, None],
        state.open_total_cost,
    )
    return state._replace(
        visited_cost=visited_cost,
        visited_total_cost=visited_total,
        visited_parent=visited_parent,
        visited_length=visited_length,
        open_total_cost=open_total,
    )


def _result_from_state(
    state: _SearchState,
    available: Array,
    *,
    maximum_path_length: int,
) -> RegionGraphAStarResult:
    batch = available.shape[0]
    batch_index = jnp.arange(batch, dtype=jnp.int32)
    accomplished = (
        state.progress == ASTAR_PROGRESS_ACCOMPLISHED
    )
    safe_goal_slot = jnp.clip(
        state.goal_slot,
        0,
        state.visited_node.shape[1] - 1,
    )
    reverse = jnp.full(
        (batch, maximum_path_length),
        _EMPTY,
        dtype=jnp.int32,
    )

    def trace_parent(
        index: int,
        carry: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        nodes, cursor = carry
        active = accomplished & (cursor >= 0)
        safe_cursor = jnp.clip(
            cursor,
            0,
            state.visited_node.shape[1] - 1,
        )
        value = state.visited_node[batch_index, safe_cursor]
        nodes = nodes.at[:, index].set(
            jnp.where(active, value, _EMPTY)
        )
        parent = state.visited_parent[batch_index, safe_cursor]
        return nodes, jnp.where(active, parent, _EMPTY)

    reverse, _ = jax.lax.fori_loop(
        0,
        maximum_path_length,
        trace_parent,
        (reverse, state.goal_slot),
    )
    path_length = jnp.where(
        accomplished,
        jnp.sum(reverse >= 0, axis=1, dtype=jnp.int32),
        jnp.int32(0),
    )
    index = jnp.arange(maximum_path_length, dtype=jnp.int32)[None, :]
    path_mask = accomplished[:, None] & (
        index < path_length[:, None]
    )
    reverse_index = jnp.clip(
        path_length[:, None] - 1 - index,
        0,
        maximum_path_length - 1,
    )
    path_node = jnp.take_along_axis(
        reverse,
        reverse_index,
        axis=1,
    )
    path_node = jnp.where(path_mask, path_node, _EMPTY)
    travel_cost = jnp.where(
        accomplished,
        state.visited_cost[batch_index, safe_goal_slot],
        jnp.float32(0.0),
    )
    return RegionGraphAStarResult(
        available=available,
        progress=state.progress,
        path_node=path_node,
        path_mask=path_mask,
        path_length=path_length,
        travel_cost=travel_cost,
        iterations=state.iterations,
        visited_count=state.visited_count,
        open_count=state.open_count,
        invalid=~available,
    )


def _native_position_key(
    position: Array,
    start_position: Array,
) -> tuple[Array, Array]:
    start_block = jnp.floor(start_position)
    key = jnp.floor(
        (position - start_block) * jnp.float32(2.0)
        + _POSITION_OFFSET
    ).astype(jnp.int32)
    valid = jnp.all(
        jnp.isfinite(position)
        & (key >= 0)
        & (key <= 2047),
        axis=1,
    )
    return key, valid


def _native_float_distance(left: Array, right: Array) -> Array:
    """Match ``Vector3d.distance`` followed by Java's ``double``-to-float cast."""

    with jax.enable_x64():
        delta = left.astype(jnp.float64) - right.astype(jnp.float64)
        distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    return distance.astype(jnp.float32)
