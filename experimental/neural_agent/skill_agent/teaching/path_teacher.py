"""The path teacher (T1, 2026-10-25): walk_to's teacher that follows the shortest path over the world's traversal graph.

The straight teacher (``neural_skills/walk_to/teacher.py``) steers at the goal. On the 0.7.2 pilot terrain it fails the
longer trips (the baseline in ``bench/region_baseline.py``). This teacher plans on the graph that
``tasks/region_tasks.py`` reads as ``RegionTraversal``:

- At build, per lane, one reversed Dijkstra from the goal node over that world's graph (scipy, host) gives every node
  its cost to the goal, ``dist``. Edges: walk, climb (the graph's own, rises up to 1.3 blocks) and drops of at most 3.0
  blocks; deeper drops are left out until fall damage is checked.
- Each tick, the node under the feet is the nearest node within 0.75 blocks horizontally and 1.5 vertically, looked up
  in the 3 by 3 block columns around the feet. The next hop is the out-edge that minimises edge cost plus the
  destination's ``dist``. The waypoint is that next hop (``waypoint_hops=1``, the default). ``waypoint_hops=2`` aims two
  hops on instead, the first design; measured 2026-10-25, it can cut a climb's corner and stall or fall. The final hop
  aims at the goal with the goal's radius.
- ``walk_to_choices`` steers the body at that target with the registered rule (``state_for``: the observed yaw) and the
  v3 stop margin, at the requested gait.

The field covers the whole graph, so a body pushed off its route is steered back by the next tick. Off the graph (no
node under the feet, or no route from it) the teacher steers straight at the goal, as the straight teacher does. Gen 7's
automatic step-up takes the climbs, so the teacher never jumps.

Edge costs are the graph's own (horizontal lengths on the pilot library, 2026-10-25). Per tick the teacher is pure JAX
over the batch axis, with the per-lane fields passed in as arrays; the field is built once on the host.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import (
    WALK_TO_V3_STOP_MARGIN_BLOCKS,
    walk_to_choices,
)
from experimental.neural_agent.skill_agent.tasks.region_tasks import RegionTraversal
from experimental.neural_agent.skill_agent.teaching.teachers import TeacherOutput, state_for
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatState

CLIMB_LIMIT_BLOCKS = 1.3
"""The most a climb edge rises: the gen-7 step-up height (``sim/gen7/mechanics/auto_step_up.py``, inferred for 0.7.2)."""
DROP_LIMIT_BLOCKS = 3.0
"""The deepest drop the teacher takes; the graph itself holds drops to 7 blocks (``worldgen/region/_traversal_edges.py``)."""
HEIGHT_TOLERANCE_BLOCKS = 1.0e-3
NODE_HORIZONTAL_REACH_BLOCKS = 0.75
"""The node under the feet lies within this horizontal distance (a block centre is 0.71 from any point of its block)."""
NODE_VERTICAL_REACH_BLOCKS = 1.5
WAYPOINT_RADIUS_BLOCKS = 0.5
"""The radius of an intermediate waypoint: the smallest walk_to radius, so the body passes it rather than parks."""
_NEIGHBOUR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


class PathTables(NamedTuple):
    """The traversal graphs as device arrays, the edges already filtered to the teacher's set."""

    node_position: jax.Array  # float32[R, N, 3]
    edge_ok: jax.Array  # bool[R, N, E]
    edge_destination: jax.Array  # int32[R, N, E] (0 where the edge is not taken)
    edge_cost: jax.Array  # float32[R, N, E]
    column_node_start: jax.Array  # int32[R, C, C]
    column_node_count: jax.Array  # int32[R, C, C]
    column_origin: jax.Array  # int32[R, 2]
    column_slot: jax.Array  # int32[K], arange(K) for the widest column


class PathPlan(NamedTuple):
    """Each lane's world, goal node and cost-to-goal field."""

    world: jax.Array  # int32[B]
    goal_node: jax.Array  # int32[B]
    dist: jax.Array  # float32[B, N], the graph cost from each node to the lane's goal (inf where no route)


def allowed_edges(traversal: RegionTraversal) -> np.ndarray:
    """bool[R, N, E]: the graph's edges the teacher takes (walk, climb up to 1.3, drops down to 3.0 blocks)."""

    worlds = np.arange(traversal.node_mask.shape[0])[:, None, None]
    destination = np.where(traversal.edge_mask, traversal.edge_destination, 0)
    rise = traversal.node_position[worlds, destination, 1] - traversal.node_position[:, :, None, 1]
    return (
        traversal.edge_mask
        & traversal.node_mask[worlds, destination]
        & (rise >= -(DROP_LIMIT_BLOCKS + HEIGHT_TOLERANCE_BLOCKS))
        & (rise <= CLIMB_LIMIT_BLOCKS + HEIGHT_TOLERANCE_BLOCKS)
    )


def path_tables(traversal: RegionTraversal, ok: np.ndarray | None = None) -> PathTables:
    """The device tables for ``traversal``; ``ok`` is ``allowed_edges(traversal)`` when given."""

    ok = allowed_edges(traversal) if ok is None else ok
    columns = traversal.column_node_count.max()
    return PathTables(
        node_position=jnp.asarray(traversal.node_position, dtype=jnp.float32),
        edge_ok=jnp.asarray(ok),
        edge_destination=jnp.asarray(np.where(ok, traversal.edge_destination, 0), dtype=jnp.int32),
        edge_cost=jnp.asarray(traversal.edge_cost, dtype=jnp.float32),
        column_node_start=jnp.asarray(traversal.column_node_start, dtype=jnp.int32),
        column_node_count=jnp.asarray(traversal.column_node_count, dtype=jnp.int32),
        column_origin=jnp.asarray(traversal.column_origin, dtype=jnp.int32),
        column_slot=jnp.arange(int(columns), dtype=jnp.int32),
    )


def _cost_to_goal(traversal: RegionTraversal, ok: np.ndarray, world: int, goals: np.ndarray) -> np.ndarray:
    """Dijkstra from each goal over the reversed graph: the cost from every node to that goal, float64 [len(goals), N]."""

    count = traversal.node_mask.shape[1]
    source = np.repeat(np.arange(count), traversal.edge_mask.shape[2])
    destination = traversal.edge_destination[world].reshape(-1).astype(np.int64)
    cost = traversal.edge_cost[world].reshape(-1).astype(np.float64)
    keep = ok[world].reshape(-1)
    source, destination, cost = source[keep], destination[keep], cost[keep]
    # An edge source -> destination is an entry (destination, source) of the reversed graph. csr_matrix sums duplicate
    # entries, so keep the cheapest edge per node pair first.
    order = np.lexsort((cost, source, destination))
    destination, source, cost = destination[order], source[order], cost[order]
    first = np.ones(cost.size, dtype=np.bool_)
    first[1:] = (destination[1:] != destination[:-1]) | (source[1:] != source[:-1])
    reversed_graph = csr_matrix((cost[first], (destination[first], source[first])), shape=(count, count))
    return dijkstra(reversed_graph, directed=True, indices=np.asarray(goals, dtype=np.int64))


def plan_paths(traversal: RegionTraversal, ok: np.ndarray | None = None) -> PathPlan:
    """Each lane's cost-to-goal field, one reversed Dijkstra per world on the host."""

    ok = allowed_edges(traversal) if ok is None else ok
    lanes = traversal.environment_world_id.shape[0]
    dist = np.full((lanes, traversal.node_mask.shape[1]), np.inf, dtype=np.float32)
    for world in np.unique(traversal.environment_world_id):
        chosen = np.flatnonzero(traversal.environment_world_id == world)
        dist[chosen] = _cost_to_goal(traversal, ok, int(world), traversal.goal_node[chosen])
    return PathPlan(
        world=jnp.asarray(traversal.environment_world_id, dtype=jnp.int32),
        goal_node=jnp.asarray(traversal.goal_node, dtype=jnp.int32),
        dist=jnp.asarray(dist),
    )


def node_under_feet(tables: PathTables, world: jax.Array, position: jax.Array) -> jax.Array:
    """int32[B]: the node under each lane's feet (position float32[B, 3]), or -1 where none is within reach."""

    origin = tables.column_origin[world]
    offsets = jnp.asarray(_NEIGHBOUR_OFFSETS, dtype=jnp.int32)
    planar = position[:, [0, 2]]
    cell = jnp.floor(planar)[:, None, :].astype(jnp.int32) + offsets[None] - origin[:, None, :]  # [B, 9, 2]
    side = tables.column_node_start.shape[1]
    inside = jnp.all((cell >= 0) & (cell < side), axis=-1)  # [B, 9]
    safe = jnp.clip(cell, 0, side - 1)
    column = (world[:, None], safe[..., 0], safe[..., 1])
    start = tables.column_node_start[column]  # [B, 9]
    count = tables.column_node_count[column]
    slot = tables.column_slot  # [K]
    node_count = tables.node_position.shape[1]
    node = jnp.clip(jnp.maximum(start, 0)[..., None] + slot, 0, node_count - 1)  # [B, 9, K]
    valid = inside[..., None] & (slot < count[..., None])
    xyz = tables.node_position[world[:, None, None], node]  # [B, 9, K, 3]
    horizontal = jnp.sqrt(jnp.sum((xyz[..., [0, 2]] - planar[:, None, None, :]) ** 2, axis=-1))
    vertical = jnp.abs(xyz[..., 1] - position[:, None, None, 1])
    fit = valid & (horizontal <= NODE_HORIZONTAL_REACH_BLOCKS) & (vertical <= NODE_VERTICAL_REACH_BLOCKS)
    score = jnp.where(fit, horizontal + vertical, jnp.inf).reshape(position.shape[0], -1)
    best = jnp.argmin(score, axis=-1)
    chosen = jnp.take_along_axis(node.reshape(position.shape[0], -1), best[:, None], axis=1)[:, 0]
    return jnp.where(jnp.isfinite(jnp.min(score, axis=-1)), chosen, -1)


def next_hop(tables: PathTables, plan: PathPlan, node: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(next node int32[B], found bool[B]): the out-edge of ``node`` minimising edge cost plus the destination's cost."""

    lanes = node.shape[0]
    safe = jnp.maximum(node, 0)
    destination = tables.edge_destination[plan.world, safe]  # [B, E]
    value = tables.edge_cost[plan.world, safe] + plan.dist[jnp.arange(lanes)[:, None], destination]
    value = jnp.where(tables.edge_ok[plan.world, safe] & (node[:, None] >= 0), value, jnp.inf)
    slot = jnp.argmin(value, axis=-1)
    best = jnp.take_along_axis(value, slot[:, None], axis=1)[:, 0]
    chosen = jnp.take_along_axis(destination, slot[:, None], axis=1)[:, 0]
    found = jnp.isfinite(best)
    return jnp.where(found, chosen, -1), found


def _choices(
    tables: PathTables,
    plan: PathPlan,
    combat: CombatState,
    target_xz: jax.Array,
    radius: jax.Array,
    gait: jax.Array,
    *,
    maximum_turn_degrees: float,
    stop_margin_blocks: float,
    waypoint_hops: int,
) -> TeacherOutput:
    lanes = plan.world.shape[0]
    position = combat.position[:, AGENT_ENTITY]
    goal = plan.goal_node
    node = node_under_feet(tables, plan.world, position)
    on_route = (node >= 0) & jnp.isfinite(plan.dist[jnp.arange(lanes), jnp.maximum(node, 0)])
    node = jnp.where(on_route, node, -1)
    hop1, found1 = next_hop(tables, plan, node)
    hop2, found2 = next_hop(tables, plan, hop1)
    at_goal = node == goal
    # The waypoint is two hops on (one, if ``waypoint_hops`` is 1), unless that is the goal or the route ends sooner; off
    # the route it is the goal.
    if waypoint_hops == 1:
        waypoint = jnp.where(at_goal | ~found1, goal, hop1)
    else:
        waypoint = jnp.where(at_goal | ~found1, goal, jnp.where(found2 & (hop1 != goal), hop2, hop1))
    final = waypoint == goal
    waypoint_xz = tables.node_position[plan.world, jnp.maximum(waypoint, 0)][:, [0, 2]]
    aim = jnp.where(final[:, None], target_xz, waypoint_xz)
    aim_radius = jnp.where(final, radius, WAYPOINT_RADIUS_BLOCKS)
    return walk_to_choices(
        state_for("walk_to", combat), aim, aim_radius, gait,
        maximum_turn_degrees=maximum_turn_degrees, stop_margin_blocks=stop_margin_blocks,
    )


_choices_jit = jax.jit(_choices, static_argnames=("maximum_turn_degrees", "stop_margin_blocks", "waypoint_hops"))


class PathTeacher:
    """The path teacher on one Region environment's traversal graphs.

    ``teacher(combat, target_xz, radius, gait)`` returns the ``TeacherOutput`` of ``walk_to_choices`` aimed along the route
    to the goal node. ``target_xz`` must be the goal node's planar position (the placement's goal).
    """

    def __init__(
        self,
        traversal: RegionTraversal,
        *,
        maximum_turn_degrees: float,
        stop_margin_blocks: float = WALK_TO_V3_STOP_MARGIN_BLOCKS,
        waypoint_hops: int = 1,
    ) -> None:
        if waypoint_hops not in (1, 2):
            raise ValueError(f"waypoint_hops must be 1 or 2, got {waypoint_hops}")
        ok = allowed_edges(traversal)
        self.tables = path_tables(traversal, ok)
        self.plan = plan_paths(traversal, ok)
        self.maximum_turn_degrees = float(maximum_turn_degrees)
        self.stop_margin_blocks = float(stop_margin_blocks)
        self.waypoint_hops = waypoint_hops

    def __call__(self, combat: CombatState, target_xz: jax.Array, radius: jax.Array, gait: jax.Array) -> TeacherOutput:
        return _choices_jit(
            self.tables, self.plan, combat, target_xz, radius, gait,
            maximum_turn_degrees=self.maximum_turn_degrees, stop_margin_blocks=self.stop_margin_blocks,
            waypoint_hops=self.waypoint_hops,
        )


def next_edge(traversal: RegionTraversal, ok: np.ndarray, dist: np.ndarray, world: int, node: int):
    """(destination, kind) of the out-edge of ``node`` that minimises edge cost plus the destination's cost, or None."""

    destination = traversal.edge_destination[world, node]
    value = np.where(ok[world, node], traversal.edge_cost[world, node] + dist[destination], np.inf)
    slot = int(np.argmin(value))
    if not np.isfinite(value[slot]):
        return None
    return int(destination[slot]), int(traversal.edge_kind[world, node, slot])


def route_edges(traversal: RegionTraversal, ok: np.ndarray, dist: np.ndarray, lane: int) -> list[tuple[int, int, int]]:
    """The (source, destination, kind) edges of one shortest route for ``lane``, read down its cost field ``dist``.

    ``dist`` is one lane's row of ``plan.dist``. The route is empty where the start has no route to the goal.
    """

    world = int(traversal.environment_world_id[lane])
    node = int(traversal.start_node[lane])
    edges: list[tuple[int, int, int]] = []
    for _ in range(traversal.node_mask.shape[1]):
        if dist[node] <= 0.0 or not np.isfinite(dist[node]):
            break
        step = next_edge(traversal, ok, dist, world, node)
        if step is None:
            break
        edges.append((node, step[0], step[1]))
        node = step[0]
    return edges


__all__ = [
    "CLIMB_LIMIT_BLOCKS",
    "DROP_LIMIT_BLOCKS",
    "PathPlan",
    "PathTables",
    "PathTeacher",
    "allowed_edges",
    "next_edge",
    "next_hop",
    "node_under_feet",
    "path_tables",
    "plan_paths",
    "route_edges",
]
