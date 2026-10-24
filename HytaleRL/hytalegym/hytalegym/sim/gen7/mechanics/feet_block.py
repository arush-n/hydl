"""Gen-7 block at the body's feet (S12 producer): the semantic id of the block in the body's feet cell.

Cell rule: ``(floor x, floor y, floor z)`` of the body's position, the feet point (inferred: a Hytale entity's position
is at its feet). Why the cell the body is in, not the support cell below it: the 0.7.2 movement audio reads
``PositionDataComponent.getInsideBlockTypeId()`` and plays MoveOut when that id changes (confirmed, javap of the
server jar: AudioSystems.txt l.96-135 in the S12 lane folder). The footstep table's audible blocks (Plant_Bush,
Deco_SpiderWeb_Flat, Fluid_Water) are passable and sit on a solid support, so a support-cell lookup reads the soil
under them and never hears them. Boundary tie: a feet point exactly on a cell face reads the cell on its + side
(floor). Airborne bodies read whatever cell they are in; no grounded gate (a cost, see the report).

Each row is one arena; the helpers take [B, E, 3] positions and return [B, E] ids.
Sources: dense ``GeometryState.block_type_id`` (9x9x9 grid in the ``cell_mask`` layout, ``origin``) and the Region
semantic atlas through ``lookup_region_block_semantics``. Out of window, no grid, outside the Region's core, or
unknown: 0. No builder attaches the result to ``world_capabilities`` yet (see heard_sounds.py).
"""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS
from hytalegym.jax.world.region.block_semantics import lookup_region_block_semantics

SIDE = CELL_RADIUS * 2 + 1


def feet_cells(position):
    """[..., 3] float feet positions -> [..., 3] int32 cells (floor)."""

    return jnp.floor(position).astype(jnp.int32)


def dense_block_under_feet(geometry, position):
    """[B, E, 3] feet positions -> [B, E] int32 semantic ids from a dense GeometryState's identity grid (0 when absent).

    Each row is one arena with its own origin and grid, so row b reads only its own grid.
    """

    batch, entities = position.shape[0], position.shape[1]
    grid = getattr(geometry, "block_type_id", None)
    if grid is None:
        return jnp.zeros((batch, entities), dtype=jnp.int32)
    origin = geometry.origin.astype(jnp.int32)[:, None, :]  # [B, 1, 3]
    local = feet_cells(position) - origin + CELL_RADIUS  # [B, E, 3]
    inside = jnp.all((local >= 0) & (local < SIDE), axis=-1)
    index = jnp.clip((local[..., 0] * SIDE + local[..., 1]) * SIDE + local[..., 2], 0, CELL_COUNT - 1)
    grid_be = jnp.broadcast_to(grid[:, None, :], (batch, entities, CELL_COUNT))
    ids = jnp.take_along_axis(grid_be, index[..., None], axis=2)[..., 0]
    return jnp.where(inside, ids, 0).astype(jnp.int32)


def region_block_under_feet(semantic_atlas, physical_atlas, environment_world_id, position):
    """[B, E, 3] feet positions -> [B, E] int32 semantic ids from a Region world (the atlas lookup of each feet cell).

    ``environment_world_id`` is [B]; every entity of row b uses that row's world. The lookup runs on the flattened
    [B*E, 3] cells (not exercised by a test yet).
    """

    batch, entities = position.shape[0], position.shape[1]
    cells = feet_cells(position).reshape(batch * entities, 3)
    worlds = jnp.broadcast_to(jnp.asarray(environment_world_id)[:, None], (batch, entities)).reshape(-1)
    selection = lookup_region_block_semantics(semantic_atlas, physical_atlas, cells, worlds)
    known = selection.available & selection.block_present
    ids = jnp.where(known, selection.block_type_id, 0)
    return ids.reshape(batch, entities).astype(jnp.int32)


def block_under_feet_for_capabilities(geometry, position):
    """The capability field (``ArsenalWorldCapabilities.block_under_feet``): [B, E] ids on the dense path, else None.

    Duck-typed on ``block_type_id``: the render mounts its own copy of the package, so an isinstance check against
    GeometryState could compare two classes. A geometry without an identity grid (or a Region) gives None.
    """

    if getattr(geometry, "block_type_id", None) is None:
        return None
    return dense_block_under_feet(geometry, position)
