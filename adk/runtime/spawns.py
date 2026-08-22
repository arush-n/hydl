"""Composable spawn sources for a scene's reset.

A scene that binds no spawn source keeps the environment's default placement.
That default suits flat ground and is *wrong on real terrain*: a Region duel
built without one spawns both fighters at the flat default height, which on a
captured Region is underground -- no line of sight, no attack ever legal,
reward identically zero, and nothing reports a problem.

A spawn source is exactly the Gym's ``RegionResetProvider``::

    keys -> ArsenalResetBatch(agent_position, target_position)

Both arrays are ``(batch, 3)``. Because a source is just that function, any
function taking a source and returning a source composes with every other one;
:func:`shifted` and :func:`swapped` are the two shipped combinators and are
also worked examples for writing more.

    spawns = shifted(
        region_pool_spawns(pool_path, batch=8),
        target=(0.0, 0.0, 1.5),
    )
    scene = kit.make_scene("duel", spawns=spawns, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.environment import ArsenalResetBatch

# ``hytalegym.jax.training.region_resets`` is imported lazily inside
# ``region_pool_spawns``.  At module scope it pulled eleven
# ``hytalegym.jax.training.*`` modules into every process that so much as
# imported ``agent``, which is the boundary ``test_import_boundaries`` exists
# to defend: the core SDK must stay usable for inference without dragging in
# the training stack.  Only this one function needs it.


#: ``keys -> ArsenalResetBatch``; identical to the Gym's ``RegionResetProvider``.
SpawnSource = Callable[[jax.Array], ArsenalResetBatch]


def _positions(value: Any, batch: int, name: str) -> jax.Array:
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.shape == (3,):
        array = jnp.broadcast_to(array, (batch, 3))
    if array.shape != (batch, 3):
        raise ValueError(
            f"{name} must have shape (3,) or ({batch}, 3), got {array.shape}"
        )
    return array


def region_pool_spawns(
    pool_path: str | Path,
    *,
    batch: int,
    world_ids: Any | None = None,
) -> SpawnSource:
    """Sample spawns from a frozen Region combat reset pool.

    This is the placement that belongs with a captured Region: the pool's pairs
    were qualified against that Region's own surface, so fighters stand on the
    terrain rather than inside it. The pool must match the Region library --
    compare the pool directory's identity against the fixture's
    ``region_library_semantic_sha256``.

    ``world_ids`` is the per-lane Region index, shape ``(batch,)``; it defaults
    to lane 0 for every environment, which is correct for a single-world pool.
    """

    from hytalegym.jax.training.region_resets import (
        load_region_combat_reset_pool,
        make_region_combat_reset_provider,
    )

    if batch < 1:
        raise ValueError("batch must be a positive integer")
    pool = load_region_combat_reset_pool(Path(pool_path))
    if world_ids is None:
        selected = jnp.zeros((batch,), dtype=jnp.int32)
    else:
        selected = jnp.asarray(world_ids, dtype=jnp.int32)
        if selected.shape != (batch,):
            raise ValueError(
                f"world_ids must have shape ({batch},), got {selected.shape}"
            )
    return make_region_combat_reset_provider(pool, selected)


#: Sky light reads ~61440 in open air and ~0 deep underground, so any
#: threshold between the two modes separates surface from cave.
DEFAULT_MINIMUM_LIGHT = 60_000

#: How far inside the captured footprint a spawn is placed, in blocks.  A
#: fight that reaches the boundary leaves the capture, and every geometry
#: query outside it is unavailable -- so collision stops being enforced and
#: the Gym marks the episode ``geometry_exhausted``.
DEFAULT_MARGIN = 24.0


def _standable_world_cells(
    region_path: Path,
    light_path: Path,
) -> tuple[Any, Any, tuple[float, float, float, float]]:
    """Return (positions, light, footprint) for every standable cell.

    Standable means: nothing to collide with at the cell, nothing directly
    above it (headroom), and a colliding block directly below. Positions are
    world-frame block centres; ``footprint`` is the captured
    ``(min_x, min_z, max_x, max_z)`` outside which every geometry query is
    unavailable.

    Blocking is decided by the capture's own shape palette rather than by
    ``code != 0``. Foliage and fluids carry a non-zero code and no collision
    box, so the cruder test both hides real headroom and offers grass as a
    floor.
    """

    import numpy as np

    region = np.load(region_path)
    illumination = np.load(light_path)
    code = region["__cell_code__"]
    known = region["__section_known__"]
    light = illumination["light_raw_yzx"]
    available = illumination["__section_available__"]

    origin = region["__core_min_chunk_xz__"]
    if not np.array_equal(origin, illumination["__core_min_chunk_xz__"]):
        raise ValueError(
            "region and light captures do not share a chunk origin: "
            f"{origin.tolist()} vs "
            f"{illumination['__core_min_chunk_xz__'].tolist()}"
        )

    chunks, sections = code.shape[0], code.shape[1]
    per_axis = int(round(chunks ** 0.5))
    size = 32
    blocking = region["shape_box_mask"][region["cell_shape_index"]].any(axis=1)
    blocking[0] = False  # code 0 is air whatever its palette entry says
    solid = blocking[code].reshape(chunks, sections, size, size, size)

    usable = known & available
    standable = np.zeros_like(solid)
    standable[:, :, 1:-1] = (
        (~solid[:, :, 1:-1]) & (~solid[:, :, 2:]) & solid[:, :, :-2]
    )
    standable &= usable[:, :, None, None, None]

    chunk_index, section, ly, lz, lx = np.nonzero(standable)
    chunk_x, chunk_z = np.divmod(chunk_index, per_axis)
    world_x = (int(origin[0]) + chunk_x) * size + lx + 0.5
    world_z = (int(origin[1]) + chunk_z) * size + lz + 0.5
    world_y = section * size + ly
    values = light[chunk_index, section, ly, lz, lx]
    span = per_axis * size
    footprint = (
        float(int(origin[0]) * size),
        float(int(origin[1]) * size),
        float(int(origin[0]) * size + span),
        float(int(origin[1]) * size + span),
    )
    return (
        np.stack((world_x, world_y.astype(np.float32), world_z), axis=1),
        values,
        footprint,
    )


def lit_surface_spawns(
    region_path: str | Path,
    light_path: str | Path,
    *,
    batch: int,
    minimum_light: int = DEFAULT_MINIMUM_LIGHT,
    separation: tuple[float, float] = (1.0, 4.0),
    include_caves: bool = False,
    margin: float = DEFAULT_MARGIN,
    pairs: int = 4096,
    seed: int = 0,
) -> SpawnSource:
    """Spawn where the sky actually reaches, on ground, facing a nearby partner.

    This is the placement a captured Region needs and its reset pool does not
    supply: the pool's pairs are qualified against whichever Region produced
    them, so sampling one for a *different* Region can bury both fighters in
    stone. Here every candidate is checked against this Region's own blocks
    and its own native light capture.

    A cell qualifies when it is clear, has headroom, has a colliding block
    beneath, and its native light is at least ``minimum_light`` -- which on a
    captured Region means open sky rather than cave. Set ``include_caves`` to
    also accept dark standable cells, for deliberately underground scenarios.

    ``margin`` insets candidates from the edge of the captured footprint.
    Outside that footprint every geometry query returns unavailable, which
    sets the Gym's sticky ``geometry_exhausted`` bit and makes the rest of the
    episode uncertified terrain -- collision included. Placing a fight next to
    the boundary walks it out within a few seconds, so the default keeps room
    to move. Pass ``margin=0.0`` to sample the whole capture deliberately.

    Pairs are precomputed on the host and sampled per lane at reset, so no
    filesystem or search work enters the compiled step.
    """

    import numpy as np

    if batch < 1:
        raise ValueError("batch must be a positive integer")
    low, high = float(separation[0]), float(separation[1])
    if not 0.0 <= low < high:
        raise ValueError("separation must be (low, high) with 0 <= low < high")

    if margin < 0.0:
        raise ValueError("margin must not be negative")

    positions, light, footprint = _standable_world_cells(
        Path(region_path), Path(light_path)
    )
    min_x, min_z, max_x, max_z = footprint
    inside = (
        (positions[:, 0] >= min_x + margin)
        & (positions[:, 0] <= max_x - margin)
        & (positions[:, 2] >= min_z + margin)
        & (positions[:, 2] <= max_z - margin)
    )
    keep = inside if include_caves else inside & (light >= minimum_light)
    chosen = positions[keep]
    if chosen.shape[0] < 2:
        raise ValueError(
            "no standable cells qualified: "
            f"{int(keep.sum())} of {positions.shape[0]} standable cells are "
            f"both lit (>= {minimum_light}) and at least {margin} blocks "
            "inside the capture; lower minimum_light or margin, or set "
            "include_caves=True"
        )

    # Bucket by a grid the size of the separation window so a partner search
    # only ever looks at neighbouring buckets.
    cell = max(1.0, high)
    keys = np.floor(chosen[:, [0, 2]] / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (bx, bz) in enumerate(map(tuple, keys)):
        buckets.setdefault((int(bx), int(bz)), []).append(index)

    rng = np.random.default_rng(seed)
    order = rng.permutation(chosen.shape[0])
    found_a: list[Any] = []
    found_b: list[Any] = []
    for index in order:
        if len(found_a) >= pairs:
            break
        bx, bz = int(keys[index, 0]), int(keys[index, 1])
        neighbourhood: list[int] = []
        for ox in (-1, 0, 1):
            for oz in (-1, 0, 1):
                neighbourhood.extend(buckets.get((bx + ox, bz + oz), ()))
        if len(neighbourhood) < 2:
            continue
        candidates = chosen[neighbourhood]
        delta = candidates - chosen[index]
        distance = np.linalg.norm(delta[:, [0, 2]], axis=1)
        ok = (distance >= low) & (distance <= high) & (np.abs(delta[:, 1]) <= 1.0)
        if not ok.any():
            continue
        partner = candidates[rng.choice(np.nonzero(ok)[0])]
        found_a.append(chosen[index])
        found_b.append(partner)

    if not found_a:
        raise ValueError(
            "no qualifying spawn pairs: widen separation or include_caves"
        )
    agent_pool = jnp.asarray(np.stack(found_a), dtype=jnp.float32)
    target_pool = jnp.asarray(np.stack(found_b), dtype=jnp.float32)
    count = agent_pool.shape[0]

    def provider(keys_in: jax.Array) -> ArsenalResetBatch:
        picks = jax.vmap(
            lambda key: jax.random.randint(key, (), 0, count, dtype=jnp.int32)
        )(keys_in)
        return ArsenalResetBatch(
            agent_position=agent_pool[picks],
            target_position=target_pool[picks],
        )

    provider.pair_count = count  # type: ignore[attr-defined]
    return provider


def fixed_spawns(
    agent_position: Sequence[float] | Any,
    target_position: Sequence[float] | Any,
    *,
    batch: int,
) -> SpawnSource:
    """Place every environment at the same explicit pair of coordinates.

    Useful for a reproducible scenario -- a fixed range, a specific ledge --
    where sampling a pool would add variance you do not want.
    """

    agent = _positions(agent_position, batch, "agent_position")
    target = _positions(target_position, batch, "target_position")

    def provider(keys: jax.Array) -> ArsenalResetBatch:
        del keys  # placement is deterministic
        return ArsenalResetBatch(agent_position=agent, target_position=target)

    return provider


def shifted(
    source: SpawnSource,
    *,
    agent: Sequence[float] = (0.0, 0.0, 0.0),
    target: Sequence[float] = (0.0, 0.0, 0.0),
) -> SpawnSource:
    """Translate a source's placements.

    Shifting only the target is how to open or close starting range without
    giving up a pool's terrain-qualified footing. Note the shift is applied
    blindly: a large offset can push a fighter into rock, because nothing
    re-qualifies the result against the geometry.
    """

    agent_delta = jnp.asarray(agent, dtype=jnp.float32)
    target_delta = jnp.asarray(target, dtype=jnp.float32)
    if agent_delta.shape != (3,) or target_delta.shape != (3,):
        raise ValueError("agent and target offsets must have shape (3,)")

    def provider(keys: jax.Array) -> ArsenalResetBatch:
        batch = source(keys)
        return ArsenalResetBatch(
            agent_position=batch.agent_position + agent_delta,
            target_position=batch.target_position + target_delta,
        )

    return provider


def swapped(source: SpawnSource) -> SpawnSource:
    """Exchange the agent and target placements.

    Both sides of a pool's pair are terrain-qualified, so swapping is safe and
    doubles the starting configurations a pool provides.
    """

    def provider(keys: jax.Array) -> ArsenalResetBatch:
        batch = source(keys)
        return ArsenalResetBatch(
            agent_position=batch.target_position,
            target_position=batch.agent_position,
        )

    return provider


def describe_spawns(source: SpawnSource, *, batch: int, seed: int = 0) -> str:
    """Sample once and report placements. HOST-SIDE: syncs, not traceable."""

    keys = jax.random.split(jax.random.key(seed), batch)
    placed = source(keys)
    agent = jnp.asarray(placed.agent_position)
    target = jnp.asarray(placed.target_position)
    separation = jnp.linalg.norm(target - agent, axis=-1)
    lines = [f"{batch} lane(s), seed {seed}"]
    for lane in range(batch):
        lines.append(
            f"  lane {lane}: agent "
            f"{[round(float(v), 2) for v in agent[lane]]} target "
            f"{[round(float(v), 2) for v in target[lane]]} "
            f"separation {float(separation[lane]):.2f}"
        )
    lines.append(
        f"  y range {float(agent[:, 1].min()):.2f}..{float(agent[:, 1].max()):.2f}"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MINIMUM_LIGHT",
    "SpawnSource",
    "describe_spawns",
    "fixed_spawns",
    "lit_surface_spawns",
    "region_pool_spawns",
    "shifted",
    "swapped",
]
