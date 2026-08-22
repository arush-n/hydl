"""Build real geometry frames from parameters.

`synthetic.py` fabricates *answers* to world queries. This builds actual cells,
so collision, support and fluid are real.

**Scale, before you design an arena around it.** A frame is a 9x9x9
neighbourhood -- `CELL_RADIUS = 4`, `CELL_COUNT = 729`
(`hytalegym/geometry/contract.py:13-15`) -- anchored at `origin` and addressed
by signed offsets in [-4, +4]. Same extent the real captures use, so it is
faithful, but it is a cube around the agent, not a map. A pit, ledge, pillar,
pool or corridor fits; a parkour *course* does not.

The loader sanctions synthetic frames: `require_exact=False` exists so callers
can "run an approximate simulator frame" (`jax/world/types.py:158`), and
`batch_size` broadcasts one immutable frame across a batch. Conversion happens
outside JIT by design, so a frame is authored once per scene, never per tick.

Nothing here writes to the captured region library; these frames are synthetic
and never claim to be captures.

    from adk.environments.frames import build
    geometry = build("lava_pool", batch_size=8)

Field semantics were **measured 2026-08-08**, and the first three attempts were
wrong. What actually matters, in order:

* `flags & FLAG_SOLID` is solidity. `runtime_block_id`, `support` and a
  collision box are not -- an agent falls through 324 such cells.
* `origin` must bracket both entities, or the combat window is exhausted at
  reset and `done` fires at step 1.
* `fluid_movement[..., 3] > 0` is required for any fluid cell, or the frame is
  rejected at load.
* `fluid_damage` has **no reader in combat** and cannot hurt anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hytalegym.geometry.contract import (
    CELL_RADIUS, CELL_SIDE, FLAG_FLUID, FLAG_OPAQUE, FLAG_SOLID,
    empty_geometry,
)

__all__ = ["Arena", "ARENAS", "build", "frame",
           "FLAT", "PIT", "LAVA_POOL", "PILLARS", "CORRIDOR", "LEDGE", "ARENA_PIT"]

#: Any non-zero id reads as a real block; the specific value is arbitrary for a
#: synthetic frame. Captures use authored ids.
STONE = 1
FLUID = 2

#: The canonical slot-zero full unit cube the loader expects.
UNIT_CUBE = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

#: Speed multiplier written into `fluid_movement[..., 3]` for fluid cells.
#:
#: Not decorative: `geometry_state_from_numpy` rejects any `FLAG_FLUID` cell
#: whose multiplier is `<= 0` (`jax/world/types.py:307-310`). Leaving the row
#: zero raises `invalid fluid speed multiplier` at load.
FLUID_SPEED_MULTIPLIER = 1.0

#: Where the 9x9x9 window sits in world space.
#:
#: **Load-bearing.** Frames previously shipped with the default `[0, 0, 0]`
#: while entities spawn at y=65, so both entities fell outside the window,
#: `_combat_window_exhausted` set `geometry_exhausted` at reset, and
#: `combat/runtime/agent.py` then rejected the whole motion
#: (`safe_position = where(geometry_exhausted, position, motion.position)`).
#: The agent froze and `done` fired at step 1 -- so every measurement taken
#: over such a frame was reading a terminated episode, not terrain. Keep the
#: window bracketing both entities.
DEFAULT_ORIGIN = (0, 65, -1)

#: (dx, dy, dz) grids over the whole 9x9x9 neighbourhood, built once.
_AXIS = np.arange(-CELL_RADIUS, CELL_RADIUS + 1)
DX, DY, DZ = np.meshgrid(_AXIS, _AXIS, _AXIS, indexing="ij")

#: Flat cell index for every (dx, dy, dz), matching `cell_index`.
_INDEX = (
    (DX + CELL_RADIUS) * CELL_SIDE * CELL_SIDE
    + (DY + CELL_RADIUS) * CELL_SIDE
    + (DZ + CELL_RADIUS)
)


@dataclass(frozen=True)
class Arena:
    """A parametric 9x9x9 arena. Frozen -- use `.tuned(**kwargs)` for sweeps.

    Features compose in a fixed order: the floor is laid, the pit is carved,
    fluid fills the carve, then pillars and walls are added on top. So a pillar
    inside a pit rises from the pit floor, and a wall is never carved away.
    """

    name: str = "flat"
    #: Local y of the ground surface. Cells at or below are solid.
    floor_y: int = -1
    #: Where the window sits in world space -- see `DEFAULT_ORIGIN`. Must
    #: bracket both entities or the episode terminates at step 1.
    origin: tuple[int, int, int] = DEFAULT_ORIGIN

    # -- carve -------------------------------------------------------------
    #: Half-width of a pit centred on the agent, in cells. 0 disables.
    pit_half_width: int = 0
    #: Cells carved below `floor_y`.
    pit_depth: int = 4
    #: Carve a trench along z instead of a square pit.
    pit_is_trench: bool = False

    # -- fluid -------------------------------------------------------------
    #: Fill the carve with fluid rather than leaving it open.
    pit_fluid: bool = False
    #: Damage per fluid cell. Only meaningful with `pit_fluid`.
    fluid_damage: int = 0

    # -- add ---------------------------------------------------------------
    #: Columns at dx = +/-`pillar_offset`, `pillar_height` cells tall. 0 off.
    pillar_offset: int = 0
    pillar_height: int = 0
    #: Solid walls at dz = +/-`wall_offset`, `wall_height` tall. 0 off.
    wall_offset: int = 0
    wall_height: int = 0
    #: Raise the floor by this much for |dx| > `step_offset`, making a ledge.
    step_offset: int = 0
    step_height: int = 0

    def tuned(self, **kwargs) -> "Arena":
        """A copy with fields replaced."""

        return replace(self, **kwargs)

    # -- masks over the whole grid, vectorised ------------------------------

    def _carved(self) -> np.ndarray:
        """True where the pit removes floor."""

        if self.pit_half_width <= 0:
            return np.zeros(DX.shape, dtype=bool)
        within = np.abs(DX) <= self.pit_half_width
        if not self.pit_is_trench:
            within &= np.abs(DZ) <= self.pit_half_width
        depth = (DY > self.floor_y - self.pit_depth) & (DY <= self.floor_y)
        return within & depth

    def _surface(self) -> np.ndarray:
        """Local y of the ground surface per column, after any step."""

        surface = np.full(DX.shape, self.floor_y, dtype=np.int32)
        if self.step_offset > 0 and self.step_height:
            surface = np.where(
                np.abs(DX) > self.step_offset,
                self.floor_y + self.step_height,
                surface,
            )
        return surface

    def _added(self) -> np.ndarray:
        """True where pillars or walls add solid above the floor."""

        added = np.zeros(DX.shape, dtype=bool)
        if self.pillar_offset > 0 and self.pillar_height > 0:
            added |= (
                (np.abs(DX) == self.pillar_offset)
                & (DZ == 0)
                & (DY > self.floor_y)
                & (DY <= self.floor_y + self.pillar_height)
            )
        if self.wall_offset > 0 and self.wall_height > 0:
            added |= (
                (np.abs(DZ) == self.wall_offset)
                & (DY > self.floor_y)
                & (DY <= self.floor_y + self.wall_height)
            )
        return added

    def masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(solid, fluid)` boolean grids over the neighbourhood."""

        solid = (DY <= self._surface()) & ~self._carved()
        solid |= self._added()
        fluid = self._carved() & bool(self.pit_fluid) & ~solid
        return solid, fluid

    def build(self) -> dict:
        """Return a geometry frame dict for `geometry_state_from_numpy`."""

        return frame(*self.masks(), fluid_damage=self.fluid_damage,
                     origin=self.origin)


def frame(solid: np.ndarray, fluid: np.ndarray, *,
          fluid_damage: int = 0,
          origin: tuple[int, int, int] = DEFAULT_ORIGIN) -> dict:
    """Assemble a geometry frame from two boolean grids.

    Public so a caller can author masks any way it likes -- noise, a heightmap,
    a hand-drawn layout -- and still get a valid frame. `solid` wins over
    `fluid` where both are set.

    **`flags` is what makes a cell real.** The sweep gates solidity on
    `(flags & FLAG_SOLID) != 0` (`jax/world/geometry/_sweep.py:95,226,251`) and
    nothing else. Measured: a frame with 324 cells carrying
    `runtime_block_id=STONE`, `support=1` and a unit collision box, but no
    flags, does **not** hold the agent -- it falls straight through, identically
    to a frame with 81 such cells. `runtime_block_id != 0` is not solidity.
    """

    if solid.shape != DX.shape or fluid.shape != DX.shape:
        raise ValueError(f"masks must have shape {DX.shape}")
    fluid = fluid & ~solid

    out = empty_geometry()
    out["available"] = 1
    out["exact_collision_shapes"] = 0  # synthetic, never native-exact
    out["origin"][:] = np.asarray(origin, dtype=np.int32)

    # `agent_bounds`/`target_bounds` are left at the `empty_geometry()` zeros.
    # Known-imperfect and deliberately not guessed at: the sweep does read
    # `geometry.agent_bounds` (`arsenal/runtime.py:1919,1983`), so a zero extent
    # is not obviously harmless -- but the authoritative value is the profile's
    # `bounding_box` (`combat/types.py:511`), not a constant, and grounding
    # measures correct without it (agent held at y=65.000, `grounded=True`).
    # Substituting a plausible humanoid AABB changed no observed behaviour and
    # did not affect the crash in #35. Populate this from the live profile if a
    # measurement ever shows it matters; do not hardcode one here.

    solid_i = _INDEX[solid]
    fluid_i = _INDEX[fluid]

    out["cell_mask"][solid_i] = 1
    # SOLID stops movement; OPAQUE stops sight. They are separate bits and the
    # occlusion test reads only the second: `line_of_sight_result` gates on
    # `cell_mask & (flags & FLAG_OPAQUE)` (`jax/world/geometry/__init__.py:283`)
    # and never looks at FLAG_SOLID. Authoring SOLID alone therefore built
    # **see-through walls** -- measured 2026-08-09, before this line: `pillars`
    # had 330 solid cells, 0 opaque, and a segment straight through a pillar
    # returned `visible=True`. PILLARS is documented as "cover to break line of
    # sight around" and provided no cover at all.
    #
    # Every block this authors is STONE, which is opaque, so the two travel
    # together here. A glass-like block would set SOLID without OPAQUE; if this
    # ever authors more than one material, that is where they must separate.
    out["flags"][solid_i] = FLAG_SOLID | FLAG_OPAQUE
    out["runtime_block_id"][solid_i] = STONE
    out["support"][solid_i] = 1
    out["collision_boxes"][solid_i, 0] = UNIT_CUBE
    out["collision_box_mask"][solid_i, 0] = 1

    out["cell_mask"][fluid_i] = 1
    out["flags"][fluid_i] = FLAG_FLUID
    out["runtime_fluid_id"][fluid_i] = FLUID
    out["fluid_level"][fluid_i] = 1
    out["fluid_fill_height"][fluid_i] = 1.0
    out["fluid_movement"][fluid_i, 3] = FLUID_SPEED_MULTIPLIER
    # Written because the field exists, but INERT: `jax/combat/` contains zero
    # readers of a cell's `fluid_damage` (verified by grep). Its only consumers
    # are `world/navigation` traversal safety and a `world/surrogate` terrain
    # feature -- neither touches entity health. Fluid changes the movement
    # medium here; it does not hurt. Do not author a hazard arena expecting
    # damage and read the resulting flat health as a policy result.
    out["fluid_damage"][fluid_i] = fluid_damage
    return out


# -- presets -----------------------------------------------------------------

#: Flat ground. The control -- every other arena should differ from this.
FLAT = Arena(name="flat")

#: A 3-wide open pit under the agent. Pairs with the `jump` minigame.
PIT = Arena(name="pit", pit_half_width=1, pit_depth=4)

#: The same pit filled with fluid. **It does not damage** -- see `frame()`;
#: there is no reader of `fluid_damage` in combat. What it does change is the
#: movement medium: a submerged agent sinks at a roughly constant rate instead
#: of accelerating. Treat this as a swimming/traversal arena, not a hazard.
#: (An earlier comment here claimed "real fluid with real damage". That was
#: wrong and had never been run.)
LAVA_POOL = Arena(name="lava_pool", pit_half_width=1, pit_depth=3,
                  pit_fluid=True, fluid_damage=2)

#: Two columns flanking the agent: cover to break line of sight around.
PILLARS = Arena(name="pillars", pillar_offset=2, pillar_height=3)

#: Walls either side, forcing engagement along one axis.
CORRIDOR = Arena(name="corridor", wall_offset=2, wall_height=4)

#: Raised ground beyond |dx| > 1 -- high ground on both flanks.
LEDGE = Arena(name="ledge", step_offset=1, step_height=2)

#: A walled arena with a lava trench across it: the composite case.
ARENA_PIT = Arena(name="arena_pit", wall_offset=3, wall_height=4,
                  pit_half_width=1, pit_is_trench=True, pit_depth=3,
                  pit_fluid=True, fluid_damage=3)

ARENAS: dict[str, Arena] = {
    a.name: a for a in
    (FLAT, PIT, LAVA_POOL, PILLARS, CORRIDOR, LEDGE, ARENA_PIT)
}


def build(arena: str | Arena, *, batch_size: int = 1):
    """Build one arena into a batched `GeometryState`.

    `require_exact=False` is mandatory: these frames are synthetic and the
    loader rejects them as native-exact by design.
    """

    from hytalegym.jax.world.types import geometry_state_from_numpy

    if isinstance(arena, str):
        try:
            arena = ARENAS[arena]
        except KeyError:
            raise ValueError(
                f"unknown arena {arena!r}; have {sorted(ARENAS)}") from None
    return geometry_state_from_numpy(
        arena.build(), batch_size=batch_size, require_exact=False)
