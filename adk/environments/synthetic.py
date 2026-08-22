"""Artificial arenas: terrain declared by a function of position.

The captured region library cannot supply these. Its manifest declares
`authority: exact_frozen_native_artifact` and
`runtime_source: artifact_only_no_seed_regeneration` -- it is frozen native
captures by contract, so writing synthetic terrain into it would forge native
provenance, which is exactly the failure this repo keeps paying for. Nothing
here touches that library.

Instead an artificial world is a **world capability provider**, the same seam
`open_flat` and `fail_closed` use (`SceneConfig.world`). A provider is

    (state, params, *, config=None) -> ArsenalWorldCapabilities

so terrain here is a *function of entity position*, evaluated per tick, rather
than a voxel grid. That is the honest shape of this seam: it can express
"is there ground under the agent", "is the agent in fluid", "is the dodge
corridor clear" -- which covers gaps, pillars and hazard fields -- and it
cannot express anything the capability struct has no field for.

Provenance: scenes built this way are synthetic and say so. Do not compare a
number from one against a Region measurement and call it terrain fidelity.

    from adk.environments import SceneConfig
    from adk.environments.synthetic import PARKOUR

    scene = SceneConfig(world=PARKOUR).build()
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

__all__ = ["SyntheticWorld", "PARKOUR", "LAVA", "ISLANDS", "PRESETS"]

#: Entity 0 is the agent; entity 1 is the opponent.
AGENT_ENTITY = 0


@dataclass(frozen=True)
class SyntheticWorld:
    """A parametric arena. Every field is a knob; defaults are a flat clearing.

    `gap_period` and `gap_width` cut trenches: ground is absent where the
    horizontal coordinate falls inside a gap, so `actor_drop_support_found` goes
    false and `actor_drop_height` reports `fall_depth`. `hazard_period` and
    `hazard_width` do the same for fluid, setting the submersion flags a lava or
    water field would.

    Set a period to 0.0 to disable that feature entirely -- the default world is
    then indistinguishable from `open_flat` except that it is labelled
    synthetic.
    """

    name: str = "synthetic_flat"
    #: Distance between gap centres, in world units. 0.0 disables gaps.
    gap_period: float = 0.0
    #: Width of each gap. Must be < gap_period or the world is all gap.
    gap_width: float = 0.0
    #: Reported drop height over a gap.
    #:
    #: **Saturates. Measured 2026-08-09** (`adk/tests/test_synthetic.py`): this
    #: reaches the policy as `min(fall_depth / 3.0, 1.0)` --
    #: 0.5->0.17, 1.5->0.50, 3.0->1.00, 3.5->1.00, 12.0->1.00, 40.0->1.00.
    #: This default (8.0) and both gap presets (12.0, 10.0) are **above the
    #: saturation point**, so as shipped the policy cannot tell them apart and
    #: the knob is effectively a boolean. Keep it under 3.0 if you want depth to
    #: be a gradient the agent can learn against.
    fall_depth: float = 8.0
    #: Distance between hazard bands. 0.0 disables hazard.
    hazard_period: float = 0.0
    #: Width of each hazard band.
    hazard_width: float = 0.0
    #: Whether hazard submerges the eyes as well as the feet.
    hazard_deep: bool = True
    #: Narrow the dodge corridor over gaps, so dodging off a ledge is illegal.
    gaps_block_dodge: bool = True
    #: Shift the band pattern along x, in world units.
    #:
    #: Load-bearing, not cosmetic. Every env spawns at **x=0.5** and all rows
    #: share it, so where the band falls relative to 0.5 decides whether the
    #: arena expresses anything at all.
    #:
    #: **Do not place a band edge exactly on spawn.** `_banded` uses a strict
    #: `<`, so an offset equal to `width / 2` is neither in nor out. The first
    #: three presets did exactly that -- margin **0.000** for all of them -- and
    #: the consequence was a probe that reported *either* 0 of 8271 observation
    #: columns *or* 40, from the same code, decided only by whether sampled
    #: actions nudged x off 0.500. Neutral actions pinned x at exactly 0.500 and
    #: saw nothing. Phases now leave a 0.5-unit margin, so spawn is
    #: unambiguously outside the band and the feature begins 0.5 units ahead.
    phase: float = 0.0

    def tuned(self, **kwargs) -> "SyntheticWorld":
        """A copy with fields replaced. `SyntheticWorld` is frozen."""

        return replace(self, **kwargs)

    # -- terrain as a function of position ---------------------------------

    def _banded(self, position, period: float, width: float):
        """True where a horizontal coordinate falls inside a band.

        Bands run along z and repeat in x, which makes them crossable in one
        direction and jumpable in the other -- a course rather than a maze.
        """

        if not period or not width:
            return jnp.zeros(position.shape[:-1], dtype=jnp.bool_)
        shifted = position[..., 0] + self.phase
        offset = jnp.abs(jnp.mod(shifted, period) - period / 2.0)
        return offset < (width / 2.0)

    def provider(self):
        """Return the callable `SceneConfig(world=...)` wants.

        Composes over `open_flat`, which supplies correctly-shaped defaults
        for every field, then overrides only the ones this arena changes. That
        keeps combat itself reachable -- `fail_closed` denies LOS and target
        selection, which would make a terrain result unreadable for a reason
        that has nothing to do with terrain.

        The cost of that choice: `open_flat` pins every world query true, so a
        field this arena does *not* override reads permissive rather than
        realistic. Only the fields listed below are meaningful here.
        """

        from hytalegym.jax.combat.arsenal.environment import (
            open_flat_arsenal_world_capabilities,
        )

        def capabilities(state, params, *, config=None):
            # Start from the permissive control so combat itself stays
            # reachable; this arena is about terrain, not about denying LOS.
            world = open_flat_arsenal_world_capabilities(
                state, params, config=config)

            # Shapes measured from inside a live provider at batch 4, not
            # assumed: position is (B, entity, 3), every `actor_*` field is
            # (B,) -- agent only, no entity axis -- and dodge_corridor_clear is
            # (B, entity, direction). Guessing the entity axis here is how this
            # repo has previously filed a gap against a phantom.
            position = state.combat.position
            agent = position[:, AGENT_ENTITY]                    # (B, 3)
            over_gap = self._banded(agent, self.gap_period, self.gap_width)
            in_hazard = self._banded(
                agent, self.hazard_period, self.hazard_width)

            updates = {
                "actor_world_state_available": jnp.ones_like(
                    jnp.asarray(world.actor_world_state_available)),
                "actor_drop_support_found": (
                    jnp.asarray(world.actor_drop_support_found) & ~over_gap),
                "actor_drop_height": jnp.where(
                    over_gap,
                    jnp.float32(self.fall_depth),
                    jnp.asarray(world.actor_drop_height, dtype=jnp.float32),
                ),
            }
            if self.hazard_period and self.hazard_width:
                updates["actor_submersion_available"] = jnp.ones_like(
                    jnp.asarray(world.actor_submersion_available))
                updates["actor_controller_in_fluid"] = in_hazard
                updates["actor_feet_submerged"] = in_hazard
                if self.hazard_deep:
                    updates["actor_eyes_submerged"] = in_hazard
            if self.gaps_block_dodge and self.gap_period and self.gap_width:
                # Per-entity here, then broadcast over the direction axis.
                blocked = self._banded(
                    position, self.gap_period, self.gap_width)   # (B, entity)
                updates["dodge_corridor_clear"] = (
                    jnp.asarray(world.dodge_corridor_clear)
                    & ~blocked[..., None])
            return world._replace(**updates)

        capabilities.__name__ = f"{self.name}_world_capabilities"
        capabilities.synthetic = self
        return capabilities


#: Trenches every 6 units, 2 wide: the agent must cross gaps to close distance.
#: Pair with the `jump` minigame -- crossing is the behaviour it pays for.
PARKOUR = SyntheticWorld(
    name="parkour", gap_period=6.0, gap_width=2.0, fall_depth=12.0,
    phase=1.0)

#: Hazard bands every 8 units, 3 wide. Submersion drives the oxygen resource,
#: which no captured scene here has ever moved (README: "oxygen never moved,
#: because no scene built here contained fluid").
LAVA = SyntheticWorld(
    name="lava", hazard_period=8.0, hazard_width=3.0, hazard_deep=True,
    phase=1.5)

#: Both at once, offset periods so gaps and hazard interleave rather than align.
ISLANDS = SyntheticWorld(
    name="islands", gap_period=7.0, gap_width=2.0, fall_depth=10.0,
    hazard_period=7.0, hazard_width=1.0, hazard_deep=False, phase=1.5)

#: name -> world, for `SceneConfig(world=...)` and the console.
PRESETS: dict[str, SyntheticWorld] = {
    world.name: world for world in (PARKOUR, LAVA, ISLANDS)
}
