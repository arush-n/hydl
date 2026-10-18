"""Bench crafting through the Region crafting action. Switch ``Constructs.bench_crafting``; row 11, stage 0.

Two engine limits keep the Region crafting action from what 0.7.2 offers at an ordinary bench:

- the commit: ``combat/arsenal/world_actions/crafting.execute_region_fieldcraft`` commits every candidate through
  ``crafting/inventory_adapter.craft_into_combat_inventory`` under the Fieldcraft context (bench type Crafting, id
  Fieldcraft, tier 0), and ``crafting/runtime._recipe_precondition`` requires the context to match one of the
  recipe's bench requirements. So a recipe the candidate provider offers because an ordinary bench is in reach
  (``_bench_contexts``) is refused at the commit: of the 0.7.2 catalog's 197 offered recipes, 2 commit (gen 5: 2 of
  154; FIDELITY-0.7.2.md (l), row 11).
- the cap: ``world/interactions/crafting.produce_actor_recipe_candidates_from_legal_mask`` publishes at most 16
  candidates (``ACTOR_RECIPE_CANDIDATE_CAPACITY``) and fails the whole actor row closed when more are legal, while the
  native bench window lists every legal recipe.

Gen 7 (``recipe_bench``) hooks both:

- ``bench_commit`` wraps ``craft_into_combat_inventory``: a lane whose context satisfies none of its recipe's bench
  requirements commits under the recipe's first requirement (bench type, id and tier). The provider offered the
  candidate because an in-reach bench meets one of those requirements, and the executor rechecks the candidate
  before the commit. A recipe the Fieldcraft context already satisfies keeps that context, so every commit that
  succeeds in gen 5 is unchanged.
- ``first_candidates`` wraps the candidate producer: when more than 16 recipes are legal it keeps the first 16 in
  table order (the packer's canonical recipe-id order), and the row stays available. ``source_count`` keeps the full
  legal count, and ``diagnostics`` carries ``ACTOR_RECIPE_DIAGNOSTIC_CAPACITY``, which in gen 7 means truncated
  rather than failed. ``capacity_exceeded`` stays False, because the policy staging
  (``observation/v3/policy/candidates/recipes.py``) drops a row that sets it. Shapes, the candidate radix and the
  contract stamps are unchanged; the candidate contract's ``"overflow": "fail_entire_actor_row_closed"`` text is gen
  5's rule.

Gen 5 (``fieldcraft_context``) returns both originals; its commit defect stays (recorded in BASELINES.md). Later
stages, not here: resource-type inputs with output exclusions, memories, bench tiers above 0, timed recipes, bench
types 1-3.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "bench_crafting"
IMPLEMENTED = True
FIELDCRAFT_CONTEXT = "fieldcraft_context"
RECIPE_BENCH = "recipe_bench"


def _enabled(profile: Any) -> bool:
    value = profile.constructs.bench_crafting
    if value == FIELDCRAFT_CONTEXT:
        return False
    if value != RECIPE_BENCH:
        raise ValueError(f"unknown bench_crafting switch {value!r}")
    return True


def bench_commit(original: Any, profile: Any) -> Any:
    """Hook for ``craft_into_combat_inventory``: commit under a bench the recipe names."""

    if not _enabled(profile):
        return original
    jnp = original.__globals__["jnp"]

    @wraps(original)
    def craft_into_combat_inventory(table, state, layout, context, recipe_index, catalog, **kwargs):
        index = jnp.clip(jnp.asarray(recipe_index, dtype=jnp.int32), 0, table.requirement_mask.shape[0] - 1)
        required = table.requirement_mask[index]
        types = table.requirement_bench_type[index]
        hashes = table.requirement_bench_id_hash[index]
        tiers = table.requirement_tier_level[index]
        satisfied = jnp.any(
            required
            & (types == context.bench_type[..., None])
            & jnp.all(hashes == context.bench_id_hash[..., None, :], axis=-1)
            & (tiers <= context.bench_tier_level[..., None]),
            axis=-1,
        )
        rebind = jnp.any(required, axis=-1) & ~satisfied
        first = jnp.argmax(required, axis=-1)[..., None]
        context = context._replace(
            bench_type=jnp.where(rebind, jnp.take_along_axis(types, first, axis=-1)[..., 0], context.bench_type),
            bench_id_hash=jnp.where(
                rebind[..., None],
                jnp.take_along_axis(hashes, first[..., None], axis=-2)[..., 0, :],
                context.bench_id_hash,
            ),
            bench_tier_level=jnp.where(
                rebind, jnp.take_along_axis(tiers, first, axis=-1)[..., 0], context.bench_tier_level
            ),
        )
        return original(table, state, layout, context, recipe_index, catalog, **kwargs)

    return craft_into_combat_inventory


def first_candidates(original: Any, profile: Any) -> Any:
    """Hook for ``produce_actor_recipe_candidates_from_legal_mask``: the first 16 legal recipes, not none."""

    if not _enabled(profile):
        return original
    env = original.__globals__
    jnp = env["jnp"]
    capacity = env["ACTOR_RECIPE_CANDIDATE_CAPACITY"]
    truncated_bit = env["ACTOR_RECIPE_DIAGNOSTIC_CAPACITY"]

    @wraps(original)
    def produce_actor_recipe_candidates_from_legal_mask(table, legal_recipe_mask, **kwargs):
        legal = jnp.asarray(legal_recipe_mask)
        visible = legal & jnp.asarray(table.recipe_mask, dtype=jnp.bool_)[None, None, :]
        # Unknown references stay in the mask, so the original still fails such a row closed.
        kept = legal & ~(visible & (jnp.cumsum(visible, axis=2) > capacity))
        result = original(table, kept, **kwargs)
        count = jnp.sum(visible, axis=2, dtype=jnp.int32)
        truncated = result.available & (count > capacity)
        return result._replace(
            source_count=jnp.where(truncated, count, result.source_count),
            diagnostics=result.diagnostics | jnp.where(truncated, truncated_bit, jnp.uint32(0)),
        )

    return produce_actor_recipe_candidates_from_legal_mask
