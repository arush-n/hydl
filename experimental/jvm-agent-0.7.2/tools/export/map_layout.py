"""Derive the observation layout empirically. There is no published offset table.

A Java assembler has to write each value at the right index, and
``arsenal_policy_observation`` composes five inputs with no documented
segmentation. This zeroes one input group at a time and records which columns
move, which yields the top-level map; then it does the same for the leaves of
the structured learner observation to break the largest block down further.

The map is the contract the Java port is written against, so it is derived from
the live code rather than read off a doc that could be stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
OUT = JVM_AGENT_ROOT / "case" / "layout.json"
PROFILE, SEED = "iron_sword", 570057


def runs(indices):
    """Collapse sorted indices into (start, end) inclusive spans."""
    spans, start, prev = [], None, None
    for index in indices:
        if start is None:
            start = prev = index
        elif index == prev + 1:
            prev = index
        else:
            spans.append((start, prev))
            start = prev = index
    if start is not None:
        spans.append((start, prev))
    return spans


def perturbed(value):
    """Fill a group with a sentinel so it is distinguishable from its neighbours.

    Zeroing cannot map a block that is *already* zero -- and in a bare combat
    scene the block, recipe and light groups are entirely zero, so zeroing them
    moves nothing and they would look like they own no columns. A non-zero
    sentinel maps them regardless of current content.
    """

    def fill(leaf):
        if leaf.dtype == jnp.bool_:
            return jnp.ones_like(leaf)
        if jnp.issubdtype(leaf.dtype, jnp.integer):
            return jnp.ones_like(leaf)
        return jnp.full_like(leaf, 0.5)

    return jax.tree_util.tree_map(fill, value)


def main() -> int:
    from hytalegym.jax.combat import (
        arsenal_runtime_config, default_combat_params, hytale_0_5_7_loadouts,
    )
    from hytalegym.jax.combat.observation.v3.policy import (
        arsenal_policy_observation,
    )
    from hytalegym.jax.combat.observation.v3.tokens.inventory import (
        inventory_policy_tokens_from_state,
    )
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment, open_flat_arsenal_world_capabilities,
    )
    from adk.validation import reset_keys

    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts((PROFILE,)))
    env = make_arsenal_ppo_environment(
        default_combat_params(microticks=1, target_active=True),
        runtime,
        world_capability_provider=open_flat_arsenal_world_capabilities,
    )
    state, observation, _ = env.reset(reset_keys(SEED, 1))
    reference = np.asarray(observation)[0]

    inventory = inventory_policy_tokens_from_state(
        state.runtime.inventory, runtime.inventory_layout,
        actor_valid=state.learner_observation.valid,
    )
    parts = {
        "learner_observation": state.learner_observation,
        "block_candidates": state.action_surface.block_candidates,
        "recipe_encoding": state.action_surface.recipe_encoding,
        "inventory_tokens": inventory,
        "light_tokens": state.light_tokens,
    }

    def compose(**override):
        merged = {**parts, **override}
        return np.asarray(arsenal_policy_observation(
            merged["learner_observation"],
            merged["block_candidates"],
            merged["recipe_encoding"],
            inventory_tokens=merged["inventory_tokens"],
            light_tokens=merged["light_tokens"],
        ))[0]

    assert np.array_equal(compose(), reference), "recomposition does not match env"
    print(f"recomposition matches env observation ({reference.size} columns)\n")

    layout, claimed = {}, np.zeros(reference.size, dtype=bool)
    for name in parts:
        moved = np.flatnonzero(compose(**{name: perturbed(parts[name])}) != reference)
        spans = runs(moved.tolist())
        layout[name] = {
            "columns": int(moved.size),
            "spans": [[int(a), int(b)] for a, b in spans],
        }
        claimed[moved] = True
        preview = ", ".join(f"{a}-{b}" for a, b in spans[:6])
        print(f"{name:22s} {moved.size:5d} cols  {len(spans):3d} span(s)  {preview}"
              + (" …" if len(spans) > 6 else ""))

    unclaimed = np.flatnonzero(~claimed)
    print(f"\nunclaimed (unmoved by any group): {unclaimed.size} columns")
    if unclaimed.size:
        spans = runs(unclaimed.tolist())
        print("  spans: " + ", ".join(f"{a}-{b}" for a, b in spans[:10])
              + (" …" if len(spans) > 10 else ""))
    layout["_unclaimed"] = {
        "columns": int(unclaimed.size),
        "spans": [[int(a), int(b)] for a, b in runs(unclaimed.tolist())],
    }
    layout["_total"] = int(reference.size)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(layout, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
