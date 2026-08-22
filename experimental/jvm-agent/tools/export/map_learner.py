"""Break `learner_observation` down field by field.

The top-level map says it owns 6,174 of 8,271 columns across 15 spans. To port
the encoder incrementally, each field needs its own span set so a Java block
can be written and diffed in isolation.

Same technique as `map_layout.py`: fill one field with a sentinel, see which
columns move. Fields whose contribution is gated behind a mask will report zero
columns -- that is information too, and it is reported rather than hidden.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from map_layout import perturbed, runs

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
OUT = JVM_AGENT_ROOT / "case" / "learner_layout.json"
PROFILE, SEED = "iron_sword", 570057


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

    learner = state.learner_observation
    inventory = inventory_policy_tokens_from_state(
        state.runtime.inventory, runtime.inventory_layout,
        actor_valid=learner.valid,
    )

    def compose(observation_value):
        return np.asarray(arsenal_policy_observation(
            observation_value,
            state.action_surface.block_candidates,
            state.action_surface.recipe_encoding,
            inventory_tokens=inventory,
            light_tokens=state.light_tokens,
        ))[0]

    assert np.array_equal(compose(learner), reference), "recomposition mismatch"
    fields = list(learner._fields)
    print(f"LearnerCombatObservationV3 has {len(fields)} fields\n")
    print(f"{'field':34s} {'cols':>6}  {'spans':>5}  first spans")
    print("-" * 92)

    layout, claimed = {}, np.zeros(reference.size, dtype=bool)
    for name in fields:
        value = getattr(learner, name)
        try:
            swapped = learner._replace(**{name: perturbed(value)})
            moved = np.flatnonzero(compose(swapped) != reference)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:34s}  {'n/a':>5}  perturb failed: {type(exc).__name__}")
            continue
        spans = runs(moved.tolist())
        claimed[moved] = True
        layout[name] = {
            "columns": int(moved.size),
            "spans": [[int(a), int(b)] for a, b in spans],
        }
        preview = ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in spans[:4])
        print(f"{name:34s} {moved.size:6d}  {len(spans):5d}  {preview}"
              + (" …" if len(spans) > 4 else ""))

    total = sum(entry["columns"] for entry in layout.values())
    print("-" * 92)
    print(f"{'sum of fields':34s} {total:6d}   (learner block owns 6174)")
    silent = [name for name, entry in layout.items() if entry["columns"] == 0]
    if silent:
        print(f"\ncontributed nothing (masked off in this scene): {silent}")

    layout["_total_columns"] = total
    OUT.write_text(json.dumps(layout, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
