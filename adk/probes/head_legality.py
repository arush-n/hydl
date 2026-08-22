"""Measure non-neutral legality on the console's production Region path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


OUTPUT = Path(__file__).with_name("head_legality_v1.json")
POSITIVE_CONTROLS = (
    "guard_off_on",
    # Dodge folded into the merged locomotion head; there is no separate dodge
    # head to control against any more.
    "locomotion_gait_compass",
    "jump_off_on",
)
BLOCK_HEADS = (
    "block_none_plus_candidates",
    "block_primary_secondary_trigger",
)


def _closure_reason(name, diagnostics, surface):
    if name == "block_primary_secondary_trigger" and not surface[
        "item_root_evidence_bound"
    ]:
        return "item_root_evidence_unbound"
    if name == "block_none_plus_candidates":
        if diagnostics["block_capacity_exceeded"]:
            return "block_candidate_capacity_exceeded"
        if diagnostics["raw_block_candidate_seen"]:
            return "candidate_present_but_action_mask_closed"
        if diagnostics["geometry_exhausted"]:
            return "no_camera_candidate_before_geometry_exhaustion"
        return "no_camera_candidate_observed"
    if name == "use_off_on" and not surface["item_root_evidence_bound"]:
        return "item_root_evidence_unbound"
    return "no_non_neutral_choice_in_action_mask"


def _episode(loaded, opponent_provider, *, seed: int, ticks: int):
    import jax
    import numpy as np

    from worlds.region import region_providers
    from hytalegym.jax.combat import neutral_arsenal_policy_action_factors
    from hytalegym.jax.combat.observation.v3.policy.surface import (
        action_surface_layout,
    )
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment,
        make_region_runtime_camera_block_candidate_provider,
    )

    environment = make_arsenal_ppo_environment(
        loaded.params,
        loaded.runtime,
        opponent_ability_provider=opponent_provider,
        **region_providers(loaded),
    )
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    state, observation, mask = environment.reset(jax.random.split(reset_key, 1))
    layout = action_surface_layout(16)
    first_tick = {name: None for name in layout.head_names}
    raw_provider = make_region_runtime_camera_block_candidate_provider(
        maximum_interaction_distance=loaded.maximum_distance
    )
    diagnostics = {
        "block_capacity_exceeded": False,
        "raw_block_candidate_seen": False,
        "geometry_exhausted": False,
        "incoming_attack_requested": False,
        "incoming_attack_accepted": False,
        "agent_damage_received": 0.0,
        "transitions": 0,
    }

    for tick in range(ticks):
        for name, row in layout.split_mask(mask).items():
            legal = np.asarray(row[0], dtype=np.bool_).copy()
            legal[layout.neutral_index(name)] = False
            if first_tick[name] is None and bool(legal.any()):
                first_tick[name] = tick

        raw, _ = raw_provider(
            state.runtime,
            state.action_surface_runtime.world,
            loaded.params,
        )
        diagnostics["block_capacity_exceeded"] |= bool(
            np.asarray(raw.capacity_exceeded).any()
        )
        diagnostics["raw_block_candidate_seen"] |= bool(
            np.asarray(raw.candidate_mask).any()
        )

        key, step_key = jax.random.split(key)
        state, observation, _, done, mask, info = environment.step_detailed(
            state,
            observation,
            neutral_arsenal_policy_action_factors(1),
            jax.random.split(step_key, 1),
        )
        diagnostics["transitions"] += 1
        diagnostics["geometry_exhausted"] |= bool(
            np.asarray(info.combat_info.geometry_exhausted).any()
        )
        requested = np.asarray(info.arsenal_info.ability_requested)
        accepted = np.asarray(info.arsenal_info.ability_accepted)
        diagnostics["incoming_attack_requested"] |= bool(requested[..., 1].any())
        diagnostics["incoming_attack_accepted"] |= bool(accepted[..., 1].any())
        damage = np.asarray(info.arsenal_info.entity_damage_received)
        diagnostics["agent_damage_received"] += float(damage[..., 0].sum())
        if bool(np.asarray(done).all()):
            break

    surface = loaded.fixture.metadata["action_surface"]
    return {
        "incoming_attack_observed": diagnostics["incoming_attack_accepted"],
        "heads": {
            name: {
                "ever_legal": first_tick[name] is not None,
                "first_tick": first_tick[name],
                "closure_reason": (
                    None
                    if first_tick[name] is not None
                    else _closure_reason(name, diagnostics, surface)
                ),
            }
            for name in layout.head_names
        },
        "diagnostics": diagnostics,
    }


def run_probe(*, seed: int = 570057, ticks: int = 96):
    from worlds.region import load_region
    from hytalegym.jax.combat.opponents.runtime.policy import (
        first_legal_opponent_ability_slots,
        inert_opponent_ability_slots,
    )

    loaded = load_region(
        weapons=("iron_sword",),
        target_weapons=("iron_sword",),
        native_evidence=False,
    )
    surface = dict(loaded.fixture.metadata["action_surface"])
    if not surface["candidate_producer_bound"]:
        raise RuntimeError("Region block candidate provider is not bound")

    environments = {
        "console_region/inert_opponent": _episode(
            loaded, inert_opponent_ability_slots, seed=seed, ticks=ticks
        ),
        "console_region/armed_opponent": _episode(
            loaded, first_legal_opponent_ability_slots, seed=seed, ticks=ticks
        ),
    }
    positive_control = {
        name: all(arm["heads"][name]["ever_legal"] for arm in environments.values())
        for name in POSITIVE_CONTROLS
    }
    if not all(positive_control.values()):
        raise RuntimeError(f"broken legality probe; positive control: {positive_control}")

    block_heads_open = all(
        arm["heads"][name]["ever_legal"]
        for arm in environments.values()
        for name in BLOCK_HEADS
    )
    incoming_arm_complete = environments["console_region/armed_opponent"][
        "incoming_attack_observed"
    ]
    return {
        "schema": "head-legality-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_ticks_per_episode": ticks,
        "action_mask_source": "PPOEnvironment reset/step action_mask",
        "provider": surface,
        "environments": environments,
        "positive_control": positive_control,
        "positive_control_passed": True,
        "block_heads_open": block_heads_open,
        "incoming_attack_arm_complete": incoming_arm_complete,
        "stop_required": not block_heads_open,
        "status": (
            "blocked_by_closed_block_heads"
            if not block_heads_open
            else "legality_probe_complete"
        ),
        "limitations": [
            "The production Region episode truncated on geometry exhaustion "
            "before the armed target produced an incoming attack."
        ]
        if not incoming_arm_complete
        else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--ticks", type=int, default=96)
    args = parser.parse_args()
    report = run_probe(seed=args.seed, ticks=args.ticks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "block_heads_open": report["block_heads_open"],
            }
        )
    )


if __name__ == "__main__":
    main()
