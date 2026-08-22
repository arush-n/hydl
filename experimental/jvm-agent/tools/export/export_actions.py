"""Export the action surface: head layout, constants, and JAX-decoded cases.

The Java hook has to turn the policy head indices into a plugin `AgentAction`.
That mapping currently lives in two Python stages -- `decode_staged_action_surface_factors`
(surface.py) turns factors into a semantic `LearnerArsenalAction`, then the native
translate step (transport.py) turns that into the wire dict `AgentAction.fromMap`
reads. In-process both stages collapse into one function, so this exports enough
to certify that function: the layout, and a set of decoded reference cases
covering legal, out-of-range, mask-illegal and arbitration-violating factors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
OUT = JVM_AGENT_ROOT / "case"


def main() -> int:
    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_BLOCK_TRIGGER_NONE,
        ARSENAL_BLOCK_TRIGGER_PRIMARY,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES,
        ARSENAL_POLICY_COMPASS_DIRECTIONS,
        ARSENAL_POLICY_LOCOMOTION_DODGE_START,
        ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS,
        DODGE_ACTION_COUNT,
        look_delta_values,
    )
    from hytalegym.jax.combat.observation.v3.policy.surface import (
        action_surface_layout,
        decode_staged_action_surface_factors,
    )
    from hytalegym.jax.combat.types import (
        ACTION_ATTACK,
        ACTION_BACK,
        ACTION_FORWARD,
        ACTION_JUMP,
        ACTION_LEFT,
        ACTION_PITCH_DELTA,
        ACTION_RIGHT,
        ACTION_SIZE,
        ACTION_YAW_DELTA,
    )
    # These moved up to the parent package in the Gym restructure --
    # `hytalegym.jax.combat.observation.contract` no longer exists, and this
    # import was a hard blocker on regenerating the fixture against the live
    # Gym. All three names are re-exported from the parent unchanged.
    from hytalegym.jax.combat.observation import (
        ACTION_DOOR_OPEN,
        DOOR_INTENT_COUNT,
        LEARNER_ACTION_COUNT,
    )
    from hytalegym.jax.combat import skills as skill_module
    from hytalegym.jax.combat.skills import (
        SKILL_APPROACH_ATTACK,
        SKILL_ATTACK,
        SKILL_COUNT,
        SKILL_IDLE,
        SKILL_RETREAT_ATTACK,
    )

    layout = action_surface_layout(block_candidate_capacity=16)
    total = int(sum(layout.head_sizes))

    manifest = {
        "head_names": list(layout.head_names),
        "head_sizes": [int(v) for v in layout.head_sizes],
        "head_offsets": {k: int(v) for k, v in layout.head_offsets.items()},
        "total_logits": total,
        "public_head_sizes": [int(v) for v in ARSENAL_POLICY_ACTION_HEAD_SIZES],
        "base_action_head_sizes": [
            int(v) for v in ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES
        ],
        "planned_non_target_heads": [
            [str(name), int(size)]
            for name, size in ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS
        ],
        "block_trigger_none": int(ARSENAL_BLOCK_TRIGGER_NONE),
        "block_trigger_primary": int(ARSENAL_BLOCK_TRIGGER_PRIMARY),
        "skill_attack": int(SKILL_ATTACK),
        "skill_approach_attack": int(SKILL_APPROACH_ATTACK),
        "skill_retreat_attack": int(SKILL_RETREAT_ATTACK),
        "yaw_neutral": int(layout.yaw_neutral),
        "pitch_neutral": int(layout.pitch_neutral),
        "block_candidate_capacity": int(layout.block_candidate_capacity),
        "skills": {
            name: int(getattr(skill_module, name))
            for name in dir(skill_module)
            if name.startswith("SKILL_")
        },
        "action_indices": {
            "forward": int(ACTION_FORWARD),
            "back": int(ACTION_BACK),
            "left": int(ACTION_LEFT),
            "right": int(ACTION_RIGHT),
            "jump": int(ACTION_JUMP),
            "attack": int(ACTION_ATTACK),
            "yaw_delta": int(ACTION_YAW_DELTA),
            "pitch_delta": int(ACTION_PITCH_DELTA),
            "size": int(ACTION_SIZE),
        },
        "skill_count": int(SKILL_COUNT),
        "action_door_open": int(ACTION_DOOR_OPEN),
        "door_intent_count": int(DOOR_INTENT_COUNT),
        "learner_action_count": int(LEARNER_ACTION_COUNT),
        "maximum_turn_degrees": 45.0,
        "look_delta_encoding": {
            "strategy": "symmetric_geometric_coarse_to_fine_v1",
            "yaw_degrees": list(look_delta_values(9)),
            "pitch_degrees": list(look_delta_values(5)),
        },
    }

    index = {name: i for i, name in enumerate(layout.head_names)}
    heads = len(layout.head_sizes)

    def factors(**overrides) -> list[int]:
        row = [0] * heads
        row[index["yaw_delta_bins"]] = int(layout.yaw_neutral)
        row[index["pitch_delta_bins"]] = int(layout.pitch_neutral)
        for key, value in overrides.items():
            row[index[key]] = int(value)
        return row

    cases: list[dict] = [
        {"name": "neutral", "factors": factors()},
        # every base_action value, so the skill -> movement/attack map is
        # certified for all 9 skills plus the 3 door-intent slots
        *(
            {"name": f"base_action_{value}", "factors": factors(base_action=value)}
            for value in range(int(layout.head_size("base_action")))
        ),
        {"name": "yaw_max", "factors": factors(yaw_delta_bins=8)},
        {"name": "yaw_min", "factors": factors(yaw_delta_bins=0)},
        {"name": "yaw_mid", "factors": factors(yaw_delta_bins=6)},
        {"name": "yaw_out_of_range", "factors": factors(yaw_delta_bins=9)},
        {"name": "pitch_max", "factors": factors(pitch_delta_bins=4)},
        {"name": "pitch_min", "factors": factors(pitch_delta_bins=0)},
        {"name": "pitch_mid", "factors": factors(pitch_delta_bins=3)},
        {"name": "guard", "factors": factors(guard_off_on=1)},
        {"name": "jump", "factors": factors(jump_off_on=1)},
        # Dodge and world movement share `locomotion_gait_compass`: choice 0 is
        # idle, 1..8 walk a compass direction, and the four dodges sit at the
        # tail. The case names keep their old numbering so the Java side still
        # certifies the same semantic actions.
        *(
            {
                "name": f"dodge_{value}",
                "factors": factors(
                    locomotion_gait_compass=(
                        0
                        if value == 0
                        else ARSENAL_POLICY_LOCOMOTION_DODGE_START + value - 1
                    )
                ),
            }
            for value in range(DODGE_ACTION_COUNT + 1)
        ),
        {"name": "ability_0", "factors": factors(ability_none_plus_slots=1)},
        {"name": "ability_7", "factors": factors(ability_none_plus_slots=8)},
        {"name": "ability_15", "factors": factors(ability_none_plus_slots=16)},
        {"name": "use", "factors": factors(use_off_on=1)},
        *(
            {
                "name": f"world_move_{value}",
                "factors": factors(locomotion_gait_compass=value),
            }
            for value in range(ARSENAL_POLICY_COMPASS_DIRECTIONS + 1)
        ),
        {"name": "block_candidate_3", "factors": factors(block_none_plus_candidates=4)},
        # Block *placement* as distinct from breaking. The trigger head is the
        # only thing separating them and it was pinned at 0 in every case above,
        # so the secondary path -- place rather than break -- decoded entirely
        # untested. Cover both triggers against a selected candidate, the first
        # and last candidate slots, and a trigger with no candidate at all.
        *(
            {
                "name": f"block_trigger_{trigger}_candidate_{candidate}",
                "factors": factors(
                    block_primary_secondary_trigger=trigger,
                    block_none_plus_candidates=candidate,
                ),
            }
            for trigger in range(
                int(layout.head_size("block_primary_secondary_trigger"))
            )
            for candidate in (0, 1, 16)
        ),
        # Candidate slot boundaries on their own head, since a 17-wide head was
        # only ever asked for slot 3.
        {"name": "block_candidate_first", "factors": factors(
            block_none_plus_candidates=1)},
        {"name": "block_candidate_last", "factors": factors(
            block_none_plus_candidates=16)},
        # The recipe candidate cases and the block/recipe arbitration case are
        # gone with `recipe_none_plus_candidates`: crafting left the policy
        # surface, so there is no longer a second world-verb root to conflict
        # against. Block/use below is now the only world-verb arbitration case.
        {
            "name": "conflict_block_use",
            "factors": factors(block_none_plus_candidates=4, use_off_on=1),
        },
        # arbitration: two standard-input roots at once must be rejected
        {
            "name": "conflict_attack_guard",
            "factors": factors(base_action=int(SKILL_ATTACK), guard_off_on=1),
        },
        {
            "name": "conflict_use_ability",
            "factors": factors(use_off_on=1, ability_none_plus_slots=1),
        },
        # out of range on a single head must reject the whole action
        {"name": "out_of_range", "factors": factors(base_action=99)},
        {"name": "negative", "factors": factors(locomotion_gait_compass=-1)},
        # movement combined with an interaction is legal (movement is not a root)
        {
            "name": "attack_while_moving",
            "factors": factors(base_action=int(SKILL_ATTACK), locomotion_gait_compass=5),
        },
    ]

    records = []
    for case in cases:
        row = jnp.asarray([case["factors"]], dtype=jnp.int32)
        mask = jnp.ones((1, total), dtype=jnp.bool_)
        decoded = decode_staged_action_surface_factors(
            row, mask, layout=layout, maximum_turn_degrees=45.0
        )
        base = decoded.base
        records.append(
            {
                "name": case["name"],
                "factors": case["factors"],
                "action_legal": bool(np.asarray(decoded.action_legal)[0]),
                "skill_id": int(np.asarray(base.skill_id)[0]),
                "ability_slot": int(np.asarray(base.ability_slot)[0]),
                "guard_held": bool(np.asarray(base.guard_held)[0]),
                "dodge_direction": int(np.asarray(base.dodge_direction)[0]),
                "jump_held": bool(np.asarray(base.jump_held)[0]),
                "world_move_direction": int(np.asarray(base.world_move_direction)[0]),
                "yaw_delta_degrees": float(np.asarray(base.yaw_delta_degrees)[0]),
                "pitch_delta_degrees": float(np.asarray(base.pitch_delta_degrees)[0]),
                "use_requested": bool(np.asarray(base.use_requested)[0]),
                "block_interaction_trigger": int(
                    np.asarray(base.block_interaction_trigger)[0]
                ),
                "recipe_candidate_index": int(
                    np.asarray(base.recipe_candidate_index)[0]
                ),
                "block_candidate_index": int(np.asarray(base.block_candidate_index)[0]),
            }
        )

    # A mask that forbids everything except the neutral index of each head, to
    # prove the Java decoder honours mask illegality and not only range.
    masked = []
    forbid = np.ones((1, total), dtype=bool)
    base_off = layout.head_offsets["base_action"]
    forbid[0, base_off + int(SKILL_ATTACK)] = False
    for case, row_values in (
        ("attack", factors(base_action=int(SKILL_ATTACK))),
        ("neutral", factors()),
    ):
        row = jnp.asarray([row_values], dtype=jnp.int32)
        decoded = decode_staged_action_surface_factors(
            row, jnp.asarray(forbid), layout=layout, maximum_turn_degrees=45.0
        )
        masked.append(
            {
                "name": f"masked_{case}",
                "factors": [int(v) for v in np.asarray(row)[0]],
                "action_legal": bool(np.asarray(decoded.action_legal)[0]),
                "skill_id": int(np.asarray(decoded.base.skill_id)[0]),
            }
        )

    # The Java port treats the movement booleans as a pure function of skill_id.
    # That is only true because skills_to_actions reads the observation solely
    # for turn_delta, which decode_learner_arsenal_action then overwrites with
    # the explicit yaw head. Verify it instead of assuming: drive the same skill
    # through several very different observations and require the movement and
    # attack columns to be bit-identical.
    from hytalegym.jax.combat.skills import skills_to_actions

    width = 64
    probes = [
        np.zeros((1, width), dtype=np.float32),
        np.ones((1, width), dtype=np.float32),
        np.full((1, width), -1.0, dtype=np.float32),
        np.linspace(-5.0, 5.0, width, dtype=np.float32)[None, :],
    ]
    movement_columns = [
        int(ACTION_FORWARD), int(ACTION_BACK),
        int(ACTION_LEFT), int(ACTION_RIGHT), int(ACTION_ATTACK),
    ]
    skill_movement = {}
    independent = True
    for skill in range(int(SKILL_COUNT)):
        rows = []
        for probe in probes:
            out = np.asarray(
                skills_to_actions(
                    jnp.asarray(probe), jnp.asarray([skill], dtype=jnp.int32)
                )
            )[0]
            rows.append([float(out[c]) for c in movement_columns])
        if any(row != rows[0] for row in rows[1:]):
            independent = False
        skill_movement[str(skill)] = {
            "forward": bool(rows[0][0] > 0.0),
            "back": bool(rows[0][1] > 0.0),
            "left": bool(rows[0][2] > 0.0),
            "right": bool(rows[0][3] > 0.0),
            "attack": bool(rows[0][4] > 0.0),
        }

    # Second stage: run the real native translate so the exported expectation is
    # the actual wire dict `AgentAction.fromMap` reads, not my reading of it.
    # Block and recipe selections are privileged -- they need a resolved
    # NativeWorldVerbRequest that only the server can produce -- so those cases
    # are expected to fail closed here, which is itself worth pinning.
    # Also moved by the restructure: native/transport.py is now
    # native/codec/transport.py. The symbols are unchanged.
    from hytalegym.jax.combat.observation.v3.native.codec.transport import (
        NATIVE_POLICY_ACTION_PROTOCOL_VERSION,
        translate_learner_arsenal_action_to_native,
    )
    from hytalegym.jax.combat.observation.v3.schema.types import LearnerArsenalAction

    backend_info = {
        "native_policy_combat_protocol_version": (
            NATIVE_POLICY_ACTION_PROTOCOL_VERSION
        ),
        "supported_actions": ",".join(
            (
                "world_move_direction", "guard_held", "dodge_direction",
                "ability_slot", "use", "place_block", "break_block",
                "craft_recipe",
            )
        ),
    }

    for record in records:
        row = record["factors"]
        legal = record["action_legal"]
        skill = record["skill_id"]
        movement_skill = (
            skill if (legal and 0 <= skill < int(SKILL_COUNT)) else int(SKILL_IDLE)
        )
        low = np.asarray(
            skills_to_actions(
                jnp.zeros((1, width), dtype=jnp.float32),
                jnp.asarray([movement_skill], dtype=jnp.int32),
            ),
            dtype=np.float32,
        ).copy()
        low[0, int(ACTION_JUMP)] = 1.0 if record["jump_held"] else 0.0
        low[0, int(ACTION_YAW_DELTA)] = record["yaw_delta_degrees"]
        low[0, int(ACTION_PITCH_DELTA)] = record["pitch_delta_degrees"]

        action = LearnerArsenalAction(
            skill_id=jnp.asarray([record["skill_id"]], dtype=jnp.int32),
            ability_slot=jnp.asarray([record["ability_slot"]], dtype=jnp.int32),
            guard_held=jnp.asarray([record["guard_held"]], dtype=jnp.bool_),
            dodge_direction=jnp.asarray(
                [record["dodge_direction"]], dtype=jnp.int32
            ),
            jump_held=jnp.asarray([record["jump_held"]], dtype=jnp.bool_),
            world_move_direction=jnp.asarray(
                [record["world_move_direction"]], dtype=jnp.int32
            ),
            yaw_delta_degrees=jnp.asarray(
                [record["yaw_delta_degrees"]], dtype=jnp.float32
            ),
            pitch_delta_degrees=jnp.asarray(
                [record["pitch_delta_degrees"]], dtype=jnp.float32
            ),
            use_requested=jnp.asarray([record["use_requested"]], dtype=jnp.bool_),
            block_interaction_trigger=jnp.asarray(
                [record["block_interaction_trigger"]], dtype=jnp.int32
            ),
            recipe_candidate_index=jnp.asarray(
                [record["recipe_candidate_index"]], dtype=jnp.int32
            ),
            block_candidate_index=jnp.asarray(
                [record["block_candidate_index"]], dtype=jnp.int32
            ),
        )
        translated = translate_learner_arsenal_action_to_native(
            action, low, backend_info, policy_legal=legal
        )
        record["native"] = {
            key: (int(value) if isinstance(value, bool) else value)
            for key, value in translated.action.items()
        }
        record["native_legal"] = bool(translated.legal)
        record["native_reject_reasons"] = list(translated.reject_reasons)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "cases": records,
        "masked_cases": masked,
        "masked_forbidden_index": int(base_off + int(SKILL_ATTACK)),
        "skill_movement": skill_movement,
        "skill_movement_observation_independent": independent,
    }
    print(
        "skill movement independent of observation: "
        f"{independent}  (over {len(probes)} probe observations)"
    )
    for skill, row in skill_movement.items():
        flags = " ".join(k for k, v in row.items() if v) or "-"
        print(f"  skill {skill}: {flags}")
    path = OUT / "actions.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Flat mirror of the same expectations. The Java side is deliberately
    # dependency-free, so the test parses tab-separated text rather than pulling
    # in a JSON library purely to read its own fixtures.
    lines = [
        "# name\tfactors\tlegal\tskill\tability\tguard\tdodge\tjump\tmove"
        "\tyaw\tpitch\tuse\tbtrig\trecipe\tblock"
        "\tnative_legal\tnf\tnb\tnl\tnr\tnattack\tnjump"
    ]
    for record in records:
        native = record["native"]
        lines.append(
            "\t".join(
                str(field)
                for field in (
                    record["name"],
                    ",".join(str(v) for v in record["factors"]),
                    int(record["action_legal"]),
                    record["skill_id"],
                    record["ability_slot"],
                    int(record["guard_held"]),
                    record["dodge_direction"],
                    int(record["jump_held"]),
                    record["world_move_direction"],
                    repr(record["yaw_delta_degrees"]),
                    repr(record["pitch_delta_degrees"]),
                    int(record["use_requested"]),
                    record["block_interaction_trigger"],
                    record["recipe_candidate_index"],
                    record["block_candidate_index"],
                    int(record["native_legal"]),
                    native["forward"],
                    native["back"],
                    native["left"],
                    native["right"],
                    native["attack"],
                    native["jump"],
                )
            )
        )
    flat = OUT / "actions.txt"
    flat.write_text("\n".join(lines) + "\n", encoding="utf-8")

    masked_lines = [f"# forbidden_logit\t{base_off + int(SKILL_ATTACK)}"]
    for record in masked:
        masked_lines.append(
            "\t".join(
                (
                    record["name"],
                    ",".join(str(v) for v in record["factors"]),
                    str(int(record["action_legal"])),
                    str(record["skill_id"]),
                )
            )
        )
    (OUT / "actions_masked.txt").write_text(
        "\n".join(masked_lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {flat} and {OUT / 'actions_masked.txt'}")

    print(f"heads       : {heads}")
    print(f"head sizes  : {list(layout.head_sizes)}")
    print(f"total logits: {total}")
    print(f"names       : {list(layout.head_names)}")
    print(f"yaw neutral : {layout.yaw_neutral}  pitch neutral: {layout.pitch_neutral}")
    print(f"wrote {path} ({len(records)} cases, {len(masked)} masked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
