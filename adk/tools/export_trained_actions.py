"""Certify the Java action decoder against the actions a TRAINED policy emits.

`ActionTest` decodes the 59 vectors in `case/actions.txt`. Those are hand-chosen
to isolate one head at a time, plus seven arbitration conflicts. The trained
checkpoint's actual output is not among them -- it emits
`[2,1,0,4,1,5,8,1,0,0,0,0]`, which moves six heads at once. So "the decoder
agrees" was established for the fixture's vectors and *inferred* for the
policy's.

This closes that by replaying the same 52 parity observations `RecurrentTest`
uses, taking the trained policy's argmax per step, and emitting a decode
expectation for every distinct action vector it actually produces. The original
59 rows are carried through verbatim -- `ActionTest` requires every decoded
field to vary before it trusts a single comparison, and trained argmax output
is far too uniform to satisfy that on its own.

    python -m adk.tools.export_trained_actions <checkpoint.npz> <out-dir>
    java -cp "out;$SERVER_JAR" ActionTest <out-dir>

Reuses captured observations, so it never opens a native session and cannot
disturb the evidence lease on 127.0.0.1:5556.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
JVM_AGENT = _ROOT / "experimental" / "jvm-agent"
CASE = JVM_AGENT / "case"
PARITY = CASE / "parity"

WIDTH = 64  # `skills_to_actions` probe width, matching export_actions.py


def trained_action_vectors(checkpoint: Path) -> tuple[list[list[int]], int]:
    """Distinct argmax action vectors over the 52 parity observations."""

    import jax.numpy as jnp
    from hytalegym.jax.training.checkpoint import load_policy_checkpoint
    from hytalegym.jax.training.policy import apply_policy

    params, config, _ = load_policy_checkpoint(checkpoint)
    sizes = list(config.action_head_sizes)

    names = []
    for line in (CASE / "recurrent" / "steps.txt").read_text(
            encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            names.append(line.split("\t")[1])

    # Same per-state mask rule as RecurrentTest.java:121 -- a state's own mask
    # wins, the case-root mask is only the fallback. Using the shared mask here
    # would take the argmax over bits this state forbids.
    shared = np.fromfile(CASE / "action_mask.bin", dtype=np.uint8).astype(bool)

    carry = jnp.zeros((1, config.recurrent_size), dtype=jnp.float32)
    seen: dict[tuple[int, ...], None] = {}
    for name in names:
        state = PARITY / name
        observation = np.fromfile(
            state / "expected_observation.bin", dtype="<f4")
        own = state / "action_mask.bin"
        mask = (np.fromfile(own, dtype=np.uint8).astype(bool)
                if own.is_file() else shared)
        carry, logits, _ = apply_policy(
            params, jnp.asarray(observation)[None], carry,
            jnp.asarray(mask)[None])

        row, offset = [], 0
        flat = np.asarray(logits)[0]
        for size in sizes:
            row.append(int(np.argmax(flat[offset:offset + size])))
            offset += size
        seen.setdefault(tuple(row), None)

    return [list(v) for v in seen], len(names)


def decode_rows(vectors: list[list[int]]) -> list[str]:
    """Decode each vector through the same two Python stages export_actions uses."""

    import jax.numpy as jnp
    from hytalegym.jax.combat.observation.v3.policy.surface import (
        action_surface_layout,
        decode_staged_action_surface_factors,
    )
    from hytalegym.jax.combat.observation.v3.native.codec.transport import (
        NATIVE_POLICY_ACTION_PROTOCOL_VERSION,
        translate_learner_arsenal_action_to_native,
    )
    from hytalegym.jax.combat.observation.v3.schema.types import LearnerArsenalAction
    from hytalegym.jax.combat.skills import SKILL_COUNT, SKILL_IDLE, skills_to_actions
    from hytalegym.jax.combat.types import (
        ACTION_JUMP, ACTION_PITCH_DELTA, ACTION_YAW_DELTA,
    )

    layout = action_surface_layout(block_candidate_capacity=16)
    total = int(sum(layout.head_sizes))
    backend = {
        "native_policy_combat_protocol_version": (
            NATIVE_POLICY_ACTION_PROTOCOL_VERSION),
        "supported_actions": ",".join((
            "world_move_direction", "guard_held", "dodge_direction",
            "ability_slot", "use", "place_block", "break_block",
            "craft_recipe")),
    }

    # ActionTest decodes with `ActionDecoder.permissive()`, so the expectation
    # has to be generated under an all-ones mask to be comparable. Mask-illegal
    # behaviour is covered separately by actions_masked.txt.
    permissive = jnp.ones((1, total), dtype=jnp.bool_)

    rows = []
    for index, vector in enumerate(vectors):
        decoded = decode_staged_action_surface_factors(
            jnp.asarray([vector], dtype=jnp.int32), permissive,
            layout=layout, maximum_turn_degrees=45.0)
        base = decoded.base

        def one(value, cast=int):
            return cast(np.asarray(value)[0])

        legal = one(decoded.action_legal, bool)
        skill = one(base.skill_id)
        jump = one(base.jump_held, bool)
        yaw = one(base.yaw_delta_degrees, float)
        pitch = one(base.pitch_delta_degrees, float)

        movement_skill = skill if (legal and 0 <= skill < int(SKILL_COUNT)) \
            else int(SKILL_IDLE)
        low = np.asarray(skills_to_actions(
            jnp.zeros((1, WIDTH), dtype=jnp.float32),
            jnp.asarray([movement_skill], dtype=jnp.int32)),
            dtype=np.float32).copy()
        low[0, int(ACTION_JUMP)] = 1.0 if jump else 0.0
        low[0, int(ACTION_YAW_DELTA)] = yaw
        low[0, int(ACTION_PITCH_DELTA)] = pitch

        action = LearnerArsenalAction(
            skill_id=jnp.asarray([skill], dtype=jnp.int32),
            ability_slot=jnp.asarray([one(base.ability_slot)], dtype=jnp.int32),
            guard_held=jnp.asarray([one(base.guard_held, bool)], dtype=jnp.bool_),
            dodge_direction=jnp.asarray(
                [one(base.dodge_direction)], dtype=jnp.int32),
            jump_held=jnp.asarray([jump], dtype=jnp.bool_),
            world_move_direction=jnp.asarray(
                [one(base.world_move_direction)], dtype=jnp.int32),
            yaw_delta_degrees=jnp.asarray([yaw], dtype=jnp.float32),
            pitch_delta_degrees=jnp.asarray([pitch], dtype=jnp.float32),
            use_requested=jnp.asarray(
                [one(base.use_requested, bool)], dtype=jnp.bool_),
            block_interaction_trigger=jnp.asarray(
                [one(base.block_interaction_trigger)], dtype=jnp.int32),
            recipe_candidate_index=jnp.asarray(
                [one(base.recipe_candidate_index)], dtype=jnp.int32),
            block_candidate_index=jnp.asarray(
                [one(base.block_candidate_index)], dtype=jnp.int32),
        )
        translated = translate_learner_arsenal_action_to_native(
            action, low, backend, policy_legal=legal)
        native = {k: (int(v) if isinstance(v, bool) else v)
                  for k, v in translated.action.items()}

        rows.append("\t".join(str(f) for f in (
            f"trained_{index}",
            ",".join(str(v) for v in vector),
            int(legal), skill, one(base.ability_slot),
            int(one(base.guard_held, bool)), one(base.dodge_direction),
            int(jump), one(base.world_move_direction),
            repr(yaw), repr(pitch),
            int(one(base.use_requested, bool)),
            one(base.block_interaction_trigger),
            one(base.recipe_candidate_index),
            one(base.block_candidate_index),
            int(bool(translated.legal)),
            native["forward"], native["back"], native["left"],
            native["right"], native["attack"], native["jump"],
        )))
    return rows


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    checkpoint, destination = Path(argv[0]), Path(argv[1])

    vectors, steps = trained_action_vectors(checkpoint)
    existing = (CASE / "actions.txt").read_text(encoding="utf-8").splitlines()
    known = {line.split("\t")[1] for line in existing if not line.startswith("#")}
    fresh = [v for v in vectors if ",".join(str(x) for x in v) not in known]

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "actions.txt").write_text(
        "\n".join(existing + decode_rows(fresh)) + "\n", encoding="utf-8")
    shutil.copy(CASE / "actions_masked.txt", destination / "actions_masked.txt")

    print(f"parity steps replayed   {steps}")
    print(f"distinct trained actions {len(vectors)}")
    print(f"not already in fixture   {len(fresh)}")
    for vector in vectors:
        print("   ", ",".join(str(v) for v in vector))
    print(f"wrote {destination / 'actions.txt'} "
          f"({len(existing) - 1} fixture + {len(fresh)} trained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
