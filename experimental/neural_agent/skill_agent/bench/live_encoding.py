"""The v3 encoding on the live 0.7.2 server (``../docs/PERCEPTION.md`` section 10.5): a native learner-v3 episode
(``kill_trork`` on the flat world, the ruleset's Kweebec_Razorleaf body, the target active) in which the agent takes
random legal actions held for several ticks. Each tick the state layer's real tick runs on the live observation
(``env.native_actor_assembly``); the coverage report has ``encoding_audit.py``'s shape, and every raw observation leaf,
the light rows, the body yaw and the actions are saved for replay. A step the native decoder refuses ends the episode
and keeps what was recorded.

Run it as the bridge launcher's follow-on command, on a free port pair, with the pinned jar
(``HytaleRL/hytale-plugin-0.7.2/tools/pin_bridge.py --check`` first)::

    cd HytaleRL/hytale-plugin-0.7.2 && HYTALERL_SERVER_VERSION=0.7.2 APPDATA="$HOME/Library/Application Support" \\
      ../../.venv/bin/python tools/launch_bridge_072.py --jar build/libs/HytaleRLBridge072-0.1.0.jar \\
      --run-dir NEW_EMPTY_DIR --bridge-port 5562 --game-port 25580 --then "/bin/bash THEN.sh"

where ``THEN.sh`` exports ``PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym``, ``JAX_PLATFORMS=cpu``
and ``HYTALERL_SERVER_VERSION=0.7.2``, changes to the repo root, and runs::

    .venv/bin/python -m experimental.neural_agent.skill_agent.bench.live_encoding \\
        --bridge-port "$HYTALERL_BRIDGE_PORT" --steps 300 --out PREFIX [--program random|no_dodge|dodge]

The run takes the shared evidence lease on its port; the server stops when the command ends.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench import encoding_audit as audit
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.state import persona as pe
from experimental.neural_agent.skill_agent.state.senses import Light
from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.jax.combat.native.evidence import acquire_combat_native_evidence_leases
from hytalegym.jax.combat.observation.v1.schema.contract import ENTITY_FLOAT_FEATURES, SELF_FLOAT_FEATURES
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_LOCOMOTION_DODGE_START,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.worldgen.region import current_native_evidence_jar_sha256

SIZES = [int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES]
STARTS = np.cumsum([0, *SIZES[:-1]])
LOCOMOTION = 4
NEUTRAL = np.asarray(neutral_arsenal_policy_action_factors(1))[0].astype(np.int32)
HEALTH = list(SELF_FLOAT_FEATURES).index("health_fraction")
TARGET_HEALTH = list(ENTITY_FLOAT_FEATURES).index("health_fraction")


def next_action(rng: np.random.Generator, mask: np.ndarray, previous: np.ndarray | None, program: str, step: int,
                hold: float = 0.875) -> np.ndarray:
    """The 12 factors of the next step: each head keeps its last choice with probability ``hold`` while it stays
    legal, else draws uniformly among its legal choices. ``no_dodge`` masks the dodge choices out; ``dodge`` is neutral
    but for a dodge on steps 10 and 11."""

    if program == "dodge":
        factors = NEUTRAL.copy()
        if step in (10, 11):
            factors[LOCOMOTION] = ARSENAL_POLICY_LOCOMOTION_DODGE_START + 2
        return factors
    mask = np.asarray(mask, bool).copy()
    if program == "no_dodge":
        mask[STARTS[LOCOMOTION] + ARSENAL_POLICY_LOCOMOTION_DODGE_START:STARTS[LOCOMOTION] + SIZES[LOCOMOTION]] = False
    out = []
    for head, (start, size) in enumerate(zip(STARTS, SIZES)):
        allowed = np.flatnonzero(mask[start:start + size])
        if previous is not None and rng.random() < hold and mask[start + previous[head]]:
            out.append(int(previous[head]))
        else:
            out.append(int(rng.choice(allowed)) if len(allowed) else 0)
    return np.asarray(out, np.int32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bridge-port", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--profile", default="iron_sword")
    parser.add_argument("--program", choices=("random", "no_dodge", "dodge"), default="random")
    parser.add_argument("--out", required=True, help="prefix of PREFIX.npz, PREFIX.coverage.json, PREFIX.info.json")
    options = parser.parse_args(argv)
    rng = np.random.default_rng(options.seed)
    tables = audit.registry_tables()
    tick = jax.jit(lambda observation, carry, action, adapter: layer.step(observation, carry, last_action=action,
                                                                         **adapter))
    coverage = audit.Coverage()
    leaves: dict[str, list] = {}
    extra: dict[str, list] = {"light_f32": [], "light_mask": [], "yaw": [], "action": [], "valid": []}
    state = layer.initial_state(1, persona=pe.sample_personas(jax.random.key(options.seed), 1))
    failure, bridge = None, ""
    with acquire_combat_native_evidence_leases((("127.0.0.1", options.bridge_port),), purpose="encoding_audit_live",
                                               evidence_kinds=("combat_fidelity", "native_actor_evidence",
                                                               "learner_v3_dynamic_row")):
        env = HytaleEnv(task="kill_trork", port=options.bridge_port, max_episode_steps=options.steps + 2,
                        ticks_per_step=1, backend="native", world="flat", combat_target_active=True,
                        learner_v3_profile=options.profile)
        try:
            _, info = env.reset(seed=options.seed)
            bridge = str(info.get("bridge_sha256", "")).upper()
            expected = current_native_evidence_jar_sha256().upper()
            if bridge != expected:
                raise SystemExit(f"bridge {bridge[:8]} is not the pinned {expected[:8]}")
            factors, action = None, audit.gen7_action(jnp.asarray(NEUTRAL[None]))
            started = time.time()
            for step in range(options.steps):
                assembly = env.native_actor_assembly
                observation = assembly.observation
                light = Light(jnp.asarray(assembly.light_tokens.token_f32),
                              jnp.asarray(assembly.light_tokens.token_mask))
                yaw = jnp.asarray([assembly.actor_yaw_degrees], jnp.float32)
                tokens, state = tick(observation, state, action, {"body_yaw_degrees": yaw, "tables": tables,
                                                                  "light": light})
                coverage.add(tokens)
                for key, value in jax.tree_util.tree_flatten_with_path(observation)[0]:
                    leaves.setdefault(jax.tree_util.keystr(key), []).append(np.asarray(value))
                extra["light_f32"].append(np.asarray(light.token_f32))
                extra["light_mask"].append(np.asarray(light.token_mask))
                extra["yaw"].append(np.asarray(yaw))
                extra["valid"].append(bool(info.get("learner_v3_observation_valid", False)))
                factors = next_action(rng, info["learner_v3_action_mask"], factors, options.program, step)
                extra["action"].append(factors)
                action = audit.gen7_action(jnp.asarray(factors[None]))
                base = observation.base
                print(f"tick {step} valid {extra['valid'][-1]} health {float(base.self_f32[0, HEALTH]):.3f} "
                      f"target {float(base.entity_f32[0, 0, TARGET_HEALTH]):.3f} action {factors.tolist()} "
                      f"{time.time() - started:.0f}s", flush=True)
                try:
                    _, _, terminated, truncated, info = env.step(factors)
                except ValueError as error:  # the native decoder refused the next frame: keep what was recorded
                    failure = f"step after tick {step}: {error}"
                    print("STEP FAILED", failure, flush=True)
                    break
                if terminated or truncated:
                    break
        finally:
            env.close()
    np.savez_compressed(options.out + ".npz", **{key: np.stack(value) for key, value in leaves.items()},
                        **{f"__{key}": np.asarray(value) for key, value in extra.items()})
    report = coverage.report()
    with open(options.out + ".coverage.json", "w") as handle:
        json.dump(report, handle, indent=1)
    with open(options.out + ".info.json", "w") as handle:
        json.dump({"bridge": bridge, "program": options.program, "seed": options.seed, "ticks": len(extra["valid"]),
                   "valid": int(sum(extra["valid"])), "failure": failure}, handle, indent=1)
    print(audit.compare({"live": report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
