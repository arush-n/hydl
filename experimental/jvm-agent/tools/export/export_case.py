"""Export everything a JVM needs to run the JAX-trained policy, plus the answer.

Proof-of-concept scope: can a network run in the JVM, take a *real* observation
captured from the live Hytale server, and produce the same outputs JAX does?

This writes three things into ``case/``:

* the trained weights, as flat little-endian float32 -- no pickle, no npz
  reader needed on the Java side;
* one real native observation and its 99-bit legality mask, captured live from
  the server, so the Java side is fed genuine server perception rather than a
  synthetic vector;
* the JAX reference outputs (logits, value, decoded 12 factors) so the Java
  result can be checked rather than eyeballed.

Run against the live Gym and record its contract identities with the exported
case.  The frozen ``613C4D9C`` snapshot is a historical comparison oracle only;
it predates ``world_actions`` and cannot produce the current flat/Region
fixtures used by this JVM port.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
CASE = JVM_AGENT_ROOT / "case"
PROFILE, SEED = "iron_sword", 570057

WEIGHTS = (
    "encoder_input_kernel", "encoder_input_bias",
    "encoder_hidden_kernel", "encoder_hidden_bias",
    "gru_input_kernel", "gru_recurrent_kernel", "gru_bias",
    "actor_kernel", "actor_bias",
    "critic_kernel", "critic_bias",
)


def write(name: str, array: np.ndarray, dtype="<f4") -> dict:
    array = np.ascontiguousarray(array, dtype=dtype)
    (CASE / f"{name}.bin").write_bytes(array.tobytes())
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def native_case():
    """One real observation + mask straight off the live server."""

    from hytalegym.envs import HytaleEnv

    env = HytaleEnv(
        task="kill_trork", backend="native", host="127.0.0.1", port=5556,
        world="flat", ticks_per_step=1, max_episode_steps=4,
        npc_role="Kweebec_Razorleaf", combat_target_active=True,
        learner_v3_profile=PROFILE,
    )
    try:
        observation, info = env.reset(seed=SEED)
        return (
            np.asarray(observation, dtype=np.float32),
            np.asarray(info["learner_v3_action_mask"], dtype=bool),
            bool(info.get("learner_v3_observation_valid")),
            str(info.get("bridge_sha256", "")).upper(),
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def main(checkpoint: str) -> int:
    import jax.numpy as jnp
    from hytalegym.jax.combat import ARSENAL_POLICY_ACTION_HEAD_SIZES
    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_STANDARD_ROOT_DISTRIBUTION,
        greedy_arsenal_action_factors,
    )
    from hytalegym.jax.training.checkpoint import load_policy_checkpoint
    from hytalegym.jax.training.policy import apply_policy

    CASE.mkdir(parents=True, exist_ok=True)
    raw = np.load(checkpoint)
    params, config, metadata = load_policy_checkpoint(Path(checkpoint))

    observation, mask, valid, bridge = native_case()
    print(f"native observation {observation.shape} valid={valid} bridge={bridge[:8]}")

    recurrent = jnp.zeros((1, config.recurrent_size), dtype=jnp.float32)
    next_recurrent, logits, value = apply_policy(
        params,
        jnp.asarray(observation)[None],
        recurrent,
        jnp.asarray(mask)[None],
    )
    logits = np.asarray(logits)[0]
    factors = np.asarray(
        greedy_arsenal_action_factors(jnp.asarray(logits)[None]),
        dtype=np.int32,
    )[0]

    manifest = {
        "profile": PROFILE,
        "seed": SEED,
        "bridge_sha256": bridge,
        "observation_valid": valid,
        "observation_size": int(observation.size),
        "recurrent_size": int(config.recurrent_size),
        "action_head_sizes": [int(s) for s in ARSENAL_POLICY_ACTION_HEAD_SIZES],
        "action_distribution": ARSENAL_STANDARD_ROOT_DISTRIBUTION,
        "checkpoint_metadata": {
            k: metadata[k] for k in (
                "observation_size", "action_size", "observation_contract_sha256",
            ) if k in metadata
        },
        "arrays": {},
        "expected": {
            "factors": factors.tolist(),
            "value": float(np.asarray(value)[0]),
        },
    }
    for name in WEIGHTS:
        manifest["arrays"][name] = write(name, raw[name])
    manifest["arrays"]["observation"] = write("observation", observation)
    manifest["arrays"]["action_mask"] = write("action_mask", mask, dtype="u1")
    manifest["arrays"]["expected_logits"] = write("expected_logits", logits)
    manifest["arrays"]["expected_recurrent"] = write(
        "expected_recurrent", np.asarray(next_recurrent)[0]
    )

    (CASE / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(f"wrote {len(manifest['arrays'])} arrays to {CASE}")
    print(f"jax factors = {factors.tolist()}")
    print(f"jax value   = {manifest['expected']['value']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
