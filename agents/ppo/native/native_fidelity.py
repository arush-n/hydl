"""Run one exact JAX checkpoint through the existing native environment loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import jax
import jaxlib
import numpy as np

from adk.evaluation import recurrent_policy
from hytalegym.jax.training.checkpoint import load_policy_checkpoint


NATIVE_POLICY_FIDELITY_SCHEMA = "hytalerl_jax_policy_native_fidelity_v1"
NATIVE_POLICY_TRACE_SCHEMA = "hytalerl_jax_policy_native_trace_v1"


def run_checkpoint_native_fidelity(
    session: Any,
    checkpoint: Path,
    bridge_jar: Path,
    *,
    seed: int,
    max_steps: int,
    decode_mode: str = "factored_argmax",
    require_terminal: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one checkpoint and drive the existing native session with JAX."""

    checkpoint, bridge_jar = Path(checkpoint).resolve(), Path(bridge_jar).resolve()
    if not checkpoint.is_file() or not bridge_jar.is_file():
        raise FileNotFoundError("checkpoint and bridge JAR must both exist")
    params, config, metadata = load_policy_checkpoint(checkpoint)
    if not (
        metadata.get("schema") == "hytalerl_native_behavior_live_policy_v2"
        and metadata.get("live_policy_compatible") is True
    ):
        raise ValueError("checkpoint is not a canonical native live policy")
    if config.observation_size != session.stamp.observation_size:
        raise ValueError("checkpoint/native observation widths differ")
    if tuple(config.action_head_sizes) != tuple(session.stamp.action_head_sizes):
        raise ValueError("checkpoint/native action heads differ")
    transfer = metadata.get("transfer_contract", {})
    observation_sha = transfer.get("observation_contract_sha256")
    if observation_sha != transfer.get("action_contract_sha256"):
        raise ValueError("checkpoint observation/action identities differ")
    dynamics_sha = transfer.get("combat_dynamics_contract_sha256")
    _require_sha256(observation_sha, "observation/action contract")
    _require_sha256(dynamics_sha, "combat dynamics contract")
    policy, carry = recurrent_policy(
        params,
        tuple(config.action_head_sizes),
        decode_mode=decode_mode,
    )
    return record_native_policy_session(
        session,
        policy,
        carry,
        key=jax.random.key(seed),
        seed=seed,
        max_steps=max_steps,
        require_terminal=require_terminal,
        policy_sha256=_sha256(checkpoint),
        bridge_jar_sha256=_sha256(bridge_jar),
        observation_contract_sha256=observation_sha,
        combat_dynamics_contract_sha256=dynamics_sha,
        decode_mode=decode_mode,
        runtime=_jax_runtime(),
    )


def record_native_policy_session(
    session: Any,
    policy: Any,
    carry: Any,
    *,
    key: jax.Array,
    seed: int,
    max_steps: int,
    require_terminal: bool,
    policy_sha256: str,
    bridge_jar_sha256: str,
    observation_contract_sha256: str,
    combat_dynamics_contract_sha256: str,
    decode_mode: str,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record a policy-specific native episode through ``step_policy``."""

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    for name, digest in (
        ("policy", policy_sha256),
        ("bridge JAR", bridge_jar_sha256),
        ("observation/action contract", observation_contract_sha256),
        ("combat dynamics contract", combat_dynamics_contract_sha256),
    ):
        _require_sha256(digest, name)
    inference_seconds: list[float] = []

    def timed_policy(current_carry, policy_input, step_key):
        started = perf_counter()
        result = policy(current_carry, policy_input, step_key)
        _block_until_ready(result)
        inference_seconds.append(perf_counter() - started)
        return result

    rows: list[dict[str, Any]] = []
    started = perf_counter()
    with session as opened:
        policy_input = opened.reset(seed=seed)
        invalid_rows = int(not _valid(policy_input))
        illegal_rows = 0
        terminated = truncated = False
        for index in range(max_steps):
            input_observation_sha = _array_sha256(policy_input.observation)
            input_mask_sha = _array_sha256(policy_input.action_mask)
            tick = perf_counter()
            carry, transition = opened.step_policy(
                timed_policy,
                carry,
                jax.random.fold_in(key, index),
            )
            total_seconds = perf_counter() - tick
            inference = inference_seconds[-1]
            policy_input = transition.policy_input
            legal = bool(transition.host_policy_action_legal)
            valid = _valid(policy_input)
            illegal_rows += int(not legal)
            invalid_rows += int(not valid)
            terminated = bool(transition.terminated)
            truncated = bool(transition.truncated)
            rows.append(
                {
                    "step": index,
                    "input_observation_sha256": input_observation_sha,
                    "input_action_mask_sha256": input_mask_sha,
                    "action_factors": np.asarray(
                        transition.action_factors, dtype=np.int32
                    ).tolist(),
                    "recurrent_sha256": _tree_sha256(carry),
                    "next_observation_sha256": _array_sha256(
                        policy_input.observation
                    ),
                    "next_action_mask_sha256": _array_sha256(
                        policy_input.action_mask
                    ),
                    "observation_valid": valid,
                    "failure_bits": int(policy_input.failure_bits),
                    "mechanics_failure_bits": int(
                        policy_input.mechanics_failure_bits
                    ),
                    "arsenal_failure_bits": int(policy_input.arsenal_failure_bits),
                    "host_policy_action_legal": legal,
                    "host_policy_action_reject_reasons": list(
                        transition.host_policy_action_reject_reasons
                    ),
                    "reward": float(transition.reward),
                    "terminated": terminated,
                    "truncated": truncated,
                    "inference_seconds": inference,
                    "native_step_seconds": max(0.0, total_seconds - inference),
                    "total_step_seconds": total_seconds,
                }
            )
            if transition.bridge_sha256 != bridge_jar_sha256:
                raise ValueError("native session used a different bridge JAR")
            if transition.done:
                break
    wall_seconds = perf_counter() - started
    warm_inference = inference_seconds[1:] or inference_seconds
    native_seconds = [row["native_step_seconds"] for row in rows]
    passed = bool(
        rows
        and invalid_rows == 0
        and illegal_rows == 0
        and not truncated
        and (terminated or not require_terminal)
        and runtime.get("backend") == "gpu"
    )
    trace = {
        "schema": NATIVE_POLICY_TRACE_SCHEMA,
        "seed": seed,
        "rows": rows,
    }
    receipt = {
        "schema": NATIVE_POLICY_FIDELITY_SCHEMA,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "competence_claim": False,
        "policy_runtime": "jax",
        "environment_backend": "native_java",
        "policy_sha256": policy_sha256,
        "bridge_jar_sha256": bridge_jar_sha256,
        "observation_contract_sha256": observation_contract_sha256,
        "combat_dynamics_contract_sha256": combat_dynamics_contract_sha256,
        "decode_mode": decode_mode,
        "runtime": dict(runtime),
        "episode": {
            "seed": seed,
            "steps": len(rows),
            "maximum_steps": max_steps,
            "terminal_required": require_terminal,
            "terminated": terminated,
            "truncated": truncated,
            "invalid_observation_rows": invalid_rows,
            "illegal_action_rows": illegal_rows,
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "steps_per_second": len(rows) / wall_seconds,
            "first_inference_seconds": inference_seconds[0],
            "warm_inference_p50_seconds": _percentile(warm_inference, 50),
            "warm_inference_p95_seconds": _percentile(warm_inference, 95),
            "native_step_p50_seconds": _percentile(native_seconds, 50),
            "native_step_p95_seconds": _percentile(native_seconds, 95),
        },
    }
    return receipt, trace


def write_native_policy_fidelity(
    output: Path,
    receipt: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Write the compact receipt and its independently hashed full trace."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_name(f"{output.stem}.trace.json")
    trace_bytes = (json.dumps(trace, indent=2, sort_keys=True) + "\n").encode()
    trace_path.write_bytes(trace_bytes)
    receipt = {
        **receipt,
        "trace": {
            "path": trace_path.resolve().as_posix(),
            "sha256": hashlib.sha256(trace_bytes).hexdigest().upper(),
            "rows": len(trace["rows"]),
        },
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["content_sha256"] = hashlib.sha256(payload).hexdigest().upper()
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _valid(policy_input: Any) -> bool:
    return bool(
        policy_input.observation_valid
        and policy_input.failure_bits == 0
        and policy_input.mechanics_failure_bits == 0
        and policy_input.arsenal_failure_bits == 0
        and not policy_input.loadout_failure
    )


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256(str(jax.tree.structure(value)).encode())
    for leaf in jax.tree.leaves(value):
        digest.update(bytes.fromhex(_array_sha256(leaf)))
    return digest.hexdigest().upper()


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree.leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _percentile(values: list[float], percentile: int) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _require_sha256(value: Any, name: str) -> None:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _jax_runtime() -> dict[str, str]:
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--ticks-per-step", type=int, default=1)
    parser.add_argument("--native-tick-rate", type=int, default=30)
    parser.add_argument("--npc-role", required=True)
    parser.add_argument("--opponent-role", required=True)
    parser.add_argument("--loadout", default="iron_sword")
    parser.add_argument("--target-profile", default="iron_sword")
    parser.add_argument("--world", default="flat")
    parser.add_argument("--worldgen-structure", default="Default")
    parser.add_argument("--allow-horizon", action="store_true")
    args = parser.parse_args()

    from adk import AgentKit, AgentSpec

    kit = AgentKit()
    spec = AgentSpec(
        name="fidelity/jax-policy-native-environment",
        scene="combat/open_flat_control",
        loadout=args.loadout,
    )
    kit.register(spec)
    handle = kit.build(spec, batch=1)
    session = handle.native_session(
        host=args.host,
        port=args.port,
        purpose="jax_policy_native_fidelity",
        npc_role=args.npc_role,
        combat_target_role=args.opponent_role,
        target_profile=args.target_profile,
        world=args.world,
        worldgen_structure=args.worldgen_structure,
        max_episode_steps=args.steps,
        ticks_per_step=args.ticks_per_step,
        native_tick_rate=args.native_tick_rate,
    )
    receipt, trace = run_checkpoint_native_fidelity(
        session,
        args.checkpoint,
        args.bridge_jar,
        seed=args.seed,
        max_steps=args.steps,
        require_terminal=not args.allow_horizon,
    )
    receipt = write_native_policy_fidelity(args.output, receipt, trace)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
