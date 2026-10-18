#!/usr/bin/env python3
"""Smoke a live HytaleRLBridge 0.7.2 server through the in-server native backend.

Run with the venv interpreter and PYTHONPATH=<HytaleRL>/hytalegym, after
tools/launch_bridge_072.py reports READY (it sets HYTALERL_BRIDGE_PORT,
HYTALERL_GAME_PORT and HYTALERL_SERVER_PID). Opens a gym session on
HytaleSurvive-v0 with backend="headless", so the bridge runs
NativeEnvironmentSession inside the real server. Records every wire request and
response type (config should return "ack", reset "observation"), steps 20 no-op
actions, optionally runs --random-steps seeded random actions that activate at
most one standard-input root (AgentAction.java rejects more than one per
action), then closes.

Default is no random steps. With --random-steps 20 and seed 0, both the 0.7.2
port and the original 0.5.10 bridge hit the same pre-existing failure at step
21 (hotbar_slot 5 is outside the NPC hotbar capacity of 5), then the episode
terminates at step 22.

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from typing import Any

import gymnasium as gym
import msgpack
import numpy as np

import hytalegym
from hytalegym.utils import connection as bridge_connection

TRACE: list[dict[str, Any]] = []
_ORIGINAL_SEND_AND_RECV = bridge_connection.BridgeConnection.send_and_recv

# HytaleEnv._legacy_step_message: place/break offsets are stored as 3 + offset,
# craft 9 means no craft, and ability_slot -1 means no ability.
NOOP_ACTION: dict[str, Any] = {
    "forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0,
    "use": 0, "guard_held": 0, "dodge_direction": 0, "ability_slot": -1,
    "world_move_direction": 0, "requested_charge_time": 0.0,
    "camera_delta_yaw": 0.0, "camera_delta_pitch": 0.0, "body_delta_yaw": 0.0,
    "hotbar_slot": 0, "place_block_x": 3, "place_block_y": 3, "place_block_z": 3,
    "place_block_type": 0, "break_block_x": 3, "break_block_y": 3,
    "break_block_z": 3, "break_block": 0, "craft_recipe_id": 9,
}
ROOT_KEYS = ("attack", "use", "place_block_type", "break_block", "craft_recipe_id", "guard_held", "ability_slot")
INACTIVE_ROOT = {"attack": 0, "use": 0, "place_block_type": 0, "break_block": 0, "craft_recipe_id": 9, "guard_held": 0, "ability_slot": -1}
INFO_KEYS = (
    "native_server_version", "world", "worldgen_provider", "worldgen_version", "npc_role",
    "backend", "task", "fidelity", "native_tick_rate", "native_time_dilation",
    "native_npc_count", "bridge_sha256", "native_hostility_authority",
    "resolved_spawn_x", "resolved_spawn_y", "resolved_spawn_z",
)


def _traced_send_and_recv(self: Any, message: dict[str, Any]) -> dict[str, Any]:
    request_type = message.get("type")
    try:
        response = _ORIGINAL_SEND_AND_RECV(self, message)
    except Exception as exception:
        TRACE.append({"request": request_type, "error": f"{type(exception).__name__}: {exception}"})
        raise
    TRACE.append({"request": request_type, "response": response.get("type"), "keys": sorted(response)})
    return response


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _single_root_action(sample: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """A random action from the Dict space with at most one standard-input root active."""
    action = {key: _plain(value) for key, value in sample.items()}
    for key in ROOT_KEYS:
        action[key] = INACTIVE_ROOT[key]
    choice = int(rng.integers(0, 8))  # 0 = no root, 1..7 = one root
    if choice == 1:
        action["attack"] = 1
    elif choice == 2:
        action["use"] = 1
    elif choice == 3:
        action["place_block_type"] = max(1, int(sample["place_block_type"]))
    elif choice == 4:
        action["break_block"] = 1
    elif choice == 5:
        action["craft_recipe_id"] = int(sample["craft_recipe_id"]) % 9
    elif choice == 6:
        action["guard_held"] = 1
    elif choice == 7:
        action["ability_slot"] = max(0, int(sample["ability_slot"]))
    return action


def _non_noop(action: dict[str, Any]) -> dict[str, Any]:
    return {key: _plain(value) for key, value in action.items() if NOOP_ACTION.get(key) != _plain(value)}


def _describe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _describe(item) for key, item in value.items()}
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _digest(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": np.round(np.asarray(obs["position"]), 3).tolist(),
        "health": float(np.asarray(obs["health"])),
    }


def _lsof(args: list[str]) -> str:
    completed = subprocess.run(["lsof", "-nP", *args], capture_output=True, text=True)
    return (completed.stdout + completed.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke the live 0.7.2 bridge.")
    parser.add_argument("--host", default=os.environ.get("HYTALERL_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HYTALERL_BRIDGE_PORT", "5556")))
    parser.add_argument("--game-port", type=int, default=int(os.environ.get("HYTALERL_GAME_PORT", "25572")))
    parser.add_argument("--server-pid", type=int, default=int(os.environ.get("HYTALERL_SERVER_PID", "0")))
    parser.add_argument("--env-id", default="HytaleSurvive-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-steps", type=int, default=0)
    parser.add_argument("--noop-steps", type=int, default=20)
    args = parser.parse_args()

    bridge_connection.BridgeConnection.send_and_recv = _traced_send_and_recv
    result: dict[str, Any] = {
        "versions": {
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "msgpack": ".".join(str(part) for part in msgpack.version),
            "hytalegym": os.path.dirname(hytalegym.__file__),
        },
        "env_id": args.env_id,
        "host": args.host,
        "port": args.port,
        "backend": "headless",
        "checks": {},
        "steps": [],
    }
    checks: dict[str, bool] = result["checks"]
    env = gym.make(args.env_id, host=args.host, port=args.port, backend="headless")
    space = env.action_space
    space.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    error = None
    try:
        obs, info = env.reset(seed=args.seed)
        space_keys = set(env.observation_space.spaces)
        checks["reset_returned_observation"] = isinstance(obs, dict)
        checks["reset_obs_covers_observation_space"] = space_keys.issubset(obs.keys())
        result["reset"] = {
            "obs": _describe(obs),
            "info_values": {key: _plain(info[key]) for key in INFO_KEYS if key in info},
            "info_key_count": len(info),
            "digest": _digest(obs),
            "extra_keys_vs_space": sorted(set(obs) - space_keys),
        }

        plan = [("noop", dict(NOOP_ACTION)) for _ in range(args.noop_steps)]
        plan += [("random_single_root", _single_root_action(space.sample(), rng)) for _ in range(args.random_steps)]
        for index, (kind, action) in enumerate(plan):
            record: dict[str, Any] = {"i": index, "kind": kind, "non_noop": _non_noop(action)}
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as exception:
                record["error"] = f"{type(exception).__name__}: {exception}"
                result["steps"].append(record)
                continue
            record.update({
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_count": info.get("step_count"),
                "obs_covers_space": space_keys.issubset(obs.keys()),
                "digest": _digest(obs),
            })
            result["steps"].append(record)

        steps = result["steps"]
        counts = [step["step_count"] for step in steps if "step_count" in step]
        checks["all_planned_steps_ran_without_error"] = len(steps) == len(plan) and all("error" not in s for s in steps)
        checks["steps_obs_covers_observation_space"] = all(s.get("obs_covers_space") for s in steps if "error" not in s)
        checks["step_count_strictly_increases"] = counts == sorted(set(counts)) and len(counts) == len(plan)
        checks["no_done_flags_in_smoke_window"] = not any(
            s.get("terminated") or s.get("truncated") for s in steps if "error" not in s
        )
    except Exception as exception:
        error = "".join(traceback.format_exception(exception)).strip()
    finally:
        try:
            env.close()
        except Exception as exception:
            error = (error or "") + f"\nclose failed: {type(exception).__name__}: {exception}"

    result["trace"] = TRACE
    checks["config_then_reset_wire_order"] = (
        len(TRACE) >= 2
        and TRACE[0].get("request") == "config"
        and TRACE[0].get("response") == "ack"
        and TRACE[1].get("request") == "reset"
        and TRACE[1].get("response") == "observation"
    )
    checks["no_bridge_errors_on_wire"] = not any("error" in item for item in TRACE)
    checks["no_exception"] = error is None
    if args.server_pid:
        tcp = _lsof(["-a", "-p", str(args.server_pid), f"-iTCP:{args.port}", "-sTCP:LISTEN"])
        udp = _lsof(["-a", "-p", str(args.server_pid), f"-iUDP:{args.game_port}"])
        result["listening"] = {"tcp_bridge": tcp, "udp_game": udp}
        checks["bridge_tcp_listening_by_server_pid"] = "TCP" in tcp and str(args.server_pid) in tcp
        checks["game_udp_bound_by_server_pid"] = "UDP" in udp and str(args.server_pid) in udp
    result["error"] = error
    result["passed"] = all(checks.values())
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
