"""Capture a role-diverse, model-neutral Java NPC behavior corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from zipfile import ZipFile

import numpy as np

from arena.imitation import (
    NATIVE_CONTEXTUAL_BEHAVIOR_VIEW,
    NATIVE_CONTEXTUAL_STEERING_VIEW,
    NATIVE_CORPUS_COMPATIBILITY_KEYS,
    TraceCorpus,
    native_concurrent_trace_sequences,
)
from hytalegym.combat.assets import HYTALE_0_5_7_ASSETS_SHA256
from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.rulesets import require_hytale_0_5_7_assets_archive
from hytalegym.worldgen import (
    NATIVE_NPC_TRACE_MAX_DRAIN,
    NativeNpcTraceCapture,
    concatenate_native_npc_traces,
    load_native_npc_trace_artifact,
    write_native_npc_trace_artifact,
)


CORPUS_SCHEMA = "arena_native_contextual_behavior_corpus_v4"


@dataclass(frozen=True, slots=True)
class NativeBehaviorScenario:
    name: str
    actor_role: str
    opponent_role: str
    seed: int
    world: str = "flat"
    fixture: str = "default"
    max_steps: int = 2048
    worldgen_structure: str = "Default"

    def __post_init__(self) -> None:
        if not self.name or self.actor_role == self.opponent_role or self.max_steps < 1:
            raise ValueError(
                "scenario needs a name, distinct roles, and positive max_steps"
            )


_MATCHUPS = (
    ("trork_hunter_kweebec", "Trork_Hunter", "Kweebec_Razorleaf", 570211),
    ("trork_warrior_kweebec", "Trork_Warrior", "Kweebec_Razorleaf", 570251),
    ("trork_brawler_kweebec", "Trork_Brawler", "Kweebec_Razorleaf", 570257),
    ("trork_shaman_kweebec", "Trork_Shaman", "Kweebec_Razorleaf", 570263),
    ("feran_sharptooth_scarak", "Feran_Sharptooth", "Scarak_Fighter", 570223),
    ("feran_longtooth_scarak", "Feran_Longtooth", "Scarak_Seeker", 570229),
    ("feran_burrower_scarak", "Feran_Burrower", "Scarak_Fighter", 570277),
    ("spider_mouse", "Spider", "Mouse", 570233),
    ("spider_cave_mouse", "Spider_Cave", "Mouse", 570281),
    ("scorpion_frog", "Scorpion", "Frog_Green", 570241),
    ("scorpion_frog_blue", "Scorpion", "Frog_Blue", 570293),
    ("rat_squirrel", "Rat", "Squirrel", 570287),
)
_WORLDS = ("flat", "hytale_generator")


def native_behavior_scenarios(
    repeats: int = 2,
    *,
    matchups: tuple[tuple[str, str, str, int], ...] = _MATCHUPS,
) -> tuple[NativeBehaviorScenario, ...]:
    """Build deterministic role/world matchups with independent episode seeds."""

    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    return tuple(
        NativeBehaviorScenario(
            f"{name}_{world}_r{repeat}",
            actor,
            opponent,
            seed + 1009 * repeat + 101 * world_index,
            world=world,
        )
        for name, actor, opponent, seed in matchups
        for world_index, world in enumerate(_WORLDS)
        for repeat in range(repeats)
    )


DEFAULT_SCENARIOS = native_behavior_scenarios()


def capture_native_behavior_corpus(
    output: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 5556,
    expected_bridge: str | None = None,
    scenarios: tuple[NativeBehaviorScenario, ...] = DEFAULT_SCENARIOS,
    native_tick_rate: int = 120,
    native_time_dilation: float = 4.0,
    capture_retries: int = 3,
) -> dict[str, Any]:
    """Capture both authored Java policies in every declared matchup."""

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    assets = require_hytale_0_5_7_assets_archive()
    roles = sorted({role for row in scenarios for role in _roles(row)})
    role_assets = _role_assets(assets, roles)
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    traces = []
    bridge = expected_bridge.upper() if expected_bridge else None
    for scenario in scenarios:
        print(f"capture {scenario.name}", flush=True)
        result, scenario_traces, observed_bridge = _capture_scenario_with_retry(
            scenario,
            destination,
            host=host,
            port=port,
            expected_bridge=bridge,
            native_tick_rate=native_tick_rate,
            native_time_dilation=native_time_dilation,
            attempts=capture_retries,
        )
        bridge = observed_bridge
        rows.append(result)
        traces.extend(scenario_traces)

    corpus = TraceCorpus(tuple(traces), NATIVE_CORPUS_COMPATIBILITY_KEYS)
    scenario_manifest = [asdict(row) for row in scenarios]
    actors = [actor for row in rows for actor in row["actors"]]
    report = {
        "schema": CORPUS_SCHEMA,
        "status": "captured_death_complete_mutually_hostile_model_neutral",
        "bridge_sha256": bridge,
        "assets_sha256": HYTALE_0_5_7_ASSETS_SHA256,
        "scenario_contract_sha256": _sha(scenario_manifest),
        "scenarios": rows,
        "scenario_manifest": scenario_manifest,
        "role_assets": role_assets,
        "corpus": {**corpus.manifest, "sha256": corpus.sha256},
        "coverage": {
            "matchups": len(rows),
            "actor_traces": len(traces),
            "roles": roles,
            "worlds": sorted({row.world for row in scenarios}),
            "fixtures": sorted({row.fixture for row in scenarios}),
            "rows": corpus.rows,
            "traces_with_movement": sum(
                actor["behavior"]["moving_rows"] > 0 for actor in actors
            ),
            "traces_with_aim_change": sum(
                actor["behavior"]["aim_change_rows"] > 0 for actor in actors
            ),
            "traces_with_attack_activation": sum(
                actor["behavior"]["attack_activation_rows"] > 0 for actor in actors
            ),
            "traces_with_exact_attack_execution": sum(
                actor["behavior"]["attack_execution_cause_rows"] > 0 for actor in actors
            ),
            "contextual_steering_valid_rows": sum(
                actor["contextual_steering_valid_rows"] for actor in actors
            ),
            "attack_activation_rows": sum(
                actor["behavior"]["attack_activation_rows"] for actor in actors
            ),
            "attack_execution_rows": sum(
                actor["behavior"]["attack_execution_rows"] for actor in actors
            ),
            "attack_execution_cause_available_rows": sum(
                actor["behavior"]["attack_execution_cause_available_rows"]
                for actor in actors
            ),
            "attack_execution_cause_rows": sum(
                actor["behavior"]["attack_execution_cause_rows"] for actor in actors
            ),
            "damage_event_rows": sum(
                actor["behavior"]["damage_event_rows"] for actor in actors
            ),
            "active_action_labels": dict(
                sum(
                    (
                        Counter(actor["behavior"]["active_action_labels"])
                        for actor in actors
                    ),
                    Counter(),
                )
            ),
            "interaction_types": dict(
                sum(
                    (
                        Counter(actor["behavior"]["interaction_types"])
                        for actor in actors
                    ),
                    Counter(),
                )
            ),
            "natural_terminal_matchups": sum(row["terminated"] for row in rows),
            "winner_traces": sum(actor["outcome"] == "winner" for actor in actors),
            "loser_traces": sum(actor["outcome"] == "loser" for actor in actors),
            "simultaneous_traces": sum(
                actor["outcome"] == "simultaneous" for actor in actors
            ),
            "mutually_hostile_matchups": sum(
                row["hostility"]["mutual"] for row in rows
            ),
        },
        "training_surface": {
            "view": "NATIVE_CONTEXTUAL_BEHAVIOR_VIEW",
            "labels": "raw concurrent Java steering, aim, and exact attack causes",
            "episode_structure": "unavailable unless separately projected from task evidence",
            "action_projection": "none; learner-specific projectors must abstain when ambiguous",
            "factored_head_projection": {
                "base_action": "exact Java combat-chain start -> ATTACK; available non-start -> IDLE",
                "locomotion_gait_compass": "exact native body steering",
                "yaw_delta_bins": "exact native head steering",
                "pitch_delta_bins": "exact native head steering",
                "remaining_heads": "abstain unless a typed Java cause is captured and joined",
            },
            "outcome_weighting": "none; both actors retained and sampled actor-uniformly",
        },
        "wall_seconds": perf_counter() - started,
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def probe_native_hostility(
    output: str | Path,
    matchups: tuple[tuple[str, str], ...],
    *,
    host: str = "127.0.0.1",
    port: int = 5556,
    expected_bridge: str | None = None,
) -> dict[str, Any]:
    """Ask the live Java server whether each role pair is mutually hostile."""

    if not matchups:
        raise ValueError("hostility probe needs at least one matchup")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    bridge = expected_bridge.upper() if expected_bridge else None
    rows = []
    for index, (actor, opponent) in enumerate(matchups):
        scenario = NativeBehaviorScenario(
            f"probe_{index}_{actor}_{opponent}", actor, opponent, 590000 + index
        )
        env = _native_environment(
            scenario,
            host=host,
            port=port,
            native_tick_rate=120,
            native_time_dilation=4.0,
        )
        try:
            try:
                _, info = env.reset(seed=scenario.seed)
            except RuntimeError as error:
                rows.append(
                    {
                        "actor_role": actor,
                        "opponent_role": opponent,
                        "accepted": False,
                        "server_rejection": str(error),
                    }
                )
            else:
                observed_bridge = str(info["bridge_sha256"]).upper()
                if bridge is not None and observed_bridge != bridge:
                    raise ValueError("live bridge does not match the requested build")
                bridge = observed_bridge
                hostility = _hostility_values(info)
                rows.append(
                    {
                        "actor_role": actor,
                        "opponent_role": opponent,
                        "accepted": hostility == _EXPECTED_HOSTILITY,
                        "hostility": hostility,
                    }
                )
        finally:
            env.close()
    report = {
        "schema": "arena_native_hostility_probe_v1",
        "bridge_sha256": bridge,
        "authority": "live Hytale WorldSupport.getAttitude",
        "matchups": rows,
        "accepted": sum(row["accepted"] for row in rows),
    }
    (destination / "hostility-probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _capture_scenario_with_retry(scenario, output, *, attempts, **connection):
    if isinstance(attempts, bool) or attempts < 1:
        raise ValueError("capture retries must be a positive integer")
    for attempt in range(1, attempts + 1):
        try:
            result = _capture_scenario(scenario, output, **connection)
            result[0]["capture_attempts"] = attempt
            return result
        except RuntimeError as error:
            transient = "World thread is not accepting tasks" in str(error)
            if not transient or attempt == attempts:
                raise
            print(
                f"retry {scenario.name} after transient world-thread closure "
                f"({attempt}/{attempts})",
                flush=True,
            )


def _capture_scenario(
    scenario: NativeBehaviorScenario,
    output: Path,
    **connection: Any,
) -> tuple[dict[str, Any], tuple[Any, Any], str]:
    expected = connection.pop("expected_bridge")
    env = _native_environment(scenario, **connection)
    try:
        _, info = env.reset(seed=scenario.seed)
        bridge = str(info["bridge_sha256"]).upper()
        if expected is not None and bridge != expected:
            raise ValueError("live bridge does not match the requested build")
        hostility = _hostility(info)
        npcs = env.capture_privileged_npcs([-256, 0, -256, 256, 512, 256])
        indexes = tuple(
            _unique_role_index(npcs.type_asset_ids, role) for role in _roles(scenario)
        )
        captures, raw_chunks, steps, terminal = _capture_pair(
            env, npcs, indexes, scenario.max_steps
        )
        outcomes = _death_outcomes(captures, terminal)
        artifacts = []
        for capture, chunks in zip(captures, raw_chunks, strict=True):
            artifact = write_native_npc_trace_artifact(
                output / scenario.name / f"{capture.role}.trace.msgpack.gz", chunks
            )
            replay = load_native_npc_trace_artifact(artifact["path"])
            if replay.trace_uuid != capture.trace_uuid or not np.array_equal(
                replay.state, capture.state
            ):
                raise AssertionError("stored trace changed during replay decode")
            artifacts.append(artifact)
        paired = native_concurrent_trace_sequences(captures)
        actor_rows = []
        for capture, opponent, trace, artifact, outcome in zip(
            captures, captures[::-1], paired, artifacts, outcomes, strict=True
        ):
            actor_rows.append(
                _actor_summary(capture, opponent, trace, artifact, outcome)
            )
        return (
            {
                "scenario": asdict(scenario),
                "steps": steps,
                "terminated": terminal[0],
                "truncated": terminal[1],
                "hostility": hostility,
                "actors": actor_rows,
            },
            paired,
            bridge,
        )
    finally:
        env.close()


_EXPECTED_HOSTILITY = {
    "actor_to_target": "HOSTILE",
    "target_to_actor": "HOSTILE",
    "mutual": True,
    "authority": "WorldSupport.getAttitude",
}


def _hostility_values(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor_to_target": str(info.get("native_actor_target_attitude", "")),
        "target_to_actor": str(info.get("native_target_actor_attitude", "")),
        "mutual": info.get("native_mutual_hostility") is True,
        "authority": str(info.get("native_hostility_authority", "")),
    }


def _hostility(info: dict[str, Any]) -> dict[str, Any]:
    result = _hostility_values(info)
    if result != _EXPECTED_HOSTILITY:
        raise RuntimeError(f"native matchup lacks authored mutual hostility: {result}")
    return result


def _native_environment(scenario: NativeBehaviorScenario, **connection: Any):
    return HytaleEnv(
        task="kill_trork",
        backend="native",
        ticks_per_step=1,
        max_episode_steps=scenario.max_steps,
        world=scenario.world,
        worldgen_structure=scenario.worldgen_structure,
        npc_role=scenario.actor_role,
        combat_target_role=scenario.opponent_role,
        combat_target_active=True,
        fidelity_fixture=scenario.fixture,
        **connection,
    )


def _capture_pair(env, npcs, indexes, ticks):
    ids, chunks = [], [[], []]
    try:
        for index in indexes:
            started = env.start_npc_trace(
                npcs.uuids[index],
                expected_role=npcs.type_asset_ids[index],
                capacity=ticks + NATIVE_NPC_TRACE_MAX_DRAIN,
            )
            ids.append(started.trace_uuid)
        terminated = truncated = False
        steps = 0
        for steps in range(1, ticks + 1):
            _, _, terminated, truncated, _ = env.step_native_npcs()
            if steps % NATIVE_NPC_TRACE_MAX_DRAIN == 0:
                for index, trace_uuid in enumerate(ids):
                    chunks[index].append(env.poll_npc_trace_wire(trace_uuid))
            if terminated or truncated:
                break
        for index, trace_uuid in enumerate(ids):
            chunks[index].append(env.stop_npc_trace_wire(trace_uuid))
        ids.clear()
        captures = tuple(
            concatenate_native_npc_traces(
                tuple(NativeNpcTraceCapture.from_response(row) for row in actor)
            )
            for actor in chunks
        )
        return (
            captures,
            tuple(tuple(actor) for actor in chunks),
            steps,
            (
                terminated,
                truncated,
            ),
        )
    finally:
        for trace_uuid in ids:
            try:
                env.stop_npc_trace(trace_uuid)
            except Exception:
                pass


def _death_outcomes(captures, terminal):
    terminated, truncated = terminal
    dead = tuple(bool(np.any(capture.next_state[:, 11] <= 0.0)) for capture in captures)
    if not terminated or truncated or not any(dead):
        raise RuntimeError(
            "native behavior episode reached its safety horizon without a death"
        )
    return tuple(
        "simultaneous" if all(dead) else "loser" if value else "winner"
        for value in dead
    )


def _actor_summary(capture, opponent, trace, artifact, outcome):
    displacement = capture.next_state[:, :3] - capture.state[:, :3]
    movement = np.linalg.norm(displacement, axis=1)
    aim_delta = np.linalg.norm(
        capture.next_state[:, 9:11] - capture.state[:, 9:11], axis=1
    )
    locked = np.asarray([value == opponent.npc_uuid for value in capture.target_uuid])
    visible = np.asarray([row.target_perceptible for row in capture.observation])
    view = NATIVE_CONTEXTUAL_BEHAVIOR_VIEW.bind(trace)
    steering = NATIVE_CONTEXTUAL_STEERING_VIEW.bind(trace)
    changes = trace.resolve("native.decision_change")
    return {
        "role": capture.role,
        "opponent_role": opponent.role,
        "outcome": outcome,
        "imitation_weight": 1.0 / capture.emitted_count,
        "artifact": artifact,
        "trace_contract_sha256": trace.contract.sha256,
        "trace_identity_sha256": trace.identity_sha256,
        "contextual_view_valid_rows": int(np.count_nonzero(view.valid)),
        "contextual_steering_valid_rows": int(np.count_nonzero(steering.valid)),
        "behavior": {
            "rows": capture.emitted_count,
            "native_control_rows": int(np.count_nonzero(capture.native_control)),
            "moving_rows": int(np.count_nonzero(movement > 1e-6)),
            "path_length": float(movement.sum()),
            "aim_change_rows": int(np.count_nonzero(aim_delta > 1e-5)),
            "opponent_lock_rows": int(np.count_nonzero(locked)),
            "opponent_visible_rows": int(np.count_nonzero(visible)),
            "decision_change_rows": int(np.count_nonzero(np.any(changes, axis=1))),
            "attack_activation_rows": int(np.count_nonzero(capture.attack_activation)),
            "attack_execution_rows": int(np.count_nonzero(capture.combat_attack)),
            "attack_execution_cause_available_rows": sum(
                cause.available for cause in capture.attack_execution_cause
            ),
            "attack_execution_cause_rows": sum(
                cause.executed for cause in capture.attack_execution_cause
            ),
            "damage_event_rows": int(np.count_nonzero(capture.damage_event_count)),
            "final_health": float(capture.next_state[-1, 11]),
            "minimum_health": float(np.min(capture.next_state[:, 11])),
            "states": dict(Counter(capture.state_name)),
            "body_instructions": dict(Counter(filter(None, capture.body_instruction))),
            "head_instructions": dict(Counter(filter(None, capture.head_instruction))),
            "active_action_labels": dict(
                Counter(label for row in capture.active_actions for label in row)
            ),
            "interaction_types": dict(
                Counter(
                    interaction.type
                    for row in capture.interactions
                    for interaction in row
                )
            ),
        },
        "examples": _examples(capture, trace, movement, aim_delta),
    }


def _examples(capture, trace, movement, aim_delta):
    candidates = np.flatnonzero(
        (movement > 1e-6) | (aim_delta > 1e-5) | capture.attack_activation
    )
    selected = (
        candidates[:3] if candidates.size else np.arange(min(1, capture.emitted_count))
    )
    phase = trace.resolve("native.combat.phase")
    return [
        {
            "tick": int(capture.tick[index]),
            "state": capture.state_name[index],
            "position": capture.state[index, :3].tolist(),
            "next_position": capture.next_state[index, :3].tolist(),
            "body_yaw": float(capture.state[index, 6]),
            "head_yaw_pitch": capture.state[index, 9:11].tolist(),
            "control": capture.control[index].tolist(),
            "control_mask": int(capture.control_mask[index]),
            "body_instruction": capture.body_instruction[index],
            "head_instruction": capture.head_instruction[index],
            "target_uuid": str(capture.target_uuid[index]),
            "target_perceptible": capture.observation[index].target_perceptible,
            "combat_phase": phase[index],
            "attack_activation": bool(capture.attack_activation[index]),
            "attack_execution_cause": bool(
                capture.attack_execution_cause[index].executed
            ),
        }
        for index in selected
    ]


def _role_assets(archive: Path, roles: list[str]) -> dict[str, Any]:
    with ZipFile(archive) as source:
        names = source.namelist()
        result = {}
        for role in roles:
            matches = [
                name
                for name in names
                if name.startswith("Server/NPC/Roles/")
                and name.endswith(f"/{role}.json")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one shipped role asset for {role}: {matches}"
                )
            payload = source.read(matches[0])
            result[role] = {
                "path": matches[0],
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
            }
        return result


def _unique_role_index(values, role):
    indexes = [index for index, value in enumerate(values) if value == role]
    if len(indexes) != 1:
        raise ValueError(f"expected one live {role}, found {len(indexes)}")
    return indexes[0]


def _roles(scenario):
    return scenario.actor_role, scenario.opponent_role


def _sha(value):
    return (
        hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
        .hexdigest()
        .upper()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--expected-bridge")
    parser.add_argument("--native-tick-rate", type=int, default=120)
    parser.add_argument("--native-time-dilation", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--capture-retries", type=int, default=3)
    parser.add_argument(
        "--matchup-name",
        action="append",
        choices=tuple(row[0] for row in _MATCHUPS),
        help="capture only a named admitted matchup; may be repeated",
    )
    parser.add_argument(
        "--probe-matchup",
        action="append",
        metavar="ACTOR:OPPONENT",
        help="probe Java hostility only; may be repeated",
    )
    args = parser.parse_args()
    if args.probe_matchup:
        matchups = tuple(_parse_matchup(value) for value in args.probe_matchup)
        report = probe_native_hostility(
            args.output,
            matchups,
            host=args.host,
            port=args.port,
            expected_bridge=args.expected_bridge,
        )
        print(json.dumps(report, sort_keys=True))
        return
    selected_matchups = tuple(
        row
        for row in _MATCHUPS
        if args.matchup_name is None or row[0] in args.matchup_name
    )
    report = capture_native_behavior_corpus(
        args.output,
        host=args.host,
        port=args.port,
        expected_bridge=args.expected_bridge,
        scenarios=native_behavior_scenarios(args.repeats, matchups=selected_matchups),
        native_tick_rate=args.native_tick_rate,
        native_time_dilation=args.native_time_dilation,
        capture_retries=args.capture_retries,
    )
    print(json.dumps(report["coverage"], sort_keys=True))


def _parse_matchup(value: str) -> tuple[str, str]:
    roles = tuple(part.strip() for part in value.split(":"))
    if len(roles) != 2 or not all(roles) or roles[0] == roles[1]:
        raise argparse.ArgumentTypeError("matchup must be ACTOR:OPPONENT")
    return roles


if __name__ == "__main__":
    main()
