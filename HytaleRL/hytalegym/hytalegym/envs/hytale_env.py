"""Core Gymnasium environment that communicates with the Hytale server plugin."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import operator
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hytalegym.audio import audio_space, parse_audio
from hytalegym.combat.telemetry import (
    COMBAT_STATE_HIGH,
    COMBAT_STATE_LOW,
    parse_combat_state,
)
from hytalegym.geometry import geometry_space, parse_geometry
from hytalegym.rulesets import load_combat_ruleset
from hytalegym.utils.connection import BridgeConnection

# Observation dimensions
INVENTORY_SIZE = 36
ENTITY_MAX = 20
NEARBY_BLOCK_MAX = 729  # 9^3 grid
NEARBY_BLOCK_RADIUS = 4
NEARBY_ENTITY_FEATURES = 4  # type, rel_x, rel_z, health
BLOCK_TYPE_COUNT = 29  # AIR(0) through STICKS(28), see BlockType.java
NATIVE_ASSET_ID_MAX = np.iinfo(np.int32).max
NEARBY_ENTITY_FIXED_POINT_SCALE = float(
    load_combat_ruleset()["observation_normalization"]["wire_fixed_point_scale"]
)

HOST_ISSUED_LEARNER_V3_ACTION_SCHEMA_ID = (
    "hytalegym-host-issued-learner-v3-action-v1"
)


def host_issued_learner_v3_action_contract_sha256() -> str:
    """Identify the host-owned exact issued-row evidence law."""

    from hytalegym.jax.combat.observation.v3.policy import (
        arsenal_policy_contract_sha256,
    )

    payload = {
        "schema": HOST_ISSUED_LEARNER_V3_ACTION_SCHEMA_ID,
        "arsenal_policy_contract_sha256": arsenal_policy_contract_sha256(),
        "fields": (
            "host_issued_learner_v3_action_factors",
            "host_issued_learner_v3_action_factors_available",
        ),
        "legal_learner_step": "exact_canonical_requested_factor_tuple_available",
        "host_rejection": "canonical_physical_noop_tuple_available",
        "reset_or_autonomous": "canonical_noop_placeholder_unavailable",
        "ownership": "host_overwrites_untrusted_server_info",
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest().upper()


class HytaleEnv(gym.Env):
    """
    Gymnasium environment for Hytale RL.

    Connects to a running Hytale server with the HytaleRLBridge plugin
    and provides step/reset/observe through a TCP MessagePack bridge.

    Args:
        task: Task ID to run (survive, mine_adamantite, kill_trork, build_house, navigate, base_builder)
        host: Bridge server host
        port: Bridge server port
        max_episode_steps: Maximum steps per episode
        ticks_per_step: Game ticks per RL step
        backend: ``simulator`` or the in-server ``native``/``headless`` backend
        world: Native world template: ``flat``, legacy ``hytale``, or V2
            ``hytale_generator`` worldgen
        worldgen_structure: V2 ``HytaleGenerator`` world-structure asset;
            ignored only at its default value for non-V2 worlds
        npc_role: Native Hytale NPC role asset to control
        combat_target_role: Optional native role for the opposing NPC. This is
            runtime/capture selection, not a learner action label.
        spawn: Optional native spawn position as ``(x, y, z)``
        combat_target_active: Whether the native kill-trork target fights back
        fidelity_fixture: Optional controlled native measurement fixture;
            ``target_memory`` requires a passive flag and runs natural target AI
        learner_v3_profile: Optional Hytale 0.5.7 Arsenal profile. When set,
            reset negotiates native actor evidence and observations/actions use
            the published learner-v3 dense/factored policy surface.
        learner_v3_target_profile: Optional target-side Arsenal profile.
        learner_v3_world_action_adapter: Optional privileged native candidate
            resolver. It owns actor-safe block/recipe/Use surfaces and fresh
            commit evidence; exact targets and String IDs stay host-only.
        native_tick_rate: Native world scheduler rate, from 1 to 2048 TPS.
        native_time_dilation: Server-authoritative simulation-time multiplier
            in ``(0.01, 4]``. Pair ``30 * speed`` TPS with ``speed`` dilation
            to fast-forward while retaining a 1/30-second logical tick.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        task: str = "survive",
        host: str = "127.0.0.1",
        port: int = 5556,
        max_episode_steps: int = 6000,
        ticks_per_step: int = 4,
        backend: str = "simulator",
        world: str = "flat",
        npc_role: str | None = None,
        combat_target_role: str | None = None,
        spawn: tuple[float, float, float] | None = None,
        combat_target_active: bool = True,
        fidelity_fixture: str = "default",
        learner_v3_profile: str | None = None,
        learner_v3_target_profile: str = "",
        learner_v3_world_geometry_config: Any | None = None,
        learner_v3_world_action_adapter: Any | None = None,
        render_mode: str | None = None,
        worldgen_structure: str = "Default",
        native_tick_rate: int = 30,
        native_time_dilation: float = 1.0,
    ):
        super().__init__()
        self.task = task
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.ticks_per_step = ticks_per_step
        self.backend = backend
        self.world = world
        if not isinstance(worldgen_structure, str) or not worldgen_structure.strip():
            raise ValueError("worldgen_structure must be a non-empty string")
        self.worldgen_structure = worldgen_structure.strip()
        if (
            isinstance(native_tick_rate, bool)
            or not isinstance(native_tick_rate, int)
            or not 1 <= native_tick_rate <= 2048
        ):
            raise ValueError("native_tick_rate must be an integer in 1..2048")
        if (
            isinstance(native_time_dilation, bool)
            or not isinstance(native_time_dilation, (int, float))
            or not np.isfinite(float(native_time_dilation))
            or not 0.01 < float(native_time_dilation) <= 4.0
        ):
            raise ValueError("native_time_dilation must be finite and in (0.01, 4]")
        if backend not in {"native", "headless"} and (
            native_tick_rate != 30 or float(native_time_dilation) != 1.0
        ):
            raise ValueError("native tick controls require a native backend")
        self.native_tick_rate = native_tick_rate
        self.native_time_dilation = float(native_time_dilation)
        if npc_role is not None:
            self.npc_role = npc_role
        elif task == "kill_trork":
            self.npc_role = str(load_combat_ruleset()["matchup"]["agent_role"])
        else:
            self.npc_role = "Trork_Unarmed"
        self.spawn = spawn
        if combat_target_role is not None:
            if not isinstance(combat_target_role, str) or not combat_target_role.strip():
                raise ValueError("combat_target_role must be a non-empty string")
            if task != "kill_trork" or backend not in {"native", "headless"}:
                raise ValueError(
                    "combat_target_role requires kill_trork and a native backend"
                )
            combat_target_role = combat_target_role.strip()
        self.combat_target_role = combat_target_role
        self.combat_target_active = combat_target_active
        self.fidelity_fixture = fidelity_fixture
        self.learner_v3_profile = learner_v3_profile
        self.learner_v3_target_profile = learner_v3_target_profile
        self._learner_v3_world_action_adapter = learner_v3_world_action_adapter
        self.render_mode = render_mode

        self._connection: BridgeConnection | None = None
        self._step_count = 0
        self._native_actor_assembler: Any | None = None
        self._native_actor_assembly: Any | None = None
        self._native_backend_info: dict[str, Any] = {}
        self._host_action_legal = True
        self._host_action_reject_reasons: tuple[str, ...] = ()
        self._host_resolved_native_ability_slot = -1
        self._learner_v3_noop_action_factors: tuple[int, ...] = ()
        self._host_issued_learner_v3_action_factors: tuple[int, ...] = ()
        self._host_issued_learner_v3_action_factors_available = False

        # Action space: base movement + extended NPC actions
        self.action_space = spaces.Dict(
            {
                # Movement
                "forward": spaces.Discrete(2),
                "back": spaces.Discrete(2),
                "left": spaces.Discrete(2),
                "right": spaces.Discrete(2),
                "jump": spaces.Discrete(2),
                "attack": spaces.Discrete(2),
                "use": spaces.Discrete(2),
                "guard_held": spaces.Discrete(2),
                "dodge_direction": spaces.Discrete(5),
                "ability_slot": spaces.Discrete(17, start=-1),
                "world_move_direction": spaces.Discrete(9),
                "requested_charge_time": spaces.Box(
                    0.0, np.inf, shape=(), dtype=np.float32
                ),
                "camera_delta_yaw": spaces.Box(
                    -180.0, 180.0, shape=(), dtype=np.float32
                ),
                "camera_delta_pitch": spaces.Box(
                    -90.0, 90.0, shape=(), dtype=np.float32
                ),
                "hotbar_slot": spaces.Discrete(9),
                # Targeted block placement (offset -3..3, type 0..BLOCK_TYPE_COUNT-1)
                "place_block_x": spaces.Discrete(7),  # mapped to -3..3
                "place_block_y": spaces.Discrete(7),
                "place_block_z": spaces.Discrete(7),
                "place_block_type": spaces.Discrete(BLOCK_TYPE_COUNT),  # 0=no placement
                # Targeted block break
                "break_block_x": spaces.Discrete(7),
                "break_block_y": spaces.Discrete(7),
                "break_block_z": spaces.Discrete(7),
                "break_block": spaces.Discrete(2),
                # Crafting
                "craft_recipe_id": spaces.Discrete(10),  # 0-8 recipes, 9=no craft
            }
        )

        # Observation space — reflects Hytale's player stats
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(-1e6, 1e6, shape=(3,), dtype=np.float64),
                "velocity": spaces.Box(-100, 100, shape=(3,), dtype=np.float64),
                "yaw": spaces.Box(-180, 180, shape=(), dtype=np.float64),
                "pitch": spaces.Box(-90, 90, shape=(), dtype=np.float64),
                "health": spaces.Box(0, 1e9, shape=(), dtype=np.float64),
                "food_buff_timer": spaces.Discrete(601),  # 0-600 ticks remaining
                "stamina": spaces.Box(0, 1e9, shape=(), dtype=np.float64),
                "mana": spaces.Box(0, 1e9, shape=(), dtype=np.float64),
                # Native 0.5.7 IDs are wider than the simulator's compact tables.
                "inventory": spaces.Box(
                    0,
                    NATIVE_ASSET_ID_MAX,
                    shape=(INVENTORY_SIZE,),
                    dtype=np.int64,
                ),
                # Nearby blocks: flattened grid [block_type_id, ...] for 9x9x9 cube
                "nearby_blocks": spaces.Box(
                    0,
                    NATIVE_ASSET_ID_MAX,
                    shape=(NEARBY_BLOCK_MAX,),
                    dtype=np.int32,
                ),
                # Nearby entities: [type, rel_x, rel_z, health] * max entities
                "nearby_entities": spaces.Box(
                    -1000,
                    1000,
                    shape=(ENTITY_MAX, NEARBY_ENTITY_FEATURES),
                    dtype=np.float32,
                ),
                "time_of_day": spaces.Discrete(24001),
                "craftable_count": spaces.Discrete(20),
                # Stable combat-chain telemetry emitted identically by the fast
                # simulator and native 0.5.7 server backend.
                "combat_state": spaces.Box(
                    COMBAT_STATE_LOW,
                    COMBAT_STATE_HIGH,
                    dtype=np.float32,
                ),
                # Versioned local collision/support/LOS state. This nested contract
                # is model-agnostic and fixed-shape even though the wire encoding is
                # sparse. Missing legacy data is explicitly marked unavailable.
                "geometry": geometry_space(),
                # Bounded, padding-first events plus per-transport capture support.
                # Missing legacy data is unavailable rather than apparent silence.
                "audio": audio_space(),
            }
        )
        if learner_v3_profile is not None:
            if backend not in {"native", "headless"}:
                raise ValueError(
                    "learner-v3 native evidence requires the native backend"
                )
            from hytalegym.jax.combat.observation.v3.native.evidence import (
                NativeActorEvidenceAssembler,
            )
            from hytalegym.jax.combat.observation.v3.native.channels.world_actions import (
                NativePolicyWorldActionAdapter,
            )
            from hytalegym.jax.combat.observation.v3.policy import (
                ARSENAL_POLICY_ACTION_HEAD_SIZES,
                arsenal_policy_observation_size,
                neutral_arsenal_policy_action_factors,
            )

            self._native_actor_assembler = NativeActorEvidenceAssembler(
                learner_v3_profile,
                target_profile=learner_v3_target_profile,
                ticks_per_step=ticks_per_step,
                world_geometry_config=learner_v3_world_geometry_config,
                native_perception_capture=self.capture_perception_channels,
            )
            if learner_v3_world_action_adapter is not None and not isinstance(
                learner_v3_world_action_adapter,
                NativePolicyWorldActionAdapter,
            ):
                raise TypeError(
                    "learner_v3_world_action_adapter must be "
                    "NativePolicyWorldActionAdapter or None"
                )
            if (
                learner_v3_world_action_adapter is not None
                and getattr(learner_v3_world_action_adapter, "actor_slot", 0)
                != 0
            ):
                raise ValueError(
                    "single-actor HytaleEnv requires a World-action adapter "
                    "for policy actor slot 0; actor-major environments must "
                    "own one adapter per policy row"
                )
            self.action_space = spaces.MultiDiscrete(
                np.asarray(
                    ARSENAL_POLICY_ACTION_HEAD_SIZES,
                    dtype=np.int64,
                )
            )
            self._learner_v3_noop_action_factors = tuple(
                int(value)
                for value in np.asarray(
                    neutral_arsenal_policy_action_factors(1)[0],
                    dtype=np.int32,
                )
            )
            self._host_issued_learner_v3_action_factors = (
                self._learner_v3_noop_action_factors
            )
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(
                    arsenal_policy_observation_size(
                        self._native_actor_assembler.world_geometry_config
                    ),
                ),
                dtype=np.float32,
            )
        elif learner_v3_world_action_adapter is not None:
            raise ValueError(
                "learner_v3_world_action_adapter requires learner_v3_profile"
            )

    @property
    def native_actor_assembly(self) -> Any | None:
        """Borrow the latest exact learner-v3 actor assembly.

        The assembly is produced by Gym's native evidence adapter and is the
        source of the returned dense observation and action mask. It remains
        ``None`` before a successful learner-v3 response (and for legacy
        environments). Callers that need structured actor input should retain
        this object rather than reconstructing it from the dense projection.
        """

        return self._native_actor_assembly

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict | np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._native_actor_assembly = None
        self._host_action_legal = True
        self._host_action_reject_reasons = ()
        self._host_resolved_native_ability_slot = -1
        self._clear_host_issued_learner_v3_action()
        if self._native_actor_assembler is not None:
            self._native_actor_assembler.reset()
        if self._learner_v3_world_action_adapter is not None:
            self._learner_v3_world_action_adapter.reset()

        if self._connection is None:
            self._connection = BridgeConnection(self.host, self.port)
            self._connection.connect()
            # Configure tick rate
            response = self._connection.send_and_recv(
                {
                    "type": "config",
                    "tick_rate": self.ticks_per_step,
                    "max_episode_steps": self.max_episode_steps,
                }
            )
            if response.get("type") != "ack":
                raise ConnectionError("Bridge did not acknowledge configuration")
        self._bind_learner_v3_world_action_adapter()

        reset_options = dict(options or {})
        reset_options.setdefault("backend", self.backend)
        reset_options.setdefault("world", self.world)
        reset_options.setdefault("worldgen_structure", self.worldgen_structure)
        reset_options.setdefault("npc_role", self.npc_role)
        if self.combat_target_role is not None:
            reset_options.setdefault("combat_target_role", self.combat_target_role)
        reset_options.setdefault("combat_target_active", self.combat_target_active)
        reset_options.setdefault("fidelity_fixture", self.fidelity_fixture)
        if self.backend in {"native", "headless"}:
            reset_options.setdefault("native_tick_rate", self.native_tick_rate)
            reset_options.setdefault("native_time_dilation", self.native_time_dilation)
        if self._native_actor_assembler is not None:
            negotiation = self._native_actor_assembler.negotiation_options
            for name, expected in negotiation.items():
                if name in reset_options and reset_options[name] != expected:
                    raise ValueError(
                        f"reset option {name!r} conflicts with learner-v3 "
                        "actor-evidence contract"
                    )
                reset_options[name] = expected
            for name, value in self._native_actor_assembler.reset_options.items():
                reset_options.setdefault(name, value)
        if self.spawn is not None:
            if len(self.spawn) != 3:
                raise ValueError("spawn must contain exactly x, y, and z")
            reset_options.setdefault("spawn_x", float(self.spawn[0]))
            reset_options.setdefault("spawn_y", float(self.spawn[1]))
            reset_options.setdefault("spawn_z", float(self.spawn[2]))

        response = self._connection.send_and_recv(
            {
                "type": "reset",
                "task_id": self.task,
                "seed": seed if seed is not None else 0,
                "options": reset_options,
            }
        )
        self._require_observation(response)

        return self._parse_policy_response(response)

    def step(
        self,
        action: dict | np.ndarray,
    ) -> tuple[dict | np.ndarray, float, bool, bool, dict]:
        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before step()")

        msg = (
            self._learner_v3_step_message(action)
            if self._native_actor_assembler is not None
            else self._legacy_step_message(action)
        )
        return self._send_step(msg)

    def step_native_npcs(
        self,
    ) -> tuple[dict | np.ndarray, float, bool, bool, dict]:
        """Advance with every Java NPC under its authored native behavior."""

        if self.backend not in {"native", "headless"}:
            raise gym.error.Error("Native NPC stepping requires a native backend")
        from hytalegym.jax.combat.observation.v3.native.channels.group import (
            native_autonomous_step_message,
        )

        self._host_resolved_native_ability_slot = -1
        self._host_action_legal = True
        self._host_action_reject_reasons = ()
        self._clear_host_issued_learner_v3_action()
        return self._send_step(native_autonomous_step_message().message)

    def _send_step(
        self,
        message: dict[str, object],
    ) -> tuple[dict | np.ndarray, float, bool, bool, dict]:
        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before step()")
        self._step_count += 1

        response = self._connection.send_and_recv(message)
        self._require_observation(response)

        obs, info = self._parse_policy_response(response)
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False))
        truncated = bool(response.get("truncated", False))

        return obs, reward, terminated, truncated, info

    def _legacy_step_message(self, action: Any) -> dict[str, Any]:
        if not isinstance(action, Mapping):
            raise TypeError("legacy HytaleEnv action must be a mapping")
        return {
            "type": "step",
            "action": {
                "forward": int(action.get("forward", 0)),
                "back": int(action.get("back", 0)),
                "left": int(action.get("left", 0)),
                "right": int(action.get("right", 0)),
                "jump": int(action.get("jump", 0)),
                "attack": int(action.get("attack", 0)),
                "use": int(action.get("use", 0)),
                "guard_held": int(action.get("guard_held", 0)),
                "dodge_direction": int(action.get("dodge_direction", 0)),
                "ability_slot": int(action.get("ability_slot", -1)),
                "world_move_direction": int(action.get("world_move_direction", 0)),
                "requested_charge_time": float(
                    action.get("requested_charge_time", 0.0)
                ),
                "camera_delta_yaw": float(action.get("camera_delta_yaw", 0.0)),
                "camera_delta_pitch": float(action.get("camera_delta_pitch", 0.0)),
                "hotbar_slot": int(action.get("hotbar_slot", 0)),
                "place_block_x": int(action.get("place_block_x", 3)) - 3,
                "place_block_y": int(action.get("place_block_y", 3)) - 3,
                "place_block_z": int(action.get("place_block_z", 3)) - 3,
                "place_block_type": int(action.get("place_block_type", 0)),
                "break_block_x": int(action.get("break_block_x", 3)) - 3,
                "break_block_y": int(action.get("break_block_y", 3)) - 3,
                "break_block_z": int(action.get("break_block_z", 3)) - 3,
                "break_block": int(action.get("break_block", 0)),
                "craft_recipe_id": (
                    -1
                    if int(action.get("craft_recipe_id", 9)) == 9
                    else int(action.get("craft_recipe_id", 9))
                ),
            },
        }

    def _learner_v3_step_message(self, factors: Any) -> dict[str, Any]:
        if self._native_actor_assembler is None or self._native_actor_assembly is None:
            raise gym.error.ResetNeeded("Call reset() before a learner-v3 step")
        action, decoded = self._native_actor_assembler.decode_policy_action(
            factors,
            self._native_actor_assembly,
        )
        values = np.asarray(factors, dtype=np.int32)
        head_sizes = np.asarray(self.action_space.nvec, dtype=np.int32)
        selected_legal: list[bool] = []
        offset = 0
        for value, size in zip(values, head_sizes, strict=True):
            selected_legal.append(
                bool(
                    self._native_actor_assembly.policy_action_mask[offset + int(value)]
                )
            )
            offset += int(size)
        reject_reasons = tuple(
            f"policy_head_masked:{index}"
            for index, legal in enumerate(selected_legal)
            if not legal
        )
        policy_legal = bool(np.asarray(decoded.valid)[0]) and not (reject_reasons)
        if not policy_legal and not reject_reasons:
            reject_reasons = ("policy_decode_rejected",)

        native_world_verb_request = None
        adapter = self._learner_v3_world_action_adapter
        if (
            policy_legal
            and adapter is not None
            and self._native_actor_assembly.action_surface_snapshot is not None
        ):
            resolution = adapter.resolve(
                action,
                self._native_actor_assembly.action_surface_snapshot,
                self._native_backend_info,
            )
            native_world_verb_request = resolution.request
            if not resolution.legal:
                policy_legal = False
                reject_reasons = tuple(
                    dict.fromkeys(reject_reasons + resolution.reject_reasons)
                )

        from hytalegym.jax.combat.observation.v3.native.codec.transport import (
            translate_learner_arsenal_action_to_native,
        )

        resolved_native_ability_slot = self._native_actor_assembler.native_ability_slot(
            action
        )
        translation = translate_learner_arsenal_action_to_native(
            action,
            decoded.low_level_action,
            self._native_backend_info,
            requested_charge_time_seconds=(
                self._native_actor_assembler.requested_charge_time_seconds(action)
            ),
            native_ability_slot=resolved_native_ability_slot,
            native_world_verb_request=native_world_verb_request,
            policy_legal=policy_legal,
            policy_reject_reasons=reject_reasons,
        )
        self._host_action_legal = translation.legal
        self._host_action_reject_reasons = translation.reject_reasons
        self._host_resolved_native_ability_slot = (
            resolved_native_ability_slot if translation.legal else -1
        )
        self._host_issued_learner_v3_action_factors = (
            tuple(int(value) for value in values)
            if translation.legal
            else self._learner_v3_noop_action_factors
        )
        self._host_issued_learner_v3_action_factors_available = True
        return {"type": "step", "action": translation.action}

    def _clear_host_issued_learner_v3_action(self) -> None:
        """Publish a canonical placeholder when no learner row was issued."""

        self._host_issued_learner_v3_action_factors = (
            self._learner_v3_noop_action_factors
        )
        self._host_issued_learner_v3_action_factors_available = False

    def _parse_policy_response(
        self,
        response: dict[str, Any],
    ) -> tuple[dict | np.ndarray, dict[str, Any]]:
        observation = self._parse_observation(response)
        info = self._parse_info(response)
        self._native_backend_info = info
        if self._native_actor_assembler is None:
            return observation, info
        raw_observation = response.get("obs")
        if not isinstance(raw_observation, Mapping):
            raise ValueError("native observation payload must be an object")
        assembly = self._native_actor_assembler.assemble(
            observation,
            raw_observation.get("native_actor_evidence"),
            info,
        )
        if self._learner_v3_world_action_adapter is not None:
            surface = self._learner_v3_world_action_adapter.surface(
                assembly,
                info,
            )
            assembly = self._native_actor_assembler.bind_policy_action_surface(
                assembly,
                surface,
                info,
            )
        self._native_actor_assembly = assembly
        info.update(
            {
                "learner_v3_profile": self.learner_v3_profile,
                "learner_v3_actor_evidence_contract_sha256": (
                    self._native_actor_assembler.negotiation_options[
                        "learner_v3_actor_evidence_contract_sha256"
                    ]
                ),
                "learner_v3_action_mask": (assembly.policy_action_mask.copy()),
                "learner_v3_observation_valid": bool(
                    np.asarray(assembly.observation.valid)[0]
                ),
                "learner_v3_failure_bits": int(
                    np.asarray(assembly.observation.failure_bits)[0]
                ),
                "learner_v3_mechanics_failure_bits": int(
                    np.asarray(assembly.observation.mechanics_failure_bits)[0]
                ),
                "learner_v3_arsenal_failure_bits": int(
                    np.asarray(assembly.observation.arsenal_failure_bits)[0]
                ),
                "learner_v3_loadout_failure": bool(
                    np.asarray(assembly.evidence.loadout_failure)[0]
                ),
                "host_policy_action_legal": self._host_action_legal,
                "host_policy_action_reject_reasons": (self._host_action_reject_reasons),
                "host_resolved_native_ability_slot": (
                    self._host_resolved_native_ability_slot
                ),
                "host_issued_learner_v3_action_factors": (
                    self._host_issued_learner_v3_action_factors
                ),
                "host_issued_learner_v3_action_factors_available": (
                    self._host_issued_learner_v3_action_factors_available
                ),
                "host_issued_learner_v3_action_contract_sha256": (
                    host_issued_learner_v3_action_contract_sha256()
                ),
            }
        )
        return assembly.policy_observation.copy(), info

    def capture_perception_channels(self, positions: np.ndarray):
        """Capture validated native light/environment rows at exact cells.

        This is a thin public host binding over the bridge's existing bounded
        ``perception_channels`` request. The World-owned encoder and decoder
        remain the single contract implementation; Combat callers use this
        method only to align those rows with actor-legal geometry tokens.
        """

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before channel capture")
        from hytalegym.worldgen import (
            native_perception_channel_capture_from_wire,
            native_perception_channel_request,
        )

        response = self._connection.send_and_recv(
            native_perception_channel_request(positions)
        )
        return native_perception_channel_capture_from_wire(response)

    def start_npc_trace(
        self,
        npc_uuid,
        *,
        expected_role: str | None = None,
        capacity: int = 4096,
    ):
        """Start a bounded native-behaviour trace for one exact NPC UUID."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before NPC tracing")
        from hytalegym.worldgen import (
            NativeNpcTraceCapture,
            native_npc_trace_start_request,
        )

        response = self._connection.send_and_recv(
            native_npc_trace_start_request(
                npc_uuid,
                expected_role=expected_role,
                capacity=capacity,
            )
        )
        return NativeNpcTraceCapture.from_response(response)

    def capture_privileged_npcs(self, bounds, *, capacity: int = 256):
        """Find exact NPC UUID, role, pose, and geometry rows in an AABB."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before NPC capture")
        from hytalegym.worldgen import (
            NativePrivilegedNpcCapture,
            native_privileged_npc_request,
        )

        response = self._connection.send_and_recv(
            native_privileged_npc_request(bounds, capacity=capacity)
        )
        return NativePrivilegedNpcCapture.from_response(response)

    def capture_privileged_npc_by_uuid(self, npc_uuid, bounds):
        """Read one NPC's role and geometry by exact UUID within an AABB."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before NPC capture")
        from hytalegym.worldgen import (
            NativePrivilegedNpcCapture,
            native_privileged_npc_by_uuid_request,
        )

        response = self._connection.send_and_recv(
            native_privileged_npc_by_uuid_request(npc_uuid, bounds)
        )
        return NativePrivilegedNpcCapture.from_uuid_response(
            response,
            npc_uuid,
            bounds,
        )

    def capture_privileged_entities(self, bounds, *, capacity: int = 256):
        """Find semantic-opaque UUID/pose/velocity rows in a native AABB."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before entity capture")
        from hytalegym.worldgen import (
            NativePrivilegedEntityCapture,
            native_privileged_entity_request,
        )

        response = self._connection.send_and_recv(
            native_privileged_entity_request(bounds, capacity=capacity)
        )
        return NativePrivilegedEntityCapture.from_response(response)

    def poll_npc_trace(self, trace_uuid, *, max_frames: int = 32):
        """Drain up to ``max_frames`` while leaving the trace active."""

        from hytalegym.worldgen import NativeNpcTraceCapture

        return NativeNpcTraceCapture.from_response(
            self.poll_npc_trace_wire(trace_uuid, max_frames=max_frames)
        )

    def poll_npc_trace_wire(self, trace_uuid, *, max_frames: int = 32):
        """Drain and return the exact replayable MessagePack response map."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before NPC tracing")
        from hytalegym.worldgen import native_npc_trace_poll_request

        return self._connection.send_and_recv(
            native_npc_trace_poll_request(trace_uuid, max_frames=max_frames)
        )

    def stop_npc_trace(self, trace_uuid, *, max_frames: int = 32):
        """Drain a final batch and remove the server-side trace marker."""

        from hytalegym.worldgen import NativeNpcTraceCapture

        return NativeNpcTraceCapture.from_response(
            self.stop_npc_trace_wire(trace_uuid, max_frames=max_frames)
        )

    def stop_npc_trace_wire(self, trace_uuid, *, max_frames: int = 32):
        """Stop and return the exact replayable MessagePack response map."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before NPC tracing")
        from hytalegym.worldgen import native_npc_trace_stop_request

        return self._connection.send_and_recv(
            native_npc_trace_stop_request(trace_uuid, max_frames=max_frames)
        )

    def capture_region_manifest(
        self,
        core_min_chunk_x: int | None = None,
        core_min_chunk_z: int | None = None,
    ) -> dict[str, Any]:
        """Select and return a native Region v1 manifest.

        With no coordinates, the server selects the 3x3 core surrounding the
        reset spawn. Supplying both coordinates selects an arbitrary valid
        core in the same active native world.
        """

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before region capture")
        if (core_min_chunk_x is None) != (core_min_chunk_z is None):
            raise ValueError(
                "core_min_chunk_x and core_min_chunk_z must be supplied together"
            )
        request: dict[str, Any] = {"type": "region_manifest"}
        if core_min_chunk_x is not None:
            request["core_min_chunk_x"] = _exact_int32(
                core_min_chunk_x,
                "core_min_chunk_x",
            )
            request["core_min_chunk_z"] = _exact_int32(
                core_min_chunk_z,
                "core_min_chunk_z",
            )
        response = self._connection.send_and_recv(request)
        if response.get("type") != "region_manifest":
            raise ConnectionError("Bridge did not return a region manifest")
        return response

    def capture_region_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> dict[str, Any]:
        """Capture one palette-compressed native 32-cubed section."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before region capture")
        response = self._connection.send_and_recv(
            {
                "type": "region_section",
                "chunk_x": int(chunk_x),
                "chunk_z": int(chunk_z),
                "section_y": int(section_y),
            }
        )
        if response.get("type") != "region_section":
            raise ConnectionError("Bridge did not return a region section")
        return response

    def capture_region_light_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> dict[str, Any]:
        """Capture one native light section with explicit readiness."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before region capture")
        response = self._connection.send_and_recv(
            {
                "type": "region_light_section",
                "chunk_x": _exact_int32(chunk_x, "chunk_x"),
                "chunk_z": _exact_int32(chunk_z, "chunk_z"),
                "section_y": _exact_int32(section_y, "section_y"),
            }
        )
        if response.get("type") != "region_light_section":
            raise ConnectionError("Bridge did not return a Region light section")
        return response

    def capture_region_block_semantic_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> dict[str, Any]:
        """Capture stable block identity aligned to one Region section."""

        if self._connection is None:
            raise gym.error.ResetNeeded("Call reset() before region capture")
        response = self._connection.send_and_recv(
            {
                "type": "region_block_semantics",
                "chunk_x": int(chunk_x),
                "chunk_z": int(chunk_z),
                "section_y": int(section_y),
            }
        )
        if response.get("type") != "region_block_semantics":
            raise ConnectionError("Bridge did not return Region block semantics")
        return response

    def close(self):
        self._unbind_learner_v3_world_action_adapter()
        if self._connection is not None:
            try:
                self._connection.send_and_recv({"type": "close"})
            except Exception:
                pass
            self._connection.close()
            self._connection = None

    def _bind_learner_v3_world_action_adapter(self) -> None:
        """Borrow this environment's socket for an opt-in capture adapter."""

        if self._connection is None:
            return
        adapter = self._learner_v3_world_action_adapter
        bind_bridge_request = (
            None if adapter is None else getattr(adapter, "bind_bridge_request", None)
        )
        if bind_bridge_request is not None:
            bind_bridge_request(self._connection.send_and_recv)

    def _unbind_learner_v3_world_action_adapter(self) -> None:
        """Release the borrowed callback before closing its owning socket."""

        if self._connection is None:
            return
        adapter = self._learner_v3_world_action_adapter
        unbind_bridge_request = (
            None if adapter is None else getattr(adapter, "unbind_bridge_request", None)
        )
        if unbind_bridge_request is not None:
            unbind_bridge_request(self._connection.send_and_recv)

    def _parse_observation(self, response: dict) -> dict:
        obs_data = response.get("obs", {})
        raw_info = response.get("info")
        step_duration_seconds = (
            raw_info.get("engine_simulated_seconds")
            if isinstance(raw_info, dict)
            else None
        )

        # Parse nearby blocks into fixed-size array
        raw_blocks = obs_data.get("nearby_blocks", [])
        nearby_blocks = np.zeros(NEARBY_BLOCK_MAX, dtype=np.int32)
        for i, block in enumerate(raw_blocks):
            if isinstance(block, (list, tuple)) and len(block) == 4:
                dx, dy, dz, block_type = (int(value) for value in block)
                if all(
                    -NEARBY_BLOCK_RADIUS <= value <= NEARBY_BLOCK_RADIUS
                    for value in (dx, dy, dz)
                ):
                    side = NEARBY_BLOCK_RADIUS * 2 + 1
                    index = (
                        (dx + NEARBY_BLOCK_RADIUS) * side * side
                        + (dy + NEARBY_BLOCK_RADIUS) * side
                        + (dz + NEARBY_BLOCK_RADIUS)
                    )
                    nearby_blocks[index] = block_type
            elif isinstance(block, int) and i < NEARBY_BLOCK_MAX:
                nearby_blocks[i] = block

        # Parse nearby entities into fixed-size array
        raw_entities = obs_data.get("nearby_entities", [])
        nearby_entities = np.zeros(
            (ENTITY_MAX, NEARBY_ENTITY_FEATURES), dtype=np.float32
        )
        for i, ent in enumerate(raw_entities):
            if i >= ENTITY_MAX:
                break
            if isinstance(ent, (list, tuple)) and len(ent) == 4:
                nearby_entities[i] = [
                    ent[0],
                    ent[1] / NEARBY_ENTITY_FIXED_POINT_SCALE,
                    ent[2] / NEARBY_ENTITY_FIXED_POINT_SCALE,
                    ent[3] / NEARBY_ENTITY_FIXED_POINT_SCALE,
                ]

        inventory = np.zeros(INVENTORY_SIZE, dtype=np.int64)
        raw_inventory = np.asarray(
            obs_data.get("inventory", []), dtype=np.int64
        ).reshape(-1)
        inventory[: min(INVENTORY_SIZE, raw_inventory.size)] = raw_inventory[
            :INVENTORY_SIZE
        ]

        raw_yaw = float(obs_data.get("yaw", 0))
        yaw = ((raw_yaw + 180.0) % 360.0) - 180.0

        return {
            "position": np.array(obs_data.get("position", [0, 0, 0]), dtype=np.float64),
            "velocity": np.array(obs_data.get("velocity", [0, 0, 0]), dtype=np.float64),
            "yaw": np.asarray(yaw, dtype=np.float64),
            "pitch": np.asarray(obs_data.get("pitch", 0), dtype=np.float64),
            "health": np.asarray(obs_data.get("health", 100), dtype=np.float64),
            "food_buff_timer": int(obs_data.get("food_buff_timer", 0)),
            "stamina": np.asarray(obs_data.get("stamina", 10), dtype=np.float64),
            "mana": np.asarray(obs_data.get("mana", 100), dtype=np.float64),
            "inventory": inventory,
            "nearby_blocks": nearby_blocks,
            "nearby_entities": nearby_entities,
            "time_of_day": int(obs_data.get("time_of_day", 6000)),
            "craftable_count": int(obs_data.get("craftable_count", 0)),
            "combat_state": parse_combat_state(response.get("info")),
            "geometry": parse_geometry(obs_data.get("geometry")),
            "audio": parse_audio(
                obs_data.get("audio"),
                step_duration_seconds=step_duration_seconds,
            ),
        }

    def _parse_info(self, response: dict) -> dict:
        raw_info = response.get("info", {})
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        raw_observation = response.get("obs")
        if (
            isinstance(raw_observation, Mapping)
            and "native_inventory" in raw_observation
        ):
            from hytalegym.jax.combat.inventory import (
                parse_native_inventory_frame,
            )

            info["native_inventory"] = parse_native_inventory_frame(
                raw_observation["native_inventory"]
            )
        info["step_count"] = self._step_count
        return info

    @staticmethod
    def _require_observation(response: dict) -> None:
        if response.get("type") != "observation":
            raise ConnectionError(
                f"Expected observation response, got {response.get('type')!r}"
            )


def _exact_int32(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if not np.iinfo(np.int32).min <= result <= np.iinfo(np.int32).max:
        raise ValueError(f"{label} exceeds int32")
    return result
