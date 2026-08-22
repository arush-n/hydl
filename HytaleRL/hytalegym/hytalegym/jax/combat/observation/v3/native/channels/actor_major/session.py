"""Owning production session for the actor-major native host composer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoderParams,
)
from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.jax.crafting import PackedRecipeTable
from hytalegym.rulesets import load_combat_ruleset
from hytalegym.utils.connection import BridgeConnection

from .composer import (
    NativeActorMajorHostComposer,
    NativeActorMajorObservation,
    NativeActorMajorStepResult,
    native_actor_major_reset_options,
)


class NativeActorMajorSession:
    """Reset, infer, and step two policy actors on one BridgeConnection.

    This is the executable session path above
    :class:`NativeActorMajorHostComposer`.  It negotiates actor UUIDs at reset,
    constructs the per-slot adapters only after that ownership fact exists,
    and returns dense/mask rows ready for an actor-major policy loop.  The
    ordinary single-actor ``HytaleEnv`` is not modified or intercepted.
    """

    def __init__(
        self,
        *,
        actor_profiles: Mapping[int, str],
        host: str = "127.0.0.1",
        port: int = 5556,
        connection: object | None = None,
        ticks_per_step: int = 4,
        max_episode_steps: int = 6000,
        world: str = "flat",
        worldgen_structure: str = "Default",
        npc_role: str | None = None,
        fidelity_fixture: str = "default",
        native_tick_rate: int = 30,
        native_time_dilation: float = 1.0,
        packed_recipes: PackedRecipeTable | None = None,
        recipe_encoder_params: RecipeCandidateEncoderParams | None = None,
        initial_request_ids: Mapping[int, int] | None = None,
    ) -> None:
        if (
            isinstance(ticks_per_step, bool)
            or not isinstance(ticks_per_step, int)
            or ticks_per_step <= 0
        ):
            raise ValueError("ticks_per_step must be a positive integer")
        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")
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
            raise ValueError(
                "native_time_dilation must be finite and in (0.01, 4]"
            )
        self._actor_profiles = dict(actor_profiles)
        self._ticks_per_step = ticks_per_step
        self._max_episode_steps = max_episode_steps
        self._world = world
        self._worldgen_structure = worldgen_structure
        self._npc_role = (
            str(load_combat_ruleset()["matchup"]["agent_role"])
            if npc_role is None
            else npc_role
        )
        self._fidelity_fixture = fidelity_fixture
        self._native_tick_rate = native_tick_rate
        self._native_time_dilation = float(native_time_dilation)
        self._packed_recipes = packed_recipes
        self._recipe_encoder_params = recipe_encoder_params
        self._initial_request_ids = initial_request_ids
        self._connection = (
            BridgeConnection(host, port) if connection is None else connection
        )
        if not callable(getattr(self._connection, "send_and_recv", None)):
            raise TypeError("connection must expose callable send_and_recv")
        self._owns_connection = connection is None
        self._configured = False
        self._composer: NativeActorMajorHostComposer | None = None
        self._reset_contract = native_actor_major_reset_options(
            self._actor_profiles,
            ticks_per_step=ticks_per_step,
        )

    @property
    def composer(self) -> NativeActorMajorHostComposer:
        if self._composer is None:
            raise RuntimeError("reset must precede actor-major session access")
        return self._composer

    def reset(
        self,
        *,
        seed: int = 0,
        options: Mapping[str, object] | None = None,
    ) -> NativeActorMajorObservation:
        """Negotiate UUID ownership and return policy-ready actor-major rows."""

        self._ensure_configured()
        reset_options: dict[str, object] = {
            "backend": "native",
            "world": self._world,
            "worldgen_structure": self._worldgen_structure,
            "npc_role": self._npc_role,
            # The current group bridge owns both rows only in the passive
            # kill-trork fixture; advertise that live precondition explicitly.
            "combat_target_active": False,
            "fidelity_fixture": self._fidelity_fixture,
            "native_tick_rate": self._native_tick_rate,
            "native_time_dilation": self._native_time_dilation,
        }
        if options is not None:
            if not isinstance(options, Mapping):
                raise TypeError("reset options must be a mapping")
            reset_options.update(options)
        for name, expected in self._reset_contract.items():
            if name in reset_options and reset_options[name] != expected:
                raise ValueError(
                    f"reset option {name!r} conflicts with actor-major contract"
                )
            reset_options[name] = expected
        if reset_options.get("combat_target_active") is not False:
            raise ValueError(
                "actor-major group control requires combat_target_active=false"
            )
        response = self._connection.send_and_recv(
            {
                "type": "reset",
                "task_id": "kill_trork",
                "seed": int(seed),
                "options": reset_options,
            }
        )
        _require_observation(response)
        identities = _negotiated_identities(response)
        if self._composer is not None:
            self._composer.unbind()
        composer = NativeActorMajorHostComposer(
            self._connection,
            actor_profiles=self._actor_profiles,
            actor_identities={slot: value for slot, value in enumerate(identities)},
            packed_recipes=self._packed_recipes,
            recipe_encoder_params=self._recipe_encoder_params,
            initial_request_ids=self._initial_request_ids,
            ticks_per_step=self._ticks_per_step,
        )
        self._composer = composer
        return composer.accept_reset_response(response)

    def step(
        self,
        factors_by_slot: Mapping[int, Any],
    ) -> NativeActorMajorStepResult:
        """Execute the production composer path for one policy transition."""

        return self.composer.step(factors_by_slot)

    def close(self) -> None:
        """Close only the connection this session owns."""

        if self._composer is not None:
            self._composer.unbind()
            self._composer = None
        if self._owns_connection:
            if self._configured:
                try:
                    self._connection.send_and_recv({"type": "close"})
                except Exception:
                    pass
            close = getattr(self._connection, "close", None)
            if callable(close):
                close()
        self._configured = False

    def _ensure_configured(self) -> None:
        if self._configured:
            return
        if self._owns_connection:
            connect = getattr(self._connection, "connect", None)
            if not callable(connect):
                raise TypeError("owned BridgeConnection has no connect method")
            connect()
        response = self._connection.send_and_recv(
            {
                "type": "config",
                "tick_rate": self._ticks_per_step,
                "max_episode_steps": self._max_episode_steps,
            }
        )
        if not isinstance(response, Mapping) or response.get("type") != "ack":
            raise ConnectionError("bridge did not acknowledge configuration")
        self._configured = True


def _negotiated_identities(response: Mapping[str, Any]) -> tuple[str, ...]:
    info = response.get("info")
    if not isinstance(info, Mapping):
        raise TypeError("reset response has no native info mapping")
    value = info.get("native_policy_actor_identities")
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise TypeError("reset response has no actor identity array")
    result = tuple(value)
    if len(result) != ENTITY_COUNT or any(
        not isinstance(identity, str) or not identity
        for identity in result
    ):
        raise ValueError("reset actor identities differ from actor capacity")
    return result


def _require_observation(response: object) -> None:
    if not isinstance(response, Mapping) or response.get("type") != "observation":
        raise ConnectionError("bridge did not return an observation")


__all__ = ["NativeActorMajorSession"]
