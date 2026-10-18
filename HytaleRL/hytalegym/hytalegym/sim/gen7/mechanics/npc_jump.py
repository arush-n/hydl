"""NPC learner first-step jump velocity (0.7.2). Switch ``Constructs.npc_walk_jump_velocity``; phase 3b.

HEAD gives every motion model the player's ``PLAYER_JUMP_FORCE`` 11.8 (``jax/combat/types.py``,
``default_combat_params``), which apexes at 2.18 m. The native NPC walk controller's first jump step
is ``sqrt(2 * jump_velocity_gravity_floor * jump_height_parameter)`` = sqrt(2 * 10.0 * 1.3) =
5.0990195 m/s, the NPC ruleset values (``rulesets/hytale_0_5_7``); the 0.7.2 NPC motion parameters
are identical to 0.5.10 (TRAINING-0.7.2.md item 6), and the 0.7.2 Walk controller of the Kweebec
template has ``Gravity`` 10 (``Server/NPC/Roles/Intelligent/Faction/Kweebec/Templates/
Template_Kweebec_Razorleaf.json``). Measured on the live bridge: native first-step vertical speed
5.09901951359 (``receipts/r4_jump_control.json``; FIDELITY-0.7.2.md row 1c).

Applies to ``motion_model="npc_walk"`` only: the player model (the skills' training model) keeps
``PLAYER_JUMP_FORCE``. Gen 5 keeps ``player_force`` (the original function, returned unchanged).
The construct is the render's hook on ``default_combat_params``.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "npc_walk_jump_velocity"
IMPLEMENTED = True
PLAYER_FORCE = "player_force"
NATIVE_NPC = "native_npc"


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; the native NPC jump velocity for ``npc_walk`` when the switch selects it."""

    value = profile.constructs.npc_walk_jump_velocity
    if value == PLAYER_FORCE:
        return original
    if value != NATIVE_NPC:
        raise ValueError(f"unknown npc_walk_jump_velocity switch {value!r}")
    env = original.__globals__
    jnp = env["jnp"]
    native = _native_npc_jump_velocity(env)

    @wraps(original)
    def default_combat_params(**kwargs: Any) -> Any:
        params = original(**kwargs)
        if kwargs.get("motion_model", "player") != "npc_walk":
            return params
        return params._replace(agent_jump_velocity=jnp.asarray(native, dtype=jnp.float32))

    return default_combat_params


def _native_npc_jump_velocity(env: dict) -> float:
    agent = env["load_combat_ruleset"]()["agent"]
    return env["math"].sqrt(2.0 * agent["jump_velocity_gravity_floor"] * agent["jump_height_parameter"])
