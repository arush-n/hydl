"""Scripted evader: a fixed rule on the 12-head surface, restricted to the mask.

Variant shipped: ``ScriptedEvaderSkill``, contract version v0001, no checkpoint.
The frozen evader network that the brief asked for is not on disk. The named run
``storage/runs/basic-20260821T174619Z-63e16a86`` holds only ``report.json`` and
``source-report.json``, and its checkpoint paths point at ``/mnt/c/...``, which
does not exist here. See SKILL.md for the evidence and the reasons.

Inputs read, from the shared belief only (``feature_index`` names):

* ``target_visible`` > 0.5, known
* ``target_bearing_sin``, ``target_bearing_cos``, both known

Rule, per lane ``b``. Let ``beta`` be the target bearing measured from body
forward, positive toward the right. The bearing is ``(sin beta, cos beta)``.

    active   = visible and both bearing bits known and sin^2 + cos^2 > 1e-6
    away     = atan2(-sin, -cos)                    retreat direction, radians
    index    = mod(round(away / (pi/4)), 8)         0 = forward, 4 = back
    base     = RETREAT (3) if active else IDLE (0)  fallback IDLE
    gait     = 9 + index if active else 0           fallback 0
    yaw, body yaw, pitch = centre bins; every other head = 0

Locomotion decode (``layout.py``): choice c >= 1 means gait = (c-1)//8 + 1 and
direction = (c-1)%8 + 1. Gait 2 is RUN, so the run choices are 9..16, and
choice 9 + index is RUN in direction index+1. ``apply_world_move_override``
replaces the four translation channels with the compass direction when the
choice is legal. A base RETREAT therefore adds no translation while a legal run
gait is active. It is the legacy retreat fallback when the gait is idle or
masked.

Restriction. Each head takes its first preference that the mask allows, in the
order shown above. If no preference is legal the lane is infeasible, and a
concrete call raises ``ValueError``. Under ``jax.jit`` the dispatcher reports
it in ``diagnostics["legal"]``.

Skill-local state. None. ``initial_state`` and every returned state are zeros.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from adk.architecture.inputs import ActorPolicyInput
from arena.training.skills.program import FactoredActionScope
from experimental.neural_agent.contracts import (
    SKILL_STATE_WIDTH,
    SharedBeliefState,
    SkillContract,
    feature_index,
)
from experimental.neural_agent.skill_library._common import (
    HEAD_NAMES,
    HEAD_OFFSETS,
    HEAD_SIZES,
    first_legal,
    neutral_choice,
    require_concrete_true,
    require_scoped_factors,
    validate_batch,
    validate_skill_inputs,
)
from hytalegym.jax.combat.observation.v1.schema.contract import LEARNER_ACTION_NAMES
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_COMPASS_DIRECTIONS,
)
from hytalegym.jax.combat.skills import SKILL_IDLE as BASE_IDLE
from hytalegym.jax.combat.skills import SKILL_RETREAT as BASE_RETREAT
from hytalegym.jax.combat.types import GAIT_RUN

EVADER_SKILL_ID: str = "evader"
EVADER_VERSION: str = "v0001"

# The base_action indices are pinned to their names in the v1 schema table.
if LEARNER_ACTION_NAMES[BASE_RETREAT] != "retreat" or LEARNER_ACTION_NAMES[BASE_IDLE] != "idle":
    raise RuntimeError("base_action indices drifted from LEARNER_ACTION_NAMES")

_BASE_HEAD: int = HEAD_NAMES.index("base_action")
_LOCOMOTION_HEAD: int = HEAD_NAMES.index("locomotion_gait_compass")
_COMPASS: int = ARSENAL_POLICY_COMPASS_DIRECTIONS
_COMPASS_STEP: float = 2.0 * math.pi / _COMPASS
# First RUN choice: 1 + (GAIT_RUN - 1) * 8 = 9. Direction 1 is body forward.
_RUN_BASE: int = 1 + (GAIT_RUN - 1) * _COMPASS
_VISIBLE: int = feature_index("target_visible")
_SIN: int = feature_index("target_bearing_sin")
_COS: int = feature_index("target_bearing_cos")
_MIN_BEARING_NORM_SQUARED: float = 1.0e-6


def _evader_rows() -> tuple[tuple[int, ...], ...]:
    """Return the allowed choices per head: the skill's declared emission set."""

    rows = []
    for name, size in zip(HEAD_NAMES, HEAD_SIZES, strict=True):
        if name == "base_action":
            rows.append((BASE_IDLE, BASE_RETREAT))
        elif name == "locomotion_gait_compass":
            rows.append((0, *range(_RUN_BASE, _RUN_BASE + _COMPASS)))
        else:
            rows.append((neutral_choice(name, size),))
    return tuple(rows)


#: The scope this skill may emit. Its contract hash is ``EVADER_SCOPE.contract_sha256``.
EVADER_SCOPE: FactoredActionScope = FactoredActionScope(_evader_rows())
_NEUTRAL_ROW: tuple[int, ...] = tuple(row[0] for row in EVADER_SCOPE.allowed_choices)


class ScriptedEvaderSkill:
    """Retreat from the target bearing with a run gait, on the 12-head surface.

    Implements the ``Skill`` protocol. It reads only the shared belief and the
    runtime action mask.
    """

    contract: SkillContract = SkillContract(
        skill_id=EVADER_SKILL_ID,
        version=EVADER_VERSION,
        observation_groups=("belief", "action_mask"),
        action_scope_sha256=EVADER_SCOPE.contract_sha256,
        param_names=(),
        termination=("goal_replaced",),
        success_signals=("separation_increases",),
        failure_signals=("target_not_visible",),
        checkpoint_sha256=None,
    )

    def initial_state(self, batch: int) -> jax.Array:
        """Return the zero skill state, float32[batch, SKILL_STATE_WIDTH].

        Args:
            batch: Number of lanes B, a positive int.
        """

        lanes = validate_batch(batch)
        return jnp.zeros((lanes, SKILL_STATE_WIDTH), dtype=jnp.float32)

    def __call__(
        self,
        state: jax.Array,
        belief: SharedBeliefState,
        actor_input: ActorPolicyInput,
        params: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(next_state, factors)`` with ``factors`` int32[B, 12].

        ``key`` is unused because the rule is deterministic.
        """

        lanes = validate_skill_inputs(
            state, belief, actor_input, params, skill_id=EVADER_SKILL_ID
        )
        mask = actor_input.action_mask
        visible = belief.feature_known[:, _VISIBLE] & (
            belief.features[:, _VISIBLE] > 0.5
        )
        bearing_known = belief.feature_known[:, _SIN] & belief.feature_known[:, _COS]
        sin = jnp.where(bearing_known, belief.features[:, _SIN], 0.0)
        cos = jnp.where(bearing_known, belief.features[:, _COS], 1.0)
        non_degenerate = (sin * sin + cos * cos) > _MIN_BEARING_NORM_SQUARED
        active = visible & bearing_known & non_degenerate
        away = jnp.where(active, jnp.arctan2(-sin, -cos), 0.0)
        index = jnp.mod(
            jnp.round(away / _COMPASS_STEP).astype(jnp.int32), _COMPASS
        )
        retreat_pref = jnp.where(active, BASE_RETREAT, BASE_IDLE).astype(jnp.int32)
        run_pref = jnp.where(active, _RUN_BASE + index, 0).astype(jnp.int32)
        idle_pref = jnp.zeros((lanes,), dtype=jnp.int32)
        columns = []
        feasible = jnp.ones((lanes,), dtype=jnp.bool_)
        for head, (offset, size) in enumerate(zip(HEAD_OFFSETS, HEAD_SIZES, strict=True)):
            if head == _BASE_HEAD:
                preferences = [retreat_pref, idle_pref]
            elif head == _LOCOMOTION_HEAD:
                preferences = [run_pref, idle_pref]
            else:
                preferences = [
                    jnp.full((lanes,), _NEUTRAL_ROW[head], dtype=jnp.int32)
                ]
            choice, ok = first_legal(preferences, mask[:, offset : offset + size])
            columns.append(choice)
            feasible = feasible & ok
        factors = jnp.stack(columns, axis=-1)
        require_concrete_true(
            feasible,
            "evader: a head has no legal choice (its preferences are all masked)",
        )
        require_scoped_factors(factors, mask, EVADER_SCOPE, "evader")
        return self.initial_state(lanes), factors


__all__ = [
    "EVADER_SCOPE",
    "EVADER_SKILL_ID",
    "EVADER_VERSION",
    "ScriptedEvaderSkill",
]
