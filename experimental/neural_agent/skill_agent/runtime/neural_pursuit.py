"""The spine's pursuit (2026-10-19, the owner's choice A): the trained walk/look network as the decision layer's
``pursuit`` skill.

The decision layer speaks ``pursuit`` with an entity target (``experimental/neural_agent/requests.py``,
``ARGUMENT_SCHEMAS``); the skill layer's trained skills are ``walk_to`` and ``look_at``. This skill composes them on
the network of the A2 C' run (``agent_skills/training_runs/skill_agent_a2/a2cp_6x128x2000_9b9169e``, the generalist
that passed every A2 bar): on each tick its walk_to slot asks to walk to the target's observed position at a run and
its look_at slot to look at the target, both from the legal observation only, and it acts by the run's own student
rule (``training/distill.py``, ``student_factors``): the masked argmax on the locomotion and gaze heads, neutral on
every other head, inside the stand-in pursuit's action scope. It implements the ``skill_library`` ``Skill`` protocol,
so the dispatcher runs it in the pursuit slot. With no target in view both slots are idle and the action is neutral.

Frames. The observation publishes the head's absolute yaw and pitch, never the body yaw (``docs/PERCEPTION_ASKS.md``,
ask 8), but it publishes the target twice: in the body frame (the nearby-entity row) and in the head frame (the combat
prefix's ``visible_target_forward`` and ``visible_target_right``). The difference of the two bearings is the head's
yaw minus the body's, so ``body_yaw`` recovers the body yaw from published fields alone while the target is visible:
exactly, over 960 visible samples of the flat world with the head up to 45 degrees off the body (``docs/LEDGER.md``,
2026-10-19). The head yaw alone is no stand-in: in the same run it sat a mean 35 degrees off the body. With no target
in view the offset is taken as 0, and the action is neutral anyway. The derivation assumes the prefix's target is
the entity in slot 0, as on the flat worlds' two-entity path.

Requests. The trained request builders take world goals and a pose. The skill gives them the target turned out of
the body frame with the body yaw, and a pose at the origin, so what they compute is the observed body-frame vector
(walk_to) and the direction in the head's frame (look_at). The goals stay in the training ranges: the run gait, a
0.5-block radius (a catch is a distance below one block), the 3-degree look tolerance; the look target is the
target's position raised by the agent's eye height. The age feature counts the skill's ticks since it was entered
(the dispatcher resets the state on entry).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import ACTION_HEADS, SKILL_STATE_WIDTH, SharedBeliefState, SkillContract
from experimental.neural_agent.skill_agent.bench.bench_tick import CAPACITIES
from experimental.neural_agent.skill_agent.contract.surface import EntityToken
from experimental.neural_agent.skill_agent.model.encoder import EncoderConfig
from experimental.neural_agent.skill_agent.model.generalist import GeneralistConfig
from experimental.neural_agent.skill_agent.model.observation_adapter import legal_tokens
from experimental.neural_agent.skill_agent.neural_skills.look_at.task import LOOK_TOLERANCE_DEGREES, LookAtGoals
from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToGoals
from experimental.neural_agent.skill_agent.tasks.simple_tasks import AGENT_EYE_HEIGHT
from experimental.neural_agent.skill_agent.training import distill
from experimental.neural_agent.skill_agent.training.blocks import load_params
from experimental.neural_agent.skill_library._common import (
    require_scoped_factors,
    require_shape,
    validate_batch,
    validate_skill_inputs,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    COMBAT_FLOAT_FEATURES,
    ENTITY_FLOAT_FEATURES,
    SELF_FLOAT_FEATURES,
)
from hytalegym.jax.combat.types import GAIT_RUN

PURSUIT_SKILL_ID = "pursuit"
RUN_DIR = Path(__file__).resolve().parents[4] / "agent_skills/training_runs/skill_agent_a2/a2cp_6x128x2000_9b9169e"
"""The A2 C' run (``docs/LEDGER.md``, 2026-10-19): its ``params.npz`` and ``report.json``."""
RADIUS_BLOCKS = 0.5
_SELF = {name: index for index, name in enumerate(SELF_FLOAT_FEATURES)}
_ENTITY = {name: index for index, name in enumerate(ENTITY_FLOAT_FEATURES)}
_COMBAT = {name: index for index, name in enumerate(COMBAT_FLOAT_FEATURES)}
_FORWARD = EntityToken.FEATURES.index("forward")
_HORIZON = 1 << 20  # the request builders read a goal's horizon only to size episodes; pursuit has none


def run_configs(run_dir: Path = RUN_DIR) -> tuple[EncoderConfig, GeneralistConfig, dict]:
    """The trunk's and the generalist's configurations, and the run's whole config, from its ``report.json``."""

    config = json.loads((run_dir / "report.json").read_text())["config"]
    encoder = EncoderConfig(**{**config["encoder"],
                               "token_types": tuple(tuple(entry) for entry in config["encoder"]["token_types"])})
    return encoder, GeneralistConfig(**config["generalist"]), config


def _world(body: jax.Array, yaw_degrees: jax.Array) -> jax.Array:
    """Body-frame (forward, right, up) vectors [B, 3] turned into world offsets with ``yaw_degrees`` [B]."""

    yaw = jnp.deg2rad(yaw_degrees)
    forward, right, up = body[:, 0], body[:, 1], body[:, 2]
    return jnp.stack([-forward * jnp.sin(yaw) + right * jnp.cos(yaw), up,
                      -forward * jnp.cos(yaw) - right * jnp.sin(yaw)], axis=1)


def body_yaw(observation: Any) -> jax.Array:
    """The body yaw in degrees [B] from published fields: the head yaw minus the head's offset from the body, which is
    the target's head-frame bearing minus its body-frame bearing while the target is visible, else 0."""

    row = observation.base.self_f32
    head_yaw = jnp.rad2deg(jnp.arctan2(row[:, _SELF["yaw_sin"]], row[:, _SELF["yaw_cos"]]))
    entity, prefix = observation.base.entity_f32[:, 0], observation.base.combat_f32
    forward, right = entity[:, _ENTITY["relative_forward"]], entity[:, _ENTITY["relative_right"]]
    body_bearing = jnp.arctan2(right, forward)
    head_bearing = jnp.arctan2(prefix[:, _COMBAT["visible_target_right"]], prefix[:, _COMBAT["visible_target_forward"]])
    visible = (prefix[:, _COMBAT["target_visible"]] > 0.5) & (jnp.hypot(forward, right) > 1e-6)
    offset = jnp.where(visible, jnp.rad2deg(jnp.arctan2(jnp.sin(head_bearing - body_bearing),
                                                        jnp.cos(head_bearing - body_bearing))), 0.0)
    return head_yaw - offset


def pursuit_factors(params: dict, observation: Any, legal: jax.Array, ticks: jax.Array, *, encoder: EncoderConfig,
                    generalist: GeneralistConfig) -> jax.Array:
    """The skill's factors int32[B, 12] for one tick, from the learner observation, the legal mask bool[B, 124]
    (the runtime mask inside the scope) and the ticks since the skill was entered, int32[B]."""

    row = observation.base.self_f32
    head_yaw = jnp.rad2deg(jnp.arctan2(row[:, _SELF["yaw_sin"]], row[:, _SELF["yaw_cos"]]))
    head_pitch = row[:, _SELF["pitch"]] * 90.0  # the observation publishes pitch / 90
    yaw = body_yaw(observation)
    tokens = legal_tokens(observation, capacities=CAPACITIES, body_yaw_degrees=yaw)
    entity, seen = tokens["entity"][0][:, 0, _FORWARD:_FORWARD + 3], tokens["entity"][1][:, 0]
    target = _world(entity, yaw)
    batch = target.shape[0]
    walk = WalkToGoals(target_xz=target[:, (0, 2)], radius=jnp.full((batch,), RADIUS_BLOCKS, jnp.float32),
                       gait=jnp.full((batch,), GAIT_RUN, jnp.int32), horizon_ticks=jnp.full((batch,), _HORIZON))
    look = LookAtGoals(target_xyz=target + jnp.asarray([0.0, AGENT_EYE_HEIGHT, 0.0], jnp.float32),
                       tolerance_degrees=jnp.full((batch,), LOOK_TOLERANCE_DEGREES, jnp.float32),
                       horizon_ticks=jnp.full((batch,), _HORIZON))
    episode = distill.make_episodes(walk, look, jnp.full((batch,), distill.BOTH, jnp.int32), yaw)
    episode = episode._replace(walk_active=seen, look_active=seen)
    age = ticks.astype(jnp.float32)[:, None]  # per lane: each lane entered the skill on its own tick
    requests = distill.build_requests(episode, jnp.zeros((batch, 3), jnp.float32), yaw, age, head_yaw,
                                      head_pitch, look_direction=True, look_frame="head")
    return distill.student_factors(params, legal, tokens, requests, seen, seen, encoder, generalist)


class NeuralPursuitSkill:
    """The ``pursuit`` slot's skill on the trained network. Implements the ``skill_library`` ``Skill`` protocol."""

    def __init__(self, contract: SkillContract, act: Callable[..., jax.Array], scope: Any) -> None:
        self.contract = contract
        self._act = act
        self._scope = scope

    def initial_state(self, batch: int) -> jax.Array:
        """Zeros float32[batch, SKILL_STATE_WIDTH]; column 0 counts the ticks since the skill was entered."""

        return jnp.zeros((validate_batch(batch), SKILL_STATE_WIDTH), dtype=jnp.float32)

    def body(self, observation: Any, mask: jax.Array, ticks: jax.Array, scalars: jax.Array | None = None) -> jax.Array:
        """The skill as an array body (``granted_actor.py``): factors int32[B, 12] from the learner observation, the
        flat action mask bool[B, 124] (restricted here to the scope) and the ticks since its grant, int32[B]; pursuit
        takes no scalar arguments."""

        return self._act(observation, self._scope.restrict(jnp.asarray(mask, dtype=jnp.bool_)), ticks)

    def __call__(self, state: jax.Array, belief: SharedBeliefState, actor_input: ActorPolicyInput, params: jax.Array,
                 key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One tick: ``(next_state, factors)``, float32[B, SKILL_STATE_WIDTH] and int32[B, 12]."""

        lanes = validate_skill_inputs(state, belief, actor_input, params, skill_id=PURSUIT_SKILL_ID)
        legal = self._scope.restrict(jnp.asarray(actor_input.action_mask, dtype=jnp.bool_))
        factors = self._act(actor_input.legal_observation, legal, state[:, 0].astype(jnp.int32))
        require_shape(factors, (lanes, ACTION_HEADS), "neural pursuit: factors")
        require_scoped_factors(factors, actor_input.action_mask, self._scope, "neural pursuit")
        return state.at[:, 0].add(1.0), factors


def load_neural_pursuit(run_dir: Path = RUN_DIR) -> NeuralPursuitSkill:
    """The skill on ``run_dir``'s network, under the stand-in pursuit's action scope and termination terms, so the
    decision layer's expectations of ``pursuit`` hold."""

    from arena.training.contracts.pursuit import pursuit_learner_action_scope

    encoder, generalist, config = run_configs(run_dir)
    template = distill.init_params(jax.random.PRNGKey(0), distill.DistillConfig(
        encoder=encoder, generalist=generalist, look_direction=config["look_direction"],
        look_frame=config["look_frame"]))
    params_path = run_dir / "params.npz"
    params = load_params(params_path, template)
    scope = pursuit_learner_action_scope(allow_jump=False)
    contract = SkillContract(
        skill_id=PURSUIT_SKILL_ID,
        version=f"neural-{run_dir.name}",
        observation_groups=("legal_observation", "action_mask"),
        action_scope_sha256=scope.contract_sha256,
        param_names=(),
        termination=("distance_below_one_block", "horizon_256_ticks"),
        success_signals=("distance_below_one_block",),
        failure_signals=("horizon_256_ticks",),
        checkpoint_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
        minimum_commitment_seconds=0.5,
        maximum_duration_seconds=10.0,
        emergency_events=("DAMAGE_RECEIVED", "TARGET_DIED", "NPC_STUCK"),
    )
    act = jax.jit(lambda observation, legal, ticks: pursuit_factors(params, observation, legal, ticks,
                                                                    encoder=encoder, generalist=generalist))
    return NeuralPursuitSkill(contract, act, scope)


__all__ = ["PURSUIT_SKILL_ID", "RUN_DIR", "NeuralPursuitSkill", "body_yaw", "load_neural_pursuit", "pursuit_factors",
           "run_configs"]
