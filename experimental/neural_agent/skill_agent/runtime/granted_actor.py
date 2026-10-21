"""The referee controls the body (2026-10-19; owner): the lower layer's actor runs only what the combiner granted.

Each tick the actor reads the combiner's grants as the active-request table (``combiner.ActiveRequestTable``: per
agent and scope, the granted verb, its arguments and the ticks since the grant) from its own state. The session writes
that state every tick through the host decider's ``on_tick`` seam (requested from the coordinator, 2026-10-19), and
``CombiningDecider`` builds the table from the combiner's grants (``grants_table``). For each scope the actor keeps the
choices of the body of the verb granted there on that scope's heads; a free scope's heads stay neutral. No stand-in
skill acts: the bodies are the skill layer's (``scripted_bodies.BODIES`` and the trained ``pursuit``), and the
decision layer's goal index is not read.

Every body runs on the whole batch each tick and the table selects per lane and scope, as the dispatcher selects its
skill: a static computation that ``jax.jit`` accepts. A body sees the ticks since its own grant. The heads are the
simulator's twelve (the gen-5 layout, which the gen-7 render keeps); a head belongs to the scope ``surface.scope_of``
names.

Reports (IDEAS.md idea 8): each tick every verb's body also reports ``scripted_bodies.REPORTS`` from the observation
and its own choices, as ``diagnostics["reports"]`` float32[B, V, R]. ``CombiningDecider`` writes a granted request's
reports onto its status on the blackboard, where the decision layer and the LLM read them.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from adk.architecture.components import ActorDecision
from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_of
from experimental.neural_agent.skill_agent.runtime.combiner import ActiveRequestTable, BodyStep, CombinedStep
from experimental.neural_agent.skill_agent.runtime.scripted_bodies import BODIES, POSITION_BODIES, Position, idle
from experimental.neural_agent.skill_agent.runtime.scripted_bodies import reports as body_reports
from hytalegym.jax.combat.observation.v3.policy import ARSENAL_POLICY_ACTION_HEAD_NAMES

ArrayBody = Callable[[Any, jax.Array, jax.Array, jax.Array, Position], jax.Array]
"""``body(observation, mask bool[B, 124], ticks int32[B], scalars float32[B, PARAM_WIDTH], position) -> factors
int32[B, 12]``; ``position`` is the granted request's world position (``scripted_bodies.Position``);
``scalars`` are the granted request's scalar arguments, in its contract's ``param_names`` order."""

HEAD_SCOPES = tuple(SCOPE_NAMES.index(scope_of(name)) for name in ARSENAL_POLICY_ACTION_HEAD_NAMES)
"""The scope index of each of the simulator's twelve heads."""


def free_table(batch: int, verbs: Sequence[str]) -> ActiveRequestTable:
    """A table with every scope free."""

    return ActiveRequestTable.from_grants([{} for _ in range(batch)], verbs, [{} for _ in range(batch)])


def grants_table(steps: Sequence[CombinedStep], live: Sequence[Mapping[int, Any]],
                 verbs: Sequence[str]) -> ActiveRequestTable:
    """The table of one tick's grants: per agent, its combiner step and its live requests by id. A granted request no
    longer live leaves its scope free."""

    rows = [{scope: requests[request_id] for scope, request_id in step.grants.items() if request_id in requests}
            for step, requests in zip(steps, live)]
    return ActiveRequestTable.from_grants(rows, verbs, [step.since for step in steps])


def granted_factors(table: ActiveRequestTable, observation: Any, mask: jax.Array,
                    bodies: Sequence[ArrayBody]) -> jax.Array:
    """The merged action int32[B, 12]: each head takes the choice of the body granted on its scope, or neutral."""

    return _granted(table, observation, mask, bodies)[0]


def _granted(table: ActiveRequestTable, observation: Any, mask: jax.Array,
             bodies: Sequence[ArrayBody]) -> tuple[jax.Array, list[jax.Array], jax.Array, list[Position]]:
    """The merged action, every body's own choices (in verb order), the neutral choices and every body's position."""

    outputs, positions = [], []
    lanes = jnp.arange(table.verb.shape[0])
    for verb, body in enumerate(bodies):
        scope = jnp.argmax(table.verb == verb, axis=1)  # the scope it is granted on (any one: they share a request)
        granted = table.verb[lanes, scope] == verb  # where it holds no scope, it has no request and no position
        positions.append(Position(table.position[lanes, scope], table.position_present[lanes, scope] & granted))
        outputs.append(body(observation, mask, table.ticks_since_grant[lanes, scope].astype(jnp.int32),
                            table.scalars[lanes, scope], positions[-1]))
    columns = []
    neutral = idle(observation, mask, table.ticks_since_grant[:, 0], table.scalars[:, 0])
    for head, scope in enumerate(HEAD_SCOPES):
        granted, choice = table.verb[:, scope], neutral[:, head]
        for verb, output in enumerate(outputs):
            choice = jnp.where(granted == verb, output[:, head], choice)
        columns.append(choice)
    return jnp.stack(columns, axis=1).astype(jnp.int32), outputs, neutral, positions


class GrantedActor:
    """The ADK ``LowLevelActor`` that runs the granted bodies (the module docstring gives the rule). Pass it to
    ``HierarchicalAgentCore(actor=...)``; its state is the active-request table."""

    def __init__(self, verbs: Sequence[str], bodies: Mapping[str, ArrayBody]) -> None:
        missing = [verb for verb in verbs if verb not in bodies]
        if missing:
            raise ValueError(f"no body for {missing}")
        self.verbs = tuple(verbs)
        self._bodies = tuple(bodies[verb] for verb in self.verbs)

    def initial_state(self, batch: int) -> ActiveRequestTable:
        return free_table(batch, self.verbs)

    def factors(self, table: ActiveRequestTable, observation: Any, mask: jax.Array) -> jax.Array:
        return granted_factors(table, observation, jnp.asarray(mask, dtype=jnp.bool_), self._bodies)

    def __call__(self, state: ActiveRequestTable, belief: Any, goal_payload: Any, actor_input: Any, key: jax.Array,
                 /) -> ActorDecision:
        observation = actor_input.legal_observation
        factors, outputs, neutral, positions = _granted(
            state, observation, jnp.asarray(actor_input.action_mask, jnp.bool_), self._bodies)
        reports = jnp.stack([body_reports(observation, output, neutral, position if verb in POSITION_BODIES else None)
                             for verb, output, position in zip(self.verbs, outputs, positions)], axis=1)
        lanes = factors.shape[0]
        zeros = jnp.zeros((lanes,), dtype=jnp.float32)
        return ActorDecision(state=state, action_factors=factors, logits=None, behavior_log_probability=zeros,
                             value=zeros, terminate_goal=jnp.zeros((lanes,), dtype=jnp.bool_),
                             diagnostics={"granted": state.verb >= 0, "reports": reports})


def continuous(request: Any, ticks_since_start: int, observation_tokens: Any, belief_tokens: Any) -> BodyStep:
    """The host combiner's body for a verb the granted actor drives: never done, no heads of its own (the actor acts).
    The request runs until the decision layer withdraws or replaces it, or it expires; success and failure signals
    from the body's own tests come later, with observation tokens on the host."""

    return BodyStep(factors={})


def referee_library(verbs: Sequence[str]) -> Any:
    """The skill library with ``continuous`` host bodies for ``verbs``, so the combiner keeps their grants."""

    from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary

    return SkillLibrary.from_catalogue(proof_of_concept=True, executors={verb: continuous for verb in verbs})


def skill_layer_actor(pursuit: Any | None = None) -> GrantedActor:
    """The actor over every skill-layer body: the scripted ones, and ``pursuit`` on the trained network when its
    ``NeuralPursuitSkill`` is given (``neural_pursuit.load_neural_pursuit()``)."""

    bodies: dict[str, ArrayBody] = dict(BODIES)
    if pursuit is not None:
        bodies["pursuit"] = pursuit.body
    return GrantedActor(tuple(bodies), bodies)


__all__ = ["HEAD_SCOPES", "ArrayBody", "GrantedActor", "continuous", "free_table", "granted_factors", "grants_table",
           "referee_library", "skill_layer_actor"]
