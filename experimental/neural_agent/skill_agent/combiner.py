"""The fixed greedy combiner: per tick and per agent, live requests -> one factored action and statuses.

``docs/SYSTEM_ONE.md`` section 9.10 (the five steps), ``skill_agent/BUILD.md`` 3.3 and the gates R1 to
R6; ``AGENT_BRIEF.md`` A5. Fixed and exact: nothing is learned here. The thinking inside a verb is its
body's own. The records and the check are the middle layer's (``experimental/neural_agent/requests.py``)
plus ``protocol.check``.

One tick ``t`` for one agent::

    1. live      the agent's requests not finished, with t < expires_tick. A request that was seen
                 unfinished and has now expired is finished with aborted("expired: ..."); one that
                 left the stream is finished with aborted("withdrawn: ...").
    2. holders   a request granted earlier keeps its scopes while it stays live and valid; a holder
                 that fails the check is aborted with the check's reason.
    3. waiting   a request in mode after:<p> whose predecessor p is live and unfinished is
                 deferred(conflict_with=p) and not checked yet (R4).
    4. check     every other request: protocol.check; a reason -> rejected(reason).
    5. grant     interrupt requests first, then the rest, each group in descending order of
                     key(r) = (urgency, confidence, priority, issued_tick, request_id)
                 so the newer request wins a tie (recency). A request is granted when every head of
                 its scopes is free. An interrupt request also takes heads from holders granted on an
                 earlier tick, which are aborted("interrupted: by request <id>"); a request whose heads
                 stay taken is deferred(conflict_with=<the holder of its first taken head>).
    6. run       each granted request's body, body(request, ticks_since_start, observation_tokens,
                 belief_tokens) with the tokens of the agent's context, fills its scopes' heads;
                 every other head is neutral.
    7. stream    accepted on the grant tick and running after, each with the heads it drove;
                 done(outcome) when its body reports done (its scopes are free on the next tick);
                 the tick's head composition (head -> request id).

The in-graph form for a compiled runner is the active-request table: per agent and per scope, the one
granted request (its id, verb index, typed arguments with presence flags, ticks since its grant).
``ActiveRequestTable.from_grants`` lowers the host's grants; ``table_factors`` computes the merged
action from the table with array bodies under ``jax.jit``. R6 checks that both give the same action.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent import requests as protocol
from experimental.neural_agent.contracts import PARAM_WIDTH, SkillOutcome
from experimental.neural_agent.skill_agent.protocol import VerbSpec, check, token_face
from experimental.neural_agent.skill_agent.surface import (
    FACES,
    SCOPE_NAMES,
    head_layout,
    legal_choices,
    scope_heads,
)

ActionRequest = protocol.ActionRequest
RequestStatus = protocol.RequestStatus


class BodyStep(NamedTuple):
    """What a body returns for one tick: choices for its heads, and whether the verb is done."""

    factors: Mapping[str, int]
    done: bool = False
    outcome: SkillOutcome = SkillOutcome.COMPLETED
    progress: Mapping[str, float] | None = None


Body = Callable[[ActionRequest, int, Any, Any], BodyStep]
"""A body: ``(request, ticks_since_start, observation_tokens, belief_tokens) -> BodyStep``.

``ticks_since_start`` counts from the request's grant (0 on the grant tick). The two token arguments are
the agent's ``AgentContext.observation_tokens`` and ``belief_tokens`` of this tick, passed through
untouched, so a body is a pure function of its arguments (testable with fixtures; the same contract as
the compiled runner). A body may fill only its request's scopes' heads.
"""


@dataclass(frozen=True)
class AgentContext:
    """What the check needs about one agent on one tick, and the tokens its bodies read.

    ``observation_tokens`` and ``belief_tokens`` are the agent's token sets for this tick (the surface
    token types; the belief section's tokens). The combiner does not read them: it hands both to every
    body it runs for the agent. None means the caller supplied none, and a body that needs them fails.
    """

    world_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    known_entities: frozenset[int] = frozenset()
    legal_scopes: frozenset[str] = frozenset(SCOPE_NAMES)
    observation_tokens: Any = None
    belief_tokens: Any = None


class CombinedStep(NamedTuple):
    """The combiner's output for one agent on one tick.

    ``grants`` is scope -> the request that held it when this tick's action was made (a request that
    finished this tick is still listed); ``since`` is request id -> ticks since its grant, for those.
    """

    factors: tuple[int, ...]
    statuses: tuple[RequestStatus, ...]
    composition: Mapping[str, int]
    grants: Mapping[str, int]
    since: Mapping[int, int]


@dataclass
class _AgentState:
    holders: dict[str, int] = field(default_factory=dict)
    unscoped: set[int] = field(default_factory=set)  # granted requests whose verb owns no scope (idle)
    done: set[int] = field(default_factory=set)  # ids that finished done: the after:<p> predecessors
    granted_at: dict[int, int] = field(default_factory=dict)
    finished: dict[int, str] = field(default_factory=dict)
    seen: dict[int, ActionRequest] = field(default_factory=dict)


def order_key(request: ActionRequest) -> tuple[float, float, float, int, int]:
    """The greedy key, compared in descending order: urgency, confidence, priority, recency."""

    return (request.urgency, request.confidence, request.priority, request.issued_tick, request.request_id)


class Combiner:
    """The host combiner. One instance carries every agent's grants across ticks.

    Args:
        specs: Verb id -> VerbSpec (``protocol.verb_specs`` over the catalogue, or a test's own).
        bodies: Verb id -> body. A granted verb without a body is aborted ("no_body: ...").
        layout: The head layout the action covers, ``"gen5"`` (today's twelve heads) or ``"gen7"``.
    """

    def __init__(self, specs: Mapping[str, VerbSpec], bodies: Mapping[str, Body], *, layout: str = "gen5") -> None:
        self.specs = dict(specs)
        self.bodies = dict(bodies)
        self.layout = layout
        self.heads = head_layout(layout)
        self.head_names = tuple(head.name for head in self.heads)
        self.neutral = tuple(head.neutral for head in self.heads)
        self._scope_heads = {scope: scope_heads(scope, layout) for scope in SCOPE_NAMES}
        self._agents: dict[int, _AgentState] = {}

    def heads_of(self, request: ActionRequest) -> tuple[str, ...]:
        """The present heads of the request's verb's scopes (none for an unknown verb)."""

        spec = self.specs.get(request.verb)
        scopes = () if spec is None else spec.scopes
        return tuple(head for scope in scopes for head in self._scope_heads[scope])

    def grants(self, agent: int) -> dict[str, int]:
        """Scope -> the granted request id, for the agent's current holders."""

        state = self._agents.get(agent)
        return {} if state is None else dict(state.holders)

    def step(self, tick: int, agent: int, requests: Sequence[ActionRequest], context: AgentContext) -> CombinedStep:
        """Combine one agent's request stream on one tick (the module docstring gives the steps)."""

        state = self._agents.setdefault(agent, _AgentState())
        statuses: list[RequestStatus] = []
        current = {request.request_id: request for request in requests if request.agent == agent}

        def finish(request_id: int, kind: str, **fields: Any) -> None:
            statuses.append(RequestStatus(agent=agent, request_id=request_id, kind=kind, tick=tick, **fields))
            state.finished[request_id] = kind
            for scope in [scope for scope, holder in state.holders.items() if holder == request_id]:
                del state.holders[scope]
            state.unscoped.discard(request_id)
            state.granted_at.pop(request_id, None)
            if kind == "done":
                state.done.add(request_id)

        for request_id, request in list(state.seen.items()):
            if request_id in state.finished:
                continue
            if request_id not in current:
                finish(request_id, "aborted", reason=f"withdrawn: request {request_id} ({request.verb}) left the stream")
            elif tick >= request.expires_tick:
                finish(request_id, "aborted",
                       reason=f"expired: request {request_id} ({request.verb}) expired at tick {request.expires_tick}")
        live = {rid: request for rid, request in current.items()
                if rid not in state.finished and tick < request.expires_tick}
        state.seen.update(live)
        done_ids = state.done

        def reason_for(request: ActionRequest) -> str | None:
            return check(request, self.specs.get(request.verb), tick=tick, world_bounds=context.world_bounds,
                         known_entities=context.known_entities, legal_scopes=context.legal_scopes,
                         done_ids=done_ids)

        for request_id in sorted(set(state.holders.values()) | state.unscoped):
            reason = reason_for(live[request_id]) if request_id in live else None
            if reason is not None and protocol.rejection_code(reason) != "predecessor_not_done":
                finish(request_id, "aborted", reason=reason)

        held = set(state.holders.values()) | state.unscoped
        candidates: list[ActionRequest] = []
        for request_id in sorted(live):
            request = live[request_id]
            if request_id in held:
                continue
            predecessor = request.after
            if predecessor is not None and predecessor in state.seen and predecessor not in state.finished:
                statuses.append(RequestStatus(agent=agent, request_id=request_id, kind="deferred", tick=tick,
                                              conflict_with=predecessor))
                continue
            reason = reason_for(request)
            if reason is not None:
                finish(request_id, "rejected", reason=reason)
                continue
            candidates.append(request)

        interrupts = sorted((r for r in candidates if r.mode == protocol.MODE_INTERRUPT), key=order_key, reverse=True)
        ordinary = sorted((r for r in candidates if r.mode != protocol.MODE_INTERRUPT), key=order_key, reverse=True)
        head_owner: dict[str, int] = {head: holder for scope, holder in state.holders.items()
                                      for head in self._scope_heads[scope]}
        for request in interrupts + ordinary:
            heads = self.heads_of(request)
            taken = [head for head in heads if head in head_owner]
            owners = {head_owner[head] for head in taken}
            if taken and request.mode == protocol.MODE_INTERRUPT and all(
                state.granted_at[owner] < tick for owner in owners
            ):
                # An interrupt pre-empts holders from earlier ticks, never a grant made this tick (which
                # outranked it in the greedy order).
                for holder in sorted(owners):
                    finish(holder, "aborted", reason=f"interrupted: by request {request.request_id} ({request.verb})")
                head_owner = {head: owner for head, owner in head_owner.items()
                              if owner not in state.finished}
                taken = [head for head in heads if head in head_owner]
            if taken:
                statuses.append(RequestStatus(agent=agent, request_id=request.request_id, kind="deferred", tick=tick,
                                              conflict_with=head_owner[taken[0]]))
                continue
            for scope in self.specs[request.verb].scopes:
                state.holders[scope] = request.request_id
            if not self.specs[request.verb].scopes:
                state.unscoped.add(request.request_id)
            state.granted_at[request.request_id] = tick
            for head in heads:
                head_owner[head] = request.request_id

        factors = list(self.neutral)
        composition: dict[str, int] = {}
        index = {name: position for position, name in enumerate(self.head_names)}
        drivers = dict(state.holders)
        since_grant = {rid: tick - state.granted_at[rid] for rid in set(drivers.values())}
        for request_id in sorted(set(state.holders.values()) | state.unscoped):
            request = live[request_id]
            body = self.bodies.get(request.verb)
            if body is None:
                finish(request_id, "aborted", reason=f"no_body: verb {request.verb!r} has no body in this combiner")
                drivers = {scope: rid for scope, rid in drivers.items() if rid != request_id}
                continue
            since = tick - state.granted_at[request_id]
            result = body(request, since, context.observation_tokens, context.belief_tokens)
            allowed = set(self.heads_of(request))
            outside = sorted(set(result.factors) - allowed)
            if outside:
                raise ValueError(f"the body of {request.verb!r} filled heads outside its scopes: {outside}")
            for head, choice in result.factors.items():
                factors[index[head]] = int(choice)
                composition[head] = request_id
            driven = tuple(head for head in self.head_names if head in result.factors)
            kind = "accepted" if since == 0 else "running"
            statuses.append(RequestStatus(agent=agent, request_id=request_id, kind=kind, tick=tick,
                                          progress=result.progress, heads=driven))
            if result.done:
                finish(request_id, "done", outcome=result.outcome)
        since_grant = {rid: value for rid, value in since_grant.items() if rid in set(drivers.values())}
        # Forget what can never matter again: a finished request that has left the stream (ids are
        # monotonic per agent and never come back). Done ids stay in ``state.done`` for after-chains, so
        # the work per tick follows the live requests, not every request the agent ever sent.
        for request_id in [rid for rid in state.seen if rid in state.finished and rid not in current]:
            del state.seen[request_id]
            del state.finished[request_id]
        return CombinedStep(tuple(factors), tuple(statuses), composition, drivers, since_grant)


# ---- stub bodies ----


@dataclass(frozen=True)
class StubBody:
    """A deterministic body for the combiner gates: choice of head h = (request_id + h) mod size.

    The formula is the one ``stub_choices`` applies to arrays, so the host path and the table path
    agree by construction. A disabled choice (gen 7's gen-5 dodges) falls back to the neutral one.
    ``done_after`` ticks after the grant the body reports done (None: never).
    """

    layout: str = "gen5"
    done_after: int | None = None
    heads: tuple[str, ...] | None = None

    def __call__(self, request: ActionRequest, ticks_since_start: int, observation_tokens: Any,
                 belief_tokens: Any) -> BodyStep:
        layout = head_layout(self.layout)
        legal = legal_choices(self.layout)
        names = self.heads if self.heads is not None else ()
        factors = {}
        for position, head in enumerate(layout):
            if head.name in names:
                choice = (request.request_id + position) % head.size
                factors[head.name] = choice if legal[position][choice] else head.neutral
        done = self.done_after is not None and ticks_since_start + 1 >= self.done_after
        return BodyStep(factors, done=done, progress={"ticks": float(ticks_since_start)})


def stub_bodies(specs: Mapping[str, VerbSpec], *, layout: str = "gen5",
                done_after: Mapping[str, int] | None = None) -> dict[str, StubBody]:
    """One StubBody per verb, filling every present head of the verb's scopes."""

    finish = dict(done_after or {})
    return {verb: StubBody(layout, finish.get(verb), tuple(head for scope in spec.scopes
                                                           for head in scope_heads(scope, layout)))
            for verb, spec in specs.items()}


# ---- the in-graph form ----


class ActiveRequestTable(NamedTuple):
    """One granted request per agent and scope, as arrays. B agents, S = len(SCOPE_NAMES) scopes.

    request_id int32[B, S] (-1 where the scope is free); verb int32[B, S] (an index into the verb list,
    -1 where free); position, direction float32[B, S, 3] with their presence bool[B, S]; entity
    int32[B, S] with presence; cell int32[B, S, 3] and face int32[B, S] (an index into surface.FACES)
    with presence; scalars float32[B, S, PARAM_WIDTH] with presence bool[B, S, PARAM_WIDTH];
    ticks_since_grant int32[B, S].
    """

    request_id: jax.Array
    verb: jax.Array
    position: jax.Array
    position_present: jax.Array
    direction: jax.Array
    direction_present: jax.Array
    entity: jax.Array
    entity_present: jax.Array
    cell: jax.Array
    face: jax.Array
    cell_present: jax.Array
    scalars: jax.Array
    scalars_present: jax.Array
    ticks_since_grant: jax.Array

    @classmethod
    def from_grants(cls, rows: Sequence[Mapping[str, ActionRequest]], verbs: Sequence[str],
                    since: Sequence[Mapping[int, int]]) -> ActiveRequestTable:
        """Lower each agent's granted requests (scope -> request) into the table.

        Args:
            rows: Per agent, scope -> the granted request.
            verbs: The verb list whose indices the table stores.
            since: Per agent, request id -> ticks since its grant.
        """

        agents, scopes = len(rows), len(SCOPE_NAMES)
        table = {
            "request_id": np.full((agents, scopes), -1, np.int32), "verb": np.full((agents, scopes), -1, np.int32),
            "position": np.zeros((agents, scopes, 3), np.float32), "position_present": np.zeros((agents, scopes), bool),
            "direction": np.zeros((agents, scopes, 3), np.float32),
            "direction_present": np.zeros((agents, scopes), bool),
            "entity": np.zeros((agents, scopes), np.int32), "entity_present": np.zeros((agents, scopes), bool),
            "cell": np.zeros((agents, scopes, 3), np.int32), "face": np.zeros((agents, scopes), np.int32),
            "cell_present": np.zeros((agents, scopes), bool),
            "scalars": np.zeros((agents, scopes, PARAM_WIDTH), np.float32),
            "scalars_present": np.zeros((agents, scopes, PARAM_WIDTH), bool),
            "ticks_since_grant": np.zeros((agents, scopes), np.int32),
        }
        for agent, grants in enumerate(rows):
            for scope, request in grants.items():
                s = SCOPE_NAMES.index(scope)
                args = request.args
                table["request_id"][agent, s] = request.request_id
                table["verb"][agent, s] = list(verbs).index(request.verb)
                table["ticks_since_grant"][agent, s] = since[agent][request.request_id]
                for name in ("position", "direction"):
                    if getattr(args, name) is not None:
                        table[name][agent, s] = getattr(args, name)
                        table[f"{name}_present"][agent, s] = True
                if args.entity is not None:
                    table["entity"][agent, s], table["entity_present"][agent, s] = args.entity, True
                if args.cell is not None:
                    table["cell"][agent, s] = args.cell
                    table["face"][agent, s] = FACES.index(token_face(args.face))
                    table["cell_present"][agent, s] = True
                for k, value in enumerate(args.scalars):
                    if value is not None:
                        table["scalars"][agent, s, k], table["scalars_present"][agent, s, k] = value, True
        return cls(**{name: jnp.asarray(value) for name, value in table.items()})


def stub_choices(table: ActiveRequestTable, layout: str = "gen5") -> jax.Array:
    """The merged action of a table of stub-body grants: int32[B, H]. Traceable under ``jax.jit``.

    Head h of scope s takes ``(request_id[:, s] + h) mod size_h`` where the scope is granted (a disabled
    choice becomes neutral), and the neutral choice where it is free, exactly as StubBody fills a head.
    """

    heads = head_layout(layout)
    legal = legal_choices(layout)
    columns = []
    for position, head in enumerate(heads):
        s = SCOPE_NAMES.index(head.scope)
        rid = table.request_id[:, s]
        choice = (rid + position) % head.size
        allowed = jnp.asarray(legal[position])[choice]
        choice = jnp.where(allowed, choice, head.neutral)
        columns.append(jnp.where(rid >= 0, choice, head.neutral))
    return jnp.stack(columns, axis=1).astype(jnp.int32)


def table_factors(table: ActiveRequestTable, layout: str = "gen5") -> jax.Array:
    """Jitted ``stub_choices``: the in-graph merge of the table's grants."""

    return _jitted(layout)(table)


_JITTED: dict[str, Any] = {}


def _jitted(layout: str) -> Any:
    if layout not in _JITTED:
        _JITTED[layout] = jax.jit(lambda table: stub_choices(table, layout))
    return _JITTED[layout]


def granted_requests(step: CombinedStep, live: Mapping[int, ActionRequest]) -> dict[str, ActionRequest]:
    """Scope -> the request object that drove it on the step's tick, for lowering into the table."""

    return {scope: live[request_id] for scope, request_id in step.grants.items()}


def with_args(request: ActionRequest, **changes: Any) -> ActionRequest:
    """A copy of ``request`` with some fields replaced (a convenience for streams in tests)."""

    return dataclasses.replace(request, **changes)


__all__ = [
    "ActiveRequestTable",
    "AgentContext",
    "Body",
    "BodyStep",
    "CombinedStep",
    "Combiner",
    "StubBody",
    "granted_requests",
    "order_key",
    "stub_bodies",
    "stub_choices",
    "table_factors",
    "with_args",
]
