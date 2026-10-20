"""The combiner on the scratchpad: live requests in from S_t's ``requests`` section, statuses out to ``skills``.

``docs/BUILD.md`` 7b, B3 (a joint step with the middle layer): the decision layer writes ``ActionRequest``
records into the ``requests`` section (writer ``"jev"``); on every tick this module reads each agent's
live requests from there, runs the combiner, and appends the ``RequestStatus`` records to the
``skills`` section's status stream (writer ``"skills"``), where the decision layer reads them and
applies its correspondence rules (``jev/correspondence.py``). Both halves meet in the state; neither
calls the other.

One tick::

    for agent in contexts:   live = S_t.requests.for_agent(agent)
                             step = combiner.step(tick, agent, live, contexts[agent])
    S_t.skills.statuses += every step's statuses (in agent order), one write as writer "skills"

Agents come from ``contexts``, not from the section, so an agent whose last request left the section
is still stepped and its holder is aborted as withdrawn. The skills section keeps its newest
``status_limit`` statuses (``shared_state.SkillsSection``).

Wiring without touching the session (B3). ``CombiningDecider`` wraps the middle layer's host decider
(``jev/programs.py`` ``ProgramExecutor``): the session binds ``scratchpad``, ``host_cadence_ticks``,
``tick_seconds``, ``host_respects_commitment`` and ``current_tick`` on it as on the executor (they are
forwarded), and each ``decide`` returns the executor's goal unchanged and then runs the combiner on S_t at
the session's clock tick, after the executor has written its requests. Each lane gets one context: the
belief's single target is a known entity (``TARGET_ENTITY``, the executor's convention) when
``ticks_since_target_seen`` is known and below saturation, the executor's own rule. A session that calls a
decider's ``on_tick(clock_tick, transition, hierarchy_state)`` on every tick (the middle layer's hook) gets the
combiner on every tick too, so ``running`` statuses reach the stream between decisions; each tick is combined
once. Once the hook has been seen, ``decide`` no longer combines: the combiner's work stays out of the decision's
timed budget (the session counts a decide over ``host_budget_ms`` as a miss and keeps the old goal, which a 512-tick
joint run showed: 3 misses under load, a diverging run). The executor reads the statuses at its next decision
either way. An older session calls only ``decide``, and the combiner then runs inside it on decision ticks.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from experimental.neural_agent.contracts import TICKS_SATURATION, feature_index
from experimental.neural_agent.shared_state import SharedState
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, CombinedStep, Combiner

SKILLS_WRITER: str = "skills"
"""The skills section's one writer (``shared_state.SECTION_WRITERS``)."""
TARGET_ENTITY: int = 0
"""The belief's single target, named by its index among the belief's targets (``jev/programs.py``, agreed
between the two sessions, 2026-10-17)."""
ANY_WORLD: tuple[tuple[float, ...], tuple[float, ...]] = ((-1.0e9,) * 3, (1.0e9,) * 3)
"""The default world box: the composition task's requests name entities, not points."""
_SEEN: int = feature_index("ticks_since_target_seen")


def combine_from_scratchpad(state: SharedState, combiner: Combiner, tick: int,
                            contexts: Mapping[int, AgentContext]) -> dict[int, CombinedStep]:
    """Run one tick of the combiner on S_t and write the statuses back.

    Args:
        state: The scratchpad S_t.
        combiner: The combiner (it carries every agent's grants across ticks).
        tick: The tick.
        contexts: Agent -> its context for this tick (world bounds, known entities, legal scopes,
            observation and belief tokens). Every agent to step this tick.

    Returns:
        Agent -> the combiner's step (factors, statuses, head composition, grants).
    """

    requests = state.requests
    steps = {agent: combiner.step(tick, agent, requests.for_agent(agent), contexts[agent])
             for agent in sorted(contexts)}
    statuses = tuple(status for agent in sorted(steps) for status in steps[agent].statuses)
    if statuses:
        skills = state.skills
        state.write("skills", dataclasses.replace(skills, statuses=skills.statuses + statuses),
                    writer=SKILLS_WRITER, tick=tick)
    return steps


def target_seen(belief: Any) -> np.ndarray:
    """Per lane: the belief's ``ticks_since_target_seen`` is known and below saturation (the executor's rule).

    The belief publishes the counter as ``counter / TICKS_SATURATION``; the rule reads it back in ticks.
    """

    features = np.asarray(belief.features, dtype=np.float32)
    known = np.asarray(belief.feature_known, dtype=bool)
    return known[:, _SEEN] & (features[:, _SEEN] * np.float32(TICKS_SATURATION) < np.float32(TICKS_SATURATION))


def lane_contexts(seen: Any, world_bounds: Any = ANY_WORLD) -> dict[int, AgentContext]:
    """One context per lane: the target entity is known exactly where ``seen`` is True."""

    return {lane: AgentContext(world_bounds=world_bounds,
                               known_entities=frozenset({TARGET_ENTITY}) if flag else frozenset())
            for lane, flag in enumerate(np.asarray(seen, dtype=bool))}


class CombiningDecider:
    """A host decider whose every decision is followed by the combiner on S_t (the module docstring).

    Args:
        inner: The host decider (``decide(transition, hierarchy_state, executive) -> goal``).
        combiner: The combiner, built once per session (it carries every lane's grants).
        world_bounds: The world box of the contexts.
        actor_verbs: The verb list of a ``granted_actor.GrantedActor`` this decider serves. With it, ``on_tick``
            returns the tick's grants as that actor's state (the active-request table), which the session's per-tick
            seam writes into the agent so the grants drive the body; without it ``on_tick`` returns None.

    ``last_steps`` holds the combiner's steps of the latest decision and ``status_counts`` counts every
    status it wrote by kind (a rejection also as ``rejected:<code>``); every other attribute reads through
    to ``inner``.
    """

    FORWARDED: tuple[str, ...] = ("scratchpad", "host_cadence_ticks", "tick_seconds", "host_respects_commitment",
                                  "current_tick")

    def __init__(self, inner: Any, combiner: Combiner, world_bounds: Any = ANY_WORLD,
                 actor_verbs: Sequence[str] | None = None) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_bound", {})
        object.__setattr__(self, "combiner", combiner)
        object.__setattr__(self, "world_bounds", world_bounds)
        object.__setattr__(self, "actor_verbs", None if actor_verbs is None else tuple(actor_verbs))
        object.__setattr__(self, "last_steps", {})
        object.__setattr__(self, "status_counts", Counter())
        object.__setattr__(self, "_combined_tick", None)
        object.__setattr__(self, "_hooked", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.FORWARDED:
            self._bound[name] = value
            if hasattr(self._inner, name):
                setattr(self._inner, name, value)
            return
        object.__setattr__(self, name, value)

    def decide(self, transition: Any, hierarchy_state: Any, executive: Any, /) -> Any:
        goal = self._inner.decide(transition, hierarchy_state, executive)
        if not self._hooked:  # a session without the per-tick hook: combine here, on decision ticks
            self._combine(int(self._bound.get("current_tick", 0)), transition)
        return goal

    def on_tick(self, clock_tick: int, transition: Any, hierarchy_state: Any) -> Any:
        """The session's per-tick hook (it calls a decider's ``on_tick`` after the tick's scratchpad sections are
        written): the combiner runs on every tick, so ``running`` statuses reach the stream between decisions.
        A tick the decision already combined is not combined again. The inner decider's own hook, if any, runs
        first. Returns the tick's grants as the served actor's state when ``actor_verbs`` is set, else None."""

        object.__setattr__(self, "_hooked", True)
        inner = getattr(self._inner, "on_tick", None)
        if callable(inner):
            inner(clock_tick, transition, hierarchy_state)
        self._combine(int(clock_tick), transition)
        state = self._bound.get("scratchpad")
        if self.actor_verbs is None or state is None:
            return None
        from experimental.neural_agent.skill_agent.runtime.granted_actor import grants_table

        agents = sorted(self.last_steps)
        live = [{request.request_id: request for request in state.requests.for_agent(agent)} for agent in agents]
        return grants_table([self.last_steps[agent] for agent in agents], live, self.actor_verbs)

    def _combine(self, tick: int, transition: Any) -> None:
        state = self._bound.get("scratchpad")
        if state is None or tick == self._combined_tick:
            return
        object.__setattr__(self, "_combined_tick", tick)
        seen = target_seen(transition.hierarchy_step.belief_update.belief)
        steps = combine_from_scratchpad(state, self.combiner, tick, lane_contexts(seen, self.world_bounds))
        object.__setattr__(self, "last_steps", steps)
        for step in steps.values():
            for status in step.statuses:
                self.status_counts[status.kind] += 1
                if status.kind == "rejected":
                    self.status_counts[f"rejected:{status.reason.split(':', 1)[0]}"] += 1


__all__ = ["ANY_WORLD", "SKILLS_WRITER", "TARGET_ENTITY", "CombiningDecider", "combine_from_scratchpad", "lane_contexts",
           "target_seen"]
