"""The combiner on the scratchpad: live requests in from S_t's ``requests`` section, statuses out to ``skills``.

``BUILD.md`` 7b, B3 (a joint step with the middle layer): the decision layer writes ``ActionRequest``
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
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

from experimental.neural_agent.shared_state import SharedState
from experimental.neural_agent.skill_agent.combiner import AgentContext, CombinedStep, Combiner

SKILLS_WRITER: str = "skills"
"""The skills section's one writer (``shared_state.SECTION_WRITERS``)."""


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


__all__ = ["SKILLS_WRITER", "combine_from_scratchpad"]
