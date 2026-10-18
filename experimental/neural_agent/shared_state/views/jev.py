"""The typed view the decision layer reads from S_t.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


from typing import NamedTuple, Sequence

import numpy as np

from experimental.neural_agent.contracts import (
    MAX_GOALS,
    SKILL_IDLE,
    ExecutiveArrays,
    ExecutiveState,
    executive_arrays,
)
from experimental.neural_agent.requests import RequestStatus

from ..sections.executive import (
    ExecutiveSection,
)
from ..sections.belief import (
    BeliefSection,
)
from ..sections.requests import (
    RequestsSection,
)
from ..sections.skills import (
    SkillsSection,
)
from ..document import (
    SharedState,
)


class JevView(NamedTuple):
    """What the decision layer reads from S_t. B lanes (the belief section's), G = MAX_GOALS.

    Attributes:
        executive: E_t without the words, the record the slow path encodes.
        arrays: ``contracts.executive_arrays(executive, skill_ids, B)``, E_t lowered as today.
        intents: G strings; ``intents[k]`` is the ``intent`` of the goal in slot k, "" in an empty slot.
        whens: G strings, aligned the same way, for ``when``.
        features: float32[B, BELIEF_FEATURE_COUNT].
        feature_known: bool[B, BELIEF_FEATURE_COUNT].
        latent_norm: float32[B].
        latent_prefix: float32[B, LATENT_PREFIX_WIDTH].
        current_skill: B skill ids: the skills section's active skill, or idle in every lane before
            the skills section is written (the carry starts with ``idle_goal``).
        current_skill_index: int32[B], positions of ``current_skill`` in ``skill_ids``.
        live_requests: B mappings; ``live_requests[b]`` maps each live request id of agent b, in id
            order, to its latest ``RequestStatus`` in the stream, or None before its first status.
    """

    executive: ExecutiveState
    arrays: ExecutiveArrays
    intents: tuple[str, ...]
    whens: tuple[str, ...]
    features: np.ndarray
    feature_known: np.ndarray
    latent_norm: np.ndarray
    latent_prefix: np.ndarray
    current_skill: tuple[str, ...]
    current_skill_index: np.ndarray
    live_requests: tuple[dict[int, RequestStatus | None], ...]




def view_for_jev(state: SharedState, skill_ids: Sequence[str]) -> JevView:
    """Return the typed view the decision layer reads; B is the belief section's lane count.

    Args:
        state: The scratchpad.
        skill_ids: The skill catalogue in dispatcher order, idle first. S_t does not hold the
            catalogue, and ``contracts.executive_arrays`` needs it to lower E_t.

    Raises:
        ValueError: If the belief section has no lanes, the skills section reports a different
            number of lanes, a live request names an agent outside the belief's lanes, or
            ``executive_arrays`` refuses the catalogue.
        KeyError: If a goal or an active skill is not in ``skill_ids``.
    """

    _version, _tick, sections = state._snapshot()
    executive: ExecutiveSection = sections["executive"]
    belief: BeliefSection = sections["belief"]
    requests: RequestsSection = sections["requests"]
    skills: SkillsSection = sections["skills"]
    lanes = len(belief.lanes)
    if lanes == 0:
        raise ValueError("the JEV view needs the belief section written for at least one lane")
    outside = sorted({request.agent for request in requests.live if request.agent >= lanes})
    if outside:
        raise ValueError(f"live requests name agents {outside}; the belief section has {lanes} lanes")
    latest = {(status.agent, status.request_id): status for status in skills.statuses}
    record = executive.to_executive_state()
    arrays = executive_arrays(record, skill_ids, lanes)
    catalogue = tuple(skill_ids)
    current = skills.active or (SKILL_IDLE,) * lanes
    if len(current) != lanes:
        raise ValueError(f"the skills section reports {len(current)} lanes; the belief section has {lanes}")
    unknown = sorted({skill for skill in current if skill not in catalogue})
    if unknown:
        raise KeyError(f"active skill(s) {unknown} are not in skill_ids")
    empty = ("",) * (MAX_GOALS - len(executive.goals))
    return JevView(
        executive=record,
        arrays=arrays,
        intents=tuple(goal.intent for goal in executive.goals) + empty,
        whens=tuple(goal.when for goal in executive.goals) + empty,
        features=np.asarray([lane.features for lane in belief.lanes], dtype=np.float32),
        feature_known=np.asarray([lane.known for lane in belief.lanes], dtype=np.bool_),
        latent_norm=np.asarray([lane.latent_norm for lane in belief.lanes], dtype=np.float32),
        latent_prefix=np.asarray([lane.latent_prefix for lane in belief.lanes], dtype=np.float32),
        current_skill=tuple(current),
        current_skill_index=np.asarray([catalogue.index(skill) for skill in current], dtype=np.int32),
        live_requests=tuple(
            {request.request_id: latest.get((lane, request.request_id)) for request in requests.for_agent(lane)}
            for lane in range(lanes)
        ),
    )


