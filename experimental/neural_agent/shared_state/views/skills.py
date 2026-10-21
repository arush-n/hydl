"""The payload view the skills read from S_t.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


from typing import Collection, NamedTuple, Sequence

import numpy as np

from experimental.neural_agent.contracts import (
    PARAM_WIDTH,
)
from experimental.neural_agent.requests import RequestStatus

from ..document import (
    SharedState,
)


class SkillsView(NamedTuple):
    """What the skills read from S_t: the goal payload fields per lane, from the decision section.

    Attributes:
        skill_id: The skill of the goal in force, per lane.
        params: float32[B, PARAM_WIDTH], zero-padded.
        executive_version: int32[B].
        goal_slot: int32[B], -1 where no executive slot is involved.
    """

    skill_id: tuple[str, ...]
    params: np.ndarray
    executive_version: np.ndarray
    goal_slot: np.ndarray




def view_for_skills(state: SharedState) -> SkillsView:
    """Return the goal payload fields per lane, from the decision section's goals in force.

    B is the number of lanes in ``decision.in_force`` (0 before the JEV's first write).
    """

    in_force = state.decision.in_force
    params = np.zeros((len(in_force), PARAM_WIDTH), dtype=np.float32)
    for lane, goal in enumerate(in_force):
        params[lane, : len(goal.params)] = goal.params
    return SkillsView(
        skill_id=tuple(goal.skill_id for goal in in_force),
        params=params,
        executive_version=np.asarray([goal.executive_version for goal in in_force], dtype=np.int32),
        goal_slot=np.asarray([goal.slot for goal in in_force], dtype=np.int32),
    )


#: The bodies' report names, in the order of ``skill_agent/runtime/scripted_bodies.py`` ``REPORTS`` (append-only there).
#: Mirrored rather than imported: the decision layer does not import the skill layer's runtime (a test checks they agree).
#: ``position_used`` (2026-10-20): the body could place its request's position (a body that reads one and the perception
#: channel that locates it); 0 means a position argument was ignored and the body acted as if none was given.
REPORT_NAMES: tuple[str, ...] = ("seen", "distance", "facing_me", "in_reach", "acting", "position_used")


def lane_reports(statuses: Sequence[RequestStatus], request_ids: Collection[int]) -> dict[str, float] | None:
    """Return the body's reports of a lane's live requests, by name, from the newest status that carries them.

    The scratchpad writes the reports into the ``progress`` of ``accepted`` and ``running`` statuses only
    (``skill_agent/runtime/scratchpad.py``, ``combine_from_scratchpad``).

    Args:
        statuses: The lane's statuses, oldest first.
        request_ids: The ids of the lane's live requests, the ones the last decision issued.

    Returns:
        The values of the names in ``REPORT_NAMES`` that the status carries, or None when no such status exists yet.
    """

    for status in reversed(statuses):
        if status.request_id not in request_ids or status.kind not in ("accepted", "running") or not status.progress:
            continue
        reports = {name: float(value) for name, value in status.progress if name in REPORT_NAMES}
        if reports:
            return reports
    return None
