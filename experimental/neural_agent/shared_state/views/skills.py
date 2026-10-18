"""The payload view the skills read from S_t.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


from typing import NamedTuple

import numpy as np

from experimental.neural_agent.contracts import (
    PARAM_WIDTH,
)

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


