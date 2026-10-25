"""What a neural skill declares, so that the shared training blocks and the runtime can use it.

Each skill folder's ``skill.py`` builds one :class:`SkillSpec`: the scope it drives, how it samples its own goals, how
its request reads from the agent's pose, its teacher and its success test. The skill's ``train.py`` is its training
strategy: it passes its own spec, and the specs of the skills that run beside it, to the training blocks
(``training/skill_dagger.py``), which train that one skill's module on the shared frozen trunk.

Owner, 2026-10-19: every skill trains its own module by its own strategy; skills compose at run time (the combiner
gives each its scope, programs chain them); a module reads the other active requests, but only of skills it was
trained beside (``model/skill_module.py``), so adding a skill never changes an existing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax

from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES

VERBS: tuple[str, ...] = ("walk_to", "look_at", "come_here")
"""A skill's verb id is its index here, in every module's verb vocabulary. Append only: an id is baked into the trained
weights of every module that has seen it."""


class Pose(NamedTuple):
    """The agent's own pose at one decision (the agent may know its own pose). Leading dimension B."""

    position: jax.Array  # float32[B, 3]
    yaw: jax.Array  # float32[B]: the body yaw, degrees
    head_yaw: jax.Array  # float32[B]: the head's absolute yaw, degrees
    head_pitch: jax.Array  # float32[B]: the head's pitch, degrees, positive up


class RequestRow(NamedTuple):
    """One skill's request for B lanes: the 15 argument features (``model.generalist.ARG_GROUPS``), which of the six
    argument groups are set, and the entity and block-candidate pointers (-1 for none)."""

    args: jax.Array  # float32[B, 15]
    present: jax.Array  # bool[B, 6]
    entity_pointer: jax.Array  # int32[B]
    cell_pointer: jax.Array  # int32[B]


@dataclass(frozen=True)
class SkillSpec:
    """A skill as the training blocks see it. Every callable is pure JAX over the batch, so it can run under ``jit``.

    Attributes:
        name: The skill's name; it must be in :data:`VERBS`, whose index is its verb id.
        scope: The scope it drives (``contract.surface.SCOPE_NAMES``): its module's heads are that scope's heads.
        sample: ``(key, batch, spawn_yaw, *, held_out) -> goals``, the skill's own goals, a NamedTuple with
            ``horizon_ticks`` int32[B]. ``held_out`` selects the skill's held-out region instead of its training region.
        request: ``(goals, pose, ticks) -> RequestRow``: the request at one decision, from the agent's own pose.
        teacher: ``(combat, goals) -> TeacherOutput``: the teacher's factors int32[B, 12] (only its scope's heads are
            read) and its done flags bool[B].
        success: ``(combat, goals) -> bool[B]``: the skill's success test.
        metrics: Optional ``(goals, active, trace, hit) -> dict``: the skill's own evaluation numbers (walk_to's pace
            and radius), from the rollout's trace (positions, yaws and tick lengths) and its first-hit ticks.
    """

    name: str
    scope: str
    sample: Callable[..., Any]
    request: Callable[..., RequestRow]
    teacher: Callable[..., Any]
    success: Callable[..., jax.Array]
    metrics: Callable[..., dict] | None = None

    def __post_init__(self) -> None:
        if self.name not in VERBS:
            raise ValueError(f"skill {self.name!r} is not in VERBS; append it there to give it a verb id")
        if self.scope not in SCOPE_NAMES:
            raise ValueError(f"scope {self.scope!r} is not one of {SCOPE_NAMES}")

    @property
    def verb(self) -> int:
        """The skill's verb id, its index in :data:`VERBS`."""

        return VERBS.index(self.name)


__all__ = ["Pose", "RequestRow", "SkillSpec", "VERBS"]
