"""Skill catalog: an ordered tuple of skills with idle always at index 0.

``SkillRegistry`` is the host-side catalog that the dispatcher and the executive
share. ``ids`` is the ``skill_ids`` argument of
``experimental.neural_agent.contracts.executive_arrays``, so an executive goal's
``skill_index`` names the same position the dispatcher uses.

``build_default_registry(batch)`` returns ``(idle, pursuit, evader)``. Loading
the pursuit candidate reads its pinned checkpoint from disk.
"""

from __future__ import annotations

from typing import Sequence

from experimental.neural_agent.contracts import (
    SKILL_IDLE,
    Skill,
    SkillContract,
    SkillControlTable,
    control_table,
)
from experimental.neural_agent.requests import COMBAT_CONTRACTS, COMBAT_VERBS
from experimental.neural_agent.skill_library._common import validate_batch
from experimental.neural_agent.skill_library.evader import ScriptedEvaderSkill
from experimental.neural_agent.skill_library.idle import IdleSkill
from experimental.neural_agent.skill_library.pursuit import load_pursuit_skill


class SkillRegistry:
    """Ordered skill catalog. Index 0 is always the idle skill (ADR-001 D6).

    Raises:
        ValueError: If the catalog is empty, index 0 is not ``"idle"``, or a
            skill id repeats.
        TypeError: If an entry has no ``SkillContract`` as ``.contract``.
    """

    def __init__(self, skills: Sequence[Skill]) -> None:
        items = tuple(skills)
        if not items:
            raise ValueError("SkillRegistry needs at least the idle skill")
        for position, skill in enumerate(items):
            contract = getattr(skill, "contract", None)
            if not isinstance(contract, SkillContract):
                raise TypeError(
                    f"skills[{position}] must expose a SkillContract as .contract"
                )
            if not callable(getattr(skill, "initial_state", None)) or not callable(skill):
                raise TypeError(
                    f"skills[{position}] must implement initial_state and __call__"
                )
        ids = tuple(skill.contract.skill_id for skill in items)
        if ids[0] != SKILL_IDLE:
            raise ValueError(f"the idle skill must be at index 0, got {ids[0]!r}")
        repeated = sorted({name for name in ids if ids.count(name) > 1})
        if repeated:
            raise ValueError(f"skill ids repeat: {repeated}")
        self._skills: tuple[Skill, ...] = items
        self._ids: tuple[str, ...] = ids

    @property
    def skills(self) -> tuple[Skill, ...]:
        """The skills in catalog order."""

        return self._skills

    @property
    def ids(self) -> tuple[str, ...]:
        """Skill ids in catalog order, the ``skill_ids`` for ``executive_arrays``."""

        return self._ids

    @property
    def contracts(self) -> tuple[SkillContract, ...]:
        """The ``SkillContract`` of every skill, in catalog order."""

        return tuple(skill.contract for skill in self._skills)

    def control_table(self) -> SkillControlTable:
        """The goal control each skill's contract declares, in catalog order (idle first).

        Pass it to a JEV instance and its runner as ``control=``, so every goal
        carries the chosen skill's own control (SYSTEM_ONE.md section 8.12, item 1).
        """

        return control_table(self.contracts)

    def index(self, skill_id: str) -> int:
        """Return the catalog position of ``skill_id``.

        Raises:
            KeyError: If ``skill_id`` is not in the catalog.
        """

        try:
            return self._ids.index(skill_id)
        except ValueError:
            raise KeyError(
                f"unknown skill id {skill_id!r}; the catalog is {self._ids}"
            ) from None

    def __len__(self) -> int:
        """Number of skills in the catalog."""

        return len(self._skills)


def build_default_registry(batch: int) -> SkillRegistry:
    """Return the catalog ``(idle, pursuit, evader)`` for ``batch`` lanes.

    Args:
        batch: Number of lanes B, a positive int. It is passed to the pursuit
            loader, which checks the pinned memory shape.

    Raises:
        TypeError: If ``batch`` is not an integer.
        ValueError: If the pinned pursuit files fail their checks.
    """

    lanes = validate_batch(batch)
    return SkillRegistry((IdleSkill(), load_pursuit_skill(lanes), ScriptedEvaderSkill()))


class DeclaredVerbSkill(IdleSkill):
    """A combat verb's catalogue slot: the idle body under the verb's declared contract (2026-10-19).

    Its outputs are the idle body's neutral choices, so the slot never acts in this package. The verb's own body is
    the skill layer's (``skill_agent/runtime/scripted_bodies.py``, commit 1bd804a).
    """

    def __init__(self, contract: SkillContract) -> None:
        self.contract = contract


def build_combat_registry(batch: int) -> SkillRegistry:
    """Return the catalogue of the combat program: idle, pursuit, evader, then the four combat verbs.

    The verbs come from ``COMBAT_CONTRACTS`` in ``COMBAT_VERBS`` order (sword, guard, punish, keep_range), so their
    dispatcher indexes are 3 to 6. The default three are unchanged, so an arm that does not run the combat program
    keeps its catalogue.

    Args:
        batch: Number of lanes B, a positive int (the pursuit loader checks its pinned memory with it).

    Raises:
        TypeError: If ``batch`` is not an integer.
        ValueError: If the pinned pursuit files fail their checks.
    """

    base = build_default_registry(batch)
    return SkillRegistry((*base.skills, *(DeclaredVerbSkill(COMBAT_CONTRACTS[verb]) for verb in COMBAT_VERBS)))


__all__ = ["DeclaredVerbSkill", "SkillRegistry", "build_combat_registry", "build_default_registry"]
