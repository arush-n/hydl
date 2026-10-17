"""Plain-language descriptions of the skills in the library, for the decision layer.

J-S1 (our own System One model, ``docs/SYSTEM_ONE.md`` section 8) reads the skill
catalogue as text and scores each skill against its description, so a skill added
later is scored without retraining (gate G3b). Until 2026-10-17 no description
existed and skills were described by their ids; these are the first. They describe
what the skill does and when the decision layer should pick it, in our own words.
"""

from __future__ import annotations

from typing import Sequence

#: skill id -> description. Keep each to one or two sentences: what it does, when to pick it.
SKILL_DESCRIPTIONS: dict[str, str] = {
    "idle": (
        "Idle: stand still and take no action. The fallback when the executive lists no valid "
        "goal or no other skill applies."
    ),
    "pursuit": (
        "Pursuit: close the distance to the selected target and catch it within one block. The "
        "engagement skill while a target is visible and no damage threat is active."
    ),
    "evader": (
        "Evader: break contact and keep a safe band of three to six blocks from the target. The "
        "defensive skill while the agent has taken damage recently and the executive allows evading."
    ),
}


def describe(skill_ids: Sequence[str]) -> tuple[str, ...]:
    """Return one description per skill id, the id itself for a skill without a description."""

    return tuple(SKILL_DESCRIPTIONS.get(skill_id, skill_id) for skill_id in skill_ids)


__all__ = ["SKILL_DESCRIPTIONS", "describe"]
