"""Patterned language: the shape the LLM is asked to give its goals, as a direction rather than a grammar.

Owner ruling (2026-10-18): the reading failures were about the structure of the words, not the models.
"They need to be structured in their outputs, however structure may break to a degree, so it needs to be
structured enough in nature that the middle layer can interpret it properly ... it does not have to follow a
defined structure, but the natural language should follow a pattern ... given as a direction/rule, so
interpretation is easier while no content or thinking of the LLM is lost."

The pattern (``PATTERN_DIRECTIONS`` is the prose the system prompt carries):
    intent   starts with the skill's catalogue id, then what the skill is for in a few words:
             ``pursuit: close the distance and keep pressure``
    when     one clause, a feature word, a comparator word, a number with its unit:
             ``when the target is closer than 3 blocks``; ``when hit``; ``always``
    note     free: anything the LLM wants kept, in any form

The pattern is read by ``jev/system_one/pattern_reader.py`` first (cheap and exact), then by the fallback
readers when it breaks. This module also *generates* patterned wordings with controlled breakage, so the
readers can be measured against how much the LLM departs from the pattern (ledger "P7"):
    breakage 0.0   the canonical forms above
    breakage b     each perturbation below is applied with probability b: the id moves later in the sentence,
                   the id is replaced by one of its aliases, the comparator by a synonym, the unit is dropped,
                   a filler phrase is prepended, the clause order is swapped (condition first), the number is
                   written in words
    description    the hard case the earlier matching experiment measured: no id at all, only a description
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from experimental.neural_agent.executive.goal_language import Predicate, _surface_forms

PATTERN_DIRECTIONS: str = (
    "How to word a goal so the decision layer reads it exactly, while you keep your own reasoning in note: "
    "start intent with the skill's catalogue id, then say in a few words what the skill is for "
    "(for example 'pursuit: close the distance and keep pressure'). Write when as one clause with a feature word, "
    "a comparator and a number with its unit (for example 'when the target is closer than 3 blocks', "
    "'when my health is below 30%', 'when the target has been unseen for 60 ticks', 'when hit'), or 'always'. "
    "These are directions, not a grammar: if you depart from them the layer still tries to read you, and asks "
    "when it cannot. Put everything else in note; nothing you write there is lost."
)

CANONICAL_WHEN: Mapping[str, str] = {
    "always": "always",
    "on_damage": "when hit",
    "target_visible": "when the target is visible",
    "target_not_visible": "when the target is not visible",
    "target_lost_for": "when the target has been unseen for {n} ticks",
    "distance_below": "when the target is closer than {n} blocks",
    "distance_above": "when the target is farther than {n} blocks",
    "self_health_below": "when my health is below {n}%",
    "target_health_below": "when the target's health is below {n}%",
}
"""The canonical condition clause per kind; ``{n}`` is the number."""

COMPARATOR_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    "closer than": ("within", "less than", "under", "nearer than"),
    "farther than": ("beyond", "more than", "further than", "past"),
    "below": ("under", "less than", "lower than", "beneath"),
    "unseen for": ("out of sight for", "not seen for", "lost for"),
}
FILLERS: tuple[str, ...] = ("right now,", "for this plan,", "as discussed,", "to be clear,", "please,")
PERTURBATIONS: tuple[str, ...] = (
    "id_later", "alias", "comparator_synonym", "unit_dropped", "filler", "clause_swapped", "number_words",
)


@dataclass(frozen=True)
class PatternedGoal:
    """One generated goal in the patterned language, with what was applied to it."""

    skill_id: str
    predicate: Predicate
    intent: str
    when: str
    perturbations: tuple[str, ...]


def canonical_intent(skill_id: str, purpose: str) -> str:
    return f"{skill_id}: {purpose}"


def canonical_when(predicate: Predicate) -> str:
    """The canonical clause for ``predicate``; numbers in digits with the kind's unit."""

    text = CANONICAL_WHEN[predicate.kind]
    if predicate.argument is None:
        return text
    if predicate.kind in ("self_health_below", "target_health_below"):
        return text.replace("{n}", f"{round(predicate.argument * 100)}")
    return text.replace("{n}", f"{predicate.argument:g}")


def patterned_goal(
    skill_id: str,
    predicate: Predicate,
    purpose: str,
    rng: np.random.Generator,
    *,
    breakage: float = 0.0,
    aliases: Sequence[str] = (),
) -> PatternedGoal:
    """Generate a patterned goal, applying each perturbation with probability ``breakage``.

    Args:
        skill_id: The catalogue id the goal names.
        predicate: The typed condition the words must mean.
        purpose: A few words on what the skill is for (any register).
        rng: The source of randomness.
        breakage: In [0, 1]; 0 gives the canonical forms.
        aliases: Other ids of the same catalogue row, for the ``alias`` perturbation.

    Raises:
        ValueError: If ``breakage`` is outside [0, 1].
    """

    if not 0.0 <= breakage <= 1.0:
        raise ValueError(f"breakage must be in [0, 1], got {breakage}")
    applied: list[str] = []
    draws = {name: bool(rng.random() < breakage) for name in PERTURBATIONS}
    name = skill_id
    if draws["alias"] and aliases:
        name = str(aliases[rng.integers(len(aliases))])
        applied.append("alias")
    if draws["id_later"]:
        intent = f"{purpose}, using {name}"
        applied.append("id_later")
    else:
        intent = canonical_intent(name, purpose)
    when = canonical_when(predicate)
    if predicate.argument is not None:
        if draws["comparator_synonym"]:
            for comparator, synonyms in COMPARATOR_SYNONYMS.items():
                if comparator in when:
                    when = when.replace(comparator, str(synonyms[rng.integers(len(synonyms))]), 1)
                    applied.append("comparator_synonym")
                    break
        if draws["number_words"]:
            forms = _surface_forms(predicate.kind, predicate.argument)
            worded = [form for form in forms if not any(ch.isdigit() for ch in form)]
            if worded:
                when = _replace_number(when, predicate, str(worded[rng.integers(len(worded))]))
                applied.append("number_words")
        if draws["unit_dropped"]:
            for unit in (" blocks", " ticks", "%"):
                if unit in when:
                    when = when.replace(unit, "", 1)
                    applied.append("unit_dropped")
                    break
    if draws["filler"]:
        intent = f"{FILLERS[rng.integers(len(FILLERS))]} {intent}"
        applied.append("filler")
    if draws["clause_swapped"] and when != "always":
        intent, when = f"{when}, {intent}", ""
        applied.append("clause_swapped")
    return PatternedGoal(skill_id, predicate, intent, when, tuple(applied))


def _replace_number(when: str, predicate: Predicate, worded: str) -> str:
    """Replace the digits of the canonical clause with ``worded`` (a surface form from the generator)."""

    if predicate.kind in ("self_health_below", "target_health_below"):
        digits = f"{round(predicate.argument * 100)}%"
        return when.replace(digits, worded, 1)
    digits = f"{predicate.argument:g}"
    for unit in (" blocks", " ticks"):
        if f"{digits}{unit}" in when:
            return when.replace(f"{digits}{unit}", worded if unit.strip() in worded else f"{worded}{unit}", 1)
    return when.replace(digits, worded, 1)


def description_only(purpose: str, predicate: Predicate) -> PatternedGoal:
    """The hard case: the intent is a description that never names the skill."""

    return PatternedGoal("", predicate, purpose, canonical_when(predicate), ("description_only",))


__all__ = [
    "CANONICAL_WHEN",
    "COMPARATOR_SYNONYMS",
    "FILLERS",
    "PATTERN_DIRECTIONS",
    "PERTURBATIONS",
    "PatternedGoal",
    "canonical_intent",
    "canonical_when",
    "description_only",
    "patterned_goal",
]
