"""The nearest skill for an intent the library cannot serve (SYSTEM_ONE 9.16).

Owner ruling (2026-10-17): when the LLM asks for something no skill serves, "then we have to choose the best
possible choice", never refuse. Today the executive store refuses a proposal whose goal names a skill id that
is not in the catalogue, and the LLM has to correct it on its next version. This module turns that refusal
into the best available choice plus a question: the harness resolves the unknown id before proposing, and
tells the LLM which skill now serves the goal and why, so the LLM can confirm or name a catalogue skill.

Resolution, in order (the first step that yields a candidate wins; `Resolution.source` names it):
    exact        the requested id is a catalogue id                                   confidence 1.0
    alias        the requested id is an alias of a catalogue row                      confidence 1.0
    tokens       the requested id's words meet a row's id words (pursue_target -> pursuit, guard_position ->
                 guard)                                                               confidence = the match,
                                                                                      at most WEAK (0.6)
    none         no id is near: no skill; the harness leaves the goal out of this version and the question
                 carries up to three rows whose id-plus-description words are nearest to the request and the
                 goal's words, as suggestions only                                    confidence 0.0

Descriptions suggest and never serve: on the seven generator families with no real skill, the nearest row by
description was noise on short phrasings ("take cover" -> loot_chest on the word "take", "make rounds" -> craft
on "make"), the same finding as 9.12 and P6 (skills by description fail). Ids are the contract, so only an id
match serves.

The match is the weighted share of the request's words that have a like word in the row (equal, or one the
other's stem: pursue/pursuit, waypoint/waypoints); a word's weight is its smoothed inverse row frequency, so
"block" in a dozen rows weighs little and "waypoint" in one weighs most. Ids beat descriptions because ids are
the contract. The resolver never guesses silently: every resolution below exact or alias is a question to the
LLM (`executive/harness.py` records it; the session's scribe writes it to the decision section).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from experimental.neural_agent.executive.goal_language import STOP_WORDS
from experimental.neural_agent.jev.system_one.condition_rules import CLEAR, WEAK
from experimental.neural_agent.jev.system_one.pattern_reader import catalogue_index

NONE: float = 0.0
_WORD = re.compile(r"[a-z][a-z0-9]*")
_MIN_STEM = 4


@dataclass(frozen=True)
class Resolution:
    """What an unknown skill request resolves to.

    Attributes:
        requested: The id the LLM wrote.
        skill_id: The catalogue row serving it, or None.
        source: ``exact``, ``alias``, ``tokens`` or ``none``.
        confidence: 1.0 for exact and alias, at most WEAK for tokens, NONE for none.
        candidates: The nearest rows in order: the token matches, or for ``none`` the description suggestions.
        reason: One or two sentences for the LLM.
    """

    requested: str
    skill_id: str | None
    source: str
    confidence: float
    candidates: tuple[str, ...]
    reason: str


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower().replace("_", " ")) if word not in STOP_WORDS}


def _alike(left: str, right: str) -> bool:
    """Equal, or one the other's stem: a common prefix of at least 4 letters leaving at most 3 letters of either
    (pursue/pursuit, waypoint/waypoints, guard/guarding)."""

    if left == right:
        return True
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared >= _MIN_STEM and len(left) - shared <= 3 and len(right) - shared <= 3


class _Words:
    """One catalogue's words: per row its id words and its id-plus-description words, and each word's weight,
    the smoothed inverse row frequency (a word in every row weighs 1, a word in no row the most)."""

    def __init__(self, catalogue: Mapping[str, Sequence[str]], descriptions: Mapping[str, str] | None) -> None:
        self.ids = {row: set().union(*(_tokens(name) for name in (row, *aliases))) for row, aliases in catalogue.items()}
        self.all = {row: words | _tokens((descriptions or {}).get(row, "")) for row, words in self.ids.items()}
        self._rows = len(self.all)
        self._counts = Counter(word for words in self.all.values() for word in words)

    def weight(self, word: str) -> float:
        return math.log((self._rows + 1) / (self._counts.get(word, 0) + 1)) + 1.0

    def ranked(self, request: set[str], table: Mapping[str, set[str]]) -> list[tuple[float, str]]:
        """Rows by the weighted share of the request's words with a like word in the row, best first."""

        total = sum(self.weight(word) for word in request)
        if total == 0.0:
            return []
        scored = [
            (sum(self.weight(word) for word in request if any(_alike(word, other) for other in words)) / total, row)
            for row, words in table.items()
        ]
        return sorted(scored, key=lambda item: (-item[0], item[1]))


def resolve(
    requested: str,
    words: str,
    catalogue: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str] | None = None,
) -> Resolution:
    """Resolve ``requested`` (a skill id the LLM wrote) against ``catalogue`` (row id -> aliases).

    Args:
        requested: The skill id as written.
        words: The goal's words (intent), added to the request side of the description step.
        catalogue: Row id -> its other ids.
        descriptions: Row id -> description, for the description step; None skips it.

    Raises:
        TypeError: If ``requested`` is not a string.
    """

    if not isinstance(requested, str):
        raise TypeError(f"requested must be a string, got {type(requested).__name__}")
    index = catalogue_index(catalogue)
    key = requested.strip().lower()
    if key in index:
        row = index[key]
        source = "exact" if row == key else "alias"
        return Resolution(requested, row, source, CLEAR, (row,), f"{requested!r} is the catalogue skill {row!r}.")
    table = _Words(catalogue, descriptions)
    request = _tokens(requested)
    ranked = table.ranked(request, table.ids)
    if ranked and ranked[0][0] > 0.0:
        best, row = ranked[0]
        return Resolution(
            requested, row, "tokens", min(best, WEAK), tuple(r for score, r in ranked[:3] if score > 0.0),
            f"{requested!r} is not a catalogue skill; serving it with {row!r}, the nearest id by its words "
            f"(match {best:.2f}). Name a catalogue skill or confirm.",
        )
    suggestions: tuple[str, ...] = ()
    if descriptions:
        ranked = table.ranked(request | _tokens(words), table.all)
        suggestions = tuple(row for score, row in ranked[:3] if score > 0.0)
    hint = f"; by description the nearest are {', '.join(suggestions)}" if suggestions else ""
    return Resolution(
        requested, None, "none", NONE, suggestions,
        f"{requested!r} is not a catalogue skill and no catalogue id is near it{hint}. The goal is left out of this "
        "version; name a catalogue skill.",
    )


def resolver_for(
    skill_ids: Sequence[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> Callable[[str, str], Resolution]:
    """A resolver bound to one catalogue: ``resolver(requested, words) -> Resolution``."""

    catalogue = {skill: tuple(aliases.get(skill, ())) if aliases else () for skill in skill_ids}

    def resolver(requested: str, words: str = "") -> Resolution:
        return resolve(requested, words, catalogue, descriptions)

    return resolver


def real_catalogue_tables() -> tuple[dict[str, tuple[str, ...]], dict[str, str], dict[str, str]] | None:
    """(id -> aliases, id -> description, id -> group) of the private catalogue's learned and command rows when the
    file exists, else None. Read at run time; nothing is copied into tracked files."""

    from experimental.neural_agent.jev.system_one.catalogue_matching import MATCHED_KINDS, read_catalogue

    rows = read_catalogue()
    if rows is None:
        return None
    matched = [row for row in rows if row.kind in MATCHED_KINDS]
    return (
        {row.ids[0]: tuple(row.ids[1:]) for row in matched},
        {row.ids[0]: row.description for row in matched},
        {row.ids[0]: row.group for row in matched},
    )


__all__ = ["NONE", "Resolution", "real_catalogue_tables", "resolve", "resolver_for"]
