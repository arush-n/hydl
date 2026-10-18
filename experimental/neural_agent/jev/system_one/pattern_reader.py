"""The pattern reader: the skill named in a goal's words, and the condition clause, read exactly when the LLM
followed the patterned language (``executive/patterned.py``), with a confidence that says when it did not.

Skill (``read_skill``): the catalogue ids and their aliases are the only vocabulary. A leading id ("pursuit:
...") is CLEAR; an id anywhere in the words ("..., using pursuit") is CLEAR; an alias is CLEAR; several
different rows named in one intent is WEAK (the first wins); no id at all is NONE, the case description
matching failed on (SYSTEM_ONE 9.11), and the reader returns no skill rather than a guess. The typed
``skill_id`` of the goal stays the selector; this reader is for words-only goals and for checking that the
words and the typed field agree.

Condition: ``condition_rules.read_condition`` (the feature, comparator and negation cue reader), which
already covers the canonical clauses and most perturbations; when the LLM swapped the clause order and the
condition sits in the intent, the intent is read too.

Gate P7 (``python -m experimental.neural_agent.jev.system_one.pattern_reader --seed 0``): agreement of the
skill and condition readings with the generated meaning over breakage levels 0, 0.2, 0.5 and 1.0 and the
description-only case, on the real catalogue when present (the private file is read, never copied), else
on the stand-in catalogue. Bars: skill and condition agreement >= 0.98 at breakage 0 and >= 0.95 at 0.2.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from experimental.neural_agent.executive.goal_language import CONDITION_FAMILIES, Predicate
from experimental.neural_agent.executive.patterned import PERTURBATIONS, PatternedGoal, description_only, patterned_goal
from experimental.neural_agent.jev.system_one.condition_rules import CLEAR, WEAK, RuleReading, read_condition

NONE: float = 0.0
_WORD = re.compile(r"[a-z][a-z0-9_]*")
_MARKED = re.compile(r"\b([a-z][a-z0-9_]*)\s*:")


@dataclass(frozen=True)
class SkillReading:
    """The skill a goal's words name.

    Attributes:
        skill_id: The catalogue row's first id, or None when no id or alias occurs.
        source: ``leading``, ``mentioned``, ``alias`` or ``none``.
        confidence: CLEAR for one row named, WEAK for several, NONE for none.
        candidates: Every row id found, in order of appearance.
    """

    skill_id: str | None
    source: str
    confidence: float
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class WordsReading:
    skill: SkillReading
    condition: RuleReading


def catalogue_index(rows: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Map every id and alias (lower case) to its row's first id. ``rows`` maps a row id to all its ids."""

    index: dict[str, str] = {}
    for row_id, ids in rows.items():
        for name in (row_id, *ids):
            index[str(name).lower()] = row_id
    return index


def read_skill(intent: str, index: Mapping[str, str]) -> SkillReading:
    """Read the skill named in ``intent`` against ``index`` (from :func:`catalogue_index`).

    Every id or alias found is a candidate. An anchored one wins: the first word of the intent, an id written
    with the pattern's colon ("..., pursuit: close in"), or one introduced by "using", "with" or "via". Without an anchor the longest
    id wins (a two-word id over a one-word id that is part of it). CLEAR when the winner is anchored or is the
    only row named, WEAK when several rows are named with no anchor, NONE when no id occurs.
    """

    text = intent.lower()
    words = _WORD.findall(text)
    marked = {m.group(1) for m in _MARKED.finditer(text)}  # ids written as "id:" (the pattern's marker)
    candidates: list[tuple[str, int, bool, str]] = []  # (row, position, anchored, source)
    for position, word in enumerate(words):
        row = index.get(word)
        if row is None:
            continue
        anchored = position == 0 or word in marked or words[position - 1] in ("using", "with", "via")
        source = "leading" if position == 0 else ("mentioned" if word == row else "alias")
        candidates.append((row, position, anchored, source))
    if not candidates:
        return SkillReading(None, "none", NONE, ())
    distinct = tuple(dict.fromkeys(row for row, _, _, _ in candidates))
    anchored = [c for c in candidates if c[2]]
    if anchored:
        row, _, _, source = anchored[0]
        return SkillReading(row, source, CLEAR, distinct)
    row, _, _, source = max(candidates, key=lambda c: (len(c[0]), -c[1]))
    return SkillReading(row, source, CLEAR if len(distinct) == 1 else WEAK, distinct)


_CLAUSE = re.compile(r"\b(?:when|whenever|if|while|once|until|unless|as soon as)\b[^,]*")


def leading_clause(intent: str) -> str | None:
    """The condition clause an LLM wrote into the intent ("when the target is visible, pursuit: ..."), or None.

    Only a clause introduced by a condition word and ending at a comma counts, so the purpose words of the
    intent ("close in", "keep your distance") are never read as a condition.
    """

    match = _CLAUSE.search(intent.lower())
    if match is None or "," not in intent[match.end():match.end() + 1] and not intent[match.end():].startswith(","):
        return None
    return match.group(0).strip()


def read_words(intent: str, when: str, index: Mapping[str, str]) -> WordsReading:
    """Read the skill from the intent and the condition from ``when``.

    With ``when`` empty, a leading condition clause in the intent is read instead (the swapped-clause
    departure); with neither, the condition is ``always`` and WEAK, since the LLM said nothing about it.
    """

    skill = read_skill(intent, index)
    if when.strip():
        condition = read_condition(when)
    else:
        clause = leading_clause(intent)
        condition = read_condition(clause) if clause else RuleReading("always", None, WEAK, ())
    return WordsReading(skill, condition)


# ---------------------------------------------------------------------------------------------- the gate


def stand_in_catalogue() -> dict[str, tuple[str, ...]]:
    from experimental.neural_agent.skill_library.descriptions import SKILL_DESCRIPTIONS

    return {skill: () for skill in SKILL_DESCRIPTIONS}


def real_catalogue() -> dict[str, tuple[str, ...]] | None:
    """The private catalogue's rows (id -> aliases) when the file exists, else None. Nothing is copied."""

    from experimental.neural_agent.jev.system_one.catalogue_matching import MATCHED_KINDS, read_catalogue

    rows = read_catalogue()
    if rows is None:
        return None
    return {row.ids[0]: tuple(row.ids[1:]) for row in rows if row.kind in MATCHED_KINDS}


def _purposes(rows: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """A short purpose phrase per row, from the stand-in descriptions when known, else generic."""

    from experimental.neural_agent.skill_library.descriptions import SKILL_DESCRIPTIONS

    neutral = ("keep at it until it is done", "this is the priority now", "carry this out carefully", "do this next")
    return {row: SKILL_DESCRIPTIONS.get(row, neutral[i % len(neutral)]) for i, row in enumerate(rows)}


def sample_goals(rows: Mapping[str, Sequence[str]], rng: np.random.Generator, count: int, breakage: float | None) -> list[PatternedGoal]:
    """``count`` goals over ``rows`` with every condition kind; ``breakage`` None is the description-only case."""

    ids = list(rows)
    purposes = _purposes(rows)
    goals = []
    for _ in range(count):
        skill = ids[rng.integers(len(ids))]
        family = CONDITION_FAMILIES[rng.integers(len(CONDITION_FAMILIES))]
        argument = None
        if family.numeric:
            step = 0.05 if family.family_id.endswith("health_below") else (10.0 if family.family_id == "target_lost_for" else 1.0)
            grid = round((family.argument_high - family.argument_low) / step)
            argument = round(family.argument_low + step * int(rng.integers(grid + 1)), 2)
        predicate = Predicate(family.family_id, argument)
        if breakage is None:
            goals.append(description_only(purposes[skill], predicate))
        else:
            goals.append(patterned_goal(skill, predicate, purposes[skill], rng, breakage=breakage, aliases=rows[skill]))
    return goals


def score(goals: Sequence[PatternedGoal], index: Mapping[str, str]) -> dict[str, Any]:
    skill_hits = condition_hits = both = unsure = 0
    per_perturbation: dict[str, list[int]] = {name: [0, 0] for name in (*PERTURBATIONS, "description_only")}
    for goal in goals:
        reading = read_words(goal.intent, goal.when, index)
        # a description-only goal names no skill: the right reading is no skill (no guess), never a wrong one
        skill_ok = reading.skill.skill_id is None if goal.skill_id == "" else reading.skill.skill_id == goal.skill_id
        predicate = reading.condition.predicate()
        condition_ok = predicate is not None and predicate == goal.predicate
        skill_hits += skill_ok
        condition_hits += condition_ok
        both += skill_ok and condition_ok
        unsure += reading.skill.confidence < CLEAR or reading.condition.confidence < CLEAR
        for name in goal.perturbations:
            per_perturbation[name][0] += 1
            per_perturbation[name][1] += skill_ok and condition_ok
    total = len(goals)
    return {
        "goals": total,
        "skill": skill_hits / total,
        "condition": condition_hits / total,
        "both": both / total,
        "unsure": unsure,
        "per_perturbation": {k: (v[1] / v[0] if v[0] else None) for k, v in per_perturbation.items() if v[0]},
    }


def gate(seed: int, count: int = 400) -> dict[str, Any]:
    rows = real_catalogue()
    source = "real catalogue" if rows else "stand-in catalogue"
    rows = rows or stand_in_catalogue()
    index = catalogue_index(rows)
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {"seed": seed, "source": source, "rows": len(rows), "levels": {}}
    for level in (0.0, 0.2, 0.5, 1.0, None):
        key = "description_only" if level is None else f"breakage_{level:g}"
        report["levels"][key] = score(sample_goals(rows, rng, count, level), index)
    at0, at2 = report["levels"]["breakage_0"], report["levels"]["breakage_0.2"]
    report["passed"] = at0["both"] >= 0.98 and at2["both"] >= 0.95
    return report


def _format(report: Mapping[str, Any]) -> str:
    lines = [f"pattern reader gate P7, seed {report['seed']}, {report['source']} ({report['rows']} rows)",
             "level | goals | skill | condition | both | unsure   (description_only: skill = the no-guess rate)"]
    for key, part in report["levels"].items():
        lines.append(f"{key} | {part['goals']} | {part['skill']:.4f} | {part['condition']:.4f} | {part['both']:.4f} | {part['unsure']}")
    worst = report["levels"]["breakage_1"]["per_perturbation"]
    lines.append("per perturbation at breakage 1 (both read): " + ", ".join(f"{k} {v:.3f}" for k, v in worst.items()))
    lines.append(f"verdict: {'PASS' if report['passed'] else 'FAIL'} (bars: both >= 0.98 at 0, >= 0.95 at 0.2)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experimental.neural_agent.jev.system_one.pattern_reader",
                                     description="P7: reading the patterned language against breakage.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=400, help="goals per level")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    report = gate(args.seed, args.count)
    print(_format(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    return 0 if report["passed"] else 1


__all__ = [
    "NONE",
    "SkillReading",
    "WordsReading",
    "catalogue_index",
    "gate",
    "read_skill",
    "read_words",
    "sample_goals",
    "score",
]


if __name__ == "__main__":
    raise SystemExit(main())
