"""Deterministic condition reader: feature, comparator and negation words to a typed condition.

The fallback reader of docs/SYSTEM_ONE.md section 9.12 for a goal that arrives with words only
(``when`` written, ``condition`` null). The typed field is the contract; this reader exists so a goal
without it still gets an exact predicate, and so that a reading the rules are unsure of reaches the LLM
as an information request instead of a silent guess. It has no learned part: a wording either carries
one of the cues below or it does not, and the result is the same on every run.

Reading, in this order (the first family with a cue wins; cue tables are the module constants):
    health      a health word (``health``, ``HP``, ``hit points``, ``life``, ``down to``, ``worn down`` ...),
                or ``hurt`` next to a number;  side: a target word -> target_health_below, else self
    visibility  a sight word or an unseen word; with a number -> target_lost_for; with an unseen word
                (``can't``, ``out of sight``, ``hidden``, ``no sign``, ``lost`` ...) -> target_not_visible;
                otherwise -> target_visible
    distance    a distance word (``closer``, ``within``, ``range``, ``gap``, ``apart``, ``beyond`` ...), a
                number in blocks, or (weak) a bare number next to a comparator; polarity by comparator
                words, below (``closer``, ``within``, ``under``, ``less than`` ...) against above
                (``farther``, ``more than``, ``beyond``, ``past``, ``over`` ...)
    damage      a damage word (``hit``, ``damage``, ``struck``, ``under fire``, ``blow`` ...) -> on_damage
    always      an always phrase (``at all times``, ``no matter what``, ``by default`` ...), or the empty text
    unit only   no cue but a number in ticks -> target_lost_for; a percentage or fraction -> self_health_below
    default     always

The argument is ``interpret.read_number`` + ``interpret.argument_for`` (the exact number reader, 1.000 in
the ledger), so this module only decides the kind.

Confidence (``RuleReading.confidence``):
    1.0  CLEAR    one family's cues, polarity resolved, the number read when the kind needs one
    0.6  WEAK     polarity unresolved (both or neither comparator side), unit-only reading, or two numbers
    0.5  DEFAULT  no cue at all (always), or a bare fraction with no side
    0.0  INVALID  a numeric kind without a readable number: no predicate (``RuleReading.predicate()`` is None)
A reading below CLEAR is what the decision layer turns into an information request (section 9.12).

Gate (section 9.12: agreement with the generator's meanings >= 0.95 on held-out wordings):
    python -m experimental.neural_agent.jev.system_one.condition_rules --seed 0 [--show-misses 20]
renders every condition template of the (training, held-out) paraphrase split that experiments P2, P3 and
P6 used (:func:`wording_split`, the draw of ``lm_reader.training_wordings``) with every grid argument in
every surface form, reads each text, and reports per split: kind agreement, exact argument agreement on
the numeric kinds, kind-and-argument agreement, the polarity confusions (visible / not visible, below /
above, self / target) and the per-family rates. The held-out numbers are the ones that count; the rule
tables were written from the vocabulary of the generator (``goal_language`` module docstring), whose
templates the author could read, so "held out" here means held out of the tuning loop, not unseen.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from experimental.neural_agent.executive.goal_language import (
    _STEPS,
    _UNITS,
    CONDITION_FAMILIES,
    Predicate,
    _surface_forms,
    split_wordings,
)
from experimental.neural_agent.jev.system_one.data import _numpy_rng
from experimental.neural_agent.jev.system_one.interpret import KINDS, NUMERIC_UNITS, ReadNumber, argument_for, read_number

CLEAR: float = 1.0
WEAK: float = 0.6
DEFAULT: float = 0.5
INVALID: float = 0.0

BAR: float = 0.95
"""Section 9.12: agreement with the generator's meanings on held-out wordings."""

POLARITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("target_visible", "target_not_visible"),
    ("distance_below", "distance_above"),
    ("self_health_below", "target_health_below"),
)
"""The confusions that cost the encoder readers the bar (section 9.12): same feature, other side."""

ALWAYS_CUES: tuple[str, ...] = (
    "at all times", "no matter what", "the whole time", "regardless", "by default", "without any condition",
    "always", "from now on", "until told otherwise", "whatever happens", "every situation", "all the time",
    "standing order", "rest of the fight", "unconditionally", "in any case",
)
DAMAGE_CUES: tuple[str, ...] = (
    "hit", "hits", "damage", "damaged", "struck", "under fire", "hurt", "hurts", "blow", "attack connects",
    "attacked", "attack lands", "taking fire",
)
HEALTH_CUES: tuple[str, ...] = (
    "health", "hp", "hit points", "hitpoints", "vitality", "life", "down to", "worn down", "weakened",
    "nearly finished", "running low", "wounded", "health bar",
)
HEALTH_WITH_NUMBER_CUES: tuple[str, ...] = ("hurt",)
"""``badly hurt, below 30%`` is a health reading; ``if you get hurt`` without a number is damage."""
TARGET_CUES: tuple[str, ...] = (
    "target", "target's", "enemy", "enemy's", "foe", "foe's", "opponent", "opponent's", "they", "they're",
    "their", "them", "its", "quarry",
)
SIGHT_CUES: tuple[str, ...] = (
    "see", "seen", "sight", "sighting", "view", "visible", "visual", "eyes on", "spot", "spotted",
    "line of sight", "field of view", "in the open", "shows itself", "watch", "looking at",
)
UNSEEN_CUES: tuple[str, ...] = (
    "can't", "cannot", "can not", "no visual", "no eyes", "out of sight", "out of view", "out of your field",
    "hidden", "hide", "hides", "behind cover", "cover", "concealed", "vanish", "vanishes", "vanished",
    "lose line of sight", "lost line of sight", "nothing is in sight", "nothing in sight", "no sign",
    "unseen", "invisible", "lost", "lose", "losing", "lost track", "haven't seen", "not seen",
    "nobody has spotted", "missing", "gone", "cold", "trail", "disappeared", "no longer", "not visible",
    "isn't visible", "is not visible", "not in sight", "not in view", "no line of sight", "no visual contact",
)
DISTANCE_CUES: tuple[str, ...] = (
    "closer", "close", "closes", "within", "away", "range", "gap", "distance", "near", "nearer", "apart",
    "inside", "farther", "further", "beyond", "outside", "distant", "from you", "from the target",
    "of the enemy", "of the target", "blocks", "block", "metres", "meters", "pulls away",
)
BELOW_CUES: tuple[str, ...] = (
    "closer", "within", "less than", "inside", "under", "below", "shrinks", "nearer", "smaller", "creeps",
    "closes", "beneath", "drops", "shorter than",
)
ABOVE_CUES: tuple[str, ...] = (
    "farther", "further", "more than", "beyond", "past", "over", "outside", "opens", "grows", "greater than",
    "pulls away", "distant", "exceeds", "out", "longer than",
)

_CUE_TABLES: dict[str, tuple[str, ...]] = {
    "always": ALWAYS_CUES,
    "damage": DAMAGE_CUES,
    "health": HEALTH_CUES,
    "health_number": HEALTH_WITH_NUMBER_CUES,
    "target": TARGET_CUES,
    "sight": SIGHT_CUES,
    "unseen": UNSEEN_CUES,
    "distance": DISTANCE_CUES,
    "below": BELOW_CUES,
    "above": ABOVE_CUES,
}
_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    name: tuple((cue, re.compile(rf"(?<![a-z'])(?:{re.escape(cue)})(?![a-z])")) for cue in cues)
    for name, cues in _CUE_TABLES.items()
}
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class RuleReading:
    """What the rules read from one ``when`` text.

    Attributes:
        kind: One of KINDS.
        argument: The number for a numeric kind (in the kind's unit), else None. None for a numeric kind
            means no number could be read: the reading is INVALID and has no predicate.
        confidence: CLEAR, WEAK, DEFAULT or INVALID (module docstring).
        evidence: The cues that decided the reading, in table order; empty for the default.
    """

    kind: str
    argument: float | None
    confidence: float
    evidence: tuple[str, ...]

    def predicate(self) -> Predicate | None:
        """The exact predicate of this reading, or None when a numeric kind has no argument."""

        if self.kind in NUMERIC_UNITS and self.argument is None:
            return None
        return Predicate(self.kind, self.argument)


def _normalise(text: str) -> str:
    return _SPACES.sub(" ", text.lower().replace("’", "'")).strip()


def _matches(text: str, table: str) -> tuple[str, ...]:
    return tuple(cue for cue, pattern in _PATTERNS[table] if pattern.search(text))


def _numeric(kind: str, number: ReadNumber | None, evidence: tuple[str, ...], confidence: float = CLEAR) -> RuleReading:
    argument = argument_for(kind, number)
    if argument is None:
        return RuleReading(kind, None, INVALID, evidence)
    if number is not None and number.ambiguous:
        confidence = min(confidence, WEAK)
    return RuleReading(kind, argument, confidence, evidence)


def read_condition(text: str) -> RuleReading:
    """Read ``text`` (a goal's ``when``) into a typed condition by the rules of the module docstring.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    norm = _normalise(text)
    if not norm:
        return RuleReading("always", None, CLEAR, ("<empty>",))
    number = read_number(norm)
    health = _matches(norm, "health") + (_matches(norm, "health_number") if number is not None else ())
    if health:
        target = _matches(norm, "target")
        return _numeric("target_health_below" if target else "self_health_below", number, health + target)
    sight, unseen = _matches(norm, "sight"), _matches(norm, "unseen")
    if sight or unseen:
        if number is not None:
            return _numeric("target_lost_for", number, sight + unseen)
        if unseen:
            return RuleReading("target_not_visible", None, CLEAR, sight + unseen)
        return RuleReading("target_visible", None, CLEAR, sight)
    distance = _matches(norm, "distance")
    below, above = _matches(norm, "below"), _matches(norm, "above")
    in_blocks = number is not None and number.unit == "blocks"
    bare = number is not None and number.unit is None and number.value >= 1.0 and (below or above)
    if distance or in_blocks or bare:
        # A bare number with only a comparator ("while it's over 3 out") is a distance reading, but a weak
        # one: nothing names the feature.
        sure = CLEAR if (distance or in_blocks) else WEAK
        if len(below) == len(above):
            return _numeric("distance_below", number, distance + below + above, confidence=WEAK)
        if len(above) > len(below):
            return _numeric("distance_above", number, distance + above, confidence=sure)
        return _numeric("distance_below", number, distance + below, confidence=sure)
    damage = _matches(norm, "damage")
    if damage:
        return RuleReading("on_damage", None, CLEAR, damage)
    always = _matches(norm, "always")
    if always:
        return RuleReading("always", None, CLEAR, always)
    if number is not None and number.unit == "ticks":
        return _numeric("target_lost_for", number, ("<ticks>",), confidence=WEAK)
    if number is not None and (number.unit == "fraction" or 0.0 < number.value < 1.0):
        return _numeric("self_health_below", number, ("<fraction>",), confidence=DEFAULT)
    return RuleReading("always", None, DEFAULT, ())


# ---------------------------------------------------------------------------------------------- the gate


def wording_split(seed: int) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """(training, held-out) paraphrases per family: the draw of ``lm_reader.training_wordings(seed)``, which is
    the split of ``interpret_benchmark.gate_sets(seed, ...)`` (experiments P2, P3 and P6)."""

    sets_key, _ = jax.random.split(jax.random.fold_in(jax.random.key(seed), 2))
    return split_wordings(_numpy_rng(jax.random.split(sets_key, 5)[0]), 0.25)


def rendered_conditions(wordings: Mapping[str, Sequence[str]]) -> Iterator[tuple[str, str, float | None]]:
    """Yield (text, kind, argument): every condition template of ``wordings`` with every grid argument in every
    surface form (``goal_language._surface_forms``); the always family also yields the empty text."""

    for family in CONDITION_FAMILIES:
        kind = family.family_id
        templates = tuple(wordings.get(kind, ()))
        if not family.numeric:
            for template in templates:
                yield template, kind, None
            if kind == "always" and templates:
                yield "", kind, None
            continue
        step = _STEPS[_UNITS[kind]]
        count = round((family.argument_high - family.argument_low) / step)
        for index in range(count + 1):
            argument = round(family.argument_low + step * index, 2)
            for template in templates:
                for form in _surface_forms(kind, argument):
                    yield template.replace("{n}", form), kind, argument


def _same_argument(read: float | None, truth: float | None) -> bool:
    if read is None or truth is None:
        return read is None and truth is None
    return abs(read - truth) <= 1e-6


def score(rows: Iterable[tuple[str, str, float | None]]) -> dict[str, Any]:
    """Read every row and return the rates, the polarity confusions, the per-family rates and the misses."""

    kind_hits = full_hits = numeric = argument_hits = 0
    per_family: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    confusions: Counter[str] = Counter()
    unsure = 0
    misses: list[dict[str, Any]] = []
    total = 0
    for text, kind, argument in rows:
        total += 1
        reading = read_condition(text)
        family_totals[kind] += 1
        ok_kind = reading.kind == kind
        ok_argument = _same_argument(reading.argument, argument)
        if kind in NUMERIC_UNITS:
            numeric += 1
            argument_hits += int(ok_argument)
        kind_hits += int(ok_kind)
        if ok_kind and ok_argument:
            full_hits += 1
            per_family[kind] += 1
        else:
            misses.append({"text": text, "truth": [kind, argument], "read": [reading.kind, reading.argument],
                           "confidence": reading.confidence, "evidence": list(reading.evidence)})
            for left, right in POLARITY_PAIRS:
                if {kind, reading.kind} == {left, right}:
                    confusions[f"{kind}->{reading.kind}"] += 1
        if reading.confidence < CLEAR:
            unsure += 1
    return {
        "total": total,
        "kind_agreement": kind_hits / total if total else float("nan"),
        "argument_exact": argument_hits / numeric if numeric else float("nan"),
        "agreement": full_hits / total if total else float("nan"),
        "unsure": unsure,
        "polarity_confusions": dict(sorted(confusions.items())),
        "per_family": {kind: per_family[kind] / family_totals[kind] for kind in KINDS if family_totals[kind]},
        "misses": misses,
    }


def gate(seed: int) -> dict[str, Any]:
    """Score the training and held-out splits of ``wording_split(seed)``; ``passed`` is the held-out bar."""

    train, held = wording_split(seed)
    report = {
        "seed": seed,
        "bar": BAR,
        "training": score(rendered_conditions(train)),
        "heldout": score(rendered_conditions(held)),
    }
    report["passed"] = report["heldout"]["kind_agreement"] >= BAR and report["heldout"]["agreement"] >= BAR
    return report


def _format(report: Mapping[str, Any], show_misses: int) -> str:
    lines = [f"condition rules gate, seed {report['seed']}, bar {report['bar']:.2f} on held-out wordings"]
    lines.append("split | texts | kind | argument exact | kind+argument | unsure | polarity confusions")
    for name in ("training", "heldout"):
        part = report[name]
        lines.append(
            f"{name} | {part['total']} | {part['kind_agreement']:.4f} | {part['argument_exact']:.4f} | "
            f"{part['agreement']:.4f} | {part['unsure']} | {part['polarity_confusions'] or '-'}"
        )
    lines.append("per family (kind+argument): " + ", ".join(
        f"{kind} {report['training']['per_family'].get(kind, float('nan')):.3f}/"
        f"{report['heldout']['per_family'].get(kind, float('nan')):.3f}" for kind in KINDS
    ) + "  (training/held-out)")
    lines.append(f"verdict: {'PASS' if report['passed'] else 'FAIL'}")
    for name in ("training", "heldout"):
        shown = report[name]["misses"][:show_misses]
        if shown:
            lines.append(f"{name} misses ({len(report[name]['misses'])}), first {len(shown)}:")
            lines.extend(f"  {json.dumps(miss, ensure_ascii=False)}" for miss in shown)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.condition_rules",
        description="Gate of the deterministic condition reader on the generator's wording split (SYSTEM_ONE 9.12).",
    )
    parser.add_argument("--seed", type=int, default=0, help="the wording split seed (experiments P2/P3/P6 used 0)")
    parser.add_argument("--show-misses", type=int, default=10, help="misses to print per split")
    parser.add_argument("--json", type=str, default=None, help="also write the report (without misses beyond --show-misses) here")
    args = parser.parse_args(argv)
    report = gate(args.seed)
    print(_format(report, args.show_misses))
    if args.json:
        trimmed = json.loads(json.dumps(report))
        for name in ("training", "heldout"):
            trimmed[name]["misses"] = trimmed[name]["misses"][: args.show_misses]
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(trimmed, handle, indent=1, ensure_ascii=False)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
