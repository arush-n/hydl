"""The goal-language generator: the procedural teacher for reading natural-language goals.

``docs/SYSTEM_ONE.md`` section 8.11 and ``docs/FORMULAS.md`` (M13). Gate G3-NL (M12) needs goals written
the way the top LLM will write them, typed fields plus unique words, before that LLM may run (section 8.10,
owner answer 4). This module writes them with the meaning known by construction: every slot of a sampled
``ExecutiveSection`` comes with a ``SlotMeaning`` (intent family, resolved skill id, predicate over B_t),
and ``teacher`` labels a belief row with the rule those meanings imply. Data-side code: numpy and the
standard library only, no JAX at call time (``contracts.py`` imports JAX when this module is imported).

Families. An intent family is one thing a skill does: a canonical description in the register of
``skill_library/descriptions.py`` and at least twelve paraphrases. pursue, evade and hold name the real
skills ``pursuit``, ``evader`` and ``idle`` and their descriptions ARE the library's; the seven others name
no real skill yet and populate the synthetic catalogues. A condition family is one predicate over B_t with
at least twelve templates; a numeric one carries the placeholder ``{n}``.

Predicates on one belief row in physical units, ``f`` the feature and ``known`` its flag::

    always                 True
    on_damage              known and f[ticks_since_damage] < 40         (J0's window, jev/scripted.py)
    target_visible         known and f[target_visible] > 0.5             (belief/features.py threshold)
    target_not_visible     known and f[target_visible] <= 0.5
    target_lost_for(n)     known and f[ticks_since_target_seen] > n      n ticks in 20..200, step 10
    distance_below(n)      known and f[target_distance_blocks] < n       n blocks in 2..12, step 1
    distance_above(n)      known and f[target_distance_blocks] > n
    self_health_below(x)   known and f[self_health_fraction] < x         x in 0.2..0.8, step 0.05
    target_health_below(x) known and f[target_health_fraction] < x

Units: the belief publishes the two tick counters as ``counter / TICKS_SATURATION`` (``belief/core.py``), so
a caller multiplies them back to ticks first, as J0 does. The other columns are already in these units.

Teacher (M13), slot k with validity v_k, priority p_k and predicate c_k::

    S_cond    = {k : v_k and c_k != always and c_k(B_t)}
    chosen    = argmax_{k in S_cond} p_k                    if S_cond is non-empty
              = argmax_{k : v_k and c_k = always} p_k      elif that set is non-empty
              = idle                                        otherwise
    y_replace = [chosen != current]                         ties on p_k go to the lower slot (argmax, as J0)

On the composition state (pursue/always at 1.0, evade/on_damage at 0.5) this is exactly (M10).

Sampling choices this module makes where section 8.11 is silent: 1..max_goals goals, uniform; families
uniform without replacement among those that resolve; a slot is unconditional with probability
ALWAYS_SHARE = 1/3, else its kind is uniform over the non-always kinds no other slot uses; numeric
arguments uniform on the grids above; priorities distinct on {0.01, ..., 1.00}; horizon_seconds an integer
in 2..30; a deadline with probability DEADLINE_RATE = 0.5 at issued_tick + U{40..600}; params empty; an
``always`` condition written as "" with probability EMPTY_ALWAYS_RATE = 0.5. Surface forms of a number,
uniform: blocks "5", "5 blocks", "five", "five blocks", "5m", "5 metres"; ticks "60 ticks", "sixty ticks",
"60", "sixty", "60 game ticks"; fractions "30%", "30 percent", "0.3", "thirty percent", and where exact
"a fifth", "a quarter", "two fifths", "half", "three fifths", "three quarters", "four fifths".

Splits (section 8.11): ``split_wordings`` holds out a fraction of every family's paraphrases (at least one
on each side); ``split_families`` holds out whole intent families, never pursue, evade or hold.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from experimental.neural_agent.contracts import (
    Condition,
    BELIEF_FEATURE_COUNT,
    MAX_GOALS,
    SKILL_IDLE,
    _finite_number,
    feature_index,
)
from experimental.neural_agent.shared_state import ExecutiveSection, GoalEntry
from experimental.neural_agent.skill_library.descriptions import SKILL_DESCRIPTIONS

DAMAGE_WINDOW_TICKS: float = 40.0
"""The damage window in ticks: ``jev/scripted.py`` EVADE_DAMAGE_WINDOW_TICKS, the J0 rule of (M10)."""
VISIBLE_THRESHOLD: float = 0.5
"""``target_visible`` above this is visible: ``belief/features.py`` VISIBLE_THRESHOLD."""
ALWAYS_SHARE: float = 1.0 / 3.0
"""Probability that a sampled slot carries no condition (the ``always`` family)."""
EMPTY_ALWAYS_RATE: float = 0.5
"""Probability that an ``always`` condition is written as the empty string."""
DEADLINE_RATE: float = 0.5
"""Probability that a sampled goal carries a deadline."""

STOP_WORDS: frozenset[str] = frozenset(
    """a an the and or but nor so if then than as because though unless until when whenever while once where
    wherever at by for from in into of off on onto out over to up down with within without about around after
    before past under below above between behind beyond through toward towards near across along against upon
    is are was were be been being am do does did done have has had having can could will would should must may
    might shall i me my we us our you your yours yourself he him his she her it its itself they them their this
    that these those there here what which who whom whatever whoever anyone anything everyone everything someone
    something nobody nothing none not no any all some each every either neither both few other another such own
    same just only also too very rather now again ever even yet more most less least much many dont doesnt didnt
    isnt arent wasnt cant wont youre youve youll theyre thats theres whats weve ive im""".split()
)
"""Function words only, lower-cased with apostrophes removed: the content-word rule of section 8.11 ignores them."""

# Condition kind -> the belief column it reads (None for always), and the unit of a numeric argument.
_FEATURE_OF: dict[str, str | None] = {
    "always": None,
    "on_damage": "ticks_since_damage",
    "target_visible": "target_visible",
    "target_not_visible": "target_visible",
    "target_lost_for": "ticks_since_target_seen",
    "distance_below": "target_distance_blocks",
    "distance_above": "target_distance_blocks",
    "self_health_below": "self_health_fraction",
    "target_health_below": "target_health_fraction",
}
_UNITS: dict[str, str] = {
    "target_lost_for": "ticks",
    "distance_below": "blocks",
    "distance_above": "blocks",
    "self_health_below": "fraction",
    "target_health_below": "fraction",
}
_STEPS: dict[str, float] = {"ticks": 10.0, "blocks": 1.0, "fraction": 0.05}
_FRACTION_WORDS: dict[int, str] = {
    20: "a fifth", 25: "a quarter", 40: "two fifths", 50: "half", 60: "three fifths", 75: "three quarters",
    80: "four fifths",
}
_ONES: tuple[str, ...] = tuple(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split()
)
_TENS: tuple[str, ...] = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


@dataclass(frozen=True)
class IntentFamily:
    """One thing a skill does, in many words.

    Attributes:
        family_id: The family's name, for example ``"pursue"``.
        description: The canonical skill description, in the register of ``SKILL_DESCRIPTIONS``.
        skill_ids: The real skill ids the family names today; empty for a family with no real skill yet.
        paraphrases: At least twelve ways the LLM might write the intent.
    """

    family_id: str
    description: str
    skill_ids: tuple[str, ...]
    paraphrases: tuple[str, ...]


@dataclass(frozen=True)
class ConditionFamily:
    """One predicate over B_t, in many words.

    Attributes:
        family_id: The predicate kind, for example ``"on_damage"``.
        paraphrases: At least twelve templates; a numeric family's templates carry ``{n}`` once.
        numeric: Whether the predicate takes an argument.
        argument_low, argument_high: The sampling range of the argument (None when not numeric).
    """

    family_id: str
    paraphrases: tuple[str, ...]
    numeric: bool
    argument_low: float | None
    argument_high: float | None


@dataclass(frozen=True)
class Predicate:
    """A condition with its argument: the meaning of a ``when`` text (module docstring, Predicates).

    Raises:
        ValueError: If the kind is unknown, a numeric kind lacks a finite argument, or another kind has one.
        TypeError: If the argument is not a real number.
    """

    kind: str
    argument: float | None = None

    def __post_init__(self) -> None:
        """Check the kind, and that exactly the numeric kinds carry a finite argument."""

        if self.kind not in _FEATURE_OF:
            raise ValueError(f"unknown condition kind {self.kind!r}; expected one of {tuple(_FEATURE_OF)}")
        if (self.kind in _UNITS) != (self.argument is not None):
            raise ValueError(f"condition {self.kind!r} {'needs an' if self.kind in _UNITS else 'takes no'} argument")
        if self.argument is not None:
            object.__setattr__(self, "argument", _finite_number(self.argument, f"Predicate({self.kind!r}).argument"))

    def evaluate(self, features: Sequence[float], known: Sequence[bool]) -> bool:
        """Return the predicate on one belief row in physical units; a feature not known makes it False.

        Raises:
            ValueError: If ``features`` or ``known`` does not hold BELIEF_FEATURE_COUNT values.
        """

        if len(features) != BELIEF_FEATURE_COUNT or len(known) != BELIEF_FEATURE_COUNT:
            raise ValueError(f"features and known must hold {BELIEF_FEATURE_COUNT} values each")
        if self.kind == "always":
            return True
        column = feature_index(_FEATURE_OF[self.kind])
        if not known[column]:
            return False
        value = float(features[column])
        if self.kind == "on_damage":
            return value < DAMAGE_WINDOW_TICKS
        if self.kind == "target_visible":
            return value > VISIBLE_THRESHOLD
        if self.kind == "target_not_visible":
            return value <= VISIBLE_THRESHOLD
        if self.kind in ("target_lost_for", "distance_above"):
            return value > self.argument
        return value < self.argument


ALWAYS = Predicate("always", None)
"""The empty condition."""


@dataclass(frozen=True)
class SlotMeaning:
    """What one goal slot means by construction: its intent family, its skill and its condition."""

    intent_family: str
    skill_id: str
    predicate: Predicate


INTENT_FAMILIES: tuple[IntentFamily, ...] = (
    IntentFamily("pursue", SKILL_DESCRIPTIONS["pursuit"], ("pursuit",), (
        "Go after the target and close in.", "We want to be right on top of the enemy.", "Chase it down.",
        "Run the enemy down and don't let up.", "Hunt them.", "Close the gap to the target as fast as you can.",
        "Stay on its heels until you catch it.", "get in the foe's face", "Pursue the target relentlessly.",
        "Track the enemy down and finish the chase.", "Engage the selected target and catch it.",
        "Don't let it get away.", "I want you chasing the enemy until you're within arm's reach.",
        "Your job: tail the target and grab it.", "run down the quarry", "Stay in pursuit of the enemy.",
    )),
    IntentFamily("evade", SKILL_DESCRIPTIONS["evader"], ("evader",), (
        "Get away from the enemy.", "Back off and put some room between you and them.",
        "Break contact and keep your distance.", "Stay out of its reach.", "Evade the target.",
        "Keep three to six blocks between you and the target.", "Disengage and don't let it get close.",
        "dodge away and stay clear of the foe", "We need space from the attacker right now.",
        "Kite it at a safe range.", "Back away and keep out of reach.", "Open up the range and stay untouchable.",
        "Play defensive and stay away from the target.", "Pull away and maintain a safe gap.", "slip away from them",
    )),
    IntentFamily("hold", SKILL_DESCRIPTIONS["idle"], ("idle",), (
        "Do nothing for now.", "Stay put.", "Stand still and take no action.", "Wait here.",
        "Sit tight until further notice.", "Hold still and wait.", "Don't move.", "Stand by and take no action yet.",
        "remain motionless", "Freeze where you are.", "Pause and wait for orders.", "Stay idle for a while.",
        "just stand there", "Take a breather and stay in place.", "Stop all activity for the moment.",
    )),
    IntentFamily(
        "reach_point",
        "Reach point: travel to the given waypoint and stop within one block of it. The movement skill when the "
        "executive names a destination and no target needs attention.",
        (),
        (
            "Head over to the marked spot.", "Get yourself to the waypoint.", "Make your way to those coordinates.",
            "Move to the location I gave you.", "Walk over to the spot and stop there.", "go to the rally point",
            "Travel to the destination.", "We need you at the marker.", "Run to the checkpoint.",
            "Navigate to the goal location.", "Reach the spot on the map.", "Arrive at the meeting place.",
            "Your task is to show up at the flagged position.", "Get to the given waypoint and stop.",
            "relocate to the new position",
        ),
    ),
    IntentFamily(
        "patrol",
        "Patrol: walk a loop through the given waypoints and watch for targets along the way. The routine skill "
        "when no target is known and an area must be covered.",
        (),
        (
            "Walk the perimeter.", "Make rounds of the camp.", "Go back and forth across the zone.",
            "Do laps around the base.", "Keep circling the route and stay alert.", "Cover the area on a loop.",
            "pace the boundary", "Keep moving between the posts.", "Run a circuit of the grounds.",
            "Go up and down the line and keep watch.", "Patrol the area.",
            "Make a regular circuit and report anything odd.", "Keep the route covered.",
            "Walk the usual loop through the waypoints.", "Do a lap of the compound every so often.",
        ),
    ),
    IntentFamily(
        "retreat_to_cover",
        "Retreat to cover: move behind the nearest solid block that breaks line of sight to the target and stay "
        "there. The recovery skill when health is low or the target outranges the agent.",
        (),
        (
            "Duck behind something solid.", "Get to cover.", "Find shelter and wait there.",
            "Hide behind the nearest wall.", "Take cover.", "Fall back behind a rock.", "Get out of its line of fire.",
            "hunker down behind an obstacle", "Pull back to somewhere protected.", "Seek shelter now.",
            "Put a wall between you and the enemy.", "Head for cover.", "go hide",
            "Retreat to a covered spot and stay hidden.", "Get behind hard terrain where it can't hit you.",
        ),
    ),
    IntentFamily(
        "guard_position",
        "Guard position: stay within a few blocks of the given position and engage anything that comes close. The "
        "defensive skill when a place, not a target, must be protected.",
        (),
        (
            "Hold this spot and fight off anyone who approaches.", "Defend the position.", "Protect the gate.",
            "Stand watch over this location and repel intruders.", "Don't let anyone take this point.",
            "Keep the area around you secure.", "Stay at your post and fend off attackers.", "watch over the doorway",
            "Hold the line here.", "Defend this ground.", "Man your post.", "Stand guard here.",
            "Keep watch on the bridge and drive back attackers.", "Make sure nobody gets past this point.",
            "Engage anything that comes close to the given position.",
        ),
    ),
    IntentFamily(
        "search_target",
        "Search target: sweep toward the last known position of the target and look around until it is seen "
        "again. The recovery skill when the target has been lost from view.",
        (),
        (
            "Go find the enemy.", "Look for it where you last saw it.", "Figure out where the foe went.",
            "Comb the area until you find it.", "Check its last known spot.", "Search the area for the target.",
            "Locate the enemy again.", "scout around until you spot them", "Retrace its steps.",
            "Look around for the target.", "Sniff out where it's hiding.", "Pick up its trail again.",
            "seek out the foe", "Get eyes back on the enemy.", "Sweep toward its last known position and find it.",
        ),
    ),
    IntentFamily(
        "flank",
        "Flank: circle around the target at medium range to come at it from the side or the back. The positioning "
        "skill before an attack when meeting the target head on is too risky.",
        (),
        (
            "Circle around and hit it from behind.", "Get to its side.", "Swing wide and strike from the rear.",
            "Work your way around behind the enemy.", "Outflank them.", "Go around and attack its back.",
            "Approach from an angle, not head on.", "loop around to its blind spot",
            "Sneak around the edge and hit them from behind.", "Take the long way around and ambush it.",
            "Move to the enemy's flank.", "Curve around the foe before you strike.",
            "Don't charge straight in; go around.", "Come at the target from the side.",
            "get behind them without being seen",
        ),
    ),
    IntentFamily(
        "follow_ally",
        "Follow ally: stay a few blocks behind the named ally and match its pace. The escort skill when the "
        "executive wants the agent to move with a friend rather than alone.",
        (),
        (
            "Tag along with your teammate.", "Stick with the ally.", "Follow your teammate.",
            "Shadow your partner and keep pace.", "Stay close to your buddy.", "Go wherever your companion goes.",
            "Keep up with the squad leader.", "trail behind your friend", "Escort our ally.",
            "Accompany the other player.", "Walk side by side with your partner.", "Don't leave your partner behind.",
            "Move together with the group.", "Be your companion's shadow.", "travel alongside the convoy",
        ),
    ),
)
"""Ten intent families; pursue, evade and hold name the real skills."""

CONDITION_FAMILIES: tuple[ConditionFamily, ...] = (
    ConditionFamily("always", (
        "at all times", "no matter what", "the whole time", "regardless of the situation", "by default",
        "without any condition", "always", "from now on", "until told otherwise", "whatever happens",
        "in every situation", "all the time", "as a standing order", "for the rest of the fight",
    ), False, None, None),
    ConditionFamily("on_damage", (
        "if you get hit", "the moment you take damage", "after being struck", "while under fire", "when damaged",
        "right after a hit lands on you", "as soon as something hurts you", "while you're taking hits",
        "when an attack connects", "if you've just been hurt", "whenever you take a blow", "upon taking damage",
        "the instant an enemy lands a hit", "if you are being attacked",
    ), False, None, None),
    ConditionFamily("target_visible", (
        "while you can see the target", "as long as the enemy is in sight", "when it's in view",
        "whenever the target is visible", "if you have eyes on them", "while there's a clear line of sight",
        "once you spot it", "when the foe shows itself", "while it stays in your field of view", "if you can see it",
        "as long as you have a visual", "while the quarry is in plain view", "when they're out in the open",
        "if the enemy can be seen",
    ), False, None, None),
    ConditionFamily("target_not_visible", (
        "while you can't see the target", "when it's out of sight", "if the enemy is hidden",
        "whenever you lose line of sight", "while it's out of view", "if you have no visual",
        "as long as the target stays hidden", "while they're behind cover", "if the foe vanishes from view",
        "when you have no eyes on it", "while nothing is in sight", "if it's concealed",
        "whenever the target can't be seen", "while it's out of your field of view",
    ), False, None, None),
    ConditionFamily("target_lost_for", (
        "if the target has been out of sight for more than {n}", "after losing it for over {n}",
        "once you haven't seen the enemy in {n}", "when it's been gone longer than {n}",
        "if they stay hidden past {n}", "after {n} without a sighting", "when the trail has been cold for {n}",
        "if nobody has spotted it for {n}", "once it has been missing for more than {n}",
        "if you lose track of the foe for longer than {n}", "when there's been no sign of them for {n}",
        "after the target stays out of view beyond {n}", "if the enemy has been invisible for over {n}",
        "should it remain unseen for longer than {n}",
    ), True, 20.0, 200.0),
    ConditionFamily("distance_below", (
        "if the target gets closer than {n}", "once the enemy is within {n}", "when it's less than {n} away",
        "if they come inside {n}", "at a range under {n}", "whenever the gap shrinks below {n}",
        "when the foe closes to within {n}", "if the distance to the target drops under {n}",
        "once it is nearer than {n}", "when you're inside {n} of the enemy",
        "while the target stays within {n} of you", "if it creeps to less than {n} from you",
        "when the two of you are under {n} apart", "as soon as the gap is smaller than {n}",
    ), True, 2.0, 12.0),
    ConditionFamily("distance_above", (
        "if the target is farther than {n}", "when the enemy is more than {n} away", "once the gap opens past {n}",
        "if they get beyond {n}", "while it's over {n} out", "when the distance grows past {n}",
        "if the foe is further away than {n}", "once it's outside {n}", "when you're more than {n} from the target",
        "if it pulls away beyond {n}", "while the target stays out past {n}",
        "as long as the range is greater than {n}", "if the two of you end up over {n} apart",
        "when it's more than {n} distant",
    ), True, 2.0, 12.0),
    ConditionFamily("self_health_below", (
        "if your health drops below {n}", "when you're under {n} health", "once you're down to less than {n} HP",
        "if you fall beneath {n} of your max health", "whenever your HP is lower than {n}",
        "once your own health slips under {n}", "when your life is under {n}", "if your hit points sink below {n}",
        "while you're at less than {n} health", "when you're badly hurt, below {n}",
        "if you are running low, under {n} health", "once you're left with under {n} of your health",
        "if your own vitality goes below {n}", "when you've lost so much that you're under {n} health",
    ), True, 0.2, 0.8),
    ConditionFamily("target_health_below", (
        "if the target's health falls under {n}", "once the enemy is below {n} health",
        "when they're weakened below {n}", "if the foe drops to less than {n} HP", "when its health bar is under {n}",
        "once the target is nearly finished, under {n}", "if their HP sinks beneath {n}",
        "whenever the enemy's life is lower than {n}", "if the target is hurt below {n} health",
        "when the opponent has less than {n} of its health left", "once they're down to under {n}",
        "if the enemy is worn down below {n}", "when the target's hit points go under {n}",
        "as soon as the foe's health is less than {n}",
    ), True, 0.2, 0.8),
)
"""The nine condition families of section 8.11, ``always`` first."""

NOTES: tuple[str, ...] = (
    "Logged for the after-action review.", "Same plan as last time.", "Map data may be stale.",
    "Revisit after the next report.", "No change from the previous version.", "Budget is fine for now.",
    "Decided after the last skirmish.", "Low confidence in this one.", "Keep the chatter to a minimum.",
    "Carried over from the earlier plan.",
)
"""Neutral remarks a goal's ``note`` carries as distractors: no intent and no condition."""

Catalogue = Mapping[str, str]
"""skill_id -> description, idle first."""
Wordings = Mapping[str, tuple[str, ...]]
"""family_id -> the paraphrases allowed for it, intent and condition families in one mapping."""

REAL_CATALOGUE: dict[str, str] = {SKILL_IDLE: SKILL_DESCRIPTIONS[SKILL_IDLE], **SKILL_DESCRIPTIONS}
"""The real catalogue, ``SKILL_DESCRIPTIONS`` with idle first."""


def synthetic_catalogue(
    rng: np.random.Generator, size: int, families: Sequence[IntentFamily] = INTENT_FAMILIES
) -> dict[str, str]:
    """Return a K-skill catalogue: idle (its real description) first, then ``skill_1 .. skill_{K-1}``.

    The ``skill_<n>`` descriptions are the canonical descriptions of distinct families drawn uniformly from
    ``families`` without the hold family (the one that names idle).

    Raises:
        ValueError: If ``size`` is below 1, or ``size - 1`` exceeds the families available.
    """

    pool = [family for family in families if SKILL_IDLE not in family.skill_ids]
    if not 1 <= size <= len(pool) + 1:
        raise ValueError(f"size must be in 1..{len(pool) + 1} for {len(pool)} families, got {size}")
    picks = rng.choice(len(pool), size=size - 1, replace=False)
    catalogue = {SKILL_IDLE: SKILL_DESCRIPTIONS[SKILL_IDLE]}
    catalogue.update({f"skill_{number}": pool[pick].description for number, pick in enumerate(picks, start=1)})
    return catalogue


def resolve(family: IntentFamily, catalogue: Catalogue) -> str | None:
    """Return the id ``family`` resolves to: a real id it names, else the id with its description, else None."""

    for skill_id in family.skill_ids:
        if skill_id in catalogue:
            return skill_id
    for skill_id, description in catalogue.items():
        if description == family.description:
            return skill_id
    return None


def all_wordings() -> dict[str, tuple[str, ...]]:
    """Return every family's paraphrases, intent families first."""

    return {family.family_id: family.paraphrases for family in (*INTENT_FAMILIES, *CONDITION_FAMILIES)}


def split_wordings(
    rng: np.random.Generator, heldout_fraction: float = 0.25
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Split every family's paraphrases into disjoint (train, held-out) parts, split (i) of section 8.11.

    A family of m paraphrases holds out round(fraction * m), clipped to 1..m-1; each part keeps the order.

    Raises:
        ValueError: If ``heldout_fraction`` is not strictly between 0 and 1.
    """

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must be in (0, 1), got {heldout_fraction}")
    train, heldout = {}, {}
    for family_id, texts in all_wordings().items():
        count = min(max(round(heldout_fraction * len(texts)), 1), len(texts) - 1)
        held = set(rng.permutation(len(texts))[:count].tolist())
        heldout[family_id] = tuple(text for index, text in enumerate(texts) if index in held)
        train[family_id] = tuple(text for index, text in enumerate(texts) if index not in held)
    return train, heldout


def split_families(rng: np.random.Generator, heldout_count: int = 2) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the intent families into (train, held-out) ids, split (ii); pursue, evade and hold always train.

    Raises:
        ValueError: If ``heldout_count`` is negative or exceeds the families that name no real skill.
    """

    eligible = [family.family_id for family in INTENT_FAMILIES if not family.skill_ids]
    if not 0 <= heldout_count <= len(eligible):
        raise ValueError(f"heldout_count must be in 0..{len(eligible)}, got {heldout_count}")
    held = {eligible[index] for index in rng.choice(len(eligible), size=heldout_count, replace=False)}
    train = tuple(family.family_id for family in INTENT_FAMILIES if family.family_id not in held)
    return train, tuple(family_id for family_id in eligible if family_id in held)


def _words(number: int) -> str:
    """English words for 0..999: 7 -> "seven", 45 -> "forty-five", 120 -> "one hundred and twenty"."""

    if number >= 100:
        hundreds, rest = divmod(number, 100)
        return f"{_ONES[hundreds]} hundred" + (f" and {_words(rest)}" if rest else "")
    if number < 20:
        return _ONES[number]
    tens, ones = divmod(number, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _surface_forms(kind: str, argument: float) -> tuple[str, ...]:
    """The ways ``{n}`` is written for ``argument`` (module docstring, surface forms)."""

    unit = _UNITS[kind]
    if unit == "fraction":
        percent = round(argument * 100)
        exact = (_FRACTION_WORDS[percent],) if percent in _FRACTION_WORDS else ()
        return (f"{percent}%", f"{percent} percent", f"{argument:g}", f"{_words(percent)} percent", *exact)
    whole = round(argument)
    if unit == "ticks":
        return (f"{whole} ticks", f"{_words(whole)} ticks", str(whole), _words(whole), f"{whole} game ticks")
    return (str(whole), f"{whole} blocks", _words(whole), f"{_words(whole)} blocks", f"{whole}m", f"{whole} metres")


def render_condition(family: ConditionFamily, template: str, argument: float | None, rng: np.random.Generator) -> str:
    """Return the ``when`` text: ``template`` with ``{n}`` written in a random surface form of ``argument``.

    An ``always`` condition is written as "" with probability EMPTY_ALWAYS_RATE.

    Raises:
        ValueError: If ``template`` is not one of the family's, or ``argument`` does not fit the family
            (missing, extra, outside the range, or off the grid of whole blocks, ticks or percent).
    """

    if template not in family.paraphrases:
        raise ValueError(f"{template!r} is not a paraphrase of {family.family_id!r}")
    Predicate(family.family_id, argument)  # the kind and the argument agree
    if family.family_id == ALWAYS.kind and rng.random() < EMPTY_ALWAYS_RATE:
        return ""
    if not family.numeric:
        return template
    scaled = argument * (100.0 if _UNITS[family.family_id] == "fraction" else 1.0)
    if not family.argument_low <= argument <= family.argument_high or abs(scaled - round(scaled)) > 1e-6:
        raise ValueError(f"argument {argument} is off the grid of {family.family_id!r}")
    forms = _surface_forms(family.family_id, argument)
    return template.replace("{n}", forms[rng.integers(len(forms))])


def _sample_argument(family: ConditionFamily, rng: np.random.Generator) -> float:
    """A uniform point of the family's grid: whole blocks, ticks in steps of 10, fractions in steps of 0.05."""

    step = _STEPS[_UNITS[family.family_id]]
    count = round((family.argument_high - family.argument_low) / step)
    return round(family.argument_low + step * int(rng.integers(count + 1)), 2)


def sample_state(
    rng: np.random.Generator,
    catalogue: Catalogue,
    wordings: Wordings,
    *,
    version: int = 1,
    issued_tick: int = 0,
    intent_families: Sequence[str] | None = None,
    max_goals: int = MAX_GOALS,
    note_rate: float = 0.25,
    hint_rate: float = 0.0,
) -> tuple[ExecutiveSection, tuple[SlotMeaning, ...]]:
    """Sample one executive section in the LLM's words, with the meaning of every slot.

    Goals name distinct skills, from families that resolve in ``catalogue`` (and are in ``intent_families``
    when given); ``intent`` and ``when`` come from ``wordings``; at most one slot carries a given non-always
    condition kind; with probability ``hint_rate`` (never for always) the goal carries its typed fields,
    ``constraints`` the kind and ``condition`` the kind with its argument (SYSTEM_ONE 9.12);
    ``note`` is one of NOTES with probability ``note_rate``. Distributions: module docstring.

    Raises:
        ValueError: If ``max_goals`` is outside 1..MAX_GOALS, a rate is outside [0, 1], an intent family is
            unknown, no family resolves, or a wording is not one of its family's paraphrases.
        KeyError: If ``wordings`` lacks a family the sample uses.
    """

    if not 1 <= max_goals <= MAX_GOALS:
        raise ValueError(f"max_goals must be in 1..{MAX_GOALS}, got {max_goals}")
    for name, rate in (("note_rate", note_rate), ("hint_rate", hint_rate)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {rate}")
    unknown = set(intent_families or ()) - {family.family_id for family in INTENT_FAMILIES}
    if unknown:
        raise ValueError(f"unknown intent families {sorted(unknown)}")
    candidates: dict[str, IntentFamily] = {}  # skill id -> the first family that resolves to it
    for family in INTENT_FAMILIES:
        skill_id = resolve(family, catalogue)
        allowed = intent_families is None or family.family_id in intent_families
        if allowed and skill_id is not None and skill_id not in candidates:
            candidates[skill_id] = family
    if not candidates:
        raise ValueError("no intent family resolves in this catalogue")
    pairs = list(candidates.items())
    count = int(rng.integers(1, min(max_goals, len(pairs)) + 1))
    picks = rng.choice(len(pairs), size=count, replace=False)
    priorities = rng.choice(np.arange(1, 101), size=count, replace=False) / 100.0
    conditions = {family.family_id: family for family in CONDITION_FAMILIES}
    goals, meanings, used = [], [], set()
    for slot, pick in enumerate(picks):
        skill_id, family = pairs[pick]
        kind = ALWAYS.kind
        if rng.random() >= ALWAYS_SHARE:
            open_kinds = [name for name in conditions if name != ALWAYS.kind and name not in used]
            kind = open_kinds[rng.integers(len(open_kinds))]
            used.add(kind)
        condition = conditions[kind]
        argument = _sample_argument(condition, rng) if condition.numeric else None
        templates = wordings[kind]
        when = render_condition(condition, templates[rng.integers(len(templates))], argument, rng)
        texts = wordings[family.family_id]
        intent = texts[rng.integers(len(texts))]
        if intent not in family.paraphrases:
            raise ValueError(f"{intent!r} is not a paraphrase of {family.family_id!r}")
        note = NOTES[rng.integers(len(NOTES))] if rng.random() < note_rate else ""
        hinted = kind != ALWAYS.kind and rng.random() < hint_rate
        deadline = issued_tick + int(rng.integers(40, 601)) if rng.random() < DEADLINE_RATE else None
        goals.append(
            GoalEntry(
                skill_id=skill_id,
                priority=float(priorities[slot]),
                constraints=(kind,) if hinted else (),
                horizon_seconds=float(rng.integers(2, 31)),
                deadline_tick=deadline,
                condition=Condition(kind, argument) if hinted else None,
                intent=intent,
                when=when,
                note=note,
            )
        )
        meanings.append(SlotMeaning(family.family_id, skill_id, Predicate(kind, argument)))
    return ExecutiveSection(version=version, issued_tick=issued_tick, goals=tuple(goals)), tuple(meanings)


def teacher(
    section: ExecutiveSection,
    meanings: Sequence[SlotMeaning],
    features: Sequence[float],
    known: Sequence[bool],
    current_skill: str,
    *,
    valid: Sequence[bool] | None = None,
) -> tuple[str, bool]:
    """Return (chosen skill id, replace) by (M13) on one belief row in physical units.

    Args:
        section: The executive section; slot k is ``section.goals[k]``.
        meanings: One SlotMeaning per goal, aligned with the slots.
        features, known: One belief row, BELIEF_FEATURE_COUNT values each (ticks in ticks, not normalized).
        current_skill: The skill in force; replace is True when the chosen skill differs from it.
        valid: One validity flag per goal; all True when None.

    Raises:
        ValueError: If ``meanings`` or ``valid`` is not one per goal, a meaning names another skill than its
            goal, or the belief row has the wrong length.
    """

    goals = section.goals
    flags = [True] * len(goals) if valid is None else [bool(flag) for flag in valid]
    if len(meanings) != len(goals) or len(flags) != len(goals):
        raise ValueError(f"need one meaning and one valid flag per goal ({len(goals)} goals)")
    for slot, (goal, meaning) in enumerate(zip(goals, meanings)):
        if goal.skill_id != meaning.skill_id:
            raise ValueError(f"slot {slot}: the goal names {goal.skill_id!r}, its meaning {meaning.skill_id!r}")
    conditional = [
        slot
        for slot, meaning in enumerate(meanings)
        if flags[slot] and meaning.predicate.kind != ALWAYS.kind and meaning.predicate.evaluate(features, known)
    ]
    pool = conditional or [slot for slot, meaning in enumerate(meanings) if flags[slot] and meaning.predicate == ALWAYS]
    chosen = goals[max(pool, key=lambda slot: (goals[slot].priority, -slot))].skill_id if pool else SKILL_IDLE
    return chosen, chosen != current_skill


def main(argv: Sequence[str] | None = None) -> int:
    """Print sampled states as JSON lines: {"section", "meanings", "catalogue"} per state."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.executive.goal_language",
        description="Dump goal-language states (docs/SYSTEM_ONE.md section 8.11) as JSON lines.",
    )
    parser.add_argument("--n", type=int, default=5, help="states to print")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--catalogue", default="real", help="real, or synthetic:K (a fresh K-skill catalogue per state)")
    parser.add_argument(
        "--heldout-wordings", type=float, default=0.25,
        help="hold out this fraction of every family's wordings and write with the rest; 0 writes with all",
    )
    args = parser.parse_args(argv)
    kind, _, size = args.catalogue.partition(":")
    if not (args.catalogue == "real" or (kind == "synthetic" and size.isdigit())):
        parser.error(f"--catalogue must be real or synthetic:K, got {args.catalogue!r}")
    rng = np.random.default_rng(args.seed)
    wordings = split_wordings(rng, args.heldout_wordings)[0] if args.heldout_wordings > 0 else all_wordings()
    for _ in range(args.n):
        catalogue = REAL_CATALOGUE if args.catalogue == "real" else synthetic_catalogue(rng, int(size))
        section, meanings = sample_state(rng, catalogue, wordings)
        record = {"section": section.to_dict(), "meanings": [dataclasses.asdict(item) for item in meanings]}
        print(json.dumps({**record, "catalogue": catalogue}))
    return 0


__all__ = [
    "ALWAYS",
    "ALWAYS_SHARE",
    "CONDITION_FAMILIES",
    "Catalogue",
    "ConditionFamily",
    "DAMAGE_WINDOW_TICKS",
    "DEADLINE_RATE",
    "EMPTY_ALWAYS_RATE",
    "INTENT_FAMILIES",
    "IntentFamily",
    "NOTES",
    "Predicate",
    "REAL_CATALOGUE",
    "STOP_WORDS",
    "SlotMeaning",
    "VISIBLE_THRESHOLD",
    "Wordings",
    "all_wordings",
    "main",
    "render_condition",
    "resolve",
    "sample_state",
    "split_families",
    "split_wordings",
    "synthetic_catalogue",
    "teacher",
]

if __name__ == "__main__":
    raise SystemExit(main())
