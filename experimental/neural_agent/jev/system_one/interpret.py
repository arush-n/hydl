"""Interpret goals once per executive version, evaluate them exactly per decision (SYSTEM_ONE.md 8.13).

The end-to-end reader of FORMULAS (M12) failed G3-NL: a pooled text embedding neither carried an unseen
paraphrase to its predicate nor kept a number exact. Reading is split in two, as section 8.13 proposes:

* **Interpret** (slow path, once per executive version, off the tick). Each goal's ``when`` text becomes a
  predicate kind of section 8.11 plus a numeric argument; its ``intent`` text becomes the catalogue skill
  whose description it matches (the catalogue-agnostic matching of (M3b), on texts). The kind comes from a
  frozen text encoder and a classifier (:class:`ConditionReader`), the argument from a deterministic number
  reader (:func:`read_number`), because numbers must be exact; the skill from :class:`IntentMatcher`.
* **Evaluate** (fast path, every decision). The interpreted predicates on one belief row, exactly
  (``goal_language.Predicate.evaluate``), then the priority rule (M13) (:func:`decide`).

Number reader (no learning). Tokens are runs of digits (one decimal point allowed), runs of letters (with
apostrophes) and ``%``; the words of one number, and a unit, follow each other with only spaces or hyphens::

    digits             "60", "7", "0.3", "0.75"
    words              zero..nineteen; twenty..ninety [one..nine] ("seventy-five", "seventy five");
                       (a | one..nine) hundred [[and] 1..99] ("one hundred and twenty", "a hundred")
    fraction words     "half"; (a | an | one..nineteen) (half | quarter[s] | fifth[s]):
                       "a quarter" 0.25, "three quarters" 0.75, "two fifths" 0.4            unit fraction
    unit after it      %, percent, per cent       value / 100                              unit fraction
                       HP, health                 value, or value / 100 above 1            unit fraction
                                                  (a hint: the word stays outside the span)
                       tick[s], game tick[s]                                               unit ticks
                       block[s], m, metre[s], meter[s]                                     unit blocks
                       none                                                                unit None
    not a number       a whole number between "the" and "of": "the two of you" (a group, not a measure)

``read_number`` returns the first number of a text; ``ambiguous`` is True when a second one follows (one
number per condition is the contract). ``argument_for(kind, number)`` is the predicate's argument: the value,
except that a fraction kind reads a unitless value above 1 as percent (30 -> 0.3).

Condition reader. With f the frozen encoder, over the distinct non-empty training (when, kind) pairs::

    x = (f(when) - mu) / sigma                   mu, sigma: per-column mean and deviation of the training x
    p = softmax(W x + b)                         over KINDS, the nine kinds of CONDITION_FAMILIES (always first)
    W, b = argmin  mean_i CE(p_i, kind_i) + (CONDITION_L2 / 2) ||W||^2     Nesterov descent from zero
    allowed = every kind                         if read_number(when) is None
            = the numeric kinds of its unit      otherwise (any numeric kind when the unit is None)
    kind = argmax_{allowed} p;  argument = argument_for(kind, read_number(when))
    confidence = p[kind], times MISSING_ARGUMENT_PENALTY when the kind is numeric and no number was read
    when.strip() == ""  ->  always, no argument, confidence 1

The unit rule is a deviation from "kind = the classifier's argmax", chosen on the training wordings only (see
``jev/system_one/SKILL.md``, 2026-10-17): a measure in blocks names a distance predicate whatever the words.

Intent matcher over a catalogue skill_id -> description, with u(t) = f(t) / ||f(t)||::

    zero-shot   s_j = u(intent) . u(desc_j)
    fitted      s_j = cos(u(intent) (I + Delta), u(desc_j)),  Delta = X^T (X X^T + MATCH_L2 I)^-1 (Y - X)
                the ridge least-squares map of intent embeddings onto description embeddings, shrunk toward the
                identity (zero-shot) map; X, Y = the u of the distinct training intents and of their skills'
                descriptions. A small ridge: it reproduces its training pairs
    skill = argmax_j s_j                         always a skill: no refusal (SYSTEM_ONE.md 8.12, item 4)

An empty intent keeps the goal's typed ``skill_id``. Texts are encoded once per distinct string and cached
per reader, ENCODE_BATCH per encoder call, shortest first.

Decision (M13) on interpretations, slot k with priority p_k (every slot valid)::

    S_cond = {k : kind_k != always and predicate_k(B_t)};  chosen = skill of argmax_{S_cond} p_k if non-empty,
    else of argmax_{k : kind_k = always} p_k, else idle;  ties to the lower slot;  replace = chosen != current

A numeric kind without an argument cannot be evaluated and is False, as an unknown feature is. Belief rows are
in physical units: the two tick columns times TICKS_SATURATION first, as ``goal_language`` requires.
Data-side code: numpy and the standard library; the encoder is any ``trunk.TrunkEncoder``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from experimental.neural_agent.contracts import SKILL_IDLE
from experimental.neural_agent.executive.goal_language import ALWAYS, CONDITION_FAMILIES, Predicate
from experimental.neural_agent.jev.system_one.trunk import TrunkEncoder
from experimental.neural_agent.shared_state import ExecutiveSection, GoalEntry

KINDS: tuple[str, ...] = tuple(family.family_id for family in CONDITION_FAMILIES)
"""The nine predicate kinds of section 8.11, in CONDITION_FAMILIES order (always first)."""
NUMERIC_UNITS: dict[str, str] = {
    "target_lost_for": "ticks",
    "distance_below": "blocks",
    "distance_above": "blocks",
    "self_health_below": "fraction",
    "target_health_below": "fraction",
}
"""The unit of each numeric kind's argument (``goal_language`` module docstring, Predicates)."""
CONDITION_L2: float = 1e-2
"""L2 weight of the condition classifier."""
MATCH_L2: float = 0.1
"""Ridge weight of the fitted intent map."""
MISSING_ARGUMENT_PENALTY: float = 0.5
"""Factor on a numeric kind's confidence when no number was read."""
ENCODE_BATCH: int = 64
"""Texts per encoder call."""
LOGISTIC_STEPS: int = 3000
"""Most Nesterov steps of the classifier fit (it stops once the gradient norm is below 1e-6)."""

_TOKEN = re.compile(r"\d+(?:\.\d+)?|%|[^\W\d_]+(?:['’][^\W\d_]+)*")
_JOIN = re.compile(r"[ \t\-]*")
_SMALL: dict[str, int] = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen".split()
    )
}
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_DENOMINATORS: dict[str, int] = {"half": 2, "halves": 2, "quarter": 4, "quarters": 4, "fifth": 5, "fifths": 5}
_UNIT_WORDS: dict[str, str] = {
    "%": "percent", "percent": "percent", "hp": "health", "health": "health", "tick": "ticks", "ticks": "ticks",
    "block": "blocks", "blocks": "blocks", "m": "blocks", "metre": "blocks", "metres": "blocks", "meter": "blocks",
    "meters": "blocks",
}


@dataclass(frozen=True)
class ReadNumber:
    """The first number of a condition text.

    Attributes:
        value: The number; a percentage or a fraction word as a fraction (30 % -> 0.3, "a quarter" -> 0.25).
        unit: ``"ticks"``, ``"blocks"``, ``"fraction"`` or None (no unit written).
        span: Character offsets [start, end) of the number and its unit in the text.
        ambiguous: True when the text holds a second number (the first is returned).
    """

    value: float
    unit: str | None
    span: tuple[int, int]
    ambiguous: bool = False


class _Tokens:
    """Lower-cased tokens of one text with their offsets."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.items = [(match.group().lower(), match.start(), match.end()) for match in _TOKEN.finditer(text)]

    def word(self, index: int) -> str:
        return self.items[index][0]

    def joined(self, index: int) -> str | None:
        """Token ``index`` when only spaces or hyphens separate it from token ``index - 1``, else None."""

        if not 0 < index < len(self.items):
            return None
        gap = self.text[self.items[index - 1][2] : self.items[index][1]]
        return self.items[index][0] if _JOIN.fullmatch(gap) else None


def _below_hundred(tokens: _Tokens, index: int, word: str | None) -> tuple[int, int] | None:
    """(value 0..99, next index) of the number words starting at token ``index`` (which is ``word``), else None."""

    if word in _TENS:
        value, end = _TENS[word], index + 1
        ones = tokens.joined(end)
        if ones in _SMALL and 1 <= _SMALL[ones] <= 9:
            value, end = value + _SMALL[ones], end + 1
        return value, end
    if word in _SMALL:
        return _SMALL[word], index + 1
    return None


def _cardinal(tokens: _Tokens, index: int) -> tuple[float, int, bool] | None:
    """(value, next index, whole) of the cardinal number starting at token ``index``, else None."""

    word = tokens.word(index)
    if word[0].isdigit():
        return float(word), index + 1, "." not in word
    if word in ("a", "an") and tokens.joined(index + 1) == "hundred":
        value, end = 1, index + 1
    else:
        parsed = _below_hundred(tokens, index, word)
        if parsed is None:
            return None
        value, end = parsed
    if 1 <= value <= 9 and tokens.joined(end) == "hundred":
        value, end = value * 100, end + 1
        start = end + 1 if tokens.joined(end) == "and" else end
        rest = _below_hundred(tokens, start, tokens.joined(start))
        if rest is not None and rest[0] > 0:
            value, end = value + rest[0], rest[1]
    return float(value), end, True


def _number_at(tokens: _Tokens, index: int) -> tuple[float, str | None, int] | None:
    """(value, unit, end index of the span) of the number expression starting at token ``index``, else None."""

    word = tokens.word(index)
    count = 1 if word in ("a", "an") else _SMALL.get(word)
    denominator = tokens.joined(index + 1)
    if count and denominator in _DENOMINATORS:
        return count / _DENOMINATORS[denominator], "fraction", index + 2
    if word == "half":
        return 0.5, "fraction", index + 1
    cardinal = _cardinal(tokens, index)
    if cardinal is None:
        return None
    value, end, whole = cardinal
    if whole and tokens.joined(index) is not None and tokens.word(index - 1) == "the" and tokens.joined(end) == "of":
        return None  # "the two of you": a group, not a measure
    after = tokens.joined(end)
    if after == "game" and tokens.joined(end + 1) in ("tick", "ticks"):
        return value, "ticks", end + 2
    if after == "per" and tokens.joined(end + 1) == "cent":
        return value / 100.0, "fraction", end + 2
    unit = _UNIT_WORDS.get(after)
    if unit == "percent":
        return value / 100.0, "fraction", end + 1
    if unit == "health":
        return (value / 100.0 if value > 1.0 else value), "fraction", end
    if unit is not None:
        return value, unit, end + 1
    return value, None, end


def _numbers(text: str) -> list[tuple[float, str | None, tuple[int, int]]]:
    """Every number expression of ``text``, left to right: (value, unit, span)."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    tokens, found, index = _Tokens(text), [], 0
    while index < len(tokens.items):
        parsed = _number_at(tokens, index)
        if parsed is None:
            index += 1
            continue
        value, unit, end = parsed
        found.append((value, unit, (tokens.items[index][1], tokens.items[end - 1][2])))
        index = end
    return found


def read_number(text: str) -> ReadNumber | None:
    """Return the first number of ``text`` with its unit and span (module docstring), or None without one.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    found = _numbers(text)
    if not found:
        return None
    value, unit, span = found[0]
    return ReadNumber(value, unit, span, ambiguous=len(found) > 1)


def argument_for(kind: str, number: ReadNumber | None) -> float | None:
    """Return the argument of predicate ``kind`` read from ``number``: None for a non-numeric kind or no number.

    Raises:
        ValueError: If ``kind`` is not one of KINDS.
    """

    if kind not in KINDS:
        raise ValueError(f"unknown condition kind {kind!r}; expected one of {KINDS}")
    if kind not in NUMERIC_UNITS or number is None:
        return None
    if NUMERIC_UNITS[kind] == "fraction" and number.unit is None and number.value > 1.0:
        return number.value / 100.0
    return number.value


def _embed(encoder: TrunkEncoder, texts: Sequence[str], cache: dict[str, np.ndarray]) -> np.ndarray:
    """Return float64[len(texts), D], encoding each new distinct text once (ENCODE_BATCH per call, shortest first)."""

    new = sorted({text for text in texts if text not in cache}, key=lambda text: (len(text), text))
    for start in range(0, len(new), ENCODE_BATCH):
        chunk = new[start : start + ENCODE_BATCH]
        cache.update(zip(chunk, np.asarray(encoder.encode(chunk), dtype=np.float32)))
    if not texts:
        return np.zeros((0, int(encoder.dim)))
    return np.stack([cache[text] for text in texts]).astype(np.float64)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows divided by their L2 norm (a zero row stays zero)."""

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _fit_logistic(x: np.ndarray, labels: np.ndarray, classes: int, l2: float) -> np.ndarray:
    """Multinomial logistic regression float64[D + 1, classes] (last row the bias): Nesterov descent from zero on
    mean CE + (l2 / 2) ||W||^2 with the step 1 / L, L = ||[x 1]||_2^2 / (2 N) + l2. Deterministic."""

    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    targets = np.eye(classes)[labels]
    penalty = np.ones((design.shape[1], 1))
    penalty[-1] = 0.0
    step = 1.0 / (0.5 * np.linalg.norm(design, 2) ** 2 / design.shape[0] + l2)
    theta = previous = np.zeros((design.shape[1], classes))
    for count in range(1, LOGISTIC_STEPS + 1):
        look = theta + (count - 1.0) / (count + 2.0) * (theta - previous)
        gradient = design.T @ (_softmax(design @ look) - targets) / design.shape[0] + l2 * penalty * look
        previous, theta = theta, look - step * gradient
        if np.linalg.norm(gradient) < 1e-6:
            break
    return theta


def _allowed(number: ReadNumber | None) -> np.ndarray:
    """bool[9]: every kind without a number, else the numeric kinds of its unit (all of them for no unit)."""

    if number is None:
        return np.ones(len(KINDS), dtype=bool)
    return np.asarray([NUMERIC_UNITS.get(kind) in ((number.unit,) if number.unit else NUMERIC_UNITS.values()) for kind in KINDS])


@dataclass(frozen=True)
class ConditionReading:
    """How one ``when`` text was read: predicate kind, argument (None unless numeric and read), confidence."""

    kind: str
    argument: float | None
    confidence: float


class ConditionReader:
    """Read ``when`` texts into predicate kinds and arguments (module docstring, Condition reader).

    Args:
        encoder: A frozen text encoder (``trunk.TrunkEncoder``).
    """

    def __init__(self, encoder: TrunkEncoder) -> None:
        self.encoder = encoder
        self._cache: dict[str, np.ndarray] = {}
        self._theta: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, texts: Sequence[str], kinds: Sequence[str]) -> ConditionReader:
        """Fit the classifier on the distinct non-empty (text, kind) pairs; return ``self``.

        Raises:
            ValueError: If the lengths differ, a kind is unknown, or no text is non-empty.
        """

        if len(texts) != len(kinds):
            raise ValueError(f"need one kind per text, got {len(texts)} texts and {len(kinds)} kinds")
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown condition kinds {sorted(unknown)}; expected some of {KINDS}")
        pairs = sorted({(text, kind) for text, kind in zip(texts, kinds) if text.strip()})
        if not pairs:
            raise ValueError("fit needs at least one non-empty condition text")
        x = _embed(self.encoder, [text for text, _ in pairs], self._cache)
        self._mean, self._scale = x.mean(axis=0), x.std(axis=0) + 1e-6
        labels = np.asarray([KINDS.index(kind) for _, kind in pairs])
        self._theta = _fit_logistic((x - self._mean) / self._scale, labels, len(KINDS), CONDITION_L2)
        return self

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Return the classifier's p over KINDS, float64[len(texts), 9]; an empty text is one-hot on always.

        Raises:
            RuntimeError: If the reader is not fitted.
        """

        if self._theta is None:
            raise RuntimeError("ConditionReader.fit must run before reading")
        out = np.zeros((len(texts), len(KINDS)))
        out[:, KINDS.index(ALWAYS.kind)] = 1.0
        rows = [row for row, text in enumerate(texts) if text.strip()]
        if rows:
            x = (_embed(self.encoder, [texts[row] for row in rows], self._cache) - self._mean) / self._scale
            out[rows] = _softmax(x @ self._theta[:-1] + self._theta[-1])
        return out

    def read(self, texts: Sequence[str]) -> list[ConditionReading]:
        """Return one ConditionReading per text: the unit rule over p, the argument, the confidence.

        Raises:
            RuntimeError: If the reader is not fitted.
        """

        readings = []
        for text, p in zip(texts, self.probabilities(texts)):
            number = read_number(text) if text.strip() else None
            kind = KINDS[int(np.argmax(np.where(_allowed(number), p, -1.0)))]
            argument = argument_for(kind, number)
            penalty = MISSING_ARGUMENT_PENALTY if kind in NUMERIC_UNITS and argument is None else 1.0
            readings.append(ConditionReading(kind, argument, float(p[KINDS.index(kind)]) * penalty))
        return readings


@dataclass(frozen=True)
class IntentReading:
    """The skill an intent matched and the score of every catalogue skill, in catalogue order."""

    skill_id: str
    scores: tuple[float, ...]


class IntentMatcher:
    """Match ``intent`` texts to catalogue skills by their descriptions (module docstring, Intent matcher).

    Args:
        encoder: A frozen text encoder (``trunk.TrunkEncoder``).
    """

    def __init__(self, encoder: TrunkEncoder) -> None:
        self.encoder = encoder
        self._cache: dict[str, np.ndarray] = {}
        self._delta: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has run."""

        return self._delta is not None

    def fit(self, intents: Sequence[str], descriptions: Sequence[str]) -> IntentMatcher:
        """Fit Delta on the distinct non-empty (intent, description of its skill) pairs; return ``self``.

        Raises:
            ValueError: If the lengths differ or there is no non-empty intent.
        """

        if len(intents) != len(descriptions):
            raise ValueError(f"need one description per intent, got {len(intents)} and {len(descriptions)}")
        pairs = sorted({(intent, text) for intent, text in zip(intents, descriptions) if intent.strip()})
        if not pairs:
            raise ValueError("fit needs at least one non-empty intent")
        x = _unit_rows(_embed(self.encoder, [intent for intent, _ in pairs], self._cache))
        y = _unit_rows(_embed(self.encoder, [text for _, text in pairs], self._cache))
        self._delta = x.T @ np.linalg.solve(x @ x.T + MATCH_L2 * np.eye(len(pairs)), y - x)
        return self

    def scores(self, intents: Sequence[str], descriptions: Sequence[str], *, zero_shot: bool = False) -> np.ndarray:
        """Return s_j per intent, float64[len(intents), len(descriptions)] (fitted unless ``zero_shot``).

        Raises:
            RuntimeError: If fitted scores are asked for before :meth:`fit`.
        """

        if not zero_shot and self._delta is None:
            raise RuntimeError("IntentMatcher.fit must run before fitted matching; pass zero_shot=True")
        x = _unit_rows(_embed(self.encoder, list(intents), self._cache))
        table = _unit_rows(_embed(self.encoder, list(descriptions), self._cache))
        return (x if zero_shot else _unit_rows(x + x @ self._delta)) @ table.T

    def match(self, intents: Sequence[str], catalogue: Mapping[str, str], *, zero_shot: bool = False) -> list[IntentReading]:
        """Return one IntentReading per intent: the best catalogue skill, never a refusal (section 8.12, item 4).

        Raises:
            ValueError: If the catalogue is empty.
            RuntimeError: As :meth:`scores`.
        """

        if not catalogue:
            raise ValueError("the catalogue is empty")
        ids = list(catalogue)
        table = self.scores(intents, [catalogue[skill_id] for skill_id in ids], zero_shot=zero_shot)
        return [IntentReading(ids[int(np.argmax(row))], tuple(float(value) for value in row)) for row in table]


@dataclass(frozen=True)
class GoalInterpretation:
    """One goal read into the vocabulary the decision layer evaluates (section 8.13)."""

    kind: str
    argument: float | None
    skill_id: str
    skill_scores: tuple[float, ...]
    condition_confidence: float


def interpret(
    entry: GoalEntry,
    catalogue: Mapping[str, str],
    reader: ConditionReader,
    matcher: IntentMatcher,
    *,
    zero_shot: bool = False,
) -> GoalInterpretation:
    """Read one goal: ``when`` by ``reader``, ``intent`` by ``matcher`` over ``catalogue``. An empty intent keeps
    the typed ``skill_id`` (scored 1, the others 0)."""

    condition = reader.read([entry.when])[0]
    if entry.intent.strip():
        intent = matcher.match([entry.intent], catalogue, zero_shot=zero_shot)[0]
    else:
        intent = IntentReading(entry.skill_id, tuple(float(skill_id == entry.skill_id) for skill_id in catalogue))
    return GoalInterpretation(condition.kind, condition.argument, intent.skill_id, intent.scores, condition.confidence)


def evaluate(interpretation: GoalInterpretation, features: Sequence[float], known: Sequence[bool]) -> bool:
    """Return the interpreted predicate on one belief row in physical units (``Predicate.evaluate``); a numeric
    kind without an argument is False."""

    if interpretation.kind in NUMERIC_UNITS and interpretation.argument is None:
        return False
    return Predicate(interpretation.kind, interpretation.argument).evaluate(features, known)


def decide(
    section: ExecutiveSection,
    interpretations: Sequence[GoalInterpretation],
    features: Sequence[float],
    known: Sequence[bool],
    current_skill: str,
) -> tuple[str, bool]:
    """Return (skill id, replace) by (M13) with the interpreted predicates and skills (module docstring).

    Raises:
        ValueError: If there is not one interpretation per goal.
    """

    goals = section.goals
    if len(interpretations) != len(goals):
        raise ValueError(f"need one interpretation per goal ({len(goals)} goals, {len(interpretations)} given)")
    conditional = [
        slot for slot, item in enumerate(interpretations) if item.kind != ALWAYS.kind and evaluate(item, features, known)
    ]
    pool = conditional or [slot for slot, item in enumerate(interpretations) if item.kind == ALWAYS.kind]
    chosen = interpretations[max(pool, key=lambda slot: (goals[slot].priority, -slot))].skill_id if pool else SKILL_IDLE
    return chosen, chosen != current_skill


__all__ = [
    "CONDITION_L2",
    "ConditionReader",
    "ConditionReading",
    "ENCODE_BATCH",
    "GoalInterpretation",
    "IntentMatcher",
    "IntentReading",
    "KINDS",
    "LOGISTIC_STEPS",
    "MATCH_L2",
    "MISSING_ARGUMENT_PENALTY",
    "NUMERIC_UNITS",
    "ReadNumber",
    "argument_for",
    "decide",
    "evaluate",
    "interpret",
    "read_number",
]
