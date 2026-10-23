"""The variable registry (``docs/VIRTUAL_HUMAN.md`` sections 10.1 and 14): every variable the agent tracks on its own
side, declared once.

What a variable is (owner, 2026-10-21): "Variables are states that are not given in game but we track locally to
modify some output or just to keep track of". A value the game gives (health, stamina, position, what is in view) is
not a variable: it reaches every layer as a token column of the senses (``tokens_v3.py``). A variable is the agent's
own: its traits and emotions now, and its drives, sleep pressure, levels, places and relationships as their steps come
(section 13).

Each variable is one scalar per agent. The state layer holds the variables of B agents as one array float32[B, V] in
:data:`NAMES` order (``state/registry.py``). The body reads those marked ``body_reads``: the ``persona`` token's
columns are the body-read variables of the self section, and the ``affect`` token's those of the inner section (rule 3
of section 14: the registry generates the layout).

The list grows and shrinks by name (owner, 2026-10-21, on the trait list: "ensure that we can expand it or compress it
later"). :func:`transfer` gives the maps that migrate a saved state, a stored row or a trained trunk from one list of
names to another (section 14).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

SECTIONS: tuple[str, ...] = ("body", "inner", "world", "self", "social")
"""``docs/VIRTUAL_HUMAN.md`` section 3. ``self`` is who the agent is, ``inner`` how it feels."""
SOURCES: tuple[str, ...] = ("birth", "rule", "written")
"""Set once when the agent is made (the LLM's persona record); computed by a fixed rule of the state layer; written by
the decision layer or the LLM (step 2)."""
WRITERS: tuple[str, ...] = ("decision", "llm")
"""The layers that write (section 3: the top two), indexed in the state by position."""


@dataclass(frozen=True)
class Variable:
    name: str
    section: str
    group: str
    source: str
    body_reads: bool
    low: float = -1.0
    high: float = 1.0
    default: float = 0.0
    writer: str | None = None  # the one layer that writes it; None for ``birth`` and ``rule``


def _group(section: str, group: str, source: str, names: Sequence[str], *, body: Sequence[str] = (),
           low: float = -1.0, high: float = 1.0) -> tuple[Variable, ...]:
    return tuple(Variable(name, section, group, source, name in body, low, high) for name in names)


_MOTOR = ("reaction_mean", "reaction_tail", "motor_noise", "smoothness", "vigilance", "lateral_bias")

VARIABLES: tuple[Variable, ...] = (
    # The traits (section 15.6, approved 2026-10-21): each in [-1, 1], 0 the population's mean. The body reads those
    # with a mechanism in it (section 15.1): the reward weights (patience, risk taking, effort sensitivity), the
    # reflex gain (negative emotionality) and the motor signature.
    *_group("self", "economic preferences", "birth",
            ("patience", "risk_taking", "positive_reciprocity", "negative_reciprocity", "altruism", "trust"),
            body=("patience", "risk_taking")),
    *_group("self", "motivation", "birth",
            ("effort_sensitivity", "impulsiveness", "self_control", "distractibility", "self_efficacy",
             "boredom_proneness"), body=("effort_sensitivity",)),
    *_group("self", "emotional reactivity", "birth", ("negative_emotionality", "positive_emotionality"),
            body=("negative_emotionality",)),
    *_group("self", "values", "birth",
            ("openness_to_change", "conservation", "self_enhancement", "self_transcendence")),
    # The motor signature, with vigilance, and the strafe-side preference as its bias (15.6: "a motor-signature bias,
    # not a trait").
    *_group("self", "motor signature", "birth", _MOTOR, body=_MOTOR),
    # The emotional state (``state/affect.py``, a proposal; step 7 trims it).
    *_group("inner", "emotion", "rule", ("fear", "anger", "startle"), body=("fear", "anger", "startle"), low=0.0),
    *_group("inner", "emotion", "rule", ("valence",), body=("valence",)),
    Variable("coping", "inner", "emotion", "rule", True, 0.0, 1.0, 0.5),
    *_group("inner", "mood", "rule", ("mood_pleasure", "mood_arousal", "mood_dominance"),
            body=("mood_pleasure", "mood_arousal", "mood_dominance")),
)
NAMES: tuple[str, ...] = tuple(variable.name for variable in VARIABLES)
BY_NAME: Mapping[str, Variable] = {variable.name: variable for variable in VARIABLES}
DEFAULTS: np.ndarray = np.asarray([variable.default for variable in VARIABLES], np.float32)

ENTITY_ENTRIES: tuple[Variable, ...] = (
    Variable("remember", "world", "tags", "written", False, 0.0, 1.0, 0.0, writer="decision"),
)
PLACE_ENTRIES: tuple[Variable, ...] = (
    Variable("remember", "world", "tags", "written", False, 0.0, 1.0, 0.0, writer="decision"),
)
"""The fields of the entries about an entity and about a place (``docs/VIRTUAL_HUMAN.md`` section 19). ``remember`` is
the priority the decision layer marks a thing with, which the stores pin and the body reads as ``tagged``; later steps
append fields by name."""

assert len(BY_NAME) == len(VARIABLES), "variable names must be unique"
assert all(v.section in SECTIONS and v.source in SOURCES and v.low <= v.default <= v.high for v in VARIABLES)
assert all(v.writer is None for v in VARIABLES if v.source in ("birth", "rule"))
assert all(v.writer in WRITERS for v in VARIABLES if v.source == "written")
for _entries in (ENTITY_ENTRIES, PLACE_ENTRIES):
    assert len({v.name for v in _entries}) == len(_entries), "entry field names must be unique"
    assert all(v.source == "written" and v.writer in WRITERS and v.low <= v.default <= v.high for v in _entries)


def names(section: str | None = None, *, body_reads: bool | None = None) -> tuple[str, ...]:
    """The variables of ``section`` (every section when None), only those the body reads (or does not) when
    ``body_reads`` is given, in :data:`NAMES` order."""

    return tuple(v.name for v in VARIABLES
                 if (section is None or v.section == section) and (body_reads is None or v.body_reads == body_reads))


def fingerprint(value: Any) -> str:
    """The first 16 hex digits of the SHA-256 of ``value`` as JSON (a list of names, or {token type: feature names}
    in sequence order), recorded with every saved state and trunk (section 14, rule 2)."""

    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:16]


def transfer(old: Sequence[str], new: Sequence[str], maps: Mapping[str, Mapping[str, float]] | None = None,
             dropped: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
    """The maps float64[len(new), len(old)] that carry columns named ``old`` to columns named ``new``.

    A new name takes the old column of the same name, or the sources ``maps[name]`` gives it ({old name: weight}), or
    nothing (a new column). An old name that no new column takes must be in ``dropped``.

    Returns:
        ``forward``: new values from old ones, ``new = forward @ old`` (a merged column is ``sum_i w_i old_i``).
        ``backward``: the rows of a linear layer that read the old columns, moved to the new ones,
        ``kernel_new = backward @ kernel_old``. A kept column keeps its row and a new one gets zeros, so the layer's
        output is unchanged; a merged column gets the sum of its sources' rows over ``sum_i w_i``, which is exact
        where its sources were equal; a dropped column's row goes, which is exact where it was 0.

    Raises:
        ValueError: If an old name disappears without a map or a drop, is taken twice, or is named in a map or
            ``dropped`` without being old; or if a map names no new column or has a zero weight or weight sum.
    """

    maps = dict(maps or {})
    position = {name: index for index, name in enumerate(old)}
    forward = np.zeros((len(new), len(old)))
    backward = np.zeros((len(new), len(old)))
    taken: set[str] = set()
    for row, name in enumerate(new):
        sources = maps.get(name, {name: 1.0} if name in position else {})
        total = sum(sources.values())
        if sources and (total == 0 or not all(sources.values())):
            raise ValueError(f"the map of {name!r} needs non-zero weights with a non-zero sum, got {dict(sources)}")
        for source, weight in sources.items():
            if source not in position:
                raise ValueError(f"{name!r} is mapped from {source!r}, which the old list does not have")
            if source in taken:
                raise ValueError(f"{source!r} is taken by two new columns")
            taken.add(source)
            forward[row, position[source]] = weight
            backward[row, position[source]] = 1.0 / total
    if set(maps) - set(new):
        raise ValueError(f"maps name columns the new list does not have: {sorted(set(maps) - set(new))}")
    if set(dropped) - set(position) or set(dropped) & taken:
        raise ValueError(f"dropped must name old columns no new column takes, got {sorted(dropped)}")
    gone = [name for name in old if name not in taken and name not in dropped]
    if gone:
        raise ValueError(f"{gone} are not in the new list: merge each into a new column with a map, or drop it by name")
    return forward, backward


def remap(values: Any, old: Sequence[str], new: Sequence[str], maps: Mapping[str, Mapping[str, float]] | None = None,
          dropped: Sequence[str] = (), fill: Any = 0.0) -> np.ndarray:
    """float32[..., len(new)]: ``values`` [..., len(old)] carried to the new columns by :func:`transfer`'s forward
    map; a new column with no source takes ``fill`` (a scalar, or one value per new column)."""

    forward, _ = transfer(old, new, maps, dropped)
    moved = np.asarray(values, np.float64) @ forward.T
    return np.where(forward.any(axis=1), moved, fill).astype(np.float32)


__all__ = ["BY_NAME", "DEFAULTS", "ENTITY_ENTRIES", "NAMES", "PLACE_ENTRIES", "SECTIONS", "SOURCES", "VARIABLES",
           "Variable", "WRITERS", "fingerprint", "names", "remap", "transfer"]
