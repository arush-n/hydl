"""The gen-7 surface: the token sets a body reads and the action heads it emits, in six scopes.

``docs/BUILD.md`` section 3.1 and ``docs/AGENT_BRIEF.md`` deliverable A1 (2026-10-17). This module is the one
layout that the simulator work and the bodies build against. It has no learned state and imports
only the gym and ``contracts``.

Two head layouts
----------------
``"gen5"`` is today's surface: the twelve heads that the gym publishes, with their names, sizes and
order exactly as ``hytalegym.jax.combat.observation.v3.policy`` gives them (124 choices).
``"gen7"`` is the PLANNED surface for Hytale 0.7.2: the same twelve heads first and unchanged, then
nine new heads. In gen 7 the locomotion head's four dodge choices (33 to 36, forward, back, left,
right) are disabled, because 0.7.2 dodges only left or right (``Server/Item/Interactions/Dodge.json``)
and the new ``dodge`` head carries that.

Six scopes
----------
A scope is a group of heads that one granted request owns at a time; the combiner grants one
request per scope. Every head belongs to exactly one scope::

    locomotion       jump, locomotion (gait x compass; the gen-5 dodges), body yaw; gen 7: mount
    gaze             camera yaw, pitch
    hands            base action (melee), hotbar, use, trigger, block candidate;
                     gen 7: duration, item_move, item_drop_give
    posture_defence  guard; gen 7: dodge
    interaction      gen 7: door, recipe
    abilities_magic  ability slot; gen 7: rune_set, cast

Two placements differ from the scope list in ``docs/BUILD.md`` 3.1, with the reason. ``use`` is in
hands, not interaction, because it acts on the block candidate that the hands scope points at, and a
verb owns one scope. Body yaw is in locomotion, so walking steers the body while gaze turns the
camera; E7 (walk to a point while looking at an entity) tests the split.

Holds
-----
A hold is a number of ticks, not a press and release per tick (``docs/BUILD.md`` 3.1). The tick is per
layout (``tick_seconds``): gen 7 runs at the game's 30 Hz, 1/30 s (owner, 2026-10-17); 0.045 s is the
gen-5 simulator's. At 30 Hz a hold of 0.1 to 3 s is 3 to 90 ticks, which ``DURATION_BUCKET_TICKS``
spans: it gives the choices of the ``duration`` head (and of ``cast``, for charged casts); bucket 0
means no hold, and a longer action (eating, 4 s) chains two holds. The buckets place at least one
choice inside every shortbow draw tier (0.1 / 0.3 / 0.6 / 0.9 / 1.2 s), which E8 needs:

    ticks(t) = ceil(t / tick_seconds(layout)); tier k holds the buckets in [ticks(t_k), ticks(t_{k+1}))

Tokens
------
Positions and velocities in tokens are in the agent's frame (forward, right, up), the frame the gym
encodes actor evidence in (``jax/combat/runtime/reset.py``): with world axes (x, y, z), y up and yaw
in radians, ``forward = (-sin yaw, 0, -cos yaw)``, ``right = (cos yaw, 0, -sin yaw)``, ``up = (0, 1,
0)``, and a world offset ``d`` has agent coordinates ``(d . forward, d . right, d_y)``.

Each token type is a frozen record with validated fields. ``features()`` returns its numeric
features in the order of ``FEATURES``; identifiers (entity handles, block kinds, item ids, cells)
are fields, not features, because they enter a network through an embedding or the resolver.
Every token has ``marked``: the instructed target is marked in its set (requirement 2 of
``SKILL_LIBRARY.md`` section 7).
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from experimental.neural_agent.contracts import TICKS_SATURATION
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)

TICK_SECONDS: float = 0.045
"""Seconds per tick of the gen-5 simulator (``loaded_dt``, measured). Gen 7 runs at 30 Hz: use ``tick_seconds``."""

TICK_SECONDS_BY_LAYOUT: Mapping[str, float] = {"gen5": TICK_SECONDS, "gen7": 1.0 / 30.0}
"""Seconds per tick by layout: gen 7 steps at the game's tick, 1/30 s (owner's 30 Hz ruling, 2026-10-17)."""

SCOPE_NAMES: tuple[str, ...] = (
    "locomotion",
    "gaze",
    "hands",
    "posture_defence",
    "interaction",
    "abilities_magic",
)
"""The six scopes, in this fixed order."""

LAYOUT_NAMES: tuple[str, ...] = ("gen5", "gen7")
"""The two head layouts."""

DURATION_BUCKET_TICKS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 11, 16, 22, 31, 44, 62, 90)
"""Hold lengths in ticks, one per choice of ``duration`` and ``cast``; choice 0 means no hold."""

SIGNED_LOOK_HEADS: frozenset[str] = frozenset({"yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"})
"""Heads whose neutral choice is the centre bin (``size // 2``); every other head neutralises at 0."""

LOCOMOTION_HEAD: str = "locomotion_gait_compass"
GEN5_DODGE_CHOICES: tuple[int, ...] = (33, 34, 35, 36)
"""Locomotion choices that dodge forward, back, left and right on the gen-5 surface."""

ENTITY_ROLES: tuple[str, ...] = ("enemy", "ally", "animal", "merchant", "player", "unknown")
FACES: tuple[str, ...] = ("none", "up", "down", "north", "south", "east", "west")
FLUIDS: tuple[str, ...] = ("none", "water", "lava", "mud", "slime", "tar")
MAX_HOLD_THRESHOLDS: int = 6
"""Hold thresholds a kit token carries; the shortbow's five draw tiers fit."""


class HeadSpec(NamedTuple):
    """One action head.

    Attributes:
        name: Head name; on the gen-5 surface, the gym's name.
        size: Number of choices.
        scope: One of ``SCOPE_NAMES``.
        neutral: The no-op choice.
        status: ``"published"`` (the gym publishes it today) or ``"planned"`` (gen 7).
        choices: Choice labels for a planned head; ``()`` for a published head (the gym owns those).
    """

    name: str
    size: int
    scope: str
    neutral: int
    status: str
    choices: tuple[str, ...] = ()


_GEN5_SCOPES: dict[str, str] = {
    "base_action": "hands",
    "ability_none_plus_slots": "abilities_magic",
    "guard_off_on": "posture_defence",
    "jump_off_on": "locomotion",
    "locomotion_gait_compass": "locomotion",
    "yaw_delta_bins": "gaze",
    "body_yaw_delta_bins": "locomotion",
    "pitch_delta_bins": "gaze",
    "hotbar_none_plus_slots": "hands",
    "use_off_on": "hands",
    "block_primary_secondary_trigger": "hands",
    "block_none_plus_candidates": "hands",
}


def neutral_choice(name: str, size: int) -> int:
    """Return a head's no-op choice: ``size // 2`` for a signed look head, else 0."""

    return size // 2 if name in SIGNED_LOOK_HEADS else 0


def _published_heads() -> tuple[HeadSpec, ...]:
    names = tuple(ARSENAL_POLICY_ACTION_HEAD_NAMES)
    sizes = tuple(int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES)
    unknown = sorted(set(names) - set(_GEN5_SCOPES))
    if unknown or len(names) != len(_GEN5_SCOPES):
        raise RuntimeError(
            f"the gym publishes heads {names}; contract/surface.py assigns scopes to {tuple(_GEN5_SCOPES)}")
    return tuple(
        HeadSpec(name, size, _GEN5_SCOPES[name], neutral_choice(name, size), "published")
        for name, size in zip(names, sizes, strict=True)
    )


def _planned(name: str, scope: str, choices: Sequence[str]) -> HeadSpec:
    return HeadSpec(name, len(choices), scope, 0, "planned", tuple(choices))


_HOLD_LABELS: tuple[str, ...] = tuple(f"hold_{ticks}_ticks" for ticks in DURATION_BUCKET_TICKS[1:])

GEN5_HEADS: tuple[HeadSpec, ...] = _published_heads()
"""Today's twelve heads, in the gym's order."""

GEN7_NEW_HEADS: tuple[HeadSpec, ...] = (
    _planned("duration", "hands", ("no_hold", *_HOLD_LABELS)),
    _planned("item_move", "hands", ("none", "to_hotbar", "to_bag", "to_container", "from_container")),
    _planned("item_drop_give", "hands", ("none", "drop_one", "drop_stack", "give")),
    _planned("recipe", "interaction", ("none", *(f"candidate_{index}" for index in range(1, 17)))),
    _planned("door", "interaction", ("none", "open", "close", "toggle")),
    _planned("mount", "locomotion", ("none", "mount", "dismount", "glider_deploy", "glider_stow", "glider_dive")),
    _planned("rune_set", "abilities_magic", ("none", *(f"rune_slot_{index}" for index in range(1, 9)))),
    _planned("cast", "abilities_magic", ("none", *(f"cast_{label}" for label in _HOLD_LABELS))),
    _planned("dodge", "posture_defence", ("none", "left", "right")),
)
"""The nine heads that the 0.7.2 surface adds (PLANNED; E1 and E8 may change the sizes)."""

_LAYOUTS: dict[str, tuple[HeadSpec, ...]] = {"gen5": GEN5_HEADS, "gen7": GEN5_HEADS + GEN7_NEW_HEADS}
_DISABLED: dict[str, dict[str, tuple[int, ...]]] = {"gen5": {}, "gen7": {LOCOMOTION_HEAD: GEN5_DODGE_CHOICES}}
_SCOPE_OF: dict[str, str] = {head.name: head.scope for head in _LAYOUTS["gen7"]}


def _layout(layout: str) -> tuple[HeadSpec, ...]:
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUT_NAMES}")
    return _LAYOUTS[layout]


def head_layout(layout: str) -> tuple[HeadSpec, ...]:
    """Return the heads of ``layout`` (``"gen5"`` or ``"gen7"``) in order."""

    return _layout(layout)


def head_names(layout: str) -> tuple[str, ...]:
    """Return the head names of ``layout`` in order."""

    return tuple(head.name for head in _layout(layout))


def head_sizes(layout: str) -> tuple[int, ...]:
    """Return the head sizes of ``layout`` in order."""

    return tuple(head.size for head in _layout(layout))


def scope_of(head: str) -> str:
    """Return the scope of ``head``.

    Raises:
        KeyError: If no layout has a head of that name.
    """

    try:
        return _SCOPE_OF[head]
    except KeyError:
        raise KeyError(f"unknown head {head!r}") from None


def scope_heads(scope: str, layout: str = "gen7") -> tuple[str, ...]:
    """Return the heads of ``scope`` that ``layout`` has, in layout order."""

    _require_scope(scope)
    return tuple(head.name for head in _layout(layout) if head.scope == scope)


def scope_mask(scope: str, present_heads: Sequence[str]) -> tuple[bool, ...]:
    """Return, for each head in ``present_heads``, whether it belongs to ``scope``.

    This is how the same code runs on today's twelve heads and on the gen-7 layout: a scope with no
    present head (``interaction`` on gen 5) gets an all-False mask.

    Raises:
        ValueError: If ``scope`` is not one of ``SCOPE_NAMES``.
        KeyError: If a present head is unknown.
    """

    _require_scope(scope)
    return tuple(scope_of(head) == scope for head in present_heads)


def legal_choices(layout: str) -> tuple[tuple[bool, ...], ...]:
    """Return, per head of ``layout``, which choices the layout allows (gen 7 disables gen-5 dodges)."""

    disabled = _DISABLED[layout] if layout in _DISABLED else {}
    return tuple(
        tuple(choice not in disabled.get(head.name, ()) for choice in range(head.size)) for head in _layout(layout)
    )


def neutral_factors(layout: str) -> tuple[int, ...]:
    """Return the no-op choice of every head of ``layout``, in order."""

    return tuple(head.neutral for head in _layout(layout))


def _require_scope(scope: str) -> None:
    if scope not in SCOPE_NAMES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPE_NAMES}")


def tick_seconds(layout: str = "gen7") -> float:
    """Return the seconds per tick of ``layout``: 1/30 s on gen 7, 0.045 s on gen 5."""

    _layout(layout)
    return TICK_SECONDS_BY_LAYOUT[layout]


def hold_ticks(seconds: float, layout: str = "gen7") -> int:
    """Return the ticks a hold of ``seconds`` needs on ``layout``: ``ceil(seconds / tick_seconds(layout))``."""

    return int(math.ceil(_number(seconds, "seconds", 0.0) / tick_seconds(layout) - 1e-9))


# ---- tokens ----


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _number(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if _is_bool(value) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    if (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {number}")
    return number


def _integer(value: Any, name: str) -> int:
    if _is_bool(value) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _count(value: Any, name: str) -> int:
    number = _integer(value, name)
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


def _flag(value: Any, name: str) -> bool:
    if not _is_bool(value):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have 3 values, got {len(items)}")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))  # type: ignore[return-value]


def _cell(value: Any, name: str) -> tuple[int, int, int]:
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have 3 values, got {len(items)}")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(items))  # type: ignore[return-value]


def _member(value: Any, options: tuple[str, ...], name: str) -> str:
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


def _one_hot(value: str, options: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(1.0 if option == value else 0.0 for option in options)


@dataclass(frozen=True)
class EntityToken:
    """A creature or person: relative position and velocity in the agent's frame, and a role."""

    handle: int
    role: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    visible: bool
    last_seen_age_ticks: int
    health_fraction: float
    marked: bool = False

    FEATURES = (*(f"role_{role}" for role in ENTITY_ROLES), "forward", "right", "up", "v_forward", "v_right", "v_up", "visible",
                "last_seen_age", "health_fraction", "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "handle", _count(self.handle, "EntityToken.handle"))
        _member(self.role, ENTITY_ROLES, "EntityToken.role")
        object.__setattr__(self, "position", _vector(self.position, "EntityToken.position"))
        object.__setattr__(self, "velocity", _vector(self.velocity, "EntityToken.velocity"))
        _flag(self.visible, "EntityToken.visible")
        object.__setattr__(self, "last_seen_age_ticks", _count(self.last_seen_age_ticks, "EntityToken.last_seen_age_ticks"))
        object.__setattr__(self, "health_fraction", _number(self.health_fraction, "EntityToken.health_fraction", 0.0, 1.0))
        _flag(self.marked, "EntityToken.marked")

    def features(self) -> tuple[float, ...]:
        """Role one-hot, position, velocity, visible, age / TICKS_SATURATION (capped at 1), health, marked."""

        age = min(self.last_seen_age_ticks, TICKS_SATURATION) / TICKS_SATURATION
        return (*_one_hot(self.role, ENTITY_ROLES), *self.position, *self.velocity, float(self.visible), age,
                self.health_fraction, float(self.marked))


@dataclass(frozen=True)
class BlockCandidateToken:
    """A block the agent can act on: its world cell, the vector to it in the agent's frame, a face."""

    cell: tuple[int, int, int]
    offset: tuple[float, float, float]
    face: str
    block_kind: int
    marked: bool = False

    FEATURES = ("forward", "right", "up", *(f"face_{face}" for face in FACES), "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _cell(self.cell, "BlockCandidateToken.cell"))
        object.__setattr__(self, "offset", _vector(self.offset, "BlockCandidateToken.offset"))
        _member(self.face, FACES, "BlockCandidateToken.face")
        object.__setattr__(self, "block_kind", _count(self.block_kind, "BlockCandidateToken.block_kind"))
        _flag(self.marked, "BlockCandidateToken.marked")

    def features(self) -> tuple[float, ...]:
        return (*self.offset, *_one_hot(self.face, FACES), float(self.marked))


@dataclass(frozen=True)
class InventorySlotToken:
    """One inventory slot; ``item_id`` 0 means empty."""

    slot: int
    item_id: int
    quantity: int
    durability: float
    marked: bool = False

    FEATURES = ("quantity", "durability", "empty", "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _count(self.slot, "InventorySlotToken.slot"))
        object.__setattr__(self, "item_id", _count(self.item_id, "InventorySlotToken.item_id"))
        object.__setattr__(self, "quantity", _count(self.quantity, "InventorySlotToken.quantity"))
        object.__setattr__(self, "durability", _number(self.durability, "InventorySlotToken.durability", 0.0, 1.0))
        _flag(self.marked, "InventorySlotToken.marked")

    def features(self) -> tuple[float, ...]:
        return (float(self.quantity), self.durability, float(self.item_id == 0), float(self.marked))


@dataclass(frozen=True)
class GeometryToken:
    """A nearby cell's shape for movement: solid, climbable, fluid, hazard, walkable top."""

    offset: tuple[float, float, float]
    solid: bool
    climbable: bool
    fluid: str
    hazard: bool
    top_walkable: bool
    marked: bool = False

    FEATURES = ("forward", "right", "up", "solid", "climbable", *(f"fluid_{fluid}" for fluid in FLUIDS), "hazard",
                "top_walkable", "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", _vector(self.offset, "GeometryToken.offset"))
        for name in ("solid", "climbable", "hazard", "top_walkable", "marked"):
            _flag(getattr(self, name), f"GeometryToken.{name}")
        _member(self.fluid, FLUIDS, "GeometryToken.fluid")

    def features(self) -> tuple[float, ...]:
        return (*self.offset, float(self.solid), float(self.climbable), *_one_hot(self.fluid, FLUIDS),
                float(self.hazard), float(self.top_walkable), float(self.marked))


@dataclass(frozen=True)
class LightToken:
    """A light level near the agent, in [0, 1]."""

    offset: tuple[float, float, float]
    level: float
    marked: bool = False

    FEATURES = ("forward", "right", "up", "level", "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", _vector(self.offset, "LightToken.offset"))
        object.__setattr__(self, "level", _number(self.level, "LightToken.level", 0.0, 1.0))
        _flag(self.marked, "LightToken.marked")

    def features(self) -> tuple[float, ...]:
        return (*self.offset, self.level, float(self.marked))


@dataclass(frozen=True)
class SelfToken:
    """The agent itself: resource meters as fractions, grounded, facing and speed."""

    health: float
    stamina: float
    mana: float
    signature_energy: float
    oxygen: float
    held_durability: float
    grounded: bool
    yaw: float
    pitch: float
    speed: float
    marked: bool = False

    FEATURES = ("health", "stamina", "mana", "signature_energy", "oxygen", "held_durability", "grounded",
                "yaw_sin", "yaw_cos", "pitch", "speed", "marked")

    def __post_init__(self) -> None:
        for name in ("health", "stamina", "mana", "signature_energy", "oxygen", "held_durability"):
            object.__setattr__(self, name, _number(getattr(self, name), f"SelfToken.{name}", 0.0, 1.0))
        _flag(self.grounded, "SelfToken.grounded")
        object.__setattr__(self, "yaw", _number(self.yaw, "SelfToken.yaw", -math.pi, math.pi))
        object.__setattr__(self, "pitch", _number(self.pitch, "SelfToken.pitch", -math.pi / 2, math.pi / 2))
        object.__setattr__(self, "speed", _number(self.speed, "SelfToken.speed", 0.0))
        _flag(self.marked, "SelfToken.marked")

    def features(self) -> tuple[float, ...]:
        return (self.health, self.stamina, self.mana, self.signature_energy, self.oxygen, self.held_durability,
                float(self.grounded), math.sin(self.yaw), math.cos(self.yaw), self.pitch, self.speed,
                float(self.marked))


@dataclass(frozen=True)
class KitToken:
    """The held item's kit: combo window, hold thresholds, charged tiers, costs, reach, projectile."""

    combo_window_seconds: float
    hold_thresholds_seconds: tuple[float, ...]
    charged_tiers: int
    stamina_cost: float
    mana_cost: float
    signature_cost: float
    reach_blocks: float
    projectile_launch: float
    projectile_gravity: float
    marked: bool = False

    FEATURES = ("combo_window_seconds", *(f"hold_threshold_{index}" for index in range(MAX_HOLD_THRESHOLDS)),
                "charged_tiers", "stamina_cost", "mana_cost", "signature_cost", "reach_blocks",
                "projectile_launch", "projectile_gravity", "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "combo_window_seconds", _number(self.combo_window_seconds, "KitToken.combo_window_seconds", 0.0))
        thresholds = tuple(
            _number(item, f"KitToken.hold_thresholds_seconds[{index}]", 0.0)
            for index, item in enumerate(self.hold_thresholds_seconds)
        )
        if len(thresholds) > MAX_HOLD_THRESHOLDS:
            raise ValueError(f"KitToken holds at most {MAX_HOLD_THRESHOLDS} thresholds, got {len(thresholds)}")
        if list(thresholds) != sorted(thresholds):
            raise ValueError(f"KitToken.hold_thresholds_seconds must ascend, got {thresholds}")
        object.__setattr__(self, "hold_thresholds_seconds", thresholds)
        object.__setattr__(self, "charged_tiers", _count(self.charged_tiers, "KitToken.charged_tiers"))
        for name in ("stamina_cost", "mana_cost", "signature_cost", "reach_blocks", "projectile_launch",
                     "projectile_gravity"):
            object.__setattr__(self, name, _number(getattr(self, name), f"KitToken.{name}", 0.0))
        _flag(self.marked, "KitToken.marked")

    def features(self) -> tuple[float, ...]:
        padded = self.hold_thresholds_seconds + (0.0,) * (MAX_HOLD_THRESHOLDS - len(self.hold_thresholds_seconds))
        return (self.combo_window_seconds, *padded, float(self.charged_tiers), self.stamina_cost, self.mana_cost,
                self.signature_cost, self.reach_blocks, self.projectile_launch, self.projectile_gravity,
                float(self.marked))


@dataclass(frozen=True)
class RouteToken:
    """A next waypoint from map memory, in the agent's frame."""

    offset: tuple[float, float, float]
    index: int
    final: bool
    marked: bool = False

    FEATURES = ("forward", "right", "up", "index", "final", "marked")

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", _vector(self.offset, "RouteToken.offset"))
        object.__setattr__(self, "index", _count(self.index, "RouteToken.index"))
        _flag(self.final, "RouteToken.final")
        _flag(self.marked, "RouteToken.marked")

    def features(self) -> tuple[float, ...]:
        return (*self.offset, float(self.index), float(self.final), float(self.marked))


TOKEN_TYPES: Mapping[str, type] = {
    "entity": EntityToken,
    "block_candidate": BlockCandidateToken,
    "inventory_slot": InventorySlotToken,
    "geometry": GeometryToken,
    "light": LightToken,
    "self": SelfToken,
    "kit": KitToken,
    "route": RouteToken,
}
"""Token type name -> record class."""


# ---- JSON export ----


def surface_dict(layout: str) -> dict[str, Any]:
    """Return the surface of ``layout`` as a JSON-shaped mapping (heads, scopes, holds, tokens)."""

    heads = _layout(layout)
    return {
        "layout": layout,
        "tick_seconds": tick_seconds(layout),
        "scopes": {scope: list(scope_heads(scope, layout)) for scope in SCOPE_NAMES},
        "heads": [
            {"name": head.name, "size": head.size, "scope": head.scope, "neutral": head.neutral,
             "status": head.status, "choices": list(head.choices)}
            for head in heads
        ],
        "disabled_choices": {name: list(choices) for name, choices in _DISABLED[layout].items()},
        "duration_bucket_ticks": list(DURATION_BUCKET_TICKS),
        "tokens": {name: {"features": list(kind.FEATURES), "width": len(kind.FEATURES)}
                   for name, kind in TOKEN_TYPES.items()},
    }


def surface_json(layout: str) -> str:
    """Return canonical JSON text of :func:`surface_dict` (sorted keys)."""

    return json.dumps(surface_dict(layout), sort_keys=True)


def surface_from_dict(mapping: Mapping[str, Any]) -> tuple[HeadSpec, ...]:
    """Parse the heads of an exported surface and check that they match this module's layout.

    Raises:
        ValueError: If the layout is unknown or any head, scope, choice or token entry differs.
    """

    layout = mapping.get("layout")
    expected = surface_dict(layout) if layout in _LAYOUTS else None
    if expected is None:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUT_NAMES}")
    if json.loads(json.dumps(dict(mapping), sort_keys=True)) != json.loads(surface_json(layout)):
        raise ValueError(f"the exported {layout} surface differs from contract/surface.py")
    return tuple(
        HeadSpec(item["name"], item["size"], item["scope"], item["neutral"], item["status"], tuple(item["choices"]))
        for item in mapping["heads"]
    )


__all__ = [
    "DURATION_BUCKET_TICKS",
    "ENTITY_ROLES",
    "FACES",
    "FLUIDS",
    "GEN5_DODGE_CHOICES",
    "GEN5_HEADS",
    "GEN7_NEW_HEADS",
    "LAYOUT_NAMES",
    "LOCOMOTION_HEAD",
    "MAX_HOLD_THRESHOLDS",
    "SCOPE_NAMES",
    "SIGNED_LOOK_HEADS",
    "TICK_SECONDS",
    "TICK_SECONDS_BY_LAYOUT",
    "TOKEN_TYPES",
    "BlockCandidateToken",
    "EntityToken",
    "GeometryToken",
    "HeadSpec",
    "InventorySlotToken",
    "KitToken",
    "LightToken",
    "RouteToken",
    "SelfToken",
    "head_layout",
    "head_names",
    "head_sizes",
    "hold_ticks",
    "legal_choices",
    "neutral_choice",
    "neutral_factors",
    "scope_heads",
    "scope_mask",
    "scope_of",
    "surface_dict",
    "surface_from_dict",
    "surface_json",
]
