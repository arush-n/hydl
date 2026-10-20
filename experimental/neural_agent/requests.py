"""The request stream between the middle and the lower layer: typed requests down, statuses up.

``docs/SYSTEM_ONE.md`` section 9.10 (owner's answers, 2026-10-17 night) and ``docs/MIDDLE_LAYER.md``
section 5. The middle layer (the decision layer, writer ``"jev"`` of S_t) turns the LLM's goals and
its behaviour programs (section 9.9) into typed :class:`ActionRequest` records in the scratchpad's
``requests`` section. The lower layer (the combiner in the dispatcher, writer ``"skills"``) checks
each live request with :func:`validate`, grants the valid ones greedily by scope, and streams one
:class:`RequestStatus` per request back into the ``skills`` section. This module is the middle
layer's side of that protocol: the two records, the argument schema of each known verb, and the
check. The combiner itself belongs to the lower layer.

Arguments travel typed, never in prose. Positions are absolute world coordinates and directions are
unit vectors, both in the game's frame (MIDDLE_LAYER.md section 5). An entity is a handle the belief
knows. A cell is an integer block coordinate, and it always comes with the face to act on. The
faces are named as Hytale's ``protocol.BlockFace`` values 1 to 6 (``FACE_NAMES[k]`` is value
``k + 1``; ``hytalegym/jax/world/interactions/blocks/candidates.py``). Scalar ``i`` is the verb's
contract's ``param_names[i]``. Each argument has a presence flag: ``None`` means absent (JSON
``null``).

Argument requirements. ``validate`` takes the verb's contract (its ``param_names`` name the scalars,
and every named scalar is needed) and an :class:`ArgumentSchema`: which of position, direction,
entity and cell the verb needs, and which scopes it drives. The schemas come from a mapping keyed by
verb, ``ARGUMENT_SCHEMAS`` by default (idle, pursuit, evader, and the combat verbs of 2026-10-19). The lower layer's catalogue parser
will supply the full mapping later. It is passed as ``validate(..., schemas=...)``, so nothing else
changes when it does.

Validation runs in this fixed order. The first failing check gives the reason, written
``"<code>: <detail>"``::

    expired               tick >= expires_tick
    verb_unknown          no contract for the verb, or no argument schema for it
    missing_argument      a needed position, direction, entity or cell (with its face) is absent,
                          or a scalar the contract names is absent
    non_finite_argument   a needed position or direction component, or a named scalar, is NaN or inf
    out_of_bounds         a needed position is outside the world box: not lower <= p <= upper on
                          every axis; or a needed cell's unit cube is outside: not lower <= c and
                          c + 1 <= upper on every axis
    entity_unknown        a needed entity is not among the entities the belief knows
    scope_illegal         a scope the verb drives is not legal under the current action mask
    predecessor_not_done  the mode is ``after:<id>`` and request <id> is not done

An argument that the verb does not need is never read, so it is not checked. ``validate`` judges
argument values. Building an ``ActionRequest`` checks only the structure (types, widths, the mode,
the ranges of the arbitration fields). A non-finite argument value can therefore be built, and the
combiner rejects it with a reason instead of failing.

Scopes. A scope is a group of action heads that one granted request owns at a time.
``SCOPES`` lists the six groups of ``skill_agent/docs/BUILD.md`` section 3.1. Section 9.10 named four
(locomotion, look, hands, posture). The six-scope plan is newer, and the lower layer's surface draft
uses the same six names. The default schemas give each proof-of-concept skill the scopes of the heads
its action scope can move off neutral. For idle that is none. For pursuit it is
``locomotion_gait_compass``, the camera and body yaw and the pitch. For evader it is ``base_action``
and ``locomotion_gait_compass``. Each head is then mapped to its scope as that draft maps the twelve
gen-5 heads: ``base_action`` to hands, body yaw to locomotion, camera yaw and pitch to gaze.

Combat verbs (2026-10-19). ``sword`` (melee combos), ``guard`` (block on the target's wind-up), ``punish`` (strike in
the target's recovery) and ``keep_range`` (step out of its reach) take the target entity. Sword and punish drive hands and abilities_magic: a weapon attacks through its
ability slots, so the strike drives the ability head (the peer's catalogue rule, uncommitted at 2026-10-19).
Guard drives posture_defence, keep_range drives locomotion. No skill implements them yet, so their contracts are
declared here (``COMBAT_CONTRACTS``: sword takes ``hold_seconds``, keep_range takes ``band_min`` and ``band_max``, the others
none; the contract's default control; no checkpoint). ``dodge`` is deferred:
the gen-7 dodge head is still a stub, so a dodge request has no schema here and ``validate`` returns ``verb_unknown``.
The id ``guard`` is the catalogue's own (section 12, scoped posture_defence by its section, per the skill layer,
2026-10-19); no rename on either side.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Collection, Mapping, Sequence

import numpy as np

from experimental.neural_agent.contracts import (
    INT32_MAX,
    MAX_GOALS,
    PARAM_WIDTH,
    SKILL_IDLE,
    SkillContract,
    SkillOutcome,
    _finite_number,
    _non_negative_int,
    _text,
)

SCOPES: tuple[str, ...] = ("locomotion", "gaze", "hands", "posture_defence", "interaction", "abilities_magic")
"""The six scopes (head groups) of skill_agent/docs/BUILD.md section 3.1. The combiner grants one request per scope."""
FACE_NAMES: tuple[str, ...] = ("up", "down", "north", "south", "east", "west")
"""Block faces; ``FACE_NAMES[k]`` is Hytale protocol.BlockFace value ``k + 1``."""
MODE_PARALLEL: str = "parallel"
"""Run beside the agent's other granted requests whose scopes are free."""
MODE_INTERRUPT: str = "interrupt"
"""Pre-empt. What the combiner does with it is the lower layer's (section 9.10 does not say)."""
AFTER_PREFIX: str = "after:"
"""The mode ``after:<id>`` runs once request ``<id>`` of the same agent is done."""
STATUS_KINDS: tuple[str, ...] = ("accepted", "rejected", "deferred", "running", "done", "aborted")
"""The status values of section 9.10. ``accepted`` marks the tick of the grant, ``running`` each tick after it."""
REJECTION_CODES: tuple[str, ...] = (
    "expired",
    "verb_unknown",
    "missing_argument",
    "non_finite_argument",
    "out_of_bounds",
    "entity_unknown",
    "scope_illegal",
    "predecessor_not_done",
)
"""The codes ``validate`` returns, in the order it checks them."""
ARGUMENT_NAMES: tuple[str, ...] = ("position", "direction", "entity", "cell")
"""The typed arguments besides the scalars, in the order ``validate`` checks them."""

_CELL_MIN: int = -INT32_MAX - 1


def _real(value: Any, name: str) -> float:
    """Return ``value`` as a float. NaN and inf are kept, because validate() judges argument values."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _int32(value: Any, name: str) -> int:
    """Return ``value`` as a signed int32. Whole floats are accepted; bools are not."""

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not _CELL_MIN <= value <= INT32_MAX:
        raise ValueError(f"{name} must fit int32, got {value}")
    return int(value)


def _triple(value: Any, name: str, item: Any) -> tuple[Any, Any, Any] | None:
    """Return None for None, else exactly three items converted by ``item``. A bare string is refused."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be three numbers, not a string")
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have 3 values, got {len(items)}")
    return tuple(item(entry, f"{name}[{index}]") for index, entry in enumerate(items))


def _unit(value: Any, name: str) -> float:
    """Return a finite float in [0, 1]."""

    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def _flag(value: Any, name: str) -> bool:
    """Return ``value`` when it is a bool."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
    return value


def _unique_texts(values: Any, name: str) -> tuple[str, ...]:
    """Return non-empty strings, each once. A bare string is refused."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not one string")
    items = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(values))
    if len(set(items)) != len(items):
        raise ValueError(f"{name} repeats a name: {items}")
    return items


# ---- the argument schema of each verb ----


@dataclass(frozen=True)
class ArgumentSchema:
    """Which typed arguments a verb needs, and which scopes it drives (section 9.10, "Contracts").

    The scalars are not listed here. They are the verb's contract's ``param_names``, and each is
    needed.

    Attributes:
        position: The verb needs ``args.position``.
        direction: The verb needs ``args.direction``.
        entity: The verb needs ``args.entity``.
        cell: The verb needs ``args.cell`` and ``args.face``.
        scopes: The scopes the verb drives, each once. ``()`` drives none: every head stays neutral.
        any_of: Groups of argument names of which at least one must be present (for example
            ``(("position", "entity"),)`` for a verb that takes a place or a target). A present
            alternative is then checked like a needed argument. Mirrors the catalogue parser's
            ``any_of`` so its contracts adapt without loss.
    """

    position: bool = False
    direction: bool = False
    entity: bool = False
    cell: bool = False
    scopes: tuple[str, ...] = ()
    any_of: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        """Check the flags, the scope names and the any-of groups."""

        for name in ARGUMENT_NAMES:
            _flag(getattr(self, name), f"ArgumentSchema.{name}")
        object.__setattr__(self, "scopes", _unique_texts(self.scopes, "ArgumentSchema.scopes"))
        groups = []
        for index, group in enumerate(self.any_of):
            names = _unique_texts(group, f"ArgumentSchema.any_of[{index}]")
            unknown = [name for name in names if name not in ARGUMENT_NAMES]
            if not names or unknown:
                raise ValueError(f"ArgumentSchema.any_of[{index}] must name one or more of {ARGUMENT_NAMES}, got {group!r}")
            groups.append(names)
        object.__setattr__(self, "any_of", tuple(groups))

    def needs(self) -> tuple[str, ...]:
        """Return the names in ``ARGUMENT_NAMES`` that the verb needs, in that order."""

        return tuple(name for name in ARGUMENT_NAMES if getattr(self, name))


ARGUMENT_SCHEMAS: Mapping[str, ArgumentSchema] = MappingProxyType(
    {
        SKILL_IDLE: ArgumentSchema(),
        "pursuit": ArgumentSchema(entity=True, scopes=("locomotion", "gaze")),
        "evader": ArgumentSchema(entity=True, scopes=("locomotion", "hands")),
        "sword": ArgumentSchema(entity=True, scopes=("hands", "abilities_magic")),
        "guard": ArgumentSchema(entity=True, scopes=("posture_defence",)),
        "punish": ArgumentSchema(entity=True, scopes=("hands", "abilities_magic")),
        "keep_range": ArgumentSchema(entity=True, scopes=("locomotion",)),
    }
)
"""The schemas of the proof-of-concept skills (``skill_library/``) and of the four combat verbs (2026-10-19), keyed
by skill id. The combat verbs take the target entity; their scalars are the ``param_names`` of ``COMBAT_CONTRACTS``, not part of the
schema. Their scopes match the peer's catalogue parser where it names
one (``VERB_SCOPES``: keep_range locomotion; it names punish hands, and the strike rule adds abilities_magic, which
these declarations use for sword and punish). The rule is the peer's ``WEAPON_SECTIONS`` and ``STRIKE_VERBS``,
uncommitted at 2026-10-19.

Pursuit closes on a target and the evader keeps its distance from one
(``skill_library/descriptions.py``). Today both read the belief's single target, so the entity names
which one is meant. Idle needs nothing and drives nothing. The module docstring says where the scopes
come from. Replace this mapping with the catalogue parser's by passing ``schemas=`` to :func:`validate`.
"""

COMBAT_VERBS: tuple[str, ...] = ("sword", "guard", "punish", "keep_range")
"""The combat verbs with a schema here. ``dodge`` is deferred (module docstring)."""


def _combat_contract(
    verb: str, observation_groups: tuple[str, ...], success: str, failure: str, scalars: tuple[str, ...] = ()
) -> SkillContract:
    """Declare one combat verb's contract: no scalars, the contract's default control, no checkpoint.

    ``action_scope_sha256`` is a digest of the verb and its declared scopes, not of a learned action scope: no
    ``FactoredActionScope`` exists for the combat verbs yet.
    """

    digest = hashlib.sha256(repr((verb, ARGUMENT_SCHEMAS[verb].scopes)).encode("utf-8")).hexdigest()
    return SkillContract(
        skill_id=verb,
        version="declared",
        observation_groups=observation_groups,
        action_scope_sha256=digest,
        param_names=scalars,
        termination=("goal_replaced",),
        success_signals=(success,),
        failure_signals=(failure,),
        checkpoint_sha256=None,
    )


COMBAT_CONTRACTS: Mapping[str, SkillContract] = MappingProxyType(
    {
        "sword": _combat_contract("sword", ("belief", "action_mask"), "target_hit", "target_out_of_reach", ("hold_seconds",)),
        "guard": _combat_contract(
            "guard", ("belief", "target_i32", "action_mask"), "attack_blocked", "target_not_winding_up"
        ),
        "punish": _combat_contract(
            "punish", ("belief", "target_i32", "action_mask"), "target_hit", "target_out_of_reach"
        ),
        "keep_range": _combat_contract(
            "keep_range", ("belief", "target_i32", "action_mask"), "distance_kept", "target_not_visible", ("band_min", "band_max")
        ),
    }
)
"""The declared contracts of the combat verbs, keyed by id. Nothing else in the decision layer or the skill layer
declares these ids yet (checked 2026-10-19). ``target_i32`` is the legal observation's target integer group, which
holds ``attack_phase`` and ``attack_index``. The success and failure labels are placeholders for the lower layer.
Scalars (2026-10-19): sword takes ``hold_seconds`` (the catalogue's section-9 group default), keep_range takes
``band_min`` and ``band_max`` (the catalogue's section-12 band tactics); punish and guard take none."""


# ---- the records ----


@dataclass(frozen=True)
class RequestArgs:
    """The typed arguments of a request. Each is None when absent (its presence flag).

    Attributes:
        position: Three floats, absolute world coordinates, or None.
        direction: Three floats, a unit vector in the game's frame, or None.
        entity: A non-negative entity handle, or None.
        cell: Three int32 block coordinates, or None. Given together with ``face``.
        face: One of ``FACE_NAMES``, or None. Given together with ``cell``.
        scalars: ``PARAM_WIDTH`` entries, each a float or None. Shorter input is padded with None.
            Scalar ``i`` is the verb's ``param_names[i]``.

    Raises:
        TypeError: If a value has the wrong type.
        ValueError: If a width is wrong, ``cell`` and ``face`` are not given together, or ``face``
            is not a face name.
    """

    position: tuple[float, float, float] | None = None
    direction: tuple[float, float, float] | None = None
    entity: int | None = None
    cell: tuple[int, int, int] | None = None
    face: str | None = None
    scalars: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        """Check the structure and normalize the sequences to tuples."""

        object.__setattr__(self, "position", _triple(self.position, "RequestArgs.position", _real))
        object.__setattr__(self, "direction", _triple(self.direction, "RequestArgs.direction", _real))
        if self.entity is not None:
            object.__setattr__(self, "entity", _non_negative_int(self.entity, "RequestArgs.entity"))
        if (self.cell is None) != (self.face is None):
            raise ValueError("RequestArgs.cell and RequestArgs.face go together: give both or neither")
        object.__setattr__(self, "cell", _triple(self.cell, "RequestArgs.cell", _int32))
        if self.face is not None and self.face not in FACE_NAMES:
            raise ValueError(f"RequestArgs.face must be one of {FACE_NAMES}, got {self.face!r}")
        if isinstance(self.scalars, (str, bytes)):
            raise TypeError("RequestArgs.scalars must be a sequence, not a string")
        scalars = tuple(self.scalars)
        if len(scalars) > PARAM_WIDTH:
            raise ValueError(f"RequestArgs.scalars has {len(scalars)} entries; the limit is {PARAM_WIDTH}")
        scalars = tuple(
            None if value is None else _real(value, f"RequestArgs.scalars[{index}]")
            for index, value in enumerate(scalars)
        )
        object.__setattr__(self, "scalars", scalars + (None,) * (PARAM_WIDTH - len(scalars)))

    def numbers(self) -> tuple[float, ...]:
        """Return every float the arguments hold: the position, the direction and the present scalars."""

        values: list[float] = []
        for vector in (self.position, self.direction):
            if vector is not None:
                values.extend(vector)
        values.extend(value for value in self.scalars if value is not None)
        return tuple(values)


@dataclass(frozen=True, kw_only=True)
class ActionRequest:
    """One request from the middle layer to the lower layer (section 9.10).

    Attributes:
        request_id: Non-negative, monotonic per agent.
        agent: The agent (lane) the request is for.
        issued_tick: The tick at which it was issued.
        expires_tick: The tick at which it expires. It is live while ``tick < expires_tick``, and
            ``expires_tick`` must be after ``issued_tick``.
        verb: A skill id of the library.
        args: The typed arguments. Default: none.
        mode: ``"parallel"`` (default), ``"interrupt"`` or ``"after:<id>"``, where <id> is an earlier
            request id of the same agent.
        urgency: In [0, 1]. The first key of the combiner's greedy order.
        confidence: In [0, 1]. The second key.
        priority: Finite. The third key. Default 1.0, as ``GoalSpec.priority``.
        goal_slot: The executive slot the request serves, or -1 for none (default).
        executive_version: The E_t version it was derived from. Default 0.
        program_step: The step of the behaviour program it serves (section 9.9). Default 0.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a value is out of range or the mode is malformed.
    """

    request_id: int
    agent: int
    issued_tick: int
    expires_tick: int
    verb: str
    args: RequestArgs = field(default_factory=RequestArgs)
    mode: str = MODE_PARALLEL
    urgency: float
    confidence: float
    priority: float = 1.0
    goal_slot: int = -1
    executive_version: int = 0
    program_step: int = 0

    def __post_init__(self) -> None:
        """Check the fields and normalize the numbers to Python types."""

        for name in ("request_id", "agent", "issued_tick", "expires_tick", "executive_version", "program_step"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), f"ActionRequest.{name}"))
        if self.expires_tick <= self.issued_tick:
            raise ValueError(
                f"ActionRequest.expires_tick ({self.expires_tick}) must be after issued_tick ({self.issued_tick})"
            )
        _text(self.verb, "ActionRequest.verb")
        if not isinstance(self.args, RequestArgs):
            raise TypeError(f"ActionRequest.args must be a RequestArgs, got {type(self.args).__name__}")
        _text(self.mode, "ActionRequest.mode")
        after = self.after
        if after is not None:
            if after >= self.request_id:
                raise ValueError(
                    f"ActionRequest.mode {self.mode!r} must name an earlier request than {self.request_id}"
                )
            object.__setattr__(self, "mode", f"{AFTER_PREFIX}{after}")
        object.__setattr__(self, "urgency", _unit(self.urgency, "ActionRequest.urgency"))
        object.__setattr__(self, "confidence", _unit(self.confidence, "ActionRequest.confidence"))
        object.__setattr__(self, "priority", _finite_number(self.priority, "ActionRequest.priority"))
        slot = self.goal_slot
        if isinstance(slot, bool):
            raise TypeError("ActionRequest.goal_slot must be an integer, got bool")
        slot = -1 if slot == -1 else _non_negative_int(slot, "ActionRequest.goal_slot")
        if slot >= MAX_GOALS:
            raise ValueError(f"ActionRequest.goal_slot must be -1 (no slot) or below {MAX_GOALS}, got {slot}")
        object.__setattr__(self, "goal_slot", slot)

    @property
    def after(self) -> int | None:
        """The predecessor's request id when the mode is ``after:<id>``, else None.

        Raises:
            ValueError: If the mode is none of the three forms.
        """

        if self.mode in (MODE_PARALLEL, MODE_INTERRUPT):
            return None
        tail = self.mode[len(AFTER_PREFIX) :]
        if not self.mode.startswith(AFTER_PREFIX) or not tail.isdigit() or not tail.isascii():
            raise ValueError(
                f"ActionRequest.mode must be {MODE_PARALLEL!r}, {MODE_INTERRUPT!r} or 'after:<id>', got {self.mode!r}"
            )
        return int(tail)


@dataclass(frozen=True, kw_only=True)
class RequestStatus:
    """One status of one request, from the lower layer back into the state (section 9.10).

    Attributes:
        agent, request_id: The request this status is about.
        kind: One of ``STATUS_KINDS``.
        tick: The tick of the status.
        reason: Non-empty for ``rejected`` and ``aborted`` (for a rejection, the ``validate``
            reason). Empty otherwise.
        conflict_with: For ``deferred`` only: the id of the request that holds the scope.
        outcome: For ``done`` only: the terminal ``SkillOutcome`` (not RUNNING).
        progress: The verb's progress signals by name, as finite numbers, or None. Given as a
            mapping (or pairs) and stored as ``(name, value)`` pairs sorted by name, so the record
            stays immutable. ``dict(status.progress)`` gives the mapping.
        signal: For ``done`` only: the termination label that fired, one of the verb's contract's
            termination labels (the catalogue's success label such as ``"arrived"``, or ``"stuck"``,
            ``"timeout"``, ``"precondition_failed"``, ``"interrupted"``). Empty otherwise.
        heads: The action heads the request drove on this tick. Only ``accepted`` and ``running``
            drive heads.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a kind-specific field is missing, present on the wrong kind, or out of range.
    """

    agent: int
    request_id: int
    kind: str
    tick: int
    reason: str = ""
    conflict_with: int | None = None
    outcome: SkillOutcome | None = None
    progress: tuple[tuple[str, float], ...] | None = None
    heads: tuple[str, ...] = ()
    signal: str = ""

    def __post_init__(self) -> None:
        """Check the fields against the kind."""

        for name in ("agent", "request_id", "tick"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), f"RequestStatus.{name}"))
        if self.kind not in STATUS_KINDS:
            raise ValueError(f"RequestStatus.kind must be one of {STATUS_KINDS}, got {self.kind!r}")
        if not isinstance(self.reason, str):
            raise TypeError(f"RequestStatus.reason must be a string, got {type(self.reason).__name__}")
        if bool(self.reason) != (self.kind in ("rejected", "aborted")):
            raise ValueError(f"RequestStatus.reason is required for rejected and aborted only (kind {self.kind!r})")
        if (self.conflict_with is not None) != (self.kind == "deferred"):
            raise ValueError(f"RequestStatus.conflict_with is required for deferred only (kind {self.kind!r})")
        if not isinstance(self.signal, str):
            raise TypeError(f"RequestStatus.signal must be a string, got {type(self.signal).__name__}")
        if self.signal and self.kind != "done":
            raise ValueError(f"RequestStatus.signal is allowed for done only (kind {self.kind!r})")
        if self.conflict_with is not None:
            object.__setattr__(
                self, "conflict_with", _non_negative_int(self.conflict_with, "RequestStatus.conflict_with")
            )
        if (self.outcome is not None) != (self.kind == "done"):
            raise ValueError(f"RequestStatus.outcome is required for done only (kind {self.kind!r})")
        if self.outcome is not None:
            if isinstance(self.outcome, bool) or not isinstance(self.outcome, (int, np.integer)):
                raise TypeError(f"RequestStatus.outcome must be a SkillOutcome, got {type(self.outcome).__name__}")
            outcome = SkillOutcome(int(self.outcome))
            if outcome is SkillOutcome.RUNNING:
                raise ValueError("RequestStatus.outcome of a done request must be terminal, not RUNNING")
            object.__setattr__(self, "outcome", outcome)
        if self.progress is not None:
            pairs = self.progress.items() if isinstance(self.progress, Mapping) else self.progress
            progress = tuple(
                sorted(
                    (
                        _text(name, "RequestStatus.progress name"),
                        _finite_number(value, f"RequestStatus.progress[{name!r}]"),
                    )
                    for name, value in pairs
                )
            )
            if len({name for name, _ in progress}) != len(progress):
                raise ValueError("RequestStatus.progress repeats a name")
            object.__setattr__(self, "progress", progress)
        heads = _unique_texts(self.heads, "RequestStatus.heads")
        if heads and self.kind not in ("accepted", "running"):
            raise ValueError(
                f"RequestStatus.heads must be empty for kind {self.kind!r}: only a granted request drives heads"
            )
        object.__setattr__(self, "heads", heads)


# ---- the check the combiner runs ----


def _world_box(world_bounds: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(lower, upper)`` as two finite triples with lower < upper on every axis."""

    corners = () if isinstance(world_bounds, (str, bytes)) else tuple(world_bounds)
    if len(corners) != 2:
        raise ValueError("world_bounds must be (lower, upper), two triples of world coordinates")
    lower, upper = (_triple(corner, f"world_bounds[{index}]", _finite_number) for index, corner in enumerate(corners))
    if lower is None or upper is None or not all(low < high for low, high in zip(lower, upper)):
        raise ValueError(f"world_bounds must have lower < upper on every axis, got {lower} and {upper}")
    return lower, upper


def rejection_code(reason: str) -> str:
    """Return the code of a rejection or abort reason: the text before the first colon."""

    return _text(reason, "reason").split(":", 1)[0].strip()


def validate(
    request: ActionRequest,
    contract: SkillContract | None,
    *,
    tick: int,
    world_bounds: Sequence[Sequence[float]],
    known_entities: Collection[int],
    legal_scopes: Collection[str],
    done_ids: Collection[int],
    schemas: Mapping[str, ArgumentSchema] = ARGUMENT_SCHEMAS,
) -> str | None:
    """Return why the combiner must reject ``request`` now, or None when it is valid.

    The checks and their order are in the module docstring. The reason is ``"<code>: <detail>"``
    with a code from ``REJECTION_CODES`` (:func:`rejection_code` reads it back). The detail names the
    request and the verb, so the reason makes sense on its own in the status stream.

    Args:
        request: The request.
        contract: The verb's contract from the catalogue, or None when the catalogue has no such verb.
        tick: The current tick.
        world_bounds: ``(lower, upper)`` corners of the world box in world coordinates.
        known_entities: The entity handles the belief knows for this agent.
        legal_scopes: The scopes that are legal under the agent's current action mask.
        done_ids: Ids of this agent's requests that are done (``after`` predecessors).
        schemas: The argument schemas keyed by verb. Default ``ARGUMENT_SCHEMAS``.

    Raises:
        TypeError: If ``request`` is not an ActionRequest or ``contract`` is not a SkillContract.
        ValueError: If ``contract`` is for another verb, or ``tick`` or ``world_bounds`` is malformed.
    """

    if not isinstance(request, ActionRequest):
        raise TypeError(f"request must be an ActionRequest, got {type(request).__name__}")
    now = _non_negative_int(tick, "tick")
    lower, upper = _world_box(world_bounds)
    verb, rid = request.verb, request.request_id
    if now >= request.expires_tick:
        return f"expired: request {rid} ({verb}) expired at tick {request.expires_tick}; it is tick {now}"
    if contract is None:
        return f"verb_unknown: request {rid} ({verb}) has no skill contract"
    if not isinstance(contract, SkillContract):
        raise TypeError(f"contract must be a SkillContract or None, got {type(contract).__name__}")
    if contract.skill_id != verb:
        raise ValueError(f"contract is for {contract.skill_id!r} but request {rid} names {verb!r}")
    schema = schemas.get(verb)
    if schema is None:
        return f"verb_unknown: request {rid} ({verb}) has no argument schema"
    args = request.args
    needed = schema.needs()
    for name in needed:
        if getattr(args, name) is None:
            what = "cell and face" if name == "cell" else name
            return f"missing_argument: request {rid} ({verb}) needs {what}"
    for group in schema.any_of:
        present = tuple(name for name in group if getattr(args, name) is not None)
        if not present:
            return f"missing_argument: request {rid} ({verb}) needs one of {group}"
        needed = tuple(needed) + tuple(name for name in present if name not in needed)
    named = tuple(enumerate(contract.param_names))
    for index, name in named:
        if args.scalars[index] is None:
            return f"missing_argument: request {rid} ({verb}) needs scalar {index} ({name})"
    for name in ("position", "direction"):
        if name in needed and not all(math.isfinite(value) for value in getattr(args, name)):
            return f"non_finite_argument: request {rid} ({verb}) {name} is {getattr(args, name)}"
    for index, name in named:
        if not math.isfinite(args.scalars[index]):
            return f"non_finite_argument: request {rid} ({verb}) scalar {index} ({name}) is {args.scalars[index]}"
    box = f"the world box {lower} to {upper}"
    if "position" in needed and not all(low <= value <= high for value, low, high in zip(args.position, lower, upper)):
        return f"out_of_bounds: request {rid} ({verb}) position {args.position} is outside {box}"
    if "cell" in needed and not all(
        low <= value and value + 1 <= high for value, low, high in zip(args.cell, lower, upper)
    ):
        return f"out_of_bounds: request {rid} ({verb}) cell {args.cell} is outside {box}"
    if "entity" in needed and args.entity not in known_entities:
        return f"entity_unknown: request {rid} ({verb}) names entity {args.entity}, which the belief does not know"
    illegal = [scope for scope in schema.scopes if scope not in legal_scopes]
    if illegal:
        return f"scope_illegal: request {rid} ({verb}) drives {illegal}, not legal under the current action mask"
    after = request.after
    if after is not None and after not in done_ids:
        return f"predecessor_not_done: request {rid} ({verb}) runs after request {after}, which is not done"
    return None


__all__ = [
    "AFTER_PREFIX",
    "ARGUMENT_NAMES",
    "ARGUMENT_SCHEMAS",
    "COMBAT_CONTRACTS",
    "COMBAT_VERBS",
    "FACE_NAMES",
    "MODE_INTERRUPT",
    "MODE_PARALLEL",
    "REJECTION_CODES",
    "SCOPES",
    "STATUS_KINDS",
    "ActionRequest",
    "ArgumentSchema",
    "RequestArgs",
    "RequestStatus",
    "rejection_code",
    "validate",
]
