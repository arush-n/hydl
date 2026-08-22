"""Parameterised goal types: change a parameter, get a different goal.

A :class:`Goal` is a named, parameterised objective that carries its own
success criterion, the action heads it implies, and the parameters it was built
from. Tasks declare goals instead of hand-writing criteria, so "survive 400
ticks" and "survive 900 ticks" are two goals from one factory rather than two
copies of the same array code.

    from arena.tasks.framework.goals import survive_for, defeat_target, all_goals

    quick   = survive_for(ticks=200)
    long    = survive_for(ticks=900)          # different goal, same factory
    flawless = all_goals(defeat_target(), keep_health_above(0.5))

Every factory returns a Goal whose ``label`` encodes its parameters, so two
goals of the same kind with different parameters never collide in a curriculum
or a run report.

Honesty about what is observable
--------------------------------
Goals here are built only from columns that exist in the current schema.
Several carry explicit caveats rather than pretending:

* :func:`collect_item` -- inventory quantity is published as
  ``quantity_log2_fraction``, a log2-encoded fraction. It is **monotonic** in
  the true count, so "more than" thresholds are sound, but the parameter is a
  fraction in [0, 1], not a raw item count. Naming one is not supported,
  because inventing a decode would be a fabricated number.
* :func:`changed_terrain_at_interaction` -- there is no "block placed" counter
  in the observation. This detects that traversability at the interaction
  target moved, which a placement or a break both produce. It proves *the world
  changed there*, not *which block is now there*.
* Target-health goals require ``target_f32.visible``. Hidden target fields are
  zero-filled, so an unavailable sample is excluded and cannot fabricate a
  defeat. A kill that becomes hidden before zero health is observed therefore
  remains unscored until exact episode-outcome evidence joins goal evaluation.
* :func:`defeat_nearby_entity` is unavailable because a zero-filled entity
  slot can mean occlusion or slot reuse, neither of which proves death.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax.numpy as np

from arena.jax_contract import GROUP_FEATURES, READABLE_GROUPS

from arena.tasks.framework.criteria import (
    Criterion,
    above,
    all_of,
    any_of,
    at_end,
    col,
    ever_below,
    maximum,
    mean,
    mark_required_groups,
    minimum,
    span,
)


GOAL_SCORING_SCHEMA = "arena_goal_scoring_v2"
_TARGET_HEALTH_AVAILABILITY = ("target_f32.visible",)


class UnavailableSurface(NotImplementedError):
    """The goal is well-formed but the observation it scores against is absent.

    Raised at construction, not at scoring: a Goal that cannot be scored must
    never reach a rollout, because the failure would land after the compute was
    already spent.
    """


@dataclass(frozen=True, slots=True)
class Goal:
    """One parameterised objective."""

    kind: str
    label: str
    description: str
    heads_implied: tuple[str, ...]
    criterion: Criterion
    parameters: Mapping[str, Any] = field(default_factory=dict)
    #: Shortest rollout this goal can be evaluated on. Goals with a time
    #: window refuse a shorter one rather than silently passing, so the
    #: conformance gate needs to know how long a probe to build.
    minimum_ticks: int = 1
    #: True when the criterion reads terrain, traversal, interaction or hazard
    #: columns. Both built-in scenes declare ``geometry: "unavailable"``, so on
    #: them those columns are structurally zero and such a goal scores against
    #: nothing. ``Task.build_scene`` warns rather than letting that pass
    #: silently. See GAP-19.
    requires_world: bool = False
    #: Named evidence gates whose unavailable rows fail closed during scoring.
    availability_requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.availability_requirements)) != len(
            self.availability_requirements
        ):
            raise ValueError("goal availability requirements must be unique")
        for requirement in self.availability_requirements:
            group, separator, feature = requirement.partition(".")
            if not separator or feature not in GROUP_FEATURES.get(group, ()):
                raise ValueError(
                    f"unknown goal availability requirement {requirement!r}"
                )

    def __call__(self, record: Mapping[str, Any]) -> np.ndarray:
        """Goals are criteria, so a Task can take one directly."""

        return self.criterion(record)

    @property
    def _arena_required_groups(self) -> frozenset[str] | None:
        return getattr(self.criterion, "_arena_required_groups", None)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "heads_implied": list(self.heads_implied),
            "parameters": dict(self.parameters),
            "minimum_ticks": self.minimum_ticks,
            "scoring": {
                "schema": GOAL_SCORING_SCHEMA,
                "availability_requirements": list(self.availability_requirements),
                "unavailable_evidence": "false",
            },
        }


# -- survival ------------------------------------------------------------------


def stay_alive() -> Goal:
    """Be alive when the episode ends."""

    return Goal(
        kind="stay_alive",
        label="stay_alive",
        description="Be alive at the end of the episode.",
        heads_implied=("locomotion_gait_compass",),
        criterion=above(at_end(col("self_f32", "alive")), 0.5),
    )


def survive_for(*, ticks: int) -> Goal:
    """Be alive continuously for at least ``ticks`` steps.

    Checks the first ``ticks`` recorded steps, so the rollout must be at least
    that long; a shorter rollout makes this trivially true and the task's own
    horizon is the guard.
    """

    if ticks < 1:
        raise ValueError("survive_for(ticks=) must be positive")

    def criterion(record: Mapping[str, Any]) -> np.ndarray:
        alive = np.asarray(col("self_f32", "alive")(record))
        if alive.shape[0] < ticks:
            raise ValueError(
                f"survive_for(ticks={ticks}) needs at least {ticks} recorded "
                f"steps; the rollout has {alive.shape[0]}"
            )
        return (alive[:ticks] > 0.5).all(axis=0)

    criterion.__name__ = f"alive_for_first_{ticks}_ticks"
    mark_required_groups(criterion, frozenset({"self_f32"}))
    return Goal(
        kind="survive_for",
        label=f"survive_for[{ticks}]",
        description=f"Stay alive for the first {ticks} ticks.",
        heads_implied=("locomotion_gait_compass",),
        criterion=criterion,
        parameters={"ticks": ticks},
        minimum_ticks=ticks,
    )


def keep_health_above(fraction: float) -> Goal:
    """Never let health drop below a fraction of maximum."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("keep_health_above(fraction) must be in [0, 1]")
    return Goal(
        kind="keep_health_above",
        label=f"keep_health_above[{fraction}]",
        description=f"Never drop below {fraction:.0%} health.",
        heads_implied=("guard_off_on", "locomotion_gait_compass"),
        criterion=above(minimum(col("self_f32", "health_fraction")), fraction),
        parameters={"fraction": fraction},
    )


# -- defeating things ----------------------------------------------------------


def defeat_target() -> Goal:
    """Observe the designated opponent at zero health."""

    return Goal(
        kind="defeat_target",
        label="defeat_target",
        description="Observe the opponent at zero health.",
        heads_implied=("ability_none_plus_slots", "base_action"),
        criterion=ever_below(
            col("target_f32", "health_fraction"),
            1e-6,
            available=col("target_f32", "visible"),
        ),
        availability_requirements=_TARGET_HEALTH_AVAILABILITY,
    )


def reduce_target_health_below(fraction: float) -> Goal:
    """Damage the designated opponent past a declared health threshold."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("reduce_target_health_below(fraction) must be in [0, 1]")
    return Goal(
        kind="reduce_target_health_below",
        label=f"reduce_target_health_below[{fraction}]",
        description=f"Observe the opponent below {fraction:.0%} health.",
        heads_implied=("ability_none_plus_slots", "base_action"),
        criterion=ever_below(
            col("target_f32", "health_fraction"),
            fraction,
            available=col("target_f32", "visible"),
        ),
        parameters={"fraction": fraction},
        availability_requirements=_TARGET_HEALTH_AVAILABILITY,
    )


def defeat_within(*, ticks: int) -> Goal:
    """Kill the opponent inside a time budget -- a pace goal, not just a win."""

    if ticks < 1:
        raise ValueError("defeat_within(ticks=) must be positive")

    criterion = ever_below(
        col("target_f32", "health_fraction"),
        1e-6,
        available=col("target_f32", "visible"),
        ticks=ticks,
        inclusive=True,
    )
    return Goal(
        kind="defeat_within",
        label=f"defeat_within[{ticks}]",
        description=f"Observe the opponent at zero health within {ticks} ticks.",
        heads_implied=("ability_none_plus_slots", "base_action"),
        criterion=criterion,
        parameters={"ticks": ticks},
        minimum_ticks=ticks,
        availability_requirements=_TARGET_HEALTH_AVAILABILITY,
    )


def defeat_nearby_entity(*, hostile_only: bool = True) -> Goal:
    """Refuse an unsound slot-disappearance proxy for entity death."""

    raise UnavailableSurface(
        "defeat_nearby_entity cannot distinguish death from occlusion or slot "
        "reuse. It needs stable entity identity plus exact death evidence; "
        "zero-filled entity slots are not proof of a kill."
    )


# -- gathering -----------------------------------------------------------------

RESOURCE_NAMES = tuple(GROUP_FEATURES["resource_f32"])


def gather_resource(*, resource: str, amount: float) -> Goal:
    """Reach at least ``amount`` of a published resource.

    Resources are normalised fractions, so ``amount`` is a fraction in [0, 1].
    """

    if resource not in RESOURCE_NAMES:
        raise ValueError(
            f"unknown resource {resource!r}; have: {', '.join(RESOURCE_NAMES)}"
        )
    if not 0.0 <= amount <= 1.0:
        raise ValueError("gather_resource(amount=) is a fraction in [0, 1]")
    return Goal(
        kind="gather_resource",
        label=f"gather_resource[{resource}>={amount}]",
        description=f"Reach at least {amount:.0%} {resource}.",
        heads_implied=("use_off_on", "locomotion_gait_compass"),
        criterion=above(maximum(col("resource_f32", resource)), amount),
        parameters={"resource": resource, "amount": amount},
    )


def spend_no_more_than(*, resource: str, amount: float) -> Goal:
    """Finish with at least ``amount`` of a resource left -- an economy goal."""

    if resource not in RESOURCE_NAMES:
        raise ValueError(
            f"unknown resource {resource!r}; have: {', '.join(RESOURCE_NAMES)}"
        )
    if not 0.0 <= amount <= 1.0:
        raise ValueError("spend_no_more_than(amount=) is a fraction in [0, 1]")
    return Goal(
        kind="spend_no_more_than",
        label=f"spend_no_more_than[{resource}>={amount}]",
        description=f"End the episode with at least {amount:.0%} {resource}.",
        heads_implied=("ability_none_plus_slots",),
        criterion=above(at_end(col("resource_f32", resource)), amount),
        parameters={"resource": resource, "amount": amount},
    )


def collect_item(*, item_id_low: float, minimum_quantity_fraction: float) -> Goal:
    """Hold an item, in at least a given (encoded) quantity.

    ``item_id_low`` matches ``inventory_token_f32.item_id_low_16``. Quantity is
    ``quantity_log2_fraction``: log2-encoded and **monotonic** in the true
    count, so a threshold means "at least this many" without naming the count.
    An exact item count is deliberately not offered -- decoding it here would
    be an invented number.

    **Not constructible today.** ``inventory_token_f32`` is named in
    ``GROUP_FEATURES`` but nothing produces it: it is absent from the
    observation tree, no Arena reader pulls it, and unlike ``light_token_f32``
    there is no state-side accessor either. Until an inventory surface exists,
    this raises here rather than returning a Goal that would pass the gate and
    then die partway through a paid rollout.
    """

    if not 0.0 <= minimum_quantity_fraction <= 1.0:
        raise ValueError("minimum_quantity_fraction is a fraction in [0, 1]")

    if "inventory_token_f32" not in READABLE_GROUPS:
        raise UnavailableSurface(
            "collect_item needs 'inventory_token_f32', which no recorder can "
            "produce: it is not on the observation tree and no Arena reader "
            "pulls it. See GAP-17. Once an inventory surface lands and the "
            "group joins READABLE_GROUPS, this factory works unchanged."
        )

    features = GROUP_FEATURES["inventory_token_f32"]
    id_index = features.index("item_id_low_16")
    qty_index = features.index("quantity_log2_fraction")
    active_index = features.index("active")

    def criterion(record: Mapping[str, Any]) -> np.ndarray:
        tokens = np.asarray(record["inventory_token_f32"])
        matches = (
            (np.abs(tokens[..., id_index] - item_id_low) < 1e-6)
            & (tokens[..., active_index] > 0.5)
            & (tokens[..., qty_index] >= minimum_quantity_fraction)
        )
        # (ticks, batch, slots) -> any slot, at any tick
        while matches.ndim > 2:
            matches = matches.any(axis=-1)
        return matches.any(axis=0)

    criterion.__name__ = f"held_item_{item_id_low}_qty>={minimum_quantity_fraction}"
    mark_required_groups(criterion, frozenset({"inventory_token_f32"}))
    return Goal(
        kind="collect_item",
        label=f"collect_item[{item_id_low}>={minimum_quantity_fraction}]",
        description=(
            f"Hold item {item_id_low} at encoded quantity "
            f">= {minimum_quantity_fraction}."
        ),
        heads_implied=("use_off_on", "locomotion_gait_compass"),
        criterion=criterion,
        parameters={
            "item_id_low": item_id_low,
            "minimum_quantity_fraction": minimum_quantity_fraction,
        },
    )


# -- world interaction ---------------------------------------------------------


def reach_interaction(*, reach: float = 0.999) -> Goal:
    """Bring the interaction target inside reach."""

    if not 0.0 <= reach <= 1.0:
        raise ValueError("reach_interaction(reach=) is a fraction in [0, 1]")

    return Goal(
        kind="reach_interaction",
        label=f"reach_interaction[{reach}]",
        description="Bring the interaction target within reach.",
        heads_implied=("locomotion_gait_compass", "use_off_on"),
        criterion=above(maximum(col("interaction_f32", "reach_fraction")), reach),
        parameters={"reach": reach},
        requires_world=True,
    )


def changed_terrain_at_interaction(*, minimum_change: float = 1e-3) -> Goal:
    """The world changed where the agent was working.

    Detects movement in ``terrain_f32.traversability``. A placement and a break
    both produce this, and no observation column names *which* block is there,
    so this proves the world changed at that spot and nothing more. Verify the
    specific block against the bridge before treating it as a build goal.
    """

    if not minimum_change > 0:
        raise ValueError(
            "changed_terrain_at_interaction minimum_change must be positive"
        )
    return Goal(
        kind="changed_terrain_at_interaction",
        label=f"changed_terrain_at_interaction[{minimum_change}]",
        description="Change terrain traversability at the interaction target.",
        heads_implied=(
            "block_none_plus_candidates",
            "block_primary_secondary_trigger",
            "use_off_on",
        ),
        criterion=above(span(col("terrain_f32", "traversability")), minimum_change),
        parameters={"minimum_change": minimum_change},
        requires_world=True,
    )


MOVEMENT_STATES = tuple(GROUP_FEATURES["movement_state_f32"])


def hold_movement_state(*, state: str, fraction: float) -> Goal:
    """Spend at least ``fraction`` of the episode in a movement state."""

    if state not in MOVEMENT_STATES:
        raise ValueError(
            f"unknown movement state {state!r}; have: {', '.join(MOVEMENT_STATES)}"
        )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("hold_movement_state(fraction=) is a fraction in [0, 1]")
    return Goal(
        kind="hold_movement_state",
        label=f"hold_movement_state[{state}>={fraction}]",
        description=f"Spend at least {fraction:.0%} of the episode {state}.",
        heads_implied=("locomotion_gait_compass", "jump_off_on"),
        criterion=above(mean(col("movement_state_f32", state)), fraction),
        parameters={"state": state, "fraction": fraction},
    )


def keep_target_visible(*, fraction: float = 0.9) -> Goal:
    """Keep the opponent in view -- an aim goal no combat objective rewards."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("keep_target_visible(fraction=) is a fraction in [0, 1]")

    return Goal(
        kind="keep_target_visible",
        label=f"keep_target_visible[{fraction}]",
        description=f"Keep the opponent visible for {fraction:.0%} of the episode.",
        # The body steer belongs here as much as the camera does: the head is
        # clamped to the model's authored window around the body (+/-45 for
        # every shipped weapon), so a target that walks past that bearing cannot
        # be kept in view by the camera alone -- the chest has to come round.
        heads_implied=(
            "yaw_delta_bins",
            "body_yaw_delta_bins",
            "pitch_delta_bins",
        ),
        criterion=above(mean(col("target_f32", "visible")), fraction),
        parameters={"fraction": fraction},
    )


def avoid_hazards(*, distance: float) -> Goal:
    """Never come closer than ``distance`` to a hazard volume."""

    if not distance >= 0:
        raise ValueError("avoid_hazards(distance=) must be non-negative")

    return Goal(
        kind="avoid_hazards",
        label=f"avoid_hazards[{distance}]",
        description=f"Never approach within {distance} of a hazard.",
        heads_implied=("locomotion_gait_compass", "jump_off_on"),
        criterion=above(minimum(col("hazard_f32", "distance")), distance),
        parameters={"distance": distance},
        requires_world=True,
    )


# -- composition ---------------------------------------------------------------


def _combine(kind: str, joiner, goals: Sequence[Goal], separator: str) -> Goal:
    if not goals:
        raise ValueError(f"{kind} needs at least one goal")
    for goal in goals:
        if not isinstance(goal, Goal):
            raise TypeError(f"{kind} takes Goal instances, got {type(goal).__name__}")
    heads: list[str] = []
    for goal in goals:
        for head in goal.heads_implied:
            if head not in heads:
                heads.append(head)
    return Goal(
        kind=kind,
        label=f"{kind}[" + separator.join(g.label for g in goals) + "]",
        description=separator.join(g.description for g in goals),
        heads_implied=tuple(heads),
        criterion=joiner(*(g.criterion for g in goals)),
        parameters={"goals": [g.describe() for g in goals]},
        minimum_ticks=max(g.minimum_ticks for g in goals),
        availability_requirements=tuple(
            dict.fromkeys(
                requirement
                for goal in goals
                for requirement in goal.availability_requirements
            )
        ),
    )


def all_goals(*goals: Goal) -> Goal:
    """Every goal must be satisfied."""

    return _combine("all_goals", all_of, goals, " AND ")


def any_goals(*goals: Goal) -> Goal:
    """Any one goal suffices."""

    return _combine("any_goals", any_of, goals, " OR ")


__all__ = [
    "MOVEMENT_STATES",
    "RESOURCE_NAMES",
    "GOAL_SCORING_SCHEMA",
    "Goal",
    "UnavailableSurface",
    "all_goals",
    "any_goals",
    "avoid_hazards",
    "changed_terrain_at_interaction",
    "collect_item",
    "defeat_nearby_entity",
    "defeat_target",
    "defeat_within",
    "gather_resource",
    "hold_movement_state",
    "keep_health_above",
    "keep_target_visible",
    "reach_interaction",
    "reduce_target_health_below",
    "spend_no_more_than",
    "stay_alive",
    "survive_for",
]
