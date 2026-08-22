"""Executable, content-addressed target programs for JAX skill lessons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from numbers import Integral
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    arsenal_policy_contract_sha256,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_GAIT_RUN,
    ARSENAL_GAIT_SPRINT,
    ARSENAL_GAIT_WALK,
    ARSENAL_POLICY_COMPASS_DIRECTIONS,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH,
    SKILL_IDLE,
    SKILL_RETREAT,
    SKILL_STRAFE_LEFT,
    SKILL_STRAFE_RIGHT,
)


FACTORED_ACTION_SCOPE_SCHEMA = "arena-factored-action-scope-v1"
TARGET_MOTION_PROGRAM_SCHEMA = "arena-target-motion-program-v3"
_HEAD_OFFSETS = tuple(
    sum(ARSENAL_POLICY_ACTION_HEAD_SIZES[:index])
    for index in range(len(ARSENAL_POLICY_ACTION_HEAD_SIZES))
)
_BASE_ACTION = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("base_action")
# Movement and dodge share one head; a non-idle choice still means motion.
_WORLD_MOVE = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("locomotion_gait_compass")
_JUMP = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("jump_off_on")
_NEUTRAL_FACTOR_CHOICES = tuple(
    # Signed look heads centre; everything else neutralises at zero. The body
    # steer is a signed look head too -- bin 0 is a hard turn, not a no-op -- so
    # leaving it out here made `neutral_allowed` false for a scope that really
    # did contain the physical neutral.
    size // 2
    if name in {"yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"}
    else 0
    for name, size in zip(
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        strict=True,
    )
)


def _hash(value: Any) -> str:
    return (
        hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    canonical = value.upper()
    if len(canonical) != 64 or any(
        character not in "0123456789ABCDEF" for character in canonical
    ):
        raise ValueError(f"{name} must be a SHA-256 string")
    return canonical


@dataclass(frozen=True, slots=True)
class FactoredActionScope:
    """Executable allowed choices for every published policy-action head."""

    allowed_choices: tuple[tuple[int, ...], ...]
    schema: str = FACTORED_ACTION_SCOPE_SCHEMA

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.allowed_choices)
        if len(rows) != len(ARSENAL_POLICY_ACTION_HEAD_SIZES):
            raise ValueError(
                "factored action scope must declare every published action head"
            )
        normalized = []
        for name, size, row in zip(
            ARSENAL_POLICY_ACTION_HEAD_NAMES,
            ARSENAL_POLICY_ACTION_HEAD_SIZES,
            rows,
            strict=True,
        ):
            if not row:
                raise ValueError(f"factored action scope head {name} cannot be empty")
            if any(
                isinstance(choice, bool) or not isinstance(choice, Integral)
                for choice in row
            ):
                raise TypeError(
                    f"factored action scope head {name} choices must be integers"
                )
            canonical = tuple(sorted(int(choice) for choice in row))
            if len(canonical) != len(set(canonical)):
                raise ValueError(
                    f"factored action scope head {name} choices must be unique"
                )
            if canonical[0] < 0 or canonical[-1] >= size:
                raise ValueError(
                    f"factored action scope head {name} choice is outside [0,{size})"
                )
            normalized.append(canonical)
        object.__setattr__(self, "allowed_choices", tuple(normalized))
        if self.schema != FACTORED_ACTION_SCOPE_SCHEMA:
            raise ValueError("factored action scope schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action_contract_sha256": arsenal_policy_contract_sha256(),
            "heads": tuple(
                {
                    "name": name,
                    "size": size,
                    "allowed_choices": choices,
                }
                for name, size, choices in zip(
                    ARSENAL_POLICY_ACTION_HEAD_NAMES,
                    ARSENAL_POLICY_ACTION_HEAD_SIZES,
                    self.allowed_choices,
                    strict=True,
                )
            ),
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())

    @property
    def neutral_allowed(self) -> bool:
        """Whether the exact physical no-op is inside this scope."""

        return all(
            neutral in choices
            for neutral, choices in zip(
                _NEUTRAL_FACTOR_CHOICES,
                self.allowed_choices,
                strict=True,
            )
        )

    def restrict(self, action_mask: jax.Array) -> jax.Array:
        """Intersect a runtime mask with this content-addressed head scope."""

        mask = jnp.asarray(action_mask)
        expected_width = sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)
        if mask.dtype != jnp.dtype(jnp.bool_):
            raise TypeError("factored action scope mask must have dtype bool")
        if mask.ndim != 2 or mask.shape[1] != expected_width:
            raise ValueError(
                f"factored action scope mask must have shape [B,{expected_width}]"
            )
        parts = []
        for offset, size, choices in zip(
            _HEAD_OFFSETS,
            ARSENAL_POLICY_ACTION_HEAD_SIZES,
            self.allowed_choices,
            strict=True,
        ):
            allowed = (
                jnp.zeros((size,), dtype=jnp.bool_)
                .at[jnp.asarray(choices, dtype=jnp.int32)]
                .set(True)
            )
            parts.append(mask[:, offset : offset + size] & allowed[None, :])
        return jnp.concatenate(parts, axis=-1)


class TargetMode(IntEnum):
    """One actor-legal target controller selected per lane."""

    STATIONARY = 0
    STRAFE = 1
    APPROACH = 2
    FLEE = 3
    FROZEN_POLICY = 4
    APPROACH_THEN_HOLD = 5
    FLEE_WEAVE = 6
    #: Flees on the locomotion head rather than the skill row, so its speed is
    #: a real gait. `FLEE_WEAVE` writes `base_action=SKILL_RETREAT`, which was
    #: measured to travel at exactly RUN speed and offers no slower option.
    FLEE_ADAPTIVE = 7
    #: Holds a standoff band instead of running away: backs off when the pursuer
    #: is inside `standoff_inner_distance`, closes when it is beyond
    #: `standoff_outer_distance`, and circles the pursuer in between. Rides the
    #: same locomotion head as `FLEE_ADAPTIVE` and differs only in the heading.
    FLEE_STANDOFF = 8


@dataclass(frozen=True, slots=True)
class TargetMotionProgram:
    """Lane-stratified motion that never mutates simulator state directly."""

    lane_modes: tuple[TargetMode, ...]
    strafe_switch_ticks: int = 32
    flee_retreat_ticks: int = 24
    flee_strafe_ticks: int = 8
    flee_jump_period_ticks: int = 48
    #: Share of each adaptive duty cycle spent at (walk, run, sprint).
    #: Measured speeds are 0.063 / 0.194 / 0.238 blocks per tick, so the mix is
    #: the whole difference between a joggable evader and one that cannot be
    #: caught. The default averages 0.194 -- exactly the speed `FLEE_WEAVE`
    #: travels at -- so the two evaders are like-for-like on pace and differ
    #: only in cadence and edge behaviour.
    adaptive_gait_mix: tuple[float, float, float] = (0.1, 0.6, 0.3)
    #: Length of one walk/sprint duty cycle. Lanes are offset within it so the
    #: whole batch does not break into a sprint on the same tick.
    adaptive_gait_period_ticks: int = 24
    #: Arena half-width the adaptive evader turns away from. `None` lets it
    #: flee in a straight line, which on a bounded arena ends in a corner.
    #: Carried here rather than passed separately so the caller that builds the
    #: mask transform does not need a second geometry argument.
    adaptive_arena_radius: float | None = None
    #: Standoff band the `FLEE_STANDOFF` evader holds, in blocks. Inside the
    #: inner edge it backs off, beyond the outer edge it closes, and between
    #: them it strafes -- so the pursuer has to track a moving bearing rather
    #: than run down a straight line.
    standoff_inner_distance: float = 6.0
    standoff_outer_distance: float = 14.0
    #: How long the evader circles one way before reversing. Lanes are offset
    #: within it, as with the gait cycle, so the batch does not turn together.
    standoff_strafe_period_ticks: int = 40
    frozen_policy_sha256: str | None = None
    frozen_factor_scope: FactoredActionScope | None = None
    approach_hold_distance: float | None = None
    schema: str = TARGET_MOTION_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        modes = tuple(self.lane_modes)
        if not modes or any(not isinstance(value, TargetMode) for value in modes):
            raise ValueError("target lane modes must be non-empty TargetMode values")
        object.__setattr__(self, "lane_modes", modes)
        if isinstance(self.strafe_switch_ticks, bool) or not isinstance(
            self.strafe_switch_ticks, Integral
        ):
            raise TypeError("strafe switch ticks must be an integer")
        switch = int(self.strafe_switch_ticks)
        if switch < 1:
            raise ValueError("strafe switch ticks must be positive")
        object.__setattr__(self, "strafe_switch_ticks", switch)
        mix = tuple(self.adaptive_gait_mix)
        if len(mix) != 3:
            raise ValueError("adaptive gait mix must be (walk, run, sprint)")
        if any(
            isinstance(share, bool) or not isinstance(share, (int, float))
            for share in mix
        ):
            raise TypeError("adaptive gait mix shares must be numbers")
        mix = tuple(float(share) for share in mix)
        if any(not math.isfinite(share) or share < 0.0 for share in mix):
            raise ValueError("adaptive gait mix shares must be finite and non-negative")
        # Refused rather than normalised: a mix that does not sum to one is a
        # typo, and quietly rescaling it would train against a cadence nobody
        # asked for while the manifest recorded the numbers that were typed.
        if abs(sum(mix) - 1.0) > 1.0e-6:
            raise ValueError(f"adaptive gait mix must sum to 1, got {sum(mix):g}")
        object.__setattr__(self, "adaptive_gait_mix", mix)
        if self.adaptive_arena_radius is not None:
            radius = float(self.adaptive_arena_radius)
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("adaptive arena radius must be finite and positive")
            object.__setattr__(self, "adaptive_arena_radius", radius)
        band = []
        for name in ("standoff_inner_distance", "standoff_outer_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
            band.append(value)
        # Ordered strictly: an inverted or empty band would leave the evader with
        # no strafe region at all, which is the straight-line flee this mode
        # exists to replace.
        if band[0] >= band[1]:
            raise ValueError(
                f"standoff inner distance must be below outer, got "
                f"{band[0]:g} >= {band[1]:g}"
            )
        for name in (
            "flee_retreat_ticks",
            "flee_strafe_ticks",
            "flee_jump_period_ticks",
            "adaptive_gait_period_ticks",
            "standoff_strafe_period_ticks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        uses_frozen = TargetMode.FROZEN_POLICY in modes
        if uses_frozen != (self.frozen_policy_sha256 is not None) or uses_frozen != (
            self.frozen_factor_scope is not None
        ):
            raise ValueError(
                "frozen-policy mode, checkpoint, and executable factor scope must be "
                "declared together"
            )
        if self.frozen_policy_sha256 is not None:
            object.__setattr__(
                self,
                "frozen_policy_sha256",
                _sha256(self.frozen_policy_sha256, "frozen policy"),
            )
            if not isinstance(self.frozen_factor_scope, FactoredActionScope):
                raise TypeError("frozen factor scope must be a FactoredActionScope")
            if not self.frozen_factor_scope.neutral_allowed:
                raise ValueError(
                    "frozen factor scope must contain the physical neutral"
                )
        uses_approach_hold = TargetMode.APPROACH_THEN_HOLD in modes
        if uses_approach_hold != (self.approach_hold_distance is not None):
            raise ValueError(
                "approach-then-hold mode and hold distance must be declared together"
            )
        if self.approach_hold_distance is not None:
            distance = float(self.approach_hold_distance)
            if not math.isfinite(distance) or distance <= 0.0:
                raise ValueError("approach hold distance must be finite and positive")
            object.__setattr__(self, "approach_hold_distance", distance)
        if self.schema != TARGET_MOTION_PROGRAM_SCHEMA:
            raise ValueError("target motion program schema is not current")

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "schema": self.schema,
            "lane_modes": tuple(value.name.lower() for value in self.lane_modes),
            "strafe_switch_ticks": self.strafe_switch_ticks,
            "flee_retreat_ticks": self.flee_retreat_ticks,
            "flee_strafe_ticks": self.flee_strafe_ticks,
            "flee_jump_period_ticks": self.flee_jump_period_ticks,
            "frozen_policy_sha256": self.frozen_policy_sha256,
            "frozen_factor_scope": (
                None
                if self.frozen_factor_scope is None
                else self.frozen_factor_scope.manifest()
            ),
            "action_contract_sha256": arsenal_policy_contract_sha256(),
            "action_heads": tuple(
                zip(
                    ARSENAL_POLICY_ACTION_HEAD_NAMES,
                    ARSENAL_POLICY_ACTION_HEAD_SIZES,
                    strict=True,
                )
            ),
            "motion_rows": {
                "stationary": SKILL_IDLE,
                "approach": SKILL_APPROACH,
                "flee": SKILL_RETREAT,
                "strafe_left": SKILL_STRAFE_LEFT,
                "strafe_right": SKILL_STRAFE_RIGHT,
            },
            "lane_schedule": (
                "lane_index_modulo_ordered_declared_modes_repetition_weights_lanes"
            ),
            "strafe_schedule": (
                "left_right_by_floor_tick_over_switch_plus_lane_parity"
            ),
            "flee_weave_schedule": (
                "retreat_for_declared_ticks_then_alternate_left_right_strafe_"
                "for_declared_ticks_with_grounded_periodic_jump_requests"
            ),
            "invalid_requested_row": "issue_neutral_and_mark_invalid",
            "state_mutation": "forbidden_actions_decode_through_normal_physics",
        }
        if TargetMode.FLEE_ADAPTIVE in self.lane_modes:
            # Declared only when the mode is used, so adding it does not move
            # the contract hash of any program authored before it existed.
            manifest["flee_adaptive"] = {
                "gait_mix_walk_run_sprint": list(self.adaptive_gait_mix),
                "gait_period_ticks": self.adaptive_gait_period_ticks,
                "gait_tick_split_walk_run": list(
                    _gait_tick_split(
                        self.adaptive_gait_period_ticks, self.adaptive_gait_mix
                    )
                ),
                "gait_schedule": (
                    "walk_then_run_then_sprint_by_rounded_mix_share_"
                    "offset_by_lane"
                ),
                "direction_source": (
                    "caller_supplied_body_relative_compass_choice_away_from_"
                    "pursuer"
                ),
                "speed_channel": "locomotion_gait_compass",
                "base_action": SKILL_IDLE,
            }
        if TargetMode.FLEE_STANDOFF in self.lane_modes:
            # Same conditional-declaration rule as the adaptive block above.
            manifest["flee_standoff"] = {
                "gait_mix_walk_run_sprint": list(self.adaptive_gait_mix),
                "gait_period_ticks": self.adaptive_gait_period_ticks,
                "gait_tick_split_walk_run": list(
                    _gait_tick_split(
                        self.adaptive_gait_period_ticks, self.adaptive_gait_mix
                    )
                ),
                "standoff_band_blocks": [
                    self.standoff_inner_distance,
                    self.standoff_outer_distance,
                ],
                "strafe_period_ticks": self.standoff_strafe_period_ticks,
                "direction_source": (
                    "caller_supplied_body_relative_compass_choice_away_inside_"
                    "band_toward_outside_band_tangent_within_band"
                ),
                "strafe_schedule": (
                    "orbit_sign_by_floor_tick_over_period_plus_lane_parity"
                ),
                "speed_channel": "locomotion_gait_compass",
                "base_action": SKILL_IDLE,
            }
        if TargetMode.APPROACH_THEN_HOLD in self.lane_modes:
            manifest["approach_then_hold"] = {
                "horizontal_distance_blocks": self.approach_hold_distance,
                "distance_source": (
                    "exact_horizontal_actor_distance_from_pre_step_arena_state"
                ),
                "outside_action": SKILL_APPROACH,
                "inside_action": SKILL_IDLE,
                "state_mutation": False,
            }
        return manifest

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


class TargetMotionOutput(NamedTuple):
    """Requested and fail-closed issued target factors for one tick."""

    factors: jax.Array
    requested_factors: jax.Array
    mode: jax.Array
    requested_legal: jax.Array
    issued_legal: jax.Array
    motion_requested: jax.Array
    motion_issued: jax.Array


def _gait_tick_split(period: int, mix: tuple[float, float, float]) -> tuple[int, int]:
    """Ticks of walk and run inside one duty cycle; sprint takes the remainder.

    Sprint is the remainder rather than a third rounding so the three always
    add back to `period` exactly. A mix that does not divide the period evenly
    is still honoured to the nearest tick -- (0.1, 0.6, 0.3) over 20 ticks is
    exact at 2/12/6, and over 24 lands at 2/14/8.
    """

    walk = int(round(period * mix[0]))
    run = int(round(period * mix[1]))
    # Clamp so a period too short for the mix cannot produce a negative
    # sprint share and silently invert the cadence.
    walk = max(0, min(walk, period))
    run = max(0, min(run, period - walk))
    return walk, run


def _unit_xz(vector: jax.Array) -> jax.Array:
    length = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return jnp.where(length > 1.0e-6, vector / jnp.maximum(length, 1.0e-6), 0.0)


def _pull_from_edge(
    heading: jax.Array,
    target_xz: jax.Array,
    *,
    arena_radius: float | None,
    boundary_fraction: float,
    centre_xz: jax.Array | None,
) -> jax.Array:
    """Blend a heading back toward the centre as the target nears the edge."""

    if arena_radius is None:
        return heading
    centre = (
        jnp.zeros_like(target_xz)
        if centre_xz is None
        else jnp.asarray(centre_xz, dtype=jnp.float32)
    )
    offset = target_xz - centre
    radius = jnp.linalg.norm(offset, axis=-1, keepdims=True)
    edge = jnp.float32(arena_radius) * jnp.float32(boundary_fraction)
    span = jnp.maximum(jnp.float32(arena_radius) - edge, 1.0e-6)
    # Ramps 0 -> 1 across the outer band, so the turn inward is gradual
    # rather than a discontinuity the pursuer could exploit.
    pull = jnp.clip((radius - edge) / span, 0.0, 1.0)
    return _unit_xz(heading * (1.0 - pull) + _unit_xz(-offset) * pull)


def _body_relative_compass(
    heading: jax.Array,
    body_yaw_degrees: jax.Array,
) -> jax.Array:
    """Quantise a world-space XZ heading to a body-relative choice, 0 or 1..8."""

    radians = jnp.deg2rad(body_yaw_degrees)
    forward = jnp.stack((-jnp.sin(radians), -jnp.cos(radians)), axis=-1)
    right = jnp.stack((-jnp.cos(radians), jnp.sin(radians)), axis=-1)
    angle = jnp.arctan2(
        jnp.sum(heading * right, axis=-1), jnp.sum(heading * forward, axis=-1)
    )
    step = 2.0 * math.pi / ARSENAL_POLICY_COMPASS_DIRECTIONS
    index = jnp.round(angle / step).astype(jnp.int32) % jnp.int32(
        ARSENAL_POLICY_COMPASS_DIRECTIONS
    )
    stationary = jnp.all(jnp.abs(heading) < 1.0e-6, axis=-1)
    return jnp.where(stationary, jnp.int32(0), index + jnp.int32(1))


def standoff_compass_choice(
    learner_xz: jax.Array,
    target_xz: jax.Array,
    body_yaw_degrees: jax.Array,
    strafe_sign: jax.Array,
    *,
    inner_distance: float,
    outer_distance: float,
    arena_radius: float | None = None,
    boundary_fraction: float = 0.7,
    centre_xz: jax.Array | None = None,
) -> jax.Array:
    """The body-relative compass choice that holds a standoff band, 1..8.

    Three regimes, switched on the horizontal separation:

    * inside `inner_distance` -- head directly away, as `flee_compass_choice`
    * beyond `outer_distance` -- head directly toward the pursuer
    * between them -- head perpendicular, circling the pursuer

    The tangent is what makes this different to fleeing: the bearing to the
    evader keeps changing while the range does not, so the pursuer has to track
    rather than run down a straight line. `strafe_sign` is +/-1 per lane and
    picks which way round; reversing it periodically stops a single sustained
    orbit the pursuer could simply lead.
    """

    learner = jnp.asarray(learner_xz, dtype=jnp.float32)
    target = jnp.asarray(target_xz, dtype=jnp.float32)
    yaw = jnp.asarray(body_yaw_degrees, dtype=jnp.float32)
    sign = jnp.asarray(strafe_sign, dtype=jnp.float32)
    if learner.shape != target.shape or learner.ndim != 2 or learner.shape[1] != 2:
        raise ValueError("standoff positions must both have shape [B,2]")
    if yaw.shape != learner.shape[:1]:
        raise ValueError("body yaw must have shape [B]")
    if sign.shape != learner.shape[:1]:
        raise ValueError("strafe sign must have shape [B]")
    inner = float(inner_distance)
    outer = float(outer_distance)
    if not 0.0 < inner < outer:
        raise ValueError("standoff band must satisfy 0 < inner < outer")

    delta = target - learner
    away = _unit_xz(delta)
    distance = jnp.linalg.norm(delta, axis=-1)
    # Rotating `away` by 90 degrees in XZ. Which of the two perpendiculars is
    # taken is the caller's `strafe_sign`, so the orbit direction is a lane
    # property rather than an artefact of the axis order here.
    tangent = jnp.stack((-away[:, 1], away[:, 0]), axis=-1) * sign[:, None]
    heading = jnp.where(
        (distance < jnp.float32(inner))[:, None],
        away,
        jnp.where((distance > jnp.float32(outer))[:, None], -away, tangent),
    )
    heading = _pull_from_edge(
        _unit_xz(heading),
        target,
        arena_radius=arena_radius,
        boundary_fraction=boundary_fraction,
        centre_xz=centre_xz,
    )
    return _body_relative_compass(heading, yaw)


def flee_compass_choice(
    learner_xz: jax.Array,
    target_xz: jax.Array,
    body_yaw_degrees: jax.Array,
    *,
    arena_radius: float | None = None,
    boundary_fraction: float = 0.7,
    centre_xz: jax.Array | None = None,
) -> jax.Array:
    """The body-relative compass choice that runs the target away, 1..8.

    Body forward is ``(-sin(yaw), -cos(yaw))`` in XZ -- taken from
    `_airborne_walk_motion` and `MotionControllerWalk`'s carry-through, which
    agree, and confirmed by a rollout where yaw 180 travelled +z on choice 1.
    Choice 1 is that forward and the rest step 45 degrees clockwise, so a pure
    retreat is choice 5.

    With `arena_radius`, a target approaching the edge blends its heading back
    toward the centre. Fleeing is otherwise a straight line, and a straight
    line on a bounded arena ends in a corner where the evader is trivially
    caught -- which is the degenerate 0.955 arena, not pursuit skill.
    """

    learner = jnp.asarray(learner_xz, dtype=jnp.float32)
    target = jnp.asarray(target_xz, dtype=jnp.float32)
    yaw = jnp.asarray(body_yaw_degrees, dtype=jnp.float32)
    if learner.shape != target.shape or learner.ndim != 2 or learner.shape[1] != 2:
        raise ValueError("flee positions must both have shape [B,2]")
    if yaw.shape != learner.shape[:1]:
        raise ValueError("body yaw must have shape [B]")

    away = _unit_xz(target - learner)
    away = _pull_from_edge(
        away,
        target,
        arena_radius=arena_radius,
        boundary_fraction=boundary_fraction,
        centre_xz=centre_xz,
    )
    return _body_relative_compass(away, yaw)


def _selected_factor_row_legal(
    factors: jax.Array,
    action_mask: jax.Array,
) -> jax.Array:
    selected = []
    for index, (offset, size) in enumerate(
        zip(_HEAD_OFFSETS, ARSENAL_POLICY_ACTION_HEAD_SIZES, strict=True)
    ):
        choice = factors[:, index]
        in_range = (choice >= 0) & (choice < jnp.int32(size))
        safe = jnp.clip(choice, 0, size - 1)
        selected.append(
            in_range
            & jnp.take_along_axis(
                action_mask[:, offset : offset + size],
                safe[:, None],
                axis=1,
            )[:, 0]
        )
    return jnp.all(jnp.stack(selected, axis=-1), axis=-1)


def target_motion_factors(
    program: TargetMotionProgram,
    tick: jax.Array,
    action_mask: jax.Array,
    *,
    sampled_factors: jax.Array | None = None,
    horizontal_distance: jax.Array | None = None,
    flee_direction: jax.Array | None = None,
) -> TargetMotionOutput:
    """Produce a target row using only public factors and current support.

    The function is scan/JIT safe. A requested row that is currently illegal
    becomes the exact physical neutral row and exposes ``requested_legal=False``
    so a lesson can truncate or resample instead of silently changing motion.
    """

    if not isinstance(program, TargetMotionProgram):
        raise TypeError("target motion needs a TargetMotionProgram")
    mask = jnp.asarray(action_mask)
    expected_width = sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)
    if mask.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("target action mask must have dtype bool")
    if mask.ndim != 2 or mask.shape[1] != expected_width:
        raise ValueError(f"target action mask must have shape [B,{expected_width}]")
    batch = mask.shape[0]
    tick_value = jnp.asarray(tick, dtype=jnp.int32)
    if tick_value.ndim == 0:
        tick_value = jnp.broadcast_to(tick_value, (batch,))
    elif tick_value.shape != (batch,):
        raise ValueError("target motion tick must be scalar or int32[B]")
    modes = jnp.asarray(tuple(int(value) for value in program.lane_modes))
    lanes = jnp.arange(batch, dtype=jnp.int32)
    lane_mode = modes[lanes % len(program.lane_modes)]
    uses_approach_hold = TargetMode.APPROACH_THEN_HOLD in program.lane_modes
    if uses_approach_hold:
        if horizontal_distance is None:
            raise ValueError(
                "approach-then-hold target motion needs horizontal distance"
            )
        distance = jnp.asarray(horizontal_distance, dtype=jnp.float32)
        if distance.ndim == 0:
            distance = jnp.broadcast_to(distance, (batch,))
        elif distance.shape != (batch,):
            raise ValueError("target horizontal distance must be scalar or float[B]")
    else:
        distance = jnp.zeros((batch,), dtype=jnp.float32)
    # Both modes ride the locomotion head off a caller-supplied heading and
    # differ only in how that heading is chosen, so everything below is shared.
    gait_driven_modes = (TargetMode.FLEE_ADAPTIVE, TargetMode.FLEE_STANDOFF)
    uses_adaptive = any(mode in program.lane_modes for mode in gait_driven_modes)
    if uses_adaptive:
        if flee_direction is None:
            raise ValueError("adaptive flee target motion needs a flee direction")
        away = jnp.asarray(flee_direction, dtype=jnp.int32)
        if away.ndim == 0:
            away = jnp.broadcast_to(away, (batch,))
        elif away.shape != (batch,):
            raise ValueError("flee direction must be scalar or int32[B]")
    else:
        if flee_direction is not None:
            raise ValueError("flee direction requires an adaptive flee lane")
        away = jnp.zeros((batch,), dtype=jnp.int32)
    neutral = neutral_arsenal_policy_action_factors(batch)
    alternating = (
        tick_value // jnp.int32(program.strafe_switch_ticks) + lanes
    ) & jnp.int32(1)
    strafe = jnp.where(
        alternating == 0,
        jnp.int32(SKILL_STRAFE_LEFT),
        jnp.int32(SKILL_STRAFE_RIGHT),
    )
    flee_cycle = jnp.int32(program.flee_retreat_ticks + program.flee_strafe_ticks)
    flee_phase = tick_value % flee_cycle
    flee_strafe = jnp.where(
        ((((tick_value // flee_cycle) + lanes) & jnp.int32(1)) == 0),
        jnp.int32(SKILL_STRAFE_LEFT),
        jnp.int32(SKILL_STRAFE_RIGHT),
    )
    flee_weave = jnp.where(
        flee_phase < jnp.int32(program.flee_retreat_ticks),
        jnp.int32(SKILL_RETREAT),
        flee_strafe,
    )
    base = jnp.select(
        (
            lane_mode == jnp.int32(TargetMode.STATIONARY),
            lane_mode == jnp.int32(TargetMode.STRAFE),
            lane_mode == jnp.int32(TargetMode.APPROACH),
            lane_mode == jnp.int32(TargetMode.FLEE),
            lane_mode == jnp.int32(TargetMode.APPROACH_THEN_HOLD),
            lane_mode == jnp.int32(TargetMode.FLEE_WEAVE),
            lane_mode == jnp.int32(TargetMode.FLEE_ADAPTIVE),
            lane_mode == jnp.int32(TargetMode.FLEE_STANDOFF),
        ),
        (
            jnp.int32(SKILL_IDLE),
            strafe,
            jnp.int32(SKILL_APPROACH),
            jnp.int32(SKILL_RETREAT),
            jnp.where(
                distance > jnp.float32(program.approach_hold_distance or 0.0),
                jnp.int32(SKILL_APPROACH),
                jnp.int32(SKILL_IDLE),
            ),
            flee_weave,
            # The gait head carries these lanes' motion, and it overrides the
            # skill row entirely -- a non-idle skill here would be dead weight
            # that only confuses the manifest.
            jnp.int32(SKILL_IDLE),
            jnp.int32(SKILL_IDLE),
        ),
        default=jnp.int32(SKILL_IDLE),
    )
    scripted = neutral.at[:, _BASE_ACTION].set(base)
    if uses_adaptive:
        period = jnp.int32(program.adaptive_gait_period_ticks)
        walk_ticks, run_ticks = _gait_tick_split(
            program.adaptive_gait_period_ticks, program.adaptive_gait_mix
        )
        # Offset by lane so the batch does not switch gait in lockstep, which
        # would make every environment sample the same speed on the same tick.
        phase = (tick_value + lanes) % period
        gait = jnp.where(
            phase < jnp.int32(walk_ticks),
            jnp.int32(ARSENAL_GAIT_WALK),
            jnp.where(
                phase < jnp.int32(walk_ticks + run_ticks),
                jnp.int32(ARSENAL_GAIT_RUN),
                jnp.int32(ARSENAL_GAIT_SPRINT),
            ),
        )
        stepping = away > jnp.int32(0)
        adaptive = jnp.where(
            stepping,
            jnp.int32(1)
            + (gait - jnp.int32(1)) * jnp.int32(ARSENAL_POLICY_COMPASS_DIRECTIONS)
            + (away - jnp.int32(1)),
            jnp.int32(0),
        )
        gait_driven = jnp.zeros_like(lane_mode, dtype=jnp.bool_)
        for mode in gait_driven_modes:
            gait_driven = gait_driven | (lane_mode == jnp.int32(mode))
        scripted = scripted.at[:, _WORLD_MOVE].set(
            jnp.where(gait_driven, adaptive, scripted[:, _WORLD_MOVE])
        )
    jump_offset = _HEAD_OFFSETS[_JUMP]
    jump_supported = mask[:, jump_offset + 1]
    jump_requested = (
        (lane_mode == jnp.int32(TargetMode.FLEE_WEAVE))
        & ((tick_value % jnp.int32(program.flee_jump_period_ticks)) == 0)
        & jump_supported
    )
    scripted = scripted.at[:, _JUMP].set(jump_requested.astype(jnp.int32))
    if TargetMode.FROZEN_POLICY in program.lane_modes:
        if sampled_factors is None:
            raise ValueError("frozen-policy target mode needs sampled factors")
        sampled = jnp.asarray(sampled_factors)
        if sampled.dtype != jnp.dtype(jnp.int32) or sampled.shape != neutral.shape:
            raise ValueError("sampled target factors must have shape int32[B,12]")
        requested = jnp.where(
            (lane_mode == jnp.int32(TargetMode.FROZEN_POLICY))[:, None],
            sampled,
            scripted,
        )
        frozen_lane = lane_mode == jnp.int32(TargetMode.FROZEN_POLICY)
        scoped_mask = program.frozen_factor_scope.restrict(mask)
        requested_mask = jnp.where(frozen_lane[:, None], scoped_mask, mask)
    else:
        if sampled_factors is not None:
            raise ValueError("sampled factors require a frozen-policy lane")
        requested = scripted
        requested_mask = mask
    legal = _selected_factor_row_legal(requested, requested_mask)
    neutral_legal = _selected_factor_row_legal(neutral, requested_mask)
    issued = jnp.where(legal[:, None], requested, neutral)
    issued_legal = jnp.where(legal, True, neutral_legal)
    motion_requested = (requested[:, _BASE_ACTION] != SKILL_IDLE) | (
        requested[:, _WORLD_MOVE] != 0
    )
    return TargetMotionOutput(
        issued,
        requested,
        lane_mode,
        legal,
        issued_legal,
        motion_requested,
        issued_legal
        & ((issued[:, _BASE_ACTION] != SKILL_IDLE) | (issued[:, _WORLD_MOVE] != 0)),
    )


__all__ = [
    "FACTORED_ACTION_SCOPE_SCHEMA",
    "TARGET_MOTION_PROGRAM_SCHEMA",
    "FactoredActionScope",
    "TargetMode",
    "TargetMotionOutput",
    "TargetMotionProgram",
    "flee_compass_choice",
    "standoff_compass_choice",
    "target_motion_factors",
]
