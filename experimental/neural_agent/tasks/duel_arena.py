"""Two-actor duel arena: both entities take external per-tick factors (task #101).

:func:`make_duel_arena` returns ``reset(keys)`` and ``step(state, factors_entity0,
factors_entity1, keys)`` over the engine's shared-arena multi-actor path
(``hytalegym.jax.training.multi_actor``). Entity 0 and entity 1 are both
policy-owned rows, so the caller's factors drive both bodies, and each row gets
its own legal learner-v3 observation and action mask from its own perspective.
The built-in opponent controller is off: ``opponent_controller_mask`` is all
False, which the multi-actor runtime needs before it accepts a policy control
for entity 1 (``multi_actor/runtime.py``: ``controlled & ~opponent_controller_mask``).

Approximations and engine gaps (each is pinned by a test or measured while building the arena):

* The open-flat fixture gates perception of the agent row only. Entity 1's
  perception of entity 0 is the same fixture evaluated on entity 1's
  actor-first view (:func:`_capabilities`).
* The actor-world scalars (medium, submersion, drop support) exist for the
  agent row only, and the multi-actor view blanks them for every other actor.
  Entity 1 copies entity 0's row, which on the flat fixture is the same floor
  (:func:`_observe`).
* The encoder divides a row by the maximum of the role it plays in the view, and the
  actor-first view keeps the agent's parameters. Entity 1's health fractions (self row,
  target row, combat block) and its agent velocity are therefore recomputed on its own
  maxima (:func:`_entity_one_on_its_own_maxima`). The gait-speed, fall-speed, chase-speed
  and attack-table normalisers have no role counterpart in ``CombatParams``, so they are
  not rescaled.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat import (
    arsenal_runtime_config,
    default_combat_params,
    hytale_0_5_7_loadouts,
)
from hytalegym.rulesets import load_combat_ruleset
from adk.policy import fields as schema
from hytalegym.jax.combat.arsenal.environment import (
    open_flat_arsenal_world_capabilities,
)
from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.arsenal.runtime import reset_arsenal_batch
from hytalegym.jax.combat.arsenal.training_inventory import training_inventory_from_config
from hytalegym.jax.combat.observation.v3.policy import ARSENAL_POLICY_ACTION_HEAD_SIZES
from hytalegym.jax.combat.runtime.math import _round_to_scale, _unit_fraction
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.jax.training.multi_actor import (
    decode_policy_actor_factors_flat,
    initialize_multi_actor_arena_state,
    observe_policy_actors_flat,
    policy_actor_assignment,
    step_multi_actor_arena,
)
from hytalegym.jax.training.multi_actor.views import actor_first_arsenal_view

#: Entity 0 is the learner row and entity 1 the opponent row.
ENTITY_COUNT: int = 2
#: AgentSpec's default turn limit (adk/core/spec.py), which the open-flat scene uses.
MAXIMUM_TURN_DEGREES: float = 45.0
#: The shortbow's per-shot arrow (``Weapon_Arrow_Crude``; the bow recipe's ARROW_ITEM). Each shot removes one.
ARROW_ITEM: int = semantic_id("Weapon_Arrow_Crude")
_HEAD_SIZES: tuple[int, ...] = tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES)
_HEAD_STARTS: tuple[int, ...] = tuple(
    sum(_HEAD_SIZES[:index]) for index in range(len(_HEAD_SIZES))
)
_SELF_HEALTH = schema.field_index("self_f32", "health_fraction")
_TARGET_HEALTH = schema.field_index("target_f32", "health_fraction")
_TARGET_VISIBLE = schema.field_index("target_f32", "visible")
_COMBAT_AGENT_HEALTH = schema.field_index("combat_f32", "agent_health_fraction")
_COMBAT_TARGET_HEALTH = schema.field_index("combat_f32", "visible_target_health_fraction")
_COMBAT_FORWARD = schema.field_index("combat_f32", "agent_forward_velocity")
_COMBAT_RIGHT = schema.field_index("combat_f32", "agent_right_velocity")

#: The CombatParams fields X that have both ``agent_X`` and ``target_X`` of one shape, enumerated from
#: ``CombatParams`` (hytalegym/jax/combat/types.py) on 2026-10-20. ``damage`` is left out: it has a counterpart
#: by name only, since agent_damage is a scalar and target_damage a five-entry table per attack.
ROLE_PAIRED_FIELDS: tuple[str, ...] = (
    "acceleration",
    "attack_pause_max_seconds",
    "attack_pause_min_seconds",
    "bounds",
    "damage_reward_scale",
    "eye_offset",
    "head_pitch_max_degrees",
    "head_pitch_min_degrees",
    "head_yaw_max_degrees",
    "head_yaw_min_degrees",
    "max_head_rotation_degrees",
    "max_health",
    "max_speed",
    "turn_speed_degrees",
)


class DuelArena(NamedTuple):
    """The two callables of one built duel: ``reset(keys)`` and ``step(...)``."""

    reset: Any
    step: Any


def factors_are_legal(mask: jax.Array, factors: jax.Array) -> jax.Array:
    """Return bool[B]: each head's factor is in range and allowed by ``mask`` (bool[B, 124])."""

    legal = jnp.ones(factors.shape[:1], dtype=jnp.bool_)
    for index, (start, size) in enumerate(zip(_HEAD_STARTS, _HEAD_SIZES, strict=True)):
        choice = factors[:, index]
        safe = jnp.clip(choice, 0, size - 1)
        allowed = jnp.take_along_axis(
            mask[:, start : start + size], safe[:, None], axis=1
        )[:, 0]
        legal = legal & (choice >= 0) & (choice < size) & allowed
    return legal


def _profiles(value: str | Sequence[str], batch: int) -> list[str]:
    names = [value] * batch if isinstance(value, str) else list(value)
    if len(names) != batch:
        raise ValueError(f"expected {batch} loadout names per entity, got {len(names)}")
    return names


def _capabilities(arsenal, locomotion, params, config, *, sector, sensor_range):
    """Open-flat capabilities with both rows of the perception pair gated.

    The fixture gates entity 0's row (``[:, AGENT, TARGET]``). Entity 1's row is
    the same fixture evaluated on entity 1's actor-first view, whose agent row is
    entity 1 (``views.actor_first_arsenal_view``), mapped back to the physical rows.
    """

    physical = open_flat_arsenal_world_capabilities(
        arsenal,
        params,
        config=config,
        legal_observation_sensor_range=sensor_range,
        view_sector_full_angle_radians=sector,
    )
    batch = arsenal.combat.health.shape[0]
    view_state, view_config, _ = actor_first_arsenal_view(
        arsenal,
        config,
        physical,
        locomotion,
        jnp.full((batch,), TARGET_ENTITY, dtype=jnp.int32),
    )
    seen = open_flat_arsenal_world_capabilities(
        view_state,
        params,
        config=view_config,
        legal_observation_sensor_range=sensor_range,
        view_sector_full_angle_radians=sector,
    )
    return physical._replace(
        target_candidate_perceptible=physical.target_candidate_perceptible.at[
            :, TARGET_ENTITY, AGENT_ENTITY
        ].set(seen.target_candidate_perceptible[:, AGENT_ENTITY, TARGET_ENTITY]),
        line_of_sight=physical.line_of_sight.at[:, TARGET_ENTITY].set(
            seen.line_of_sight[:, AGENT_ENTITY]
        ),
        selector_line_of_sight=physical.selector_line_of_sight.at[:, TARGET_ENTITY].set(
            seen.selector_line_of_sight[:, AGENT_ENTITY]
        ),
    )


def _entity_one_on_its_own_maxima(structured, arsenal, locomotion, params):
    """Recompute entity 1's role-dependent columns from the raw state, on entity 1's own maxima.

    The encoder divides by the maximum of the role a row plays, and the actor-first view
    keeps the agent's parameters. For entity 1 the agent role is its own, so its self health
    divides by ``target_max_health``, its target row (entity 0) by ``agent_max_health``, and its
    combat block the same way. Velocities divide by ``max_speed`` for the agent role, so entity 1
    divides by ``target_max_speed``. Each value uses the encoder's helpers (``_unit_fraction``
    clamps to the unit interval, ``_round_to_scale`` is the wire rounding of the combat block),
    and the target columns stay zero unless entity 0 is perceived (the ``visible`` column).
    """

    combat = arsenal.combat
    own_health = combat.health[:, TARGET_ENTITY]
    other_health = combat.health[:, AGENT_ENTITY]
    visible = structured.base.target_f32[:, TARGET_ENTITY, _TARGET_VISIBLE] > jnp.float32(0.5)
    yaw = jnp.deg2rad(locomotion.head_yaw[:, TARGET_ENTITY])
    velocity = combat.velocity[:, TARGET_ENTITY]
    forward = (
        velocity[:, 0] * -jnp.sin(yaw) + velocity[:, 2] * -jnp.cos(yaw)
    ) / params.target_max_speed
    right = (
        velocity[:, 0] * jnp.cos(yaw) + velocity[:, 2] * -jnp.sin(yaw)
    ) / params.target_max_speed
    base = structured.base
    combat_f32 = (
        base.combat_f32.at[:, TARGET_ENTITY, _COMBAT_AGENT_HEALTH]
        .set(_unit_fraction(own_health, params.target_max_health))
        .at[:, TARGET_ENTITY, _COMBAT_TARGET_HEALTH]
        .set(
            jnp.where(
                visible,
                _unit_fraction(
                    _round_to_scale(other_health, params.wire_fixed_point_scale),
                    params.agent_max_health,
                ),
                jnp.float32(0.0),
            )
        )
        .at[:, TARGET_ENTITY, _COMBAT_FORWARD]
        .set(forward)
        .at[:, TARGET_ENTITY, _COMBAT_RIGHT]
        .set(right)
    )
    base = base._replace(
        self_f32=base.self_f32.at[:, TARGET_ENTITY, _SELF_HEALTH].set(
            own_health / jnp.maximum(jnp.abs(params.target_max_health), jnp.float32(1.0e-6))
        ),
        target_f32=base.target_f32.at[:, TARGET_ENTITY, _TARGET_HEALTH].set(
            jnp.where(
                visible,
                _unit_fraction(other_health, params.agent_max_health),
                jnp.float32(0.0),
            )
        ),
        combat_f32=combat_f32,
    )
    return structured._replace(base=base)


def _observe(arena, capabilities, params, config, assignment):
    """Per-actor learner-v3 structure and action mask, actor-major ``[B, 2, ...]``."""

    observation = observe_policy_actors_flat(
        arena.arsenal, params, capabilities, config, arena.locomotion, assignment
    )
    structured = observation.structured
    # The engine's actor-world scalars are zeroed for every non-agent row. Copy the
    # agent row's values, which describe the same flat floor.
    structured = structured._replace(
        actor_world_f32=structured.actor_world_f32.at[:, TARGET_ENTITY].set(
            structured.actor_world_f32[:, AGENT_ENTITY]
        ),
        actor_world_mask=structured.actor_world_mask.at[:, TARGET_ENTITY].set(
            structured.actor_world_mask[:, AGENT_ENTITY]
        ),
    )
    structured = _entity_one_on_its_own_maxima(
        structured, arena.arsenal, arena.locomotion, params
    )
    return observation._replace(structured=structured)


def _factor_rows(value, batch: int, name: str) -> jax.Array:
    factors = jnp.asarray(value)
    if factors.dtype != jnp.int32:
        raise TypeError(f"{name} must be int32, got {factors.dtype}")
    if factors.shape != (batch, len(_HEAD_SIZES)):
        raise ValueError(f"{name} must have shape {(batch, len(_HEAD_SIZES))}")
    return factors


def _matchup_roles() -> tuple[str, str]:
    """The matchup's (agent role, target role) ids, read from the loaded ruleset."""

    matchup = load_combat_ruleset()["matchup"]
    return matchup["agent_role"], matchup["target_role"]


def _role_paired_params(params, roles, agent_role):
    """Give each slot the paired fields of the role it plays (``roles`` is the pair of role ids).

    Slot 0 is read through ``agent_X`` and slot 1 through ``target_X``. A slot whose role is the agent role keeps its
    values; a slot playing the target role takes the counterpart value, so after the rebuild both slots hold the value
    of the role they play.
    """

    update = {}
    for name in ROLE_PAIRED_FIELDS:
        agent_value = getattr(params, f"agent_{name}")
        target_value = getattr(params, f"target_{name}")
        update[f"agent_{name}"] = agent_value if roles[0] == agent_role else target_value
        update[f"target_{name}"] = agent_value if roles[1] == agent_role else target_value
    return params._replace(**update)


def _arrow_inventory(config, arrows):
    """``training_inventory_from_config(config)`` with each entity's arrow stack set to its count.

    The inventory is ``[batch, entity, container slot]``. A loadout holds its arrow stack in a slot as the outer root's
    budget (quantity 1); every slot holding :data:`ARROW_ITEM` takes that entity's count, so a shot removes one arrow.
    An entity with a positive count whose loadout holds no arrow stack raises ``ValueError``.
    """

    inventory = training_inventory_from_config(config)
    arrow_slot = inventory.item_id == ARROW_ITEM
    for entity, count in enumerate(arrows):
        if count > 0 and not bool(jnp.all(jnp.any(arrow_slot[:, entity], axis=-1))):
            raise ValueError(
                f"arrows[{entity}] is {count} but entity {entity}'s loadout has no arrow stack ({ARROW_ITEM})"
            )
    counts = jnp.asarray(arrows, dtype=inventory.quantity.dtype)[None, :, None]
    return inventory._replace(quantity=jnp.where(arrow_slot, counts, inventory.quantity))


def _target_head_on_body(arsenal):
    """Set ``combat.target_head_yaw`` to entity 1's body yaw, ``combat.yaw[:, TARGET_ENTITY]`` (#112).

    Why: the progressive attack selector (``arsenal/effects/events.py``, ``_progressive_selector_result``) takes
    slot 0's facing from ``combat.yaw`` (the body) and slot 1's from ``combat.target_head_yaw``. The built-in
    target AI, the only writer of that leaf in the engine, is off in the duel (``target_active`` False), so the
    leaf kept its reset seed while entity 1's body turned. Measured on the mirror placement (sword PRESSURE, 200
    ticks): entity 1's body turned 36 degrees and the leaf stayed at its spawn heading, up to 58.7 degrees off.

    The same leaf also sets entity 1's view sector (``arsenal/environment.py``, ``head_yaw`` from ``combat.yaw``),
    which now follows the body as slot 0's does. It also sets entity 1's projectile muzzle yaw (``environment.py``,
    ``muzzle_yaw``) and entity 0's observation of the target's yaw (``multi_actor/views.py``, the legacy actor's
    ``target_yaw``). Slot 0's muzzle reads its look (``agent_head_yaw``) and entity 1's observation as seen by entity
    0 is read from its look by the engine's non-legacy path. In the duel the look and the body differ by up to 45
    degrees (measured), so entity 1's arrows now leave along its body, not its look. Nothing in the engine serves the
    selector (body, as slot 0's is) and the muzzle (look) from one leaf; this sync keeps the selector.

    When: ``step`` calls this on the state it receives, before the projection, the decode and the engine step, and
    on the state it returns, before the next projection, so the next observation is current. ``reset`` is not
    changed: its ``entity_yaw`` seed stands. Lag: none at the selector. The selector reads both slots' yaw before
    the agent's own motion (``process_direct_events`` at ``arsenal/runtime.py:1302``, ``_microtick`` at ``:1519``),
    and entity 1's locomotion turn runs after the engine step (``multi_actor/runtime.py``), so both read the
    start-of-tick body yaw (measured: the selector's yaw equals the step input for both slots, max difference 0.0).

    Pitch is not synced. Slot 0's selector pitch is ``combat.pitch``, the agent's body pitch, which the engine holds
    level (``runtime/agent.py``, ``pitch = zeros_like``). Entity 1 has no body-pitch leaf: its look pitch lives in
    the locomotion bank, which the selector does not read for either seat. ``target_head_pitch`` is zero for the
    whole duel (seeded at ``runtime/reset.py:157``, never written with ``target_active`` False), which matches.
    """

    combat = arsenal.combat
    return arsenal._replace(combat=combat._replace(target_head_yaw=combat.yaw[:, TARGET_ENTITY]))


def make_duel_arena(
    batch: int,
    loadouts: Sequence[str | Sequence[str]],
    *,
    view_sector_full_angle_radians: float | None,
    legal_observation_sensor_range: bool,
    combat_params: Any | None = None,
    max_health: tuple[float, float] | None = None,
    roles: tuple[str, str] | None = None,
    arrows: tuple[int, int] | None = None,
) -> DuelArena:
    """Build the two-entity duel on the open-flat fixture.

    Args:
        batch: Number of duels B, a positive int.
        loadouts: ``(entity 0 loadouts, entity 1 loadouts)``. Each is one profile name
            for every row, or a sequence of ``batch`` names.
        view_sector_full_angle_radians: The view sector, radians, centred on each
            entity's head yaw. ``None`` leaves the fixture's sector open.
        legal_observation_sensor_range: Hide a row beyond ``params.sensor_range``.
        combat_params: The ``CombatParams`` to duel under (``microticks`` must be 1, one
            decision per tick); ``None`` takes ``default_combat_params(microticks=1)``. The
            duel sets ``target_active`` False whatever is passed: the built-in target motion
            is gated by it, not by the opponent-controller mask (#103b, 2026-10-20).
        max_health: ``(entity 0 maximum, entity 1 maximum)`` overriding
            ``agent_max_health`` and ``target_max_health`` (the league's equal roles,
            2026-10-20); the reset health defaults follow them, and so do the
            fractions each entity reads.
        roles: ``(entity 0 role, entity 1 role)``, each the matchup's agent or target role id as the loaded ruleset
            names them; any other value raises ``ValueError``. It sets the statuses (``entity_role_ids``) and gives
            each slot the paired fields of the role it plays (:data:`ROLE_PAIRED_FIELDS`): a slot playing the target
            role takes the ``target_X`` value, a slot playing the agent role keeps ``agent_X``. It applies before
            ``max_health``. ``None`` keeps the matchup as it is.

            The fields ``roles`` leaves as they are (no counterpart): the agent-only fields (``knockback_scale``, the
            walk, air-drag and force-drag fields, ``hit_delays``, ``attack_ranges``, ``half_angles``,
            ``jump_velocity``, ``min_walk_speed``, ``min_hit_slowdown``, ``maximum_gait_speed``,
            ``movement_velocity_resistance``, ``steering_relative_turn_speed``, ``spawn``, ``motion_model``,
            ``variable_jump_fall_gravity``); the target-only fields (``knockback_force``, ``knockback_relative_x``,
            ``knockback_relative_z``, ``knockback_velocity_y``, the chase, maintain, strafe, activation, attack-table
            and sweep fields, ``offset``, ``initial_distance``, ``active``); and ``damage`` (a scalar against a
            five-entry table). The motion kernel runs every entity row through the agent slot
            (``multi_actor/locomotion.py``, ``_materialize_entity_zero``) and reads ``agent_max_speed`` for both
            slots, so a role pair moves both bodies through ``agent_*``; the ``target_*`` motion fields are read only by
            the built-in target's motion, which ``params.target_active`` switches on (see tasks/README.md).
        arrows: ``(entity 0 arrows, entity 1 arrows)``, two non-negative ints, or ``None`` (the bare factory
            inventory, which holds no arrows). Each count is the quantity of the entity's arrow stack
            (:data:`ARROW_ITEM`) in ``training_inventory_from_config(config)``, passed to the reset as its
            ``initial_inventory``. A shot removes one arrow. ``None`` keeps the reset bit-identical. An entity with
            a positive count whose loadout holds no arrow stack raises ``ValueError``.

    Returns:
        A :class:`DuelArena`. ``reset(keys, *, entity_position=None, entity_health=None,
        entity_yaw=None)`` returns ``(state, observation_entity0, observation_entity1,
        action_mask_entity0, action_mask_entity1)``. ``step(state, factors_entity0,
        factors_entity1, keys)`` returns ``(state, observation_entity0,
        observation_entity1, action_mask_entity0, action_mask_entity1, info, done)``.
        ``done`` is bool[B] and ``info`` holds [B, 2] arrays ``action_valid``,
        ``reward``, ``terminated``, ``truncated``, ``health``.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be a positive int, got {batch!r}")
    if len(loadouts) != ENTITY_COUNT:
        raise ValueError("loadouts must be (entity 0 loadouts, entity 1 loadouts)")
    if not isinstance(legal_observation_sensor_range, bool):
        raise TypeError("legal_observation_sensor_range must be a bool")
    sector = view_sector_full_angle_radians
    if sector is not None and (isinstance(sector, bool) or not isinstance(sector, (int, float))):
        raise TypeError("view_sector_full_angle_radians must be a float or None")
    if arrows is not None and (
        not isinstance(arrows, (tuple, list))
        or len(arrows) != ENTITY_COUNT
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in arrows)
    ):
        raise ValueError(
            f"arrows must be (entity 0 arrows, entity 1 arrows), two non-negative ints, got {arrows!r}"
        )

    params = default_combat_params(microticks=1) if combat_params is None else combat_params
    if int(params.microticks) != 1:
        raise ValueError("combat_params.microticks must be 1: the duel takes one decision per tick")
    if roles is not None:
        agent_role, target_role = _matchup_roles()
        if len(roles) != ENTITY_COUNT or any(role not in (agent_role, target_role) for role in roles):
            raise ValueError(
                f"roles must be two role ids, each {agent_role!r} or {target_role!r}, got {tuple(roles)!r}"
            )
        params = _role_paired_params(params, tuple(roles), agent_role)
    if max_health is not None:
        if len(max_health) != ENTITY_COUNT or any(float(value) <= 0.0 for value in max_health):
            raise ValueError("max_health must be two positive maxima, (entity 0, entity 1)")
        params = params._replace(
            agent_max_health=jnp.asarray(float(max_health[0]), dtype=jnp.float32),
            target_max_health=jnp.asarray(float(max_health[1]), dtype=jnp.float32),
        )
    # The built-in target motion (chase, strafe, maintain; opponents/legacy/motion.py) is gated by
    # params.target_active, not by the opponent-controller mask, and switches on at the target's activation
    # tick (36 to 44, a reset draw). The duel drives entity 1 by its factors alone, so it is off here (2026-10-20,
    # #103b: under neutral factors entity 1 drifted 0.65 blocks in 100 ticks at 6 and 10 blocks of separation).
    params = params._replace(target_active=jnp.asarray(False, dtype=jnp.bool_))
    config = arsenal_runtime_config(
        hytale_0_5_7_loadouts(
            _profiles(loadouts[0], batch),
            target_profiles=_profiles(loadouts[1], batch),
        ),
        opponent_controller_mask=jnp.zeros((batch, ENTITY_COUNT), dtype=jnp.bool_),
        legal_observation=True,
        entity_role_ids=None if roles is None else tuple(roles),
    )
    seeded_inventory = None if arrows is None else _arrow_inventory(config, tuple(arrows))
    entity_index = jnp.tile(jnp.arange(ENTITY_COUNT, dtype=jnp.int32), (batch, 1))
    assignment = policy_actor_assignment(
        entity_index,
        entity_index,
        trainable=jnp.zeros((batch, ENTITY_COUNT), dtype=jnp.bool_),
        entity_count=ENTITY_COUNT,
    )

    def project(arena):
        capabilities = _capabilities(
            arena.arsenal,
            arena.locomotion,
            params,
            config,
            sector=sector,
            sensor_range=legal_observation_sensor_range,
        )
        return _observe(arena, capabilities, params, config, assignment), capabilities

    def policy_rows(observation):
        structured = observation.structured
        rows = []
        for slot in range(ENTITY_COUNT):
            rows.append(
                (
                    jax.tree_util.tree_map(lambda value: value[:, slot], structured),
                    observation.action_mask[:, slot],
                )
            )
        return rows

    def reset_core(keys, entity_position, entity_health, entity_yaw):
        arsenal, _ = reset_arsenal_batch(
            keys,
            params,
            config,
            entity_position=entity_position,
            entity_health=entity_health,
            entity_yaw=entity_yaw,
            initial_inventory=seeded_inventory,
        )
        if entity_yaw is not None:
            # Every head starts on its body (multi_actor/locomotion.py:115-123). reset_arsenal_batch
            # sets only the body yaw, so the view direction (agent_head_yaw, target_head_yaw) is seeded here.
            combat = arsenal.combat._replace(
                agent_head_yaw=entity_yaw[:, AGENT_ENTITY],
                target_head_yaw=entity_yaw[:, TARGET_ENTITY],
            )
            arsenal = arsenal._replace(combat=combat)
        arena = initialize_multi_actor_arena_state(arsenal, params)
        observation, _ = project(arena)
        (observation0, mask0), (observation1, mask1) = policy_rows(observation)
        return arena, observation0, observation1, mask0, mask1

    compiled_reset = jax.jit(reset_core)

    def reset(keys, *, entity_position=None, entity_health=None, entity_yaw=None):
        """Reset every duel. The optional arrays place bodies: [B, 2, 3] positions, [B, 2] health, [B, 2] yaw."""

        def as_float(value):
            return None if value is None else jnp.asarray(value, dtype=jnp.float32)

        return compiled_reset(
            keys,
            as_float(entity_position),
            as_float(entity_health),
            as_float(entity_yaw),
        )

    def step(arena, factors_entity0, factors_entity1, keys):
        """One duel tick. Entity 1's selector facing is synced first (:func:`_target_head_on_body`)."""

        arena = arena._replace(arsenal=_target_head_on_body(arena.arsenal))
        factors = jnp.stack(
            (
                _factor_rows(factors_entity0, batch, "factors_entity0"),
                _factor_rows(factors_entity1, batch, "factors_entity1"),
            ),
            axis=1,
        )
        observation, capabilities = project(arena)
        controls = decode_policy_actor_factors_flat(
            arena.arsenal,
            observation.structured,
            factors,
            capabilities,
            config,
            arena.locomotion,
            assignment,
            maximum_turn_degrees=MAXIMUM_TURN_DEGREES,
        )
        commands = empty_arsenal_commands(batch, entity_count=ENTITY_COUNT)._replace(
            world=capabilities
        )
        transition = step_multi_actor_arena(
            arena, controls, assignment, keys, params, commands, config
        )
        next_state = transition.state._replace(
            arsenal=_target_head_on_body(transition.state.arsenal)
        )
        next_observation, _ = project(next_state)
        legal = jnp.stack(
            [
                factors_are_legal(observation.action_mask[:, slot], factors[:, slot])
                for slot in range(ENTITY_COUNT)
            ],
            axis=1,
        )
        info = {
            "action_valid": controls.valid & legal,
            "reward": transition.reward,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "health": transition.state.arsenal.combat.health,
        }
        done = jnp.any(transition.terminated | transition.truncated, axis=1)
        (observation0, mask0), (observation1, mask1) = policy_rows(next_observation)
        return (
            next_state,
            observation0,
            observation1,
            mask0,
            mask1,
            info,
            done,
        )

    return DuelArena(reset=reset, step=jax.jit(step))


__all__ = ["ARROW_ITEM", "DuelArena", "ENTITY_COUNT", "ROLE_PAIRED_FIELDS", "factors_are_legal", "make_duel_arena"]
