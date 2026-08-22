"""The world as a player would describe it, not as tensors.

The observation is complete but not legible: health lives in column 3 of
``self_f32``, "am I swimming" is column 13 of a 23-wide movement block, and
"is something dangerous near me" is a masked reduction over an entity set.
An agent author ends up rewriting the same translation every time, and the
NPC they are building is trying to answer ordinary player questions:

    How hurt am I?  What am I standing on?  Is anything hostile close?
    Can I reach that?  Is my weapon ready?

This module answers those, in those terms.  Nothing here is new information --
every value is assembled from :mod:`adk.policy.fields` and
:mod:`adk.policy.queries` -- but it is the difference between a schema and a
situation.

Because it describes a *player*, it works for any controlled entity: the same
view is what a scripted NPC, a behaviour tree, or a learned policy needs.
There is no algorithm-specific shaping here and no reward.

Every field is a batched array and nothing synchronizes with the device, so a
view can be built inside a rollout and recorded, or used directly as a compact
feature vector via :func:`as_features`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.types import AGENT_ENTITY

from adk.policy import queries
from adk.policy.fields import field
from adk.policy.observations import observation_sets, observation_vectors
from adk.policy.surfaces import light_tokens


class Vitals(NamedTuple):
    """How the player is doing."""

    health: jax.Array  # 0..1
    alive: jax.Array
    guarding: jax.Array
    stamina_broken: jax.Array
    invulnerable: jax.Array  # dodge i-frames
    recovering: jax.Array  # staggered / in damage recovery
    control_locked: jax.Array  # knocked back, cannot steer


class Resources(NamedTuple):
    """What the player can spend, as fractions of the authored maximum.

    A channel a weapon family never defines reads ``-1.0``, not ``0.0``: the
    Gym zeroes and masks-off missing stats, so the raw value alone cannot tell
    "out of mana" from "this family has no mana at all".  Distinguishing them
    matters -- an empty bow must reload, a sword must not.
    """

    stamina: jax.Array
    mana: jax.Array
    magic_charges: jax.Array
    signature_energy: jax.Array
    signature_charges: jax.Array
    ammo: jax.Array
    out_of_ammo: jax.Array  # has an ammo stat AND it is empty


class Stance(NamedTuple):
    """What the player's body is doing right now."""

    on_ground: jax.Array
    jumping: jax.Array
    falling: jax.Array
    swimming: jax.Array
    in_fluid: jax.Array
    climbing: jax.Array
    sprinting: jax.Array
    crouching: jax.Array
    sliding: jax.Array
    mantling: jax.Array
    gliding: jax.Array
    speed: jax.Array  # planar speed magnitude


class Threats(NamedTuple):
    """What might hurt the player, and where it is."""

    target_visible: jax.Array
    target_distance: jax.Array
    target_health: jax.Array
    hostiles_near: jax.Array  # count
    hostiles_ahead: jax.Array  # count inside the view cone
    nearest_hostile_distance: jax.Array
    nearest_hostile_found: jax.Array
    hazards_near: jax.Array  # count
    hazard_damage_rate: jax.Array  # summed over live hazards


class Surroundings(NamedTuple):
    """The immediate physical situation.

    ``submersion_known`` and ``drop_known`` are the environment's own validity
    flags.  They matter: without a world provider the underlying floats are
    zero, and an ungated ``drop_support_found < 0.5`` would report a cliff
    ahead in an empty scene.  Absent data reads as "not known", never as a
    hazard.
    """

    feet_submerged: jax.Array
    eyes_submerged: jax.Array
    submersion_known: jax.Array
    drop_ahead: jax.Array
    drop_height: jax.Array
    drop_known: jax.Array
    interactables_in_reach: jax.Array
    nearest_interactable_distance: jax.Array
    nearest_interactable_found: jax.Array
    light_level: jax.Array  # mean sky light where geometry is known; 0 if none


class Equipment(NamedTuple):
    """What the player is holding and what it can do."""

    weapon_id: jax.Array
    weapon_family: jax.Array
    abilities_ready: jax.Array  # off cooldown
    abilities_affordable: jax.Array  # off cooldown and resource-affordable


class PlayerView(NamedTuple):
    """One player's situation, assembled from the published observation."""

    vitals: Vitals
    resources: Resources
    stance: Stance
    threats: Threats
    surroundings: Surroundings
    equipment: Equipment


#: How close something must be to count as "near" by default.  Distances in
#: the observation are already normalized against the scene's sensor range, so
#: this is a fraction of that range rather than metres.
NEAR_DISTANCE = 0.5

#: Half-angle of the "ahead" cone, in degrees.
VIEW_CONE_DEGREES = 60.0


def _self(vectors: dict[str, jax.Array], name: str) -> jax.Array:
    return field(vectors["self"], "self_f32", name)


def _movement(vectors: dict[str, jax.Array], name: str) -> jax.Array:
    return field(vectors["movement_state"], "movement_state_f32", name) > 0.5


def _actor_slice(values: jax.Array, actor: int) -> jax.Array:
    """Take the controlled entity's row from a group with an entity axis."""

    return values[:, actor]


def vitals(legal_observation: Any, *, actor: int = AGENT_ENTITY) -> Vitals:
    """Health, guard, and control state for the controlled entity."""

    vectors = observation_vectors(legal_observation)
    defense = _actor_slice(vectors["defense"], actor)
    return Vitals(
        health=_self(vectors, "health_fraction"),
        alive=_self(vectors, "alive") > 0.5,
        guarding=field(defense, "defense_f32", "guard_active") > 0.5,
        stamina_broken=field(defense, "defense_f32", "stamina_broken") > 0.5,
        invulnerable=(
            field(defense, "defense_f32", "dodge_invulnerability_fraction") > 0.0
        ),
        recovering=_self(vectors, "damage_recovery") > 0.0,
        control_locked=_self(vectors, "knockback_control_lock") > 0.0,
    )


def resources(legal_observation: Any, *, actor: int = AGENT_ENTITY) -> Resources:
    """Spendable resource levels for the controlled entity.

    Unavailable channels are reported as ``-1.0`` rather than ``0.0``; see
    :class:`Resources`.
    """

    values = _actor_slice(observation_vectors(legal_observation)["resource"], actor)
    available = _actor_slice(legal_observation.resource_mask, actor)

    def channel(name: str) -> jax.Array:
        return jnp.where(
            field(available, "resource_mask", name),
            field(values, "resource_f32", name),
            jnp.float32(-1.0),
        )

    ammo = channel("ammo")
    return Resources(
        stamina=channel("stamina"),
        mana=channel("mana"),
        magic_charges=channel("magic_charges"),
        signature_energy=channel("signature_energy"),
        signature_charges=channel("signature_charges"),
        ammo=ammo,
        out_of_ammo=(ammo >= 0.0) & (ammo <= 0.0),
    )


def stance(legal_observation: Any) -> Stance:
    """Locomotion state, read from the 23 published movement flags."""

    vectors = observation_vectors(legal_observation)
    forward = _self(vectors, "forward_velocity")
    right = _self(vectors, "right_velocity")
    return Stance(
        on_ground=_movement(vectors, "on_ground"),
        jumping=_movement(vectors, "jumping"),
        falling=_movement(vectors, "falling"),
        swimming=_movement(vectors, "swimming"),
        in_fluid=_movement(vectors, "in_fluid"),
        climbing=_movement(vectors, "climbing"),
        sprinting=_movement(vectors, "sprinting"),
        crouching=_movement(vectors, "crouching"),
        sliding=_movement(vectors, "sliding"),
        mantling=_movement(vectors, "mantling"),
        gliding=_movement(vectors, "gliding"),
        speed=jnp.sqrt(forward * forward + right * right),
    )


def threats(
    legal_observation: Any,
    *,
    near: float = NEAR_DISTANCE,
    cone_degrees: float = VIEW_CONE_DEGREES,
) -> Threats:
    """Hostiles and hazards, counted and located.

    "Hostile" is the entity set's own published flag, not an inference from
    behaviour, so this reports what the environment asserts rather than what a
    heuristic guesses.
    """

    vectors = observation_vectors(legal_observation)
    sets = observation_sets(legal_observation)
    target = vectors["target"]

    entities = sets["entity"]
    hostile_mask = queries.matching(entities, "entity_f32", "hostile", at_least=0.5)
    hostiles = queries.refine(entities, hostile_mask)
    close_hostiles = queries.refine(
        hostiles,
        queries.matching(hostiles, "entity_f32", "distance", at_most=near),
    )
    ahead = queries.refine(
        hostiles,
        queries.field_of_view(
            hostiles,
            "entity_f32",
            forward="relative_forward",
            lateral="relative_right",
            half_angle_degrees=cone_degrees,
        ),
    )
    nearest_hostile = queries.nearest(hostiles, "entity_f32")

    hazards = sets["hazard"]
    close_hazards = queries.refine(
        hazards,
        queries.matching(hazards, "hazard_f32", "distance", at_most=near),
    )

    return Threats(
        target_visible=field(target, "target_f32", "visible") > 0.5,
        target_distance=field(target, "target_f32", "distance"),
        target_health=field(target, "target_f32", "health_fraction"),
        hostiles_near=queries.count(close_hostiles),
        hostiles_ahead=queries.count(ahead),
        nearest_hostile_distance=jnp.where(
            nearest_hostile.found, nearest_hostile.value, jnp.inf
        ),
        nearest_hostile_found=nearest_hostile.found,
        hazards_near=queries.count(close_hazards),
        hazard_damage_rate=queries.masked_sum(
            close_hazards, "hazard_f32", "damage_rate"
        ),
    )


def surroundings(
    legal_observation: Any,
    state: Any = None,
    *,
    near: float = NEAR_DISTANCE,
) -> Surroundings:
    """Water, drops, reachable things, and how bright it is.

    ``state`` is optional and only supplies light, which lives on the
    environment state rather than the observation.  Without it -- or without a
    light provider -- ``light_level`` is zero rather than invented.
    """

    vectors = observation_vectors(legal_observation)
    sets = observation_sets(legal_observation)
    actor_world = vectors["actor_world"]

    interactions = sets["interaction"]
    in_reach = queries.refine(
        interactions,
        queries.matching(interactions, "interaction_f32", "distance", at_most=near),
    )
    nearest_interactable = queries.nearest(interactions, "interaction_f32")

    brightness = jnp.zeros_like(field(actor_world, "actor_world_f32", "drop_height_fraction"))
    if state is not None:
        tokens = light_tokens(state)
        if tokens is not None:
            from adk.policy.fields import field_index

            sky = tokens.values[..., field_index("light_token_f32", "sky_light")]
            occupied = tokens.mask.astype(sky.dtype)
            brightness = jnp.sum(sky * occupied, axis=-1) / jnp.maximum(
                jnp.sum(occupied, axis=-1), 1.0
            )

    # Not in VECTOR_GROUPS (it is a bool mask, not a feature block), but it is
    # registered in GROUP_FEATURES so its columns are still named.
    mask = legal_observation.actor_world_mask
    submersion_known = field(mask, "actor_world_mask", "submersion_available")
    drop_known = field(mask, "actor_world_mask", "drop_available")

    return Surroundings(
        feet_submerged=submersion_known
        & (field(actor_world, "actor_world_f32", "feet_submerged") > 0.5),
        eyes_submerged=submersion_known
        & (field(actor_world, "actor_world_f32", "eyes_submerged") > 0.5),
        submersion_known=submersion_known,
        # Unknown ground is not a cliff: gate on the validity flag.
        drop_ahead=drop_known
        & (field(actor_world, "actor_world_f32", "drop_support_found") < 0.5),
        drop_height=jnp.where(
            drop_known,
            field(actor_world, "actor_world_f32", "drop_height_fraction"),
            0.0,
        ),
        drop_known=drop_known,
        interactables_in_reach=queries.count(in_reach),
        nearest_interactable_distance=jnp.where(
            nearest_interactable.found, nearest_interactable.value, jnp.inf
        ),
        nearest_interactable_found=nearest_interactable.found,
        light_level=brightness,
    )


def equipment(legal_observation: Any, *, actor: int = AGENT_ENTITY) -> Equipment:
    """What is held, and how many abilities are actually usable now.

    "Ready" means off cooldown; "affordable" additionally requires the
    resource cost to be payable, which is the environment's own ``affordable``
    flag rather than a recomputation.
    """

    weapon = _actor_slice(legal_observation.weapon_i32, actor)
    ability_values = _actor_slice(legal_observation.ability_f32, actor)
    ability_mask = _actor_slice(legal_observation.ability_mask, actor)

    cooldown = ability_values[..., _ability_index("cooldown_remaining_fraction")]
    affordable = ability_values[..., _ability_index("affordable")]
    ready = ability_mask & (cooldown <= 0.0)

    return Equipment(
        weapon_id=weapon[..., _weapon_index("weapon_id")],
        weapon_family=weapon[..., _weapon_index("weapon_family")],
        abilities_ready=jnp.sum(ready.astype(jnp.int32), axis=-1),
        abilities_affordable=jnp.sum(
            (ready & (affordable > 0.5)).astype(jnp.int32), axis=-1
        ),
    )


def _ability_index(name: str) -> int:
    from adk.policy.fields import field_index

    return field_index("ability_f32", name)


def _weapon_index(name: str) -> int:
    from adk.policy.fields import field_index

    return field_index("weapon_i32", name)


def player_view(
    actor_input: Any,
    state: Any = None,
    *,
    actor: int = AGENT_ENTITY,
    near: float = NEAR_DISTANCE,
    cone_degrees: float = VIEW_CONE_DEGREES,
) -> PlayerView:
    """The whole situation in one call.

    Pass ``state`` to include light.  ``actor`` selects which entity's
    perspective to take, so the same function describes the opponent by
    passing ``TARGET_ENTITY`` -- useful for self-play and for opponent models.
    """

    legal = actor_input.legal_observation
    return PlayerView(
        vitals=vitals(legal, actor=actor),
        resources=resources(legal, actor=actor),
        stance=stance(legal),
        threats=threats(legal, near=near, cone_degrees=cone_degrees),
        surroundings=surroundings(legal, state, near=near),
        equipment=equipment(legal, actor=actor),
    )


def as_features(view: PlayerView) -> jax.Array:
    """Flatten a view into one ``float32[batch, n]`` vector.

    A compact, human-meaningful alternative to the full observation: useful as
    a policy input on its own, as an auxiliary head, or as a logging record.
    Booleans become 0/1 and non-finite distances become -1, so the result is
    always finite and safe to feed to a network.
    """

    columns = []
    for group in view:
        for value in group:
            array = jnp.asarray(value).astype(jnp.float32)
            array = jnp.where(jnp.isfinite(array), array, -1.0)
            columns.append(array)
    return jnp.stack(columns, axis=-1)


def feature_names(view: PlayerView | None = None) -> tuple[str, ...]:
    """Names matching :func:`as_features`, in order.  Pure Python.

    Derived from the NamedTuple fields themselves, so the names cannot drift
    away from the values.
    """

    template = view if view is not None else PlayerView(
        Vitals(*[None] * len(Vitals._fields)),
        Resources(*[None] * len(Resources._fields)),
        Stance(*[None] * len(Stance._fields)),
        Threats(*[None] * len(Threats._fields)),
        Surroundings(*[None] * len(Surroundings._fields)),
        Equipment(*[None] * len(Equipment._fields)),
    )
    names: list[str] = []
    for group_name, group in zip(PlayerView._fields, template):
        for field_name in type(group)._fields:
            names.append(f"{group_name}.{field_name}")
    return tuple(names)


__all__ = [
    "NEAR_DISTANCE",
    "VIEW_CONE_DEGREES",
    "Equipment",
    "PlayerView",
    "Resources",
    "Stance",
    "Surroundings",
    "Threats",
    "Vitals",
    "as_features",
    "equipment",
    "feature_names",
    "player_view",
    "resources",
    "stance",
    "surroundings",
    "threats",
    "vitals",
]
