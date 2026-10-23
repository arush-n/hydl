"""Token schema v3 (``skill_tokens_v3``): what the neural layer perceives, knows about itself and remembers
(``docs/PERCEPTION.md`` section 10; owner, 2026-10-20 night).

Four decisions frame it: one spatial convention, the ``where`` block of ``state/frames.py``, on every spatial
token (the body and the head frame); a stored spatial memory, so entities, geometry cells and block candidates that
leave view stay tokens, ``remembered``, with their age; the agent's own recent actions as ``act`` tokens (which head,
the last choice, how long it has been held); and every legal field the simulator publishes, with a seam (zero, and
``known`` 0) for what it does not publish yet (the asks S1 to S9, the coordinator's).

A token type here is its feature list, ``FEATURES``; ``state/senses.py`` fills the arrays and ``model/encoder.py``
reads them with ``EncoderConfig(token_types=token_widths(TOKEN_SCHEMA_V3))``. Every list ends in ``marked``. The order
of the types the trunk reads is ``TRUNK_TYPE_NAMES_V3``. v1 and v2 are unchanged beside it.
"""

from __future__ import annotations

from typing import Mapping

from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.contract.surface import (
    ATTACK_PHASES,
    ATTITUDES,
    BLOCK_PROPERTIES,
    CREATURE_PROPERTIES,
    HELD_KINDS,
    InventorySlotTokenV2,
    head_layout,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    DEFENSE_FLOAT_FEATURES,
    MOVEMENT_STATE_FEATURES,
    OBSERVATION_CAPACITY,
    RESOURCE_COUNT,
)
from hytalegym.jax.combat.observation.v3.tokens.light import ACTOR_LIGHT_POLICY_FEATURES
from hytalegym.jax.combat.observation.v3.tokens.world import WORLD_GEOMETRY_TOKEN_CAPACITY

TOKEN_SCHEMA_V3: str = "skill_tokens_v3"

DESCRIPTION_WIDTH: int = 64
"""As v2's (PERCEPTION.md decision 1)."""
RESERVE_WIDTH: int = 8
"""Spare features per content token, zero until a later version assigns them."""
RESOURCES: tuple[str, ...] = ("stamina", "mana", "magic_charges", "signature_energy", "signature_charges", "ammo",
                              "oxygen")
"""The gym's seven resource slots, in its order (``mechanics/schema/contract.py``)."""
ITEM_PROPERTIES_V3: tuple[str, ...] = (
    "block", "tool", "weapon", "armor", "consumable", "utility", "furniture", "non_gameplay", "quality", "item_level",
    "max_stack_log2", "max_durability_log10", "armor_head", "armor_chest", "armor_legs", "armor_hands", "armor_health",
    "armor_physical", "armor_projectile", "tool_best_power", "tool_gather_types",
)
"""The registry's item property row (equal to ``surface.ITEM_PROPERTIES``; a test checks)."""
GEOMETRY_FLAGS: tuple[str, ...] = ("solid", "opaque", "fluid", "damaging", "climbable", "bouncy", "trigger",
                                   "protrudes", "has_movement", "has_fluid_movement")
"""The cell flag word's bits, in bit order (``hytalegym/geometry/contract.py``)."""
GEOMETRY_EDGES: int = 12
GEOMETRY_EDGE_FIELDS: tuple[str, ...] = ("mask", "to_forward", "to_right", "to_up", "cost", "kind", "flags")
GEOMETRY_BOXES: int = 9
GEOMETRY_BOX_FIELDS: tuple[str, ...] = ("mask", "x0", "y0", "z0", "x1", "y1", "z1")
"""A geometry token's traversal edges and detail boxes, from the gym's world tokens
(``observation/v3/tokens/world.py``). The gym gives an edge's destination as an index into its own token list, which
the trunk cannot resolve (a token carries no index); the adapter gives instead the destination's displacement from
this cell in the body frame (blocks over 16, as the ``where`` block), which the stored memory turns with the body. The
boxes are the cell's own shape, in world axes around the cell (a shape, not a place, so it is not rotated)."""
AFFORDANCES: tuple[str, ...] = ("breakable", "ore", "harvestable", "soft", "has_break_drop",
                                "use_default_drop_when_placed", "door", "stateful")
"""The block affordance tags with an assigned bit (``worldgen/block_affordances.py``; bits 8 to 15 are free)."""
GATHER_BITS: int = 8
"""The gather type's index as the gym's candidates carry it, in binary."""
RECIPE_WIDTH: int = 32
"""The gym's recipe candidate encoding (``observation/v3/policy/layout.py``)."""
ABILITY_SLOTS: int = OBSERVATION_CAPACITY
"""Own ability slots, 16."""
HOLD_THRESHOLDS: int = 2
"""Hold thresholds per ability slot the kit token carries (every 0.7.2 melee slot has at most two)."""
STATUS_FIELDS: tuple[str, ...] = ("remaining_fraction", "cycle_progress", "damage_fraction", "healing_fraction",
                                  "resource_delta_fraction", "speed_multiplier")
STATUS_FLAGS: tuple[str, ...] = ("invulnerable", "disable_movement", "disable_abilities", "ignore_knockback", "debuff",
                                 "disable_sprint", "control_immunity_gated")
"""The status flag word's bits, in bit order (``hytalegym/jax/combat/mechanics/schema/contract.py``)."""
WHERE_FEATURES: tuple[str, ...] = ("forward", "right", "up", "log_distance", "body_bearing_sin", "body_bearing_cos",
                                   "head_bearing_sin", "head_bearing_cos", "head_elevation_sin", "head_elevation_cos")
"""Every spatial token's ``where`` block, computed only by ``state/frames.py`` (``where``): the position in the
body frame, the log distance, the bearing in the body and the head frame, and the elevation from the eye against the
head's pitch."""
MEMORY_FEATURES: tuple[str, ...] = ("remembered", "age", "tagged")
"""On every remembered type: ``remembered`` (not perceived this tick, from the stored memory), ``age`` (log1p of the
ticks since it was last perceived, over log1p of the memory's horizon) and ``tagged`` (the priority the middle layer
gave it with a ``remember`` request, 0 when untagged; PERCEPTION.md section 10.3)."""

PERSONA_FEATURES: tuple[str, ...] = variables.names("self", body_reads=True)
"""The ``persona`` token's columns: the traits the body has a mechanism for (``docs/VIRTUAL_HUMAN.md`` section 15.1),
generated from the variable registry (``contract/variables.py``). The other traits stay in the state layer for the
decision layer and the LLM."""
AFFECT_FEATURES: tuple[str, ...] = variables.names("inner", body_reads=True)
"""The agent's emotional state (``state/affect.py``), generated from the variable registry: fear, anger, startle in
[0, 1]; valence (joy or distress) and the mood's pleasure, arousal and dominance in [-1, 1]; coping, its sense of
control over the threat, in [0, 1]."""
FEELINGS: tuple[str, ...] = ("hostility", "fear", "trust", "anger")
"""The agent's feelings toward one known entity (``state/affect.py``), on that entity's token as ``feel_*``."""
EFFECT_FEATURES: tuple[str, ...] = (
    "debuff", "infinite", "duration_log10", "overlap_overwrite", "overlap_extend", "overlap_ignore",
    "damage_over_time", "modifies_stats", "modifies_damage_taken", "modifies_damage_dealt", "speed_multiplier",
    "invulnerable", "changes_model", "conditional",
)
"""A status effect's registry properties, ``perception/effects.py``'s ``EFFECT_FEATURES``. Copied here, not imported:
that module imports the perception adapter, which imports the encoder, which reads this one. A test asserts equality."""
PLACE_KINDS: tuple[str, ...] = ("stay", "danger", "death", "find", "marked")
"""A long-term place's kinds, ``state/places.py``'s ``PLACE_KINDS`` (copied as above; a test asserts equality)."""
EVENT_KINDS: tuple[str, ...] = ("damage_taken", "damage_dealt", "item_gained", "item_lost", "ate", "request_done",
                                "request_failed", "request_aborted", "death")
"""An event's kinds, ``state/events.py``'s ``EVENT_KINDS`` (copied as above; a test asserts equality)."""

_DESCRIPTION = tuple(f"description_{index}" for index in range(DESCRIPTION_WIDTH))
_RESERVE = tuple(f"reserve_{index}" for index in range(RESERVE_WIDTH))


class SelfTokenV3:
    """The agent itself: its motion in the body frame, contact and medium, its movement states, control, its head and
    heading, vitals, resources, defense, attack state, and the item it holds."""

    FEATURES = ("v_forward", "v_right", "v_up", "speed", "grounded", "fall_speed", "applied_vertical_velocity",
                "motion_delta", "in_fluid", "feet_submerged", "eyes_submerged", "drop_support", "drop_height",
                "medium_known", "submersion_known", "drop_known",
                *(f"moving_{name}" for name in MOVEMENT_STATE_FEATURES), "movement_known",
                "knockback_control_lock", "damage_recovery", "alive",
                "head_yaw_left_sin", "head_yaw_left_cos", "head_pitch_sin", "head_pitch_cos",
                "heading_sin", "heading_cos", "heading_known", "position_known",
                "health_fraction", "max_health_log",
                *(f"resource_{name}" for name in RESOURCES), *(f"resource_{name}_known" for name in RESOURCES),
                *(f"defense_{name}" for name in DEFENSE_FLOAT_FEATURES),
                "attack_executing", "attack_cooldown",
                "holds", *(f"held_item_{name}" for name in ITEM_PROPERTIES_V3), "held_known", *_DESCRIPTION,
                *_RESERVE, "marked")


class ActTokenV3:
    """One action head of the gen-7 layout: which head, the agent's choice on the last tick (as a fraction of the
    head's choices, and whether neutral), whether it changed, and how long it has been held (log1p ticks over log1p
    of 90)."""

    HEADS = tuple(head.name for head in head_layout("gen7"))
    FEATURES = (*(f"head_{name}" for name in HEADS), "choice_fraction", "neutral", "changed", "held_ticks", "marked")


class KitTokenV3:
    """One of the agent's own ability slots: which slot, its live state (duration, cooldown, active progress,
    affordable, legal, costs), and what the compiled loadout says of it (hold thresholds, a charged child, a
    projectile's speed and gravity, the measured reach), which the agent knows of its own weapon."""

    FEATURES = (*(f"slot_{index}" for index in range(ABILITY_SLOTS)), "duration_fraction",
                "cooldown_remaining_fraction", "active_progress", "affordable", "legal",
                *(f"cost_{name}" for name in RESOURCES), "hold_count",
                *(f"hold_threshold_{index}" for index in range(HOLD_THRESHOLDS)), "hold_indefinite", "charged",
                "projectile", "projectile_speed", "projectile_gravity", "reach", "known", "marked")


class StatusTokenV3:
    """One of the agent's own status effects: its timing and per-cycle effects, what it does to me (the flag bits:
    invulnerable, movement, abilities or sprint disabled, knockback ignored, a debuff), and which effect it is: its
    registry properties (``EFFECT_FEATURES``) and its description, looked up by the id the observation publishes in the
    effects registry (``perception/effects.py``; ``docs/VIRTUAL_HUMAN.md`` sections 11.4 and 24). ``effect_known`` is 0
    for an id the registry does not hold, and the effect columns read zero where no registry is given."""

    FEATURES = (*STATUS_FIELDS, *(f"flag_{name}" for name in STATUS_FLAGS), *_RESERVE,
                *(f"effect_{name}" for name in EFFECT_FEATURES),
                *(f"effect_description_{index}" for index in range(DESCRIPTION_WIDTH)), "effect_known", "marked")


class EntityTokenV3:
    """A creature or person, perceived this tick or remembered: where it is, its motion, its facing (relative to my
    body, and toward me), its health and attack (phase, progress, which attack), whether the engine marks it hostile
    to me, what it is (kind, attitude toward me, player: seams until role ids are published, S2), what it holds (S4) and
    its description."""

    FEATURES = (*WHERE_FEATURES, "v_forward", "v_right", "v_up", "yaw_sin", "yaw_cos", "facing_error",
                "head_facing_error", "visible", *MEMORY_FEATURES, "health_fraction", "alive",
                *(f"phase_{phase}" for phase in ATTACK_PHASES), "attack_progress", "attack_index",
                *(f"attitude_{attitude}" for attitude in ATTITUDES), "player", "known", "hostile", *(f"feel_{name}" for name in FEELINGS),
                *(f"kind_{name}" for name in CREATURE_PROPERTIES), *(f"held_kind_{kind}" for kind in HELD_KINDS),
                *(f"held_item_{name}" for name in ITEM_PROPERTIES_V3), *_DESCRIPTION, *_RESERVE, "marked")


class ProjectileTokenV3:
    """A projectile in flight: where it is, its velocity in the body frame, when and how close it passes me
    (``approach_time`` in seconds over 2, ``approach_distance`` in blocks over 4, both clipped), and whose it is."""

    FEATURES = (*WHERE_FEATURES, "v_forward", "v_right", "v_up", "speed", "approach_time", "approach_distance",
                "hostile", "mine", "damage_fraction", "radius_fraction", "age_fraction", "lifetime_remaining",
                "marked")


class HazardTokenV3:
    """An area: where it is, its half extents (world axes), intensity, duration left, damage rate, activation, and
    whose it is."""

    FEATURES = (*WHERE_FEATURES, "half_x", "half_y", "half_z", "intensity", "duration_remaining", "damage_rate",
                "activation", "hostile", "mine", "marked")


class GeometryTokenV3:
    """A local cell, perceived or remembered: where it is, its token kind, its clearance, the cell flags, whether a
    door or other dynamic part blocks it, its traversal edges, its shape (detail boxes), and its light."""

    FEATURES = (*WHERE_FEATURES, *MEMORY_FEATURES, "kind", "provenance", "clearance",
                *(f"flag_{name}" for name in GEOMETRY_FLAGS), "dynamic_blocked",
                *(f"edge_{edge}_{field}" for edge in range(GEOMETRY_EDGES) for field in GEOMETRY_EDGE_FIELDS),
                *(f"box_{box}_{field}" for box in range(GEOMETRY_BOXES) for field in GEOMETRY_BOX_FIELDS),
                *(f"light_{name}" for name in ACTOR_LIGHT_POLICY_FEATURES), "marked")


class BlockCandidateTokenV3:
    """A block the agent could act on, perceived or remembered: where it is, its affordances, its gather type (the
    gym's index bits), the tool quality it needs, its registry properties, my tools' relations to it and its
    description (the block id seam, S5)."""

    FEATURES = (*WHERE_FEATURES, *MEMORY_FEATURES, *(f"affordance_{name}" for name in AFFORDANCES),
                *(f"gather_bit_{bit}" for bit in range(GATHER_BITS)), "required_quality",
                *(f"block_{name}" for name in BLOCK_PROPERTIES), "best_power", "best_quality_margin", "held_power",
                "held_quality_margin", "known", *_DESCRIPTION, *_RESERVE, "marked")


class InventorySlotTokenV3:
    """One slot of the agent's own inventory, as v2's (``surface.InventorySlotTokenV2``)."""

    FEATURES = InventorySlotTokenV2.FEATURES


class PersonaTokenV3:
    """Who the agent is, as far as the body reads it (``PERSONA_FEATURES``), the same for its whole life. One token; it
    also conditions every other token's embedding (``EncoderConfig.condition_types``)."""

    FEATURES = (*PERSONA_FEATURES, "marked")


class AffectTokenV3:
    """How the agent feels now (``AFFECT_FEATURES``). One token; it also conditions every other token's embedding."""

    FEATURES = (*AFFECT_FEATURES, "marked")


class RecipeTokenV3:
    """A recipe the agent could craft here: the gym's candidate encoding."""

    FEATURES = (*(f"recipe_{index}" for index in range(RECIPE_WIDTH)), "marked")


class PlaceTokenV3:
    """A long-term place (``state/places.py``, section 22): where it is (its ``where`` block, from me), its kinds, its
    strength, its visits and the time since its last one (both log-scaled), and whether I am inside it. The body reads
    the few places that matter most to it (section 24): the marked ones first, then the strong and the near."""

    FEATURES = (*WHERE_FEATURES, *(f"place_{kind}" for kind in PLACE_KINDS), "strength", "visits_log",
                "since_visit_log", "inside", *_RESERVE, "marked")


class EventTokenV3:
    """One of the last events (``state/events.py``, section 21): its kind, its age (log ticks), its amount, whether it
    was inferred, whether the other party is a known entity now, and where it happened (its ``where`` block, and
    ``where_known`` 0 when it has no place)."""

    FEATURES = (*WHERE_FEATURES, *(f"event_{kind}" for kind in EVENT_KINDS), "age_log", "amount", "inferred",
                "other_known", "where_known", *_RESERVE, "marked")


TOKEN_TYPES_V3: Mapping[str, type] = {
    "self": SelfTokenV3,
    "act": ActTokenV3,
    "kit": KitTokenV3,
    "status": StatusTokenV3,
    "entity": EntityTokenV3,
    "projectile": ProjectileTokenV3,
    "hazard": HazardTokenV3,
    "geometry": GeometryTokenV3,
    "block_candidate": BlockCandidateTokenV3,
    "inventory_slot": InventorySlotTokenV3,
    "recipe": RecipeTokenV3,
    "persona": PersonaTokenV3,
    "affect": AffectTokenV3,
    "place": PlaceTokenV3,
    "event": EventTokenV3,
}
"""Token type name -> its feature list under ``skill_tokens_v3``. The places and events come last, so no type's
position changes (``docs/VIRTUAL_HUMAN.md`` section 14)."""

TRUNK_TYPE_NAMES_V3: tuple[str, ...] = tuple(TOKEN_TYPES_V3)
"""The token types the trunk reads under v3, in its sequence order."""

REMEMBERED_GEOMETRY_CELLS: int = 32
"""Remembered terrain cells beside this tick's: the background's cache of nearby cells (``docs/VIRTUAL_HUMAN.md``
section 18; 84 until 2026-10-22)."""

CAPACITIES_V3: Mapping[str, int] = {
    "self": 1, "act": len(ActTokenV3.HEADS), "kit": ABILITY_SLOTS, "status": 8, "entity": 16, "projectile": 16,
    "hazard": 8, "geometry": WORLD_GEOMETRY_TOKEN_CAPACITY + REMEMBERED_GEOMETRY_CELLS, "block_candidate": 16 + 16,
    "inventory_slot": 76, "recipe": 16, "persona": 1, "affect": 1, "place": 8, "event": 8,
}
"""Default capacities. Geometry and block candidates hold this tick's plus remembered ones. This tick's geometry is the
gym's world-geometry capacity for the server line (``WORLD_GEOMETRY_TOKEN_CAPACITY``: 96 on 0.7.2 since 04c3b18, from
a 20-seed census, ``docs/VIRTUAL_HUMAN.md`` 18.4; 44 on 0.5.7), read at import like the gym's own; block candidates
are 16. The remembered ones are ``REMEMBERED_GEOMETRY_CELLS`` and 16. Entities hold perceived and remembered together.
The body reads 8 places and the last 8 events (``docs/VIRTUAL_HUMAN.md`` section 24). A capacity is a configuration:
the trunk's parameters never depend on it."""

assert RESOURCE_COUNT == len(RESOURCES)
assert all(cls.FEATURES[-1] == "marked" for cls in TOKEN_TYPES_V3.values())
assert all(len(set(cls.FEATURES)) == len(cls.FEATURES) for cls in TOKEN_TYPES_V3.values())

__all__ = ["ABILITY_SLOTS", "AFFECT_FEATURES", "AFFORDANCES", "CAPACITIES_V3", "DESCRIPTION_WIDTH", "EFFECT_FEATURES",
           "EVENT_KINDS", "FEELINGS", "GATHER_BITS", "GEOMETRY_BOXES", "GEOMETRY_BOX_FIELDS", "GEOMETRY_EDGES",
           "GEOMETRY_EDGE_FIELDS", "GEOMETRY_FLAGS", "HOLD_THRESHOLDS", "ITEM_PROPERTIES_V3", "MEMORY_FEATURES",
           "PERSONA_FEATURES", "PLACE_KINDS", "RECIPE_WIDTH", "REMEMBERED_GEOMETRY_CELLS", "RESERVE_WIDTH", "RESOURCES",
           "STATUS_FIELDS", "STATUS_FLAGS", "TOKEN_SCHEMA_V3", "TOKEN_TYPES_V3", "TRUNK_TYPE_NAMES_V3",
           "WHERE_FEATURES", "ActTokenV3", "AffectTokenV3", "BlockCandidateTokenV3", "EntityTokenV3", "EventTokenV3",
           "GeometryTokenV3", "HazardTokenV3", "InventorySlotTokenV3", "KitTokenV3", "PersonaTokenV3", "PlaceTokenV3",
           "ProjectileTokenV3", "RecipeTokenV3", "SelfTokenV3", "StatusTokenV3"]
