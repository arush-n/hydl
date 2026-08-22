"""Array-only ranged-controller state and command PyTrees."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class RangedControllerRules(NamedTuple):
    weapon_family: Array
    active: Array
    max_ammo: Array
    max_durability: Array
    durability_loss_per_projectile: Array
    signature_energy_cost: Array


class RangedControllerState(NamedTuple):
    phase: Array
    phase_elapsed_seconds: Array
    primary_was_held: Array
    primary_cooldown_seconds: Array
    pending_inventory_arrow: Array
    arrow_inventory: Array
    arrow_add_capacity: Array
    ammo: Array
    durability: Array
    signature_energy: Array
    signature_charges: Array
    failure_bits: Array


class RangedControllerCommands(NamedTuple):
    primary_held: Array
    signature_pressed: Array
    reload_pressed: Array
    secondary_pressed: Array
    swap_away: Array
    interrupted: Array


class RangedActionMask(NamedTuple):
    legal: Array


class RangedControllerInfo(NamedTuple):
    ability_slot: Array
    ability_requested: Array
    charge_level: Array
    projectile_count: Array
    projectile_damage_multiplier: Array
    signature_activated: Array
    ammo_loaded: Array
    arrows_removed: Array
    arrows_returned: Array
    arrows_dropped: Array
    swap_accepted: Array
    swap_blocked: Array
    failure_bits: Array
    valid: Array
