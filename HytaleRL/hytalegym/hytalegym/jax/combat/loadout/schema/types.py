"""Array-only types for per-environment melee equipment."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class MeleeLoadoutBatch(NamedTuple):
    """Episode-pinned weapon profiles with batch-first fixed arrays."""

    weapon_id: Array
    weapon_kind: Array
    equipped_mask: Array

    attack_mask: Array
    hit_delay_ticks: Array
    range_blocks: Array
    half_angle_degrees: Array
    damage: Array
    cooldown_min_seconds: Array
    cooldown_max_seconds: Array
    requires_line_of_sight: Array

    overflow: Array


class MeleeLoadoutObservation(NamedTuple):
    """Normalized learner projection for one episode-pinned loadout."""

    schema_version: Array
    weapon_i32: Array
    equipped_mask: Array
    attack_f32: Array
    attack_i32: Array
    attack_mask: Array
    valid: Array
    failure_bits: Array
