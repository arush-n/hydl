"""Exact joint distribution for the factored Arsenal action surface.

The public policy keeps twelve fixed categorical heads, but six of those
heads can request a Hytale standard-input root.  Sampling those six heads
independently can request mutually exclusive roots in one control tick.  This
module conditions the product of their categorical logits on the decoder's
joint legality rule while leaving motion, look, jump, Dodge, and the retained
block-trigger factor independent.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_ACTION_SIZE,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH_ATTACK,
    SKILL_ATTACK,
    SKILL_RETREAT_ATTACK,
)


ARSENAL_STANDARD_ROOT_DISTRIBUTION = "arsenal_standard_root_v1"

_MASKED_LOGIT_CUTOFF = -5.0e8
_INVALID_LOG_PROBABILITY = -1.0e9

_BASE = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("base_action")
_ABILITY = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("ability_none_plus_slots")
_GUARD = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("guard_off_on")
_JUMP = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("jump_off_on")
_LOCOMOTION = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("locomotion_gait_compass")
_YAW = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("yaw_delta_bins")
_PITCH = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("pitch_delta_bins")
_USE = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("use_off_on")
_BLOCK_TRIGGER = ARSENAL_POLICY_ACTION_HEAD_NAMES.index(
    "block_primary_secondary_trigger"
)
_BLOCK = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("block_none_plus_candidates")

_CATEGORY_NONE = 0
_CATEGORY_BASE_ATTACK = 1
_CATEGORY_ABILITY = 2
_CATEGORY_GUARD = 3
_CATEGORY_BLOCK_OR_USE = 4
_CATEGORY_COUNT = 5

_NON_ROOT_HEADS = (_LOCOMOTION, _JUMP, _YAW, _PITCH, _BLOCK_TRIGGER)
_ATTACK_CHOICES = (SKILL_ATTACK, SKILL_APPROACH_ATTACK, SKILL_RETREAT_ATTACK)


def sample_arsenal_action_factors(
    key: jax.Array,
    logits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample only decoder-valid Arsenal factors and return their joint log-p."""

    values = _validate_logits(logits)
    heads = _split_heads(values)
    category_logits, _, _ = _root_category_statistics(heads)
    keys = jax.random.split(key, len(heads) + 1)
    category = jax.random.categorical(
        keys[0],
        category_logits,
        axis=-1,
    ).astype(jnp.int32)
    allowed = _allowed_masks_for_category(heads, category)
    factors = jnp.stack(
        tuple(
            _sample_head(head_key, head, head_allowed)
            for head_key, head, head_allowed in zip(
                keys[1:],
                heads,
                allowed,
                strict=True,
            )
        ),
        axis=-1,
    )
    return factors, arsenal_action_log_probabilities(values, factors)


def arsenal_action_log_probabilities(
    logits: jax.Array,
    factors: jax.Array,
) -> jax.Array:
    """Return log-p under the product distribution conditioned on legality."""

    values = _validate_logits(logits)
    selected = _validate_factors(factors, values.shape[:-1])
    heads = _split_heads(values)
    category_logits, _, _ = _root_category_statistics(heads)
    root_log_normalizer = jax.nn.logsumexp(category_logits, axis=-1)

    selected_score = jnp.zeros(values.shape[:-1], dtype=values.dtype)
    selected_available = jnp.ones(values.shape[:-1], dtype=jnp.bool_)
    for index, head in enumerate(heads):
        choice = selected[..., index]
        safe_choice = jnp.clip(choice, 0, head.shape[-1] - 1)
        chosen_logit = jnp.take_along_axis(
            head,
            safe_choice[..., None],
            axis=-1,
        )[..., 0]
        selected_score += chosen_logit
        selected_available &= (
            (choice >= 0)
            & (choice < head.shape[-1])
            & _logit_available(chosen_logit)
        )

    non_root_log_normalizer = jnp.zeros(
        values.shape[:-1],
        dtype=values.dtype,
    )
    non_root_available = jnp.ones(values.shape[:-1], dtype=jnp.bool_)
    for index in _NON_ROOT_HEADS:
        log_mass, _, _ = _subset_statistics(
            heads[index],
            jnp.ones((heads[index].shape[-1],), dtype=jnp.bool_),
        )
        non_root_log_normalizer += log_mass
        non_root_available &= jnp.isfinite(log_mass)

    legal = (
        arsenal_standard_root_legal(selected)
        & selected_available
        & jnp.isfinite(root_log_normalizer)
        & non_root_available
    )
    log_probability = (
        selected_score
        - root_log_normalizer
        - non_root_log_normalizer
    )
    return jnp.where(
        legal,
        log_probability,
        jnp.asarray(_INVALID_LOG_PROBABILITY, dtype=values.dtype),
    )


def arsenal_action_entropy(logits: jax.Array) -> jax.Array:
    """Return exact entropy of the legality-conditioned joint distribution."""

    values = _validate_logits(logits)
    heads = _split_heads(values)
    category_logits, conditional_entropies, _ = _root_category_statistics(heads)
    category_probabilities, category_entropy = _categorical_statistics(
        category_logits,
    )
    entropy = category_entropy + jnp.sum(
        category_probabilities * conditional_entropies,
        axis=-1,
    )
    for index in _NON_ROOT_HEADS:
        _, head_entropy, _ = _subset_statistics(
            heads[index],
            jnp.ones((heads[index].shape[-1],), dtype=jnp.bool_),
        )
        entropy += head_entropy
    return entropy


def greedy_arsenal_action_factors(logits: jax.Array) -> jax.Array:
    """Return the highest-scoring legal joint factor vector."""

    values = _validate_logits(logits)
    heads = _split_heads(values)
    _, _, category_scores = _root_category_statistics(heads)
    category = jnp.argmax(category_scores, axis=-1).astype(jnp.int32)
    allowed = _allowed_masks_for_category(heads, category)
    return jnp.stack(
        tuple(
            _greedy_head(head, head_allowed)
            for head, head_allowed in zip(heads, allowed, strict=True)
        ),
        axis=-1,
    )


def arsenal_standard_root_legal(factors: jax.Array) -> jax.Array:
    """Check factor ranges and the decoder's one-standard-root rule."""

    selected = jnp.asarray(factors)
    if selected.ndim < 1 or selected.shape[-1] != len(
        ARSENAL_POLICY_ACTION_HEAD_SIZES
    ):
        raise ValueError(
            "Arsenal factors must end with exactly "
            f"{len(ARSENAL_POLICY_ACTION_HEAD_SIZES)} heads"
        )
    if not jnp.issubdtype(selected.dtype, jnp.integer):
        raise TypeError("Arsenal factors must use an integer dtype")
    selected = selected.astype(jnp.int32)
    in_range = jnp.ones(selected.shape[:-1], dtype=jnp.bool_)
    for index, size in enumerate(ARSENAL_POLICY_ACTION_HEAD_SIZES):
        value = selected[..., index]
        in_range &= (value >= 0) & (value < size)

    base = selected[..., _BASE]
    legacy_attack = (
        (base == SKILL_ATTACK)
        | (base == SKILL_APPROACH_ATTACK)
        | (base == SKILL_RETREAT_ATTACK)
    )
    ability = selected[..., _ABILITY] > 0
    guard = selected[..., _GUARD] == 1
    use = selected[..., _USE] == 1
    block_target = selected[..., _BLOCK] > 0
    block_request = block_target & ~use
    standard_root_count = (
        legacy_attack.astype(jnp.int32)
        + ability.astype(jnp.int32)
        + guard.astype(jnp.int32)
        + use.astype(jnp.int32)
        + block_request.astype(jnp.int32)
    )
    return in_range & (standard_root_count <= 1) & (~use | block_target)


def validate_arsenal_standard_root_distribution(
    head_sizes: Sequence[int],
    transport: str,
) -> None:
    """Fail early when the distribution is bound to a different ABI."""

    if tuple(head_sizes) != tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES):
        raise ValueError(
            f"{ARSENAL_STANDARD_ROOT_DISTRIBUTION} requires Arsenal head sizes "
            f"{ARSENAL_POLICY_ACTION_HEAD_SIZES}"
        )
    if transport != "factors":
        raise ValueError(
            f"{ARSENAL_STANDARD_ROOT_DISTRIBUTION} requires factor transport"
        )


def _validate_logits(logits: jax.Array) -> jax.Array:
    values = jnp.asarray(logits)
    if values.ndim < 1 or values.shape[-1] != ARSENAL_POLICY_ACTION_SIZE:
        raise ValueError(
            "Arsenal logits must end with the exact published width "
            f"{ARSENAL_POLICY_ACTION_SIZE}"
        )
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("Arsenal logits must use a floating dtype")
    return values


def _validate_factors(
    factors: jax.Array,
    batch_shape: tuple[int, ...],
) -> jax.Array:
    selected = jnp.asarray(factors)
    expected_shape = batch_shape + (len(ARSENAL_POLICY_ACTION_HEAD_SIZES),)
    if selected.shape != expected_shape:
        raise ValueError(
            "Arsenal factors must match logits batch axes and head count: "
            f"{selected.shape} != {expected_shape}"
        )
    if not jnp.issubdtype(selected.dtype, jnp.integer):
        raise TypeError("Arsenal factors must use an integer dtype")
    return selected.astype(jnp.int32)


def _split_heads(logits: jax.Array) -> tuple[jax.Array, ...]:
    total = 0
    boundaries = []
    for size in ARSENAL_POLICY_ACTION_HEAD_SIZES[:-1]:
        total += size
        boundaries.append(total)
    return tuple(jnp.split(logits, tuple(boundaries), axis=-1))


def _logit_available(logit: jax.Array) -> jax.Array:
    return jnp.isfinite(logit) & (logit > _MASKED_LOGIT_CUTOFF)


def _subset_statistics(
    head: jax.Array,
    allowed: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    valid = _logit_available(head) & jnp.asarray(allowed, dtype=jnp.bool_)
    any_valid = jnp.any(valid, axis=-1)
    fallback = jnp.zeros_like(valid).at[..., 0].set(True)
    safe_valid = valid | (~any_valid)[..., None] & fallback
    masked = jnp.where(safe_valid, head, -jnp.inf)
    log_mass = jax.nn.logsumexp(masked, axis=-1)
    log_probabilities = jax.nn.log_softmax(masked, axis=-1)
    probabilities = jnp.where(safe_valid, jnp.exp(log_probabilities), 0.0)
    safe_log_probabilities = jnp.where(
        safe_valid,
        log_probabilities,
        0.0,
    )
    entropy = -jnp.sum(
        probabilities * safe_log_probabilities,
        axis=-1,
    )
    maximum = jnp.max(masked, axis=-1)
    return (
        jnp.where(any_valid, log_mass, -jnp.inf),
        jnp.where(any_valid, entropy, 0.0),
        jnp.where(any_valid, maximum, -jnp.inf),
    )


def _categorical_statistics(
    logits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    valid = jnp.isfinite(logits)
    any_valid = jnp.any(valid, axis=-1)
    fallback = jnp.zeros_like(valid).at[..., 0].set(True)
    safe_valid = valid | (~any_valid)[..., None] & fallback
    masked = jnp.where(safe_valid, logits, -jnp.inf)
    log_probabilities = jax.nn.log_softmax(masked, axis=-1)
    probabilities = jnp.where(safe_valid, jnp.exp(log_probabilities), 0.0)
    safe_log_probabilities = jnp.where(
        safe_valid,
        log_probabilities,
        0.0,
    )
    entropy = -jnp.sum(
        probabilities * safe_log_probabilities,
        axis=-1,
    )
    return probabilities, jnp.where(any_valid, entropy, 0.0)


def _head_masks(head: jax.Array) -> tuple[jax.Array, jax.Array]:
    choices = jnp.arange(head.shape[-1], dtype=jnp.int32)
    return choices == 0, choices > 0


def _base_masks(head: jax.Array) -> tuple[jax.Array, jax.Array]:
    choices = jnp.arange(head.shape[-1], dtype=jnp.int32)
    attack = jnp.zeros_like(choices, dtype=jnp.bool_)
    for choice in _ATTACK_CHOICES:
        attack |= choices == choice
    return ~attack, attack


def _root_category_statistics(
    heads: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    base_neutral_mask, base_attack_mask = _base_masks(heads[_BASE])
    ability_none_mask, ability_active_mask = _head_masks(heads[_ABILITY])
    guard_off_mask, guard_on_mask = _head_masks(heads[_GUARD])
    use_off_mask, _ = _head_masks(heads[_USE])
    use_all_mask = jnp.ones((heads[_USE].shape[-1],), dtype=jnp.bool_)
    block_none_mask, block_active_mask = _head_masks(heads[_BLOCK])

    base_neutral = _subset_statistics(heads[_BASE], base_neutral_mask)
    base_attack = _subset_statistics(heads[_BASE], base_attack_mask)
    ability_none = _subset_statistics(heads[_ABILITY], ability_none_mask)
    ability_active = _subset_statistics(heads[_ABILITY], ability_active_mask)
    guard_off = _subset_statistics(heads[_GUARD], guard_off_mask)
    guard_on = _subset_statistics(heads[_GUARD], guard_on_mask)
    use_off = _subset_statistics(heads[_USE], use_off_mask)
    use_all = _subset_statistics(heads[_USE], use_all_mask)
    block_none = _subset_statistics(heads[_BLOCK], block_none_mask)
    block_active = _subset_statistics(heads[_BLOCK], block_active_mask)

    fixed_none_mass = (
        ability_none[0]
        + guard_off[0]
        + use_off[0]
        + block_none[0]
    )
    fixed_none_score = (
        ability_none[2]
        + guard_off[2]
        + use_off[2]
        + block_none[2]
    )
    category_logits = jnp.stack(
        (
            base_neutral[0] + fixed_none_mass,
            base_attack[0] + fixed_none_mass,
            base_neutral[0] + ability_active[0] + (
                guard_off[0] + use_off[0] + block_none[0]
            ),
            base_neutral[0] + guard_on[0] + (
                ability_none[0] + use_off[0] + block_none[0]
            ),
            base_neutral[0] + use_all[0] + block_active[0] + (
                ability_none[0] + guard_off[0]
            ),
        ),
        axis=-1,
    )
    conditional_entropies = jnp.stack(
        (
            base_neutral[1],
            base_attack[1],
            base_neutral[1] + ability_active[1],
            base_neutral[1],
            base_neutral[1] + use_all[1] + block_active[1],
        ),
        axis=-1,
    )
    category_scores = jnp.stack(
        (
            base_neutral[2] + fixed_none_score,
            base_attack[2] + fixed_none_score,
            base_neutral[2] + ability_active[2] + (
                guard_off[2] + use_off[2] + block_none[2]
            ),
            base_neutral[2] + guard_on[2] + (
                ability_none[2] + use_off[2] + block_none[2]
            ),
            base_neutral[2] + use_all[2] + block_active[2] + (
                ability_none[2] + guard_off[2]
            ),
        ),
        axis=-1,
    )
    if category_logits.shape[-1] != _CATEGORY_COUNT:
        raise RuntimeError("standard-root category count drift")
    return category_logits, conditional_entropies, category_scores


def _allowed_masks_for_category(
    heads: tuple[jax.Array, ...],
    category: jax.Array,
) -> tuple[jax.Array, ...]:
    base_neutral, base_attack = _base_masks(heads[_BASE])
    ability_none, ability_active = _head_masks(heads[_ABILITY])
    guard_off, guard_on = _head_masks(heads[_GUARD])
    use_off, _ = _head_masks(heads[_USE])
    use_all = jnp.ones((heads[_USE].shape[-1],), dtype=jnp.bool_)
    block_none, block_active = _head_masks(heads[_BLOCK])
    all_masks = tuple(
        jnp.ones((head.shape[-1],), dtype=jnp.bool_) for head in heads
    )
    allowed = list(all_masks)
    allowed[_BASE] = _select_category_mask(
        category,
        _CATEGORY_BASE_ATTACK,
        base_attack,
        base_neutral,
    )
    allowed[_ABILITY] = _select_category_mask(
        category,
        _CATEGORY_ABILITY,
        ability_active,
        ability_none,
    )
    allowed[_GUARD] = _select_category_mask(
        category,
        _CATEGORY_GUARD,
        guard_on,
        guard_off,
    )
    allowed[_USE] = _select_category_mask(
        category,
        _CATEGORY_BLOCK_OR_USE,
        use_all,
        use_off,
    )
    allowed[_BLOCK] = _select_category_mask(
        category,
        _CATEGORY_BLOCK_OR_USE,
        block_active,
        block_none,
    )
    return tuple(allowed)


def _select_category_mask(
    category: jax.Array,
    active_category: int,
    active_mask: jax.Array,
    inactive_mask: jax.Array,
) -> jax.Array:
    return jnp.where(
        (category == active_category)[..., None],
        active_mask,
        inactive_mask,
    )


def _sample_head(
    key: jax.Array,
    head: jax.Array,
    allowed: jax.Array,
) -> jax.Array:
    valid = _logit_available(head) & allowed
    any_valid = jnp.any(valid, axis=-1)
    fallback = jnp.zeros_like(valid).at[..., 0].set(True)
    safe_valid = valid | (~any_valid)[..., None] & fallback
    masked = jnp.where(safe_valid, head, -jnp.inf)
    return jax.random.categorical(key, masked, axis=-1).astype(jnp.int32)


def _greedy_head(head: jax.Array, allowed: jax.Array) -> jax.Array:
    valid = _logit_available(head) & allowed
    any_valid = jnp.any(valid, axis=-1)
    fallback = jnp.zeros_like(valid).at[..., 0].set(True)
    safe_valid = valid | (~any_valid)[..., None] & fallback
    masked = jnp.where(safe_valid, head, -jnp.inf)
    return jnp.argmax(masked, axis=-1).astype(jnp.int32)


__all__ = [
    "ARSENAL_STANDARD_ROOT_DISTRIBUTION",
    "arsenal_action_entropy",
    "arsenal_action_log_probabilities",
    "arsenal_standard_root_legal",
    "greedy_arsenal_action_factors",
    "sample_arsenal_action_factors",
    "validate_arsenal_standard_root_distribution",
]
