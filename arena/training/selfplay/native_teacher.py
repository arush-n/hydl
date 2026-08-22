"""Install a contract-checked native behavior policy as a frozen JAX teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from arena.training.imitation.native_replay import NativeBehaviorPolicy
from arena.training.imitation.native_transfer import (
    NativeArsenalAbilityBinding,
    transplant_native_behavior_policy,
)
from hytalegym.jax.training.types import PPOConfig, RecurrentPolicyParams


@dataclass(frozen=True, slots=True)
class NativeTeacherInstall:
    """Audit receipt for one immutable teacher row in a policy bank."""

    policy_id: int
    checkpoint_sha256: str
    style: tuple[str, str] | None
    transfer: dict[str, Any]
    trainable: bool = False


def install_native_teacher_policy(
    policy_bank: RecurrentPolicyParams,
    policy: NativeBehaviorPolicy,
    target_config: PPOConfig,
    combat_params: Any,
    *,
    policy_id: int,
    style: tuple[str, str] | None = None,
    ability_binding: NativeArsenalAbilityBinding | None = None,
    target_ability_slot_contract_sha256: str | None = None,
) -> tuple[RecurrentPolicyParams, NativeTeacherInstall]:
    """Transplant a native policy into one frozen multi-actor policy-bank row.

    The caller remains responsible for assigning this row with
    ``trainable=False``.  The receipt states that invariant explicitly and the
    production league already preserves frozen rows byte-for-byte.
    """

    if isinstance(policy_id, bool) or not isinstance(policy_id, int):
        raise TypeError("native teacher policy_id must be an integer")
    leaves = jax.tree_util.tree_leaves(policy_bank)
    if not leaves or any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("native teacher requires a policy bank with a K axis")
    count = leaves[0].shape[0]
    if any(leaf.shape[0] != count for leaf in leaves):
        raise ValueError("policy-bank leaves must share their K axis")
    if not 0 <= policy_id < count:
        raise ValueError("native teacher policy_id is outside the policy bank")

    target = jax.tree_util.tree_map(lambda value: value[policy_id], policy_bank)
    transplanted, transfer = transplant_native_behavior_policy(
        policy,
        target,
        target_config,
        combat_params,
        style,
        ability_binding=ability_binding,
        target_ability_slot_contract_sha256=(target_ability_slot_contract_sha256),
    )
    updated = jax.tree_util.tree_map(
        lambda bank, value: bank.at[policy_id].set(
            jnp.asarray(value, dtype=bank.dtype)
        ),
        policy_bank,
        transplanted,
    )
    return updated, NativeTeacherInstall(
        policy_id=policy_id,
        checkpoint_sha256=policy.checkpoint_sha256,
        style=style,
        transfer=transfer,
    )


__all__ = ["NativeTeacherInstall", "install_native_teacher_policy"]
