"""Reward streams that compose into a lesson.

These are not standalone contracts: none of them owns an action scope or a
reset. They are the reward *terms* a training run mixes -- engine-grounded
combat shaping, an online behaviour prior, exact-imitation alignment, and the
learned exploration/imitation kernels (RND, RIDE, GAIL/AIRL).

A full lesson -- reward law plus the heads it trains plus its manifest -- lives
in :mod:`arena.training.contracts` instead.
"""

from __future__ import annotations

from arena.training.rewards.behavior_prior import (
    BehaviorPriorConfig,
    BehaviorPriorState,
    make_behavior_prior_environment,
)
from arena.training.rewards.combat_fundamentals import (
    COMBAT_ATTACK_ENVELOPE_SCHEMA,
    COMBAT_FUNDAMENTALS_REWARD_SCHEMA,
    CombatAttackEnvelope,
    CombatFundamentalsConfig,
    CombatFundamentalsState,
    CombatFundamentalsTerms,
    arsenal_basic_attack_envelope,
    combat_fundamentals_training_feature,
    combat_fundamentals_training_profile,
    make_combat_fundamentals_environment,
)
from arena.training.rewards.exact_imitation import (
    ExactImitationConfig,
    ExactImitationTerms,
    exact_imitation_reward,
)
from arena.training.rewards.learned_rewards import (
    RewardMixConfig,
    adversarial_imitation_reward,
    discriminator_loss,
    impact_reward,
    mix_rewards,
    prediction_error,
    prediction_loss,
    prediction_reward,
)


__all__ = [
    "BehaviorPriorConfig",
    "BehaviorPriorState",
    "COMBAT_ATTACK_ENVELOPE_SCHEMA",
    "COMBAT_FUNDAMENTALS_REWARD_SCHEMA",
    "CombatAttackEnvelope",
    "CombatFundamentalsConfig",
    "CombatFundamentalsState",
    "CombatFundamentalsTerms",
    "ExactImitationConfig",
    "ExactImitationTerms",
    "RewardMixConfig",
    "adversarial_imitation_reward",
    "arsenal_basic_attack_envelope",
    "combat_fundamentals_training_feature",
    "combat_fundamentals_training_profile",
    "discriminator_loss",
    "exact_imitation_reward",
    "impact_reward",
    "make_behavior_prior_environment",
    "make_combat_fundamentals_environment",
    "mix_rewards",
    "prediction_error",
    "prediction_loss",
    "prediction_reward",
]
