"""Composable arena skill contracts for an end-to-end combat agent.

A *contract* here is one lesson, complete: a tunable reward law, the action
heads it trains, and a pure scoring function over engine evidence. Each module
is named after its lesson and publishes that lesson's pieces under the same
names -- see :mod:`arena.training.contracts.common` for the exact shape and for
the reward conventions every module obeys.

Combat lessons (1v1)
    ``guard_duel``      attack timing versus block timing, both sides trainable
    ``punish_window``   counter-attacking inside the opponent's recovery
    ``ability_landing`` aimed ability usage against a deterministic mover
    ``evasion``         damage avoidance without leaving the engagement band

Locomotion lessons
    ``pursuit``          chase and flee against a sampled actor
    ``checkpoint_route`` reach a generated goal block as fast as possible

Support
    ``common``      shared validators, ``head_scope``, ``normalized_manifest``
    ``observation`` the omniscient target view every lesson trains against
    ``probes``      deterministic policy comparisons that check a reward law is
                    not farmable; run with
                    ``python -m arena.training.contracts.probes``

Not to be confused with :mod:`arena.training.runtime.contracts`, a different
thing under a similar name: that module versions a *training run* (metrics,
stop-loss, feature programs), while this package defines what a policy is being
paid to do.

Only ``pursuit`` is bound to a collector. The other lessons expose their config,
scoring function and action scope; wiring them into a rollout is the caller's
decision, exactly as ``bind_pursuit_collector`` does for pursuit.
"""

from __future__ import annotations

from arena.training.contracts.ability_landing import (
    ABILITY_LANDING_SCHEMA,
    ABILITY_NONE,
    AbilityLandingConfig,
    AbilityLandingSignals,
    ability_landing_action_scope,
    ability_landing_signals,
)
from arena.training.contracts.checkpoint_route import (
    CHECKPOINT_ROUTE_SCHEMA,
    CheckpointRouteConfig,
    CheckpointRouteSignals,
    checkpoint_route_action_scope,
    checkpoint_route_signals,
    goal_distance,
)
from arena.training.contracts.catalog import (
    CATALOG_SCHEMA,
    REGISTRATION_SCHEMA,
    training_contract,
    training_contracts,
)
from arena.training.contracts.common import (
    LOCOMOTION_HEADS,
    HorizonConfig,
    band_fraction,
    contract_hash,
    head_scope,
    normalized_manifest,
)
from arena.training.contracts.evasion import (
    DODGE_NONE,
    EVASION_SCHEMA,
    EvasionConfig,
    EvasionSignals,
    evasion_action_scope,
    evasion_signals,
)
from arena.training.contracts.guard_duel import (
    GUARD_DUEL_SCHEMA,
    GuardDuelConfig,
    GuardDuelSignals,
    advance_guard_streak,
    guard_duel_attacker_action_scope,
    guard_duel_blocker_action_scope,
    guard_duel_signals,
)
from arena.training.contracts.observation import (
    OMNISCIENT_CONTEXT_SCHEMA,
    make_omniscient_observation_transform,
    omniscient_context_manifest,
)
from arena.training.contracts.punish_window import (
    PUNISH_WINDOW_SCHEMA,
    PunishWindowConfig,
    PunishWindowSignals,
    punish_window_action_scope,
    punish_window_signals,
)
from arena.training.contracts.pursuit import (
    PURSUIT_CONTEXT_SCHEMA,
    PURSUIT_DEFAULT_TICKS,
    PURSUIT_MAXIMUM_TICKS,
    PURSUIT_STAGE_SCHEMA,
    PursuitLessonSignals,
    PursuitStageConfig,
    bind_pursuit_collector,
    make_pursuit_action_mask_transform,
    make_pursuit_observation_transform,
    make_pursuit_transition_transform,
    pursuit_learner_action_scope,
    pursuit_lesson_signals,
)


#: Every lesson config in this package, keyed by its module name.
CONTRACT_CONFIGS = {
    "guard_duel": GuardDuelConfig,
    "punish_window": PunishWindowConfig,
    "ability_landing": AbilityLandingConfig,
    "evasion": EvasionConfig,
    "checkpoint_route": CheckpointRouteConfig,
    "pursuit": PursuitStageConfig,
}


def contract_manifest() -> dict[str, object]:
    """Describe every default contract, for recording on a training artifact.

    ``pursuit`` is included but does not share the normalized manifest
    skeleton: it predates :func:`normalized_manifest` and publishes its own key
    set. Compare the other five key-by-key; compare pursuit only with itself.
    """

    return {
        "schema": "arena-contract-suite-v1",
        "observation": omniscient_context_manifest(),
        "contracts": {
            name: config().manifest() for name, config in CONTRACT_CONFIGS.items()
        },
    }


__all__ = [
    "ABILITY_LANDING_SCHEMA",
    "ABILITY_NONE",
    "AbilityLandingConfig",
    "AbilityLandingSignals",
    "CHECKPOINT_ROUTE_SCHEMA",
    "CATALOG_SCHEMA",
    "CONTRACT_CONFIGS",
    "CheckpointRouteConfig",
    "CheckpointRouteSignals",
    "DODGE_NONE",
    "EVASION_SCHEMA",
    "EvasionConfig",
    "EvasionSignals",
    "GUARD_DUEL_SCHEMA",
    "GuardDuelConfig",
    "GuardDuelSignals",
    "HorizonConfig",
    "LOCOMOTION_HEADS",
    "OMNISCIENT_CONTEXT_SCHEMA",
    "PUNISH_WINDOW_SCHEMA",
    "PURSUIT_CONTEXT_SCHEMA",
    "PURSUIT_DEFAULT_TICKS",
    "PURSUIT_MAXIMUM_TICKS",
    "PURSUIT_STAGE_SCHEMA",
    "REGISTRATION_SCHEMA",
    "PunishWindowConfig",
    "PunishWindowSignals",
    "PursuitLessonSignals",
    "PursuitStageConfig",
    "ability_landing_action_scope",
    "ability_landing_signals",
    "advance_guard_streak",
    "band_fraction",
    "bind_pursuit_collector",
    "checkpoint_route_action_scope",
    "checkpoint_route_signals",
    "contract_hash",
    "contract_manifest",
    "evasion_action_scope",
    "evasion_signals",
    "goal_distance",
    "guard_duel_attacker_action_scope",
    "guard_duel_blocker_action_scope",
    "guard_duel_signals",
    "head_scope",
    "make_omniscient_observation_transform",
    "make_pursuit_action_mask_transform",
    "make_pursuit_observation_transform",
    "make_pursuit_transition_transform",
    "normalized_manifest",
    "omniscient_context_manifest",
    "punish_window_action_scope",
    "punish_window_signals",
    "training_contract",
    "training_contracts",
    "pursuit_learner_action_scope",
    "pursuit_lesson_signals",
]
