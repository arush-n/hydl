"""JAX self-play contracts for frozen leagues and two-live-agent duels.

League snapshots are immutable opponents. A duel is deliberately separate:
both policy IDs are live, both act in the same environment tick, and each owns
an independent optimizer. Keeping these modes explicit prevents a historical
snapshot from being rewritten accidentally.
"""

from arena.training.selfplay.duel import duel_assignment, make_duel_ppo_trainer
from arena.training.selfplay.native_teacher import (
    NativeTeacherInstall,
    install_native_teacher_policy,
)
from arena.training.selfplay.league import (
    Matchups,
    SelfPlayConfig,
    SelfPlayMetrics,
    SelfPlayState,
    activate_snapshot,
    frozen_opponent_assignment,
    initialize_self_play,
    sample_opponents,
    snapshot_policy,
    update_ratings,
)
from arena.training.selfplay.production import (
    LeagueBookkeeping,
    LeaguePolicyOwner,
    LeaguePolicyRole,
    LeagueSnapshot,
    LeagueUpdatePlan,
    ProductionLeagueConfig,
    ProductionLeagueLoop,
    ProductionLeagueMetrics,
    ProductionLeagueState,
    apply_league_outcomes,
    assignment_content_sha256,
    initialize_league_bookkeeping,
    league_policy_owner_manifest,
    league_update_plan_for_opponent,
    schedule_league_update,
)
from arena.training.selfplay.checkpoint import (
    PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA,
    PRODUCTION_LEAGUE_CHECKPOINT_VERSION,
    load_production_league_checkpoint,
    production_league_checkpoint_contract_manifest,
    production_league_checkpoint_contract_sha256,
    save_production_league_checkpoint,
)

__all__ = [
    "Matchups",
    "NativeTeacherInstall",
    "LeagueBookkeeping",
    "LeaguePolicyOwner",
    "LeaguePolicyRole",
    "LeagueSnapshot",
    "LeagueUpdatePlan",
    "PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA",
    "PRODUCTION_LEAGUE_CHECKPOINT_VERSION",
    "ProductionLeagueConfig",
    "ProductionLeagueLoop",
    "ProductionLeagueMetrics",
    "ProductionLeagueState",
    "SelfPlayConfig",
    "SelfPlayMetrics",
    "SelfPlayState",
    "activate_snapshot",
    "apply_league_outcomes",
    "assignment_content_sha256",
    "duel_assignment",
    "frozen_opponent_assignment",
    "initialize_self_play",
    "install_native_teacher_policy",
    "initialize_league_bookkeeping",
    "league_policy_owner_manifest",
    "league_update_plan_for_opponent",
    "load_production_league_checkpoint",
    "make_duel_ppo_trainer",
    "sample_opponents",
    "snapshot_policy",
    "production_league_checkpoint_contract_manifest",
    "production_league_checkpoint_contract_sha256",
    "save_production_league_checkpoint",
    "schedule_league_update",
    "update_ratings",
]
