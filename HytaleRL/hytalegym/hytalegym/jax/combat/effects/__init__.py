"""Fixed-capacity projectiles and hazards isolated from world geometry."""

from hytalegym.jax.combat.effects.schema.contract import (
    COMBAT_EFFECTS_SCHEMA,
    COMBAT_EFFECTS_VERSION,
    EFFECT_FAILURE_HAZARD_OVERFLOW,
    EFFECT_FAILURE_INVALID_COMMAND,
    EFFECT_FAILURE_INVALID_STATE,
    EFFECT_FAILURE_PROJECTILE_OVERFLOW,
    EFFECT_FAILURE_UNSUPPORTED_COLLISION,
    HAZARD_CAPACITY,
    HAZARD_KIND_AABB_DAMAGE,
    HAZARD_KIND_NONE,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
    OUTLANDER_ARROW_ASSET_SHA256,
    OUTLANDER_ARROW_AUTHORED_LIFETIME_SECONDS,
    OUTLANDER_ARROW_DAMAGE,
    OUTLANDER_ARROW_DEAD_TIME_SECONDS,
    OUTLANDER_ARROW_DEPTH_OFFSET,
    OUTLANDER_ARROW_GRAVITY,
    OUTLANDER_ARROW_MODEL_HALF_EXTENT,
    OUTLANDER_ARROW_MUZZLE_VELOCITY,
    OUTLANDER_ARROW_SERVER_DESPAWN_SECONDS,
    OUTLANDER_ARROW_TERMINAL_VELOCITY,
    OUTLANDER_ARROW_VERTICAL_OFFSET,
    PROJECTILE_CAPACITY,
    PROJECTILE_FLAG_IMPACTED,
    PROJECTILE_KIND_NONE,
    PROJECTILE_KIND_OUTLANDER_HUNTER_ARROW,
)
from hytalegym.jax.combat.effects.factory import (
    empty_effect_commands,
    empty_effects_state,
)
from hytalegym.jax.combat.effects.runtime.learner import (
    LearnerCombatEffectsTransitionV2,
    reset_learner_combat_effects_batch_v2,
    step_learner_combat_effects_batch_v2,
)
from hytalegym.jax.combat.effects.catalog.profiles import (
    outlander_hunter_arrow_commands,
)
from hytalegym.jax.combat.effects.runtime.engine import (
    reset_combat_effects_batch,
    rollout_batch_effects,
    rollout_batch_melee_loadout_effects,
    step_batch_effects,
    step_batch_melee_loadout_effects,
)
from hytalegym.jax.combat.effects.runtime.scene import (
    combat_effects_scene,
    merge_combat_effects_scene,
)
from hytalegym.jax.combat.effects.schema.spec import (
    combat_effects_contract_json,
    combat_effects_contract_manifest,
    combat_effects_contract_sha256,
)
from hytalegym.jax.combat.effects.schema.types import (
    CombatEffectCommands,
    CombatEffectsEnvironmentState,
    CombatEffectsInfo,
    CombatEffectsState,
    CombatEffectsTrajectory,
    CombatEffectsTransition,
    HazardSpawn,
    HazardState,
    ProjectileSpawn,
    ProjectileState,
)
from hytalegym.jax.combat.effects.schema.validation import validate_effect_commands

__all__ = [name for name in globals() if not name.startswith("_")]
