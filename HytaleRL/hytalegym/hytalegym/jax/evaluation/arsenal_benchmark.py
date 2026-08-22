"""Frozen flat Arsenal benchmarks with framework-neutral action sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import (
    CELL_COUNT,
    FLUID_MOVEMENT_FEATURES,
    MOVEMENT_FEATURES,
)
from hytalegym.jax.combat import (
    AGENT_ENTITY,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_ACTION_SIZE,
    ARSENAL_POLICY_OBSERVATION_SIZE,
    PROFILE_NAMES,
    arsenal_policy_contract_sha256,
    arsenal_runtime_config,
    combat_arsenal_contract_sha256,
    default_combat_params,
    hytale_0_5_7_loadouts,
    learner_observation_v3_contract_sha256,
)
from hytalegym.jax.combat.contracts.publication import (
    verify_combat_bridge_artifact_identity,
)
from hytalegym.jax.combat.observation.v3.policy.distribution import (
    ARSENAL_STANDARD_ROOT_DISTRIBUTION,
)
from hytalegym.jax.training.arsenal import (
    geometry_arsenal_world_capabilities,
)
from hytalegym.jax.training.arsenal_evaluation import (
    make_arsenal_evaluator,
)
from hytalegym.jax.training.baselines import (
    BASELINE_IDENTITIES,
    LEGACY_BASELINE_IDENTITIES,
    always_idle_arsenal_action_source,
    uniform_legal_arsenal_action_source,
)
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    current_combat_checkpoint_contract,
)
from hytalegym.jax.training.evaluation import (
    ActionCarryInitializer,
    ActionSource,
    evaluation_statistics,
    pure_jax_process_provenance,
)
from hytalegym.jax.training.policy import initialize_policy
from hytalegym.jax.training.types import PPOConfig
from hytalegym.jax.world import COLLISION_SHAPE_NONE, GeometryState
from hytalegym.rulesets import combat_ruleset_sha256

from .suite_publication import (
    CONTENT_ID_BASIS,
    content_addressed_suite_id,
    is_content_addressed_suite,
    load_frozen_suite_from_artifact,
    publish_current_trained_suite,
    verify_content_addressed_suite_manifest,
    verify_current_trained_suite_publication,
)


ARSENAL_FLAT_V1 = "hytalerl_arsenal_flat_v1"
ARSENAL_FLAT_V2 = "hytalerl_arsenal_flat_v2"
ARSENAL_FLAT_V3 = "hytalerl_arsenal_flat_v3"
ARSENAL_FLAT_V4 = "hytalerl_arsenal_flat_v4"
ARSENAL_FLAT_V5 = "hytalerl_arsenal_flat_v5"
ARSENAL_FLAT_V6 = "hytalerl_arsenal_flat_v6"
ARSENAL_FLAT_V7 = "hytalerl_arsenal_flat_v7"
ARSENAL_FLAT_V8 = "hytalerl_arsenal_flat_v8"
ARSENAL_FLAT_V9 = "hytalerl_arsenal_flat_v9"
ARSENAL_FLAT_V10 = "hytalerl_arsenal_flat_v10"
ARSENAL_FLAT_V11 = "hytalerl_arsenal_flat_v11"
ARSENAL_FLAT_V12 = "hytalerl_arsenal_flat_v12"
CURRENT_TRAINED_ARSENAL_FLAT = "hytalerl_arsenal_flat_current_trained"
CURRENT_TRAINED_ARSENAL_FLAT_VERSION = 13
BENCHMARK_ARTIFACT_SCHEMA = "hytalerl_benchmark_artifact_v1"
BENCHMARK_ARTIFACT_VERSION = 1
ARSENAL_FLAT_V1_SUITE_SHA256 = (
    "A794987FFEF2E8A9FDC649E44E1BCFDF3C501063C9F22A5CD8C5EC609DC4E7E2"
)
ARSENAL_FLAT_V2_SUITE_SHA256 = (
    "BD268DBEDEF38BE65ED9149311A0CD174B7694A4E56CEF1425B9B5332912E87E"
)
ARSENAL_FLAT_V3_SUITE_SHA256 = (
    "B3B003AE9084A7159555F0D7669107DEB1628B9DFC37442B4A865B7AEE40001C"
)
ARSENAL_FLAT_V4_SUITE_SHA256 = (
    "C63DD5F3F873FF641235C5FB296B775BFF276A26B2693F68B6910375D978B1AF"
)
ARSENAL_FLAT_V5_SUITE_SHA256 = (
    "4E92C0799F40151677DC4D557C326B11F6E42D3AE7F5C8F91395BD558B686095"
)
ARSENAL_FLAT_V6_SUITE_SHA256 = (
    "E87218A7A376AB7D83C799FE61F9B7E033B4D3395A46B462B4FEBABADC02E960"
)
ARSENAL_FLAT_V7_SUITE_SHA256 = (
    "F58211C0905C02DEC0F561795B18016DDC5935058A240CA19236E01E79A1359E"
)
ARSENAL_FLAT_V8_SUITE_SHA256 = (
    "FBB6C923F2CA28A0F7310BFF2F1917847EE08DFFD18DB167597B9277BBA27186"
)
ARSENAL_FLAT_V9_SUITE_SHA256 = (
    "AA66D18F5D664B3715238DBC1D474B95FBCCBE60FA4B8FA5FA99C7763C63650E"
)
ARSENAL_FLAT_V10_SUITE_SHA256 = (
    "273A16D5FA97DA840E68FFC274C0A07ECC60899B5CD7DF1B0C83477B1BB260AA"
)
ARSENAL_FLAT_V11_SUITE_SHA256 = (
    "FE20BC915DC9ED460C19D0D2DCFFA6ED96A75F25E89B47ECDE5004FC912A2DDF"
)
ARSENAL_FLAT_V12_SUITE_SHA256 = (
    "62EACACA2A4C31696CBDF1D5640675DC5F3F509FB59F81DB03A60B8F82B9F836"
)
ARSENAL_FLAT_V1_REFERENCE_RESULTS_SHA256 = (
    "3137732CD79D69FF50DFDD25186F7E703AEF892C074731894135C4C5709DC874"
)
ARSENAL_FLAT_V2_REFERENCE_RESULTS_SHA256 = (
    "3137732CD79D69FF50DFDD25186F7E703AEF892C074731894135C4C5709DC874"
)
ARSENAL_FLAT_V3_REFERENCE_RESULTS_SHA256 = (
    "3137732CD79D69FF50DFDD25186F7E703AEF892C074731894135C4C5709DC874"
)
ARSENAL_FLAT_V4_REFERENCE_RESULTS_SHA256 = (
    "3137732CD79D69FF50DFDD25186F7E703AEF892C074731894135C4C5709DC874"
)
ARSENAL_FLAT_V5_REFERENCE_RESULTS_SHA256 = (
    "3137732CD79D69FF50DFDD25186F7E703AEF892C074731894135C4C5709DC874"
)
ARSENAL_FLAT_V2_MEASUREMENT_PROVENANCE = {
    "mode": "inherited_reference_rows",
    "source_suite_id": ARSENAL_FLAT_V1,
    "source_results_sha256": ARSENAL_FLAT_V1_REFERENCE_RESULTS_SHA256,
    "independent_v2_execution": False,
    "statement": (
        "v2 reuses v1's stored reference result rows; their identical digest "
        "does not demonstrate reproduction under v2's semantic contracts"
    ),
}
ARSENAL_FLAT_V3_MEASUREMENT_PROVENANCE = {
    "mode": "independent_suite_execution",
    "execution_suite_id": ARSENAL_FLAT_V3,
    "execution_suite_sha256": ARSENAL_FLAT_V3_SUITE_SHA256,
    "execution_results_sha256": ARSENAL_FLAT_V3_REFERENCE_RESULTS_SHA256,
    "independent_v3_execution": True,
    "statement": (
        "v3 reference rows were independently executed under the embedded "
        "v3 suite; matching the historical digest is a measured reproduction"
    ),
}
ARSENAL_FLAT_V4_MEASUREMENT_PROVENANCE = {
    "mode": "independent_suite_execution",
    "execution_suite_id": ARSENAL_FLAT_V4,
    "execution_suite_sha256": ARSENAL_FLAT_V4_SUITE_SHA256,
    "execution_results_sha256": ARSENAL_FLAT_V4_REFERENCE_RESULTS_SHA256,
    "independent_v4_execution": True,
    "statement": (
        "v4 reference rows were independently executed under the embedded "
        "v4 suite; matching the historical digest is a measured reproduction"
    ),
}
ARSENAL_FLAT_V5_MEASUREMENT_PROVENANCE = {
    "mode": "independent_suite_execution",
    "execution_suite_id": ARSENAL_FLAT_V5,
    "execution_suite_sha256": ARSENAL_FLAT_V5_SUITE_SHA256,
    "execution_results_sha256": ARSENAL_FLAT_V5_REFERENCE_RESULTS_SHA256,
    "independent_v5_execution": True,
    "statement": (
        "v5 reference rows were independently executed under the embedded "
        "v5 suite; matching the historical digest is a measured reproduction"
    ),
}
REFERENCE_MEASUREMENT_PROVENANCE_BY_ID = {
    ARSENAL_FLAT_V2: ARSENAL_FLAT_V2_MEASUREMENT_PROVENANCE,
    ARSENAL_FLAT_V3: ARSENAL_FLAT_V3_MEASUREMENT_PROVENANCE,
    ARSENAL_FLAT_V4: ARSENAL_FLAT_V4_MEASUREMENT_PROVENANCE,
    ARSENAL_FLAT_V5: ARSENAL_FLAT_V5_MEASUREMENT_PROVENANCE,
}
FROZEN_SUITE_SHA256_BY_ID = {
    ARSENAL_FLAT_V1: ARSENAL_FLAT_V1_SUITE_SHA256,
    ARSENAL_FLAT_V2: ARSENAL_FLAT_V2_SUITE_SHA256,
    ARSENAL_FLAT_V3: ARSENAL_FLAT_V3_SUITE_SHA256,
    ARSENAL_FLAT_V4: ARSENAL_FLAT_V4_SUITE_SHA256,
    ARSENAL_FLAT_V5: ARSENAL_FLAT_V5_SUITE_SHA256,
    ARSENAL_FLAT_V6: ARSENAL_FLAT_V6_SUITE_SHA256,
    ARSENAL_FLAT_V7: ARSENAL_FLAT_V7_SUITE_SHA256,
    ARSENAL_FLAT_V8: ARSENAL_FLAT_V8_SUITE_SHA256,
    ARSENAL_FLAT_V9: ARSENAL_FLAT_V9_SUITE_SHA256,
    ARSENAL_FLAT_V10: ARSENAL_FLAT_V10_SUITE_SHA256,
    ARSENAL_FLAT_V11: ARSENAL_FLAT_V11_SUITE_SHA256,
    ARSENAL_FLAT_V12: ARSENAL_FLAT_V12_SUITE_SHA256,
}
REFERENCE_RESULTS_SHA256_BY_ID = {
    ARSENAL_FLAT_V1: ARSENAL_FLAT_V1_REFERENCE_RESULTS_SHA256,
    ARSENAL_FLAT_V2: ARSENAL_FLAT_V2_REFERENCE_RESULTS_SHA256,
    ARSENAL_FLAT_V3: ARSENAL_FLAT_V3_REFERENCE_RESULTS_SHA256,
    ARSENAL_FLAT_V4: ARSENAL_FLAT_V4_REFERENCE_RESULTS_SHA256,
    ARSENAL_FLAT_V5: ARSENAL_FLAT_V5_REFERENCE_RESULTS_SHA256,
}
REFERENCE_BASELINES = ("uniform_legal", "always_idle", "untrained_init")
TRAINED_COMPARISON_ARMS = ("trained_policy", *REFERENCE_BASELINES)
EVALUATION_SEEDS = (170003, 170029, 170057)
ROLLOUT_SEED = 744903660
UNTRAINED_INITIALIZATION_SEED = 15485863
MAX_POLICY_STEPS = 256
REFERENCE_TARGET_MAX_HEALTH = 61.0
ARSENAL_FLAT_V2_GENERATED_WORLD_LIMITATION = (
    "native filler-root capture and stationary standability/clearance are "
    "certified; generated-world scores remain unavailable until a fixed-shape "
    "Region graph producer certifies moving step/drop edges and supplies the "
    "policy traversal row"
)
ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION = (
    "this flat suite does not produce generated-world scores; a separate "
    "versioned suite must pin a generated-world selection and its current "
    "native traversal evidence"
)
ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION = (
    "the exact all-air frame is used for capability correctness, not speed; "
    "a separate batch-128 control measured 563.93 transitions/s for exact "
    "geometry without tokens versus 12,636.83 for open-flat"
)
_FROZEN_INTERPRETATION_LIMITATIONS_BY_ID = {
    ARSENAL_FLAT_V1: {
        "generated_world_scores": (
            ARSENAL_FLAT_V2_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V2: {
        "generated_world_scores": (
            ARSENAL_FLAT_V2_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V3: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V4: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V5: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V6: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V7: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V8: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V9: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V10: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V11: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
    ARSENAL_FLAT_V12: {
        "generated_world_scores": (
            ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
        ),
        "exact_geometry_throughput": (
            ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
        ),
    },
}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _suite_contract_for(
    *,
    suite_id: str,
    version: int,
) -> dict[str, Any]:
    profiles = tuple(PROFILE_NAMES)
    loadouts = hytale_0_5_7_loadouts(profiles)
    weapon_ids = np.asarray(loadouts.weapon_id)[:, AGENT_ENTITY]
    params = default_combat_params(microticks=1, target_active=True)
    contract = {
        "schema": f"hytalerl_benchmark_suite_v{version}",
        "version": version,
        "id": suite_id,
        "environment": "jax_arsenal",
        "task": "kweebec_razorleaf_vs_trork_brawler",
        "opponent": {
            "role": "Trork_Brawler",
            "selector": "ordered_legacy_attack_sequence",
        },
        "episode_seeds": list(EVALUATION_SEEDS),
        "rollout_seed": ROLLOUT_SEED,
        "episodes_per_profile": len(EVALUATION_SEEDS),
        "episode_count": len(profiles) * len(EVALUATION_SEEDS),
        "maximum_policy_steps": MAX_POLICY_STEPS,
        "microticks": 1,
        "target_active": True,
        "profiles": list(profiles),
        "profile_weapon_ids": {
            profile: int(weapon_id)
            for profile, weapon_id in zip(
                profiles,
                weapon_ids,
                strict=True,
            )
        },
        "observation": {
            "schema": "LearnerCombatObservationV3",
            "contract_sha256": (
                learner_observation_v3_contract_sha256().upper()
            ),
        },
        "action": (
            {
                "encoding": "explicit_per_head_int32",
                "head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
                "combination_count": math.prod(
                    ARSENAL_POLICY_ACTION_HEAD_SIZES
                ),
                **(
                    {
                        "distribution": ARSENAL_STANDARD_ROOT_DISTRIBUTION,
                        "legal_combination_count": 4_835_700,
                    }
                    if version >= CURRENT_TRAINED_ARSENAL_FLAT_VERSION
                    else {}
                ),
            }
            if version >= 8
            else {
                "encoding": "packed_categorical_heads",
                "head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
                "packed_action_count": math.prod(
                    ARSENAL_POLICY_ACTION_HEAD_SIZES
                ),
            }
        ),
        "capability_provider": {
            "id": "exact_complete_empty_local_geometry_v1",
            "frame": "complete_all_air_local_geometry",
            "open_flat_control": False,
            "world_geometry_tokens": "unavailable_fail_closed",
        },
        "reward_objective": {
            "target_damage": float(params.target_damage_reward_scale),
            "agent_damage": float(params.agent_damage_reward_scale),
            "completion": float(params.completion_reward),
            "death": float(params.death_reward),
        },
        "contracts": {
            "combat_ruleset_sha256": combat_ruleset_sha256().upper(),
            "arsenal_sha256": combat_arsenal_contract_sha256().upper(),
            "reference_policy_surface_sha256": (
                arsenal_policy_contract_sha256().upper()
            ),
        },
        "reference_baselines": {
            name: (
                BASELINE_IDENTITIES[name]
                if version >= CURRENT_TRAINED_ARSENAL_FLAT_VERSION
                else LEGACY_BASELINE_IDENTITIES[name]
            )
            for name in REFERENCE_BASELINES
        },
        "limitations": {
            "profile_content_held_out": False,
            "geometry": (
                "controlled exact all-air local frame, not generated terrain"
            ),
            "generated_world_scores": (
                ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
            ),
            "exact_geometry_throughput": (
                ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
            ),
            "world_geometry_tokens": "unavailable_fail_closed",
            "entity_count": 2,
        },
    }
    if version >= 6:
        contract["publication"] = {
            "kind": "trained_comparison",
            "required_arms": list(TRAINED_COMPARISON_ARMS),
            "trained_policy_checkpoint": {
                "match": "exact_current",
                "contract": current_combat_checkpoint_contract(
                    policy_surface=ARSENAL_POLICY_SURFACE,
                ),
            },
            "trained_policy_execution": "public_action_source",
        }
    return contract


def _current_trained_suite_contract() -> dict[str, Any]:
    contract = _suite_contract_for(
        suite_id=CONTENT_ID_BASIS,
        version=CURRENT_TRAINED_ARSENAL_FLAT_VERSION,
    )
    contract["id"] = content_addressed_suite_id(contract)
    return contract


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def publish_current_trained_suite_contract(
    repository_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Atomically publish the complete live trained-suite contract.

    The live suite is derived here rather than accepted from a caller, so the
    stable pointer cannot be advanced with a hand-edited or partially current
    manifest.
    """

    root = _repository_root() if repository_root is None else Path(repository_root)
    verify_combat_bridge_artifact_identity(
        root,
        require_deployed=os.name == "nt",
    )
    return publish_current_trained_suite(
        root,
        _current_trained_suite_contract(),
    )


def suite_contract() -> dict[str, Any]:
    """Return the byte-verified immutable historical v5 suite."""

    return load_frozen_suite_from_artifact(
        _repository_root(),
        "hytalerl_arsenal_flat_v5_reference.json",
        expected_id=ARSENAL_FLAT_V5,
        expected_sha256=ARSENAL_FLAT_V5_SUITE_SHA256,
    )


def trained_suite_contract() -> dict[str, Any]:
    """Return the byte-verified immutable historical v6 suite."""

    return load_frozen_suite_from_artifact(
        _repository_root(),
        "hytalerl_arsenal_flat_v6_trained.json",
        expected_id=ARSENAL_FLAT_V6,
        expected_sha256=ARSENAL_FLAT_V6_SUITE_SHA256,
    )


def current_trained_suite_contract() -> dict[str, Any]:
    """Return the current content-addressed suite via its stable pointer."""

    live = _current_trained_suite_contract()
    _, published = verify_current_trained_suite_publication(
        _repository_root(),
        live,
    )
    return published


def exact_empty_local_geometry(params) -> GeometryState:
    """Return complete all-air local evidence for the public flat suite."""

    return GeometryState(
        origin=jnp.asarray(((0, 66, -1),), dtype=jnp.int32),
        cell_mask=jnp.ones((1, CELL_COUNT), dtype=jnp.bool_),
        flags=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        fluid_level=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        support=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        block_damage=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        fluid_damage=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        movement=jnp.zeros(
            (1, CELL_COUNT, MOVEMENT_FEATURES),
            dtype=jnp.float32,
        ),
        fluid_movement=jnp.zeros(
            (1, CELL_COUNT, FLUID_MOVEMENT_FEATURES),
            dtype=jnp.float32,
        ),
        collision_shape_index=jnp.full(
            (1, CELL_COUNT),
            COLLISION_SHAPE_NONE,
            dtype=jnp.int16,
        ),
        collision_full_cube_cell=jnp.full(
            (1, 1),
            -1,
            dtype=jnp.int16,
        ),
        collision_exception_cell=jnp.full(
            (1, 1),
            -1,
            dtype=jnp.int16,
        ),
        collision_exception_box_index=jnp.full(
            (1, 1, 1),
            -1,
            dtype=jnp.int16,
        ),
        collision_exception_boxes=jnp.zeros(
            (1, 1, 6),
            dtype=jnp.float32,
        ),
        collision_exception_box_cell=jnp.full(
            (1, 1),
            -1,
            dtype=jnp.int16,
        ),
        collision_world_order=jnp.asarray(((0, 1),), dtype=jnp.int16),
        agent_bounds=jnp.broadcast_to(params.agent_bounds, (1, 6)),
        target_bounds=jnp.broadcast_to(params.target_bounds, (1, 6)),
        agent_los_offset=jnp.broadcast_to(
            params.agent_eye_offset,
            (1, 3),
        ),
        target_los_offset=jnp.broadcast_to(
            params.target_eye_offset,
            (1, 3),
        ),
    )


def run_arsenal_flat_v5(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer | None = None,
    reference_artifact: Mapping[str, Any] | None = None,
    compile_step: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v5."""

    del (
        action_source,
        subject,
        action_carry_initializer,
        reference_artifact,
        compile_step,
    )
    raise RuntimeError(
        f"{ARSENAL_FLAT_V5} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v5_references(
    *,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject regeneration of historical v5 reference data."""

    del compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V5} is historical; do not refresh its references"
    )


def run_arsenal_flat_v6_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v6."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V6} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v7_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v7."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V7} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v8_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v8."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V8} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v9_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v9."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V9} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_current_arsenal_flat_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Run the current trained ActionSource and all frozen controls."""

    suite = current_trained_suite_contract()
    return _run_trained_comparison(
        suite,
        action_source,
        subject,
        action_carry_initializer=action_carry_initializer,
        compile=compile,
    )


def run_arsenal_flat_v10_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v10."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V10} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v11_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v11."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V11} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v12_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v12."""

    del action_source, subject, action_carry_initializer, compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V12} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def _run_trained_comparison(
    suite: Mapping[str, Any],
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool,
) -> dict[str, Any]:
    _validate_trained_subject(subject, suite=suite)
    runtime, reset_keys, labels, row_seeds, params, provider = (
        _suite_runtime()
    )
    trained_evaluator = make_arsenal_evaluator(
        params,
        runtime,
        world_capability_provider=provider,
        max_policy_steps=MAX_POLICY_STEPS,
        compile=compile,
        action_source=action_source,
        action_carry_initializer=action_carry_initializer,
    )
    trained_result = trained_evaluator(
        None,
        reset_keys,
        jax.random.key(ROLLOUT_SEED),
    )
    jax.block_until_ready(trained_result)
    reference_results, reference_subjects = _run_reference_arms(
        runtime,
        reset_keys,
        labels,
        row_seeds,
        params,
        provider,
        compile=compile,
    )
    results = {
        "trained_policy": evaluation_statistics(
            trained_result,
            labels=labels,
            seeds=row_seeds,
        ),
        **reference_results,
    }
    artifact = _artifact(
        kind="trained_comparison",
        subjects={
            "trained_policy": dict(subject),
            **reference_subjects,
        },
        results=results,
        suite=suite,
    )
    artifact["comparisons"] = {
        name: _paired_comparison(results["trained_policy"], results[name])
        for name in REFERENCE_BASELINES
    }
    _require_valid(artifact, expected_kind="trained_comparison")
    return artifact


def run_arsenal_flat_v4(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v4."""

    del args, kwargs
    raise RuntimeError(
        f"{ARSENAL_FLAT_V4} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v4_references(
    *,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject regeneration of historical v4 reference data."""

    del compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V4} is historical; do not refresh its references"
    )


def run_arsenal_flat_v3(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v3."""

    del args, kwargs
    raise RuntimeError(
        f"{ARSENAL_FLAT_V3} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v3_references(
    *,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject regeneration of historical v3 reference data."""

    del compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V3} is historical; do not refresh its references"
    )


def run_arsenal_flat_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject new runs that would relabel current semantics as v2."""

    del args, kwargs
    raise RuntimeError(
        f"{ARSENAL_FLAT_V2} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v2_references(
    *,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject regeneration of historical v2 reference data."""

    del compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V2} is historical; do not refresh its references"
    )


def run_arsenal_flat_v1(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject new runs that would relabel current combat semantics as v1."""

    del args, kwargs
    raise RuntimeError(
        f"{ARSENAL_FLAT_V1} is historical; current source must use "
        f"{CURRENT_TRAINED_ARSENAL_FLAT}"
    )


def run_arsenal_flat_v1_references(
    *,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject regeneration of historical v1 reference data."""

    del compile
    raise RuntimeError(
        f"{ARSENAL_FLAT_V1} is historical; do not refresh its references"
    )


def load_reference_baselines(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = benchmark_artifact_errors(
        artifact,
        expected_kind="reference",
    )
    if errors:
        raise ValueError("invalid reference artifact:\n- " + "\n- ".join(errors))
    return artifact


def _require_reference_suite(
    artifact: Mapping[str, Any],
    *,
    suite_sha256: str,
) -> None:
    errors = benchmark_artifact_errors(
        artifact,
        expected_kind="reference",
    )
    if errors:
        raise ValueError(
            "reference artifact is invalid:\n- " + "\n- ".join(errors)
        )
    if artifact["suite_sha256"] != suite_sha256:
        raise ValueError(
            "reference artifact belongs to a different benchmark suite"
        )


def benchmark_artifact_errors(
    artifact: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> list[str]:
    errors = []
    if artifact.get("schema") != BENCHMARK_ARTIFACT_SCHEMA:
        errors.append("artifact schema differs")
    if artifact.get("version") != BENCHMARK_ARTIFACT_VERSION:
        errors.append("artifact version differs")
    kind = artifact.get("kind")
    if expected_kind is not None and kind != expected_kind:
        errors.append(f"artifact kind must be {expected_kind!r}")
    suite = artifact.get("suite")
    if not isinstance(suite, Mapping):
        return errors + ["suite must be a mapping"]
    suite_hash = artifact.get("suite_sha256")
    if suite_hash != canonical_sha256(suite):
        errors.append("suite_sha256 does not match the embedded suite")
    suite_id = suite.get("id")
    expected_suite_hash = FROZEN_SUITE_SHA256_BY_ID.get(suite_id)
    content_addressed = is_content_addressed_suite(suite)
    if expected_suite_hash is None and not content_addressed:
        errors.append("artifact benchmark suite is unsupported")
    elif expected_suite_hash is not None and suite_hash != expected_suite_hash:
        errors.append("artifact is not for its frozen Arsenal suite")
    elif content_addressed:
        try:
            verify_content_addressed_suite_manifest(
                _repository_root(),
                suite,
            )
        except RuntimeError:
            errors.append(
                "artifact content-addressed Arsenal suite is unpublished"
            )
    contract = artifact.get("contract")
    if not isinstance(contract, Mapping):
        return errors + ["contract must be a mapping"]
    if artifact.get("contract_sha256") != canonical_sha256(contract):
        errors.append("contract_sha256 does not match the contract")
    if contract.get("suite_sha256") != suite_hash:
        errors.append("contract suite identity differs")
    if (
        kind != "reference"
        and contract.get("native_server_process")
        != pure_jax_process_provenance()
    ):
        errors.append(
            "pure-JAX artifact must declare native-server uptime independence"
        )
    results = artifact.get("results")
    subjects = contract.get("subjects")
    if not isinstance(results, Mapping) or not isinstance(subjects, Mapping):
        return errors + ["results and contract subjects must be mappings"]
    results_sha256 = artifact.get("results_sha256")
    if results_sha256 != canonical_sha256(results):
        errors.append("results_sha256 does not match the embedded results")
    expected_results_sha256 = REFERENCE_RESULTS_SHA256_BY_ID.get(suite_id)
    if (
        kind == "reference"
        and expected_results_sha256
        and results_sha256 != expected_results_sha256
    ):
        errors.append("reference results differ from the published suite data")
    expected_provenance = REFERENCE_MEASUREMENT_PROVENANCE_BY_ID.get(
        suite_id
    )
    if kind == "reference" and expected_provenance is not None:
        if (
            not isinstance(artifact.get("interpretation"), Mapping)
            or artifact["interpretation"].get("measurement_provenance")
            != expected_provenance
        ):
            errors.append("reference measurement provenance differs")
    if set(results) != set(subjects):
        errors.append("result subjects differ from contract subjects")
    if kind == "reference" and set(results) != set(REFERENCE_BASELINES):
        errors.append("reference artifact must contain all three baselines")
    if kind == "trained_comparison":
        if suite_id not in (
            ARSENAL_FLAT_V6,
            ARSENAL_FLAT_V7,
            ARSENAL_FLAT_V8,
            ARSENAL_FLAT_V9,
            ARSENAL_FLAT_V10,
            ARSENAL_FLAT_V11,
            ARSENAL_FLAT_V12,
        ) and not content_addressed:
            errors.append("trained comparison must use a frozen trained suite")
        if set(results) != set(TRAINED_COMPARISON_ARMS):
            errors.append(
                "trained comparison must contain trained policy and controls"
            )
        trained_subject = subjects.get("trained_policy")
        if not isinstance(trained_subject, Mapping):
            errors.append("trained policy subject must be a mapping")
        else:
            try:
                _validate_trained_subject(trained_subject, suite=suite)
            except (TypeError, ValueError) as error:
                errors.append(f"trained policy subject is invalid: {error}")
    expected_episodes = suite.get("episode_count")
    profiles = suite.get("profiles")
    seeds = suite.get("episode_seeds")
    expected_keys = (
        [
            (profile, seed)
            for seed in seeds
            for profile in profiles
        ]
        if isinstance(profiles, list) and isinstance(seeds, list)
        else None
    )
    objective = suite.get("reward_objective")
    for name, result in results.items():
        if not isinstance(result, Mapping):
            errors.append(f"result {name!r} must be a mapping")
            continue
        if result.get("episode_count") != expected_episodes:
            errors.append(f"result {name!r} episode count differs")
        episodes = result.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != expected_episodes:
            errors.append(f"result {name!r} episode rows differ")
            continue
        keys = [(row.get("label"), row.get("seed")) for row in episodes]
        if expected_keys is None or keys != expected_keys:
            errors.append(f"result {name!r} episode keys differ")
        if not isinstance(objective, Mapping) or not _reward_recomposes(
            episodes,
            objective,
        ):
            errors.append(f"result {name!r} reward components do not recompose")
    if kind == "trained_comparison" and set(results) == set(
        TRAINED_COMPARISON_ARMS
    ):
        try:
            expected_comparisons = {
                name: _paired_comparison(
                    results["trained_policy"],
                    results[name],
                )
                for name in REFERENCE_BASELINES
            }
        except (KeyError, TypeError, ValueError):
            errors.append("trained comparisons are not derivable")
        else:
            if artifact.get("comparisons") != expected_comparisons:
                errors.append(
                    "trained comparisons differ from paired result rows"
                )
    if kind in ("reference", "trained_comparison"):
        try:
            expected_interpretation = _artifact_interpretation(
                kind,
                suite,
                results,
            )
        except (KeyError, TypeError, ValueError):
            errors.append("reference interpretation facts are not derivable")
        else:
            if artifact.get("interpretation") != expected_interpretation:
                errors.append(
                    (
                        "reference"
                        if kind == "reference"
                        else "trained comparison"
                    )
                    + " interpretation differs from result facts"
                )
    return errors


def _suite_runtime():
    profiles = tuple(PROFILE_NAMES)
    labels = tuple(
        profile for _ in EVALUATION_SEEDS for profile in profiles
    )
    row_seeds = tuple(
        seed for seed in EVALUATION_SEEDS for _ in profiles
    )
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(labels))
    params = default_combat_params(microticks=1, target_active=True)
    geometry = exact_empty_local_geometry(params)
    provider = partial(
        geometry_arsenal_world_capabilities,
        geometry=geometry,
        config=runtime,
    )
    reset_keys = jnp.stack(
        tuple(jax.random.key(seed) for seed in row_seeds),
        axis=0,
    )
    return runtime, reset_keys, labels, row_seeds, params, provider


def _run_reference_arms(
    runtime,
    reset_keys,
    labels,
    row_seeds,
    params,
    provider,
    *,
    compile: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs = {
        "world_capability_provider": provider,
        "max_policy_steps": MAX_POLICY_STEPS,
        "compile": compile,
    }
    policy_evaluator = make_arsenal_evaluator(params, runtime, **kwargs)
    evaluators = {
        "uniform_legal": (
            make_arsenal_evaluator(
                params,
                runtime,
                action_source=uniform_legal_arsenal_action_source,
                **kwargs,
            ),
            None,
        ),
        "always_idle": (
            make_arsenal_evaluator(
                params,
                runtime,
                action_source=always_idle_arsenal_action_source,
                **kwargs,
            ),
            None,
        ),
    }
    config = _reference_policy_config(len(labels))
    evaluators["untrained_init"] = (
        policy_evaluator,
        initialize_policy(
            jax.random.key(UNTRAINED_INITIALIZATION_SEED),
            config,
        ),
    )
    results = {}
    for name, (evaluator, policy_params) in evaluators.items():
        result = evaluator(
            policy_params,
            reset_keys,
            jax.random.key(ROLLOUT_SEED),
        )
        jax.block_until_ready(result)
        results[name] = evaluation_statistics(
            result,
            labels=labels,
            seeds=row_seeds,
        )
    subjects = {
        name: {
            **BASELINE_IDENTITIES[name],
            **(
                {
                    "initialization_seed": UNTRAINED_INITIALIZATION_SEED,
                    "policy_config": asdict(config),
                }
                if name == "untrained_init"
                else {}
            ),
        }
        for name in REFERENCE_BASELINES
    }
    return results, subjects


def _reference_policy_config(batch_size: int) -> PPOConfig:
    return PPOConfig(
        num_envs=batch_size,
        rollout_steps=1,
        update_epochs=1,
        num_minibatches=3,
        observation_size=ARSENAL_POLICY_OBSERVATION_SIZE,
        action_size=ARSENAL_POLICY_ACTION_SIZE,
        action_head_sizes=ARSENAL_POLICY_ACTION_HEAD_SIZES,
        action_transport="factors",
        action_distribution=ARSENAL_STANDARD_ROOT_DISTRIBUTION,
    )


def _artifact(
    *,
    kind: str,
    subjects: Mapping[str, Any],
    results: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    suite_sha256 = canonical_sha256(suite)
    contract = {
        "schema": "hytalerl_benchmark_run_contract_v1",
        "version": 1,
        "kind": kind,
        "suite_sha256": suite_sha256,
        "action_source_interface": (
            "(LearnerCombatObservationV3, carry, jax_key) "
            "-> (explicit_action_factors, carry)"
            if int(suite["version"]) >= 8
            else "(LearnerCombatObservationV3, carry, jax_key) "
            "-> (packed_action_ids, carry)"
        ),
        "subjects": dict(subjects),
    }
    if kind != "reference":
        contract["native_server_process"] = pure_jax_process_provenance()
    return {
        "schema": BENCHMARK_ARTIFACT_SCHEMA,
        "version": BENCHMARK_ARTIFACT_VERSION,
        "kind": kind,
        "suite": dict(suite),
        "suite_sha256": suite_sha256,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "results": dict(results),
        "results_sha256": canonical_sha256(results),
        "interpretation": _artifact_interpretation(kind, suite, results),
    }


def _artifact_interpretation(
    kind: str,
    suite: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    limitations = dict(suite["limitations"])
    limitations.update(
        _FROZEN_INTERPRETATION_LIMITATIONS_BY_ID.get(
            suite["id"],
            {
                "generated_world_scores": (
                    ARSENAL_FLAT_V3_GENERATED_WORLD_LIMITATION
                ),
                "exact_geometry_throughput": (
                    ARSENAL_FLAT_V2_EXACT_GEOMETRY_THROUGHPUT_LIMITATION
                ),
            },
        )
    )
    interpretation = {
        "covers": (
            "31 shipped loadout profiles, three fixed episode seeds, "
            "natural ordered Brawler attacks, raw reward components, and "
            "one exact controlled local geometry frame"
        ),
        "limitations": limitations,
    }
    if kind == "reference":
        provenance = REFERENCE_MEASUREMENT_PROVENANCE_BY_ID.get(suite["id"])
        if provenance is not None:
            interpretation["measurement_provenance"] = provenance
        interpretation["reference_findings"] = _reference_findings(
            results,
            suite,
        )
    elif kind == "trained_comparison":
        suite_version = suite["version"]
        suite_id = suite["id"]
        interpretation["measurement_provenance"] = {
            "mode": "independent_suite_execution",
            "execution_suite_id": suite_id,
            "execution_suite_sha256": canonical_sha256(suite),
            "execution_results_sha256": canonical_sha256(results),
            f"independent_v{suite_version}_execution": True,
            "statement": (
                f"v{suite_version} trained and control rows were "
                f"independently executed under the embedded v{suite_version} "
                "suite"
            ),
        }
        interpretation["trained_comparison_findings"] = _trained_findings(
            results
        )
    return interpretation


def _reference_findings(
    results: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    interaction_by_arm = {}
    maximum_target_damage = 0.0
    for name in REFERENCE_BASELINES:
        episodes = results[name]["episodes"]
        target_damage = [
            float(row["reward_components"]["target_damage"])
            for row in episodes
        ]
        maximum = max(target_damage)
        maximum_target_damage = max(maximum_target_damage, maximum)
        interaction_by_arm[name] = {
            "episode_count": len(episodes),
            "episodes_with_target_damage": sum(
                damage > 0.0 for damage in target_damage
            ),
            "maximum_target_damage": maximum,
            "death_count": sum(
                bool(row["reward_components"]["death"])
                for row in episodes
            ),
        }
    target_max_health = REFERENCE_TARGET_MAX_HEALTH
    required_action_size = sum(suite["action"]["head_sizes"])
    return {
        "interaction_by_arm": interaction_by_arm,
        "attack_reachability": (
            "demonstrated by positive target damage in uniform-legal and "
            "untrained-init episodes"
        ),
        "zero_success": (
            "not a no-interaction result: maximum cumulative target damage "
            f"is {maximum_target_damage:g}, below target max health "
            f"{target_max_health:g} within the 256-decision ceiling"
        ),
        "trained_policy": {
            "included": False,
            "reason": (
                "the published subjects are controls; no stored checkpoint "
                "matched the suite's 38-logit policy contract at publication"
            ),
            "checkpoint_census_at_publication": {
                "examined": 19,
                "compatible": 0,
                "stored_action_sizes": [33, 35],
                "required_action_size": required_action_size,
            },
            "consequence": (
                "this reference artifact cannot answer whether training "
                "beats random; a compatible trained policy must run through "
                "the submission ActionSource"
            ),
            "required_policy_contract_sha256": (
                suite["contracts"]["reference_policy_surface_sha256"]
            ),
        },
    }


def _trained_findings(results: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = {
        name: _paired_comparison(results["trained_policy"], results[name])
        for name in REFERENCE_BASELINES
    }
    uniform_delta = comparisons["uniform_legal"][
        "episode_return_delta"
    ]["mean"]
    return {
        "arms": {
            name: {
                "episode_return": dict(results[name]["episode_return"]),
                "success_rate": float(results[name]["success_rate"]),
            }
            for name in TRAINED_COMPARISON_ARMS
        },
        "paired_against_controls": comparisons,
        "training_beats_uniform_on_mean_return": uniform_delta > 0.0,
        "statement": (
            "the trained arm is published regardless of sign; training "
            "budget and checkpoint identity are pinned in contract subjects"
        ),
    }


def _paired_comparison(
    subject: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    subject_rows = subject["episodes"]
    reference_rows = reference["episodes"]
    subject_keys = [
        (row["label"], row["seed"]) for row in subject_rows
    ]
    reference_keys = [
        (row["label"], row["seed"]) for row in reference_rows
    ]
    if subject_keys != reference_keys:
        raise ValueError("benchmark episode keys are not identical")
    delta = np.asarray(
        [
            left["episode_return"] - right["episode_return"]
            for left, right in zip(
                subject_rows,
                reference_rows,
                strict=True,
            )
        ],
        dtype=np.float32,
    )
    return {
        "episode_return_delta": {
            "mean": float(np.mean(delta)),
            "standard_deviation": float(np.std(delta)),
            "minimum": float(np.min(delta)),
            "maximum": float(np.max(delta)),
        },
        "success_rate_delta": (
            float(subject["success_rate"])
            - float(reference["success_rate"])
        ),
    }


def _validate_subject(subject: Mapping[str, Any]) -> None:
    missing = {"schema", "version", "name"} - set(subject)
    if missing:
        raise ValueError(f"subject contract missing fields: {sorted(missing)}")
    try:
        json.dumps(subject, sort_keys=True)
    except TypeError as error:
        raise TypeError("subject contract must be JSON-compatible") from error


def _validate_trained_subject(
    subject: Mapping[str, Any],
    *,
    suite: Mapping[str, Any] | None = None,
) -> None:
    _validate_subject(subject)
    if subject.get("schema") != "hytalerl_trained_arsenal_subject_v1":
        raise ValueError("trained subject schema differs")
    checkpoint = subject.get("policy_checkpoint")
    if not isinstance(checkpoint, Mapping) or not _is_sha256(
        checkpoint.get("sha256")
    ):
        raise ValueError(
            "trained subject policy_checkpoint.sha256 must be 64-hex"
        )
    filename = checkpoint.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(
            "trained subject policy_checkpoint.filename must be nonempty"
        )
    metadata = checkpoint.get("metadata")
    config = checkpoint.get("config")
    if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
        raise ValueError(
            "trained subject checkpoint metadata and config must be mappings"
        )
    budget = subject.get("training_budget")
    if not isinstance(budget, Mapping):
        raise ValueError("trained subject training_budget must be a mapping")
    seed = budget.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("trained subject training_budget.seed must be an int")
    for name in ("updates", "environment_steps"):
        value = budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"trained subject training_budget.{name} must be positive"
            )
        if metadata.get(name) != value:
            raise ValueError(
                f"trained subject training_budget.{name} differs from metadata"
            )
    if budget.get("seed") != metadata.get("seed"):
        raise ValueError("trained subject training seed differs from metadata")
    config_fields = {field.name for field in fields(PPOConfig)}
    suite_id = suite.get("id") if isinstance(suite, Mapping) else None
    allowed_missing = {"action_distribution"}
    if suite_id in (ARSENAL_FLAT_V6, ARSENAL_FLAT_V7):
        allowed_missing.add("action_transport")
    missing = config_fields - set(config)
    unexpected = set(config) - config_fields
    if unexpected or not missing.issubset(allowed_missing):
        raise ValueError(
            "trained subject checkpoint config fields differ from PPOConfig"
        )
    config_values = dict(config)
    config_values.setdefault("action_transport", "scalar")
    config_values.setdefault("action_distribution", "independent")
    try:
        validated_config = PPOConfig(**config_values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "trained subject checkpoint config is invalid"
        ) from error
    if suite is None:
        return
    publication = suite.get("publication")
    checkpoint_gate = (
        publication.get("trained_policy_checkpoint")
        if isinstance(publication, Mapping)
        else None
    )
    expected_metadata = (
        checkpoint_gate.get("contract")
        if isinstance(checkpoint_gate, Mapping)
        else None
    )
    if not isinstance(expected_metadata, Mapping):
        raise ValueError("suite trained checkpoint contract is unavailable")
    for name, expected in expected_metadata.items():
        actual = metadata.get(name)
        if actual != expected:
            raise ValueError(
                f"trained subject metadata.{name} differs from suite"
            )
    expected_config = {
        "observation_size": expected_metadata.get("observation_size"),
        "action_size": expected_metadata.get("action_size"),
    }
    for name, expected in expected_config.items():
        if getattr(validated_config, name) != expected:
            raise ValueError(
                f"trained subject config.{name} differs from suite"
            )
    suite_action = suite.get("action")
    expected_head_sizes = (
        suite_action.get("head_sizes")
        if isinstance(suite_action, Mapping)
        else None
    )
    if not isinstance(expected_head_sizes, list) or tuple(
        validated_config.action_head_sizes
    ) != tuple(expected_head_sizes):
        raise ValueError(
            "trained subject config.action_head_sizes differs from suite"
        )
    expected_transport = (
        "factors"
        if suite_action.get("encoding") == "explicit_per_head_int32"
        else "scalar"
    )
    if validated_config.action_transport != expected_transport:
        raise ValueError(
            "trained subject config.action_transport differs from suite"
        )
    expected_distribution = suite_action.get("distribution")
    if (
        expected_distribution is not None
        and validated_config.action_distribution != expected_distribution
    ):
        raise ValueError(
            "trained subject config.action_distribution differs from suite"
        )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _require_valid(
    artifact: Mapping[str, Any],
    *,
    expected_kind: str,
) -> None:
    errors = benchmark_artifact_errors(
        artifact,
        expected_kind=expected_kind,
    )
    if errors:
        raise RuntimeError(
            "benchmark artifact failed its contract:\n- "
            + "\n- ".join(errors)
        )


def _reward_recomposes(
    episodes: list[Mapping[str, Any]],
    objective: Mapping[str, Any],
) -> bool:
    for row in episodes:
        components = row.get("reward_components")
        if not isinstance(components, Mapping):
            return False
        recomposed = (
            float(components["target_damage"]) * float(objective["target_damage"])
            + float(components["agent_damage"]) * float(objective["agent_damage"])
            + float(bool(components["completion"])) * float(objective["completion"])
            + float(bool(components["death"])) * float(objective["death"])
        )
        if not np.isclose(
            recomposed,
            float(row["episode_return"]),
            atol=1.0e-4,
            rtol=1.0e-6,
        ):
            return False
    return True
