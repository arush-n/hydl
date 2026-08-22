"""Frozen Region-context Arsenal benchmark and fail-closed publication gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import FLAG_OPAQUE
from hytalegym.jax.combat import (
    AGENT_ENTITY,
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    PROFILE_NAMES,
    arsenal_runtime_config,
    default_combat_params,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.evaluation.arsenal_benchmark import (
    EVALUATION_SEEDS,
    MAX_POLICY_STEPS,
    ROLLOUT_SEED,
    TRAINED_COMPARISON_ARMS,
    UNTRAINED_INITIALIZATION_SEED,
    _reference_policy_config,
    _validate_trained_subject,
)
from hytalegym.jax.training.arsenal import (
    geometry_arsenal_world_capabilities,
    geometry_arsenal_world_tokens,
)
from hytalegym.jax.training.arsenal_evaluation import make_arsenal_evaluator
from hytalegym.jax.training.baselines import (
    always_idle_arsenal_action_source,
    uniform_legal_arsenal_action_source,
)
from hytalegym.jax.training.evaluation import (
    ActionCarryInitializer,
    ActionSource,
)
from hytalegym.jax.training.policy import initialize_policy
from hytalegym.jax.world import (
    REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION,
    REGION_TRAVERSAL_POLICY_STAGING_CAPACITY,
    WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    RegionLibraryEnvironmentSelection,
    RegionGeometryState,
    query_region_traversal_tokens,
    region_environments_from_artifact_manifest,
    region_geometry_from_atlas,
)
from hytalegym.jax.world.navigation.adapter import (
    surrogate_region_walk_target_navigation_provider,
)
from hytalegym.jax.world.region.los import (
    REGION_PERCEPTION_LOS_DISTANCE_BLOCKS,
)
from ._contracts import (  # noqa: F401  (re-exported: callers unchanged)
    ARSENAL_REGION_V1,
    ARSENAL_REGION_V1_SUITE_SHA256,
    ARSENAL_REGION_V2,
    ARSENAL_REGION_V2_SUITE_SHA256,
    ARSENAL_REGION_V3,
    ARSENAL_REGION_V3_SUITE_SHA256,
    ARSENAL_REGION_V4,
    ARSENAL_REGION_V4_SUITE_SHA256,
    ARSENAL_REGION_V5,
    ARSENAL_REGION_V5_SUITE_SHA256,
    ARSENAL_REGION_V6,
    ARSENAL_REGION_V6_SUITE_SHA256,
    EDGE_SELECTION_METHOD,
    EDGE_SELECTION_NAMESPACE,
    EDGE_SELECTION_PAYLOAD_V1,
    EDGE_SELECTION_PAYLOAD_V2,
    FROZEN_REGION_WORLDS,
    FrozenRegionWorld,
    REGION_ASSIGNMENT_KEY,
    REGION_CAPTURE_EVIDENCE_BRIDGE_SHA256,
    REGION_EPISODE_COUNT,
    REGION_LIBRARY_SEMANTIC_SHA256,
    REGION_SELECTION_KEY,
    REGION_WORLD_COUNT,
    _checked_region_v1_suite_contract,
    _checked_region_v2_suite_contract,
    _checked_region_v3_suite_contract,
    _checked_region_v4_suite_contract,
    _checked_region_v5_suite_contract,
    _checked_region_v6_suite_contract,
    _manifest_sha,
    _region_suite_contract,
    _region_v1_suite_contract,
    _region_v2_suite_contract,
    _region_v3_suite_contract,
    _region_v4_suite_contract,
    _region_v5_suite_contract,
    _region_v6_suite_contract,
    region_v1_suite_contract,
    region_v2_suite_contract,
    region_v3_suite_contract,
    region_v4_suite_contract,
    region_v5_suite_contract,
    region_v6_suite_contract,
)
from ._artifact import (  # noqa: F401  (re-exported: callers unchanged)
    FROZEN_REGION_SUITE_SHA256_BY_ID,
    _baseline_subjects,
    _episode_key,
    _expected_episode_keys,
    _paired_region_comparison,
    _paired_row_summary,
    _region_artifact,
    _region_comparisons,
    _region_interpretation,
    _region_statistics,
    _rows_match_world_contract,
    _select_world_edge,
    _validate_loaded_worlds,
    _world_rows,
    region_benchmark_artifact_errors,
)


ARSENAL_REGION_LOCOMOTION_BLOCKER = (
    "Region publication requires physical_geometry_bound=true, measured "
    "from the selected RegionGeometryState passed to Arsenal physical motion; "
    "a missing or false binding fails closed."
)


@dataclass(frozen=True)
class RegionBenchmarkContext:
    selection: RegionLibraryEnvironmentSelection
    region_manifest: Mapping[str, Any]
    traversal_manifest: Mapping[str, Any]
    suite: Mapping[str, Any]


def region_v1_runtime_blockers(
    *,
    physical_geometry_bound: bool,
) -> list[str]:
    """Return blockers that make an exact-terrain publication misleading."""

    if physical_geometry_bound is True:
        return []
    return [ARSENAL_REGION_LOCOMOTION_BLOCKER]


def load_region_v1_context(
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
) -> RegionBenchmarkContext:
    """Load and verify historical v1 only while source still matches it."""

    region_path = Path(region_manifest_path)
    traversal_path = Path(traversal_manifest_path)
    region_manifest = json.loads(region_path.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(
        traversal_path.read_text(encoding="utf-8")
    )
    suite = _checked_region_v1_suite_contract(traversal_manifest)
    return _load_region_context(
        region_path,
        traversal_path,
        region_manifest,
        traversal_manifest,
        suite,
    )


def load_region_v2_context(
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
) -> RegionBenchmarkContext:
    """Load historical v2 only while source still matches its identity."""

    region_path = Path(region_manifest_path)
    traversal_path = Path(traversal_manifest_path)
    region_manifest = json.loads(region_path.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(
        traversal_path.read_text(encoding="utf-8")
    )
    suite = _checked_region_v2_suite_contract(traversal_manifest)
    return _load_region_context(
        region_path,
        traversal_path,
        region_manifest,
        traversal_manifest,
        suite,
    )


def load_region_v3_context(
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
) -> RegionBenchmarkContext:
    """Load historical v3 only while source still matches its identity."""

    region_path = Path(region_manifest_path)
    traversal_path = Path(traversal_manifest_path)
    region_manifest = json.loads(region_path.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(
        traversal_path.read_text(encoding="utf-8")
    )
    suite = _checked_region_v3_suite_contract(traversal_manifest)
    return _load_region_context(
        region_path,
        traversal_path,
        region_manifest,
        traversal_manifest,
        suite,
    )


def load_region_v4_context(
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
) -> RegionBenchmarkContext:
    """Load historical v4 only while source still matches its identity."""

    region_path = Path(region_manifest_path)
    traversal_path = Path(traversal_manifest_path)
    region_manifest = json.loads(region_path.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(
        traversal_path.read_text(encoding="utf-8")
    )
    suite = _checked_region_v4_suite_contract(traversal_manifest)
    return _load_region_context(
        region_path,
        traversal_path,
        region_manifest,
        traversal_manifest,
        suite,
    )


def load_region_v5_context(
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
) -> RegionBenchmarkContext:
    """Load the current explicit-factor held-out Region corpus."""

    region_path = Path(region_manifest_path)
    traversal_path = Path(traversal_manifest_path)
    region_manifest = json.loads(region_path.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(
        traversal_path.read_text(encoding="utf-8")
    )
    suite = _checked_region_v5_suite_contract(traversal_manifest)
    return _load_region_context(
        region_path,
        traversal_path,
        region_manifest,
        traversal_manifest,
        suite,
    )


def load_region_v6_context(
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
) -> RegionBenchmarkContext:
    """Load the current Region corpus with target navigation bound."""

    region_path = Path(region_manifest_path)
    traversal_path = Path(traversal_manifest_path)
    region_manifest = json.loads(region_path.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(
        traversal_path.read_text(encoding="utf-8")
    )
    suite = _checked_region_v6_suite_contract(traversal_manifest)
    return _load_region_context(
        region_path,
        traversal_path,
        region_manifest,
        traversal_manifest,
        suite,
    )


def _load_region_context(
    region_path: Path,
    traversal_path: Path,
    region_manifest: Mapping[str, Any],
    traversal_manifest: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> RegionBenchmarkContext:
    corpus = suite["region_corpus"]
    _require_manifest_value(
        region_manifest,
        ("schema",),
        corpus["region_library"]["schema"],
    )
    _require_manifest_value(
        region_manifest,
        ("version",),
        corpus["region_library"]["version"],
    )
    _require_manifest_sha(
        region_manifest,
        ("library_semantic_sha256",),
        REGION_LIBRARY_SEMANTIC_SHA256,
    )
    _require_manifest_sha(
        region_manifest,
        ("capture_contract", "native_evidence_jar_sha256"),
        REGION_CAPTURE_EVIDENCE_BRIDGE_SHA256,
    )
    _require_manifest_value(
        traversal_manifest,
        ("schema",),
        corpus["traversal_library"]["schema"],
    )
    _require_manifest_value(
        traversal_manifest,
        ("version",),
        corpus["traversal_library"]["version"],
    )
    _require_manifest_sha(
        traversal_manifest,
        ("library_semantic_sha256",),
        suite["region_corpus"]["traversal_library"]["semantic_sha256"],
    )
    _require_manifest_sha(
        traversal_manifest,
        ("source_region_library_semantic_sha256",),
        REGION_LIBRARY_SEMANTIC_SHA256,
    )
    _require_manifest_sha(
        traversal_manifest,
        ("native_evidence_jar_sha256",),
        suite["region_corpus"]["traversal_library"][
            "capture_evidence_bridge_sha256"
        ],
    )
    selection = region_environments_from_artifact_manifest(
        region_path,
        usage=REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION,
        expected_library_semantic_sha256=REGION_LIBRARY_SEMANTIC_SHA256,
        working_set_capacity=REGION_WORLD_COUNT,
        environment_count=REGION_WORLD_COUNT,
        selection_key=REGION_SELECTION_KEY,
        assignment_key=REGION_ASSIGNMENT_KEY,
        traversal_manifest_path=traversal_path,
        expected_traversal_library_semantic_sha256=(
            suite["region_corpus"]["traversal_library"]["semantic_sha256"]
        ),
    )
    _validate_loaded_worlds(selection, traversal_manifest)
    return RegionBenchmarkContext(
        selection=selection,
        region_manifest=region_manifest,
        traversal_manifest=traversal_manifest,
        suite=suite,
    )


def run_arsenal_region_v1_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new executions that would relabel current semantics as v1."""

    del (
        action_source,
        subject,
        region_manifest_path,
        traversal_manifest_path,
        action_carry_initializer,
        compile,
    )
    raise RuntimeError(
        f"{ARSENAL_REGION_V1} is historical; current source must use "
        f"{ARSENAL_REGION_V6}"
    )


def run_arsenal_region_v2_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new executions that would relabel current semantics as v2."""

    del (
        action_source,
        subject,
        region_manifest_path,
        traversal_manifest_path,
        action_carry_initializer,
        compile,
    )
    raise RuntimeError(
        f"{ARSENAL_REGION_V2} is historical; current source must use "
        f"{ARSENAL_REGION_V6}"
    )


def run_arsenal_region_v3_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new executions that would relabel current semantics as v3."""

    del (
        action_source,
        subject,
        region_manifest_path,
        traversal_manifest_path,
        action_carry_initializer,
        compile,
    )
    raise RuntimeError(
        f"{ARSENAL_REGION_V3} is historical; current source must use "
        f"{ARSENAL_REGION_V6}"
    )


def run_arsenal_region_v4_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new executions that would relabel current semantics as v4."""

    del (
        action_source,
        subject,
        region_manifest_path,
        traversal_manifest_path,
        action_carry_initializer,
        compile,
    )
    raise RuntimeError(
        f"{ARSENAL_REGION_V4} is historical; current source must use "
        f"{ARSENAL_REGION_V6}"
    )


def run_arsenal_region_v5_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Reject new executions that would relabel v6 semantics as v5."""

    del (
        action_source,
        subject,
        region_manifest_path,
        traversal_manifest_path,
        action_carry_initializer,
        compile,
    )
    raise RuntimeError(
        f"{ARSENAL_REGION_V5} is historical; current source must use "
        f"{ARSENAL_REGION_V6}"
    )


def run_arsenal_region_v6_trained(
    action_source: ActionSource,
    subject: Mapping[str, Any],
    *,
    region_manifest_path: str | Path,
    traversal_manifest_path: str | Path,
    action_carry_initializer: ActionCarryInitializer,
    compile: bool = True,
) -> dict[str, Any]:
    """Run Region v6 with exact geometry and target navigation bound."""

    context = load_region_v6_context(
        region_manifest_path,
        traversal_manifest_path,
    )
    suite = context.suite
    _validate_trained_subject(subject, suite=suite)
    reference_config = _reference_policy_config(
        len(PROFILE_NAMES) * len(EVALUATION_SEEDS)
    )
    untrained_policy = initialize_policy(
        jax.random.key(UNTRAINED_INITIALIZATION_SEED),
        reference_config,
    )
    rows_by_arm: dict[str, list[dict[str, Any]]] = {
        name: [] for name in TRAINED_COMPARISON_ARMS
    }
    for world in FROZEN_REGION_WORLDS:
        (
            params,
            runtime,
            reset_keys,
            labels,
            row_seeds,
            geometry,
            capability_provider,
            token_provider,
            target_navigation_provider,
        ) = _world_runtime(context, world)
        physical_geometry_bound = isinstance(geometry, RegionGeometryState)
        blockers = region_v1_runtime_blockers(
            physical_geometry_bound=physical_geometry_bound,
        )
        if blockers:
            raise RuntimeError(
                "Arsenal Region v6 publication is blocked:\n- "
                + "\n- ".join(blockers)
            )
        common = {
            "geometry_provider": geometry,
            "world_capability_provider": capability_provider,
            "world_token_provider": token_provider,
            "target_navigation_provider": target_navigation_provider,
            "max_policy_steps": MAX_POLICY_STEPS,
            "compile": compile,
        }
        evaluators = {
            "trained_policy": (
                make_arsenal_evaluator(
                    params,
                    runtime,
                    action_source=action_source,
                    action_carry_initializer=action_carry_initializer,
                    **common,
                ),
                None,
            ),
            "uniform_legal": (
                make_arsenal_evaluator(
                    params,
                    runtime,
                    action_source=uniform_legal_arsenal_action_source,
                    **common,
                ),
                None,
            ),
            "always_idle": (
                make_arsenal_evaluator(
                    params,
                    runtime,
                    action_source=always_idle_arsenal_action_source,
                    **common,
                ),
                None,
            ),
            "untrained_init": (
                make_arsenal_evaluator(params, runtime, **common),
                untrained_policy,
            ),
        }
        rollout_key = jax.random.fold_in(
            jax.random.key(ROLLOUT_SEED),
            world.evaluation_order,
        )
        for arm, (evaluator, policy) in evaluators.items():
            result = evaluator(policy, reset_keys, rollout_key)
            jax.block_until_ready(result)
            rows_by_arm[arm].extend(
                _world_rows(
                    result,
                    labels=labels,
                    seeds=row_seeds,
                    world=world,
                    arm=arm,
                    physical_geometry_bound=physical_geometry_bound,
                )
            )
    results = {
        arm: _region_statistics(rows, suite)
        for arm, rows in rows_by_arm.items()
    }
    subjects = {
        "trained_policy": dict(subject),
        **_baseline_subjects(reference_config),
    }
    artifact = _region_artifact(
        suite=suite,
        subjects=subjects,
        results=results,
    )
    errors = region_benchmark_artifact_errors(
        artifact,
        expected_kind="trained_comparison",
    )
    if errors:
        raise RuntimeError(
            "Region benchmark artifact failed its contract:\n- "
            + "\n- ".join(errors)
        )
    return artifact


def _world_runtime(
    context: RegionBenchmarkContext,
    world: FrozenRegionWorld,
):
    profiles = tuple(PROFILE_NAMES)
    labels = tuple(
        profile for _ in EVALUATION_SEEDS for profile in profiles
    )
    row_seeds = tuple(
        seed for seed in EVALUATION_SEEDS for _ in profiles
    )
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(labels))
    source = jnp.asarray(world.source_position, dtype=jnp.float32)
    destination = jnp.asarray(world.destination_position, dtype=jnp.float32)
    params = default_combat_params(
        microticks=1,
        target_active=True,
    )._replace(
        agent_spawn=source,
        target_offset=destination - source,
        floor_y=source[1],
    )
    world_ids = jnp.full(
        (len(labels),),
        world.selection_slot,
        dtype=jnp.int32,
    )
    geometry = region_geometry_from_atlas(
        context.selection.atlas,
        world_ids,
        agent_bounds=params.agent_bounds,
        target_bounds=params.target_bounds,
        agent_los_offset=params.agent_eye_offset,
        target_los_offset=params.target_eye_offset,
        # Perception budget, not block reach -- see the constant's note.
        max_los_distance=REGION_PERCEPTION_LOS_DISTANCE_BLOCKS,
    )
    role_mask = (
        context.selection.atlas.cell_flags.astype(jnp.int32)
        & jnp.int32(FLAG_OPAQUE)
    ) != 0
    capability_provider = partial(
        geometry_arsenal_world_capabilities,
        geometry=geometry,
        config=runtime,
        role_opaque_mask=role_mask,
    )
    traversal_atlas = context.selection.traversal_atlas
    if traversal_atlas is None:
        raise RuntimeError("Region traversal atlas is unavailable")

    def token_provider(state, combat_params):
        actor_position = state.combat.position[
            :,
            AGENT_ENTITY : AGENT_ENTITY + 1,
            :,
        ]
        traversal = query_region_traversal_tokens(
            traversal_atlas,
            world_ids,
            actor_position,
            token_capacity=(
                REGION_TRAVERSAL_POLICY_STAGING_CAPACITY
            ),
            max_distance=(
                DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance
            ),
        )
        return geometry_arsenal_world_tokens(
            state,
            combat_params,
            geometry=geometry,
            role_opaque_mask=role_mask,
            geometry_provenance=(
                WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY
            ),
            traversal=traversal,
            policy_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
        )

    reset_keys = jnp.stack(
        tuple(
            jax.random.fold_in(
                jax.random.key(seed),
                world.evaluation_order,
            )
            for seed in row_seeds
        ),
        axis=0,
    )
    return (
        params,
        runtime,
        reset_keys,
        labels,
        row_seeds,
        geometry,
        capability_provider,
        token_provider,
        surrogate_region_walk_target_navigation_provider,
    )


def _require_manifest_value(
    manifest: Mapping[str, Any],
    path: tuple[str, ...],
    expected: Any,
) -> None:
    value: Any = manifest
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            raise ValueError(f"manifest field {'.'.join(path)} is missing")
        value = value[name]
    if value != expected:
        raise ValueError(f"manifest field {'.'.join(path)} changed")


def _require_manifest_sha(
    manifest: Mapping[str, Any],
    path: tuple[str, ...],
    expected: str,
) -> None:
    value: Any = manifest
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            raise ValueError(f"manifest field {'.'.join(path)} is missing")
        value = value[name]
    if not isinstance(value, str) or value.upper() != expected:
        raise ValueError(f"manifest field {'.'.join(path)} changed")


__all__ = [
    "ARSENAL_REGION_LOCOMOTION_BLOCKER",
    "ARSENAL_REGION_V1",
    "ARSENAL_REGION_V1_SUITE_SHA256",
    "ARSENAL_REGION_V2",
    "ARSENAL_REGION_V2_SUITE_SHA256",
    "ARSENAL_REGION_V3",
    "ARSENAL_REGION_V3_SUITE_SHA256",
    "ARSENAL_REGION_V4",
    "ARSENAL_REGION_V4_SUITE_SHA256",
    "ARSENAL_REGION_V5",
    "ARSENAL_REGION_V5_SUITE_SHA256",
    "ARSENAL_REGION_V6",
    "ARSENAL_REGION_V6_SUITE_SHA256",
    "FROZEN_REGION_WORLDS",
    "load_region_v1_context",
    "load_region_v2_context",
    "load_region_v3_context",
    "load_region_v4_context",
    "load_region_v5_context",
    "load_region_v6_context",
    "region_benchmark_artifact_errors",
    "region_v1_runtime_blockers",
    "region_v1_suite_contract",
    "region_v2_suite_contract",
    "region_v3_suite_contract",
    "region_v4_suite_contract",
    "region_v5_suite_contract",
    "region_v6_suite_contract",
    "run_arsenal_region_v1_trained",
    "run_arsenal_region_v2_trained",
    "run_arsenal_region_v3_trained",
    "run_arsenal_region_v4_trained",
    "run_arsenal_region_v5_trained",
    "run_arsenal_region_v6_trained",
]
