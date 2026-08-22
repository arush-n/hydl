"""Leaf helpers extracted verbatim from arsenal_region_benchmark.py."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
from typing import Any
import jax
import numpy as np
from hytalegym.jax.evaluation.arsenal_benchmark import BENCHMARK_ARTIFACT_SCHEMA, BENCHMARK_ARTIFACT_VERSION, REFERENCE_BASELINES, TRAINED_COMPARISON_ARMS, UNTRAINED_INITIALIZATION_SEED, canonical_sha256, _reward_recomposes, _validate_trained_subject
from hytalegym.jax.training.baselines import BASELINE_IDENTITIES
from hytalegym.jax.training.evaluation import evaluation_statistics, pure_jax_process_provenance, summarize_evaluation_rows
from hytalegym.jax.world import RegionLibraryEnvironmentSelection
from hytalegym.worldgen.region import CHUNK_SIZE, CORE_BLOCKS_PER_AXIS
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
    _manifest_sha,
    _region_suite_contract,
    _region_v1_suite_contract,
    _region_v2_suite_contract,
    _region_v3_suite_contract,
    _region_v4_suite_contract,
    _region_v5_suite_contract,
    region_v1_suite_contract,
    region_v2_suite_contract,
    region_v3_suite_contract,
    region_v4_suite_contract,
    region_v5_suite_contract,
)


FROZEN_REGION_SUITE_SHA256_BY_ID = {
    ARSENAL_REGION_V1: ARSENAL_REGION_V1_SUITE_SHA256,
    ARSENAL_REGION_V2: ARSENAL_REGION_V2_SUITE_SHA256,
    ARSENAL_REGION_V3: ARSENAL_REGION_V3_SUITE_SHA256,
    ARSENAL_REGION_V4: ARSENAL_REGION_V4_SUITE_SHA256,
    ARSENAL_REGION_V5: ARSENAL_REGION_V5_SUITE_SHA256,
    ARSENAL_REGION_V6: ARSENAL_REGION_V6_SUITE_SHA256,
}


def region_benchmark_artifact_errors(
    artifact: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> list[str]:
    """Return every frozen Region artifact-contract violation."""

    errors: list[str] = []
    if artifact.get("schema") != BENCHMARK_ARTIFACT_SCHEMA:
        errors.append("artifact schema differs")
    if artifact.get("version") != BENCHMARK_ARTIFACT_VERSION:
        errors.append("artifact version differs")
    kind = artifact.get("kind")
    if expected_kind is not None and kind != expected_kind:
        errors.append(f"artifact kind must be {expected_kind!r}")
    if kind != "trained_comparison":
        errors.append("Region benchmark requires a trained comparison")
    suite = artifact.get("suite")
    if not isinstance(suite, Mapping):
        return errors + ["suite must be a mapping"]
    suite_sha256 = artifact.get("suite_sha256")
    if suite_sha256 != canonical_sha256(suite):
        errors.append("suite_sha256 does not match the embedded suite")
    suite_id = suite.get("id")
    expected_suite_sha256 = FROZEN_REGION_SUITE_SHA256_BY_ID.get(suite_id)
    if expected_suite_sha256 is None or suite_sha256 != expected_suite_sha256:
        errors.append("artifact is not for a frozen Arsenal Region suite")
    providers = suite.get("providers")
    physical_motion = (
        providers.get("physical_motion")
        if isinstance(providers, Mapping)
        else None
    )
    if (
        not isinstance(physical_motion, Mapping)
        or physical_motion.get("publication_ready") is not True
    ):
        errors.append(
            "Region publication is blocked because exact geometry is not "
            "bound to Arsenal physical motion"
        )
    contract = artifact.get("contract")
    if not isinstance(contract, Mapping):
        return errors + ["contract must be a mapping"]
    if artifact.get("contract_sha256") != canonical_sha256(contract):
        errors.append("contract_sha256 does not match the contract")
    if contract.get("suite_sha256") != suite_sha256:
        errors.append("contract suite identity differs")
    if contract.get("native_server_process") != pure_jax_process_provenance():
        errors.append(
            "Region pure-JAX artifact must declare uptime independence"
        )
    results = artifact.get("results")
    subjects = contract.get("subjects")
    if not isinstance(results, Mapping) or not isinstance(subjects, Mapping):
        return errors + ["results and contract subjects must be mappings"]
    if set(results) != set(TRAINED_COMPARISON_ARMS):
        errors.append("Region result must contain trained policy and controls")
    if set(subjects) != set(TRAINED_COMPARISON_ARMS):
        errors.append("Region subjects must contain trained policy and controls")
    if artifact.get("results_sha256") != canonical_sha256(results):
        errors.append("results_sha256 does not match the embedded results")
    trained = subjects.get("trained_policy")
    if isinstance(trained, Mapping):
        try:
            _validate_trained_subject(trained, suite=suite)
        except (TypeError, ValueError) as error:
            errors.append(f"trained policy subject is invalid: {error}")
    else:
        errors.append("trained policy subject must be a mapping")
    expected_keys = _expected_episode_keys(suite)
    for arm in TRAINED_COMPARISON_ARMS:
        result = results.get(arm)
        if not isinstance(result, Mapping):
            errors.append(f"result {arm!r} must be a mapping")
            continue
        rows = result.get("episodes")
        if not isinstance(rows, list):
            errors.append(f"result {arm!r} episodes must be a list")
            continue
        if result.get("episode_count") != suite.get("episode_count"):
            errors.append(f"result {arm!r} episode count differs")
        if len(rows) != suite.get("episode_count"):
            errors.append(f"result {arm!r} episode rows differ")
            continue
        if not all(isinstance(row, Mapping) for row in rows):
            errors.append(f"result {arm!r} episode rows must be mappings")
            continue
        keys = [_episode_key(row) for row in rows]
        if keys != expected_keys:
            errors.append(f"result {arm!r} episode keys differ")
        if any(
            row.get("world_capabilities_ready") is not True
            for row in rows
        ):
            errors.append(f"result {arm!r} has unavailable capability rows")
        if any(row.get("world_geometry_ready") is not True for row in rows):
            errors.append(f"result {arm!r} has unavailable World-token rows")
        if any(
            row.get("physical_geometry_bound") is not True for row in rows
        ):
            errors.append(
                f"result {arm!r} has unavailable physical-geometry rows"
            )
        if not _rows_match_world_contract(rows, suite):
            errors.append(f"result {arm!r} world/profile metadata differs")
        objective = suite.get("reward_objective")
        if not isinstance(objective, Mapping) or not _reward_recomposes(
            rows,
            objective,
        ):
            errors.append(f"result {arm!r} reward components do not recompose")
        try:
            expected_result = _region_statistics(rows, suite)
        except (KeyError, TypeError, ValueError):
            errors.append(f"result {arm!r} summaries are not derivable")
        else:
            if result != expected_result:
                errors.append(f"result {arm!r} summaries differ from rows")
    if set(results) == set(TRAINED_COMPARISON_ARMS):
        try:
            comparisons = _region_comparisons(results, suite)
        except (AttributeError, KeyError, TypeError, ValueError):
            errors.append("Region comparisons are not derivable")
        else:
            if artifact.get("comparisons") != comparisons:
                errors.append("Region comparisons differ from episode rows")
            expected_interpretation = _region_interpretation(
                suite,
                results,
                comparisons,
            )
            if artifact.get("interpretation") != expected_interpretation:
                errors.append("Region interpretation differs from result facts")
    return errors


def _world_rows(
    result,
    *,
    labels: Sequence[str],
    seeds: Sequence[int],
    world: FrozenRegionWorld,
    arm: str,
    physical_geometry_bound: bool,
) -> list[dict[str, Any]]:
    capabilities = np.asarray(
        jax.device_get(result.world_capabilities_ready),
        dtype=np.bool_,
    )
    geometry = np.asarray(
        jax.device_get(result.world_geometry_ready),
        dtype=np.bool_,
    )
    if not np.all(capabilities):
        raise RuntimeError(
            f"{arm} world {world.semantic_sha256} lost capability availability"
        )
    if not np.all(geometry):
        raise RuntimeError(
            f"{arm} world {world.semantic_sha256} lost World-token availability"
        )
    statistics = evaluation_statistics(
        result,
        labels=labels,
        seeds=seeds,
    )
    rows = statistics["episodes"]
    for index, row in enumerate(rows):
        row.update(
            {
                "world_semantic_sha256": world.semantic_sha256,
                "world_graph_semantic_sha256": (
                    world.graph_semantic_sha256
                ),
                "world_seed": world.seed,
                "world_evaluation_order": world.evaluation_order,
                "world_capabilities_ready": bool(capabilities[index]),
                "world_geometry_ready": bool(geometry[index]),
                "physical_geometry_bound": physical_geometry_bound,
            }
        )
    return rows


def _region_statistics(
    rows: Sequence[Mapping[str, Any]],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    summary = summarize_evaluation_rows(values)
    summary["episodes"] = values
    summary["by_world"] = {
        world["semantic_sha256"]: summarize_evaluation_rows(
            [
                row
                for row in values
                if row["world_semantic_sha256"]
                == world["semantic_sha256"]
            ]
        )
        for world in suite["region_corpus"]["worlds"]
    }
    summary["by_profile"] = {
        label: summarize_evaluation_rows(
            [row for row in values if row["label"] == label]
        )
        for label in suite["profiles"]
    }
    return summary


def _region_artifact(
    *,
    suite: Mapping[str, Any],
    subjects: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    suite_sha256 = canonical_sha256(suite)
    contract = {
        "schema": "hytalerl_region_benchmark_run_contract_v1",
        "version": 1,
        "kind": "trained_comparison",
        "suite_sha256": suite_sha256,
        "action_source_interface": (
            "(LearnerCombatObservationV3, carry, jax_key) "
            + (
                "-> (explicit_action_factors, carry)"
                if int(suite["version"]) >= 2
                else "-> (packed_action_ids, carry)"
            )
        ),
        "subjects": dict(subjects),
        "native_server_process": pure_jax_process_provenance(),
    }
    comparisons = _region_comparisons(results, suite)
    return {
        "schema": BENCHMARK_ARTIFACT_SCHEMA,
        "version": BENCHMARK_ARTIFACT_VERSION,
        "kind": "trained_comparison",
        "suite": dict(suite),
        "suite_sha256": suite_sha256,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "results": dict(results),
        "results_sha256": canonical_sha256(results),
        "comparisons": comparisons,
        "interpretation": _region_interpretation(
            suite,
            results,
            comparisons,
        ),
    }


def _region_comparisons(
    results: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        control: _paired_region_comparison(
            results["trained_policy"],
            results[control],
            suite,
        )
        for control in REFERENCE_BASELINES
    }


def _paired_region_comparison(
    subject: Mapping[str, Any],
    reference: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    subject_rows = subject["episodes"]
    reference_rows = reference["episodes"]
    if [_episode_key(row) for row in subject_rows] != [
        _episode_key(row) for row in reference_rows
    ]:
        raise ValueError("Region benchmark episode keys are not identical")
    return {
        **_paired_row_summary(subject_rows, reference_rows),
        "by_world": {
            world["semantic_sha256"]: _paired_row_summary(
                [
                    row
                    for row in subject_rows
                    if row["world_semantic_sha256"]
                    == world["semantic_sha256"]
                ],
                [
                    row
                    for row in reference_rows
                    if row["world_semantic_sha256"]
                    == world["semantic_sha256"]
                ],
            )
            for world in suite["region_corpus"]["worlds"]
        },
        "by_profile": {
            profile: _paired_row_summary(
                [row for row in subject_rows if row["label"] == profile],
                [row for row in reference_rows if row["label"] == profile],
            )
            for profile in suite["profiles"]
        },
    }


def _paired_row_summary(
    subject_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [_episode_key(row) for row in subject_rows] != [
        _episode_key(row) for row in reference_rows
    ]:
        raise ValueError("paired Region rows differ")
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
            sum(bool(row["success"]) for row in subject_rows)
            / len(subject_rows)
            - sum(bool(row["success"]) for row in reference_rows)
            / len(reference_rows)
        ),
    }


def _region_interpretation(
    suite: Mapping[str, Any],
    results: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    suite_id = str(suite["id"])
    suite_version = int(suite["version"])
    return {
        "covers": (
            "four exact held-out Region worlds, 31 shipped loadout profiles, "
            "three fixed episode seeds, native-exact traversal policy tokens, "
            "raw reward components, and deterministic legal spawn edges"
        ),
        "limitations": dict(suite["limitations"]),
        "measurement_provenance": {
            "mode": "independent_suite_execution",
            "execution_suite_id": suite_id,
            "execution_suite_sha256": canonical_sha256(suite),
            "execution_results_sha256": canonical_sha256(results),
            f"independent_region_v{suite_version}_execution": True,
            "statement": (
                f"Region v{suite_version} trained and control rows were "
                "independently executed under the embedded suite"
            ),
        },
        "trained_comparison_findings": {
            "arms": {
                name: {
                    "episode_return": dict(
                        results[name]["episode_return"]
                    ),
                    "success_rate": float(results[name]["success_rate"]),
                }
                for name in TRAINED_COMPARISON_ARMS
            },
            "paired_against_controls": dict(comparisons),
            "training_beats_uniform_on_mean_return": (
                comparisons["uniform_legal"]["episode_return_delta"][
                    "mean"
                ]
                > 0.0
            ),
            "statement": (
                "the flat-trained arm is published regardless of sign; "
                "this is a held-out Region transfer probe, not convergence"
            ),
        },
    }


def _baseline_subjects(reference_config) -> dict[str, Any]:
    return {
        name: {
            **BASELINE_IDENTITIES[name],
            **(
                {
                    "initialization_seed": UNTRAINED_INITIALIZATION_SEED,
                    "policy_config": asdict(reference_config),
                }
                if name == "untrained_init"
                else {}
            ),
        }
        for name in REFERENCE_BASELINES
    }


def _expected_episode_keys(
    suite: Mapping[str, Any],
) -> list[tuple[str, str, int]]:
    return [
        (world["semantic_sha256"], profile, seed)
        for world in suite["region_corpus"]["worlds"]
        for seed in suite["episode_seeds"]
        for profile in suite["profiles"]
    ]


def _episode_key(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("world_semantic_sha256"),
        row.get("label"),
        row.get("seed"),
    )


def _rows_match_world_contract(
    rows: Sequence[Mapping[str, Any]],
    suite: Mapping[str, Any],
) -> bool:
    worlds = {
        world["semantic_sha256"]: world
        for world in suite["region_corpus"]["worlds"]
    }
    weapon_ids = suite["profile_weapon_ids"]
    for row in rows:
        world = worlds.get(row.get("world_semantic_sha256"))
        label = row.get("label")
        if world is None or label not in weapon_ids:
            return False
        if (
            row.get("world_graph_semantic_sha256")
            != world["graph_semantic_sha256"]
            or row.get("world_seed") != world["seed"]
            or row.get("world_evaluation_order")
            != world["evaluation_order"]
            or row.get("profile_weapon_id") != weapon_ids[label]
        ):
            return False
    return True


def _validate_loaded_worlds(
    selection: RegionLibraryEnvironmentSelection,
    traversal_manifest: Mapping[str, Any],
) -> None:
    traversal_atlas = selection.traversal_atlas
    if traversal_atlas is None:
        raise RuntimeError("Region selection lacks its traversal graph")
    world_order = tuple(
        int(value)
        for value in np.asarray(
            jax.device_get(selection.environment_world_id)
        ).tolist()
    )
    expected_order = tuple(
        world.selection_slot for world in FROZEN_REGION_WORLDS
    )
    if world_order != expected_order:
        raise ValueError("held-out Region world assignment changed")
    graph_entries = {
        entry["source_region_semantic_sha256"].upper(): entry
        for entry in traversal_manifest["entries"]
    }
    for world in FROZEN_REGION_WORLDS:
        actual_sha = selection.working_set.artifact_semantic_sha256[
            world.selection_slot
        ].upper()
        actual_seed = selection.working_set.artifact_seed[
            world.selection_slot
        ]
        if actual_sha != world.semantic_sha256 or actual_seed != world.seed:
            raise ValueError(
                "held-out Region selection changed at slot "
                f"{world.selection_slot}: expected "
                f"{world.semantic_sha256!r}/{world.seed!r} "
                f"({type(world.seed).__name__}), got "
                f"{actual_sha!r}/{actual_seed!r} "
                f"({type(actual_seed).__name__})"
            )
        graph = graph_entries.get(world.semantic_sha256)
        if (
            graph is None
            or graph["graph_semantic_sha256"].upper()
            != world.graph_semantic_sha256
        ):
            raise ValueError("held-out Region traversal graph changed")
        selected = _select_world_edge(selection, world.selection_slot)
        if selected != world:
            raise ValueError(
                f"Region spawn edge changed for {world.semantic_sha256}"
            )


def _select_world_edge(
    selection: RegionLibraryEnvironmentSelection,
    selection_slot: int,
) -> FrozenRegionWorld:
    atlas = selection.traversal_atlas
    if atlas is None:
        raise RuntimeError("Region selection lacks its traversal graph")
    semantic_sha256 = selection.working_set.artifact_semantic_sha256[
        selection_slot
    ].upper()
    seed = selection.working_set.artifact_seed[selection_slot]
    node_mask = np.asarray(
        jax.device_get(atlas.node_mask[selection_slot]),
        dtype=np.bool_,
    )
    edge_mask = np.asarray(
        jax.device_get(atlas.edge_mask[selection_slot]),
        dtype=np.bool_,
    )
    positions = np.asarray(
        jax.device_get(atlas.node_position[selection_slot]),
        dtype=np.float32,
    )
    core_origin = (
        np.asarray(
            jax.device_get(atlas.core_min_chunk_xz[selection_slot]),
            dtype=np.int32,
        )
        * CHUNK_SIZE
    )
    blocks = np.floor(positions).astype(np.int32)
    inside_core = np.all(
        (blocks[:, (0, 2)] >= core_origin)
        & (blocks[:, (0, 2)] < core_origin + CORE_BLOCKS_PER_AXIS),
        axis=1,
    )
    sources = np.flatnonzero(
        node_mask & inside_core & np.any(edge_mask, axis=1)
    )
    candidates = (
        (
            hashlib.sha256(
                (
                    f"{EDGE_SELECTION_NAMESPACE}\0"
                    f"{semantic_sha256.lower()}\0"
                    f"{source}\0{slot}"
                ).encode("ascii")
            ).digest(),
            int(source),
            int(slot),
        )
        for source in sources
        for slot in np.flatnonzero(edge_mask[source])
    )
    digest, source, slot = min(candidates)
    destination = int(
        jax.device_get(
            atlas.edge_destination[selection_slot, source, slot]
        )
    )
    evaluation_order = next(
        world.evaluation_order
        for world in FROZEN_REGION_WORLDS
        if world.selection_slot == selection_slot
    )
    return FrozenRegionWorld(
        evaluation_order=evaluation_order,
        selection_slot=selection_slot,
        seed=int(seed),
        semantic_sha256=semantic_sha256,
        graph_semantic_sha256=next(
            world.graph_semantic_sha256
            for world in FROZEN_REGION_WORLDS
            if world.selection_slot == selection_slot
        ),
        legal_core_source_nodes=int(sources.size),
        directed_edges=int(
            np.count_nonzero(edge_mask & node_mask[:, None])
        ),
        source_node=source,
        edge_slot=slot,
        destination_node=destination,
        source_position=tuple(
            float(value) for value in positions[source]
        ),
        destination_position=tuple(
            float(value) for value in positions[destination]
        ),
        edge_kind=int(
            jax.device_get(atlas.edge_kind[selection_slot, source, slot])
        ),
        edge_flags=int(
            jax.device_get(atlas.edge_flags[selection_slot, source, slot])
        ),
        edge_cost=float(
            jax.device_get(atlas.edge_cost[selection_slot, source, slot])
        ),
        edge_selection_sha256=digest.hex().upper(),
    )
