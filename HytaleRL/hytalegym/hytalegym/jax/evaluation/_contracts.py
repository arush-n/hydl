"""Leaf helpers extracted verbatim from arsenal_region_benchmark.py."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from hytalegym.jax.combat import AGENT_ENTITY, ARSENAL_POLICY_ACTION_HEAD_SIZES, DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG, PROFILE_NAMES, arsenal_policy_contract_sha256, combat_arsenal_contract_sha256, default_combat_params, hytale_0_5_7_loadouts, learner_observation_v3_contract_sha256
from hytalegym.jax.combat.observation.v3 import (
    world_geometry_policy_config_manifest,
)
from hytalegym.jax.evaluation.arsenal_benchmark import EVALUATION_SEEDS, MAX_POLICY_STEPS, REFERENCE_BASELINES, ROLLOUT_SEED, TRAINED_COMPARISON_ARMS, canonical_sha256
from hytalegym.jax.training.baselines import LEGACY_BASELINE_IDENTITIES
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    current_combat_checkpoint_contract,
)
from hytalegym.jax.world import REGION_ARTIFACT_ASSIGNMENT_METHOD, REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION, REGION_TRAVERSAL_POLICY_STAGING_CAPACITY
from hytalegym.rulesets import combat_ruleset_sha256


ARSENAL_REGION_V1 = "hytalerl_arsenal_region_v1"


ARSENAL_REGION_V1_SUITE_SHA256 = (
    "1B38F0923FCC16B70112435D34B6474F9214D31FBFB004B20B70386B9C9DA447"
)


ARSENAL_REGION_V2 = "hytalerl_arsenal_region_v2"


ARSENAL_REGION_V2_SUITE_SHA256 = (
    "904598AB8332839593802C21A832DC2E31A9360E93B37FB387338A9733258DCF"
)


ARSENAL_REGION_V3 = "hytalerl_arsenal_region_v3"


ARSENAL_REGION_V3_SUITE_SHA256 = (
    "2C93CFB9FB63AEA4401FE1032B986DAE46EB650A9DB4220C42D2ED3EC39A4C5D"
)


ARSENAL_REGION_V4 = "hytalerl_arsenal_region_v4"


ARSENAL_REGION_V4_SUITE_SHA256 = (
    "FECDE4FD5939FC96D18206C5B9BE313774E49823082BADDD7FCED6AF8A8CD8A9"
)


ARSENAL_REGION_V5 = "hytalerl_arsenal_region_v5"


ARSENAL_REGION_V5_SUITE_SHA256 = (
    "A2AD0DB9F1E644E8E9B08FD07B30D9039B98CCC2E0728D007AF8D798650D06D1"
)


ARSENAL_REGION_V6 = "hytalerl_arsenal_region_v6"


ARSENAL_REGION_V6_SUITE_SHA256 = (
    "4A52B017C74A149C848AECB0DC435AD2FA6F5964E12DBAB041C49609CD1B4494"
)


REGION_LIBRARY_SEMANTIC_SHA256 = (
    "F0EA0526AC1EE6184FC575C085607EE7B3F9843D78B0D4F806FFF15B51C8616C"
)


REGION_CAPTURE_EVIDENCE_BRIDGE_SHA256 = (
    "1644DC99158B50A946828DB4E8A3931188C574B856303A69B0BBA6D39B67B245"
)


REGION_SELECTION_KEY = 1197923983


REGION_ASSIGNMENT_KEY = 1197923984


REGION_WORLD_COUNT = 4


REGION_EPISODE_COUNT = (
    REGION_WORLD_COUNT * len(PROFILE_NAMES) * len(EVALUATION_SEEDS)
)


EDGE_SELECTION_METHOD = "sha256_min_lexicographic_v1"


EDGE_SELECTION_NAMESPACE = ARSENAL_REGION_V1


EDGE_SELECTION_PAYLOAD_V1 = (
    "suite_id + NUL + world_semantic_sha256 + NUL + "
    "source_node_decimal + NUL + edge_slot_decimal"
)


EDGE_SELECTION_PAYLOAD_V2 = (
    "edge_selection_namespace + NUL + world_semantic_sha256 + NUL + "
    "source_node_decimal + NUL + edge_slot_decimal"
)


@dataclass(frozen=True)
class FrozenRegionWorld:
    evaluation_order: int
    selection_slot: int
    seed: int
    semantic_sha256: str
    graph_semantic_sha256: str
    legal_core_source_nodes: int
    directed_edges: int
    source_node: int
    edge_slot: int
    destination_node: int
    source_position: tuple[float, float, float]
    destination_position: tuple[float, float, float]
    edge_kind: int
    edge_flags: int
    edge_cost: float
    edge_selection_sha256: str


FROZEN_REGION_WORLDS = (
    FrozenRegionWorld(
        evaluation_order=0,
        selection_slot=1,
        seed=1176664667,
        semantic_sha256=(
            "2262DDAFBA8D1C63141541185CDA1661ADACEF830AB308FF6EEF0AB503B222B2"
        ),
        graph_semantic_sha256=(
            "03D7265A1BA1E892DB4D8A0D0597F1631808C15B0598CA7253AA2954F4340E76"
        ),
        legal_core_source_nodes=21027,
        directed_edges=224612,
        source_node=17846,
        edge_slot=5,
        destination_node=18173,
        source_position=(11.5, 122.0, 5.5),
        destination_position=(12.5, 123.0, 4.5),
        edge_kind=2,
        edge_flags=1,
        edge_cost=1.4142135381698608,
        edge_selection_sha256=(
            "0000215B95492C58AF00AA04471DFE2D73CDB4EADFF64AF8E68684E486D9AC91"
        ),
    ),
    FrozenRegionWorld(
        evaluation_order=1,
        selection_slot=3,
        seed=1921897808,
        semantic_sha256=(
            "24153767C7E69D6B6380B70D31B664C426BEE3C0A4953003733028E69A3E1A7B"
        ),
        graph_semantic_sha256=(
            "8C0CEB854F73AD0AAE3CDA2753DAEB78B3A33EDF7F02CFDC461881B0688CE20B"
        ),
        legal_core_source_nodes=26955,
        directed_edges=269254,
        source_node=16398,
        edge_slot=2,
        destination_node=16116,
        source_position=(-29.5, 138.0, 18.5),
        destination_position=(-30.5, 138.0, 19.5),
        edge_kind=1,
        edge_flags=1,
        edge_cost=1.4142135381698608,
        edge_selection_sha256=(
            "000018F9375A1C9EA3D03F2CDBEF523ACFDAF3ADF1D80BD4C046F4408CBCE0BB"
        ),
    ),
    FrozenRegionWorld(
        evaluation_order=2,
        selection_slot=0,
        seed=1397671741,
        semantic_sha256=(
            "9D4BAC549FDD9D8BDDDF6802D230018B8B96FCA9D2B24521DF80C3CACE8C468C"
        ),
        graph_semantic_sha256=(
            "57A3A6D0726CDA0724FF9D7705F0F0B97BF73E87F501136D59C75C2FFDAD7649"
        ),
        legal_core_source_nodes=23227,
        directed_edges=238432,
        source_node=6802,
        edge_slot=0,
        destination_node=6448,
        source_position=(-58.5, 64.0, 22.5),
        destination_position=(-59.5, 63.0, 21.5),
        edge_kind=3,
        edge_flags=1,
        edge_cost=1.4142135381698608,
        edge_selection_sha256=(
            "00006AD5EB699041EB4E9465BE7C3DE317CA59392DE503FFA80D39E17E1BF77A"
        ),
    ),
    FrozenRegionWorld(
        evaluation_order=3,
        selection_slot=2,
        seed=1274646439,
        semantic_sha256=(
            "CB99FE50BC5CD81C1F851DB7B1324F85997E287EA375254C44AF409F43BC3579"
        ),
        graph_semantic_sha256=(
            "43D1C3CEBF04EF935B6745415032BB7C907D783FF8BAC47D6BCE5205AA771945"
        ),
        legal_core_source_nodes=21772,
        directed_edges=237910,
        source_node=6759,
        edge_slot=0,
        destination_node=6473,
        source_position=(-25.5, 89.0, 32.5),
        destination_position=(-26.5, 90.0, 31.5),
        edge_kind=2,
        edge_flags=1,
        edge_cost=1.4142135381698608,
        edge_selection_sha256=(
            "00004127B0FC9FBE73CD30CD8698E52F6EDB00DB6A0AF73C50C98F2F72100624"
        ),
    ),
)


def _region_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
    suite_id: str,
    version: int,
) -> dict[str, Any]:
    profiles = tuple(PROFILE_NAMES)
    loadouts = hytale_0_5_7_loadouts(profiles)
    weapon_ids = np.asarray(loadouts.weapon_id)[:, AGENT_ENTITY]
    params = default_combat_params(microticks=1, target_active=True)
    traversal_semantic_sha256 = _manifest_sha(
        traversal_manifest,
        "library_semantic_sha256",
    )
    traversal_evidence_bridge_sha256 = _manifest_sha(
        traversal_manifest,
        "native_evidence_jar_sha256",
    )
    return {
        "schema": f"hytalerl_arsenal_region_suite_v{version}",
        "version": version,
        "id": suite_id,
        "environment": "jax_arsenal",
        "task": "kweebec_razorleaf_vs_trork_brawler",
        "opponent": {
            "role": "Trork_Brawler",
            "selector": "ordered_legacy_attack_sequence",
        },
        "usage": REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION,
        "episode_key_fields": [
            "world_semantic_sha256",
            "label",
            "seed",
        ],
        "episode_seeds": list(EVALUATION_SEEDS),
        "rollout_seed": ROLLOUT_SEED,
        "reset_key_derivation": (
            "jax.random.fold_in(jax.random.key(episode_seed), "
            "world_evaluation_order)"
        ),
        "rollout_key_derivation": (
            "jax.random.fold_in(jax.random.key(rollout_seed), "
            "world_evaluation_order)"
        ),
        "episodes_per_world_profile": len(EVALUATION_SEEDS),
        "world_count": REGION_WORLD_COUNT,
        "episode_count": REGION_EPISODE_COUNT,
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
            }
            if version >= 2
            else {
                "encoding": "packed_categorical_heads",
                "head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
                "packed_action_count": math.prod(
                    ARSENAL_POLICY_ACTION_HEAD_SIZES
                ),
            }
        ),
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
        "region_corpus": {
            "region_library": {
                "schema": "hytalerl_region_artifact_library_v1",
                "version": 1,
                "semantic_sha256": REGION_LIBRARY_SEMANTIC_SHA256,
                "capture_evidence_bridge_sha256": (
                    REGION_CAPTURE_EVIDENCE_BRIDGE_SHA256
                ),
            },
            "traversal_library": {
                "schema": "hytalerl_native_region_traversal_library_v2",
                "version": 2,
                "semantic_sha256": traversal_semantic_sha256,
                "source_region_library_semantic_sha256": (
                    REGION_LIBRARY_SEMANTIC_SHA256
                ),
                "capture_evidence_bridge_sha256": (
                    traversal_evidence_bridge_sha256
                ),
            },
            "selection": {
                "split": "heldout",
                "usage": REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION,
                "selection_method": "sha256_rank_v1",
                "selection_key": REGION_SELECTION_KEY,
                "assignment_method": REGION_ARTIFACT_ASSIGNMENT_METHOD,
                "assignment_key": REGION_ASSIGNMENT_KEY,
                "edge_selection_method": EDGE_SELECTION_METHOD,
                "edge_selection_payload": (
                    EDGE_SELECTION_PAYLOAD_V2
                    if version >= 2
                    else EDGE_SELECTION_PAYLOAD_V1
                ),
                **(
                    {
                        "edge_selection_namespace": (
                            EDGE_SELECTION_NAMESPACE
                        )
                    }
                    if version >= 2
                    else {}
                ),
            },
            "worlds": [asdict(world) for world in FROZEN_REGION_WORLDS],
        },
        "providers": {
            "capabilities": {
                "id": "exact_region_geometry_v1",
                "availability": "required_throughout_active_episode",
            },
            "world_geometry": {
                "id": "native_exact_region_traversal_tokens_v1",
                "runtime_provenance": "native_exact_geometry",
                "availability": "required_nonempty_throughout_active_episode",
                "traversal_staging_capacity": (
                    REGION_TRAVERSAL_POLICY_STAGING_CAPACITY
                ),
                "checkpoint_compatible_policy_config": (
                    world_geometry_policy_config_manifest(
                        DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG
                    )
                ),
                "provenance_boundary": (
                    "checkpoint policy_config retains its surrogate training "
                    "label; runtime tokens carry native_exact_geometry"
                ),
            },
            "physical_motion": {
                "id": "native_exact_region_geometry_v1",
                "publication_ready": True,
                "runtime_provenance": "native_exact_region_geometry",
                "required_runtime_provenance": "native_exact_region_geometry",
                "availability": "required_throughout_active_episode",
                "measurement_field": "physical_geometry_bound",
                "evidence": [
                    (
                        "jax/training/arsenal_evaluation.py:"
                        "make_arsenal_evaluator(geometry_provider=)"
                    ),
                    (
                        "jax/combat/arsenal/runtime.py:"
                        "reset_arsenal_batch(geometry=)"
                    ),
                    (
                        "jax/combat/arsenal/runtime.py:"
                        "step_arsenal_batch(geometry=)"
                    ),
                ],
                "binding": (
                    "the selected RegionGeometryState is passed as "
                    "geometry_provider to every arm"
                ),
            },
            **(
                {
                    "target_navigation": {
                        "id": "surrogate_region_walk_navigation_v1",
                        "runtime_provenance": (
                            "native_exact_region_graph_projected_by_"
                            "surrogate_region_walk"
                        ),
                        "availability": "bound_to_every_evaluation_arm",
                        "certification_scope": (
                            "bounded_base_walk_arrival_not_full_native_"
                            "MotionControllerWalk"
                        ),
                    }
                }
                if version >= 6
                else {}
            ),
        },
        "cost_budget": {
            "basis": "pre_motion_seam_region_context_ppo_update_v1",
            "evidence": {
                "filename": "p5-region-training-cost-570057.json",
                "report_semantic_sha256": (
                    "FB8BF09566E165F3E07467179BCC377C367256FD6353DF39BFF899ACE5D1FDE3"
                ),
                "contention_recorded": True,
                "isolated_capacity_claim": False,
                "reported_geometry_label": "exact_native_region",
                "physical_motion_provenance": "open_flat_floor",
                "post_motion_seam_remeasurement_required": True,
            },
            "measurement_scope": {
                "world_split": "training",
                "environment_count": 32,
                "rollout_decisions": 4,
                "transitions": 128,
                "optimizer_updates": 1,
                "actor_legal_native_graph_tokens": True,
                "includes_policy_collection": True,
                "includes_optimizer": True,
                "exact_region_capability_queries": True,
                "exact_region_physical_motion": False,
                "region_v1_timing": False,
            },
            "pre_seam_planning_lower_bound": {
                "transitions_per_second": 446.875259658133,
                "historical_flat_transitions_per_second": (
                    2603.351000650383
                ),
                "cost_multiplier_vs_historical_flat": (
                    5.825677175867802
                ),
                "maximum_suite_policy_transitions": (
                    len(TRAINED_COMPARISON_ARMS)
                    * REGION_EPISODE_COUNT
                    * MAX_POLICY_STEPS
                ),
                "suite_seconds_at_pre_seam_rate": (
                    len(TRAINED_COMPARISON_ARMS)
                    * REGION_EPISODE_COUNT
                    * MAX_POLICY_STEPS
                    / 446.875259658133
                ),
                "interpretation": (
                    "optimistic historical lower bound on elapsed time; the "
                    "exact-physical-motion suite has not been remeasured"
                ),
                "excludes": [
                    "cold compilation",
                    "checkpoint I/O",
                    "artifact writing",
                ],
            },
            "elapsed_share_of_realistic_update": {
                "exact_region_capability_queries": 0.6685529690264581,
                "actor_legal_token_path": 0.16208654438010625,
                "matched_flat_rollout": 0.12615672614369286,
                "optimizer": 0.04320376044974284,
            },
            "post_landing_transition_evidence": {
                "scope": (
                    "complete Arsenal PPO environment transitions; "
                    "tokens off; no policy collection or optimizer; "
                    "physical motion remains open-flat"
                ),
                "reports": [
                    {
                        "filename": (
                            "p4-static-flattening-postlanding-570057.json"
                        ),
                        "report_semantic_sha256": (
                            "DDE8B3DD85C63D1F2ACE508F9EB0011175740E12A8511AD9E64177ECD9027324"
                        ),
                        "production_transitions_per_second": (
                            814.5084314302134
                        ),
                    },
                    {
                        "filename": (
                            "p4-static-flattening-postlanding-570058.json"
                        ),
                        "report_semantic_sha256": (
                            "93FB057431F87A645DB24CB59F8C164992AA624EB86371E8E8DAA625F28E2B75"
                        ),
                        "production_transitions_per_second": (
                            1059.798729321554
                        ),
                    },
                ],
                "contention_recorded": True,
                "isolated_capacity_claim": False,
                "projection_comparison_valid": False,
                "projection_invalid_reason": (
                    "the measured comparison covered traversal moves from "
                    "EF2FF66F4E3750C31C73837DD1A283AA89B468941DCBC35DEC951D09241783B1 "
                    "to "
                    "2378C2DCB7E0DF5688BC1D47131F7F9B1B2552249975C631A0A99E52BB018F1E; "
                    "the current Region suite now pins "
                    f"{traversal_semantic_sha256}"
                ),
            },
        },
        "reference_baselines": {
            name: LEGACY_BASELINE_IDENTITIES[name]
            for name in REFERENCE_BASELINES
        },
        "publication": {
            "kind": "trained_comparison",
            "required_arms": list(TRAINED_COMPARISON_ARMS),
            "trained_policy_checkpoint": {
                "match": "exact_current",
                "contract": current_combat_checkpoint_contract(
                    policy_surface=ARSENAL_POLICY_SURFACE,
                ),
            },
            "trained_policy_execution": "public_action_source",
        },
        "limitations": {
            "policy_training_geometry": "open_flat_control",
            "policy_training_traversal_provenance": "surrogate",
            "evaluation_traversal_provenance": "native_exact",
            "train_split_checkpoint": False,
            "profile_content_held_out": False,
            "entity_count": 2,
            "native_transfer": False,
            "evaluation_physical_motion": "native_exact_region_geometry",
            "moving_traversal": (
                "exact Region physical motion is bound; graph tokens certify "
                "static arrivals, not saved native per-tick trajectories"
            ),
            "target_navigation": (
                "surrogate_region_walk_provider_bound_to_every_arm_with_"
                "native_exact_graph_and_bounded_base_walk_scope"
                if version >= 6
                else (
                    "target_navigation_unsupported remains a known port gap "
                    "and must accompany any reported result"
                )
            ),
            "throughput": (
                "446.875259658133 transitions/s is a contention-recorded "
                "pre-motion-seam lower-bound basis, not isolated capacity "
                "or a measured exact-motion Region suite rate"
            ),
        },
    }


def _region_v1_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _region_suite_contract(
        traversal_manifest=traversal_manifest,
        suite_id=ARSENAL_REGION_V1,
        version=1,
    )


def _region_v2_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _region_suite_contract(
        traversal_manifest=traversal_manifest,
        suite_id=ARSENAL_REGION_V2,
        version=2,
    )


def _region_v3_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _region_suite_contract(
        traversal_manifest=traversal_manifest,
        suite_id=ARSENAL_REGION_V3,
        version=3,
    )


def _region_v4_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _region_suite_contract(
        traversal_manifest=traversal_manifest,
        suite_id=ARSENAL_REGION_V4,
        version=4,
    )


def _region_v5_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _region_suite_contract(
        traversal_manifest=traversal_manifest,
        suite_id=ARSENAL_REGION_V5,
        version=5,
    )


def _region_v6_suite_contract(
    *,
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _region_suite_contract(
        traversal_manifest=traversal_manifest,
        suite_id=ARSENAL_REGION_V6,
        version=6,
    )


def region_v1_suite_contract(
    *,
    traversal_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return Region v1 only while source contracts match its frozen identity."""

    traversal_manifest = json.loads(
        Path(traversal_manifest_path).read_text(encoding="utf-8")
    )
    return _checked_region_v1_suite_contract(traversal_manifest)


def region_v2_suite_contract(
    *,
    traversal_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return historical v2 only while source still matches its identity."""

    traversal_manifest = json.loads(
        Path(traversal_manifest_path).read_text(encoding="utf-8")
    )
    return _checked_region_v2_suite_contract(traversal_manifest)


def region_v3_suite_contract(
    *,
    traversal_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return historical v3 only while source still matches its identity."""

    traversal_manifest = json.loads(
        Path(traversal_manifest_path).read_text(encoding="utf-8")
    )
    return _checked_region_v3_suite_contract(traversal_manifest)


def region_v4_suite_contract(
    *,
    traversal_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return current Region v4 while its full source contract matches."""

    traversal_manifest = json.loads(
        Path(traversal_manifest_path).read_text(encoding="utf-8")
    )
    return _checked_region_v4_suite_contract(traversal_manifest)


def region_v5_suite_contract(
    *,
    traversal_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return current Region v5 while its full source contract matches."""

    traversal_manifest = json.loads(
        Path(traversal_manifest_path).read_text(encoding="utf-8")
    )
    return _checked_region_v5_suite_contract(traversal_manifest)


def region_v6_suite_contract(
    *,
    traversal_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return current Region v6 with target navigation bound."""

    traversal_manifest = json.loads(
        Path(traversal_manifest_path).read_text(encoding="utf-8")
    )
    return _checked_region_v6_suite_contract(traversal_manifest)


def _checked_region_v1_suite_contract(
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    suite = _region_v1_suite_contract(
        traversal_manifest=traversal_manifest,
    )
    actual = canonical_sha256(suite)
    if actual != ARSENAL_REGION_V1_SUITE_SHA256:
        raise RuntimeError(
            f"{ARSENAL_REGION_V1} drifted: expected "
            f"{ARSENAL_REGION_V1_SUITE_SHA256}, current {actual}; "
            "publish a new suite version"
        )
    return suite


def _checked_region_v2_suite_contract(
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    suite = _region_v2_suite_contract(
        traversal_manifest=traversal_manifest,
    )
    actual = canonical_sha256(suite)
    if actual != ARSENAL_REGION_V2_SUITE_SHA256:
        raise RuntimeError(
            f"{ARSENAL_REGION_V2} drifted: expected "
            f"{ARSENAL_REGION_V2_SUITE_SHA256}, current {actual}; "
            "publish a new suite version"
        )
    return suite


def _checked_region_v3_suite_contract(
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    suite = _region_v3_suite_contract(
        traversal_manifest=traversal_manifest,
    )
    actual = canonical_sha256(suite)
    if actual != ARSENAL_REGION_V3_SUITE_SHA256:
        raise RuntimeError(
            f"{ARSENAL_REGION_V3} drifted: expected "
            f"{ARSENAL_REGION_V3_SUITE_SHA256}, current {actual}; "
            "publish a new suite version"
        )
    return suite


def _checked_region_v4_suite_contract(
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    suite = _region_v4_suite_contract(
        traversal_manifest=traversal_manifest,
    )
    actual = canonical_sha256(suite)
    if actual != ARSENAL_REGION_V4_SUITE_SHA256:
        raise RuntimeError(
            f"{ARSENAL_REGION_V4} drifted: expected "
            f"{ARSENAL_REGION_V4_SUITE_SHA256}, current {actual}; "
            "publish a new suite version"
        )
    return suite


def _checked_region_v5_suite_contract(
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    suite = _region_v5_suite_contract(
        traversal_manifest=traversal_manifest,
    )
    actual = canonical_sha256(suite)
    if actual != ARSENAL_REGION_V5_SUITE_SHA256:
        raise RuntimeError(
            f"{ARSENAL_REGION_V5} drifted: expected "
            f"{ARSENAL_REGION_V5_SUITE_SHA256}, current {actual}; "
            "publish a new suite version"
        )
    return suite


def _checked_region_v6_suite_contract(
    traversal_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    suite = _region_v6_suite_contract(
        traversal_manifest=traversal_manifest,
    )
    actual = canonical_sha256(suite)
    if actual != ARSENAL_REGION_V6_SUITE_SHA256:
        raise RuntimeError(
            f"{ARSENAL_REGION_V6} drifted: expected "
            f"{ARSENAL_REGION_V6_SUITE_SHA256}, current {actual}; "
            "publish a new suite version"
        )
    return suite


def _manifest_sha(
    manifest: Mapping[str, Any],
    name: str,
) -> str:
    value = manifest.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"manifest field {name} is not a SHA-256")
    return value.upper()
