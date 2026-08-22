"""Content-bound league sidecar around population checkpoint v1.

The population checkpoint remains the sole tensor/optimizer format.  This
sidecar adds only host state that v1 intentionally does not own: ratings,
roles, snapshot lineage, deterministic scheduler position, and the exact
assignment that was active at the saved fresh-arena boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from arena.training.selfplay.league import SelfPlayState
from arena.training.selfplay.production import (
    LeagueBookkeeping,
    LeaguePolicyOwner,
    LeaguePolicyRole,
    LeagueSnapshot,
    ProductionLeagueConfig,
    ProductionLeagueLoop,
    ProductionLeagueState,
    canonical_json,
    config_manifest,
    league_update_plan_for_opponent,
    league_policy_owner_manifest,
)


PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA = (
    "hytalerl_production_selfplay_league_checkpoint_v1"
)
PRODUCTION_LEAGUE_CHECKPOINT_VERSION = 1

_LEAGUE_MANIFEST = "league.json"
_LEAGUE_MANIFEST_SHA = "league.sha256"
_POPULATION_DIRECTORY = "population"


def production_league_checkpoint_contract_manifest() -> dict[str, Any]:
    """Describe the stable sidecar and its population-v1 ownership boundary."""

    return {
        "schema": PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA,
        "version": PRODUCTION_LEAGUE_CHECKPOINT_VERSION,
        "files": {
            "league_manifest": _LEAGUE_MANIFEST,
            "league_manifest_sha256": _LEAGUE_MANIFEST_SHA,
            "population_checkpoint_v1": _POPULATION_DIRECTORY,
        },
        "state": {
            "ratings": "float32[K]",
            "games": "int32[K]",
            "active": "bool[K] derived from policy roles",
            "roles": "trainable/frozen/anchor/reserve on fixed K",
            "scheduler_update_index": "next deterministic host update",
            "snapshots": "append-only source/target/context lineage",
            "current_plan": "last assignment saved by population v1",
        },
        "population_boundary": (
            "policy bank, optimizer, counters, assignment, Combat contracts, "
            "and selected upload export remain population checkpoint v1"
        ),
        "resume_boundary": (
            "fresh arena; deterministic schedule/train/reset keys derive from "
            "the persisted seed and scheduler update index"
        ),
    }


def production_league_checkpoint_contract_sha256() -> str:
    return hashlib.sha256(
        canonical_json(production_league_checkpoint_contract_manifest())
    ).hexdigest().upper()


def save_production_league_checkpoint(
    path: str | Path,
    loop: ProductionLeagueLoop,
    state: ProductionLeagueState,
    *,
    metadata: Mapping[str, Any] | None = None,
    arsenal_runtime_capacity: Any = None,
    world_geometry_config: Any = None,
) -> Path:
    """Save a new league artifact without replacing an existing generation."""

    destination = Path(path)
    _prepare_empty_directory(destination)
    entrypoint = loop.entrypoint_for(state.current_plan)
    population_path = entrypoint.save_checkpoint(
        destination / _POPULATION_DIRECTORY,
        state.population,
        selected_policy_id=loop.config.learner_policy_id,
        metadata=metadata,
        arsenal_runtime_capacity=arsenal_runtime_capacity,
        world_geometry_config=world_geometry_config,
    )
    population_manifest_sha256 = (
        population_path / "manifest.sha256"
    ).read_text(encoding="ascii").strip()
    manifest = _manifest(
        loop.config,
        state,
        population_manifest_sha256=population_manifest_sha256,
    )
    encoded = canonical_json(manifest) + b"\n"
    (destination / _LEAGUE_MANIFEST).write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest().upper()
    (destination / _LEAGUE_MANIFEST_SHA).write_text(
        digest + "\n",
        encoding="ascii",
        newline="\n",
    )
    return destination


def load_production_league_checkpoint(
    path: str | Path,
    loop: ProductionLeagueLoop,
    *,
    arsenal_runtime_capacity: Any = None,
    world_geometry_config: Any = None,
) -> ProductionLeagueState:
    """Verify the sidecar, rebuild its assignment, then delegate tensors to v1."""

    source = Path(path)
    manifest_path = source / _LEAGUE_MANIFEST
    encoded = manifest_path.read_bytes()
    expected_sha256 = (source / _LEAGUE_MANIFEST_SHA).read_text(
        encoding="ascii"
    ).strip()
    if hashlib.sha256(encoded).hexdigest().upper() != expected_sha256:
        raise ValueError("league manifest SHA-256 mismatch")
    manifest = json.loads(encoded)
    if encoded != canonical_json(manifest) + b"\n":
        raise ValueError("league manifest is not canonical JSON")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "version",
            "checkpoint_contract_sha256",
            "config",
            "state",
            "current_plan",
            "population",
            "provenance",
        },
        "league manifest",
    )
    if manifest["schema"] != PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported league checkpoint schema")
    if manifest["version"] != PRODUCTION_LEAGUE_CHECKPOINT_VERSION:
        raise ValueError("unsupported league checkpoint version")
    if (
        manifest["checkpoint_contract_sha256"]
        != production_league_checkpoint_contract_sha256()
    ):
        raise ValueError("league checkpoint contract identity mismatch")
    if manifest["config"] != config_manifest(loop.config):
        raise ValueError("league checkpoint config mismatch")

    bookkeeping = _decode_bookkeeping(manifest["state"], loop.config)
    plan_record = manifest["current_plan"]
    _require_exact_keys(
        plan_record,
        {
            "scheduler_update_index",
            "opponent_policy_id",
            "assignment_content_sha256",
        },
        "league current plan",
    )
    plan_index = _integer(
        plan_record["scheduler_update_index"],
        name="current plan scheduler update index",
        minimum=0,
    )
    expected_plan_index = max(bookkeeping.scheduler_update_index - 1, 0)
    if plan_index != expected_plan_index:
        raise ValueError("league current plan is not the last update boundary")
    plan = league_update_plan_for_opponent(
        bookkeeping,
        loop.config,
        opponent_policy_id=_integer(
            plan_record["opponent_policy_id"],
            name="current opponent policy ID",
            minimum=0,
        ),
        scheduler_update_index=plan_index,
    )
    if plan.assignment_content_sha256 != plan_record["assignment_content_sha256"]:
        raise ValueError("league current assignment identity mismatch")

    population_record = manifest["population"]
    _require_exact_keys(
        population_record,
        {"directory", "manifest_sha256", "update_count", "actor_steps"},
        "league population",
    )
    if population_record["directory"] != _POPULATION_DIRECTORY:
        raise ValueError("league population directory is unsupported")
    population_path = source / _POPULATION_DIRECTORY
    actual_population_manifest_sha = (
        population_path / "manifest.sha256"
    ).read_text(encoding="ascii").strip()
    if actual_population_manifest_sha != population_record["manifest_sha256"]:
        raise ValueError("league population manifest identity mismatch")
    entrypoint = loop.entrypoint_for(plan)
    loaded = entrypoint.load_checkpoint(
        population_path,
        _resume_reset_key(loop.config, bookkeeping.scheduler_update_index),
        arsenal_runtime_capacity=arsenal_runtime_capacity,
        world_geometry_config=world_geometry_config,
    )
    update_count = np.asarray(jax.device_get(loaded.state.update_count))
    actor_steps = np.asarray(
        jax.device_get(loaded.state.total_trainable_actor_steps)
    )
    if update_count.tolist() != population_record["update_count"]:
        raise ValueError("league population update counters differ")
    if actor_steps.tolist() != population_record["actor_steps"]:
        raise ValueError("league population actor-step counters differ")

    provenance = manifest["provenance"]
    _require_exact_keys(
        provenance,
        {
            "runtime_config_content_sha256",
            "learner_context",
            "learner_context_attestation",
            "opponent_owner",
        },
        "league provenance",
    )
    return loop.restore_loaded_state(
        bookkeeping=bookkeeping,
        population=loaded.state,
        current_plan=plan,
        learner_context=provenance["learner_context"],
        learner_context_attestation=(
            provenance["learner_context_attestation"]
        ),
        opponent_owner=_owner(provenance["opponent_owner"]),
        runtime_config_content_sha256=(
            provenance["runtime_config_content_sha256"]
        ),
    )


def _manifest(
    config: ProductionLeagueConfig,
    state: ProductionLeagueState,
    *,
    population_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA,
        "version": PRODUCTION_LEAGUE_CHECKPOINT_VERSION,
        "checkpoint_contract_sha256": (
            production_league_checkpoint_contract_sha256()
        ),
        "config": config_manifest(config),
        "state": {
            "ratings": _array_list(state.bookkeeping.ratings.ratings, float),
            "games": _array_list(state.bookkeeping.ratings.games, int),
            "active": _array_list(state.bookkeeping.ratings.active, bool),
            "roles": [role.value for role in state.bookkeeping.roles],
            "scheduler_update_index": (
                state.bookkeeping.scheduler_update_index
            ),
            "snapshots": [
                {
                    "generation": value.generation,
                    "source_policy_id": value.source_policy_id,
                    "target_policy_id": value.target_policy_id,
                    "scheduler_update_index": value.scheduler_update_index,
                    "learner_update_count": value.learner_update_count,
                    "source_context_content_sha256": (
                        value.source_context_content_sha256
                    ),
                    "source_runtime_config_content_sha256": (
                        value.source_runtime_config_content_sha256
                    ),
                    "target_context_content_sha256": (
                        value.target_context_content_sha256
                    ),
                    "target_owner": league_policy_owner_manifest(
                        value.target_owner
                    ),
                    "source_profile": value.source_profile,
                    "target_profile": value.target_profile,
                    "source_team_id": value.source_team_id,
                    "target_team_id": value.target_team_id,
                    "source_controller": value.source_controller,
                    "target_controller": value.target_controller,
                    "transfer_reason": value.transfer_reason,
                }
                for value in state.bookkeeping.snapshots
            ],
        },
        "current_plan": {
            "scheduler_update_index": (
                state.current_plan.scheduler_update_index
            ),
            "opponent_policy_id": state.current_plan.opponent_policy_id,
            "assignment_content_sha256": (
                state.current_plan.assignment_content_sha256
            ),
        },
        "population": {
            "directory": _POPULATION_DIRECTORY,
            "manifest_sha256": population_manifest_sha256,
            "update_count": _array_list(state.population.update_count, int),
            "actor_steps": _array_list(
                state.population.total_trainable_actor_steps,
                int,
            ),
        },
        "provenance": {
            "runtime_config_content_sha256": (
                state.runtime_config_content_sha256
            ),
            "learner_context": state.learner_context,
            "learner_context_attestation": (
                state.learner_context_attestation
            ),
            "opponent_owner": league_policy_owner_manifest(
                state.opponent_owner
            ),
        },
    }


def _decode_bookkeeping(
    value: Any,
    config: ProductionLeagueConfig,
) -> LeagueBookkeeping:
    if not isinstance(value, dict):
        raise TypeError("league state must be an object")
    _require_exact_keys(
        value,
        {
            "ratings",
            "games",
            "active",
            "roles",
            "scheduler_update_index",
            "snapshots",
        },
        "league state",
    )
    count = len(config.policy_roles)
    ratings = _numeric_vector(value["ratings"], count, np.float32, "ratings")
    if not np.all(np.isfinite(ratings)):
        raise ValueError("league ratings must be finite")
    games = _numeric_vector(value["games"], count, np.int32, "games")
    if np.any(games < 0):
        raise ValueError("league games cannot be negative")
    active = _boolean_vector(value["active"], count, "active")
    if not isinstance(value["roles"], list) or len(value["roles"]) != count:
        raise ValueError("league roles must cover policy-bank K")
    roles = tuple(LeaguePolicyRole(role) for role in value["roles"])
    snapshots_value = value["snapshots"]
    if not isinstance(snapshots_value, list):
        raise TypeError("league snapshots must be a list")
    snapshots = tuple(_decode_snapshot(row) for row in snapshots_value)
    bookkeeping = LeagueBookkeeping(
        ratings=SelfPlayState(
            ratings=jnp.asarray(ratings, dtype=jnp.float32),
            games=jnp.asarray(games, dtype=jnp.int32),
            active=jnp.asarray(active, dtype=jnp.bool_),
        ),
        roles=roles,
        scheduler_update_index=_integer(
            value["scheduler_update_index"],
            name="scheduler update index",
            minimum=0,
        ),
        snapshots=snapshots,
    )
    # The loop's restore path performs the complete role/active/lineage audit.
    return bookkeeping


def _decode_snapshot(value: Any) -> LeagueSnapshot:
    if not isinstance(value, dict):
        raise TypeError("league snapshot must be an object")
    fields = {
        "generation",
        "source_policy_id",
        "target_policy_id",
        "scheduler_update_index",
        "learner_update_count",
        "source_context_content_sha256",
        "source_runtime_config_content_sha256",
        "target_context_content_sha256",
        "target_owner",
        "source_profile",
        "target_profile",
        "source_team_id",
        "target_team_id",
        "source_controller",
        "target_controller",
        "transfer_reason",
    }
    _require_exact_keys(value, fields, "league snapshot")
    return LeagueSnapshot(
        generation=_integer(value["generation"], name="generation", minimum=1),
        source_policy_id=_integer(
            value["source_policy_id"], name="source policy ID", minimum=0
        ),
        target_policy_id=_integer(
            value["target_policy_id"], name="target policy ID", minimum=0
        ),
        scheduler_update_index=_integer(
            value["scheduler_update_index"],
            name="snapshot scheduler update index",
            minimum=1,
        ),
        learner_update_count=_integer(
            value["learner_update_count"],
            name="snapshot learner update count",
            minimum=1,
        ),
        source_context_content_sha256=_sha256(
            value["source_context_content_sha256"],
            name="snapshot context identity",
        ),
        source_runtime_config_content_sha256=_sha256(
            value["source_runtime_config_content_sha256"],
            name="snapshot runtime identity",
        ),
        target_context_content_sha256=_sha256(
            value["target_context_content_sha256"],
            name="snapshot target context identity",
        ),
        target_owner=_owner(value["target_owner"]),
        source_profile=_label(value["source_profile"], name="source profile"),
        target_profile=_label(value["target_profile"], name="target profile"),
        source_team_id=_integer(
            value["source_team_id"], name="source team ID", minimum=-2147483648
        ),
        target_team_id=_integer(
            value["target_team_id"], name="target team ID", minimum=-2147483648
        ),
        source_controller=_label(
            value["source_controller"], name="source controller"
        ),
        target_controller=_label(
            value["target_controller"], name="target controller"
        ),
        transfer_reason=_optional_reason(value["transfer_reason"]),
    )


def _resume_reset_key(config: ProductionLeagueConfig, update: int) -> jax.Array:
    # Deliberately separate from production.py's private domain constants.  A
    # load always starts a fresh arena, so only deterministic repeatability is
    # required here; the next sampled/train keys remain unchanged.
    key = jax.random.key(np.uint32(config.seed))
    key = jax.random.fold_in(key, np.uint32(0x4C4F4144))
    return jax.random.fold_in(key, np.uint32(update))


def _prepare_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError("league checkpoint path must be a directory")
        if any(path.iterdir()):
            raise FileExistsError(
                "league checkpoint directory must be absent or empty"
            )
    else:
        path.mkdir(parents=True)


def _array_list(value: Any, scalar: type) -> list[Any]:
    array = np.asarray(jax.device_get(value))
    if array.ndim != 1:
        raise ValueError("league checkpoint arrays must be vectors")
    return [scalar(item) for item in array.tolist()]


def _numeric_vector(
    value: Any,
    count: int,
    dtype: Any,
    name: str,
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"league {name} must have length K")
    if any(isinstance(item, bool) for item in value):
        raise TypeError(f"league {name} must be numeric")
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"league {name} has invalid numeric values") from error
    if result.tolist() != value:
        raise ValueError(f"league {name} is not canonical {np.dtype(dtype).name}")
    return result


def _boolean_vector(value: Any, count: int, name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, bool) for item in value)
    ):
        raise TypeError(f"league {name} must be bool[K]")
    return np.asarray(value, dtype=np.bool_)


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _label(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{name} must be nonempty and trimmed")
    return value


def _optional_reason(value: Any) -> str | None:
    if value is None:
        return None
    return _label(value, name="snapshot transfer reason")


def _owner(value: Any) -> LeaguePolicyOwner:
    if not isinstance(value, dict) or set(value) != {
        "entity_index",
        "profile",
        "team_id",
        "controller",
        "trainable",
    }:
        raise ValueError("snapshot target owner fields differ")
    return LeaguePolicyOwner(
        entity_index=_integer(
            value["entity_index"], name="owner entity index", minimum=0
        ),
        profile=_label(value["profile"], name="owner profile"),
        team_id=_integer(
            value["team_id"], name="owner team ID", minimum=-2147483648
        ),
        controller=_label(
            value["controller"], name="owner controller"
        ),
        trainable=value["trainable"]
        if isinstance(value["trainable"], bool)
        else (_raise_owner_trainable()),
    )


def _raise_owner_trainable():
    raise TypeError("snapshot target owner trainable must be boolean")


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.upper()
        or any(character not in "0123456789ABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be an uppercase SHA-256")
    return value


def _require_exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} fields differ from the current contract")


__all__ = [
    "PRODUCTION_LEAGUE_CHECKPOINT_SCHEMA",
    "PRODUCTION_LEAGUE_CHECKPOINT_VERSION",
    "load_production_league_checkpoint",
    "production_league_checkpoint_contract_manifest",
    "production_league_checkpoint_contract_sha256",
    "save_production_league_checkpoint",
]
