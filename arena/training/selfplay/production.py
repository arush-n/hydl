"""Host-side production loop for rating-aware frozen-opponent self-play.

The JIT owns one PPO update.  This module owns everything that must remain
outside it: immutable historical roles, matchmaking, update-boundary
assignment changes, ratings, and snapshots.  League mode deliberately has one
live learner; independently live policies remain the separate duel mode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from hytalegym.jax.training.multi_actor import (
    MultiActorPopulationEpisodeOutcomes,
    MultiActorPopulationTrainingState,
    PolicyActorAssignment,
    policy_training_context_content_sha256,
    validate_population_policy_bank,
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


class LeaguePolicyRole(str, Enum):
    """Update ownership for one fixed-capacity policy-bank row."""

    TRAINABLE = "trainable"
    FROZEN = "frozen"
    ANCHOR = "anchor"
    RESERVE = "reserve"


@dataclass(frozen=True, slots=True)
class LeaguePolicyOwner:
    """Immutable runtime-derived owner row for one physical policy seat."""

    entity_index: int
    profile: str
    team_id: int
    controller: str
    trainable: bool


@dataclass(frozen=True, slots=True)
class ProductionLeagueConfig:
    """Static league topology and deterministic host scheduling contract."""

    policy_roles: tuple[LeaguePolicyRole | str, ...]
    actor_indices: tuple[int, int]
    entity_count: int
    num_envs: int
    seed: int
    rating: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    policy_slot_order: tuple[str, str] = ("learner", "opponent")
    snapshot_interval_updates: int = 0
    include_anchors_in_training: bool = False
    snapshot_transfer_reason: str | None = None

    def __post_init__(self) -> None:
        roles = tuple(LeaguePolicyRole(role) for role in self.policy_roles)
        object.__setattr__(self, "policy_roles", roles)
        if len(roles) < 2:
            raise ValueError("a production league needs at least two policy rows")
        if roles.count(LeaguePolicyRole.TRAINABLE) != 1:
            raise ValueError(
                "frozen-opponent league mode needs exactly one trainable row; "
                "use duel mode for multiple live optimizers"
            )
        if not any(role is LeaguePolicyRole.FROZEN for role in roles):
            raise ValueError("a production league needs an initial frozen opponent")
        if (
            not isinstance(self.actor_indices, tuple)
            or len(self.actor_indices) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.actor_indices
            )
        ):
            raise TypeError("actor_indices must be two integer entity IDs")
        if min(self.actor_indices) < 0 or self.actor_indices[0] == self.actor_indices[1]:
            raise ValueError("rated learner and opponent need distinct entity IDs")
        if isinstance(self.entity_count, bool) or not isinstance(self.entity_count, int):
            raise TypeError("entity_count must be an integer")
        if self.entity_count <= max(self.actor_indices):
            raise ValueError("a rated actor index is outside the entity axis")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise TypeError("num_envs must be an integer")
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must fit uint32")
        if tuple(self.policy_slot_order) not in {
            ("learner", "opponent"),
            ("opponent", "learner"),
        }:
            raise ValueError(
                "policy_slot_order must contain learner and opponent exactly once"
            )
        if (
            isinstance(self.snapshot_interval_updates, bool)
            or not isinstance(self.snapshot_interval_updates, int)
        ):
            raise TypeError("snapshot_interval_updates must be an integer")
        if self.snapshot_interval_updates < 0:
            raise ValueError("snapshot_interval_updates cannot be negative")
        if not isinstance(self.include_anchors_in_training, bool):
            raise TypeError("include_anchors_in_training must be boolean")
        if self.snapshot_transfer_reason is not None and (
            not isinstance(self.snapshot_transfer_reason, str)
            or not self.snapshot_transfer_reason.strip()
            or self.snapshot_transfer_reason != self.snapshot_transfer_reason.strip()
        ):
            raise ValueError(
                "snapshot_transfer_reason must be absent or nonempty and trimmed"
            )

    @property
    def learner_policy_id(self) -> int:
        return self.policy_roles.index(LeaguePolicyRole.TRAINABLE)


@dataclass(frozen=True, slots=True)
class LeagueSnapshot:
    """One immutable source-to-reserve copy at a successful update boundary."""

    generation: int
    source_policy_id: int
    target_policy_id: int
    scheduler_update_index: int
    learner_update_count: int
    source_context_content_sha256: str
    source_runtime_config_content_sha256: str
    target_context_content_sha256: str
    target_owner: LeaguePolicyOwner
    source_profile: str
    target_profile: str
    source_team_id: int
    target_team_id: int
    source_controller: str
    target_controller: str
    transfer_reason: str | None


@dataclass(frozen=True, slots=True)
class LeagueBookkeeping:
    """Small resumable host state; tensor/optimizer state stays in checkpoint v1."""

    ratings: SelfPlayState
    roles: tuple[LeaguePolicyRole, ...]
    scheduler_update_index: int = 0
    snapshots: tuple[LeagueSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class LeagueUpdatePlan:
    """One immutable assignment sampled before entering a compiled update."""

    scheduler_update_index: int
    opponent_policy_id: int
    matchups: Matchups
    assignment: PolicyActorAssignment
    assignment_content_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionLeagueState:
    """League bookkeeping plus the existing population trainer state."""

    bookkeeping: LeagueBookkeeping
    population: MultiActorPopulationTrainingState
    current_plan: LeagueUpdatePlan
    learner_context: dict[str, Any]
    learner_context_attestation: dict[str, Any]
    opponent_owner: LeaguePolicyOwner
    runtime_config_content_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionLeagueMetrics:
    """One update's optimizer, outcome, rating, and snapshot facts."""

    scheduler_update_index: int
    opponent_policy_id: int
    update_applied: bool
    rated_episodes: int
    unrated_other_episodes: int
    rating: SelfPlayMetrics | None
    snapshot: LeagueSnapshot | None
    population: Any


EntryPointFactory = Callable[[PolicyActorAssignment], Any]

_KEY_DOMAIN_SCHEDULE = 0x53434844
_KEY_DOMAIN_TRAIN = 0x54524149
_KEY_DOMAIN_RESET = 0x52455345


def initialize_league_bookkeeping(
    config: ProductionLeagueConfig,
    *,
    initial_rating: float = 1200.0,
) -> LeagueBookkeeping:
    """Create ratings whose active mask is derived solely from immutable roles."""

    active_ids = tuple(
        index
        for index, role in enumerate(config.policy_roles)
        if role is not LeaguePolicyRole.RESERVE
    )
    ratings = initialize_self_play(
        len(config.policy_roles),
        active_ids=active_ids,
        initial_rating=initial_rating,
    )
    return LeagueBookkeeping(ratings=ratings, roles=config.policy_roles)


def schedule_league_update(
    bookkeeping: LeagueBookkeeping,
    config: ProductionLeagueConfig,
) -> LeagueUpdatePlan:
    """Draw one opponent and broadcast it across the static PPO arena batch.

    A single opponent per update bounds compiled assignment templates by the
    fixed bank capacity instead of the combinatorial number of per-arena
    schedules.  Match diversity still changes at every host update boundary.
    """

    _validate_bookkeeping(bookkeeping, config)
    learner = config.learner_policy_id
    eligible_roles = {LeaguePolicyRole.FROZEN}
    if config.include_anchors_in_training:
        eligible_roles.add(LeaguePolicyRole.ANCHOR)
    eligible = np.asarray(
        [role in eligible_roles for role in bookkeeping.roles],
        dtype=np.bool_,
    )
    if not np.any(eligible):
        raise ValueError("league has no eligible immutable opponent")
    active = jnp.asarray(eligible, dtype=jnp.bool_).at[learner].set(True)
    sampling_state = bookkeeping.ratings._replace(active=active)
    sampled = sample_opponents(
        sampling_state,
        learner,
        _league_key(
            config.seed,
            bookkeeping.scheduler_update_index,
            _KEY_DOMAIN_SCHEDULE,
        ),
        batch_size=1,
        config=config.rating,
    )
    if not bool(np.asarray(jax.device_get(sampled.valid))[0]):
        raise RuntimeError("rating-aware opponent sampler returned no match")
    opponent = int(np.asarray(jax.device_get(sampled.opponent_id))[0])
    return league_update_plan_for_opponent(
        bookkeeping,
        config,
        opponent_policy_id=opponent,
    )


def league_update_plan_for_opponent(
    bookkeeping: LeagueBookkeeping,
    config: ProductionLeagueConfig,
    *,
    opponent_policy_id: int,
    scheduler_update_index: int | None = None,
) -> LeagueUpdatePlan:
    """Rebuild a recorded static plan without re-running matchmaking."""

    _validate_bookkeeping(bookkeeping, config)
    if isinstance(opponent_policy_id, bool) or not isinstance(
        opponent_policy_id, int
    ):
        raise TypeError("opponent_policy_id must be an integer")
    if not 0 <= opponent_policy_id < len(bookkeeping.roles):
        raise ValueError("opponent policy ID is outside the policy bank")
    eligible = {LeaguePolicyRole.FROZEN}
    if config.include_anchors_in_training:
        eligible.add(LeaguePolicyRole.ANCHOR)
    if bookkeeping.roles[opponent_policy_id] not in eligible:
        raise ValueError("recorded opponent role is not eligible for training")
    if scheduler_update_index is None:
        scheduler_update_index = bookkeeping.scheduler_update_index
    if isinstance(scheduler_update_index, bool) or not isinstance(
        scheduler_update_index, int
    ):
        raise TypeError("scheduler_update_index must be an integer")
    if scheduler_update_index < 0:
        raise ValueError("scheduler_update_index cannot be negative")
    matchups = Matchups(
        learner_id=jnp.full(
            (config.num_envs,), config.learner_policy_id, dtype=jnp.int32
        ),
        opponent_id=jnp.full(
            (config.num_envs,), opponent_policy_id, dtype=jnp.int32
        ),
        valid=jnp.ones((config.num_envs,), dtype=jnp.bool_),
    )
    assignment = frozen_opponent_assignment(
        matchups,
        actor_indices=config.actor_indices,
        policy_slot_order=config.policy_slot_order,
        entity_count=config.entity_count,
    )
    return LeagueUpdatePlan(
        scheduler_update_index=scheduler_update_index,
        opponent_policy_id=opponent_policy_id,
        matchups=matchups,
        assignment=assignment,
        assignment_content_sha256=assignment_content_sha256(assignment),
    )


def apply_league_outcomes(
    bookkeeping: LeagueBookkeeping,
    plan: LeagueUpdatePlan,
    outcomes: MultiActorPopulationEpisodeOutcomes,
    config: ProductionLeagueConfig,
) -> tuple[LeagueBookkeeping, SelfPlayMetrics | None, int, int]:
    """Rate exact completed learner episodes and skip classified ``other`` rows."""

    if plan.scheduler_update_index != bookkeeping.scheduler_update_index:
        raise ValueError("league outcome plan is not the current update")
    arrays = {
        name: np.asarray(jax.device_get(getattr(outcomes, name)), dtype=np.bool_)
        for name in outcomes._fields
    }
    expected = (1, None, config.num_envs)
    for name, value in arrays.items():
        if value.ndim != 3 or value.shape[0] != expected[0] or value.shape[2] != expected[2]:
            raise ValueError(
                f"population outcome {name} must have shape [1,T,num_envs]"
            )
    done = arrays["done"][0]
    classified = (
        arrays["success"].astype(np.int8)
        + arrays["death"].astype(np.int8)
        + arrays["simultaneous"].astype(np.int8)
        + arrays["other"].astype(np.int8)
    )[0]
    if np.any(classified[done] != 1) or np.any(classified[~done] != 0):
        raise ValueError("completed league episodes are not disjoint and exhaustive")

    ratable = done & ~arrays["other"][0]
    scores = np.where(
        arrays["success"][0][ratable],
        np.float32(1.0),
        np.where(
            arrays["simultaneous"][0][ratable],
            np.float32(0.5),
            np.float32(0.0),
        ),
    )
    rated = int(scores.size)
    other = int(np.count_nonzero(done & arrays["other"][0]))
    next_ratings = bookkeeping.ratings
    rating_metrics = None
    if rated:
        event_matchups = Matchups(
            learner_id=jnp.full(
                (rated,), config.learner_policy_id, dtype=jnp.int32
            ),
            opponent_id=jnp.full(
                (rated,), plan.opponent_policy_id, dtype=jnp.int32
            ),
            valid=jnp.ones((rated,), dtype=jnp.bool_),
        )
        next_ratings, rating_metrics = update_ratings(
            bookkeeping.ratings,
            event_matchups,
            jnp.asarray(scores, dtype=jnp.float32),
            config.rating,
        )
        if int(rating_metrics.invalid_results) != 0:
            raise RuntimeError("validated league outcomes produced invalid Elo rows")
    return (
        LeagueBookkeeping(
            ratings=next_ratings,
            roles=bookkeeping.roles,
            scheduler_update_index=bookkeeping.scheduler_update_index + 1,
            snapshots=bookkeeping.snapshots,
        ),
        rating_metrics,
        rated,
        other,
    )


def assignment_content_sha256(assignment: PolicyActorAssignment) -> str:
    """Hash one static update assignment without relying on device layout."""

    digest = hashlib.sha256(b"HYTALERL_LEAGUE_UPDATE_ASSIGNMENT\0")
    for name in assignment._fields:
        value = np.asarray(jax.device_get(getattr(assignment, name)))
        canonical = np.ascontiguousarray(
            value.astype(
                value.dtype
                if value.dtype.byteorder == "|"
                else value.dtype.newbyteorder("<"),
                copy=False,
            )
        )
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        dtype = canonical.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(canonical.ndim.to_bytes(4, "big"))
        for dimension in canonical.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        raw = canonical.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest().upper()


class ProductionLeagueLoop:
    """Compose matchmaking with existing population PPO and checkpoint v1."""

    def __init__(
        self,
        config: ProductionLeagueConfig,
        entrypoint_factory: EntryPointFactory,
    ) -> None:
        if not callable(entrypoint_factory):
            raise TypeError("entrypoint_factory must be callable")
        self.config = config
        self._factory = entrypoint_factory
        self._entrypoints: dict[str, Any] = {}

    def initialize(
        self,
        policy_bank: Any,
        *,
        initial_rating: float = 1200.0,
    ) -> ProductionLeagueState:
        """Initialize one deterministic schedule and the existing PPO state."""

        bookkeeping = initialize_league_bookkeeping(
            self.config,
            initial_rating=initial_rating,
        )
        plan = schedule_league_update(bookkeeping, self.config)
        entrypoint = self._entrypoint(plan)
        if validate_population_policy_bank(policy_bank, entrypoint.ppo_config) != len(
            self.config.policy_roles
        ):
            raise ValueError("policy-bank K axis differs from league roles")
        population = entrypoint.initialize(
            policy_bank,
            _league_key(self.config.seed, 0, _KEY_DOMAIN_RESET),
        )
        context, attestation = _require_upload_provenance(
            entrypoint,
            self.config.learner_policy_id,
            self.config.actor_indices[0],
        )
        opponent_owner = _require_opponent_provenance(
            entrypoint,
            self.config,
            snapshots=(),
        )
        return ProductionLeagueState(
            bookkeeping=bookkeeping,
            population=population,
            current_plan=plan,
            learner_context=context,
            learner_context_attestation=attestation,
            opponent_owner=opponent_owner,
            runtime_config_content_sha256=(
                entrypoint.runtime_config_content_sha256
            ),
        )

    def train_update(
        self,
        state: ProductionLeagueState,
    ) -> tuple[ProductionLeagueState, ProductionLeagueMetrics]:
        """Sample, bind, train, rate, and optionally snapshot one update."""

        _validate_bookkeeping(
            state.bookkeeping,
            self.config,
            learner_context=state.learner_context,
            runtime_config_content_sha256=(
                state.runtime_config_content_sha256
            ),
        )
        plan = schedule_league_update(state.bookkeeping, self.config)
        entrypoint = self._entrypoint(plan)
        context, attestation = _require_upload_provenance(
            entrypoint,
            self.config.learner_policy_id,
            self.config.actor_indices[0],
        )
        opponent_owner = _require_opponent_provenance(
            entrypoint,
            self.config,
            snapshots=state.bookkeeping.snapshots,
        )
        if context != state.learner_context or attestation != state.learner_context_attestation:
            raise ValueError("league assignment changed learner upload provenance")
        if (
            entrypoint.runtime_config_content_sha256
            != state.runtime_config_content_sha256
        ):
            raise ValueError("league assignment changed runtime config identity")
        if state.opponent_owner != opponent_owner:
            raise ValueError("league assignment changed opponent owner provenance")

        population = state.population
        if (
            plan.assignment_content_sha256
            != state.current_plan.assignment_content_sha256
        ):
            population = _rebind_population_state(
                population,
                entrypoint,
                _league_key(
                    self.config.seed,
                    state.bookkeeping.scheduler_update_index,
                    _KEY_DOMAIN_RESET,
                ),
                learner_policy_id=self.config.learner_policy_id,
            )

        _preflight_snapshot_capacity(population, state.bookkeeping, self.config)
        population, population_metrics = entrypoint.train_step(
            population,
            _league_key(
                self.config.seed,
                state.bookkeeping.scheduler_update_index,
                _KEY_DOMAIN_TRAIN,
            ),
        )
        next_bookkeeping, rating_metrics, rated, other = apply_league_outcomes(
            state.bookkeeping,
            plan,
            population_metrics.episode_outcomes,
            self.config,
        )
        snapshot = None
        if _snapshot_due(population, next_bookkeeping, self.config):
            population, next_bookkeeping, snapshot = _publish_snapshot(
                population,
                next_bookkeeping,
                self.config,
                learner_context=context,
                learner_context_attestation=attestation,
                runtime_config_content_sha256=(
                    entrypoint.runtime_config_content_sha256
                ),
                target_owner=opponent_owner,
                transfer_reason=self.config.snapshot_transfer_reason,
            )
        applied = bool(
            np.asarray(jax.device_get(population_metrics.update_applied))[0]
        )
        next_state = ProductionLeagueState(
            bookkeeping=next_bookkeeping,
            population=population,
            current_plan=plan,
            learner_context=context,
            learner_context_attestation=attestation,
            opponent_owner=opponent_owner,
            runtime_config_content_sha256=(
                entrypoint.runtime_config_content_sha256
            ),
        )
        return next_state, ProductionLeagueMetrics(
            scheduler_update_index=plan.scheduler_update_index,
            opponent_policy_id=plan.opponent_policy_id,
            update_applied=applied,
            rated_episodes=rated,
            unrated_other_episodes=other,
            rating=rating_metrics,
            snapshot=snapshot,
            population=population_metrics,
        )

    def entrypoint_for(self, plan: LeagueUpdatePlan) -> Any:
        """Return the verified static entrypoint used for checkpointing."""

        return self._entrypoint(plan)

    def restore_loaded_state(
        self,
        *,
        bookkeeping: LeagueBookkeeping,
        population: MultiActorPopulationTrainingState,
        current_plan: LeagueUpdatePlan,
        learner_context: Mapping[str, Any],
        learner_context_attestation: Mapping[str, Any],
        opponent_owner: LeaguePolicyOwner,
        runtime_config_content_sha256: str,
    ) -> ProductionLeagueState:
        """Verify a decoded sidecar against the freshly loaded v1 population."""

        _validate_bookkeeping(
            bookkeeping,
            self.config,
            learner_context=learner_context,
            runtime_config_content_sha256=runtime_config_content_sha256,
        )
        entrypoint = self._entrypoint(current_plan)
        context, attestation = _require_upload_provenance(
            entrypoint,
            self.config.learner_policy_id,
            self.config.actor_indices[0],
        )
        if dict(learner_context) != context:
            raise ValueError("league checkpoint learner context differs")
        if dict(learner_context_attestation) != attestation:
            raise ValueError("league checkpoint learner attestation differs")
        current_opponent_owner = _require_opponent_provenance(
            entrypoint,
            self.config,
            snapshots=bookkeeping.snapshots,
        )
        if opponent_owner != current_opponent_owner:
            raise ValueError("league checkpoint opponent owner differs")
        if runtime_config_content_sha256 != entrypoint.runtime_config_content_sha256:
            raise ValueError("league checkpoint runtime identity differs")
        if validate_population_policy_bank(
            population.policy_bank,
            entrypoint.ppo_config,
        ) != len(self.config.policy_roles):
            raise ValueError("loaded population K axis differs from league roles")
        return ProductionLeagueState(
            bookkeeping=bookkeeping,
            population=population,
            current_plan=current_plan,
            learner_context=context,
            learner_context_attestation=attestation,
            opponent_owner=current_opponent_owner,
            runtime_config_content_sha256=runtime_config_content_sha256,
        )

    def _entrypoint(self, plan: LeagueUpdatePlan) -> Any:
        entrypoint = self._entrypoints.get(plan.assignment_content_sha256)
        if entrypoint is None:
            entrypoint = self._factory(plan.assignment)
            _require_entrypoint_contract(entrypoint, plan, self.config)
            self._entrypoints[plan.assignment_content_sha256] = entrypoint
        return entrypoint


def _require_entrypoint_contract(
    entrypoint: Any,
    plan: LeagueUpdatePlan,
    config: ProductionLeagueConfig,
) -> None:
    required = (
        "assignment",
        "entity_count",
        "ppo_config",
        "trainer",
        "runtime_config_content_sha256",
        "policy_training_contexts",
        "policy_training_context_attestations",
        "training_entity_specs",
    )
    if any(not hasattr(entrypoint, name) for name in required):
        raise TypeError("entrypoint factory did not return a population entrypoint")
    if assignment_content_sha256(entrypoint.assignment) != plan.assignment_content_sha256:
        raise ValueError("entrypoint assignment differs from sampled league plan")
    if entrypoint.entity_count != config.entity_count:
        raise ValueError("entrypoint entity_count differs from league")
    if entrypoint.ppo_config.num_envs != config.num_envs:
        raise ValueError("entrypoint PPO num_envs differs from league")
    if entrypoint.trainer.trainable_policy_ids != (config.learner_policy_id,):
        raise ValueError("league entrypoint must optimize only its learner row")
    if not entrypoint.training_entity_specs:
        raise ValueError("league entrypoint has no runtime-derived entity specs")
    _require_upload_provenance(
        entrypoint,
        config.learner_policy_id,
        config.actor_indices[0],
    )
    _require_opponent_provenance(entrypoint, config, snapshots=())


def _require_upload_provenance(
    entrypoint: Any,
    learner_policy_id: int,
    learner_entity_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contexts = {
        int(value["policy_id"]): dict(value)
        for value in entrypoint.policy_training_contexts
    }
    attestations = {
        int(value["policy_id"]): dict(value)
        for value in entrypoint.policy_training_context_attestations
    }
    if learner_policy_id not in contexts or learner_policy_id not in attestations:
        raise ValueError("league learner has no runtime-derived upload attestation")
    context = contexts[learner_policy_id]
    owners = context.get("owners")
    if not isinstance(owners, list) or not owners:
        raise ValueError("league learner provenance has no owners")
    if any(
        owner.get("entity_index") != learner_entity_index
        or owner.get("trainable") is not True
        for owner in owners
    ):
        raise ValueError("league learner provenance differs from its fixed owner")
    attestation = attestations[learner_policy_id]
    if (
        attestation.get("runtime_config_content_sha256")
        != entrypoint.runtime_config_content_sha256
    ):
        raise ValueError("league learner attests a different runtime")
    if (
        attestation.get("context_content_sha256")
        != policy_training_context_content_sha256(context)
    ):
        raise ValueError("league learner context attestation is invalid")
    return context, attestation


def _require_opponent_provenance(
    entrypoint: Any,
    config: ProductionLeagueConfig,
    *,
    snapshots: tuple[LeagueSnapshot, ...],
) -> LeaguePolicyOwner:
    """Require a verified frozen-owner context for every seatable bank row.

    The actual opponent entity is fixed even though the sampled policy row
    changes.  A source snapshot may be used there only when its profile and
    controller match that physical owner, or the league records an explicit
    transfer reason.  Opposing team IDs are allowed to differ because the
    shipped actor-first view permutes the actual targeting rules with the
    actor; the policy never receives an absolute learner-team label.
    """

    contexts = tuple(dict(value) for value in entrypoint.policy_training_contexts)
    attestations = tuple(
        dict(value) for value in entrypoint.policy_training_context_attestations
    )
    learner_context = _context_for_policy(contexts, config.learner_policy_id)
    learner_owner = _single_context_owner(
        learner_context,
        entity_index=config.actor_indices[0],
        trainable=True,
    )
    snapshot_by_policy = {
        record.target_policy_id: record
        for record in snapshots
    }
    opponent_rows = tuple(
        value
        for value in contexts
        if any(
            owner.get("entity_index") == config.actor_indices[1]
            and owner.get("trainable") is False
            for owner in value.get("owners", ())
        )
    )
    if len(opponent_rows) != 1:
        raise ValueError(
            "current league assignment needs one runtime-derived opponent context"
        )
    opponent_owner: LeaguePolicyOwner | None = None
    for context in opponent_rows:
        policy_id = int(context["policy_id"])
        owner = _single_context_owner(
            context,
            entity_index=config.actor_indices[1],
            trainable=False,
        )
        attestation = _context_for_policy(attestations, policy_id)
        if (
            attestation.get("runtime_config_content_sha256")
            != entrypoint.runtime_config_content_sha256
            or attestation.get("context_content_sha256")
            != policy_training_context_content_sha256(context)
        ):
            raise ValueError("league opponent context attestation is invalid")
        recorded = snapshot_by_policy.get(policy_id)
        if recorded is not None:
            if _league_policy_owner(owner) != recorded.target_owner:
                raise ValueError("seated snapshot owner differs from its lineage")
            if (
                policy_training_context_content_sha256(context)
                != recorded.target_context_content_sha256
            ):
                raise ValueError(
                    "seated snapshot context differs from its recorded target"
                )
        immutable_owner = _league_policy_owner(owner)
        snapshot_transfer_needed = (
            config.snapshot_interval_updates > 0 or recorded is not None
        )
        if (
            snapshot_transfer_needed
            and (
                immutable_owner.profile != learner_owner["profile"]
                or immutable_owner.controller != learner_owner["controller"]
            )
            and config.snapshot_transfer_reason is None
        ):
            raise ValueError(
                "learner snapshot and opponent owner differ in profile or "
                "controller; provide an explicit snapshot transfer reason"
            )
        opponent_owner = immutable_owner
    assert opponent_owner is not None
    return opponent_owner


def _context_for_policy(
    values: tuple[Mapping[str, Any], ...],
    policy_id: int,
) -> dict[str, Any]:
    matches = [dict(value) for value in values if value.get("policy_id") == policy_id]
    if len(matches) != 1:
        raise ValueError("policy context/attestation is missing or repeated")
    return matches[0]


def _single_context_owner(
    context: Mapping[str, Any],
    *,
    entity_index: int,
    trainable: bool,
) -> dict[str, Any]:
    owners = [
        dict(owner)
        for owner in context.get("owners", ())
        if owner.get("entity_index") == entity_index
        and owner.get("trainable") is trainable
    ]
    if len(owners) != 1:
        raise ValueError("policy context has no unique expected owner")
    return owners[0]


def _league_policy_owner(value: Mapping[str, Any]) -> LeaguePolicyOwner:
    return LeaguePolicyOwner(
        entity_index=int(value["entity_index"]),
        profile=str(value["profile"]),
        team_id=int(value["team_id"]),
        controller=str(value["controller"]),
        trainable=bool(value["trainable"]),
    )


def league_policy_owner_manifest(value: LeaguePolicyOwner) -> dict[str, Any]:
    """Return the canonical context-row mapping for hashing/JSON."""

    if not isinstance(value, LeaguePolicyOwner):
        raise TypeError("league policy owner must be LeaguePolicyOwner")
    return {
        "entity_index": value.entity_index,
        "profile": value.profile,
        "team_id": value.team_id,
        "controller": value.controller,
        "trainable": value.trainable,
    }


def _rebind_population_state(
    state: MultiActorPopulationTrainingState,
    entrypoint: Any,
    key: jax.Array,
    *,
    learner_policy_id: int,
) -> MultiActorPopulationTrainingState:
    """Reset match-local state while retaining the learner optimizer exactly."""

    if entrypoint.trainer.trainable_policy_ids != (learner_policy_id,):
        raise ValueError("cannot rebind a different trainable policy plan")
    template = entrypoint.trainer.initialize(
        state.policy_bank,
        entrypoint._reset_arena(key),
    )
    if len(state.optimizer_states) != 1:
        raise ValueError("frozen-opponent league expects one optimizer state")
    if state.update_count.shape != (1,) or state.total_trainable_actor_steps.shape != (1,):
        raise ValueError("league population counters must have one learner row")
    return template._replace(
        optimizer_states=state.optimizer_states,
        update_count=state.update_count,
        total_trainable_actor_steps=state.total_trainable_actor_steps,
    )


def _preflight_snapshot_capacity(
    population: MultiActorPopulationTrainingState,
    bookkeeping: LeagueBookkeeping,
    config: ProductionLeagueConfig,
) -> None:
    if config.snapshot_interval_updates == 0:
        return
    applied = int(np.asarray(jax.device_get(population.update_count))[0])
    next_applied = applied + 1
    if next_applied % config.snapshot_interval_updates != 0:
        return
    if not any(role is LeaguePolicyRole.RESERVE for role in bookkeeping.roles):
        raise ValueError("next successful update needs a reserve snapshot row")


def _snapshot_due(
    population: MultiActorPopulationTrainingState,
    bookkeeping: LeagueBookkeeping,
    config: ProductionLeagueConfig,
) -> bool:
    if config.snapshot_interval_updates == 0:
        return False
    applied = int(np.asarray(jax.device_get(population.update_count))[0])
    return (
        applied > 0
        and applied % config.snapshot_interval_updates == 0
        and len(bookkeeping.snapshots) < applied // config.snapshot_interval_updates
    )


def _publish_snapshot(
    population: MultiActorPopulationTrainingState,
    bookkeeping: LeagueBookkeeping,
    config: ProductionLeagueConfig,
    *,
    learner_context: Mapping[str, Any],
    learner_context_attestation: Mapping[str, Any],
    runtime_config_content_sha256: str,
    target_owner: LeaguePolicyOwner,
    transfer_reason: str | None,
) -> tuple[
    MultiActorPopulationTrainingState,
    LeagueBookkeeping,
    LeagueSnapshot,
]:
    target = next(
        (
            index
            for index, role in enumerate(bookkeeping.roles)
            if role is LeaguePolicyRole.RESERVE
        ),
        None,
    )
    if target is None:
        raise ValueError("snapshot is due but no reserve policy row remains")
    source = config.learner_policy_id
    source_owner = _single_context_owner(
        learner_context,
        entity_index=config.actor_indices[0],
        trainable=True,
    )
    normalized_target_owner = league_policy_owner_manifest(target_owner)
    if (
        normalized_target_owner.get("entity_index") != config.actor_indices[1]
        or normalized_target_owner.get("trainable") is not False
    ):
        raise ValueError("snapshot target context differs from opponent owner")
    if (
        normalized_target_owner.get("profile") != source_owner["profile"]
        or normalized_target_owner.get("controller")
        != source_owner["controller"]
    ) and transfer_reason is None:
        raise ValueError(
            "snapshot source and target differ in profile or controller "
            "without an explicit transfer reason"
        )
    source_context_sha = policy_training_context_content_sha256(
        learner_context
    )
    if (
        learner_context_attestation.get("context_content_sha256")
        != source_context_sha
        or learner_context_attestation.get("runtime_config_content_sha256")
        != runtime_config_content_sha256
    ):
        raise ValueError("snapshot learner context attestation is invalid")
    target_context = {
        "policy_id": target,
        "owners": [normalized_target_owner],
    }
    target_context_sha = policy_training_context_content_sha256(
        target_context
    )
    bank = snapshot_policy(
        population.policy_bank,
        source_id=source,
        target_id=target,
    )
    ratings = activate_snapshot(
        bookkeeping.ratings,
        source_id=source,
        target_id=target,
    )
    roles = list(bookkeeping.roles)
    roles[target] = LeaguePolicyRole.FROZEN
    applied = int(np.asarray(jax.device_get(population.update_count))[0])
    record = LeagueSnapshot(
        generation=len(bookkeeping.snapshots) + 1,
        source_policy_id=source,
        target_policy_id=target,
        scheduler_update_index=bookkeeping.scheduler_update_index,
        learner_update_count=applied,
        source_context_content_sha256=source_context_sha,
        source_runtime_config_content_sha256=runtime_config_content_sha256,
        target_context_content_sha256=target_context_sha,
        target_owner=target_owner,
        source_profile=str(source_owner["profile"]),
        target_profile=str(normalized_target_owner["profile"]),
        source_team_id=int(source_owner["team_id"]),
        target_team_id=int(normalized_target_owner["team_id"]),
        source_controller=str(source_owner["controller"]),
        target_controller=str(normalized_target_owner["controller"]),
        transfer_reason=transfer_reason,
    )
    return (
        population._replace(policy_bank=bank),
        LeagueBookkeeping(
            ratings=ratings,
            roles=tuple(roles),
            scheduler_update_index=bookkeeping.scheduler_update_index,
            snapshots=bookkeeping.snapshots + (record,),
        ),
        record,
    )


def _validate_bookkeeping(
    bookkeeping: LeagueBookkeeping,
    config: ProductionLeagueConfig,
    *,
    learner_context: Mapping[str, Any] | None = None,
    runtime_config_content_sha256: str | None = None,
) -> None:
    count = len(config.policy_roles)
    if len(bookkeeping.roles) != count:
        raise ValueError("league role count differs from policy-bank capacity")
    if bookkeeping.roles[config.learner_policy_id] is not LeaguePolicyRole.TRAINABLE:
        raise ValueError("league learner role changed")
    if any(
        current is LeaguePolicyRole.ANCHOR and initial is not LeaguePolicyRole.ANCHOR
        or initial is LeaguePolicyRole.ANCHOR and current is not LeaguePolicyRole.ANCHOR
        for current, initial in zip(
            bookkeeping.roles,
            config.policy_roles,
            strict=True,
        )
    ):
        raise ValueError("anchor roles are immutable")
    ratings = bookkeeping.ratings
    if ratings.ratings.shape != (count,) or ratings.games.shape != (count,) or ratings.active.shape != (count,):
        raise ValueError("league rating arrays differ from policy-bank capacity")
    host_active = np.asarray(jax.device_get(ratings.active), dtype=np.bool_)
    expected_active = np.asarray(
        [role is not LeaguePolicyRole.RESERVE for role in bookkeeping.roles],
        dtype=np.bool_,
    )
    if not np.array_equal(host_active, expected_active):
        raise ValueError("league active mask differs from immutable roles")
    if bookkeeping.scheduler_update_index < 0:
        raise ValueError("scheduler update index cannot be negative")
    targets = [record.target_policy_id for record in bookkeeping.snapshots]
    if len(targets) != len(set(targets)):
        raise ValueError("a snapshot target was overwritten")
    for generation, record in enumerate(bookkeeping.snapshots, start=1):
        if record.generation != generation:
            raise ValueError("snapshot generations are not contiguous")
        if record.source_policy_id != config.learner_policy_id:
            raise ValueError("snapshot source differs from league learner")
        if not 0 <= record.target_policy_id < count:
            raise ValueError("snapshot target is outside the policy bank")
        if bookkeeping.roles[record.target_policy_id] is not LeaguePolicyRole.FROZEN:
            raise ValueError("published snapshot is not frozen")
        if len(record.source_context_content_sha256) != 64:
            raise ValueError("snapshot source context identity is malformed")
        if len(record.source_runtime_config_content_sha256) != 64:
            raise ValueError("snapshot runtime identity is malformed")
        if len(record.target_context_content_sha256) != 64:
            raise ValueError("snapshot target context identity is malformed")
        expected_target_context = {
            "policy_id": record.target_policy_id,
            "owners": [league_policy_owner_manifest(record.target_owner)],
        }
        if (
            policy_training_context_content_sha256(expected_target_context)
            != record.target_context_content_sha256
        ):
            raise ValueError("snapshot target context identity differs")
        if (
            record.target_owner.entity_index != config.actor_indices[1]
            or record.target_owner.trainable is not False
        ):
            raise ValueError("snapshot target owner differs from opponent seat")
        if (
            record.source_profile != record.target_profile
            or record.source_controller != record.target_controller
        ) and record.transfer_reason is None:
            raise ValueError("cross-context snapshot has no transfer reason")
        if (
            record.target_profile != record.target_owner.profile
            or record.target_team_id != record.target_owner.team_id
            or record.target_controller != record.target_owner.controller
        ):
            raise ValueError("snapshot target labels differ from target owner")
        if learner_context is not None:
            learner_owner = _single_context_owner(
                learner_context,
                entity_index=config.actor_indices[0],
                trainable=True,
            )
            if (
                record.source_context_content_sha256
                != policy_training_context_content_sha256(learner_context)
                or record.source_profile != learner_owner["profile"]
                or record.source_team_id != learner_owner["team_id"]
                or record.source_controller != learner_owner["controller"]
            ):
                raise ValueError("snapshot source lineage differs from learner")
        if (
            runtime_config_content_sha256 is not None
            and record.source_runtime_config_content_sha256
            != runtime_config_content_sha256
        ):
            raise ValueError("snapshot runtime lineage differs from current runtime")


def _league_key(seed: int, update: int, domain: int) -> jax.Array:
    key = jax.random.key(np.uint32(seed))
    key = jax.random.fold_in(key, np.uint32(domain))
    return jax.random.fold_in(key, np.uint32(update))


def config_manifest(config: ProductionLeagueConfig) -> dict[str, Any]:
    """Return the canonical JSON-ready configuration used by checkpoints."""

    return {
        "policy_roles": [role.value for role in config.policy_roles],
        "actor_indices": list(config.actor_indices),
        "entity_count": config.entity_count,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "rating": {
            "temperature": config.rating.temperature,
            "uniform_mix": config.rating.uniform_mix,
            "k_factor": config.rating.k_factor,
            "rating_scale": config.rating.rating_scale,
        },
        "policy_slot_order": list(config.policy_slot_order),
        "snapshot_interval_updates": config.snapshot_interval_updates,
        "include_anchors_in_training": config.include_anchors_in_training,
        "snapshot_transfer_reason": config.snapshot_transfer_reason,
    }


def config_from_manifest(value: Mapping[str, Any]) -> ProductionLeagueConfig:
    """Parse a persisted league config without accepting unknown semantics."""

    if not isinstance(value, Mapping):
        raise TypeError("league config must be an object")
    expected = {
        "policy_roles",
        "actor_indices",
        "entity_count",
        "num_envs",
        "seed",
        "rating",
        "policy_slot_order",
        "snapshot_interval_updates",
        "include_anchors_in_training",
        "snapshot_transfer_reason",
    }
    if set(value) != expected:
        raise ValueError("league config fields differ from the current contract")
    raw_rating = value["rating"]
    if not isinstance(raw_rating, Mapping) or set(raw_rating) != {
        "temperature",
        "uniform_mix",
        "k_factor",
        "rating_scale",
    }:
        raise ValueError("league rating config fields differ")
    parsed = ProductionLeagueConfig(
        policy_roles=tuple(value["policy_roles"]),
        actor_indices=tuple(value["actor_indices"]),
        entity_count=value["entity_count"],
        num_envs=value["num_envs"],
        seed=value["seed"],
        rating=SelfPlayConfig(**dict(raw_rating)),
        policy_slot_order=tuple(value["policy_slot_order"]),
        snapshot_interval_updates=value["snapshot_interval_updates"],
        include_anchors_in_training=value["include_anchors_in_training"],
        snapshot_transfer_reason=value["snapshot_transfer_reason"],
    )
    if config_manifest(parsed) != dict(value):
        raise ValueError("league config is not canonical")
    return parsed


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


__all__ = [
    "LeagueBookkeeping",
    "LeaguePolicyRole",
    "LeaguePolicyOwner",
    "LeagueSnapshot",
    "LeagueUpdatePlan",
    "ProductionLeagueConfig",
    "ProductionLeagueLoop",
    "ProductionLeagueMetrics",
    "ProductionLeagueState",
    "apply_league_outcomes",
    "assignment_content_sha256",
    "canonical_json",
    "config_from_manifest",
    "config_manifest",
    "initialize_league_bookkeeping",
    "league_update_plan_for_opponent",
    "league_policy_owner_manifest",
    "schedule_league_update",
]
