"""End-to-end recurrent IL -> PPO training on a WorldGen V2 Arena task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

from agents import training_runtime as _training_runtime  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from adk import AgentKit, AgentSpec
from agents.ppo.contracts import task_contract
from agents.ppo.runtime import default_compilation_cache
from arena.imitation import TraceCorpus, TraceSequence
from arena.jax_env import ArenaScene
from arena.tasks.framework.base import Task
from arena.tasks.games.world.generated import PLAINS_DUEL
from arena.training import (
    BehaviorPriorConfig,
    CombatFundamentalsConfig,
    ExactImitationConfig,
    arsenal_basic_attack_envelope,
    behavior_cloning_head_accuracy,
    behavior_cloning_loss,
    load_native_behavior_policy,
    load_native_replay_corpus,
    make_behavior_prior_environment,
    make_combat_fundamentals_environment,
    make_replicated_ppo_step,
    native_arsenal_profile_basic_attack_binding,
    native_behavior_observation_projector,
    transplant_native_behavior_policy,
    repeated_keys,
    split_replicas,
    stack_replicas,
)
from agents.ppo.worldgen.worldgen_trace import (
    MappedDemonstrations,
    WORLDGEN_PPO_PROJECTION_SCHEMA,
    WORLDGEN_PPO_PROJECTION_SHA256,
    capture_worldgen_trace,
    map_worldgen_trace,
)
from arena.jax_contract import HEAD_SPANS
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    current_combat_checkpoint_contract,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
)
from hytalegym.jax.training.policy import apply_policy_sequence


REPORT_SCHEMA = "hytalerl_worldgen_il_ppo_run_v2"
CHECKPOINT_SCHEMA = "hytalerl_worldgen_il_ppo_checkpoint_v2"
CURVE_CHECKPOINT_SCHEMA = "hytalerl_worldgen_il_ppo_curve_checkpoints_v1"
CHAIN_SCHEMA = "hytalerl_worldgen_il_ppo_deployment_chain_v1"
LIVE_EXECUTION_SCHEMA = "hytalerl_isolated_policy_execution_v1"
_ENVIRONMENT_HEAD_ALIASES = {
    "guard_held": "guard_off_on",
    "jump_held": "jump_off_on",
}


@dataclass(frozen=True, slots=True)
class WorldgenILPPOSettings:
    batch: int = 2
    trace_steps: int = 4
    cloning_steps: int = 32
    ppo_updates: int = 2
    evaluation_updates: tuple[int, ...] = ()
    evaluation_policy_batch_size: int = 1
    rollout_steps: int = 2
    ppo_update_epochs: int = 2
    ppo_num_minibatches: int = 1
    encoder_size: int = 8
    recurrent_size: int = 8
    cloning_learning_rate: float = 3.0e-4
    cloning_label_smoothing: float = 0.05
    cloning_entropy_weight: float = 0.01
    cloning_target_accuracy: float = 0.80
    cloning_min_steps: int = 1
    imitation_heads: tuple[str, ...] = tuple(HEAD_SPANS)
    ppo_learning_rate: float = 3.0e-4
    ppo_entropy_coefficient: float = 1.0e-2
    ppo_maximum_kl: float = 2.0e-2
    native_behavior_checkpoint: Path | None = None
    native_behavior_corpus: Path | None = None
    native_behavior_live_checkpoint: Path | None = None
    native_behavior_style: tuple[str, str] | None = None
    native_behavior_action_mismatch_scale: float = 0.0025
    native_behavior_aim_error_scale: float = 0.0025
    native_behavior_excess_spin_scale: float = 0.005
    native_behavior_spin_tolerance_degrees: float = 1.0
    native_behavior_lock_unsupervised_heads: bool = True
    native_behavior_online_prior: bool = True
    combat_fundamentals: CombatFundamentalsConfig | None = None
    seed: int = 570057
    output_dir: Path = Path("agents/ppo/artifacts/worldgen-il-run")
    compilation_cache: Path | None = field(default_factory=default_compilation_cache)

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "trace_steps",
            "ppo_updates",
            "rollout_steps",
            "ppo_update_epochs",
            "ppo_num_minibatches",
            "encoder_size",
            "recurrent_size",
            "evaluation_policy_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cloning_steps", "cloning_min_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.cloning_min_steps > self.cloning_steps:
            raise ValueError("cloning_min_steps must not exceed cloning_steps")
        for name in ("cloning_learning_rate", "ppo_learning_rate"):
            value = getattr(self, name)
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
        if self.ppo_entropy_coefficient < 0.0 or not np.isfinite(
            self.ppo_entropy_coefficient
        ):
            raise ValueError("ppo_entropy_coefficient must be nonnegative and finite")
        if self.ppo_maximum_kl <= 0.0 or not np.isfinite(self.ppo_maximum_kl):
            raise ValueError("ppo_maximum_kl must be positive and finite")
        if (self.native_behavior_checkpoint is None) != (
            self.native_behavior_corpus is None
        ):
            raise ValueError(
                "native behavior checkpoint and corpus must be supplied together"
            )
        if self.native_behavior_live_checkpoint is not None and (
            self.native_behavior_checkpoint is None
            or self.native_behavior_style is None
        ):
            raise ValueError(
                "native live checkpoint requires a source checkpoint, corpus, and style"
            )
        if self.cloning_steps == 0 and self.native_behavior_live_checkpoint is None:
            raise ValueError("zero cloning steps require a native live checkpoint")
        if self.native_behavior_style is not None and (
            self.native_behavior_checkpoint is None
            or not isinstance(self.native_behavior_style, tuple)
            or len(self.native_behavior_style) != 2
            or not all(self.native_behavior_style)
        ):
            raise ValueError(
                "native behavior style needs a checkpoint and two nonempty roles"
            )
        for name in (
            "native_behavior_action_mismatch_scale",
            "native_behavior_aim_error_scale",
            "native_behavior_excess_spin_scale",
            "native_behavior_spin_tolerance_degrees",
        ):
            value = getattr(self, name)
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.native_behavior_lock_unsupervised_heads, bool):
            raise TypeError("native_behavior_lock_unsupervised_heads must be a bool")
        if not isinstance(self.native_behavior_online_prior, bool):
            raise TypeError("native_behavior_online_prior must be a bool")
        if self.combat_fundamentals is not None and not isinstance(
            self.combat_fundamentals, CombatFundamentalsConfig
        ):
            raise TypeError("combat_fundamentals must be CombatFundamentalsConfig")
        if (
            self.combat_fundamentals is not None
            and self.rollout_steps != self.combat_fundamentals.rollout_horizon_ticks
        ):
            raise ValueError(
                "combat fundamentals require rollout_steps="
                f"{self.combat_fundamentals.rollout_horizon_ticks}"
            )
        if not 0.0 <= self.cloning_label_smoothing < 1.0:
            raise ValueError("cloning_label_smoothing must be in [0, 1)")
        if self.cloning_entropy_weight < 0.0 or not np.isfinite(
            self.cloning_entropy_weight
        ):
            raise ValueError("cloning_entropy_weight must be finite and nonnegative")
        if not 0.0 < self.cloning_target_accuracy <= 1.0:
            raise ValueError("cloning_target_accuracy must be in (0, 1]")
        if (
            not self.imitation_heads
            or len(self.imitation_heads) != len(set(self.imitation_heads))
            or not set(self.imitation_heads) <= set(HEAD_SPANS)
        ):
            raise ValueError("imitation_heads must be a unique nonempty policy subset")
        if self.batch % self.ppo_num_minibatches:
            raise ValueError("batch must be divisible by ppo_num_minibatches")
        if not isinstance(self.evaluation_updates, tuple) or any(
            isinstance(update, bool) or not isinstance(update, int)
            for update in self.evaluation_updates
        ):
            raise TypeError("evaluation_updates must be a tuple of integers")
        if self.evaluation_updates != tuple(sorted(set(self.evaluation_updates))):
            raise ValueError("evaluation_updates must be sorted and unique")
        if any(
            update < 0 or update > self.ppo_updates
            for update in self.evaluation_updates
        ):
            raise ValueError("evaluation_updates must be in [0, ppo_updates]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


class WorldgenILPPOAgent:
    """ADK-facing trainer that can ingest any compatible Arena trace."""

    def __init__(
        self,
        settings: WorldgenILPPOSettings | None = None,
        *,
        scene: ArenaScene | None = None,
        task: Task | None = None,
        difficulty: str = "inert",
    ) -> None:
        started = perf_counter()
        self.settings = settings or WorldgenILPPOSettings()
        if scene is not None and task is not None:
            raise ValueError("pass either scene or task, not both")
        self.kit = AgentKit(compilation_cache=self.settings.compilation_cache)
        self.task = PLAINS_DUEL if scene is None and task is None else task
        self.difficulty = difficulty
        if scene is None:
            assert self.task is not None
            scene = self.task.build_scene(difficulty, batch=self.settings.batch)
        self.scene = scene
        agent_name = self.scene.name.replace("/", "-").replace(".", "-")
        self.spec = AgentSpec(
            name=f"worldgen-il-ppo/{agent_name}",
            loadout=self.scene.expected_loadout,
            scene="combat/open_flat_control",
        )
        self.kit.register(self.spec)
        self.handle = self.kit.build(self.spec, batch=self.settings.batch)
        self._validate_surface()
        self.build_wall_seconds = perf_counter() - started

    def capture_trace(self, key: jax.Array):
        return capture_worldgen_trace(
            self.scene, key, self.settings.trace_steps, compile=True
        )

    def ingest_trace(self, trace, ppo, cloning) -> MappedDemonstrations:
        return map_worldgen_trace(
            self.scene,
            trace,
            ppo,
            cloning,
            supervised_heads=self.settings.imitation_heads,
        )

    def cloning_config(self):
        settings = self.settings
        return self.handle.arena.behavior_cloning_config(
            learning_rate=settings.cloning_learning_rate,
            label_smoothing=settings.cloning_label_smoothing,
            entropy_weight=settings.cloning_entropy_weight,
        )

    def ppo_config(self):
        settings = self.settings
        config = self.handle.arena.ppo_config(
            self.scene,
            rollout_steps=settings.rollout_steps,
            update_epochs=settings.ppo_update_epochs,
            num_minibatches=settings.ppo_num_minibatches,
            encoder_size=settings.encoder_size,
            recurrent_size=settings.recurrent_size,
            learning_rate=settings.ppo_learning_rate,
            entropy_coefficient=settings.ppo_entropy_coefficient,
            action_distribution="arsenal_standard_root_v1",
            episode_horizon_ticks=(
                None
                if settings.combat_fundamentals is None
                else settings.combat_fundamentals.rollout_horizon_ticks
            ),
        )
        if (
            settings.combat_fundamentals is not None
            and abs(float(config.gamma) - float(settings.combat_fundamentals.discount))
            > 1.0e-9
        ):
            raise ValueError("combat reward discount must equal PPO gamma")
        return config

    def run(
        self,
        trace=None,
        *,
        trace_capture: Callable[[jax.Array], Any] | None = None,
        evaluate: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Clone a trace, warm-start the same policy tree, then update it with PPO."""

        run_started = perf_counter()
        source_snapshot = _loaded_source_hashes()
        settings = self.settings
        task_manifest = task_contract(self.task, self.difficulty)
        keys = jax.random.split(jax.random.key(settings.seed), 2 + settings.ppo_updates)
        ppo = self.ppo_config()
        cloning = self.cloning_config()

        trace_started = perf_counter()
        if trace is not None and trace_capture is not None:
            raise ValueError("pass trace or trace_capture, not both")
        trace = (
            trace
            if trace is not None
            else (
                self.capture_trace(keys[0])
                if trace_capture is None
                else trace_capture(keys[0])
            )
        )
        mapped = self.ingest_trace(trace, ppo, cloning)
        trace_wall_seconds = perf_counter() - trace_started

        training = self.handle.arena.training
        rl_scene = self.scene
        combat_reward = None
        if settings.combat_fundamentals is not None:
            attack_envelope = arsenal_basic_attack_envelope(
                self.scene.runtime_config,
                settings.combat_fundamentals.basic_ability_slot,
            )
            combat_reward = settings.combat_fundamentals.describe()
            combat_reward["sha256"] = settings.combat_fundamentals.sha256
            combat_reward["attack_envelope"] = attack_envelope.describe()
            rl_scene = replace(
                self.scene,
                environment=make_combat_fundamentals_environment(
                    self.scene.environment,
                    settings.combat_fundamentals,
                    attack_envelope=attack_envelope,
                    agent_max_health=self.scene.combat_params.agent_max_health,
                    target_max_health=self.scene.combat_params.target_max_health,
                ),
            )
        initial_state = training.initialize_ppo(keys[1], rl_scene, ppo)
        initial_params = initial_state.policy_params
        cloning_initial_params = initial_params
        il_scene = rl_scene
        native_prior = None
        if settings.native_behavior_checkpoint is not None:
            cache_hits = _native_behavior_source.cache_info().hits
            corpus, policy = _native_behavior_source(
                str(settings.native_behavior_checkpoint.resolve()),
                str(settings.native_behavior_corpus.resolve()),
            )
            source_cache_hit = _native_behavior_source.cache_info().hits > cache_hits
            ability_binding = (
                None
                if settings.native_behavior_style is None
                or "ability_none_plus_slots" not in policy.supervised_heads
                else native_arsenal_profile_basic_attack_binding(
                    corpus,
                    settings.native_behavior_style[0],
                    self.scene.expected_loadout,
                )
            )
            expected_live_params, transfer = transplant_native_behavior_policy(
                policy,
                initial_params,
                ppo,
                self.scene.combat_params,
                settings.native_behavior_style,
                ability_binding=ability_binding,
                target_ability_slot_contract_sha256=(
                    None
                    if ability_binding is None
                    else ability_binding.target_ability_slot_contract_sha256
                ),
            )
            live_checkpoint = None
            if settings.native_behavior_live_checkpoint is None:
                cloning_initial_params = expected_live_params
            else:
                from agents.ppo.native.native_live_policy import load_live_policy

                assert settings.native_behavior_style is not None
                cloning_initial_params, live_checkpoint = load_live_policy(
                    settings.native_behavior_live_checkpoint,
                    expected_live_params,
                    ppo,
                    scene=self.scene,
                    task_manifest=task_manifest,
                    native_checkpoint_sha256=policy.checkpoint_sha256,
                    corpus=corpus,
                    style=settings.native_behavior_style,
                    transfer=transfer,
                    seed=settings.seed,
                    ppo_updates=settings.ppo_updates,
                )
            imitation = ExactImitationConfig(
                action_mismatch_scale=(settings.native_behavior_action_mismatch_scale),
                aim_error_scale=settings.native_behavior_aim_error_scale,
                excess_spin_scale=settings.native_behavior_excess_spin_scale,
                spin_tolerance_degrees=(
                    settings.native_behavior_spin_tolerance_degrees
                ),
            )
            if settings.native_behavior_online_prior:
                environment = make_behavior_prior_environment(
                    rl_scene.environment,
                    policy.params,
                    policy.config,
                    native_behavior_observation_projector(
                        policy, settings.native_behavior_style
                    ),
                    BehaviorPriorConfig(
                        tuple(transfer["transferred_heads"]),
                        imitation,
                        lock_unsupervised_heads=(
                            settings.native_behavior_lock_unsupervised_heads
                        ),
                    ),
                )
                il_scene = replace(rl_scene, environment=environment)
            fundamentals_scope = bool(
                settings.combat_fundamentals is not None
                and settings.combat_fundamentals.restrict_to_fundamentals
            )
            prior_scope = bool(
                settings.native_behavior_online_prior
                and settings.native_behavior_lock_unsupervised_heads
            )
            native_prior = {
                "checkpoint": str(settings.native_behavior_checkpoint),
                "corpus": str(settings.native_behavior_corpus),
                "source_cache_hit": source_cache_hit,
                "style": (
                    None
                    if settings.native_behavior_style is None
                    else list(settings.native_behavior_style)
                ),
                "transfer": transfer,
                "live_checkpoint": live_checkpoint,
                "initial_parameter_l2_delta": _tree_delta(
                    initial_params,
                    cloning_initial_params,
                ),
                "online_during_ppo": settings.native_behavior_online_prior,
                "action_scope": {
                    "locked_to_neutral": prior_scope or fundamentals_scope,
                    "trainable_heads": (
                        [
                            "ability_none_plus_slots",
                            "locomotion_gait_compass",
                            "yaw_delta_bins",
                            "pitch_delta_bins",
                        ]
                        if fundamentals_scope
                        else list(transfer["transferred_heads"])
                    ),
                    "other_heads": (
                        "choice_zero_only"
                        if prior_scope or fundamentals_scope
                        else "trainable_from_neutral_prior"
                    ),
                    "source": (
                        "combat_fundamentals"
                        if fundamentals_scope
                        else "online_native_prior"
                        if prior_scope
                        else "unrestricted"
                    ),
                },
                "reward": {
                    "schema": "arena_online_exact_behavior_prior_v1",
                    "enabled": settings.native_behavior_online_prior,
                    "action_mismatch_scale": imitation.action_mismatch_scale,
                    "aim_error_scale": imitation.aim_error_scale,
                    "excess_spin_scale": imitation.excess_spin_scale,
                    "spin_tolerance_degrees": imitation.spin_tolerance_degrees,
                    "alignment": "expert_queried_on_current_policy_state",
                    "unknown_heads": "abstain",
                },
            }

        def apply(candidate, observations, action_mask):
            initial_carry = jnp.zeros(
                (observations.shape[1], ppo.recurrent_size), dtype=jnp.float32
            )
            _, logits, _ = apply_policy_sequence(
                candidate,
                observations,
                initial_carry,
                mapped.episode_start,
                action_mask,
            )
            return logits

        trainer = self.handle.arena.behavior_cloning_trainer(
            apply, cloning, compile=True
        )
        optimizer_state = trainer.initialize(cloning_initial_params)
        head_labels = _head_label_diagnostics(mapped, ppo)
        stop_heads = (
            tuple(
                name
                for name in mapped.supervised_heads
                if sum(value > 0 for value in head_labels[name]["histogram"]) > 1
            )
            or mapped.supervised_heads
        )
        stop_indices = tuple(_action_head_names().index(name) for name in stop_heads)

        def head_accuracy(candidate):
            return behavior_cloning_head_accuracy(
                apply(
                    candidate,
                    mapped.batch.observation,
                    mapped.batch.action_mask,
                ),
                mapped.batch,
                cloning,
            )

        head_accuracy = jax.jit(head_accuracy)
        initial_head_accuracy, head_support = head_accuracy(cloning_initial_params)
        initial_loss = behavior_cloning_loss(
            apply(
                cloning_initial_params,
                mapped.batch.observation,
                mapped.batch.action_mask,
            ),
            mapped.batch,
            cloning,
        )
        cloning_times = []
        params = cloning_initial_params
        cloning_steps = 0
        for _ in range(settings.cloning_steps):
            started = perf_counter()
            next_params, next_optimizer_state, cloning_metrics = trainer.step(
                params, optimizer_state, mapped.batch
            )
            jax.block_until_ready(cloning_metrics.loss.loss)
            if cloning_steps >= settings.cloning_min_steps:
                current_head_accuracy, _ = head_accuracy(params)
                jax.block_until_ready(current_head_accuracy)
                if (
                    min(_scalar(current_head_accuracy[index]) for index in stop_indices)
                    >= settings.cloning_target_accuracy
                ):
                    break
            params, optimizer_state = next_params, next_optimizer_state
            cloning_times.append(perf_counter() - started)
            cloning_steps += 1
        final_loss = behavior_cloning_loss(
            apply(params, mapped.batch.observation, mapped.batch.action_mask),
            mapped.batch,
            cloning,
        )
        jax.block_until_ready(final_loss.loss)
        final_head_accuracy, _ = head_accuracy(params)
        il_parameter_delta = _tree_delta(initial_params, params)

        il_initial_state = (
            initial_state
            if native_prior is None or not settings.native_behavior_online_prior
            else training.initialize_ppo(keys[1], il_scene, ppo)
        )
        if _tree_delta(initial_params, il_initial_state.policy_params) != 0.0:
            raise RuntimeError("paired PPO arms did not share random initialization")
        il_state = training.warm_start_ppo(il_initial_state, params, ppo)
        scratch_state = initial_state
        shared_environment = il_scene.environment is rl_scene.environment
        if shared_environment:
            ppo_step = make_replicated_ppo_step(il_scene, ppo, 2, compile=True)
            replica_state = stack_replicas((il_state, scratch_state))
            training_execution = {
                **ppo_step.describe(),
                "arms": ["il_ppo", "ppo_only"],
                "common_random_numbers": True,
            }
        else:
            ppo_step = training.make_ppo_step(il_scene, ppo, compile=True)
            scratch_step = training.make_ppo_step(rl_scene, ppo, compile=True)
            replica_state = None
            training_execution = {
                "schema": "arena_replicated_training_v1",
                "mode": "serial_contract_groups",
                "replicas": 1,
                "groups": 2,
                "reason": "training environments differ",
                "common_random_numbers": True,
            }
        ppo_rows, scratch_rows = [], []
        warm_params = params
        evaluation_cache: dict[tuple[str, int], Any] = {}
        evaluation_execution = None
        evaluation_curve = []
        curve_parameters = {}
        evaluation_update_set = set(settings.evaluation_updates)
        defer_evaluation = (
            evaluate is not None and settings.evaluation_policy_batch_size > 1
        )
        if 0 in evaluation_update_set:
            curve_parameters[0] = _host_policy_pair(warm_params, initial_params)
            if evaluate is not None and not defer_evaluation:
                evaluation_curve.append(
                    _evaluation_curve_row(
                        evaluate,
                        evaluation_cache,
                        update=0,
                        environment_steps=0,
                        il_ppo=warm_params,
                        ppo_only=initial_params,
                    )
                )
        for index, key in enumerate(keys[2 : 2 + settings.ppo_updates], start=1):
            if replica_state is not None:
                replica_state, rows = _ppo_replicated_update(
                    ppo_step,
                    replica_state,
                    key,
                    index,
                    settings.batch * settings.rollout_steps,
                )
                il_state, scratch_state = split_replicas(replica_state, 2)
                il_row, scratch_row = rows
            else:
                il_state, il_row = _ppo_update(
                    ppo_step,
                    il_state,
                    key,
                    index,
                    settings.batch * settings.rollout_steps,
                )
                scratch_state, scratch_row = _ppo_update(
                    scratch_step,
                    scratch_state,
                    key,
                    index,
                    settings.batch * settings.rollout_steps,
                )
            _require_safe_kl(il_row, "native_prior", settings.ppo_maximum_kl)
            _require_safe_kl(scratch_row, "scratch", settings.ppo_maximum_kl)
            ppo_rows.append(il_row)
            scratch_rows.append(scratch_row)
            if index in evaluation_update_set:
                curve_parameters[index] = _host_policy_pair(
                    il_state.policy_params,
                    scratch_state.policy_params,
                )
                if evaluate is not None and not defer_evaluation:
                    evaluation_curve.append(
                        _evaluation_curve_row(
                            evaluate,
                            evaluation_cache,
                            update=index,
                            environment_steps=(
                                index * settings.batch * settings.rollout_steps
                            ),
                            il_ppo=il_state.policy_params,
                            ppo_only=scratch_state.policy_params,
                        )
                    )
        if defer_evaluation:
            requests = {
                ("ppo_only", 0): initial_params,
                ("il_ppo", 0): warm_params,
                ("il_ppo", settings.ppo_updates): il_state.policy_params,
                ("ppo_only", settings.ppo_updates): scratch_state.policy_params,
            }
            for update, pair in curve_parameters.items():
                requests[("il_ppo", update)] = pair["il_ppo"]
                requests[("ppo_only", update)] = pair["ppo_only"]
            evaluation_execution = _populate_batched_evaluations(
                evaluate,
                evaluation_cache,
                requests,
                settings.evaluation_policy_batch_size,
            )
            evaluation_curve = [
                _evaluation_curve_row(
                    evaluate,
                    evaluation_cache,
                    update=update,
                    environment_steps=(
                        update * settings.batch * settings.rollout_steps
                    ),
                    il_ppo=curve_parameters[update]["il_ppo"],
                    ppo_only=curve_parameters[update]["ppo_only"],
                )
                for update in settings.evaluation_updates
            ]

        ppo_parameter_delta = _tree_delta(warm_params, il_state.policy_params)
        scratch_parameter_delta = _tree_delta(
            initial_params, scratch_state.policy_params
        )
        evaluations = _legacy_evaluations(
            evaluate,
            evaluation_cache,
            initial_params=initial_params,
            warm_params=warm_params,
            il_ppo=il_state.policy_params,
            ppo_only=scratch_state.policy_params,
            final_update=settings.ppo_updates,
        )
        if evaluate is not None and evaluation_execution is None:
            evaluation_execution = _serial_evaluation_execution(evaluation_cache)

        expected_updates = settings.ppo_updates
        expected_steps = settings.batch * settings.rollout_steps * expected_updates
        for name, candidate in (
            ("il_ppo", il_state),
            ("ppo_only", scratch_state),
        ):
            if (
                _scalar(candidate.update_count) != expected_updates
                or _scalar(candidate.total_environment_steps) != expected_steps
            ):
                raise RuntimeError(f"{name} PPO counters do not match the run")
        initialized = (
            settings.cloning_steps > 0
            and _scalar(final_loss.loss) < _scalar(initial_loss.loss)
        ) or (
            settings.cloning_steps == 0
            and native_prior is not None
            and native_prior["initial_parameter_l2_delta"] > 0.0
        )
        learned = (
            initialized
            and il_parameter_delta > 0.0
            and ppo_parameter_delta > 0.0
            and scratch_parameter_delta > 0.0
            and all(
                np.isfinite(row["total_loss"]) for row in (*ppo_rows, *scratch_rows)
            )
        )
        if not learned:
            raise RuntimeError("IL -> PPO run completed without a learning signal")
        _require_unchanged_sources(source_snapshot)

        output = settings.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_base = {
            "schema": CHECKPOINT_SCHEMA,
            "transfer_contract": current_combat_checkpoint_contract(
                arsenal_runtime_capacity=self.scene.runtime_capacity,
                policy_surface=ARSENAL_POLICY_SURFACE,
                world_geometry_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
            ),
            "scene_contract_sha256": self.scene.contract_sha256,
            "world": dict(self.scene.world_identity),
            "trace_identity_sha256": _trace_identity_sha256(trace),
            "trace_projection_sha256": WORLDGEN_PPO_PROJECTION_SHA256,
            "task_contract_sha256": (
                None if task_manifest is None else task_manifest["sha256"]
            ),
            "combat_fundamentals_reward_sha256": (
                None if combat_reward is None else combat_reward["sha256"]
            ),
            "updates": expected_updates,
            "environment_steps": expected_steps,
        }
        checkpoints = {
            "il_ppo": _write_checkpoint(
                output / "il_ppo_policy.npz",
                il_state.policy_params,
                ppo,
                {
                    **checkpoint_base,
                    "training_arm": (
                        "il_warm_start_ppo"
                        if native_prior is None
                        else (
                            "native_behavior_prior_ppo"
                            if settings.native_behavior_online_prior
                            else "native_initialized_ppo"
                        )
                    ),
                    "native_behavior_prior": native_prior,
                },
            ),
            "ppo_only": _write_checkpoint(
                output / "ppo_only_policy.npz",
                scratch_state.policy_params,
                ppo,
                {**checkpoint_base, "training_arm": "ppo_only"},
            ),
        }
        curve_checkpoints = (
            None
            if not curve_parameters
            else _write_curve_checkpoints(
                output,
                curve_parameters,
                ppo,
                checkpoint_base,
                transitions_per_update=settings.batch * settings.rollout_steps,
            )
        )

        cache = self.kit.jax_runtime_settings
        report = {
            "schema": REPORT_SCHEMA,
            "learned": True,
            "settings": {
                "seed": settings.seed,
                "batch": settings.batch,
                "trace_steps": settings.trace_steps,
                "cloning_steps": settings.cloning_steps,
                "cloning_min_steps": settings.cloning_min_steps,
                "cloning_learning_rate": settings.cloning_learning_rate,
                "cloning_label_smoothing": settings.cloning_label_smoothing,
                "cloning_entropy_weight": settings.cloning_entropy_weight,
                "cloning_target_accuracy": settings.cloning_target_accuracy,
                "imitation_heads": list(settings.imitation_heads),
                "ppo_updates": settings.ppo_updates,
                "evaluation_updates": list(settings.evaluation_updates),
                "evaluation_policy_batch_size": (settings.evaluation_policy_batch_size),
                "rollout_steps": settings.rollout_steps,
                "ppo_update_epochs": settings.ppo_update_epochs,
                "ppo_num_minibatches": settings.ppo_num_minibatches,
                "ppo_learning_rate": settings.ppo_learning_rate,
                "ppo_entropy_coefficient": settings.ppo_entropy_coefficient,
                "ppo_maximum_kl": settings.ppo_maximum_kl,
                "encoder_size": settings.encoder_size,
                "recurrent_size": settings.recurrent_size,
                "native_behavior_checkpoint": (
                    None
                    if settings.native_behavior_checkpoint is None
                    else str(settings.native_behavior_checkpoint)
                ),
                "native_behavior_corpus": (
                    None
                    if settings.native_behavior_corpus is None
                    else str(settings.native_behavior_corpus)
                ),
                "native_behavior_live_checkpoint": (
                    None
                    if settings.native_behavior_live_checkpoint is None
                    else str(settings.native_behavior_live_checkpoint)
                ),
                "native_behavior_style": (
                    None
                    if settings.native_behavior_style is None
                    else list(settings.native_behavior_style)
                ),
                "native_behavior_lock_unsupervised_heads": (
                    settings.native_behavior_lock_unsupervised_heads
                ),
                "native_behavior_online_prior": (settings.native_behavior_online_prior),
                "combat_fundamentals_reward": combat_reward,
            },
            "task_contract": task_manifest,
            "scene": {
                "name": self.scene.name,
                "contract_sha256": self.scene.contract_sha256,
                "task": None if self.task is None else self.task.name,
                "difficulty": self.difficulty,
                "world": dict(self.scene.world_identity),
            },
            "surface": {
                "observation_schema": self.scene.environment.spec.observation_schema,
                "observation_size": ppo.observation_size,
                "action_schema": self.scene.environment.spec.action_schema,
                "action_head_names": list(self.handle.action_head_names),
                "environment_action_head_names": [
                    item.name for item in self.scene.environment.spec.action_components
                ],
                "environment_head_aliases": _ENVIRONMENT_HEAD_ALIASES,
                "action_head_sizes": list(ppo.action_head_sizes),
                "action_size": ppo.action_size,
                "distribution": ppo.action_distribution,
            },
            "trace": {
                "contract_sha256": _trace_contract_sha256(trace),
                "identity_sha256": _trace_identity_sha256(trace),
                "rows": trace.rows,
                "source_trace_identity_sha256": list(mapped.trace_identity_sha256),
                "episode_lengths": list(mapped.lengths),
                "projection_schema": WORLDGEN_PPO_PROJECTION_SCHEMA,
                "projection_sha256": WORLDGEN_PPO_PROJECTION_SHA256,
                "current_next_stateful": True,
                "selected_actions_legal": True,
                "capture_wall_seconds": trace_wall_seconds,
                "transitions_per_second": trace.rows / trace_wall_seconds,
                "teacher_rollout": _trace_diagnostics(trace),
                "demonstrator": _trace_demonstrator(trace),
            },
            "imitation": {
                "steps_requested": settings.cloning_steps,
                "steps": cloning_steps,
                "early_stopped": cloning_steps < settings.cloning_steps,
                "stop_rule": "variable_head_accuracy_before_next_update_v2",
                "stop_heads": list(stop_heads),
                "supervised_heads": list(mapped.supervised_heads),
                "head_labels": head_labels,
                "head_accuracy": _head_metric_diagnostics(
                    initial_head_accuracy,
                    final_head_accuracy,
                    head_support,
                ),
                "initial_loss": _scalar(initial_loss.loss),
                "final_loss": _scalar(final_loss.loss),
                "initial_negative_log_likelihood": _scalar(
                    initial_loss.negative_log_likelihood
                ),
                "final_negative_log_likelihood": _scalar(
                    final_loss.negative_log_likelihood
                ),
                "initial_entropy": _scalar(initial_loss.entropy),
                "final_entropy": _scalar(final_loss.entropy),
                "initial_accuracy": _scalar(initial_loss.accuracy),
                "final_accuracy": _scalar(final_loss.accuracy),
                "parameter_l2_delta": il_parameter_delta,
                "worldgen_bc_parameter_l2_delta": _tree_delta(
                    cloning_initial_params,
                    params,
                ),
                "first_step_wall_seconds": (
                    None if not cloning_times else cloning_times[0]
                ),
                "warm_step_mean_wall_seconds": (
                    None
                    if not cloning_times
                    else float(np.mean(cloning_times[1:] or cloning_times))
                ),
            },
            "ppo": {
                "updates": ppo_rows,
                "parameter_l2_delta": ppo_parameter_delta,
                "warm_started_same_parameter_tree": True,
                "transitions_per_arm": expected_steps,
                "common_random_numbers": True,
                "update_epochs": ppo.update_epochs,
                "num_minibatches": ppo.num_minibatches,
                "learning_rate": ppo.learning_rate,
                "entropy_coefficient": ppo.entropy_coefficient,
                "scratch_control": {
                    "updates": scratch_rows,
                    "parameter_l2_delta": scratch_parameter_delta,
                    "same_initial_parameter_tree": True,
                },
            },
            "checkpoints": checkpoints,
            "native_behavior_prior": native_prior,
            "combat_fundamentals_reward": combat_reward,
            "adk": self.handle.arena.describe(),
            "runtime": {
                "jax_version": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "compilation_cache": (
                    None
                    if cache is None
                    else {
                        "enabled": cache.enabled,
                        "path": (
                            None
                            if cache.compilation_cache is None
                            else str(cache.compilation_cache)
                        ),
                        "runtime_namespace": cache.runtime_namespace,
                        "disabled_reason": cache.disabled_reason,
                    }
                ),
                "build_wall_seconds": self.build_wall_seconds,
                "run_wall_seconds": perf_counter() - run_started,
                "evaluation_execution": evaluation_execution,
                "training_execution": training_execution,
                "loaded_source_sha256": _source_snapshot_sha256(source_snapshot),
            },
        }
        if evaluations is not None:
            report["evaluation"] = evaluations
        if evaluation_curve:
            report["evaluation_curve"] = {
                "common_random_numbers": True,
                "updates": list(settings.evaluation_updates),
                "rows": evaluation_curve,
            }
        if curve_checkpoints is not None:
            report["curve_checkpoints"] = curve_checkpoints
        report_path = output / "report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    def _validate_surface(self) -> None:
        spec = self.scene.environment.spec
        if self.scene.expected_batch != self.settings.batch:
            raise ValueError("Arena scene batch differs from the agent settings")
        names = tuple(
            _ENVIRONMENT_HEAD_ALIASES.get(item.name, item.name)
            for item in spec.action_components
        )
        sizes = tuple(item.size for item in spec.action_components)
        if names != tuple(self.handle.action_head_names) or sizes != tuple(
            self.handle.action_head_sizes
        ):
            raise ValueError("ADK and Arena action contracts differ")
        if spec.observation_size != self.handle.observation_size:
            raise ValueError("ADK and Arena observation widths differ")


def _cached_evaluation(
    evaluate: Callable[[Any], dict[str, Any]],
    cache: dict[tuple[str, int], Any],
    arm: str,
    update: int,
    parameters: Any,
):
    key = (arm, update)
    if key not in cache:
        cache[key] = evaluate(parameters)
    return cache[key]


@lru_cache(maxsize=4)
def _native_behavior_source(checkpoint: str, corpus: str):
    """Verify and project an immutable native prior once per Python process."""

    replay = load_native_replay_corpus(corpus)
    return replay, load_native_behavior_policy(checkpoint, replay)


def _evaluation_curve_row(
    evaluate: Callable[[Any], dict[str, Any]],
    cache: dict[tuple[str, int], Any],
    *,
    update: int,
    environment_steps: int,
    il_ppo: Any,
    ppo_only: Any,
) -> dict[str, Any]:
    return {
        "update": update,
        "environment_steps_per_arm": environment_steps,
        "il_ppo": _cached_evaluation(evaluate, cache, "il_ppo", update, il_ppo),
        "ppo_only": _cached_evaluation(evaluate, cache, "ppo_only", update, ppo_only),
    }


def _populate_batched_evaluations(
    evaluate: Callable[[Any], dict[str, Any]],
    cache: dict[tuple[str, int], Any],
    requests: dict[tuple[str, int], Any],
    batch_size: int,
) -> dict[str, Any]:
    evaluate_many = getattr(evaluate, "evaluate_many", None)
    if evaluate_many is None:
        raise TypeError("evaluation batching requires an evaluate_many callable")
    items = list(requests.items())
    calls = []
    for offset in range(0, len(items), batch_size):
        chunk = items[offset : offset + batch_size]
        reports = evaluate_many([parameters for _, parameters in chunk])
        if len(reports) != len(chunk):
            raise RuntimeError("batched evaluator returned the wrong policy count")
        batch = reports[0].get("evaluation_batch", {})
        if (
            batch.get("policies") != len(chunk)
            or not batch.get("common_random_numbers")
            or not np.isfinite(batch.get("wall_seconds", np.nan))
        ):
            raise RuntimeError("batched evaluator omitted its execution receipt")
        calls.append(
            {
                "policies": len(chunk),
                "wall_seconds": float(batch["wall_seconds"]),
            }
        )
        cache.update(
            (key, report)
            for (key, _parameters), report in zip(chunk, reports, strict=True)
        )
    wall_seconds = sum(call["wall_seconds"] for call in calls)
    return {
        "schema": "hytalerl_policy_evaluation_execution_v1",
        "mode": "batched",
        "common_random_numbers": True,
        "requested_batch_size": batch_size,
        "policy_rollouts": len(items),
        "batch_invocations": len(calls),
        "maximum_policies_per_invocation": max(
            (call["policies"] for call in calls), default=0
        ),
        "wall_seconds": wall_seconds,
        "policy_rollouts_per_second": len(items) / max(wall_seconds, 1.0e-9),
        "calls": calls,
    }


def _serial_evaluation_execution(
    cache: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    wall_seconds = sum(float(report["wall_seconds"]) for report in cache.values())
    count = len(cache)
    return {
        "schema": "hytalerl_policy_evaluation_execution_v1",
        "mode": "serial",
        "common_random_numbers": True,
        "requested_batch_size": 1,
        "policy_rollouts": count,
        "batch_invocations": count,
        "maximum_policies_per_invocation": min(count, 1),
        "wall_seconds": wall_seconds,
        "policy_rollouts_per_second": count / max(wall_seconds, 1.0e-9),
        "calls": [
            {"policies": 1, "wall_seconds": float(report["wall_seconds"])}
            for report in cache.values()
        ],
    }


def _legacy_evaluations(
    evaluate: Callable[[Any], dict[str, Any]] | None,
    cache: dict[tuple[str, int], Any],
    *,
    initial_params: Any,
    warm_params: Any,
    il_ppo: Any,
    ppo_only: Any,
    final_update: int,
) -> dict[str, Any] | None:
    if evaluate is None:
        return None
    return {
        "before_il": _cached_evaluation(evaluate, cache, "ppo_only", 0, initial_params),
        "after_il": _cached_evaluation(evaluate, cache, "il_ppo", 0, warm_params),
        "after_ppo": _cached_evaluation(
            evaluate, cache, "il_ppo", final_update, il_ppo
        ),
        "ppo_only": _cached_evaluation(
            evaluate, cache, "ppo_only", final_update, ppo_only
        ),
    }


def _host_policy_pair(il_ppo: Any, ppo_only: Any) -> dict[str, Any]:
    def copy_to_host(value):
        return np.array(jax.device_get(value), copy=True)

    return {
        "il_ppo": jax.tree.map(copy_to_host, il_ppo),
        "ppo_only": jax.tree.map(copy_to_host, ppo_only),
    }


def _ppo_update(step, state, key, index: int, transitions: int):
    started = perf_counter()
    state, metrics = step(state, key)
    jax.block_until_ready(metrics.total_loss)
    elapsed = perf_counter() - started
    return state, _ppo_metric_row(metrics, index, transitions, elapsed)


def _ppo_replicated_update(step, state, key, index: int, transitions: int):
    started = perf_counter()
    state, metrics = step(state, repeated_keys(key, step.replicas))
    jax.block_until_ready(metrics.total_loss)
    elapsed = perf_counter() - started
    rows = tuple(
        _ppo_metric_row(
            jax.tree.map(lambda leaf, replica=replica: leaf[replica], metrics),
            index,
            transitions,
            elapsed,
            aggregate_transitions=transitions * step.replicas,
        )
        for replica in range(step.replicas)
    )
    return state, rows


def _ppo_metric_row(
    metrics,
    index: int,
    transitions: int,
    elapsed: float,
    *,
    aggregate_transitions: int | None = None,
):
    row = {
        "index": index,
        "wall_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
        **{name: _scalar(value) for name, value in metrics._asdict().items()},
    }
    if aggregate_transitions is not None:
        row["aggregate_transitions_per_second"] = aggregate_transitions / elapsed
    return row


def _head_label_diagnostics(mapped: MappedDemonstrations, config) -> dict[str, Any]:
    actions = np.asarray(jax.device_get(mapped.batch.action))
    supervised = np.asarray(jax.device_get(mapped.batch.supervision_mask))
    action_mask = np.asarray(jax.device_get(mapped.batch.action_mask))
    result = {}
    offset = 0
    for index, (name, size) in enumerate(
        zip(_action_head_names(), config.action_head_sizes, strict=True)
    ):
        known = supervised[..., index]
        selected = actions[..., index][known]
        histogram = np.bincount(selected, minlength=size).astype(int)
        legal = np.flatnonzero(
            np.any(action_mask[..., offset : offset + size][known], axis=0)
        )
        probabilities = histogram[histogram > 0] / max(1, selected.size)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        result[name] = {
            "labels": int(selected.size),
            "histogram": histogram.tolist(),
            "unique_labels": int(np.count_nonzero(histogram)),
            "observed_legal_choices": legal.astype(int).tolist(),
            "legal_choice_coverage": float(
                np.count_nonzero(histogram) / max(1, legal.size)
            ),
            "active_fraction": (
                None if selected.size == 0 else float(np.mean(selected != 0))
            ),
            "normalized_label_entropy": (
                None
                if selected.size == 0
                else entropy / np.log(size)
                if size > 1
                else 0.0
            ),
        }
        offset += size
    return result


def _head_metric_diagnostics(initial, final, support) -> dict[str, Any]:
    return {
        name: {
            "initial_accuracy": _scalar(initial[index]),
            "final_accuracy": _scalar(final[index]),
            "weighted_support": _scalar(support[index]),
        }
        for index, name in enumerate(_action_head_names())
    }


def _trace_diagnostics(trace) -> dict[str, Any]:
    if isinstance(trace, TraceCorpus):
        rows = [_trace_diagnostics(item) for item in trace.traces]
        episodes = sum(row["episodes"] for row in rows)
        return {
            "reward_component": rows[0]["reward_component"],
            "trace_count": len(rows),
            "episodes": episodes,
            "terminated_episodes": sum(row["terminated_episodes"] for row in rows),
            "truncated_episodes": sum(row["truncated_episodes"] for row in rows),
            "mean_return": sum(row["mean_return"] * row["episodes"] for row in rows)
            / episodes,
            "minimum_return": min(row["minimum_return"] for row in rows),
            "maximum_return": max(row["maximum_return"] for row in rows),
        }
    reward_names = tuple(
        spec.name
        for spec in trace.contract.components
        if spec.name.startswith("reward.")
    )
    if len(reward_names) != 1:
        raise ValueError("WorldGen trace must declare exactly one reward component")
    rewards = np.asarray(trace.components[reward_names[0]].current, dtype=np.float32)
    episode_ids = tuple(dict.fromkeys(int(value) for value in trace.episode_id))
    returns = np.asarray(
        [rewards[trace.episode_id == episode].sum() for episode in episode_ids],
        dtype=np.float32,
    )
    return {
        "reward_component": reward_names[0],
        "episodes": len(episode_ids),
        "terminated_episodes": int(np.sum(trace.terminated)),
        "truncated_episodes": int(np.sum(trace.truncated)),
        "mean_return": float(np.mean(returns)),
        "minimum_return": float(np.min(returns)),
        "maximum_return": float(np.max(returns)),
    }


def _trace_contract_sha256(trace: TraceSequence | TraceCorpus) -> str:
    return (
        trace.contract.sha256
        if isinstance(trace, TraceSequence)
        else trace.contract_sha256
    )


def _trace_identity_sha256(trace: TraceSequence | TraceCorpus) -> str:
    return trace.identity_sha256 if isinstance(trace, TraceSequence) else trace.sha256


def _trace_demonstrator(trace: TraceSequence | TraceCorpus) -> dict[str, Any]:
    def describe(item):
        return {
            key.removeprefix("arena.demonstrator."): value
            for key, value in item.parameters.items()
            if key.startswith("arena.demonstrator.")
        }

    if isinstance(trace, TraceSequence):
        return describe(trace)
    return {
        "kind": "trace_corpus",
        "trace_count": len(trace.traces),
        "corpus_sha256": trace.sha256,
        "sources": [describe(item) for item in trace.traces],
    }


def _action_head_names() -> tuple[str, ...]:
    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
    )

    return tuple(ARSENAL_POLICY_ACTION_HEAD_NAMES)


def _require_safe_kl(row: dict[str, Any], arm: str, maximum: float) -> None:
    observed = float(row["approximate_kl"])
    if not np.isfinite(observed) or observed > maximum:
        raise RuntimeError(
            f"{arm} PPO update {row.get('index', 'unknown')} KL {observed:.6f} "
            f"exceeds safety limit {maximum:.6f}"
        )


def _write_checkpoint(path: Path, params: Any, config: Any, metadata: dict[str, Any]):
    save_policy_checkpoint(path, params, config, metadata=metadata)
    loaded, loaded_config, loaded_metadata = load_policy_checkpoint(path)
    expected_metadata = json.loads(json.dumps(metadata))
    if (
        loaded_config != config
        or loaded_metadata != expected_metadata
        or not _trees_equal(params, loaded)
    ):
        raise RuntimeError("WorldGen PPO checkpoint failed its round-trip gate")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "roundtrip_exact": True,
    }


def _write_curve_checkpoints(
    output: Path,
    snapshots: dict[int, dict[str, Any]],
    config: Any,
    checkpoint_base: dict[str, Any],
    *,
    transitions_per_update: int,
) -> dict[str, Any]:
    omitted = (
        "optimizer_state",
        "environment_state",
        "recurrent_state",
        "rng_position",
    )
    rows = []
    for update in sorted(snapshots):
        environment_steps = update * transitions_per_update
        arms = {}
        for arm, training_arm in (
            ("il_ppo", "il_warm_start_ppo"),
            ("ppo_only", "ppo_only"),
        ):
            metadata = {
                **checkpoint_base,
                "checkpoint_role": "evaluation_curve_policy_snapshot",
                "training_arm": training_arm,
                "updates": update,
                "environment_steps": environment_steps,
                "training_state_resumable": False,
                "omitted_training_state": omitted,
            }
            arms[arm] = _write_checkpoint(
                output
                / "curve_checkpoints"
                / f"update-{update:06d}"
                / f"{arm}_policy.npz",
                snapshots[update][arm],
                config,
                metadata,
            )
        rows.append(
            {
                "update": update,
                "environment_steps_per_arm": environment_steps,
                **arms,
            }
        )
    return {
        "schema": CURVE_CHECKPOINT_SCHEMA,
        "policy_only": True,
        "training_state_resumable": False,
        "omitted_training_state": list(omitted),
        "rows": rows,
    }


def _loaded_source_hashes() -> dict[str, str]:
    """Fingerprint loaded project sources so a long run cannot cross an edit."""

    root = Path(__file__).resolve().parents[3]
    prefixes = tuple(
        (root / relative).resolve()
        for relative in ("agents", "arena", "adk", "HytaleRL/hytalegym/hytalegym")
    )
    snapshot = {}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix != ".py" or not any(path.is_relative_to(p) for p in prefixes):
            continue
        snapshot[path.relative_to(root).as_posix()] = (
            hashlib.sha256(path.read_bytes()).hexdigest().upper()
        )
    return dict(sorted(snapshot.items()))


def _source_snapshot_sha256(snapshot: dict[str, str]) -> str:
    return (
        hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def _require_unchanged_sources(
    expected: dict[str, str], actual: dict[str, str] | None = None
) -> None:
    actual = _loaded_source_hashes() if actual is None else actual
    changed = sorted(
        name for name, digest in expected.items() if actual.get(name) != digest
    )
    if changed:
        names = ", ".join(changed[:5])
        suffix = " ..." if len(changed) > 5 else ""
        raise RuntimeError(
            f"loaded runtime source changed while training: {names}{suffix}"
        )


def _trees_equal(left: Any, right: Any) -> bool:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    return left_tree == right_tree and all(
        np.array_equal(np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b)))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_delta(before: Any, after: Any) -> float:
    squared = sum(
        float(jnp.sum(jnp.square(right - left)))
        for left, right in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )
    return float(np.sqrt(squared))


def _scalar(value: Any) -> float | int:
    result = np.asarray(jax.device_get(value))
    return int(result) if result.dtype.kind in "iu" else float(result)


def attach_deployment_smoke(
    report: dict[str, Any],
    *,
    fixture: Path,
    destination: Path,
    checkpoint: Path | None = None,
    profile: Path | None = None,
    java: Path | None = None,
    java_classes: Path | None = None,
    server_jar: Path | None = None,
) -> dict[str, Any]:
    """Continue one training receipt through bundle and Java inference."""

    from adk.deploy.bundle import certify_java, export, verify

    checkpoint = (
        _local_path(report["checkpoints"]["il_ppo"]["path"])
        if checkpoint is None
        else Path(checkpoint)
    )
    manifest = export(
        checkpoint,
        destination,
        role="Kweebec_Razorleaf",
        decision=2,
        fixture=fixture,
        profile=profile,
    )
    problems = verify(destination)
    if problems:
        raise RuntimeError(f"deployment bundle failed verification: {problems}")
    java_inputs = (java, java_classes, server_jar)
    if any(java_inputs) and not all(java_inputs):
        raise ValueError("java, java_classes and server_jar must be passed together")
    certificate = None
    if all(java_inputs):
        certificate = certify_java(
            destination,
            java=java,
            classes=java_classes,
            server_jar=server_jar,
        )
    report["deployment_chain"] = {
        "schema": CHAIN_SCHEMA,
        "stages": {
            "jax_trace_capture": {"passed": report["trace"]["rows"] > 0},
            "behavior_cloning": {"passed": report["imitation"]["steps"] > 0},
            "ppo": {"passed": report["learned"]},
            "bundle": {
                "passed": True,
                "path": str(destination.resolve()),
                "manifest": manifest,
            },
            "java_inference": {
                "passed": certificate is not None,
                "status": "passed" if certificate is not None else "not_run",
                "certificate": certificate,
            },
            "live_server_execution": {
                "passed": False,
                "status": "not_run",
                "reason": "requires an explicitly authorized isolated deployment",
            },
        },
        "complete_through": "java_inference" if certificate else "bundle",
        "competence_claim": False,
    }
    _local_path(report["report_path"]).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def attach_live_execution(report: dict[str, Any], receipt: Path) -> dict[str, Any]:
    """Pin a passing isolated-server receipt to the same exported bundle."""

    receipt = Path(receipt).resolve()
    live = json.loads(receipt.read_text(encoding="utf-8"))
    chain = report.get("deployment_chain")
    if not isinstance(chain, dict):
        raise ValueError("deployment smoke must precede live execution")
    bundle = chain["stages"]["bundle"]
    expected = Path(bundle["path"]).resolve()
    actual = Path(live["inputs"]["bundle"]["path"]).resolve()
    if actual != expected:
        raise ValueError("live receipt executed a different policy bundle")
    if not (
        live.get("schema") == LIVE_EXECUTION_SCHEMA
        and live.get("passed") is True
        and live.get("status") == "passed"
        and live.get("isolated") is True
        and live.get("shared_server_modified") is False
        and live.get("exit_code") == 0
    ):
        raise ValueError("live receipt is not a passing isolated execution")

    metrics = live["evidence"]["metrics"]
    contract = bundle["manifest"]["contract"]
    widths = {
        "observation_size": "observation_size",
        "action_mask_size": "action_size",
        "encoder_size": "encoder_size",
        "recurrent_size": "recurrent_size",
    }
    if any(
        int(metrics[target]) != int(contract[source])
        for source, target in widths.items()
    ):
        raise ValueError("live metrics do not match the exported policy contract")
    if not (
        int(metrics.get("schema_version", 0)) >= 3
        and int(metrics.get("policy_ticks", 0)) > 0
        and int(metrics.get("locomotion_requests", 0)) > 0
        and bool(metrics.get("moved"))
        and bool(metrics.get("last_action_legal"))
        and int(metrics.get("rejected_illegal", 0)) == 0
        and int(live["evidence"].get("lifecycle_rows", 0)) > 0
    ):
        raise ValueError("live receipt lacks non-vacuous policy behavior")

    chain["stages"]["live_server_execution"] = {
        "passed": True,
        "status": "passed",
        "receipt": {
            "path": str(receipt),
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest().upper(),
        },
        "policy_ticks": int(metrics["policy_ticks"]),
        "locomotion_requests": int(metrics["locomotion_requests"]),
        "horizontal_path": float(metrics["motion_horizontal_path"]),
        "world_tps": float(metrics["world_tps"]),
        "policy_tps": float(metrics["policy_ticks_per_second"]),
        "throughput_available": bool(
            float(metrics["world_tps"]) > 0.0
            and float(metrics["policy_ticks_per_second"]) > 0.0
        ),
        "lifecycle_rows": int(live["evidence"]["lifecycle_rows"]),
    }
    chain["complete_through"] = "live_server_execution"
    chain["competence_claim"] = False
    _local_path(report["report_path"]).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _local_path(value: str | Path) -> Path:
    """Resolve a shared WSL `/mnt/<drive>` path on a Windows caller."""

    text = str(value)
    if os.name == "nt" and len(text) > 6 and text.startswith("/mnt/"):
        drive, tail = text[5], text[6:]
        if drive.isalpha() and tail.startswith("/"):
            return Path(f"{drive.upper()}:{tail}")
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agents/ppo/artifacts/worldgen-il-run"),
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--trace-steps", type=int, default=4)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--ppo-update-epochs", type=int, default=2)
    parser.add_argument("--ppo-num-minibatches", type=int, default=1)
    parser.add_argument("--cloning-steps", type=int, default=32)
    parser.add_argument("--encoder-size", type=int, default=8)
    parser.add_argument("--recurrent-size", type=int, default=8)
    parser.add_argument("--combat-fundamentals-reward", action="store_true")
    parser.add_argument("--native-behavior-initialization-only", action="store_true")
    parser.add_argument("--deployment-fixture", type=Path)
    parser.add_argument("--deployment-profile", type=Path)
    parser.add_argument("--java", type=Path)
    parser.add_argument("--java-classes", type=Path)
    parser.add_argument("--server-jar", type=Path)
    arguments = parser.parse_args()
    report = WorldgenILPPOAgent(
        WorldgenILPPOSettings(
            output_dir=arguments.output_dir,
            batch=arguments.batch,
            trace_steps=arguments.trace_steps,
            ppo_updates=arguments.ppo_updates,
            rollout_steps=arguments.rollout_steps,
            ppo_update_epochs=arguments.ppo_update_epochs,
            ppo_num_minibatches=arguments.ppo_num_minibatches,
            cloning_steps=arguments.cloning_steps,
            encoder_size=arguments.encoder_size,
            recurrent_size=arguments.recurrent_size,
            combat_fundamentals=(
                CombatFundamentalsConfig()
                if arguments.combat_fundamentals_reward
                else None
            ),
            native_behavior_online_prior=(
                not arguments.native_behavior_initialization_only
            ),
        )
    ).run()
    if arguments.deployment_fixture is not None:
        report = attach_deployment_smoke(
            report,
            fixture=arguments.deployment_fixture,
            destination=arguments.output_dir / "deployment" / "policy-agent",
            profile=arguments.deployment_profile,
            java=arguments.java,
            java_classes=arguments.java_classes,
            server_jar=arguments.server_jar,
        )
    elif any(
        (
            arguments.deployment_profile,
            arguments.java,
            arguments.java_classes,
            arguments.server_jar,
        )
    ):
        parser.error("--deployment-fixture is required for a deployment smoke")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CHAIN_SCHEMA",
    "WorldgenILPPOAgent",
    "WorldgenILPPOSettings",
    "attach_deployment_smoke",
    "attach_live_execution",
]
