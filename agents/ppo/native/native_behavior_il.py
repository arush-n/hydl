"""Train recurrent PPO to imitate death-complete native behavior on JAX replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

# Imported for its import-time side effect: configure_training_runtime()
# must run before JAX is imported. Not unused -- do not delete.
from agents import training_runtime as _training_runtime  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from arena.jax_contract import HEAD_SPANS
from arena.training import (
    BehaviorCloningConfig,
    ExactImitationConfig,
    NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
    NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
    NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
    NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE,
    NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
    NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
    NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256,
    NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE,
    behavior_cloning_head_accuracy,
    behavior_cloning_loss,
    make_behavior_cloning_trainer,
    make_native_replay_environment,
    native_arsenal_conditioned_observation_sha256,
    native_arsenal_visual_conditioned_observation_sha256,
    native_replay_ppo_config,
    native_replay_action_head_coverage,
    native_replay_sequence,
    pack_native_replay_sequence,
    warm_start_ppo,
)
from arena.training.imitation.native_replay_cache import compile_or_load_native_replay
from hytalegym.jax.training.checkpoint import save_policy_checkpoint
from hytalegym.jax.training.policy import apply_policy_sequence
from hytalegym.jax.training.ppo import initialize_training, make_train_step


REPORT_SCHEMA = "hytalerl_native_behavior_il_ppo_v17"


def _device_memory_report() -> dict[str, int]:
    memory = dict(jax.devices()[0].memory_stats() or {})
    return {
        key: int(memory[key])
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if key in memory
    }


@dataclass(frozen=True, slots=True)
class Settings:
    seed: int = 570251
    num_envs: int = 1024
    cloning_steps: int = 128
    cloning_sequence_lanes: int = 32
    ppo_updates: int = 64
    rollout_steps: int = 64
    update_epochs: int = 2
    num_minibatches: int = 8
    encoder_size: int = 64
    recurrent_size: int = 64
    cloning_learning_rate: float = 3.0e-4
    ppo_learning_rate: float = 3.0e-4
    demonstration_replay_steps: int = 1
    natural_frame_weight: float = 0.5
    observation_view: str = "arsenal_visual_conditioned"
    action_mismatch_scale: float = 1.0
    aim_error_scale: float = 1.0
    excess_spin_scale: float = 1.0
    spin_tolerance_degrees: float = 1.0
    replay_cache: str = "~/.cache/hytalerl/native-replay"

    def __post_init__(self) -> None:
        positive = (
            "num_envs",
            "cloning_steps",
            "cloning_sequence_lanes",
            "ppo_updates",
            "rollout_steps",
            "update_epochs",
            "num_minibatches",
            "encoder_size",
            "recurrent_size",
        )
        if any(
            isinstance(getattr(self, name), bool) or getattr(self, name) < 1
            for name in positive
        ):
            raise ValueError("training sizes must be positive integers")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0
            for name in ("cloning_learning_rate", "ppo_learning_rate")
        ):
            raise ValueError("learning rates must be finite and positive")
        if (
            isinstance(self.demonstration_replay_steps, bool)
            or self.demonstration_replay_steps < 0
        ):
            raise ValueError("demonstration_replay_steps must be nonnegative")
        if not math.isfinite(self.natural_frame_weight) or not (
            0.0 <= self.natural_frame_weight <= 1.0
        ):
            raise ValueError("natural_frame_weight must be in [0, 1]")
        if self.observation_view not in {
            "omniscient",
            "arsenal_shared",
            "arsenal_conditioned",
            "arsenal_visual_shared",
            "arsenal_visual_conditioned",
        }:
            raise ValueError(
                "observation_view must be omniscient, arsenal_shared, or "
                "arsenal_conditioned, arsenal_visual_shared, or "
                "arsenal_visual_conditioned"
            )
        ExactImitationConfig(
            action_mismatch_scale=self.action_mismatch_scale,
            aim_error_scale=self.aim_error_scale,
            excess_spin_scale=self.excess_spin_scale,
            spin_tolerance_degrees=self.spin_tolerance_degrees,
        )
        if not self.replay_cache:
            raise ValueError("replay_cache must be a nonempty path")


_DEFAULTS = Settings()


def train(corpus_path: Path, output: Path, settings: Settings) -> dict:
    """Warm-start with actor-balanced BC, then refine exact replay reward."""

    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    compiled = compile_or_load_native_replay(
        corpus_path, settings.replay_cache, settings.observation_view
    )
    training_corpus, heldout_corpus = compiled.training, compiled.heldout
    compiled_manifest = compiled.manifest
    projected = compiled_manifest["projected"]
    head_readiness = compiled_manifest["action_head_runtime_readiness"]
    action_coverage = compiled_manifest["action_head_coverage"]
    imitation = ExactImitationConfig(
        action_mismatch_scale=settings.action_mismatch_scale,
        aim_error_scale=settings.aim_error_scale,
        excess_spin_scale=settings.excess_spin_scale,
        spin_tolerance_degrees=settings.spin_tolerance_degrees,
    )
    environment = make_native_replay_environment(training_corpus, imitation)
    ppo = native_replay_ppo_config(
        training_corpus,
        num_envs=settings.num_envs,
        rollout_steps=settings.rollout_steps,
        update_epochs=settings.update_epochs,
        num_minibatches=settings.num_minibatches,
        encoder_size=settings.encoder_size,
        recurrent_size=settings.recurrent_size,
        learning_rate=settings.ppo_learning_rate,
        gamma=0.0,
        gae_lambda=0.0,
        entropy_coefficient=0.0,
    )
    cloning = BehaviorCloningConfig(
        learning_rate=settings.cloning_learning_rate,
        label_smoothing=0.02,
    )
    sequence_started = perf_counter()
    demonstrations = native_replay_sequence(training_corpus, cloning, device=False)
    heldout = native_replay_sequence(heldout_corpus, cloning, device=False)
    sequence_seconds = perf_counter() - sequence_started
    objective = _training_objective(demonstrations, settings.natural_frame_weight)
    heldout_objective = _training_objective(
        heldout, settings.natural_frame_weight
    )
    packing_started = perf_counter()
    packed_objective = pack_native_replay_sequence(
        objective,
        settings.cloning_sequence_lanes,
    )
    packed_heldout = pack_native_replay_sequence(
        heldout_objective,
        min(8, settings.cloning_sequence_lanes),
    )
    packing_seconds = perf_counter() - packing_started
    training_evaluation, training_roles = _packed_evaluation_sequence(
        packed_objective, objective, training_corpus
    )
    heldout_evaluation, heldout_roles = _packed_evaluation_sequence(
        packed_heldout, heldout_objective, heldout_corpus
    )
    keys = jax.random.split(jax.random.key(settings.seed), settings.ppo_updates + 2)
    state = initialize_training(keys[0], ppo, environment=environment)
    initial_policy_params = state.policy_params

    def apply(params, observation, action_mask):
        carry = jnp.zeros(
            (observation.shape[1], settings.recurrent_size), dtype=jnp.float32
        )
        _, logits, _ = apply_policy_sequence(
            params,
            observation,
            carry,
            packed_objective.episode_start,
            action_mask,
        )
        return logits

    trainer = make_behavior_cloning_trainer(apply, cloning)
    optimizer = trainer.initialize(state.policy_params)
    before = _evaluate(
        state.policy_params,
        training_evaluation,
        training_corpus,
        cloning,
        settings.recurrent_size,
        training_roles,
    )
    before_heldout = _evaluate(
        state.policy_params,
        heldout_evaluation,
        heldout_corpus,
        cloning,
        settings.recurrent_size,
        heldout_roles,
    )
    cloning_started = perf_counter()
    cloning_metrics = None
    for _ in range(settings.cloning_steps):
        state_params, optimizer, cloning_metrics = trainer.step(
            state.policy_params, optimizer, packed_objective.batch
        )
        state = state._replace(policy_params=state_params)
    jax.block_until_ready(cloning_metrics.loss.loss)
    cloning_seconds = perf_counter() - cloning_started
    after_cloning = _evaluate(
        state.policy_params,
        training_evaluation,
        training_corpus,
        cloning,
        settings.recurrent_size,
        training_roles,
    )
    after_cloning_heldout = _evaluate(
        state.policy_params,
        heldout_evaluation,
        heldout_corpus,
        cloning,
        settings.recurrent_size,
        heldout_roles,
    )
    visual_learning = _visual_learning_report(
        compiled_manifest["visual_context"],
        heldout_evaluation,
        heldout_corpus,
        initial_policy_params,
        state.policy_params,
        cloning,
        settings.recurrent_size,
        training_corpus.observation_schema,
        heldout_roles,
    )
    state = warm_start_ppo(state, state.policy_params, ppo)
    bc_checkpoint = _save(
        output / "after_bc.npz",
        state.policy_params,
        ppo,
        training_corpus,
        "after_bc",
        0,
        tensor_sha256=projected["tensor_sha256"],
        coverage=action_coverage,
    )

    step = make_train_step(ppo, environment=environment)
    update_rows = []
    replay_steps = 0
    replay_seconds = 0.0
    ppo_started = perf_counter()
    for update, key in enumerate(keys[2:], 1):
        tick = perf_counter()
        state, metrics = step(state, key)
        jax.block_until_ready(metrics.total_loss)
        replay_started = perf_counter()
        replay_metrics = None
        for _ in range(settings.demonstration_replay_steps):
            params, optimizer, replay_metrics = trainer.step(
                state.policy_params, optimizer, packed_objective.batch
            )
            state = state._replace(policy_params=params)
            replay_steps += 1
        if replay_metrics is not None:
            jax.block_until_ready(replay_metrics.loss.loss)
        replay_seconds += perf_counter() - replay_started
        update_rows.append(
            {
                "update": update,
                "environment_steps": int(metrics.total_environment_steps),
                "episodes_completed": int(metrics.episodes_completed),
                "mean_reward": float(metrics.mean_rollout_reward),
                "loss": float(metrics.total_loss),
                "policy_loss": float(metrics.policy_loss),
                "value_loss": float(metrics.value_loss),
                "entropy": float(metrics.entropy),
                "approximate_kl": float(metrics.approximate_kl),
                "wall_seconds": perf_counter() - tick,
            }
        )
    ppo_seconds = perf_counter() - ppo_started
    after_ppo = _evaluate(
        state.policy_params,
        training_evaluation,
        training_corpus,
        cloning,
        settings.recurrent_size,
        training_roles,
    )
    after_ppo_heldout = _evaluate(
        state.policy_params,
        heldout_evaluation,
        heldout_corpus,
        cloning,
        settings.recurrent_size,
        heldout_roles,
    )
    checkpoint = _save(
        output / "after_ppo.npz",
        state.policy_params,
        ppo,
        training_corpus,
        "after_ppo",
        settings.ppo_updates,
        tensor_sha256=projected["tensor_sha256"],
        coverage=action_coverage,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed_jax_replay_training_not_live_deployment",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_memory": _device_memory_report(),
        "settings": asdict(settings),
        "corpus": {
            "path": corpus_path.as_posix(),
            "source_report_sha256": compiled_manifest["source"][
                "source_report_sha256"
            ],
            "source_tensor_sha256": compiled_manifest["source"]["tensor_sha256"],
            "tensor_sha256": projected["tensor_sha256"],
            "observation_schema": projected["observation_schema"],
            "observation_size": projected["observation_size"],
            "native_actor_episodes": projected["episodes"],
            "native_rows": projected["rows"],
            "roles": projected["roles"],
            "outcome_use": "audit only; no win/loss observation or reward",
            "native_episode_weighting": "uniform across both actors and outcomes",
            "action_head_coverage": action_coverage,
            "action_head_runtime_readiness": head_readiness,
            "split": {
                "method": "latest_seed_per_directed_role_world_group",
                "training_episodes": training_corpus.episodes,
                "heldout_episodes": heldout_corpus.episodes,
                "training_ids": list(training_corpus.episode_ids),
                "heldout_ids": list(heldout_corpus.episode_ids),
            },
        },
        "data_compilation": {
            **compiled.report(),
            "sequence_seconds": sequence_seconds,
            "packing_seconds": packing_seconds,
            "host_arrays": "read_only_mmap_projected_splits",
            "accelerator_transfer": "packed_recurrent_objective_plus_JAX_environment",
        },
        "behavior_cloning": {
            "checkpoint": bc_checkpoint,
            "before": before,
            "before_heldout": before_heldout,
            "after": after_cloning,
            "after_heldout": after_cloning_heldout,
            "steps": settings.cloning_steps,
            "wall_seconds": cloning_seconds,
            "sequence_packing": {
                "training": packed_objective.report(),
                "heldout": packed_heldout.report(),
            },
            "objective_weighting": {
                "schema": "arena_native_behavior_objective_v2",
                "episode_and_head_balanced": True,
                "decision_opportunities_only": True,
                "forced_lifecycle_choices": "zero_weight",
                "natural_frame_fraction": settings.natural_frame_weight,
                "present_class_balanced_fraction": 1.0 - settings.natural_frame_weight,
            },
            "visual_learning": visual_learning,
        },
        "ppo": {
            "updates": update_rows,
            "environment_steps": int(state.total_environment_steps),
            "replayed_episodes": sum(row["episodes_completed"] for row in update_rows),
            "after": after_ppo,
            "after_heldout": after_ppo_heldout,
            "wall_seconds": ppo_seconds,
            "transitions_per_second": int(state.total_environment_steps) / ppo_seconds,
            "reward": "negative aligned action/aim/excess-spin error only",
            "reward_contract": {
                "schema": "arena_exact_imitation_reward_v1",
                **asdict(imitation),
                "unknown_heads": "abstain",
                "alignment": "exact_native_replay_row",
            },
            "scope": "recorded-context replay; not novel native world dynamics",
            "demonstration_replay": {
                "order": "after_ppo_before_next_rollout",
                "steps_per_update": settings.demonstration_replay_steps,
                "applied_steps": replay_steps,
                "wall_seconds": replay_seconds,
                "actor_episode_weighting": "uniform; independent of outcome",
            },
        },
        "checkpoint": checkpoint,
        "wall_seconds": perf_counter() - started,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _packed_evaluation_sequence(packed, source, corpus):
    episode = np.asarray(packed.source_episode, dtype=np.int32)
    present = episode >= 0
    safe_episode = np.maximum(episode, 0)
    outcome = np.where(present, np.asarray(source.outcome)[safe_episode], 0)
    role = np.where(
        present,
        np.asarray(corpus.tensors.role, dtype=np.int32)[safe_episode],
        0,
    )
    return (
        source._replace(
            batch=packed.batch,
            episode_start=packed.episode_start,
            outcome=jnp.asarray(outcome),
            frame_weight=packed.frame_weight,
        ),
        jnp.asarray(role),
    )


def _evaluate(params, demonstrations, corpus, config, recurrent_size, frame_role):
    observation = demonstrations.batch.observation
    carry = jnp.zeros((observation.shape[1], recurrent_size), dtype=jnp.float32)
    _, logits, _ = apply_policy_sequence(
        params,
        observation,
        carry,
        demonstrations.episode_start,
        demonstrations.batch.action_mask,
    )
    loss = behavior_cloning_loss(logits, demonstrations.batch, config)
    accuracy, support = behavior_cloning_head_accuracy(
        logits, demonstrations.batch, config
    )
    jax.block_until_ready(loss.loss)
    behavior = _behavior_metrics(logits, demonstrations)
    return {
        "loss": float(loss.loss),
        "negative_log_likelihood": float(loss.negative_log_likelihood),
        "class_balanced_accuracy": float(loss.accuracy),
        "supervision_coverage": _supervision_coverage(demonstrations),
        "heads": {
            name: {
                "accuracy": float(accuracy[index]),
                "support": float(support[index]),
                **behavior["heads"][name],
            }
            for index, name in enumerate(HEAD_SPANS)
            if float(support[index]) > 0.0
        },
        "exact_frame_accuracy": behavior["exact_frame_accuracy"],
        "episode_balanced_exact_frame_accuracy": behavior[
            "episode_balanced_exact_frame_accuracy"
        ],
        "unnecessary_spin_rate": behavior["unnecessary_spin_rate"],
        "episode_balanced_unnecessary_spin_rate": behavior[
            "episode_balanced_unnecessary_spin_rate"
        ],
        "head_accuracy_weighting": "per_actor_per_present_class",
        "winner": _outcome_accuracy(logits, demonstrations, 1, config),
        "loser": _outcome_accuracy(logits, demonstrations, -1, config),
        "roles": {
            role: _selected_accuracy(
                logits,
                demonstrations,
                frame_role == index,
                config,
            )
            for index, role in enumerate(corpus.roles)
        },
    }


def _visual_learning_report(
    source_visual,
    heldout,
    heldout_corpus,
    initial_params,
    trained_params,
    config,
    recurrent_size,
    observation_schema,
    frame_role,
):
    visual_schemas = {
        NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
        NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
    }
    if observation_schema not in visual_schemas:
        return {"enabled": False}
    if source_visual.get("enabled") is not True:
        raise ValueError("visual replay view has no compiled visual evidence")
    ablated = heldout._replace(
        batch=heldout.batch._replace(
            observation=jnp.asarray(heldout.batch.observation)
            .at[
                ...,
                NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE:(
                    NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE
                ),
            ]
            .set(0.0)
        )
    )
    without_visual = _evaluate(
        trained_params,
        ablated,
        heldout_corpus,
        config,
        recurrent_size,
        frame_role,
    )
    before_kernel = np.asarray(initial_params.encoder_input.kernel)
    after_kernel = np.asarray(trained_params.encoder_input.kernel)

    def branch(start, stop):
        before = before_kernel[start:stop]
        after = after_kernel[start:stop]
        return {
            "features": stop - start,
            "initial_l2": float(np.linalg.norm(before)),
            "final_l2": float(np.linalg.norm(after)),
            "delta_l2": float(np.linalg.norm(after - before)),
        }

    return {
        "enabled": True,
        "available_frames": source_visual["available_frames"],
        "valid_frames": source_visual["valid_frames"],
        "available_fraction": source_visual["available_fraction"],
        "mean_visible_tokens": source_visual["mean_visible_tokens"],
        "encoder_input_branches": {
            "shared_physical": branch(0, NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE),
            "shared_visual": branch(
                NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE,
                NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE,
            ),
            "role_conditioning": branch(
                NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE,
                before_kernel.shape[0],
            ),
        },
        "heldout_ablation": {
            "method": "zero_only_shared_visual_input_columns",
            "loss_without_visual": without_visual["loss"],
            "negative_log_likelihood_without_visual": without_visual[
                "negative_log_likelihood"
            ],
        },
    }


def _training_objective(demonstrations, natural_frame_weight):
    """Blend literal and class-balanced weights over genuine decisions only."""

    action = np.asarray(demonstrations.batch.action, dtype=np.int32)
    known = np.asarray(demonstrations.batch.supervision_mask, dtype=np.bool_)
    frame = np.asarray(demonstrations.frame_weight, dtype=np.float32)
    mask = np.asarray(demonstrations.batch.action_mask, dtype=np.bool_)
    if action.ndim != 3 or frame.shape != action.shape[:2]:
        raise ValueError("native objective requires [time, episode, head] rows")

    natural = np.zeros_like(known, dtype=np.float32)
    balanced = np.zeros_like(natural)
    offset = 0
    for head, (_, size) in enumerate(HEAD_SPANS.values()):
        decision = (
            known[..., head]
            & (frame > 0.0)
            & (mask[..., offset : offset + size].sum(axis=-1) > 1)
        )
        for episode in range(action.shape[1]):
            rows = decision[:, episode]
            if not np.any(rows):
                continue
            values = frame[rows, episode]
            natural[rows, episode, head] = values / values.sum()
            labels = np.unique(action[rows, episode, head])
            for label in labels:
                selected = rows & (action[:, episode, head] == label)
                values = frame[selected, episode]
                balanced[selected, episode, head] = (
                    values / values.sum() / labels.size
                )
        offset += size

    for weights in (natural, balanced):
        total = weights.sum(axis=(0, 1), keepdims=True)
        np.divide(weights, total, out=weights, where=total > 0.0)
    weight = (
        np.float32(1.0 - natural_frame_weight) * balanced
        + np.float32(natural_frame_weight) * natural
    )
    return demonstrations._replace(batch=demonstrations.batch._replace(weight=weight))


def _supervision_coverage(demonstrations):
    known = jnp.asarray(demonstrations.batch.supervision_mask, dtype=jnp.bool_)
    valid_frame = jnp.asarray(demonstrations.frame_weight) > 0.0
    known &= valid_frame[..., None]
    valid_pairs = jnp.sum(valid_frame) * known.shape[-1]
    supervised_heads = jnp.any(known, axis=tuple(range(known.ndim - 1)))
    return {
        "supervised_pairs": int(jnp.sum(known)),
        "padded_pair_capacity": int(known.size),
        "valid_frame_pair_capacity": int(valid_pairs),
        "fraction_of_padded_pairs": float(jnp.mean(known)),
        "fraction_of_valid_frame_pairs": float(
            jnp.sum(known) / jnp.maximum(valid_pairs, 1)
        ),
        "supervised_heads": int(jnp.sum(supervised_heads)),
        "head_capacity": int(known.shape[-1]),
    }


def _behavior_metrics(logits, demonstrations):
    action = demonstrations.batch.action
    known = demonstrations.batch.supervision_mask
    frame_weight = demonstrations.frame_weight
    correct = jnp.zeros_like(known)
    heads = {}
    offset = 0
    neutral = {
        "locomotion_gait_compass": 0,
        "yaw_delta_bins": HEAD_SPANS["yaw_delta_bins"][1] // 2,
        "pitch_delta_bins": HEAD_SPANS["pitch_delta_bins"][1] // 2,
    }
    for index, (name, (_, size)) in enumerate(HEAD_SPANS.items()):
        head_logits = logits[..., offset : offset + size]
        probability = jax.nn.softmax(head_logits, axis=-1)
        choice = jnp.argmax(head_logits, axis=-1)
        head_known = known[..., index]
        head_correct = choice == action[..., index]
        correct = correct.at[..., index].set(head_correct)
        active = head_known & (action[..., index] != neutral.get(name, 0))
        active_weight = jnp.where(active, frame_weight, 0.0)
        active_support = jnp.sum(active_weight)
        active_count = jnp.sum(active)
        natural_weight = jnp.where(head_known, frame_weight, 0.0)
        natural_support = jnp.sum(natural_weight)
        frame_count = jnp.sum(head_known)
        label_weight = jnp.sum(
            natural_weight[..., None]
            * jax.nn.one_hot(action[..., index], size, dtype=jnp.float32),
            axis=tuple(range(natural_weight.ndim)),
        )
        expert_probability = jnp.take_along_axis(
            probability, action[..., index, None], axis=-1
        )[..., 0]
        inactive_weight = jnp.where(head_known & ~active, frame_weight, 0.0)
        inactive_support = jnp.sum(inactive_weight)
        inactive = head_known & ~active
        inactive_count = jnp.sum(inactive)
        head_action_mask = jnp.asarray(
            demonstrations.batch.action_mask[..., offset : offset + size]
        )
        inactive_opportunity = inactive & jnp.any(
            head_action_mask.at[..., neutral.get(name, 0)].set(False),
            axis=-1,
        )
        inactive_opportunity_count = jnp.sum(inactive_opportunity)
        true_positive = jnp.sum(active & head_correct)
        false_positive = jnp.sum(inactive & (choice != neutral.get(name, 0)))
        predicted_active = jnp.sum(head_known & (choice != neutral.get(name, 0)))
        legal = jnp.take_along_axis(
            head_action_mask,
            choice[..., None],
            axis=-1,
        )[..., 0]
        heads[name] = {
            "frame_accuracy": float(
                jnp.sum(head_known & head_correct) / jnp.maximum(frame_count, 1)
            ),
            "frame_count": int(frame_count),
            "episode_balanced_frame_accuracy": float(
                jnp.sum(natural_weight * head_correct)
                / jnp.maximum(natural_support, 1.0)
            ),
            "episode_balanced_majority_accuracy": float(
                jnp.max(label_weight) / jnp.maximum(natural_support, 1.0)
            ),
            "active_event_accuracy": float(
                jnp.sum(active & head_correct) / jnp.maximum(active_count, 1)
            ),
            "active_event_count": int(active_count),
            "episode_balanced_active_accuracy": float(
                jnp.sum(active_weight * head_correct) / jnp.maximum(active_support, 1.0)
            ),
            "episode_balanced_active_support": float(active_support),
            "episode_balanced_expert_softmax_score": float(
                jnp.sum(natural_weight * expert_probability)
                / jnp.maximum(natural_support, 1.0)
            ),
            "episode_balanced_active_expert_softmax_score": float(
                jnp.sum(active_weight * expert_probability)
                / jnp.maximum(active_support, 1.0)
            ),
            "episode_balanced_inactive_non_neutral_softmax_score": float(
                jnp.sum(
                    inactive_weight * (1.0 - probability[..., neutral.get(name, 0)])
                )
                / jnp.maximum(inactive_support, 1.0)
            ),
            "inactive_false_positive_rate": float(
                jnp.sum(inactive & (choice != neutral.get(name, 0)))
                / jnp.maximum(inactive_count, 1)
            ),
            "inactive_frame_count": int(inactive_count),
            "inactive_decision_opportunity_count": int(inactive_opportunity_count),
            "inactive_decision_false_positive_rate": float(
                jnp.sum(inactive_opportunity & (choice != neutral.get(name, 0)))
                / jnp.maximum(inactive_opportunity_count, 1)
            ),
            "true_positive_count": int(true_positive),
            "false_positive_count": int(false_positive),
            "predicted_active_count": int(predicted_active),
            "active_precision": float(true_positive / jnp.maximum(predicted_active, 1)),
            "false_positive_to_true_positive_ratio": float(
                false_positive / jnp.maximum(true_positive, 1)
            ),
            "illegal_choice_count": int(jnp.sum(head_known & ~legal)),
            "episode_balanced_inactive_false_positive_rate": float(
                jnp.sum(inactive_weight * (choice != neutral.get(name, 0)))
                / jnp.maximum(inactive_support, 1.0)
            ),
        }
        offset += size
    row_known = jnp.any(known, axis=-1)
    exact = jnp.all(~known | correct, axis=-1)
    row_weight = jnp.where(row_known, frame_weight, 0.0)
    yaw = tuple(HEAD_SPANS).index("yaw_delta_bins")
    yaw_neutral = neutral["yaw_delta_bins"]
    yaw_choice = jnp.argmax(
        logits[
            ...,
            HEAD_SPANS["yaw_delta_bins"][0] : sum(HEAD_SPANS["yaw_delta_bins"]),
        ],
        axis=-1,
    )
    no_turn = known[..., yaw] & (action[..., yaw] == yaw_neutral)
    no_turn_weight = jnp.where(no_turn, frame_weight, 0.0)
    return {
        "heads": heads,
        "exact_frame_accuracy": float(
            jnp.sum(row_known & exact) / jnp.maximum(jnp.sum(row_known), 1)
        ),
        "episode_balanced_exact_frame_accuracy": float(
            jnp.sum(row_weight * exact) / jnp.maximum(jnp.sum(row_weight), 1.0)
        ),
        "unnecessary_spin_rate": float(
            jnp.sum(no_turn & (yaw_choice != yaw_neutral))
            / jnp.maximum(jnp.sum(no_turn), 1)
        ),
        "episode_balanced_unnecessary_spin_rate": float(
            jnp.sum(no_turn_weight * (yaw_choice != yaw_neutral))
            / jnp.maximum(jnp.sum(no_turn_weight), 1.0)
        ),
    }


def _outcome_accuracy(logits, demonstrations, outcome, config):
    return _selected_accuracy(
        logits, demonstrations, demonstrations.outcome == outcome, config
    )


def _selected_accuracy(logits, demonstrations, selected, config):
    selected = jnp.asarray(selected)
    while selected.ndim < demonstrations.batch.action.ndim - 1:
        selected = selected[None]
    if demonstrations.batch.weight.ndim == demonstrations.batch.action.ndim:
        selected = selected[..., None]
    batch = demonstrations.batch._replace(
        weight=jnp.where(selected, demonstrations.batch.weight, 0.0)
    )
    accuracy, support = behavior_cloning_head_accuracy(logits, batch, config)
    return {
        name: {"accuracy": float(accuracy[index]), "support": float(support[index])}
        for index, name in enumerate(HEAD_SPANS)
        if float(support[index]) > 0.0
    }


def _save(
    path,
    params,
    config,
    corpus,
    stage,
    updates,
    *,
    tensor_sha256=None,
    coverage=None,
):
    coverage = coverage or native_replay_action_head_coverage(corpus)
    save_policy_checkpoint(
        path,
        params,
        config,
        metadata={
            "schema": REPORT_SCHEMA,
            "stage": stage,
            "ppo_updates": updates,
            "native_replay_tensor_sha256": tensor_sha256 or corpus.tensor_sha256,
            "native_replay_observation_schema": corpus.observation_schema,
            "native_arsenal_shared_observation_sha256": (
                NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256
                if corpus.observation_schema
                in {
                    NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
                    NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
                    NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
                    NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
                }
                else None
            ),
            "native_arsenal_conditioned_observation_sha256": (
                native_arsenal_conditioned_observation_sha256(corpus.roles)
                if corpus.observation_schema
                == NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA
                else None
            ),
            "native_arsenal_visual_shared_observation_sha256": (
                NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256
                if corpus.observation_schema
                in {
                    NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
                    NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
                }
                else None
            ),
            "native_arsenal_visual_conditioned_observation_sha256": (
                native_arsenal_visual_conditioned_observation_sha256(corpus.roles)
                if corpus.observation_schema
                == NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA
                else None
            ),
            "native_behavior_style_roles": (
                list(corpus.roles)
                if corpus.observation_schema
                in {
                    NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
                    NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
                }
                else None
            ),
            "native_replay_action_schema": corpus.action_schema,
            "native_replay_ability_slot_contract_sha256": (
                corpus.ability_slot_contract_sha256 or None
            ),
            "native_replay_supervised_heads": [
                name for name in HEAD_SPANS if coverage[name]["labelled_rows"] > 0
            ],
            "deployable_to_arsenal": False,
            "consumer": "arena.training.load_native_behavior_policy",
            "scope": "native_replay_context_behavior_prior",
        },
    )
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "stage": stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--num-envs", type=int, default=_DEFAULTS.num_envs)
    parser.add_argument("--cloning-steps", type=int, default=_DEFAULTS.cloning_steps)
    parser.add_argument(
        "--cloning-sequence-lanes",
        type=int,
        default=_DEFAULTS.cloning_sequence_lanes,
    )
    parser.add_argument("--ppo-updates", type=int, default=_DEFAULTS.ppo_updates)
    parser.add_argument("--rollout-steps", type=int, default=_DEFAULTS.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=_DEFAULTS.update_epochs)
    parser.add_argument(
        "--num-minibatches", type=int, default=_DEFAULTS.num_minibatches
    )
    parser.add_argument("--encoder-size", type=int, default=_DEFAULTS.encoder_size)
    parser.add_argument("--recurrent-size", type=int, default=_DEFAULTS.recurrent_size)
    parser.add_argument(
        "--cloning-learning-rate",
        type=float,
        default=_DEFAULTS.cloning_learning_rate,
    )
    parser.add_argument(
        "--ppo-learning-rate", type=float, default=_DEFAULTS.ppo_learning_rate
    )
    parser.add_argument(
        "--demonstration-replay-steps",
        type=int,
        default=_DEFAULTS.demonstration_replay_steps,
    )
    parser.add_argument(
        "--natural-frame-weight",
        type=float,
        default=_DEFAULTS.natural_frame_weight,
    )
    parser.add_argument(
        "--observation-view",
        choices=(
            "omniscient",
            "arsenal_shared",
            "arsenal_conditioned",
            "arsenal_visual_shared",
            "arsenal_visual_conditioned",
        ),
        default=_DEFAULTS.observation_view,
    )
    parser.add_argument("--replay-cache", default=_DEFAULTS.replay_cache)
    for name in (
        "action_mismatch_scale",
        "aim_error_scale",
        "excess_spin_scale",
        "spin_tolerance_degrees",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=getattr(_DEFAULTS, name),
        )
    args = parser.parse_args()
    report = train(
        args.corpus,
        args.output,
        Settings(
            seed=args.seed,
            num_envs=args.num_envs,
            cloning_steps=args.cloning_steps,
            cloning_sequence_lanes=args.cloning_sequence_lanes,
            ppo_updates=args.ppo_updates,
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            num_minibatches=args.num_minibatches,
            encoder_size=args.encoder_size,
            recurrent_size=args.recurrent_size,
            cloning_learning_rate=args.cloning_learning_rate,
            ppo_learning_rate=args.ppo_learning_rate,
            demonstration_replay_steps=args.demonstration_replay_steps,
            natural_frame_weight=args.natural_frame_weight,
            observation_view=args.observation_view,
            replay_cache=args.replay_cache,
            action_mismatch_scale=args.action_mismatch_scale,
            aim_error_scale=args.aim_error_scale,
            excess_spin_scale=args.excess_spin_scale,
            spin_tolerance_degrees=args.spin_tolerance_degrees,
        ),
    )
    print(
        json.dumps({"status": report["status"], "ppo": report["ppo"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
