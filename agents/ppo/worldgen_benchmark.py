"""Compare generic recurrent IL -> PPO training across WorldGen V2 tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from agents import training_runtime as _training_runtime  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from agents.ppo.worldgen_il import (
    _ENVIRONMENT_HEAD_ALIASES,
    WorldgenILPPOAgent,
    WorldgenILPPOSettings,
)
from agents.ppo.worldgen_trace import capture_recurrent_worldgen_trace
from agents.ppo.contracts import task_contract
from agents.ppo.runtime import default_compilation_cache
from arena.evaluation import (
    ArmResult,
    EvaluationReplicate,
    PromotionGate,
    assess_promotion,
)
from arena.jax_contract import GROUP_FEATURES, HEAD_SPANS, observation_groups
from arena.jax_env import ArenaHandle
from arena.training import (
    CombatFundamentalsConfig,
    arsenal_basic_attack_envelope,
    make_combat_fundamentals_environment,
)
from arena.league.tasks.framework.base import Task
from arena.league.tasks.framework.criteria import criterion_groups
from arena.league.tasks.games.world.generated import GENERATED
from arena.worlds import WorldSpec, world_spec_for_split
from hytalegym.jax.combat import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.jax.combat.observation.v3.policy import (
    decode_look_delta,
    greedy_arsenal_action_factors,
)
from hytalegym.jax.training.policy import apply_policy, sample_configured_actions
from hytalegym.jax.training.checkpoint import load_policy_checkpoint
from hytalegym.jax.training.ppo import combat_episode_outcome


REPORT_SCHEMA = "hytalerl_worldgen_generic_il_benchmark_v25"
MINIGAME_BEHAVIOR_SCHEMA = "hytalerl_worldgen_minigame_behavior_v2"
STAGES = ("before_il", "after_il", "after_ppo", "ppo_only")
TERMINAL_CAUSE_NAMES = (
    "success",
    "death",
    "simultaneous",
    "geometry_exhausted",
    "target_navigation_unsupported",
    "invalid",
    "unclassified",
)


@dataclass(frozen=True, slots=True)
class WorldgenBenchmarkSettings:
    batch: int = 2
    trace_steps: int = 8
    cloning_steps: int = 64
    cloning_learning_rate: float = 3.0e-4
    cloning_label_smoothing: float = 0.05
    cloning_entropy_weight: float = 0.01
    cloning_target_accuracy: float = 0.80
    cloning_min_steps: int = 8
    imitation_heads: tuple[str, ...] = tuple(HEAD_SPANS)
    demonstrator_checkpoint: Path | None = None
    demonstrator_mode: str = "greedy"
    demonstrator_minimum_success_rate: float = 0.70
    ppo_updates: int = 8
    rollout_steps: int = 4
    ppo_update_epochs: int = 2
    ppo_num_minibatches: int = 1
    evaluation_steps: int = 32
    evaluation_batch: int | None = None
    evaluation_updates: tuple[int, ...] = ()
    evaluation_policy_batch_size: int = 1
    evaluation_mode: str = "sample"
    encoder_size: int = 8
    recurrent_size: int = 8
    ppo_learning_rate: float = 3.0e-4
    ppo_entropy_coefficient: float = 1.0e-2
    ppo_maximum_kl: float = 2.0e-2
    native_behavior_checkpoint: Path | None = None
    native_behavior_corpus: Path | None = None
    native_behavior_live_checkpoint: Path | None = None
    native_behavior_live_checkpoint_dir: Path | None = None
    native_behavior_style: tuple[str, str] | None = None
    native_behavior_action_mismatch_scale: float = 0.0025
    native_behavior_aim_error_scale: float = 0.0025
    native_behavior_excess_spin_scale: float = 0.005
    native_behavior_spin_tolerance_degrees: float = 1.0
    native_behavior_lock_unsupervised_heads: bool = True
    native_behavior_online_prior: bool = True
    combat_fundamentals: CombatFundamentalsConfig | None = None
    difficulty: str = "armed"
    seed: int = 570057
    training_seeds: tuple[int, ...] = ()
    evaluation_seed: int = 580057
    evaluation_split: str = "training"
    heldout_world_count: int = 4
    output_dir: Path = Path("agents/ppo/artifacts/worldgen-il-benchmark")
    compilation_cache: Path | None = field(default_factory=default_compilation_cache)

    def __post_init__(self) -> None:
        if not isinstance(self.difficulty, str) or not self.difficulty:
            raise ValueError("difficulty must be a nonempty string")
        for name in (
            "batch",
            "trace_steps",
            "ppo_updates",
            "rollout_steps",
            "ppo_update_epochs",
            "ppo_num_minibatches",
            "evaluation_steps",
            "encoder_size",
            "recurrent_size",
            "evaluation_policy_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.evaluation_batch is not None and (
            isinstance(self.evaluation_batch, bool)
            or not isinstance(self.evaluation_batch, int)
            or self.evaluation_batch < 1
        ):
            raise ValueError("evaluation_batch must be a positive integer or None")
        for name in ("cloning_steps", "cloning_min_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.evaluation_seed, bool) or not isinstance(
            self.evaluation_seed, int
        ):
            raise ValueError("evaluation_seed must be an integer")
        if not isinstance(self.training_seeds, tuple) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in self.training_seeds
        ):
            raise TypeError("training_seeds must be a tuple of integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be unique")
        if self.evaluation_split not in {"training", "heldout"}:
            raise ValueError("evaluation_split must be 'training' or 'heldout'")
        if (
            isinstance(self.heldout_world_count, bool)
            or not isinstance(self.heldout_world_count, int)
            or not 1 <= self.heldout_world_count <= 4
        ):
            raise ValueError("heldout_world_count must be in [1, 4]")
        if self.batch % self.ppo_num_minibatches:
            raise ValueError("batch must be divisible by ppo_num_minibatches")
        if self.evaluation_mode not in {"sample", "greedy"}:
            raise ValueError("evaluation_mode must be 'sample' or 'greedy'")
        if (
            not isinstance(self.evaluation_updates, tuple)
            or any(
                isinstance(update, bool) or not isinstance(update, int)
                for update in self.evaluation_updates
            )
            or tuple(sorted(set(self.evaluation_updates))) != self.evaluation_updates
            or any(
                update < 0 or update > self.ppo_updates
                for update in self.evaluation_updates
            )
        ):
            raise ValueError(
                "evaluation_updates must be sorted unique integers in [0, ppo_updates]"
            )
        for name in ("ppo_learning_rate", "ppo_entropy_coefficient"):
            value = getattr(self, name)
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.ppo_learning_rate == 0.0:
            raise ValueError("ppo_learning_rate must be positive")
        if self.ppo_maximum_kl <= 0.0 or not np.isfinite(self.ppo_maximum_kl):
            raise ValueError("ppo_maximum_kl must be positive and finite")
        if (self.native_behavior_checkpoint is None) != (
            self.native_behavior_corpus is None
        ):
            raise ValueError(
                "native behavior checkpoint and corpus must be supplied together"
            )
        if (
            self.native_behavior_live_checkpoint is not None
            and self.native_behavior_live_checkpoint_dir is not None
        ):
            raise ValueError("pass one native live checkpoint source")
        if (
            self.native_behavior_live_checkpoint is not None
            or self.native_behavior_live_checkpoint_dir is not None
        ) and (
            self.native_behavior_checkpoint is None
            or self.native_behavior_style is None
        ):
            raise ValueError(
                "native live checkpoint requires a source checkpoint, corpus, and style"
            )
        if self.cloning_steps == 0 and not (
            self.native_behavior_live_checkpoint
            or self.native_behavior_live_checkpoint_dir
        ):
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
        if (
            self.combat_fundamentals is not None
            and self.evaluation_steps != self.combat_fundamentals.rollout_horizon_ticks
        ):
            raise ValueError(
                "combat fundamentals require evaluation_steps="
                f"{self.combat_fundamentals.rollout_horizon_ticks}"
            )
        if self.cloning_min_steps > self.cloning_steps:
            raise ValueError("cloning_min_steps must not exceed cloning_steps")
        if self.cloning_learning_rate <= 0.0 or not np.isfinite(
            self.cloning_learning_rate
        ):
            raise ValueError("cloning_learning_rate must be positive and finite")
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
        if self.demonstrator_mode not in {"greedy", "sample"}:
            raise ValueError("demonstrator_mode must be 'greedy' or 'sample'")
        if not 0.0 <= self.demonstrator_minimum_success_rate <= 1.0:
            raise ValueError("demonstrator_minimum_success_rate must be in [0, 1]")

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.training_seeds or (self.seed,)


def run_benchmark(
    settings: WorldgenBenchmarkSettings | None = None,
    *,
    tasks: tuple[Task, ...] = GENERATED,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train one matching-loadout policy per task and compare paired rollouts."""

    settings = settings or WorldgenBenchmarkSettings()
    if not tasks:
        raise ValueError("tasks must not be empty")
    if settings.native_behavior_live_checkpoint is not None and (
        len(tasks) != 1 or len(settings.seeds) != 1
    ):
        raise ValueError(
            "one native live checkpoint can cover only one task and training seed; "
            "use native_behavior_live_checkpoint_dir"
        )
    started = perf_counter()
    task_reports = []
    total = len(tasks) * len(settings.seeds)
    run_index = 0
    for task_index, task in enumerate(tasks):
        for training_seed in settings.seeds:
            run_index += 1
            if progress is not None:
                progress(
                    f"[{run_index}/{total}] training {task.name} "
                    f"seed={training_seed} (armed)"
                )
            live_checkpoint = _native_live_checkpoint(settings, task, training_seed)
            run_output = settings.output_dir / task.name
            if len(settings.seeds) > 1:
                run_output /= f"seed-{training_seed}"
            agent_settings = WorldgenILPPOSettings(
                batch=settings.batch,
                trace_steps=settings.trace_steps,
                cloning_steps=settings.cloning_steps,
                cloning_learning_rate=settings.cloning_learning_rate,
                cloning_label_smoothing=settings.cloning_label_smoothing,
                cloning_entropy_weight=settings.cloning_entropy_weight,
                cloning_target_accuracy=settings.cloning_target_accuracy,
                cloning_min_steps=settings.cloning_min_steps,
                imitation_heads=settings.imitation_heads,
                ppo_updates=settings.ppo_updates,
                evaluation_updates=settings.evaluation_updates,
                evaluation_policy_batch_size=settings.evaluation_policy_batch_size,
                rollout_steps=settings.rollout_steps,
                ppo_update_epochs=settings.ppo_update_epochs,
                ppo_num_minibatches=settings.ppo_num_minibatches,
                encoder_size=settings.encoder_size,
                recurrent_size=settings.recurrent_size,
                ppo_learning_rate=settings.ppo_learning_rate,
                ppo_entropy_coefficient=settings.ppo_entropy_coefficient,
                ppo_maximum_kl=settings.ppo_maximum_kl,
                native_behavior_checkpoint=settings.native_behavior_checkpoint,
                native_behavior_corpus=settings.native_behavior_corpus,
                native_behavior_live_checkpoint=live_checkpoint,
                native_behavior_style=settings.native_behavior_style,
                native_behavior_action_mismatch_scale=(
                    settings.native_behavior_action_mismatch_scale
                ),
                native_behavior_aim_error_scale=(
                    settings.native_behavior_aim_error_scale
                ),
                native_behavior_excess_spin_scale=(
                    settings.native_behavior_excess_spin_scale
                ),
                native_behavior_spin_tolerance_degrees=(
                    settings.native_behavior_spin_tolerance_degrees
                ),
                native_behavior_lock_unsupervised_heads=(
                    settings.native_behavior_lock_unsupervised_heads
                ),
                native_behavior_online_prior=settings.native_behavior_online_prior,
                combat_fundamentals=settings.combat_fundamentals,
                seed=training_seed,
                output_dir=run_output,
                compilation_cache=settings.compilation_cache,
            )
            agent = WorldgenILPPOAgent(
                agent_settings,
                task=task,
                difficulty=settings.difficulty,
            )
            evaluation_task = _evaluation_task(task, settings, task_index)
            evaluation_batch = settings.evaluation_batch or settings.batch
            evaluation_scene = (
                agent.scene
                if evaluation_task is task and evaluation_batch == settings.batch
                else evaluation_task.build_scene(
                    settings.difficulty, batch=evaluation_batch
                )
            )
            _require_policy_compatible(agent.scene, evaluation_scene)
            if settings.combat_fundamentals is not None:
                attack_envelope = arsenal_basic_attack_envelope(
                    evaluation_scene.runtime_config,
                    settings.combat_fundamentals.basic_ability_slot,
                )
                evaluation_scene = replace(
                    evaluation_scene,
                    environment=make_combat_fundamentals_environment(
                        evaluation_scene.environment,
                        settings.combat_fundamentals,
                        attack_envelope=attack_envelope,
                        agent_max_health=(
                            evaluation_scene.combat_params.agent_max_health
                        ),
                        target_max_health=(
                            evaluation_scene.combat_params.target_max_health
                        ),
                    ),
                )
            evaluation_seed = settings.evaluation_seed + task_index
            evaluate = make_worldgen_evaluator(
                evaluation_scene,
                evaluation_task,
                steps=settings.evaluation_steps,
                recurrent_size=settings.recurrent_size,
                config=agent.ppo_config(),
                mode=settings.evaluation_mode,
                key=jax.random.key(evaluation_seed),
            )
            trace_capture = _checkpoint_trace_capture(
                agent,
                task,
                settings.demonstrator_checkpoint,
                settings.demonstrator_mode,
            )
            demonstrator_admission = _admit_demonstrator(
                trace_capture,
                agent,
                task,
                settings,
                evaluation_seed + 100_000,
            )
            report = agent.run(
                trace_capture=trace_capture,
                evaluate=evaluate,
            )
            if progress is not None:
                evaluation = report["evaluation"]
                progress(
                    f"[{run_index}/{total}] {task.name} seed={training_seed}: "
                    f"IL accuracy {report['imitation']['final_accuracy']:.3f}, "
                    f"native [{_progress_behavior(evaluation['after_il'])}], "
                    f"IL-PPO [{_progress_behavior(evaluation['after_ppo'])}], "
                    f"PPO-only [{_progress_behavior(evaluation['ppo_only'])}]"
                )
            task_reports.append(
                {
                    "task": task.name,
                    "task_contract": task_contract(task, "armed"),
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "loadout": task.loadout,
                    "scene_contract_sha256": agent.scene.contract_sha256,
                    "world": dict(agent.scene.world_identity),
                    "evaluation_scene_contract_sha256": (
                        evaluation_scene.contract_sha256
                    ),
                    "evaluation_world": dict(evaluation_scene.world_identity),
                    "trace_rows": report["trace"]["rows"],
                    "demonstrator": report["trace"]["demonstrator"],
                    "demonstrator_admission": demonstrator_admission,
                    "imitation": report["imitation"],
                    "ppo": report["ppo"],
                    "checkpoints": report["checkpoints"],
                    "curve_checkpoints": report.get("curve_checkpoints"),
                    "evaluation": report["evaluation"],
                    "evaluation_curve": report.get("evaluation_curve"),
                    "native_behavior_prior": report.get("native_behavior_prior"),
                    "combat_fundamentals_reward": report.get(
                        "combat_fundamentals_reward"
                    ),
                    "native_behavior_live_checkpoint": (
                        None if live_checkpoint is None else str(live_checkpoint)
                    ),
                    "runtime": report["runtime"],
                    "report_path": report["report_path"],
                }
            )

    report = {
        "schema": REPORT_SCHEMA,
        "settings": {
            "batch": settings.batch,
            "trace_steps": settings.trace_steps,
            "cloning_steps": settings.cloning_steps,
            "cloning_learning_rate": settings.cloning_learning_rate,
            "cloning_label_smoothing": settings.cloning_label_smoothing,
            "cloning_entropy_weight": settings.cloning_entropy_weight,
            "cloning_target_accuracy": settings.cloning_target_accuracy,
            "cloning_min_steps": settings.cloning_min_steps,
            "imitation_heads": list(settings.imitation_heads),
            "demonstrator_checkpoint": (
                None
                if settings.demonstrator_checkpoint is None
                else str(settings.demonstrator_checkpoint)
            ),
            "demonstrator_mode": settings.demonstrator_mode,
            "demonstrator_minimum_success_rate": (
                settings.demonstrator_minimum_success_rate
            ),
            "ppo_updates": settings.ppo_updates,
            "rollout_steps": settings.rollout_steps,
            "ppo_update_epochs": settings.ppo_update_epochs,
            "ppo_num_minibatches": settings.ppo_num_minibatches,
            "ppo_learning_rate": settings.ppo_learning_rate,
            "ppo_entropy_coefficient": settings.ppo_entropy_coefficient,
            "ppo_maximum_kl": settings.ppo_maximum_kl,
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
            "native_behavior_live_checkpoint_dir": (
                None
                if settings.native_behavior_live_checkpoint_dir is None
                else str(settings.native_behavior_live_checkpoint_dir)
            ),
            "native_behavior_lock_unsupervised_heads": (
                settings.native_behavior_lock_unsupervised_heads
            ),
            "native_behavior_online_prior": settings.native_behavior_online_prior,
            "native_behavior_style": (
                None
                if settings.native_behavior_style is None
                else list(settings.native_behavior_style)
            ),
            "native_behavior_action_mismatch_scale": (
                settings.native_behavior_action_mismatch_scale
            ),
            "native_behavior_aim_error_scale": (
                settings.native_behavior_aim_error_scale
            ),
            "native_behavior_excess_spin_scale": (
                settings.native_behavior_excess_spin_scale
            ),
            "native_behavior_spin_tolerance_degrees": (
                settings.native_behavior_spin_tolerance_degrees
            ),
            "combat_fundamentals_reward": (
                None
                if settings.combat_fundamentals is None
                else {
                    **settings.combat_fundamentals.describe(),
                    "sha256": settings.combat_fundamentals.sha256,
                }
            ),
            "evaluation_steps": settings.evaluation_steps,
            "evaluation_batch": settings.evaluation_batch or settings.batch,
            "evaluation_updates": list(settings.evaluation_updates),
            "evaluation_policy_batch_size": settings.evaluation_policy_batch_size,
            "evaluation_mode": settings.evaluation_mode,
            "encoder_size": settings.encoder_size,
            "recurrent_size": settings.recurrent_size,
            "seed": settings.seed,
            "training_seeds": list(settings.seeds),
            "evaluation_seed": settings.evaluation_seed,
            "evaluation_split": settings.evaluation_split,
            "heldout_world_count": settings.heldout_world_count,
            "difficulty": settings.difficulty,
            "paired_evaluation_rollouts": True,
        },
        "tasks": task_reports,
        "task_contracts": {
            task.name: task_contract(task, settings.difficulty) for task in tasks
        },
        "aggregate": _aggregate(task_reports),
        "minigame_assessment": _aggregate_minigame_behavior(task_reports),
        "runtime": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "wall_seconds": perf_counter() - started,
            **_aggregate_runtime(task_reports),
        },
    }
    if settings.evaluation_split == "heldout":
        report["promotion"] = assess_promotion(
            _promotion_rows(task_reports),
            PromotionGate(minimum_tasks=len(GENERATED)),
        )
    output = settings.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "benchmark.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _aggregate_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [row["runtime"] for row in rows]
    evaluations = [
        runtime["evaluation_execution"]
        for runtime in runtimes
        if runtime.get("evaluation_execution") is not None
    ]
    return {
        "task_runs": len(runtimes),
        "summed_agent_run_wall_seconds": sum(
            float(runtime["run_wall_seconds"]) for runtime in runtimes
        ),
        "summed_build_wall_seconds": sum(
            float(runtime["build_wall_seconds"]) for runtime in runtimes
        ),
        "evaluation_execution": {
            "schema": "hytalerl_policy_evaluation_execution_aggregate_v1",
            "modes": sorted({item["mode"] for item in evaluations}),
            "policy_rollouts": sum(item["policy_rollouts"] for item in evaluations),
            "batch_invocations": sum(item["batch_invocations"] for item in evaluations),
            "maximum_policies_per_invocation": max(
                (item["maximum_policies_per_invocation"] for item in evaluations),
                default=0,
            ),
            "wall_seconds": sum(item["wall_seconds"] for item in evaluations),
        },
    }


def _progress_behavior(report: dict[str, Any]) -> str:
    behavior = report["minigame_behavior"]
    return (
        f"damage={1.0 - report['mean_final_raw_target_health_fraction']:.3f} "
        f"self={report['mean_final_raw_self_health_fraction']:.3f} "
        f"travel={behavior['locomotion']['mean_planar_distance_traveled']:.1f} "
        f"aim={behavior['tracking']['mean_mean_actor_aim_error_degrees']:.1f}deg "
        f"attack/100={behavior['basic_attack']['requests_per_100_active_steps']:.2f} "
        f"eff={behavior['damage_exchange']['damage_events_per_attack_request']:.3f} "
        f"censor={report['right_censoring_rate']:.3f} "
        f"nav={report['support_diagnostic_counts']['target_navigation_unsupported']}"
    )


def _native_live_checkpoint(
    settings: WorldgenBenchmarkSettings, task: Task, seed: int
) -> Path | None:
    """Resolve the exact pre-training policy for one task/seed contract."""

    if settings.native_behavior_live_checkpoint_dir is None:
        return settings.native_behavior_live_checkpoint
    path = (
        settings.native_behavior_live_checkpoint_dir
        / task.name
        / f"seed-{seed}"
        / "live_policy.npz"
    )
    if not path.is_file() or not path.with_suffix(path.suffix + ".json").is_file():
        raise FileNotFoundError(f"native live checkpoint is incomplete: {path}")
    return path


def _base_environment_state(state: Any) -> Any:
    """Unwrap reward/prior state without changing actor-visible tensors."""

    for _ in range(4):
        if hasattr(state, "runtime"):
            return state
        if not hasattr(state, "environment"):
            break
        state = state.environment
    raise TypeError("WorldGen evaluator could not reach its Arsenal state")


def make_worldgen_evaluator(
    scene,
    task: Task,
    *,
    steps: int,
    recurrent_size: int,
    config,
    mode: str,
    key,
):
    required = criterion_groups(task.success)
    if required is None:
        raise ValueError(f"task {task.name!r} has no static scoring contract")
    groups = tuple(sorted(required | {"self_f32", "target_f32"}))
    self_health = GROUP_FEATURES["self_f32"].index("health_fraction")
    target_health = GROUP_FEATURES["target_f32"].index("health_fraction")
    target_visible = GROUP_FEATURES["target_f32"].index("visible")
    target_facing = GROUP_FEATURES["target_f32"].index("facing_error")
    target_head_facing = GROUP_FEATURES["target_f32"].index("head_facing_error")
    self_attack_executing = GROUP_FEATURES["self_f32"].index("attack_executing")
    head_names = tuple(
        _ENVIRONMENT_HEAD_ALIASES.get(item.name, item.name)
        for item in scene.environment.spec.action_components
    )
    if head_names != tuple(HEAD_SPANS):
        raise ValueError("WorldGen evaluator action heads differ from Arena's ABI")
    ability_head = head_names.index("ability_none_plus_slots")
    movement_head = head_names.index("world_move_none_plus_compass")
    ability_start, ability_size = HEAD_SPANS["ability_none_plus_slots"]
    if ability_size < 2:
        raise ValueError("ability head has no basic-attack choice")
    basic_attack_choice = 1
    basic_attack_mask_index = ability_start + basic_attack_choice
    requires_visible_defeat = _goal_requires_visible_defeat(task.goal)

    def policy(parameters, carry, actor_input, policy_key):
        carry, logits, _ = apply_policy(
            parameters,
            actor_input.observation,
            carry,
            actor_input.action_mask,
        )
        if mode == "sample":
            factors, _ = sample_configured_actions(policy_key, logits, config)
        else:
            factors = greedy_arsenal_action_factors(logits)
        return carry, factors

    def record(transition):
        current_state = _base_environment_state(transition.state)
        next_state = _base_environment_state(transition.next_state)
        current = observation_groups(
            transition.actor_input.legal_observation,
            ("self_f32", "target_f32"),
        )
        combat = next_state.runtime.combat
        current_combat = current_state.runtime.combat
        runtime = next_state.runtime
        outcome = combat_episode_outcome(
            combat,
            transition.done,
            valid=transition.info.arsenal_info.valid,
        )
        return {
            **observation_groups(
                transition.next_actor_input.legal_observation,
                groups,
            ),
            "action": transition.action_factors,
            "reward": transition.reward,
            "done": transition.done,
            "current_self_f32": current["self_f32"],
            "current_target_visible": current["target_f32"][:, target_visible],
            "current_position": current_combat.position[
                :, (AGENT_ENTITY, TARGET_ENTITY)
            ],
            "next_position": combat.position[:, (AGENT_ENTITY, TARGET_ENTITY)],
            "current_yaw": current_combat.yaw[:, (AGENT_ENTITY, TARGET_ENTITY)],
            "next_yaw": combat.yaw[:, (AGENT_ENTITY, TARGET_ENTITY)],
            "basic_attack_legal": transition.actor_input.action_mask[
                :, basic_attack_mask_index
            ],
            "action_surface_legal": transition.info.action_surface_legal,
            "basic_attack_accepted": transition.info.arsenal_info.ability_accepted[
                :, AGENT_ENTITY
            ],
            "current_active_ability_root": (
                current_state.runtime.arsenal.active_ability_root_slot[:, AGENT_ENTITY]
            ),
            "next_active_ability_root": (
                runtime.arsenal.active_ability_root_slot[:, AGENT_ENTITY]
            ),
            "current_basic_attack_cooldown": (
                current_state.runtime.arsenal.ability_cooldown_seconds[
                    :, AGENT_ENTITY, 0
                ]
            ),
            "current_raw_target_health": (current_combat.health[:, TARGET_ENTITY]),
            "current_raw_self_health": current_combat.health[:, AGENT_ENTITY],
            "next_raw_target_health": combat.health[:, TARGET_ENTITY],
            "next_raw_self_health": combat.health[:, AGENT_ENTITY],
            "terminal_success": outcome.success,
            "terminal_death": outcome.death,
            "terminal_simultaneous": outcome.simultaneous,
            "terminal_geometry_exhausted": outcome.geometry_exhausted,
            "terminal_target_navigation_unsupported": (
                outcome.target_navigation_unsupported
            ),
            "terminal_invalid": outcome.invalid,
            "terminal_unclassified": outcome.unclassified,
            "geometry_exhausted": combat.geometry_exhausted,
            "target_navigation_unsupported": (combat.target_navigation_unsupported),
            "arsenal_failure_bits": runtime.arsenal.failure_bits,
            "mechanics_failure_bits": runtime.mechanics.failure_bits,
            "inventory_failure_bits": runtime.inventory.failure_bits,
        }

    def score(trajectory):
        actions = trajectory["action"]
        changes = actions[1:] != actions[:-1]
        done = trajectory["done"]
        active = ~jnp.concatenate(
            (
                jnp.zeros_like(done[:1]),
                jnp.cumsum(done[:-1], axis=0) > 0,
            ),
            axis=0,
        )
        active_steps = jnp.sum(active, axis=0)
        pair_active = active[1:]
        target_crossed_zero = (trajectory["current_raw_target_health"] > 0.0) & (
            trajectory["next_raw_target_health"] <= 0.0
        )
        visible_target_defeat_event = target_crossed_zero & (
            trajectory["current_target_visible"] > 0.5
        )
        visible_target_defeat = jnp.any(visible_target_defeat_event, axis=0)
        terminal_causes = {
            name: jnp.any(trajectory[f"terminal_{name}"], axis=0)
            for name in TERMINAL_CAUSE_NAMES
        }
        encountered_geometry_exhausted = jnp.any(
            trajectory["geometry_exhausted"], axis=0
        )
        encountered_target_navigation_unsupported = jnp.any(
            trajectory["target_navigation_unsupported"], axis=0
        )
        support_failure = (
            encountered_geometry_exhausted
            | encountered_target_navigation_unsupported
            | terminal_causes["invalid"]
            | terminal_causes["unclassified"]
        )
        criterion_success = task.success(trajectory)
        success = criterion_success & ~support_failure
        if requires_visible_defeat:
            success &= visible_target_defeat & terminal_causes["success"]
        terminal_step = jnp.where(
            jnp.any(done, axis=0),
            jnp.argmax(done, axis=0) + 1,
            -1,
        )
        visible_target_defeat_step = jnp.where(
            visible_target_defeat,
            jnp.argmax(visible_target_defeat_event, axis=0) + 1,
            -1,
        )
        first_frozen_step = jnp.where(
            (terminal_step > 0) & (terminal_step < done.shape[0]),
            terminal_step + 1,
            -1,
        )
        active_denominator = jnp.maximum(active_steps, 1)
        pair_denominator = jnp.maximum(jnp.sum(pair_active, axis=0), 1)
        target_visibility = trajectory["target_f32"][:, :, target_visible]
        active_visible = active & (target_visibility > 0.5)
        visible_denominator = jnp.maximum(jnp.sum(active_visible, axis=0), 1)
        positions = trajectory["current_position"]
        next_positions = trajectory["next_position"]
        current_delta = positions[:, :, TARGET_ENTITY] - positions[:, :, AGENT_ENTITY]
        next_delta = (
            next_positions[:, :, TARGET_ENTITY] - next_positions[:, :, AGENT_ENTITY]
        )
        current_planar_distance = jnp.linalg.norm(current_delta[..., (0, 2)], axis=-1)
        next_planar_distance = jnp.linalg.norm(next_delta[..., (0, 2)], axis=-1)

        def actor_facing_error(delta, yaw):
            bearing = jnp.rad2deg(jnp.arctan2(-delta[..., 0], -delta[..., 2]))
            return jnp.abs((bearing - yaw + 180.0) % 360.0 - 180.0)

        current_actor_facing_error = actor_facing_error(
            current_delta, trajectory["current_yaw"][:, :, AGENT_ENTITY]
        )
        next_actor_facing_error = actor_facing_error(
            next_delta, trajectory["next_yaw"][:, :, AGENT_ENTITY]
        )
        next_opponent_facing_error = actor_facing_error(
            -next_delta, trajectory["next_yaw"][:, :, TARGET_ENTITY]
        )
        planar_motion = jnp.linalg.norm(
            (next_positions[:, :, AGENT_ENTITY] - positions[:, :, AGENT_ENTITY])[
                ..., (0, 2)
            ],
            axis=-1,
        )
        final_active_index = jnp.maximum(active_steps - 1, 0)[None]
        final_planar_distance = jnp.take_along_axis(
            next_planar_distance, final_active_index, axis=0
        )[0]
        basic_attack_request = actions[..., ability_head] == basic_attack_choice
        movement_request = actions[..., movement_head] != 0
        basic_attack_legal = trajectory["basic_attack_legal"]
        attack_executing = (
            trajectory["current_self_f32"][:, :, self_attack_executing] > 0.5
        )
        damage_dealt = jnp.maximum(
            trajectory["current_raw_target_health"]
            - trajectory["next_raw_target_health"],
            0.0,
        )
        damage_received = jnp.maximum(
            trajectory["current_raw_self_health"] - trajectory["next_raw_self_health"],
            0.0,
        )
        damage_dealt_event = damage_dealt > 1.0e-6
        damage_received_event = damage_received > 1.0e-6
        basic_attack_accepted = (
            basic_attack_request & trajectory["basic_attack_accepted"]
        )
        current_busy = trajectory["current_active_ability_root"] >= 0
        basic_attack_active = trajectory["next_active_ability_root"] == 0
        cooldown_request = basic_attack_request & (
            trajectory["current_basic_attack_cooldown"] > 1.0e-6
        )
        busy_request = basic_attack_request & current_busy
        illegal_joint_action = (
            basic_attack_request & ~trajectory["action_surface_legal"]
        )
        other_rejected_request = (
            basic_attack_request
            & ~basic_attack_accepted
            & ~cooldown_request
            & ~busy_request
            & ~illegal_joint_action
        )

        def attack_lifecycle(carry, row):
            opened, hit = carry
            accepted, still_active, damaged, terminal = row
            opened |= accepted
            hit |= opened & damaged
            resolved = opened & ~still_active & ~terminal
            missed = resolved & ~hit
            return (
                jnp.where(resolved | terminal, False, opened),
                jnp.where(resolved | terminal, False, hit),
            ), (resolved, missed)

        (_, _), (attack_resolved, attack_missed) = jax.lax.scan(
            attack_lifecycle,
            (
                jnp.zeros_like(done[0]),
                jnp.zeros_like(done[0]),
            ),
            (basic_attack_accepted, basic_attack_active, damage_dealt_event, done),
        )
        yaw_delta = decode_look_delta(
            actions[..., head_names.index("yaw_delta_bins")],
            HEAD_SPANS["yaw_delta_bins"][1],
        )
        excess_spin = (
            jnp.maximum(
                jnp.abs(yaw_delta) - current_actor_facing_error - 2.0,
                0.0,
            )
            / 45.0
        )
        well_timed_attack = (
            basic_attack_accepted
            & (current_planar_distance >= 1.25)
            & (current_planar_distance <= 3.25)
            & (current_actor_facing_error <= 25.0)
        )

        def active_sum(value):
            return jnp.sum(jnp.where(active, value, 0), axis=0)

        def first_active_step(event):
            event = event & active
            return jnp.where(jnp.any(event, axis=0), jnp.argmax(event, axis=0) + 1, -1)

        raw_self_fraction = (
            trajectory["next_raw_self_health"] / scene.combat_params.agent_max_health
        )
        raw_target_fraction = (
            trajectory["next_raw_target_health"] / scene.combat_params.target_max_health
        )
        observed_target_error = jnp.abs(
            trajectory["target_f32"][:, :, target_health] - raw_target_fraction
        )
        visible = target_visibility > 0.5
        return {
            "success": success,
            "success_available": ~support_failure & jnp.any(done, axis=0),
            "support_clean": ~support_failure,
            "right_censored": ~jnp.any(done, axis=0),
            "criterion_success": criterion_success,
            "visible_target_defeat": visible_target_defeat,
            "return": jnp.sum(trajectory["reward"], axis=0),
            "final_self_health": trajectory["self_f32"][-1, :, self_health],
            "minimum_self_health": jnp.min(
                trajectory["self_f32"][:, :, self_health], axis=0
            ),
            "final_target_health": trajectory["target_f32"][-1, :, target_health],
            "minimum_target_health": jnp.min(
                trajectory["target_f32"][:, :, target_health], axis=0
            ),
            "visible_fraction": jnp.mean(target_visibility, axis=0),
            "active_visible_fraction": (
                jnp.sum(target_visibility * active, axis=0) / active_denominator
            ),
            "head_active_fraction": jnp.mean(actions != 0, axis=0),
            "head_change_fraction": jnp.mean(changes, axis=0),
            "decision_change_fraction": jnp.mean(jnp.any(changes, axis=-1), axis=0),
            "active_head_active_fraction": (
                jnp.sum((actions != 0) & active[..., None], axis=0)
                / active_denominator[:, None]
            ),
            "active_head_change_fraction": (
                jnp.sum(changes & pair_active[..., None], axis=0)
                / pair_denominator[:, None]
            ),
            "active_decision_change_fraction": (
                jnp.sum(jnp.any(changes, axis=-1) & pair_active, axis=0)
                / pair_denominator
            ),
            "planar_distance_traveled": active_sum(planar_motion),
            "net_closing_distance": current_planar_distance[0] - final_planar_distance,
            "mean_planar_distance": active_sum(next_planar_distance)
            / active_denominator,
            "minimum_planar_distance": jnp.min(
                jnp.where(active, next_planar_distance, jnp.inf), axis=0
            ),
            "final_planar_distance": final_planar_distance,
            "movement_requested_steps": active_sum(movement_request),
            "movement_stall_steps": active_sum(
                movement_request & (planar_motion <= 1.0e-4)
            ),
            "mean_actor_aim_error_degrees": active_sum(next_actor_facing_error)
            / active_denominator,
            "mean_visible_actor_aim_error_degrees": (
                jnp.sum(
                    jnp.where(
                        active_visible,
                        next_actor_facing_error,
                        0.0,
                    ),
                    axis=0,
                )
                / visible_denominator
            ),
            "mean_opponent_facing_error_degrees": active_sum(next_opponent_facing_error)
            / active_denominator,
            "mean_visible_opponent_facing_error": (
                jnp.sum(
                    jnp.where(
                        active_visible,
                        jnp.abs(trajectory["target_f32"][:, :, target_facing]),
                        0.0,
                    ),
                    axis=0,
                )
                / visible_denominator
            ),
            "mean_visible_opponent_head_facing_error": (
                jnp.sum(
                    jnp.where(
                        active_visible,
                        jnp.abs(trajectory["target_f32"][:, :, target_head_facing]),
                        0.0,
                    ),
                    axis=0,
                )
                / visible_denominator
            ),
            "aligned_engagement_steps": active_sum(
                (next_planar_distance >= 1.25)
                & (next_planar_distance <= 3.25)
                & (next_actor_facing_error <= 25.0)
            ),
            "excess_spin_fraction": active_sum(excess_spin) / active_denominator,
            "basic_attack_legal_steps": active_sum(basic_attack_legal),
            "basic_attack_request_steps": active_sum(basic_attack_request),
            "basic_attack_accepted_steps": active_sum(basic_attack_accepted),
            "basic_attack_resolved_steps": active_sum(attack_resolved),
            "basic_attack_missed_steps": active_sum(attack_missed),
            "basic_attack_well_timed_steps": active_sum(well_timed_attack),
            "basic_attack_cooldown_request_steps": active_sum(cooldown_request),
            "basic_attack_busy_request_steps": active_sum(busy_request),
            "basic_attack_rejected_request_steps": active_sum(other_rejected_request),
            "basic_attack_illegal_joint_action_steps": active_sum(illegal_joint_action),
            "basic_attack_illegal_request_steps": active_sum(
                basic_attack_request & ~basic_attack_legal
            ),
            "basic_attack_while_executing_steps": active_sum(
                basic_attack_request & attack_executing
            ),
            "attack_executing_steps": active_sum(attack_executing),
            "damage_dealt_events": active_sum(damage_dealt_event),
            "damage_received_events": active_sum(damage_received_event),
            "damage_dealt_fraction": active_sum(damage_dealt)
            / scene.combat_params.target_max_health,
            "damage_received_fraction": active_sum(damage_received)
            / scene.combat_params.agent_max_health,
            "first_basic_attack_request_step": first_active_step(basic_attack_request),
            "first_damage_dealt_step": first_active_step(damage_dealt_event),
            "terminal": jnp.any(done, axis=0),
            "terminal_step": terminal_step,
            "first_frozen_step": first_frozen_step,
            "visible_target_defeat_step": visible_target_defeat_step,
            "active_steps": active_steps,
            "encountered_geometry_exhausted": encountered_geometry_exhausted,
            "encountered_target_navigation_unsupported": (
                encountered_target_navigation_unsupported
            ),
            "first_geometry_exhausted_step": jnp.where(
                encountered_geometry_exhausted,
                jnp.argmax(trajectory["geometry_exhausted"], axis=0) + 1,
                -1,
            ),
            "first_target_navigation_unsupported_step": jnp.where(
                encountered_target_navigation_unsupported,
                jnp.argmax(trajectory["target_navigation_unsupported"], axis=0) + 1,
                -1,
            ),
            **{f"terminal_{name}": value for name, value in terminal_causes.items()},
            **{
                f"terminal_{name}_failure_bits": jnp.bitwise_or.reduce(
                    jnp.where(done, trajectory[f"{name}_failure_bits"], 0), axis=0
                )
                for name in ("arsenal", "mechanics", "inventory")
            },
            "final_raw_self_health_fraction": raw_self_fraction[-1],
            "final_raw_target_health_fraction": raw_target_fraction[-1],
            "mean_self_health_observation_error": jnp.mean(
                jnp.abs(trajectory["self_f32"][:, :, self_health] - raw_self_fraction),
                axis=0,
            ),
            "mean_visible_target_health_observation_error": (
                jnp.sum(jnp.where(visible, observed_target_error, 0.0), axis=0)
                / jnp.maximum(jnp.sum(visible, axis=0), 1)
            ),
        }

    handle = ArenaHandle(scene)
    runner = handle.compile_parameterized_evaluator(
        policy,
        record,
        score,
        steps,
        initial_carry=jnp.zeros(
            (scene.expected_batch, recurrent_size), dtype=jnp.float32
        ),
    )

    def evaluate(parameters, _runner=runner) -> dict[str, Any]:
        started = perf_counter()
        values = _runner(parameters, key)
        jax.block_until_ready(values["success"])
        values = jax.device_get(values)
        success = np.asarray(values["success"])
        success_available = np.asarray(values["success_available"], dtype=np.bool_)
        support_clean = np.asarray(values["support_clean"], dtype=np.bool_)
        right_censored = np.asarray(values["right_censored"], dtype=np.bool_)
        criterion_success = np.asarray(values["criterion_success"])
        visible_target_defeat = np.asarray(values["visible_target_defeat"])
        returns = np.asarray(values["return"])
        terminal = np.asarray(values["terminal"], dtype=np.bool_)
        causes = {
            name: np.asarray(values[f"terminal_{name}"], dtype=np.bool_)
            for name in TERMINAL_CAUSE_NAMES
        }
        cause_total = sum(value.astype(np.int8) for value in causes.values())
        if np.any(cause_total != terminal.astype(np.int8)):
            raise RuntimeError("terminal cause taxonomy is not exhaustive and disjoint")
        terminal_cause = np.full(success.shape, "not_terminal", dtype=object)
        for name, value in causes.items():
            terminal_cause[value] = name
        failure_bit_histograms = {}
        for name in ("arsenal", "mechanics", "inventory"):
            bits = np.asarray(values[f"terminal_{name}_failure_bits"], dtype=np.uint32)
            unique, counts = np.unique(bits[bits != 0], return_counts=True)
            failure_bit_histograms[name] = {
                f"0x{int(value):08X}": int(count)
                for value, count in zip(unique, counts, strict=True)
            }
        minigame_behavior = _minigame_behavior_report(values)
        return {
            "mode": mode,
            "episodes": int(success.size),
            "success_rate": float(success.mean()),
            "successes": int(success.sum()),
            "success_by_lane": success.tolist(),
            "success_available_by_lane": success_available.tolist(),
            "support_clean_by_lane": support_clean.tolist(),
            "right_censored_by_lane": right_censored.tolist(),
            "right_censored": int(right_censored.sum()),
            "right_censoring_rate": float(right_censored.mean()),
            "success_definition": (
                "task_criterion_and_terminal_combat_victory_and_visible_target_defeat"
                if requires_visible_defeat
                else "task_criterion_and_supported_episode"
            ),
            "criterion_success_rate": float(criterion_success.mean()),
            "criterion_successes": int(criterion_success.sum()),
            "criterion_success_by_lane": criterion_success.tolist(),
            "visible_target_defeat_rate": float(visible_target_defeat.mean()),
            "visible_target_defeats": int(visible_target_defeat.sum()),
            "visible_target_defeat_by_lane": visible_target_defeat.tolist(),
            "mean_return": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_by_lane": returns.tolist(),
            "mean_final_self_health": float(
                np.asarray(values["final_self_health"]).mean()
            ),
            "mean_minimum_self_health": float(
                np.asarray(values["minimum_self_health"]).mean()
            ),
            "mean_final_target_health": float(
                np.asarray(values["final_target_health"]).mean()
            ),
            "mean_minimum_target_health": float(
                np.asarray(values["minimum_target_health"]).mean()
            ),
            "mean_visible_fraction": float(
                np.asarray(values["visible_fraction"]).mean()
            ),
            "mean_active_visible_fraction": float(
                np.asarray(values["active_visible_fraction"]).mean()
            ),
            "mean_decision_change_fraction": float(
                np.asarray(values["decision_change_fraction"]).mean()
            ),
            "mean_active_decision_change_fraction": float(
                np.asarray(values["active_decision_change_fraction"]).mean()
            ),
            "head_active_fraction": _head_means(
                head_names, values["head_active_fraction"]
            ),
            "head_change_fraction": _head_means(
                head_names, values["head_change_fraction"]
            ),
            "active_head_active_fraction": _head_means(
                head_names, values["active_head_active_fraction"]
            ),
            "active_head_change_fraction": _head_means(
                head_names, values["active_head_change_fraction"]
            ),
            "minigame_behavior": minigame_behavior,
            "terminal_rate": float(terminal.mean()),
            "terminal_cause_counts": {
                name: int(value.sum()) for name, value in causes.items()
            },
            "support_diagnostic_counts": {
                "geometry_exhausted": int(
                    np.asarray(values["encountered_geometry_exhausted"]).sum()
                ),
                "target_navigation_unsupported": int(
                    np.asarray(
                        values["encountered_target_navigation_unsupported"]
                    ).sum()
                ),
            },
            "geometry_exhausted_by_lane": np.asarray(
                values["encountered_geometry_exhausted"], dtype=np.bool_
            ).tolist(),
            "target_navigation_unsupported_by_lane": np.asarray(
                values["encountered_target_navigation_unsupported"], dtype=np.bool_
            ).tolist(),
            "first_geometry_exhausted_steps": np.asarray(
                values["first_geometry_exhausted_step"], dtype=int
            ).tolist(),
            "first_target_navigation_unsupported_steps": np.asarray(
                values["first_target_navigation_unsupported_step"], dtype=int
            ).tolist(),
            "terminal_failure_bit_histograms": failure_bit_histograms,
            "terminal_steps": np.asarray(values["terminal_step"], dtype=int).tolist(),
            "first_frozen_steps": np.asarray(
                values["first_frozen_step"], dtype=int
            ).tolist(),
            "visible_target_defeat_steps": np.asarray(
                values["visible_target_defeat_step"], dtype=int
            ).tolist(),
            "terminal_causes": terminal_cause.tolist(),
            "active_steps": np.asarray(values["active_steps"], dtype=int).tolist(),
            "mean_active_steps": float(np.asarray(values["active_steps"]).mean()),
            "mean_final_raw_self_health_fraction": float(
                np.asarray(values["final_raw_self_health_fraction"]).mean()
            ),
            "final_raw_self_health_fraction_by_lane": np.asarray(
                values["final_raw_self_health_fraction"]
            ).tolist(),
            "mean_final_raw_target_health_fraction": float(
                np.asarray(values["final_raw_target_health_fraction"]).mean()
            ),
            "final_raw_target_health_fraction_by_lane": np.asarray(
                values["final_raw_target_health_fraction"]
            ).tolist(),
            "mean_self_health_observation_error": float(
                np.asarray(values["mean_self_health_observation_error"]).mean()
            ),
            "mean_visible_target_health_observation_error": float(
                np.asarray(
                    values["mean_visible_target_health_observation_error"]
                ).mean()
            ),
            "world_seed_breakdown": _world_seed_metrics(
                scene.world_identity,
                success,
                returns,
                criterion_success=criterion_success,
                visible_target_defeat=visible_target_defeat,
                terminal_cause=terminal_cause,
                geometry_exhausted=np.asarray(
                    values["encountered_geometry_exhausted"], dtype=np.bool_
                ),
                target_navigation_unsupported=np.asarray(
                    values["encountered_target_navigation_unsupported"],
                    dtype=np.bool_,
                ),
                right_censored=right_censored,
                behavior={
                    name: np.asarray(values[name]) for name in _WORLD_BEHAVIOR_FIELDS
                },
            ),
            "wall_seconds": perf_counter() - started,
        }

    batched_runners: dict[int, Any] = {}

    def evaluate_many(parameters: list[Any]) -> list[dict[str, Any]]:
        """Evaluate multiple policies together with identical rollout randomness."""

        if not parameters:
            return []
        policy_count = len(parameters)
        if policy_count not in batched_runners:
            compile_many = getattr(handle, "compile_batched_parameterized_evaluator")
            batched_runners[policy_count] = compile_many(
                policy,
                record,
                score,
                steps,
                initial_carry=jnp.zeros(
                    (scene.expected_batch, recurrent_size), dtype=jnp.float32
                ),
            )
        batched_runner = batched_runners[policy_count]
        stacked = jax.tree.map(lambda *items: jnp.stack(items), *parameters)
        started = perf_counter()
        values = batched_runner(stacked, key)
        jax.block_until_ready(values["success"])
        elapsed = perf_counter() - started
        reports = []
        for index in range(len(parameters)):
            item = jax.tree.map(lambda value: value[index], values)
            report = evaluate(None, lambda _parameters, _key, item=item: item)
            report["wall_seconds"] = elapsed / policy_count
            report["evaluation_batch"] = {
                "policies": policy_count,
                "wall_seconds": elapsed,
                "common_random_numbers": True,
            }
            reports.append(report)
        return reports

    evaluate.evaluate_many = evaluate_many
    return evaluate


@dataclass(frozen=True, slots=True)
class _CheckpointTraceCapture:
    scene: Any
    steps: int
    parameters: Any
    config: Any
    checkpoint_sha256: str
    mode: str

    def __call__(self, key):
        return capture_recurrent_worldgen_trace(
            self.scene,
            key,
            self.steps,
            self.parameters,
            self.config,
            checkpoint_sha256=self.checkpoint_sha256,
            mode=self.mode,
        )


def _checkpoint_trace_capture(
    agent: WorldgenILPPOAgent,
    task: Task,
    checkpoint: Path | None,
    mode: str,
):
    if checkpoint is None:
        return None
    source = checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"demonstrator checkpoint does not exist: {source}")
    params, config, metadata = load_policy_checkpoint(source)
    expected_task = task_contract(task, "armed")
    if metadata.get("transfer_contract") != dict(agent.scene.policy_contract):
        raise ValueError(
            "demonstrator checkpoint policy transfer contract does not match"
        )
    if metadata.get("scene_contract_sha256") != agent.scene.contract_sha256:
        raise ValueError("demonstrator checkpoint scene contract does not match")
    if metadata.get("task_contract_sha256") != expected_task["sha256"]:
        raise ValueError("demonstrator checkpoint task contract does not match")
    digest = hashlib.sha256(source.read_bytes()).hexdigest().upper()

    return _CheckpointTraceCapture(
        agent.scene,
        agent.settings.trace_steps,
        params,
        config,
        digest,
        mode,
    )


def _admit_demonstrator(capture, agent, task, settings, seed):
    if capture is None:
        return None
    evaluate = make_worldgen_evaluator(
        agent.scene,
        task,
        steps=settings.evaluation_steps,
        recurrent_size=capture.config.recurrent_size,
        config=capture.config,
        mode=capture.mode,
        key=jax.random.key(seed),
    )
    evaluation = evaluate(capture.parameters)
    rates = [row["success_rate"] for row in evaluation["world_seed_breakdown"].values()]
    minimum = settings.demonstrator_minimum_success_rate
    admission = {
        "passed": evaluation["success_rate"] >= minimum and min(rates) >= minimum,
        "minimum_success_rate": minimum,
        "pooled_success_rate": evaluation["success_rate"],
        "minimum_world_success_rate": min(rates),
        "evaluation_seed": seed,
        "evaluation": evaluation,
    }
    if not admission["passed"]:
        raise RuntimeError(
            "demonstrator failed calibration admission: "
            f"pooled={evaluation['success_rate']:.3f}, "
            f"minimum_world={min(rates):.3f}, required={minimum:.3f}"
        )
    return admission


def _evaluation_task(
    task: Task, settings: WorldgenBenchmarkSettings, task_index: int
) -> Task:
    if settings.evaluation_split == "training":
        return task
    if not isinstance(task.world, WorldSpec):
        raise TypeError("held-out evaluation requires a WorldGen V2 WorldSpec")
    world = world_spec_for_split(
        f"{task.world.name}_heldout",
        task.world.structure,
        "heldout",
        count=settings.heldout_world_count,
        selection_key=task_index,
    )
    overlap = set(task.world.seeds) & set(world.seeds)
    if overlap:
        raise RuntimeError(f"training and held-out WorldGen seeds overlap: {overlap}")
    return replace(
        task,
        name=f"{task.name}_heldout",
        description=f"Held-out evaluation for {task.name}.",
        world=world,
    )


def _require_policy_compatible(training_scene, evaluation_scene) -> None:
    if dict(training_scene.policy_contract) != dict(evaluation_scene.policy_contract):
        raise ValueError("training and held-out policy transfer contracts differ")
    training = training_scene.environment.spec
    evaluation = evaluation_scene.environment.spec
    training_surface = (
        training.observation_schema,
        training.observation_size,
        training.action_schema,
        tuple((head.name, head.size) for head in training.action_components),
    )
    evaluation_surface = (
        evaluation.observation_schema,
        evaluation.observation_size,
        evaluation.action_schema,
        tuple((head.name, head.size) for head in evaluation.action_components),
    )
    if training_surface != evaluation_surface:
        raise ValueError("training and held-out policy surfaces differ")


def _promotion_rows(task_reports: list[dict[str, Any]]):
    return tuple(
        EvaluationReplicate(
            task=row["task"],
            training_seed=row["training_seed"],
            evaluation_seed=row["evaluation_seed"],
            split=row["evaluation_world"]["split"],
            arms={
                stage: ArmResult(
                    successes=row["evaluation"][stage]["successes"],
                    episodes=row["evaluation"][stage]["episodes"],
                )
                for stage in ("after_ppo", "ppo_only", "before_il")
            },
        )
        for row in task_reports
    )


def _head_means(names: tuple[str, ...], values: Any) -> dict[str, float]:
    means = np.asarray(values).mean(axis=0)
    return {name: float(value) for name, value in zip(names, means, strict=True)}


_WORLD_BEHAVIOR_FIELDS = (
    "planar_distance_traveled",
    "net_closing_distance",
    "mean_actor_aim_error_degrees",
    "basic_attack_request_steps",
    "basic_attack_missed_steps",
    "damage_dealt_events",
    "damage_dealt_fraction",
    "damage_received_fraction",
)


def _minigame_behavior_report(values: dict[str, Any]) -> dict[str, Any]:
    arrays = {
        name: np.asarray(values[name])
        for name in (
            "active_steps",
            "planar_distance_traveled",
            "net_closing_distance",
            "mean_planar_distance",
            "minimum_planar_distance",
            "final_planar_distance",
            "movement_requested_steps",
            "movement_stall_steps",
            "mean_actor_aim_error_degrees",
            "mean_visible_actor_aim_error_degrees",
            "mean_opponent_facing_error_degrees",
            "mean_visible_opponent_facing_error",
            "mean_visible_opponent_head_facing_error",
            "aligned_engagement_steps",
            "excess_spin_fraction",
            "basic_attack_legal_steps",
            "basic_attack_request_steps",
            "basic_attack_accepted_steps",
            "basic_attack_resolved_steps",
            "basic_attack_missed_steps",
            "basic_attack_well_timed_steps",
            "basic_attack_cooldown_request_steps",
            "basic_attack_busy_request_steps",
            "basic_attack_rejected_request_steps",
            "basic_attack_illegal_joint_action_steps",
            "basic_attack_illegal_request_steps",
            "basic_attack_while_executing_steps",
            "attack_executing_steps",
            "damage_dealt_events",
            "damage_received_events",
            "damage_dealt_fraction",
            "damage_received_fraction",
            "first_basic_attack_request_step",
            "first_damage_dealt_step",
        )
    }

    def summary(*names: str) -> dict[str, Any]:
        return {f"mean_{name}": float(arrays[name].mean()) for name in names} | {
            f"{name}_by_lane": arrays[name].tolist() for name in names
        }

    active = max(int(arrays["active_steps"].sum()), 1)
    movement_requests = int(arrays["movement_requested_steps"].sum())
    legal = int(arrays["basic_attack_legal_steps"].sum())
    requests = int(arrays["basic_attack_request_steps"].sum())
    damage_events = int(arrays["damage_dealt_events"].sum())
    return {
        "schema": MINIGAME_BEHAVIOR_SCHEMA,
        "locomotion": {
            **summary(
                "planar_distance_traveled",
                "net_closing_distance",
                "mean_planar_distance",
                "minimum_planar_distance",
                "final_planar_distance",
                "movement_requested_steps",
                "movement_stall_steps",
            ),
            "movement_stall_fraction": float(
                arrays["movement_stall_steps"].sum() / max(movement_requests, 1)
            ),
        },
        "tracking": summary(
            "mean_actor_aim_error_degrees",
            "mean_visible_actor_aim_error_degrees",
            "mean_opponent_facing_error_degrees",
            "mean_visible_opponent_facing_error",
            "mean_visible_opponent_head_facing_error",
            "aligned_engagement_steps",
            "excess_spin_fraction",
        ),
        "basic_attack": {
            "head": "ability_none_plus_slots",
            "choice": 1,
            "semantics": "certified_default_ability_slot_zero",
            **summary(
                "basic_attack_legal_steps",
                "basic_attack_request_steps",
                "basic_attack_accepted_steps",
                "basic_attack_resolved_steps",
                "basic_attack_missed_steps",
                "basic_attack_well_timed_steps",
                "basic_attack_cooldown_request_steps",
                "basic_attack_busy_request_steps",
                "basic_attack_rejected_request_steps",
                "basic_attack_illegal_joint_action_steps",
                "basic_attack_illegal_request_steps",
                "basic_attack_while_executing_steps",
                "attack_executing_steps",
            ),
            "requests": requests,
            "requests_per_100_active_steps": float(100.0 * requests / active),
            "request_fraction_when_legal": float(requests / max(legal, 1)),
            "illegal_requests": int(arrays["basic_attack_illegal_request_steps"].sum()),
            "missed_attacks": int(arrays["basic_attack_missed_steps"].sum()),
            "cooldown_requests": int(
                arrays["basic_attack_cooldown_request_steps"].sum()
            ),
            "busy_requests": int(arrays["basic_attack_busy_request_steps"].sum()),
            "illegal_joint_actions": int(
                arrays["basic_attack_illegal_joint_action_steps"].sum()
            ),
            "requests_while_executing": int(
                arrays["basic_attack_while_executing_steps"].sum()
            ),
            "first_request_steps": arrays["first_basic_attack_request_step"].tolist(),
        },
        "damage_exchange": {
            **summary(
                "damage_dealt_events",
                "damage_received_events",
                "damage_dealt_fraction",
                "damage_received_fraction",
            ),
            "damage_dealt_events": damage_events,
            "damage_events_per_attack_request": float(damage_events / max(requests, 1)),
            "first_damage_dealt_steps": arrays["first_damage_dealt_step"].tolist(),
        },
    }


def _world_seed_metrics(
    world: dict[str, Any],
    success: np.ndarray,
    returns: np.ndarray,
    *,
    criterion_success: np.ndarray | None = None,
    visible_target_defeat: np.ndarray | None = None,
    terminal_cause: np.ndarray | None = None,
    geometry_exhausted: np.ndarray | None = None,
    target_navigation_unsupported: np.ndarray | None = None,
    right_censored: np.ndarray | None = None,
    behavior: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    cycle = np.asarray(world["assignment_seed_cycle"], dtype=np.int64)
    if cycle.ndim != 1 or cycle.size == 0:
        raise ValueError("world assignment seed cycle must be nonempty")
    lane_seed = cycle[np.arange(success.size) % cycle.size]
    artifacts = {int(row["seed"]): row for row in world["artifacts"]}
    result = {}
    for seed in dict.fromkeys(map(int, cycle)):
        select = lane_seed == seed
        result[str(seed)] = {
            "episodes": int(select.sum()),
            "successes": int(success[select].sum()),
            "success_rate": float(success[select].mean()),
            "mean_return": float(returns[select].mean()),
            "region_semantic_sha256": artifacts[seed]["region_semantic_sha256"],
        }
        if criterion_success is not None:
            result[str(seed)]["criterion_success_rate"] = float(
                criterion_success[select].mean()
            )
        if visible_target_defeat is not None:
            result[str(seed)]["visible_target_defeat_rate"] = float(
                visible_target_defeat[select].mean()
            )
        if terminal_cause is not None:
            selected_causes = terminal_cause[select]
            result[str(seed)]["terminal_cause_counts"] = {
                name: int(np.sum(selected_causes == name))
                for name in (*TERMINAL_CAUSE_NAMES, "not_terminal")
            }
        if geometry_exhausted is not None:
            result[str(seed)]["geometry_exhausted_diagnostic_count"] = int(
                geometry_exhausted[select].sum()
            )
        if target_navigation_unsupported is not None:
            result[str(seed)]["target_navigation_unsupported_diagnostic_count"] = int(
                target_navigation_unsupported[select].sum()
            )
        if right_censored is not None:
            result[str(seed)]["right_censored_count"] = int(
                right_censored[select].sum()
            )
        if behavior is not None:
            if any(
                np.asarray(value).shape != success.shape for value in behavior.values()
            ):
                raise ValueError("world behavior metrics must have one value per lane")
            result[str(seed)]["behavior_means"] = {
                name: float(np.asarray(value)[select].mean())
                for name, value in behavior.items()
            }
    return result


def _goal_requires_visible_defeat(goal: Any) -> bool:
    if goal is None:
        return False
    description = goal.describe()
    if description["kind"] in {"defeat_target", "defeat_within"}:
        return True
    return any(
        _described_goal_requires_visible_defeat(child)
        for child in description["parameters"].get("goals", ())
    )


def _described_goal_requires_visible_defeat(goal: dict[str, Any]) -> bool:
    if goal["kind"] in {"defeat_target", "defeat_within"}:
        return True
    return any(
        _described_goal_requires_visible_defeat(child)
        for child in goal.get("parameters", {}).get("goals", ())
    )


def _aggregate(task_reports: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for stage in STAGES:
        rows = [report["evaluation"][stage] for report in task_reports]
        aggregate[stage] = {
            name: float(np.mean([row[name] for row in rows]))
            for name in (
                "success_rate",
                "criterion_success_rate",
                "visible_target_defeat_rate",
                "mean_return",
                "mean_final_raw_self_health_fraction",
                "mean_final_raw_target_health_fraction",
                "mean_active_visible_fraction",
                "mean_active_decision_change_fraction",
                "mean_active_steps",
                "terminal_rate",
            )
        }
    aggregate["delta_after_il"] = _delta(aggregate["before_il"], aggregate["after_il"])
    aggregate["delta_after_ppo"] = _delta(aggregate["after_il"], aggregate["after_ppo"])
    aggregate["delta_ppo_only"] = _delta(aggregate["before_il"], aggregate["ppo_only"])
    aggregate["il_ppo_advantage"] = _delta(
        aggregate["ppo_only"], aggregate["after_ppo"]
    )
    return aggregate


_BEHAVIOR_AGGREGATES = {
    "locomotion": (
        "mean_planar_distance_traveled",
        "mean_net_closing_distance",
        "mean_mean_planar_distance",
        "mean_minimum_planar_distance",
        "mean_final_planar_distance",
        "mean_movement_requested_steps",
        "mean_movement_stall_steps",
        "movement_stall_fraction",
    ),
    "tracking": (
        "mean_mean_actor_aim_error_degrees",
        "mean_mean_visible_actor_aim_error_degrees",
        "mean_mean_opponent_facing_error_degrees",
        "mean_aligned_engagement_steps",
        "mean_excess_spin_fraction",
    ),
    "basic_attack": (
        "mean_basic_attack_legal_steps",
        "mean_basic_attack_request_steps",
        "mean_basic_attack_accepted_steps",
        "mean_basic_attack_resolved_steps",
        "mean_basic_attack_missed_steps",
        "mean_basic_attack_well_timed_steps",
        "mean_basic_attack_cooldown_request_steps",
        "mean_basic_attack_busy_request_steps",
        "mean_basic_attack_rejected_request_steps",
        "mean_basic_attack_illegal_joint_action_steps",
        "mean_attack_executing_steps",
        "requests_per_100_active_steps",
        "request_fraction_when_legal",
        "illegal_requests",
        "requests_while_executing",
    ),
    "damage_exchange": (
        "mean_damage_dealt_events",
        "mean_damage_received_events",
        "mean_damage_dealt_fraction",
        "mean_damage_received_fraction",
        "damage_events_per_attack_request",
    ),
}


def _aggregate_minigame_behavior(
    task_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep the study verdict multidimensional and task-resolved."""

    def vector(rows: list[dict[str, Any]], stage: str) -> dict[str, float]:
        reports = [row["evaluation"][stage] for row in rows]
        behavior = [report["minigame_behavior"] for report in reports]
        values = {
            f"{group}.{name}": float(np.mean([item[group][name] for item in behavior]))
            for group, names in _BEHAVIOR_AGGREGATES.items()
            for name in names
        }
        values.update(
            {
                name: float(
                    np.mean([_evaluation_dimension(report, name) for report in reports])
                )
                for name in _EVALUATION_DIMENSIONS
            }
        )
        return values

    by_task = {}
    for task in dict.fromkeys(row["task"] for row in task_reports):
        rows = [row for row in task_reports if row["task"] == task]
        stages = {stage: vector(rows, stage) for stage in STAGES}
        by_task[task] = {
            "training_seeds": [row["training_seed"] for row in rows],
            "stages": stages,
            "il_ppo_change": _delta(stages["after_il"], stages["after_ppo"]),
            "il_ppo_advantage": _delta(stages["ppo_only"], stages["after_ppo"]),
        }
    stages = {stage: vector(task_reports, stage) for stage in STAGES}
    return {
        "schema": "hytalerl_worldgen_minigame_assessment_v3",
        "dimensions": [*_BEHAVIOR_AGGREGATES, "episode", "support"],
        "by_task": by_task,
        "macro": {
            "stages": stages,
            "il_ppo_change": _delta(stages["after_il"], stages["after_ppo"]),
            "il_ppo_advantage": _delta(stages["ppo_only"], stages["after_ppo"]),
        },
        "interpretation": (
            "No scalar promotion score: read locomotion, tracking, legal attack "
            "timing, damage exchange, outcomes, censoring, and support together."
        ),
    }


_EVALUATION_DIMENSIONS = (
    "episode.strict_success_rate",
    "episode.criterion_success_rate",
    "episode.visible_target_defeat_rate",
    "episode.mean_final_raw_self_health_fraction",
    "episode.mean_final_raw_target_health_fraction",
    "episode.mean_active_visible_fraction",
    "episode.terminal_rate",
    "episode.right_censoring_rate",
    "episode.mean_active_steps",
    "support.clean_fraction",
    "support.geometry_exhausted_fraction",
    "support.target_navigation_unsupported_fraction",
    "support.invalid_terminal_fraction",
    "support.unclassified_terminal_fraction",
)


def _evaluation_dimension(report: dict[str, Any], name: str) -> float:
    direct = {
        "episode.strict_success_rate": "success_rate",
        "episode.criterion_success_rate": "criterion_success_rate",
        "episode.visible_target_defeat_rate": "visible_target_defeat_rate",
        "episode.mean_final_raw_self_health_fraction": (
            "mean_final_raw_self_health_fraction"
        ),
        "episode.mean_final_raw_target_health_fraction": (
            "mean_final_raw_target_health_fraction"
        ),
        "episode.mean_active_visible_fraction": "mean_active_visible_fraction",
        "episode.terminal_rate": "terminal_rate",
        "episode.right_censoring_rate": "right_censoring_rate",
        "episode.mean_active_steps": "mean_active_steps",
    }
    if name in direct:
        return float(report[direct[name]])
    episodes = max(int(report["episodes"]), 1)
    counts = report["support_diagnostic_counts"]
    terminal = report["terminal_cause_counts"]
    support = {
        "support.clean_fraction": float(np.mean(report["support_clean_by_lane"])),
        "support.geometry_exhausted_fraction": counts["geometry_exhausted"] / episodes,
        "support.target_navigation_unsupported_fraction": (
            counts["target_navigation_unsupported"] / episodes
        ),
        "support.invalid_terminal_fraction": terminal["invalid"] / episodes,
        "support.unclassified_terminal_fraction": terminal["unclassified"] / episodes,
    }
    return float(support[name])


def _delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - value for name, value in before.items()}


def _combat_fundamentals_config(
    enabled: bool, path: Path | None
) -> CombatFundamentalsConfig | None:
    if path is None:
        return CombatFundamentalsConfig() if enabled else None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("combat reward config must be a JSON object")
    weights = data.get("weights", data)
    if not isinstance(weights, dict):
        raise ValueError("combat reward weights must be a JSON object")
    return CombatFundamentalsConfig(**weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--trace-steps", type=int, default=8)
    parser.add_argument("--cloning-steps", type=int, default=64)
    parser.add_argument("--cloning-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--cloning-label-smoothing", type=float, default=0.05)
    parser.add_argument("--cloning-entropy-weight", type=float, default=0.01)
    parser.add_argument("--cloning-target-accuracy", type=float, default=0.80)
    parser.add_argument("--cloning-min-steps", type=int, default=8)
    parser.add_argument(
        "--imitation-heads",
        nargs="+",
        choices=tuple(HEAD_SPANS),
        default=tuple(HEAD_SPANS),
    )
    parser.add_argument("--demonstrator-checkpoint", type=Path)
    parser.add_argument(
        "--demonstrator-mode", choices=("greedy", "sample"), default="greedy"
    )
    parser.add_argument("--demonstrator-minimum-success-rate", type=float, default=0.70)
    parser.add_argument("--ppo-updates", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--ppo-update-epochs", type=int, default=2)
    parser.add_argument("--ppo-num-minibatches", type=int, default=1)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ppo-entropy-coefficient", type=float, default=1.0e-2)
    parser.add_argument("--ppo-maximum-kl", type=float, default=2.0e-2)
    parser.add_argument("--native-behavior-checkpoint", type=Path)
    parser.add_argument("--native-behavior-corpus", type=Path)
    parser.add_argument("--native-behavior-live-checkpoint", type=Path)
    parser.add_argument("--native-behavior-live-checkpoint-dir", type=Path)
    parser.add_argument(
        "--native-behavior-style",
        nargs=2,
        metavar=("ACTOR_ROLE", "OPPONENT_ROLE"),
    )
    parser.add_argument(
        "--native-behavior-action-mismatch-scale", type=float, default=0.0025
    )
    parser.add_argument("--native-behavior-aim-error-scale", type=float, default=0.0025)
    parser.add_argument(
        "--native-behavior-excess-spin-scale", type=float, default=0.005
    )
    parser.add_argument(
        "--native-behavior-spin-tolerance-degrees", type=float, default=1.0
    )
    parser.add_argument(
        "--native-behavior-allow-unsupervised-heads",
        action="store_true",
        help="allow non-fundamental action heads during native-prior PPO",
    )
    parser.add_argument(
        "--native-behavior-initialization-only",
        action="store_true",
        help="use the native transplant only as PPO initialization",
    )
    parser.add_argument(
        "--combat-fundamentals-reward",
        action="store_true",
        help="add dense JAX approach, aim, attack, hit, miss, and cooldown reward",
    )
    parser.add_argument(
        "--combat-fundamentals-config",
        type=Path,
        help="JSON CombatFundamentalsConfig overrides; also enables the reward",
    )
    parser.add_argument("--evaluation-steps", type=int, default=32)
    parser.add_argument("--evaluation-batch", type=int)
    parser.add_argument("--evaluation-updates", nargs="+", type=int, default=())
    parser.add_argument("--evaluation-policy-batch-size", type=int, default=1)
    parser.add_argument(
        "--evaluation-mode", choices=("sample", "greedy"), default="sample"
    )
    parser.add_argument("--encoder-size", type=int, default=8)
    parser.add_argument("--recurrent-size", type=int, default=8)
    parser.add_argument("--training-seeds", nargs="+", type=int)
    parser.add_argument("--evaluation-seed", type=int, default=580057)
    parser.add_argument(
        "--evaluation-split", choices=("training", "heldout"), default="training"
    )
    parser.add_argument("--heldout-world-count", type=int, default=4)
    task_by_name = {task.name: task for task in GENERATED}
    parser.add_argument("--tasks", nargs="+", choices=tuple(task_by_name))
    parser.add_argument(
        "--difficulty",
        default="armed",
        help="task difficulty/rung to build (for example inert or armed)",
    )
    parser.add_argument("--no-compilation-cache", action="store_true")
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=default_compilation_cache(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agents/ppo/artifacts/worldgen-il-benchmark"),
    )
    arguments = parser.parse_args()
    combat_fundamentals = _combat_fundamentals_config(
        arguments.combat_fundamentals_reward,
        arguments.combat_fundamentals_config,
    )
    report = run_benchmark(
        WorldgenBenchmarkSettings(
            batch=arguments.batch,
            trace_steps=arguments.trace_steps,
            cloning_steps=arguments.cloning_steps,
            cloning_learning_rate=arguments.cloning_learning_rate,
            cloning_label_smoothing=arguments.cloning_label_smoothing,
            cloning_entropy_weight=arguments.cloning_entropy_weight,
            cloning_target_accuracy=arguments.cloning_target_accuracy,
            cloning_min_steps=arguments.cloning_min_steps,
            imitation_heads=tuple(arguments.imitation_heads),
            demonstrator_checkpoint=arguments.demonstrator_checkpoint,
            demonstrator_mode=arguments.demonstrator_mode,
            demonstrator_minimum_success_rate=(
                arguments.demonstrator_minimum_success_rate
            ),
            ppo_updates=arguments.ppo_updates,
            rollout_steps=arguments.rollout_steps,
            ppo_update_epochs=arguments.ppo_update_epochs,
            ppo_num_minibatches=arguments.ppo_num_minibatches,
            ppo_learning_rate=arguments.ppo_learning_rate,
            ppo_entropy_coefficient=arguments.ppo_entropy_coefficient,
            ppo_maximum_kl=arguments.ppo_maximum_kl,
            native_behavior_checkpoint=arguments.native_behavior_checkpoint,
            native_behavior_corpus=arguments.native_behavior_corpus,
            native_behavior_live_checkpoint=(arguments.native_behavior_live_checkpoint),
            native_behavior_live_checkpoint_dir=(
                arguments.native_behavior_live_checkpoint_dir
            ),
            native_behavior_style=(
                None
                if arguments.native_behavior_style is None
                else tuple(arguments.native_behavior_style)
            ),
            native_behavior_action_mismatch_scale=(
                arguments.native_behavior_action_mismatch_scale
            ),
            native_behavior_aim_error_scale=(arguments.native_behavior_aim_error_scale),
            native_behavior_excess_spin_scale=(
                arguments.native_behavior_excess_spin_scale
            ),
            native_behavior_spin_tolerance_degrees=(
                arguments.native_behavior_spin_tolerance_degrees
            ),
            native_behavior_lock_unsupervised_heads=(
                not arguments.native_behavior_allow_unsupervised_heads
            ),
            native_behavior_online_prior=(
                not arguments.native_behavior_initialization_only
            ),
            combat_fundamentals=combat_fundamentals,
            difficulty=arguments.difficulty,
            evaluation_steps=arguments.evaluation_steps,
            evaluation_batch=arguments.evaluation_batch,
            evaluation_updates=tuple(arguments.evaluation_updates),
            evaluation_policy_batch_size=arguments.evaluation_policy_batch_size,
            evaluation_mode=arguments.evaluation_mode,
            encoder_size=arguments.encoder_size,
            recurrent_size=arguments.recurrent_size,
            training_seeds=tuple(arguments.training_seeds or ()),
            evaluation_seed=arguments.evaluation_seed,
            evaluation_split=arguments.evaluation_split,
            heldout_world_count=arguments.heldout_world_count,
            output_dir=arguments.output_dir,
            compilation_cache=(
                None if arguments.no_compilation_cache else arguments.compilation_cache
            ),
        ),
        tasks=(
            GENERATED
            if arguments.tasks is None
            else tuple(task_by_name[name] for name in arguments.tasks)
        ),
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "aggregate": report["aggregate"],
                "tasks": [
                    {
                        "task": row["task"],
                        "evaluation": row["evaluation"],
                    }
                    for row in report["tasks"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "WorldgenBenchmarkSettings",
    "make_worldgen_evaluator",
    "run_benchmark",
]
