"""CUDA throughput screen for fixed-shape Arena PPO updates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

# Imported for its import-time side effect: configure_training_runtime()
# must run before JAX is imported. Not unused -- do not delete.
from agents import training_runtime as _training_runtime  # noqa: F401

import jax
import numpy as np

from agents.ppo import PPOAgent, PPOSettings
from agents.ppo.contracts import task_contract
from agents.ppo.runtime import default_compilation_cache
from arena.tasks.games.world.generated import PLAINS_DUEL


SCHEMA = "hytalerl_ppo_cuda_shape_screen_v1"
SELECTION_SCHEMA = "hytalerl_ppo_cuda_shape_selection_v1"
PRODUCTION_SHAPES = ((64, 32), (96, 32), (64, 64))
MEASURED_TRANSITIONS = 12_288
MAX_PEAK_BYTES = 14 * 1024**3
REPLICATES = 3


@dataclass(frozen=True, slots=True)
class ScreenShape:
    environments: int
    rollout_steps: int

    @classmethod
    def parse(cls, label: str) -> "ScreenShape":
        try:
            environments, rollout_steps = map(int, label.lower().split("x"))
        except ValueError as exc:
            raise ValueError("shape must be formatted as ENVIRONMENTSxSTEPS") from exc
        shape = cls(environments, rollout_steps)
        if (shape.environments, shape.rollout_steps) not in PRODUCTION_SHAPES:
            raise ValueError(f"unsupported production shape: {label}")
        return shape

    @property
    def label(self) -> str:
        return f"{self.environments}x{self.rollout_steps}"

    @property
    def transitions_per_update(self) -> int:
        return self.environments * self.rollout_steps

    def measured_updates(self, transitions: int = MEASURED_TRANSITIONS) -> int:
        updates, remainder = divmod(transitions, self.transitions_per_update)
        if updates < 2 or remainder:
            raise ValueError("measured transitions must divide the shape exactly")
        return updates


def run_shape(
    shape: ScreenShape,
    *,
    output_dir: Path,
    compilation_cache: Path,
    seed: int = 570057,
    measured_transitions: int = MEASURED_TRANSITIONS,
    replicate: int = 1,
) -> dict[str, Any]:
    """Compile once, then measure an equal transition budget on CUDA."""

    if jax.default_backend() != "gpu":
        raise RuntimeError("production PPO shape screening requires a JAX GPU")
    measured_updates = shape.measured_updates(measured_transitions)
    if not 1 <= replicate <= REPLICATES:
        raise ValueError(f"replicate must be from 1 to {REPLICATES}")
    settings = PPOSettings(
        batch=shape.environments,
        trace_steps=1,
        cloning_steps=1,
        ppo_updates=1,
        rollout_steps=shape.rollout_steps,
        ppo_update_epochs=4,
        ppo_num_minibatches=4,
        encoder_size=128,
        recurrent_size=128,
        seed=seed,
        output_dir=output_dir,
        compilation_cache=compilation_cache,
    )
    agent = PPOAgent(settings, task=PLAINS_DUEL, difficulty="armed")
    config = agent.ppo_config()

    started = perf_counter()
    state = agent.handle.arena.training.initialize_ppo(
        jax.random.key(seed), agent.scene, config
    )
    jax.block_until_ready(state.observation)
    initialization_seconds = perf_counter() - started
    step = agent.handle.arena.training.make_ppo_step(
        agent.scene, config, compile=True
    )
    keys = jax.random.split(jax.random.key(seed + 1), measured_updates + 1)

    state, compile_row = _timed_update(step, state, keys[0], shape)
    warm_rows = []
    for key in keys[1:]:
        state, row = _timed_update(step, state, key, shape)
        warm_rows.append(row)

    memory = dict(jax.devices()[0].memory_stats() or {})
    expected_updates = measured_updates + 1
    expected_steps = expected_updates * shape.transitions_per_update
    counters_exact = (
        int(jax.device_get(state.update_count)) == expected_updates
        and int(jax.device_get(state.total_environment_steps)) == expected_steps
    )
    finite = all(row["finite"] for row in (compile_row, *warm_rows))
    stable_kl = all(
        row["metrics"]["approximate_kl"] < 0.02
        for row in (compile_row, *warm_rows)
    )
    peak_bytes = int(memory.get("peak_bytes_in_use", 0))
    valid = (
        finite
        and stable_kl
        and counters_exact
        and 0 < peak_bytes <= MAX_PEAK_BYTES
    )
    warm_seconds = [row["wall_seconds"] for row in warm_rows]
    warm_tps = [row["transitions_per_second"] for row in warm_rows]
    cache = agent.kit.jax_runtime_settings
    body = {
        "schema": SCHEMA,
        "valid": valid,
        "shape": {
            "label": shape.label,
            "environments": shape.environments,
            "rollout_steps": shape.rollout_steps,
            "transitions_per_update": shape.transitions_per_update,
        },
        "measurement": {
            "replicate": replicate,
            "equal_transition_budget": measured_transitions,
            "measured_updates": measured_updates,
            "compile": compile_row,
            "warm_updates": warm_rows,
            "median_warm_transitions_per_second": float(median(warm_tps)),
            "aggregate_warm_transitions_per_second": (
                measured_transitions / sum(warm_seconds)
            ),
        },
        "gates": {
            "finite_metrics": finite,
            "approximate_kl_below_0.02": stable_kl,
            "counters_exact": counters_exact,
            "peak_memory_below_14_gib": peak_bytes <= MAX_PEAK_BYTES,
        },
        "memory": memory,
        "config": {
            "seed": seed,
            "encoder_size": config.encoder_size,
            "recurrent_size": config.recurrent_size,
            "update_epochs": config.update_epochs,
            "num_minibatches": config.num_minibatches,
            "learning_rate": config.learning_rate,
            "observation_size": config.observation_size,
            "action_size": config.action_size,
            "action_head_sizes": list(config.action_head_sizes),
        },
        "task_contract": task_contract(PLAINS_DUEL, "armed"),
        "scene_contract_sha256": agent.scene.contract_sha256,
        "world": dict(agent.scene.world_identity),
        "runtime": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "agent_build_seconds": agent.build_wall_seconds,
            "initialization_seconds": initialization_seconds,
            "compilation_cache": (
                None if cache is None or cache.compilation_cache is None
                else str(cache.compilation_cache)
            ),
        },
    }
    body["sha256"] = _digest(body)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{shape.label}-r{replicate}.json"
    body["report_path"] = str(path.resolve())
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    if not valid:
        raise RuntimeError(f"PPO shape {shape.label} failed its screen gates")
    return body


def select_shape(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the fastest valid candidate by the declared warm median."""

    expected = {f"{envs}x{steps}" for envs, steps in PRODUCTION_SHAPES}
    counts = {
        label: sum(row["shape"]["label"] == label for row in reports)
        for label in expected
    }
    if set(counts) != expected or any(count != REPLICATES for count in counts.values()):
        raise ValueError("selection requires three replicates of every production shape")
    replicate_ids = {
        label: {
            row["measurement"]["replicate"]
            for row in reports
            if row["shape"]["label"] == label
        }
        for label in expected
    }
    if any(ids != set(range(1, REPLICATES + 1)) for ids in replicate_ids.values()):
        raise ValueError("shape replicate identities are incomplete")
    valid = [row for row in reports if row.get("valid")]
    if len(valid) != len(reports):
        raise ValueError("cannot select from a failed shape")
    identities = {_selection_identity(row) for row in reports}
    if len(identities) != 1:
        raise ValueError("shape reports do not share one comparable contract")
    grouped = {
        label: [row for row in valid if row["shape"]["label"] == label]
        for label in expected
    }
    scores = {
        label: float(
            median(
                row["measurement"]["median_warm_transitions_per_second"]
                for row in rows
            )
        )
        for label, rows in grouped.items()
    }
    winner_label = max(scores, key=scores.__getitem__)
    winner = grouped[winner_label][0]
    body = {
        "schema": SELECTION_SCHEMA,
        "selection_rule": "maximum_median_of_three_process_medians",
        "shared_identity": {
            "seed": winner["config"]["seed"],
            "task_contract_sha256": winner["task_contract"]["sha256"],
            "world_pool_semantic_sha256": winner["world"]["pool_semantic_sha256"],
            "jax_version": winner["runtime"]["jax_version"],
            "equal_transition_budget": winner["measurement"][
                "equal_transition_budget"
            ],
        },
        "selected_shape": winner["shape"],
        "candidates": [
            {
                "shape": label,
                "replicate_median_transitions_per_second": [
                    row["measurement"]["median_warm_transitions_per_second"]
                    for row in sorted(
                        rows, key=lambda item: item["measurement"]["replicate"]
                    )
                ],
                "median_warm_transitions_per_second": scores[label],
                "median_aggregate_warm_transitions_per_second": float(
                    median(
                        row["measurement"]["aggregate_warm_transitions_per_second"]
                        for row in rows
                    )
                ),
                "maximum_peak_bytes_in_use": max(
                    row["memory"]["peak_bytes_in_use"] for row in rows
                ),
                "reports": [
                    {
                        "replicate": row["measurement"]["replicate"],
                        "sha256": row["sha256"],
                        "path": row["report_path"],
                    }
                    for row in sorted(
                        rows, key=lambda item: item["measurement"]["replicate"]
                    )
                ],
            }
            for label, rows in sorted(grouped.items())
        ],
    }
    body["sha256"] = _digest(body)
    return body


def _timed_update(step, state, key, shape: ScreenShape):
    started = perf_counter()
    state, metrics = step(state, key)
    jax.block_until_ready(metrics.total_loss)
    elapsed = perf_counter() - started
    values = {
        name: float(np.asarray(jax.device_get(value)))
        for name, value in metrics._asdict().items()
    }
    return state, {
        "wall_seconds": elapsed,
        "transitions_per_second": shape.transitions_per_update / elapsed,
        "finite": bool(all(np.isfinite(value) for value in values.values())),
        "metrics": values,
    }


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest().upper()


def _selection_identity(report: dict[str, Any]) -> tuple[Any, ...]:
    config = report["config"]
    return (
        report["schema"],
        config["seed"],
        config["encoder_size"],
        config["recurrent_size"],
        config["update_epochs"],
        config["num_minibatches"],
        config["learning_rate"],
        config["observation_size"],
        config["action_size"],
        tuple(config["action_head_sizes"]),
        report["task_contract"]["sha256"],
        report["world"]["pool_semantic_sha256"],
        report["runtime"]["jax_version"],
        report["runtime"]["backend"],
        report["measurement"]["equal_transition_budget"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shape", choices=("64x32", "96x32", "64x64"))
    source.add_argument("--select", action="store_true")
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--replicate", type=int, choices=range(1, REPLICATES + 1), default=1)
    parser.add_argument("--measured-transitions", type=int, default=MEASURED_TRANSITIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agents/ppo/artifacts/production-shape-screen-v1"),
    )
    parser.add_argument(
        "--compilation-cache", type=Path, default=default_compilation_cache()
    )
    args = parser.parse_args()
    if args.select:
        reports = [
            json.loads(
                (args.output_dir / f"{envs}x{steps}-r{replicate}.json").read_text()
            )
            for envs, steps in PRODUCTION_SHAPES
            for replicate in range(1, REPLICATES + 1)
        ]
        report = select_shape(reports)
        path = args.output_dir / "selection.json"
        report["report_path"] = str(path.resolve())
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    assert args.shape is not None
    report = run_shape(
        ScreenShape.parse(args.shape),
        output_dir=args.output_dir,
        compilation_cache=args.compilation_cache,
        seed=args.seed,
        measured_transitions=args.measured_transitions,
        replicate=args.replicate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "MEASURED_TRANSITIONS",
    "PRODUCTION_SHAPES",
    "REPLICATES",
    "ScreenShape",
    "run_shape",
    "select_shape",
]
