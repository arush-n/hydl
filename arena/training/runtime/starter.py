"""One-call recursive train, evaluate, audit, tune, and promote loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Protocol, Sequence

import jax

from arena.params import validate_params
from arena.training.runtime.audit import (
    AuditFlag,
    RewardAudit,
    TrialResult,
    audit_reward,
)
from arena.training.runtime.contracts import TrainingContract
from arena.training.runtime.execution import recommended_replicas
from arena.training.runtime.monitor import TrainingMonitor
from arena.training.runtime.tuning import propose_parameters, trial_utility


RECURSIVE_TRAINING_SCHEMA = "arena_recursive_training_run_v2"


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    cycle: int
    replicas: int
    vram_fraction: float
    output_dir: Path


class TrainingAdapter(Protocol):
    """Strategy adapter supplied by an agent, never by a task-name branch."""

    contract: TrainingContract
    initial_parameters: Mapping[str, Any]
    output_dir: Path

    def train_and_evaluate(
        self,
        parameters: Sequence[Mapping[str, Any]],
        request: ExecutionRequest,
    ) -> Sequence[TrialResult]: ...

    def promote(self, result: TrialResult) -> Any: ...


@dataclass(frozen=True, slots=True)
class StopPolicy:
    """Generic rollback/stop-loss gates over normalized declared utility."""

    minimum_utility_improvement: float = 0.01
    maximum_utility_drawdown: float = 0.50
    patience_cycles: int = 3
    maximum_consecutive_rejected_cycles: int = 2
    maximum_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_utility_improvement < 0.0:
            raise ValueError("minimum_utility_improvement must be nonnegative")
        if self.maximum_utility_drawdown < 0.0:
            raise ValueError("maximum_utility_drawdown must be nonnegative")
        for name in ("patience_cycles", "maximum_consecutive_rejected_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_wall_seconds is not None and self.maximum_wall_seconds <= 0.0:
            raise ValueError("maximum_wall_seconds must be positive or None")

    def describe(self) -> dict[str, Any]:
        return {
            "minimum_utility_improvement": self.minimum_utility_improvement,
            "maximum_utility_drawdown": self.maximum_utility_drawdown,
            "patience_cycles": self.patience_cycles,
            "maximum_consecutive_rejected_cycles": (
                self.maximum_consecutive_rejected_cycles
            ),
            "maximum_wall_seconds": self.maximum_wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class RecursiveTrainingConfig:
    cycles: int = 4
    candidates_per_cycle: int = 2
    maximum_replicas: int = 8
    vram_fraction: float = 0.90
    seed: int = 570057
    stop: StopPolicy = field(default_factory=StopPolicy)

    def __post_init__(self) -> None:
        for name in ("cycles", "candidates_per_cycle", "maximum_replicas"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.vram_fraction <= 1.0:
            raise ValueError("vram_fraction must be in (0, 1]")
        if not isinstance(self.stop, StopPolicy):
            raise TypeError("stop must be a StopPolicy")


def train(training_type: str, agent: Any) -> dict[str, Any]:
    """Train an agent recursively; callers supply only strategy and agent."""

    if not isinstance(training_type, str) or not training_type:
        raise ValueError("training_type must be a nonempty string")
    factory = getattr(agent, "training_adapter", None)
    if not callable(factory):
        raise TypeError("agent must expose training_adapter(training_type)")
    adapter = factory(training_type)
    contract = adapter.contract
    if contract.strategy != training_type:
        raise ValueError("agent adapter returned a different training strategy")
    config = getattr(adapter, "recursive_config", RecursiveTrainingConfig())
    if not isinstance(config, RecursiveTrainingConfig):
        raise TypeError("adapter recursive_config must be RecursiveTrainingConfig")
    if jax.default_backend() != "gpu":
        raise RuntimeError("recursive training requires a JAX GPU backend")

    output = Path(adapter.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(output)
    state_path = output / "recursive-training.json"
    history = _load_history(state_path, contract)
    initial = validate_params(contract.parameter_space, adapter.initial_parameters)
    memory = dict(jax.devices()[0].memory_stats() or {})
    bytes_limit = int(memory.get("bytes_limit", 1))
    started = monotonic()
    best_utility = max(
        (
            trial_utility(contract, result)
            for result, audit, _ in history
            if audit.promotable
        ),
        default=-float("inf"),
    )
    stale_cycles = 0
    rejected_cycles = 0
    stop_reason = None
    monitor.emit(
        "running",
        {
            "strategy": training_type,
            "contract_sha256": contract.sha256,
            "resumed_trials": len(history),
        },
        phase="train",
    )

    for cycle in range(len({row[2] for row in history}), config.cycles):
        peak = max(
            (row.peak_vram_bytes or 0 for row, _, _ in history),
            default=0,
        )
        replicas = recommended_replicas(
            bytes_limit=bytes_limit,
            measured_peak_bytes=peak or None,
            maximum=config.maximum_replicas,
            vram_fraction=config.vram_fraction,
        )
        if not history:
            candidates = (initial,)
            proposal = None
        else:
            proposal = propose_parameters(
                contract,
                [(result, audit) for result, audit, _ in history],
                count=config.candidates_per_cycle,
                seed=config.seed + cycle,
            )
            candidates = proposal.parameters
        request = ExecutionRequest(cycle, replicas, config.vram_fraction, output)
        monitor.emit(
            "cycle_started",
            {"replicas": replicas, "candidates": len(candidates)},
            cycle=cycle,
            phase="train",
        )
        results = tuple(adapter.train_and_evaluate(candidates, request))
        if len(results) != len(candidates):
            raise RuntimeError("training adapter returned the wrong result count")
        for parameters, result in zip(candidates, results, strict=True):
            if dict(parameters) != dict(result.parameters):
                raise RuntimeError("training result parameters differ from its request")
            audit = audit_reward(
                contract,
                result,
                [previous for previous, _, _ in history],
            )
            if (
                result.peak_vram_bytes is not None
                and result.peak_vram_bytes > bytes_limit * config.vram_fraction
            ):
                audit = RewardAudit(
                    audit.flags
                    + (
                        AuditFlag(
                            "vram_budget_exceeded",
                            "error",
                            f"{result.peak_vram_bytes}/{bytes_limit} bytes",
                        ),
                    ),
                    audit.sample_count,
                    audit.reward_progress_correlation,
                )
            history.append((result, audit, cycle))
            requested_stop = next(
                (flag for flag in audit.flags if flag.stop_training), None
            )
            if requested_stop is not None and stop_reason is None:
                stop_reason = requested_stop.code
            monitor.emit(
                "trial_completed" if audit.promotable else "trial_rejected",
                {
                    "parameters": dict(result.parameters),
                    "metrics": dict(result.metrics),
                    "stop_loss_flags": dict(result.stop_loss_flags),
                    "audit": audit.describe(),
                },
                cycle=cycle,
                phase="evaluate",
            )
        _write_state(state_path, contract, config, history, proposal)

        admitted_cycle = [
            result
            for result, audit, trial_cycle in history
            if trial_cycle == cycle and audit.promotable
        ]
        if stop_reason is not None:
            pass
        elif not admitted_cycle:
            rejected_cycles += 1
            if rejected_cycles >= config.stop.maximum_consecutive_rejected_cycles:
                stop_reason = "consecutive_rejected_cycles"
        else:
            rejected_cycles = 0
            cycle_best = max(trial_utility(contract, row) for row in admitted_cycle)
            if cycle_best < best_utility - config.stop.maximum_utility_drawdown:
                stop_reason = "utility_stop_loss"
            elif cycle_best >= best_utility + config.stop.minimum_utility_improvement:
                best_utility = cycle_best
                stale_cycles = 0
            else:
                best_utility = max(best_utility, cycle_best)
                stale_cycles += 1
                if stale_cycles >= config.stop.patience_cycles:
                    stop_reason = "improvement_patience_exhausted"
        if (
            stop_reason is None
            and config.stop.maximum_wall_seconds is not None
            and monotonic() - started >= config.stop.maximum_wall_seconds
        ):
            stop_reason = "wall_time_stop_loss"
        if stop_reason is not None:
            monitor.emit(
                "stopped",
                {"reason": stop_reason, "incumbent_utility": best_utility},
                cycle=cycle,
                phase="stopped",
            )
            break

    admitted = [(row, audit) for row, audit, _ in history if audit.promotable]
    if not admitted:
        raise RuntimeError("every candidate failed reward/validity admission")
    best, best_audit = max(admitted, key=lambda item: trial_utility(contract, item[0]))
    promoted = adapter.promote(best)
    report = _state_document(contract, config, history, None)
    report["stop_reason"] = stop_reason or "cycle_budget_complete"
    report["selection"] = {
        "parameters": dict(best.parameters),
        "utility": trial_utility(contract, best),
        "audit": best_audit.describe(),
        "promoted": promoted,
    }
    state_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    monitor.emit(
        "completed",
        {
            "stop_reason": report["stop_reason"],
            "selection": report["selection"],
        },
        phase="completed",
    )
    return report


def _state_document(contract, config, history, proposal):
    return {
        "schema": RECURSIVE_TRAINING_SCHEMA,
        "contract": {**contract.describe(), "sha256": contract.sha256},
        "config": {
            "cycles": config.cycles,
            "candidates_per_cycle": config.candidates_per_cycle,
            "maximum_replicas": config.maximum_replicas,
            "vram_fraction": config.vram_fraction,
            "seed": config.seed,
            "stop": config.stop.describe(),
        },
        "last_proposal": None if proposal is None else proposal.describe(),
        "trials": [
            {
                "cycle": cycle,
                "parameters": dict(result.parameters),
                "metrics": dict(result.metrics),
                "artifact": None if result.artifact is None else str(result.artifact),
                "checkpoint": None
                if result.checkpoint is None
                else str(result.checkpoint),
                "peak_vram_bytes": result.peak_vram_bytes,
                "stop_loss_flags": dict(result.stop_loss_flags),
                "audit": audit.describe(),
            }
            for result, audit, cycle in history
        ],
    }


def _write_state(path, contract, config, history, proposal):
    value = _state_document(contract, config, history, proposal)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_history(path, contract):
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != RECURSIVE_TRAINING_SCHEMA:
        raise ValueError("recursive training state schema differs")
    if value.get("contract", {}).get("sha256") != contract.sha256:
        raise ValueError("recursive training state contract differs")
    rows = []
    for item in value.get("trials", []):
        audit_value = item["audit"]
        audit = RewardAudit(
            tuple(AuditFlag(**flag) for flag in audit_value["flags"]),
            int(audit_value["sample_count"]),
            audit_value["reward_progress_correlation"],
        )
        result = TrialResult(
            item["parameters"],
            item["metrics"],
            artifact=None if item["artifact"] is None else Path(item["artifact"]),
            checkpoint=(
                None if item["checkpoint"] is None else Path(item["checkpoint"])
            ),
            peak_vram_bytes=item["peak_vram_bytes"],
            stop_loss_flags=item.get("stop_loss_flags", {}),
        )
        rows.append((result, audit, int(item["cycle"])))
    return rows


__all__ = [
    "ExecutionRequest",
    "RECURSIVE_TRAINING_SCHEMA",
    "RecursiveTrainingConfig",
    "StopPolicy",
    "TrainingAdapter",
    "train",
]
