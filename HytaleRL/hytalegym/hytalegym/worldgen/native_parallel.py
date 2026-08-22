"""Bounded parallel collection across independent native Hytale environments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any

from hytalegym.worldgen.native_npc_traces import (
    NATIVE_NPC_TRACE_MAX_ACTIVE,
    NATIVE_NPC_TRACE_MAX_CAPACITY,
    NATIVE_NPC_TRACE_MAX_DRAIN,
    concatenate_native_npc_traces,
)


@dataclass(frozen=True, slots=True)
class NativeTraceTarget:
    npc_uuid: Any
    expected_role: str | None = None


@dataclass(frozen=True, slots=True)
class NativeTraceJob:
    """One reset environment and the NPCs to record within it."""

    environment: Any
    targets: tuple[NativeTraceTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets:
            raise ValueError("a native trace job requires at least one target")
        identities = tuple(str(target.npc_uuid) for target in self.targets)
        if len(self.targets) > NATIVE_NPC_TRACE_MAX_ACTIVE:
            raise ValueError(
                f"one world supports at most {NATIVE_NPC_TRACE_MAX_ACTIVE} traces"
            )
        if len(identities) != len(set(identities)):
            raise ValueError("native trace targets must be unique within one world")


@dataclass(frozen=True, slots=True)
class NativeTraceResult:
    captures: tuple[Any, ...]
    rpc_steps: int
    engine_ticks: int
    terminated: bool
    truncated: bool
    engine_seconds: float
    transfer_seconds: float

    @property
    def engine_ticks_per_second(self) -> float:
        return self.engine_ticks / max(self.engine_seconds, 1.0e-12)


@dataclass(frozen=True, slots=True)
class NativeParallelCollection:
    results: tuple[NativeTraceResult, ...]
    elapsed_seconds: float
    workers: int

    @property
    def engine_ticks(self) -> int:
        return sum(result.engine_ticks for result in self.results)

    @property
    def trace_rows(self) -> int:
        return sum(
            getattr(capture, "emitted_count", 0)
            for result in self.results
            for capture in result.captures
        )

    @property
    def engine_ticks_per_second(self) -> float:
        return self.engine_ticks / max(self.elapsed_seconds, 1.0e-12)

    @property
    def trace_rows_per_second(self) -> float:
        return self.trace_rows / max(self.elapsed_seconds, 1.0e-12)

    @property
    def mean_world_engine_ticks_per_second(self) -> float:
        seconds = sum(result.engine_seconds for result in self.results)
        return self.engine_ticks / max(seconds, 1.0e-12)

    @property
    def transfer_seconds(self) -> float:
        return sum(result.transfer_seconds for result in self.results)


def collect_native_npc_traces(
    jobs: tuple[NativeTraceJob, ...] | list[NativeTraceJob],
    *,
    engine_ticks: int,
    capacity: int | None = None,
    max_workers: int | None = None,
) -> NativeParallelCollection:
    """Trace multiple worlds concurrently with one socket owner per worker.

    ``ticks_per_step`` batches engine ticks inside Java, while this bounded pool
    overlaps independent worlds. Exact tick counts require divisibility.
    """

    jobs = tuple(jobs)
    if not jobs:
        raise ValueError("at least one native trace job is required")
    if (
        isinstance(engine_ticks, bool)
        or not isinstance(engine_ticks, int)
        or engine_ticks < 1
    ):
        raise ValueError("engine_ticks must be a positive integer")
    workers = _workers(len(jobs), max_workers)
    requested_capacity = (
        min(engine_ticks, NATIVE_NPC_TRACE_MAX_CAPACITY)
        if capacity is None
        else capacity
    )
    if (
        isinstance(requested_capacity, bool)
        or not isinstance(requested_capacity, int)
        or not 1 <= requested_capacity <= NATIVE_NPC_TRACE_MAX_CAPACITY
    ):
        raise ValueError("capacity must fit the native trace limit")
    for job in jobs:
        ticks_per_step = getattr(job.environment, "ticks_per_step", None)
        if (
            isinstance(ticks_per_step, bool)
            or not isinstance(ticks_per_step, int)
            or ticks_per_step < 1
            or engine_ticks % ticks_per_step
            or requested_capacity < ticks_per_step
        ):
            raise ValueError(
                "engine_ticks must be divisible by every ticks_per_step and capacity"
                " must cover one step"
            )

    started = perf_counter()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="hytalerl-trace"
    ) as pool:
        results = tuple(
            pool.map(
                lambda job: _collect(job, engine_ticks, requested_capacity),
                jobs,
            )
        )
    return NativeParallelCollection(results, perf_counter() - started, workers)


def _collect(
    job: NativeTraceJob, engine_ticks: int, capacity: int
) -> NativeTraceResult:
    env = job.environment
    trace_ids: list[Any] = []
    parts: list[list[Any]] = [[] for _ in job.targets]
    try:
        for target in job.targets:
            started = env.start_npc_trace(
                target.npc_uuid,
                expected_role=target.expected_role,
                capacity=capacity,
            )
            trace_ids.append(started.trace_uuid)
        rpc_steps = engine_ticks // env.ticks_per_step
        drain_limit = min(capacity, NATIVE_NPC_TRACE_MAX_DRAIN)
        terminated = truncated = False
        completed = 0
        buffered = [0] * len(trace_ids)
        engine_seconds = transfer_seconds = 0.0
        for step in range(rpc_steps):
            step_started = perf_counter()
            _, _, terminated, truncated, _ = env.step_native_npcs()
            engine_seconds += perf_counter() - step_started
            completed += env.ticks_per_step
            buffered = [value + env.ticks_per_step for value in buffered]
            if terminated or truncated:
                break
            if step + 1 < rpc_steps and any(
                value + env.ticks_per_step > capacity for value in buffered
            ):
                transfer_started = perf_counter()
                for index, trace_uuid in enumerate(trace_ids):
                    batch = env.poll_npc_trace(
                        trace_uuid,
                        max_frames=drain_limit,
                    )
                    parts[index].append(batch)
                    buffered[index] = max(
                        0, buffered[index] - batch.emitted_count
                    )
                transfer_seconds += perf_counter() - transfer_started
        while any(value > drain_limit for value in buffered):
            transfer_started = perf_counter()
            emitted = 0
            for index, trace_uuid in enumerate(trace_ids):
                if buffered[index] <= drain_limit:
                    continue
                batch = env.poll_npc_trace(
                    trace_uuid,
                    max_frames=drain_limit,
                )
                parts[index].append(batch)
                emitted += batch.emitted_count
                buffered[index] = max(
                    0, buffered[index] - batch.emitted_count
                )
            transfer_seconds += perf_counter() - transfer_started
            if emitted == 0:
                break
        transfer_started = perf_counter()
        for index, trace_uuid in enumerate(trace_ids):
            parts[index].append(
                env.stop_npc_trace(
                    trace_uuid,
                    max_frames=drain_limit,
                )
            )
        transfer_seconds += perf_counter() - transfer_started
        captures = tuple(concatenate_native_npc_traces(rows) for rows in parts)
        trace_ids.clear()
        return NativeTraceResult(
            captures,
            completed // env.ticks_per_step,
            completed,
            terminated,
            truncated,
            engine_seconds,
            transfer_seconds,
        )
    finally:
        for trace_uuid in trace_ids:
            try:
                env.stop_npc_trace(
                    trace_uuid,
                    max_frames=min(capacity, NATIVE_NPC_TRACE_MAX_DRAIN),
                )
            except Exception:
                pass


def _workers(count: int, requested: int | None) -> int:
    default = min(count, max(1, min(4, (os.cpu_count() or 2) // 2)))
    if requested is None:
        return default
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError("max_workers must be an integer")
    if not 1 <= requested <= count:
        raise ValueError("max_workers must be between one and the job count")
    return requested


__all__ = [
    "NativeParallelCollection",
    "NativeTraceJob",
    "NativeTraceResult",
    "NativeTraceTarget",
    "collect_native_npc_traces",
]
