"""Background evaluation over immutable policy snapshots."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import jax
import numpy as np


ASYNC_EVALUATION_SCHEMA = "arena_asynchronous_evaluation_v1"
Result = TypeVar("Result")


def immutable_host_snapshot(value: Any) -> Any:
    """Copy a JAX pytree to read-only host arrays, detached from training state."""

    def freeze(leaf: Any) -> np.ndarray:
        array = np.asarray(jax.device_get(leaf)).copy()
        array.setflags(write=False)
        return array

    return jax.tree.map(freeze, value)


@dataclass(frozen=True, slots=True)
class EvaluationTicket(Generic[Result]):
    milestone: int
    snapshot: Any
    future: Future[Result]


class AsyncEvaluationQueue(Generic[Result]):
    """Evaluate frozen snapshots without exposing live trainer state or RNG."""

    def __init__(
        self,
        evaluator: Callable[[int, Any], Result],
        *,
        thread_name: str = "arena-evaluation",
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._evaluator = evaluator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._tickets: list[EvaluationTicket[Result]] = []
        self._closed = False

    def submit(self, milestone: int, value: Any) -> EvaluationTicket[Result]:
        if self._closed:
            raise RuntimeError("evaluation queue is closed")
        if isinstance(milestone, bool) or not isinstance(milestone, int) or milestone < 0:
            raise ValueError("evaluation milestone must be a nonnegative integer")
        if self._tickets and milestone <= self._tickets[-1].milestone:
            raise ValueError("evaluation milestones must be strictly increasing")
        snapshot = immutable_host_snapshot(value)
        ticket = EvaluationTicket(
            milestone,
            snapshot,
            self._executor.submit(self._evaluator, milestone, snapshot),
        )
        self._tickets.append(ticket)
        return ticket

    def results(self) -> tuple[tuple[int, Result], ...]:
        """Wait in milestone order; callers should invoke this after training."""

        return tuple((ticket.milestone, ticket.future.result()) for ticket in self._tickets)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)

    def describe(self) -> dict[str, Any]:
        return {
            "schema": ASYNC_EVALUATION_SCHEMA,
            "mode": "single_background_worker",
            "parameter_source": "immutable_read_only_host_snapshot",
            "optimizer_state_shared": False,
            "environment_state_shared": False,
            "recurrent_state_shared": False,
            "rng_shared": False,
            "training_consumes_results": False,
            "publication_waits_for_results": True,
        }

    def __enter__(self) -> AsyncEvaluationQueue[Result]:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "ASYNC_EVALUATION_SCHEMA",
    "AsyncEvaluationQueue",
    "EvaluationTicket",
    "immutable_host_snapshot",
]
