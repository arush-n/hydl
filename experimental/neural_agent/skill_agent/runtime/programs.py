"""Running activities: a program of steps becomes a stream of requests and reads the status stream back.

``experimental/neural_agent/docs/SYSTEM_ONE.md`` section 9.9 (behaviour programs) and the catalogue's program syntax:
stages run in order; the steps of a stage run at the same time; ``(for each ...)`` repeats a step over targets the
caller supplies. These are the sequence, parallel and repeat nodes of behaviour trees (Colledanchise and Ogren 2018) and
the ordered methods of HTN planning (Erol, Hendler and Nau 1994), kept to the three forms the catalogue uses.

Who fills the arguments. A program names *which* verbs run and in what order; *where* (a position, an
entity, a cell) comes from the caller's state through a binder::

    binder(step, verb, iteration) -> RequestArgs | None

For a plain step the binder returns its arguments (None skips the step). For a repeating step it is
called with iteration 0, 1, 2, ... until it returns None. The middle layer's binder reads the belief and
world memory; the tests use a binder built from each verb's schema.

The runner, per tick::

    requests()      the live requests: one per running step of the current stage (a repeating step has
                    one iteration live at a time). Stage k + 1 starts when every step of stage k is done.
    observe(s)      reads the status stream: done advances that step; rejected or aborted fails the
                    program with the step and the reason; accepted, running and deferred wait.
    state           "running", "done" or "failed"; ``reason`` names the failing step.

Re-sampling a failed program is the middle layer's choice (section 9.9), not the runner's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from experimental.neural_agent import requests as middle
from experimental.neural_agent.skill_agent.contract.catalogue import ActivityStep
from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary

Binder = Callable[[ActivityStep, str, int], "middle.RequestArgs | None"]


@dataclass
class _StepRun:
    step: ActivityStep
    verb: str
    iteration: int = 0
    live: middle.ActionRequest | None = None
    finished: bool = False


@dataclass
class ProgramRunner:
    """Run one program for one agent through a library's request path.

    Args:
        library: The skill library (verbs and programs).
        program: The program id.
        agent: The agent the requests are for.
        binder: Supplies each step's arguments (see the module docstring).
        first_request_id: The first request id to use; ids are monotonic per agent.
        urgency, confidence, priority: The arbitration fields of every request the program issues.
    """

    library: SkillLibrary
    program: str
    agent: int
    binder: Binder
    first_request_id: int = 0
    urgency: float = 0.5
    confidence: float = 0.5
    priority: float = 1.0
    state: str = "running"
    reason: str = ""
    trace: list[tuple[int, str, int, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stages = self.library.expand(self.program)
        self._stage = -1
        self._runs: list[_StepRun] = []
        self._next_id = self.first_request_id
        self._tick = 0

    @property
    def stage(self) -> int:
        """The index of the running stage in the expanded program."""

        return self._stage

    def _issue(self, run: _StepRun) -> None:
        args = self.binder(run.step, run.verb, run.iteration)
        if args is None:
            run.finished = True
            run.live = None
            return
        rid = self._next_id
        self._next_id += 1
        run.live = self.library.request(run.verb, agent=self.agent, request_id=rid, tick=self._tick, args=args,
                                        urgency=self.urgency, confidence=self.confidence, priority=self.priority,
                                        program_step=max(0, self._stage))

    def _advance(self) -> None:
        while self.state == "running" and all(run.finished for run in self._runs):
            self._stage += 1
            if self._stage >= len(self._stages):
                self.state = "done"
                return
            self._runs = [_StepRun(step, verb) for step, verb in self._stages[self._stage]]
            for run in self._runs:
                self._issue(run)

    def requests(self, tick: int) -> tuple[middle.ActionRequest, ...]:
        """The program's live requests on ``tick`` (start or advance first)."""

        self._tick = tick
        self._advance()
        return tuple(run.live for run in self._runs if run.live is not None)

    def observe(self, statuses: Iterable[middle.RequestStatus]) -> None:
        """Read the status stream: done advances a step; rejected or aborted fails the program."""

        by_id = {run.live.request_id: run for run in self._runs if run.live is not None}
        for status in statuses:
            run = by_id.get(status.request_id)
            if run is None or status.agent != self.agent:
                continue
            self.trace.append((status.tick, run.step.text, status.request_id, status.kind))
            if status.kind == "done":
                if run.step.repeat:
                    run.iteration += 1
                    self._issue(run)
                else:
                    run.finished, run.live = True, None
            elif status.kind in ("rejected", "aborted"):
                self.state = "failed"
                self.reason = f"step {run.step.text!r} ({run.verb}): {status.reason}"
                for other in self._runs:
                    other.live = None
                return
        self._advance()


__all__ = ["Binder", "ProgramRunner"]
