"""Executive backends: the seam protocol, the usage record, and two deterministic baselines (ADR-001 P5).

Formulas
--------
OfflinePlanBackend, with plan P, and current state E of version v, at tick t::

    next = ExecutiveState(version=v + 1, issued_tick=t, goals=P.goals,
                          budget=P.budget, facts=P.facts)

RuleBackend, with current goals g_0 .. g_{k-1} at version v, at tick t::

    rank   = goal slots sorted by (-priority, slot); rank[0] is the leader
    failed = some GoalReport r has r.skill_id == g_{rank[0]}.skill_id,
             r.outcome in {FAILED, TIMEOUT}, and r.executive_version == v
    if failed and k > 1:
        rotated = rank[1:] + rank[:1]                  # the failed leader goes last
        values  = [priority of slot i for i in rank]   # the same values, in descending order
        new slot r holds goal g_{rotated[r]} with priority values[r]
    version = v + 1, issued_tick = t

The next goal therefore leads, the failed goal gets the lowest priority value, and no
value is invented. Ties rank by slot. The rule runs once per call. A report with an
older executive_version cannot rotate the ranking a second time.

Citations: ADR-001 P5 ("three deterministic baselines: fixed cadence, per-transition,
offline-plan-only"); ADR-001 D3 (E_t is versioned, and each decision reports the version
it served). The RuleBackend rule is an engineering choice made here, not a published result.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple, Protocol, Sequence, runtime_checkable

from experimental.neural_agent.contracts import ExecutiveState, GoalReport, GoalSpec, SkillOutcome
from experimental.neural_agent.executive.reports import EventRecord

FAILURE_OUTCOMES: frozenset[SkillOutcome] = frozenset({SkillOutcome.FAILED, SkillOutcome.TIMEOUT})


class CallUsage(NamedTuple):
    """Usage of one executive call: the requested knobs, and the token counts from the API.

    Attributes:
        model: Requested model id, or "none" for a non-LLM baseline.
        effort: Requested effort level, or "none".
        input_tokens: Uncached input tokens.
        output_tokens: Output tokens.
        cache_read_tokens: Input tokens served from the prompt cache.
        cache_creation_tokens: Input tokens written to the prompt cache.
    """

    model: str
    effort: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


NO_USAGE = CallUsage(model="none", effort="none")


class ExecutiveBackend(Protocol):
    """Anything that proposes the next executive state.

    ``propose`` must return a state whose version is ``current.version + 1``. The
    store rejects any other version.
    """

    def propose(
        self,
        current: ExecutiveState,
        reports: Sequence[GoalReport | EventRecord],
        tick: int,
    ) -> ExecutiveState:
        """Return the next E_t.

        Args:
            current: The state in force.
            reports: The reports drained since the last call, oldest first.
            tick: The simulation tick of this call.

        Returns:
            The proposed ExecutiveState.
        """


@runtime_checkable
class UsageReportingBackend(Protocol):
    """A backend that also returns the usage of each call, so the harness can log it.

    Attributes:
        model: The model id the backend requests.
        effort: The effort level the backend requests.
    """

    model: str
    effort: str

    def propose_with_usage(
        self,
        current: ExecutiveState,
        reports: Sequence[GoalReport | EventRecord],
        tick: int,
    ) -> tuple[ExecutiveState, CallUsage]:
        """Return ``(state, usage)`` for one call. The same contract as ``propose``."""


def _ranking(goals: Sequence[GoalSpec]) -> list[int]:
    """Return the goal slots in descending priority, with ties in slot order."""

    return sorted(range(len(goals)), key=lambda slot: (-goals[slot].priority, slot))


class OfflinePlanBackend:
    """Return the same plan's goals on every call. Only the version and the tick advance.

    This is ADR-001 P5 baseline 3, the offline-plan-only arm.

    Args:
        plan: The ExecutiveState whose goals, budget and facts are kept. Its version is
            ignored, because the store assigns versions.

    Raises:
        TypeError: If ``plan`` is not an ExecutiveState.
    """

    def __init__(self, plan: ExecutiveState) -> None:
        if not isinstance(plan, ExecutiveState):
            raise TypeError(f"plan must be an ExecutiveState, got {type(plan).__name__}")
        self._plan = plan

    def propose(
        self,
        current: ExecutiveState,
        reports: Sequence[GoalReport | EventRecord],
        tick: int,
    ) -> ExecutiveState:
        """Return the plan's goals at version ``current.version + 1`` and issued tick ``tick``."""

        return ExecutiveState(
            version=current.version + 1,
            issued_tick=tick,
            goals=self._plan.goals,
            budget=self._plan.budget,
            facts=self._plan.facts,
        )


class RuleBackend:
    """Deterministic rule: when the leading goal failed or timed out, rotate the ranking.

    The rule has no model and no randomness. If the leader failed or timed out under the
    current version, the next goal leads and the failed goal moves to the back. Otherwise
    the goals are kept. Each call returns version ``current.version + 1``.
    """

    def propose(
        self,
        current: ExecutiveState,
        reports: Sequence[GoalReport | EventRecord],
        tick: int,
    ) -> ExecutiveState:
        """Apply the rotation rule and return the next state. See the module docstring."""

        goals = current.goals
        if len(goals) > 1 and _leader_failed(current, reports):
            goals = _rotate_ranking(goals)
        return ExecutiveState(
            version=current.version + 1,
            issued_tick=tick,
            goals=goals,
            budget=current.budget,
            facts=current.facts,
        )


def _leader_failed(current: ExecutiveState, reports: Sequence[GoalReport | EventRecord]) -> bool:
    """Return True when a current-version report says the leading goal failed or timed out."""

    leader = current.goals[_ranking(current.goals)[0]].skill_id
    return any(
        isinstance(item, GoalReport)
        and item.skill_id == leader
        and item.outcome in FAILURE_OUTCOMES
        and item.executive_version == current.version
        for item in reports
    )


def _rotate_ranking(goals: Sequence[GoalSpec]) -> tuple[GoalSpec, ...]:
    """Rotate the ranking by one, and reassign the same priority values in descending order."""

    order = _ranking(goals)
    rotated = order[1:] + order[:1]
    values = [goals[slot].priority for slot in order]
    return tuple(
        replace(goals[slot], priority=values[position]) for position, slot in enumerate(rotated)
    )
