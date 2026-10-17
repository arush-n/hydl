"""Correspondence: how the middle layer reacts to the lower layer's status stream (SYSTEM_ONE.md 9.10).

Every cadence tick (the host decision cadence, 11 ticks = 0.495 s at 0.045 s per tick), the middle
layer reads, for each agent's current program step, the latest status of every request it has in
flight for that step, and :func:`correspond` turns them into typed actions for the program executor
(section 9.9). The functions here are pure and deterministic: no clock, no randomness, no state. The
executor carries what must persist between cadence ticks: the step's earlier rejections, its hold
time, and its own status history when the scratchpad's bounded stream is too short. It also applies
the actions; re-sampling a step is its job.

Actions (typed records; ``kind`` is the rule's name in section 9.10)::

    ResampleStep(reason, keep_verb)   re-sample the current step. keep_verb True: keep the verb and
                                      fill its args again from the state ("fix the args"); False:
                                      choose another verb
    RaiseUrgency(request_id, by)      raise a deferred request's urgency by ``by``
    Drop(request_id)                  withdraw a deferred request
    Advance()                         the step is done: advance the program
    InformationRequest(reason)        ask the LLM (the section 8.13 gaps): the step keeps being rejected

Rules, by precedence. The first rule that applies decides; only rule 5 can return several actions::

    1 rejected      some request's latest status is rejected. With r = step_rejections + the number
                    of rejected requests: r >= REPEATED_REJECTIONS -> InformationRequest; otherwise
                    ResampleStep with keep_verb = (every rejection code is in ARGUMENT_CODES)
    2 unsuccessful  some request is done with an outcome other than COMPLETED -> ResampleStep,
                    keep_verb False (section 9.9: a step is re-sampled when its skill reports
                    failure); otherwise some request is aborted -> ResampleStep, keep_verb True (the
                    lower layer withdrew a valid request)
    3 done          every request is done (COMPLETED) -> Advance
    4 stalled       a running request whose newest progress differs by at most STALL_EPSILON, on
                    every signal it shares, from its newest granted status at or before
                    tick - STALL_TICKS, with every status from that one on granted -> ResampleStep,
                    keep_verb False
    5 deferred      a request deferred continuously since tick d, with tick - d > hold_ticks:
                    urgency < 1 -> RaiseUrgency(id, min(URGENCY_STEP, 1 - urgency)); urgency = 1 ->
                    Drop(id). One action per such request, in id order
    6 otherwise     () : keep acting

"Latest status" is the last status of that request in ``statuses`` at or before ``tick``. A
request with no status yet is pending and triggers nothing. "Granted" means accepted or running.
The newest rejection (by tick, then id) gives the reason of rules 1 and 2. The deferral start ``d``
is the earliest status of the trailing run of deferred statuses in the history given. If the history
is shorter than the run, ``d`` is later than the true start, so rule 5 can fire late but never
early. Rule 4 needs a status at or before tick - STALL_TICKS, so it can miss a stall but never
invent one. The scratchpad keeps the newest ``shared_state.STATUS_HISTORY`` statuses. An executor
that needs a longer reach (holds of 4 or 8 s, many agents) passes its own history.

Thresholds (deterministic, documented):

    REPEATED_REJECTIONS = 3   the third rejection of one step asks the LLM. The first two re-sample
                              (an argument fix, then one more try), because LLM calls are rationed
                              (ComputeBudget.executive_calls_per_minute)
    STALL_TICKS = 45          ceil(2.0 s / 0.045 s): the longest minimum commitment among the
                              proof-of-concept skills (evader, 2.0 s), so a stall is never called
                              while a skill's own contract still commits to it
    STALL_EPSILON = 1e-3      absolute, in the signal's own units (blocks for a distance)
    URGENCY_STEP = 0.25       four raises take any urgency to 1, and the next over-hold cadence tick
                              drops the request. A deferred request is therefore granted or
                              withdrawn within 5 cadence ticks after its hold is exceeded
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence

from experimental.neural_agent.contracts import SkillOutcome, _non_negative_int
from experimental.neural_agent.requests import ActionRequest, RequestStatus, rejection_code

REPEATED_REJECTIONS: int = 3
"""Rejections of one step at which the middle layer asks the LLM instead of re-sampling."""
STALL_TICKS: int = 45
"""Window of the stall rule, in ticks: ceil(2.0 s / 0.045 s)."""
STALL_EPSILON: float = 1.0e-3
"""Largest change of a progress signal over STALL_TICKS that still counts as no progress."""
URGENCY_STEP: float = 0.25
"""Urgency added to a request deferred beyond its step's hold time, per cadence tick."""
ARGUMENT_CODES: frozenset[str] = frozenset(
    {"expired", "missing_argument", "non_finite_argument", "out_of_bounds", "entity_unknown", "predecessor_not_done"}
)
"""Rejection codes fixed by filling the args again (keep the verb). Every other code changes the verb."""

_GRANTED: frozenset[str] = frozenset({"accepted", "running"})


@dataclass(frozen=True)
class ResampleStep:
    """Re-sample the current program step; keep the verb (fix the args) or choose another."""

    reason: str
    keep_verb: bool
    kind: ClassVar[str] = "resample_step"


@dataclass(frozen=True)
class RaiseUrgency:
    """Raise the urgency of deferred request ``request_id`` by ``by`` (urgency stays at most 1)."""

    request_id: int
    by: float
    kind: ClassVar[str] = "raise_urgency"


@dataclass(frozen=True)
class Drop:
    """Withdraw deferred request ``request_id``."""

    request_id: int
    kind: ClassVar[str] = "drop"


@dataclass(frozen=True)
class Advance:
    """Every request of the step is done: advance the program to its next step."""

    kind: ClassVar[str] = "advance"


@dataclass(frozen=True)
class InformationRequest:
    """Ask the LLM: the step was rejected REPEATED_REJECTIONS times (section 8.13 gaps)."""

    reason: str
    kind: ClassVar[str] = "information_request"


CorrespondenceAction = ResampleStep | RaiseUrgency | Drop | Advance | InformationRequest
"""Any of the five actions."""


def _newest(statuses: Sequence[RequestStatus]) -> RequestStatus:
    """Return the newest status by (tick, request_id)."""

    return max(statuses, key=lambda status: (status.tick, status.request_id))


def _stalled(history: Sequence[RequestStatus], cutoff: int) -> int | None:
    """Return the reference tick when the newest status shows no progress since ``cutoff``, else None.

    The reference is the newest status at or before ``cutoff``. It and every status after it must be
    granted, and it and the newest must both carry progress with at least one shared signal.
    """

    newest = history[-1]
    for index in range(len(history) - 1, -1, -1):
        if history[index].tick <= cutoff:
            break
    else:
        return None
    reference = history[index]
    if any(status.kind not in _GRANTED for status in history[index:]):
        return None
    if reference.progress is None or newest.progress is None:
        return None
    before, after = dict(reference.progress), dict(newest.progress)
    shared = sorted(set(before) & set(after))
    if not shared or any(abs(after[name] - before[name]) > STALL_EPSILON for name in shared):
        return None
    return reference.tick


def correspond(
    requests: Sequence[ActionRequest],
    statuses: Sequence[RequestStatus],
    *,
    tick: int,
    hold_ticks: int,
    step_rejections: int = 0,
) -> tuple[CorrespondenceAction, ...]:
    """Return the middle layer's actions for one agent's current program step at cadence tick ``tick``.

    The rules, their precedence and the thresholds are in the module docstring.

    Args:
        requests: The requests the executor has in flight for the step. They must share the agent,
            the executive version and the program step. An empty sequence gives ``()``.
        statuses: The status stream, newest last: ``SharedState.skills.statuses``, or the executor's
            own longer history. Statuses of other agents or requests, and statuses after ``tick``,
            are ignored.
        tick: The current cadence tick.
        hold_ticks: The step's hold time in ticks (its hold-time bucket, section 9.9), at least 1.
        step_rejections: Rejections of this step that the executor has already acted on.

    Returns:
        A tuple of actions. It is empty when nothing calls for one.

    Raises:
        TypeError: If an item is not an ActionRequest or a RequestStatus, or a number is not an integer.
        ValueError: If the requests mix steps or repeat an id, or a number is out of range.
    """

    items = tuple(requests)
    for index, request in enumerate(items):
        if not isinstance(request, ActionRequest):
            raise TypeError(f"requests[{index}] must be an ActionRequest, got {type(request).__name__}")
    now = _non_negative_int(tick, "tick")
    hold = _non_negative_int(hold_ticks, "hold_ticks")
    if hold < 1:
        raise ValueError(f"hold_ticks must be at least 1, got {hold}")
    prior = _non_negative_int(step_rejections, "step_rejections")
    if not items:
        return ()
    steps = {(request.agent, request.executive_version, request.program_step) for request in items}
    if len(steps) != 1:
        raise ValueError(f"requests must belong to one agent's one program step, got {sorted(steps)}")
    items = tuple(sorted(items, key=lambda request: request.request_id))
    ids = [request.request_id for request in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"requests repeat an id: {ids}")
    agent = items[0].agent
    history: dict[int, list[RequestStatus]] = {request_id: [] for request_id in ids}
    for index, status in enumerate(statuses):
        if not isinstance(status, RequestStatus):
            raise TypeError(f"statuses[{index}] must be a RequestStatus, got {type(status).__name__}")
        if status.agent == agent and status.request_id in history and status.tick <= now:
            history[status.request_id].append(status)
    latest = [history[request_id][-1] for request_id in ids if history[request_id]]

    rejected = [status for status in latest if status.kind == "rejected"]
    if rejected:
        count = prior + len(rejected)
        reason = _newest(rejected).reason
        if count >= REPEATED_REJECTIONS:
            return (InformationRequest(f"the step was rejected {count} times; the newest: {reason}"),)
        keep = all(rejection_code(status.reason) in ARGUMENT_CODES for status in rejected)
        return (ResampleStep(reason, keep_verb=keep),)

    failed = [status for status in latest if status.kind == "done" and status.outcome is not SkillOutcome.COMPLETED]
    if failed:
        status = _newest(failed)
        return (ResampleStep(f"request {status.request_id} done with {status.outcome.name}", keep_verb=False),)
    aborted = [status for status in latest if status.kind == "aborted"]
    if aborted:
        return (ResampleStep(_newest(aborted).reason, keep_verb=True),)

    if len(latest) == len(ids) and all(status.kind == "done" for status in latest):
        return (Advance(),)

    for request_id in ids:
        seen = history[request_id]
        if seen and seen[-1].kind == "running":
            since = _stalled(seen, now - STALL_TICKS)
            if since is not None:
                reason = f"stalled: request {request_id} made no progress from tick {since} to {seen[-1].tick}"
                return (ResampleStep(reason, keep_verb=False),)

    actions: list[CorrespondenceAction] = []
    for request in items:
        seen = history[request.request_id]
        if not seen or seen[-1].kind != "deferred":
            continue
        start = seen[-1].tick
        for status in reversed(seen[:-1]):
            if status.kind != "deferred":
                break
            start = status.tick
        if now - start > hold:
            if request.urgency >= 1.0:
                actions.append(Drop(request.request_id))
            else:
                actions.append(RaiseUrgency(request.request_id, min(URGENCY_STEP, 1.0 - request.urgency)))
    return tuple(actions)


__all__ = [
    "ARGUMENT_CODES",
    "REPEATED_REJECTIONS",
    "STALL_EPSILON",
    "STALL_TICKS",
    "URGENCY_STEP",
    "Advance",
    "CorrespondenceAction",
    "Drop",
    "InformationRequest",
    "RaiseUrgency",
    "ResampleStep",
    "correspond",
]
