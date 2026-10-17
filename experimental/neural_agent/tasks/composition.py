"""Composition task: pursue, evade when damaged, then resume pursuit (ADR-001 P2).

The task is an executive state plus a success rule. The executive writes two
goals, a pursuit goal and an evader goal that is gated by the constraint
``on_damage``. The JEV decides which goal is in force. This module does not
choose a phase, and it does not run a policy.

Thesis (ADR-001 section 1, verbatim): the research question is "how an
intelligent agent uses time and compute: when to think, when to act, and when to
reuse an existing decision. Task completion is the instrument, not the
question." Composition is the instrument for sequencing and interruption. The
agent must pre-empt pursuit when damage arrives, run the evader, and resume
pursuit afterwards. See ``SKILL.md`` for the formulas and citations.

The objective that closes a composition depends on the arena. On
``flat_pursuit`` it is the catch, which ends the episode. On ``combat_baseline``
DAMAGE_RECEIVED fires without injection but nothing can be caught, so the
objective is re-engagement (owner decision, 2026-10-16): the legal target
distance, read from the belief's ``target_distance_blocks`` feature, is at most
``REENGAGE_DISTANCE_BLOCKS`` for ``REENGAGE_RUN_STEPS`` consecutive decision
steps after the resumed pursuit and before the episode ends
(:func:`reengage_start_tick`, :func:`reengaged_tick`). Damage injection stays an
ablation. Every tick window in this module counts environment decision steps of
0.045 s.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from experimental.neural_agent.contracts import (
    ComputeBudget,
    ExecutiveState,
    GoalSpec,
    executive_arrays,
)

#: The composition phases, in order. ``resume`` follows ``evade``.
PHASES: tuple[str, ...] = ("pursue", "evade", "resume")
PURSUIT_SKILL: str = "pursuit"
EVADER_SKILL: str = "evader"
#: Constraint on the evader goal: it is in force only after damage (ADR-001 P2).
EVADER_CONSTRAINT: str = "on_damage"
PURSUIT_PRIORITY: float = 1.0
EVADER_PRIORITY: float = 0.5
#: Verdicts returned by :meth:`CompositionTask.classify`. Each is read against the
#: episode's objective: the catch, or, on an episode with no catch, the
#: re-engagement tick ``reengaged_tick`` (``combat_baseline``).
#: ``composed``: the objective follows a damage-triggered evade entry and a resumed
#: pursuit (the strict rule). The only verdict that counts as success.
VERDICT_COMPOSED: str = "composed"
#: ``composed_loose``: the objective follows an evade entry, but no entry meets the
#: strict rule. Reported, never counted as success.
VERDICT_COMPOSED_LOOSE: str = "composed_loose"
#: ``pursue_only``: the objective with no evade entry before it and no damage in the
#: episode. On ``combat_baseline``: re-engaged with no damage.
VERDICT_PURSUE_ONLY: str = "pursue_only"

#: Re-engagement, the objective on ``combat_baseline`` (owner decision, 2026-10-16):
#: the known target distance (the belief's ``target_distance_blocks``) is at most
#: this many blocks ...
REENGAGE_DISTANCE_BLOCKS: float = 2.0
#: ... on this many consecutive environment decision steps of 0.045 s.
REENGAGE_RUN_STEPS: int = 5

#: An evade entry counts as damage-triggered when a damage event happened within
#: this many ticks before it (the same window J0 uses, jev/scripted.py).
EVADE_TRIGGER_WINDOW_TICKS: int = 40
#: ``failed``: every other episode, including every episode with no objective.
VERDICT_FAILED: str = "failed"
_METRIC_KEYS: tuple[str, ...] = ("caught", "catch_tick", "evade_entry_ticks", "damage_events")


def _count(value: Any, name: str) -> int:
    """Return ``value`` as a non-negative int. Bools and floats are rejected."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


def _episode_metrics(metrics: Mapping[str, Any]) -> tuple[bool, int | None, tuple[int, ...], int]:
    """Validate one episode's metrics and return them as typed values.

    Raises:
        TypeError: If ``metrics`` is not a mapping, or a field has the wrong type.
        ValueError: If a key is missing, a count is negative, or ``catch_tick`` is
            inconsistent with ``caught``.
    """

    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")
    missing = [key for key in _METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"metrics is missing {missing}; required keys are {list(_METRIC_KEYS)}")
    caught = metrics["caught"]
    if not isinstance(caught, (bool, np.bool_)):
        raise TypeError("metrics['caught'] must be a bool")
    catch_tick = metrics["catch_tick"]
    if caught:
        catch_tick = _count(catch_tick, "metrics['catch_tick']")
    elif catch_tick is not None:
        raise ValueError("metrics['catch_tick'] must be None when no catch occurred")
    entries = metrics["evade_entry_ticks"]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("metrics['evade_entry_ticks'] must be a sequence of ticks")
    entry_ticks = tuple(
        _count(entry, f"metrics['evade_entry_ticks'][{index}]")
        for index, entry in enumerate(entries)
    )
    damage_events = _count(metrics["damage_events"], "metrics['damage_events']")
    return bool(caught), catch_tick, entry_ticks, damage_events



def _optional_ticks(metrics: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    """Return ``metrics[key]`` as a tuple of non-negative ints, or None when absent."""

    if key not in metrics:
        return None
    value = metrics[key]
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{key} must be a sequence of ints")
    ticks = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int,)) and not hasattr(item, "__index__"):
            raise TypeError(f"{key} must contain ints")
        tick = int(item)
        if tick < 0:
            raise ValueError(f"{key} must be non-negative")
        ticks.append(tick)
    return tuple(ticks)


def _optional_tick(metrics: Mapping[str, Any], key: str) -> int | None:
    """Return ``metrics[key]`` as a non-negative int, or None when it is absent or None."""

    value = metrics.get(key)
    return None if value is None else _count(value, f"metrics[{key!r}]")


def _damage_triggered(entry: int, damage_ticks: Sequence[int]) -> bool:
    """Return True when a damage tick ``d`` has ``entry - EVADE_TRIGGER_WINDOW_TICKS <= d <= entry``."""

    return any(entry - EVADE_TRIGGER_WINDOW_TICKS <= d <= entry for d in damage_ticks)


def reengage_start_tick(
    evade_entry_ticks: Sequence[int],
    pursuit_entry_ticks: Sequence[int],
    damage_ticks: Sequence[int],
) -> int | None:
    """Return the first episode tick from which a re-engagement counts, or None.

    The owner's rule reads the re-engagement after the resumed pursuit:

    * with no evade entry, the episode start, 0;
    * otherwise the resume: the first pursuit entry after the first
      damage-triggered evade entry, or after the first evade entry when no entry
      was damage-triggered (the episode is then ``composed_loose`` at best);
    * None when no pursuit entry follows that evade entry, so no re-engagement counts.

    Args:
        evade_entry_ticks: Episode ticks where the evader became active.
        pursuit_entry_ticks: Episode ticks where pursuit became active.
        damage_ticks: Episode ticks with DAMAGE_RECEIVED active and known.

    Returns:
        An episode tick, or None.
    """

    if len(evade_entry_ticks) == 0:
        return 0
    triggered = [entry for entry in evade_entry_ticks if _damage_triggered(entry, damage_ticks)]
    evade = min(triggered or evade_entry_ticks)
    return next((tick for tick in sorted(pursuit_entry_ticks) if tick > evade), None)


def reengaged_tick(distance_blocks: Any, distance_known: Any, *, start: int = 0) -> int | None:
    """Return the tick that completes the first re-engagement run at or after ``start``, or None.

    A re-engagement run is ``REENGAGE_RUN_STEPS`` consecutive ticks ``t >= start``
    with ``distance_known[t]`` and ``distance_blocks[t] <= REENGAGE_DISTANCE_BLOCKS``.
    A shorter run does not count, and an unknown tick breaks a run. The ticks are
    environment decision steps of 0.045 s. The distance is the belief's
    ``target_distance_blocks`` feature, which is known only while the target is
    visible.

    Args:
        distance_blocks: Float array [T], one lane-episode of the target distance in blocks.
        distance_known: Bool array [T], the feature's known flag on the same ticks.
        start: The first tick a run may use, a non-negative int
            (:func:`reengage_start_tick`).

    Returns:
        The run's last tick, an index into the arrays, or None when no run completes.

    Raises:
        ValueError: If the arrays are not one-dimensional with equal shapes.
        TypeError: If ``distance_known`` is not boolean, or ``start`` is not an integer.
    """

    distance = np.asarray(distance_blocks, dtype=np.float64)
    known = np.asarray(distance_known)
    if distance.ndim != 1 or known.shape != distance.shape:
        raise ValueError(
            f"distance_blocks {distance.shape} and distance_known {known.shape} must be equal 1-D shapes"
        )
    if known.dtype != np.bool_:
        raise TypeError(f"distance_known must be boolean, got {known.dtype}")
    run = 0
    for tick in range(_count(start, "start"), distance.shape[0]):
        close = bool(known[tick]) and distance[tick] <= REENGAGE_DISTANCE_BLOCKS
        run = run + 1 if close else 0
        if run == REENGAGE_RUN_STEPS:
            return tick
    return None


@dataclass(frozen=True)
class CompositionTask:
    """The pursue, evade, resume task and its termination rule.

    Attributes:
        phases: Must be exactly ``("pursue", "evade", "resume")``. Any other value
            is refused, so a reordered task cannot be run under this name.
        tick_budget: Decision ticks before the episode ends, a positive int. Build
            the arena with ``ticks=tick_budget`` so that the environment's own
            truncation agrees with this rule.

    Raises:
        ValueError: If ``phases`` differs from the fixed order, or ``tick_budget``
            is not positive.
        TypeError: If ``tick_budget`` is not an integer.
    """

    phases: tuple[str, ...] = PHASES
    tick_budget: int = 256

    def __post_init__(self) -> None:
        """Validate the phases and the budget."""

        if tuple(self.phases) != PHASES:
            raise ValueError(f"phases must be exactly {PHASES}, got {tuple(self.phases)}")
        object.__setattr__(self, "phases", PHASES)
        object.__setattr__(
            self, "tick_budget", _count(self.tick_budget, "CompositionTask.tick_budget")
        )
        if self.tick_budget < 1:
            raise ValueError("CompositionTask.tick_budget must be positive")

    def initial_executive_state(self, skill_ids: Sequence[str]) -> ExecutiveState:
        """Return the executive state that opens the task.

        The goals are ``pursuit`` at priority 1.0 with no constraint, and ``evader``
        at priority 0.5 with the constraint ``on_damage``. The state has version 1,
        because version 0 means that no state has been written
        (``empty_executive_arrays``). The budget is the ``ComputeBudget`` default.

        Args:
            skill_ids: The skill catalog in dispatcher order. It must contain
                ``pursuit`` and ``evader``. The usual catalog is
                ``("idle", "pursuit", "evader")``.

        Returns:
            The ExecutiveState. It was checked by lowering it with
            ``executive_arrays`` against ``skill_ids`` (the lowered arrays are
            discarded), so it is valid for that catalog.

        Raises:
            KeyError: If a goal names a skill that is not in ``skill_ids``.
            ValueError: If ``skill_ids`` repeats an id.
        """

        state = ExecutiveState(
            version=1,
            issued_tick=0,
            goals=(
                GoalSpec(skill_id=PURSUIT_SKILL, priority=PURSUIT_PRIORITY),
                GoalSpec(
                    skill_id=EVADER_SKILL,
                    priority=EVADER_PRIORITY,
                    constraints=(EVADER_CONSTRAINT,),
                ),
            ),
            budget=ComputeBudget(),
            facts=("composition: pursue; evade when damaged; resume pursuit",),
        )
        executive_arrays(state, skill_ids, batch=1)
        return state

    def classify(self, metrics: Mapping[str, Any]) -> str:
        """Return the verdict for one episode: composed, composed_loose, pursue_only or failed.

        The objective ``O`` is the catch. On an episode with no catch and a
        ``reengaged_tick`` (``combat_baseline``, where nothing can be caught), ``O``
        is the re-engagement, and its tick takes the catch tick's place. Without
        ``reengaged_tick`` the rules are the flat-arena rules, unchanged. Rules,
        with ``E`` the evade entries and ``D`` the damage events:

        * ``composed``: ``O`` happened after an evade entry that meets the strict
          rule below.
        * ``composed_loose``: ``O`` happened after an evade entry, but no entry
          meets the strict rule. Reported, never counted as success.
        * ``pursue_only``: ``O`` happened, no evade phase came before it, and
          ``D == 0``. The episode is reported separately from ``composed``. On
          ``combat_baseline`` it is a re-engagement with no damage.
        * ``failed``: every other episode, including every episode with no ``O``.

        Strict rule (2026-10-17): ``composed`` requires an evade entry ``e`` before
        ``O`` that was damage-triggered (a damage tick ``d`` with
        ``e - EVADE_TRIGGER_WINDOW_TICKS <= d <= e``) and a resumed pursuit (a
        pursuit entry ``p`` with ``e < p < O``). This needs the optional keys
        ``damage_ticks`` and ``pursuit_entry_ticks`` (sequences of ints). Without
        them, or when no evade entry satisfies both, ``O`` after an evade entry
        is ``composed_loose``: reported, never counted as success. The loose rule
        is what let the shuffled-goal control score 3/5 on 2026-10-17.

        Args:
            metrics: A mapping with the keys ``caught`` (bool), ``catch_tick`` (int
                when ``caught``, else None), ``evade_entry_ticks`` (a sequence of
                ints) and ``damage_events`` (non-negative int), and optionally
                ``damage_ticks``, ``pursuit_entry_ticks`` and ``reengaged_tick``
                (a non-negative int or None, from :func:`reengaged_tick`). Ticks
                are episode-relative decision steps. The caller reduces the
                per-step records to these values.

        Returns:
            One of ``VERDICT_COMPOSED``, ``VERDICT_COMPOSED_LOOSE``,
            ``VERDICT_PURSUE_ONLY`` or ``VERDICT_FAILED``.

        Raises:
            TypeError: If ``metrics`` is not a mapping, or a value has the wrong type.
            ValueError: If a required key is missing, or a count is negative, or
                ``catch_tick`` contradicts ``caught``.
        """

        caught, catch_tick, entries, damage = _episode_metrics(metrics)
        reengaged = _optional_tick(metrics, "reengaged_tick")
        # The objective is the catch. With no catch, a re-engagement takes its place
        # (combat_baseline, where nothing can be caught; owner decision 2026-10-16).
        reached, objective_tick = caught, catch_tick
        if not caught and reengaged is not None:
            reached, objective_tick = True, reengaged
        damage_ticks = _optional_ticks(metrics, "damage_ticks")
        pursuit_entries = _optional_ticks(metrics, "pursuit_entry_ticks")
        if reached and any(entry < objective_tick for entry in entries):
            if damage_ticks is None or pursuit_entries is None:
                # The caller did not record when damage happened or when pursuit
                # resumed, so only the loose rule can be applied. The 256-tick gate
                # run of 2026-10-17 showed the shuffled-goal control scoring 3/5 under
                # the loose rule; it is reported separately and never as success.
                return VERDICT_COMPOSED_LOOSE
            for entry in entries:
                if entry >= objective_tick:
                    continue
                triggered = _damage_triggered(entry, damage_ticks)
                resumed = any(entry < p < objective_tick for p in pursuit_entries)
                if triggered and resumed:
                    return VERDICT_COMPOSED
            return VERDICT_COMPOSED_LOOSE
        if reached and damage == 0:
            return VERDICT_PURSUE_ONLY
        return VERDICT_FAILED

    def success(self, metrics: Mapping[str, Any]) -> bool:
        """Return True only for a composed success (see :meth:`classify`).

        A ``pursue_only`` episode is a catch (or a re-engagement) with no damage,
        and it is not a composed success. Read it through :meth:`classify`.

        Args:
            metrics: As for :meth:`classify`.

        Returns:
            True if and only if :meth:`classify` returns ``VERDICT_COMPOSED``.
        """

        return self.classify(metrics) == VERDICT_COMPOSED

    def budget_exhausted(self, tick: int) -> bool:
        """Return True once ``tick`` reaches the tick budget.

        Args:
            tick: The number of decision ticks elapsed, a non-negative int.

        Returns:
            ``tick >= tick_budget``.

        Raises:
            TypeError: If ``tick`` is not an integer.
            ValueError: If ``tick`` is negative.
        """

        return _count(tick, "tick") >= self.tick_budget


__all__ = [
    "CompositionTask",
    "EVADER_CONSTRAINT",
    "EVADER_PRIORITY",
    "EVADER_SKILL",
    "PHASES",
    "PURSUIT_PRIORITY",
    "PURSUIT_SKILL",
    "REENGAGE_DISTANCE_BLOCKS",
    "REENGAGE_RUN_STEPS",
    "VERDICT_COMPOSED",
    "VERDICT_COMPOSED_LOOSE",
    "EVADE_TRIGGER_WINDOW_TICKS",
    "VERDICT_FAILED",
    "VERDICT_PURSUE_ONLY",
    "reengage_start_tick",
    "reengaged_tick",
]
