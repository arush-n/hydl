"""Pure metric functions over recorded per-tick arrays for the composition arms.

Arrays are host numpy values of shape [T, B]: T ticks, B lanes. Results are
plain ints, floats, lists or dicts. Nothing here imports JAX, so the functions
run on records copied out of a compiled collector.

Formulas and citations are in ``eval/README.md`` and in the generated
``REPORT.md``. Each docstring states the exact rule that the function applies.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Sequence

import numpy as np

#: Two-sided 95% normal quantile. recipe.py's advance rule uses the same value.
DEFAULT_Z: float = 1.96


def _tick_array(
    value: Any,
    name: str,
    *,
    boolean: bool = False,
    integer: bool = False,
) -> np.ndarray:
    """Return ``value`` as a numpy array of shape [T, B], rejecting the wrong kind.

    Raises:
        ValueError: If the array is not two-dimensional.
        TypeError: If ``boolean`` or ``integer`` is set and the dtype does not match.
    """

    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T, B], got {array.shape}")
    if boolean and array.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean array, got {array.dtype}")
    if integer and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array, got {array.dtype}")
    return array


def _onsets(column: np.ndarray) -> np.ndarray:
    """Return the tick indices where a boolean column turns on.

    An onset at tick t means ``column[t]`` is True and ``column[t - 1]`` is
    False. The tick before the record counts as False.
    """

    previous = np.zeros_like(column)
    previous[1:] = column[:-1]
    return np.flatnonzero(column & ~previous)


def skill_changes(active_skill: Any, goal_valid: Any) -> np.ndarray:
    """Return where the active skill index changed on a tick with a goal already in force.

    Args:
        active_skill: Integer array [T, B]. The skill index in force after the
            decision at each tick.
        goal_valid: Boolean array [T, B]. True where a goal was already in force
            before the decision at that tick. False marks the forced first goal of
            an episode, which the core sets after an autoreset.

    Returns:
        Boolean array [T, B]. Entry ``[t, b]`` is True when ``t >= 1``,
        ``active_skill[t, b] != active_skill[t - 1, b]`` and ``goal_valid[t, b]``.
        Row 0 is all False.

    Raises:
        ValueError: If the arrays are not two-dimensional, or their shapes differ.
        TypeError: If ``active_skill`` is not integer-valued, or ``goal_valid`` is
            not boolean.
    """

    skills = _tick_array(active_skill, "active_skill", integer=True)
    valid = _tick_array(goal_valid, "goal_valid", boolean=True)
    if valid.shape != skills.shape:
        raise ValueError(f"goal_valid {valid.shape} must match active_skill {skills.shape}")
    changed = np.zeros(skills.shape, dtype=np.bool_)
    changed[1:] = (skills[1:] != skills[:-1]) & valid[1:]
    return changed


def goal_switches(active_skill: Any, goal_valid: Any) -> np.ndarray:
    """Count goal switches per lane as changes of the active skill index.

    Args:
        active_skill: Integer array [T, B]. The skill index in force after the
            decision at each tick.
        goal_valid: Boolean array [T, B]. True where a goal was already in force
            before the decision at that tick. False marks the forced first goal of
            an episode, which the core sets after an autoreset.

    Returns:
        int64 array [B]. Entry ``b`` is the number of ticks t in [1, T) where
        ``active_skill[t, b] != active_skill[t - 1, b]`` and ``goal_valid[t, b]``.

    Notes:
        A replacement with the same skill index is not counted, because only
        indices are given. A forced first goal is not counted either: its index is
        compared with the previous episode's last index, and that change is a reset,
        not a switch. The first tick of the record is never counted.

    Raises:
        ValueError: If the arrays are not two-dimensional, or their shapes differ.
        TypeError: If ``active_skill`` is not integer-valued, or ``goal_valid`` is
            not boolean.
    """

    return np.sum(skill_changes(active_skill, goal_valid), axis=0, dtype=np.int64)


def interrupt_latency_ticks(event_active: Any, switched: Any) -> list[int]:
    """Return the tick gap from each event onset to the next goal switch, per lane.

    An onset is a tick t where ``event_active[t, b]`` is True and
    ``event_active[t - 1, b]`` is False (see ``_onsets``). Its latency is
    ``t' - t``, where ``t'`` is the first tick >= t with ``switched[t', b]``
    True. A latency of 0 means the goal changed on the tick the event became
    active. An onset with no later switch in the record gives no entry. Those
    onsets are the unanswered ones, and their count is
    ``interrupt_onset_count`` minus the length of this list.

    Args:
        event_active: Boolean array [T, B]. True where an interrupt event is active.
        switched: Boolean array [T, B]. True where the goal in force was answered. The
            composition reducer passes a core replacement (``goal_switched`` and
            ``goal_valid``) or a skill change on a tick with a goal in force
            (:func:`skill_changes`); the second is how a host-written goal, which is
            not a core replacement, answers. A forced first goal is not a reply to an
            event, so the caller excludes it.

    Returns:
        A list of non-negative ints. Lane 0 comes first, and onsets within a
        lane are in time order.

    Raises:
        ValueError: If the arrays differ in shape or are not two-dimensional.
        TypeError: If either array is not boolean.
    """

    active = _tick_array(event_active, "event_active", boolean=True)
    switch = _tick_array(switched, "switched", boolean=True)
    if active.shape != switch.shape:
        raise ValueError(
            f"event_active {active.shape} and switched {switch.shape} must match"
        )
    gaps: list[int] = []
    for lane in range(active.shape[1]):
        for onset in _onsets(active[:, lane]):
            later = np.flatnonzero(switch[onset:, lane])
            if later.size:
                gaps.append(int(later[0]))
    return gaps


def interrupt_onset_count(event_active: Any) -> int:
    """Count event onsets over all lanes.

    Args:
        event_active: Boolean array [T, B]. True where an interrupt event is active.

    Returns:
        The number of (tick, lane) pairs that are onsets, as defined in
        :func:`interrupt_latency_ticks`.

    Raises:
        ValueError: If the array is not two-dimensional.
        TypeError: If the array is not boolean.
    """

    active = _tick_array(event_active, "event_active", boolean=True)
    return int(sum(_onsets(active[:, lane]).size for lane in range(active.shape[1])))


def commitment_violations(goal_age_at_switch: Any, minimum_commitment: Any) -> int:
    """Count ordinary goal switches made before the minimum commitment elapsed.

    Args:
        goal_age_at_switch: Float array of any shape. The elapsed age of the goal
            being replaced, in seconds, at each ordinary switch. Ordinary means
            valid, not timed out, and not an emergency interrupt. The caller
            filters out the bypasses, and this function does not see them.
        minimum_commitment: A scalar, or an array that broadcasts against
            ``goal_age_at_switch``, of minimum commitment seconds. +inf is allowed.

    Returns:
        The number of entries where ``age < minimum``. This is the negation of
        the core's commitment test ``commit = age >= minimum``
        (adk/architecture/hierarchy.py:273).

    Raises:
        ValueError: If an age is NaN, or a minimum is NaN or negative.
    """

    ages = np.asarray(goal_age_at_switch, dtype=np.float64)
    minimum = np.asarray(minimum_commitment, dtype=np.float64)
    if np.any(np.isnan(ages)):
        raise ValueError("goal_age_at_switch must not contain NaN")
    if np.any(np.isnan(minimum)) or np.any(minimum < 0.0):
        raise ValueError("minimum_commitment must be non-negative and not NaN")
    return int(np.count_nonzero(ages < minimum))


def wilson_lower_bound(successes: int, n: int, z: float = DEFAULT_Z) -> float:
    """Return the lower bound of the Wilson score interval for a success rate.

    With ``p = successes / n``::

        lower = (p + z^2 / (2n) - z * sqrt((p (1 - p) + z^2 / (4n)) / n))
                / (1 + z^2 / n)

    This is the same formula as ``agent_skills/training_runs/recipe.py``
    ``wilson_lower``. That function's advance rule compares it with
    ``advance_lower_bound``. The interval is Wilson (1927), cited in
    ``eval/README.md``.

    Args:
        successes: Number of successes, an integer with 0 <= successes <= n.
        n: Number of trials, an integer with n >= 1.
        z: Normal quantile, finite and positive. The default 1.96 gives 95%.

    Returns:
        The lower bound, a float. It is 0.0 when there are no successes.

    Raises:
        TypeError: If ``successes`` or ``n`` is not an integer (bools are rejected).
        ValueError: If n < 1, successes is outside [0, n], or z is not a finite positive number.
    """

    for name, value in (("successes", successes), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be within [0, {n}], got {successes}")
    if (
        isinstance(z, bool)
        or not isinstance(z, numbers.Real)
        or not math.isfinite(float(z))
        or float(z) <= 0.0
    ):
        raise ValueError(f"z must be a finite positive number, got {z!r}")
    if successes == 0:
        return 0.0
    p = successes / n
    z_squared = z * z
    centre = p + z_squared / (2.0 * n)
    spread = z * math.sqrt((p * (1.0 - p) + z_squared / (4.0 * n)) / n)
    return (centre - spread) / (1.0 + z_squared / n)


def completed_episodes(
    done: Any,
    terminated: Any,
    active_skill: Any,
    damage_active: Any,
    evader_index: int,
) -> list[dict[str, Any]]:
    """Reduce per-tick records to the per-episode metrics of ``CompositionTask.classify``.

    An episode of one lane runs from its first tick through the first tick at
    which ``done`` is True. That tick is the last tick of the episode, and the
    next episode starts on the tick after it. An episode that is still running
    at the end of the record is not completed, so it is not returned.

    For a completed episode that runs from tick ``s`` to tick ``e`` on lane ``b``:

    * ``caught`` is True if ``terminated[t, b]`` holds for some ``t`` in ``[s, e]``.
    * ``catch_tick`` is the first such ``t`` minus ``s``, or None if not caught.
    * ``evade_entry_ticks`` lists each ``t - s`` where ``active_skill[t, b]`` equals
      ``evader_index`` and the previous tick does not (the first tick of the
      episode counts as a previous tick that does not match).
    * ``damage_events`` counts the rising edges of ``damage_active`` inside ``[s, e]``.
      The first tick of the episode is an edge when damage is active there, because
      the episode is read on its own and the tick before it is not read.

    Args:
        done: Boolean [T, B]. The episode boundary (``EpisodeBoundary.done``).
        terminated: Boolean [T, B]. The environment's terminated flag, which
            ``info.terminated`` carries when the boundary is collapsed.
        active_skill: Integer [T, B]. The skill index in force after the decision.
        damage_active: Boolean [T, B]. True where DAMAGE_RECEIVED is active and known.
        evader_index: The catalogue index of the evader skill, a non-negative int.

    Returns:
        A list of dicts, in lane order and then time order. Each dict has the
        keys ``lane``, ``start``, ``end``, ``caught``, ``catch_tick``,
        ``evade_entry_ticks`` (a tuple of ints) and ``damage_events``. The
        ``start`` and ``end`` values are record ticks, and the other tick values
        are relative to ``start``.

    Raises:
        ValueError: If the arrays differ in shape or are not two-dimensional, or
            ``evader_index`` is not a non-negative int.
        TypeError: If a boolean or integer array has the wrong dtype.
    """

    done_array = _tick_array(done, "done", boolean=True)
    terminated_array = _tick_array(terminated, "terminated", boolean=True)
    skills = _tick_array(active_skill, "active_skill", integer=True)
    damage = _tick_array(damage_active, "damage_active", boolean=True)
    shape = done_array.shape
    for name, array in (
        ("terminated", terminated_array),
        ("active_skill", skills),
        ("damage_active", damage),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} {array.shape} must match done {shape}")
    if isinstance(evader_index, bool) or not isinstance(evader_index, numbers.Integral) or evader_index < 0:
        raise ValueError(f"evader_index must be a non-negative int, got {evader_index!r}")

    ticks, lanes = shape
    episodes: list[dict[str, Any]] = []
    for lane in range(lanes):
        start = 0
        for end in range(ticks):
            if not done_array[end, lane]:
                continue
            window = slice(start, end + 1)
            caught_mask = terminated_array[window, lane]
            caught = bool(caught_mask.any())
            catch_tick = int(np.flatnonzero(caught_mask)[0]) if caught else None
            evading = skills[window, lane] == evader_index
            entering = evading & ~np.concatenate(([False], evading[:-1]))
            damage_window = damage[window, lane]
            damage_onsets = damage_window & ~np.concatenate(([False], damage_window[:-1]))
            episodes.append(
                {
                    "lane": lane,
                    "start": start,
                    "end": end,
                    "caught": caught,
                    "catch_tick": catch_tick,
                    "evade_entry_ticks": tuple(int(tick) for tick in np.flatnonzero(entering)),
                    "damage_events": int(np.count_nonzero(damage_onsets)),
                }
            )
            start = end + 1
    return episodes


def summarize(
    *,
    lane_runs: int,
    episodes_completed: int,
    successes: int | None,
    pursue_only: int | None,
    catches: int | None,
    goal_switches: int,
    core_switches: int,
    interrupt_latencies: Sequence[int],
    interrupt_onsets: int,
    commitment_violation_count: int,
    jev_decisions: int | None,
    jev_misses: int | None,
    jev_latency_ms_p50: float | None,
    jev_latency_ms_p95: float | None,
    wall_seconds: float,
    warmup_seconds: float,
    ticks: int,
) -> dict[str, Any]:
    """Reduce one arm's totals to the values written to ``metrics.csv``.

    Args:
        lane_runs: Lanes times runs, a positive int. A lane-run is one lane over
            one collector run of ``ticks`` steps. Switches are reported per lane-run.
        episodes_completed: Completed episodes, a non-negative int (see
            :func:`completed_episodes`). Success and catch rates use this count.
        successes: Completed episodes with a composed verdict, or None when the
            task is not supplied.
        pursue_only: Completed episodes with a ``pursue_only`` verdict, or None.
        catches: Completed episodes with ``caught`` True, or None.
        goal_switches: Total changes of the active skill index over all lane-runs.
        core_switches: Total core replacements over all lane-runs: ticks where the core
            switched while a goal was already in force. This counts the core's
            ``goal_switched`` flag, includes a same-index replacement, and excludes
            the forced first goal of each episode.
        interrupt_latencies: Answered latencies in ticks, from
            :func:`interrupt_latency_ticks`.
        interrupt_onsets: Total event onsets, from :func:`interrupt_onset_count`.
        commitment_violation_count: Total from :func:`commitment_violations`.
        jev_decisions: JEV decisions measured, or None when they are not measured.
        jev_misses: JEV decisions over the latency budget, or None.
        jev_latency_ms_p50: Median JEV latency in ms, or None when not measured.
        jev_latency_ms_p95: 95th-percentile JEV latency in ms, or None.
        wall_seconds: Summed wall time of the timed collector runs.
        warmup_seconds: Wall time of the untimed first run, which includes JIT compilation.
        ticks: Ticks per collector run.

    Returns:
        A dict with the value of each derived column. ``success_rate`` is
        ``successes / episodes_completed``, and ``success_wilson_lower`` is
        :func:`wilson_lower_bound` on the same counts. A rate is None when its
        denominator is zero or its count is None. ``jev_miss_rate`` is
        ``jev_misses / jev_decisions``.

    Raises:
        ValueError: If lane_runs < 1, a count is negative, or answered latencies
            outnumber onsets.
    """

    if isinstance(lane_runs, bool) or not isinstance(lane_runs, numbers.Integral) or lane_runs < 1:
        raise ValueError(f"lane_runs must be a positive integer, got {lane_runs!r}")
    if episodes_completed < 0:
        raise ValueError(f"episodes_completed must be non-negative, got {episodes_completed}")
    latencies = [int(value) for value in interrupt_latencies]
    if any(value < 0 for value in latencies):
        raise ValueError("interrupt latencies must be non-negative")
    if interrupt_onsets < len(latencies):
        raise ValueError(
            f"{len(latencies)} answered latencies cannot exceed {interrupt_onsets} onsets"
        )
    success_rate = None
    wilson = None
    if successes is not None and episodes_completed >= 1:
        success_rate = successes / episodes_completed
        wilson = wilson_lower_bound(successes, episodes_completed)
    miss_rate = None
    if jev_decisions and jev_misses is not None:
        miss_rate = jev_misses / jev_decisions
    return {
        "lane_runs": int(lane_runs),
        "episodes_completed": int(episodes_completed),
        "successes": successes,
        "success_rate": success_rate,
        "success_wilson_lower": wilson,
        "pursue_only": pursue_only,
        "catches": catches,
        "catches_per_episode": (
            catches / episodes_completed
            if catches is not None and episodes_completed >= 1
            else None
        ),
        "goal_switches_per_lane_run": goal_switches / lane_runs,
        "core_switches_per_lane_run": core_switches / lane_runs,
        "interrupt_onsets": int(interrupt_onsets),
        "interrupt_answered": len(latencies),
        "interrupt_unanswered": int(interrupt_onsets) - len(latencies),
        "interrupt_latency_median_ticks": (
            float(np.median(latencies)) if latencies else None
        ),
        "commitment_violations": int(commitment_violation_count),
        "jev_decisions": jev_decisions,
        "jev_misses": jev_misses,
        "jev_miss_rate": miss_rate,
        "jev_latency_ms_p50": jev_latency_ms_p50,
        "jev_latency_ms_p95": jev_latency_ms_p95,
        "wall_seconds": float(wall_seconds),
        "warmup_seconds": float(warmup_seconds),
        "ticks": int(ticks),
    }


__all__ = [
    "DEFAULT_Z",
    "commitment_violations",
    "completed_episodes",
    "goal_switches",
    "interrupt_latency_ticks",
    "interrupt_onset_count",
    "skill_changes",
    "summarize",
    "wilson_lower_bound",
]
