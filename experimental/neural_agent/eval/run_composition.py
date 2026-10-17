"""Run the comparison arms on the composition task and write metrics.csv and REPORT.md.

Command (CPU; set the environment first, see SKILL.md)::

    python -m experimental.neural_agent.eval.run_composition \
        --arms flat,j0,shuffled --episodes 2 --batch 8 --ticks 256 --seed 0 \
        --arena combat_baseline --runner compiled --out DIR

The composition task (``tasks/composition.py``) sets the executive, E_t, and the
verdict of each episode. ``--arena`` picks the environment (``tasks/arenas.py``):
``combat_baseline``, the default, where DAMAGE_RECEIVED fires without injection
but nothing can be caught, so the task's objective there is re-engagement in
place of the catch (``tasks/composition.py`` ``reengaged_tick``), or
``flat_pursuit``, where a catch ends an episode but no actor takes damage. The
horizon is ``--ticks``, the same number that
``CompositionTask.tick_budget`` takes. The default is 256 ticks: the 256-tick
gate run (``eval/out/gate256``) caught once per episode on ``flat_pursuit``.

Each arm is composed as ``HierarchicalAgentCore(belief, manager, actor,
manager_cadence_seconds=0.5)``. ``--runner compiled``, the default, runs it
through ``compile_hierarchical_collector``. Run ``e`` uses the key
``fold_in(key(seed), e)``. One untimed warm-up run compiles the collector, and
each timed run is clocked with ``time.perf_counter``. ``--runner eager`` runs
it through ``session.run_session``, the Python step loop: one untimed one-step
warm-up session, then one timed session of ``episodes * ticks`` steps with the
key ``fold_in(key(seed), 0)``. A host-mode arm (``j_lm``) needs the eager
runner, and the compiled runner refuses it. Its JEV columns are the session's
host metrics.

The sibling modules are imported inside the functions that use them, so
``--help`` runs without them.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import HierarchicalCarry
from adk.architecture.decision_context import DecisionEvent
from adk.architecture.hierarchy import (
    HierarchicalAgentCore,
    HierarchicalCoreState,
    initialize_hierarchical_state,
)
from adk.architecture.inputs import actor_policy_input
from adk.runtime.hierarchical import (
    HierarchicalActorInputFn,
    HierarchicalContextFn,
    HierarchicalLoopTransition,
    compile_hierarchical_collector,
    jax_hierarchical_context,
)
from experimental.neural_agent.contracts import ACTION_HEADS, JevStats, feature_index, idle_goal
from experimental.neural_agent.eval.arms import (
    ARMS,
    BELIEF_PARAMS_SEED,
    J0_LATENCY_BUDGET_MS,
    LM_BACKENDS,
    Arm,
    ArmContext,
    arm_label,
    build_arm_context,
    get_arm,
)
from experimental.neural_agent.eval.metrics import (
    commitment_violations,
    completed_episodes,
    goal_switches,
    interrupt_latency_ticks,
    interrupt_onset_count,
    skill_changes,
    summarize,
)

#: Manager cadence of the composed core, in seconds (the composition brief).
MANAGER_CADENCE_SECONDS: float = 0.5

#: Events that count as interrupts. ADR-001 section 8 P2 names them as the
#: evade triggers ("evade on PROJECTILE_THREAT_ENTERED or DAMAGE_RECEIVED").
INTERRUPT_EVENTS: tuple[DecisionEvent, ...] = (
    DecisionEvent.PROJECTILE_THREAT_ENTERED,
    DecisionEvent.DAMAGE_RECEIVED,
)
INTERRUPT_EVENT_INDICES: tuple[int, ...] = tuple(int(event) for event in INTERRUPT_EVENTS)
DAMAGE_EVENT_INDEX: int = int(DecisionEvent.DAMAGE_RECEIVED)

DEFAULT_ARMS: tuple[str, ...] = ("flat", "j0", "shuffled")
DEFAULT_OUT_DIR: Path = Path(__file__).resolve().parent / "out"

#: The names of ``tasks/arenas.py`` ARENA_NAMES, repeated so that ``--help`` needs no sibling.
ARENA_CHOICES: tuple[str, ...] = ("flat_pursuit", "combat_baseline", "combat_open_flat")  # must match tasks.arenas.ARENA_NAMES
DEFAULT_ARENA: str = "combat_open_flat"  # the shipped scene where the target is visible and damage occurs
RUNNER_CHOICES: tuple[str, ...] = ("compiled", "eager")
DEFAULT_RUNNER: str = "compiled"
#: Default horizon. The 256-tick gate run caught once per episode on flat_pursuit.
DEFAULT_TICKS: int = 512  # 2026-10-17: at 256 ticks (11.5 s) re-engagement after an evade rarely completes on the default
# arena (J0 composed 1/4); at 512 it composes 33/64 vs 16/64 for the shuffled control (docs/LEDGER.md). Was 256.

#: Sibling modules that the real run needs. Each is checked before a run starts.
SIBLING_MODULES: tuple[str, ...] = (
    "experimental.neural_agent.belief.core",
    "experimental.neural_agent.skill_library.library",
    "experimental.neural_agent.skill_library.dispatcher",
    "experimental.neural_agent.jev.runner",
    "experimental.neural_agent.jev.scripted",
    "experimental.neural_agent.tasks.composition",
    "experimental.neural_agent.tasks.flat_arena",
    "experimental.neural_agent.tasks.arenas",
    "experimental.neural_agent.session",
    "experimental.neural_agent.jev.network",
    "experimental.neural_agent.jev.lm.instance",
)

#: Column order of metrics.csv. Every key of a run row must appear here.
COLUMNS: tuple[str, ...] = (
    "arm",
    "episodes",
    "batch",
    "seed",
    "ticks",
    "lane_runs",
    "episodes_completed",
    "successes",
    "success_rate",
    "success_wilson_lower",
    "composed_loose",
    "pursue_only",
    "catches",
    "catches_per_episode",
    "damage_events",
    "goal_switches_per_lane_run",
    "core_switches_per_lane_run",
    "interrupt_onsets",
    "interrupt_answered",
    "interrupt_unanswered",
    "interrupt_latency_median_ticks",
    "commitment_violations",
    "jev_decisions",
    "jev_misses",
    "jev_miss_rate",
    "jev_lane_failures",
    "jev_latency_ms_p50",
    "jev_latency_ms_p95",
    "wall_seconds",
    "warmup_seconds",
)

THESIS_TEXT: str = (
    "ADR-001 section 1 (COMMITTED): \"The research question is how an intelligent agent uses time "
    "and compute: when to think, when to act, and when to reuse an existing decision. Task completion "
    "is the instrument, not the question.\" This report measures the instrument: goal switching, interrupt "
    "response and JEV latency on the composition task. A structural run is not evidence that the hierarchy "
    "works (ADR-001 sections 0 and 6.5). ADR-001 is a local record and is not in the repository."
)

FORMULA_TEXT: str = """\
lane-run                     = one lane over one collector run of `ticks` steps; lane_runs = episodes x batch      [src: run_arm]
                               eager: one lane over `ticks` steps of one session of episodes x ticks steps  [src: run_arm_eager]
episode                      = the ticks from a reset through the first tick with done (a catch, or the horizon)   [src: metrics.completed_episodes]
episodes_completed           = episodes that reached done inside the run; a partial episode at the end is not counted  [src: metrics.completed_episodes]
success_rate                 = successes / episodes_completed      (successes: CompositionTask verdict "composed")  [src: metrics.summarize]
success_wilson_lower         = (p + z^2/(2n) - z*sqrt((p(1-p) + z^2/(4n))/n)) / (1 + z^2/n),  p = successes/n, z = 1.96  [src: metrics.wilson_lower_bound; Wilson 1927]
objective O                  = flat_pursuit: the catch (terminated);  combat_baseline: the re-engagement, since terminated
                               is a death or a world error there, never a catch  [src: run_composition._score_run]
composed                     = O reached, and some evade entry e < O has a damage tick d with e - W <= d <= e
                               and a pursuit entry p with e < p < O;  W = EVADE_TRIGGER_WINDOW_TICKS  [src: tasks/composition.py classify]
composed_loose               = O reached after an evade entry, but no entry meets the composed rule; never a success  [src: tasks/composition.py classify]
pursue_only                  = O reached and no evade entry before it, with zero damage events  [src: tasks/composition.py classify]
reengaged_tick               = first t >= s0 ending R consecutive ticks with distance_known and distance <= D;  distance = the
                               belief's target_distance_blocks, D = REENGAGE_DISTANCE_BLOCKS, R = REENGAGE_RUN_STEPS  [src: tasks/composition.py reengaged_tick]
s0                           = 0 with no evade entry; else the first pursuit entry after the first damage-triggered evade entry
                               (after the first evade entry if none was triggered); none: no re-engagement  [src: tasks/composition.py reengage_start_tick]
re-engagements               = completed episodes with a reengaged_tick; the catches columns on the combat arenas  [src: run_composition._score_run]
damage_events                = sum over completed episodes of the rising edges of DAMAGE_RECEIVED in the episode  [src: metrics.completed_episodes]
damage_ticks                 = episode ticks t with DAMAGE_RECEIVED active and known at t  [src: run_composition._score_run]
pursuit_entry_ticks          = episode ticks t with skill[t] = pursuit and skill[t-1] != pursuit (the first tick counts)  [src: run_composition._score_run]
goal_switches_per_lane_run   = sum over lane-runs of sum_{t=1}^{T-1} [skill[t] != skill[t-1]] * [goal_valid[t]] / lane_runs  [src: metrics.goal_switches]
replacement                  = goal_switched[t] and goal_valid[t]   (a forced first goal is excluded)  [src: run_composition._score_run; hierarchy.py:356 and :372]
interrupt onset              = event_active[t] and not event_active[t-1]   (PROJECTILE_THREAT_ENTERED, DAMAGE_RECEIVED)  [src: metrics.interrupt_onset_count]
interrupt answer             = replacement[t] or (skill[t] != skill[t-1] and goal_valid[t])  [src: run_composition._score_run; metrics.skill_changes]
interrupt latency            = t' - t,  t' = min{ t' >= t : answer[t'] }   (no such t': unanswered onset)  [src: metrics.interrupt_latency_ticks; decision_context.py:415]
commitment violations        = #{ ordinary switches : goal_age < minimum_commitment }
                               ordinary = valid goal, not timed out, not an emergency interrupt  [src: metrics.commitment_violations; hierarchy.py:273 and :347-356]
jev_miss_rate                = misses / decisions;  a miss is latency_ms > budget_ms (strict; not measured in-graph)  [src: jev/runner.py JevRunner.__call__]
eager host-mode arm          = a decision is due when step % host_cadence_ticks = 0 or any event is active; latency_ms is the
                               wall time of the decide call; a miss is latency_ms > host_budget_ms, a decider error, or no goal  [src: session.py run_session]
jev_lane_failures            = lanes on which the host instance kept the goal after a backend failure or timeout  [src: eval/arms.py LanguageModelHostDecider]"""


CITATION_LINES: tuple[str, ...] = (
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. "
    "Journal of the American Statistical Association 22(158): 209-212. DOI 10.1080/01621459.1927.10502953 "
    "(from a third-party bibliographic record in a web search; not checked at the publisher). "
    "[formula recalled; not re-checked against the paper in this session]",
    "Sutton, R. S., Precup, D., Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal "
    "abstraction in reinforcement learning. Artificial Intelligence 112(1-2): 181-211. Public copy: "
    "https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf . The DOI "
    "10.1016/S0004-3702(99)00052-1 is recalled and not confirmed. [cited in ADR-001 section 11, O1; "
    "the local record was not re-checked here]",
    "ADR-001 three-level agent, local record, not in the repository (no URL): section 8 P2 exit (goal switches, "
    "interrupt latency, commitment violations); section 9 (metric families and ablation matrix).",
    "experimental/hierarchical_planner/DESIGN.md section 6: the shuffled-goal honesty control (repository path).",
    "agent_skills/training_runs/recipe.py: wilson_lower, and the advance rule that compares it with "
    "advance_lower_bound (repository path).",
    "experimental/neural_agent/tasks/composition.py: CompositionTask.classify, the verdicts and the executive "
    "(repository path).",
    "experimental/neural_agent/tasks/events.py: DAMAGE_RECEIVED is not produced by the flat arena; "
    "inject_damage_event is an ablation (ADR-001 D5) (repository path).",
    "experimental/neural_agent/probes/out/events.json, written by probes/probe_events.py: on combat/baseline a "
    "neutral actor received DAMAGE_RECEIVED on 17 of 1,024 lane-steps (batch 8, 128 steps), with 0 terminated and "
    "8 truncated episodes (local probe output).",
    "experimental/neural_agent/tasks/arenas.py (the two arenas) and experimental/neural_agent/session.py run_session "
    "(the eager runner and its host metrics) (repository paths).",
    "adk/architecture/hierarchy.py:273-282 (commitment, timeout, emergency), :347-356 (switch rule) and :372 "
    "(goal_valid update); adk/architecture/decision_context.py:415 (manager due); adk/runtime/hierarchical.py "
    "(collector). Repository paths.",
)


class ArmRun(NamedTuple):
    """The result of running one arm.

    Attributes:
        name: The arm name, as on the command line.
        row: The metrics row, keyed by :data:`COLUMNS`.
        records: The host arrays of the last timed run, shaped [T, B]. They are
            kept for inspection and tests.
        notes: Lines for the report's caveats that only this run can state.
        scratchpad: The shared state S_t of the timed eager session (``shared_state.py``),
            or None for the compiled runner, which has no session.
    """

    name: str
    row: dict[str, Any]
    records: dict[str, np.ndarray]
    notes: tuple[str, ...] = ()
    scratchpad: Any = None


@dataclasses.dataclass
class _Totals:
    """Running totals over the timed runs of one arm."""

    switches: int = 0
    core_switches: int = 0
    latencies: list[int] = dataclasses.field(default_factory=list)
    onsets: int = 0
    violations: int = 0
    completed: int = 0
    successes: int = 0
    composed_loose: int = 0
    pursue_only: int = 0
    #: Episodes that reached the objective: catches, or re-engagements on combat_baseline.
    catches: int = 0
    damage_events: int = 0
    #: Lane-ticks with the re-engagement distance known, and lane-ticks read (combat_baseline).
    distance_known: int = 0
    distance_ticks: int = 0
    #: The classify input and verdict of every completed episode, for inspection and tests.
    episodes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    last: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)


def missing_siblings() -> list[str]:
    """Return the sibling modules that cannot be found, in :data:`SIBLING_MODULES` order.

    A sibling whose parent package does not exist is reported as missing too.
    """

    missing: list[str] = []
    for name in SIBLING_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(name)
    return missing


def load_context(
    batch: int,
    ticks: int,
    *,
    seed: int = 0,
    lm_backend: str = "fake",
    j1_params: str | Path | None = None,
    js1_heads: str | Path | None = None,
) -> tuple[ArmContext, Any]:
    """Return the arm context and the composition task, built from the siblings.

    The executive is the one that ``CompositionTask.initial_executive_state``
    returns for the registry's catalogue. The task's tick budget is ``ticks``,
    which is the arena horizon as well. ``seed``, ``lm_backend``, ``j1_params``
    and ``js1_heads`` are passed to :func:`build_arm_context` for the ``j1``,
    ``j_lm`` and ``js1`` arms.

    Returns:
        ``(ctx, task)``.

    Raises:
        ImportError: If a sibling module is missing.
        ValueError: From :func:`build_arm_context` when the catalogue is not idle-first.
    """

    from experimental.neural_agent.skill_library.library import build_default_registry
    from experimental.neural_agent.tasks.composition import CompositionTask

    registry = build_default_registry(batch)
    task = CompositionTask(tick_budget=ticks)
    skill_ids = tuple(registry.ids)
    ctx = build_arm_context(
        skill_ids=skill_ids,
        registry=registry,
        batch=batch,
        executive=task.initial_executive_state(skill_ids),
        seed=seed,
        lm_backend=lm_backend,
        j1_params=j1_params,
        js1_heads=js1_heads,
    )
    return ctx, task


def load_arena(name: str, batch: int, seed: int, ticks: int) -> Any:
    """Return the named arena (``tasks/arenas.py`` ``BuiltArena``), with horizon ``ticks``.

    Raises:
        ImportError: If ``experimental.neural_agent.tasks.arenas`` is missing.
        ValueError: If the name is unknown, or ``ticks`` is outside the arena's range.
    """

    from experimental.neural_agent.tasks.arenas import build_arena

    return build_arena(name, batch, ticks, seed)


def _component_state(component: Any, role: str, batch: int) -> Any:
    """Return ``component.initial_state(batch)``, failing closed if it is missing."""

    initial = getattr(component, "initial_state", None)
    if not callable(initial):
        raise TypeError(f"the {role} must provide initial_state(batch)")
    return initial(batch)


def initial_hierarchy_state(
    batch: int,
    *,
    belief: Any,
    manager: Any,
    actor: Any,
) -> HierarchicalCoreState:
    """Return the reset hierarchy state for ``batch`` lanes.

    Shapes: ``previous_action`` is int32[B, ACTION_HEADS]. ``goal_age`` and
    ``action_age`` are float32[B]. The goal is :func:`idle_goal`, and
    ``manager_diagnostics`` is ``()``, the structure every arm manager returns.

    Raises:
        TypeError: If a component has no callable ``initial_state``.
    """

    carry = HierarchicalCarry(
        belief_state=_component_state(belief, "belief", batch),
        manager_state=_component_state(manager, "manager", batch),
        actor_state=_component_state(actor, "actor", batch),
        world_model_state=(),
        goal=idle_goal(batch),
        previous_action=jnp.zeros((batch, ACTION_HEADS), dtype=jnp.int32),
        goal_age=jnp.zeros((batch,), dtype=jnp.float32),
        action_age=jnp.zeros((batch,), dtype=jnp.float32),
    )
    return initialize_hierarchical_state(
        carry,
        manager_diagnostics=(),
        goal_valid=False,
        action_valid=False,
    )


def _terminated_flag(info: Any) -> jax.Array:
    """Return ``info.terminated`` as bool[B], or fail closed when the info lacks it.

    The flat arena collapses ``done`` and publishes the cause split in ``info``,
    so this flag is read from there.
    """

    flag = getattr(info, "terminated", None)
    if flag is None:
        raise TypeError(
            "the environment's info must publish 'terminated'; the composition metrics "
            "cannot be read without it"
        )
    return jnp.asarray(flag, dtype=jnp.bool_)


def composition_record_fn() -> Callable[[HierarchicalLoopTransition], dict[str, jax.Array]]:
    """Return the collector's record function for the composition metrics.

    Each returned array is shaped [B] or [B, 14] for one tick, and the
    collector stacks it to [T, ...]. ``active_skill`` is the goal in force
    after the decision, which is the goal that acted this tick.
    ``minimum_commitment`` is the minimum of the goal in force before the
    decision, since the core's commitment test uses that goal's control.
    ``terminated`` is the boundary's split when the producer publishes one,
    and ``info.terminated`` otherwise. ``target_distance_blocks`` and
    ``target_distance_known`` are the ``target_distance_blocks`` column of the
    belief that this tick's decision read (``belief_update.belief``, a
    ``SharedBeliefState``); the re-engagement objective reads them. The eager
    runner passes this same function to ``session.run_session``, so both
    runners record the same keys.
    """

    distance_column = feature_index("target_distance_blocks")

    def record(transition: HierarchicalLoopTransition) -> dict[str, jax.Array]:
        step = transition.hierarchy_step
        runtime = step.goal_runtime
        belief = step.belief_update.belief
        boundary = transition.boundary
        terminated = (
            boundary.terminated
            if boundary.split_known
            else _terminated_flag(transition.info)
        )
        return {
            "active_skill": step.active_goal.payload.skill_index,
            "goal_switched": step.goal_switched,
            "goal_valid": runtime.valid,
            "goal_age": runtime.age_seconds,
            "minimum_commitment": (
                transition.hierarchy_state.carry.goal.control.minimum_commitment_seconds
            ),
            "timed_out": runtime.timed_out,
            "emergency": runtime.emergency_interrupt,
            "active_events": runtime.active_events,
            "done": transition.done,
            "terminated": terminated,
            "reward": transition.reward,
            "target_distance_blocks": belief.features[:, distance_column],
            "target_distance_known": belief.feature_known[:, distance_column],
        }

    return record


def _verdict_names() -> tuple[str, str, str]:
    """Return ``(VERDICT_COMPOSED, VERDICT_COMPOSED_LOOSE, VERDICT_PURSUE_ONLY)`` from the task."""

    from experimental.neural_agent.tasks.composition import (
        VERDICT_COMPOSED,
        VERDICT_COMPOSED_LOOSE,
        VERDICT_PURSUE_ONLY,
    )

    return VERDICT_COMPOSED, VERDICT_COMPOSED_LOOSE, VERDICT_PURSUE_ONLY


def _entry_ticks(active: np.ndarray) -> tuple[int, ...]:
    """Return the ticks where a boolean column turns on. The first tick counts when it is on."""

    previous = np.concatenate(([False], active[:-1]))
    return tuple(int(tick) for tick in np.flatnonzero(active & ~previous))


def _score_run(
    host: Mapping[str, np.ndarray],
    totals: _Totals,
    task: Any,
    evader_index: int,
    pursuit_index: int,
    *,
    reengage: bool = False,
) -> None:
    """Add one timed run's host records to ``totals``.

    An ordinary switch is valid, not timed out, and not an emergency interrupt.
    Those switches are the only ones checked for commitment violations. An
    interrupt onset is answered by a core replacement, or by a skill change on a
    tick where a goal was in force. Each completed episode is classified by
    ``task.classify`` with the four keys of ``metrics.completed_episodes`` plus
    ``damage_ticks`` and ``pursuit_entry_ticks``, which the strict composed rule
    needs. Both are episode-relative, like the evade entries.

    With ``reengage`` (``combat_baseline``), ``terminated`` is a death or a world
    error, so ``caught`` is False, and the episode gets ``reengaged_tick`` read
    from the records ``target_distance_blocks`` and ``target_distance_known``
    from ``reengage_start_tick`` on: after the resume when the episode has an
    evade entry, else from the episode start. ``totals.catches`` then counts
    re-engagements.

    Raises:
        ValueError: From the metric reductions when the record shapes are wrong.
        KeyError: With ``reengage``, when the records lack the target distance.
    """

    from experimental.neural_agent.tasks.composition import reengage_start_tick, reengaged_tick

    valid = np.asarray(host["goal_valid"], dtype=np.bool_)
    totals.switches += int(np.sum(goal_switches(host["active_skill"], valid)))
    events = np.any(host["active_events"][..., list(INTERRUPT_EVENT_INDICES)], axis=-1)
    switched = np.asarray(host["goal_switched"], dtype=np.bool_)
    # A replacement is a switch on a tick where a goal was in force. The forced first goal
    # of an episode follows an autoreset, so it is neither a switch nor a reply to an event.
    replaced = switched & valid
    totals.core_switches += int(np.sum(replaced))
    # A host-mode arm (j_lm) writes its goal between steps, so the goal shows as a skill
    # change on the next tick and never as a core replacement. For an in-graph arm the skill
    # change adds nothing: the core changes the active goal only when it switches
    # (adk/architecture/hierarchy.py:356-357).
    answered = replaced | skill_changes(host["active_skill"], valid)
    totals.latencies.extend(interrupt_latency_ticks(events, answered))
    totals.onsets += interrupt_onset_count(events)
    ordinary = (
        switched
        & np.asarray(host["goal_valid"], dtype=np.bool_)
        & ~np.asarray(host["timed_out"], dtype=np.bool_)
        & ~np.asarray(host["emergency"], dtype=np.bool_)
    )
    totals.violations += commitment_violations(
        host["goal_age"][ordinary],
        host["minimum_commitment"][ordinary],
    )
    damage = host["active_events"][..., DAMAGE_EVENT_INDEX]
    episodes = completed_episodes(
        host["done"],
        host["terminated"],
        host["active_skill"],
        damage,
        evader_index,
    )
    composed, composed_loose, pursue_only = _verdict_names()
    skills = np.asarray(host["active_skill"])
    if reengage:
        distance = np.asarray(host["target_distance_blocks"])
        known = np.asarray(host["target_distance_known"])
        totals.distance_known += int(np.count_nonzero(known))
        totals.distance_ticks += int(known.size)
    for episode in episodes:
        lane = episode["lane"]
        window = slice(episode["start"], episode["end"] + 1)
        metrics = {
            key: episode[key]
            for key in ("caught", "catch_tick", "evade_entry_ticks", "damage_events")
        }
        metrics["damage_ticks"] = tuple(int(tick) for tick in np.flatnonzero(damage[window, lane]))
        metrics["pursuit_entry_ticks"] = _entry_ticks(skills[window, lane] == pursuit_index)
        if reengage:
            metrics["caught"], metrics["catch_tick"] = False, None
            start = reengage_start_tick(
                metrics["evade_entry_ticks"], metrics["pursuit_entry_ticks"], metrics["damage_ticks"]
            )
            metrics["reengaged_tick"] = (
                None if start is None else reengaged_tick(distance[window, lane], known[window, lane], start=start)
            )
        verdict = task.classify(metrics)
        totals.completed += 1
        totals.successes += int(verdict == composed)
        totals.composed_loose += int(verdict == composed_loose)
        totals.pursue_only += int(verdict == pursue_only)
        totals.catches += int(metrics["caught"] or metrics.get("reengaged_tick") is not None)
        totals.damage_events += episode["damage_events"]
        totals.episodes.append(
            {"lane": lane, "start": episode["start"], "end": episode["end"], **metrics, "verdict": verdict}
        )
    totals.last = dict(host)


def _distance_note(label: str, totals: _Totals) -> str:
    """Return the run note on how often the re-engagement distance was known."""

    return (
        f"`{label}`: the target distance was known on {totals.distance_known} of {totals.distance_ticks} "
        "lane-ticks. A re-engagement needs consecutive known ticks, so none can count where it is never known."
    )


def _finite_or_none(value: float) -> float | None:
    """Return ``value`` as a float, or None when it is NaN."""

    number = float(value)
    return None if math.isnan(number) else number


def _positive(value: int, name: str) -> int:
    """Return ``value`` as a positive int, or raise ValueError."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def host_arm_refusal(names: Sequence[str]) -> str:
    """Return the message that refuses host-mode arms on the compiled runner."""

    return (
        f"arm(s) {', '.join(names)} run a host-mode decision model between steps; the compiled "
        "collector calls its manager inside jax.lax.cond and cannot call it. Re-run with --runner eager."
    )


class _Composed(NamedTuple):
    """One arm composed for a run."""

    core: HierarchicalAgentCore
    manager: Any
    initial: HierarchicalCoreState
    evader_index: int
    pursuit_index: int


def _compose(
    arm: Arm,
    ctx: ArmContext,
    *,
    built: Any,
    ticks: int,
    episodes: int,
    seed: int,
) -> _Composed:
    """Check the run inputs and compose the arm's core. Both runners start here."""

    from experimental.neural_agent.tasks.composition import EVADER_SKILL, PURSUIT_SKILL

    _positive(ticks, "ticks")
    _positive(episodes, "episodes")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if getattr(built, "batch", None) != ctx.batch:
        raise ValueError(
            f"built.batch is {getattr(built, 'batch', None)!r}; the arm context has batch {ctx.batch}"
        )
    for skill in (EVADER_SKILL, PURSUIT_SKILL):
        if skill not in ctx.skill_ids:
            raise ValueError(f"the catalogue must name {skill!r}; it has {ctx.skill_ids}")

    belief = arm.make_belief(ctx)
    actor = arm.make_actor(ctx)
    manager = arm.make_manager(ctx)
    core = HierarchicalAgentCore(
        belief=belief,
        manager=manager,
        actor=actor,
        manager_cadence_seconds=MANAGER_CADENCE_SECONDS,
    )
    return _Composed(
        core=core,
        manager=manager,
        initial=initial_hierarchy_state(ctx.batch, belief=belief, manager=manager, actor=actor),
        evader_index=ctx.skill_ids.index(EVADER_SKILL),
        pursuit_index=ctx.skill_ids.index(PURSUIT_SKILL),
    )


def _arm_row(
    arm: Arm,
    ctx: ArmContext,
    totals: _Totals,
    *,
    episodes: int,
    seed: int,
    ticks: int,
    stats: JevStats | None,
    lane_failures: int | None,
    wall_seconds: float,
    warmup_seconds: float,
) -> dict[str, Any]:
    """Reduce an arm's totals to its metrics row. On combat_baseline the catch columns count re-engagements."""

    metrics = summarize(
        lane_runs=episodes * ctx.batch,
        episodes_completed=totals.completed,
        successes=totals.successes,
        pursue_only=totals.pursue_only,
        catches=totals.catches,
        goal_switches=totals.switches,
        core_switches=totals.core_switches,
        interrupt_latencies=totals.latencies,
        interrupt_onsets=totals.onsets,
        commitment_violation_count=totals.violations,
        jev_decisions=None if stats is None else int(stats.decisions),
        jev_misses=None if stats is None else int(stats.misses),
        jev_latency_ms_p50=None if stats is None else _finite_or_none(stats.latency_ms_p50),
        jev_latency_ms_p95=None if stats is None else _finite_or_none(stats.latency_ms_p95),
        wall_seconds=wall_seconds,
        warmup_seconds=warmup_seconds,
        ticks=ticks,
    )
    return {
        "arm": arm_label(arm.name, ctx),
        "episodes": episodes,
        "batch": ctx.batch,
        "seed": seed,
        **metrics,
        "composed_loose": totals.composed_loose,
        "damage_events": totals.damage_events,
        "jev_lane_failures": lane_failures,
    }


def run_arm(
    arm: Arm,
    ctx: ArmContext,
    *,
    built: Any,
    ticks: int,
    episodes: int,
    seed: int,
    task: Any,
    context_fn: HierarchicalContextFn = jax_hierarchical_context,
    actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
    catches_known: bool = True,
    host_respects_commitment: bool = False,  # accepted for a uniform call; the compiled runner has no host seam
) -> ArmRun:
    """Compose one arm, run it ``episodes`` times through the compiled collector, and return its row.

    Args:
        arm: The arm to run. A host-mode arm is refused.
        ctx: The arm context. ``ctx.batch`` must equal ``built.batch``, and the
            catalogue must name ``"evader"`` and ``"pursuit"``.
        built: The arena environment, as ``adk.runtime.hierarchical`` requires.
        ticks: Ticks per run, a positive int. It equals the arena horizon.
        episodes: Timed runs, a positive int. Each uses its own key.
        seed: Root seed, a non-negative int.
        task: A ``CompositionTask``. Its ``classify`` gives the episode verdicts.
        context_fn: Maps the step info to a DecisionContext.
        actor_input_fn: Maps the environment state to an ActorPolicyInput.
        catches_known: The arena's ``catch_from_terminated``. True: a terminated
            lane is a catch, and the catch is the objective. False
            (``combat_baseline``): terminated is a death or a world error, the
            objective is re-engagement (``tasks/composition.py``
            ``reengaged_tick``), and the catch columns count re-engagements.

    Returns:
        An :class:`ArmRun`. Its row has the keys in :data:`COLUMNS`.

    Raises:
        ValueError: If the arm is host-mode, ``built.batch`` differs from
            ``ctx.batch``, a count is not positive, the seed is negative, or the
            catalogue lacks ``"evader"`` or ``"pursuit"``.
        TypeError: If a component has no callable ``initial_state``.
    """

    if arm.make_host_decider is not None:
        raise ValueError(host_arm_refusal([arm.name]))
    composed = _compose(arm, ctx, built=built, ticks=ticks, episodes=episodes, seed=seed)
    collector = compile_hierarchical_collector(
        built,
        composed.core,
        composition_record_fn(),
        ticks,
        initial_state=composed.initial,
        context_fn=context_fn,
        actor_input_fn=actor_input_fn,
    )
    root = jax.random.key(seed)
    started = time.perf_counter()
    jax.block_until_ready(collector(jax.random.fold_in(root, 0)))
    warmup_seconds = time.perf_counter() - started

    totals = _Totals()
    wall_seconds = 0.0
    for episode in range(episodes):
        started = time.perf_counter()
        _, records = collector(jax.random.fold_in(root, episode))
        jax.block_until_ready(records)
        wall_seconds += time.perf_counter() - started
        host = {key: np.asarray(value) for key, value in records.items()}
        _score_run(
            host, totals, task, composed.evader_index, composed.pursuit_index, reengage=not catches_known
        )

    row = _arm_row(
        arm,
        ctx,
        totals,
        episodes=episodes,
        seed=seed,
        ticks=ticks,
        stats=composed.manager.jev_stats(),
        lane_failures=None,
        wall_seconds=wall_seconds,
        warmup_seconds=warmup_seconds,
    )
    notes = () if catches_known else (_distance_note(row["arm"], totals),)
    return ArmRun(name=arm.name, row=row, records=totals.last, notes=notes)


def run_arm_eager(
    arm: Arm,
    ctx: ArmContext,
    *,
    built: Any,
    ticks: int,
    episodes: int,
    seed: int,
    task: Any,
    catches_known: bool = True,
    host_respects_commitment: bool = False,
) -> ArmRun:
    """Compose one arm and run it through ``session.run_session``, the eager loop.

    One untimed one-step session warms the per-operation caches. The timed
    session then runs ``episodes * ticks`` steps with the key
    ``fold_in(key(seed), 0)``, so every lane covers ``episodes`` horizons. A
    host-mode arm gets a fresh decider from ``arm.make_host_decider`` for each
    session, and its JEV columns are the session's host metrics. Any other arm
    reports ``manager.jev_stats()``, as the compiled runner does.

    Arguments, returns and errors are those of :func:`run_arm`, except that
    host-mode arms are accepted. The session uses the default context and actor
    input functions, so this runner needs an arena whose info the default
    ``jax_hierarchical_context`` reads.
    """

    from experimental.neural_agent.session import SessionConfig, run_session
    from experimental.neural_agent.shared_state import ExecutiveSection, SharedState

    composed = _compose(arm, ctx, built=built, ticks=ticks, episodes=episodes, seed=seed)
    record_fn = composition_record_fn()
    key = jax.random.fold_in(jax.random.key(seed), 0)

    def session(steps: int, shared: SharedState | None = None) -> tuple[Any, Any]:
        decider = None if arm.make_host_decider is None else arm.make_host_decider(ctx, steps)
        result = run_session(
            built,
            composed.core,
            composed.initial,
            key,
            SessionConfig(
                steps=steps,
                host_budget_ms=J0_LATENCY_BUDGET_MS,
                host_respects_commitment=host_respects_commitment,
            ),
            skill_ids=ctx.skill_ids,
            host_decider=decider,
            shared_state=shared,
            record_fn=record_fn,
        )
        return result, decider

    started = time.perf_counter()
    session(1)
    warmup_seconds = time.perf_counter() - started
    # The scratchpad opens with the arm's fixed E_t in the executive section (no harness runs
    # here); the session writes the sections it owns.
    scratchpad = SharedState(executive=ExecutiveSection.from_executive_state(ctx.executive))
    result, decider = session(episodes * ticks, scratchpad)
    host = {name: np.asarray(value) for name, value in result.records.items()}
    totals = _Totals()
    _score_run(
        host, totals, task, composed.evader_index, composed.pursuit_index, reengage=not catches_known
    )

    notes: tuple[str, ...] = ()
    lane_failures = None
    stats = composed.manager.jev_stats()
    if decider is not None:
        measured = result.metrics.host
        stats = JevStats(
            decisions=measured.decisions,
            misses=measured.misses,
            latency_ms_p50=measured.latency_ms_p50,
            latency_ms_p95=measured.latency_ms_p95,
            compute_units=0,
        )
        lane_failures = getattr(decider, "lane_failures", None)
        notes = (
            f"`{arm_label(arm.name, ctx)}` (eager session): {measured.decisions} host decisions, "
            f"{measured.misses} misses, {measured.errors} decider errors; "
            f"{lane_failures} lane failures in {getattr(decider, 'lane_calls', 'n/a')} lane calls; "
            f"first failure: {getattr(decider, 'first_failure', None) or 'none'}.",
        )
    row = _arm_row(
        arm,
        ctx,
        totals,
        episodes=episodes,
        seed=seed,
        ticks=ticks,
        stats=stats,
        lane_failures=lane_failures,
        wall_seconds=result.metrics.wall_seconds,
        warmup_seconds=warmup_seconds,
    )
    if not catches_known:
        notes += (_distance_note(row["arm"], totals),)
    return ArmRun(name=arm.name, row=row, records=totals.last, notes=notes, scratchpad=scratchpad)


def _cell(value: Any) -> str:
    """Format one CSV cell. None becomes ``n/a``, and floats get six decimals."""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_metrics_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write one CSV row per arm, with the columns in :data:`COLUMNS`.

    Raises:
        ValueError: If a row has a key that is not in :data:`COLUMNS`.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            unexpected = sorted(set(row) - set(COLUMNS))
            if unexpected:
                raise ValueError(f"row has keys outside COLUMNS: {unexpected}")
            writer.writerow({key: _cell(row.get(key)) for key in COLUMNS})
    return target


def _rate_cell(rate: float | None, lower: float | None, successes: int | None, completed: int) -> str:
    """Format success as ``successes/completed rate (Wilson lower)``, or n/a."""

    if rate is None or lower is None:
        return f"n/a (0/{completed} completed)" if completed == 0 else "n/a"
    return f"{successes}/{completed} = {rate:.3f} ({lower:.3f})"


def _latency_cell(row: Mapping[str, Any]) -> str:
    """Format the interrupt latency median with answered and onset counts."""

    median = row["interrupt_latency_median_ticks"]
    median_text = "n/a" if median is None else f"{median:g}"
    return f"{median_text} ({row['interrupt_answered']} / {row['interrupt_onsets']})"


def _misses_cell(row: Mapping[str, Any]) -> str:
    """Format JEV misses over decisions, or n/a when they are not measured."""

    if row["jev_decisions"] is None:
        return "n/a (not measured)"
    cell = f"{row['jev_misses']} / {row['jev_decisions']}"
    if row.get("jev_lane_failures") is not None:
        cell += f" (lane failures {row['jev_lane_failures']})"
    return cell


def _latency_ms_cell(row: Mapping[str, Any]) -> str:
    """Format the JEV p50 and p95 latency in milliseconds, or n/a when not measured."""

    if row["jev_latency_ms_p50"] is None:
        return "n/a (not measured)"
    return f"{row['jev_latency_ms_p50']:.3f} / {row['jev_latency_ms_p95']:.3f}"


def _caveats(runs: Sequence[ArmRun], *, arena: str, runner: str) -> list[str]:
    """Return the caveat lines for this arena, runner and set of arms."""

    from experimental.neural_agent.tasks.composition import REENGAGE_DISTANCE_BLOCKS, REENGAGE_RUN_STEPS

    labels = {run.row["arm"] for run in runs}
    lines = [
        "- Episodes completed counts only the episodes that reached `done` inside the run. "
        "Success, composed-loose and pursue-only are read from the composition task's `classify`. "
        "Composed-loose is a catch after an evade entry that misses the strict rule; it is never a success.",
    ]
    if arena == "flat_pursuit":
        lines += [
            "- Terminated is read as a catch. This is an inference, not a test: the flat arena closes combat, "
            "so no actor can die there (`tasks/flat_arena.py` step_detailed and `tasks/events.py`).",
            "- The flat arena produces no DAMAGE_RECEIVED event (`tasks/events.py`). This run injects none, "
            "because the injection is an ablation (ADR-001 D5). So the interrupt columns are empty here, and "
            "an episode cannot be composed without an injected damage event.",
        ]
    else:
        from experimental.neural_agent.tasks.arenas import COMBAT_BASELINE_AGENT, COMBAT_OPEN_FLAT, COMBAT_OPEN_FLAT_SCENE

        # Before 2026-10-17 this branch named combat_baseline for every combat arena.
        kit_name = COMBAT_OPEN_FLAT_SCENE if arena == COMBAT_OPEN_FLAT else COMBAT_BASELINE_AGENT
        damage_note = (
            "" if arena == COMBAT_OPEN_FLAT
            else " (`probes/out/events.json`: 17 of 1,024 lane-steps for a neutral actor)"
        )
        lines += [
            f"- The arena is `{arena}` (`tasks/arenas.py`): AgentKit's `{kit_name}` with an episode "
            f"limit of `ticks`" + (" in the Gym's legal observation mode (`legal_observation=True`, 2026-10-17)" if arena == COMBAT_OPEN_FLAT else "") + f". DAMAGE_RECEIVED fires there without injection{damage_note}, so the evade "
            "trigger and the interrupt columns are live.",
            f"- Nothing can be caught on `{arena}`: `terminated` is a death or a world error "
            "(`adk/runtime/env_adapter.py` BuiltAgent.step_detailed), so it is never read as a catch. The objective "
            "there is re-engagement (owner decision, 2026-10-16): the known target distance, read from the belief's "
            f"`target_distance_blocks`, is at most {REENGAGE_DISTANCE_BLOCKS:g} blocks on {REENGAGE_RUN_STEPS} "
            "consecutive decision steps of 0.045 s, counted after the resumed pursuit when the episode has an evade "
            "entry and from the episode start otherwise (`tasks/composition.py` `reengage_start_tick`, "
            "`reengaged_tick`). Success, composed-loose and pursue-only are read with the re-engagement in place of "
            "the catch, and the catch columns of `metrics.csv` (`catches`, `catches_per_episode`) count "
            "re-engagements here. Damage injection stays an ablation.",
            "- The distance feature is known only while the target is visible (`belief/features.py`), so a tick "
            "without sight breaks a re-engagement run.",
        ]
    if runner == "compiled":
        lines.append(
            "- JEV decisions, misses and latency are not measured for in-graph instances (J0, J1 and the "
            "shuffled arm). `JevRunner` counts eager calls only, and its latency is NaN for in-graph instances "
            "(`jev/runner.py` `stats`)."
        )
    else:
        lines.append(
            "- Eager runner: each arm ran one session of episodes x ticks steps through `session.run_session`, "
            "after a one-step warm-up session. For a host-mode arm the JEV columns are the session's host "
            "metrics; when a host decision switches any lane, the session writes the goal with goal age 0 on "
            "every lane, so a continuing lane's age restarts too (`session.py` run_session). A host-written goal changes the "
            "skill-changes column but is not a core replacement; interrupt latency credits it through that skill "
            "change (see the interrupt caveat). In-graph instances stay unmeasured."
        )
    if "j1-untrained" in labels:
        lines.append(
            "- `j1-untrained` is NetworkJev with `init_j1` random parameters from the run seed. No trained "
            "checkpoint was supplied (`--j1-params`), so its row tests the wiring, not a learned JEV."
        )
    if "j_lm-fake" in labels:
        lines.append(
            "- `j_lm-fake` is LanguageModelJev over FakeDecisionLM, scripted to answer pursuit on every call. "
            "No language model runs, so its latency is the host overhead of one decision."
        )
    if "j_lm-ollama" in labels:
        lines.append(
            "- `j_lm-ollama` calls qwen3:4b on a local Ollama server, each call bounded by the "
            f"{J0_LATENCY_BUDGET_MS:g} ms budget. `jev/lm/ollama_qwen.py` measured 3.7 to 9.9 s per decision, "
            "so lane failures (timeouts) are expected (inferred, not measured here)."
        )
    lines += [
        "- The belief parameters are a random initialization with seed "
        f"{BELIEF_PARAMS_SEED}. The arena publishes no speed scales, so the belief's self-speed and closing-speed "
        "features are unknown in this run (`belief/core.py` docstring).",
        "- Skill changes count a change of skill index on ticks where a goal was already in force. Core "
        "replacements count the core's switch flag on those ticks, so a same-index replacement appears there and "
        "not in skill changes. The forced first goal of each episode (an autoreset) is excluded from both and "
        "from interrupt latency. The shuffled arm is read from the core column.",
        "- Interrupt latency runs from an onset to its answer: the first tick at or after it with a core "
        "replacement, or with a skill change on a tick where a goal was in force. The skill change credits a "
        "host-mode arm (`j_lm`), whose goal is written between steps and shows on the next tick, so its latency is "
        "at least 1; a host write that keeps the skill is not an answer. For an in-graph arm it adds nothing, "
        "because the core changes the active goal only when it switches (`adk/architecture/hierarchy.py:356-357`).",
        "- Commitment violations count only ordinary switches. The core's switch rule "
        "(`adk/architecture/hierarchy.py:347-356`) should make this zero, so a nonzero value points at "
        "the core or at a bypass that was recorded wrongly (inferred from reading the rule, not tested "
        "against a live arena).",
        "- The flat arm has no JEV, so its JEV columns are n/a by design.",
    ]
    for run in runs:
        lines += [f"- {note}" for note in run.notes]
    return lines


def render_report(
    runs: Sequence[ArmRun],
    *,
    config: Mapping[str, Any],
    generated_at: str,
    arena: str = DEFAULT_ARENA,
    runner: str = DEFAULT_RUNNER,
) -> str:
    """Return the text of REPORT.md for the runs.

    The report has a configuration list, a per-arm table, the caveats of this
    run (which depend on ``arena``, ``runner`` and the arm labels), the thesis,
    the formulas and the citations.
    """

    lines: list[str] = [
        "# Composition arms: run report",
        "",
        f"Generated {generated_at}. Source: `experimental/neural_agent/eval/run_composition.py`.",
        "",
        "## Run configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    objective = "Re-engagements / episode" if arena == "combat_baseline" else "Catches / episode"
    lines += [
        "",
        "## Per-arm table",
        "",
        "| Arm | Lane-runs | Episodes completed | Success: n/completed = rate (Wilson lower 95%) "
        f"| Composed-loose | Pursue-only | {objective} | Damage events | Skill changes / lane-run "
        "| Core replacements / lane-run "
        "| Interrupt latency median ticks (answered / onsets) | Commitment violations "
        "| JEV misses / decisions | JEV p50 / p95 ms | Wall s (warm-up s) | Ticks |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        row = run.row
        catches = (
            "n/a"
            if row["catches_per_episode"] is None
            else f"{row['catches_per_episode']:.3f}"
        )
        pursue = "n/a" if row["pursue_only"] is None else str(row["pursue_only"])
        loose = "n/a" if row.get("composed_loose") is None else str(row["composed_loose"])
        damage = "n/a" if row.get("damage_events") is None else str(row["damage_events"])
        lines.append(
            f"| {row['arm']} | {row['lane_runs']} | {row['episodes_completed']} "
            f"| {_rate_cell(row['success_rate'], row['success_wilson_lower'], row['successes'], row['episodes_completed'])} "
            f"| {loose} | {pursue} | {catches} | {damage} | {row['goal_switches_per_lane_run']:.3f} | {row['core_switches_per_lane_run']:.3f} | {_latency_cell(row)} "
            f"| {row['commitment_violations']} | {_misses_cell(row)} | {_latency_ms_cell(row)} "
            f"| {row['wall_seconds']:.3f} ({row['warmup_seconds']:.3f}) | {row['ticks']} |"
        )
    lines += [
        "",
        "## Status and caveats",
        "",
        *_caveats(runs, arena=arena, runner=runner),
        "",
        "## Thesis",
        "",
        THESIS_TEXT,
        "",
        "## Formulas",
        "",
        "```text",
        FORMULA_TEXT,
        "```",
        "",
        "## Citations",
        "",
    ]
    lines += [f"- {citation}" for citation in CITATION_LINES]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line. ``--help`` exits with status 0."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.eval.run_composition",
        description="Run the composition arms and write metrics.csv and REPORT.md.",
    )
    parser.add_argument(
        "--arms",
        type=_parse_arms,
        default=DEFAULT_ARMS,
        help=(
            "comma-separated arms from: flat, j0, shuffled, j1, j_lm, js1 (default flat,j0,shuffled); "
            "j_lm and js1 run on the host and need --runner eager"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=2,
        help="timed collector runs per arm, each with its own key (default 2)",
    )
    parser.add_argument("--batch", type=_positive_int, default=8, help="lanes per run (default 8)")
    parser.add_argument(
        "--ticks",
        type=_positive_int,
        default=DEFAULT_TICKS,
        help=(
            f"ticks per run, also the arena horizon (default {DEFAULT_TICKS}: at 256 the default arena rarely "
            "completes a re-engagement after an evade, see docs/LEDGER.md); flat_pursuit accepts 9 to 1000"
        ),
    )
    parser.add_argument("--seed", type=_non_negative_int, default=0, help="root seed (default 0)")
    parser.add_argument(
        "--arena",
        choices=ARENA_CHOICES,
        default=DEFAULT_ARENA,
        help=(
            "combat_open_flat (default): the target is visible and DAMAGE_RECEIVED fires, so "
            "re-engagement can be scored; combat_baseline: damage fires but the target is never "
            "visible (the fail_closed scene); flat_pursuit: a catch ends an episode, no damage happens"
        ),
    )
    parser.add_argument(
        "--runner",
        choices=RUNNER_CHOICES,
        default=DEFAULT_RUNNER,
        help=(
            "compiled (default): compile_hierarchical_collector; eager: session.run_session, "
            "the Python step loop, which host-mode arms (j_lm) require"
        ),
    )
    parser.add_argument(
        "--lm-backend",
        choices=LM_BACKENDS,
        default="fake",
        help=(
            "j_lm backend: fake (default; FakeDecisionLM answering pursuit, no network) or "
            "ollama (qwen3:4b on a local Ollama server)"
        ),
    )
    parser.add_argument(
        "--j1-params",
        type=Path,
        default=None,
        help=(
            "trained J1 parameters, an .npz keyed by the J1Params field names; without it the j1 "
            "arm uses init_j1 random parameters and is labelled j1-untrained"
        ),
    )
    parser.add_argument(
        "--host-respects-commitment",
        action="store_true",
        help="eager runner: apply a host decider's goal per lane only where the skill's commitment allows a switch (mirrors the core)",
    )
    parser.add_argument(
        "--js1-heads",
        type=Path,
        default=None,
        help=(
            "trained J-S1 heads, an .npz keyed by the HeadsParams field names; without it the js1 "
            "arm uses init_heads random parameters and is labelled js1-untrained"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for metrics.csv and REPORT.md (default eval/out, which is git-ignored)",
    )
    return parser.parse_args(argv)


def _positive_int(text: str) -> int:
    """argparse type: a positive integer."""

    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    """argparse type: a non-negative integer."""

    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return value


def _parse_arms(text: str) -> tuple[str, ...]:
    """argparse type: a comma-separated list of registered arm names, with no repeats."""

    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("--arms must name at least one arm")
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown arm(s) {unknown}; expected a subset of {list(ARMS)}")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"--arms repeats an arm: {names}")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested arms and write ``metrics.csv`` and ``REPORT.md``.

    Returns:
        0 on success. 2 when a sibling module is missing, or when a host-mode
        arm is asked for the compiled runner. The message goes to stderr, and
        nothing is written.
    """

    args = parse_args(argv)
    missing = missing_siblings()
    if missing:
        sys.stderr.write(
            "cannot run: sibling modules are not written yet: " + ", ".join(missing) + "\n"
        )
        return 2
    arms = [get_arm(name) for name in args.arms]
    host_arms = [arm.name for arm in arms if arm.make_host_decider is not None]
    if host_arms and args.runner == "compiled":
        sys.stderr.write(host_arm_refusal(host_arms) + "\n")
        return 2
    ctx, task = load_context(
        args.batch,
        args.ticks,
        seed=args.seed,
        lm_backend=args.lm_backend,
        j1_params=args.j1_params,
        js1_heads=args.js1_heads,
    )
    arena = load_arena(args.arena, args.batch, args.seed, args.ticks)
    run = run_arm if args.runner == "compiled" else run_arm_eager
    runs = [
        run(
            arm,
            ctx,
            built=arena.built,
            ticks=args.ticks,
            episodes=args.episodes,
            seed=args.seed,
            task=task,
            catches_known=arena.catch_from_terminated,
            host_respects_commitment=args.host_respects_commitment,
        )
        for arm in arms
    ]
    out_dir = Path(args.out)
    write_metrics_csv(out_dir / "metrics.csv", [result.row for result in runs])
    for result in runs:
        if result.scratchpad is not None:
            from experimental.neural_agent.shared_state import view_for_llm

            (out_dir / f"scratchpad_{result.name}.json").write_text(result.scratchpad.to_json(), encoding="utf-8")
            (out_dir / f"scratchpad_{result.name}_llm_view.md").write_text(
                view_for_llm(result.scratchpad), encoding="utf-8"
            )
    from experimental.neural_agent.tasks.composition import (
        EVADE_TRIGGER_WINDOW_TICKS,
        REENGAGE_DISTANCE_BLOCKS,
        REENGAGE_RUN_STEPS,
    )

    config = {
        "arms": ", ".join(result.row["arm"] for result in runs),
        "arena": args.arena,
        "runner": args.runner,
        "episodes per arm": args.episodes,
        "batch (lanes)": args.batch,
        "ticks per run (arena horizon)": args.ticks,
        "seed": args.seed,
        **(
            {"j1 parameters": "init_j1 random (j1-untrained)" if args.j1_params is None else str(args.j1_params)}
            if "j1" in args.arms
            else {}
        ),
        **({"j_lm backend": args.lm_backend} if "j_lm" in args.arms else {}),
        **(
            {"js1 heads": "init_heads random (js1-untrained)" if args.js1_heads is None else str(args.js1_heads)}
            if "js1" in args.arms
            else {}
        ),
        "evade trigger window W (ticks)": EVADE_TRIGGER_WINDOW_TICKS,
        **(
            {
                "objective (re-engagement)": (
                    f"known target distance <= {REENGAGE_DISTANCE_BLOCKS:g} blocks on "
                    f"{REENGAGE_RUN_STEPS} consecutive decision steps"
                )
            }
            if args.arena == "combat_baseline"
            else {}
        ),
        "manager cadence (s)": MANAGER_CADENCE_SECONDS,
        "backend": jax.default_backend(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "numpy": np.__version__,
    }
    report = render_report(
        runs,
        config=config,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        arena=args.arena,
        runner=args.runner,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    sys.stdout.write(f"wrote {out_dir / 'metrics.csv'} and {out_dir / 'REPORT.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARENA_CHOICES",
    "ArmRun",
    "COLUMNS",
    "DEFAULT_ARENA",
    "DEFAULT_RUNNER",
    "DEFAULT_TICKS",
    "INTERRUPT_EVENTS",
    "MANAGER_CADENCE_SECONDS",
    "RUNNER_CHOICES",
    "composition_record_fn",
    "host_arm_refusal",
    "initial_hierarchy_state",
    "load_arena",
    "load_context",
    "main",
    "missing_siblings",
    "parse_args",
    "render_report",
    "run_arm",
    "run_arm_eager",
    "write_metrics_csv",
]
