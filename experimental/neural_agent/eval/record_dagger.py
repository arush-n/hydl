"""DAgger-style relabelling for the learned decider ``js1``: the rule labels the learner's own states.

Why (``docs/LEDGER.md``, P1 to P1d). The js1 heads (``SystemOneJev``, structured heads over
``HashedBagTrunk``) agree with the scripted rule J0 on every recorded held-out row, yet compose
worse online (0.40 against 0.53). They were cloned from J0's own rollouts, they under-react to the
evade transition, and once they miss a switch they sit in states that the J0 traces never contain.
That is the covariate shift of behaviour cloning. DAgger (Ross, Gordon, Bagnell, AISTATS 2011,
``docs/REFERENCES.md`` [24]) removes it by labelling the states that the learner itself visits with
the teacher. J0 is a pure function of (B_t, the goal in force, E_t), so its labels cost nothing.

One round k = 1 .. R. pi_0 is the ``--heads`` file::

    S_k  = the states pi_{k-1} decides on online: every host consultation of one eager session
           (session.run_session via run_composition.run_arm_eager, commitment-gated), every lane,
           on recording seed ``--seed + k``
    D_k  = {(s, J0(s)) : s in S_k}      J0(s) = (chosen skill, terminate) on the same B_t, goal in force and lowered E_t
    D    = D_J0 U D_1 U ... U D_k       D_J0 = the recorded J0 training rows (eval/out/traces/train/dataset.npz)
    pi_k = train_supervised(D)          the G4 command: --trunk hashed --goal-reading 4096 --text-path off, seed 0
    stop at the first k where, in one run_composition evaluation (--arms j0,js1 --runner eager
    --host-respects-commitment, seed ``--seed``, never a recording seed):
           rate(js1) >= WilsonLower95(j0)

The learner drives the episode. :class:`RelabelingDecider` wraps the js1 host decider that
``eval/arms.py`` builds (``get_arm("js1").make_host_decider``) and returns its goal object unchanged.
It keeps references to what the learner read: B_t, the goal in force, and the lowered E_t. JAX arrays
are immutable and the eager loop donates no buffer, so the references stay valid. The rule runs once,
after the session (:meth:`RelabelingDecider.labelled`), on those inputs. Labelling therefore adds
nothing to the timed decision, and the session counts a decision over 20 ms as a miss. J0 reads only
the belief's features and known flags, the current skill and the lowered E_t: ``jev/scripted.py``
``__call__`` deletes the actor input, the key, the decision context and the goal runtime. The
deferred call is therefore the call it would have made on the tick.

Rows. The layout is ``eval/traces.py`` TRACE_KEYS, the layout that ``record_traces.merge_decisions``
reads. Row = call * B + lane:

* ``features``, ``feature_known``, ``latent`` and ``exec_*``: as the learner read them.
* ``current_skill``: the goal in force.
* ``target_skill``: J0's chosen skill.
* ``target_replace``: J0's terminate, which is chosen != current. It is the rule's request, before
  any commitment gate.
* ``manager_due``: True, because the host decider was consulted.

The recorded J0 rows label ``target_replace`` with the core's ``goal_switched`` instead, which a
commitment can hold. On 2026-10-17, 61 of the 2,966 J0 train rows asked for a change that the
commitment held and carry False. ``decisions.npz`` sits beside ``dataset.npz``, row-aligned, with the
columns that ``write_trace`` does not accept: ``call``, ``lane``, ``tick``, ``learner_switch``,
``learner_skill``, ``rule_switch``, ``rule_skill`` and ``disagree``. ``tick`` is the eager session's
0-based step index; its clock tick is ``tick + 1``.

Disagreement, per lane-decision, learner L against rule R. A switch is a requested change of the goal
in force. The skill is the skill in force after the decision: the requested skill where L or R
switches, else the current one::

    disagree            = [switch_L != switch_R]  or  [skill_L != skill_R]
    switch learner-only = switch_L and not switch_R
    switch rule-only    = switch_R and not switch_L
    skill               = switch_L and switch_R and skill_L != skill_R

The P1d shadow (ledger) counted a different "skill" rate: the raw argmax of every lane, including
lanes that did not switch. Its two switch rates are the same quantities as here.

Run (CPU, from the repository root)::

    PYTHONPATH=. JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.eval.record_dagger \\
        --heads <heads.npz> --rounds 3 --out <dir> --seed 0

Each round writes these files under ``<out>/round_<k>/``:

* ``dataset.npz``: D_k, written with ``write_trace`` and checked with ``validate_dataset``.
* ``decisions.npz``: the row-aligned extra columns above.
* ``disagreement.json``: the disagreement counts and rates.
* ``record/metrics.csv``, ``record/REPORT.md``: the recording run's table.
* ``merged.npz``: D.
* ``train/heads.npz``, ``train/metrics.csv``, ``train/train.log``: pi_k and the trainer's output.
* ``eval/metrics.csv``, ``eval/REPORT.md``: the evaluation run.

``<out>/summary.json`` collects every round and is rewritten after each one.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.contracts import ExecutiveArrays, GoalPayload, SharedBeliefState
from experimental.neural_agent.eval.record_traces import merge_decisions
from experimental.neural_agent.eval.traces import TRACE_KEYS, write_trace
from experimental.neural_agent.jev.train_j1 import FIELDS, validate_dataset

_EVAL_DIR = Path(__file__).resolve().parent
#: The recorded J0 rows the G4 heads were trained on, and the recorded held-out rows of gate G1.
DEFAULT_J0_DATA: Path = _EVAL_DIR / "out" / "traces" / "train" / "dataset.npz"
DEFAULT_HELDOUT: Path = _EVAL_DIR / "out" / "traces" / "heldout" / "dataset.npz"
#: Gate G1: the recorded held-out agreement (skill and replace) that each retrained heads file keeps.
HELDOUT_AGREEMENT_BAR: float = 0.95
#: The G4 heads' training command (ledger P1 / G4): goal-reading states, text path off, hashed trunk, seed 0.
DEFAULT_GOAL_READING: int = 4096
TRAIN_SEED: int = 0
#: The arm that the evaluation compares against, and the learned arm.
RULE_ARM: str = "j0"
LEARNED_ARM: str = "js1"

_HELDOUT_LINE = re.compile(
    r"^heldout .+?: (?P<rows>\d+) rows, skill_agreement (?P<skill>[0-9.]+), replace_agreement (?P<replace>[0-9.]+)",
    re.MULTILINE,
)


class RelabelingDecider:
    """A ``session.HostDecider`` that lets the learner decide and keeps every consultation for the rule.

    ``decide`` calls the wrapped decider and returns its goal unchanged, so the learner drives the
    episode. It also appends references to the learner's inputs: the belief B_t, the goal in force,
    the lowered E_t (``carry.manager_state``) and the goal it returned. The inputs are read from the
    same places as ``arms.LanguageModelHostDecider``. :meth:`labelled` runs the rule on them after the
    session.

    Args:
        decider: The learner's host decider. For js1 it is ``get_arm("js1").make_host_decider(ctx, steps)``.
        rule: The teacher, called once on all recorded inputs: ``ScriptedJev`` (J0).
        skill_ids: The catalogue in dispatcher order.
    """

    def __init__(self, decider: Any, rule: Any, skill_ids: Sequence[str]) -> None:
        self._decider = decider
        self._rule = rule
        self.skill_ids = tuple(skill_ids)
        self._calls: list[tuple[Any, Any, Any, Any]] = []

    @property
    def calls(self) -> int:
        """Consultations recorded so far."""

        return len(self._calls)

    # ``run_composition.run_arm_eager`` reads these three for its report note, so pass them through.
    @property
    def lane_calls(self) -> Any:
        return getattr(self._decider, "lane_calls", None)

    @property
    def lane_failures(self) -> Any:
        return getattr(self._decider, "lane_failures", None)

    @property
    def first_failure(self) -> Any:
        return getattr(self._decider, "first_failure", None)

    def decide(self, transition: Any, hierarchy_state: Any, executive: Any, /) -> Any:
        """Return the learner's goal (or None) unchanged and keep the inputs it decided on.

        A raise in the learner propagates before anything is kept. The session counts it as an
        error, and :meth:`labelled` then finds one consultation fewer than the session made.
        """

        goal = self._decider.decide(transition, hierarchy_state, executive)
        self._calls.append(
            (
                transition.hierarchy_step.belief_update.belief,
                hierarchy_state.carry.goal.payload,
                hierarchy_state.carry.manager_state,
                None if goal is None else goal.payload,
            )
        )
        return goal

    def labelled(self, steps: Sequence[int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Label every recorded lane-decision with the rule. Return ``(trace, decisions)``.

        Args:
            steps: The session step index of each consultation, in order: one per recorded call.
                :func:`consulted_steps` gives them.

        Returns:
            ``trace`` holds the TRACE_KEYS arrays for ``write_trace``. ``decisions`` holds the
            row-aligned extra columns of the module docstring.

        Raises:
            ValueError: If nothing was recorded, or ``steps`` does not have one entry per call.
        """

        calls = len(self._calls)
        if calls == 0:
            raise ValueError("the host decider was never consulted; there is nothing to label")
        steps_array = np.asarray(steps, dtype=np.int64)
        if steps_array.shape != (calls,):
            raise ValueError(
                f"{calls} consultations were recorded but {steps_array.shape} consulted steps were given; "
                "a decide call that raised is not recorded"
            )
        beliefs, currents, lowered, proposed = zip(*self._calls)

        def stack(leaves: Sequence[Any], dtype: Any) -> np.ndarray:
            return np.concatenate([np.asarray(leaf) for leaf in leaves], axis=0).astype(dtype)

        lanes = int(np.shape(currents[0].skill_index)[0])
        rows = calls * lanes
        features = stack([b.features for b in beliefs], np.float32)
        known = stack([b.feature_known for b in beliefs], np.bool_)
        latent = stack([b.latent for b in beliefs], np.float32)
        current = GoalPayload(
            skill_index=stack([c.skill_index for c in currents], np.int32),
            params=stack([c.params for c in currents], np.float32),
            executive_version=stack([c.executive_version for c in currents], np.int32),
            goal_slot=stack([c.goal_slot for c in currents], np.int32),
        )
        executive = ExecutiveArrays(
            skill_index=stack([e.skill_index for e in lowered], np.int32),
            priority=stack([e.priority for e in lowered], np.float32),
            params=stack([e.params for e in lowered], np.float32),
            goal_valid=stack([e.goal_valid for e in lowered], np.bool_),
            budget=stack([e.budget for e in lowered], np.float32),
            version=stack([e.version for e in lowered], np.int32),
        )
        # The learner's decision: the goal it returned, or the goal in force where it returned None.
        chosen = [current_payload if goal is None else goal for current_payload, goal in zip(currents, proposed)]
        learner_skill = stack([g.skill_index for g in chosen], np.int32)
        learner_switch = (
            (learner_skill != current.skill_index)
            | (stack([g.goal_slot for g in chosen], np.int32) != current.goal_slot)
            | np.any(stack([g.params for g in chosen], np.float32) != current.params, axis=-1)
        )
        decision = self._rule(
            self._rule.initial_state(rows),
            SharedBeliefState(features=jnp.asarray(features), feature_known=jnp.asarray(known), latent=jnp.asarray(latent)),
            GoalPayload(*(jnp.asarray(leaf) for leaf in current)),
            None,
            None,
            decision_context=None,
            goal_runtime=None,
            executive=ExecutiveArrays(*(jnp.asarray(leaf) for leaf in executive)),
        )
        rule_skill = np.asarray(decision.goal.payload.skill_index, dtype=np.int32)
        rule_switch = np.asarray(decision.terminate, dtype=np.bool_)
        trace = {
            "features": features,
            "feature_known": known,
            "latent": latent,
            "exec_skill_index": executive.skill_index,
            "exec_priority": executive.priority,
            "exec_params": executive.params,
            "exec_goal_valid": executive.goal_valid,
            "exec_budget": executive.budget,
            "exec_version": executive.version,
            "current_skill": current.skill_index,
            "target_skill": rule_skill,
            "target_replace": rule_switch,
            "manager_due": np.ones((rows,), dtype=np.bool_),
        }
        decisions = {
            "call": np.repeat(np.arange(calls, dtype=np.int32), lanes),
            "lane": np.tile(np.arange(lanes, dtype=np.int32), calls),
            "tick": np.repeat(steps_array.astype(np.int32), lanes),
            "learner_switch": learner_switch,
            "learner_skill": learner_skill,
            "rule_switch": rule_switch,
            "rule_skill": rule_skill,
            "disagree": disagree_flags(learner_switch, learner_skill, rule_switch, rule_skill),
        }
        return {key: trace[key] for key in TRACE_KEYS}, decisions


def disagree_flags(learner_switch: Any, learner_skill: Any, rule_switch: Any, rule_skill: Any) -> np.ndarray:
    """bool[N]: the learner's switch or its skill in force after the decision differs from the rule's.

    ``learner_skill`` is the skill of the goal the learner left in force. ``rule_skill`` is the rule's
    chosen skill, which equals the current skill wherever the rule does not switch.
    """

    return (np.asarray(learner_switch, np.bool_) != np.asarray(rule_switch, np.bool_)) | (
        np.asarray(learner_skill) != np.asarray(rule_skill)
    )


def disagreement(decisions: Mapping[str, np.ndarray], skill_ids: Sequence[str]) -> dict[str, Any]:
    """Return the disagreement counts and rates of one round, JSON-ready (module docstring)."""

    learner_switch = np.asarray(decisions["learner_switch"], np.bool_)
    rule_switch = np.asarray(decisions["rule_switch"], np.bool_)
    learner_skill = np.asarray(decisions["learner_skill"])
    rule_skill = np.asarray(decisions["rule_skill"])
    rows = int(learner_switch.shape[0])
    learner_only = learner_switch & ~rule_switch
    rule_only = rule_switch & ~learner_switch
    counts = {
        "learner_switch": int(learner_switch.sum()),
        "rule_switch": int(rule_switch.sum()),
        "both_switch": int((learner_switch & rule_switch).sum()),
        "switch_learner_only": int(learner_only.sum()),
        "switch_rule_only": int(rule_only.sum()),
        "skill": int((learner_switch & rule_switch & (learner_skill != rule_skill)).sum()),
        "disagree": int(np.asarray(decisions["disagree"], np.bool_).sum()),
    }

    def by_skill(mask: np.ndarray, skills: np.ndarray) -> dict[str, int]:
        found = np.bincount(skills[mask], minlength=len(skill_ids))
        return {skill_ids[i]: int(found[i]) for i in range(len(skill_ids))}

    return {
        "consultations": int(np.unique(np.asarray(decisions["call"])).shape[0]),
        "lane_decisions": rows,
        **counts,
        "rates": {name: value / rows for name, value in counts.items()},
        "switch_learner_only_to": by_skill(learner_only, learner_skill),
        "switch_rule_only_to": by_skill(rule_only, rule_skill),
    }


def consulted_steps(active_events: Any, *, cadence: int, on_events: bool) -> np.ndarray:
    """Return the session step indices at which ``session.run_session`` consulted its host decider.

    The session's rule (``session.py``, hook step 4): step s is due when s % cadence == 0, or, with
    ``on_events``, when a known event is active on any lane. ``active_events`` is the composition
    record of that name, bool[T, B, E]. It is the step's ``goal_runtime.active_events``, which is
    what the session reads.
    """

    events = np.asarray(active_events, dtype=np.bool_)
    steps = np.arange(events.shape[0])
    due = steps % int(cadence) == 0
    if on_events:
        due = due | events.reshape(events.shape[0], -1).any(axis=1)
    return steps[due]


def merge_rounds(j0: Mapping[str, Any], round_paths: Sequence[str | Path], *, num_skills: int) -> dict[str, np.ndarray]:
    """Return D = D_J0 U D_1 U ... U D_k: the recorded J0 rows first, then each round's consulted rows.

    ``merge_decisions`` keeps the rows of each round whose ``manager_due`` is set, that is the rows
    where the decider was consulted.

    Raises:
        ValueError: If ``j0`` or the merged set fails ``validate_dataset``, or ``j0`` scores another catalogue.
    """

    validate_dataset(j0)
    if int(np.asarray(j0["num_skills"])) != int(num_skills):
        raise ValueError(f"the J0 dataset has {int(np.asarray(j0['num_skills']))} skills; the rounds have {num_skills}")
    rounds = merge_decisions(round_paths, num_skills=num_skills)
    merged = {name: np.concatenate([np.asarray(j0[name]), rounds[name]], axis=0) for name in FIELDS}
    merged["num_skills"] = np.asarray(num_skills, dtype=np.int32)
    validate_dataset(merged)
    return merged


def relabeling_arm() -> tuple[Any, list[tuple[int, RelabelingDecider]]]:
    """Return the js1 arm with its host decider wrapped, and the list that each new decider joins.

    Every factory of the arm stays js1's own. Only ``make_host_decider`` changes: it wraps the decider
    of ``get_arm("js1")`` in a :class:`RelabelingDecider`. The rule is ``ScriptedJev(ctx.skill_ids,
    control=...)``, with the registry's control table, or the default table for a context without a
    registry. The control fields do not affect the labels. ``run_arm_eager`` builds one decider per
    session, a one-step warm-up and then the timed session, so the timed session's decider is the
    last entry, ``(steps, decider)``.
    """

    from experimental.neural_agent.eval.arms import get_arm
    from experimental.neural_agent.jev.scripted import ScriptedJev

    base = get_arm(LEARNED_ARM)
    made: list[tuple[int, RelabelingDecider]] = []

    def make_host_decider(ctx: Any, steps: int) -> RelabelingDecider:
        table_of = getattr(ctx.registry, "control_table", None)
        rule = ScriptedJev(ctx.skill_ids, control=None if table_of is None else table_of())
        decider = RelabelingDecider(base.make_host_decider(ctx, steps), rule, ctx.skill_ids)
        made.append((steps, decider))
        return decider

    return base._replace(make_host_decider=make_host_decider), made


def record_round(
    heads: Path, out_dir: Path, *, arena: str, batch: int, ticks: int, episodes: int, seed: int
) -> dict[str, Any]:
    """Run js1 with ``heads`` wrapped by :class:`RelabelingDecider` and write D_k and the run's table.

    The run is ``run_composition.run_arm_eager`` with ``host_respects_commitment=True``, the same
    session as the evaluation's js1 arm. Writes ``dataset.npz``, ``decisions.npz``,
    ``disagreement.json``, ``record/metrics.csv`` and ``record/REPORT.md`` into ``out_dir``.

    Returns:
        The round's recording summary, JSON-ready.

    Raises:
        RuntimeError: If the consultations recorded, the steps rebuilt from the session records, and
            the session's host decision count disagree.
    """

    from experimental.neural_agent.eval.run_composition import (
        load_arena,
        load_context,
        render_report,
        run_arm_eager,
        write_metrics_csv,
    )
    from experimental.neural_agent.session import SessionConfig

    ctx, task = load_context(batch, ticks, seed=seed, js1_heads=heads)
    built = load_arena(arena, batch, seed, ticks)
    arm, made = relabeling_arm()
    run = run_arm_eager(
        arm,
        ctx,
        built=built.built,
        ticks=ticks,
        episodes=episodes,
        seed=seed,
        task=task,
        catches_known=built.catch_from_terminated,
        host_respects_commitment=True,
    )
    timed_steps, relabeler = made[-1]
    defaults = SessionConfig(steps=1)  # run_arm_eager leaves the cadence and the event rule at their defaults
    steps = consulted_steps(
        run.records["active_events"], cadence=defaults.host_cadence_ticks, on_events=defaults.host_on_events
    )
    decisions_made = run.row["jev_decisions"]
    if timed_steps != episodes * ticks or not (relabeler.calls == len(steps) == decisions_made):
        raise RuntimeError(
            f"consultations disagree: recorded {relabeler.calls}, rebuilt from the records {len(steps)}, "
            f"session host decisions {decisions_made} (timed session of {timed_steps} steps)"
        )
    trace, decisions = relabeler.labelled(steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_trace(out_dir / "dataset.npz", trace)
    checked = merge_decisions([path], num_skills=len(ctx.skill_ids))  # runs validate_dataset on the consulted rows
    np.savez(out_dir / "decisions.npz", **decisions)
    summary = disagreement(decisions, ctx.skill_ids)
    (out_dir / "disagreement.json").write_text(_json(summary), encoding="utf-8")
    write_metrics_csv(out_dir / "record" / "metrics.csv", [run.row])
    config = {
        "arms": f"{LEARNED_ARM} (recording run: RelabelingDecider over the js1 host decider)",
        "arena": arena,
        "runner": "eager",
        "episodes per arm": episodes,
        "batch (lanes)": batch,
        "ticks per run (arena horizon)": ticks,
        "seed": seed,
        "js1 heads": str(heads),
        "host respects commitment": True,
        "labels": "ScriptedJev (J0) on the recorded inputs, after the session (eval/record_dagger.py)",
    }
    report = render_report(
        [run], config=config, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        arena=arena, runner="eager",
    )
    (out_dir / "record" / "REPORT.md").write_text(report, encoding="utf-8")
    counts = np.bincount(trace["target_skill"], minlength=len(ctx.skill_ids))
    return {
        "seed": seed,
        "heads": str(heads),
        "row": run.row,
        "dataset": str(path),
        "rows": int(checked["current_skill"].shape[0]),
        "consultations": relabeler.calls,
        "target_skill_counts": {ctx.skill_ids[i]: int(counts[i]) for i in range(len(ctx.skill_ids))},
        "replace_rate": float(np.mean(trace["target_replace"])),
        "disagreement": summary,
    }


def retrain(data: Path, heldout: Path, out: Path, *, goal_reading: int, updates: int | None) -> dict[str, Any]:
    """Train heads on ``data`` with the G4 command and return the recorded held-out agreement (G1).

    The command is ``train_supervised --data <data> --heldout <heldout> --trunk hashed --goal-reading N
    --text-path off --seed 0 --out <out>``, plus ``--updates`` when given. The trainer's output goes
    to stdout and to ``train.log`` beside ``out``.

    Raises:
        RuntimeError: If the trainer fails or prints no held-out agreement line.
    """

    from experimental.neural_agent.jev.system_one import train_supervised

    argv = [
        "--data", str(data), "--heldout", str(heldout), "--trunk", "hashed",
        "--goal-reading", str(goal_reading), "--text-path", "off", "--seed", str(TRAIN_SEED), "--out", str(out),
    ]
    if updates is not None:
        argv += ["--updates", str(updates)]
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = train_supervised.main(argv)
    text = buffer.getvalue()
    out.with_name("train.log").write_text("train_supervised " + " ".join(argv) + "\n" + text, encoding="utf-8")
    sys.stdout.write(text)
    match = _HELDOUT_LINE.search(text)
    if code != 0 or match is None:
        raise RuntimeError(f"train_supervised exited {code} without a held-out agreement line")
    skill, replace = float(match["skill"]), float(match["replace"])
    return {
        "heads": str(out),
        "rows": int(match["rows"]),
        "skill_agreement": skill,
        "replace_agreement": replace,
        "g1_pass": skill >= HELDOUT_AGREEMENT_BAR and replace >= HELDOUT_AGREEMENT_BAR,
    }


def _cell(text: str) -> Any:
    """A metrics.csv cell as a number, or None for ``n/a``."""

    if text == "n/a":
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def evaluate(
    heads: Path, out_dir: Path, *, arena: str, batch: int, ticks: int, episodes: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Run ``run_composition --arms j0,js1 --runner eager --host-respects-commitment`` and return its rows by arm.

    Raises:
        RuntimeError: If the composition run does not exit 0.
    """

    from experimental.neural_agent.eval.run_composition import main as compose

    argv = [
        "--arms", f"{RULE_ARM},{LEARNED_ARM}", "--runner", "eager", "--arena", arena, "--ticks", str(ticks),
        "--batch", str(batch), "--episodes", str(episodes), "--seed", str(seed), "--js1-heads", str(heads),
        "--host-respects-commitment", "--out", str(out_dir),
    ]
    code = compose(argv)
    if code != 0:
        raise RuntimeError(f"run_composition exited {code}")
    with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return {row["arm"]: {key: _cell(value) for key, value in row.items()} for row in csv.DictReader(handle)}


def bar_check(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The bar on one evaluation run: rate(js1) >= Wilson lower(j0), and js1 made 0 misses."""

    rate = rows[LEARNED_ARM]["success_rate"]
    lower = rows[RULE_ARM]["success_wilson_lower"]
    misses = rows[LEARNED_ARM]["jev_misses"]
    rate_ok = rate is not None and lower is not None and rate >= lower
    return {
        "js1_success_rate": rate,
        "j0_wilson_lower": lower,
        "rate_ok": rate_ok,
        "js1_misses": misses,
        "met": rate_ok and misses == 0,
    }


def _number(value: Any) -> str:
    """A table cell: ``n/a`` for None, three decimals for a float."""

    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _table_line(label: str, row: Mapping[str, Any]) -> str:
    """One arm's line of the console table: successes/completed, rate, Wilson lower, switches, JEV columns."""

    return (
        f"  {label:<16} {row['successes']}/{row['episodes_completed']} = {_number(row['success_rate'])} "
        f"(Wilson lower {_number(row['success_wilson_lower'])}), switches/lane-run "
        f"{_number(row['goal_switches_per_lane_run'])}, decisions {_number(row['jev_decisions'])}, misses "
        f"{_number(row['jev_misses'])}, latency p50/p95 ms {_number(row['jev_latency_ms_p50'])}/"
        f"{_number(row['jev_latency_ms_p95'])}"
    )


def _json(value: Any) -> str:
    """JSON text; numpy scalars become Python numbers."""

    return json.dumps(value, indent=2, default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""

    from experimental.neural_agent.eval.run_composition import ARENA_CHOICES

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.eval.record_dagger",
        description="DAgger rounds for js1: record its own decision states, label them with J0, retrain, evaluate.",
    )
    parser.add_argument("--heads", type=Path, required=True, help="js1 heads .npz that records round 1")
    parser.add_argument("--rounds", type=int, default=3, help="most rounds; stops at the first that meets the rate bar (default 3)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (round_<k>/ and summary.json)")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="evaluation seed (default 0, the G4 runs' seed); round k records on seed + k",
    )
    parser.add_argument("--arena", choices=ARENA_CHOICES, default="combat_open_flat")
    parser.add_argument("--ticks", type=int, default=512, help="ticks per run, the arena horizon (default 512)")
    parser.add_argument("--batch", type=int, default=8, help="lanes (default 8)")
    parser.add_argument("--episodes", type=int, default=8, help="horizons per lane in each run (default 8)")
    parser.add_argument("--j0-data", type=Path, default=DEFAULT_J0_DATA, help="recorded J0 training rows (trainer format)")
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT, help="recorded held-out rows, gate G1")
    parser.add_argument(
        "--goal-reading", type=int, default=DEFAULT_GOAL_READING,
        help=f"train_supervised --goal-reading (default {DEFAULT_GOAL_READING}, the G4 heads' value; smaller for smoke tests)",
    )
    parser.add_argument(
        "--updates", type=int, default=None,
        help="train_supervised --updates (default: the trainer's own, as for the G4 heads; smaller for smoke tests)",
    )
    args = parser.parse_args(argv)
    for name in ("rounds", "ticks", "batch", "episodes", "goal_reading"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run up to ``--rounds`` DAgger rounds and write the per-round files and ``summary.json``."""

    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.j0_data) as archive:
        j0 = {name: archive[name] for name in archive.files}
    num_skills = int(np.asarray(j0["num_skills"]))
    summary: dict[str, Any] = {
        "config": {
            "heads": str(args.heads), "rounds": args.rounds, "eval_seed": args.seed, "arena": args.arena,
            "ticks": args.ticks, "batch": args.batch, "episodes": args.episodes, "j0_data": str(args.j0_data),
            "heldout": str(args.heldout), "goal_reading": args.goal_reading, "updates": args.updates,
            "train_seed": TRAIN_SEED, "host_respects_commitment": True,
        },
        "j0_rows": int(j0["current_skill"].shape[0]),
        "rounds": [],
        "bar_met_round": None,
        "rate_ok_round": None,
    }
    heads: Path = args.heads
    round_paths: list[Path] = []
    run = {"arena": args.arena, "batch": args.batch, "ticks": args.ticks, "episodes": args.episodes}
    for k in range(1, args.rounds + 1):
        round_dir = out / f"round_{k}"
        print(f"round {k}: recording js1 ({heads}) on seed {args.seed + k}", flush=True)
        recorded = record_round(heads, round_dir, seed=args.seed + k, **run)
        round_paths.append(round_dir / "dataset.npz")
        merged = merge_rounds(j0, round_paths, num_skills=num_skills)
        np.savez(round_dir / "merged.npz", **merged)
        print(f"round {k}: {recorded['rows']} rows recorded, {int(merged['current_skill'].shape[0])} merged; retraining", flush=True)
        trained = retrain(
            round_dir / "merged.npz", args.heldout, round_dir / "train" / "heads.npz",
            goal_reading=args.goal_reading, updates=args.updates,
        )
        print(f"round {k}: evaluating {trained['heads']} on seed {args.seed}", flush=True)
        rows = evaluate(Path(trained["heads"]), round_dir / "eval", seed=args.seed, **run)
        check = bar_check(rows)
        summary["rounds"].append(
            {
                "round": k,
                "record": recorded,
                "merged": {
                    "path": str(round_dir / "merged.npz"),
                    "rows": int(merged["current_skill"].shape[0]),
                    "j0_rows": summary["j0_rows"],
                    "dagger_rows": int(merged["current_skill"].shape[0]) - summary["j0_rows"],
                },
                "trained": trained,
                "eval": {"seed": args.seed, "dir": str(round_dir / "eval"), "rows": rows},
                "bar": check,
            }
        )
        if check["met"] and summary["bar_met_round"] is None:
            summary["bar_met_round"] = k
        if check["rate_ok"] and summary["rate_ok_round"] is None:
            summary["rate_ok_round"] = k
        (out / "summary.json").write_text(_json(summary), encoding="utf-8")
        found = recorded["disagreement"]
        rates = found["rates"]
        lines = [
            f"round {k} summary",
            f"  disagreement in the recording run ({found['lane_decisions']} lane-decisions, "
            f"{found['consultations']} consultations): switch learner-only {rates['switch_learner_only']:.4f}, "
            f"rule-only {rates['switch_rule_only']:.4f}, skill {rates['skill']:.4f}, any {rates['disagree']:.4f}",
            f"  rows: round {recorded['rows']}, merged {summary['rounds'][-1]['merged']['rows']} "
            f"(J0 {summary['j0_rows']})",
            f"  held-out recorded agreement: skill {trained['skill_agreement']:.4f}, replace "
            f"{trained['replace_agreement']:.4f} (G1 {'pass' if trained['g1_pass'] else 'FAIL'})",
            f"  recording run, seed {recorded['seed']}:",
            _table_line(LEARNED_ARM + " (recording)", recorded["row"]),
            f"  evaluation run, seed {args.seed}:",
            *(_table_line(arm, rows[arm]) for arm in (RULE_ARM, LEARNED_ARM)),
            f"  bar: js1 rate {_number(check['js1_success_rate'])} vs j0 Wilson lower "
            f"{_number(check['j0_wilson_lower'])}: {'not below' if check['rate_ok'] else 'below'}; js1 misses "
            f"{check['js1_misses']}; met {check['met']}",
        ]
        print("\n".join(lines), flush=True)
        heads = Path(trained["heads"])
        if check["rate_ok"]:
            break
    print(f"wrote {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HELDOUT",
    "DEFAULT_J0_DATA",
    "HELDOUT_AGREEMENT_BAR",
    "RelabelingDecider",
    "bar_check",
    "consulted_steps",
    "disagree_flags",
    "disagreement",
    "evaluate",
    "main",
    "merge_rounds",
    "parse_args",
    "record_round",
    "relabeling_arm",
    "retrain",
]
