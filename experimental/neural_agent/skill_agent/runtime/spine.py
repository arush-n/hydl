"""The working spine (2026-10-19, the owner's choice A): a trained skill acting end to end through the three layers.

``eval.run_composition`` runs with the same arguments up to three ways (``--runs``): ``stand_in``, as shipped, where
the stand-in PPO pursuit acts; ``neural``, with the neural pursuit (``neural_pursuit.py``: the A2 C' network composing
walk_to and look_at) in the registry's pursuit slot, so every arm's dispatcher runs it; and ``referee``, where the grant
gates the action (2026-10-19, the owner: the grant controls the body, no stand-ins): every arm's actor is the skill
layer's ``GrantedActor`` (``granted_actor.skill_layer_actor``: the neural pursuit and the scripted bodies), whose state
is the combiner's grants, written each tick through the session's ``on_tick`` seam; the host combiner holds
``continuous`` bodies, so a grant lasts until the request is withdrawn or replaced. All of it is patched in at run
time, as ``joint_check.py`` does, so the eval driver's files are untouched. In every run each host-mode arm's decider
(the ``jprog`` arm's program executor) is wrapped in ``scratchpad.CombiningDecider``, so the loop is whole: the
executive's goals in S_t (scripted; no live LLM before the lower level is finished) -> the program executor's requests
in S_t -> the combiner, which checks and grants them and streams their statuses back to the executor -> the skill
acting. In the stand-in and neural runs the compiled dispatcher acts on the manager's goal and the combiner only adds
the status stream (B3, ``scratchpad.py``); the referee refuses an arm without a host decider, which has no combiner.

The neural and referee runs load no stand-in checkpoint, so they also run on a gen-7 render (``gen7.py``). The
stand-in run is gen 5 only: the stand-in's pinned action-scope digest (F002D864BCA0) is gen 5's, and it refuses a
gen-7 render, whose pursuit scope digests differently.

The report (``<out>/spine_report.json`` and stdout): each arm's ``parity.BRIEF_COLUMNS``, one value per run in
``--runs`` order, and the statuses the combiner wrote, by kind, in each run.

CLI (gen 5; on gen 7 run it under ``gen7.py`` with ``--runs neural referee``)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym NA_PROGRAM=J0 JAX_PLATFORMS=cpu \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.spine --out <dir> \\
        [--runs stand_in neural referee] --arms jprog --runner eager --ticks 512 --batch 8 --episodes 2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from experimental.neural_agent.skill_agent.runtime.granted_actor import referee_library, skill_layer_actor
from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary
from experimental.neural_agent.skill_agent.runtime.neural_pursuit import load_neural_pursuit
from experimental.neural_agent.skill_agent.runtime.parity import BRIEF_COLUMNS, read_metrics
from experimental.neural_agent.skill_agent.runtime.scratchpad import CombiningDecider

RUNS: tuple[str, ...] = ("stand_in", "neural", "referee")
"""The runs ``--runs`` may name; the default is the first two."""


def run(argv: Sequence[str], *, neural: bool, referee: bool = False,
        layout: str = "gen5") -> tuple[int, list[CombiningDecider]]:
    """Run ``eval.run_composition.main(argv)`` with every host decider behind the combiner on S_t; with ``neural`` or
    ``referee``, the neural pursuit in the registry's pursuit slot (no stand-in checkpoint loads); with ``referee``,
    the skill layer's ``GrantedActor`` as every arm's actor, acting on the combiner's grants. Returns the exit code
    and the combining deciders made."""

    from experimental.neural_agent.eval import run_composition
    from experimental.neural_agent.skill_library import library
    from experimental.neural_agent.skill_library.evader import ScriptedEvaderSkill
    from experimental.neural_agent.skill_library.idle import IdleSkill

    made: list[CombiningDecider] = []
    skill = load_neural_pursuit() if neural or referee else None
    actor = skill_layer_actor(skill) if referee else None
    original_arm, original_registry = run_composition.get_arm, library.build_default_registry

    def wrap(arm: Any) -> Any:
        changes: dict[str, Any] = {}
        if arm.make_host_decider is not None:
            base = arm.make_host_decider

            def make_decider(ctx: Any, steps: int) -> CombiningDecider:
                if actor is None:
                    decider = CombiningDecider(base(ctx, steps),
                                               SkillLibrary.from_catalogue(proof_of_concept=True).combiner(layout))
                else:
                    decider = CombiningDecider(base(ctx, steps), referee_library(actor.verbs).combiner(layout),
                                               actor_verbs=actor.verbs)
                made.append(decider)
                return decider

            changes["make_host_decider"] = make_decider
        if actor is not None:
            if arm.make_host_decider is None:
                raise ValueError(f"the referee acts on the combiner's grants; the {arm.name} arm has no host decider")
            changes["make_actor"] = lambda ctx: actor
        return arm._replace(**changes)

    run_composition.get_arm = lambda name: wrap(original_arm(name))
    if skill is not None:
        library.build_default_registry = lambda batch: library.SkillRegistry(
            (IdleSkill(), skill, ScriptedEvaderSkill()))
    try:
        return run_composition.main(list(argv)), made
    finally:
        run_composition.get_arm, library.build_default_registry = original_arm, original_registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The working spine: the stand-in, the neural pursuit, the referee.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS[:2]),
                        help="the runs in order (default: stand_in neural); referee: the combiner's grants drive the body")
    args, rest = parser.parse_known_args(argv)
    metrics: dict[str, Any] = {}
    statuses: dict[str, dict[str, int]] = {}
    for name in args.runs:
        directory = args.out / name
        code, made = run([*rest, "--out", str(directory)], neural=name == "neural", referee=name == "referee")
        if code != 0:
            sys.stderr.write(f"the {name} run exited with {code}\n")
            return code
        metrics[name] = read_metrics(directory / "metrics.csv")
        counts: Counter[str] = Counter()
        for decider in made:
            counts.update(decider.status_counts)
        statuses[name] = dict(sorted(counts.items()))
    skill = load_neural_pursuit()
    report = {"metrics": metrics, "statuses": statuses, "neural_pursuit": {
        "version": skill.contract.version, "checkpoint_sha256": skill.contract.checkpoint_sha256}}
    (args.out / "spine_report.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["arm | " + " | ".join(BRIEF_COLUMNS) + f" ({' / '.join(args.runs)})"]
    for arm in metrics[args.runs[0]]:
        lines.append(f"{arm} | " + " | ".join(" / ".join(str(metrics[name].get(arm, {}).get(column))
                                                         for name in args.runs) for column in BRIEF_COLUMNS))
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"statuses written by the combiner: {statuses}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RUNS", "main", "run"]
