"""The working spine (2026-10-19, the owner's choice A): a trained skill acting end to end through the three layers.

``eval.run_composition`` runs twice with the same arguments: as shipped, where the stand-in PPO pursuit acts, and with
the neural pursuit (``neural_pursuit.py``: the A2 C' network composing walk_to and look_at) in the pursuit slot of
every arm's dispatcher. Both are patched in at run time, as ``joint_check.py`` does, so the eval driver's files are
untouched. In both runs each host-mode arm's decider (the ``jprog`` arm's program executor) is wrapped in
``scratchpad.CombiningDecider``, so the loop is whole: the executive's goals in S_t (scripted; no live LLM before the
lower level is finished) -> the program executor's ``pursuit`` requests in S_t -> the combiner, which checks and
grants them and streams their statuses back to the executor -> the dispatcher's skill acting in the gen-5 world.

Not yet: the combiner's grant does not gate the action. In the composition run the compiled dispatcher acts on the
manager's goal and the combiner adds the status stream (B3, ``scratchpad.py``); the in-graph active-request table
(``combiner.ActiveRequestTable``) is where a grant would gate it.

The report (``<out>/spine_report.json`` and stdout): each arm's ``parity.BRIEF_COLUMNS`` as stand-in / neural, and the
statuses the combiner wrote, by kind, in each run.

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym NA_PROGRAM=J0 JAX_PLATFORMS=cpu \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.spine --out <dir> \\
        --arms jprog --runner eager --ticks 512 --batch 8 --episodes 2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary
from experimental.neural_agent.skill_agent.runtime.neural_pursuit import PURSUIT_SKILL_ID, load_neural_pursuit
from experimental.neural_agent.skill_agent.runtime.parity import BRIEF_COLUMNS, read_metrics
from experimental.neural_agent.skill_agent.runtime.scratchpad import CombiningDecider


def run(argv: Sequence[str], *, neural: bool, layout: str = "gen5") -> tuple[int, list[CombiningDecider]]:
    """Run ``eval.run_composition.main(argv)`` with every host decider behind the combiner on S_t; with ``neural``, the
    neural pursuit in every arm's pursuit slot. Returns the exit code and the combining deciders made."""

    from experimental.neural_agent.eval import run_composition
    from experimental.neural_agent.skill_library.dispatcher import SkillDispatcher
    from experimental.neural_agent.skill_library.library import SkillRegistry

    made: list[CombiningDecider] = []
    skill = load_neural_pursuit() if neural else None
    original = run_composition.get_arm

    def wrap(arm: Any) -> Any:
        changes: dict[str, Any] = {}
        if arm.make_host_decider is not None:
            base = arm.make_host_decider

            def make_decider(ctx: Any, steps: int) -> CombiningDecider:
                decider = CombiningDecider(base(ctx, steps),
                                           SkillLibrary.from_catalogue(proof_of_concept=True).combiner(layout))
                made.append(decider)
                return decider

            changes["make_host_decider"] = make_decider
        if skill is not None:
            def make_actor(ctx: Any) -> Any:
                skills = tuple(skill if entry.contract.skill_id == PURSUIT_SKILL_ID else entry
                               for entry in ctx.registry.skills)
                return SkillDispatcher(SkillRegistry(skills))

            changes["make_actor"] = make_actor
        return arm._replace(**changes)

    run_composition.get_arm = lambda name: wrap(original(name))
    try:
        return run_composition.main(list(argv)), made
    finally:
        run_composition.get_arm = original


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The working spine: the stand-in pursuit against the neural pursuit.")
    parser.add_argument("--out", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    metrics: dict[str, Any] = {}
    statuses: dict[str, dict[str, int]] = {}
    for name, neural in (("stand_in", False), ("neural", True)):
        directory = args.out / name
        code, made = run([*rest, "--out", str(directory)], neural=neural)
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
    lines = ["arm | " + " | ".join(BRIEF_COLUMNS) + " (stand-in / neural)"]
    for arm, row in metrics["stand_in"].items():
        other = metrics["neural"].get(arm, {})
        lines.append(f"{arm} | " + " | ".join(f"{row.get(column)} / {other.get(column)}" for column in BRIEF_COLUMNS))
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"statuses written by the combiner: {statuses}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
