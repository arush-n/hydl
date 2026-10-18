"""B3's joint check: the middle layer's program executor and this layer's combiner meet in S_t.

``BUILD.md`` 7b, B3. ``run_composition`` runs twice with the same arguments: as shipped, and with every
host-mode arm's decider (the ``jprog`` arm's ``ProgramExecutor``) wrapped in
``scratchpad.CombiningDecider``, patched in at run time (the eval driver's file is untouched, as
``parity.run`` does). With the wrapper the executor's requests reach the combiner through S_t's requests
section and the combiner's statuses reach the executor through the skills section, where its
correspondence rules read them.

The gate: every non-timing column of ``metrics.csv`` is identical (in the composition run the compiled
dispatcher acts; the combiner adds the status stream), and the statuses the combiner wrote are reported by
kind and rejection code. A rejection the executor reacts to would change its decisions, so identical
metrics also say the stream carried nothing it had to correct.

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym NA_PROGRAM=J0 JAX_PLATFORMS=cpu \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.joint_check --out <dir> \\
        --arms flat,j0,shuffled,jprog --runner eager --ticks 64 --batch 8 --episodes 1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from experimental.neural_agent.skill_agent.library import SkillLibrary
from experimental.neural_agent.skill_agent.parity import BRIEF_COLUMNS, TIMING_COLUMNS, compare, read_metrics
from experimental.neural_agent.skill_agent.scratchpad import CombiningDecider


def run(argv: Sequence[str], *, joint: bool, layout: str = "gen5") -> tuple[int, list[CombiningDecider]]:
    """Run ``eval.run_composition.main(argv)``; with ``joint``, every host decider is wrapped. Returns the
    exit code and the wrappers made (one per session, the warm-up session's included)."""

    from experimental.neural_agent.eval import run_composition

    if not joint:
        return run_composition.main(list(argv)), []
    made: list[CombiningDecider] = []
    original = run_composition.get_arm

    def wrap(arm: Any) -> Any:
        if arm.make_host_decider is None:
            return arm
        base = arm.make_host_decider

        def make(ctx: Any, steps: int) -> CombiningDecider:
            combiner = SkillLibrary.from_catalogue(proof_of_concept=True).combiner(layout)
            decider = CombiningDecider(base(ctx, steps), combiner)
            made.append(decider)
            return decider

        return arm._replace(make_host_decider=make)

    run_composition.get_arm = lambda name: wrap(original(name))
    try:
        return run_composition.main(list(argv)), made
    finally:
        run_composition.get_arm = original


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Composition runs with and without the combiner on S_t.")
    parser.add_argument("--out", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    outputs: dict[str, Any] = {}
    wrappers: list[CombiningDecider] = []
    for name, joint in (("baseline", False), ("joint", True)):
        directory = args.out / name
        code, made = run([*rest, "--out", str(directory)], joint=joint)
        if code != 0:
            sys.stderr.write(f"the {name} run exited with {code}\n")
            return code
        wrappers.extend(made)
        outputs[name] = read_metrics(directory / "metrics.csv")
    differences = compare(outputs["baseline"], outputs["joint"])
    counts: Counter[str] = Counter()
    for wrapper in wrappers:
        counts.update(wrapper.status_counts)
    lines = ["arm | " + " | ".join(BRIEF_COLUMNS) + " | identical"]
    for arm, row in outputs["baseline"].items():
        other = outputs["joint"][arm]
        same = all(row.get(column) == other.get(column) for column in row if column not in TIMING_COLUMNS)
        lines.append(f"{arm} | " + " | ".join(f"{row.get(c)} / {other.get(c)}" for c in BRIEF_COLUMNS)
                     + f" | {'yes' if same else 'NO'}")
    report = {"differences": differences, "status_counts": dict(sorted(counts.items())),
              "sessions_wrapped": len(wrappers)}
    (args.out / "joint_report.json").write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"non-timing differences: {len(differences)}\n" + "".join(f"  {d}\n" for d in differences))
    sys.stdout.write(f"statuses written by the combiner: {report['status_counts']}\n")
    return 0 if not differences else 1


__all__ = ["main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
