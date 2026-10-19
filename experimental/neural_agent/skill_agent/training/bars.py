"""Score a distillation report on the A2 bars, on held success (owner, 2026-10-18), with the first hit beside it.

The bars (docs/TRAINING_PLAN.md 2c; README 2.16), per mode (walk_only, look_only, both), for the student:
- T5: the teacher's rate less the student's is at most ``T5_POINTS`` (0.10);
- held out: the student's held-out rate is at least ``HELD_OUT_RATIO`` (0.90) of its seen rate;
- E4 (walk_to's arguments): pace within 5 % on at least ``E4_SHARE`` (0.95) of the steady lanes, and the radius kept
  on at least that share of the successes.
Composition is reported, not barred: the "both" rate against walk x look in the same lanes.

A report written before held success existed (schema v1) has nothing to score and is refused.
Usage: python -m experimental.neural_agent.skill_agent.training.bars REPORT.json [--skill walk_to|look_at]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

T5_POINTS = 0.10
HELD_OUT_RATIO = 0.90
E4_SHARE = 0.95
MODES = ("walk_only", "look_only", "both")
#: The modes and checks that belong to each skill; "both" belongs to both.
SKILL_MODES = {"walk_to": ("walk_only", "both"), "look_at": ("look_only", "both")}
#: The run options printed with the bars (the size and the opt-in changes).
CONFIG_KEYS = ("iterations", "lanes", "steps", "seed", "walk_stop_margin", "look_direction", "look_frame")


def score(report: dict[str, Any]) -> dict[str, Any]:
    """The bars of one report: per mode T5 and held out (held success, the first hit beside), E4 and composition."""

    if report.get("schema") != "skill_agent_distill_v2":
        raise ValueError(f"no held success to score: schema {report.get('schema')!r}, needs skill_agent_distill_v2")
    evaluation = report["evaluation"]
    seen, teacher = evaluation["seen"]["student"], evaluation["seen"]["teacher"]
    held_out = evaluation["held_out"]["student"]
    modes = {}
    for mode in MODES:
        row = {}
        for rule, key in (("held", "held"), ("first", "success")):
            gap = teacher[mode][key] - seen[mode][key]
            ratio = held_out[mode][key] / seen[mode][key] if seen[mode][key] else float("nan")
            row[rule] = {"t5_gap": gap, "t5_pass": gap <= T5_POINTS, "held_out_ratio": ratio,
                         "held_out_pass": ratio >= HELD_OUT_RATIO}
        modes[mode] = row
    pace = seen["walk_pace"]["within_5_percent"]
    radius = seen["walk_radius"]["within_radius"]
    return {
        "modes": modes,
        "e4": {"pace": pace, "pace_pass": pace is not None and pace >= E4_SHARE,
               "radius": radius, "radius_pass": radius is not None and radius >= E4_SHARE},
        "composition": {"both": seen["both"]["held"],
                        "walk_times_look": seen["both"]["walk_held"] * seen["both"]["look_held"]},
    }


def _verdict(passed: bool) -> str:
    return "pass" if passed else "FAIL"


def format_bars(report: dict[str, Any], skill: str | None = None) -> str:
    """The report's rates and bars as text; ``skill`` keeps only that skill's modes (and E4 for walk_to)."""

    bars = score(report)
    modes = SKILL_MODES[skill] if skill else MODES
    evaluation = report["evaluation"]
    teachers = {name: entry.get("version") for name, entry in report.get("teachers", {}).items()}
    options = {key: report["config"].get(key) for key in CONFIG_KEYS}
    lines = [f"config {options} | teachers {teachers}"]
    for region in ("seen", "held_out"):
        for who in ("student", "teacher"):
            e = evaluation[region][who]
            rates = " ".join(f"{mode} {e[mode]['held']:.3f} [{e[mode]['success']:.3f}]" for mode in modes)
            lines.append(f"{region:8s} {who:7s} held [first]: {rates} | lanes "
                         f"{'/'.join(str(e[mode]['lanes']) for mode in modes)}")
    lines.append("bars, student on held success (first hit in brackets):")
    for mode in modes:
        held, first = bars["modes"][mode]["held"], bars["modes"][mode]["first"]
        lines.append(f"  {mode:9s} T5 {held['t5_gap']:+.3f} {_verdict(held['t5_pass'])} [{first['t5_gap']:+.3f}]; "
                     f"held out / seen {held['held_out_ratio']:.3f} {_verdict(held['held_out_pass'])} "
                     f"[{first['held_out_ratio']:.3f}]")
    if "both" in modes:
        composition = bars["composition"]
        lines.append(f"  composition: both {composition['both']:.3f} against walk x look "
                     f"{composition['walk_times_look']:.3f}")
    if skill in (None, "walk_to"):
        e4 = bars["e4"]
        lines.append(f"  E4 pace {e4['pace']} {_verdict(e4['pace_pass'])}, radius {e4['radius']} "
                     f"{_verdict(e4['radius_pass'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="a distill report (report.json), or the run folder holding it")
    parser.add_argument("--skill", choices=tuple(SKILL_MODES), help="show only this skill's modes")
    args = parser.parse_args(argv)
    path = args.report / "report.json" if args.report.is_dir() else args.report
    try:
        text = format_bars(json.loads(path.read_text()), args.skill)
    except ValueError as error:
        parser.error(str(error))
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["E4_SHARE", "HELD_OUT_RATIO", "MODES", "SKILL_MODES", "T5_POINTS", "format_bars", "main", "score"]
