"""Score a distillation report on the A2 bars, on held success (owner, 2026-10-18), with the first hit beside it.

The bars (docs/TRAINING_PLAN.md 2c; README 2.16), per mode (walk_only, look_only, both), for the student:
- T5: the teacher's rate less the student's is at most ``T5_POINTS`` (0.10);
- held out: the student's held-out rate is at least ``HELD_OUT_RATIO`` (0.90) of its seen rate;
- E4 (walk_to's arguments): pace within 5 % on at least ``E4_SHARE`` (0.95) of the steady lanes, and the radius kept
  on at least that share of the successes.
Composition is reported, not barred: the "both" rate against walk x look in the same lanes.

A report written before held success existed (schema v1) has nothing to score and is refused.

A skill module's report (schema ``skill_module_v1``, ``training/skill_dagger.run_main``) is scored per lane kind: T5
(the teacher's held rate less the module's) and held out against seen, for the trained skill; the held rate of each
skill beside it under the module and under the teacher, so interference shows; and the skill's own metrics (walk_to's
pace and radius, against E4).
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


def _share_bars(teacher: float | None, module: float | None, held_out: float | None) -> dict[str, Any]:
    """A holding skill's bars on its share: the teacher's share less the module's, at most ``T5_POINTS``, and the held-out
    share over the seen one, at least ``HELD_OUT_RATIO``. A lane kind with no counted tick has no share, so it fails."""

    if teacher is None or module is None or held_out is None:
        return {"t5_gap": None, "t5_pass": False, "held_out_ratio": None, "held_out_pass": False}
    gap = teacher - module
    ratio = held_out / module if module else float("nan")
    return {"t5_gap": gap, "t5_pass": gap <= T5_POINTS, "held_out_ratio": ratio,
            "held_out_pass": ratio >= HELD_OUT_RATIO}


def _figure(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "none"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def score_module(report: dict[str, Any]) -> dict[str, Any]:
    """The bars of one skill module's report (``skill_module_v1``): per lane kind, T5 and held out for the trained skill
    (held success, the first hit beside; for a holding skill, its share in view), the skills beside it under the module and
    the teacher, and its metrics."""

    if report.get("schema") != "skill_module_v1":
        raise ValueError(f"not a skill module report: schema {report.get('schema')!r}")
    skill = report["skill"]
    evaluation = report["evaluation"]
    seen, teacher, held_out = (evaluation["seen"]["module"], evaluation["seen"]["teacher"],
                               evaluation["held_out"]["module"])
    kinds = {}
    for kind, row in seen["kinds"].items():
        own, own_teacher, own_held_out = (row["skills"][skill], teacher["kinds"][kind]["skills"][skill],
                                          held_out["kinds"][kind]["skills"][skill])
        if "share" in own:  # a holding skill (track): barred on its share in view within reach, not on held success
            bars = {"share": _share_bars(own_teacher["share"], own["share"], own_held_out["share"])}
        else:
            bars = {}
            for rule, key in (("held", "held"), ("first", "success")):
                gap = own_teacher[key] - own[key]
                ratio = own_held_out[key] / own[key] if own[key] else float("nan")
                bars[rule] = {"t5_gap": gap, "t5_pass": gap <= T5_POINTS, "held_out_ratio": ratio,
                              "held_out_pass": ratio >= HELD_OUT_RATIO}
        beside = {name: {"module": rates["held"], "teacher": teacher["kinds"][kind]["skills"][name]["held"]}
                  for name, rates in row["skills"].items() if name != skill}
        kinds[kind] = {"bars": bars, "beside": beside, "all_held": row["all_held"]}
    result = {"skill": skill, "kinds": kinds}
    metrics = seen.get("metrics")
    if metrics and "pace" in metrics:
        pace, radius = metrics["pace"]["within_5_percent"], metrics["radius"]["within_radius"]
        result["e4"] = {"pace": pace, "pace_pass": pace is not None and pace >= E4_SHARE,
                        "radius": radius, "radius_pass": radius is not None and radius >= E4_SHARE}
    return result


def format_module(report: dict[str, Any]) -> str:
    """A skill module's report and its bars as text."""

    scored = score_module(report)
    lines = [f"{scored['skill']} module | trunk {report['trunk_sha256'][:12]} | module "
             f"{report['module_sha256'][:12]} | beside {report['beside']} | config {report['config']}"]
    for kind, row in scored["kinds"].items():
        beside = "; ".join(f"{name} held {rates['module']:.3f} beside the module, {rates['teacher']:.3f} beside "
                           f"the teacher" for name, rates in row["beside"].items())
        if "share" in row["bars"]:
            share = row["bars"]["share"]
            lines.append(f"  {kind:14s} share T5 {_figure(share['t5_gap'], True)} {_verdict(share['t5_pass'])}; "
                         f"held out / seen {_figure(share['held_out_ratio'])} {_verdict(share['held_out_pass'])}"
                         + (f"; {beside}" if beside else ""))
            continue
        held, first = row["bars"]["held"], row["bars"]["first"]
        lines.append(f"  {kind:14s} T5 {held['t5_gap']:+.3f} {_verdict(held['t5_pass'])} [{first['t5_gap']:+.3f}]; "
                     f"held out / seen {held['held_out_ratio']:.3f} {_verdict(held['held_out_pass'])} "
                     f"[{first['held_out_ratio']:.3f}]" + (f"; {beside}" if beside else ""))
    if "e4" in scored:
        e4 = scored["e4"]
        lines.append(f"  E4 pace {e4['pace']} {_verdict(e4['pace_pass'])}, radius {e4['radius']} "
                     f"{_verdict(e4['radius_pass'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="a report.json, or the run folder holding it")
    parser.add_argument("--skill", choices=tuple(SKILL_MODES), help="show only this skill's modes (A2 reports)")
    args = parser.parse_args(argv)
    path = args.report / "report.json" if args.report.is_dir() else args.report
    report = json.loads(path.read_text())
    try:
        if report.get("schema") == "skill_module_v1":
            if args.skill and args.skill != report["skill"]:
                parser.error(f"{path} is {report['skill']}'s module, not {args.skill}'s")
            text = format_module(report)
        else:
            text = format_bars(report, args.skill)
    except ValueError as error:
        parser.error(str(error))
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "E4_SHARE",
    "HELD_OUT_RATIO",
    "MODES",
    "SKILL_MODES",
    "T5_POINTS",
    "format_bars",
    "format_module",
    "main",
    "score",
    "score_module",
]
