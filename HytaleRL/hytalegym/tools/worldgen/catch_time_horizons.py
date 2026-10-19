#!/usr/bin/env python3
"""Catch-time horizon rungs from measured time-to-success, for one server line.

Tracked stand-in for the private `catch_time.py` that the skill docs name
(`experimental/neural_agent/skill_agent/docs/BUILD.md:147, 221`). This tool does not
read or copy that file. Its rule is the one stated below.

Input: --input FILE, a JSON array or JSON Lines, one object per successful episode:
  {"seconds": <finite number > 0>}     time from reset to success, in seconds
`--example` prints an illustrative five-line input. Its values are NOT measured.
No measured times exist in the repository yet. Pursuit runs write aggregate counts
(`first_episode_counts`, arena/training/runs/pursuit_run.py:1813-1839) and per-tick
replays (pursuit_run.py:3007), not a per-episode time list (read from the code, not run).

Tick base (seconds per environment step), imported from the profiles:
  0.7.2 -> 1/30 s   hytalegym/sim/gen7/profile.py:71 (one step duration for gen 7).
  0.5.7 -> 0.045 s  hytalegym/sim/common/profile.py:163, the gen-5 LOADED tick. Gen 5
                    also steps 1/30 s on profile 0 (profile.py:155-164), so 0.045 is the
                    slower of its two steps, not an average.

Rule (an inferred choice; the repository does not state one):
  1. ticks_i = max(1, ceil(seconds_i / tick - 1e-9)). The 1e-9 absorbs float error at exact multiples.
  2. For each quantile q in --quantiles (default 0.5,0.9,1.0), T_q is the nearest-rank quantile
     of the sorted ticks, index ceil(q*n - 1e-9) - 1, with no interpolation. Each rung is therefore
     a measured time. q = 1.0 admits every measured success.
  3. pow2_ticks = the smallest power of two >= T_q.
  4. ticks = min(pow2_ticks, effective ceiling). Each rung reports `capped`, and the result reports
     `cap_applied`. Nothing is capped silently.

Ceilings (confirmed by reading):
  RUNNER_CEILING = 1000  arena/training/contracts/pursuit.py:60 (PURSUIT_MAXIMUM_TICKS), enforced at
                         arena/training/runs/pursuit_run.py:647-649.
  CONSOLE_CEILING = 512  console/core/execution/jobs.py:585 (a literal), refused at :612-615.
  The effective ceiling is min of the two, so 512 today.
The present pursuer `close` rung (768) cannot come out of this rule: 768 is not a power of two. The
present 1024 rungs cannot come out of it either: 1024 is above both ceilings.
No floor is encoded. The curriculum's 128-tick floor (arena/curriculum/pursuit.py:43-48) is a stated policy.

Usage (repo venv, from any directory):
  /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/catch_time_horizons.py --input <file> --server-version 0.7.2 --json <new file in scratch>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from hytalegym.sim.common.profile import StepDuration  # noqa: E402
from hytalegym.sim.gen7.profile import PROFILE as GEN7_PROFILE  # noqa: E402

SCHEMA = "catch-time-horizons-v1"
TICK_SECONDS = {
    "0.7.2": GEN7_PROFILE.step_duration.nominal_seconds,
    "0.5.7": StepDuration().loaded_seconds,
}
RUNNER_CEILING = 1000
CONSOLE_CEILING = 512
DEFAULT_QUANTILES = (0.5, 0.9, 1.0)
RULE = (
    "ticks_i = max(1, ceil(seconds_i / tick - 1e-9)); T_q = nearest-rank q-quantile of the sorted ticks "
    "(no interpolation); pow2 = smallest power of two >= T_q; rung = min(pow2, effective ceiling), "
    "capped reported; effective ceiling = min(runner 1000, console 512)"
)
EXAMPLE = (
    '{"seconds": 3.1}\n{"seconds": 4.6}\n{"seconds": 6.2}\n{"seconds": 9.8}\n{"seconds": 15.0}\n'
)


def read_seconds(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    seconds: list[float] = []
    for index, record in enumerate(records):
        value = record.get("seconds") if isinstance(record, dict) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: record {index} needs a finite positive 'seconds', got {record!r}"
            )
        seconds.append(float(value))
    if not seconds:
        raise ValueError(f"{path}: no records")
    return seconds


def compute(
    seconds: list[float],
    server_version: str,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    if server_version not in TICK_SECONDS:
        raise ValueError(f"server_version must be one of {sorted(TICK_SECONDS)}")
    if not seconds:
        raise ValueError("no measured times")
    if any(not 0 < q <= 1 for q in quantiles):
        raise ValueError(f"quantiles must lie in (0, 1], got {list(quantiles)}")
    tick = TICK_SECONDS[server_version]
    ticks = sorted(max(1, math.ceil(value / tick - 1e-9)) for value in seconds)
    effective = min(RUNNER_CEILING, CONSOLE_CEILING)
    rungs = []
    for q in quantiles:
        measured = ticks[max(0, math.ceil(q * len(ticks) - 1e-9) - 1)]
        pow2 = 1 << (measured - 1).bit_length()
        chosen = min(pow2, effective)
        rungs.append({
            "quantile": q,
            "measured_ticks": measured,
            "pow2_ticks": pow2,
            "ticks": chosen,
            "seconds": round(chosen * tick, 3),
            "capped": pow2 > effective,
        })
    return {
        "schema": SCHEMA,
        "server_version": server_version,
        "tick_seconds": tick,
        "records": len(seconds),
        "measured_ticks_sorted": ticks,
        "ceilings": {
            "runner": RUNNER_CEILING,
            "console": CONSOLE_CEILING,
            "effective": effective,
        },
        "cap_applied": any(rung["capped"] for rung in rungs),
        "rules": RULE,
        "rungs": rungs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Catch-time horizon rungs from measured time-to-success."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help='JSON array or JSON Lines of {"seconds": ...} records',
    )
    parser.add_argument(
        "--example", action="store_true", help="print an illustrative input and exit"
    )
    parser.add_argument("--server-version", choices=sorted(TICK_SECONDS), default="0.7.2")
    parser.add_argument(
        "--quantiles",
        default=",".join(str(q) for q in DEFAULT_QUANTILES),
        help="comma-separated quantiles in (0, 1]",
    )
    parser.add_argument("--json", type=Path, metavar="OUT", help="write the result as JSON to OUT")
    args = parser.parse_args(argv)
    if args.example:
        print(EXAMPLE, end="")
        return 0
    if args.input is None:
        parser.error("--input is required unless --example is given")
    quantiles = tuple(float(part) for part in args.quantiles.split(",") if part.strip())
    result = compute(read_seconds(args.input), args.server_version, quantiles)
    result["input"] = {
        "path": str(args.input),
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    for rung in result["rungs"]:
        note = " (CAPPED)" if rung["capped"] else ""
        print(
            f"q={rung['quantile']}: measured {rung['measured_ticks']} ticks -> "
            f"pow2 {rung['pow2_ticks']} -> rung {rung['ticks']} ticks = {rung['seconds']} s{note}"
        )
    print(
        f"server {args.server_version}, tick {result['tick_seconds']} s, records {result['records']}, "
        f"effective ceiling {result['ceilings']['effective']} ticks, cap applied: {result['cap_applied']}"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
