"""Regenerate the swap plan and re-pin its step-clock counts (run from HytaleRL/hytalegym/tools).

After a commit that touches a module under hytalegym/jax/ (even a comment), the plan's HEAD line numbers can
go stale. This runs content_extract.redirects into a temporary file and prints what moved against the committed
plan. Unless --dry-run, it writes the plan, re-pins tests/sim/test_render.py if the counts changed, and checks
that hytalegym.sim.gen7.overlay reads them. Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
HYTALEGYM = TOOLS.parent
OVERLAY = HYTALEGYM / "hytalegym/sim/common/overlay.py"
PLAN = HYTALEGYM / "hytalegym/sim/common/content/redirects.json"
TEST = HYTALEGYM / "tests/sim/test_render.py"
PIN_TEST = "def test_step_clock_rows_are_the_swap_plan_source_sites"
PIN = re.compile(r"assert STEP_CLOCK_SITES == (\{[^}]*\})")
COUNT_KEYS = ("sites", "in_source", "redirected", "consumers", "outside_source")
PREFIX = "HytaleRL/hytalegym/hytalegym/"
ENV = dict(os.environ, PYTHONPATH=str(HYTALEGYM), JAX_PLATFORMS="cpu")
# overlay.py is stdlib-only, so it is loaded by path and counting does not import jax; --overlay imports the package.
CHILD = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("sim_overlay", sys.argv[1])
overlay = importlib.util.module_from_spec(spec)
sys.modules["sim_overlay"] = overlay
spec.loader.exec_module(overlay)
plans = [a for a in sys.argv[2:] if not a.startswith("--")]
result = {"counts": [overlay.step_clock_redirects(overlay.load_swap_plan(Path(p)))[1] for p in plans]}
if "--overlay" in sys.argv:
    from hytalegym.sim.gen7.overlay import STEP_CLOCK_SITES
    result["overlay_counts"] = STEP_CLOCK_SITES
print(json.dumps(result, sort_keys=True))
"""


def _child(*args: str) -> dict:
    done = subprocess.run([sys.executable, "-c", CHILD, str(OVERLAY), *args], cwd=TOOLS, env=ENV,
                          capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError("step-clock count failed:\n" + done.stderr[-2000:])
    return json.loads(done.stdout.splitlines()[-1])


def _pin_match(text: str) -> re.Match:
    start = text.index(PIN_TEST)
    end = text.find("\ndef ", start + 1)
    match = PIN.search(text, start, len(text) if end < 0 else end)
    if match is None:
        raise RuntimeError(f"no 'assert STEP_CLOCK_SITES == {{...}}' in {PIN_TEST} ({TEST})")
    return match


def _repinned(text: str, counts: dict) -> str:
    match = _pin_match(text)
    line_start = text.rfind("\n", 0, match.start(1)) + 1
    indent = " " * (match.start(1) - line_start + 1)  # the continuation line stays aligned under the first key
    first = ", ".join(f'"{k}": {counts[k]}' for k in COUNT_KEYS[:4])
    literal = "{" + first + ",\n" + indent + f'"outside_source": {counts["outside_source"]}' + "}"
    return text[: match.start(1)] + literal + text[match.end(1):]


def _diff(old_items: list, new_items: list, key) -> tuple[list, int, int]:
    """Head-line moves (key, old, new), and the head lines added or removed, for items grouped by key."""
    old, new = defaultdict(list), defaultdict(list)
    for item in old_items:
        old[key(item)].append(item["head_line"])
    for item in new_items:
        new[key(item)].append(item["head_line"])
    moves = [(k, a, b) for k in sorted(old.keys() & new.keys()) for a, b in zip(sorted(old[k]), sorted(new[k])) if a != b]
    added = sum(max(0, len(new[k]) - len(old.get(k, []))) for k in new)
    removed = sum(max(0, len(old[k]) - len(new.get(k, []))) for k in old)
    return moves, added, removed


def _report(old: dict, new: dict, old_counts: dict, counts: dict, pinned: dict, dry: bool) -> None:
    records = _diff(old["redirects"], new["redirects"], lambda r: (r["module"], r["symbol"], r["target"]))
    sites = _diff(old["step_duration"]["sites"], new["step_duration"]["sites"],
                  lambda s: (s["module"], s["symbol"], s["kind"], s["scope"], s["token"], s["text"]))
    print(f"regen_swap_plan: {'dry run, nothing written' if dry else 'plan written'}")
    print(f"head: {old['head']} -> {new['head']}" + (" (unchanged)" if old["head"] == new["head"] else ""))
    print(f"redirect records: {len(new['redirects'])} (committed {len(old['redirects'])}); "
          f"status {dict(Counter(r['status'] for r in new['redirects']))}")
    print(f"step-clock redirects: {counts['redirected']} (committed {old_counts['redirected']}); step_duration sites: "
          f"{counts['sites']} (committed {old_counts['sites']}): in_source {counts['in_source']}, "
          f"consumers {counts['consumers']}, outside_source {counts['outside_source']}")
    for label, (moves, added, removed) in (("redirect records", records), ("step_duration sites", sites)):
        print(f"head_line moves, {label}: {len(moves)}; added {added}, removed {removed}")
        grouped: dict[str, list[str]] = defaultdict(list)
        for (module, symbol, *_), before, after in moves:
            grouped[module.removeprefix(PREFIX)].append(f"{symbol} {before}->{after}")
        for module, items in sorted(grouped.items()):
            print(f"    {module}: {len(items)} ({'; '.join(items[:3])}{'; ...' if len(items) > 3 else ''})")
    state = "unchanged" if counts == pinned else f"{'would change' if dry else 'rewritten'} to {counts}"
    print(f"test pin (tests/sim/test_render.py): {state}")


def _run(repo: Path, dry: bool) -> None:
    old = json.loads(PLAN.read_text(encoding="utf-8"))
    test_text = TEST.read_text(encoding="utf-8")
    pinned = ast.literal_eval(_pin_match(test_text).group(1))
    with tempfile.TemporaryDirectory(prefix="regen_swap_plan_") as scratch:
        out = Path(scratch) / "redirects.json"
        done = subprocess.run([sys.executable, "-m", "content_extract.redirects", "--repo", str(repo),
                               "--out", str(out)], cwd=TOOLS, env=ENV, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError("content_extract.redirects failed:\n" + done.stderr[-2000:])
        new = json.loads(out.read_text(encoding="utf-8"))
        old_counts, counts = _child(str(PLAN), str(out))["counts"]
        _report(old, new, old_counts, counts, pinned, dry)
        if not dry:
            shutil.copyfile(out, PLAN)
    if not dry:
        if counts != pinned:
            TEST.write_text(_repinned(test_text, counts), encoding="utf-8")
        verified = _child(str(PLAN), "--overlay")["overlay_counts"]
        if verified != counts:
            raise RuntimeError(f"gen7 overlay reads {verified}, expected {counts}")
        print("gen7 overlay reads the new counts: yes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the swap plan and re-pin its step-clock counts.")
    parser.add_argument("--repo", type=Path, help="git work tree containing HytaleRL/ (default: git toplevel of the cwd)")
    parser.add_argument("--dry-run", action="store_true", help="regenerate to a temporary file and report; write nothing")
    args = parser.parse_args(argv)
    try:
        if args.repo is None:
            top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
            if top.returncode:
                raise RuntimeError("not inside a git work tree; pass --repo")
            args.repo = Path(top.stdout.strip())
        _run(args.repo.resolve(), args.dry_run)
    except RuntimeError as error:
        print(f"regen_swap_plan: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
