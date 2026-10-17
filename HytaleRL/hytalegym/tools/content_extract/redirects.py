"""Write the machine-readable redirect list (the render overlay's swap plan).

One record per engine constant or table that the content layer replaces, plus
every site that reads the engine step duration (``step_duration``, from
``step_sites.py``):
module, symbol, line in ``HEAD`` (read with ``git show HEAD:<path>``, never the
working tree, so uncommitted work is not baked into a render), line in the
working tree, whether the file is clean or uncommitted, the adapter, how the
value is held (``kind``) and the content-table fields it reads.

Read-only on git (``rev-parse``, ``show``, ``status``). Imports ``hytalegym``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SCHEMA = "hytalerl_render_redirects_v1"
PACKAGE_PREFIX = "HytaleRL/hytalegym/hytalegym/"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _line(text: str | None, symbol: str, kind: str) -> int | None:
    if text is None:
        return None
    lines = text.splitlines()
    if kind == "module_constant":
        pattern = re.compile(rf"^\s*{re.escape(symbol)}\s*(:[^=]*)?=")
        for number, line in enumerate(lines, start=1):
            if pattern.match(line):
                return number
    for number, line in enumerate(lines, start=1):
        if symbol in line:
            return number
    return None


def _repo_path(module_path: str) -> str:
    return str(Path(PACKAGE_PREFIX + module_path).as_posix()).replace(
        "hytalegym/hytalegym/../tests/", "hytalegym/tests/"
    )


def build(repo: str | Path) -> dict:
    from hytalegym.sim.common.content import adapters

    root = Path(repo)
    head = _git(root, "rev-parse", "HEAD").strip()
    status_lines = _git(root, "status", "--porcelain", "--", "HytaleRL").splitlines()
    status = {line[3:].strip(): line[:2] for line in status_lines}
    records = []
    head_cache: dict[str, str | None] = {}
    for entry in adapters.SWAP_PLAN:
        path = _repo_path(entry.module_path)
        if path not in head_cache:
            try:
                head_cache[path] = _git(root, "show", f"HEAD:{path}")
            except subprocess.CalledProcessError:
                head_cache[path] = None
        working = (root / path).read_text(encoding="utf-8") if (root / path).is_file() else None
        code = status.get(path)
        if head_cache[path] is None:
            state = "absent_in_head"
        elif code is None:
            state = "clean"
        else:
            state = "uncommitted" if code.strip() else "clean"
        records.append(
            {
                "target": entry.target,
                "module": path,
                "symbol": entry.symbol,
                "kind": entry.kind,
                "head_line": _line(head_cache[path], entry.symbol, entry.kind),
                "worktree_line": _line(working, entry.symbol, entry.kind),
                "status": state,
                "adapter": entry.adapter,
                "content": list(adapters.ADAPTER_SOURCES[entry.adapter]),
            }
        )
    from .step_sites import scan as scan_step_sites

    step_sites = scan_step_sites(root)
    return {
        "schema": SCHEMA,
        "head": head,
        "step_duration": {
            "content": "SimProfile.step_duration (gen 5: nominal 1/30 s, loaded 0.045 s, 3 per-lane timing profiles; gen 7: one value, 1/30 s)",
            "site_count": step_sites["site_count"],
            "kinds": step_sites["kinds"],
            "sites": step_sites["sites"],
        },
        "source_rule": "render reads every module with `git show HEAD:<module>`; head_line is the line in that text",
        "redirect_count": len(records),
        "redirects": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="the git work tree (contains HytaleRL/)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = build(args.repo)
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    states: dict[str, int] = {}
    for record in document["redirects"]:
        states[record["status"]] = states.get(record["status"], 0) + 1
    print(json.dumps({"head": document["head"], "redirects": document["redirect_count"], "status": states}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
