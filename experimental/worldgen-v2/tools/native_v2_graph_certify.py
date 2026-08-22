"""Freeze or validate live evidence for one compiled procedural V2 graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


WORLDGEN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = WORLDGEN_ROOT.parents[1]
for entry in (WORKSPACE, WORLDGEN_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from native_graph_compiler import (  # noqa: E402
    build_live_validation,
    validate_live_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--capture-report", type=Path)
    parser.add_argument("--analysis-report", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_live_validation(args.output)
    else:
        required = {
            "evidence_root": args.evidence_root,
            "pack": args.pack,
            "pilot_report": args.pilot_report,
            "capture_report": args.capture_report,
            "analysis_report": args.analysis_report,
            "server_log": args.server_log,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("missing required build arguments: " + ", ".join(missing))
        result = build_live_validation(
            evidence_root=args.evidence_root,
            pack=args.pack,
            pilot_report=args.pilot_report,
            capture_report=args.capture_report,
            analysis_report=args.analysis_report,
            server_log=args.server_log,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
