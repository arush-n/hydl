"""Drive a trained checkpoint as a live server NPC.

    python -m npc --agent combat/duellist --checkpoint runs/latest.msgpack

Requires the native listener already serving on ``--port``. This takes the
evidence lease for the duration of the episode; if another lane holds it the
session raises rather than stealing it.
"""

from __future__ import annotations

import argparse
import json

from adk import AgentKit
from npc.loop import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m npc", description=__doc__)
    parser.add_argument("--agent", required=True, help="registered agent name")
    parser.add_argument("--checkpoint", required=True, help="path to a saved policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decode",
        default="factored_argmax",
        help="factored_argmax (greedy, the sane default) or a sampling mode",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on the first action the server rejects",
    )
    args = parser.parse_args(argv)

    handle = AgentKit().make(args.agent, num_envs=1, backend="jax")
    report = serve(
        handle,
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        steps=args.steps,
        seed=args.seed,
        decode_mode=args.decode,
        strict=args.strict,
    )
    print(json.dumps(report.describe(), indent=2))
    return 0 if report.rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
