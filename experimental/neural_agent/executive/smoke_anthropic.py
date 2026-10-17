"""One real executive call, if ANTHROPIC_API_KEY is set. The tests never run this file.

Run from the repository root::

    export PYTHONPATH="/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym" JAX_PLATFORMS=cpu
    /Users/arush/hydl/.venv/bin/python -m experimental.neural_agent.executive.smoke_anthropic

Without a key it prints one line and exits 0, with no network call. With a key it makes one
Messages call (model claude-opus-5-5, effort medium, max_tokens 4096). That call costs money.
It prints the proposed E_t and the usage, then offers the reply to the store. The exit status
is 1 if the store rejects the reply.
"""

from __future__ import annotations

import json
import os
import sys

from experimental.neural_agent.contracts import GoalReport, SkillOutcome
from experimental.neural_agent.executive.anthropic_backend import AnthropicExecutiveBackend
from experimental.neural_agent.executive.store import ExecutiveStateStore

SMOKE_SKILLS = ("idle", "pursuit", "checkpoint_route")


def main() -> int:
    """Run one call when a key is set. Return the process exit status."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; no call made.")
        return 0
    store = ExecutiveStateStore(SMOKE_SKILLS)
    backend = AnthropicExecutiveBackend(skill_ids=SMOKE_SKILLS)
    report = GoalReport(
        tick=120,
        lane=0,
        skill_id="pursuit",
        executive_version=0,
        outcome=SkillOutcome.TIMEOUT,
        duration_ticks=300,
        detail="smoke: target out of sight",
    )
    state, usage = backend.propose_with_usage(store.current(), [report], tick=120)
    print(state.to_json())
    print(json.dumps(dict(usage._asdict()), sort_keys=True))
    try:
        store.propose(state)
    except ValueError as error:
        print(f"store rejected the reply: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
