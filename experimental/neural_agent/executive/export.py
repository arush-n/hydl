"""Export the executive's E_t history as JSON lines, one state per line.

The top LLM writes goal states through the executive harness; J-S1 (System One
decision model, ``docs/SYSTEM_ONE.md`` section 8.4, stage A source 2) trains on
decisions rendered under those states. This module moves the states from the
store to a file and back. Each line is ``ExecutiveState.to_json()``; the first
line is version 0 (the empty state the store starts with) unless ``skip_initial``.

    from experimental.neural_agent.executive.export import export_history, load_history
    count = export_history(store, "eval/out/executive/history.jsonl")
    states = load_history("eval/out/executive/history.jsonl")
"""

from __future__ import annotations

from pathlib import Path

from experimental.neural_agent.contracts import ExecutiveState
from experimental.neural_agent.executive.store import ExecutiveStateStore


def export_history(store: ExecutiveStateStore, path: str | Path, *, skip_initial: bool = True) -> int:
    """Write every stored E_t version to ``path`` as JSON lines and return the count written.

    Args:
        store: The executive state store whose ``history()`` is exported.
        path: Output path; parent directories are created. Overwritten.
        skip_initial: Leave out version 0, the empty state every store starts with.

    Returns:
        The number of lines written.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    states = store.history()
    if skip_initial:
        states = tuple(state for state in states if state.version != 0)
    target.write_text("".join(state.to_json() + "\n" for state in states))
    return len(states)


def load_history(path: str | Path) -> tuple[ExecutiveState, ...]:
    """Read JSON lines written by :func:`export_history` back into states, in file order.

    Raises:
        ValueError: From ``ExecutiveState.from_json`` on a malformed line, naming the line number.
    """

    states: list[ExecutiveState] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            states.append(ExecutiveState.from_json(line))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return tuple(states)


__all__ = ["export_history", "load_history"]
