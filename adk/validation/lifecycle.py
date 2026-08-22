"""Lifecycle edges — the difference between "asked" and "ran".

A mechanic's outcome is only readable once you can say it *executed*. These
helpers exist because the naive edge count is wrong in a way that reads as a
dead mechanic:

    starts = ((slot[1:] >= 0) & (slot[:-1] < 0)).sum()      # WRONG

An ability requested from tick 0 is already active in the first recorded tick,
so the only rising edge in the run is the one this expression cannot see. It
reported ``0`` for runs where the ability demonstrably ran, and that zero was
written up as "the scene cannot start abilities" before the neutral-arm
discriminator showed the channel was correct all along.

The fix is to seed the comparison with the state *before* the trace begins.
``active_ability_slot`` is initialised to ``IDLE_SLOT``
(``entities/abilities/factory.py:256``) and reset to it when a program finishes
(``scheduler.py:452``), so the seed is known rather than guessed.
"""

from __future__ import annotations

import numpy as np

#: Idle sentinel for ``active_ability_slot``, from the Gym's own factory.
IDLE_SLOT = -1

#: Entity axis index of the learning agent.  Channels are ``(T, B, ENTITY)``
#: and entity 1 is the scripted target; reducing before indexing reports the
#: wrong actor's column.
AGENT = 0


def agent_column(series: np.ndarray) -> np.ndarray:
    """Select the agent's ``(T, B)`` view, before any reduction."""

    values = np.asarray(series)
    return values[..., AGENT] if values.ndim == 3 else values


def rising_edges(series: np.ndarray, *, seed: int = IDLE_SLOT) -> np.ndarray:
    """Count transitions from idle to active, per batch element.

    ``seed`` is the value in effect immediately before the first recorded tick.
    Leaving it out is what made a mechanic that fires on tick 0 look dead.
    """

    values = agent_column(series)
    prior = np.concatenate([np.full((1, *values.shape[1:]), seed, values.dtype), values[:-1]])
    return ((values >= 0) & (prior < 0)).sum(axis=0)


def assert_reachable(series: np.ndarray, *, what: str) -> np.ndarray:
    """The positive control every outcome claim depends on.

    Raises rather than returning a flag: a run whose mechanic never executed is
    **invalid**, not negative, and must not reach a comparison.
    """

    values = agent_column(series)
    starts = rising_edges(values)
    if not (values >= 0).any():
        raise AssertionError(
            f"{what} never became active (all values {np.unique(values).tolist()}); "
            "this run is INVALID, not a negative result -- classify it 5, "
            "inconclusive, and say what would make it conclusive"
        )
    return starts


def completions(series: np.ndarray, *, seed: int = IDLE_SLOT) -> np.ndarray:
    """Count transitions from active back to idle.

    Starts without completions means the program is still running at the end of
    the window -- either the horizon is too short or the mechanic is stuck, and
    those need different responses.
    """

    values = agent_column(series)
    prior = np.concatenate([np.full((1, *values.shape[1:]), seed, values.dtype), values[:-1]])
    return ((values < 0) & (prior >= 0)).sum(axis=0)
