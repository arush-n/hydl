"""Array-only opponent-controller state."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class OpponentMemoryState(NamedTuple):
    """Private, entity-indexed controller memory.

    None of these fields are actor observations. A source row may acquire a
    target coordinate only through validity-masked legal perception evidence.
    """

    mode: Array
    home_position: Array
    home_valid: Array
    last_seen_position: Array
    last_seen_valid: Array
    pursuit_elapsed_ticks: Array
    search_elapsed_seconds: Array
    search_timeout_seconds: Array
    navigation_unavailable: Array


__all__ = ["OpponentMemoryState"]
