"""Fixed-shape, evidence-gated opponent controller memory."""

from hytalegym.jax.combat.opponents.schema.contract import (
    OPPONENT_MODE_CHASE,
    OPPONENT_MODE_INACTIVE,
    OPPONENT_MODE_RETURN_HOME_REQUIRED,
    OPPONENT_MODE_SEARCH_REQUIRED,
    SEARCH_TIMEOUT_MAX_SECONDS,
    SEARCH_TIMEOUT_MIN_SECONDS,
)
from hytalegym.jax.combat.opponents.runtime.factory import reset_opponent_memory
from hytalegym.jax.combat.opponents.runtime.kernel import (
    pairwise_hearing_evidence,
    step_opponent_memory,
)
from hytalegym.jax.combat.opponents.schema.types import OpponentMemoryState


__all__ = [
    "OPPONENT_MODE_CHASE",
    "OPPONENT_MODE_INACTIVE",
    "OPPONENT_MODE_RETURN_HOME_REQUIRED",
    "OPPONENT_MODE_SEARCH_REQUIRED",
    "OpponentMemoryState",
    "SEARCH_TIMEOUT_MAX_SECONDS",
    "SEARCH_TIMEOUT_MIN_SECONDS",
    "pairwise_hearing_evidence",
    "reset_opponent_memory",
    "step_opponent_memory",
]
