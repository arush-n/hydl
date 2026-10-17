"""Shared belief B_t for experimental/neural_agent (ADR-001 D6, experimental).

``SharedBelief`` is the ADK ``BeliefCore``. It reads only the legal
``ActorPolicyInput`` and never ``JaxTrainingInput``. ``features`` holds the
history-free feature readers and ``core`` holds the carry, the GRU and the
belief itself. SKILL.md records the column sources and the formulas.
"""

from experimental.neural_agent.belief.core import (
    ACTION_HEAD_SIZES,
    DRIVE_WIDTH,
    PREVIOUS_ACTION_WIDTH,
    SHARED_BELIEF_CONTRACT,
    BeliefParams,
    SharedBelief,
    SharedBeliefCarry,
    init_belief_params,
    previous_action_one_hot,
)
from experimental.neural_agent.belief.features import (
    ObservationReadings,
    observation_readings,
)

__version__ = "0.1.0"

__all__ = [
    "ACTION_HEAD_SIZES",
    "DRIVE_WIDTH",
    "PREVIOUS_ACTION_WIDTH",
    "SHARED_BELIEF_CONTRACT",
    "BeliefParams",
    "ObservationReadings",
    "SharedBelief",
    "SharedBeliefCarry",
    "init_belief_params",
    "observation_readings",
    "previous_action_one_hot",
]
