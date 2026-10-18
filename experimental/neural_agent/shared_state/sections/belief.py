"""The belief section of S_t: the belief features per lane, written by the session from the belief module.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_FEATURE_NAMES,
    BELIEF_LATENT_WIDTH,
    SharedBeliefState,
    _finite_tuple,
)
from experimental.neural_agent.executive.store import validate_against_schema

from .._common import (
    LATENT_PREFIX_WIDTH,
    _bounded,
    _flag,
    _instances,
)
from ..schema import (
    _BELIEF_SECTION_SCHEMA,
)


# ---- belief section: written by the belief module ----


@dataclass(frozen=True)
class BeliefLane:
    """One lane of the belief summary.

    Attributes:
        features: ``BELIEF_FEATURE_COUNT`` finite values, in BELIEF_FEATURE_NAMES order.
        known: ``BELIEF_FEATURE_COUNT`` flags; False means not published, which is not zero.
        latent_norm: The L2 norm of the lane's latent, finite and non-negative.
        latent_prefix: The first ``LATENT_PREFIX_WIDTH`` latent values.
    """

    features: tuple[float, ...]
    known: tuple[bool, ...]
    latent_norm: float
    latent_prefix: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the widths and values and normalize them to Python floats and bools."""

        features = _finite_tuple(self.features, BELIEF_FEATURE_COUNT, "BeliefLane.features")
        known = tuple(_flag(flag, f"BeliefLane.known[{index}]") for index, flag in enumerate(self.known))
        prefix = _finite_tuple(self.latent_prefix, LATENT_PREFIX_WIDTH, "BeliefLane.latent_prefix")
        if len(features) != BELIEF_FEATURE_COUNT or len(known) != BELIEF_FEATURE_COUNT:
            raise ValueError(
                f"BeliefLane needs {BELIEF_FEATURE_COUNT} features and known flags, "
                f"got {len(features)} and {len(known)}"
            )
        if len(prefix) != LATENT_PREFIX_WIDTH:
            raise ValueError(f"BeliefLane.latent_prefix needs {LATENT_PREFIX_WIDTH} values, got {len(prefix)}")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "known", known)
        object.__setattr__(self, "latent_norm", _bounded(self.latent_norm, "BeliefLane.latent_norm"))
        object.__setattr__(self, "latent_prefix", prefix)


@dataclass(frozen=True)
class BeliefSection:
    """The belief module's section: a lane-wise summary of B_t (``lanes[b]`` is lane b)."""

    lanes: tuple[BeliefLane, ...] = ()

    def __post_init__(self) -> None:
        """Normalize ``lanes`` to a tuple of BeliefLane."""

        object.__setattr__(self, "lanes", _instances(self.lanes, BeliefLane, "BeliefSection.lanes"))

    @classmethod
    def from_belief(cls, belief: SharedBeliefState) -> BeliefSection:
        """Summarize B_t per lane: features, known flags, latent L2 norm and the first 8 latent values.

        Raises:
            TypeError: If ``belief`` is not a SharedBeliefState.
            ValueError: If a shape or the known dtype is wrong, or a value is not finite.
        """

        if not isinstance(belief, SharedBeliefState):
            raise TypeError(f"belief must be a SharedBeliefState, got {type(belief).__name__}")
        features = np.asarray(belief.features)
        known = np.asarray(belief.feature_known)
        latent = np.asarray(belief.latent, dtype=np.float64)
        lanes = features.shape[0] if features.ndim == 2 else -1
        expected = ((lanes, BELIEF_FEATURE_COUNT), (lanes, BELIEF_FEATURE_COUNT), (lanes, BELIEF_LATENT_WIDTH))
        if (features.shape, known.shape, latent.shape) != expected or known.dtype != np.bool_:
            raise ValueError(
                f"belief must hold features [B, {BELIEF_FEATURE_COUNT}], feature_known bool "
                f"[B, {BELIEF_FEATURE_COUNT}] and latent [B, {BELIEF_LATENT_WIDTH}]; got {features.shape}, "
                f"{known.shape} {known.dtype} and {latent.shape}"
            )
        norms = np.linalg.norm(latent, axis=1)
        return cls(
            tuple(
                BeliefLane(
                    features=tuple(features[lane]),
                    known=tuple(known[lane]),
                    latent_norm=float(norms[lane]),
                    latent_prefix=tuple(latent[lane, :LATENT_PREFIX_WIDTH]),
                )
                for lane in range(lanes)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping; features and known flags are keyed by BELIEF_FEATURE_NAMES."""

        return {
            "lanes": [
                {
                    "features": dict(zip(BELIEF_FEATURE_NAMES, lane.features)),
                    "known": dict(zip(BELIEF_FEATURE_NAMES, lane.known)),
                    "latent_norm": lane.latent_norm,
                    "latent_prefix": list(lane.latent_prefix),
                }
                for lane in self.lanes
            ]
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> BeliefSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _BELIEF_SECTION_SCHEMA, "$.belief")
        return cls(
            tuple(
                BeliefLane(
                    features=tuple(item["features"][name] for name in BELIEF_FEATURE_NAMES),
                    known=tuple(item["known"][name] for name in BELIEF_FEATURE_NAMES),
                    latent_norm=item["latent_norm"],
                    latent_prefix=tuple(item["latent_prefix"]),
                )
                for item in mapping["lanes"]
            )
        )

