"""Independent black-box validation of JAX-vs-native fidelity."""

from adk.validation.baselines import Arm, discriminates, report
from adk.validation.identity import Identity, IdentityDrift, assert_stable, capture
from adk.validation.keys import env_key, reset_keys, stream
from worlds.region import (
    RegionScene,
    authored_reach,
    available,
    region_scene,
    world_action_heads,
)
from adk.validation.verdict import Check, Finding, Report, Verdict, decide
from adk.validation.lifecycle import (
    AGENT,
    IDLE_SLOT,
    agent_column,
    assert_reachable,
    completions,
    rising_edges,
)

__all__ = [
    "AGENT",
    "Arm",
    "discriminates",
    "report",
    "Check",
    "Finding",
    "Report",
    "Verdict",
    "decide",
    "IDLE_SLOT",
    "Identity",
    "IdentityDrift",
    "RegionScene",
    "agent_column",
    "assert_reachable",
    "assert_stable",
    "authored_reach",
    "available",
    "capture",
    "region_scene",
    "world_action_heads",
    "env_key",
    "reset_keys",
    "stream",
    "completions",
    "rising_edges",
]
