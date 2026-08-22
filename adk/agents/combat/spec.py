"""Combat agents. The only family playable on the current ABI."""

from adk.core.registry import register
from adk.core.spec import AgentSpec

BASELINE = register(AgentSpec(name="combat/baseline", loadout="iron_sword"))
