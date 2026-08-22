"""Learning from native NPC behaviour, end to end.

``native_behavior_il``   train recurrent PPO to imitate death-complete traces
``native_jax_pipeline``  the native-data to live-JAX-policy workflow, resumable
``native_live_policy``   persist a native prior after transplant to live JAX
``native_fidelity``      run one JAX checkpoint back through native and compare
``pipeline``             bind native data, JAX policy and boundary evidence
"""

from __future__ import annotations

from agents.ppo.native import native_behavior_il as native_behavior_il
from agents.ppo.native import native_fidelity as native_fidelity
from agents.ppo.native import native_jax_pipeline as native_jax_pipeline
from agents.ppo.native import native_live_policy as native_live_policy
from agents.ppo.native import pipeline as pipeline


__all__ = [
    "native_behavior_il",
    "native_fidelity",
    "native_jax_pipeline",
    "native_live_policy",
    "pipeline",
]
