"""Run a trained JAX policy as a live Hytale server NPC.

**Most of this port already existed.** Do not rebuild it:

* ``NativeBridgeSession`` (``adk/runtime/bridge_client/__init__.py``) owns the
  evidence lease, opens the socket, resets, validates the policy input, exposes
  ``make_action_factors`` and ``action_receipt``, and already has
  ``step_policy(policy, carry, key)``.
* The Java side hooks real server NPC combat --
  ``NativePolicyActorState.java`` drives
  ``com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack``.
* ``test_native_learner_v3_full_row_matches_source_aligned_jax`` certifies that
  the observation row the server hands back matches the JAX row.

What did **not** exist is anything that actually drives a *trained* policy
through it. The only caller, ``agents/ppo/agent.py``, constructs the
session under ``purpose="simple_ppo_agent_reference_only"`` and closes it
without opening -- its own report records ``socket_opened: false``. This module
is the missing loop, and nothing more.

Two things it insists on, because both fail silently:

* **Head-order agreement.** The ADK and the server each publish an ordered list
  of action heads. They agree today. If they ever stop agreeing, a "jump"
  arrives as a "guard" and nothing raises -- the NPC just behaves wrongly
  forever. Checked by name before the socket opens.
* **Rejection accounting.** An action the server refuses is not a no-op; it
  means the policy's legality mask disagrees with the server's. Rejections are
  counted and reported, and ``strict`` turns the first one into a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np


class ActionSurfaceMismatch(RuntimeError):
    """The ADK and the server disagree about the action space."""


def assert_action_surface_matches() -> tuple[str, ...]:
    """Refuse to serve if the two action surfaces are ordered differently.

    Compared **by name**, never by width: two surfaces can carry the same head
    count and the same total logit width while meaning different things per
    position.
    """

    from adk.probes.policies import HEAD_SPANS
    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
    )

    adk_heads = tuple(HEAD_SPANS)
    native_heads = tuple(ARSENAL_POLICY_ACTION_HEAD_NAMES)
    if adk_heads != native_heads:
        raise ActionSurfaceMismatch(
            "ADK and native action surfaces differ, so every action would be "
            "misrouted silently.\n"
            f"  ADK    : {adk_heads}\n"
            f"  native : {native_heads}"
        )
    return native_heads


@dataclass(slots=True)
class ServeReport:
    """What the policy did on the live server."""

    steps: int = 0
    accepted: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    checkpoint: str | None = None
    checkpoint_schema: str | None = None
    spec_sha256: str | None = None
    bridge_sha256: str | None = None

    @property
    def acceptance_rate(self) -> float | None:
        return None if self.steps == 0 else self.accepted / self.steps

    def describe(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.acceptance_rate,
            "reject_reasons": dict(
                sorted(self.reject_reasons.items(), key=lambda kv: -kv[1])
            ),
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "checkpoint": self.checkpoint,
            "checkpoint_schema": self.checkpoint_schema,
            "spec_sha256": self.spec_sha256,
            "bridge_sha256": self.bridge_sha256,
        }


def serve(
    handle: Any,
    *,
    checkpoint: str | Path | None = None,
    policy: Any = None,
    host: str = "127.0.0.1",
    port: int = 5556,
    steps: int = 256,
    seed: int = 0,
    decode_mode: str = "factored_argmax",
    strict: bool = False,
    purpose: str = "npc_serve",
) -> ServeReport:
    """Drive one live episode with a trained policy.

    Supply either ``checkpoint`` (loaded through the ADK, which verifies its
    identity against this handle) or an already-built ``policy``.

    ``decode_mode`` defaults to ``factored_argmax`` -- greedy. A live NPC that
    samples will look erratic in a way that reads as a broken policy rather
    than as exploration.
    """

    assert_action_surface_matches()
    if (checkpoint is None) == (policy is None):
        raise ValueError("pass exactly one of checkpoint= or policy=")

    report = ServeReport()

    if checkpoint is not None:
        loaded = handle.ppo.load(checkpoint)
        policy = handle.ppo.policy(loaded.policy_params, decode_mode=decode_mode)
        report.checkpoint = str(Path(checkpoint).resolve())
        adk_metadata = loaded.metadata.get("adk", {})
        report.checkpoint_schema = adk_metadata.get("schema")
        report.spec_sha256 = adk_metadata.get("spec_sha256")

    session = handle.native_session(host=host, port=port, purpose=purpose)
    try:
        session.open()
        frame = session.reset(seed=seed)
        report.bridge_sha256 = getattr(frame, "bridge_sha256", None)

        carry = getattr(policy, "initial_carry", None)
        key = jax.random.key(seed)
        for index in range(steps):
            key, step_key = jax.random.split(key)
            carry, transition = session.step_policy(policy, carry, step_key)
            report.steps += 1
            report.total_reward += float(np.asarray(transition.reward).sum())

            receipt = transition.action_receipt
            accepted = bool(getattr(receipt, "host_policy_action_legal", True))
            if accepted:
                report.accepted += 1
            else:
                report.rejected += 1
                reasons = tuple(
                    getattr(receipt, "host_policy_action_reject_reasons", ())
                ) or ("<unnamed>",)
                for reason in reasons:
                    report.reject_reasons[reason] = (
                        report.reject_reasons.get(reason, 0) + 1
                    )
                if strict:
                    raise RuntimeError(
                        f"server rejected the policy action at step {index}: "
                        f"{list(reasons)}. The policy's legality mask "
                        "disagrees with the server's."
                    )

            if transition.done:
                boundary = transition.boundary
                report.terminated = bool(getattr(boundary, "terminated", False))
                report.truncated = bool(getattr(boundary, "truncated", False))
                break
    finally:
        session.close()

    return report


__all__ = [
    "ActionSurfaceMismatch",
    "ServeReport",
    "assert_action_surface_matches",
    "serve",
]
