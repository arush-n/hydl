"""Drive a scene with a probe policy and report what the environment broke.

The loop is the ADK's own compiled collector, so an audit costs about what a
rollout costs: the invariants ride along as recorded keys rather than as a
separate pass.

Two kinds of question are answered here:

* **Per-step properties** -- :func:`audit` runs the invariants at every step and
  reports where they failed.
* **Whole-run properties** -- determinism and reset stability cannot be seen
  from one step, so they compare two runs instead.

Reporting is host-side and says so; the collection itself never synchronizes.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from adk.policy.actions import action
from adk.policy.fields import field
from adk.probes import invariants, liveness


class Violation(NamedTuple):
    """One invariant that failed, with enough detail to reproduce it."""

    name: str
    steps: int  # how many steps had at least one violating row
    rows: int  # total (step, row) pairs affected
    first_step: int


class AuditReport(NamedTuple):
    """Outcome of one audit run.

    ``clean`` means no invariant fired.  It does **not** mean the run proved
    anything -- check ``conclusive`` for that, because a frozen environment
    violates nothing.
    """

    steps: int
    batch: int
    violations: tuple[Violation, ...]
    liveness: liveness.Liveness
    rewards: jax.Array
    dones: jax.Array

    @property
    def clean(self) -> bool:
        """No invariant was violated."""

        return not self.violations

    @property
    def conclusive(self) -> bool:
        """Clean *and* the run actually exercised the environment."""

        return self.clean and not self.liveness.vacuous

    def describe(self) -> str:
        """A short human-readable verdict.  Host-side."""

        lines = [
            f"{self.steps} steps x {self.batch} envs, "
            f"{len(self.violations)} invariant(s) violated"
        ]
        if self.clean:
            lines.append("  no violations")
        for violation in self.violations:
            lines.append(
                f"  {violation.name:30s} first at step {violation.first_step:4d}"
                f"  ({violation.steps} steps, {violation.rows} rows)"
            )
        notes = self.liveness.warnings()
        if notes:
            lines.append(
                "  VACUOUS -- a clean result here proves nothing:"
                if self.liveness.vacuous
                else "  what this run never reached:"
            )
            for note in notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)


def invariant_recorder(transition: Any) -> dict[str, jax.Array]:
    """Record every invariant plus reward and done.  Traced.

    Checks the *resulting* observation, because a bug shows up in the state an
    action produced, not the one it was chosen from.
    """

    observation = transition.next_actor_input.legal_observation
    records = {
        f"invariant.{name}": flag
        for name, flag in invariants.check(observation, transition.next_state).items()
    }
    records["reward"] = transition.reward
    records["done"] = transition.done
    # Liveness evidence: enough to tell a real run from a frozen one.
    records["liveness.observed"] = observation.base.self_f32
    records["liveness.action"] = transition.action_factors
    records["liveness.target_visible"] = (
        field(observation.base.combat_f32, "combat_f32", "target_visible") > 0.5
    )
    records["liveness.attack_legal"] = action(
        observation.base.action_mask, "action_mask", "attack"
    )
    return records


def audit(
    handle: Any,
    policy: Any,
    steps: int = 64,
    *,
    key: jax.Array | None = None,
    initial_carry: Any = None,
) -> AuditReport:
    """Run a probe policy and report every invariant it broke.

    **Host-side**: reduces the recorded flags to Python integers at the end, so
    it synchronizes once when the run is over -- not per step.

    ``initial_carry`` is forwarded to the collector; policies that need one
    publish it as ``policy.initial_carry``.
    """

    if key is None:
        key = jax.random.key(0)
    if initial_carry is None:
        initial_carry = getattr(policy, "initial_carry", None)

    collector = handle.compile_collector(
        policy, invariant_recorder, steps, initial_carry=initial_carry
    )
    _final, records = collector(key)

    found: list[Violation] = []
    for name in invariants.invariant_names():
        flags = records[f"invariant.{name}"]
        per_step = jnp.any(flags, axis=1)
        total = int(jnp.sum(flags))
        if total == 0:
            continue
        found.append(
            Violation(
                name=name,
                steps=int(jnp.sum(per_step)),
                rows=total,
                first_step=int(jnp.argmax(per_step)),
            )
        )

    rewards = records["reward"]
    return AuditReport(
        steps=steps,
        batch=int(rewards.shape[1]),
        violations=tuple(found),
        liveness=liveness.summarize(records),
        rewards=rewards,
        dones=records["done"],
    )


def _trajectory(handle, policy, steps, key, initial_carry):
    def record(transition):
        return {
            "action": transition.action_factors,
            "reward": transition.reward,
            "done": transition.done,
            "health": transition.next_actor_input.legal_observation.base.self_f32,
        }

    collector = handle.compile_collector(
        policy, record, steps, initial_carry=initial_carry
    )
    return collector(key)[1]


def is_deterministic(
    handle: Any,
    policy: Any,
    steps: int = 32,
    *,
    key: jax.Array | None = None,
    initial_carry: Any = None,
) -> dict[str, bool]:
    """Does the same key produce a bitwise-identical run twice?

    **Host-side.**  A false here invalidates every other result: a replay that
    does not reproduce cannot be used to diagnose anything, and a "fixed" bug
    cannot be confirmed fixed.  Compared exactly, not within a tolerance --
    float drift between identical inputs is itself the bug.
    """

    if key is None:
        key = jax.random.key(0)
    if initial_carry is None:
        initial_carry = getattr(policy, "initial_carry", None)

    first = _trajectory(handle, policy, steps, key, initial_carry)
    second = _trajectory(handle, policy, steps, key, initial_carry)
    return {
        name: bool(jnp.array_equal(first[name], second[name])) for name in first
    }


def reset_is_deterministic(handle: Any, key: jax.Array | None = None) -> bool:
    """Does ``reset`` with one key produce the same starting state twice?

    **Host-side.**  Checked separately from :func:`is_deterministic` because a
    reset bug and a step bug need different fixes, and a run that starts
    differently will diverge for reasons that have nothing to do with the
    policy.
    """

    if key is None:
        key = jax.random.key(0)
    first_state, first_input = handle.reset(key)
    second_state, second_input = handle.reset(key)
    leaves_a = jax.tree.leaves((first_state, first_input))
    leaves_b = jax.tree.leaves((second_state, second_input))
    if len(leaves_a) != len(leaves_b):
        return False
    return all(
        bool(jnp.array_equal(a, b)) for a, b in zip(leaves_a, leaves_b)
    )


def responds_to_actions(
    handle: Any,
    steps: int = 32,
    *,
    key: jax.Array | None = None,
) -> bool:
    """Do different actions produce different trajectories?

    **Host-side.**  Runs a do-nothing policy against a uniform-legal one from
    the same key and compares.  Identical trajectories mean the environment
    ignores the action entirely -- the single most damaging bug possible, and
    one that every per-step invariant passes straight through.

    A ``False`` here invalidates any training result from this scene.
    """

    from adk.probes.policies import uniform_legal

    if key is None:
        key = jax.random.key(0)

    idle = handle.neutral_policy()
    still = _trajectory(handle, idle, steps, key, None)
    moving = _trajectory(handle, uniform_legal, steps, key, None)
    return not bool(jnp.array_equal(still["health"], moving["health"]))


def compare_policies(
    handle: Any,
    policies: Sequence[tuple[str, Any]],
    steps: int = 64,
    *,
    key: jax.Array | None = None,
) -> dict[str, AuditReport]:
    """Audit several probes against the same scene and key.

    A violation that only one probe reaches tells you which factor causes it,
    which is the whole reason to keep several trivial policies around.
    """

    return {
        name: audit(handle, policy, steps, key=key) for name, policy in policies
    }


__all__ = [
    "AuditReport",
    "Violation",
    "audit",
    "compare_policies",
    "invariant_recorder",
    "is_deterministic",
    "reset_is_deterministic",
    "responds_to_actions",
]
