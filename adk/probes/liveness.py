"""Did anything actually happen?  The check that stops a vacuous pass.

Per-step invariants answer "did the environment do something wrong". They
cannot answer "did the environment do anything at all", and a run where
nothing happens satisfies every one of them.  That failure mode is the easy
one to ship: a scene with no providers bound, a target that never becomes
visible, an action space that turns out to be movement-only.  The audit reports
``no violations`` and the result means nothing.

So these are *run-level* properties, computed from a whole trajectory rather
than a step.  They are warnings, not violations: a movement-only scene is a
legitimate thing to build, and only the caller knows whether it is what they
asked for.  What is not legitimate is not being told.

All host-side: they reduce recorded arrays to Python values.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp


class Liveness(NamedTuple):
    """What a run did, in the terms that decide whether its result is meaningful."""

    observation_changed: bool
    reward_ever_nonzero: bool
    episode_ever_ended: bool
    target_ever_visible: bool
    attack_ever_legal: bool
    distinct_actions: int

    @property
    def vacuous(self) -> bool:
        """True when the run exercised so little that a clean audit proves nothing."""

        return not (self.observation_changed and self.distinct_actions > 1)

    def warnings(self) -> tuple[str, ...]:
        """Human-readable notes about what the run never reached."""

        notes: list[str] = []
        if not self.observation_changed:
            notes.append(
                "observation never changed -- the environment is frozen, and "
                "every invariant below passed vacuously"
            )
        if self.distinct_actions <= 1:
            notes.append(
                f"only {self.distinct_actions} distinct action(s) played -- the "
                "policy is not exploring, or the mask permits nothing"
            )
        if not self.reward_ever_nonzero:
            notes.append(
                "reward was exactly zero for the whole run -- nothing rewarding "
                "or penalised occurred"
            )
        if not self.episode_ever_ended:
            notes.append(
                "no episode ended -- termination and autoreset were never exercised"
            )
        if not self.target_ever_visible:
            notes.append(
                "the target was never visible -- no engagement happened, so "
                "combat behaviour is untested"
            )
        if not self.attack_ever_legal:
            notes.append(
                "attack was never legal -- the reachable action space was "
                "movement-only for this entire run"
            )
        return tuple(notes)


def summarize(records: Mapping[str, Any]) -> Liveness:
    """Reduce a recorded trajectory to its liveness facts.  Host-side.

    Expects the keys produced by :func:`adk.probes.audit.invariant_recorder`.
    """

    observed = jnp.asarray(records["liveness.observed"])
    actions = jnp.asarray(records["liveness.action"])
    flat = actions.reshape(-1, actions.shape[-1])
    distinct = len({tuple(row) for row in flat.tolist()})

    return Liveness(
        observation_changed=not bool(
            jnp.array_equal(observed[0], observed[-1])
        ),
        reward_ever_nonzero=bool(jnp.any(jnp.asarray(records["reward"]) != 0.0)),
        episode_ever_ended=bool(jnp.any(jnp.asarray(records["done"]))),
        target_ever_visible=bool(jnp.any(jnp.asarray(records["liveness.target_visible"]))),
        attack_ever_legal=bool(jnp.any(jnp.asarray(records["liveness.attack_legal"]))),
        distinct_actions=distinct,
    )


__all__ = ["Liveness", "summarize"]
