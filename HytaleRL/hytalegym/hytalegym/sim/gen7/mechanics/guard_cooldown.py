"""Guard re-entry cooldown (gen 7, row 5d): a new guard claim waits 0.35 s after the previous claim.

Switch ``Constructs.guard_reentry_cooldown``: ``none`` (gen 5: no re-entry cooldown; a re-raise is refused only while the
guard is active or an operation is pending) or ``claim_0_35s`` (gen 7).

Measured in FIDELITY-0.7.2.md section (af) on the live 0.7.2 bridge: 24 learner-row guard runs (release 1 to 12, seeds
570057 and 913771). The claim is tick 1 (the guard generation increments there), the activation is tick 3, and the entry
choice reopened on claim + 10 ticks in 17 runs and on claim + 11 in 7. The loaded server's ticks were 0.035 to 0.037 s long,
so its 0.35 s timer elapsed in 10 ticks; the simulator runs at the exact 1/30 s tick, where 0.35 s from the claim reopens
on claim + 11. That one-tick difference is recorded as load jitter in section (af); the reference is the claim (START),
the asset constant (DEFAULT_COOLDOWN, 0.35 s).

Reference (section (af), decision 3 of the timing lane): START, the claim, with the asset constant. On the replayed native
deltas no single threshold fits START (seed 570057 n=7 reopens on tick 11 at 0.367 s of reported time, n=10 on tick 12 at
0.371 s), while ACTIVATION (tick 3) plus 0.297 s fits all 24 runs. At the simulator's exact 1/30 s clock both references
predict the same tick (claim + 11, equal to activation + 9), so the choice is invisible to the simulator. Decisive check
(queued follow-up): re-measure one guard sweep and the kunai bank on an idle machine, where ticks are 1/30 s.

State: ``CombatMechanicsState.guard_reentry_seconds`` (float32 countdown, patched onto mechanics/schema/types.py and
mechanics/factory.py). Hook: ``apply_defense_commands`` (combat/mechanics/runtime/motion.py), once per engine tick at
microticks 1. Per tick, a new claim (a rising request while not held) is refused while the countdown stored after the
previous tick is positive. After the admission, an admitted claim sets the countdown to 0.35 s; otherwise the countdown falls
by one nominal tick (1 / server_ticks_per_second) and floors at zero. The guard-on legality of the policy mask requires a zero
countdown for a new claim (encoder.py patch), so the mask reopens on the tick the countdown reaches zero and a held request is
admitted on the tick after that.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "guard_reentry_cooldown"
IMPLEMENTED = True
NONE = "none"
CLAIM_0_35S = "claim_0_35s"
#: InteractionTypeUtils.DEFAULT_COOLDOWN, counted from the guard claim.
COOLDOWN_SECONDS = 0.35


def _enabled(profile: Any) -> bool:
    value = profile.constructs.guard_reentry_cooldown
    if value == NONE:
        return False
    if value != CLAIM_0_35S:
        raise ValueError(f"unknown guard_reentry_cooldown switch {value!r}")
    return True


def refused(xp: Any, stored: Any, requested: Any, held: Any) -> Any:
    """A new claim is refused while the countdown stored after the previous tick is positive."""

    return requested & ~held & (stored > 0)


def countdown_after(xp: Any, stored: Any, admitted: Any, tick_seconds: Any) -> Any:
    """The countdown after one engine tick: an admitted claim sets the cooldown, otherwise it falls and floors at zero."""

    return xp.where(
        admitted,
        xp.float32(COOLDOWN_SECONDS),
        xp.maximum(stored - tick_seconds, xp.float32(0.0)),
    )


def tick_seconds(xp: Any, rules: Any) -> Any:
    """The nominal engine tick, 1 / server_ticks_per_second (the simulator's exact 1/30 s)."""

    ticks = rules.server_ticks_per_second
    safe = xp.where(xp.isfinite(ticks) & (ticks > 0), ticks, xp.float32(1.0))
    return xp.float32(1.0) / safe


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``claim_0_35s`` the admission that holds a new claim during the cooldown."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _apply_defense_commands(state: Any, commands: Any, yaw_degrees: Any, alive: Any, rules: Any, params: Any) -> Any:
        blocked = refused(jnp, state.guard_reentry_seconds, commands.guard_held, state.guard_held)
        held = commands.guard_held & ~blocked
        updated, info = original(state, commands._replace(guard_held=held), yaw_degrees, alive, rules, params)
        admitted = updated.guard_held & ~state.guard_held
        following = countdown_after(
            jnp,
            state.guard_reentry_seconds,
            admitted,
            tick_seconds(jnp, rules),
        )
        return updated._replace(guard_reentry_seconds=following), info

    return _apply_defense_commands
