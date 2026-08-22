"""Derived combat statistics — the numbers fighting games actually score on.

Damage and hits describe *what landed*.  They say nothing about whether the
agent is playing well, and a policy optimised on damage alone learns to trade
hits rather than win exchanges.  Everything here is derived from a recorded
trajectory and is **jittable**, so it can serve as a reward term or as an
evaluation statistic without a second implementation.

Conventions, all of which have bitten this project:

* Arrays are ``(T,)`` or ``(T, B)`` — time first.  Ability arrays arrive as
  ``(T, B, ENTITY)``; index the agent before calling anything here.
* **Admission and occupancy are not activation.**  ``ability_accepted`` is a
  request-*admission* event -- the request entering the interaction queue,
  granted readily -- and ``projectile_count`` is live-projectile occupancy.
  Neither counts activations.  Use :func:`rising_edges` over
  ``active_ability_slot >= 0``.
* Windows are in **ticks**, and the server is 30 TPS.  A window quoted in
  seconds is ``ceil(seconds * 30)`` ticks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: Server tick rate.  Every seconds-denominated window converts through this.
TICKS_PER_SECOND = 30


def ticks(seconds: float) -> int:
    """Whole ticks spanning ``seconds``, rounded up.

    A boundary that falls mid-tick is only reached on the following tick, so
    0.25 s is 8 ticks and 0.35 s is 11.
    """

    return int(jnp.ceil(seconds * TICKS_PER_SECOND))


def rising_edges(active: jax.Array) -> jax.Array:
    """Ticks where ``active`` turns on — activations, not occupied ticks.

    Summing an occupancy flag counts how long something was live.  Forcing one
    ability for 256 ticks yields 512 "acceptances" across a batch of 2 and
    exactly as many requests, which reads as an unenforced cooldown and is
    wrong.  Rising edges are the activation count.
    """

    previous = jnp.concatenate(
        [jnp.zeros_like(active[:1]), active[:-1]], axis=0
    )
    return active & jnp.logical_not(previous)


def combo_lengths(hit: jax.Array, *, gap: int) -> jax.Array:
    """Length of the combo each hit belongs to, at the tick the combo ends.

    A *combo* is a run of hits separated by no more than ``gap`` idle ticks —
    the standard fighting-game notion of a chain the opponent never escaped.
    ``gap`` should be the opponent's recovery window; hits further apart are
    two exchanges, not one combo.

    Returns ``(T, ...)`` where a combo's total length appears on its final
    tick and every other tick is 0, so ``result.max()`` is the longest combo
    and ``(result > 0).sum()`` is the number of combos.
    """

    if gap < 0:
        raise ValueError("gap must be >= 0")

    def step(carry, current):
        run, idle = carry
        # A hit extends the run; idle resets only once the gap is exceeded.
        extended = jnp.where(current, run + 1, run)
        idle_next = jnp.where(current, 0, idle + 1)
        broken = idle_next > gap
        emit = jnp.where(broken & (extended > 0), extended, 0)
        run_next = jnp.where(broken, 0, extended)
        return (run_next, idle_next), emit

    initial = (jnp.zeros(hit.shape[1:], jnp.int32), jnp.zeros(hit.shape[1:], jnp.int32))
    (trailing, _), emitted = jax.lax.scan(step, initial, hit.astype(bool))
    # A combo still open at the end of the episode is real; credit it on the
    # last tick rather than discarding it.
    return emitted.at[-1].set(jnp.maximum(emitted[-1], trailing))


def combo_count(hit: jax.Array, *, gap: int) -> jax.Array:
    """How many distinct combos landed."""

    return (combo_lengths(hit, gap=gap) > 0).sum(axis=0)


def longest_combo(hit: jax.Array, *, gap: int) -> jax.Array:
    """Longest chain of hits the opponent never escaped."""

    return combo_lengths(hit, gap=gap).max(axis=0)


def punish_rate(hit: jax.Array, target_vulnerable: jax.Array) -> jax.Array:
    """Fraction of hits landed while the opponent could not answer.

    *Punishing* — hitting during recovery or a whiffed windup — is the
    clearest single separator between a strong player and one who merely
    trades.  ``target_vulnerable`` is typically the recovery phase column.
    """

    landed = hit.astype(jnp.float32).sum(axis=0)
    punished = (hit & target_vulnerable.astype(bool)).astype(jnp.float32).sum(axis=0)
    return jnp.where(landed > 0, punished / jnp.maximum(landed, 1.0), 0.0)


def whiff_rate(started: jax.Array, hit: jax.Array, *, window: int) -> jax.Array:
    """Fraction of ability activations that landed nothing within ``window``.

    Whiffing is not merely wasted damage: the recovery frames are when the
    opponent punishes.  A policy blind to this learns to spam.

    ``started`` must be activations (:func:`rising_edges`), not occupancy.
    """

    if window < 1:
        raise ValueError("window must be >= 1")
    landed_soon = _any_within(hit.astype(bool), window)
    activations = started.astype(jnp.float32).sum(axis=0)
    connected = (started.astype(bool) & landed_soon).astype(jnp.float32).sum(axis=0)
    return jnp.where(
        activations > 0, 1.0 - connected / jnp.maximum(activations, 1.0), 0.0
    )


def _any_within(flag: jax.Array, window: int) -> jax.Array:
    """True at tick t if ``flag`` is true anywhere in ``[t, t + window)``."""

    padded = jnp.concatenate(
        [flag, jnp.zeros((window,) + flag.shape[1:], flag.dtype)], axis=0
    )
    windows = jnp.stack([padded[i : i + flag.shape[0]] for i in range(window)], axis=0)
    return windows.any(axis=0)


def damage_per_stamina(damage: jax.Array, stamina: jax.Array) -> jax.Array:
    """Damage produced per unit of stamina spent.

    Resource efficiency separates a policy that wins an exchange from one that
    wins it and can still defend.  Only *falls* in stamina count as spend;
    regeneration is not a cost.
    """

    spend = jnp.clip(-jnp.diff(stamina, axis=0), a_min=0.0).sum(axis=0)
    dealt = damage.sum(axis=0)
    return jnp.where(spend > 1e-6, dealt / jnp.maximum(spend, 1e-6), 0.0)


def time_to_first_hit(hit: jax.Array) -> jax.Array:
    """Ticks until the first hit lands; ``T`` if none ever does.

    Opening speed, and a cheap proxy for whether the agent can find its
    opponent at all.
    """

    horizon = hit.shape[0]
    index = jnp.argmax(hit.astype(bool), axis=0)
    return jnp.where(hit.astype(bool).any(axis=0), index, horizon)


def engagement_fraction(distance: jax.Array, *, near: float, far: float) -> jax.Array:
    """Fraction of the episode spent inside a chosen range band.

    *Spacing* is the neutral game.  An agent that wins by standing outside
    threat range and one that wins by controlling it look identical in damage
    and completely different here.
    """

    if not near < far:
        raise ValueError("near must be < far")
    inside = (distance >= near) & (distance <= far)
    return inside.astype(jnp.float32).mean(axis=0)


def aggression(started: jax.Array) -> jax.Array:
    """Ability activations per tick — the tempo the policy plays at.

    Multiply by ``TICKS_PER_SECOND`` for actions per second.  Read alongside
    :func:`whiff_rate`: high aggression with a high whiff rate is spam, and
    the two together separate pressure from panic.
    """

    return started.astype(jnp.float32).mean(axis=0)
