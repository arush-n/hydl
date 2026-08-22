"""Shared native interaction tick clocks for compiled combat programs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    EVENT_FLAG_INJECTED_SELECTOR,
    EVENT_FLAG_PARALLEL_FORK,
    EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR,
    EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR,
    EVENT_FLAG_SERVER_SELECTOR,
    EVENT_SCHEDULER_CLOCK_COUNT,
    INTERACTION_QUEUE_DELAY_TICKS,
    NON_PLAYER_SELECTOR_START_DELAY_TICKS,
    PARALLEL_FORK_START_DELAY_TICKS,
    PLAYER_SELECTOR_CLIENT_SYNC_DELAY_TICKS,
    POSITIVE_RUNTIME_START_DELAY_TICKS,
)


# Float32 clocks accumulate one delta per tick. Decimal rates such as 1/30 can
# land a few ULPs below their mathematically exact boundary; treating that as
# another full engine tick violates the duration-ceil contract. This epsilon
# is far smaller than one 30 TPS tick and is applied on both sides of a crossing
# so a one-shot event cannot fire again on the following tick.
SCHEDULER_CLOCK_EPSILON_SECONDS = jnp.float32(1.0e-5)


def event_uses_progressive_selector(event_flags: jax.Array) -> jax.Array:
    """Return whether an event advances an authored selector over time."""

    return (
        event_flags
        & jnp.uint32(
            EVENT_FLAG_INJECTED_SELECTOR
            | EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR
            | EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR
        )
    ) != 0


def event_dispatch_delay_ticks(
    event_time_seconds: jax.Array,
    event_flags: jax.Array,
    player_backed_actor_mask: jax.Array | bool = False,
) -> jax.Array:
    """Return native scheduler boundaries crossed before an event can run.

    Player-backed actors replace an authored server selector with
    ``ClientSourcedSelector``.  Only selector rows receive the extra
    request/response boundaries; ordinary events keep their existing clock.
    The actor mask may be scalar or omit trailing event/resource axes.
    """

    positive_runtime = event_time_seconds > jnp.float32(0.0)
    server_selector = (event_flags & jnp.uint32(EVENT_FLAG_SERVER_SELECTOR)) != 0
    parallel_fork = (event_flags & jnp.uint32(EVENT_FLAG_PARALLEL_FORK)) != 0
    player_backed = jnp.asarray(player_backed_actor_mask, dtype=jnp.bool_)
    while player_backed.ndim < event_time_seconds.ndim:
        player_backed = player_backed[..., None]
    return (
        jnp.int32(INTERACTION_QUEUE_DELAY_TICKS)
        + positive_runtime.astype(jnp.int32)
        * jnp.int32(POSITIVE_RUNTIME_START_DELAY_TICKS)
        + server_selector.astype(jnp.int32)
        * jnp.int32(NON_PLAYER_SELECTOR_START_DELAY_TICKS)
        + parallel_fork.astype(jnp.int32) * jnp.int32(PARALLEL_FORK_START_DELAY_TICKS)
        + (server_selector & player_backed).astype(jnp.int32)
        * jnp.int32(PLAYER_SELECTOR_CLIENT_SYNC_DELAY_TICKS)
    )


def advance_scheduler_clocks(
    active: jax.Array,
    scheduler_tick: jax.Array,
    clocks: jax.Array,
    dt_seconds: jax.Array,
) -> jax.Array:
    delay_axis = jnp.arange(
        1,
        EVENT_SCHEDULER_CLOCK_COUNT + 1,
        dtype=jnp.int32,
    )
    advances = active[..., None] & (scheduler_tick[..., None] >= delay_axis)
    return clocks + jnp.where(
        advances,
        dt_seconds[..., None],
        jnp.float32(0.0),
    )


def scheduler_clock(
    clocks: jax.Array,
    delay_ticks: jax.Array,
) -> jax.Array:
    index = jnp.clip(
        delay_ticks - jnp.int32(1),
        jnp.int32(0),
        jnp.int32(EVENT_SCHEDULER_CLOCK_COUNT - 1),
    )
    expanded = clocks
    while expanded.ndim < index.ndim + 1:
        expanded = expanded[..., None, :]
    return jnp.take_along_axis(
        expanded,
        index[..., None],
        axis=-1,
    )[..., 0]


def event_clock_crossed(
    prior_clock: jax.Array,
    after_clock: jax.Array,
    event_time_seconds: jax.Array,
) -> jax.Array:
    """Return whether an advancing native clock crossed an event boundary.

    Native interactions finish when their elapsed time reaches the authored
    runtime.  The interval is therefore open on the prior clock and closed on
    the advanced clock.  Time-zero interactions are the sole exception: they
    execute on the first clock advance from zero.
    """

    advanced = after_clock > prior_clock
    return advanced & (
        (
            (event_time_seconds == jnp.float32(0.0))
            & (prior_clock == jnp.float32(0.0))
        )
        | (
            (
                event_time_seconds
                > prior_clock + SCHEDULER_CLOCK_EPSILON_SECONDS
            )
            & (
                event_time_seconds
                <= after_clock + SCHEDULER_CLOCK_EPSILON_SECONDS
            )
        )
    )


def scheduler_clock_reached(
    after_clock: jax.Array,
    scheduled_time_seconds: jax.Array,
) -> jax.Array:
    """Return whether scheduled work has had an executable clock sample.

    A non-zero timestamp is complete once its scheduler clock reaches that
    timestamp.  Time-zero work still needs one real clock advance: treating an
    untouched zero-valued clock as complete would let a short parent root end
    before a delayed instant child (for example a parallel resource commit)
    ever executes.
    """

    return (
        after_clock + SCHEDULER_CLOCK_EPSILON_SECONDS
        >= scheduled_time_seconds
    ) & (
        (scheduled_time_seconds > jnp.float32(0.0))
        | (after_clock > jnp.float32(0.0))
    )


__all__ = [
    "advance_scheduler_clocks",
    "event_clock_crossed",
    "event_dispatch_delay_ticks",
    "event_uses_progressive_selector",
    "scheduler_clock_reached",
    "scheduler_clock",
    "SCHEDULER_CLOCK_EPSILON_SECONDS",
]
