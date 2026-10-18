"""Ability chain start (gen 7, row 5f): an accepted ability claims on its request tick and starts one drain later.

Switch ``Constructs.ability_start_split``: ``claim_tick`` (gen 5: the chain starts on the claim tick) or ``next_drain``
(gen 7).

Measured in FIDELITY-0.7.2.md section (af) on the live 0.7.2 bridge, seeds 570057 and 913771: the kunai slot zero claims
on tick 1 (the slot is held on ticks 1 to 10), is accepted and started on tick 2 and finishes on tick 11; the iron sword
claims on tick 1, starts on tick 2 and finishes on tick 14; the shortbow claims on tick 1 and starts on tick 2. Every
measured chain is shifted one tick later at its start and its finish, with the same duration as the gen-5 program.

Mechanism: the admission (``start_abilities``) sets the scheduler to ``-prelude``; the lifecycle reads ``scheduler + prelude``
everywhere, so one more tick of scheduler offset at admission moves the whole chain by one tick, its start and its finish
together. The offset is applied only to the rows the original admitted (``accepted``). Charges are spent at admission and
the recovery timer runs from the claim, both unchanged.

Recharge (record, section (af)): the charge recovery is unchanged at the nominal 1.0 s from the claim (the
``ability_charge_times_seconds`` value), which is 30 ticks at the simulator's 1/30 s tick. On the loaded server the kunai
bank's fifth and sixth charges returned 27 to 29 ticks after their claims (1.016 to 1.027 s of engine time); the simulator's
legal-on ticks are 1 to 3 ticks later than the native ones in all four events (offsets +3, +3, +1, +2 against seeds
570057 and 913771). The 1.02 s fit and the offsets are load jitter of the server's timers, not constants of 0.7.2.

Hook: ``start_abilities`` (combat/arsenal/programs/ability.py). No state leaf: the scheduler already carries the offset.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "ability_start_split"
IMPLEMENTED = True
CLAIM_TICK = "claim_tick"
NEXT_DRAIN = "next_drain"


def _enabled(profile: Any) -> bool:
    value = profile.constructs.ability_start_split
    if value == CLAIM_TICK:
        return False
    if value != NEXT_DRAIN:
        raise ValueError(f"unknown ability_start_split switch {value!r}")
    return True


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``next_drain`` the admission that delays each accepted chain by one drain."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _start_abilities(*args: Any, **kwargs: Any) -> Any:
        arsenal, mechanics, requested, accepted = original(*args, **kwargs)
        shifted = arsenal.ability_scheduler_tick - jnp.int32(1)
        arsenal = arsenal._replace(
            ability_scheduler_tick=jnp.where(
                accepted,
                shifted,
                arsenal.ability_scheduler_tick,
            ),
        )
        return arsenal, mechanics, requested, accepted

    return _start_abilities
