"""Shield bash offered only while the guard is held and active (0.7.2 fork of the held guard). Switch
``Constructs.shield_bash_root``; row 5d (FIDELITY-0.7.2.md section (x) and its addendum "the plain-root bash withdrawn").

0.7.2 authors ``Weapon_Shield_Secondary_Guard_Bash`` only as a fork of the held guard. The gen-5 engine carries that as
``ability_guard_fork_type`` (``combat/arsenal/profiles/hytale_0_5_7.py:2956``, ``guard_fork_type=INTERACTION_TYPE_PRIMARY``;
the only authored fork in the catalog, grep 2026-10-18). Its root gate ``_ability_root_admission_mask``
(``combat/arsenal/programs/ability.py:162``; the fork term at lines 172-173) admits a fork while ``guard_held`` is true, so
the stock mask offers the bash from the first tick of a hold. Held is not active: ``guard_active`` rises on tick 3 of a hold
requested on tick 1 (``combat/mechanics/runtime/motion.py:397-418``; measured on the render, ``tests/sim/test_shield_bash_fork.py``).
The live bridge advertises the bash only from that active tick.

Hook: ``_ability_root_admission_mask`` (``programs/ability.py:162``). The mask is the function both the observable legal mask
(``ability_legal_mask``, line 71) and admission read (``start_abilities``, line 328: ``requested_root_legal``, line 367), so one
hook gates both. The construct keeps the stock mask and adds one term: a guard fork root also needs ``guard_active``. The
stock fork term already needs ``guard_held``, so the result is held and active. Not a Patch: the rule is a conjunction on an
existing mask, which the wrapper expresses without editing the source text.

``plain_root`` (gen 5 default) returns the original, so the stock mask (held level, no active term) is kept bit for bit.
``held_guard_fork`` (gen 7) adds the active term. The name ``plain_root`` is the spec's; the stock mask does not offer the
bash as a plain root (no guard held: not legal on tick 0 or after release), so the gap it covers is held-but-not-active.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "shield_bash_root"
IMPLEMENTED = True
PLAIN_ROOT = "plain_root"
HELD_GUARD_FORK = "held_guard_fork"


def _enabled(profile: Any) -> bool:
    value = profile.constructs.shield_bash_root
    if value == PLAIN_ROOT:
        return False
    if value != HELD_GUARD_FORK:
        raise ValueError(f"unknown shield_bash_root switch {value!r}")
    return True


def root_admission(original: Any, profile: Any) -> Any:
    """Hook for ``_ability_root_admission_mask``: a guard fork root is legal only while the guard is also active."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _ability_root_admission_mask(arsenal: Any, mechanics: Any, loadout: Any, alive: Any) -> Any:
        admitted = original(arsenal, mechanics, loadout, alive)
        fork = loadout.ability_guard_fork_type >= 0
        active = mechanics.guard_active[:, :, None]
        return admitted & (~fork | active)

    return _ability_root_admission_mask
