"""Guard cancels a melee attack only while the attack's root winds up (0.7.2). Switch ``Constructs.guard_cancel_window``; row 5d.

0.7.2 removes ``Rules.Interrupting: [Primary]`` from the Sword, Mace, Battleaxe and Daggers guard roots (the overlay's
``guard_*_no_primary_interrupt`` patches) and declares ``Rules.InterruptedBy: [Secondary]`` on each of their Primary
attack roots: a ``Simple`` interaction whose ``RunTime`` is the wind-up before the rule-free damage selector (sword
swings 0.111 s, sword down 0.361 s, mace 0.536 s, battleaxe 0.348 and 0.4 s, daggers 0.066 s; the weapons table rows
``interaction:<id>`` of ``sim/gen7/data``). Measured on the live 0.7.2 server (2026-10-18; attack on tick 1, Guard
held from onset k): Guard cancels the attack exactly while its root runs (sword left and right k = 2..5, sword down
2..12, mace 2..18, battleaxe down-left 2..12) and never later. Gen 5 cancels at any time through the guard root's
mask; the patched gen-7 render cancelled at no time, because nothing carried the attack roots' rule.

The engine already holds an ability's ``InterruptedBy`` mask with an "interruptible after" boundary
(``programs/interaction_rules.active_operation_interrupted_by``; the crossbow reload uses it). Gen 7
(``attack_root``) adds the "interruptible until" case with two hooks:

- ``ability_root_window`` wraps the rendered profile's ``_ability``: an attack of the four families whose 0.7.2 root
  declares ``InterruptedBy`` Secondary with a ``RunTime``, and that sets no mask of its own, gets the Secondary bit
  and the root's ``RunTime`` as its boundary (both from the content table, not per call site);
- ``operation_window`` wraps ``active_operation_interrupted_by``: for a Primary attack of the four families that
  carries exactly the Secondary bit and a positive boundary, the boundary ends the window instead of opening it.

Gen 5 (``guard_root``) returns both originals unchanged. Not modelled here (kept as separate rows): a Guard held
through the end of an uncancelled attack never rises natively, while the engine raises it when the attack ends; the
engine resolves a one-tick thrust request through the primary hold table to the left swing, so its "thrust" carries
the left swing's window; native admits an attack one tick before the engine's legality view, so the first window tick
(onset 2 for a swing) is not cancelled by the engine.
"""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any

SWITCH = "guard_cancel_window"
IMPLEMENTED = True
GUARD_ROOT = "guard_root"
ATTACK_ROOT = "attack_root"
#: The families whose 0.7.2 guard roots lost ``Interrupting: [Primary]`` (the overlay's guard patches).
ATTACK_PREFIXES = (
    "Weapon_Sword_Primary_",
    "Weapon_Mace_Primary_",
    "Weapon_Battleaxe_Primary_",
    "Weapon_Daggers_Primary_",
)


def _enabled(profile: Any) -> bool:
    value = profile.constructs.guard_cancel_window
    if value == GUARD_ROOT:
        return False
    if value != ATTACK_ROOT:
        raise ValueError(f"unknown guard_cancel_window switch {value!r}")
    return True


def root_window_seconds(tables: Any, asset: str) -> float | None:
    """The 0.7.2 wind-up of an attack root that ``InterruptedBy`` Secondary, or None."""

    if not asset.startswith(ATTACK_PREFIXES):
        return None
    try:
        document = tables.row("weapons", "interaction:" + asset)["document"]
    except KeyError:
        return None
    rules = document.get("Rules") or {}
    run_time = document.get("RunTime")
    if (
        document.get("Type") != "Simple"
        or "Secondary" not in (rules.get("InterruptedBy") or ())
        or isinstance(run_time, bool)
        or not isinstance(run_time, (int, float))
        or run_time <= 0
    ):
        return None
    return float(run_time)


def ability_root_window(original: Any, profile: Any) -> Any:
    """Hook for the profile's ``_ability``: stamp the attack roots' Secondary window."""

    if not _enabled(profile):
        return original
    from hytalegym.sim.common.content import load

    tables = load(profile)
    secondary_bit = 1 << original.__globals__["INTERACTION_TYPE_SECONDARY"]
    windows: dict[str, float | None] = {}

    @wraps(original)
    def _ability(asset: str, duration: float, *events: dict, **kwargs: Any) -> dict:
        if asset not in windows:
            windows[asset] = root_window_seconds(tables, asset)
        window = windows[asset]
        if window is not None and not kwargs.get("interrupted_by_type_mask", 0):
            kwargs = {**kwargs, "interrupted_by_type_mask": secondary_bit, "interruptible_after_seconds": window}
        return original(asset, duration, *events, **kwargs)

    return _ability


def operation_window(original: Any, profile: Any) -> Any:
    """Hook for ``active_operation_interrupted_by``: an attack root's boundary ends its window."""

    if not _enabled(profile):
        return original
    env = original.__globals__
    jnp = env["jnp"]
    contract = import_module(original.__module__.rsplit(".programs.", 1)[0] + ".schema.contract")
    active_type = env["active_ability_interaction_type"]
    secondary_bit = jnp.uint32(1 << contract.INTERACTION_TYPE_SECONDARY)
    families = jnp.asarray(
        (contract.FAMILY_SWORD, contract.FAMILY_MACE, contract.FAMILY_BATTLEAXE, contract.FAMILY_DAGGERS),
        dtype=jnp.int32,
    )

    @wraps(original)
    def active_operation_interrupted_by(admitted_arsenal, loadout, incoming_type):
        after = original(admitted_arsenal, loadout, incoming_type)
        # Active, a valid incoming type in the mask, with no boundary: the window's other three conditions.
        typed = original(
            admitted_arsenal,
            loadout._replace(
                ability_interruptible_after_seconds=jnp.zeros_like(loadout.ability_interruptible_after_seconds)
            ),
            incoming_type,
        )
        capacity = loadout.ability_interrupted_by_type_mask.shape[2]
        slot = jnp.clip(admitted_arsenal.active_ability_slot, 0, capacity - 1)[..., None]
        mask = jnp.take_along_axis(loadout.ability_interrupted_by_type_mask, slot, axis=2)[..., 0]
        boundary = jnp.take_along_axis(loadout.ability_interruptible_after_seconds, slot, axis=2)[..., 0]
        windowed = (
            (mask.astype(jnp.uint32) == secondary_bit)
            & (boundary > 0.0)
            & jnp.any(loadout.weapon_family.astype(jnp.int32)[..., None] == families, axis=-1)
            & (active_type(admitted_arsenal, loadout) == contract.INTERACTION_TYPE_PRIMARY)
        )
        before_end = admitted_arsenal.ability_elapsed_seconds < boundary
        while windowed.ndim < after.ndim:
            windowed = windowed[..., None]
            before_end = before_end[..., None]
        return jnp.where(windowed, typed & before_end, after)

    return active_operation_interrupted_by
