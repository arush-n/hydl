"""Profile-gated constructs for behaviour gen 5 does not model (design stubs).

One module per ``Constructs`` field that gen 7 sets away from its gen-5
default. Each module names its switch (``SWITCH``), phase and design section in
``../mechanics_072.md`` and exposes typed, unwired interfaces that raise
``NotImplementedError``. Nothing here is imported by the engine yet; a render
(phase 2) adds the hook that calls a construct when its switch is on.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: Constructs field -> module under this package that owns it.
CONSTRUCT_MODULES = {
    "hit_interrupt": "hit_interrupt",
    "mana_pool": "mana_pool",
    "rune_casting": "mana",
    "hookshot": "hookshot",
    "dodge_action_head": "dodge",
    "line_of_sight": "line_of_sight",
    "climbing": "climbing",
    "gliding": "gliding",
    "riding": "riding",
    "npc_velocity_model": "npc_velocity",
    "npc_walk_jump_velocity": "npc_jump",
    "dagger_swing_damage": "dagger_damage",
    "guard_cancel_window": "guard_window",
    "native_binding_types": "native_bindings",
    "walk_descent": "walk_descent",
    "slope_limits": "slope_limits",
    "block_health_reset_on_write": "block_health",
}


def construct_module(switch: str) -> Any:
    return import_module(f"{__name__}.{CONSTRUCT_MODULES[switch]}")


def require_construct(profile: Any, switch: str) -> Any:
    """Return the owning module when ``profile`` turns ``switch`` away from gen 5.

    Raises ``NotImplementedError`` while the construct is a stub, so a render
    cannot silently run gen-7 content with gen-5 behaviour.
    """

    module = construct_module(switch)
    if not getattr(module, "IMPLEMENTED", False):
        raise NotImplementedError(
            f"construct {switch!r} ({module.__name__}) is designed but not implemented; "
            f"see sim/gen7/mechanics_072.md"
        )
    return module
