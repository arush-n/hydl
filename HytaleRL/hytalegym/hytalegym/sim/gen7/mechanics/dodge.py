"""Dodge action head size. Switch ``Constructs.dodge_action_head``; P3.

Not a new mechanic: the seven dodge assets are identical in 0.5.10 and 0.7.2
(``weapons/interaction:Dodge`` routes only Left/Right to ``Dodge_Left``/
``Dodge_Right``: Costs.Stamina 2, ApplyForce 13 Set, Dodge_Invulnerability
0.25 s, StaminaRegenDelay Set -0.7) and the engine already executes left and
right only (``DODGE_AUTHORED_ACTION_MASK``). Gen 5 keeps four policy slots
(forward/back are authored no-ops); gen 7 exposes two. Hook: the locomotion
head layout (``observation/v3/schema/contract.py``, ``policy/layout.py``).
Differential: the existing dodge boundary tests on 0.7.2 (FIDELITY row 9).
"""

from __future__ import annotations

from typing import Any

SWITCH = "dodge_action_head"
IMPLEMENTED = False


def action_head(choices: tuple[str, ...]) -> Any:
    """The dodge sub-head for the given choices (gen 7: left, right)."""

    raise NotImplementedError("dodge head construct: design only (mechanics_072.md)")
