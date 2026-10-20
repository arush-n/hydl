"""Hit interrupt (gen 7, row 5a): the 0.7.2 status rows and the attack rows the construct reads.

An explicit family, not a manifest family: ``FAMILY_MODULES`` does not list it, so the gen-5 and gen-7 content identities are
unchanged. The construct reads ``sim/gen7/data/hit_interrupt.json`` directly and never the archive. Write it from a scratch
directory with an explicit family list, then copy the one file::

    python -m content_extract extract --assets <Assets.zip> --families hit_interrupt --out <scratch>
    cp <scratch>/hit_interrupt.json ../hytalegym/sim/gen7/data/

Rows: ``effect:<name>`` for the eight ``Server/Entity/Effects/Status/Hit_Interrupt_*.json`` assets (Duration, OverlapBehavior,
the HorizontalSpeedMultiplier of ApplicationEffects, Infinite, Debuff), and ``attack:<interaction>`` for every weapon or ability
interaction whose parent chain carries ``DamageEntityParent_Hit_Interrupt`` (the weapons family's derived ``hit_interrupt``
flag, tools/content_extract/weapons.py). The eligible roles are not a row: they are the construct's configured set.
"""

from __future__ import annotations

from typing import Any

from . import weapons as weapon_family

FAMILY = "hit_interrupt"
VERSION = 1
NOTES = ("extracted from the installed 0.7.2 Assets.zip (hit_interrupt v1); read by the gen-7 construct",)
STATUS_PREFIX = "Server/Entity/Effects/Status/"
EFFECTS = (
    "Hit_Interrupt_Eligible",
    "Hit_Interrupt_Flinch",
    "Hit_Interrupt_Stunned",
    "Hit_Interrupt_Recent_Reaction_A",
    "Hit_Interrupt_Recent_Reaction_B",
    "Hit_Interrupt_Recovery_Pending",
    "Hit_Interrupt_Enraged",
    "Hit_Interrupt_Protected_Attack",
)


def _effect_row(data: dict[str, Any], path: str) -> dict[str, Any]:
    effects = data.get("ApplicationEffects") or {}
    return {
        "duration_seconds": float(data["Duration"]),
        "overlap": str(data["OverlapBehavior"]),
        "speed_multiplier": float(effects.get("HorizontalSpeedMultiplier", 1.0)),
        "infinite": bool(data.get("Infinite", False)),
        "debuff": bool(data.get("Debuff", False)),
        "source": path,
    }


def extract(archive: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in EFFECTS:
        path = f"{STATUS_PREFIX}{name}.json"
        rows[f"effect:{name}"] = _effect_row(archive.read_json(path), path)
    for key, row in weapon_family.extract(archive).items():
        if key.startswith("interaction:") and row.get("hit_interrupt"):
            name = key.split(":", 1)[1]
            rows[f"attack:{name}"] = {"interaction": name, "parent_chain": list(row.get("parent_chain") or []),
                                      "source": row.get("source")}
    return rows
