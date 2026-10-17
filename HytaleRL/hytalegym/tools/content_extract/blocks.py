"""Block properties the simulator reads: solidity, LOS opacity, movement.

* ``block:<id>``: every item with a ``BlockType`` (item ``Parent`` chain merged
  recursively, the rule of ``worldgen/surrogate/semantics.py``): ``Material``,
  ``Opacity`` (LOS: non-``Transparent`` blocks sight in 0.5.x), ``HitboxType``,
  ``DrawType``, ``IsDoor``, ``MovementSettings`` (``IsClimbable``, speed,
  jump and drag modifiers, e.g. spider webs), collision interactions, state
  names. Absent fields are omitted; engine defaults (Material ``Empty``,
  Opacity ``Solid``, HitboxType ``Full``, IsDoor false) are applied by the
  content layer, not here.
* ``hitbox:<id>``: ``Server/Item/Block/Hitboxes`` detail boxes
  ``[minX, minY, minZ, maxX, maxY, maxZ]`` per ``HitboxType``.
* ``fluid:<id>`` and ``fluidfx:<id>``: ``Server/Item/Block/Fluids`` and
  ``Server/Item/Block/FluidFX`` (swim up/down, sink, speed multipliers).
"""

from __future__ import annotations

from typing import Any

from .archive import GameArchive
from .common import gameplay, resolve_parented

FAMILY = "blocks"
VERSION = 1
ITEM_PREFIX = "Server/Item/Items/"
FLUID_PREFIX = "Server/Item/Block/Fluids/"
FLUIDFX_PREFIX = "Server/Item/Block/FluidFX/"
HITBOX_PREFIX = "Server/Item/Block/Hitboxes/"


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def extract(archive: GameArchive) -> dict[str, Any]:
    items = archive.index(ITEM_PREFIX)
    rows: dict[str, Any] = {}
    for asset_id in sorted(items):
        raw = archive.read_json(items[asset_id])
        if not isinstance(raw, dict):
            continue
        if "BlockType" not in raw and not isinstance(raw.get("Parent"), str):
            continue
        item, chain = resolve_parented(asset_id, items, archive.read_json)
        block = item.get("BlockType")
        if not isinstance(block, dict):
            continue
        movement = block.get("MovementSettings")
        interactions = block.get("Interactions")
        collision = interactions.get("Collision") if isinstance(interactions, dict) else None
        state = block.get("State")
        definitions = state.get("Definitions") if isinstance(state, dict) else None
        rows[f"block:{asset_id}"] = _compact(
            {
                "source": items[asset_id],
                "parent_chain": list(chain) or None,
                "material": block.get("Material"),
                "opacity": block.get("Opacity"),
                "hitbox_type": block.get("HitboxType"),
                "draw_type": block.get("DrawType"),
                "is_door": block.get("IsDoor"),
                "movement_settings": movement if isinstance(movement, dict) else None,
                "climbable": True
                if isinstance(movement, dict) and movement.get("IsClimbable") is True
                else None,
                "damage_to_entities": block.get("DamageToEntities"),
                "collision_interactions": gameplay(collision) if collision is not None else None,
                "states": sorted(definitions) if isinstance(definitions, dict) else None,
            }
        )
    # Detail hitboxes per HitboxType (the 0.7.2 line-of-sight rule tests a ray
    # against these instead of the whole cell).
    hitboxes = archive.index(HITBOX_PREFIX)
    for stem in sorted(hitboxes):
        document = archive.read_json(hitboxes[stem])
        boxes = []
        for box in document.get("Boxes") or []:
            low, high = box.get("Min", {}), box.get("Max", {})
            boxes.append([float(low.get(a, 0.0)) for a in "XYZ"] + [float(high.get(a, 0.0)) for a in "XYZ"])
        rows[f"hitbox:{stem}"] = {"source": hitboxes[stem], "boxes": boxes}
    for prefix, kind in ((FLUID_PREFIX, "fluid"), (FLUIDFX_PREFIX, "fluidfx")):
        index = archive.index(prefix)
        for stem in sorted(index):
            document, chain = resolve_parented(stem, index, archive.read_json)
            if kind == "fluidfx":
                document = {"MovementSettings": document.get("MovementSettings")}
            rows[f"{kind}:{stem}"] = _compact(
                {
                    "source": index[stem],
                    "parent_chain": list(chain) or None,
                    "document": gameplay(document),
                }
            )
    return rows
