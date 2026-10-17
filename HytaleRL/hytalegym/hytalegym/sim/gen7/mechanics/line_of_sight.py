"""0.7.2 line-of-sight rule. Switch ``Constructs.line_of_sight``; P2.

Gen 5 ("non_transparent_blocks"): any block whose Opacity is not Transparent
stops sight across its whole cell (``jax/world/geometry`` occluders). Gen 7
("detail_hitbox"): a non-transparent block occludes only where the ray hits one
of its detail hitboxes (0.7.2 ``PositionCache`` uses
``BlockBoundingBoxes$RotatedVariantBoxes.getDetailBoxes``). Inputs (gen-7
tables): ``blocks/block:*`` ``opacity``, ``hitbox_type``; ``blocks/hitbox:*``
``boxes``; rotation from the captured Region cell. Differential: an occlusion
fixture with slabs, stairs and fences, every ray agreeing with native
(FIDELITY rows 2a, 2b); the catch rule and the ``hidden`` opening follow.
"""

from __future__ import annotations

from typing import Any

SWITCH = "line_of_sight"
IMPLEMENTED = False


def cell_occludes(cell_flags: Any, detail_boxes: Any, ray_origin: Any, ray_direction: Any) -> Any:
    """Per cell: does the ray intersect an occluding detail box."""

    raise NotImplementedError("line-of-sight construct: design only (mechanics_072.md)")
