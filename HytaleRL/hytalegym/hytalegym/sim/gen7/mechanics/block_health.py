"""Block health reset on write (0.7.2 rule). Switch ``block_health_reset_on_write``.

Engine rule, no table input: 0.7.2 drops a cell's health entry once health
leaves (0, 1) and on any block write (PORT-0.7.2.md). Differential: the
mutable-block native compare re-baselined on 0.7.2 (P5).
"""

from __future__ import annotations

from typing import Any

SWITCH = "block_health_reset_on_write"
IMPLEMENTED = False


def on_block_write(state: Any, cell: Any) -> Any:
    raise NotImplementedError("block health construct: design only (mechanics_072.md)")
