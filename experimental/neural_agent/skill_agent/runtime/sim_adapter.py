"""The head adapter between the combiner and a simulator that implements only some of the gen-7 heads.

``skill_agent/README.md`` section 3 is the contract. The combiner emits one choice per head of the layout (21 on
gen 7). The simulator publishes a per-choice legality mask only for the heads it implements, so the adapter applies
one rule to every agent's choices::

    implemented head               -> the choice passes through unchanged
    absent head, neutral choice    -> the neutral choice (``surface.neutral_factors``); not a drop
    absent head, any other choice  -> the neutral choice, and ``dropped[head]`` counts it

The count is the point: a skill that depends on a missing head shows up as drops, not as a silently inert skill.
``legal_scopes`` is ``protocol.legal_scopes``, so a present head whose only legal choice is neutral keeps its scope
legal (a cooldown, not a missing mechanic). ``adapt`` is the reference for one agent; ``adapt_batch`` applies the same
rule to a (B, H) array and counts over every row.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experimental.neural_agent.skill_agent.contract import protocol
from experimental.neural_agent.skill_agent.contract.surface import head_names, head_sizes, neutral_factors


def _check_masks(head_masks: Mapping[str, Sequence[bool]], layout: str) -> None:
    """Raise ValueError unless every head is a head of ``layout`` and its mask has the head's size."""

    sizes = dict(zip(head_names(layout), head_sizes(layout)))
    for head, mask in head_masks.items():
        if head not in sizes:
            raise ValueError(f"the mask names {head!r}, which is not a head of layout {layout!r}")
        if len(mask) != sizes[head]:
            raise ValueError(f"the mask of {head!r} has {len(mask)} choices; the head has {sizes[head]}")


class HeadAdapter:
    """Maps the combiner's choices onto the heads a simulator implements (the module docstring gives the rule).

    Args:
        layout: The combiner's head layout, ``"gen5"`` or ``"gen7"``.
        implemented: The names of the heads the simulator implements, each a head of ``layout``.

    Attributes:
        implemented: The implemented heads, in layout order.
        absent: The other heads of ``layout``, in layout order.
        dropped: Absent head -> the non-neutral choices the combiner made on it, over every agent adapted so far.
    """

    def __init__(self, layout: str = "gen7", implemented: Iterable[str] = ()) -> None:
        names = head_names(layout)
        present = set(implemented)
        unknown = sorted(present - set(names))
        if unknown:
            raise ValueError(f"{unknown} are not heads of layout {layout!r}")
        self.layout = layout
        self.implemented: tuple[str, ...] = tuple(name for name in names if name in present)
        self.absent: tuple[str, ...] = tuple(name for name in names if name not in present)
        self.dropped: Counter[str] = Counter()
        self._names = names
        self._neutral = neutral_factors(layout)
        self._absent_positions = tuple(position for position, name in enumerate(names) if name not in present)

    @classmethod
    def from_mask(cls, head_masks: Mapping[str, Sequence[bool]], layout: str = "gen7") -> HeadAdapter:
        """Build the adapter from the simulator's mask: the heads it publishes are the implemented ones.

        Raises:
            ValueError: If a head is not a head of ``layout``, or its mask does not have the head's size.
        """

        _check_masks(head_masks, layout)
        return cls(layout, head_masks.keys())

    @staticmethod
    def legal_scopes(head_masks: Mapping[str, Sequence[bool]]) -> frozenset[str]:
        """The scopes legal under the simulator's mask, by ``protocol.legal_scopes`` (after the gen-7 mask check).

        Raises:
            ValueError: If a head is not a gen-7 head, or its mask does not have the head's size.
        """

        _check_masks(head_masks, "gen7")
        return protocol.legal_scopes(head_masks)

    def adapt(self, factors: Sequence[int]) -> tuple[int, ...]:
        """One agent's choices in layout order -> the choices the simulator takes.

        Raises:
            ValueError: If ``factors`` does not have one choice per head of the layout.
        """

        if len(factors) != len(self._neutral):
            raise ValueError(f"expected {len(self._neutral)} choices for layout {self.layout!r}, got {len(factors)}")
        out = [int(choice) for choice in factors]
        for position in self._absent_positions:
            if out[position] != self._neutral[position]:
                self.dropped[self._names[position]] += 1
            out[position] = self._neutral[position]
        return tuple(out)

    def adapt_batch(self, factors: np.ndarray) -> np.ndarray:
        """``adapt`` for B agents at once: ``factors`` is (B, H) in layout order; drops count over every row.

        Raises:
            ValueError: If ``factors`` is not (B, H) for the layout's H heads.
        """

        out = np.array(factors, copy=True)
        heads = len(self._neutral)
        if out.ndim != 2 or out.shape[1] != heads:
            raise ValueError(f"expected choices of shape (B, {heads}) for layout {self.layout!r}, got {out.shape}")
        absent = np.asarray(self._absent_positions, dtype=np.intp)
        neutral = np.asarray(self._neutral, dtype=np.intp)[absent]
        drops = np.count_nonzero(out[:, absent] != neutral, axis=0)
        out[:, absent] = neutral
        for position, count in zip(self._absent_positions, drops.tolist()):
            if count:
                self.dropped[self._names[position]] += count
        return out

    def report(self) -> dict[str, Any]:
        """The adapter's state, JSON-ready: implemented and absent heads, and the drops per absent head (0 if none)."""

        return {"implemented": list(self.implemented), "absent": list(self.absent),
                "dropped": {head: self.dropped[head] for head in self.absent}}


__all__ = ["HeadAdapter"]
