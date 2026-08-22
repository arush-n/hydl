"""Build explicit action-factor vectors with every head pinned but one.

The action transport is ``int32[B, len(HEAD_SPANS)]`` -- one option index per
head, in ``HEAD_SPANS`` order.  :func:`adk.probes.repeat` emits such a vector
unchanged every step, which is the only way to *force* an action; ``prefer()``
biases logits and the mask still wins.
"""

from __future__ import annotations

import numpy as np

from adk.probes.policies import HEAD_SPANS

#: Head order is the action transport's column order.
HEAD_ORDER: tuple[str, ...] = tuple(HEAD_SPANS)

_HEAD_INDEX = {name: index for index, name in enumerate(HEAD_ORDER)}

#: Heads whose neutral option is the *centre* of a signed delta ladder rather
#: than index 0.  Measured: yaw centre (index 4 of 9) produces 0.00 deg drift
#: over 200 ticks, so the centre really is "no change".
_CENTRED_HEADS = frozenset({"yaw_delta_bins", "pitch_delta_bins"})


def _neutral_option(head: str) -> int:
    """The option that makes ``head`` do nothing.

    ``none``/``off`` heads neutralise at 0.  Signed delta ladders neutralise at
    their midpoint.
    """

    size = HEAD_SPANS[head][1]
    return size // 2 if head in _CENTRED_HEADS else 0


#: Per-head do-nothing option.  Everything here was confirmed inert by
#: measurement, not assumed: forcing all of it yields exactly 0.00000 total
#: displacement and 0.00 yaw drift.
NEUTRAL_OPTION: dict[str, int] = {head: _neutral_option(head) for head in HEAD_ORDER}


def neutral_factors(batch: int) -> np.ndarray:
    """An ``int32[batch, heads]`` action in which every head does nothing."""

    if batch < 1:
        raise ValueError("batch must be >= 1")
    row = np.array(
        [NEUTRAL_OPTION[head] for head in HEAD_ORDER],
        dtype=np.int32,
    )
    return np.broadcast_to(row, (batch, len(HEAD_ORDER))).copy()


def override(batch: int, **heads: int) -> np.ndarray:
    """Neutral everywhere, except the heads named as keyword arguments.

    >>> override(2, jump_off_on=1, locomotion_gait_compass=3).shape
    (2, 10)

    Raises on an unknown head or an out-of-range option, because a silent
    typo here produces a plausible-looking measurement of the wrong thing.
    """

    factors = neutral_factors(batch)
    for head, option in heads.items():
        if head not in _HEAD_INDEX:
            raise KeyError(
                f"unknown action head {head!r}; known: {sorted(HEAD_ORDER)}"
            )
        size = HEAD_SPANS[head][1]
        if not 0 <= int(option) < size:
            raise ValueError(
                f"option {option} out of range for {head!r} (size {size})"
            )
        factors[:, _HEAD_INDEX[head]] = int(option)
    return factors


def isolate(head: str, option: int, *, batch: int) -> np.ndarray:
    """Force one head to one option; pin every other head neutral.

    This is the measurement that the cross-head constraints demand.  Anything
    less controlled cannot tell a *suppressed* action from a *wrong* one -- the
    two are indistinguishable in aggregate, which is how two runs of a compass
    probe reported scattered headings for a mapping that is exactly correct.
    """

    return override(batch, **{head: option})


def head_column(head: str) -> int:
    """Column of ``head`` in the action-factor vector."""

    try:
        return _HEAD_INDEX[head]
    except KeyError as error:
        raise KeyError(
            f"unknown action head {head!r}; known: {sorted(HEAD_ORDER)}"
        ) from error
