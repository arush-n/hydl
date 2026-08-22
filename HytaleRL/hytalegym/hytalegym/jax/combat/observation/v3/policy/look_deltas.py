"""Reusable multiscale look-delta codebook for discrete policy heads."""

from __future__ import annotations

import math
import operator
import struct

import jax.numpy as jnp


LOOK_DELTA_FINE_DEGREES = 1.0


def _wire_float32(value: float) -> float:
    """Return the exact Python value represented by one wire float32."""

    return struct.unpack("!f", struct.pack("!f", value))[0]


def look_delta_values(
    size: int,
    *,
    maximum_degrees: float = 45.0,
    fine_degrees: float = LOOK_DELTA_FINE_DEGREES,
) -> tuple[float, ...]:
    """Return a symmetric coarse-to-fine delta codebook.

    Linear bins make small view corrections unreachable when a low-cardinality
    head also needs a large turn range. Geometric positive magnitudes retain
    both: the outer bins provide the authored maximum and the inner bins give
    one-degree corrections. The definition depends only on head cardinality
    and look limits, never on a weapon or target fixture.
    """

    try:
        count = operator.index(size)
    except TypeError as exc:
        raise ValueError("look delta head size must be an odd integer") from exc
    maximum = float(maximum_degrees)
    fine = float(fine_degrees)
    if count < 3 or count % 2 == 0:
        raise ValueError("look delta head size must be odd and at least 3")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_degrees must be finite and positive")
    if not math.isfinite(fine) or fine <= 0.0 or fine > maximum:
        raise ValueError(
            "fine_degrees must be finite, positive, and at most maximum"
        )
    positive_count = (count - 1) // 2
    if positive_count > 1 and fine >= maximum:
        raise ValueError(
            "fine_degrees must be less than maximum_degrees for heads wider "
            "than 3"
        )
    if positive_count == 1:
        positive = (maximum,)
    else:
        ratio = (maximum / fine) ** (1.0 / (positive_count - 1))
        generated = [fine * ratio**index for index in range(positive_count)]
        generated[0] = fine
        generated[-1] = maximum
        positive = tuple(generated)
    positive = tuple(_wire_float32(value) for value in positive)
    if any(left >= right for left, right in zip(positive, positive[1:])):
        raise ValueError(
            "look delta magnitudes must remain distinct after float32 rounding"
        )
    return tuple(-value for value in reversed(positive)) + (0.0,) + positive


def decode_look_delta(
    choice,
    size: int,
    *,
    maximum_degrees: float = 45.0,
    fine_degrees: float = LOOK_DELTA_FINE_DEGREES,
):
    """Decode integer choices through the shared multiscale codebook."""

    table = jnp.asarray(
        look_delta_values(
            size,
            maximum_degrees=maximum_degrees,
            fine_degrees=fine_degrees,
        ),
        dtype=jnp.float32,
    )
    bounded = jnp.clip(jnp.asarray(choice, dtype=jnp.int32), 0, size - 1)
    return table[bounded]


def encode_look_delta(
    value,
    size: int,
    *,
    maximum_degrees: float = 45.0,
    fine_degrees: float = LOOK_DELTA_FINE_DEGREES,
):
    """Select the nearest representable look delta, ties to lower index."""

    table = jnp.asarray(
        look_delta_values(
            size,
            maximum_degrees=maximum_degrees,
            fine_degrees=fine_degrees,
        ),
        dtype=jnp.float32,
    )
    values = jnp.asarray(value, dtype=jnp.float32)
    return jnp.argmin(
        jnp.abs(values[..., None] - table),
        axis=-1,
    ).astype(jnp.int32)


__all__ = [
    "LOOK_DELTA_FINE_DEGREES",
    "decode_look_delta",
    "encode_look_delta",
    "look_delta_values",
]
