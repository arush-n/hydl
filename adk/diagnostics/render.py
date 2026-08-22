"""Look at what the agent sees: image arrays and text views.

Two kinds of output, kept apart on purpose:

* **Array producers** (:func:`top_down_rgb`, :func:`top_down_occupancy`) build
  an image-shaped array with a scatter.  They are traced and sync-free, so they
  can run inside a rollout and be recorded like any other tensor, then written
  out later by whatever plotting or video tool you prefer.  The ADK deliberately
  takes no plotting dependency.
* **Text renderers** (``render_*``) return strings for a terminal or a log.
  They synchronize with the device and say so; they are for debugging, not for
  the hot loop.

Nothing here is specific to a policy or algorithm -- these read the published
observation and produce plain arrays or plain text.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from adk.policy.observations import TokenSet


#: Dark to light.  Used by the ASCII renderers.
DENSITY_RAMP = " .:-=+*#%@"


def _bin_positions(positions: jax.Array, resolution: int) -> jax.Array:
    """Map [-1, 1] coordinates onto [0, resolution - 1] integer cells."""

    cells = jnp.floor((positions + 1.0) * 0.5 * resolution).astype(jnp.int32)
    return jnp.clip(cells, 0, resolution - 1)


def top_down_occupancy(
    positions: jax.Array,
    mask: jax.Array,
    *,
    resolution: int = 32,
    axes: tuple[int, int] = (0, 2),
) -> jax.Array:
    """Scatter tokens into a ``(batch, resolution, resolution)`` count map.

    ``axes`` picks which two of the three relative coordinates form the plane;
    the default ``(0, 2)`` is the usual top-down X/Z view.  Traced and
    allocation-bounded: cost is one scatter over the token axis.
    """

    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError("resolution must be an integer")
    if resolution < 1:
        raise ValueError("resolution must be positive")
    if positions.ndim != 3:
        raise ValueError(
            f"positions must be (batch, slots, 3), got {positions.shape}"
        )

    planar = jnp.stack(
        [positions[..., axes[0]], positions[..., axes[1]]], axis=-1
    )
    cells = _bin_positions(planar, resolution)
    weights = mask.astype(jnp.float32)

    def scatter(cell, weight):
        grid = jnp.zeros((resolution, resolution), dtype=jnp.float32)
        return grid.at[cell[:, 0], cell[:, 1]].add(weight)

    return jax.vmap(scatter)(cells, weights)


def top_down_rgb(
    positions: jax.Array,
    rgb: jax.Array,
    mask: jax.Array,
    *,
    resolution: int = 32,
    axes: tuple[int, int] = (0, 2),
) -> jax.Array:
    """Scatter per-token colour into a ``(batch, resolution, resolution, 3)`` image.

    Colour is averaged over the tokens landing in each cell, so a crowded cell
    does not saturate.  Empty cells stay black.  Pair it with
    :func:`adk.policy.surfaces.light_rgb` to see the lighting the agent has,
    or with any other three normalized channels.
    """

    if rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have three channels, got {rgb.shape}")

    counts = top_down_occupancy(
        positions, mask, resolution=resolution, axes=axes
    )
    planar = jnp.stack(
        [positions[..., axes[0]], positions[..., axes[1]]], axis=-1
    )
    cells = _bin_positions(planar, resolution)
    weighted = jnp.where(mask[..., None], rgb, 0.0)

    def scatter(cell, colour):
        grid = jnp.zeros((resolution, resolution, 3), dtype=jnp.float32)
        return grid.at[cell[:, 0], cell[:, 1]].add(colour)

    totals = jax.vmap(scatter)(cells, weighted)
    return totals / jnp.maximum(counts[..., None], 1.0)


def token_positions(tokens: TokenSet, columns: Sequence[int]) -> jax.Array:
    """Pull the three relative-position columns out of a token set.

    A thin O(1) view, so callers do not have to remember which columns carry
    geometry versus block-candidate positions.
    """

    if len(columns) != 3:
        raise ValueError("expected three position columns")
    return tokens.values[..., list(columns)]


# -- text ---------------------------------------------------------------------


def render_ascii(plane: Any, *, ramp: str = DENSITY_RAMP) -> str:
    """Render one 2-D array as an ASCII density map.

    **Host-side**: transfers the array.  Values are normalized against the
    plane's own maximum, so an all-zero plane renders blank rather than
    dividing by zero.
    """

    values = jnp.asarray(plane)
    if values.ndim != 2:
        raise ValueError(f"render_ascii needs a 2-D array, got {values.shape}")
    peak = float(jnp.max(values))
    if peak <= 0.0:
        return "\n".join(" " * values.shape[1] for _ in range(values.shape[0]))
    scaled = (values / peak) * (len(ramp) - 1)
    rows = []
    for row in scaled.tolist():
        rows.append("".join(ramp[int(round(cell))] for cell in row))
    return "\n".join(rows)


def render_availability(report: Mapping[str, Any]) -> str:
    """Tabulate a :func:`describe_availability` report.

    **Host-side.**  The report itself already synchronized, so this adds no
    further cost.
    """

    if not report:
        return "(no surfaces reported)"
    width = max(len(name) for name in report)
    lines = []
    for name, status in report.items():
        mark = "yes" if getattr(status, "available", False) else "no "
        occupied = getattr(status, "occupied", 0)
        detail = getattr(status, "detail", "") or ""
        note = f"  -- {detail}" if occupied == 0 and detail else ""
        lines.append(
            f"{name:<{width}}  available={mark}  occupied={occupied:<6}{note}"
        )
    return "\n".join(lines)


def render_fields(values: Any, group: str, *, row: int = 0) -> str:
    """Tabulate one group's named columns for a single environment.

    **Host-side.**  Intended for "what does this observation actually say"
    moments; it transfers the group it is given, not the whole observation.
    """

    from adk.policy.fields import field_names

    array = jnp.asarray(getattr(values, "values", values))
    names = field_names(group)
    if array.shape[-1] != len(names):
        raise ValueError(
            f"{group!r} publishes {len(names)} columns but the array has "
            f"{array.shape[-1]}"
        )
    selected = array[row]
    while selected.ndim > 1:
        selected = selected[0]
    width = max(len(name) for name in names)
    return "\n".join(
        f"{name:<{width}}  {float(value): .5f}"
        for name, value in zip(names, selected.tolist())
    )


def render_summary(summary: Mapping[str, Any]) -> str:
    """Tabulate a :func:`adk.diagnostics.trace.summarize` result.

    **Host-side.**  Sorted so successive runs line up for eyeball diffing.
    """

    if not summary:
        return "(empty summary)"
    width = max(len(name) for name in summary)
    return "\n".join(
        f"{name:<{width}}  {float(jnp.asarray(value)): .6f}"
        for name, value in sorted(summary.items())
    )


__all__ = [
    "DENSITY_RAMP",
    "render_ascii",
    "render_availability",
    "render_fields",
    "render_summary",
    "token_positions",
    "top_down_occupancy",
    "top_down_rgb",
]
