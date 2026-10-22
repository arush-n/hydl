"""A trained trunk carried to another token layout by feature name, without retraining (``docs/VIRTUAL_HUMAN.md``
section 14, rules 1, 2 and 5).

A token type's features enter the trunk through one linear layer (``features @ kernel + bias``, the first layer of
``encoder._token_mlp``), and the conditioning vector through one more per type (``EncoderParams.condition``). Nothing
else in the trunk depends on a feature's position. So migrating a type rewrites the rows of those kernels by name
(``contract/variables.transfer``):
- a kept feature keeps its row, and a new feature gets zeros, so a trunk that only grew computes exactly what it did:
  the new inputs start out ignored (the input-side case of Net2Net's function-preserving growth, Chen, Goodfellow and
  Shlens, ICLR 2016);
- a feature merged into a new one by a map ``new = sum_i w_i old_i`` gives it the sum of its sources' rows over
  ``sum_i w_i``: the trunk is unchanged for an input whose merged features were equal, and changed otherwise, by an
  amount the update measures before and after (rule 5);
- a dropped feature's row goes: the trunk computes what it did with that feature at 0 (a trait's population mean).
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.variables import transfer
from experimental.neural_agent.skill_agent.model.encoder import DenseParams, EncoderConfig, EncoderParams


def _rows(back: np.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(back @ np.asarray(kernel, np.float64), jnp.float32)


def migrate_trunk(params: EncoderParams, config: EncoderConfig, old: Mapping[str, Sequence[str]],
                  new: Mapping[str, Sequence[str]], *,
                  maps: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
                  dropped: Mapping[str, Sequence[str]] | None = None) -> tuple[EncoderParams, EncoderConfig]:
    """``params`` read under the layout ``old`` ({type: feature names}, ``encoder.token_layout``), carried to ``new``.

    Args:
        params: The trunk's parameters.
        config: Its sizes; ``old`` must give each of its types, in its order, with its width.
        old: The layout it was trained on.
        new: The layout to carry it to: the same types in the same order.
        maps: Per type, {new feature: {old feature: weight}} for merged features.
        dropped: Per type, the old features dropped on purpose.

    Returns:
        The migrated parameters and their configuration (the new widths; every other size unchanged).

    Raises:
        ValueError: If the types differ (adding or removing a token type is not a column migration), a width does not
            match, or a feature disappears without a map or a drop.
    """

    maps, dropped = maps or {}, dropped or {}
    names = [name for name, _ in config.token_types]
    if list(old) != names or list(new) != names:
        raise ValueError(f"old and new must give the trunk's types {names} in order, got {list(old)} and {list(new)}")
    for name, width in config.token_types:
        if len(old[name]) != width:
            raise ValueError(f"{name}: the old layout has {len(old[name])} features, the trunk reads {width}")
    back = {name: transfer(old[name], new[name], maps.get(name), dropped.get(name, ()))[1] for name in names}
    types = {}
    for name in names:
        embedding = params.types[name].embedding
        moved = embedding.input._replace(kernel=_rows(back[name], embedding.input.kernel))
        types[name] = params.types[name]._replace(embedding=embedding._replace(input=moved))
    condition = params.condition
    if condition is not None:  # the conditioning vector is the condition types' features, concatenated
        widths = dict(config.token_types)
        bounds, start = {}, 0
        for name in config.condition_types:
            bounds[name] = (start, start + widths[name])
            start += widths[name]

        def carry(kernel: jnp.ndarray) -> jnp.ndarray:
            return jnp.concatenate([_rows(back[name], kernel[slice(*bounds[name])]) for name in config.condition_types])

        condition = {name: DenseParams(carry(dense.kernel), dense.bias) for name, dense in condition.items()}
    token_types = tuple((name, len(new[name])) for name in names)
    return params._replace(types=types, condition=condition), dataclasses.replace(config, token_types=token_types)


__all__ = ["migrate_trunk"]
