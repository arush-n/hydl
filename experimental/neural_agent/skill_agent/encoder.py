"""The token-set encoder: one trunk that reads a body's token sets (``BUILD.md`` phase B, deliverable B1).

A body reads its agent's token sets: any number of tokens of each type, a token being a feature row whose last
feature is ``marked`` (``surface.py``). The trunk is pure JAX: parameters are NamedTuples of arrays, and
``apply_encoder`` is a function of them. For a token of type ``t`` with features ``f`` (width ``F_t``), validity
``m`` and model width ``W``::

    h = MLP_t(f) + type_t + [f_marked > 0.5] * marked       MLP_t: Dense(F_t -> W), LayerNorm, GELU, Dense(W -> W)

The tokens of all types form one sequence ``X`` with validity ``M`` (in ``TRUNK_TYPES`` order), and ``layers``
pre-LayerNorm blocks follow::

    X <- X + MHA(LN(X); M)          q, k, v and o projections; a key with M False gets logit -inf
    X <- X + MLP(LN(X))             Dense(W -> r W), GELU, Dense(r W -> W)

A query with no valid key (an agent with no valid token) gets 0 from attention. After a final LayerNorm, the
readouts over the valid tokens are::

    mean    = sum(M X) / max(1, sum M)               max = max over valid X (0 if none)
    mark    = masked mean over valid, marked X       (0 if none)
    self    = X at the first valid token of type "self"   (0 if none)
    context = Dense(4 W -> O)(concat(mean, max, mark, self))

and ``tokens[t]`` is ``X`` restricted to the slots of type ``t``: the per-token embeddings that a pointer head can
index (a block-candidate body picks one candidate by its slot).

Masking is exact. A padded slot's features are zeroed before the first layer, so no output depends on what a padded
slot holds, and no padded key is ever attended to. The parameters depend on the widths only, never on token counts,
so a change of capacity is a change of packing, not of shape.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.surface import TOKEN_TYPES as SURFACE_TOKEN_TYPES

TRUNK_TYPES: tuple[tuple[str, type], ...] = tuple(
    (name, SURFACE_TOKEN_TYPES[name])
    for name in ("entity", "block_candidate", "inventory_slot", "geometry", "self", "kit")
)
"""The token types the trunk reads, in its sequence order: (name in ``surface.TOKEN_TYPES``, record class)."""

assert all(cls.FEATURES[-1] == "marked" for _, cls in TRUNK_TYPES), "each token's last feature must be 'marked'"

_LAYER_NORM_EPSILON: float = 1e-5


def _default_token_types() -> tuple[tuple[str, int], ...]:
    return tuple((name, len(cls.FEATURES)) for name, cls in TRUNK_TYPES)


def _positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class EncoderConfig:
    """The trunk's sizes. The parameters depend on these, never on token counts.

    Attributes:
        token_types: (type name, feature width) in sequence order. The default is ``TRUNK_TYPES`` with
            ``len(cls.FEATURES)`` widths.
        width: The model width ``W``.
        heads: Attention heads; ``width`` must be a multiple of it.
        layers: Pre-LayerNorm transformer blocks.
        mlp_ratio: The block MLP's hidden width is ``mlp_ratio * width``.
        output: The width of the context vector.

    Raises:
        ValueError: If a size is not a positive integer, ``width`` is not a multiple of ``heads``, a feature
            width is not positive, or two token types share a name.
    """

    token_types: tuple[tuple[str, int], ...] = _default_token_types()
    width: int = 64
    heads: int = 4
    layers: int = 2
    mlp_ratio: int = 2
    output: int = 128

    def __post_init__(self) -> None:
        for name in ("width", "heads", "layers", "mlp_ratio", "output"):
            _positive(getattr(self, name), name)
        for name, features in self.token_types:
            _positive(features, f"feature width of {name!r}")
        names = [name for name, _ in self.token_types]
        if len(set(names)) != len(names):
            raise ValueError(f"token type names must be unique, got {names}")
        if self.width % self.heads:
            raise ValueError(f"width {self.width} must be a multiple of heads {self.heads}")


class DenseParams(NamedTuple):
    kernel: jax.Array
    bias: jax.Array


class LayerNormParams(NamedTuple):
    scale: jax.Array
    offset: jax.Array


class TokenMLPParams(NamedTuple):
    """Dense(F -> W), LayerNorm, GELU, Dense(W -> W): shared by the tokens of one type."""

    input: DenseParams
    norm: LayerNormParams
    output: DenseParams


class TypeParams(NamedTuple):
    """One token type: its embedding MLP and its learned type vector [W]."""

    embedding: TokenMLPParams
    type_vector: jax.Array


class AttentionParams(NamedTuple):
    query: DenseParams
    key: DenseParams
    value: DenseParams
    output: DenseParams


class BlockParams(NamedTuple):
    attention_norm: LayerNormParams
    attention: AttentionParams
    mlp_norm: LayerNormParams
    mlp_in: DenseParams
    mlp_out: DenseParams


class EncoderParams(NamedTuple):
    types: dict[str, TypeParams]
    marked_vector: jax.Array
    blocks: tuple[BlockParams, ...]
    norm: LayerNormParams
    readout: DenseParams


class EncoderOutput(NamedTuple):
    """``context`` is float32[B, output]; ``tokens`` maps each configured type to float32[B, N_t, width]."""

    context: jax.Array
    tokens: dict[str, jax.Array]


def _init_normal(key: jax.Array, shape: tuple[int, ...], fan_in: int) -> jax.Array:
    return jax.random.normal(key, shape, dtype=jnp.float32) / math.sqrt(fan_in)


def _init_dense(key: jax.Array, fan_in: int, fan_out: int) -> DenseParams:
    return DenseParams(_init_normal(key, (fan_in, fan_out), fan_in), jnp.zeros((fan_out,), dtype=jnp.float32))


def _init_norm(size: int) -> LayerNormParams:
    return LayerNormParams(jnp.ones((size,), dtype=jnp.float32), jnp.zeros((size,), dtype=jnp.float32))


def init_encoder(key: jax.Array, config: EncoderConfig) -> EncoderParams:
    """Initialise the parameters: kernels and vectors ~ N(0, 1/fan_in), zero biases, unit LayerNorm scales.

    Args:
        key: A JAX PRNG key.
        config: The trunk's sizes.

    Returns:
        The parameters. Their shapes depend on ``config`` and on no token count.
    """

    width, hidden = config.width, config.mlp_ratio * config.width
    # One key per dense kernel and per type or marked vector, consumed in the order below.
    keys = iter(jax.random.split(key, 3 * len(config.token_types) + 1 + 6 * config.layers + 1))
    types = {}
    for name, features in config.token_types:
        embedding = TokenMLPParams(
            input=_init_dense(next(keys), features, width),
            norm=_init_norm(width),
            output=_init_dense(next(keys), width, width),
        )
        types[name] = TypeParams(embedding=embedding, type_vector=_init_normal(next(keys), (width,), width))
    marked_vector = _init_normal(next(keys), (width,), width)
    blocks = []
    for _ in range(config.layers):
        blocks.append(
            BlockParams(
                attention_norm=_init_norm(width),
                attention=AttentionParams(
                    query=_init_dense(next(keys), width, width),
                    key=_init_dense(next(keys), width, width),
                    value=_init_dense(next(keys), width, width),
                    output=_init_dense(next(keys), width, width),
                ),
                mlp_norm=_init_norm(width),
                mlp_in=_init_dense(next(keys), width, hidden),
                mlp_out=_init_dense(next(keys), hidden, width),
            )
        )
    return EncoderParams(
        types=types,
        marked_vector=marked_vector,
        blocks=tuple(blocks),
        norm=_init_norm(width),
        readout=_init_dense(next(keys), 4 * width, config.output),
    )


def _linear(params: DenseParams, inputs: jax.Array) -> jax.Array:
    return inputs @ params.kernel + params.bias


def _layer_norm(params: LayerNormParams, inputs: jax.Array) -> jax.Array:
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
    return (inputs - mean) * jax.lax.rsqrt(variance + _LAYER_NORM_EPSILON) * params.scale + params.offset


def _token_mlp(params: TokenMLPParams, features: jax.Array) -> jax.Array:
    hidden = jax.nn.gelu(_layer_norm(params.norm, _linear(params.input, features)))
    return _linear(params.output, hidden)


def _attention(params: AttentionParams, inputs: jax.Array, valid: jax.Array, heads: int) -> jax.Array:
    """Masked multi-head self-attention over the token axis; a batch row with no valid token gives 0."""

    batch, tokens, width = inputs.shape
    head_width = width // heads

    def split_heads(projected: jax.Array) -> jax.Array:
        return projected.reshape(batch, tokens, heads, head_width).transpose(0, 2, 1, 3)

    query = split_heads(_linear(params.query, inputs))
    key = split_heads(_linear(params.key, inputs))
    value = split_heads(_linear(params.value, inputs))
    logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(head_width)
    any_valid = jnp.any(valid, axis=1)
    logits = jnp.where(valid[:, None, None, :], logits, -jnp.inf)
    # A row with no valid key is all -inf. Zero its logits so the softmax stays finite; its output is dropped below.
    logits = jnp.where(any_valid[:, None, None, None], logits, 0.0)
    weights = jax.nn.softmax(logits, axis=-1)
    mixed = jnp.einsum("bhqk,bhkd->bhqd", weights, value).transpose(0, 2, 1, 3).reshape(batch, tokens, width)
    return jnp.where(any_valid[:, None, None], _linear(params.output, mixed), 0.0)


def _block(params: BlockParams, inputs: jax.Array, valid: jax.Array, heads: int) -> jax.Array:
    inputs = inputs + _attention(params.attention, _layer_norm(params.attention_norm, inputs), valid, heads)
    hidden = jax.nn.gelu(_linear(params.mlp_in, _layer_norm(params.mlp_norm, inputs)))
    return inputs + _linear(params.mlp_out, hidden)


def apply_encoder(params: EncoderParams, batch: Mapping[str, tuple[Any, Any]], config: EncoderConfig) -> EncoderOutput:
    """Encode the token sets of B agents: the context float32[B, O] and each type's token embeddings.

    Args:
        params: The parameters from :func:`init_encoder` for this ``config``.
        batch: Type name -> (features float32[B, N_t, F_t], mask bool[B, N_t]), as :func:`stack_batch` returns.
            A configured type missing from the mapping counts as empty. A padded slot may hold anything.
        config: The trunk's sizes (static under :func:`encoder_apply_jit`).

    Returns:
        An :class:`EncoderOutput`.

    Raises:
        ValueError: If the batch names an unknown type or no type at all, a shape does not match ``config``, or the
            batch has no token slot.
    """

    names = [name for name, _ in config.token_types]
    unknown = sorted(set(batch) - set(names))
    if unknown:
        raise ValueError(f"the batch names token types {unknown} that the config does not have")
    present = [name for name in names if name in batch]
    if not present:
        raise ValueError("the batch has no token set")
    batch_size = jnp.shape(batch[present[0]][0])[0]

    features_of: dict[str, jax.Array] = {}
    valid_of: dict[str, jax.Array] = {}
    for name, width in config.token_types:
        if name in batch:
            values = jnp.asarray(batch[name][0], dtype=jnp.float32)
            valid = jnp.asarray(batch[name][1], dtype=jnp.bool_)
        else:
            values = jnp.zeros((batch_size, 0, width), dtype=jnp.float32)
            valid = jnp.zeros((batch_size, 0), dtype=jnp.bool_)
        if values.ndim != 3 or values.shape[0] != batch_size or values.shape[2] != width or valid.shape != values.shape[:2]:
            raise ValueError(
                f"token type {name!r} needs features [B, N, {width}] and mask [B, N] with B={batch_size}; "
                f"got {values.shape} and {valid.shape}"
            )
        features_of[name] = values
        valid_of[name] = valid

    slots: dict[str, tuple[int, int]] = {}
    total = 0
    for name in names:
        count = valid_of[name].shape[1]
        slots[name] = (total, total + count)
        total += count
    if total == 0:
        raise ValueError("the batch has no token slot")

    sequence, valid_parts, marked_parts = [], [], []
    for name in names:
        valid = valid_of[name]
        values = jnp.where(valid[..., None], features_of[name], 0.0)  # a padded slot reads as zeros
        marked = valid & (values[..., -1] > 0.5)
        embedding = params.types[name]
        hidden = _token_mlp(embedding.embedding, values) + embedding.type_vector
        sequence.append(hidden + jnp.where(marked[..., None], params.marked_vector, 0.0))
        valid_parts.append(valid)
        marked_parts.append(marked)
    x = jnp.concatenate(sequence, axis=1)
    valid_tokens = jnp.concatenate(valid_parts, axis=1)
    marked_tokens = jnp.concatenate(marked_parts, axis=1)

    for block in params.blocks:
        x = _block(block, x, valid_tokens, config.heads)
    x = _layer_norm(params.norm, x)

    keep = valid_tokens[..., None]
    count = jnp.sum(valid_tokens, axis=1, keepdims=True).astype(jnp.float32)
    mean = jnp.sum(jnp.where(keep, x, 0.0), axis=1) / jnp.maximum(count, 1.0)
    maximum = jnp.max(jnp.where(keep, x, jnp.finfo(jnp.float32).min), axis=1)
    maximum = jnp.where(count > 0.0, maximum, 0.0)
    marked_count = jnp.sum(marked_tokens, axis=1, keepdims=True).astype(jnp.float32)
    mark = jnp.sum(jnp.where(marked_tokens[..., None], x, 0.0), axis=1) / jnp.maximum(marked_count, 1.0)
    self_vector = jnp.zeros_like(mean)
    start, stop = slots.get("self", (0, 0))
    if stop > start:
        self_valid = valid_tokens[:, start:stop]
        first = start + jnp.argmax(self_valid.astype(jnp.int32), axis=1)
        picked = x[jnp.arange(batch_size), first]
        self_vector = jnp.where(jnp.any(self_valid, axis=1)[:, None], picked, 0.0)

    summary = jnp.concatenate([mean, maximum, mark, self_vector], axis=-1)
    tokens = {name: x[:, first_slot:last_slot] for name, (first_slot, last_slot) in slots.items()}
    return EncoderOutput(context=_linear(params.readout, summary), tokens=tokens)


@functools.lru_cache(maxsize=None)
def encoder_apply_jit(config: EncoderConfig) -> Callable[..., EncoderOutput]:
    """A cached ``jax.jit`` of :func:`apply_encoder` with ``config`` static: call it as ``f(params, batch)``."""

    return jax.jit(functools.partial(apply_encoder, config=config))


def parameter_count(params: Any) -> int:
    """Count the scalars of a parameter pytree."""

    return sum(math.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))


def pack_tokens(tokens: Sequence[Any], capacity: int, cls: type) -> tuple[np.ndarray, np.ndarray]:
    """Pack one agent's tokens of one type into fixed-size arrays.

    Args:
        tokens: Token records, each an instance of ``cls``.
        capacity: The number of slots. The tokens fill the first slots in order.
        cls: The record class (``surface.py``). Its ``features()`` give one row.

    Returns:
        ``(features float32[capacity, F], mask bool[capacity])`` with ``F = len(cls.FEATURES)``. The slots past the
        tokens hold zeros and False.

    Raises:
        ValueError: If a token is not an instance of ``cls``, or there are more tokens than ``capacity``.
    """

    if len(tokens) > capacity:
        raise ValueError(f"{len(tokens)} {cls.__name__} tokens do not fit in capacity {capacity}")
    features = np.zeros((capacity, len(cls.FEATURES)), dtype=np.float32)
    mask = np.zeros((capacity,), dtype=np.bool_)
    for slot, token in enumerate(tokens):
        if not isinstance(token, cls):
            raise ValueError(f"token {slot} is a {type(token).__name__}, expected {cls.__name__}")
        features[slot] = token.features()
        mask[slot] = True
    return features, mask


def stack_batch(
    rows: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Stack per-agent packed token sets into a batch on the default device.

    Args:
        rows: One mapping per agent: type name -> (features, mask) from :func:`pack_tokens`. Every row must hold the
            same types, with the same shapes.

    Returns:
        ``{type: (features float32[B, N, F], mask bool[B, N])}``.

    Raises:
        ValueError: If there are no rows, or the rows disagree on their types.
    """

    if not rows:
        raise ValueError("stack_batch needs at least one agent")
    names = set(rows[0])
    if any(set(row) != names for row in rows):
        raise ValueError("every row must hold the same token types")
    batch = {}
    for name in rows[0]:
        features = np.stack([np.asarray(row[name][0], dtype=np.float32) for row in rows])
        mask = np.stack([np.asarray(row[name][1], dtype=np.bool_) for row in rows])
        batch[name] = (jnp.asarray(features), jnp.asarray(mask))
    return batch


__all__ = [
    "TRUNK_TYPES",
    "AttentionParams",
    "BlockParams",
    "DenseParams",
    "EncoderConfig",
    "EncoderOutput",
    "EncoderParams",
    "LayerNormParams",
    "TokenMLPParams",
    "TypeParams",
    "apply_encoder",
    "encoder_apply_jit",
    "init_encoder",
    "pack_tokens",
    "parameter_count",
    "stack_batch",
]
