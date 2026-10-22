"""The request-conditioned generalist: one network that reads a tick's open requests and the trunk's token sets.

Each tick the middle layer holds up to ``R`` requests for an agent, and the combiner has granted each one a scope. The
generalist reads them with the trunk's token sets (``model/encoder.py``) and answers two things: the action heads, read
from the scope query of the scope each head belongs to, and each request's status and progress. It is pure JAX, as
``model/encoder.py`` is: parameters are NamedTuples of arrays, and :func:`apply_generalist` is a function of them.

A request token, for slot ``r`` of agent ``b`` (width ``W``)::

    q = Embed_verb(verb) + Dense(args * present) + Embed_scope(scope) + T_entity[entity_pointer] + T_cell[cell_pointer]

``present`` expands the six argument-group flags to the 15 argument features (``ARG_GROUPS``). A pointer adds nothing
when it is -1, out of range, or names a padded slot. The request tokens and six learned scope queries form
``X = [q_1 .. q_R; s_1 .. s_6]``, and ``layers`` pre-LayerNorm blocks follow::

    X <- X + SelfAttn(LN(X); V)         V: the valid request slots and every scope query
    X <- X + CrossAttn(LN(X); T, M)     T: every token the trunk outputs, in its order; M: their validity
    X <- X + MLP(LN(X))                 MLP: Dense(W -> 2 W), GELU, Dense(2 W -> W)

After a final LayerNorm, the readouts are::

    head_logits[h] = Dense_h(s_scope(h))           float32[B, size_h], for each head of surface.head_layout("gen7")
    status         = Dense(W -> 3)(q)              logits of done, stuck, success: float32[B, R, 3]
    progress       = sigmoid(Dense(W -> 1)(q))     float32[B, R]

Masking is exact, as in the encoder. An invalid request's token is zeroed before the first layer, so none of its fields
reaches the network; it is never a key; and its status and progress read 0. An absent argument group is replaced by
zeros with ``where``, not multiplied, so whatever it holds never reaches the network. The parameters depend on the
widths, the verb count and the head layout, never on the token counts or on ``max_requests``.

Choices the brief left open: the granted scope enters as a learned embedding added to each request token, since without
it a scope query cannot tell which request it owns; the trunk's ``context`` is not read (the generalist attends to the
token sets); the progress readout is a sigmoid; the final LayerNorm before the readouts follows the pre-LN convention.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, head_layout, scope_of
from experimental.neural_agent.skill_agent.model.encoder import (
    TRUNK_TYPES,
    AttentionParams,
    DenseParams,
    EncoderConfig,
    EncoderOutput,
    LayerNormParams,
    _init_dense,
    _init_norm,
    _init_normal,
    _layer_norm,
    _linear,
)

__all__ = [
    "ARG_FEATURES",
    "ARG_GROUPS",
    "AttentionParams",
    "GeneralistBlockParams",
    "GeneralistConfig",
    "GeneralistOutput",
    "GeneralistParams",
    "GeneralistRequests",
    "apply_generalist",
    "gather_pointed",
    "generalist_apply_jit",
    "init_generalist",
]

ARG_GROUPS: tuple[tuple[str, int], ...] = (
    ("position", 3),
    ("direction", 3),
    ("scalars", 4),
    ("mode", 3),
    ("urgency", 1),
    ("age", 1),
)
"""The six argument groups and their feature widths, in feature order: position and direction in the agent frame,
four scalars, the mode one-hot (parallel, after, interrupt), urgency, and age in ticks / 256."""

ARG_FEATURES: int = sum(width for _, width in ARG_GROUPS)
"""The argument feature width: 15."""

_GROUP_OF_FEATURE: np.ndarray = np.repeat(np.arange(len(ARG_GROUPS)), [width for _, width in ARG_GROUPS])
"""The group of each argument feature, so ``present[..., _GROUP_OF_FEATURE]`` expands the six group flags."""

_HEADS = head_layout("gen7")
_MLP_RATIO: int = 2
_STATUS_LOGITS: int = 3


def _positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class GeneralistConfig:
    """The generalist's sizes. The parameters depend on these, never on the token counts or on ``max_requests``.

    Attributes:
        width: The model width ``W``. It must equal the trunk's width (:func:`init_generalist` checks).
        heads: Attention heads; ``width`` must be a multiple of it.
        layers: Pre-LayerNorm blocks.
        max_requests: The request slots ``R`` of every call. A producer pads short sets with ``valid`` False.
        verb_count: The verb vocabulary size.
        arg_width: The argument feature width. It is the 15 features of ``ARG_GROUPS``, so any other value is refused.

    Raises:
        ValueError: If a size is not a positive integer, ``width`` is not a multiple of ``heads``, or ``arg_width`` is
            not :data:`ARG_FEATURES`.
    """

    width: int = 64
    heads: int = 4
    layers: int = 2
    max_requests: int = 6
    verb_count: int = 160
    arg_width: int = ARG_FEATURES

    def __post_init__(self) -> None:
        for name in ("width", "heads", "layers", "max_requests", "verb_count"):
            _positive(getattr(self, name), name)
        if self.width % self.heads:
            raise ValueError(f"width {self.width} must be a multiple of heads {self.heads}")
        if self.arg_width != ARG_FEATURES:
            raise ValueError(f"arg_width must be {ARG_FEATURES}, the features of ARG_GROUPS, got {self.arg_width}")


class GeneralistRequests(NamedTuple):
    """The request slots of B agents. Every field has the leading dimensions [B, R].

    Attributes:
        verb: int32[B, R]; a valid request's verb is in [0, verb_count).
        args: float32[B, R, arg_width]: position, direction, scalars, mode one-hot, urgency and age, in ``ARG_GROUPS``
            order.
        arg_present: bool[B, R, 6]: whether each argument group is set. An unset group reads as zeros.
        entity_pointer: int32[B, R]: the entity slot the request names, or -1.
        cell_pointer: int32[B, R]: the block-candidate slot the request names, or -1.
        scope: int32[B, R] in [0, 6): the scope the combiner granted the request.
        valid: bool[B, R]: False for an empty slot.
    """

    verb: jax.Array
    args: jax.Array
    arg_present: jax.Array
    entity_pointer: jax.Array
    cell_pointer: jax.Array
    scope: jax.Array
    valid: jax.Array


class GeneralistOutput(NamedTuple):
    """The generalist's readouts.

    Attributes:
        head_logits: Head name -> float32[B, size], for every head of ``surface.head_layout("gen7")``.
        status: float32[B, R, 3], logits of done, stuck and success. Zero for an invalid request.
        progress: float32[B, R] in (0, 1). Zero for an invalid request.
    """

    head_logits: dict[str, jax.Array]
    status: jax.Array
    progress: jax.Array


class GeneralistBlockParams(NamedTuple):
    self_norm: LayerNormParams
    self_attention: AttentionParams
    cross_norm: LayerNormParams
    cross_attention: AttentionParams
    mlp_norm: LayerNormParams
    mlp_in: DenseParams
    mlp_out: DenseParams


class GeneralistParams(NamedTuple):
    verbs: jax.Array  # float32[verb_count, W]
    scopes: jax.Array  # float32[6, W]: the granted scope, added to each request token
    arguments: DenseParams  # Dense(arg_width -> W)
    scope_queries: jax.Array  # float32[6, W]
    blocks: tuple[GeneralistBlockParams, ...]
    norm: LayerNormParams
    heads: dict[str, DenseParams]  # head name -> Dense(W -> size)
    status: DenseParams  # Dense(W -> 3)
    progress: DenseParams  # Dense(W -> 1)


def _init_attention(keys: Iterator[jax.Array], width: int) -> AttentionParams:
    return AttentionParams(
        query=_init_dense(next(keys), width, width),
        key=_init_dense(next(keys), width, width),
        value=_init_dense(next(keys), width, width),
        output=_init_dense(next(keys), width, width),
    )


def init_generalist(key: jax.Array, config: GeneralistConfig, encoder_config: EncoderConfig) -> GeneralistParams:
    """Initialise the parameters as the trunk's are: kernels and embeddings ~ N(0, 1/fan_in), zero biases, unit scales.

    Args:
        key: A JAX PRNG key.
        config: The generalist's sizes.
        encoder_config: The trunk the generalist reads. Its width must be ``config.width``, since the trunk's tokens join
            the request tokens, and it must have every trunk type.

    Returns:
        The parameters. Their shapes depend on ``config`` and on the head layout, never on a token count.

    Raises:
        ValueError: If the widths differ, or the trunk lacks a token type the generalist reads.
    """

    if encoder_config.width != config.width:
        raise ValueError(f"the trunk's width {encoder_config.width} must equal the generalist's width {config.width}")
    names = {name for name, _ in encoder_config.token_types}
    missing = sorted(name for name, _ in TRUNK_TYPES if name not in names)
    if missing:
        raise ValueError(f"the trunk lacks the token types {missing} that the generalist reads")

    width = config.width
    keys = iter(jax.random.split(key, 4 + 10 * config.layers + len(_HEADS) + 2))
    verbs = _init_normal(next(keys), (config.verb_count, width), width)
    scopes = _init_normal(next(keys), (len(SCOPE_NAMES), width), width)
    arguments = _init_dense(next(keys), config.arg_width, width)
    scope_queries = _init_normal(next(keys), (len(SCOPE_NAMES), width), width)
    blocks = tuple(
        GeneralistBlockParams(
            self_norm=_init_norm(width),
            self_attention=_init_attention(keys, width),
            cross_norm=_init_norm(width),
            cross_attention=_init_attention(keys, width),
            mlp_norm=_init_norm(width),
            mlp_in=_init_dense(next(keys), width, _MLP_RATIO * width),
            mlp_out=_init_dense(next(keys), _MLP_RATIO * width, width),
        )
        for _ in range(config.layers)
    )
    heads = {head.name: _init_dense(next(keys), width, head.size) for head in _HEADS}
    return GeneralistParams(
        verbs=verbs,
        scopes=scopes,
        arguments=arguments,
        scope_queries=scope_queries,
        blocks=blocks,
        norm=_init_norm(width),
        heads=heads,
        status=_init_dense(next(keys), width, _STATUS_LOGITS),
        progress=_init_dense(next(keys), width, 1),
    )


def _attend(params: AttentionParams, queries: jax.Array, memory: jax.Array, valid: jax.Array, heads: int) -> jax.Array:
    """Masked multi-head attention from ``queries`` float32[B, Q, W] over ``memory`` float32[B, K, W].

    ``valid`` is bool[B, K]: a key with False gets logit -inf. A row with no valid key gives 0. Self-attention passes the
    same array as both.
    """

    batch, query_count, width = queries.shape
    key_count = memory.shape[1]
    head_width = width // heads

    def split_heads(projected: jax.Array, count: int) -> jax.Array:
        return projected.reshape(batch, count, heads, head_width).transpose(0, 2, 1, 3)

    query = split_heads(_linear(params.query, queries), query_count)
    key = split_heads(_linear(params.key, memory), key_count)
    value = split_heads(_linear(params.value, memory), key_count)
    logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(head_width)
    any_valid = jnp.any(valid, axis=1)
    logits = jnp.where(valid[:, None, None, :], logits, -jnp.inf)
    # A row with no valid key is all -inf. Zero its logits so the softmax stays finite; its output is dropped below.
    logits = jnp.where(any_valid[:, None, None, None], logits, 0.0)
    weights = jax.nn.softmax(logits, axis=-1)
    mixed = jnp.einsum("bhqk,bhkd->bhqd", weights, value).transpose(0, 2, 1, 3).reshape(batch, query_count, width)
    return jnp.where(any_valid[:, None, None], _linear(params.output, mixed), 0.0)


def gather_pointed(tokens: jax.Array, token_valid: jax.Array, pointer: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The trunk token that each request points at.

    Args:
        tokens: float32[B, N, W], one trunk type's token embeddings (``EncoderOutput.tokens``).
        token_valid: bool[B, N], that type's validity.
        pointer: int32[B, R], the slot each request names, or -1.

    Returns:
        ``(vectors float32[B, R, W], present bool[B, R])``. A pointer that is -1, past the last slot, or at a padded slot
        gives a zero vector and False.
    """

    batch, slots, width = tokens.shape
    requests = pointer.shape[1]
    if slots == 0:
        return jnp.zeros((batch, requests, width), dtype=jnp.float32), jnp.zeros((batch, requests), dtype=jnp.bool_)
    index = jnp.clip(pointer, 0, slots - 1)
    present = (pointer >= 0) & (pointer < slots) & jnp.take_along_axis(token_valid, index, axis=1)
    vectors = tokens[jnp.arange(batch)[:, None], index]
    return jnp.where(present[..., None], vectors, 0.0), present


def _check_shapes(encoder_output: EncoderOutput, requests: GeneralistRequests, config: GeneralistConfig) -> None:
    batch, slots = jnp.shape(requests.verb)
    if slots != config.max_requests:
        raise ValueError(f"the requests have {slots} slots; the config has max_requests={config.max_requests}")
    expected = {
        "verb": (batch, slots),
        "args": (batch, slots, config.arg_width),
        "arg_present": (batch, slots, len(ARG_GROUPS)),
        "entity_pointer": (batch, slots),
        "cell_pointer": (batch, slots),
        "scope": (batch, slots),
        "valid": (batch, slots),
    }
    for field, shape in expected.items():
        actual = tuple(jnp.shape(getattr(requests, field)))
        if actual != shape:
            raise ValueError(f"request field {field!r} needs shape {shape}, got {actual}")
    missing = [name for name, _ in TRUNK_TYPES if name not in encoder_output.tokens]
    if missing:
        raise ValueError(f"the encoder output lacks the trunk types {missing}")
    if tuple(jnp.shape(encoder_output.context))[0] != batch:
        raise ValueError(f"the encoder output has a different batch size from the requests ({batch})")


def _trunk_masks(encoder_output: EncoderOutput, token_masks: Mapping[str, Any]) -> dict[str, jax.Array]:
    """Each token type the trunk output carries, in its order (the six v1 types for a v1 trunk, every v3 type for a v3
    one; 2026-10-22, step 1b of ``docs/VIRTUAL_HUMAN.md`` section 13), with its validity mask."""

    masks: dict[str, jax.Array] = {}
    for name in encoder_output.tokens:
        shape = jnp.shape(encoder_output.tokens[name])[:2]
        if name in token_masks:
            mask = jnp.asarray(token_masks[name], dtype=jnp.bool_)
            if mask.shape != shape:
                raise ValueError(f"the token mask {name!r} needs shape {shape}, got {mask.shape}")
        else:
            mask = jnp.zeros(shape, dtype=jnp.bool_)
        masks[name] = mask
    return masks


def apply_generalist(
    params: GeneralistParams,
    encoder_output: EncoderOutput,
    token_masks: Mapping[str, Any],
    requests: GeneralistRequests,
    config: GeneralistConfig,
) -> GeneralistOutput:
    """Run the generalist on B agents' requests and trunk token sets.

    Args:
        params: The parameters from :func:`init_generalist` for this ``config``.
        encoder_output: The trunk's output for the same B agents (``encoder.apply_encoder``).
        token_masks: Trunk type name -> bool[B, N_t], the validity of each token slot, as the encoder batch holds them.
            A trunk type missing from the mapping counts as empty.
        requests: The request slots, with R = ``config.max_requests``.
        config: The generalist's sizes (static under :func:`generalist_apply_jit`).

    Returns:
        A :class:`GeneralistOutput`.

    Raises:
        ValueError: If a shape does not match ``config`` or ``encoder_output``, a mask has the wrong shape, or the trunk
            lacks a type.
    """

    _check_shapes(encoder_output, requests, config)
    batch, slots = jnp.shape(requests.verb)
    masks = _trunk_masks(encoder_output, token_masks)
    memory = jnp.concatenate([encoder_output.tokens[name] for name in masks], axis=1)
    memory_valid = jnp.concatenate(list(masks.values()), axis=1)

    arguments = jnp.where(requests.arg_present[..., _GROUP_OF_FEATURE], requests.args, 0.0)
    token = params.verbs[requests.verb] + _linear(params.arguments, arguments) + params.scopes[requests.scope]
    entity, _ = gather_pointed(encoder_output.tokens["entity"], masks["entity"], requests.entity_pointer)
    cell, _ = gather_pointed(encoder_output.tokens["block_candidate"], masks["block_candidate"], requests.cell_pointer)
    request_tokens = jnp.where(requests.valid[..., None], token + entity + cell, 0.0)

    scope_count = len(SCOPE_NAMES)
    scope_queries = jnp.broadcast_to(params.scope_queries, (batch, scope_count, config.width))
    x = jnp.concatenate([request_tokens, scope_queries], axis=1)
    key_valid = jnp.concatenate([requests.valid, jnp.ones((batch, scope_count), dtype=jnp.bool_)], axis=1)
    for block in params.blocks:
        normed = _layer_norm(block.self_norm, x)
        x = x + _attend(block.self_attention, normed, normed, key_valid, config.heads)
        normed = _layer_norm(block.cross_norm, x)
        x = x + _attend(block.cross_attention, normed, memory, memory_valid, config.heads)
        hidden = jax.nn.gelu(_linear(block.mlp_in, _layer_norm(block.mlp_norm, x)))
        x = x + _linear(block.mlp_out, hidden)
    x = _layer_norm(params.norm, x)

    states = x[:, :slots]
    scope_states = x[:, slots:]
    status = jnp.where(requests.valid[..., None], _linear(params.status, states), 0.0)
    progress = jnp.where(requests.valid, jax.nn.sigmoid(_linear(params.progress, states)[..., 0]), 0.0)
    head_logits = {}
    for head in _HEADS:
        scope_index = SCOPE_NAMES.index(scope_of(head.name))
        head_logits[head.name] = _linear(params.heads[head.name], scope_states[:, scope_index])
    return GeneralistOutput(head_logits=head_logits, status=status, progress=progress)


@functools.lru_cache(maxsize=None)
def generalist_apply_jit(config: GeneralistConfig) -> Callable[..., GeneralistOutput]:
    """A cached ``jax.jit`` of :func:`apply_generalist` with ``config`` static: call it as ``f(params, encoder_output,
    token_masks, requests)``."""

    return jax.jit(functools.partial(apply_generalist, config=config))
