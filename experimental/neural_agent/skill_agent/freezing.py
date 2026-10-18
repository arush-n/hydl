"""Freezing: one shared trunk and one adapter per verb, so that adding a verb never changes a frozen verb.

Requirement 7 of ``SKILL_LIBRARY.md`` says a frozen skill never changes (``BUILD.md`` sections 3.2 and 7b, B4; E13 on
stubs). The mechanism is one shared trunk and one small adapter per verb. The trunk is the set encoder of ``encoder.py``
(B1): ``Encoder(x)`` is the context float32[B, O] of B agents' token sets. A verb ``v`` owns one adapter, and its head
logits read only the trunk and that adapter::

    logits_v(x) = Adapter_v(Encoder(x))           Adapter_v: Dense(O -> H), GELU, then one Dense(H -> size_h)
                                                  for each head h the verb owns (its scope's heads on gen 7)

Three rules make the guarantee exact rather than approximate:

- A verb's compiled program takes the trunk and that verb's adapter as arguments, and is cached by
  ``(verb, config, head names)``. It never sees another adapter, and adapters are never stacked across verbs, so the
  program does not depend on how many verbs exist.
- ``add_verb`` returns a new network whose trunk and existing adapters are the same objects as before.
- Training verb ``v`` differentiates its adapter only, and the trunk too while the trunk is not frozen::

      L_v = sum_{b, h} m_bh (-log softmax(logits_v,h(x_b))[y_bh])  /  max(1, sum_{b, h} m_bh)

  A frozen verb cannot be trained, and a frozen trunk receives no gradient. Training any verb while the trunk is not
  frozen moves the trunk, and with it every verb's logits: freeze the trunk before a body grows.

``params_sha256`` hashes a parameter tree, so a test can show that a frozen verb's weights, or the trunk's, did not
change.

This module proves the weights side of requirement 7. The contract side (a frozen skill never changes under the same
version) is enforced by ``library.py``.
"""

from __future__ import annotations

import functools
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.skill_agent.encoder import EncoderConfig, EncoderParams, apply_encoder, init_encoder
from experimental.neural_agent.skill_agent.surface import head_layout

__all__ = [
    "AdapterParams",
    "DenseLayer",
    "SkillNetwork",
    "add_verb",
    "freeze",
    "new_network",
    "params_sha256",
    "train_steps",
    "verb_logits",
]

_HEAD_SIZES: dict[str, int] = {head.name: head.size for head in head_layout("gen7")}
"""The gen-7 head names and sizes (``surface.head_layout``)."""


class DenseLayer(NamedTuple):
    """A dense layer: ``weight`` float32[fan_in, fan_out] and ``bias`` float32[fan_out]."""

    weight: jax.Array
    bias: jax.Array


class AdapterParams(NamedTuple):
    """One verb's adapter: ``hidden`` is Dense(O -> H); ``heads`` maps each head name to Dense(H -> size_h)."""

    hidden: DenseLayer
    heads: dict[str, DenseLayer]


@dataclass(frozen=True)
class SkillNetwork:
    """A trunk, one adapter per verb, and what is frozen. Functions here return new networks and never mutate one.

    Attributes:
        config: The trunk's sizes (``encoder.EncoderConfig``).
        trunk: The encoder's parameters.
        adapters: Verb -> its adapter. Never mutated after construction.
        frozen: The verbs whose adapters never change.
        trunk_frozen: True when the trunk never changes.
    """

    config: EncoderConfig
    trunk: EncoderParams
    adapters: Mapping[str, AdapterParams]
    frozen: frozenset[str]
    trunk_frozen: bool


def _init_dense(key: jax.Array, fan_in: int, fan_out: int) -> DenseLayer:
    """Kernel ~ N(0, 1/fan_in) and zero bias, as ``encoder.init_encoder`` draws its dense layers."""

    return DenseLayer(
        weight=jax.random.normal(key, (fan_in, fan_out), dtype=jnp.float32) / math.sqrt(fan_in),
        bias=jnp.zeros((fan_out,), dtype=jnp.float32),
    )


def _head_names(heads: Sequence[str]) -> tuple[str, ...]:
    names = tuple(heads)
    if not names:
        raise ValueError("a verb owns at least one head")
    if len(set(names)) != len(names):
        raise ValueError(f"a verb's heads must be distinct, got {names}")
    unknown = [name for name in names if name not in _HEAD_SIZES]
    if unknown:
        raise ValueError(f"unknown gen-7 heads {unknown}; expected names from surface.head_layout('gen7')")
    return names


def _init_adapter(key: jax.Array, config: EncoderConfig, heads: Sequence[str], hidden: int) -> AdapterParams:
    names = _head_names(heads)
    keys = jax.random.split(key, len(names) + 1)
    return AdapterParams(
        hidden=_init_dense(keys[0], config.output, hidden),
        heads={name: _init_dense(keys[index + 1], hidden, _HEAD_SIZES[name]) for index, name in enumerate(names)},
    )


def new_network(
    key: jax.Array, config: EncoderConfig, verbs: Mapping[str, Sequence[str]] = {}, hidden: int = 64
) -> SkillNetwork:
    """A trunk plus one adapter per verb; nothing is frozen.

    Args:
        key: A JAX PRNG key.
        config: The trunk's sizes.
        verbs: Verb -> the gen-7 head names it owns.
        hidden: The width H of every adapter's hidden layer.

    Returns:
        The network, with ``frozen`` empty and ``trunk_frozen`` False.

    Raises:
        ValueError: If a verb owns no head, names an unknown head, or names a head twice.
    """

    keys = jax.random.split(key, len(verbs) + 1)
    adapters = {
        verb: _init_adapter(keys[index + 1], config, heads, hidden) for index, (verb, heads) in enumerate(verbs.items())
    }
    return SkillNetwork(
        config=config,
        trunk=init_encoder(keys[0], config),
        adapters=adapters,
        frozen=frozenset(),
        trunk_frozen=False,
    )


def add_verb(
    network: SkillNetwork, verb: str, heads: Sequence[str], key: jax.Array, hidden: int = 64
) -> SkillNetwork:
    """A new network with ``verb``'s adapter added. The trunk and every existing adapter are the same objects.

    Raises:
        ValueError: If the verb exists, or its heads are invalid (see :func:`new_network`).
    """

    if verb in network.adapters:
        raise ValueError(f"verb {verb!r} exists; a verb's adapter is never replaced")
    adapter = _init_adapter(key, network.config, heads, hidden)
    return replace(network, adapters={**network.adapters, verb: adapter})


def freeze(network: SkillNetwork, verbs: Sequence[str] = (), trunk: bool = True) -> SkillNetwork:
    """A new network with ``verbs`` frozen and, when ``trunk`` is True, the trunk frozen.

    Freezing is one-way: ``trunk=False`` leaves the trunk's state as it is.

    Raises:
        ValueError: If a named verb is unknown (freezing a name that is not yet added would freeze the later verb).
    """

    unknown = sorted(set(verbs) - set(network.adapters))
    if unknown:
        raise ValueError(f"cannot freeze unknown verbs {unknown}")
    return replace(network, frozen=network.frozen | frozenset(verbs), trunk_frozen=network.trunk_frozen or trunk)


def _adapter_of(network: SkillNetwork, verb: str) -> AdapterParams:
    if verb not in network.adapters:
        raise ValueError(f"unknown verb {verb!r}; the network has {sorted(network.adapters)}")
    return network.adapters[verb]


def _rows(batch: Mapping[str, tuple[Any, Any]]) -> int:
    if not batch:
        raise ValueError("the batch has no token set")
    return int(jnp.shape(next(iter(batch.values()))[0])[0])


def _adapter_logits(
    trunk: EncoderParams, adapter: AdapterParams, batch: Mapping[str, tuple[Any, Any]], config: EncoderConfig
) -> dict[str, jax.Array]:
    context = apply_encoder(trunk, batch, config).context
    hidden = jax.nn.gelu(context @ adapter.hidden.weight + adapter.hidden.bias)
    return {name: hidden @ layer.weight + layer.bias for name, layer in adapter.heads.items()}


@functools.lru_cache(maxsize=None)
def _verb_program(verb: str, config: EncoderConfig, heads: tuple[str, ...]) -> Callable[..., dict[str, jax.Array]]:
    """The cached jitted logits of one verb. ``verb`` and ``heads`` key the cache; the trunk and adapter are arguments."""

    return jax.jit(functools.partial(_adapter_logits, config=config))


def verb_logits(network: SkillNetwork, verb: str, batch: Mapping[str, tuple[Any, Any]]) -> dict[str, jax.Array]:
    """The verb's head logits for a token batch: head -> float32[B, size]. Reads only the trunk and the verb's adapter.

    Args:
        network: The network.
        verb: A verb of ``network``.
        batch: Type name -> (features, mask), the mapping ``encoder.apply_encoder`` takes.

    Raises:
        ValueError: If the verb is unknown.
    """

    adapter = _adapter_of(network, verb)
    return _verb_program(verb, network.config, tuple(adapter.heads))(network.trunk, adapter, batch)


@functools.lru_cache(maxsize=None)
def _train_program(
    verb: str, config: EncoderConfig, heads: tuple[str, ...], trunk_frozen: bool, learning_rate: float
) -> tuple[Callable[..., Any], optax.GradientTransformation]:
    """The jitted Adam step of one verb, and its optimizer. ``verb`` keys the cache like :func:`_verb_program`."""

    optimizer = optax.adam(learning_rate)

    def loss(trunk: EncoderParams, adapter: AdapterParams, batch: Any, labels: Any, mask: Any) -> jax.Array:
        logits = _adapter_logits(trunk, adapter, batch, config)
        total, count = 0.0, 0.0
        for name in heads:
            picked = jnp.take_along_axis(jax.nn.log_softmax(logits[name]), labels[name][:, None], axis=1)[:, 0]
            weight = mask[name].astype(jnp.float32)
            total = total - jnp.sum(picked * weight)
            count = count + jnp.sum(weight)
        return total / jnp.maximum(count, 1.0)

    def step(trunk: EncoderParams, adapter: AdapterParams, opt_state: Any, batch: Any, labels: Any, mask: Any):
        if trunk_frozen:
            grad = jax.grad(loss, argnums=1)(trunk, adapter, batch, labels, mask)
            updates, opt_state = optimizer.update(grad, opt_state, adapter)
            return trunk, optax.apply_updates(adapter, updates), opt_state
        trunk_grad, adapter_grad = jax.grad(loss, argnums=(0, 1))(trunk, adapter, batch, labels, mask)
        updates, opt_state = optimizer.update((trunk_grad, adapter_grad), opt_state, (trunk, adapter))
        new_trunk, new_adapter = optax.apply_updates((trunk, adapter), updates)
        return new_trunk, new_adapter, opt_state

    return jax.jit(step), optimizer


def train_steps(
    network: SkillNetwork,
    verb: str,
    batches: Mapping[str, tuple[Any, Any]],
    labels: Mapping[str, Any],
    steps: int,
    learning_rate: float = 1e-3,
    mask: Mapping[str, Any] | None = None,
) -> SkillNetwork:
    """Train ``verb`` for ``steps`` Adam steps on one token batch, and return the new network.

    The loss is the masked cross-entropy of the verb's heads (the module docstring). Gradients flow into the verb's
    adapter, and into the trunk while the trunk is not frozen. Each call starts a fresh Adam state.

    Args:
        network: The network.
        verb: The verb to train. It must not be frozen.
        batches: One token batch (the mapping ``encoder.apply_encoder`` takes); every step trains on it.
        labels: Head -> int array [B] of choices in [0, size), one entry for each head of the verb.
        steps: The number of Adam steps.
        learning_rate: The Adam learning rate.
        mask: Optional head -> bool array [B]; a True row is supervised. A head it does not name is supervised on
            every row.

    Returns:
        The new network: the verb's adapter replaced, and the trunk too when it is not frozen. Every other adapter is
        the same object as before; a frozen trunk is the same object too.

    Raises:
        ValueError: If the verb is frozen or unknown, the labels do not name exactly the verb's heads, the mask names
            a head the verb does not own, or a label or mask does not have one entry per row.
    """

    if verb in network.frozen:
        raise ValueError(f"verb {verb!r} is frozen; its adapter never changes")
    adapter = _adapter_of(network, verb)
    heads = tuple(adapter.heads)
    rows = _rows(batches)
    if set(labels) != set(heads):
        raise ValueError(f"labels must name the heads {heads} of verb {verb!r}, got {tuple(labels)}")
    label_arrays = {name: jnp.asarray(labels[name], dtype=jnp.int32) for name in heads}
    mask_arrays = {name: jnp.ones((rows,), dtype=jnp.bool_) for name in heads}
    if mask is not None:
        unknown = sorted(set(mask) - set(heads))
        if unknown:
            raise ValueError(f"mask names heads {unknown} that verb {verb!r} does not own")
        mask_arrays.update({name: jnp.asarray(mask[name], dtype=jnp.bool_) for name in mask})
    for name in heads:
        if label_arrays[name].shape != (rows,) or mask_arrays[name].shape != (rows,):
            raise ValueError(f"head {name!r} needs one label and one mask entry per row ({rows} rows)")

    program, optimizer = _train_program(verb, network.config, heads, network.trunk_frozen, float(learning_rate))
    trainable = adapter if network.trunk_frozen else (network.trunk, adapter)
    opt_state = optimizer.init(trainable)
    trunk, trained = network.trunk, adapter
    for _ in range(steps):
        trunk, trained, opt_state = program(trunk, trained, opt_state, batches, label_arrays, mask_arrays)
    if network.trunk_frozen:
        trunk = network.trunk
    return replace(network, trunk=trunk, adapters={**network.adapters, verb: trained})


def params_sha256(tree: Any) -> str:
    """The SHA-256 of a parameter tree: for each leaf, in sorted key-path order, its path, dtype, shape and bytes.

    Equal digests mean the leaves agree bit for bit, up to SHA-256 collisions.
    """

    digest = hashlib.sha256()
    leaves = sorted(jax.tree_util.tree_leaves_with_path(tree), key=lambda item: jax.tree_util.keystr(item[0]))
    for path, leaf in leaves:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(f"{jax.tree_util.keystr(path)}|{array.dtype}|{array.shape}|".encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()
