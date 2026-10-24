"""Training through time on stored rollouts: the machinery and its exactness proof (``docs/TRAINING_PLAN.md`` 8.2 item 2,
stage G1; no training runs here).

The v3 trunk is recurrent. Each tick ``runtime/state_tick.tick`` runs the state layer (no parameters), then the trunk,
which reads and writes three carried things: the latent memory, the background summary, and each remembered object's
latent, read and written through the object's store slot. To train the trunk through time, a stored sequence must let the
trunk be re-run over the same decisions and reproduce what the rollout computed, with the same parameters:

- :func:`record` keeps, per decision and after the state layer, the tokens, each object type's row slots, which store
  slots took a new object (the layer zeroed their latent), the reset flags, and the carried state the trunk step starts
  from.
- :func:`cut` cuts each lane's decisions into sequences of up to L decisions that never cross a reset, each storing the
  carried state at its first decision.
- :func:`unroll` re-runs the trunk over a batch of sequences with ``lax.scan``, with the same reads and writes as
  ``tick``, in pure JAX so that a loss can differentiate through it.
- :func:`pack` and :func:`unpack` store the tokens as their valid rows only, in a storage dtype (float16 by default,
  section 8.6), and restore dense float32 arrays with zeros where a row is invalid.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderConfig,
    EncoderOutput,
    EncoderParams,
    apply_encoder,
    initial_background,
    initial_memory,
)
from experimental.neural_agent.skill_agent.runtime.state_tick import _carried, _gather, _scatter
from experimental.neural_agent.skill_agent.state import layer


class PackedRows(NamedTuple):
    """One token type's valid rows of one decision: ``index`` int32[M, 2] holds each valid row's (lane, row),
    ``features`` [M, F] its values in the storage dtype, and ``shape`` the dense (B, N, F)."""

    index: np.ndarray
    features: np.ndarray
    shape: tuple


class Decision(NamedTuple):
    """One decision of a rollout for every lane (B lanes, host arrays)."""

    tokens: dict  # token type -> (features float32[B, N, F], valid bool[B, N]), or PackedRows when recorded packed
    slots: dict  # object type -> int32[B, N]: each row's store slot, -1 for none
    restart: dict  # object type -> bool[B, C]: the store slots that took a new object this tick
    reset: np.ndarray  # bool[B]
    latent: np.ndarray | None  # float32[B, S, W]: the memory the trunk step starts from, before any reset; None if not kept
    background: np.ndarray | None  # float32[B, K, W]: the background the trunk step starts from, before any reset; ditto
    stores: dict | None  # object type -> float32[B, C, W]: the store latents as the trunk step reads them; None if not kept


class Recording:
    """The decisions of one rollout, in order, as :func:`record` keeps them."""

    def __init__(self, config: EncoderConfig, batch: int) -> None:
        self.config = config
        self.batch = batch
        self.decisions: list[Decision] = []


def _stores(state: layer.AgentState) -> dict:
    """The object stores by object type, named as ``state_tick.trunk_step`` names them."""

    return {"entity": state.memory.entity, "block_candidate": state.memory.block}


def _restarted(before: Any, after: Any) -> np.ndarray:
    """bool[B, C]: the slots whose latent was not all zero before the layer step and is all zero after it. The layer
    zeroes a slot for a new object (``state/memory.py``), and a reset empties the stores."""

    held = np.any(np.asarray(before) != 0, axis=-1)
    empty = ~np.any(np.asarray(after) != 0, axis=-1)
    return held & empty


def _kept_tokens(tokens: dict, dtype: Any) -> dict:
    """The tokens as :func:`record` keeps them: dense float32 when ``dtype`` is None, else each type's valid rows only."""

    if dtype is None:
        return {name: (np.asarray(features, np.float32), np.asarray(valid, bool))
                for name, (features, valid) in tokens.items()}
    kept = {}
    for name, (features, valid) in tokens.items():
        valid, values = np.asarray(valid, bool), np.asarray(features, np.float32)
        kept[name] = PackedRows(np.argwhere(valid).astype(np.int32), values[valid].astype(dtype), values.shape)
    return kept


def record(recording: Recording, before: layer.AgentState, tokens: dict, after: layer.AgentState,
           reset: Any = None, *, dtype: Any = None, carry_at: Any = None) -> None:
    """Keep one decision. ``before`` is the state the layer step started from, ``after`` the state it returned, and
    ``tokens`` its token dict. Call it between ``layer.step`` and the trunk step, with the ``reset`` the step took
    (None: no lane resets).

    ``dtype`` (None: dense float32, as before) keeps each token type's valid rows only, in that dtype, which is what
    makes a rollout's memory affordable (``docs/TRAINING_PLAN.md`` 8.6). ``carry_at`` (a set of decision indices; None:
    every decision) keeps the carried state (latent, background, store latents) only at those decisions. :func:`cut`
    needs it at each sequence's first decision."""

    objects = recording.config.object_types
    stores_before, stores_after = _stores(before), _stores(after)
    kept = carry_at is None or len(recording.decisions) in carry_at
    recording.decisions.append(Decision(
        tokens=_kept_tokens(tokens, dtype),
        slots={name: np.asarray(after.object_slots[name], np.int32) for name in objects},
        restart={name: _restarted(stores_before[name].latent, stores_after[name].latent) for name in objects},
        reset=np.zeros(recording.batch, bool) if reset is None else np.asarray(reset, bool),
        latent=None if after.latent is None or not kept else np.asarray(after.latent, np.float32),
        background=None if after.background is None or not kept else np.asarray(after.background, np.float32),
        stores=None if not kept else {name: np.asarray(stores_after[name].latent, np.float32) for name in objects}))


class Sequences(NamedTuple):
    """Sequences of up to L decisions: host arrays with a leading axis of n sequences. Steps past a sequence's
    ``steps`` are padding (no valid rows, no reset, slot -1, no restart). ``tokens`` is dense ``(features, valid)``
    per type, or :class:`PackedTokens` once packed."""

    lane: np.ndarray  # int32[n]: the lane each sequence was cut from
    first: np.ndarray  # int32[n]: its first decision in that lane's rollout
    steps: np.ndarray  # int32[n]: the decisions it holds, 1 to L
    reset: np.ndarray  # bool[n, L]
    tokens: dict  # token type -> (float32[n, L, N, F], bool[n, L, N]), or PackedTokens
    slots: dict  # object type -> int32[n, L, N]
    restart: dict  # object type -> bool[n, L, C]
    latent: np.ndarray | None  # float32[n, S, W]: the memory before the first decision's trunk step
    background: np.ndarray | None  # float32[n, K, W]: the background before the first decision's trunk step
    stores: dict  # object type -> float32[n, C, W]: the store latents the first decision's trunk step reads


def cut(recording: Recording, length: int, *, stride: int | None = None, ends: Any = None) -> Sequences:
    """Cut each lane's decisions into sequences of up to ``length`` (L) decisions. A reset starts a new sequence, so no
    sequence crosses one. Each sequence stores the carried state at its first decision.

    ``stride`` (default ``length``, today's cut) is the distance between a lane's sequence starts; below ``length`` the
    sequences overlap, each by ``length - stride`` (R2D2's burn-in, section 8.3 V1). ``ends`` (int[B], default all) is
    each lane's number of decisions to use. From a packed recording (:func:`record` with ``dtype``) the tokens come back
    packed, with no dense intermediate.

    Raises:
        ValueError: If ``length`` or ``stride`` is not a positive integer, ``ends`` is not one count per lane within the
            recording, nothing was recorded, no decision is inside a lane's horizon, or a sequence starts at a decision
            whose carried state was not kept (``record``'s ``carry_at``).
    """

    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"length must be a positive integer, got {length!r}")
    stride = length if stride is None else stride
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride!r}")
    decisions = recording.decisions
    if not decisions:
        raise ValueError("nothing was recorded")
    horizon = np.full(recording.batch, len(decisions), np.int64) if ends is None else np.asarray(ends, np.int64)
    if horizon.shape != (recording.batch,) or horizon.min() < 0 or horizon.max() > len(decisions):
        raise ValueError(f"ends must be one count per lane, each at most {len(decisions)}, got {ends!r}")
    spans = []  # (lane, first decision, steps)
    for lane in range(recording.batch):
        segment = 0  # a reset, or the lane's horizon, closes a segment; its sequences start every ``stride``
        for step in range(1, int(horizon[lane]) + 1):
            if step == horizon[lane] or decisions[step].reset[lane]:
                spans.extend((lane, begin, min(length, step - begin)) for begin in range(segment, step, stride))
                segment = step
    if not spans:
        raise ValueError("no decision is inside a lane's horizon")
    for _, start, _ in spans:
        if decisions[start].stores is None:
            raise ValueError(f"the carried state was not kept at decision {start}, where a sequence starts")
    count, objects, head = len(spans), recording.config.object_types, decisions[0]
    packed = any(isinstance(value, PackedRows) for value in head.tokens.values())
    if packed:
        tokens = {name: _packed_tokens(decisions, spans, length, name) for name in head.tokens}
    else:
        tokens = {name: (np.zeros((count, length) + features.shape[1:], np.float32),
                         np.zeros((count, length) + valid.shape[1:], bool))
                  for name, (features, valid) in head.tokens.items()}
    slots = {name: np.full((count, length, head.slots[name].shape[1]), -1, np.int32) for name in objects}
    restart = {name: np.zeros((count, length, head.restart[name].shape[1]), bool) for name in objects}
    reset = np.zeros((count, length), bool)
    latent = None if head.latent is None else np.zeros((count,) + head.latent.shape[1:], np.float32)
    background = None if head.background is None else np.zeros((count,) + head.background.shape[1:], np.float32)
    stores = {name: np.zeros((count,) + head.stores[name].shape[1:], np.float32) for name in objects}
    for s, (lane, start, steps) in enumerate(spans):
        for pos in range(steps):
            decision = decisions[start + pos]
            if not packed:
                for name, (features, valid) in decision.tokens.items():
                    tokens[name][0][s, pos] = features[lane]
                    tokens[name][1][s, pos] = valid[lane]
            for name in objects:
                slots[name][s, pos] = decision.slots[name][lane]
                restart[name][s, pos] = decision.restart[name][lane]
            reset[s, pos] = decision.reset[lane]
        first = decisions[start]
        if latent is not None:
            latent[s] = first.latent[lane]
        if background is not None:
            background[s] = first.background[lane]
        for name in objects:
            stores[name][s] = first.stores[name][lane]
    lanes, firsts, steps = (np.array(column, np.int32) for column in zip(*spans))
    return Sequences(lanes, firsts, steps, reset, tokens, slots, restart, latent, background, stores)


class PackedTokens(NamedTuple):
    """One token type's valid rows: ``index`` int32[M, 3] holds each valid row's (sequence, step, row), ``features``
    [M, F] its values in the storage dtype, and ``shape`` the dense (n, L, N, F)."""

    index: np.ndarray
    features: np.ndarray
    shape: tuple


def pack(sequences: Sequences, dtype: Any = np.float16) -> Sequences:
    """The tokens as their valid rows only, in ``dtype`` (section 8.6: float16 costs nothing that matters). The other
    fields are unchanged; a float32 pack is exact."""

    tokens = {}
    for name, (features, valid) in sequences.tokens.items():
        tokens[name] = PackedTokens(np.argwhere(valid).astype(np.int32), features[valid].astype(dtype), features.shape)
    return sequences._replace(tokens=tokens)


def unpack(sequences: Sequences) -> Sequences:
    """Dense float32 tokens restored from :func:`pack`: each valid row's values, zeros where a row is invalid."""

    tokens = {}
    for name, packed in sequences.tokens.items():
        if not isinstance(packed, PackedTokens):
            tokens[name] = packed
            continue
        features = np.zeros(packed.shape, np.float32)
        valid = np.zeros(packed.shape[:3], bool)
        rows = tuple(packed.index.T)
        features[rows] = packed.features
        valid[rows] = True
        tokens[name] = (features, valid)
    return sequences._replace(tokens=tokens)


def _packed_tokens(decisions: list, spans: list, length: int, name: str) -> PackedTokens:
    """One token type's packed rows for the sequences: each (sequence, step) takes its decision's rows of its lane, in
    sequence order (the index is sorted, so :func:`take` can slice it)."""

    parts, values, lookups = [], [], {}
    for s, (lane, start, steps) in enumerate(spans):
        for pos in range(steps):
            decision = start + pos
            entry = decisions[decision].tokens[name]
            if decision not in lookups:  # where each lane's rows begin in this decision's index
                lookups[decision] = np.searchsorted(entry.index[:, 0], np.arange(entry.shape[0] + 1))
            lo, hi = lookups[decision][lane], lookups[decision][lane + 1]
            if hi > lo:
                rows = entry.index[lo:hi, 1]
                parts.append(np.column_stack([np.full(hi - lo, s, np.int32), np.full(hi - lo, pos, np.int32), rows]))
                values.append(entry.features[lo:hi])
    template = decisions[0].tokens[name]
    index = np.concatenate(parts).astype(np.int32) if parts else np.zeros((0, 3), np.int32)
    features = np.concatenate(values) if values else np.zeros((0, template.shape[-1]), template.features.dtype)
    return PackedTokens(index, features, (len(spans), length) + tuple(template.shape[1:]))


def take(sequences: Sequences, indices: Any) -> Sequences:
    """The chosen sequences in the order given (repeats allowed), still packed where they were. A packed token type's
    rows are sliced with ``np.searchsorted`` on its sequence index. ``unpack(take(...))`` is a dense batch."""

    chosen = np.asarray(indices, np.int64)
    count = len(chosen)
    tokens = {}
    for name, value in sequences.tokens.items():
        if not isinstance(value, PackedTokens):
            tokens[name] = (value[0][chosen], value[1][chosen])
            continue
        lo = np.searchsorted(value.index[:, 0], chosen, side="left")
        hi = np.searchsorted(value.index[:, 0], chosen, side="right")
        rows = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)] + [np.zeros(0, np.int64)])
        new_sequence = np.repeat(np.arange(count, dtype=np.int32), hi - lo)
        index = np.column_stack([new_sequence, value.index[rows, 1:]]).astype(np.int32)
        tokens[name] = PackedTokens(index, value.features[rows], (count,) + tuple(value.shape[1:]))
    return Sequences(
        lane=sequences.lane[chosen], first=sequences.first[chosen], steps=sequences.steps[chosen],
        reset=sequences.reset[chosen], tokens=tokens,
        slots={name: value[chosen] for name, value in sequences.slots.items()},
        restart={name: value[chosen] for name, value in sequences.restart.items()},
        latent=None if sequences.latent is None else sequences.latent[chosen],
        background=None if sequences.background is None else sequences.background[chosen],
        stores={name: value[chosen] for name, value in sequences.stores.items()})


def step(params: EncoderParams, config: EncoderConfig, carry: tuple, decision: dict) -> tuple[tuple, EncoderOutput]:
    """One decision of the unroll for a batch of sequences: the carry ``(memory, background, store latents)`` and the
    decision's inputs, each with the batch on its leading axis (``tokens`` by type, ``slots`` and ``restart`` by object
    type, ``reset``). It does what ``state_tick.trunk_step`` does: the carried memory and background start over where
    ``reset`` is True, the stores' restarting slots are emptied, each object row reads its slot's latent, the encoder
    runs, and each row's next latent goes back to its slot. Returns the next carry and the decision's outputs.
    :func:`unroll` scans it; run one decision at a time it is the rollout's own computation."""

    latent, background, stores = carry
    objects, count = config.object_types, decision["reset"].shape[0]
    latent = _carried(latent, initial_memory(params, config, count), decision["reset"])
    background = _carried(background, initial_background(params, config, count), decision["reset"])
    stores = {name: jnp.where(decision["restart"][name][..., None], 0.0, stores[name]) for name in objects}
    held = {name: _gather(stores[name], decision["slots"][name]) for name in objects}
    encoded = apply_encoder(params, decision["tokens"], config, memory=latent, background=background,
                            objects=held or None)
    if objects:
        stores = {name: _scatter(stores[name], decision["slots"][name], encoded.objects[name]) for name in objects}
    return (encoded.memory, encoded.background, stores), encoded


def unroll(params: EncoderParams, config: EncoderConfig, sequences: Sequences) -> EncoderOutput:
    """The trunk re-run over every sequence with ``lax.scan`` over its L decisions (:func:`step`): each output is
    float32[n, L, ...], and the padding steps' outputs are not meaningful. Pure JAX, so it differentiates in ``params``.

    Raises:
        TypeError: If the tokens are still packed (:func:`unpack` restores them).
    """

    if any(isinstance(packed, PackedTokens) for packed in sequences.tokens.values()):
        raise TypeError("unroll needs dense tokens: unpack the sequences first")
    objects = config.object_types

    def time_major(leaf: Any) -> jax.Array:
        return jnp.swapaxes(jnp.asarray(leaf), 0, 1)

    xs = {
        "tokens": {name: (time_major(features), time_major(valid)) for name, (features, valid) in sequences.tokens.items()},
        "slots": {name: time_major(sequences.slots[name]) for name in objects},
        "restart": {name: time_major(sequences.restart[name]) for name in objects},
        "reset": time_major(sequences.reset),
    }
    carry = (None if sequences.latent is None else jnp.asarray(sequences.latent),
             None if sequences.background is None else jnp.asarray(sequences.background),
             {name: jnp.asarray(sequences.stores[name]) for name in objects})
    _, outputs = jax.lax.scan(functools.partial(step, params, config), carry, xs)
    return jax.tree_util.tree_map(lambda leaf: jnp.swapaxes(leaf, 0, 1), outputs)


__all__ = ["Decision", "PackedRows", "PackedTokens", "Recording", "Sequences", "cut", "pack", "record", "step", "take",
           "unpack", "unroll"]
