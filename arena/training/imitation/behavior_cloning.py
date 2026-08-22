"""Generic behavior cloning over Arena's factored action contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from arena.imitation.contract import TraceSequence
from arena.jax_contract import HEAD_SPANS

NATIVE_HEAD_SIZES = tuple(size for _, size in HEAD_SPANS.values())


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    """Tunable optimizer and categorical-loss settings."""

    head_sizes: tuple[int, ...] = NATIVE_HEAD_SIZES
    learning_rate: float = 3e-4
    max_gradient_norm: float = 1.0
    label_smoothing: float = 0.0
    entropy_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.head_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in self.head_sizes
        ):
            raise ValueError("head_sizes must contain positive widths")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not math.isfinite(self.entropy_weight) or self.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative")


class DemonstrationBatch(NamedTuple):
    """Dense factored demonstrations with row- or per-head sample weights."""

    observation: jax.Array
    action: jax.Array
    action_mask: jax.Array
    supervision_mask: jax.Array
    weight: jax.Array


@dataclass(frozen=True, slots=True)
class ProjectedDemonstrations:
    """A fail-closed BC batch bound to versioned trace projections."""

    batch: DemonstrationBatch
    source_index: npt.NDArray[np.int64]
    head_names: tuple[str, ...]
    observation_reference: str
    action_references: tuple[tuple[str, str], ...]
    action_mask_reference: str | None
    weight_reference: str | None
    trace_identity_sha256: str


class BehaviorCloningLoss(NamedTuple):
    loss: jax.Array
    negative_log_likelihood: jax.Array
    entropy: jax.Array
    accuracy: jax.Array
    valid_fraction: jax.Array


class BehaviorCloningMetrics(NamedTuple):
    loss: BehaviorCloningLoss
    gradient_norm: jax.Array
    update_applied: jax.Array


@dataclass(frozen=True, slots=True)
class BehaviorCloningTrainer:
    initialize: Callable
    step: Callable


def demonstration_batch(
    observation,
    action,
    *,
    action_mask=None,
    supervision_mask=None,
    weight=None,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
    device: bool = True,
) -> DemonstrationBatch:
    """Validate demonstrations, optionally deferring their accelerator transfer."""

    array = jnp.asarray if device else np.asarray
    observations = array(observation, dtype=np.float32)
    actions = array(action, dtype=np.int32)
    leading = actions.shape[:-1]
    if actions.ndim < 1 or actions.shape[-1] != len(config.head_sizes):
        raise ValueError("action must have shape [..., number_of_heads]")
    if observations.ndim < 1 or observations.shape[:-1] != leading:
        raise ValueError("observation and action leading axes must match")
    total_actions = sum(config.head_sizes)
    masks = (
        (jnp.ones if device else np.ones)(leading + (total_actions,), dtype=np.bool_)
        if action_mask is None
        else array(action_mask, dtype=np.bool_)
    )
    weights = (
        (jnp.ones if device else np.ones)(leading, dtype=np.float32)
        if weight is None
        else array(weight, dtype=np.float32)
    )
    supervised = (
        (jnp.ones if device else np.ones)(
            leading + (len(config.head_sizes),), dtype=np.bool_
        )
        if supervision_mask is None
        else array(supervision_mask, dtype=np.bool_)
    )
    if masks.shape != leading + (total_actions,):
        raise ValueError("action_mask must have shape [..., sum(head_sizes)]")
    if supervised.shape != leading + (len(config.head_sizes),):
        raise ValueError(
            "supervision_mask must have shape [..., number_of_heads]"
        )
    if weights.shape not in {leading, leading + (len(config.head_sizes),)}:
        raise ValueError("weight must match rows or rows plus number_of_heads")
    return DemonstrationBatch(
        observations,
        actions,
        masks,
        supervised,
        weights,
    )


def projected_demonstrations(
    trace: TraceSequence,
    *,
    observation: str,
    action_labels: Mapping[str, str],
    action_mask: str | None = None,
    weight: str | None = None,
    head_names: Sequence[str] | None = None,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
) -> ProjectedDemonstrations:
    """Bind exact, independently valid action heads from a composable trace.

    Projectors may supervise any subset of heads. Ambiguous rows abstain via
    component validity; incompatible exact labels fail instead of disappearing
    inside the loss.
    """

    if not isinstance(trace, TraceSequence):
        raise TypeError("trace must be TraceSequence")
    names = tuple(HEAD_SPANS) if head_names is None else tuple(head_names)
    if len(names) != len(config.head_sizes) or len(names) != len(set(names)):
        raise ValueError("head_names must uniquely name every configured head")
    labels = dict(action_labels)
    if not labels or not set(labels) <= set(names):
        raise ValueError("action_labels must name a nonempty subset of heads")

    observation_component = _projected_component(
        trace, observation, kinds={"observation", "state", "context"}
    )
    observations = np.asarray(observation_component.current)
    if observations.ndim != 2:
        raise ValueError("projected observations must have shape [rows, features]")
    row_valid = observation_component.validity().copy()
    total_actions = sum(config.head_sizes)
    if action_mask is None:
        legal = np.ones((trace.rows, total_actions), dtype=np.bool_)
    else:
        mask_component = _projected_component(trace, action_mask, kinds={"mask"})
        legal = np.asarray(mask_component.current)
        if legal.dtype != np.bool_ or legal.shape != (trace.rows, total_actions):
            raise ValueError(
                "projected action mask must be bool [rows, sum(head_sizes)]"
            )
        row_valid &= mask_component.validity()

    actions = np.zeros((trace.rows, len(names)), dtype=np.int32)
    supervised = np.zeros_like(actions, dtype=np.bool_)
    offset = 0
    for index, (head, size) in enumerate(
        zip(names, config.head_sizes, strict=True)
    ):
        reference = labels.get(head)
        if reference is not None:
            component = _projected_component(trace, reference, kinds={"action"})
            values = np.asarray(component.current)
            if values.dtype.kind not in "iu" or values.shape != (trace.rows,):
                raise ValueError("projected action labels must be scalar integers")
            known = component.validity()
            if np.any(known & ((values < 0) | (values >= size))):
                raise ValueError(f"projected {head} labels are out of range")
            actions[known, index] = values[known]
            comparable = known & row_valid
            chosen = np.clip(values, 0, size - 1)
            if np.any(comparable & ~legal[np.arange(trace.rows), offset + chosen]):
                raise ValueError(f"projected {head} label is illegal under its mask")
            supervised[:, index] = known
        offset += size

    weights = np.ones(trace.rows, dtype=np.float32)
    if weight is not None:
        weight_component = _projected_component(
            trace, weight, kinds={"context", "event", "mask"}
        )
        weights = np.asarray(weight_component.current, dtype=np.float32)
        if weights.shape != (trace.rows,) or np.any(~np.isfinite(weights)) or np.any(
            weights < 0.0
        ):
            raise ValueError("projected weights must be finite nonnegative scalars")
        row_valid &= weight_component.validity()

    source_index = np.flatnonzero(row_valid & np.any(supervised, axis=1)).astype(
        np.int64
    )
    if source_index.size == 0:
        raise ValueError("trace has no valid projected action labels")
    source_index.flags.writeable = False
    batch = demonstration_batch(
        observations[source_index],
        actions[source_index],
        action_mask=legal[source_index],
        supervision_mask=supervised[source_index],
        weight=weights[source_index],
        config=config,
    )
    return ProjectedDemonstrations(
        batch=batch,
        source_index=source_index,
        head_names=names,
        observation_reference=observation,
        action_references=tuple((name, labels[name]) for name in names if name in labels),
        action_mask_reference=action_mask,
        weight_reference=weight,
        trace_identity_sha256=trace.identity_sha256,
    )


def _projected_component(
    trace: TraceSequence, reference: str, *, kinds: set[str]
):
    if not isinstance(reference, str) or reference.endswith(".next"):
        raise ValueError("BC projector references must use current trace components")
    try:
        component = trace.components[reference]
    except KeyError as error:
        raise KeyError(f"trace has no component {reference!r}") from error
    if component.spec.kind not in kinds or not isinstance(component.current, np.ndarray):
        raise ValueError(f"trace component {reference!r} has an incompatible schema")
    return component


def behavior_cloning_loss(
    logits: jax.Array,
    batch: DemonstrationBatch,
    config: BehaviorCloningConfig,
) -> BehaviorCloningLoss:
    """Masked categorical imitation loss, reduced over examples and heads."""

    logits = jnp.asarray(logits, dtype=jnp.float32)
    if logits.shape != batch.action_mask.shape:
        raise ValueError("logits must have the same shape as action_mask")

    nlls, entropies, correct, valid = [], [], [], []
    offset = 0
    for index, size in enumerate(config.head_sizes):
        head_logits = logits[..., offset : offset + size]
        head_mask = batch.action_mask[..., offset : offset + size]
        action = batch.action[..., index]
        safe_action = jnp.clip(action, 0, size - 1)
        masked = jnp.where(head_mask, head_logits, jnp.finfo(jnp.float32).min)
        log_probability = jax.nn.log_softmax(masked, axis=-1)
        chosen = jnp.take_along_axis(log_probability, safe_action[..., None], axis=-1)[
            ..., 0
        ]
        legal = jnp.take_along_axis(head_mask, safe_action[..., None], axis=-1)[..., 0]
        legal_count = jnp.maximum(jnp.sum(head_mask, axis=-1), 1)
        smooth = -jnp.sum(jnp.where(head_mask, log_probability, 0.0), axis=-1)
        smooth /= legal_count
        nlls.append(
            -(1.0 - config.label_smoothing) * chosen + config.label_smoothing * smooth
        )
        probability = jnp.where(head_mask, jnp.exp(log_probability), 0.0)
        entropies.append(-jnp.sum(probability * log_probability, axis=-1))
        correct.append(jnp.argmax(masked, axis=-1) == safe_action)
        valid.append(
            (action >= 0)
            & (action < size)
            & legal
            & jnp.any(head_mask, -1)
            & batch.supervision_mask[..., index]
        )
        offset += size

    nll = jnp.stack(nlls, axis=-1)
    entropy = jnp.stack(entropies, axis=-1)
    accuracy = jnp.stack(correct, axis=-1)
    valid_rows = jnp.stack(valid, axis=-1)
    sample_weight = (
        batch.weight[..., None]
        if batch.weight.shape == batch.action.shape[:-1]
        else batch.weight
    )
    valid_rows &= jnp.isfinite(sample_weight) & (sample_weight >= 0.0)
    weights = jnp.where(valid_rows, sample_weight, 0.0)
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    mean_nll = jnp.sum(weights * nll) / denominator
    mean_entropy = jnp.sum(weights * entropy) / denominator
    return BehaviorCloningLoss(
        loss=mean_nll - config.entropy_weight * mean_entropy,
        negative_log_likelihood=mean_nll,
        entropy=mean_entropy,
        accuracy=jnp.sum(weights * accuracy) / denominator,
        valid_fraction=jnp.mean(valid_rows),
    )


def behavior_cloning_head_accuracy(
    logits: jax.Array,
    batch: DemonstrationBatch,
    config: BehaviorCloningConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return weighted accuracy and label support for every action head."""

    logits = jnp.asarray(logits, dtype=jnp.float32)
    if logits.shape != batch.action_mask.shape:
        raise ValueError("logits must have the same shape as action_mask")
    accuracies, supports = [], []
    offset = 0
    for index, size in enumerate(config.head_sizes):
        head_mask = batch.action_mask[..., offset : offset + size]
        action = batch.action[..., index]
        safe_action = jnp.clip(action, 0, size - 1)
        legal = jnp.take_along_axis(head_mask, safe_action[..., None], axis=-1)[..., 0]
        sample_weight = (
            batch.weight
            if batch.weight.shape == batch.action.shape[:-1]
            else batch.weight[..., index]
        )
        valid = (
            (action >= 0)
            & (action < size)
            & legal
            & jnp.any(head_mask, axis=-1)
            & batch.supervision_mask[..., index]
            & jnp.isfinite(sample_weight)
            & (sample_weight >= 0.0)
        )
        weight = jnp.where(valid, sample_weight, 0.0)
        support = jnp.sum(weight)
        choice = jnp.argmax(
            jnp.where(head_mask, logits[..., offset : offset + size], -jnp.inf),
            axis=-1,
        )
        accuracies.append(
            jnp.sum(weight * (choice == safe_action)) / jnp.maximum(support, 1.0)
        )
        supports.append(support)
        offset += size
    return jnp.stack(accuracies), jnp.stack(supports)


def make_behavior_cloning_trainer(
    apply: Callable,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
    *,
    optimizer: optax.GradientTransformation | None = None,
    compile: bool = True,
) -> BehaviorCloningTrainer:
    """Build an update for ``apply(params, observations, action_mask) -> logits``."""

    optimizer = optimizer or optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )

    def step(params: Any, optimizer_state: Any, batch: DemonstrationBatch):
        def objective(candidate):
            metrics = behavior_cloning_loss(
                apply(candidate, batch.observation, batch.action_mask), batch, config
            )
            return metrics.loss, metrics

        (loss_value, loss), gradients = jax.value_and_grad(objective, has_aux=True)(
            params
        )
        gradient_norm = optax.tree.norm(gradients)
        finite = jnp.isfinite(loss_value) & jnp.isfinite(gradient_norm)

        def update(_):
            changes, next_optimizer = optimizer.update(
                gradients, optimizer_state, params
            )
            return optax.apply_updates(params, changes), next_optimizer

        next_params, next_optimizer = jax.lax.cond(
            finite, update, lambda _: (params, optimizer_state), operand=None
        )
        return (
            next_params,
            next_optimizer,
            BehaviorCloningMetrics(
                loss=loss,
                gradient_norm=gradient_norm,
                update_applied=finite,
            ),
        )

    return BehaviorCloningTrainer(
        initialize=optimizer.init,
        step=jax.jit(step) if compile else step,
    )


__all__ = [
    "BehaviorCloningConfig",
    "BehaviorCloningLoss",
    "BehaviorCloningMetrics",
    "BehaviorCloningTrainer",
    "DemonstrationBatch",
    "ProjectedDemonstrations",
    "NATIVE_HEAD_SIZES",
    "behavior_cloning_loss",
    "behavior_cloning_head_accuracy",
    "demonstration_batch",
    "make_behavior_cloning_trainer",
    "projected_demonstrations",
]
