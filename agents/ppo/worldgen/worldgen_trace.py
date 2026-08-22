"""Exact WorldGen Arena traces mapped to recurrent PPO imitation batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from adk.policy.actions import action_index
from adk.policy.fields import field
from arena.imitation import (
    TraceContract,
    TraceCorpus,
    TraceSequence,
    numeric_component,
    reward_component,
)
from arena.jax_contract import HEAD_SPANS
from arena.jax_env import ArenaHandle, ArenaScene, collect
from arena.training import (
    BehaviorCloningConfig,
    DemonstrationBatch,
    demonstration_batch,
)
from hytalegym.jax.combat.observation.v3.policy import (
    arsenal_standard_root_legal,
    greedy_arsenal_action_factors,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.training.types import PPOConfig
from hytalegym.jax.training.policy import apply_policy, sample_configured_actions


_HEADS = tuple(HEAD_SPANS)
_BASE = _HEADS.index("base_action")
_ABILITY = _HEADS.index("ability_none_plus_slots")
_BASE_OFFSET, _BASE_SIZE = HEAD_SPANS["base_action"]
_ABILITY_OFFSET, _ABILITY_SIZE = HEAD_SPANS["ability_none_plus_slots"]
_IDLE = action_index("action_mask", "idle")
_APPROACH = action_index("action_mask", "approach")
_FACE = action_index("action_mask", "face_target")
_WORLDGEN_GENERATORS = frozenset({"worldgen_v2", "worldgen_v2_publication"})

WORLDGEN_PPO_PROJECTION_SCHEMA = "hytalerl_worldgen_ppo_demonstration_v5"
SCRIPTED_DEMONSTRATOR_SCHEMA = "hytalerl_worldgen_scripted_demonstrator_v1"
WORLDGEN_CORPUS_COMPATIBILITY_KEYS = (
    "arena.scene_contract_sha256",
    "arena.observation_schema",
    "arena.action_schema",
    "arena.observation_size",
    "arena.action_head_names",
    "arena.action_head_sizes",
    "arena.world_pool_semantic_sha256",
    "arena.policy_transfer_contract_sha256",
    "arena.observation_contract_sha256",
    "arena.action_contract_sha256",
)
_PROJECTION_MANIFEST = {
    "schema": WORLDGEN_PPO_PROJECTION_SCHEMA,
    "requires": (
        "arena.policy_observation.current_next",
        "arena.action_mask.current_next",
        "arena.action_factors",
        "episode_id",
        "episode_start",
        "arena.policy_transfer_contract_sha256",
        "arena.observation_contract_sha256",
        "arena.action_contract_sha256",
        "arena.trace_content_sha256",
    ),
    "output": "arena.training.DemonstrationBatch",
    "head_supervision": "explicit_nonempty_projector_subset",
    "invalid_rows": "drop_then_reject_recurrent_gaps",
}
WORLDGEN_PPO_PROJECTION_SHA256 = hashlib.sha256(
    json.dumps(_PROJECTION_MANIFEST, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class MappedDemonstrations:
    """Episode-major recurrent BC inputs plus their original trace rows."""

    batch: DemonstrationBatch
    episode_start: jax.Array
    source_index: np.ndarray
    source_trace_index: np.ndarray
    lengths: tuple[int, ...]
    supervised_heads: tuple[str, ...]
    trace_identity_sha256: tuple[str, ...]


def worldgen_trace_corpus(*traces: TraceSequence) -> TraceCorpus:
    """Group diverse demonstrators without losing their row provenance."""

    identities = tuple(item.identity_sha256 for item in traces)
    if len(identities) != len(set(identities)):
        raise ValueError("WorldGen trace corpus sources must be unique")
    return TraceCorpus(tuple(traces), WORLDGEN_CORPUS_COMPATIBILITY_KEYS)


def capture_worldgen_trace(
    scene: ArenaScene,
    key: jax.Array,
    steps: int,
    *,
    compile: bool = True,
) -> TraceSequence:
    """Record a legal scripted teacher through the actual generated JAX scene."""

    _validate_scene(scene)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    def run(seed):
        return collect(scene, _teacher, _record_transition, seed, steps)

    _, trajectory = (jax.jit(run) if compile else run)(key)
    return worldgen_transition_trace(
        scene,
        jax.device_get(trajectory),
        extra_parameters={
            "arena.demonstrator.kind": SCRIPTED_DEMONSTRATOR_SCHEMA,
            "arena.demonstrator.selection_mode": "deterministic_factors",
        },
    )


def capture_recurrent_worldgen_trace(
    scene: ArenaScene,
    key: jax.Array,
    steps: int,
    policy_params: Any,
    ppo: PPOConfig,
    *,
    checkpoint_sha256: str,
    mode: str = "greedy",
    compile: bool = True,
) -> TraceSequence:
    """Capture any compatible recurrent JAX checkpoint as an IL demonstrator."""

    _validate_scene(scene)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if mode not in {"greedy", "sample"}:
        raise ValueError("mode must be 'greedy' or 'sample'")
    if not isinstance(checkpoint_sha256, str):
        raise TypeError("checkpoint_sha256 must be a string")
    digest = checkpoint_sha256.upper()
    if len(digest) != 64 or any(char not in "0123456789ABCDEF" for char in digest):
        raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
    spec = scene.environment.spec
    if (
        ppo.num_envs != scene.expected_batch
        or ppo.observation_size != spec.observation_size
        or ppo.action_size != spec.action_size
        or tuple(ppo.action_head_sizes)
        != tuple(component.size for component in spec.action_components)
    ):
        raise ValueError("checkpoint PPO shapes do not match the environment")

    def policy(params, carry, actor_input, policy_key):
        carry, logits, _ = apply_policy(
            params,
            actor_input.observation,
            carry,
            actor_input.action_mask,
        )
        factors = (
            sample_configured_actions(policy_key, logits, ppo)[0]
            if mode == "sample"
            else greedy_arsenal_action_factors(logits)
        )
        return carry, factors

    initial_carry = jnp.zeros(
        (scene.expected_batch, ppo.recurrent_size), dtype=jnp.float32
    )
    if compile:
        runner = ArenaHandle(scene).compile_parameterized_collector(
            policy,
            _record_transition,
            steps,
            initial_carry=initial_carry,
        )
        _, trajectory = runner(policy_params, key)
    else:
        def bound(carry, actor_input, policy_key):
            return policy(policy_params, carry, actor_input, policy_key)

        _, trajectory = collect(
            scene,
            bound,
            _record_transition,
            key,
            steps,
            initial_carry=initial_carry,
        )
    return worldgen_transition_trace(
        scene,
        jax.device_get(trajectory),
        extra_parameters={
            "arena.demonstrator.kind": "recurrent_policy_checkpoint",
            "arena.demonstrator.checkpoint_sha256": digest,
            "arena.demonstrator.selection_mode": mode,
            "arena.demonstrator.action_distribution": ppo.action_distribution,
            "arena.demonstrator.encoder_size": ppo.encoder_size,
            "arena.demonstrator.recurrent_size": ppo.recurrent_size,
        },
    )


def map_worldgen_trace(
    scene: ArenaScene,
    trace: TraceSequence | TraceCorpus,
    ppo: PPOConfig,
    cloning: BehaviorCloningConfig,
    *,
    supervised_heads: tuple[str, ...] = _HEADS,
) -> MappedDemonstrations:
    """Fail closed, then arrange valid transitions for recurrent behavior cloning."""

    _validate_scene(scene)
    if isinstance(trace, TraceCorpus):
        return _merge_mapped(
            tuple(
                map_worldgen_trace(
                    scene,
                    item,
                    ppo,
                    cloning,
                    supervised_heads=supervised_heads,
                )
                for item in trace.traces
            ),
            cloning,
        )
    if not isinstance(trace, TraceSequence):
        raise TypeError("trace must be a TraceSequence or TraceCorpus")
    if trace.parameters.get("arena.trace_content_sha256") != _trace_content_sha256(
        trace
    ):
        raise ValueError("trace content SHA-256 does not match its rows")
    spec = scene.environment.spec
    expected = {
        "arena.scene_contract_sha256": scene.contract_sha256,
        "arena.observation_schema": spec.observation_schema,
        "arena.action_schema": spec.action_schema,
        "arena.observation_size": spec.observation_size,
        "arena.action_head_names": tuple(item.name for item in spec.action_components),
        "arena.action_head_sizes": tuple(item.size for item in spec.action_components),
        "arena.world_pool_semantic_sha256": scene.world_identity[
            "pool_semantic_sha256"
        ],
        "arena.policy_transfer_contract_sha256": _policy_contract_sha256(scene),
        "arena.observation_contract_sha256": scene.policy_contract[
            "observation_contract_sha256"
        ],
        "arena.action_contract_sha256": scene.policy_contract[
            "action_contract_sha256"
        ],
    }
    for name, value in expected.items():
        if trace.parameters.get(name) != value:
            raise ValueError(f"trace parameter {name!r} does not match the environment")
    head_sizes = expected["arena.action_head_sizes"]
    if (
        ppo.observation_size != spec.observation_size
        or ppo.action_size != spec.action_size
        or tuple(ppo.action_head_sizes) != head_sizes
        or ppo.num_envs != scene.expected_batch
    ):
        raise ValueError("PPO shapes do not match the ingested environment")
    if tuple(cloning.head_sizes) != head_sizes:
        raise ValueError("behavior-cloning heads do not match the environment")
    selected_heads = tuple(supervised_heads)
    if (
        not selected_heads
        or len(selected_heads) != len(set(selected_heads))
        or not set(selected_heads) <= set(_HEADS)
    ):
        raise ValueError("supervised_heads must be a unique nonempty policy subset")
    selected_indices = tuple(_HEADS.index(name) for name in selected_heads)
    trace.require_episode_structure("episode_id", "episode_start")

    observation = _component(
        trace,
        "arena.policy_observation",
        (spec.observation_size,),
        kind="observation",
        dtype="float32",
        stateful=True,
    )
    action_mask = _component(
        trace,
        "arena.action_mask",
        (spec.action_size,),
        kind="mask",
        dtype="bool",
        stateful=True,
    )
    action = _component(
        trace,
        "arena.action_factors",
        (len(head_sizes),),
        kind="action",
        dtype="int32",
        stateful=False,
    )
    if observation.next is None or action_mask.next is None:
        raise ValueError("policy observation and action mask must carry next state")
    current = np.asarray(observation.current, dtype=np.float32)
    following = np.asarray(observation.next, dtype=np.float32)
    masks = np.asarray(action_mask.current, dtype=np.bool_)
    actions = np.asarray(action.current, dtype=np.int32)
    if not np.isfinite(current).all() or not np.isfinite(following).all():
        raise ValueError("trace observations must be finite")

    valid = observation.valid & observation.next_valid
    valid &= action_mask.valid & action_mask.next_valid & action.valid
    source = np.flatnonzero(valid)
    if source.size == 0:
        raise ValueError("trace has no complete policy transitions")
    _validate_actions(actions[source], masks[source], head_sizes)

    episode_ids = tuple(dict.fromkeys(int(value) for value in trace.episode_id[source]))
    rows = [source[trace.episode_id[source] == episode] for episode in episode_ids]
    if any(np.any(np.diff(indexes) != 1) for indexes in rows):
        raise ValueError("valid trace rows must not skip within a recurrent episode")
    if any(not trace.episode_start[indexes[0]] for indexes in rows):
        raise ValueError("each mapped episode must start at an episode boundary")
    if any(np.any(trace.episode_start[indexes[1:]]) for indexes in rows):
        raise ValueError("one episode id contains multiple episode starts")

    lengths = tuple(len(indexes) for indexes in rows)
    time, lanes = max(lengths), len(lengths)
    observations = np.zeros((time, lanes, spec.observation_size), np.float32)
    padded_masks = np.ones((time, lanes, spec.action_size), np.bool_)
    factors = np.zeros((time, lanes, len(head_sizes)), np.int32)
    supervised = np.zeros_like(factors, np.bool_)
    weights = np.zeros((time, lanes), np.float32)
    starts = np.zeros((time, lanes), np.bool_)
    source_index = np.full((time, lanes), -1, np.int64)
    for lane, indexes in enumerate(rows):
        size = len(indexes)
        observations[:size, lane] = current[indexes]
        padded_masks[:size, lane] = masks[indexes]
        factors[:size, lane] = actions[indexes]
        supervised[:size, lane, selected_indices] = True
        weights[:size, lane] = 1.0
        starts[:size, lane] = trace.episode_start[indexes]
        source_index[:size, lane] = indexes
    source_index.flags.writeable = False
    source_trace_index = np.where(source_index >= 0, 0, -1).astype(np.int64)
    source_trace_index.flags.writeable = False
    return MappedDemonstrations(
        demonstration_batch(
            observations,
            factors,
            action_mask=padded_masks,
            supervision_mask=supervised,
            weight=weights,
            config=cloning,
        ),
        jnp.asarray(starts),
        source_index,
        source_trace_index,
        lengths,
        selected_heads,
        (trace.identity_sha256,),
    )


def _merge_mapped(
    items: tuple[MappedDemonstrations, ...], cloning: BehaviorCloningConfig
) -> MappedDemonstrations:
    time = max(item.batch.observation.shape[0] for item in items)
    lanes = sum(item.batch.observation.shape[1] for item in items)
    observation_size = items[0].batch.observation.shape[-1]
    head_count = items[0].batch.action.shape[-1]
    action_size = items[0].batch.action_mask.shape[-1]
    observations = np.zeros((time, lanes, observation_size), np.float32)
    actions = np.zeros((time, lanes, head_count), np.int32)
    masks = np.ones((time, lanes, action_size), np.bool_)
    supervised = np.zeros((time, lanes, head_count), np.bool_)
    weights = np.zeros((time, lanes), np.float32)
    starts = np.zeros((time, lanes), np.bool_)
    source_index = np.full((time, lanes), -1, np.int64)
    source_trace_index = np.full((time, lanes), -1, np.int64)
    cursor = 0
    for trace_index, item in enumerate(items):
        item_time, item_lanes = item.batch.observation.shape[:2]
        target = slice(cursor, cursor + item_lanes)
        observations[:item_time, target] = item.batch.observation
        actions[:item_time, target] = item.batch.action
        masks[:item_time, target] = item.batch.action_mask
        supervised[:item_time, target] = item.batch.supervision_mask
        weights[:item_time, target] = item.batch.weight
        starts[:item_time, target] = item.episode_start
        source_index[:item_time, target] = item.source_index
        valid = item.source_index >= 0
        source_trace_index[:item_time, target] = np.where(valid, trace_index, -1)
        cursor += item_lanes
    source_index.flags.writeable = False
    source_trace_index.flags.writeable = False
    return MappedDemonstrations(
        demonstration_batch(
            observations,
            actions,
            action_mask=masks,
            supervision_mask=supervised,
            weight=weights,
            config=cloning,
        ),
        jnp.asarray(starts),
        source_index,
        source_trace_index,
        tuple(length for item in items for length in item.lengths),
        items[0].supervised_heads,
        tuple(identity for item in items for identity in item.trace_identity_sha256),
    )


def _teacher(carry: Any, actor_input: Any, _key: jax.Array):
    observation, mask = actor_input.legal_observation, actor_input.action_mask
    visible = field(observation.base.combat_f32, "combat_f32", "target_visible") > 0.5
    distance = field(
        observation.base.combat_f32,
        "combat_f32",
        "visible_target_planar_distance",
    )
    requested = jnp.where(visible, jnp.where(distance > 0.15, _APPROACH, _FACE), _IDLE)
    base_mask = mask[:, _BASE_OFFSET : _BASE_OFFSET + _BASE_SIZE]
    legal = jnp.take_along_axis(base_mask, requested[:, None], axis=1)[:, 0]
    ability_mask = mask[:, _ABILITY_OFFSET : _ABILITY_OFFSET + _ABILITY_SIZE]
    real_ability = ability_mask.at[:, 0].set(False)
    ability = jnp.where(
        visible & jnp.any(real_ability, axis=1), jnp.argmax(real_ability, axis=1), 0
    )
    factors = neutral_arsenal_policy_action_factors(mask.shape[0])
    factors = factors.at[:, _BASE].set(jnp.where(legal, requested, _IDLE))
    return carry, factors.at[:, _ABILITY].set(ability)


def worldgen_transition_trace(
    scene: ArenaScene,
    trajectory: tuple[Any, ...],
    *,
    extra_parameters: Mapping[str, Any] | None = None,
) -> TraceSequence:
    """Ingest time-major environment transitions into the generic trace contract."""

    _validate_scene(scene)
    if not isinstance(trajectory, (tuple, list)) or len(trajectory) != 8:
        raise ValueError("trajectory must contain eight aligned transition tensors")
    (
        observation,
        mask,
        action,
        reward,
        runtime_terminated,
        runtime_truncated,
        next_observation,
        next_mask,
    ) = map(np.asarray, trajectory)
    spec = scene.environment.spec
    done = runtime_terminated | runtime_truncated
    if done.ndim != 2 or done.shape[1] != scene.expected_batch or done.shape[0] < 1:
        raise ValueError("trajectory done must have shape [steps, scene batch]")
    leading = done.shape
    expected_shapes = (
        (observation, leading + (spec.observation_size,)),
        (mask, leading + (spec.action_size,)),
        (action, leading + (len(spec.action_components),)),
        (reward, leading),
        (runtime_terminated, leading),
        (runtime_truncated, leading),
        (next_observation, leading + (spec.observation_size,)),
        (next_mask, leading + (spec.action_size,)),
    )
    if any(value.shape != shape for value, shape in expected_shapes):
        raise ValueError("trajectory tensors do not match the environment ABI")
    if (
        observation.dtype != np.float32
        or next_observation.dtype != np.float32
        or mask.dtype != np.bool_
        or next_mask.dtype != np.bool_
        or action.dtype != np.int32
        or reward.dtype != np.float32
        or runtime_terminated.dtype != np.bool_
        or runtime_truncated.dtype != np.bool_
    ):
        raise TypeError("trajectory tensors do not use the environment ABI dtypes")
    if np.any(runtime_terminated & runtime_truncated):
        raise ValueError("trajectory cannot be both terminated and truncated")
    if not (
        np.isfinite(observation).all()
        and np.isfinite(next_observation).all()
        and np.isfinite(reward).all()
    ):
        raise ValueError("trajectory contains non-finite values")
    steps, batch = done.shape
    lengths = [
        int(hit[0] + 1) if (hit := np.flatnonzero(done[:, lane])).size else steps
        for lane in range(batch)
    ]

    def rows(value):
        return np.concatenate(
            [value[:length, lane] for lane, length in enumerate(lengths)]
        )

    terminated = rows(runtime_terminated).astype(np.bool_)
    truncated = rows(runtime_truncated).astype(np.bool_)
    offset = 0
    for length in lengths:
        final = offset + length - 1
        truncated[final] |= not (terminated[final] or truncated[final])
        offset += length
    components = (
        numeric_component(
            "arena.policy_observation",
            rows(observation),
            next=rows(next_observation),
            kind="observation",
            layer="transition",
        ),
        numeric_component(
            "arena.action_mask",
            rows(mask),
            next=rows(next_mask),
            kind="mask",
            layer="decision",
        ),
        numeric_component(
            "arena.action_factors", rows(action), kind="action", layer="decision"
        ),
        reward_component(rows(reward)),
    )
    world = scene.world_identity
    nominal_dt = float(np.asarray(scene.combat_params.nominal_dt))
    microticks = int(np.asarray(scene.combat_params.microticks))
    tick = np.concatenate([np.arange(length) for length in lengths])
    delta_seconds = np.full(sum(lengths), nominal_dt * microticks, np.float32)
    episode_id = np.concatenate(
        [np.full(length, lane, np.int64) for lane, length in enumerate(lengths)]
    )
    episode_start = np.concatenate([np.arange(length) == 0 for length in lengths])
    available = np.ones(sum(lengths), dtype=np.bool_)
    columns = {
        "tick": tick,
        "delta_seconds": delta_seconds,
        "episode_id": episode_id,
        "episode_start": episode_start,
        "terminated": terminated,
        "truncated": truncated,
        "episode_id_available": available,
        "episode_start_available": available,
        "terminated_available": available,
        "truncated_available": available,
    }
    parameters = {
        "arena.scene_contract_sha256": scene.contract_sha256,
        "arena.observation_schema": spec.observation_schema,
        "arena.action_schema": spec.action_schema,
        "arena.observation_size": spec.observation_size,
        "arena.action_head_names": tuple(item.name for item in spec.action_components),
        "arena.action_head_sizes": tuple(item.size for item in spec.action_components),
        "arena.world_generator": world["generator"],
        "arena.world_seeds": tuple(world["seeds"]),
        "arena.world_pool_semantic_sha256": world["pool_semantic_sha256"],
        "arena.policy_transfer_contract_sha256": _policy_contract_sha256(scene),
        "arena.observation_contract_sha256": scene.policy_contract[
            "observation_contract_sha256"
        ],
        "arena.action_contract_sha256": scene.policy_contract[
            "action_contract_sha256"
        ],
        "arena.trace_content_sha256": _trace_content_sha256(columns, components),
    }
    extra = dict(extra_parameters or {})
    overlap = set(parameters) & set(extra)
    if overlap:
        raise ValueError(f"trace parameters supplied twice: {sorted(overlap)}")
    parameters.update(extra)
    return TraceSequence(
        contract=TraceContract(
            tuple(component.spec for component in components), tuple(parameters)
        ),
        **columns,
        components={component.spec.name: component for component in components},
        parameters=parameters,
    )


def _trace_content_sha256(trace_or_columns, components=None) -> str:
    if isinstance(trace_or_columns, TraceSequence):
        trace = trace_or_columns
        columns = {
            name: getattr(trace, name)
            for name in (
                "tick",
                "delta_seconds",
                "episode_id",
                "episode_start",
                "terminated",
                "truncated",
                "episode_id_available",
                "episode_start_available",
                "terminated_available",
                "truncated_available",
            )
        }
        components = trace.components.values()
    else:
        columns = trace_or_columns
    digest = hashlib.sha256()
    for name, value in columns.items():
        _update_content_digest(digest, name, value)
    for component in sorted(components, key=lambda item: item.spec.name):
        for suffix, value in (
            ("current", component.current),
            ("next", component.next),
            ("valid", component.valid),
            ("next_valid", component.next_valid),
        ):
            if value is not None:
                _update_content_digest(
                    digest, f"{component.spec.name}.{suffix}", value
                )
    return digest.hexdigest().upper()


def _update_content_digest(digest, name: str, value) -> None:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": array.shape},
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))


def _component(
    trace: TraceSequence,
    name: str,
    shape: tuple[int, ...],
    *,
    kind: str,
    dtype: str,
    stateful: bool,
):
    try:
        component = trace.components[name]
    except KeyError as error:
        raise ValueError(f"trace is missing required component {name!r}") from error
    if (
        component.spec.shape != shape
        or component.spec.kind != kind
        or component.spec.dtype != dtype
        or component.spec.stateful != stateful
    ):
        raise ValueError(f"trace component {name!r} has the wrong contract")
    return component


def _record_transition(transition):
    return (
        transition.actor_input.observation,
        transition.actor_input.action_mask,
        transition.action_factors,
        transition.reward,
        transition.terminated,
        transition.truncated,
        transition.next_actor_input.observation,
        transition.next_actor_input.action_mask,
    )


def _validate_actions(actions: np.ndarray, mask: np.ndarray, sizes: tuple[int, ...]):
    offset = 0
    for head, size in enumerate(sizes):
        choice = actions[:, head]
        if np.any((choice < 0) | (choice >= size)):
            raise ValueError(f"trace action head {head} is out of range")
        if not np.all(mask[np.arange(len(actions)), offset + choice]):
            raise ValueError(f"trace action head {head} selects a masked choice")
        offset += size
    if not np.all(np.asarray(arsenal_standard_root_legal(jnp.asarray(actions)))):
        raise ValueError("trace action factors violate joint Arsenal legality")


def _validate_scene(scene: ArenaScene) -> None:
    if not isinstance(scene, ArenaScene):
        raise TypeError("scene must be an ArenaScene")
    if (
        scene.world_identity is None
        or scene.world_identity.get("generator") not in _WORLDGEN_GENERATORS
        or scene.policy_contract is None
    ):
        raise ValueError(
            "the IL smoke requires an exact WorldGen V2 world or publication"
        )


def _policy_contract_sha256(scene: ArenaScene) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(scene.policy_contract), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest().upper()


__all__ = [
    "MappedDemonstrations",
    "SCRIPTED_DEMONSTRATOR_SCHEMA",
    "WORLDGEN_CORPUS_COMPATIBILITY_KEYS",
    "WORLDGEN_PPO_PROJECTION_SCHEMA",
    "WORLDGEN_PPO_PROJECTION_SHA256",
    "capture_recurrent_worldgen_trace",
    "capture_worldgen_trace",
    "map_worldgen_trace",
    "worldgen_transition_trace",
    "worldgen_trace_corpus",
]
