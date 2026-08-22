"""JAX-resident replay environments for death-complete native NPC traces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from arena.imitation import native_concurrent_trace_sequences
from arena.jax_contract import HEAD_SPANS
from arena.training.rewards.exact_imitation import (
    ExactImitationConfig,
    exact_imitation_reward,
)
from arena.training.imitation.native_imitation import native_npc_demonstrations
from arena.training.imitation import (
    BehaviorCloningConfig,
    DemonstrationBatch,
    demonstration_batch,
)
from hytalegym.jax.combat.environment import ActionComponentSpec, EnvironmentSpec
from hytalegym.jax.training.checkpoint import load_policy_checkpoint
from hytalegym.jax.training.policy import apply_policy_sequence
from hytalegym.jax.training.ppo import PPOEnvironment
from hytalegym.jax.training.types import PPOConfig
from hytalegym.geometry import parse_geometry
from hytalegym.geometry.contract import CELL_COUNT, MAX_DETAIL_BOXES
from hytalegym.jax.world import (
    WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    geometry_state_from_numpy,
    produce_actor_world_geometry_tokens,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    encode_world_geometry_policy_tokens,
    world_geometry_shared_visual,
    world_geometry_shared_visual_manifest,
    world_geometry_shared_visual_sha256,
    world_geometry_shared_visual_size,
)
from hytalegym.worldgen import load_native_npc_trace_artifact


NATIVE_REPLAY_SCHEMA = "arena_native_behavior_replay_v12"
NATIVE_REPLAY_ACTION_SCHEMA = "arena_native_replay_factored_action_v5"
_EXACT_ABILITY_ACTION_SCHEMA = "arena_native_replay_factored_action_v3"
NATIVE_REPLAY_ABILITY_SLOT_SCHEMA = "arena_native_role_ability_slots_v2"
_COMPATIBLE_REPLAY_SCHEMAS = {
    "arena_native_behavior_replay_v4",
    "arena_native_behavior_replay_v5",
    "arena_native_behavior_replay_v6",
    "arena_native_behavior_replay_v7",
    "arena_native_behavior_replay_v8",
    "arena_native_behavior_replay_v9",
    "arena_native_behavior_replay_v10",
    "arena_native_behavior_replay_v11",
    NATIVE_REPLAY_SCHEMA,
}
NATIVE_REPLAY_BASE_OBSERVATION_SCHEMA = "arena_native_omniscient_behavior_context_v2"
NATIVE_REPLAY_OBSERVATION_SCHEMA = "arena_native_omniscient_behavior_context_v6"
NATIVE_REPLAY_OBSERVATION_SCHEMAS = frozenset(
    (
        NATIVE_REPLAY_BASE_OBSERVATION_SCHEMA,
        "arena_native_omniscient_behavior_context_v3",
        "arena_native_omniscient_behavior_context_v4",
        "arena_native_omniscient_behavior_context_v5",
        NATIVE_REPLAY_OBSERVATION_SCHEMA,
    )
)
NATIVE_REPLAY_ATTACK_BLACKBOARD_FEATURES = (
    "combat_lifecycle_available",
    "attack_pause_seconds/5",
    "attack_executing",
    "candidate_count/16",
    "candidate_overflow",
    "any_candidate_active",
    "any_candidate_triggered",
    "any_candidate_ready",
    "minimum_aiming_seconds_remaining/5",
    "maximum_charge_seconds/5",
)
_HEAD_SIZES = tuple(size for _, size in HEAD_SPANS.values())
_ABILITY_HEAD = tuple(HEAD_SPANS).index("ability_none_plus_slots")
_PRIVILEGED_OPPONENT_HEADS = (
    "base_action",
    "ability_none_plus_slots",
    "locomotion_gait_compass",
    "yaw_delta_bins",
    "pitch_delta_bins",
)
_NATIVE_VISUAL_CHUNK = 32
_NATIVE_VISUAL_EXCEPTION_CAPACITY = 128
_NATIVE_VISUAL_EXCEPTION_BOX_CAPACITY = (
    _NATIVE_VISUAL_EXCEPTION_CAPACITY * MAX_DETAIL_BOXES
)
NATIVE_REPLAY_VISUAL_SIZE = world_geometry_shared_visual_size()
NATIVE_REPLAY_VISUAL_SHA256 = world_geometry_shared_visual_sha256()


class NativeReplayTensors(NamedTuple):
    observation: Any
    expert_action: Any
    supervision_mask: Any
    basic_attack_legal: Any
    control: Any
    control_mask: Any
    valid: Any
    episode_frame_weight: Any
    head_weight: Any
    length: Any
    outcome: Any
    role: Any
    opponent_role: Any


class NativeReplayState(NamedTuple):
    episode: Any
    row: Any


class NativeReplayBatch(NamedTuple):
    observation: Any
    expert_action: Any
    supervision_mask: Any
    basic_attack_legal: Any
    control: Any
    control_mask: Any
    outcome: Any
    role: Any
    opponent_role: Any


class NativeReplayDemonstrations(NamedTuple):
    batch: DemonstrationBatch
    episode: Any
    row: Any
    outcome: Any


class NativeReplaySequence(NamedTuple):
    batch: DemonstrationBatch
    episode_start: Any
    outcome: Any
    frame_weight: Any


@dataclass(frozen=True, slots=True)
class PackedNativeReplaySequence:
    """Equivalent recurrent training rows packed into fewer accelerator lanes."""

    batch: DemonstrationBatch
    episode_start: Any
    source_episode: Any
    source_row: Any
    frame_weight: Any
    source_capacity: int
    valid_frames: int

    def report(self) -> dict[str, Any]:
        packed_capacity = int(np.asarray(self.frame_weight).size)
        return {
            "schema": "arena_native_replay_sequence_packing_v1",
            "lanes": int(np.asarray(self.frame_weight).shape[1]),
            "time_steps": int(np.asarray(self.frame_weight).shape[0]),
            "valid_frames": self.valid_frames,
            "source_capacity": self.source_capacity,
            "packed_capacity": packed_capacity,
            "padding_fraction": 1.0 - self.valid_frames / packed_capacity,
            "capacity_reduction_fraction": 1.0
            - packed_capacity / self.source_capacity,
            "recurrent_reset": "every_source_episode_start",
        }


@dataclass(frozen=True, slots=True)
class NativeReplayCorpus:
    tensors: NativeReplayTensors
    roles: tuple[str, ...]
    feature_layout: tuple[str, ...]
    source_report_sha256: str
    tensor_sha256: str
    episode_ids: tuple[str, ...] = ()
    episode_worlds: tuple[str, ...] = ()
    episode_seeds: tuple[int, ...] = ()
    observation_schema: str = NATIVE_REPLAY_OBSERVATION_SCHEMA
    worlds: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()
    worldgen_structures: tuple[str, ...] = ()
    action_schema: str = NATIVE_REPLAY_ACTION_SCHEMA
    ability_slot_layout: tuple[tuple[tuple[str, str, str, str], ...], ...] = ()
    ability_slot_contract_sha256: str = ""

    @property
    def episodes(self) -> int:
        return int(np.asarray(self.tensors.length).size)

    @property
    def rows(self) -> int:
        return int(np.asarray(self.tensors.valid).sum())

    @property
    def observation_size(self) -> int:
        return int(np.asarray(self.tensors.observation).shape[-1])


@dataclass(frozen=True, slots=True)
class NativeBehaviorPolicy:
    """Contract-checked recurrent behavior prior for native replay context."""

    params: Any
    config: PPOConfig
    checkpoint_sha256: str
    metadata: Mapping[str, Any]
    supervised_heads: tuple[str, ...]


class NativeReplaySplit(NamedTuple):
    training: NativeReplayCorpus
    heldout: NativeReplayCorpus


def build_native_replay_corpus(report_path: str | Path) -> NativeReplayCorpus:
    """Convert exact native traces into padded, outcome-neutral JAX episodes."""

    source_path = Path(report_path)
    source_bytes = source_path.read_bytes()
    report = json.loads(source_bytes)
    if report.get("schema") != "arena_native_contextual_behavior_corpus_v4":
        raise ValueError("native replay requires a causal, death-complete corpus v4")
    scenarios = tuple(report.get("scenarios", ()))
    if not scenarios or any(
        not row.get("terminated") or row.get("truncated") for row in scenarios
    ):
        raise ValueError("every replay source matchup must end in a natural death")
    roles = tuple(sorted(report["coverage"]["roles"]))
    role_index = {name: index for index, name in enumerate(roles)}
    worlds = tuple(sorted({row["scenario"]["world"] for row in scenarios}))
    fixtures = tuple(sorted({row["scenario"]["fixture"] for row in scenarios}))
    structures = tuple(
        sorted({row["scenario"]["worldgen_structure"] for row in scenarios})
    )
    ability_slots_by_role = {}
    ability_roots_by_role = {}
    episodes = []
    for scenario in scenarios:
        specification = scenario["scenario"]
        actors = tuple(scenario["actors"])
        if len(actors) != 2:
            raise ValueError("native replay matchups require exactly two actors")
        captures = tuple(
            load_native_npc_trace_artifact(
                _report_artifact_path(source_path, actor["artifact"]["path"])
            )
            for actor in actors
        )
        for capture in captures:
            layout = _native_ability_slot_layout(capture)
            previous = ability_slots_by_role.setdefault(capture.role, layout)
            if previous != layout:
                raise ValueError("native role ability-slot identity changed in corpus")
            roots = ability_roots_by_role.setdefault(
                capture.role, [set() for _ in layout]
            )
            if len(roots) != len(layout):
                raise ValueError("native role ability-slot count changed in corpus")
            for slot, identities in enumerate(
                _native_ability_slot_roots(capture, layout)
            ):
                roots[slot].update(identities)
                if len(roots[slot]) > 1:
                    raise ValueError(
                        "native role ability slot resolved to multiple interaction roots"
                    )
        traces = native_concurrent_trace_sequences(captures)
        demonstrations = tuple(
            native_npc_demonstrations(
                capture,
                _trace=trace,
                project_basic_attacks=True,
            )
            for capture, trace in zip(captures, traces, strict=True)
        )
        for index, (actor, capture, trace, demonstration) in enumerate(
            zip(actors, captures, traces, demonstrations, strict=True)
        ):
            opponent_index = 1 - index
            opponent_capture = captures[opponent_index]
            if (
                actor["opponent_role"] != opponent_capture.role
                or actors[opponent_index]["opponent_role"] != capture.role
            ):
                raise ValueError("native replay actors are not mirrored opponents")
            episode = _episode(
                actor,
                capture,
                trace,
                demonstration,
                demonstrations[opponent_index],
                specification,
                role_index,
                len(roles),
                worlds,
                fixtures,
                structures,
            )
            episode["id"] = f"{specification['name']}::{capture.role}"
            episode["world"] = specification["world"]
            episode["seed"] = int(specification["seed"])
            episodes.append(episode)
    tensors, feature_layout = _pad(episodes)
    digest = _tensor_sha256(tensors)
    ability_slot_layout = tuple(
        _resolved_ability_slot_layout(
            ability_slots_by_role[role], ability_roots_by_role[role]
        )
        for role in roles
    )
    ability_slot_contract_sha256 = native_replay_ability_slot_contract_sha256(
        roles, ability_slot_layout
    )
    return NativeReplayCorpus(
        tensors=tensors,
        roles=roles,
        feature_layout=feature_layout,
        source_report_sha256=hashlib.sha256(source_bytes).hexdigest().upper(),
        tensor_sha256=digest,
        episode_ids=tuple(row["id"] for row in episodes),
        episode_worlds=tuple(row["world"] for row in episodes),
        episode_seeds=tuple(row["seed"] for row in episodes),
        observation_schema=NATIVE_REPLAY_OBSERVATION_SCHEMA,
        worlds=worlds,
        fixtures=fixtures,
        worldgen_structures=structures,
        action_schema=NATIVE_REPLAY_ACTION_SCHEMA,
        ability_slot_layout=ability_slot_layout,
        ability_slot_contract_sha256=ability_slot_contract_sha256,
    )


def _report_artifact_path(report_path: Path, recorded: str) -> Path:
    """Resolve same-corpus absolute paths across Windows and WSL."""

    direct = Path(recorded)
    if direct.is_file():
        return direct
    parts = recorded.replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2:
        raise FileNotFoundError(recorded)
    local = report_path.parent / parts[-2] / parts[-1]
    if (
        not local.is_file()
        or local.resolve().parent.parent != report_path.parent.resolve()
    ):
        raise FileNotFoundError(recorded)
    return local


def sample_native_replay_rows(
    tensors: NativeReplayTensors,
    key: jax.Array,
    batch_size: int,
) -> NativeReplayBatch:
    """Sample actors uniformly, then one valid row uniformly per actor."""

    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    episode_key, row_key = jax.random.split(key)
    count = tensors.length.shape[0]
    episode = jax.random.randint(episode_key, (batch_size,), 0, count)
    length = tensors.length[episode]
    row = jnp.floor(jax.random.uniform(row_key, (batch_size,)) * length).astype(
        jnp.int32
    )
    return _gather(tensors, episode, row)


def native_replay_demonstrations(
    corpus: NativeReplayCorpus,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
) -> NativeReplayDemonstrations:
    """Flatten valid frames while giving each actor episode equal total weight."""

    if tuple(config.head_sizes) != _HEAD_SIZES:
        raise ValueError("native replay demonstrations require Arena's head ABI")
    tensors = corpus.tensors
    episode, row = np.nonzero(tensors.valid)
    weights = tensors.head_weight[episode, row]
    batch = demonstration_batch(
        tensors.observation[episode, row],
        tensors.expert_action[episode, row],
        supervision_mask=tensors.supervision_mask[episode, row],
        weight=weights,
        config=config,
    )
    return NativeReplayDemonstrations(
        batch,
        episode.astype(np.int32),
        row.astype(np.int32),
        tensors.outcome[episode],
    )


def native_replay_sequence(
    corpus: NativeReplayCorpus,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
    *,
    device: bool = True,
) -> NativeReplaySequence:
    """Return time-major episodes, optionally deferring accelerator transfer."""

    if tuple(config.head_sizes) != _HEAD_SIZES:
        raise ValueError("native replay sequences require Arena's head ABI")
    tensors = corpus.tensors
    weight = tensors.head_weight
    batch = NativeReplayBatch(
        tensors.observation,
        tensors.expert_action,
        tensors.supervision_mask,
        tensors.basic_attack_legal,
        tensors.control,
        tensors.control_mask,
        tensors.outcome,
        tensors.role,
        tensors.opponent_role,
    )
    action_mask = (
        np.asarray(
            _action_mask(
                jax.tree.map(jnp.asarray, batch),
                jnp.asarray(_role_action_mask(corpus)),
            )
        )
        if device
        else _numpy_action_mask(batch, _role_action_mask(corpus))
    )
    demonstrations = demonstration_batch(
        np.swapaxes(tensors.observation, 0, 1),
        np.swapaxes(tensors.expert_action, 0, 1),
        action_mask=np.swapaxes(action_mask, 0, 1),
        supervision_mask=np.swapaxes(tensors.supervision_mask, 0, 1),
        weight=np.swapaxes(weight, 0, 1),
        config=config,
        device=device,
    )
    episode_start = np.zeros(tensors.valid.T.shape, dtype=np.bool_)
    episode_start[0] = True
    return NativeReplaySequence(
        demonstrations,
        episode_start,
        tensors.outcome,
        np.swapaxes(tensors.episode_frame_weight, 0, 1),
    )


def pack_native_replay_sequence(
    sequence: NativeReplaySequence,
    lanes: int = 32,
) -> PackedNativeReplaySequence:
    """Pack complete episodes without changing recurrent or loss semantics."""

    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
        raise ValueError("native replay packing lanes must be a positive integer")
    frame_weight = np.asarray(sequence.frame_weight, dtype=np.float32)
    valid = frame_weight > 0.0
    if valid.ndim != 2 or valid.shape[1] < 1:
        raise ValueError("native replay packing requires [time, episode] rows")
    lengths = valid.sum(axis=0, dtype=np.int32)
    expected = np.arange(valid.shape[0])[:, None] < lengths[None, :]
    if np.any(lengths < 1) or not np.array_equal(valid, expected):
        raise ValueError("native replay episodes must be nonempty contiguous prefixes")
    starts = np.asarray(sequence.episode_start, dtype=np.bool_)
    expected_starts = np.zeros_like(valid)
    expected_starts[0] = True
    if not np.array_equal(starts, expected_starts):
        raise ValueError("source replay must carry one start at each episode row zero")

    lane_count = min(lanes, valid.shape[1])
    assignments: list[list[int]] = [[] for _ in range(lane_count)]
    loads = np.zeros(lane_count, dtype=np.int32)
    for episode in sorted(range(valid.shape[1]), key=lambda i: (-lengths[i], i)):
        lane = int(np.argmin(loads))
        assignments[lane].append(episode)
        loads[lane] += lengths[episode]
    for episodes in assignments:
        episodes.sort()
    time_steps = int(loads.max())

    packed = {}
    for name, source in zip(
        sequence.batch._fields,
        sequence.batch,
        strict=True,
    ):
        array = np.asarray(source)
        fill = True if name == "action_mask" else 0
        packed[name] = np.full(
            (time_steps, lane_count, *array.shape[2:]),
            fill,
            dtype=array.dtype,
        )
    packed_start = np.zeros((time_steps, lane_count), dtype=np.bool_)
    packed_episode = np.full((time_steps, lane_count), -1, dtype=np.int32)
    packed_row = np.full((time_steps, lane_count), -1, dtype=np.int32)
    packed_frame_weight = np.zeros((time_steps, lane_count), dtype=np.float32)
    for lane, episodes in enumerate(assignments):
        cursor = 0
        for episode in episodes:
            length = int(lengths[episode])
            target = slice(cursor, cursor + length)
            for name, source in zip(
                sequence.batch._fields,
                sequence.batch,
                strict=True,
            ):
                packed[name][target, lane] = np.asarray(source)[:length, episode]
            packed_start[cursor, lane] = True
            packed_episode[target, lane] = episode
            packed_row[target, lane] = np.arange(length, dtype=np.int32)
            packed_frame_weight[target, lane] = frame_weight[:length, episode]
            cursor += length

    return PackedNativeReplaySequence(
        DemonstrationBatch(*(jnp.asarray(packed[name]) for name in sequence.batch._fields)),
        jnp.asarray(packed_start),
        jnp.asarray(packed_episode),
        jnp.asarray(packed_row),
        jnp.asarray(packed_frame_weight),
        int(valid.size),
        int(valid.sum()),
    )


def native_replay_action_head_coverage(
    corpus: NativeReplayCorpus,
) -> dict[str, dict[str, Any]]:
    """Describe which factored heads have exact Java-derived labels."""

    return _action_head_coverage(corpus.tensors, corpus.action_schema)


def split_native_replay_corpus(corpus: NativeReplayCorpus) -> NativeReplaySplit:
    """Hold out one repeat per directed role/world group."""

    if not (
        len(corpus.episode_ids)
        == len(corpus.episode_worlds)
        == len(corpus.episode_seeds)
        == corpus.episodes
    ):
        raise ValueError("native replay lacks episode provenance for a heldout split")
    role = np.asarray(corpus.tensors.role)
    opponent = np.asarray(corpus.tensors.opponent_role)
    groups: dict[tuple[int, int, str], list[int]] = {}
    for index, world in enumerate(corpus.episode_worlds):
        groups.setdefault((int(role[index]), int(opponent[index]), world), []).append(
            index
        )
    if any(len(indexes) < 2 for indexes in groups.values()):
        raise ValueError("each directed role/world group needs at least two repeats")
    heldout = {
        max(
            indexes,
            key=lambda index: (corpus.episode_seeds[index], corpus.episode_ids[index]),
        )
        for indexes in groups.values()
    }
    training = [index for index in range(corpus.episodes) if index not in heldout]
    return NativeReplaySplit(
        _subset_corpus(corpus, training),
        _subset_corpus(corpus, sorted(heldout)),
    )


def make_native_replay_environment(
    corpus: NativeReplayCorpus,
    config: ExactImitationConfig = ExactImitationConfig(),
) -> PPOEnvironment:
    """Expose exact replay as a vectorized PPO environment on accelerator."""

    tensors = jax.tree.map(jnp.asarray, corpus.tensors)
    role_action_mask = jnp.asarray(_role_action_mask(corpus))
    mean_length = jnp.mean(tensors.length, dtype=jnp.float32)

    def reset(keys):
        episode = jax.vmap(lambda key: jax.random.randint(key, (), 0, corpus.episodes))(
            keys
        )
        row = jnp.zeros_like(episode, dtype=jnp.int32)
        batch = _gather(tensors, episode, row)
        return (
            NativeReplayState(episode, row),
            batch.observation,
            _action_mask(batch, role_action_mask),
        )

    def step(state, _observation, action, _keys):
        batch = _gather(tensors, state.episode, state.row)
        terms = exact_imitation_reward(
            action,
            batch.expert_action,
            batch.supervision_mask,
            jnp.ones_like(state.row, dtype=jnp.bool_),
            config,
        )
        scale = mean_length * tensors.episode_frame_weight[state.episode, state.row]
        reward = terms.reward * scale
        done = state.row + 1 >= tensors.length[state.episode]
        next_row = jnp.minimum(state.row + 1, tensors.length[state.episode] - 1)
        next_batch = _gather(tensors, state.episode, next_row)
        return (
            NativeReplayState(state.episode, next_row),
            next_batch.observation,
            reward,
            done,
            _action_mask(next_batch, role_action_mask),
        )

    components = tuple(
        ActionComponentSpec(name, "discrete", size)
        for name, (_, size) in HEAD_SPANS.items()
    )
    return PPOEnvironment(
        reset=reset,
        step=step,
        spec=EnvironmentSpec(
            observation_schema=corpus.observation_schema,
            action_schema=corpus.action_schema,
            action_components=components,
            observation_size=corpus.observation_size,
            action_size=sum(_HEAD_SIZES),
        ),
    )


def native_replay_ppo_config(
    corpus: NativeReplayCorpus,
    *,
    num_envs: int,
    **overrides: Any,
) -> PPOConfig:
    environment = make_native_replay_environment(corpus)
    return PPOConfig.from_environment_spec(
        environment.spec,
        num_envs=num_envs,
        action_distribution="arsenal_standard_root_v1",
        **overrides,
    )


def write_native_replay_corpus(
    corpus: NativeReplayCorpus,
    output: str | Path,
) -> dict[str, Any]:
    """Persist portable NumPy tensors plus a content-pinned manifest."""

    if corpus.observation_schema != NATIVE_REPLAY_OBSERVATION_SCHEMA:
        raise ValueError("portable replay v12 stores the omniscient source view only")
    if corpus.action_schema != NATIVE_REPLAY_ACTION_SCHEMA:
        raise ValueError("portable replay v12 requires the current action projection")
    _require_native_visual_contract(corpus)
    if _tensor_sha256(corpus.tensors) != corpus.tensor_sha256:
        raise ValueError("native replay tensor identity is stale")
    if not (
        len(corpus.episode_ids)
        == len(corpus.episode_worlds)
        == len(corpus.episode_seeds)
        == corpus.episodes
    ):
        raise ValueError("replay v12 requires provenance for every episode")
    if (
        len(corpus.ability_slot_layout) != len(corpus.roles)
        or corpus.ability_slot_contract_sha256
        != native_replay_ability_slot_contract_sha256(
            corpus.roles, corpus.ability_slot_layout
        )
    ):
        raise ValueError("replay v12 requires a pinned role ability-slot layout")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **corpus.tensors._asdict())
    manifest = {
        "schema": NATIVE_REPLAY_SCHEMA,
        "observation_schema": corpus.observation_schema,
        "action_schema": corpus.action_schema,
        "combat_projection": {
            "decision": "causally_proven_attack_onset",
            "factor": "ability_none_plus_slots",
            "encoding": "zero_none_one_basic_attack",
            "authored_root": "retained_as_trace_context_not_a_fundamental_label",
            "ongoing_execution": "none_not_repeated_input",
            "observation_boundary": "blackboard_before_native_policy_decision",
            "legality": (
                "Java attack pause and active-action state suppress onset; a causal "
                "start receipt remains legal at its decision boundary"
            ),
        },
        "ability_slot_schema": NATIVE_REPLAY_ABILITY_SLOT_SCHEMA,
        "ability_slot_contract_sha256": corpus.ability_slot_contract_sha256,
        "ability_slot_layout": _ability_slot_manifest(
            corpus.roles, corpus.ability_slot_layout
        ),
        "source_report_sha256": corpus.source_report_sha256,
        "tensor_sha256": corpus.tensor_sha256,
        "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest().upper(),
        "episodes": corpus.episodes,
        "rows": corpus.rows,
        "observation_size": corpus.observation_size,
        "roles": list(corpus.roles),
        "feature_layout": list(corpus.feature_layout),
        "episode_ids": list(corpus.episode_ids),
        "episode_worlds": list(corpus.episode_worlds),
        "episode_seeds": list(corpus.episode_seeds),
        "worlds": list(corpus.worlds),
        "fixtures": list(corpus.fixtures),
        "worldgen_structures": list(corpus.worldgen_structures),
        "action_head_coverage": _action_head_coverage(
            corpus.tensors, corpus.action_schema
        ),
        "sampling": "uniform actor episode; natural-frame PPO and per-head class-balanced BC",
        "outcome_use": "provenance only; absent from observation and reward",
        "terminal": "last native frame before the Java death boundary",
        "privileged_context": {
            "attack_blackboard": {
                "features": list(NATIVE_REPLAY_ATTACK_BLACKBOARD_FEATURES),
                "boundary": "authoritative_pre_native_policy_decision",
                "legacy_capture": "lifecycle_unavailable_but_candidates_remain_causal",
                "post_decision_fields": "excluded",
            },
            "opponent_action": {
                "heads": list(_PRIVILEGED_OPPONENT_HEADS),
                "alignment": "latest_native_control_tick_strictly_before_actor_tick",
                "same_tick": "excluded_to_avoid_entity_iteration_order_leakage",
                "missing": "zero_with_independent_availability",
                "includes": [
                    "normalized_native_control",
                    "control_mask_bits",
                    "exact_action_one_hot",
                    "head_availability",
                    "attack_lifecycle",
                    "row_availability_and_age",
                ],
            },
            "world": {
                "worlds": list(corpus.worlds),
                "fixtures": list(corpus.fixtures),
                "worldgen_structures": list(corpus.worldgen_structures),
                "includes": [
                    "local_role_opaque_geometry_9x9x9",
                    "compact_exact_JAX_world_visual_tokens",
                    "world_fixture_structure_one_hot",
                    "world_seed_u32_bytes",
                    "episode_step_budget",
                ],
                "shared_visual": {
                    **world_geometry_shared_visual_manifest(),
                    "sha256": NATIVE_REPLAY_VISUAL_SHA256,
                    "source": "per_frame_exact_Java_9x9x9_geometry",
                },
            },
            "deployment": (
                "teacher_or_critic_only; actor-safe Arsenal projections omit this suffix"
            ),
        },
    }
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **manifest,
        "path": destination.as_posix(),
        "manifest": manifest_path.as_posix(),
    }


def load_native_replay_corpus(path: str | Path) -> NativeReplayCorpus:
    """Load a portable replay corpus and verify both file and tensor content."""

    source = Path(path)
    manifest = json.loads(source.with_suffix(source.suffix + ".json").read_text())
    if manifest.get("schema") not in _COMPATIBLE_REPLAY_SCHEMAS:
        raise ValueError("unsupported native replay corpus schema")
    schema = manifest.get("schema")
    current = schema == NATIVE_REPLAY_SCHEMA
    has_ability_slot_contract = schema in {
        "arena_native_behavior_replay_v8",
        "arena_native_behavior_replay_v10",
        "arena_native_behavior_replay_v11",
        NATIVE_REPLAY_SCHEMA,
    }
    if current and (
        manifest.get("observation_schema") != NATIVE_REPLAY_OBSERVATION_SCHEMA
        or manifest.get("action_schema") != NATIVE_REPLAY_ACTION_SCHEMA
        or manifest.get("combat_projection", {}).get("encoding")
        != "zero_none_one_basic_attack"
        or manifest.get("ability_slot_schema") != NATIVE_REPLAY_ABILITY_SLOT_SCHEMA
        or not manifest.get("worlds")
        or not manifest.get("fixtures")
        or not manifest.get("worldgen_structures")
    ):
        raise ValueError("native replay observation/action schema mismatch")
    if hashlib.sha256(source.read_bytes()).hexdigest().upper() != manifest.get(
        "file_sha256"
    ):
        raise ValueError("native replay file hash mismatch")
    migrated_legacy_mask = False
    with np.load(source, allow_pickle=False) as archive:
        names = set(archive.files)
        expected = set(NativeReplayTensors._fields)
        legacy = expected - {"basic_attack_legal"}
        if names == expected:
            tensors = NativeReplayTensors(
                *(np.asarray(archive[name]) for name in NativeReplayTensors._fields)
            )
        elif not current and names == legacy:
            legacy_values = {
                name: np.asarray(archive[name])
                for name in NativeReplayTensors._fields
                if name != "basic_attack_legal"
            }
            if _tensor_items_sha256(legacy_values.items()) != manifest.get(
                "tensor_sha256"
            ):
                raise ValueError("native replay tensor hash mismatch")
            legacy_values["basic_attack_legal"] = np.ones_like(
                legacy_values["valid"], dtype=np.bool_
            )
            tensors = NativeReplayTensors(
                *(legacy_values[name] for name in NativeReplayTensors._fields)
            )
            migrated_legacy_mask = True
        else:
            raise ValueError("native replay tensor members differ")
    if not migrated_legacy_mask and _tensor_sha256(tensors) != manifest.get(
        "tensor_sha256"
    ):
        raise ValueError("native replay tensor hash mismatch")
    action_schema = manifest.get(
        "action_schema", "arena_native_replay_factored_action_v2"
    )
    expected_coverage = _action_head_coverage(tensors, action_schema)
    if not current:
        expected_coverage = _legacy_action_head_coverage(expected_coverage)
    if manifest.get("action_head_coverage") != expected_coverage:
        raise ValueError("native replay action-head coverage mismatch")
    for value in tensors:
        value.flags.writeable = False
    roles = tuple(manifest["roles"])
    ability_slot_layout = (
        _ability_slot_layout_from_manifest(manifest, roles)
        if has_ability_slot_contract
        else ()
    )
    ability_slot_contract_sha256 = (
        native_replay_ability_slot_contract_sha256(roles, ability_slot_layout)
        if has_ability_slot_contract
        else ""
    )
    if has_ability_slot_contract and ability_slot_contract_sha256 != manifest.get(
        "ability_slot_contract_sha256"
    ):
        raise ValueError("native replay ability-slot contract mismatch")
    if has_ability_slot_contract and action_schema == _EXACT_ABILITY_ACTION_SCHEMA:
        _require_labelled_ability_roots(tensors, ability_slot_layout)
    corpus = NativeReplayCorpus(
        tensors=tensors,
        roles=roles,
        feature_layout=tuple(manifest["feature_layout"]),
        source_report_sha256=manifest["source_report_sha256"],
        tensor_sha256=manifest["tensor_sha256"],
        episode_ids=tuple(manifest.get("episode_ids", ())),
        episode_worlds=tuple(manifest.get("episode_worlds", ())),
        episode_seeds=tuple(manifest.get("episode_seeds", ())),
        observation_schema=manifest["observation_schema"],
        worlds=tuple(manifest.get("worlds", ())),
        fixtures=tuple(manifest.get("fixtures", ())),
        worldgen_structures=tuple(manifest.get("worldgen_structures", ())),
        action_schema=action_schema,
        ability_slot_layout=ability_slot_layout,
        ability_slot_contract_sha256=ability_slot_contract_sha256,
    )
    if (
        corpus.episodes != manifest["episodes"]
        or corpus.rows != manifest["rows"]
        or corpus.observation_size != manifest["observation_size"]
    ):
        raise ValueError("native replay manifest shape mismatch")
    if current and not (
        len(corpus.episode_ids)
        == len(corpus.episode_worlds)
        == len(corpus.episode_seeds)
        == corpus.episodes
    ):
        raise ValueError("native replay episode provenance mismatch")
    if current:
        visual = (
            manifest.get("privileged_context", {})
            .get("world", {})
            .get("shared_visual", {})
        )
        if visual.get("sha256") != NATIVE_REPLAY_VISUAL_SHA256:
            raise ValueError("native replay visual contract mismatch")
        _require_native_visual_contract(corpus)
    return corpus


def _require_native_visual_contract(corpus: NativeReplayCorpus) -> None:
    marker = f"jax_shared_world_visual[{NATIVE_REPLAY_VISUAL_SIZE}]"
    observation = np.asarray(corpus.tensors.observation)
    if (
        not corpus.feature_layout
        or corpus.feature_layout[-1] != marker
        or observation.shape[-1] < NATIVE_REPLAY_VISUAL_SIZE
    ):
        raise ValueError("native replay v12 requires exact JAX visual context")
    visual = observation[..., -NATIVE_REPLAY_VISUAL_SIZE:]
    valid = np.asarray(corpus.tensors.valid, dtype=np.bool_)
    if (
        not np.all(np.isfinite(visual))
        or np.any(np.abs(visual[valid]) > np.float32(1.0))
        or np.any(visual[~valid] != 0.0)
    ):
        raise ValueError("native replay visual rows violate the shared contract")


def load_native_behavior_policy(
    path: str | Path, corpus: NativeReplayCorpus
) -> NativeBehaviorPolicy:
    """Load a native-context prior and reject incompatible game-policy use."""

    source = Path(path)
    params, config, metadata = load_policy_checkpoint(source)
    if metadata.get("native_replay_observation_schema") != corpus.observation_schema:
        from arena.training.imitation.native_transfer import (
            NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
            native_arsenal_conditioned_replay_corpus,
            native_arsenal_shared_replay_corpus,
            native_arsenal_visual_conditioned_replay_corpus,
            native_arsenal_visual_shared_replay_corpus,
        )

        schema = metadata.get("native_replay_observation_schema")
        if schema == NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA:
            corpus = native_arsenal_shared_replay_corpus(corpus)
        elif schema == NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA:
            corpus = native_arsenal_conditioned_replay_corpus(corpus)
        elif schema == NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA:
            corpus = native_arsenal_visual_shared_replay_corpus(corpus)
        elif schema == NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA:
            corpus = native_arsenal_visual_conditioned_replay_corpus(corpus)
        else:
            raise ValueError("native behavior checkpoint observation contract mismatch")
    coverage = native_replay_action_head_coverage(corpus)
    supervised_heads = tuple(
        name for name, row in coverage.items() if row["labelled_rows"] > 0
    )
    expected = {
        "native_replay_tensor_sha256": corpus.tensor_sha256,
        "native_replay_observation_schema": corpus.observation_schema,
        "native_replay_action_schema": corpus.action_schema,
        "deployable_to_arsenal": False,
        "native_replay_supervised_heads": list(supervised_heads),
    }
    if corpus.ability_slot_contract_sha256:
        expected["native_replay_ability_slot_contract_sha256"] = (
            corpus.ability_slot_contract_sha256
        )
    if corpus.observation_schema != NATIVE_REPLAY_OBSERVATION_SCHEMA:
        from arena.training.imitation.native_transfer import (
            NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256,
            native_arsenal_conditioned_observation_sha256,
            native_arsenal_visual_conditioned_observation_sha256,
        )

        expected["native_arsenal_shared_observation_sha256"] = (
            NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256
        )
        if corpus.observation_schema in {
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
        }:
            expected["native_arsenal_visual_shared_observation_sha256"] = (
                NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256
            )
        if corpus.observation_schema in {
            NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
        }:
            expected.update(
                {
                    "native_behavior_style_roles": list(corpus.roles),
                }
            )
        if corpus.observation_schema == NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA:
            expected["native_arsenal_conditioned_observation_sha256"] = (
                native_arsenal_conditioned_observation_sha256(corpus.roles)
            )
        if (
            corpus.observation_schema
            == NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA
        ):
            expected["native_arsenal_visual_conditioned_observation_sha256"] = (
                native_arsenal_visual_conditioned_observation_sha256(corpus.roles)
            )
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("native behavior checkpoint contract mismatch")
    if (
        config.observation_size != corpus.observation_size
        or tuple(config.action_head_sizes) != _HEAD_SIZES
        or config.action_distribution != "arsenal_standard_root_v1"
    ):
        raise ValueError("native behavior checkpoint policy shape mismatch")
    return NativeBehaviorPolicy(
        params,
        config,
        hashlib.sha256(source.read_bytes()).hexdigest().upper(),
        MappingProxyType(dict(metadata)),
        supervised_heads,
    )


def native_behavior_policy_logits(
    policy: NativeBehaviorPolicy, corpus: NativeReplayCorpus
) -> jax.Array:
    """Evaluate a behavior prior on its exact actor-major replay context."""

    if policy.metadata["native_replay_observation_schema"] != corpus.observation_schema:
        from arena.training.imitation.native_transfer import (
            NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
            native_arsenal_conditioned_replay_corpus,
            native_arsenal_shared_replay_corpus,
            native_arsenal_visual_conditioned_replay_corpus,
            native_arsenal_visual_shared_replay_corpus,
        )

        schema = policy.metadata["native_replay_observation_schema"]
        projectors = {
            NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA: (
                native_arsenal_shared_replay_corpus
            ),
            NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA: (
                native_arsenal_conditioned_replay_corpus
            ),
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA: (
                native_arsenal_visual_shared_replay_corpus
            ),
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA: (
                native_arsenal_visual_conditioned_replay_corpus
            ),
        }
        if schema not in projectors:
            raise ValueError("native behavior checkpoint observation contract mismatch")
        corpus = projectors[schema](corpus)
    if policy.metadata["native_replay_tensor_sha256"] != corpus.tensor_sha256:
        raise ValueError("native behavior policy belongs to another replay corpus")
    sequence = native_replay_sequence(corpus)
    observation = jnp.asarray(sequence.batch.observation)
    carry = jnp.zeros(
        (observation.shape[1], policy.config.recurrent_size), dtype=jnp.float32
    )
    _, logits, _ = apply_policy_sequence(
        policy.params,
        observation,
        carry,
        jnp.asarray(sequence.episode_start),
        jnp.asarray(sequence.batch.action_mask),
    )
    return logits


def _episode(
    actor,
    capture,
    trace,
    demonstration,
    opponent_demonstration,
    specification,
    role_index,
    role_count,
    worlds,
    fixtures,
    structures,
):
    if actor.get("outcome") not in {"winner", "loser", "simultaneous"}:
        raise ValueError("native replay actor lacks a death-derived outcome")
    source = demonstration.source_index
    state = np.asarray(trace.resolve("native.actor_state"))[source]
    relative = np.asarray(trace.resolve("spatial.opponent_relative"))[source]
    opaque = np.asarray(trace.resolve("native.role_opaque_cell_mask"))[source]
    visible = np.asarray(
        [capture.observation[index].target_perceptible for index in source],
        dtype=np.float32,
    )[:, None]
    health = state[:, 11:12] / np.maximum(state[:, 12:13], 1.0)
    dynamics = np.concatenate(
        (
            state[:, 3:6] / 16.0,
            np.sin(state[:, 6:7]),
            np.cos(state[:, 6:7]),
            np.sin(state[:, 9:10]),
            np.cos(state[:, 9:10]),
            state[:, 10:11] / np.pi,
            health,
            state[:, 13:14],
        ),
        axis=1,
    )
    scaled_relative = relative.copy()
    scaled_relative[:, :3] /= 32.0
    scaled_relative[:, 3:6] /= 16.0
    scaled_relative[:, 6:8] /= 32.0
    role = np.zeros((source.size, role_count), dtype=np.float32)
    opponent = np.zeros_like(role)
    role[:, role_index[capture.role]] = 1.0
    opponent[:, role_index[actor["opponent_role"]]] = 1.0
    actor_context = np.stack(
        [_actor_context(capture.actor_evidence[index]) for index in source]
    )
    target_context = np.stack(
        [_actor_context(capture.target_actor_evidence[index]) for index in source]
    )
    internal_context = np.stack(
        [_internal_context(capture.internal_state[index]) for index in source]
    )
    attack_context = np.stack([_attack_context(capture, index) for index in source])
    opponent_action = _previous_opponent_context(
        demonstration,
        opponent_demonstration,
    )
    world_context = np.broadcast_to(
        _world_context(specification, worlds, fixtures, structures),
        (source.size, _world_context_size(worlds, fixtures, structures)),
    )
    visual_context = _native_visual_context(capture, source)
    observation = np.concatenate(
        (
            dynamics,
            scaled_relative,
            visible,
            opaque.astype(np.float32),
            actor_context,
            target_context,
            internal_context,
            attack_context,
            role,
            opponent,
            opponent_action,
            world_context,
            visual_context,
        ),
        axis=1,
    ).astype(np.float32)
    outcome = {"loser": -1, "simultaneous": 0, "winner": 1}[actor["outcome"]]
    actions = np.asarray(demonstration.batch.action, dtype=np.int32)
    supervised = np.asarray(demonstration.batch.supervision_mask, dtype=np.bool_)
    basic_start = supervised[:, _ABILITY_HEAD] & (actions[:, _ABILITY_HEAD] == 1)
    observations = tuple(capture.observation[index] for index in source)
    actor_evidence = tuple(capture.actor_evidence[index] for index in source)
    predecision_active = np.asarray(
        [
            row.combat_attack_executing
            or evidence["active_ability_slot"] >= 0
            or any(candidate.active for candidate in row.attack_candidates)
            for row, evidence in zip(observations, actor_evidence, strict=True)
        ],
        dtype=np.bool_,
    )
    pause_clear = np.asarray(
        [
            not row.combat_lifecycle_available
            or row.attack_pause_seconds <= 1.0e-6
            for row in observations
        ],
        dtype=np.bool_,
    )
    basic_attack_legal = (~predecision_active & pause_clear) | basic_start
    return {
        "observation": observation,
        "expert_action": actions,
        "supervision_mask": supervised,
        "basic_attack_legal": basic_attack_legal,
        "control": np.asarray(capture.control[source], dtype=np.float32),
        "control_mask": np.asarray(capture.control_mask[source], dtype=np.int32),
        "head_weight": _head_balanced_weights(
            np.asarray(demonstration.batch.action, dtype=np.int32),
            np.asarray(demonstration.batch.supervision_mask, dtype=np.bool_),
        ),
        "outcome": outcome,
        "role": role_index[capture.role],
        "opponent_role": role_index[actor["opponent_role"]],
    }


def _pad(episodes):
    count = len(episodes)
    maximum = max(row["observation"].shape[0] for row in episodes)
    features = episodes[0]["observation"].shape[1]
    heads = len(_HEAD_SIZES)
    arrays = NativeReplayTensors(
        observation=np.zeros((count, maximum, features), dtype=np.float32),
        expert_action=np.zeros((count, maximum, heads), dtype=np.int32),
        supervision_mask=np.zeros((count, maximum, heads), dtype=np.bool_),
        basic_attack_legal=np.zeros((count, maximum), dtype=np.bool_),
        control=np.zeros((count, maximum, 9), dtype=np.float32),
        control_mask=np.zeros((count, maximum), dtype=np.int32),
        valid=np.zeros((count, maximum), dtype=np.bool_),
        episode_frame_weight=np.zeros((count, maximum), dtype=np.float32),
        head_weight=np.zeros((count, maximum, heads), dtype=np.float32),
        length=np.zeros((count,), dtype=np.int32),
        outcome=np.zeros((count,), dtype=np.int8),
        role=np.zeros((count,), dtype=np.int32),
        opponent_role=np.zeros((count,), dtype=np.int32),
    )
    for index, episode in enumerate(episodes):
        length = episode["observation"].shape[0]
        arrays.observation[index, :length] = episode["observation"]
        arrays.expert_action[index, :length] = episode["expert_action"]
        arrays.supervision_mask[index, :length] = episode["supervision_mask"]
        arrays.basic_attack_legal[index, :length] = episode["basic_attack_legal"]
        arrays.control[index, :length] = episode["control"]
        arrays.control_mask[index, :length] = episode["control_mask"]
        arrays.valid[index, :length] = True
        arrays.episode_frame_weight[index, :length] = 1.0 / length
        arrays.head_weight[index, :length] = episode["head_weight"]
        arrays.length[index] = length
        arrays.outcome[index] = episode["outcome"]
        arrays.role[index] = episode["role"]
        arrays.opponent_role[index] = episode["opponent_role"]
    for value in arrays:
        value.flags.writeable = False
    layout = (
        "self_velocity_x/16",
        "self_velocity_y/16",
        "self_velocity_z/16",
        "body_yaw_sin",
        "body_yaw_cos",
        "head_yaw_sin",
        "head_yaw_cos",
        "head_pitch/pi",
        "self_health_fraction",
        "on_ground",
        "opponent_relative[11]",
        "target_perceptible",
        "role_opaque_cell_mask[729]",
        "actor_stateful_context[83]",
        "target_stateful_context[83]",
        "internal_decision_context[15]",
        "predecision_attack_blackboard[10]",
        "actor_role_one_hot",
        "opponent_role_one_hot",
        "opponent_previous_native_action_context",
        "static_world_context",
        f"jax_shared_world_visual[{NATIVE_REPLAY_VISUAL_SIZE}]",
    )
    return arrays, layout


def causal_opponent_row_indices(actor_tick, opponent_tick):
    """Align each actor row to the opponent decision strictly before it.

    Native actors may receive callbacks on different ticks.  Same-tick rows
    are deliberately excluded: feeding one callback into the other would
    depend on Java entity iteration order and would be circular in a joint JAX
    policy.  ``-1`` explicitly marks rows before the opponent's first decision.
    """

    actor = np.asarray(actor_tick, dtype=np.int64)
    opponent = np.asarray(opponent_tick, dtype=np.int64)
    if actor.ndim != 1 or opponent.ndim != 1:
        raise ValueError("native opponent alignment requires one-dimensional ticks")
    if (
        actor.size > 1
        and np.any(actor[1:] <= actor[:-1])
        or opponent.size > 1
        and np.any(opponent[1:] <= opponent[:-1])
    ):
        raise ValueError("native opponent alignment requires increasing ticks")
    return np.searchsorted(opponent, actor, side="left").astype(np.int64) - 1


def _previous_opponent_context(actor, opponent):
    index = causal_opponent_row_indices(actor.tick, opponent.tick)
    available = index >= 0
    safe = np.maximum(index, 0)
    action = np.asarray(opponent.batch.action, dtype=np.int32)[safe]
    supervised = np.asarray(opponent.batch.supervision_mask, dtype=np.bool_)[safe]
    supervised &= available[:, None]

    action_parts = []
    availability_parts = []
    names = tuple(HEAD_SPANS)
    for name in _PRIVILEGED_OPPONENT_HEADS:
        head = names.index(name)
        size = HEAD_SPANS[name][1]
        known = supervised[:, head]
        one_hot = np.eye(size, dtype=np.float32)[action[:, head]]
        action_parts.append(one_hot * known[:, None])
        availability_parts.append(known[:, None].astype(np.float32))

    control = np.asarray(opponent.native_labels.control, dtype=np.float32)[safe]
    control = control.copy()
    control[:, :3] = np.clip(control[:, :3], -1.0, 1.0)
    control[:, 3:7] = (
        (control[:, 3:7] + np.float32(np.pi)) % np.float32(2.0 * np.pi)
        - np.float32(np.pi)
    ) / np.float32(np.pi)
    control[:, 7:9] = np.clip(control[:, 7:9] / 4.0, -1.0, 1.0)
    control *= available[:, None]
    control_mask = np.asarray(opponent.native_labels.control_mask, dtype=np.int32)[safe]
    control_bits = np.stack(
        [((control_mask >> bit) & 1) for bit in range(7)], axis=-1
    ).astype(np.float32)
    control_bits *= available[:, None]
    cause = tuple(opponent.native_labels.attack_execution_cause[row] for row in safe)
    lifecycle = np.stack(
        (
            np.asarray(opponent.native_labels.attack_action_active)[safe],
            np.asarray(opponent.native_labels.attack_activation)[safe],
            np.asarray(opponent.native_labels.combat_attack)[safe],
            np.asarray([row.available for row in cause]),
            np.asarray([row.executed for row in cause]),
        ),
        axis=-1,
    ).astype(np.float32)
    lifecycle *= available[:, None]
    age = np.where(
        available, np.asarray(actor.tick) - np.asarray(opponent.tick)[safe], 0
    )
    timing = np.stack(
        (
            available.astype(np.float32),
            np.clip(age.astype(np.float32) / 30.0, 0.0, 1.0),
        ),
        axis=-1,
    )
    return np.concatenate(
        (
            control,
            control_bits,
            *action_parts,
            *availability_parts,
            lifecycle,
            timing,
        ),
        axis=-1,
    ).astype(np.float32)


@jax.jit
def _native_visual_rows(geometry, position, eye, forward, role_opaque_mask):
    source = produce_actor_world_geometry_tokens(
        geometry,
        position[:, None, :],
        eye[:, None, :],
        forward[:, None, :],
        role_opaque_mask=role_opaque_mask,
        geometry_provenance=WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
        token_capacity=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.token_capacity,
        maximum_distance=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance,
        view_sector_full_angle_radians=math.tau,
    )
    encoded = encode_world_geometry_policy_tokens(
        source,
        actor_index=0,
        config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    )
    return world_geometry_shared_visual(encoded)


def _native_visual_context(capture, source):
    """Encode Java geometry with the exact live JAX visual-token primitive."""

    indexes = np.asarray(source, dtype=np.int64)
    if indexes.ndim != 1 or indexes.size == 0:
        raise ValueError("native visual projection requires source rows")
    output = np.zeros((indexes.size, NATIVE_REPLAY_VISUAL_SIZE), dtype=np.float32)
    state = np.asarray(capture.state, dtype=np.float32)
    for start in range(0, indexes.size, _NATIVE_VISUAL_CHUNK):
        stop = min(start + _NATIVE_VISUAL_CHUNK, indexes.size)
        selected = indexes[start:stop]
        padded = np.pad(
            selected,
            (0, _NATIVE_VISUAL_CHUNK - selected.size),
            mode="edge",
        )
        rows = tuple(capture.observation[int(index)] for index in padded)
        available = np.asarray(
            [row.role_opaque_cell_mask_available for row in rows], dtype=np.bool_
        )
        frames = tuple(parse_geometry(dict(row.geometry)) for row in rows)
        geometry_available = np.asarray(
            [bool(int(frame.get("available", 0))) for frame in frames],
            dtype=np.bool_,
        )
        available &= geometry_available
        if not np.any(available):
            continue
        fallback = frames[int(np.flatnonzero(geometry_available)[0])]
        frames = tuple(
            frame if bool(int(frame.get("available", 0))) else fallback
            for frame in frames
        )
        geometry = geometry_state_from_numpy(
            frames,
            collision_full_cube_capacity=CELL_COUNT,
            collision_exception_capacity=_NATIVE_VISUAL_EXCEPTION_CAPACITY,
            collision_exception_box_capacity=(
                _NATIVE_VISUAL_EXCEPTION_BOX_CAPACITY
            ),
        )
        position = state[padded, :3]
        eye = position + np.asarray(geometry.agent_los_offset)
        yaw = state[padded, 6]
        forward = np.stack(
            (-np.sin(yaw), np.zeros_like(yaw), -np.cos(yaw)), axis=1
        ).astype(np.float32)
        role_mask = np.stack(
            [
                row.role_opaque_cell_mask
                if row.role_opaque_cell_mask_available
                else np.zeros((CELL_COUNT,), dtype=np.bool_)
                for row in rows
            ],
            axis=0,
        ).astype(np.bool_)
        projected = np.asarray(
            _native_visual_rows(
                geometry,
                position,
                eye,
                forward,
                role_mask,
            )
        ).copy()
        projected[~available] = 0.0
        output[start:stop] = projected[: selected.size]
    return output


def _world_context(specification, worlds, fixtures, structures):
    seed = int(specification["seed"])
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("native replay world seed must fit uint32")
    value = np.concatenate(
        (
            np.eye(len(worlds), dtype=np.float32)[worlds.index(specification["world"])],
            np.eye(len(fixtures), dtype=np.float32)[
                fixtures.index(specification["fixture"])
            ],
            np.eye(len(structures), dtype=np.float32)[
                structures.index(specification["worldgen_structure"])
            ],
            np.asarray(
                [((seed >> shift) & 0xFF) / 255.0 for shift in (0, 8, 16, 24)],
                dtype=np.float32,
            ),
            np.asarray(
                [min(max(int(specification["max_steps"]), 0), 65535) / 65535.0],
                dtype=np.float32,
            ),
        )
    )
    if value.size != _world_context_size(worlds, fixtures, structures):
        raise RuntimeError("native replay world-context width drift")
    return value


def _world_context_size(worlds, fixtures, structures):
    return len(worlds) + len(fixtures) + len(structures) + 5


def _actor_context(evidence):
    resources = np.asarray(evidence["resource_values"], dtype=np.float32)
    maxima = np.asarray(evidence["resource_maximums"], dtype=np.float32)
    resource_available = np.asarray(evidence["resource_available"], dtype=np.float32)
    resource_fraction = np.where(
        resource_available > 0.0,
        resources / np.maximum(maxima, 1.0),
        0.0,
    )
    defense_available = np.asarray(evidence["defense_available"], dtype=np.float32)
    defense = np.asarray(evidence["defense_values"], dtype=np.float32)
    defense = np.where(defense_available > 0.0, defense, 0.0)
    world_available = np.asarray(evidence["actor_world_available"], dtype=np.float32)
    expanded_world_available = world_available[[0, 1, 1, 2, 2]]
    world = np.where(
        expanded_world_available > 0.0,
        np.asarray(evidence["actor_world_values"], dtype=np.float32),
        0.0,
    )
    movement = evidence["movement_states"]
    movement_available = float(movement["available"])
    movement_bits = (
        np.asarray(
            [(movement["bits"] >> bit) & 1 for bit in range(23)], dtype=np.float32
        )
        * movement_available
    )
    motion = evidence["motion_force"]
    motion_names = (
        "legacy_external",
        "configured_applied",
        "pending_knockback",
        "projected",
    )
    motion_available = np.asarray(
        [motion[f"{name}_available"] for name in motion_names], dtype=np.float32
    )
    motion_values = np.concatenate(
        [
            np.asarray(motion[f"{name}_velocity"], dtype=np.float32) / 16.0
            for name in motion_names
        ]
    ) * np.repeat(motion_available, 3)
    maximum_health = max(float(evidence["max_health"]), 1.0)
    pose = np.asarray(
        (
            float(evidence["health"]) / maximum_health,
            np.sin(np.deg2rad(float(evidence["yaw_degrees"]))),
            np.cos(np.deg2rad(float(evidence["yaw_degrees"]))),
            float(evidence["pitch_degrees"]) / 90.0,
            (int(evidence["active_ability_slot"]) + 1) / 16.0,
        ),
        dtype=np.float32,
    )
    return np.concatenate(
        (
            pose,
            resource_fraction,
            resource_available,
            defense,
            defense_available,
            world,
            expanded_world_available,
            movement_bits,
            np.asarray((movement_available,), dtype=np.float32),
            motion_values,
            motion_available,
        )
    )


def _internal_context(state):
    maximum_speed = max(float(state.maximum_speed), 1.0)
    return np.asarray(
        (
            state.busy,
            state.transitioning,
            state.role_change_requested,
            state.terminal_action,
            state.backing_away,
            state.motion_in_progress,
            state.obstructed,
            float(state.current_speed) / maximum_speed,
            *(float(value) / maximum_speed for value in state.avoidance_steering),
            *(float(value) / maximum_speed for value in state.separation_steering),
            min(len(state.marked_targets), 16) / 16.0,
        ),
        dtype=np.float32,
    )


def _attack_context(capture, index):
    observation = capture.observation[index]
    candidates = observation.attack_candidates
    aiming = [max(float(row.aiming_seconds_remaining), 0.0) for row in candidates]
    charge = [max(float(row.charge_seconds), 0.0) for row in candidates]
    return np.asarray(
        (
            observation.combat_lifecycle_available,
            max(float(observation.attack_pause_seconds), 0.0) / 5.0,
            observation.combat_attack_executing,
            min(len(candidates), 16) / 16.0,
            observation.attack_candidate_overflow,
            any(row.active for row in candidates),
            any(row.triggered for row in candidates),
            any(row.ready for row in candidates),
            min(aiming, default=0.0) / 5.0,
            max(charge, default=0.0) / 5.0,
        ),
        dtype=np.float32,
    )


def _head_balanced_weights(action, supervised):
    weight = np.zeros_like(action, dtype=np.float32)
    for head in range(action.shape[1]):
        known = supervised[:, head]
        labels, counts = np.unique(action[known, head], return_counts=True)
        for label, count in zip(labels, counts, strict=True):
            weight[known & (action[:, head] == label), head] = 1.0 / (
                labels.size * count
            )
    return weight


def _gather(tensors, episode, row):
    return NativeReplayBatch(
        tensors.observation[episode, row],
        tensors.expert_action[episode, row],
        tensors.supervision_mask[episode, row],
        tensors.basic_attack_legal[episode, row],
        tensors.control[episode, row],
        tensors.control_mask[episode, row],
        tensors.outcome[episode],
        tensors.role[episode],
        tensors.opponent_role[episode],
    )


def _action_mask(batch, role_action_mask):
    mask = role_action_mask[batch.role]
    while mask.ndim < batch.expert_action.ndim:
        mask = jnp.expand_dims(mask, -2)
    result = jnp.broadcast_to(
        mask,
        batch.expert_action.shape[:-1] + (_HEAD_SIZES_SUM,),
    )
    if result.shape[-1] > HEAD_SPANS["ability_none_plus_slots"][0] + 1:
        basic = HEAD_SPANS["ability_none_plus_slots"][0] + 1
        result = result.at[..., basic].set(
            result[..., basic] & batch.basic_attack_legal
        )
    return result


def _numpy_action_mask(batch, role_action_mask):
    mask = np.asarray(role_action_mask)[np.asarray(batch.role)]
    while mask.ndim < np.asarray(batch.expert_action).ndim:
        mask = np.expand_dims(mask, -2)
    result = np.broadcast_to(
        mask,
        np.asarray(batch.expert_action).shape[:-1] + (_HEAD_SIZES_SUM,),
    ).copy()
    if result.shape[-1] > HEAD_SPANS["ability_none_plus_slots"][0] + 1:
        basic = HEAD_SPANS["ability_none_plus_slots"][0] + 1
        result[..., basic] &= np.asarray(batch.basic_attack_legal)
    return result


def _role_action_mask(corpus):
    mask = np.ones((len(corpus.roles), _HEAD_SIZES_SUM), dtype=np.bool_)
    if len(corpus.ability_slot_layout) != len(corpus.roles):
        return mask
    start, size = HEAD_SPANS["ability_none_plus_slots"]
    mask[:, start : start + size] = False
    mask[:, start] = True
    if corpus.action_schema == NATIVE_REPLAY_ACTION_SCHEMA:
        for role, slots in enumerate(corpus.ability_slot_layout):
            mask[role, start + 1] = any(root for _, _, root, _ in slots)
        return mask
    for role, slots in enumerate(corpus.ability_slot_layout):
        for slot, (_, _, root, _) in enumerate(slots[: size - 1], 1):
            mask[role, start + slot] = bool(root)
    return mask


_HEAD_SIZES_SUM = sum(_HEAD_SIZES)
_EXACT_NATIVE_HEADS = {
    "base_action": "java_combat_chain_start",
    "ability_none_plus_slots": "java_causal_basic_attack_onset",
    "locomotion_gait_compass": "native_body_steering",
    "yaw_delta_bins": "native_head_steering",
    "pitch_delta_bins": "native_head_steering",
}
_HEAD_ROLE_RELEVANCE = {
    "base_action": (
        "legacy_coarse_alternative",
        "not jointly supervised with the exact role-local ability root",
    ),
    "ability_none_plus_slots": (
        "combat_relevant_exercised",
        "Java activation edges and instantaneous causal starts label attack now; "
        "authored swing identity remains trace context for advanced policies",
    ),
    "guard_off_on": (
        "not_observed_in_captured_role_decisions",
        "no causal native guard request is exposed for these roles",
    ),
    "jump_off_on": (
        "not_observed_in_captured_role_decisions",
        "movement state is an effect and does not prove a jump request",
    ),
    "use_off_on": (
        "not_observed_in_captured_role_decisions",
        "no causal native use request is exposed for these roles",
    ),
    "block_primary_secondary_trigger": (
        "not_observed_in_captured_role_decisions",
        "no causal native block-interaction trigger is exposed for these roles",
    ),
    "block_none_plus_candidates": (
        "not_observed_in_captured_role_decisions",
        "no causal native block candidate is exposed for these roles",
    ),
    "locomotion_gait_compass": (
        "combat_relevant_exercised",
        "native body steering directly supplies the requested planar motion",
    ),
    "yaw_delta_bins": (
        "combat_relevant_exercised",
        "native head steering directly supplies requested yaw",
    ),
    "body_yaw_delta_bins": (
        "not_observed_in_captured_role_decisions",
        "the bridge commands body and head to one yaw and exposes only "
        "camera_delta_yaw, so no separate native body steer is captured; "
        "locomotion_gait_compass carries native body TRANSLATION, not its yaw",
    ),
    "pitch_delta_bins": (
        "combat_relevant_exercised",
        "native head steering directly supplies requested pitch",
    ),
    "hotbar_none_plus_slots": (
        "not_observed_in_captured_role_decisions",
        "the captured roles never swap weapons: their loadout is pinned to hotbar "
        "slot zero for the whole episode, so no slot-change edge exists to label",
    ),
}
if set(_HEAD_ROLE_RELEVANCE) != set(HEAD_SPANS):
    raise RuntimeError("native action-head relevance ledger differs from Arena's ABI")


def _action_head_coverage(tensors, action_schema=NATIVE_REPLAY_ACTION_SCHEMA):
    valid = np.asarray(tensors.valid, dtype=np.bool_)
    actions = np.asarray(tensors.expert_action)
    supervised = np.asarray(tensors.supervision_mask) & valid[..., None]
    result = {}
    for index, (name, (_, size)) in enumerate(HEAD_SPANS.items()):
        known = supervised[..., index]
        counts = np.bincount(actions[..., index][known], minlength=size)
        source = _EXACT_NATIVE_HEADS.get(name)
        if (
            action_schema == "arena_native_replay_factored_action_v2"
            and name == "ability_none_plus_slots"
        ):
            source = None
        result[name] = {
            "status": (
                "exact_causal_labels"
                if np.any(known)
                else "not_exercised"
                if source is not None
                else "unavailable_no_exact_native_cause"
            ),
            "source": source,
            "role_relevance": _HEAD_ROLE_RELEVANCE[name][0],
            "relevance_reason": _HEAD_ROLE_RELEVANCE[name][1],
            "training_behavior": (
                "supervise_exact_rows" if np.any(known) else "abstain_without_penalty"
            ),
            "labelled_rows": int(np.count_nonzero(known)),
            "unique_labels": int(np.count_nonzero(counts)),
            "histogram": counts.tolist(),
            "policy_mask": (
                "role_basic_attack_only; live_JAX_ability_legality_applies_cooldown"
                if action_schema == NATIVE_REPLAY_ACTION_SCHEMA
                and name == "ability_none_plus_slots"
                else "all_choices; supervision_and_reward_abstain_when_unknown"
            ),
        }
    return result


def _legacy_action_head_coverage(coverage):
    added = {"role_relevance", "relevance_reason", "training_behavior"}
    return {
        name: {key: value for key, value in row.items() if key not in added}
        for name, row in coverage.items()
    }


def _subset_corpus(corpus, indexes):
    tensors = NativeReplayTensors(
        *(np.asarray(value)[indexes].copy() for value in corpus.tensors)
    )
    for value in tensors:
        value.flags.writeable = False
    return NativeReplayCorpus(
        tensors=tensors,
        roles=corpus.roles,
        feature_layout=corpus.feature_layout,
        source_report_sha256=corpus.source_report_sha256,
        tensor_sha256=_tensor_sha256(tensors),
        episode_ids=tuple(corpus.episode_ids[index] for index in indexes),
        episode_worlds=tuple(corpus.episode_worlds[index] for index in indexes),
        episode_seeds=tuple(corpus.episode_seeds[index] for index in indexes),
        observation_schema=corpus.observation_schema,
        worlds=corpus.worlds,
        fixtures=corpus.fixtures,
        worldgen_structures=corpus.worldgen_structures,
        action_schema=corpus.action_schema,
        ability_slot_layout=corpus.ability_slot_layout,
        ability_slot_contract_sha256=corpus.ability_slot_contract_sha256,
    )


def native_replay_ability_slot_contract_sha256(roles, layouts):
    """Pin role-local Java ActionAttack slots without guessing global meaning."""

    if len(roles) != len(layouts):
        raise ValueError("native ability-slot layouts must align with roles")
    payload = {
        "schema": NATIVE_REPLAY_ABILITY_SLOT_SCHEMA,
        "roles": [
            {
                "role": role,
                "slots": [
                    {
                        "slot": index,
                        "label": slot[0],
                        "interaction_type": slot[1],
                        "interaction_id": slot[2],
                        "path": slot[3],
                    }
                    for index, slot in enumerate(layout)
                ],
            }
            for role, layout in zip(roles, layouts, strict=True)
        ],
        "factor_encoding": "choice_zero_none_else_role_local_slot_plus_one",
        "positive_edge": (
            "ActionAttack_activation_or_instant_causally_joined_chain_start"
        ),
        "ongoing_execution": "none_not_repeated_input",
    }
    return (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def _native_ability_slot_layout(capture):
    layouts = set()
    for observation in (*capture.observation, *capture.next_observation):
        if observation.attack_candidate_overflow:
            raise ValueError("native role attack candidates overflow ability capacity")
        if (
            observation.attack_candidate_count
            > HEAD_SPANS["ability_none_plus_slots"][1] - 1
        ):
            raise ValueError("native role has more attacks than Arena ability slots")
        layouts.add(
            tuple(
                (candidate.label, candidate.interaction_type, candidate.path)
                for candidate in observation.attack_candidates
            )
        )
    if len(layouts) != 1:
        raise ValueError("native role ability-slot ordering changed within a trace")
    return layouts.pop()


def _native_ability_slot_roots(capture, layout):
    roots = [set() for _ in layout]
    for observation in (*capture.observation, *capture.next_observation):
        for index, candidate in enumerate(observation.attack_candidates):
            if candidate.interaction_id:
                _add_ability_root(roots, layout, index, candidate)
    for cause in capture.attack_execution_cause:
        if cause.available and cause.executed:
            _add_ability_root(roots, layout, cause.candidate_index, cause)
    return tuple(frozenset(values) for values in roots)


def _add_ability_root(roots, layout, index, evidence):
    if not 0 <= index < len(layout):
        raise ValueError("native ability execution names an invalid candidate slot")
    if (
        evidence.interaction_type != layout[index][1]
        or evidence.path != layout[index][2]
        or not evidence.interaction_id
    ):
        raise ValueError("native ability execution disagrees with its candidate slot")
    roots[index].add((evidence.interaction_id, evidence.interaction_type))


def _resolved_ability_slot_layout(layout, roots):
    result = []
    for slot, identities in zip(layout, roots, strict=True):
        if len(identities) > 1:
            raise ValueError("native ability slot has ambiguous interaction roots")
        interaction_id = next(iter(identities))[0] if identities else ""
        result.append((slot[0], slot[1], interaction_id, slot[2]))
    return tuple(result)


def _require_labelled_ability_roots(tensors, layouts):
    actions = np.asarray(tensors.expert_action)[..., _ABILITY_HEAD]
    supervised = np.asarray(tensors.supervision_mask)[..., _ABILITY_HEAD]
    valid = np.asarray(tensors.valid, dtype=np.bool_)
    for episode, role in enumerate(np.asarray(tensors.role, dtype=np.int32)):
        choices = np.unique(actions[episode][valid[episode] & supervised[episode]])
        for choice in choices[choices > 0]:
            slot = int(choice) - 1
            if slot >= len(layouts[role]) or not layouts[role][slot][2]:
                raise ValueError("labelled native ability lacks an exact root identity")


def _ability_slot_manifest(roles, layouts):
    return [
        {
            "role": role,
            "slots": [
                {
                    "slot": index,
                    "label": slot[0],
                    "interaction_type": slot[1],
                    "interaction_id": slot[2],
                    "interaction_id_available": bool(slot[2]),
                    "path": slot[3],
                }
                for index, slot in enumerate(layout)
            ],
        }
        for role, layout in zip(roles, layouts, strict=True)
    ]


def _ability_slot_layout_from_manifest(manifest, roles):
    rows = manifest.get("ability_slot_layout")
    if not isinstance(rows, list) or len(rows) != len(roles):
        raise ValueError("native replay ability-slot layout is missing")
    result = []
    for role, row in zip(roles, rows, strict=True):
        if not isinstance(row, dict) or row.get("role") != role:
            raise ValueError("native replay ability-slot role ordering changed")
        slots = row.get("slots")
        if not isinstance(slots, list):
            raise ValueError("native replay ability slots must be a list")
        values = []
        for index, slot in enumerate(slots):
            if (
                not isinstance(slot, dict)
                or slot.get("slot") != index
                or not all(
                    isinstance(slot.get(key), str)
                    for key in (
                        "label",
                        "interaction_type",
                        "interaction_id",
                        "path",
                    )
                )
                or slot.get("interaction_id_available") != bool(slot["interaction_id"])
            ):
                raise ValueError("native replay ability-slot identity is invalid")
            values.append(
                (
                    slot["label"],
                    slot["interaction_type"],
                    slot["interaction_id"],
                    slot["path"],
                )
            )
        result.append(tuple(values))
    return tuple(result)


def _tensor_sha256(tensors):
    return _tensor_items_sha256(zip(tensors._fields, tensors, strict=True))


def _tensor_items_sha256(items):
    digest = hashlib.sha256()
    for name, value in items:
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        if array.flags.c_contiguous:
            digest.update(memoryview(array).cast("B"))
        else:
            for chunk in array:
                digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest().upper()


def native_replay_tensor_sha256(tensors: NativeReplayTensors) -> str:
    """Hash replay tensors without materializing a second byte-sized copy."""

    return _tensor_sha256(tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = write_native_replay_corpus(
        build_native_replay_corpus(arguments.source), arguments.output
    )
    print(json.dumps(manifest, sort_keys=True))


__all__ = [
    "NATIVE_REPLAY_ABILITY_SLOT_SCHEMA",
    "NATIVE_REPLAY_BASE_OBSERVATION_SCHEMA",
    "NATIVE_REPLAY_OBSERVATION_SCHEMA",
    "NATIVE_REPLAY_OBSERVATION_SCHEMAS",
    "NATIVE_REPLAY_ACTION_SCHEMA",
    "NATIVE_REPLAY_SCHEMA",
    "NATIVE_REPLAY_VISUAL_SHA256",
    "NATIVE_REPLAY_VISUAL_SIZE",
    "NativeBehaviorPolicy",
    "NativeReplayBatch",
    "NativeReplayCorpus",
    "NativeReplayDemonstrations",
    "NativeReplayState",
    "NativeReplaySplit",
    "NativeReplaySequence",
    "NativeReplayTensors",
    "PackedNativeReplaySequence",
    "build_native_replay_corpus",
    "causal_opponent_row_indices",
    "load_native_behavior_policy",
    "load_native_replay_corpus",
    "make_native_replay_environment",
    "native_replay_ppo_config",
    "native_replay_action_head_coverage",
    "native_replay_demonstrations",
    "native_behavior_policy_logits",
    "native_replay_ability_slot_contract_sha256",
    "native_replay_sequence",
    "native_replay_tensor_sha256",
    "pack_native_replay_sequence",
    "sample_native_replay_rows",
    "split_native_replay_corpus",
    "write_native_replay_corpus",
]


if __name__ == "__main__":
    main()
