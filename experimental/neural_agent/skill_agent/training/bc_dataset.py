"""Behaviour-cloning dataset from the recorded native NPC traces, on the gen-7 surface (BUILD.md phase B, B2).

The 32 traces in ``agent_skills/training_runs/traces_072/`` (private) are 16 scenario folders, one per fight,
each holding the two NPCs that were recorded concurrently: ``<scenario>/<Role>.trace.msgpack.gz``. A scenario
name is ``<matchup>_<world>_r<repeat>`` with ``<world>`` ``flat`` or ``hytale_generator``. The capture's own
``world`` field is an opaque per-run instance id, so the world kind is read from the folder name. Folders that
are not scenarios (``probe/``) and the ``summary-*.json`` files are ignored, and every trace must come from
server 0.7.2 (``build_dataset`` refuses any other).

Rows
    One row per frame of a trace whose ``native_control`` is True and whose opponent (the other trace of the
    folder) has a frame at the same server tick: an inner join on ``tick``. ``trace_joined_ticks`` records how
    many ticks join for each trace. Traces are in scenario-name order, rows in tick order within a trace.

Tokens, in the agent frame of the row's NPC (``surface.py``; the frame is the one ``references.py`` gives)
    ``self_tokens`` float32[R, 1, 12], ``SelfToken.FEATURES`` order: health = health / max_health in [0, 1];
    stamina, mana, signature_energy, oxygen and held_durability are 0.0, because NPC traces carry no such
    meters; grounded = on_ground > 0.5; yaw_sin and yaw_cos of the body yaw wrapped to [-pi, pi]; pitch = the
    head pitch clipped to [-pi/2, pi/2]; speed = hypot(vx, vz); marked = 0.
    ``entity_tokens`` float32[R, 1, 16], ``EntityToken.FEATURES`` order, for the opponent: role_enemy = 1 (every
    matchup is mutually hostile); position = the opponent's offset from this NPC in its frame (forward, right,
    up); velocity = the opponent's velocity rotated into that frame, not normalised; visible = 1;
    last_seen_age = 0; health_fraction = opponent health / max_health in [0, 1]; marked = 1 (the instructed
    target).

Labels on the gen-7 layout (``surface.head_names("gen7")``, 21 heads)
    ``actions`` int32[R, 21] and ``supervised`` bool[R, 21] start at ``surface.neutral_factors("gen7")`` with
    no supervision. Four heads take the rules of arena's ``native_npc_demonstrations(capture,
    project_attack_execution=True)``, copied here and checked row for row by the tests (this module never
    imports arena): ``locomotion_gait_compass`` is the movement compass (0 standing still, 1 to 8 the eight
    45-degree sectors of the heading); ``yaw_delta_bins`` and ``pitch_delta_bins`` are the look-delta bin nearest
    the requested head angle minus the current one; ``base_action`` is SKILL_ATTACK on an executed Primary
    interaction and SKILL_IDLE where the attack is available and not executed, supervised where it is available
    and (not executed or Primary). A fifth rule is ours, not arena's (``body_yaw=True``, the default since v2):
    ``body_yaw_delta_bins`` on every body-steering row is the bin nearest the requested body yaw (control-mask bit
    1) minus the current one, else the centre bin, as head steering labels yaw. Every other head stays neutral and
    unsupervised. Arena zero-fills the unlabelled cells of body_yaw_delta_bins (0 is its -45 degree bin); the
    dataset keeps the gen-7 no-op there (the centre). ``body_yaw=False`` gives exactly arena's labels.

Native jitter
    Per supervised head: the fraction of pairs of rows of one trace at consecutive ticks (t and t + 1), both
    supervised for that head, whose label changes. ``nan`` when a head has no such pair.

Storage
    ``save_npz`` and ``load_npz`` keep every field: the arrays, the per-trace tuples, ``head_names``, and the
    receipt ``{"server_version": "0.7.2", "bridge_sha256": [...]}`` (the distinct bridge hashes, sorted).

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.training.bc_dataset [--root DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experimental.neural_agent.skill_agent import surface
from experimental.neural_agent.skill_agent.training.traces import (
    CONTROL_COLUMNS,
    DEFAULT_TRACE_ROOT,
    STATE_COLUMNS,
    trace_paths,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values
from hytalegym.jax.combat.skills import SKILL_ATTACK, SKILL_IDLE
from hytalegym.worldgen import load_native_npc_trace_artifact

SERVER_VERSION = "0.7.2"
WORLDS: tuple[str, ...] = ("flat", "hytale_generator")
MOVEMENT_EPSILON = 1.0e-6
DEFAULT_OUT = DEFAULT_TRACE_ROOT / "bc_dataset_v2.npz"  # v2: the body-yaw label (v1: arena's four heads only)
SCENARIO = re.compile(r"^(?P<matchup>.+)_(?P<world>flat|hytale_generator)_r(?P<repeat>\d+)$")

HEAD_NAMES: tuple[str, ...] = surface.head_names("gen7")
SELF_FEATURES: tuple[str, ...] = surface.SelfToken.FEATURES
ENTITY_FEATURES: tuple[str, ...] = surface.EntityToken.FEATURES
_NEUTRAL = np.asarray(surface.neutral_factors("gen7"), dtype=np.int32)
_HEAD_SIZE = dict(zip(HEAD_NAMES, surface.head_sizes("gen7"), strict=True))
_MOVE = HEAD_NAMES.index("locomotion_gait_compass")
_YAW = HEAD_NAMES.index("yaw_delta_bins")
_PITCH = HEAD_NAMES.index("pitch_delta_bins")
_BODY_YAW = HEAD_NAMES.index("body_yaw_delta_bins")
_BASE = HEAD_NAMES.index("base_action")
_STATE = {name: index for index, name in enumerate(STATE_COLUMNS)}
_CONTROL = {name: index for index, name in enumerate(CONTROL_COLUMNS)}
_SELF = {name: index for index, name in enumerate(SELF_FEATURES)}
_ENTITY = {name: index for index, name in enumerate(ENTITY_FEATURES)}
_POSITION = [_STATE["x"], _STATE["y"], _STATE["z"]]
_VELOCITY = [_STATE["vx"], _STATE["vy"], _STATE["vz"]]
# Bits of a trace's control_mask (NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT).
_BODY_TRANSLATION = 1 << 0
_BODY_YAW_REQUESTED = 1 << 1
_HEAD_YAW_REQUESTED = 1 << 3
_HEAD_PITCH_REQUESTED = 1 << 4
_BODY_STEERING = 1 << 5
_HEAD_STEERING = 1 << 6
_ROW_ARRAYS = ("self_tokens", "entity_tokens", "actions", "supervised", "tick", "trace_index")
_ARRAY_DTYPES = {
    "self_tokens": np.float32,
    "entity_tokens": np.float32,
    "actions": np.int32,
    "supervised": np.bool_,
    "tick": np.int64,
    "trace_index": np.int32,
}


@dataclass(frozen=True, eq=False)
class BCDataset:
    """The rows and the per-trace records of the dataset (the module docstring gives the schema)."""

    self_tokens: np.ndarray
    entity_tokens: np.ndarray
    actions: np.ndarray
    supervised: np.ndarray
    tick: np.ndarray
    trace_index: np.ndarray
    trace_scenario: tuple[str, ...]
    trace_role: tuple[str, ...]
    trace_world: tuple[str, ...]
    trace_repeat: tuple[int, ...]
    trace_joined_ticks: tuple[int, ...]
    head_names: tuple[str, ...]
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        for name, dtype in _ARRAY_DTYPES.items():
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        rows, traces = self.tick.shape[0], len(self.trace_scenario)
        shapes = {
            "self_tokens": (rows, 1, len(SELF_FEATURES)),
            "entity_tokens": (rows, 1, len(ENTITY_FEATURES)),
            "actions": (rows, len(HEAD_NAMES)),
            "supervised": (rows, len(HEAD_NAMES)),
            "trace_index": (rows,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"BCDataset.{name} must have shape {shape}, got {getattr(self, name).shape}")
        for name in ("trace_role", "trace_world", "trace_repeat", "trace_joined_ticks"):
            if len(getattr(self, name)) != traces:
                raise ValueError(f"BCDataset.{name} must have one entry per trace ({traces})")

    def world_of_rows(self) -> np.ndarray:
        """The world of every row, ``flat`` or ``hytale_generator``, from its trace's scenario name."""

        return np.asarray(self.trace_world)[self.trace_index]

    def split(self, held_out_world: str = "hytale_generator") -> tuple[np.ndarray, np.ndarray]:
        """Row indices ``(train, test)``. The test half is the held-out world's traces, so no trace is in both."""

        if held_out_world not in WORLDS:
            raise ValueError(f"held_out_world must be one of {WORLDS}, got {held_out_world!r}")
        worlds = self.world_of_rows()
        return np.flatnonzero(worlds != held_out_world), np.flatnonzero(worlds == held_out_world)

    def token_batch(self, index: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The token mapping an encoder takes for the rows ``index``, with all-True masks of shape [n, 1]."""

        self_tokens, entity_tokens = self.self_tokens[index], self.entity_tokens[index]
        return {
            "self": (self_tokens, np.ones((self_tokens.shape[0], 1), dtype=bool)),
            "entity": (entity_tokens, np.ones((entity_tokens.shape[0], 1), dtype=bool)),
        }

    def native_jitter(self) -> dict[str, float]:
        """Per supervised head, the fraction of consecutive-tick row pairs within a trace whose label changes."""

        same_trace = self.trace_index[:-1] == self.trace_index[1:]
        consecutive = same_trace & (self.tick[:-1] + 1 == self.tick[1:])
        both = self.supervised[:-1] & self.supervised[1:] & consecutive[:, None]
        changed = both & (self.actions[:-1] != self.actions[1:])
        pairs, changes = both.sum(axis=0), changed.sum(axis=0)
        return {
            name: float(changes[index] / pairs[index]) if pairs[index] else float("nan")
            for index, name in enumerate(HEAD_NAMES)
            if self.supervised[:, index].any()
        }

    def stats(self) -> dict[str, Any]:
        """Rows, rows per world, and per head the supervised count and the label histogram of its supervised rows."""

        worlds = self.world_of_rows()
        heads: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(HEAD_NAMES):
            labels, counts = np.unique(self.actions[self.supervised[:, index], index], return_counts=True)
            heads[name] = {
                "supervised": int(counts.sum()),
                "labels": {int(label): int(count) for label, count in zip(labels, counts, strict=True)},
            }
        return {
            "rows": int(self.tick.shape[0]),
            "traces": len(self.trace_scenario),
            "joined_ticks": int(sum(self.trace_joined_ticks)),
            "rows_per_world": {world: int(np.count_nonzero(worlds == world)) for world in WORLDS},
            "heads": heads,
        }

    def save_npz(self, path: str | Path) -> Path:
        """Write every field to a compressed ``.npz`` and return the path."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            self_tokens=self.self_tokens,
            entity_tokens=self.entity_tokens,
            actions=self.actions,
            supervised=self.supervised,
            tick=self.tick,
            trace_index=self.trace_index,
            trace_scenario=np.asarray(self.trace_scenario, dtype=str),
            trace_role=np.asarray(self.trace_role, dtype=str),
            trace_world=np.asarray(self.trace_world, dtype=str),
            trace_repeat=np.asarray(self.trace_repeat, dtype=np.int64),
            trace_joined_ticks=np.asarray(self.trace_joined_ticks, dtype=np.int64),
            head_names=np.asarray(self.head_names, dtype=str),
            receipt=np.asarray(json.dumps(self.receipt, sort_keys=True)),
        )
        return destination

    @classmethod
    def load_npz(cls, path: str | Path) -> BCDataset:
        """Read a file written by :meth:`save_npz`."""

        with np.load(Path(path), allow_pickle=False) as data:
            arrays = {name: data[name] for name in _ROW_ARRAYS}
            texts = {
                name: tuple(str(item) for item in data[name])
                for name in ("trace_scenario", "trace_role", "trace_world", "head_names")
            }
            numbers = {name: tuple(int(item) for item in data[name]) for name in ("trace_repeat", "trace_joined_ticks")}
            receipt = json.loads(str(data["receipt"]))
        return cls(**arrays, **texts, **numbers, receipt=receipt)


def _compass_choice(x: np.ndarray, z: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """The movement compass: 0 when not moving, else 1 + the 45-degree sector of atan2(x, -z)."""

    angle = np.mod(np.arctan2(x, -z), 2.0 * np.pi)
    choice = (np.floor(angle / (np.pi / 4.0) + 0.5).astype(np.int32) % 8) + 1
    return np.where(moving, choice, 0)


def _signed_choice(desired: np.ndarray, current: np.ndarray, size: int, *, wrap: bool) -> np.ndarray:
    """The look-delta bin nearest to ``desired - current`` (radians, wrapped to [-pi, pi] when ``wrap``)."""

    delta = desired - current
    if wrap:
        delta = np.arctan2(np.sin(delta), np.cos(delta))
    degrees = np.asarray(np.rad2deg(delta), dtype=np.float32)
    table = np.asarray(look_delta_values(size), dtype=np.float32)
    return np.argmin(np.abs(degrees[..., None] - table), axis=-1).astype(np.int32)


def _agent_frame(vectors: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate world vectors [n, 3] into the agent frame of ``yaw`` [n]: (forward, right, up), without normalising.

    The gym's frame (``references.direction_to_agent_frame``): forward = (-sin yaw, 0, -cos yaw),
    right = (cos yaw, 0, -sin yaw), up = y.
    """

    sine, cosine = np.sin(yaw), np.cos(yaw)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return np.stack((-sine * x - cosine * z, cosine * x - sine * z, y), axis=1)


def label_frames(capture: Any, frames: Sequence[int] | np.ndarray, *,
                 body_yaw: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Gen-7 labels of the frames ``frames`` of ``capture``: ``(actions int32[n, 21], supervised bool[n, 21])``.

    The rules of arena's ``native_npc_demonstrations(capture, project_attack_execution=True)`` for its four heads
    (the module docstring lists them). Every other head keeps its neutral choice, unsupervised.
    """

    index = np.asarray(frames, dtype=np.int64)
    actions = np.tile(_NEUTRAL, (index.size, 1))
    supervised = np.zeros((index.size, len(HEAD_NAMES)), dtype=bool)
    control = np.asarray(capture.control, dtype=np.float64)[index]
    mask = np.asarray(capture.control_mask, dtype=np.int64)[index]
    state = np.asarray(capture.state, dtype=np.float64)[index]

    # Movement: a row with the body-steering bit labels the compass when it has no translation, or only a planar one.
    body = (mask & _BODY_STEERING) != 0
    translation = (mask & _BODY_TRANSLATION) != 0
    planar = np.hypot(control[:, _CONTROL["body_x"]], control[:, _CONTROL["body_z"]])
    planar_only = np.abs(control[:, _CONTROL["body_y"]]) <= MOVEMENT_EPSILON
    move = body & (~translation | planar_only)
    actions[move, _MOVE] = _compass_choice(
        control[move, _CONTROL["body_x"]],
        control[move, _CONTROL["body_z"]],
        translation[move] & (planar[move] > MOVEMENT_EPSILON),
    )
    supervised[move, _MOVE] = True

    # Look: a head-steering row labels the bins. Without a request the desired angle is the current one (centre bin).
    head = (mask & _HEAD_STEERING) != 0
    yaw_now = state[:, _STATE["head_yaw"]]
    pitch_now = state[:, _STATE["head_pitch"]]
    yaw_wanted = np.where((mask & _HEAD_YAW_REQUESTED) != 0, control[:, _CONTROL["head_yaw"]], yaw_now)
    pitch_wanted = np.where((mask & _HEAD_PITCH_REQUESTED) != 0, control[:, _CONTROL["head_pitch"]], pitch_now)
    actions[head, _YAW] = _signed_choice(yaw_wanted[head], yaw_now[head], _HEAD_SIZE["yaw_delta_bins"], wrap=True)
    actions[head, _PITCH] = _signed_choice(
        pitch_wanted[head], pitch_now[head], _HEAD_SIZE["pitch_delta_bins"], wrap=False
    )
    supervised[head, _YAW] = True
    supervised[head, _PITCH] = True

    # Body turn (not an arena rule; added 2026-10-17 with the owner's go-ahead): a body-steering row labels the
    # body-yaw bins as head steering labels the yaw bins, since the gym decodes both heads with one signed-delta
    # codebook (observation/v3/policy/surface.py): desired = the requested body yaw (control-mask bit 1), else
    # the current body yaw (the centre bin).
    if body_yaw:
        body_now = state[:, _STATE["body_yaw"]]
        body_wanted = np.where((mask & _BODY_YAW_REQUESTED) != 0, control[:, _CONTROL["body_yaw"]], body_now)
        actions[body, _BODY_YAW] = _signed_choice(
            body_wanted[body], body_now[body], _HEAD_SIZE["body_yaw_delta_bins"], wrap=True
        )
        supervised[body, _BODY_YAW] = True

    # Base action: only a Primary interaction proves the melee attack; other authored roots are not labelled.
    causes = [capture.attack_execution_cause[int(frame)] for frame in index]
    available = np.asarray([cause.available for cause in causes], dtype=bool)
    executed = np.asarray([cause.executed for cause in causes], dtype=bool)
    primary = np.asarray([cause.interaction_type == "Primary" for cause in causes], dtype=bool)
    attack = executed & primary
    known = available & (~executed | attack)
    actions[known, _BASE] = SKILL_IDLE
    actions[attack, _BASE] = SKILL_ATTACK
    supervised[known, _BASE] = True
    return actions, supervised


def token_features(own_state: Any, other_state: Any) -> tuple[np.ndarray, np.ndarray]:
    """Self tokens [n, 12] of the NPC rows ``own_state`` and opponent entity tokens [n, 16] of ``other_state``.

    The formulas are the module docstring's, in the order of ``SelfToken.FEATURES`` and ``EntityToken.FEATURES``.
    """

    own = np.asarray(own_state, dtype=np.float64)
    other = np.asarray(other_state, dtype=np.float64)
    body_yaw = own[:, _STATE["body_yaw"]]
    wrapped = np.arctan2(np.sin(body_yaw), np.cos(body_yaw))
    self_tokens = np.zeros((own.shape[0], len(SELF_FEATURES)))
    self_tokens[:, _SELF["health"]] = np.clip(own[:, _STATE["health"]] / own[:, _STATE["max_health"]], 0.0, 1.0)
    self_tokens[:, _SELF["grounded"]] = own[:, _STATE["on_ground"]] > 0.5
    self_tokens[:, _SELF["yaw_sin"]] = np.sin(wrapped)
    self_tokens[:, _SELF["yaw_cos"]] = np.cos(wrapped)
    self_tokens[:, _SELF["pitch"]] = np.clip(own[:, _STATE["head_pitch"]], -np.pi / 2, np.pi / 2)
    self_tokens[:, _SELF["speed"]] = np.hypot(own[:, _STATE["vx"]], own[:, _STATE["vz"]])

    entity_tokens = np.zeros((own.shape[0], len(ENTITY_FEATURES)))
    entity_tokens[:, _ENTITY["role_enemy"]] = 1.0
    entity_tokens[:, [_ENTITY["forward"], _ENTITY["right"], _ENTITY["up"]]] = _agent_frame(
        other[:, _POSITION] - own[:, _POSITION], body_yaw
    )
    entity_tokens[:, [_ENTITY["v_forward"], _ENTITY["v_right"], _ENTITY["v_up"]]] = _agent_frame(
        other[:, _VELOCITY], body_yaw
    )
    entity_tokens[:, _ENTITY["visible"]] = 1.0
    # last_seen_age stays 0: the opponent is seen on the frame it is in.
    entity_tokens[:, _ENTITY["health_fraction"]] = np.clip(
        other[:, _STATE["health"]] / other[:, _STATE["max_health"]], 0.0, 1.0
    )
    entity_tokens[:, _ENTITY["marked"]] = 1.0
    return self_tokens, entity_tokens


def _scenarios(root: str | Path) -> list[tuple[str, list[Path]]]:
    """The scenario folders under ``root`` with their two traces, in name order. Other folders (``probe/``) are ignored."""

    folders: dict[str, list[Path]] = {}
    for path in trace_paths(root):
        if SCENARIO.match(path.parent.name):
            folders.setdefault(path.parent.name, []).append(path)
    for scenario, paths in folders.items():
        if len(paths) != 2:
            raise ValueError(f"scenario {scenario} has {len(paths)} traces; expected the actor and its opponent")
    return [(scenario, folders[scenario]) for scenario in sorted(folders)]


def _load_capture(path: Path) -> Any:
    """Load one trace and refuse any server version but ``SERVER_VERSION``."""

    capture = load_native_npc_trace_artifact(path)
    if capture.server_version != SERVER_VERSION:
        raise ValueError(f"{path} was recorded on server {capture.server_version!r}; expected {SERVER_VERSION!r}")
    return capture


def _trace_rows(own: Any, other: Any, index: int, body_yaw: bool = True) -> tuple[dict[str, np.ndarray], int]:
    """The rows of ``own`` joined to ``other`` on tick, and the number of ticks that joined."""

    _, own_frames, other_frames = np.intersect1d(own.tick, other.tick, return_indices=True)
    native = np.asarray(own.native_control, dtype=bool)[own_frames]
    frames, partners = own_frames[native], other_frames[native]
    actions, supervised = label_frames(own, frames, body_yaw=body_yaw)
    self_tokens, entity_tokens = token_features(own.state[frames], other.state[partners])
    rows = {
        "self_tokens": self_tokens[:, None, :],
        "entity_tokens": entity_tokens[:, None, :],
        "actions": actions,
        "supervised": supervised,
        "tick": np.asarray(own.tick, dtype=np.int64)[frames],
        "trace_index": np.full(frames.size, index, dtype=np.int32),
    }
    return rows, int(own_frames.size)


def build_dataset(root: str | Path = DEFAULT_TRACE_ROOT, *, body_yaw: bool = True) -> BCDataset:
    """Build the dataset from every scenario folder under ``root`` (the module docstring gives the rules).

    ``body_yaw`` adds the body-yaw label (``label_frames``); without it the labels are exactly arena's.
    """

    scenarios = _scenarios(root)
    if not scenarios:
        raise ValueError(f"no scenario folder with two traces under {root}")
    parts: list[dict[str, np.ndarray]] = []
    names: list[str] = []
    roles: list[str] = []
    worlds: list[str] = []
    repeats: list[int] = []
    joined: list[int] = []
    bridges: set[str] = set()
    for scenario, paths in scenarios:
        match = SCENARIO.match(scenario)
        captures = [_load_capture(path) for path in paths]
        for own_index in (0, 1):
            own, other = captures[own_index], captures[1 - own_index]
            rows, joined_ticks = _trace_rows(own, other, len(names), body_yaw)
            parts.append(rows)
            names.append(scenario)
            roles.append(own.role)
            worlds.append(match.group("world"))
            repeats.append(int(match.group("repeat")))
            joined.append(joined_ticks)
            bridges.add(own.bridge_sha256)
    arrays = {name: np.concatenate([part[name] for part in parts]) for name in _ROW_ARRAYS}
    return BCDataset(
        **arrays,
        trace_scenario=tuple(names),
        trace_role=tuple(roles),
        trace_world=tuple(worlds),
        trace_repeat=tuple(repeats),
        trace_joined_ticks=tuple(joined),
        head_names=HEAD_NAMES,
        receipt={"server_version": SERVER_VERSION, "bridge_sha256": sorted(bridges),
                 "label_rules": ["arena_native_npc_demonstrations", *(["body_yaw_from_body_steering"] if body_yaw else [])]},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Behaviour-cloning dataset from the recorded native NPC traces.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arena-only", action="store_true", help="arena's labels only: no body-yaw label")
    args = parser.parse_args(argv)
    dataset = build_dataset(args.root, body_yaw=not args.arena_only)
    destination = dataset.save_npz(args.out)
    stats = dataset.stats()
    jitter = dataset.native_jitter()
    joined = dataset.trace_joined_ticks
    lines = [
        f"traces {stats['traces']}  rows {stats['rows']}  joined ticks {stats['joined_ticks']} "
        f"(per trace min {min(joined)}, max {max(joined)})",
        "rows per world: " + "  ".join(f"{world} {count}" for world, count in stats["rows_per_world"].items()),
        "supervised heads (label histogram over the supervised rows):",
    ]
    for name, head in stats["heads"].items():
        if head["supervised"]:
            lines.append(f"  {name}: supervised {head['supervised']} labels {head['labels']}")
    lines.append("native jitter (fraction of consecutive-tick pairs whose label changes):")
    lines.extend(f"  {name}: {value:.4f}" for name, value in jitter.items())
    lines.append(f"wrote {destination}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


__all__ = [
    "DEFAULT_OUT",
    "ENTITY_FEATURES",
    "HEAD_NAMES",
    "SELF_FEATURES",
    "SERVER_VERSION",
    "WORLDS",
    "BCDataset",
    "build_dataset",
    "label_frames",
    "main",
    "token_features",
]


if __name__ == "__main__":
    raise SystemExit(main())
