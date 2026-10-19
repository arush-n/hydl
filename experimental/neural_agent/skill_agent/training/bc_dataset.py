"""Behaviour-cloning dataset from the recorded native NPC traces, on the gen-7 surface (docs/BUILD.md phase B, B2).

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

Tokens, in the agent frame of the row's NPC (``contract/surface.py``; the frame is the one
``contract/references.py`` gives)
    ``self_tokens`` float32[R, 1, 13], ``SelfToken.FEATURES`` order: health = health / max_health in [0, 1];
    stamina, mana, signature_energy and oxygen = the frame's actor_evidence ``resource_values`` over
    ``resource_maximums`` in the slots RESOURCE_STAMINA, RESOURCE_MANA, RESOURCE_SIGNATURE_ENERGY and
    RESOURCE_OXYGEN (``hytalegym/jax/combat/mechanics/schema/contract.py``), 0 where the maximum is 0, clipped to
    [0, 1] (``meter_features``); held_durability is 0.0, since the traces carry no such field; grounded =
    on_ground > 0.5; yaw_sin and yaw_cos of the body yaw wrapped to [-pi, pi]; pitch = the head pitch clipped to
    [-pi/2, pi/2]; speed = hypot(vx, vz); head_yaw_offset = the head yaw minus the body yaw, both in radians from the
    state (head_yaw, body_yaw), wrapped to [-pi, pi]; marked = 0.
    ``entity_tokens`` float32[R, 1, 16], ``EntityToken.FEATURES`` order, for the opponent: role_enemy = 1 (every
    matchup is mutually hostile); position = the opponent's offset from this NPC in its frame (forward, right,
    up); velocity = the opponent's velocity rotated into that frame, not normalised; visible = the NPC's
    observation.target_perceptible on the frame; last_seen_age = min(age, TICKS_SATURATION) / TICKS_SATURATION,
    where age is the number of frames since the target was last perceptible, counted over the whole trace (0 on a
    perceptible frame, TICKS_SATURATION before the first one; ``last_seen_ages``); health_fraction = opponent
    health / max_health in [0, 1]; marked = 1 (the instructed target).
    ``geometry_tokens`` float32[R, 64, 14] and ``geometry_mask`` bool[R, 64], ``GeometryToken.FEATURES`` order: the
    64 candidate cells nearest the NPC, nearest first (``geometry_features``; a stable sort, so ties keep the grid
    order). The grid is ``observation.geometry`` parsed by ``hytalegym.geometry.parse_geometry(dict(...))``. A cell
    is a candidate when its cell_mask is set and it is solid, fluid, climbable or damaging. The grid origin is the
    NPC's floored position (``protocol/messages.md``, geometry v5; the native producer in
    ``NativeEnvironmentSession``; on every recorded frame the origin equals floor(state x, y, z)), and cell
    (dx, dy, dz) occupies [origin + d, origin + d + 1) on each axis, so its centre is origin + (dx, dy, dz) + 0.5.
    Features: (forward, right, up) = the vector from the NPC to that centre in the agent frame; solid; climbable;
    fluid_none = 1 unless the cell has FLAG_FLUID; the fluid kinds (water, lava, mud, slime, tar) stay 0 on every
    token, because the gym publishes a runtime fluid id but no id-to-name table, so a fluid cell has fluid_none 0
    and no kind (``stats()`` counts the fluid tokens); hazard = FLAG_DAMAGING; top_walkable = solid and the cell
    above (dy + 1) is not solid, the cells above the grid counting as not solid; marked = 0. A frame whose geometry
    is unavailable has an all-False mask and ``geometry_available`` False (bool[R]).

Labels on the gen-7 layout (``surface.head_names("gen7")``, 21 heads)
    ``actions`` int32[R, 21] and ``supervised`` bool[R, 21] start at ``surface.neutral_factors("gen7")`` with
    no supervision. Four heads take the rules of arena's ``native_npc_demonstrations(capture,
    project_attack_execution=True)``, copied here and checked row for row by the tests (this module never
    imports arena): ``locomotion_gait_compass`` is the movement compass (0 standing still, 1 to 8 the eight
    45-degree sectors of the heading; see the compass note below); ``yaw_delta_bins`` and ``pitch_delta_bins`` are the look-delta bin nearest
    the requested head angle minus the current one; ``base_action`` is SKILL_ATTACK on an executed Primary
    interaction and SKILL_IDLE where the attack is available and not executed, supervised where it is available
    and (not executed or Primary). A fifth rule is ours, not arena's (``body_yaw=True``, the default since v2):
    ``body_yaw_delta_bins`` on every body-steering row is the bin nearest the requested body yaw (control-mask bit
    1) minus the current one, else the centre bin, as head steering labels yaw. Every other head stays neutral and
    unsupervised. Arena zero-fills the unlabelled cells of body_yaw_delta_bins (0 is its -45 degree bin); the
    dataset keeps the gen-7 no-op there (the centre). ``body_yaw=False`` gives exactly arena's labels.

Compass frame (v4, 2026-10-18)
    The trace's steering translation (control body_x, body_z) is a WORLD heading: on the 5,445 moving native frames
    the NPC's next velocity points along it (median 0.0 degrees apart; 93.4 % within 30). The gym's compass is
    BODY-relative (``apply_world_move_override``: choice d moves along the body angle (d - 1) * 45 degrees, resolved
    against the commanded body yaw). Arena's rule takes the world heading's sector, which is right only at body yaw 0,
    and the NPCs' median body yaw is 107 degrees, so on 90.6 % of those frames it labelled the wrong sector. Since v4 the
    label is the heading's sector in the commanded body frame: the current body yaw plus the delta this frame's
    body-yaw label decodes to (``body_frame_compass=True``, the default). ``body_frame_compass=False`` keeps arena's
    rule for the parity test.

Native jitter
    Per supervised head: the fraction of pairs of rows of one trace at consecutive ticks (t and t + 1), both
    supervised for that head, whose label changes. ``nan`` when a head has no such pair.

Storage
    ``save_npz`` and ``load_npz`` keep every field: the arrays, the per-trace tuples, ``head_names``, and the
    receipt ``{"server_version": "0.7.2", "bridge_sha256": [...], "label_rules": [...], "observation": [...],
    "features": {...}}`` (the distinct bridge hashes, sorted; the observation token sets: self, entity and geometry;
    each set's feature names in order, so a reader can tell the dataset versions apart).

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.training.bc_dataset [--root DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experimental.neural_agent.contracts import TICKS_SATURATION
from experimental.neural_agent.skill_agent.contract import surface
from experimental.neural_agent.skill_agent.training.traces import (
    CONTROL_COLUMNS,
    DEFAULT_TRACE_ROOT,
    STATE_COLUMNS,
    trace_paths,
)
from hytalegym.geometry.contract import (
    CELL_RADIUS,
    CELL_SIDE,
    FLAG_CLIMBABLE,
    FLAG_DAMAGING,
    FLAG_FLUID,
    FLAG_SOLID,
    parse_geometry,
)
from hytalegym.jax.combat.mechanics.schema.contract import (
    RESOURCE_COUNT,
    RESOURCE_MANA,
    RESOURCE_OXYGEN,
    RESOURCE_SIGNATURE_ENERGY,
    RESOURCE_STAMINA,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values
from hytalegym.jax.combat.skills import SKILL_ATTACK, SKILL_IDLE
from hytalegym.worldgen import load_native_npc_trace_artifact

SERVER_VERSION = "0.7.2"
WORLDS: tuple[str, ...] = ("flat", "hytale_generator")
MOVEMENT_EPSILON = 1.0e-6
DEFAULT_OUT = DEFAULT_TRACE_ROOT / "bc_dataset_v5.npz"  # v5: the self token's head_yaw_offset (v4: compass in the body
# frame; v3: geometry, meters, visibility; v2: body yaw; v1: arena's)
GEOMETRY_CAPACITY = 64
"""Geometry tokens per row: the candidate cells nearest the NPC (``geometry_features``)."""
SCENARIO = re.compile(r"^(?P<matchup>.+)_(?P<world>flat|hytale_generator)_r(?P<repeat>\d+)$")

HEAD_NAMES: tuple[str, ...] = surface.head_names("gen7")
SELF_FEATURES: tuple[str, ...] = surface.SelfToken.FEATURES
ENTITY_FEATURES: tuple[str, ...] = surface.EntityToken.FEATURES
GEOMETRY_FEATURES: tuple[str, ...] = surface.GeometryToken.FEATURES
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
_GEOM = {name: index for index, name in enumerate(GEOMETRY_FEATURES)}
_POSITION = [_STATE["x"], _STATE["y"], _STATE["z"]]
_VELOCITY = [_STATE["vx"], _STATE["vy"], _STATE["vz"]]
# The self token's meters, in the order stamina, mana, signature energy, oxygen (the resource slots 0, 1, 3, 6).
_RESOURCE_SLOTS = [RESOURCE_STAMINA, RESOURCE_MANA, RESOURCE_SIGNATURE_ENERGY, RESOURCE_OXYGEN]
_METER_COLUMNS = [_SELF["stamina"], _SELF["mana"], _SELF["signature_energy"], _SELF["oxygen"]]
_GEOMETRY_OFFSET = [_GEOM["forward"], _GEOM["right"], _GEOM["up"]]
# Bits of a trace's control_mask (NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT).
_BODY_TRANSLATION = 1 << 0
_BODY_YAW_REQUESTED = 1 << 1
_HEAD_YAW_REQUESTED = 1 << 3
_HEAD_PITCH_REQUESTED = 1 << 4
_BODY_STEERING = 1 << 5
_HEAD_STEERING = 1 << 6
_ROW_ARRAYS = (
    "self_tokens", "entity_tokens", "geometry_tokens", "geometry_mask", "geometry_available",
    "actions", "supervised", "tick", "trace_index",
)
_ARRAY_DTYPES = {
    "self_tokens": np.float32,
    "entity_tokens": np.float32,
    "geometry_tokens": np.float32,
    "geometry_mask": np.bool_,
    "geometry_available": np.bool_,
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
    geometry_tokens: np.ndarray
    geometry_mask: np.ndarray
    geometry_available: np.ndarray
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
            "geometry_tokens": (rows, GEOMETRY_CAPACITY, len(GEOMETRY_FEATURES)),
            "geometry_mask": (rows, GEOMETRY_CAPACITY),
            "geometry_available": (rows,),
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
        """The token mapping an encoder takes for the rows ``index``: self and entity with all-True masks of shape
        [n, 1], and the geometry tokens with their masks."""

        self_tokens, entity_tokens = self.self_tokens[index], self.entity_tokens[index]
        return {
            "self": (self_tokens, np.ones((self_tokens.shape[0], 1), dtype=bool)),
            "entity": (entity_tokens, np.ones((entity_tokens.shape[0], 1), dtype=bool)),
            "geometry": (self.geometry_tokens[index], self.geometry_mask[index]),
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
        """Rows, rows per world, the observation's coverage, and per head the supervised count and label histogram.

        ``geometry``: the rows whose frame had geometry, the rows without it, the available rows with no candidate
        cell, the mean number of geometry tokens per available row (at most ``GEOMETRY_CAPACITY``), and the fluid
        tokens (the kept tokens on fluid cells). ``visible_fraction``: the share of rows with the opponent perceptible.
        """

        worlds = self.world_of_rows()
        heads: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(HEAD_NAMES):
            labels, counts = np.unique(self.actions[self.supervised[:, index], index], return_counts=True)
            heads[name] = {
                "supervised": int(counts.sum()),
                "labels": {int(label): int(count) for label, count in zip(labels, counts, strict=True)},
            }
        available = self.geometry_available
        candidates = self.geometry_mask.sum(axis=1)
        fluid = self.geometry_mask & (self.geometry_tokens[:, :, _GEOM["fluid_none"]] == 0)
        rows = int(self.tick.shape[0])
        return {
            "rows": rows,
            "traces": len(self.trace_scenario),
            "joined_ticks": int(sum(self.trace_joined_ticks)),
            "rows_per_world": {world: int(np.count_nonzero(worlds == world)) for world in WORLDS},
            "geometry": {
                "rows_available": int(available.sum()),
                "rows_unavailable": int((~available).sum()),
                "rows_without_candidate": int((available & (candidates == 0)).sum()),
                "mean_candidates": float(candidates[available].mean()) if available.any() else float("nan"),
                "fluid_tokens": int(fluid.sum()),
            },
            "visible_fraction": float(self.entity_tokens[:, 0, _ENTITY["visible"]].mean()) if rows else float("nan"),
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
            geometry_tokens=self.geometry_tokens,
            geometry_mask=self.geometry_mask,
            geometry_available=self.geometry_available,
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
    """Arena's movement compass: 0 when not moving, else 1 + the 45-degree sector of atan2(x, -z) of the WORLD heading.

    That equals the body-relative compass the gym decodes only at body yaw 0 (``_body_compass_choice``).
    """

    angle = np.mod(np.arctan2(x, -z), 2.0 * np.pi)
    choice = (np.floor(angle / (np.pi / 4.0) + 0.5).astype(np.int32) % 8) + 1
    return np.where(moving, choice, 0)


def _body_compass_choice(x: np.ndarray, z: np.ndarray, yaw: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """The gym's movement compass: 0 when not moving, else 1 + the 45-degree sector of the world heading (x, z) seen
    from a body at ``yaw`` (radians): 1 forward, 2 forward-right, 3 right, ... 8 forward-left.

    This is the frame ``apply_world_move_override`` decodes: choice d moves along the body angle (d - 1) * 45 degrees,
    resolved against the commanded body heading (``jax/combat/runtime/step.py``). The forward and right vectors are
    ``_agent_frame``'s. At yaw 0 it equals ``_compass_choice``.
    """

    forward = -np.sin(yaw) * x - np.cos(yaw) * z
    right = np.cos(yaw) * x - np.sin(yaw) * z
    angle = np.mod(np.arctan2(right, forward), 2.0 * np.pi)
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
                 body_yaw: bool = True, body_frame_compass: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Gen-7 labels of the frames ``frames`` of ``capture``: ``(actions int32[n, 21], supervised bool[n, 21])``.

    The rules of arena's ``native_npc_demonstrations(capture, project_attack_execution=True)`` for its four heads
    (the module docstring lists them), except the compass when ``body_frame_compass``: then the heading's sector is
    taken in the commanded body frame, the frame the gym decodes it in. Every other head keeps its neutral choice,
    unsupervised.
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
    moving = translation[move] & (planar[move] > MOVEMENT_EPSILON)
    if body_frame_compass:
        # The engine resolves the compass against the commanded body yaw after this tick's body-yaw delta
        # (step.py), so the frame is the current body yaw plus the delta the body-yaw label decodes to (none when
        # that label is off, or the row requests no body yaw: the centre bin decodes to 0).
        frame_yaw = state[:, _STATE["body_yaw"]].copy()
        if body_yaw:
            body_now = state[:, _STATE["body_yaw"]]
            requested = (mask & _BODY_YAW_REQUESTED) != 0
            wanted = np.where(requested, control[:, _CONTROL["body_yaw"]], body_now)
            table = np.asarray(look_delta_values(_HEAD_SIZE["body_yaw_delta_bins"]), dtype=np.float64)
            delta = table[_signed_choice(wanted, body_now, _HEAD_SIZE["body_yaw_delta_bins"], wrap=True)]
            frame_yaw = np.where(body, body_now + np.deg2rad(delta), body_now)
        actions[move, _MOVE] = _body_compass_choice(
            control[move, _CONTROL["body_x"]], control[move, _CONTROL["body_z"]], frame_yaw[move], moving
        )
    else:
        actions[move, _MOVE] = _compass_choice(control[move, _CONTROL["body_x"]], control[move, _CONTROL["body_z"]], moving)
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
    """Self tokens [n, 13] of the NPC rows ``own_state`` and opponent entity tokens [n, 16] of ``other_state``.

    The formulas are the module docstring's, in the order of ``SelfToken.FEATURES`` and ``EntityToken.FEATURES``. The
    state holds neither the meters nor the visibility: those columns stay 0 here, and ``_trace_rows`` fills them.
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
    head_offset = own[:, _STATE["head_yaw"]] - wrapped
    self_tokens[:, _SELF["head_yaw_offset"]] = np.arctan2(np.sin(head_offset), np.cos(head_offset))

    entity_tokens = np.zeros((own.shape[0], len(ENTITY_FEATURES)))
    entity_tokens[:, _ENTITY["role_enemy"]] = 1.0
    entity_tokens[:, [_ENTITY["forward"], _ENTITY["right"], _ENTITY["up"]]] = _agent_frame(
        other[:, _POSITION] - own[:, _POSITION], body_yaw
    )
    entity_tokens[:, [_ENTITY["v_forward"], _ENTITY["v_right"], _ENTITY["v_up"]]] = _agent_frame(
        other[:, _VELOCITY], body_yaw
    )
    entity_tokens[:, _ENTITY["health_fraction"]] = np.clip(
        other[:, _STATE["health"]] / other[:, _STATE["max_health"]], 0.0, 1.0
    )
    entity_tokens[:, _ENTITY["marked"]] = 1.0
    return self_tokens, entity_tokens


def last_seen_ages(perceptible: Any) -> np.ndarray:
    """Frames since the target was last perceptible, one per frame of a trace (float64[n]).

    Frame i has age 0 when the target is perceptible on it, else the number of frames since the last perceptible frame
    before it; while no frame up to i is perceptible the age is ``TICKS_SATURATION``.
    """

    seen = np.asarray(perceptible, dtype=bool)
    positions = np.arange(seen.size)
    last = np.maximum.accumulate(np.where(seen, positions, -1))
    return np.where(last >= 0, positions - last, TICKS_SATURATION).astype(np.float64)


def _visibility(observations: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """A trace's per-frame perceptibility and the entity token's last_seen_age feature, over every frame of the trace."""

    perceptible = np.asarray([bool(item.target_perceptible) for item in observations], dtype=bool)
    return perceptible, np.minimum(last_seen_ages(perceptible), TICKS_SATURATION) / TICKS_SATURATION


def meter_features(evidence: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The self token's meters [n, 4] from the frames' actor_evidence: stamina, mana, signature energy, oxygen.

    Each is ``resource_values[slot] / resource_maximums[slot]`` clipped to [0, 1], and 0 where the maximum is 0.
    """

    values = np.asarray([row["resource_values"] for row in evidence], dtype=np.float64)
    maxima = np.asarray([row["resource_maximums"] for row in evidence], dtype=np.float64)
    values = values.reshape(len(evidence), RESOURCE_COUNT)[:, _RESOURCE_SLOTS]
    maxima = maxima.reshape(len(evidence), RESOURCE_COUNT)[:, _RESOURCE_SLOTS]
    ratio = np.zeros_like(values)
    np.divide(values, maxima, out=ratio, where=maxima != 0.0)
    return np.clip(ratio, 0.0, 1.0)


def geometry_features(geometries: Sequence[Mapping[str, Any]], states: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The geometry tokens of n frames: ``(features float32[n, K, 14], mask bool[n, K], available bool[n])``.

    ``geometries`` holds each frame's ``observation.geometry`` (a read-only mapping) and ``states`` the frames' state rows
    float64[n, 14]: the NPC's position and body yaw are the frame's own. The module docstring gives the cell rules.
    """

    count = len(geometries)
    states = np.asarray(states, dtype=np.float64).reshape(count, len(STATE_COLUMNS))
    tokens = np.zeros((count, GEOMETRY_CAPACITY, len(GEOMETRY_FEATURES)), dtype=np.float32)
    mask = np.zeros((count, GEOMETRY_CAPACITY), dtype=bool)
    available = np.zeros(count, dtype=bool)
    for row, geometry in enumerate(geometries):
        parsed = parse_geometry(dict(geometry))
        available[row] = bool(parsed["available"])
        if available[row]:
            tokens[row], mask[row] = _nearest_cells(parsed, states[row, _POSITION], states[row, _STATE["body_yaw"]])
    return tokens, mask, available


def _nearest_cells(parsed: Mapping[str, Any], position: np.ndarray, body_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """One parsed frame's ``GEOMETRY_CAPACITY`` nearest candidate cells: the feature rows and their mask."""

    present = np.asarray(parsed["cell_mask"], dtype=bool)
    flags = np.asarray(parsed["flags"], dtype=np.int64)
    solid = present & ((flags & FLAG_SOLID) != 0)
    fluid = present & ((flags & FLAG_FLUID) != 0)
    climbable = present & ((flags & FLAG_CLIMBABLE) != 0)
    damaging = present & ((flags & FLAG_DAMAGING) != 0)
    # top_walkable: solid, and the cell above (dy + 1) is not solid; the cells above the grid count as not solid.
    grid = solid.reshape(CELL_SIDE, CELL_SIDE, CELL_SIDE)
    above = np.zeros_like(grid)
    above[:, :-1, :] = grid[:, 1:, :]
    walkable = (grid & ~above).reshape(-1)
    candidates = np.flatnonzero(solid | fluid | climbable | damaging)
    cells = np.stack(np.unravel_index(candidates, (CELL_SIDE, CELL_SIDE, CELL_SIDE)), axis=1) - CELL_RADIUS
    offsets = np.asarray(parsed["origin"], dtype=np.float64) + cells + 0.5 - position
    nearest = np.argsort(np.linalg.norm(offsets, axis=1), kind="stable")[:GEOMETRY_CAPACITY]
    chosen = candidates[nearest]
    count = chosen.size
    features = np.zeros((GEOMETRY_CAPACITY, len(GEOMETRY_FEATURES)), dtype=np.float32)
    features[:count, _GEOMETRY_OFFSET] = _agent_frame(offsets[nearest], np.full(count, body_yaw))
    features[:count, _GEOM["solid"]] = solid[chosen]
    features[:count, _GEOM["climbable"]] = climbable[chosen]
    features[:count, _GEOM["fluid_none"]] = ~fluid[chosen]  # a fluid cell has no kind: the gym names none
    features[:count, _GEOM["hazard"]] = damaging[chosen]
    features[:count, _GEOM["top_walkable"]] = walkable[chosen]
    mask = np.zeros(GEOMETRY_CAPACITY, dtype=bool)
    mask[:count] = True
    return features, mask


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


def _trace_rows(own: Any, other: Any, index: int, body_yaw: bool = True,
                body_frame_compass: bool = True) -> tuple[dict[str, np.ndarray], int]:
    """The rows of ``own`` joined to ``other`` on tick, and the number of ticks that joined."""

    _, own_frames, other_frames = np.intersect1d(own.tick, other.tick, return_indices=True)
    native = np.asarray(own.native_control, dtype=bool)[own_frames]
    frames, partners = own_frames[native], other_frames[native]
    actions, supervised = label_frames(own, frames, body_yaw=body_yaw, body_frame_compass=body_frame_compass)
    self_tokens, entity_tokens = token_features(own.state[frames], other.state[partners])
    self_tokens[:, _METER_COLUMNS] = meter_features([own.actor_evidence[int(frame)] for frame in frames])
    perceptible, seen = _visibility(own.observation)  # the age counts every frame of the trace, not only the rows
    entity_tokens[:, _ENTITY["visible"]] = perceptible[frames]
    entity_tokens[:, _ENTITY["last_seen_age"]] = seen[frames]
    geometry, geometry_mask, geometry_available = geometry_features(
        [own.observation[int(frame)].geometry for frame in frames], own.state[frames]
    )
    rows = {
        "self_tokens": self_tokens[:, None, :],
        "entity_tokens": entity_tokens[:, None, :],
        "geometry_tokens": geometry,
        "geometry_mask": geometry_mask,
        "geometry_available": geometry_available,
        "actions": actions,
        "supervised": supervised,
        "tick": np.asarray(own.tick, dtype=np.int64)[frames],
        "trace_index": np.full(frames.size, index, dtype=np.int32),
    }
    return rows, int(own_frames.size)


def build_dataset(root: str | Path = DEFAULT_TRACE_ROOT, *, body_yaw: bool = True,
                  body_frame_compass: bool = True) -> BCDataset:
    """Build the dataset from every scenario folder under ``root`` (the module docstring gives the rules).

    ``body_yaw`` adds the body-yaw label, and ``body_frame_compass`` takes the compass in the commanded body frame
    (``label_frames``). With both False the labels are exactly arena's.
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
            rows, joined_ticks = _trace_rows(own, other, len(names), body_yaw, body_frame_compass)
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
                 "label_rules": ["arena_native_npc_demonstrations",
                                 *(["body_yaw_from_body_steering"] if body_yaw else []),
                                 *(["compass_in_commanded_body_frame"] if body_frame_compass else [])],
                 "observation": ["self", "entity", "geometry"],
                 "features": {"self": list(SELF_FEATURES), "entity": list(ENTITY_FEATURES),
                              "geometry": list(GEOMETRY_FEATURES)}},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Behaviour-cloning dataset from the recorded native NPC traces.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arena-only", action="store_true",
                        help="arena's labels only: no body-yaw label, and arena's world-frame compass")
    args = parser.parse_args(argv)
    dataset = build_dataset(args.root, body_yaw=not args.arena_only, body_frame_compass=not args.arena_only)
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
    geometry = stats["geometry"]
    lines.append(
        f"geometry: rows with geometry {geometry['rows_available']} of {stats['rows']} "
        f"(unavailable {geometry['rows_unavailable']}), without a candidate cell {geometry['rows_without_candidate']}, "
        f"mean candidates {geometry['mean_candidates']:.2f}, fluid tokens {geometry['fluid_tokens']}"
    )
    lines.append(f"visible fraction {stats['visible_fraction']:.4f}")
    lines.append("native jitter (fraction of consecutive-tick pairs whose label changes):")
    lines.extend(f"  {name}: {value:.4f}" for name, value in jitter.items())
    lines.append(f"wrote {destination}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


__all__ = [
    "DEFAULT_OUT",
    "ENTITY_FEATURES",
    "GEOMETRY_CAPACITY",
    "GEOMETRY_FEATURES",
    "HEAD_NAMES",
    "SELF_FEATURES",
    "SERVER_VERSION",
    "WORLDS",
    "BCDataset",
    "build_dataset",
    "geometry_features",
    "label_frames",
    "last_seen_ages",
    "main",
    "meter_features",
    "token_features",
]


if __name__ == "__main__":
    raise SystemExit(main())
