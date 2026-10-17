"""Native NPC trace ingestion: recorded NpcTraceFrame batches -> arrays with a documented schema (A7).

``AGENT_BRIEF.md`` A7 and ``BUILD.md`` 3.4 (behaviour cloning warm start from native NPC traces, the
only demonstration source for now). The bridge records an NPC's transitions as ``NpcTraceFrame``
batches (``HytaleRL/hytale-plugin*/src/.../imitation/NpcTraceFrame.java``); the capture tool writes
one artifact per NPC per scenario with ``hytalegym.worldgen.write_native_npc_trace_artifact`` (a gzip
msgpack envelope around the columnar drains, validated again on load). This module reads such
artifacts into ``TraceArrays``. The live 0.7.2 recordings live in
``agent_skills/training_runs/traces_072/<scenario>/<role>.trace.msgpack.gz`` (private, never staged).

Schema of ``TraceArrays`` (N frames, one per native transition):

    tick                 int64[N]        the server tick of the frame
    delta_seconds        float32[N]      the frame's simulated duration
    state, next_state    float64[N, 14]  STATE_COLUMNS: x y z vx vy vz body_yaw body_pitch body_roll
                                         head_yaw head_pitch health max_health on_ground
    control              float64[N, 9]   CONTROL_COLUMNS: body_x body_y body_z body_yaw body_pitch
                                         head_yaw head_pitch body_turn_speed head_turn_speed
    control_mask         int32[N]        bits: 0 body translation, 1 body yaw, 2 body pitch, 3 head yaw,
                                         4 head pitch, 5 body steering present, 6 head steering present
    native_control       bool[N]         the NPC's own controller produced the control
    attack_activation    bool[N]         an attack started on this frame
    combat_attack        bool[N]         a combat attack was active
    attack_pause_seconds float32[N]
    damage_event_count   int32[N]

plus the receipt fields: scenario, role, server version, bridge sha256, world, seed, status and the
total captured count. ``seconds`` (the sum of ``delta_seconds``) and ``frames`` are the duration and
the count the A7 gate reports. The column names come from the gym's own layout strings, so a schema
change in the bridge shows up here as a different column tuple.

CLI (the inventory of a trace directory)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.training.traces [--root DIR] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hytalegym.worldgen import (
    NATIVE_NPC_TRACE_CONTROL_LAYOUT,
    NATIVE_NPC_TRACE_STATE_LAYOUT,
    load_native_npc_trace_artifact,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRACE_ROOT = ROOT / "agent_skills" / "training_runs" / "traces_072"
TRACE_SUFFIX = ".trace.msgpack.gz"
STATE_COLUMNS: tuple[str, ...] = tuple(NATIVE_NPC_TRACE_STATE_LAYOUT.split(","))
CONTROL_COLUMNS: tuple[str, ...] = tuple(NATIVE_NPC_TRACE_CONTROL_LAYOUT.split(","))
ARRAY_FIELDS: tuple[str, ...] = (
    "tick", "delta_seconds", "state", "next_state", "control", "control_mask", "native_control",
    "attack_activation", "combat_attack", "attack_pause_seconds", "damage_event_count",
)
_DTYPES = {
    "tick": np.int64, "delta_seconds": np.float32, "state": np.float64, "next_state": np.float64,
    "control": np.float64, "control_mask": np.int32, "native_control": np.bool_, "attack_activation": np.bool_,
    "combat_attack": np.bool_, "attack_pause_seconds": np.float32, "damage_event_count": np.int32,
}


@dataclass(frozen=True, eq=False)
class TraceArrays:
    """One NPC's recorded transitions as arrays (the module docstring gives the schema)."""

    scenario: str
    role: str
    server_version: str
    bridge_sha256: str
    world: str
    seed: int
    status: str
    total_captured: int
    tick: np.ndarray
    delta_seconds: np.ndarray
    state: np.ndarray
    next_state: np.ndarray
    control: np.ndarray
    control_mask: np.ndarray
    native_control: np.ndarray
    attack_activation: np.ndarray
    combat_attack: np.ndarray
    attack_pause_seconds: np.ndarray
    damage_event_count: np.ndarray

    def __post_init__(self) -> None:
        frames = int(np.asarray(self.tick).shape[0])
        for name in ARRAY_FIELDS:
            value = np.asarray(getattr(self, name), dtype=_DTYPES[name])
            if value.shape[0] != frames:
                raise ValueError(f"TraceArrays.{name} has {value.shape[0]} rows; tick has {frames}")
            object.__setattr__(self, name, value)
        for name, columns in (("state", STATE_COLUMNS), ("next_state", STATE_COLUMNS), ("control", CONTROL_COLUMNS)):
            if getattr(self, name).shape != (frames, len(columns)):
                raise ValueError(f"TraceArrays.{name} must have shape ({frames}, {len(columns)}), "
                                 f"got {getattr(self, name).shape}")

    @property
    def frames(self) -> int:
        """The number of recorded frames."""

        return int(self.tick.shape[0])

    @property
    def seconds(self) -> float:
        """The recorded duration: the sum of the frames' ``delta_seconds``."""

        return float(np.sum(self.delta_seconds, dtype=np.float64))

    def receipt(self) -> dict[str, Any]:
        """The A7 receipt row: scenario, role, frames, seconds, ticks spanned, server, bridge prefix, status."""

        span = int(self.tick[-1] - self.tick[0] + 1) if self.frames else 0
        return {"scenario": self.scenario, "role": self.role, "frames": self.frames, "seconds": round(self.seconds, 3),
                "ticks_spanned": span, "server_version": self.server_version, "bridge_sha256": self.bridge_sha256[:12],
                "status": self.status, "total_captured": self.total_captured,
                "attack_activations": int(self.attack_activation.sum()),
                "frames_with_damage": int((self.damage_event_count > 0).sum())}

    def save_npz(self, path: str | Path) -> Path:
        """Write every field to a compressed ``.npz`` (strings as 0-d arrays) and return the path."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **{item.name: np.asarray(getattr(self, item.name)) for item in fields(self)})
        return destination

    @classmethod
    def load_npz(cls, path: str | Path) -> TraceArrays:
        """Read a file written by :meth:`save_npz`."""

        with np.load(Path(path), allow_pickle=False) as data:
            values = {item.name: data[item.name] for item in fields(cls)}
        for name in ("scenario", "role", "server_version", "bridge_sha256", "world", "status"):
            values[name] = str(values[name])
        for name in ("seed", "total_captured"):
            values[name] = int(values[name])
        return cls(**values)

    def equals(self, other: TraceArrays) -> bool:
        """True when every field is equal (arrays element-wise, with their dtypes)."""

        for item in fields(self):
            mine, theirs = getattr(self, item.name), getattr(other, item.name)
            if isinstance(mine, np.ndarray):
                if mine.dtype != theirs.dtype or not np.array_equal(mine, theirs):
                    return False
            elif mine != theirs:
                return False
        return True


def from_capture(capture: Any, scenario: str = "") -> TraceArrays:
    """Arrays from a ``hytalegym.worldgen.NativeNpcTraceCapture``."""

    return TraceArrays(
        scenario=scenario, role=capture.role, server_version=capture.server_version,
        bridge_sha256=capture.bridge_sha256, world=capture.world, seed=int(capture.seed), status=capture.status,
        total_captured=int(capture.total_captured),
        **{name: getattr(capture, name) for name in ARRAY_FIELDS},
    )


def load_trace(path: str | Path) -> TraceArrays:
    """Load one artifact; the scenario is its parent directory's name."""

    location = Path(path)
    return from_capture(load_native_npc_trace_artifact(location), scenario=location.parent.name)


def trace_paths(root: str | Path = DEFAULT_TRACE_ROOT) -> list[Path]:
    """Every artifact under ``root``, sorted."""

    return sorted(Path(root).glob(f"*/*{TRACE_SUFFIX}"))


def inventory(root: str | Path = DEFAULT_TRACE_ROOT) -> list[dict[str, Any]]:
    """One receipt row per artifact under ``root``."""

    return [load_trace(path).receipt() | {"path": str(path)} for path in trace_paths(root)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory of recorded native NPC traces.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = inventory(args.root)
    if args.json:
        sys.stdout.write(json.dumps(rows, indent=1) + "\n")
    else:
        for row in rows:
            sys.stdout.write(" ".join(f"{key}={value}" for key, value in row.items() if key != "path") + "\n")
        sys.stdout.write(f"{len(rows)} artifacts, {sum(r['frames'] for r in rows)} frames, "
                         f"{sum(r['seconds'] for r in rows):.1f} s\n")
    return 0


__all__ = [
    "ARRAY_FIELDS",
    "CONTROL_COLUMNS",
    "DEFAULT_TRACE_ROOT",
    "STATE_COLUMNS",
    "TraceArrays",
    "from_capture",
    "inventory",
    "load_trace",
    "trace_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
