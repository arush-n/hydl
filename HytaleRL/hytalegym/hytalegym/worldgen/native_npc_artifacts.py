"""Deterministic, replay-decodable storage for native NPC trace drains."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import msgpack

from hytalegym.worldgen.native_npc_traces import (
    NativeNpcTraceCapture,
    concatenate_native_npc_traces,
)


NATIVE_NPC_TRACE_ARTIFACT_SCHEMA = "hytalerl_native_npc_trace_artifact_v1"
NATIVE_NPC_TRACE_ARTIFACT_VERSION = 1


def encode_native_npc_trace_artifact(chunks: Sequence[Mapping[str, Any]]) -> bytes:
    """Validate and store exact bridge response maps without dropping fields."""

    rows = tuple(dict(chunk) for chunk in chunks)
    capture = _capture(rows)
    envelope = {
        "schema": NATIVE_NPC_TRACE_ARTIFACT_SCHEMA,
        "version": NATIVE_NPC_TRACE_ARTIFACT_VERSION,
        "trace_uuid": str(capture.trace_uuid),
        "npc_uuid": str(capture.npc_uuid),
        "role": capture.role,
        "rows": capture.emitted_count,
        "bridge_sha256": capture.bridge_sha256,
        "observation_contract_sha256": capture.observation_contract_sha256,
        "chunks": rows,
    }
    return gzip.compress(msgpack.packb(envelope, use_bin_type=True), mtime=0)


def decode_native_npc_trace_artifact(payload: bytes) -> NativeNpcTraceCapture:
    """Decode an artifact through the same strict parser used for live traces."""

    value = msgpack.unpackb(gzip.decompress(payload), raw=False)
    if not isinstance(value, dict) or (value.get("schema"), value.get("version")) != (
        NATIVE_NPC_TRACE_ARTIFACT_SCHEMA,
        NATIVE_NPC_TRACE_ARTIFACT_VERSION,
    ):
        raise ValueError("unsupported native NPC trace artifact")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not all(isinstance(row, dict) for row in chunks):
        raise ValueError("native NPC trace artifact chunks are invalid")
    capture = _capture(chunks)
    expected = {
        "trace_uuid": str(capture.trace_uuid),
        "npc_uuid": str(capture.npc_uuid),
        "role": capture.role,
        "rows": capture.emitted_count,
        "bridge_sha256": capture.bridge_sha256,
        "observation_contract_sha256": capture.observation_contract_sha256,
    }
    if any(
        value.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError("native NPC trace artifact manifest does not match its chunks")
    return capture


def write_native_npc_trace_artifact(
    path: str | Path, chunks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Write an exact trace and return its content-addressed receipt."""

    destination = Path(path)
    payload = encode_native_npc_trace_artifact(chunks)
    capture = decode_native_npc_trace_artifact(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "schema": NATIVE_NPC_TRACE_ARTIFACT_SCHEMA,
        "path": destination.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
        "bytes": len(payload),
        "rows": capture.emitted_count,
        "role": capture.role,
        "trace_uuid": str(capture.trace_uuid),
        "npc_uuid": str(capture.npc_uuid),
    }


def load_native_npc_trace_artifact(path: str | Path) -> NativeNpcTraceCapture:
    return decode_native_npc_trace_artifact(Path(path).read_bytes())


def _capture(chunks: Sequence[Mapping[str, Any]]) -> NativeNpcTraceCapture:
    if not chunks:
        raise ValueError("native NPC trace artifact requires at least one chunk")
    return concatenate_native_npc_traces(
        tuple(NativeNpcTraceCapture.from_response(chunk) for chunk in chunks)
    )


__all__ = [
    "NATIVE_NPC_TRACE_ARTIFACT_SCHEMA",
    "NATIVE_NPC_TRACE_ARTIFACT_VERSION",
    "decode_native_npc_trace_artifact",
    "encode_native_npc_trace_artifact",
    "load_native_npc_trace_artifact",
    "write_native_npc_trace_artifact",
]
