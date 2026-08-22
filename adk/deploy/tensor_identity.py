"""Canonical content identity for the policy tensors loaded by Java.

The checkpoint SHA-256 identifies the source ``.npz`` container.  It cannot
prove that the eleven raw ``.bin`` tensors beside ``agent.json`` still contain
the parameters exported from that checkpoint.  This module defines the binary
identity stream used by the exporter.  The standalone Java mod implements the
same documented format independently and recomputes it before arming.

This is a model-layout contract, not a weapon or profile catalogue.  New model
tensors must be added to ``POLICY_TENSOR_NAMES`` and require a bundle-schema
revision; weapon-specific values remain ordinary tensor contents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from typing import Iterable


POLICY_TENSOR_IDENTITY_SCHEMA = "hytalerl_policy_tensor_content_v1"

# Canonical model order.  It mirrors the parameters consumed by Policy.java;
# never derive it from filesystem iteration order.
POLICY_TENSOR_NAMES = (
    "encoder_input_kernel",
    "encoder_input_bias",
    "encoder_hidden_kernel",
    "encoder_hidden_bias",
    "gru_input_kernel",
    "gru_recurrent_kernel",
    "gru_bias",
    "actor_kernel",
    "actor_bias",
    "critic_kernel",
    "critic_bias",
)

_DOMAIN = b"HYTALERL_POLICY_TENSOR_CONTENT\0"
_FORMAT_VERSION = 1


def _update_u32(identity: "hashlib._Hash", value: int) -> None:
    identity.update(struct.pack(">I", value))


def _update_u64(identity: "hashlib._Hash", value: int) -> None:
    identity.update(struct.pack(">Q", value))


def policy_tensor_content_sha256(
    directory: str | Path,
    *,
    tensor_names: Iterable[str] = POLICY_TENSOR_NAMES,
) -> str:
    """Hash the canonical ordered tensor names and their exact raw bytes.

    The byte stream is:

    ``domain || u32be(version) || u32be(count) ||`` for each tensor,
    ``u32be(UTF-8-name-length) || UTF-8-name || u64be(content-length) || bytes``.

    Length prefixes make name/content boundaries unambiguous.  The explicit
    order means a reordered export is a different identity even when all raw
    bytes are otherwise present.  Missing, duplicate, or non-file entries are
    errors rather than being silently skipped.
    """

    root = Path(directory)
    names = tuple(tensor_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("policy tensor names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("policy tensor names must be unique")

    identity = hashlib.sha256()
    identity.update(_DOMAIN)
    _update_u32(identity, _FORMAT_VERSION)
    _update_u32(identity, len(names))
    for name in names:
        encoded = name.encode("utf-8")
        path = root / f"{name}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"missing policy tensor {path}")
        _update_u32(identity, len(encoded))
        identity.update(encoded)
        _update_u64(identity, path.stat().st_size)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                identity.update(block)
    return identity.hexdigest().upper()


__all__ = [
    "POLICY_TENSOR_IDENTITY_SCHEMA",
    "POLICY_TENSOR_NAMES",
    "policy_tensor_content_sha256",
]
