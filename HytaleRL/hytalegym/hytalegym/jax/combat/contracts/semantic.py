"""Stable host-visible semantic identifiers for authored asset names."""

from __future__ import annotations


def semantic_id(asset_id: str) -> int:
    """Return the positive 31-bit FNV-1a ID used by Combat device rows."""

    if not isinstance(asset_id, str) or not asset_id:
        raise TypeError("asset_id must be a nonempty string")
    value = 0x811C9DC5
    for byte in asset_id.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value & 0x7FFFFFFF or 1


__all__ = ["semantic_id"]
