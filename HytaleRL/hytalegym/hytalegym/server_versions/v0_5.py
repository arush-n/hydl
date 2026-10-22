"""Hytale 0.5.x: the original line. Every gate keeps its own 0.5 constants."""

from hytalegym.server_versions import ServerProfile

PROFILE = ServerProfile(
    release_line=(0, 5),
    certified_version="0.5.7",
    world_geometry_token_capacity=44,
    world_geometry_capacity_evidence=(
        "artifacts/worldgen/world-token-capacity-v3.json:first_zero_overflow"
    ),
)
