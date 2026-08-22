"""Fixed-capacity native spawn-marker metadata for surrogate entity worlds."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
import operator

import numpy as np

from hytalegym.worldgen.surrogate.assets import SpawnMarkerAsset
from hytalegym.worldgen.surrogate.contract import (
    DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY,
    DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY,
)
from hytalegym.worldgen.surrogate.entities import entity_type_identity_words


@dataclass(frozen=True)
class CompiledSpawnMarkerCatalog:
    """Padded marker choices; selection is metadata, not NPC lifecycle behavior."""

    marker_ids: tuple[str, ...]
    asset_sha256: tuple[str, ...]
    role_ids: tuple[tuple[str | None, ...], ...]
    flock_json: tuple[tuple[str | None, ...], ...]
    marker_mask: np.ndarray
    marker_identity_words: np.ndarray
    asset_sha256_words: np.ndarray
    realtime_respawn: np.ndarray
    manual_trigger: np.ndarray
    exclusion_radius: np.ndarray
    maximum_drop_height: np.ndarray
    deactivation_distance: np.ndarray
    deactivation_seconds: np.ndarray
    choice_mask: np.ndarray
    role_present: np.ndarray
    role_identity_words: np.ndarray
    choice_weight: np.ndarray
    respawn_seconds: np.ndarray
    flock_required: np.ndarray

    @property
    def logical_bytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.marker_mask,
                self.marker_identity_words,
                self.asset_sha256_words,
                self.realtime_respawn,
                self.manual_trigger,
                self.exclusion_radius,
                self.maximum_drop_height,
                self.deactivation_distance,
                self.deactivation_seconds,
                self.choice_mask,
                self.role_present,
                self.role_identity_words,
                self.choice_weight,
                self.respawn_seconds,
                self.flock_required,
            )
        )


def compile_spawn_marker_catalog(
    assets: Sequence[SpawnMarkerAsset],
    *,
    known_role_ids: Collection[str],
    marker_capacity: int,
    choice_capacity: int,
) -> CompiledSpawnMarkerCatalog:
    """Compile exact asset metadata, rejecting unknown roles and all overflow."""

    marker_limit = _positive_int(marker_capacity, "spawn marker capacity")
    choice_limit = _positive_int(choice_capacity, "spawn marker choice capacity")
    if isinstance(assets, (str, bytes)) or any(
        not isinstance(asset, SpawnMarkerAsset) for asset in assets
    ):
        raise TypeError("assets must contain SpawnMarkerAsset values")
    if isinstance(known_role_ids, (str, bytes)):
        raise TypeError("known_role_ids must be a collection of role IDs")
    roles = frozenset(known_role_ids)
    if any(not isinstance(role_id, str) or not role_id for role_id in roles):
        raise TypeError("known_role_ids must contain non-empty strings")
    ordered = tuple(sorted(assets, key=lambda asset: asset.asset_id))
    if len(ordered) > marker_limit:
        raise ValueError(
            f"spawn markers exceed catalog capacity {marker_limit}"
        )
    if len({asset.asset_id for asset in ordered}) != len(ordered):
        raise ValueError("spawn marker catalog contains duplicate asset IDs")

    marker_mask = np.zeros(marker_limit, dtype=np.bool_)
    marker_identity = np.zeros((marker_limit, 2), dtype=np.uint32)
    asset_hash = np.zeros((marker_limit, 8), dtype=np.uint32)
    realtime = np.zeros(marker_limit, dtype=np.bool_)
    manual = np.zeros(marker_limit, dtype=np.bool_)
    exclusion = np.zeros(marker_limit, dtype=np.float32)
    maximum_drop = np.zeros(marker_limit, dtype=np.float32)
    deactivation_distance = np.zeros(marker_limit, dtype=np.float32)
    deactivation_seconds = np.zeros(marker_limit, dtype=np.float32)
    choice_mask = np.zeros((marker_limit, choice_limit), dtype=np.bool_)
    role_present = np.zeros((marker_limit, choice_limit), dtype=np.bool_)
    role_identity = np.zeros((marker_limit, choice_limit, 2), dtype=np.uint32)
    choice_weight = np.zeros((marker_limit, choice_limit), dtype=np.float32)
    respawn_seconds = np.zeros((marker_limit, choice_limit), dtype=np.float32)
    flock_required = np.zeros((marker_limit, choice_limit), dtype=np.bool_)
    marker_ids: list[str] = []
    asset_sha256: list[str] = []
    role_ids: list[tuple[str | None, ...]] = []
    flock_json: list[tuple[str | None, ...]] = []

    identities: set[tuple[int, int]] = set()
    role_identities: dict[tuple[int, int], str] = {}
    for marker_index, asset in enumerate(ordered):
        if len(asset.configurations) > choice_limit:
            raise ValueError(
                f"spawn marker {asset.asset_id!r} exceeds choice capacity "
                f"{choice_limit}"
            )
        marker_values = (
            asset.exclusion_radius,
            asset.maximum_drop_height,
            asset.deactivation_distance,
            asset.deactivation_seconds,
        )
        if (
            not all(np.isfinite(value) for value in marker_values)
            or asset.exclusion_radius < 0.0
            or any(value <= 0.0 for value in marker_values[1:])
        ):
            raise ValueError(
                f"spawn marker {asset.asset_id!r} has invalid physical metadata"
            )
        identity = spawn_marker_identity_words(asset.asset_id)
        if identity in identities:
            raise ValueError("spawn marker semantic identity collision")
        identities.add(identity)
        marker_mask[marker_index] = True
        marker_identity[marker_index] = identity
        asset_hash[marker_index] = np.frombuffer(
            bytes.fromhex(asset.provenance.content_sha256),
            dtype="<u4",
        )
        realtime[marker_index] = asset.realtime_respawn
        manual[marker_index] = asset.manual_trigger
        exclusion[marker_index] = asset.exclusion_radius
        maximum_drop[marker_index] = asset.maximum_drop_height
        deactivation_distance[marker_index] = asset.deactivation_distance
        deactivation_seconds[marker_index] = asset.deactivation_seconds
        marker_ids.append(asset.asset_id)
        asset_sha256.append(asset.provenance.content_sha256)
        role_ids.append(tuple(choice.role_id for choice in asset.configurations))
        flock_json.append(
            tuple(choice.flock_json for choice in asset.configurations)
        )
        for choice_index, choice in enumerate(asset.configurations):
            if choice.role_id is not None and choice.role_id not in roles:
                raise ValueError(
                    f"spawn marker {asset.asset_id!r} references unknown role "
                    f"{choice.role_id!r}"
                )
            choice_mask[marker_index, choice_index] = True
            if not np.isfinite(choice.weight) or choice.weight <= 0.0:
                raise ValueError("spawn marker choice weight must be positive")
            active_respawn = (
                choice.realtime_respawn_seconds
                if asset.realtime_respawn
                else choice.game_time_respawn_seconds
            )
            if (
                active_respawn is None
                or not np.isfinite(active_respawn)
                or active_respawn <= 0.0
            ):
                raise ValueError(
                    f"spawn marker {asset.asset_id!r} lacks active respawn timing"
                )
            choice_weight[marker_index, choice_index] = choice.weight
            respawn_seconds[marker_index, choice_index] = active_respawn
            flock_required[marker_index, choice_index] = (
                choice.flock_json is not None
            )
            if choice.role_id is not None:
                role_present[marker_index, choice_index] = True
                role_key = entity_type_identity_words(choice.role_id)
                previous_role = role_identities.setdefault(
                    role_key,
                    choice.role_id,
                )
                if previous_role != choice.role_id:
                    raise ValueError("spawn marker role semantic identity collision")
                role_identity[marker_index, choice_index] = role_key

    arrays = (
        marker_mask,
        marker_identity,
        asset_hash,
        realtime,
        manual,
        exclusion,
        maximum_drop,
        deactivation_distance,
        deactivation_seconds,
        choice_mask,
        role_present,
        role_identity,
        choice_weight,
        respawn_seconds,
        flock_required,
    )
    if any(
        not np.all(np.isfinite(value))
        for value in (
            exclusion,
            maximum_drop,
            deactivation_distance,
            deactivation_seconds,
            choice_weight,
            respawn_seconds,
        )
    ):
        raise ValueError("spawn marker catalog contains non-finite values")
    for value in arrays:
        value.flags.writeable = False
    return CompiledSpawnMarkerCatalog(
        marker_ids=tuple(marker_ids),
        asset_sha256=tuple(asset_sha256),
        role_ids=tuple(role_ids),
        flock_json=tuple(flock_json),
        marker_mask=marker_mask,
        marker_identity_words=marker_identity,
        asset_sha256_words=asset_hash,
        realtime_respawn=realtime,
        manual_trigger=manual,
        exclusion_radius=exclusion,
        maximum_drop_height=maximum_drop,
        deactivation_distance=deactivation_distance,
        deactivation_seconds=deactivation_seconds,
        choice_mask=choice_mask,
        role_present=role_present,
        role_identity_words=role_identity,
        choice_weight=choice_weight,
        respawn_seconds=respawn_seconds,
        flock_required=flock_required,
    )


def spawn_marker_identity_words(asset_id: str) -> tuple[int, int]:
    """Return the entity-type identity used by placed spawn-marker entities."""

    if not isinstance(asset_id, str) or not asset_id:
        raise TypeError("spawn marker asset_id must be a non-empty string")
    return entity_type_identity_words(asset_id)


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


__all__ = [
    "CompiledSpawnMarkerCatalog",
    "DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY",
    "DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY",
    "compile_spawn_marker_catalog",
    "spawn_marker_identity_words",
]
