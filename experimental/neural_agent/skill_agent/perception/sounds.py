"""The sound registry: every ``SoundEvent`` of a server version's assets as fixed-width properties and a description
vector, looked up by the sound's asset id (``docs/VIRTUAL_HUMAN.md`` section 28, step 6's first part).

The hearing rule needs two things of a sound: whether it carries to the agent (how far, and where its fade starts) and
what it is (its description). The live frame names a sound by the server's runtime ``sound_event_index``, whose
index-to-id table is not available yet, so nothing here is keyed by that index.

Built offline from the game's own asset archive, like the effects registry (``effects.py``), whose helpers and defaults
this module reuses from ``registry.py``. The engine's own values are read from its classes in the 0.7.2 server jar
(``SoundEvent``, ``AudioUtil``); each constant below names its source.

- **Sounds** (``Server/Audio/SoundEvents/**.json``) are resolved through their ``Parent`` chains, a child's keys over its
  parent's, as effects are. A chain that names a missing sound or loops ends there; both are counted in the manifest.
  A key that no sound in the chain sets takes the engine's default, the value ``SoundEvent``'s protected no-argument
  constructor gives it (the codec's supplier is that constructor).
- **Features** (``SOUND_FEATURES``, in this order):
  - ``max_distance``: ``MaxDistance`` in blocks over ``SOUND_RANGE`` (70, the loudest preset's): the distance at which the
    sound is attenuated to zero. Without one in its chain, ``ENGINE_MAX_DISTANCE``.
  - ``start_attenuation``: ``StartAttenuationDistance`` in blocks over ``SOUND_RANGE``: where the fade begins. Without
    one, ``ENGINE_START_ATTENUATION``.
  - ``distance_known``: 1 when the chain sets a ``MaxDistance``; 0 when the two distances are the engine's defaults.
  - ``volume``: ``Volume`` in decibels, as the archive gives it (the engine's own description: "Volume adjustment of the
    sound event in decibels"; its setter converts it with ``AudioUtil.decibelsToLinearGain``). Without one, 0 dB,
    the engine's linear gain of 1.0.
  - ``spatial_blend``: ``SpatialBlend``, 1.0 fully 3D (a point source) and 0.0 fully diffuse (centred on the player), by
    the engine's own description. Without one, ``ENGINE_SPATIAL_BLEND``.
  - ``preset_<name>``, one for each of ``PRESETS`` and ``preset_none``: one-hot over the attenuation preset, the first
    ancestor named ``SFX_Attn_<name>`` walking up from the sound itself. A preset is its own preset, as the census counts
    it.
  - ``category_<AudioCategory>``, one for each category the archive uses (sorted), after ``SOUND_FEATURES``. The manifest
    stores the list and ``sound_feature_names`` gives every column's name, so a later archive's new category does not
    shift the columns silently.
- **Descriptions**: each sound's name through the language file, the folders it sits in and its id split into words,
  embedded by the pinned sentence encoder and projected to ``description_width`` numbers. The 0.7.2 language file names
  no sound event, so in 0.7.2 the text is the folders and the id. The name key (``soundEvents.<id>``) is a guess, not
  taken from the archive.
- **Lookup**: ``sound_tables`` keys each sound by its asset id under ``semantic_id`` (the scheme the observation's ids
  use) for ``adapter.lookup``. It refuses ids that collide.

Saved as ``<prefix>.npz`` (the tables) with ``<prefix>.json`` (the manifest and the sound ids).
Usage: python -m experimental.neural_agent.skill_agent.perception.sounds --assets ASSETS.zip --out PREFIX
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.perception import adapter
from experimental.neural_agent.skill_agent.perception import registry as rg

SOUNDS_SCHEMA = "skill_sound_registry_v1"
SOUND_PREFIX = "Server/Audio/SoundEvents/"
PRESET_PREFIX = "SFX_Attn_"
PRESETS = ("ExtremelyQuiet", "VeryQuiet", "Quiet", "Moderate", "Loud", "VeryLoud")
SOUND_RANGE = 70.0
"""Blocks: the loudest preset's ``MaxDistance`` (``SFX_Attn_VeryLoud``). The two distance features are stored over it."""
ENGINE_MAX_DISTANCE = 16.0
ENGINE_START_ATTENUATION = 2.0
ENGINE_SPATIAL_BLEND = 0.6
ENGINE_VOLUME_DB = 0.0
"""The engine's defaults: the values ``SoundEvent``'s protected no-argument constructor gives its fields (0.7.2
``HytaleServer.jar``). Its volume is linear 1.0, which is 0 dB."""
SOUND_FEATURES = (
    "max_distance", "start_attenuation", "distance_known", "volume", "spatial_blend",
    *(f"preset_{name}" for name in (*PRESETS, "none")),
)


class SoundRecord(NamedTuple):
    """One sound: its file, its document over its ``Parent`` chain, its attenuation preset (None when the chain has
    none) and how the chain ended (None at a root, ``"missing"`` or ``"cycle"``)."""

    path: str
    value: dict
    preset: str | None
    chain_end: str | None


class SoundRegistry(NamedTuple):
    """The sound tables of one server version. Rows of ``sound_features`` and ``sound_descriptions`` follow
    ``sound_ids``; the columns of ``sound_features`` are ``sound_feature_names(manifest["categories"])``."""

    manifest: dict
    sound_ids: tuple[str, ...]
    sound_features: np.ndarray  # float32[N, len(SOUND_FEATURES) + len(categories)]
    sound_descriptions: np.ndarray  # float32[N, description_width]
    projection_mean: np.ndarray  # float32[E]: the sounds' description projection, for content added later
    projection_basis: np.ndarray  # float32[E, description_width]


class SoundTables(NamedTuple):
    """The sound registry on the device, keyed by semantic id. Row 0 is the unknown row (zeros); ``keys`` are the
    sorted semantic ids and ``rows`` the table row of each."""

    keys: jax.Array  # int32[N]
    rows: jax.Array  # int32[N]
    features: jax.Array  # float32[1 + N, len(sound_feature_names)]
    descriptions: jax.Array  # float32[1 + N, description_width]


def sound_feature_names(categories: Sequence[str]) -> tuple[str, ...]:
    """The column names of ``sound_features``: ``SOUND_FEATURES``, then one ``category_<AudioCategory>`` each."""

    return (*SOUND_FEATURES, *(f"category_{category}" for category in categories))


def _number(sound_id: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sound {sound_id}: {key} must be a number, not {value!r}")
    return float(value)


def _distance(sound_id: str, value: dict, key: str, default: float) -> float:
    if key not in value:
        return default
    distance = _number(sound_id, key, value[key])
    if distance < 0:
        raise ValueError(f"sound {sound_id}: {key} must not be negative, not {distance!r}")
    return distance


def _chain(sound_id: str, raw: dict[str, dict]) -> tuple[list[str], str | None]:
    """The sound and its ancestors, nearest first, and how the chain ended: None at a root, ``"missing"`` when a
    ``Parent`` names no sound, ``"cycle"`` when it names one already in the chain."""

    chain = [sound_id]
    while True:
        parent = raw[chain[-1]].get("Parent")
        if not isinstance(parent, str):
            return chain, None
        if parent not in raw:
            return chain, "missing"
        if parent in chain:
            return chain, "cycle"
        chain.append(parent)


def read_sounds(archive: zipfile.ZipFile) -> dict[str, SoundRecord]:
    """Every sound under ``Server/Audio/SoundEvents/`` by id (its file name), resolved through its ``Parent`` chain."""

    documents = rg._documents(archive, SOUND_PREFIX)
    raw = {sound: value for sound, (_, value) in documents.items()}
    resolved = rg._resolve_parents(raw)
    records = {}
    for sound, (path, _) in documents.items():
        chain, chain_end = _chain(sound, raw)
        preset = next((name for name in chain if name.startswith(PRESET_PREFIX)), None)
        records[sound] = SoundRecord(path, resolved[sound], preset, chain_end)
    return records


def sound_row(sound_id: str, record: SoundRecord) -> np.ndarray:
    """One sound's ``SOUND_FEATURES``. Refuses a preset the registry does not name, and a field of the wrong kind."""

    value = record.value
    preset = "none" if record.preset is None else record.preset[len(PRESET_PREFIX):]
    if preset not in (*PRESETS, "none"):
        raise ValueError(f"sound {sound_id}: attenuation preset {record.preset} is not one of {PRESET_PREFIX}*, "
                         f"{PRESETS}")
    spatial_blend = _number(sound_id, "SpatialBlend", value.get("SpatialBlend", ENGINE_SPATIAL_BLEND))
    if not 0.0 <= spatial_blend <= 1.0:
        raise ValueError(f"sound {sound_id}: SpatialBlend must be between 0 and 1, not {spatial_blend!r}")
    values = [
        _distance(sound_id, value, "MaxDistance", ENGINE_MAX_DISTANCE) / SOUND_RANGE,
        _distance(sound_id, value, "StartAttenuationDistance", ENGINE_START_ATTENUATION) / SOUND_RANGE,
        float("MaxDistance" in value),
        _number(sound_id, "Volume", value.get("Volume", ENGINE_VOLUME_DB)),
        spatial_blend,
        *(float(preset == name) for name in (*PRESETS, "none")),
    ]
    return np.asarray(values, dtype=np.float32)


def sound_text(sound_id: str, path: str, language: dict[str, str]) -> str:
    """The words a sound is described by: its name, the folders it sits in and its id split into words, in a fixed
    order."""

    relative = path[len(SOUND_PREFIX):]
    folders = relative.rsplit("/", 1)[0] if "/" in relative else ""
    parts = (language.get(f"soundEvents.{sound_id}", ""), folders.replace("/", " ").replace("_", " "),
             sound_id.replace("_", " "))
    return ". ".join(part.strip() for part in parts if part.strip())


def _tables_sha(registry: SoundRegistry) -> str:
    return rg._sha(*(value for value in registry if isinstance(value, np.ndarray)))


def build_sounds(archive_path: Path, config: rg.RegistryConfig = rg.RegistryConfig(),
                 encoder: rg.Encoder | None = None) -> SoundRegistry:
    """Read the archive and build every table. ``encoder`` defaults to the pinned sentence encoder."""

    archive_bytes = Path(archive_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        sounds = read_sounds(archive)
        language = rg.read_language(archive)
    sound_ids = tuple(sorted(sounds))
    categories = tuple(sorted({sounds[sound].value["AudioCategory"] for sound in sound_ids
                               if "AudioCategory" in sounds[sound].value}))
    fixed = np.stack([sound_row(sound, sounds[sound]) for sound in sound_ids])
    category_columns = np.asarray([[float(sounds[sound].value.get("AudioCategory") == category)
                                    for category in categories] for sound in sound_ids], dtype=np.float32)
    features = np.concatenate([fixed, category_columns], axis=1)
    texts = [sound_text(sound, sounds[sound].path, language) for sound in sound_ids]
    encode = encoder or rg.sentence_encoder(config.encoder)
    embedded = encode(texts)
    mean, basis = rg.fit_projection(embedded, config.description_width)
    descriptions = rg.project(embedded, mean, basis)

    names = sound_feature_names(categories)
    column = {name: index for index, name in enumerate(names)}
    registry = SoundRegistry({}, sound_ids, features, descriptions, mean, basis)
    chain_ends = [sounds[sound].chain_end for sound in sound_ids]
    manifest = {
        "schema": SOUNDS_SCHEMA,
        "version": config.version,
        "assets_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "encoder": config.encoder if encoder is None else getattr(encoder, "__name__", "injected"),
        "description_width": config.description_width,
        "sound_features": list(names),
        "categories": list(categories),
        "sound_range": SOUND_RANGE,
        "engine_defaults": {"MaxDistance": ENGINE_MAX_DISTANCE, "StartAttenuationDistance": ENGINE_START_ATTENUATION,
                            "SpatialBlend": ENGINE_SPATIAL_BLEND, "Volume": ENGINE_VOLUME_DB},
        "counts": {
            "sounds": len(sound_ids),
            "presets": {name: int(features[:, column[f"preset_{name}"]].sum()) for name in (*PRESETS, "none")},
            "distance_known": int(features[:, column["distance_known"]].sum()),
            "categories": len(categories),
            "no_category": sum(1 for sound in sound_ids if "AudioCategory" not in sounds[sound].value),
            "missing_parents": chain_ends.count("missing"),
            "cyclic_parents": chain_ends.count("cycle"),
        },
        "tables_sha256": _tables_sha(registry),
    }
    return registry._replace(manifest=manifest)


def save_sounds(prefix: Path, registry: SoundRegistry) -> None:
    arrays = {name: getattr(registry, name) for name in SoundRegistry._fields
              if isinstance(getattr(registry, name), np.ndarray)}
    np.savez(Path(prefix).with_suffix(".npz"), **arrays)
    meta = {**registry.manifest, "sound_ids": list(registry.sound_ids)}
    Path(prefix).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_sounds(prefix: Path) -> SoundRegistry:
    meta = json.loads(Path(prefix).with_suffix(".json").read_text())
    arrays = np.load(Path(prefix).with_suffix(".npz"))
    manifest = {key: value for key, value in meta.items() if key != "sound_ids"}
    registry = SoundRegistry(manifest=manifest, sound_ids=tuple(meta["sound_ids"]),
                             **{name: arrays[name] for name in arrays.files})
    if _tables_sha(registry) != manifest["tables_sha256"]:
        raise ValueError(f"{prefix}: the tables' SHA-256 differs from the manifest's")
    return registry


def sound_tables(registry: SoundRegistry) -> SoundTables:
    """The tables ``adapter.lookup`` reads, keyed by asset id. Refuses ids that collide under ``semantic_id``."""

    keys, rows = adapter._keyed(registry.sound_ids, "sound")
    return SoundTables(keys=jnp.asarray(keys), rows=jnp.asarray(rows),
                       features=adapter._padded(registry.sound_features),
                       descriptions=adapter._padded(registry.sound_descriptions))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", type=Path, required=True, help="the server version's Assets.zip")
    parser.add_argument("--out", type=Path, required=True, help="the output prefix (.npz and .json)")
    parser.add_argument("--version", default=rg.DEFAULT_VERSION)
    parser.add_argument("--description-width", type=int, default=rg.DESCRIPTION_WIDTH)
    args = parser.parse_args(argv)
    registry = build_sounds(args.assets, rg.RegistryConfig(version=args.version,
                                                           description_width=args.description_width))
    save_sounds(args.out, registry)
    print(json.dumps(registry.manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENGINE_MAX_DISTANCE",
    "ENGINE_SPATIAL_BLEND",
    "ENGINE_START_ATTENUATION",
    "ENGINE_VOLUME_DB",
    "PRESETS",
    "PRESET_PREFIX",
    "SOUNDS_SCHEMA",
    "SOUND_FEATURES",
    "SOUND_PREFIX",
    "SOUND_RANGE",
    "SoundRecord",
    "SoundRegistry",
    "SoundTables",
    "build_sounds",
    "load_sounds",
    "main",
    "read_sounds",
    "save_sounds",
    "sound_feature_names",
    "sound_row",
    "sound_tables",
    "sound_text",
]
