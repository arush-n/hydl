"""The status-effect registry: every ``EntityEffect`` of a server version's assets as fixed-width properties and a
description vector, looked up by the effect's semantic id (``docs/VIRTUAL_HUMAN.md`` section 11.4).

Today the status token knows that an effect is active but not which one. The observation publishes the effect's id
(``status_i32`` column ``effect_id``, the gym's ``semantic_id`` of the asset id); this registry gives each effect its
fixed properties and a description vector, looked up by that id. Wiring it into the status token is a later step.

Built offline from the game's own asset archive, like the content registry (``registry.py``), whose helpers and
defaults this module reuses: version, description encoder and width are its ``RegistryConfig`` defaults.

- **Effects** (``Server/Entity/Effects/**.json``) are resolved through their ``Parent`` chains and described by
  ``EFFECT_FEATURES``. An absent ``OverlapBehavior`` is ``DEFAULT_OVERLAP``, the simulator's own reading
  (``hytalegym/combat/assets/weapon_archetypes/deployables/compiler.py``, ``_compile_status``); that function also
  defaults ``Debuff`` to False and ``HorizontalSpeedMultiplier`` to 1.0, as ``effect_row`` does.
- **Duration**: an effect without one reads ``duration_log10`` 0, as an infinite one does. In 0.7.2 four are neither
  (``Burn_Template``, ``Tests/Damage``, ``Freeze``, ``Mana_Drain``); their length is presumably set where they are
  applied (not checked).
- **Descriptions**: each effect's name (through the language file), the folders it sits in and its id split into
  words, embedded by the pinned sentence encoder and projected to ``description_width`` numbers. The projection (mean
  and basis) is stored with the registry, so content added later is projected into the same space. In 0.7.2 only
  ``Poison`` has a name (``entityEffects.Poison``); the other 259 are described by their folders and id.

Saved as ``<prefix>.npz`` (the tables) with ``<prefix>.json`` (the manifest and the effect ids).
Usage: python -m experimental.neural_agent.skill_agent.perception.effects --assets ASSETS.zip --out PREFIX
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.perception import adapter
from experimental.neural_agent.skill_agent.perception import registry as rg

EFFECTS_SCHEMA = "skill_effect_registry_v1"
EFFECT_PREFIX = "Server/Entity/Effects/"
OVERLAPS = ("Overwrite", "Extend", "Ignore")
DEFAULT_OVERLAP = "Ignore"
"""The simulator's own reading of an absent ``OverlapBehavior`` (``hytalegym/combat/assets/weapon_archetypes/
deployables/compiler.py``, ``_compile_status``); that function also defaults ``Debuff`` to False and
``HorizontalSpeedMultiplier`` to 1.0."""
EFFECT_FEATURES = (
    "debuff", "infinite", "duration_log10", "overlap_overwrite", "overlap_extend", "overlap_ignore",
    "damage_over_time", "modifies_stats", "modifies_damage_taken", "modifies_damage_dealt", "speed_multiplier",
    "invulnerable", "changes_model", "conditional",
)


class EffectRegistry(NamedTuple):
    """The effect tables of one server version. Rows of ``effect_features`` and ``effect_descriptions`` follow
    ``effect_ids``."""

    manifest: dict
    effect_ids: tuple[str, ...]
    effect_features: np.ndarray  # float32[N, len(EFFECT_FEATURES)]
    effect_descriptions: np.ndarray  # float32[N, description_width]
    projection_mean: np.ndarray  # float32[E]: the effects' description projection, for content added later
    projection_basis: np.ndarray  # float32[E, description_width]


class EffectTables(NamedTuple):
    """The effect registry on the device, keyed by semantic id. Row 0 is the unknown row (zeros); ``keys`` are the
    sorted semantic ids and ``rows`` the table row of each."""

    keys: jax.Array  # int32[N]
    rows: jax.Array  # int32[N]
    features: jax.Array  # float32[1 + N, len(EFFECT_FEATURES)]
    descriptions: jax.Array  # float32[1 + N, description_width]


def _flag(effect_id: str, effect: dict, key: str) -> float:
    value = effect.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"effect {effect_id}: {key} must be true or false, not {value!r}")
    return float(value)


def _number(effect_id: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"effect {effect_id}: {key} must be a number, not {value!r}")
    return float(value)


def _filled(value: Any, kind: type) -> bool:
    return isinstance(value, kind) and len(value) > 0


def effect_row(effect_id: str, effect: dict) -> np.ndarray:
    """One effect's ``EFFECT_FEATURES``, from its document resolved through its ``Parent`` chain. Refuses a field of
    the wrong kind, an overlap behavior the simulator does not know, and a negative or non-numeric duration."""

    infinite = _flag(effect_id, effect, "Infinite")
    duration_log10 = 0.0
    if "Duration" in effect:
        duration = _number(effect_id, "Duration", effect["Duration"])
        if duration < 0:
            raise ValueError(f"effect {effect_id}: Duration must not be negative, not {duration!r}")
        duration_log10 = 0.0 if infinite else math.log10(1.0 + duration)
    overlap = effect.get("OverlapBehavior", DEFAULT_OVERLAP)
    if overlap not in OVERLAPS:
        raise ValueError(f"effect {effect_id}: OverlapBehavior must be one of {OVERLAPS}, not {overlap!r}")
    application = effect.get("ApplicationEffects") or {}
    values = {
        "debuff": _flag(effect_id, effect, "Debuff"),
        "infinite": infinite,
        "duration_log10": duration_log10,
        **{f"overlap_{name.lower()}": float(overlap == name) for name in OVERLAPS},
        "damage_over_time": float(_filled(effect.get("DamageCalculator"), dict)),
        "modifies_stats": float(_filled(effect.get("StatModifiers"), dict)
                               or _filled(effect.get("RawStatModifiers"), dict)),
        "modifies_damage_taken": float(_filled(effect.get("DamageResistance"), dict)),
        "modifies_damage_dealt": float(_filled(effect.get("OutgoingDamage"), dict)),
        "speed_multiplier": _number(effect_id, "ApplicationEffects.HorizontalSpeedMultiplier",
                                    application.get("HorizontalSpeedMultiplier", 1.0)),
        "invulnerable": _flag(effect_id, effect, "Invulnerable"),
        "changes_model": float(effect.get("ModelChange") is not None or effect.get("ModelOverride") is not None),
        "conditional": float(_filled(effect.get("ApplyConditions"), list)),
    }
    return np.asarray([values[name] for name in EFFECT_FEATURES], dtype=np.float32)


def read_effects(archive: zipfile.ZipFile) -> dict[str, tuple[str, dict]]:
    """Every effect under ``Server/Entity/Effects/`` by id (its file name), with its path, each resolved through its
    ``Parent`` chain."""

    documents = rg._documents(archive, EFFECT_PREFIX)
    resolved = rg._resolve_parents({effect: value for effect, (_, value) in documents.items()})
    return {effect: (documents[effect][0], value) for effect, value in resolved.items()}


def effect_text(effect_id: str, path: str, language: dict[str, str]) -> str:
    """The words an effect is described by: its name, the folders it sits in and its id split into words, in a fixed
    order."""

    relative = path[len(EFFECT_PREFIX):]
    folders = relative.rsplit("/", 1)[0] if "/" in relative else ""
    parts = (language.get(f"entityEffects.{effect_id}", ""), folders.replace("/", " ").replace("_", " "),
             effect_id.replace("_", " "))
    return ". ".join(part.strip() for part in parts if part.strip())


def _tables_sha(registry: EffectRegistry) -> str:
    return rg._sha(*(value for value in registry if isinstance(value, np.ndarray)))


def build_effects(archive_path: Path, config: rg.RegistryConfig = rg.RegistryConfig(),
                  encoder: rg.Encoder | None = None) -> EffectRegistry:
    """Read the archive and build every table. ``encoder`` defaults to the pinned sentence encoder."""

    archive_bytes = Path(archive_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        effects = read_effects(archive)
        language = rg.read_language(archive)
    effect_ids = tuple(sorted(effects))
    features = (np.stack([effect_row(effect, effects[effect][1]) for effect in effect_ids]) if effect_ids
                else np.zeros((0, len(EFFECT_FEATURES)), dtype=np.float32))
    texts = [effect_text(effect, effects[effect][0], language) for effect in effect_ids]
    encode = encoder or rg.sentence_encoder(config.encoder)
    embedded = encode(texts)
    mean, basis = rg.fit_projection(embedded, config.description_width)
    descriptions = rg.project(embedded, mean, basis)

    column = {name: index for index, name in enumerate(EFFECT_FEATURES)}
    registry = EffectRegistry({}, effect_ids, features, descriptions, mean, basis)
    manifest = {
        "schema": EFFECTS_SCHEMA,
        "version": config.version,
        "assets_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "encoder": config.encoder if encoder is None else getattr(encoder, "__name__", "injected"),
        "description_width": config.description_width,
        "effect_features": list(EFFECT_FEATURES),
        "counts": {"effects": len(effect_ids),
                   "debuffs": int(features[:, column["debuff"]].sum()),
                   "infinite": int(features[:, column["infinite"]].sum()),
                   "damage_over_time": int(features[:, column["damage_over_time"]].sum())},
        "tables_sha256": _tables_sha(registry),
    }
    return registry._replace(manifest=manifest)


def save_effects(prefix: Path, registry: EffectRegistry) -> None:
    arrays = {name: getattr(registry, name) for name in EffectRegistry._fields
              if isinstance(getattr(registry, name), np.ndarray)}
    np.savez(Path(prefix).with_suffix(".npz"), **arrays)
    meta = {**registry.manifest, "effect_ids": list(registry.effect_ids)}
    Path(prefix).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_effects(prefix: Path) -> EffectRegistry:
    meta = json.loads(Path(prefix).with_suffix(".json").read_text())
    arrays = np.load(Path(prefix).with_suffix(".npz"))
    manifest = {key: value for key, value in meta.items() if key != "effect_ids"}
    registry = EffectRegistry(manifest=manifest, effect_ids=tuple(meta["effect_ids"]),
                              **{name: arrays[name] for name in arrays.files})
    if _tables_sha(registry) != manifest["tables_sha256"]:
        raise ValueError(f"{prefix}: the tables' SHA-256 differs from the manifest's")
    return registry


def effect_tables(registry: EffectRegistry) -> EffectTables:
    """The tables ``adapter.lookup`` reads. Refuses ids that collide under ``semantic_id``."""

    keys, rows = adapter._keyed(registry.effect_ids, "effect")
    return EffectTables(keys=jnp.asarray(keys), rows=jnp.asarray(rows),
                        features=adapter._padded(registry.effect_features),
                        descriptions=adapter._padded(registry.effect_descriptions))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", type=Path, required=True, help="the server version's Assets.zip")
    parser.add_argument("--out", type=Path, required=True, help="the output prefix (.npz and .json)")
    parser.add_argument("--version", default=rg.DEFAULT_VERSION)
    parser.add_argument("--description-width", type=int, default=rg.DESCRIPTION_WIDTH)
    args = parser.parse_args(argv)
    registry = build_effects(args.assets, rg.RegistryConfig(version=args.version,
                                                            description_width=args.description_width))
    save_effects(args.out, registry)
    print(json.dumps(registry.manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OVERLAP",
    "EFFECTS_SCHEMA",
    "EFFECT_FEATURES",
    "EFFECT_PREFIX",
    "EffectRegistry",
    "EffectTables",
    "OVERLAPS",
    "build_effects",
    "effect_row",
    "effect_tables",
    "effect_text",
    "load_effects",
    "main",
    "read_effects",
    "save_effects",
]
