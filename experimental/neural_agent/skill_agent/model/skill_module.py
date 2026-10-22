"""A skill's own module, and the shared trunk every module reads.

Owner, 2026-10-19: every skill trains its own module by its own strategy, on a shared, frozen trunk, and skills compose
at run time. A module is the request-reading head of ``model/generalist.py`` with this skill's own weights. Each tick
the trunk (``model/encoder.py``) runs once, and every active skill's module reads the trunk's token sets and the tick's
requests:

- it drives only its own scope's heads, and reads its own request's done logit;
- it sees the other active requests (owner: "skills see each other"), but only of the skills it was trained beside
  (``known``). A slot of any other verb is invisible to it, so adding a skill never changes an existing module;
- it records the SHA-256 of the trunk it was trained on (``trunk_sha256``) and refuses any other trunk.

Saved as ``<name>.npz`` (the parameters) beside ``<name>.json`` (what the module is and the two fingerprints).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import scope_of
from experimental.neural_agent.skill_agent.contract.variables import fingerprint
from experimental.neural_agent.skill_agent.model.encoder import EncoderConfig, EncoderOutput, init_encoder
from experimental.neural_agent.skill_agent.model.freezing import params_sha256
from experimental.neural_agent.skill_agent.model.migrate import migrate_trunk
from experimental.neural_agent.skill_agent.model.generalist import (
    GeneralistConfig,
    GeneralistParams,
    GeneralistRequests,
    apply_generalist,
    init_generalist,
)
from experimental.neural_agent.skill_agent.neural_skills.spec import SkillSpec
from experimental.neural_agent.skill_agent.training.blocks import HEAD_NAMES, REQUEST_SLOTS, load_params, save_params

MODULE_CONFIG = GeneralistConfig(max_requests=REQUEST_SLOTS, verb_count=160)
"""A module's sizes: the generalist head's (width 64, 2 layers, 4 heads) with room for 160 verbs, the catalogue's size,
so that a verb id never outgrows a module."""


class Trunk(NamedTuple):
    """The shared, frozen trunk: the token encoder's sizes, its parameters and their SHA-256."""

    config: EncoderConfig
    params: Any
    sha256: str


class SkillModule(NamedTuple):
    """One skill's module. ``known`` is the verb ids it reads: its own and those of the skills it was trained beside."""

    skill: str
    scope: str
    verb: int
    known: tuple[int, ...]
    config: GeneralistConfig
    params: GeneralistParams
    trunk_sha256: str


class ModuleOutput(NamedTuple):
    head_logits: dict[str, jax.Array]  # every head of the layout; the module drives only its scope's
    done: jax.Array  # float32[B]: the done logit of its own request (0 where it has none)
    own: jax.Array  # bool[B]: whether the tick has a valid request of its verb


def make_trunk(params: Any, config: EncoderConfig) -> Trunk:
    return Trunk(config=config, params=params, sha256=params_sha256(params))


def trunk_from_run(params_path: Path, config: EncoderConfig = EncoderConfig()) -> Trunk:
    """The trunk of a saved run of the A2 reference trainer (``training/distill.py``; its leaves are ``trunk.*``)."""

    template = {"trunk": init_encoder(jax.random.key(0), config)}
    return make_trunk(load_params(params_path, template)["trunk"], config)


def scope_heads(scope: str) -> np.ndarray:
    """bool[12]: the gym heads (``blocks.HEAD_NAMES``) that belong to ``scope``."""

    return np.asarray([scope_of(name) == scope for name in HEAD_NAMES])


def init_module(key: jax.Array, spec: SkillSpec, beside: Sequence[SkillSpec], trunk: Trunk,
                config: GeneralistConfig = MODULE_CONFIG) -> SkillModule:
    """A fresh module for ``spec`` that will read its own requests and those of ``beside``."""

    known = tuple(sorted({spec.verb, *(other.verb for other in beside)}))
    return SkillModule(skill=spec.name, scope=spec.scope, verb=spec.verb, known=known, config=config,
                       params=init_generalist(key, config, trunk.config), trunk_sha256=trunk.sha256)


def visible_requests(requests: GeneralistRequests, known: Sequence[int]) -> GeneralistRequests:
    """The requests as a module sees them: a slot whose verb is not in ``known`` is invalid, so none of its fields
    reaches the module (``apply_generalist`` zeroes an invalid slot and never attends to it)."""

    seen = jnp.isin(requests.verb, jnp.asarray(tuple(known), dtype=jnp.int32))
    return requests._replace(valid=requests.valid & seen)


def module_outputs(params: GeneralistParams, config: GeneralistConfig, known: tuple[int, ...], verb: int,
                   encoded: EncoderOutput, token_masks: dict, requests: GeneralistRequests) -> ModuleOutput:
    """The module's readouts for B agents: every head's logits (the caller keeps its scope's) and its own request's done
    logit. Pure JAX; ``known`` and ``verb`` are static."""

    out = apply_generalist(params, encoded, token_masks, visible_requests(requests, known), config)
    own_slots = (requests.verb == verb) & requests.valid
    slot = jnp.argmax(own_slots, axis=1)
    done = out.status[jnp.arange(slot.shape[0]), slot, 0]
    own = jnp.any(own_slots, axis=1)
    return ModuleOutput(head_logits=out.head_logits, done=jnp.where(own, done, 0.0), own=own)


def check_trunk(module: SkillModule, trunk: Trunk) -> None:
    if module.trunk_sha256 != trunk.sha256:
        raise ValueError(f"{module.skill}'s module was trained on trunk {module.trunk_sha256[:12]}, not "
                         f"{trunk.sha256[:12]}: retrain it on this trunk, or load its own")


def _config_dict(config: Any) -> dict:
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


def save_trunk(path: Path, trunk: Trunk, layout: Mapping[str, Sequence[str]] | None = None) -> None:
    """``path`` (.npz) and its .json sidecar: the encoder's sizes and the SHA-256, and ``layout`` ({type: feature
    names}, ``encoder.token_layout``) with its fingerprint when given, so a later schema can migrate the trunk by name
    (``docs/VIRTUAL_HUMAN.md`` section 14, rule 2)."""

    meta = {"config": _config_dict(trunk.config), "sha256": trunk.sha256}
    if layout is not None:
        if tuple((name, len(features)) for name, features in layout.items()) != trunk.config.token_types:
            raise ValueError(f"the layout's widths are not the trunk's {trunk.config.token_types}")
        meta["layout"] = {name: list(features) for name, features in layout.items()}
        meta["layout_fingerprint"] = fingerprint(meta["layout"])
    save_params(path, trunk.params)
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_trunk(path: Path, layout: Mapping[str, Sequence[str]] | None = None, *,
               maps: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
               dropped: Mapping[str, Sequence[str]] | None = None) -> Trunk:
    """A saved trunk, checked against its SHA-256. With ``layout`` (the schema it will read), a trunk saved under
    another layout is migrated to it by name (``model/migrate.py``; ``maps`` and ``dropped`` as there), and one saved
    without a layout must have its widths."""

    meta = json.loads(Path(path).with_suffix(".json").read_text())
    raw = meta["config"]
    config = EncoderConfig(**{**raw, "token_types": tuple(tuple(entry) for entry in raw["token_types"]),
                              "condition_types": tuple(raw.get("condition_types", ()))})
    trunk = make_trunk(load_params(path, init_encoder(jax.random.key(0), config)), config)
    if trunk.sha256 != meta["sha256"]:
        raise ValueError(f"{path}: the parameters' SHA-256 differs from the sidecar's")
    if layout is None:
        return trunk
    current = {name: list(features) for name, features in layout.items()}
    saved = meta.get("layout")
    if saved is None:
        if tuple((name, len(features)) for name, features in current.items()) != config.token_types:
            raise ValueError(f"{path} was saved without its layout and its widths differ, so it cannot be migrated")
        return trunk
    if fingerprint(saved) != meta["layout_fingerprint"]:
        raise ValueError(f"{path}: the layout differs from its fingerprint")
    if saved == current:
        return trunk
    params, config = migrate_trunk(trunk.params, config, saved, current, maps=maps, dropped=dropped)
    return make_trunk(params, config)


def save_module(path: Path, module: SkillModule) -> None:
    """``path`` (.npz) and its .json sidecar: the skill, its scope and verb, the verbs it reads, its sizes, and the
    SHA-256 of its parameters and of its trunk."""

    save_params(path, module.params)
    meta = {"skill": module.skill, "scope": module.scope, "verb": module.verb, "known": list(module.known),
            "config": _config_dict(module.config), "trunk_sha256": module.trunk_sha256,
            "params_sha256": params_sha256(module.params)}
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_module(path: Path, trunk: Trunk) -> SkillModule:
    """A saved module, checked against its fingerprint and against ``trunk`` (it must be the one it was trained on)."""

    meta = json.loads(Path(path).with_suffix(".json").read_text())
    config = GeneralistConfig(**meta["config"])
    params = load_params(path, init_generalist(jax.random.key(0), config, trunk.config))
    if params_sha256(params) != meta["params_sha256"]:
        raise ValueError(f"{path}: the parameters' SHA-256 differs from the sidecar's")
    module = SkillModule(skill=meta["skill"], scope=meta["scope"], verb=meta["verb"], known=tuple(meta["known"]),
                         config=config, params=params, trunk_sha256=meta["trunk_sha256"])
    check_trunk(module, trunk)
    return module


__all__ = [
    "MODULE_CONFIG",
    "ModuleOutput",
    "SkillModule",
    "Trunk",
    "check_trunk",
    "init_module",
    "load_module",
    "load_trunk",
    "make_trunk",
    "module_outputs",
    "save_module",
    "save_trunk",
    "scope_heads",
    "trunk_from_run",
    "visible_requests",
]
