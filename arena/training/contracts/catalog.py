"""One registry through which training contracts become discoverable.

Reward/action contracts are algorithm-neutral.  A registration separately
declares whether an executable collector currently binds that contract, so UI
clients can list unfinished lessons without falsely offering a launch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
)

from arena.training.contracts.ability_landing import (
    AbilityLandingConfig,
    ability_landing_action_scope,
)
from arena.training.contracts.checkpoint_route import (
    CheckpointRouteConfig,
    checkpoint_route_action_scope,
)
from arena.training.contracts.evasion import EvasionConfig, evasion_action_scope
from arena.training.contracts.guard_duel import (
    GuardDuelConfig,
    guard_duel_attacker_action_scope,
    guard_duel_blocker_action_scope,
)
from arena.training.contracts.punish_window import (
    PunishWindowConfig,
    punish_window_action_scope,
)
from arena.training.contracts.pursuit import (
    PursuitStageConfig,
    pursuit_learner_action_scope,
)


CATALOG_SCHEMA = "arena_training_contract_catalog_v1"
REGISTRATION_SCHEMA = "arena_training_contract_registration_v1"


@dataclass(frozen=True, slots=True)
class _Registration:
    contract_id: str
    title: str
    kind: str
    config: type
    scopes: tuple[tuple[str, Callable[[Any], Any]], ...]
    collector_stage: str | None = None
    blockers: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        config = self.config()
        roles = []
        open_heads: set[str] = set()
        for role, factory in self.scopes:
            scope = factory(config)
            choices = {
                name: list(row)
                for name, row in zip(
                    ARSENAL_POLICY_ACTION_HEAD_NAMES,
                    scope.allowed_choices,
                    strict=True,
                )
                if len(row) > 1
            }
            roles.append({"role": role, "open_heads": choices})
            open_heads.update(choices)
        manifest = config.manifest()
        return {
            "schema": REGISTRATION_SCHEMA,
            "contract_id": self.contract_id,
            "title": self.title,
            "kind": self.kind,
            "contract_schema": manifest["schema"],
            "contract_sha256": config.contract_sha256,
            "manifest": manifest,
            "defaults": asdict(config),
            "action_heads": sorted(open_heads),
            "roles": roles,
            "algorithm_contract": "agent_selected_jax_learner",
            "collector": {
                "status": "bound" if self.collector_stage else "unbound",
                "stage": self.collector_stage,
                "launchable": self.collector_stage is not None,
                "blockers": list(self.blockers),
            },
        }


_TRANSITION_BLOCKER = (
    "bind engine evidence, action scope, reward signals and episode boundaries "
    "to a JIT collector"
)

_REGISTRATIONS = (
    _Registration(
        "pursuit",
        "Pursuit tracking",
        "locomotion",
        PursuitStageConfig,
        (("learner", lambda _config: pursuit_learner_action_scope()),),
        collector_stage="pursuit_tracking",
    ),
    _Registration(
        "guard_duel",
        "Guard duel",
        "combat",
        GuardDuelConfig,
        (
            ("attacker", lambda _config: guard_duel_attacker_action_scope()),
            ("blocker", lambda _config: guard_duel_blocker_action_scope()),
        ),
        blockers=(_TRANSITION_BLOCKER,),
    ),
    _Registration(
        "punish_window",
        "Punish window",
        "combat",
        PunishWindowConfig,
        (("learner", lambda _config: punish_window_action_scope()),),
        blockers=(_TRANSITION_BLOCKER,),
    ),
    _Registration(
        "ability_landing",
        "Ability landing",
        "combat",
        AbilityLandingConfig,
        (("learner", ability_landing_action_scope),),
        blockers=(
            _TRANSITION_BLOCKER,
            "bind deterministic target motion and doubled target health at reset",
        ),
    ),
    _Registration(
        "evasion",
        "Evasion",
        "combat",
        EvasionConfig,
        (("learner", lambda _config: evasion_action_scope()),),
        blockers=(_TRANSITION_BLOCKER,),
    ),
    _Registration(
        "checkpoint_route",
        "Checkpoint route",
        "locomotion",
        CheckpointRouteConfig,
        (("learner", lambda _config: checkpoint_route_action_scope()),),
        blockers=(
            _TRANSITION_BLOCKER,
            "bind a generated-terrain goal block and exact goal distance",
        ),
    ),
)


def training_contracts() -> dict[str, Any]:
    """Return the stable, JSON-ready contract catalog."""

    return {
        "schema": CATALOG_SCHEMA,
        "contracts": [registration.describe() for registration in _REGISTRATIONS],
        "registration_rule": (
            "add one _Registration; Train discovery and readiness are automatic"
        ),
    }


def training_contract(contract_id: str) -> dict[str, Any]:
    """Resolve one registered contract or fail with the available ids."""

    catalog = training_contracts()["contracts"]
    found = next(
        (item for item in catalog if item["contract_id"] == contract_id), None
    )
    if found is None:
        available = [item["contract_id"] for item in catalog]
        raise ValueError(f"unknown training contract {contract_id!r}; choose {available}")
    return found


__all__ = [
    "CATALOG_SCHEMA",
    "REGISTRATION_SCHEMA",
    "training_contract",
    "training_contracts",
]
