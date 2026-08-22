"""Live contract capture and the fail-closed guard.

Hashes are read from gym accessors at call time. Never hardcode one here:
contracts are bridge-derived, so a deploy moves them without any edit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

# The Gym moved this to `arsenal/schema/spec.py` (see its PACKAGE-MAP.md);
# older trees still have it at `arsenal/spec.py`. Accept both so the ADK imports
# against the current Gym as well as a pinned snapshot.
try:
    from hytalegym.jax.combat.arsenal.schema.spec import (
        combat_arsenal_contract_sha256,
    )
except ModuleNotFoundError:
    from hytalegym.jax.combat.arsenal.spec import combat_arsenal_contract_sha256
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    arsenal_policy_contract_sha256,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.world_tokens import (
    WorldGeometryPolicyConfig,
)
from hytalegym.jax.combat.observation.v3 import (
    learner_observation_v3_actor_evidence_contract_sha256,
    learner_observation_v3_contract_sha256,
)
from hytalegym.jax.crafting import crafting_contract_sha256
from hytalegym.jax.world import world_geometry_token_contract_sha256
from hytalegym.rulesets import combat_ruleset_sha256
from hytalegym.worldgen.block_affordances import block_affordance_dictionary_sha256
from hytalegym.worldgen.region.stability import current_native_evidence_jar_sha256


@dataclass(frozen=True, slots=True)
class ContractStamp:
    """What an artifact was produced against."""

    contracts: Mapping[str, str]
    action_head_names: tuple[str, ...]
    action_head_sizes: tuple[int, ...]
    observation_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.contracts, Mapping):
            raise TypeError("contracts must be a mapping")
        normalized = {}
        for name, value in self.contracts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("contract names must be non-empty strings")
            if not isinstance(value, str):
                raise TypeError(f"contract {name!r} must be a SHA-256 string")
            digest = value.strip().upper()
            if len(digest) != 64 or any(
                character not in "0123456789ABCDEF" for character in digest
            ):
                raise ValueError(f"contract {name!r} must be 64 hex characters")
            normalized[name] = digest
        object.__setattr__(
            self,
            "contracts",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        names = tuple(self.action_head_names)
        sizes = tuple(self.action_head_sizes)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("action head names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("action head names must be unique")
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError("action head sizes must be integers")
            if size < 1:
                raise ValueError("action head sizes must be positive")
        object.__setattr__(self, "action_head_names", names)
        object.__setattr__(self, "action_head_sizes", sizes)
        if len(self.action_head_names) != len(self.action_head_sizes):
            raise ValueError("action head names and sizes must have equal length")
        if isinstance(self.observation_size, bool) or not isinstance(
            self.observation_size,
            int,
        ):
            raise TypeError("observation_size must be an integer")
        if self.observation_size < 1:
            raise ValueError("observation_size must be positive")

    def mismatches(self, other: "ContractStamp") -> list[str]:
        """Field names that differ. Empty means identical."""
        drift = [
            key
            for key in set(self.contracts) | set(other.contracts)
            if self.contracts.get(key) != other.contracts.get(key)
        ]
        if self.action_head_names != other.action_head_names:
            drift.append("action_head_names")
        if self.action_head_sizes != other.action_head_sizes:
            drift.append("action_head_sizes")
        if self.observation_size != other.observation_size:
            drift.append("observation_size")
        return sorted(drift)

    def to_dict(self) -> dict[str, object]:
        return {
            "contracts": dict(self.contracts),
            "action_head_names": list(self.action_head_names),
            "action_head_sizes": list(self.action_head_sizes),
            "observation_size": self.observation_size,
        }

    def stamp_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest().upper()


def current_stamp(
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
) -> ContractStamp:
    """Capture live identities for one exact dense policy configuration."""

    return ContractStamp(
        contracts={
            "bridge_jar": current_native_evidence_jar_sha256().upper(),
            "combat_arsenal": combat_arsenal_contract_sha256().upper(),
            "arsenal_policy": arsenal_policy_contract_sha256(
                world_geometry_config
            ).upper(),
            "learner_observation_v3": (
                learner_observation_v3_contract_sha256().upper()
            ),
            "learner_v3_actor_evidence": (
                learner_observation_v3_actor_evidence_contract_sha256().upper()
            ),
            "world_geometry_token": world_geometry_token_contract_sha256().upper(),
            "block_affordance_dictionary": block_affordance_dictionary_sha256().upper(),
            "crafting": crafting_contract_sha256().upper(),
            "combat_ruleset": combat_ruleset_sha256().upper(),
        },
        action_head_names=tuple(ARSENAL_POLICY_ACTION_HEAD_NAMES),
        action_head_sizes=tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        observation_size=int(
            arsenal_policy_observation_size(world_geometry_config)
        ),
    )


def require_current(
    stamp: ContractStamp,
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
) -> None:
    """Raise unless ``stamp`` matches the live contracts. Never warn."""

    drift = stamp.mismatches(current_stamp(world_geometry_config))
    if drift:
        raise ValueError(f"contract drift, artifact is historical: {drift}")
