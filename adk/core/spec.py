"""Declarative agent specification.

An agent is a hashable spec, not a subclass. The digest goes into every
artifact so two runs can be told apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """What distinguishes one agent from another.

    Deliberately narrow: every field here changes what ``build()`` produces.
    A ``reward_terms`` field was removed because nothing consumed it -- the
    environment supplies reward directly, and a spec must not advertise a
    shape it does not implement.
    """

    name: str
    loadout: str = "iron_sword"
    scene: str = "combat/fail_closed"
    microticks: int = 1
    maximum_turn_degrees: float = 45.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name != self.name.strip()
            or self.name.count("/") != 1
            or any(not part for part in self.name.split("/", 1))
        ):
            raise ValueError(f"agent name must be '<family>/<variant>': {self.name!r}")
        if (
            not isinstance(self.loadout, str)
            or not self.loadout
            or self.loadout != self.loadout.strip()
        ):
            raise ValueError("loadout must be a non-empty profile name")
        if (
            not isinstance(self.scene, str)
            or self.scene != self.scene.strip()
            or self.scene.count("/") != 1
            or any(not part for part in self.scene.split("/", 1))
        ):
            raise ValueError(
                f"scene must be a namespaced registry key: {self.scene!r}"
            )
        if isinstance(self.microticks, bool) or not isinstance(self.microticks, int):
            raise TypeError("microticks must be an integer")
        if self.microticks < 1:
            raise ValueError(f"microticks must be >= 1: {self.microticks}")
        if (
            isinstance(self.maximum_turn_degrees, bool)
            or not isinstance(self.maximum_turn_degrees, (int, float))
            or not math.isfinite(float(self.maximum_turn_degrees))
            or self.maximum_turn_degrees <= 0.0
        ):
            raise ValueError("maximum_turn_degrees must be finite and positive")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def spec_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @property
    def family(self) -> str:
        return self.name.split("/", 1)[0]
