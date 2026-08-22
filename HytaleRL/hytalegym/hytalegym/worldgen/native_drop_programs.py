"""Typed native evidence for installed Hytale block-drop distributions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Mapping

from hytalegym.worldgen.block_actions import BLOCK_DROP_OUTPUT_CAPACITY
from hytalegym.worldgen.drop_programs import (
    BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
    LocalDropProgram,
    LocalDropStack,
    block_drop_program_contract_sha256,
)


NATIVE_DROP_PROGRAM_SCHEMA = "hytalerl_native_drop_program_evidence_v1"
NATIVE_DROP_PROGRAM_VERSION = 1
NATIVE_DROP_PROGRAM_MAX_SAMPLES = 65_536
NATIVE_DROP_PROGRAM_ROUTES = ("breaking", "soft", "harvest")
NATIVE_DROP_PROGRAM_RNG = "java.util.concurrent.ThreadLocalRandom"


@dataclass(frozen=True)
class NativeDropStack:
    """One semantic non-empty stack in native list order."""

    item_asset_id: str
    quantity: int
    durability: float
    max_durability: float
    metadata_json_sha256: str

    @classmethod
    def from_response(cls, value: Mapping[str, Any]) -> NativeDropStack:
        metadata = _string(
            value,
            "metadata_json_sha256",
            allow_empty=True,
        )
        if metadata and (
            len(metadata) != 64
            or any(character not in "0123456789abcdef" for character in metadata)
        ):
            raise ValueError("native drop metadata hash is not lowercase SHA-256")
        quantity = _integer(value, "quantity")
        durability = _number(value, "durability")
        maximum = _number(value, "max_durability")
        if quantity < 1 or durability < 0.0 or maximum < 0.0:
            raise ValueError("native drop stack is outside its domain")
        return cls(
            item_asset_id=_string(value, "item_asset_id"),
            quantity=quantity,
            durability=durability,
            max_durability=maximum,
            metadata_json_sha256=metadata,
        )

    def signature(self) -> tuple[object, ...]:
        return (
            self.item_asset_id,
            self.quantity,
            self.durability,
            self.max_durability,
            self.metadata_json_sha256,
        )


@dataclass(frozen=True)
class NativeDropOutcome:
    """One native output bundle and its empirical count."""

    count: int
    stacks: tuple[NativeDropStack, ...]

    @classmethod
    def from_response(cls, value: Mapping[str, Any]) -> NativeDropOutcome:
        raw_stacks = value.get("stacks")
        if not isinstance(raw_stacks, (list, tuple)):
            raise ValueError("native drop stacks must be an array")
        stacks = tuple(
            NativeDropStack.from_response(_mapping(item))
            for item in raw_stacks
        )
        count = _integer(value, "count")
        if count < 1 or len(stacks) > BLOCK_DROP_OUTPUT_CAPACITY:
            raise ValueError("native drop outcome is outside capacity")
        return cls(count=count, stacks=stacks)

    def signature(self) -> tuple[tuple[object, ...], ...]:
        return tuple(stack.signature() for stack in self.stacks)


@dataclass(frozen=True)
class NativeDropProgramEvidence:
    """One bridge-attributed empirical histogram."""

    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    block_asset_id: str
    route: str
    authored_quantity: int
    resolved_item_id: str
    resolved_drop_list_id: str
    sample_count: int
    outcomes: tuple[NativeDropOutcome, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeDropProgramEvidence:
        if value.get("type") != "drop_program_evidence":
            raise ValueError("bridge returned the wrong drop evidence type")
        if value.get("schema") != NATIVE_DROP_PROGRAM_SCHEMA:
            raise ValueError("bridge returned the wrong drop evidence schema")
        if value.get("version") != NATIVE_DROP_PROGRAM_VERSION:
            raise ValueError("bridge returned the wrong drop evidence version")
        bridge_sha256 = _sha256(
            _string(value, "bridge_sha256"),
            "bridge_sha256",
        )
        if (
            expected_bridge_sha256 is not None
            and bridge_sha256.lower() != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("drop evidence came from a different bridge")
        if value.get("rng") != NATIVE_DROP_PROGRAM_RNG:
            raise ValueError("native drop RNG provenance changed")
        if value.get("same_seed_replayable") is not False:
            raise ValueError("native drop evidence overclaims seed replay")
        if value.get("empty_stack_semantics") != "elided_semantic_no_drop":
            raise ValueError("native empty-stack semantics changed")
        route = _string(value, "route")
        if route not in NATIVE_DROP_PROGRAM_ROUTES:
            raise ValueError("native drop route is unsupported")
        sample_count = _integer(value, "sample_count")
        authored_quantity = _integer(value, "authored_quantity")
        raw_outcomes = value.get("outcomes")
        if not isinstance(raw_outcomes, (list, tuple)):
            raise ValueError("native drop outcomes must be an array")
        outcomes = tuple(
            NativeDropOutcome.from_response(_mapping(item))
            for item in raw_outcomes
        )
        signatures = tuple(outcome.signature() for outcome in outcomes)
        if (
            not 1 <= sample_count <= NATIVE_DROP_PROGRAM_MAX_SAMPLES
            or authored_quantity < 0
            or not 1 <= len(outcomes) <= BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY
            or sum(outcome.count for outcome in outcomes) != sample_count
            or len(set(signatures)) != len(signatures)
        ):
            raise ValueError("native drop histogram is invalid")
        return cls(
            bridge_sha256=bridge_sha256,
            server_version=_string(value, "server_version"),
            world=_string(value, "world"),
            worldgen_provider=_string(value, "worldgen_provider"),
            worldgen_version=_string(value, "worldgen_version"),
            seed=_integer(value, "seed"),
            block_asset_id=_string(value, "block_asset_id"),
            route=route,
            authored_quantity=authored_quantity,
            resolved_item_id=_string(
                value,
                "resolved_item_id",
                allow_empty=True,
            ),
            resolved_drop_list_id=_string(
                value,
                "resolved_drop_list_id",
                allow_empty=True,
            ),
            sample_count=sample_count,
            outcomes=outcomes,
        )

    def empirical_probabilities(
        self,
    ) -> dict[tuple[tuple[object, ...], ...], float]:
        return {
            outcome.signature(): outcome.count / self.sample_count
            for outcome in self.outcomes
        }


@dataclass(frozen=True)
class NativeDropDistributionComparison:
    """Finite-sample comparison against an exact installed distribution."""

    accepted: bool
    alpha: float
    simultaneous_absolute_tolerance: float
    maximum_absolute_error: float
    total_variation_distance: float
    outcome_count: int


def native_drop_program_evidence_request(
    block_asset_id: str,
    route: str,
    *,
    sample_count: int = NATIVE_DROP_PROGRAM_MAX_SAMPLES,
) -> dict[str, object]:
    """Build one bounded native histogram request."""

    if not isinstance(block_asset_id, str) or not block_asset_id:
        raise ValueError("block_asset_id cannot be blank")
    if route not in NATIVE_DROP_PROGRAM_ROUTES:
        raise ValueError("unsupported native drop route")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= NATIVE_DROP_PROGRAM_MAX_SAMPLES
    ):
        raise ValueError("native drop sample_count exceeds capacity")
    return {
        "type": "drop_program_evidence",
        "block_asset_id": block_asset_id,
        "route": route,
        "sample_count": sample_count,
    }


def compare_native_drop_distribution(
    evidence: NativeDropProgramEvidence,
    program: LocalDropProgram,
    *,
    alpha: float = 1.0e-6,
) -> NativeDropDistributionComparison:
    """Compare one empirical histogram using a simultaneous Hoeffding bound."""

    if not isinstance(evidence, NativeDropProgramEvidence):
        raise TypeError("evidence has the wrong type")
    if not isinstance(program, LocalDropProgram):
        raise TypeError("program has the wrong type")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    expected: dict[tuple[tuple[object, ...], ...], float] = {}
    for outcome in program.outcomes:
        signature = tuple(_local_stack_signature(stack) for stack in outcome.drops)
        expected[signature] = float(Fraction(
            outcome.probability_numerator,
            outcome.probability_denominator,
        ))
    empirical = evidence.empirical_probabilities()
    signatures = set(expected) | set(empirical)
    count = max(1, len(signatures))
    tolerance = math.sqrt(
        math.log(2.0 * count / alpha) / (2.0 * evidence.sample_count)
    )
    errors = [
        abs(expected.get(key, 0.0) - empirical.get(key, 0.0))
        for key in signatures
    ]
    maximum = max(errors, default=0.0)
    return NativeDropDistributionComparison(
        accepted=maximum <= tolerance,
        alpha=alpha,
        simultaneous_absolute_tolerance=tolerance,
        maximum_absolute_error=maximum,
        total_variation_distance=0.5 * sum(errors),
        outcome_count=len(signatures),
    )


def native_drop_program_evidence_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_DROP_PROGRAM_SCHEMA,
        "version": NATIVE_DROP_PROGRAM_VERSION,
        "server_version": "0.5.7",
        "source": {
            "native_api": "BlockHarvestUtils.getDrops",
            "rng": NATIVE_DROP_PROGRAM_RNG,
            "drop_program_contract_sha256": (
                block_drop_program_contract_sha256()
            ),
        },
        "fixed_capacity": {
            "samples": NATIVE_DROP_PROGRAM_MAX_SAMPLES,
            "outcomes": BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
            "stacks_per_outcome": BLOCK_DROP_OUTPUT_CAPACITY,
        },
        "histogram": (
            "ordered_nonempty_stack_bundles_with_empirical_counts"
        ),
        "comparison": (
            "exact_installed_distribution_with_simultaneous_Hoeffding_bound"
        ),
        "same_seed_native_replay": "unsupported",
        "metadata": (
            "native_BSON_JSON_SHA256_transport; installed_0_5_7_drop_"
            "programs_have_no_metadata; nonempty_metadata_comparison_fails_closed"
        ),
        "provenance": "native_runtime_bridge_plus_installed_assets",
    }


def native_drop_program_evidence_contract_sha256() -> str:
    return _canonical_sha256(native_drop_program_evidence_contract())


def _local_stack_signature(stack: LocalDropStack) -> tuple[object, ...]:
    if any(stack.metadata_hash):
        raise ValueError(
            "native metadata canonicalization is not certified"
        )
    return (
        stack.item_asset_id,
        stack.quantity,
        stack.durability,
        stack.max_durability,
        "",
    )


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("native drop row must be an object")
    return value


def _string(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise ValueError(f"{key} must be a string")
    return result


def _integer(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def _number(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(result)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{name} must be a SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


__all__ = [
    "NATIVE_DROP_PROGRAM_MAX_SAMPLES",
    "NATIVE_DROP_PROGRAM_RNG",
    "NATIVE_DROP_PROGRAM_ROUTES",
    "NATIVE_DROP_PROGRAM_SCHEMA",
    "NATIVE_DROP_PROGRAM_VERSION",
    "NativeDropDistributionComparison",
    "NativeDropOutcome",
    "NativeDropProgramEvidence",
    "NativeDropStack",
    "compare_native_drop_distribution",
    "native_drop_program_evidence_contract",
    "native_drop_program_evidence_contract_sha256",
    "native_drop_program_evidence_request",
]
