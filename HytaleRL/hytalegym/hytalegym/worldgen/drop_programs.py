"""Exact installed drop-program distributions with explicit JAX sampling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import struct

from hytalegym.worldgen.block_actions import (
    BLOCK_DROP_METADATA_HASH_WORDS,
    BLOCK_DROP_OUTPUT_CAPACITY,
    stable_asset_id,
)


BLOCK_DROP_PROGRAM_SCHEMA = "hytalerl_block_drop_program_v1"
BLOCK_DROP_PROGRAM_VERSION = 1
BLOCK_DROP_PROGRAM_CAPACITY = 256
BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY = 75

NamedDropLookup = Callable[[str], Mapping[str, object] | None]
ItemLookup = Callable[[str], Mapping[str, object]]


@dataclass(frozen=True)
class LocalDropStack:
    item_asset_id: str
    item_id: int
    quantity: int
    item_max_stack: int
    durability: float
    max_durability: float
    metadata_hash: tuple[int, ...]


@dataclass(frozen=True)
class LocalDropOutcome:
    probability_numerator: int
    probability_denominator: int
    drops: tuple[LocalDropStack, ...]


@dataclass(frozen=True)
class LocalDropProgram:
    semantic_sha256: str
    randomized: bool
    missing_named_assets: tuple[str, ...]
    outcomes: tuple[LocalDropOutcome, ...]


@dataclass(frozen=True)
class _DropSpec:
    item_asset_id: str
    quantity_min: int
    quantity_max: int
    metadata: object | None


_TraversalState = tuple[tuple[_DropSpec, ...], frozenset[str]]
_TraversalDistribution = dict[_TraversalState, Fraction]


class LocalDropProgramCompiler:
    """Compile native container programs into bounded exact distributions."""

    def __init__(
        self,
        named_drop_lookup: NamedDropLookup,
        item_lookup: ItemLookup,
    ):
        if not callable(named_drop_lookup) or not callable(item_lookup):
            raise TypeError("drop and item lookups must be callable")
        self._named_drop_lookup = named_drop_lookup
        self._item_lookup = item_lookup
        self._named_cache: dict[str, Mapping[str, object] | None] = {}
        self._item_cache: dict[str, tuple[int, float]] = {}

    def compile(
        self,
        value: str | Mapping[str, object],
    ) -> LocalDropProgram:
        """Compile one named or embedded ``ItemDropList`` value."""

        missing: set[str] = set()
        if isinstance(value, str):
            identifier = _asset_id(value, "drop list ID")
            document = self._named(identifier, missing)
            distribution = (
                _unit_distribution(frozenset({identifier}))
                if document is None
                else self._document(
                    document,
                    frozenset({identifier}),
                    missing,
                )
            )
        elif isinstance(value, Mapping):
            distribution = self._document(value, frozenset(), missing)
        else:
            raise TypeError("drop list must be a string or object")
        outcomes = self._materialize(distribution)
        semantic = {
            "schema": BLOCK_DROP_PROGRAM_SCHEMA,
            "version": BLOCK_DROP_PROGRAM_VERSION,
            "missing_named_assets": sorted(missing),
            "outcomes": [asdict(outcome) for outcome in outcomes],
        }
        return LocalDropProgram(
            semantic_sha256=_canonical_sha256(semantic),
            randomized=len(outcomes) > 1,
            missing_named_assets=tuple(sorted(missing)),
            outcomes=outcomes,
        )

    def _document(
        self,
        value: Mapping[str, object],
        visited: frozenset[str],
        missing: set[str],
    ) -> _TraversalDistribution:
        document = _mapping(value, "drop list")
        unknown = set(document).difference({"Container"})
        if unknown:
            raise ValueError(
                f"drop list has unsupported fields: {sorted(unknown)}"
            )
        container = document.get("Container")
        if container is None:
            return _unit_distribution(visited)
        return self._container(
            _mapping(container, "drop list Container"),
            visited,
            missing,
        )

    def _container(
        self,
        value: Mapping[str, object],
        visited: frozenset[str],
        missing: set[str],
    ) -> _TraversalDistribution:
        kind = value.get("Type")
        if kind == "Single":
            _reject_unknown(value, {"Type", "Weight", "Item"}, "Single")
            item = _mapping(value.get("Item"), "Single.Item")
            _reject_unknown(
                item,
                {"ItemId", "Metadata", "QuantityMin", "QuantityMax"},
                "Single.Item",
            )
            minimum = _nonnegative_int(
                item.get("QuantityMin", 1),
                "Single.Item.QuantityMin",
            )
            maximum = _positive_int(
                item.get("QuantityMax", 1),
                "Single.Item.QuantityMax",
            )
            if minimum > maximum:
                raise ValueError("drop quantity minimum exceeds maximum")
            spec = _DropSpec(
                item_asset_id=_asset_id(
                    item.get("ItemId"),
                    "Single.Item.ItemId",
                ),
                quantity_min=minimum,
                quantity_max=maximum,
                metadata=item.get("Metadata"),
            )
            return {((spec,), visited): Fraction(1)}
        if kind == "Empty":
            _reject_unknown(value, {"Type", "Weight"}, "Empty")
            return _unit_distribution(visited)
        if kind == "Droplist":
            _reject_unknown(
                value,
                {"Type", "Weight", "DroplistId"},
                "Droplist",
            )
            identifier = _asset_id(
                value.get("DroplistId"),
                "Droplist.DroplistId",
            )
            if identifier in visited:
                return _unit_distribution(visited)
            next_visited = visited | {identifier}
            document = self._named(identifier, missing)
            if document is None:
                return _unit_distribution(next_visited)
            return self._document(document, next_visited, missing)
        if kind == "Multiple":
            return self._multiple(value, visited, missing)
        if kind == "Choice":
            return self._choice(value, visited, missing)
        raise ValueError(f"unsupported drop container Type {kind!r}")

    def _multiple(
        self,
        value: Mapping[str, object],
        visited: frozenset[str],
        missing: set[str],
    ) -> _TraversalDistribution:
        _reject_unknown(
            value,
            {
                "Type",
                "Weight",
                "Containers",
                "MinCount",
                "MaxCount",
            },
            "Multiple",
        )
        minimum = _nonnegative_int(
            value.get("MinCount", 1),
            "Multiple.MinCount",
        )
        maximum = _nonnegative_int(
            value.get("MaxCount", 1),
            "Multiple.MaxCount",
        )
        if minimum > maximum:
            raise ValueError("Multiple.MinCount exceeds MaxCount")
        children = _container_array(value.get("Containers"), "Multiple")
        result: _TraversalDistribution = {}
        for count, count_probability in _rounded_count_distribution(
            minimum,
            maximum,
        ).items():
            current = _unit_distribution(visited)
            for _ in range(count):
                for child in children:
                    current = self._apply_multiple_child(
                        current,
                        child,
                        missing,
                    )
            _merge_scaled(result, current, count_probability)
        return _bounded(result)

    def _apply_multiple_child(
        self,
        source: _TraversalDistribution,
        child: Mapping[str, object],
        missing: set[str],
    ) -> _TraversalDistribution:
        probability = _weight(child) / 100
        probability = min(max(probability, Fraction(0)), Fraction(1))
        result: _TraversalDistribution = {}
        for (drops, visited), state_probability in source.items():
            if probability < 1:
                _add(
                    result,
                    (drops, visited),
                    state_probability * (1 - probability),
                )
            if probability > 0:
                nested = self._container(child, visited, missing)
                for (added, next_visited), nested_probability in nested.items():
                    _add(
                        result,
                        (drops + added, next_visited),
                        state_probability * probability * nested_probability,
                    )
        return _bounded(result)

    def _choice(
        self,
        value: Mapping[str, object],
        visited: frozenset[str],
        missing: set[str],
    ) -> _TraversalDistribution:
        _reject_unknown(
            value,
            {
                "Type",
                "Weight",
                "Containers",
                "RollsMin",
                "RollsMax",
            },
            "Choice",
        )
        minimum = _positive_int(value.get("RollsMin", 1), "Choice.RollsMin")
        maximum = _positive_int(value.get("RollsMax", 1), "Choice.RollsMax")
        if minimum > maximum:
            raise ValueError("Choice.RollsMin exceeds RollsMax")
        children = _container_array(value.get("Containers"), "Choice")
        weights = tuple(_weight(child) for child in children)
        total = sum(weights, Fraction())
        if total <= 0:
            raise ValueError("Choice requires positive total weight")
        result: _TraversalDistribution = {}
        count_probability = Fraction(1, maximum - minimum + 1)
        for count in range(minimum, maximum + 1):
            current = _unit_distribution(visited)
            for _ in range(count):
                next_distribution: _TraversalDistribution = {}
                for (drops, state_visited), state_probability in current.items():
                    for child, weight in zip(
                        children,
                        weights,
                        strict=True,
                    ):
                        if weight <= 0:
                            continue
                        nested = self._container(
                            child,
                            state_visited,
                            missing,
                        )
                        for (
                            (added, next_visited),
                            nested_probability,
                        ) in nested.items():
                            _add(
                                next_distribution,
                                (drops + added, next_visited),
                                state_probability
                                * weight
                                / total
                                * nested_probability,
                            )
                current = _bounded(next_distribution)
            _merge_scaled(result, current, count_probability)
        return _bounded(result)

    def _named(
        self,
        asset_id: str,
        missing: set[str],
    ) -> Mapping[str, object] | None:
        if asset_id not in self._named_cache:
            value = self._named_drop_lookup(asset_id)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError("named drop lookup must return an object or None")
            self._named_cache[asset_id] = value
        value = self._named_cache[asset_id]
        if value is None:
            missing.add(asset_id)
        return value

    def _materialize(
        self,
        source: _TraversalDistribution,
    ) -> tuple[LocalDropOutcome, ...]:
        probabilities: dict[tuple[LocalDropStack, ...], Fraction] = {}
        for (specs, _visited), traversal_probability in source.items():
            materialized: dict[tuple[LocalDropStack, ...], Fraction] = {
                (): Fraction(1)
            }
            for spec in specs:
                item_max_stack, durability = self._item(spec.item_asset_id)
                next_materialized: dict[
                    tuple[LocalDropStack, ...],
                    Fraction,
                ] = {}
                quantity_probability = Fraction(
                    1,
                    spec.quantity_max - spec.quantity_min + 1,
                )
                metadata_hash = _metadata_hash(spec.metadata)
                for drops, probability in materialized.items():
                    for quantity in range(
                        spec.quantity_min,
                        spec.quantity_max + 1,
                    ):
                        stack = drops
                        if quantity > 0:
                            stack += (
                                LocalDropStack(
                                    item_asset_id=spec.item_asset_id,
                                    item_id=stable_asset_id(spec.item_asset_id),
                                    quantity=quantity,
                                    item_max_stack=item_max_stack,
                                    durability=durability,
                                    max_durability=durability,
                                    metadata_hash=metadata_hash,
                                ),
                            )
                        _add(
                            next_materialized,
                            stack,
                            probability * quantity_probability,
                        )
                materialized = next_materialized
            for drops, quantity_probability in materialized.items():
                if len(drops) > BLOCK_DROP_OUTPUT_CAPACITY:
                    raise ValueError("drop program exceeds output capacity")
                _add(
                    probabilities,
                    drops,
                    traversal_probability * quantity_probability,
                )
        if sum(probabilities.values(), Fraction()) != 1:
            raise ValueError("drop program probabilities do not sum to one")
        if len(probabilities) > BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY:
            raise ValueError("drop program exceeds outcome capacity")
        return tuple(
            LocalDropOutcome(
                probability_numerator=probability.numerator,
                probability_denominator=probability.denominator,
                drops=drops,
            )
            for drops, probability in sorted(
                probabilities.items(),
                key=lambda row: tuple(
                    (
                        drop.item_asset_id,
                        drop.item_id,
                        drop.quantity,
                        drop.item_max_stack,
                        drop.durability,
                        drop.max_durability,
                        drop.metadata_hash,
                    )
                    for drop in row[0]
                ),
            )
        )

    def _item(self, asset_id: str) -> tuple[int, float]:
        if asset_id not in self._item_cache:
            item = _mapping(self._item_lookup(asset_id), "drop item")
            authored_stack = item.get("MaxStack")
            if authored_stack is None:
                unique = any(
                    item.get(key) is not None
                    for key in (
                        "Tool",
                        "Weapon",
                        "Armor",
                        "BuilderTool",
                        "BlockSelectorTool",
                    )
                )
                maximum_stack = 1 if unique else 100
            else:
                maximum_stack = _positive_int(
                    authored_stack,
                    "drop item MaxStack",
                )
            durability = _nonnegative_number(
                item.get("MaxDurability", 0.0),
                "drop item MaxDurability",
            )
            self._item_cache[asset_id] = maximum_stack, durability
        return self._item_cache[asset_id]


def block_drop_program_contract() -> dict[str, object]:
    return {
        "schema": BLOCK_DROP_PROGRAM_SCHEMA,
        "version": BLOCK_DROP_PROGRAM_VERSION,
        "server_version": "0.5.7",
        "source": {
            "container_types": [
                "Single",
                "Multiple",
                "Choice",
                "Droplist",
                "Empty",
            ],
            "server_path": (
                "ItemModule.getRandomItemDrops->"
                "ItemDropContainer.populateDrops->ItemDrop.getRandomQuantity"
            ),
            "rng": "native_ThreadLocalRandom_is_not_seed_synchronizable",
        },
        "fixed_capacity": {
            "programs": BLOCK_DROP_PROGRAM_CAPACITY,
            "outcomes_per_program": BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
            "outputs_per_outcome": BLOCK_DROP_OUTPUT_CAPACITY,
            "installed_0_5_7_observed": {
                "authored_droplist_routes": 634,
                "unique_route_sources": 247,
                "deduplicated_semantic_programs": 239,
                "stochastic_routes": 367,
                "stochastic_semantic_programs": 68,
                "native_empty_missing_named_assets": 38,
                "maximum_outcomes": 60,
                "maximum_outputs": 4,
            },
        },
        "sampling": {
            "publication": "exact_output_distribution",
            "device_input": "explicit_float32_sample_in_half_open_[0,1)",
            "draw_count": (
                "collapsed_to_one_categorical_draw_per_route; "
                "native_individual_draw_sequence_not_claimed"
            ),
            "same_seed_native_replay": "unsupported",
        },
        "missing_named_asset": (
            "known_native_empty_result_from_ItemModule_null_lookup"
        ),
        "failure": (
            "unknown_fields_invalid_values_missing_items_or_capacity_fail_closed"
        ),
        "provenance": "native_source_plus_installed_assets",
    }


def block_drop_program_contract_sha256() -> str:
    return _canonical_sha256(block_drop_program_contract())


def _unit_distribution(visited: frozenset[str]) -> _TraversalDistribution:
    return {((), visited): Fraction(1)}


def _rounded_count_distribution(
    minimum: int,
    maximum: int,
) -> dict[int, Fraction]:
    if minimum == maximum:
        return {minimum: Fraction(1)}
    width = maximum - minimum
    return {
        value: (
            Fraction(1, 2 * width)
            if value in (minimum, maximum)
            else Fraction(1, width)
        )
        for value in range(minimum, maximum + 1)
    }


def _container_array(
    value: object,
    label: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}.Containers must be a nonempty array")
    return tuple(
        _mapping(child, f"{label}.Containers[{index}]")
        for index, child in enumerate(value)
    )


def _weight(value: Mapping[str, object]) -> Fraction:
    raw = value.get("Weight", 100)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("drop container Weight must be numeric")
    numeric = float(raw)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("drop container Weight must be finite and nonnegative")
    return Fraction(str(raw))


def _metadata_hash(value: object | None) -> tuple[int, ...]:
    if value is None:
        return (0,) * BLOCK_DROP_METADATA_HASH_WORDS
    if not isinstance(value, Mapping):
        raise ValueError("drop metadata must be an object")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    size = BLOCK_DROP_METADATA_HASH_WORDS * struct.calcsize(">I")
    return struct.unpack(
        f">{BLOCK_DROP_METADATA_HASH_WORDS}I",
        hashlib.sha256(payload).digest()[:size],
    )


def _merge_scaled(
    target: _TraversalDistribution,
    source: _TraversalDistribution,
    scale: Fraction,
) -> None:
    for state, probability in source.items():
        _add(target, state, probability * scale)


def _bounded(
    value: _TraversalDistribution,
) -> _TraversalDistribution:
    if len(value) > BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY * 4:
        raise ValueError("drop traversal exceeds bounded intermediate support")
    return value


def _add(target: dict, key: object, probability: Fraction) -> None:
    if probability:
        target[key] = target.get(key, Fraction()) + probability


def _reject_unknown(
    value: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value).difference(allowed | {"$Comment"})
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {sorted(unknown)}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _asset_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "/\\\0")
    ):
        raise ValueError(f"{label} must be a stable asset ID")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "BLOCK_DROP_PROGRAM_CAPACITY",
    "BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY",
    "BLOCK_DROP_PROGRAM_SCHEMA",
    "BLOCK_DROP_PROGRAM_VERSION",
    "LocalDropOutcome",
    "LocalDropProgram",
    "LocalDropProgramCompiler",
    "LocalDropStack",
    "block_drop_program_contract",
    "block_drop_program_contract_sha256",
]
