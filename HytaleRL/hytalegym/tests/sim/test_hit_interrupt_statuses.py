"""Hit-interrupt status rows in the stock entity-interaction table (FIDELITY (an) part 1, row 5a).

The eight ``Hit_Interrupt_*`` statuses are rows of ``entity_interaction_status_programs_by_semantic_id()`` for the 0.7.2
server line only (``_EXTRA_ROWS_BY_CERTIFIED_VERSION`` in ``jax/combat/entities/interactions/schema/status_programs.py``);
gen 5 (0.5.7) adds none. The native learner decoder reads that table (``observation/v3/native/codec/decode.py``), so under
``HYTALERL_SERVER_VERSION=0.7.2`` it knows the eight statuses and does not fail closed on them. The archive test reads the
eight files from the installed 0.7.2 Assets.zip and skips when it is absent.
"""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.entities.interactions.schema import status_programs as table
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_COMBO_1_EFFECT_ID,
    CROSSBOW_COMBO_2_EFFECT_ID,
)
from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_PHYSICAL,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)
from hytalegym.jax.combat.observation.v3.native.codec import decode
from hytalegym.server_versions import SERVER_VERSION_ENV

from .support import ARCHIVES

#: name -> (duration seconds, OverlapBehavior as the asset writes it, horizontal speed multiplier). Every effect is finite,
#: not a debuff and carries no damage; only Flinch has a speed multiplier (HorizontalSpeedMultiplier 0).
EXPECTED = {
    "Hit_Interrupt_Eligible": (1.0, "Overwrite", 1.0),
    "Hit_Interrupt_Flinch": (0.5, "Ignore", 0.0),
    "Hit_Interrupt_Stunned": (0.7, "Overwrite", 1.0),
    "Hit_Interrupt_Recent_Reaction_A": (4.0, "Ignore", 1.0),
    "Hit_Interrupt_Recent_Reaction_B": (4.0, "Ignore", 1.0),
    "Hit_Interrupt_Recovery_Pending": (2.0, "Ignore", 1.0),
    "Hit_Interrupt_Enraged": (8.0, "Overwrite", 1.0),
    "Hit_Interrupt_Protected_Attack": (8.0, "Ignore", 1.0),
}
OVERLAP = {"Ignore": STATUS_OVERLAP_IGNORE, "Overwrite": STATUS_OVERLAP_OVERWRITE}


def _decoder_programs() -> dict:
    """The stock native decoder's status table for an empty compiled event table: only the role and interaction rows."""

    shape = (1, 1, 1)
    loadout = SimpleNamespace(
        event_mask=np.zeros(shape, dtype=bool),
        event_f32=np.zeros(shape, dtype=np.float32),
        event_i32=np.zeros(shape, dtype=np.int32),
        event_flags=np.zeros(shape, dtype=np.uint32),
    )
    return decode._status_programs(loadout)


def test_the_table_adds_nothing_on_0_5_7_and_the_eight_on_0_7_2(monkeypatch) -> None:
    crossbow = {CROSSBOW_COMBO_1_EFFECT_ID, CROSSBOW_COMBO_2_EFFECT_ID}
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.5.7")
    assert set(table.entity_interaction_status_programs_by_semantic_id()) == crossbow
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.7.2")
    assert set(table.entity_interaction_status_programs_by_semantic_id()) == crossbow | {
        semantic_id(name) for name in EXPECTED
    }


def test_0_5_7_decoder_has_no_program_for_any_of_them(monkeypatch) -> None:
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.5.7")
    assert not {semantic_id(name) for name in EXPECTED} & set(_decoder_programs())


def test_declared_rows_carry_the_asset_facts(monkeypatch) -> None:
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.7.2")
    rows = table.entity_interaction_status_programs_by_semantic_id()
    for name, (duration, overlap, speed) in EXPECTED.items():
        row = rows[semantic_id(name)]
        assert (row.duration_seconds, row.overlap_mode, row.speed_multiplier) == (duration, OVERLAP[overlap], speed), name
        assert (row.infinite, row.flags, row.value_percent, row.source_entity_is_other) == (False, 0, False, False), name
        assert (row.damage_per_cycle, row.healing_per_cycle, row.resource_id) == (0.0, 0.0, -1), name
        assert row.damage_cause == DAMAGE_PHYSICAL, name


@pytest.mark.skipif(not ARCHIVES["gen7"].exists(), reason="the 0.7.2 Assets.zip is not installed")
def test_asset_facts_match_the_installed_0_7_2_archive() -> None:
    with zipfile.ZipFile(ARCHIVES["gen7"]) as archive:
        members = archive.namelist()
        for name, (duration, overlap, speed) in EXPECTED.items():
            (member,) = [m for m in members if m.endswith(f"Server/Entity/Effects/Status/{name}.json")]
            data = json.loads(archive.read(member).decode("utf-8-sig"))
            assert float(data["Duration"]) == duration, name
            assert data["OverlapBehavior"] == overlap, name
            assert data.get("Infinite", False) is False and data.get("Debuff", False) is False, name
            assert float(data.get("ApplicationEffects", {}).get("HorizontalSpeedMultiplier", 1.0)) == speed, name


def test_a_semantic_id_collision_raises(monkeypatch) -> None:
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.7.2")
    entry = table._HIT_INTERRUPT_STATUSES_072[0]
    monkeypatch.setitem(table._EXTRA_ROWS_BY_CERTIFIED_VERSION, "0.7.2", (entry, entry))
    with pytest.raises(ValueError, match="collision"):
        table.entity_interaction_status_programs_by_semantic_id()


def test_an_unsupported_server_line_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.6.9")
    with pytest.raises(ValueError, match="unsupported"):
        table.entity_interaction_status_programs_by_semantic_id()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_0_7_2_decoder_knows_each_status_as_a_finite_program(monkeypatch, name) -> None:
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.7.2")
    program = _decoder_programs().get(semantic_id(name))
    assert program is not None, name
    _, overlap, speed = EXPECTED[name]
    assert program.infinite is False and program.flags == 0 and program.damage == 0.0, name
    assert program.overlap_mode == OVERLAP[overlap] and program.speed_multiplier == speed, name
