"""Server version / ServerProfile -> generation profile -> data directory."""

from __future__ import annotations

import pytest

from hytalegym.server_versions import SERVER_VERSION_ENV, server_profile
from hytalegym.server_versions import _MODULES as SERVER_LINES
from hytalegym.sim.common.content import GENERATION_MODULES, data_directory, generations, load, sim_profile
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.profile import PROFILE as GEN7


@pytest.mark.parametrize(
    ("version", "generation"),
    [("0.5.7", "gen5"), ("0.5.10", "gen5"), ("0.7.2", "gen7"), ("0.7.0", "gen7")],
)
def test_versions_select_their_generation(version: str, generation: str) -> None:
    assert sim_profile(version).generation == generation
    assert sim_profile(server_profile(version)).generation == generation


def test_unsupported_versions_fail_closed() -> None:
    for version in ("0.6.9", "1.0.0", "latest"):
        with pytest.raises(ValueError):
            sim_profile(version)


def test_environment_default_follows_server_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SERVER_VERSION_ENV, raising=False)
    assert sim_profile().generation == "gen5"
    monkeypatch.setenv(SERVER_VERSION_ENV, "0.7.2")
    assert sim_profile().generation == "gen7"


def test_every_server_release_line_has_a_generation() -> None:
    assert set(GENERATION_MODULES) == set(SERVER_LINES)
    assert [profile.generation for profile in generations()] == ["gen5", "gen7"]


def test_profiles_bind_their_server_profile_and_data() -> None:
    assert GEN5.server_profile().release_line == (0, 5) and GEN5.deprecated
    gen7 = GEN7.server_profile()
    assert gen7.certified_version == "0.7.2" and gen7.release_line == (0, 7)
    assert data_directory("0.7.2") == GEN7.data_dir and GEN7.data_dir.is_dir()
    # The bridge profile's native crafting catalogue size is the extracted one.
    assert gen7.crafting_recipe_count == len(load(GEN7).keys("crafting", "recipe"))


def test_content_identities_are_stable_and_distinct() -> None:
    gen5, gen7 = load("0.5.7"), load("0.7.2")
    assert gen5.content_sha256 == load(GEN5).content_sha256
    assert gen5.content_sha256 != gen7.content_sha256
    assert gen7.content_sha256 == gen7.extracted_content_sha256  # gen 7 has no overlay
    assert gen5.content_sha256 != gen5.extracted_content_sha256  # gen 5 carries one
    assert gen5.game_version == "0.5.10" and gen7.game_version == "0.7.2"


def test_step_duration_is_profile_content() -> None:
    from . import gen5_live as live

    engine = live.ruleset()["engine"]
    step = GEN5.step_duration
    assert (step.nominal_seconds, step.loaded_seconds, step.timing_profile_count) == (
        engine["nominal_delta_seconds"],
        engine["loaded_delta_seconds"],
        engine["motion_timing_profile_count"],
    )
    gen7 = GEN7.step_duration
    assert gen7.nominal_seconds == gen7.loaded_seconds == 1.0 / 30.0 and gen7.timing_profile_count == 1


def test_content_identity_matches_the_profile_pin() -> None:
    from hytalegym.sim.gen5 import profile as gen5_profile
    from hytalegym.sim.gen7 import profile as gen7_profile

    assert load(GEN5).content_sha256 == gen5_profile.CONTENT_SHA256
    assert load(GEN7).content_sha256 == gen7_profile.CONTENT_SHA256
