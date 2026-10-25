"""Reset option fixture_blocks (opt-in): HytaleEnv forwards it to the bridge, and rejects a bad shape before connecting.

The bridge checks the ranges and the asset ids (EnvironmentOptions, FixtureBlocksCodec). A fake connection records the
reset message and stops it there, so no bridge is needed.
"""

from __future__ import annotations

import pytest

from hytalegym.envs import hytale_env
from hytalegym.envs.hytale_env import HytaleEnv


class _StopAtReset(Exception):
    pass


class _FakeConnection:
    connections: list[_FakeConnection] = []
    messages: list[dict] = []

    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        _FakeConnection.connections.append(self)

    def connect(self) -> None:
        return None

    def send_and_recv(self, message: dict) -> dict:
        _FakeConnection.messages.append(message)
        if message["type"] == "config":
            return {"type": "ack"}
        raise _StopAtReset


@pytest.fixture
def fake_bridge(monkeypatch):
    _FakeConnection.connections = []
    _FakeConnection.messages = []
    monkeypatch.setattr(hytale_env, "BridgeConnection", _FakeConnection)
    return _FakeConnection


def _reset_message(fake_bridge, options) -> dict:
    env = HytaleEnv()
    with pytest.raises(_StopAtReset):
        env.reset(options=options)
    return next(m for m in fake_bridge.messages if m["type"] == "reset")


def test_absent_fixture_blocks_is_not_sent(fake_bridge) -> None:
    assert "fixture_blocks" not in _reset_message(fake_bridge, {})["options"]


def test_fixture_blocks_are_forwarded_as_plain_entries(fake_bridge) -> None:
    blocks = [(0, 65, 0, "Plant_Bush"), [-32, 60, 32, "Deco_SpiderWeb_Flat"]]
    sent = _reset_message(fake_bridge, {"fixture_blocks": blocks})["options"]["fixture_blocks"]
    assert sent == [[0, 65, 0, "Plant_Bush"], [-32, 60, 32, "Deco_SpiderWeb_Flat"]]


def test_numpy_integers_are_normalized_to_ints(fake_bridge) -> None:
    import numpy as np

    blocks = [(np.int64(1), np.int32(65), np.int64(-2), "Plant_Bush")]
    sent = _reset_message(fake_bridge, {"fixture_blocks": blocks})["options"]["fixture_blocks"]
    assert sent == [[1, 65, -2, "Plant_Bush"]]
    assert all(type(c) is int for c in sent[0][:3])


@pytest.mark.parametrize(
    "value",
    [
        "Plant_Bush",
        [[0, 65, 0]],
        [[0, 65, 0, "Plant_Bush", "extra"]],
        [[0.0, 65, 0, "Plant_Bush"]],
        [[True, 65, 0, "Plant_Bush"]],
        [[0, 65, 0, ""]],
        [[0, 65, 0, 7]],
    ],
)
def test_a_bad_shape_is_refused_before_any_connection(fake_bridge, value) -> None:
    env = HytaleEnv()
    with pytest.raises(ValueError, match="fixture_blocks"):
        env.reset(options={"fixture_blocks": value})
    assert fake_bridge.connections == []
    assert fake_bridge.messages == []
