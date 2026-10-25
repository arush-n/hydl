"""Async loader for the bridge's sound-event table.

The bridge's 3-D audio events carry a server-internal ``sound_event_index``. This module fetches the server's table
(index -> asset id) over its own short-lived connection, so the env socket is never shared across threads. It mirrors
``block_type_table.py``: lookups never block, and fetch errors are recorded, never raised into the tick path.
"""

from __future__ import annotations

import threading

from hytalegym.utils.connection import BridgeConnection

SOUND_EVENT_TABLE_SCHEMA = "hytalerl_sound_event_table_v1"
SOUND_EVENT_TABLE_VERSION = 1


def fetch_sound_event_table(
    host: str,
    port: int,
    timeout: float = 30.0,
) -> list[tuple[int, str]]:
    """Fetch and validate the table; returns ``[(sound_event_index, asset_id), ...]``."""

    connection = BridgeConnection(host, port, timeout=timeout)
    connection.connect()
    try:
        response = connection.send_and_recv({"type": "sound_event_table"})
    finally:
        connection.close()
    if response.get("type") != "sound_event_table":
        raise ValueError(f"unexpected sound-event response: {response.get('type')!r}")
    if response.get("schema") != SOUND_EVENT_TABLE_SCHEMA:
        raise ValueError(f"unexpected sound-event schema: {response.get('schema')!r}")
    if response.get("version") != SOUND_EVENT_TABLE_VERSION:
        raise ValueError(f"unexpected sound-event version: {response.get('version')!r}")
    entries = []
    for row in response.get("entries", []):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("sound-event entry must be [index, asset_id]")
        index, asset_id = row
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(asset_id, str):
            raise ValueError("sound-event entry has the wrong field types")
        entries.append((index, asset_id))
    if len(entries) != response.get("count", len(entries)):
        raise ValueError("sound-event count does not match its entries")
    return entries


class SoundEventTableLoader:
    """Fetches the sound-event table on a daemon thread; lookups never block.

    The table is cached on the class, keyed by (host, port), so one server process is fetched once. Fetch errors are
    recorded in ``error`` and leave the loader not ready.
    """

    _table_cache: dict[tuple[str, int], dict[int, str]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.error: str | None = None
        self._table: dict[int, str] | None = None
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._table is not None

    @property
    def asset_ids(self) -> dict[int, str]:
        """index -> asset id; empty until the table has arrived."""

        return self._table if self._table is not None else {}

    def start(self) -> None:
        """Begin loading; a no-op after the first call or once cached."""

        if self._thread is not None or self._table is not None:
            return
        key = (self.host, self.port)
        with type(self)._cache_lock:
            cached = type(self)._table_cache.get(key)
        if cached is not None:
            self._table = cached
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"sound-event-table-{self.host}:{self.port}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            table = dict(fetch_sound_event_table(self.host, self.port, self.timeout))
        except Exception as exc:  # recorded, never raised into the tick path
            self.error = f"{type(exc).__name__}: {exc}"
            return
        with type(self)._cache_lock:
            type(self)._table_cache[(self.host, self.port)] = table
        self._table = table


__all__: list[str] = [
    "SOUND_EVENT_TABLE_SCHEMA",
    "SOUND_EVENT_TABLE_VERSION",
    "SoundEventTableLoader",
    "fetch_sound_event_table",
]
