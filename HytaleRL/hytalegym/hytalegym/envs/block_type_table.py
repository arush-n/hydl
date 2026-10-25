"""Async loader for the bridge's block-type table.

The bridge's ``nearby_blocks`` rows carry server-internal runtime block ids.
This module fetches the server's block-type table (runtime id -> asset id) over
its own short-lived connection, so the env socket is never shared across
threads, and maps each asset id to its semantic id. It lives beside
``hytale_env.py`` because the env is its only consumer.
"""

from __future__ import annotations

import threading
import numpy as np

from hytalegym.utils.connection import BridgeConnection

BLOCK_TYPE_TABLE_SCHEMA = "hytalerl_block_type_table_v1"
BLOCK_TYPE_TABLE_VERSION = 1


def fetch_block_type_table(
    host: str,
    port: int,
    timeout: float = 30.0,
) -> list[tuple[int, str]]:
    """Fetch and validate the table; returns ``[(runtime_id, asset_id), ...]``."""

    connection = BridgeConnection(host, port, timeout=timeout)
    connection.connect()
    try:
        response = connection.send_and_recv({"type": "block_type_table"})
    finally:
        connection.close()
    if response.get("type") != "block_type_table":
        raise ValueError(f"unexpected block-type response: {response.get('type')!r}")
    if response.get("schema") != BLOCK_TYPE_TABLE_SCHEMA:
        raise ValueError(f"unexpected block-type schema: {response.get('schema')!r}")
    if response.get("version") != BLOCK_TYPE_TABLE_VERSION:
        raise ValueError(f"unexpected block-type version: {response.get('version')!r}")
    entries = []
    for row in response.get("entries", []):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("block-type entry must be [runtime_id, asset_id]")
        runtime_id, asset_id = row
        if not isinstance(runtime_id, int) or not isinstance(asset_id, str):
            raise ValueError("block-type entry has the wrong field types")
        entries.append((runtime_id, asset_id))
    if len(entries) != response.get("count", len(entries)):
        raise ValueError("block-type count does not match its entries")
    return entries


def build_semantic_table(entries: list[tuple[int, str]]) -> np.ndarray:
    """Return an int32 array indexed by runtime id holding semantic ids.

    Runtime id 0 is air and maps to 0. Ids absent from the table map to 0.
    """

    # Imported here: ``import hytalegym`` must not load ``hytalegym.jax``, or a gen-7 render's hooks read the
    # profile before a launcher can switch constructs on (tests/sim/test_import_order.py).
    from hytalegym.jax.combat.contracts.semantic import semantic_id

    size = max((runtime_id for runtime_id, _ in entries), default=0) + 1
    table = np.zeros(size, dtype=np.int32)
    for runtime_id, asset_id in entries:
        if runtime_id > 0:
            table[runtime_id] = semantic_id(asset_id)
    return table


class BlockTypeIdLoader:
    """Fetches the block-type table on a daemon thread; lookups never block.

    The table is cached on the class, keyed by (host, port), so one server
    process is fetched once. Fetch errors are recorded in ``error`` and leave
    the loader not ready; they are never raised into the caller.
    """

    _table_cache: dict[tuple[str, int], np.ndarray] = {}
    _cache_lock = threading.Lock()

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.error: str | None = None
        self._table: np.ndarray | None = None
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._table is not None

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
            name=f"block-type-table-{self.host}:{self.port}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            table = build_semantic_table(
                fetch_block_type_table(self.host, self.port, self.timeout)
            )
        except Exception as exc:  # recorded, never raised into the tick path
            self.error = f"{type(exc).__name__}: {exc}"
            return
        with type(self)._cache_lock:
            type(self)._table_cache[(self.host, self.port)] = table
        self._table = table

    def lookup(self, runtime_ids: np.ndarray) -> tuple[np.ndarray, int]:
        """Map runtime ids to semantic ids of the same shape.

        Returns ``(semantic_ids, unknown_count)``. ``unknown_count`` is the
        number of nonzero runtime ids that have no table entry. Before ready,
        every semantic id is 0.
        """

        ids = np.asarray(runtime_ids, dtype=np.int64)
        table = self._table
        if table is None:
            return np.zeros(ids.shape, dtype=np.int32), 0
        in_range = (ids >= 0) & (ids < table.shape[0])
        semantic = np.where(in_range, table[np.clip(ids, 0, table.shape[0] - 1)], 0)
        semantic = semantic.astype(np.int32)
        unknown = int(np.count_nonzero((ids != 0) & (semantic == 0)))
        return semantic, unknown


__all__: list[str] = [
    "BLOCK_TYPE_TABLE_SCHEMA",
    "BLOCK_TYPE_TABLE_VERSION",
    "BlockTypeIdLoader",
    "build_semantic_table",
    "fetch_block_type_table",
]
