"""Backend resolution and metadata for ADK entry points.

The key invariants:

* Backend choice is explicit in config, not inferred from a port number.
* ``native`` is an evidence-only lane and cannot be used for training entry points.
* Host/port are validated before any transport is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Literal

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - exercised where JAX is absent
    jax = None


Backend = Literal["jax", "simulator", "native"]
TRAINING_ENTRY_POINTS: Final[frozenset[str]] = frozenset(
    {
        "train",
        "trainer",
        "training",
        "self_play",
        "policy_optimization",
    }
)


def _normalize_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("backend port must be an integer")
    if not (1 <= value <= 65535):
        raise ValueError("backend port must be within [1,65535]")
    return int(value)


def _normalize_host(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("backend host must be a non-empty string")
    host = value.strip()
    if not host:
        raise ValueError("backend host must be a non-empty string")
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return "127.0.0.1"
    return host


def _normalize_backend(value: object) -> Backend:
    if not isinstance(value, str):
        raise TypeError("backend must be one of 'jax', 'simulator', 'native'")
    normalized = value.strip().lower()
    if normalized not in {"jax", "simulator", "native"}:
        raise ValueError(f"unknown backend {value!r}")
    return normalized  # type: ignore[return-value]


def _normalize_entry_point(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _current_device_kind() -> str | None:
    if jax is None:
        return None
    try:
        return jax.devices()[0].device_kind
    except Exception:
        return None


def resolve_backend(config: Mapping[str, Any] | None = None) -> "BackendSelection":
    """Resolve runtime backend from explicit config.

    Parameters are intentionally generic so callers can pass either a full
    object config mapping or a lightweight per-call dict.
    """

    options: Mapping[str, Any] = {} if config is None else config
    backend = _normalize_backend(options.get("backend", "jax"))
    entry_point = _normalize_entry_point(options.get("entry_point"))
    host = _normalize_host(options.get("host", "127.0.0.1"))
    port = _normalize_port(options.get("port", 5556))

    if backend == "native" and (entry_point in TRAINING_ENTRY_POINTS):
        raise RuntimeError(
            "native backend is evidence-only; training entry points must use jax"
        )

    return BackendSelection(
        backend=backend,
        host=host,
        port=port,
        entry_point=entry_point,
        device_kind=_current_device_kind() if backend == "jax" else None,
    )


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """Runtime backend choice plus port+host metadata."""

    backend: Backend
    host: str
    port: int
    entry_point: str = ""
    device_kind: str | None = None

    @property
    def is_training(self) -> bool:
        return self.entry_point in TRAINING_ENTRY_POINTS


__all__ = [
    "Backend",
    "BackendSelection",
    "TRAINING_ENTRY_POINTS",
    "resolve_backend",
]
