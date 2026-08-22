"""Shared cross-process lease for native evidence on one Hytale listener."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import secrets
from threading import RLock
from typing import BinaryIO, Mapping, Sequence

NATIVE_EVIDENCE_LEASE_SCHEMA = "hytalerl_native_evidence_lease_v1"

_TOKEN_ENV = "HYTALERL_NATIVE_EVIDENCE_LEASE_TOKEN"
_HOST_ENV = "HYTALERL_NATIVE_EVIDENCE_LEASE_HOST"
_PORT_ENV = "HYTALERL_NATIVE_EVIDENCE_LEASE_PORT"
_ROOT_ENV = "HYTALERL_NATIVE_EVIDENCE_LEASE_ROOT"
_ENVIRONMENT_KEYS = (_TOKEN_ENV, _HOST_ENV, _PORT_ENV, _ROOT_ENV)
_LeaseKey = tuple[str, str, int]
_PROCESS_OWNERS: dict[_LeaseKey, NativeEvidenceLease] = {}
_PROCESS_OWNERS_LOCK = RLock()


class NativeEvidenceLease:
    """An owning OS lock or a validated child-process join."""

    __slots__ = (
        "_closed",
        "_owner_signal_json",
        "_previous_environment",
        "_registry_key",
        "_stream",
        "joined",
        "signal_path",
        "token",
    )

    def __init__(
        self,
        *,
        stream: BinaryIO | None,
        signal_path: Path,
        token: str,
        joined: bool,
        previous_environment: Mapping[str, str | None] | None = None,
        owner_signal: Mapping[str, object] | None = None,
        registry_key: _LeaseKey | None = None,
    ) -> None:
        self._stream = stream
        self._owner_signal_json = (
            None
            if owner_signal is None
            else json.dumps(owner_signal, sort_keys=True)
        )
        self._previous_environment = dict(previous_environment or {})
        self._registry_key = registry_key
        self._closed = False
        self.signal_path = signal_path
        self.token = token
        self.joined = joined

    def __enter__(self) -> NativeEvidenceLease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        # The registry lock makes owner removal and OS-lock release one atomic
        # transition from the perspective of a same-process join. Otherwise a
        # racing join can find the owner after ``_closed`` becomes true but
        # before it leaves the registry, or join while its stream is closing.
        with _PROCESS_OWNERS_LOCK:
            if self._closed:
                return
            self._closed = True
            if self.joined:
                _restore_environment(self._previous_environment)
                return
            _unregister_process_owner(self)
            try:
                _unlink_owned_signal(self.signal_path, self.token)
            finally:
                if self._stream is not None:
                    self._stream.close()
                _restore_environment(self._previous_environment)


def acquire_native_evidence_lease(
    repository_root: str | Path | None = None,
    *,
    host: str,
    port: int,
    purpose: str,
    stage: str | Path | None = None,
    evidence_kinds: Sequence[str] = (),
) -> NativeEvidenceLease:
    """Acquire or join the non-blocking lease for one native listener."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else native_evidence_repository_root()
    )
    normalized_host = _host(host)
    normalized_port = _port(port)
    normalized_purpose = _purpose(purpose)
    registry_key = (str(root), normalized_host, normalized_port)
    signal_path, lock_path = _lease_paths(root, normalized_host, normalized_port)
    joined = _join_process_owner(registry_key, signal_path)
    if joined is not None:
        return joined
    joined = _join_inherited_lease(
        root,
        normalized_host,
        normalized_port,
        signal_path,
    )
    if joined is not None:
        return joined

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"\n")
        stream.flush()
    stream.seek(0)
    try:
        _lock_stream(stream)
    except OSError as error:
        stream.close()
        owner = _read_signal_or_empty(signal_path)
        raise RuntimeError(
            "native evidence already active for "
            f"{normalized_host}:{normalized_port}; "
            f"owner_pid={owner.get('pid', 'unknown')} "
            f"purpose={owner.get('purpose', 'unknown')}"
        ) from error

    token = secrets.token_hex(16)
    previous_environment = {
        name: os.environ.get(name) for name in _ENVIRONMENT_KEYS
    }
    now = datetime.now(UTC).isoformat()
    owner_signal = {
        "schema": NATIVE_EVIDENCE_LEASE_SCHEMA,
        "host": normalized_host,
        "port": normalized_port,
        "pid": os.getpid(),
        "purpose": normalized_purpose,
        "token": token,
        "stage": None if stage is None else str(Path(stage).resolve()),
        "started_at_utc": now,
        "updated_at_utc": now,
        "phase": "startup",
        "phase_status": "running",
        "evidence_kinds": [str(value) for value in evidence_kinds],
        "detail": {},
    }
    try:
        _write_json(
            signal_path,
            owner_signal,
        )
        os.environ[_TOKEN_ENV] = token
        os.environ[_HOST_ENV] = normalized_host
        os.environ[_PORT_ENV] = str(normalized_port)
        os.environ[_ROOT_ENV] = str(root)
    except Exception:
        stream.close()
        _restore_environment(previous_environment)
        raise
    lease = NativeEvidenceLease(
        stream=stream,
        signal_path=signal_path,
        token=token,
        joined=False,
        previous_environment=previous_environment,
        owner_signal=owner_signal,
        registry_key=registry_key,
    )
    _register_process_owner(lease)
    return lease


def update_native_evidence_lease(
    signal_path: str | Path,
    *,
    phase: str,
    phase_status: str,
    detail: Mapping[str, object] | None = None,
) -> None:
    """Update owner progress without changing lease identity."""

    _validate_progress(phase, phase_status)
    path = Path(signal_path)
    value = _read_json(path)
    if value.get("schema") != NATIVE_EVIDENCE_LEASE_SCHEMA:
        raise RuntimeError("native evidence lease schema changed")
    _apply_progress(value, phase, phase_status, detail)
    _write_json(path, value)


def update_owned_native_evidence_lease(
    lease: NativeEvidenceLease,
    *,
    phase: str,
    phase_status: str,
    detail: Mapping[str, object] | None = None,
) -> None:
    """Update owner progress, repairing only a deleted owner signal.

    The open owner stream is the handle that retains the OS byte-range lock.
    Joined children have no such stream or immutable owner envelope and cannot
    use this recovery path. Existing corrupt or foreign signals remain strict
    failures; only a genuinely absent signal is recreated.
    """

    _validate_progress(phase, phase_status)
    owner = _owned_signal_envelope(lease)
    path = lease.signal_path
    try:
        value = _read_json(path)
    except FileNotFoundError:
        value = owner
        _apply_progress(value, phase, phase_status, detail)
        try:
            _write_json_if_missing(path, value)
            return
        except FileExistsError:
            # Another writer restored the signal after our read. It must still
            # be the exact owner envelope before this lease may update it.
            value = _read_json(path)
    _require_owner_signal(value, owner)
    _apply_progress(value, phase, phase_status, detail)
    _write_json(path, value)


def native_evidence_repository_root() -> Path:
    """Resolve the HytaleRL root without relying on the current directory."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "hytalegym" / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("HytaleRL repository root is unavailable")


def require_native_reset_bridge_identity(
    reset: Mapping[str, object],
    expected_sha256: str,
) -> Mapping[str, object]:
    """Return reset info only when it names the expected runtime bridge."""

    expected = _sha256(expected_sha256)
    info = reset.get("info")
    if not isinstance(info, Mapping):
        raise ValueError("native reset omitted info")
    observed = info.get("bridge_sha256")
    if not isinstance(observed, str) or _sha256(observed) != expected:
        raise ValueError("native reset reported a different bridge identity")
    return info


def select_native_control_seed(
    supplied: int | None,
    *,
    excluded_seeds: Sequence[int],
) -> int:
    """Select one bounded control seed distinct from named evidence seeds."""

    excluded = frozenset(_control_seed(value) for value in excluded_seeds)
    if supplied is not None:
        result = _control_seed(supplied)
        if result in excluded:
            raise ValueError("control seed must be distinct from evidence seeds")
        return result
    result = secrets.randbelow(2**31 - 1) + 1
    while result in excluded:
        result = 1 if result == 2**31 - 1 else result + 1
    return result


def _join_inherited_lease(
    root: Path,
    host: str,
    port: int,
    signal_path: Path,
) -> NativeEvidenceLease | None:
    token = os.environ.get(_TOKEN_ENV)
    if token is None:
        return None
    inherited_endpoint = (
        os.environ.get(_HOST_ENV),
        os.environ.get(_PORT_ENV),
        os.environ.get(_ROOT_ENV),
    )
    requested_endpoint = (host, str(port), str(root))
    if inherited_endpoint != requested_endpoint:
        return None
    try:
        signal = _read_json(signal_path)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("inherited native evidence lease is stale or invalid") from error
    if (
        signal.get("schema") != NATIVE_EVIDENCE_LEASE_SCHEMA
        or signal.get("token") != token
        or signal.get("host") != host
        or signal.get("port") != port
    ):
        raise RuntimeError("inherited native evidence lease is stale or invalid")
    return NativeEvidenceLease(
        stream=None,
        signal_path=signal_path,
        token=token,
        joined=True,
    )


def _join_process_owner(
    key: _LeaseKey,
    signal_path: Path,
) -> NativeEvidenceLease | None:
    """Join this process's live owner even when another endpoint is current.

    The environment transport can name only one endpoint. A multi-listener
    probe therefore temporarily points it at the most recently acquired
    endpoint. Without this registry, nesting back into an earlier endpoint can
    acquire a second same-process owner and unlink the original owner's signal
    on close. The retained stream proves that the OS lock is still live, so it
    is also sufficient authority to restore an accidentally removed signal.
    """

    with _PROCESS_OWNERS_LOCK:
        owner = _PROCESS_OWNERS.get(key)
        if owner is None:
            return None
        envelope = _owned_signal_envelope(owner)
        try:
            signal = _read_json(signal_path)
        except FileNotFoundError:
            try:
                _write_json_if_missing(signal_path, envelope)
                signal = envelope
            except FileExistsError:
                signal = _read_json(signal_path)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "same-process native evidence lease signal is invalid"
            ) from error
        _require_owner_signal(signal, envelope)
        previous_environment = {
            name: os.environ.get(name) for name in _ENVIRONMENT_KEYS
        }
        root, host, port = key
        os.environ[_TOKEN_ENV] = owner.token
        os.environ[_HOST_ENV] = host
        os.environ[_PORT_ENV] = str(port)
        os.environ[_ROOT_ENV] = root
        return NativeEvidenceLease(
            stream=None,
            signal_path=signal_path,
            token=owner.token,
            joined=True,
            previous_environment=previous_environment,
        )


def _register_process_owner(lease: NativeEvidenceLease) -> None:
    key = lease._registry_key
    if key is None:
        raise RuntimeError("native evidence owner omitted its registry key")
    with _PROCESS_OWNERS_LOCK:
        existing = _PROCESS_OWNERS.get(key)
        if existing is not None and existing is not lease:
            raise RuntimeError("native evidence endpoint already has a process owner")
        _PROCESS_OWNERS[key] = lease


def _unregister_process_owner(lease: NativeEvidenceLease) -> None:
    key = lease._registry_key
    if key is None:
        return
    with _PROCESS_OWNERS_LOCK:
        if _PROCESS_OWNERS.get(key) is lease:
            del _PROCESS_OWNERS[key]


def _lease_paths(root: Path, host: str, port: int) -> tuple[Path, Path]:
    safe_host = "".join(value if value.isalnum() else "_" for value in host)
    stem = f"native-world-recertification-{safe_host}-{port}"
    directory = root / ".codex-local"
    return directory / f"{stem}.json", directory / f"{stem}.lock"


def _lock_stream(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlink_owned_signal(path: Path, token: str) -> None:
    try:
        value = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if value.get("token") == token:
        path.unlink(missing_ok=True)


def _restore_environment(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _read_signal_or_empty(path: Path) -> dict[str, object]:
    try:
        return _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _validate_progress(phase: str, phase_status: str) -> None:
    if not phase or phase_status not in {"running", "passed", "failed"}:
        raise ValueError("native evidence phase/status is invalid")


def _apply_progress(
    value: dict[str, object],
    phase: str,
    phase_status: str,
    detail: Mapping[str, object] | None,
) -> None:
    value["phase"] = phase
    value["phase_status"] = phase_status
    value["updated_at_utc"] = datetime.now(UTC).isoformat()
    value["detail"] = dict(detail or {})


def _owned_signal_envelope(lease: NativeEvidenceLease) -> dict[str, object]:
    if lease.joined:
        raise RuntimeError("joined native evidence lease cannot repair owner signal")
    if (
        lease._closed
        or lease._stream is None
        or lease._stream.closed
        or lease._owner_signal_json is None
    ):
        raise RuntimeError("native evidence OS lock is no longer held by lease")
    value = json.loads(lease._owner_signal_json)
    if not isinstance(value, dict) or value.get("token") != lease.token:
        raise RuntimeError("native evidence owner envelope is invalid")
    return value


def _require_owner_signal(
    value: Mapping[str, object],
    owner: Mapping[str, object],
) -> None:
    immutable_fields = (
        "schema",
        "host",
        "port",
        "pid",
        "purpose",
        "token",
        "stage",
        "started_at_utc",
        "evidence_kinds",
    )
    if any(value.get(field) != owner.get(field) for field in immutable_fields):
        raise RuntimeError("native evidence owner signal changed")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_if_missing(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _host(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("native evidence host must be non-empty")
    normalized = value.strip().casefold().rstrip(".")
    if normalized in {"localhost", "localhost.localdomain", "::1", "[::1]"}:
        return "127.0.0.1"
    return normalized


def _port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("native evidence port must be an integer")
    if value <= 0 or value > 65535:
        raise ValueError("native evidence port is outside [1,65535]")
    return value


def _purpose(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("native evidence purpose must be non-empty")
    return value.strip()


def _control_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("control seed must be an integer")
    if not 0 < value < 2**31:
        raise ValueError("control seed must be in (0, 2^31)")
    return value


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("bridge identity must be a SHA-256")
    return value.upper()


__all__ = [
    "NATIVE_EVIDENCE_LEASE_SCHEMA",
    "NativeEvidenceLease",
    "acquire_native_evidence_lease",
    "native_evidence_repository_root",
    "require_native_reset_bridge_identity",
    "select_native_control_seed",
    "update_native_evidence_lease",
    "update_owned_native_evidence_lease",
]
