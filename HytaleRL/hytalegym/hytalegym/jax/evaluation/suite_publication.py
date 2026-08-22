"""Stable publication and immutable lookup for flat Arsenal suites."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


PUBLICATION_SCHEMA = "hytalerl_arsenal_flat_suite_publication_v1"
CURRENT_TRAINED_PUBLICATION_FILENAME = (
    "hytalerl-arsenal-flat-current-trained.json"
)
CONTENT_ID_BASIS = "hytalerl_arsenal_flat_content"
CONTENT_ID_PREFIX = f"{CONTENT_ID_BASIS}_"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical JSON identity used by benchmark artifacts."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    """Return the byte identity of one immutable publication file."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def content_addressed_suite_id(suite: Mapping[str, Any]) -> str:
    """Derive a stable suite ID from every field except the ID itself."""

    identity_payload = dict(suite)
    identity_payload["id"] = CONTENT_ID_BASIS
    identity = canonical_sha256(identity_payload)
    return f"{CONTENT_ID_PREFIX}{identity[:16].lower()}"


def is_content_addressed_suite(suite: Mapping[str, Any]) -> bool:
    """Return whether ``suite.id`` matches its complete semantic content."""

    return suite.get("id") == content_addressed_suite_id(suite)


def current_trained_suite_publication_path(
    repository_root: str | Path,
) -> Path:
    """Return the stable pointer to the current immutable trained suite."""

    return (
        Path(repository_root)
        / "artifacts"
        / "evaluation"
        / CURRENT_TRAINED_PUBLICATION_FILENAME
    )


def content_addressed_suite_path(
    repository_root: str | Path,
    suite: Mapping[str, Any],
) -> Path:
    """Return the immutable manifest path implied by one complete suite."""

    suite_sha256 = canonical_sha256(suite)
    return (
        Path(repository_root)
        / "artifacts"
        / "evaluation"
        / f"suite-content-{suite_sha256[:8].lower()}"
        / "suite.json"
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def verify_current_trained_suite_publication(
    repository_root: str | Path,
    live_suite: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the stable pointer, immutable bytes, and complete live suite."""

    root = Path(repository_root)
    pointer_path = current_trained_suite_publication_path(root)
    if not pointer_path.is_file():
        raise RuntimeError(
            f"current trained suite publication is missing: {pointer_path}"
        )
    pointer = _json_object(pointer_path)
    if pointer.get("schema") != PUBLICATION_SCHEMA:
        raise RuntimeError(
            f"current trained suite publication schema drifted: {pointer_path}"
        )

    live = dict(live_suite)
    if not is_content_addressed_suite(live):
        raise RuntimeError("live trained suite ID does not cover its content")
    current_sha256 = canonical_sha256(live)
    published_sha256 = pointer.get("suite_sha256")
    if published_sha256 != current_sha256:
        raise RuntimeError(
            "current trained suite publication is stale at stable pointer "
            f"{pointer_path}: published={published_sha256}, "
            f"current={current_sha256}"
        )
    if pointer.get("suite_id") != live["id"]:
        raise RuntimeError(
            f"current trained suite publication ID drifted: {pointer_path}"
        )

    expected_reference = (
        f"suite-content-{current_sha256[:8].lower()}/suite.json"
    )
    if pointer.get("suite") != expected_reference:
        raise RuntimeError(
            "current trained suite publication does not reference its "
            f"immutable content address: {pointer_path}"
        )
    suite_path = content_addressed_suite_path(root, live)
    if not suite_path.is_file():
        raise RuntimeError(f"published trained suite is missing: {suite_path}")
    if pointer.get("suite_file_sha256") != file_sha256(suite_path):
        raise RuntimeError(f"published trained suite bytes changed: {suite_path}")
    published_suite = verify_content_addressed_suite_manifest(root, live)
    return pointer, published_suite


def publish_current_trained_suite(
    repository_root: str | Path,
    live_suite: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Publish immutable suite bytes, then atomically advance the pointer.

    Existing content-addressed bytes are accepted only when they are exactly
    equal to ``live_suite``. The stable pointer is replaced last, so readers
    can observe either the previous complete publication or the new complete
    publication, never a pointer to missing suite bytes.
    """

    root = Path(repository_root)
    suite = dict(live_suite)
    if not is_content_addressed_suite(suite):
        raise ValueError("live trained suite ID does not cover its content")
    suite_path = content_addressed_suite_path(root, suite)
    pointer_path = current_trained_suite_publication_path(root)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if suite_path.exists():
        if _json_object(suite_path) != suite:
            raise RuntimeError(
                f"content-addressed suite path is occupied: {suite_path}"
            )
    else:
        _write_json_atomic(suite_path, suite)

    suite_sha256 = canonical_sha256(suite)
    pointer = {
        "schema": PUBLICATION_SCHEMA,
        "suite": f"suite-content-{suite_sha256[:8].lower()}/suite.json",
        "suite_file_sha256": file_sha256(suite_path),
        "suite_id": suite["id"],
        "suite_sha256": suite_sha256,
    }
    _write_json_atomic(pointer_path, pointer)
    verify_current_trained_suite_publication(root, suite)
    return pointer_path, suite_path


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_content_addressed_suite_manifest(
    repository_root: str | Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an intrinsically identified suite to have immutable bytes."""

    expected = dict(suite)
    if not is_content_addressed_suite(expected):
        raise RuntimeError("suite ID does not cover its complete content")
    suite_path = content_addressed_suite_path(repository_root, expected)
    if not suite_path.is_file():
        raise RuntimeError(
            f"content-addressed suite manifest is missing: {suite_path}"
        )
    published = _json_object(suite_path)
    if canonical_sha256(published) != canonical_sha256(expected):
        raise RuntimeError(
            f"content-addressed suite manifest drifted: {suite_path}"
        )
    if not is_content_addressed_suite(published):
        raise RuntimeError(
            f"published suite ID does not cover its content: {suite_path}"
        )
    if published != expected:
        raise RuntimeError(
            f"content-addressed suite differs from manifest: {suite_path}"
        )
    return published


def load_frozen_suite_from_artifact(
    repository_root: str | Path,
    artifact_filename: str,
    *,
    expected_id: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load and verify an immutable suite embedded in a historical artifact."""

    artifact_path = (
        Path(repository_root)
        / "artifacts"
        / "evaluation"
        / artifact_filename
    )
    if not artifact_path.is_file():
        raise RuntimeError(f"historical suite artifact is missing: {artifact_path}")
    artifact = _json_object(artifact_path)
    suite = artifact.get("suite")
    if not isinstance(suite, dict):
        raise RuntimeError(
            f"historical artifact has no suite object: {artifact_path}"
        )
    if suite.get("id") != expected_id:
        raise RuntimeError(f"historical suite ID drifted: {artifact_path}")
    actual_sha256 = canonical_sha256(suite)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"historical suite content drifted: {artifact_path}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    artifact_suite_sha256 = artifact.get("suite_sha256")
    if (
        artifact_suite_sha256 is not None
        and artifact_suite_sha256 != expected_sha256
    ):
        raise RuntimeError(
            f"historical artifact suite pin drifted: {artifact_path}"
        )
    return suite


__all__ = [
    "CONTENT_ID_BASIS",
    "CONTENT_ID_PREFIX",
    "CURRENT_TRAINED_PUBLICATION_FILENAME",
    "PUBLICATION_SCHEMA",
    "canonical_sha256",
    "content_addressed_suite_path",
    "content_addressed_suite_id",
    "current_trained_suite_publication_path",
    "file_sha256",
    "is_content_addressed_suite",
    "load_frozen_suite_from_artifact",
    "publish_current_trained_suite",
    "verify_content_addressed_suite_manifest",
    "verify_current_trained_suite_publication",
]
