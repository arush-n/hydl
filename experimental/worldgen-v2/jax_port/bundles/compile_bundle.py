"""Command-line compiler for an analyzed WorldGen V2 capture corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from jax_port.bundles.bundle import compile_capture_bundle
else:
    from .bundle import compile_capture_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile an exact WorldGen V2 capture into a small direct JAX "
            "bundle catalog. Source artifacts are referenced, never copied."
        )
    )
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-id")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="skip hashing all 300 source artifacts; intended only for tests",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    bundle = compile_capture_bundle(
        args.capture_root,
        args.output,
        bundle_id=args.bundle_id,
        verify_artifact_hashes=not args.manifest_only,
    )
    elapsed = time.perf_counter() - started
    result = {
        "status": "passed",
        "bundle_manifest": str(bundle.manifest_path),
        "bundle_semantic_sha256": bundle.semantic_sha256,
        "structure_count": len(bundle.structure_ids),
        "artifact_count": len(bundle.artifacts),
        "compile_seconds": elapsed,
        "source_artifacts_hashed": not args.manifest_only,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
