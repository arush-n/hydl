"""Which installed 0.5.x files are provably identical to the 0.5.7 pins.

The simulator is calibrated to 0.5.7 (``Assets.zip`` SHA-256 ``1B8802C2...``);
this machine has 0.5.10. Every 0.5.7 hash the repository already pins is
compared against the installed archive:

* raw-byte SHA-256 per asset file in the fidelity source audits
  (``ASSET_SHA256`` / ``TRACE_ASSET_SHA256`` dicts, parsed with ``ast``, not
  imported);
* the combat asset catalog's per-item semantic hashes (resolved item plus its
  whole interaction graph) and auxiliary-file hashes recorded on 0.5.7 in
  ``HytaleRL/artifacts/combat_asset_audit_rev13.json``, recomputed on the
  installed archive with ``build_combat_asset_catalog``;
* ``OUTLANDER_ARROW_ASSET_SHA256`` (``jax/combat/effects/schema/contract.py``).

Unlike the extractors this module imports ``hytalegym``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

SCHEMA = "hytalerl_gen5_equivalence_v1"
PINNED_057_ASSETS_SHA256 = "1B8802C284C228AE4549DAC037716C175BC6B94B0FC9AAC2B7039AC6BD2FFD5D"
_AUDIT_TABLES = ("ASSET_SHA256", "TRACE_ASSET_SHA256")
_OUTLANDER_PATHS = (
    "Server/Projectiles/NPCs/Intelligent/Outlander_Hunter/Outlander_Hunter_Arrow.json",
    "Server/Projectiles/NPCs/Intelligent/Outlander_Stalker/Outlander_Stalker_Arrow.json",
)


def _literal_pins(source: Path) -> dict[str, str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    pins: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if not names & set(_AUDIT_TABLES):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            try:
                path, sha = ast.literal_eval(key), ast.literal_eval(value)
            except ValueError:
                continue
            if isinstance(path, str) and path.startswith("Server/") and isinstance(sha, str):
                pins[path] = sha.upper()
    return pins


def collect(assets_zip: str | Path, hytalerl_root: str | Path, *, assets_sha256: str) -> dict[str, Any]:
    root = Path(hytalerl_root)
    fidelity = root / "hytalegym" / "tests" / "fidelity"
    file_rows = []
    with ZipFile(assets_zip) as archive:
        names = set(archive.namelist())

        def installed(path: str) -> str | None:
            return hashlib.sha256(archive.read(path)).hexdigest().upper() if path in names else None

        for module in sorted(fidelity.glob("surrogate_*_source_audit.py")):
            for path, pinned in sorted(_literal_pins(module).items()):
                actual = installed(path)
                file_rows.append(
                    {
                        "path": path,
                        "pinned_by": str(module.relative_to(root)),
                        "pinned_057_sha256": pinned,
                        "installed_sha256": actual,
                        "identical": actual == pinned,
                    }
                )
        from hytalegym.jax.combat.effects.schema.contract import OUTLANDER_ARROW_ASSET_SHA256

        for path in _OUTLANDER_PATHS:
            actual = installed(path)
            file_rows.append(
                {
                    "path": path,
                    "pinned_by": "hytalegym/hytalegym/jax/combat/effects/schema/contract.py",
                    "pinned_057_sha256": OUTLANDER_ARROW_ASSET_SHA256.upper(),
                    "installed_sha256": actual,
                    "identical": actual == OUTLANDER_ARROW_ASSET_SHA256.upper(),
                }
            )

    from hytalegym.combat.assets.catalog import build_combat_asset_catalog

    catalog = build_combat_asset_catalog(assets_zip, expected_sha256=assets_sha256, verify_sha256=False)
    reference = json.loads((root / "artifacts" / "combat_asset_audit_rev13.json").read_text(encoding="utf-8"))
    pinned = {row["asset_id"]: row["source_semantic_sha256"] for row in reference["records"]}
    current = {record.asset_id: record.source_semantic_sha256 for record in catalog.records}
    weapon_rows = [
        {
            "asset_id": asset_id,
            "pinned_057_semantic_sha256": pinned[asset_id],
            "installed_semantic_sha256": current.get(asset_id),
            "identical": current.get(asset_id) == pinned[asset_id],
        }
        for asset_id in sorted(pinned)
    ]
    auxiliary_pinned = reference["auxiliary_source_semantic_sha256"]
    auxiliary_current = dict(catalog.auxiliary_source_semantic_sha256)
    auxiliary_rows = [
        {
            "path": path,
            "pinned_057_semantic_sha256": auxiliary_pinned[path],
            "installed_semantic_sha256": auxiliary_current.get(path),
            "identical": auxiliary_current.get(path) == auxiliary_pinned[path],
        }
        for path in sorted(auxiliary_pinned)
    ]
    return {
        "schema": SCHEMA,
        "pinned_057_assets_sha256": PINNED_057_ASSETS_SHA256,
        "installed_assets_sha256": assets_sha256.upper(),
        "reference_artifact": "HytaleRL/artifacts/combat_asset_audit_rev13.json",
        "files": file_rows,
        "weapon_graphs": weapon_rows,
        "weapon_graphs_not_in_057": sorted(set(current) - set(pinned)),
        "auxiliary_files": auxiliary_rows,
        "summary": {
            "files_identical": sum(row["identical"] for row in file_rows),
            "files_pinned": len(file_rows),
            "weapon_graphs_identical": sum(row["identical"] for row in weapon_rows),
            "weapon_graphs_pinned": len(weapon_rows),
            "auxiliary_identical": sum(row["identical"] for row in auxiliary_rows),
            "auxiliary_pinned": len(auxiliary_rows),
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--assets-sha256", required=True)
    parser.add_argument("--hytalerl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.assets, args.hytalerl_root, assets_sha256=args.assets_sha256)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
