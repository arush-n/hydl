#!/usr/bin/env python3
"""Capture one Region's block-semantic sidecar (block identity per cell) from a running 0.7.2 bridge.

Task #130 item 5: the 0.7.2 Region library has no block-semantics sidecar, so block candidates are empty on 0.7.2
Regions. The stored Region NPZs keep a physical palette only, so the sidecar must be captured live
(README section "Block-semantic sidecar"). This is the live step, one seed per fresh bridge launch:

1. reset the native env on the seed (the same env as examples/capture_native_region.py);
2. capture_projected_native_region_block_semantics: one live physical pass, checked against the stored Region NPZ
   (only non-damaging fluid drift may differ), then 250 block-identity sections;
3. save the sidecar, reload it, and check that the reloaded semantic digest equals the captured one.

The output goes to --output (a stage folder, never the shared library); nothing is promoted or bound. Run it as the
launcher's --then command:
    launch_bridge_072.py ... --bridge-port 5556 --game-port 25574 --then \
      "<python> capture_block_semantics.py --port 5556 --seed S --region-npz <library>/artifacts/seed_S.npz \
       --native-evidence-jar <deployed jar> --output <stage>/sidecars/seed_S.block-semantics.npz"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.worldgen.native_lease import acquire_native_evidence_lease
from hytalegym.worldgen.region import capture_projected_native_region_block_semantics, current_native_evidence_jar_sha256
from hytalegym.worldgen.region.block_semantics import NativeRegionBlockSemanticSnapshot
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", default="Kweebec_Razorleaf")
    parser.add_argument("--region-npz", type=Path, required=True, help="the stored 0.7.2 Region NPZ for this seed")
    parser.add_argument("--native-evidence-jar", type=Path, required=True, help="the deployed bridge jar")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output exists, refusing to overwrite: {args.output}")
    evidence = hashlib.sha256(args.native_evidence_jar.read_bytes()).hexdigest().upper()
    if evidence != current_native_evidence_jar_sha256():
        parser.error(f"bridge jar {evidence[:8]} is not the current pin {current_native_evidence_jar_sha256()[:8]}")
    physical = NativeRegionSnapshot.load(args.region_npz)
    with acquire_native_evidence_lease(host=args.host, port=args.port, purpose="capture_block_semantics"):
        env = HytaleEnv(task="kill_trork", backend="native", world="hytale", host=args.host, port=args.port,
                        ticks_per_step=1, max_episode_steps=100, npc_role=args.role, combat_target_active=False,
                        fidelity_fixture="static_region")
        try:
            env.reset(seed=args.seed)
            capture = capture_projected_native_region_block_semantics(
                env, physical, evidence_bridge_sha256=evidence, progress=_progress)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    captured = capture.snapshot.semantic_sha256()
    destination = capture.snapshot.save(args.output)
    reloaded = NativeRegionBlockSemanticSnapshot.load(destination).semantic_sha256()
    receipt = {
        "seed": args.seed,
        "output": str(destination),
        "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "semantic_sha256": captured,
        "reloaded_semantic_sha256": reloaded,
        "reload_matches": reloaded == captured,
        "region_npz": str(args.region_npz),
        "region_semantic_sha256": physical.semantic_digest(),
        "evidence_bridge_sha256": evidence,
        "physical_comparison": repr(capture.physical_comparison)[:2000],
    }
    destination.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ("seed", "semantic_sha256", "reload_matches", "file_sha256")}))
    return 0 if receipt["reload_matches"] else 1


def _progress(done: int, total: int, _task: object) -> None:
    if done == 1 or done == total or done % 50 == 0:
        print(f"captured {done}/{total} sections", flush=True)


if __name__ == "__main__":
    sys.exit(main())
