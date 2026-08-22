"""Pin the observation contract the Java port is certified against.

**The hole this fills.** `case/manifest.json` records an
`observation_contract_sha256`, but its `checkpoint_metadata` carries only three
keys -- `action_size`, `observation_size`, `observation_contract_sha256`. The
arsenal hash is computed as
`arsenal_policy_contract_sha256(world_geometry_config, entity_count=...)`, and
neither parameter is recorded. So the pin **cannot be recomputed**, which means
it cannot be checked, which means nothing noticed when the contract moved
7818 -> 8259 -> 8271.

Equal width is not equal meaning. A checkpoint trained against one column layout
loads happily against another whenever the widths agree, because `Policy.load`
derives shapes from file lengths by design. That is the only *silent* failure
mode left in the port: everything else fails loudly.

Two directions of drift, two checks:

  * **Gym drift** -- the Gym's contract changes and our fixtures/weights do not.
    Caught here by `--check`, which recomputes from the live Gym and diffs.
  * **Java drift** -- Java's implemented layout stops matching the pin. Caught
    by `ContractTest`, which needs no Python and runs in the normal gate.

Write the pin:

    cd ~/hydl
    # from the repository root (use ":" instead of ";" on POSIX):
    PYTHONPATH="HytaleRL/hytalegym;." \\
        python -u experimental/jvm-agent/tools/export/export_contract.py

Verify it still holds (this is the one to run after touching the Gym):

    ... python -u experimental/jvm-agent/tools/export/export_contract.py --check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
OUT = JVM_AGENT_ROOT / "case"


def _contract() -> dict:
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        ARSENAL_POLICY_SCHEMA,
        ARSENAL_POLICY_VERSION,
        arsenal_policy_contract_sha256,

    )
    from hytalegym.jax.combat import ARSENAL_POLICY_ACTION_HEAD_SIZES

    # Record the *parameters* alongside the hash, which is exactly what
    # checkpoint_metadata omits. Without them the digest is an opaque string
    # nobody can reproduce, and an unverifiable pin is not a pin.
    #
    # The width deliberately does not come from the contract manifest: that
    # structure has changed shape before, and a nested lookup into it is one
    # more thing that can silently move. It is taken from the fixture Java is
    # certified against instead, below.
    return {
        "schema": ARSENAL_POLICY_SCHEMA,
        "version": ARSENAL_POLICY_VERSION,
        "arsenal_policy_contract_sha256": arsenal_policy_contract_sha256(),
        "action_head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        "action_size": int(sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)),
        "entity_count_default": True,
        "world_geometry_config_default": True,
    }


def main() -> int:
    check = len(sys.argv) > 1 and sys.argv[1] == "--check"
    contract = _contract()

    # The manifest shape has moved before, so do not trust a nested lookup to
    # supply the width. Take it from the fixture Java is actually certified
    # against, which is the number that matters.
    observation_candidates = (
        OUT / "observation.bin",
        OUT / "profile" / "expected_observation.bin",
    )
    observation_widths = {
        path.stat().st_size // 4
        for path in observation_candidates
        if path.is_file()
    }
    if len(observation_widths) > 1:
        raise SystemExit(
            "root and production-profile observation widths disagree: "
            f"{sorted(observation_widths)}"
        )
    if observation_widths:
        contract["observation_size"] = observation_widths.pop()

    pin_path = OUT / "contract.json"

    if check:
        if not pin_path.is_file():
            raise SystemExit(
                "no case/contract.json to check against -- run without --check "
                "first to write the pin"
            )
        pinned = json.loads(pin_path.read_text(encoding="utf-8"))
        drift = [
            key
            for key in sorted(set(pinned) | set(contract))
            if pinned.get(key) != contract.get(key)
        ]
        if drift:
            print("CONTRACT DRIFT -- the live Gym no longer matches the pin:")
            for key in drift:
                print(f"  {key}:")
                print(f"    pinned : {pinned.get(key)}")
                print(f"    live   : {contract.get(key)}")
            print()
            print("The Java port is certified against the pinned contract. Either")
            print("re-export every fixture against the live Gym and re-pin, or")
            print("keep the Gym where it was. Do not just overwrite the pin.")
            return 1
        print(f"contract pin holds: {pinned['arsenal_policy_contract_sha256']}")
        print(f"observation {pinned['observation_size']}, "
              f"action {pinned['action_size']}, "
              f"heads {pinned['action_head_sizes']}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    pin_path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"pinned contract to {pin_path}")
    for key, value in sorted(contract.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
