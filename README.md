# HYDL

HYDL is a high-throughput software twin of Hytale for training autonomous game
agents.

It reimplements Hytale's structured game state and rules in JAX so many worlds
can run in parallel inside compiled array code. A Java bridge then runs the
policy against the authoritative Hytale server for validation.

The current policy uses numerical state, action-allowed flags, and target
features. It does not use screenshots or video.

## What is proven

- The JAX combat, locomotion, and world state machine runs in batched,
  JIT-compatible array code.
- The simulator has separate player and NPC locomotion models, including the
  native NPC walk behavior used by the current motion lane.
- Disabled targets are handled safely instead of producing invalid geometry
  transitions.
- Bounded geometry and locomotion checks pass against an installed Hytale 0.5.9
  server for the fixtures exercised so far.
- A native 0.5.9 policy trace reproduces **296/296** action decisions when
  replayed through the JAX policy.
- The Java policy runtime executes the policy in Hytale and drives locomotion
  under explicit test accommodations.
- Validation fails closed when the policy, simulator version, or task setup does
  not match.

This proves a working, narrow policy and locomotion lane. It does not prove
that the complete JAX environment and Hytale server behave identically.

## Gaps to close

1. Start JAX and Hytale from the same state and compare what the agent sees on
   every tick.
2. Record what the server actually applied, not only what the policy requested.
3. Compare movement, damage, death, rewards, inventory, block interactions, and
   other world effects.
4. Run a fair pursuit evaluation with the same start distance, target behavior,
   episode length, game version, and held-out trials.
5. Validate a native scripted fleeing target without special target information.
6. Add vision so an agent can learn from screenshots or video.
7. Measure the speed advantage on one matched task. JAX is the high-throughput
   training path; Hytale is limited by its real-time server loop, the cost of
   running many separate worlds, and CPU/bridge/device transfers.

The policy, simulator version, and native task setup must be aligned before a
matched native pursuit result can be certified.

## Architecture

```text
Hytale authored data
        |
        v
HytaleGym: batched JAX state machine
        |
        v
Arena: tasks, rewards, policies, training, evaluation
        |
        v
PPO / self-play / rollouts

Hytale server <-> Java bridge <-> policy runtime <-> validation evidence
```

### Main directories

| Directory | Responsibility |
|---|---|
| `HytaleRL/hytalegym/` | JAX environment: combat, motion, collision, observations, actions, geometry, and world stepping. |
| `arena/` | Tasks, rewards, world binding, rollout collection, PPO, and evaluation. |
| `adk/` | Deployment bundles, runtime setup, provenance, and transfer validation. |
| `experimental/jvm-agent/` | Java policy execution, native state projection, action decoding, and server validation. |
| `console/` | Training-job API and run management. |
| `agents/` | Training profiles and policy artifacts. |
| `worlds/` and `npc/` | World providers and NPC/task behavior surfaces. |

The current end-to-end training lesson is pursuit. Other task interfaces exist,
but they are not all runnable end to end.

## Setup

Requirements:

- Python 3.11 or newer
- JAX, NumPy, Gymnasium, and Optax
- A GPU is recommended for training

From the repository root:

```bash
python -m pip install -e ".[dev]"
export PYTHONPATH="$PWD:$PWD/HytaleRL"
```

PowerShell:

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "$PWD;$PWD\HytaleRL"
```

`hytalegym` is shipped in this repository rather than resolved from a package
index, so the `HytaleRL` path is required.

## Train in JAX

Training is owned by the Console so the task, environment, policy, and evidence
contracts are recorded together.

```bash
python -m console.server --help
```

Use the Console API documented in [`console/docs/API.md`](console/docs/API.md)
to run the `basic` pursuit profile. JAX training does not require a running
Hytale server.

For an environment throughput measurement:

```bash
python tools/bench_env_throughput.py --out results
```

## Test the simulator

```bash
python -m pytest \
  HytaleRL/hytalegym/tests/jax/combat/runtime/test_airborne_player_motion.py \
  HytaleRL/hytalegym/tests/jax/combat/motion/test_target_vertical.py \
  HytaleRL/hytalegym/tests/jax/test_geometry.py -q
```

```bash
python -m ruff check HytaleRL/hytalegym/hytalegym
```

## Native validation

Native validation requires an installed Hytale server, matching assets, a
compatible JDK, and bridge/plugin build outputs. Those runtime files are not
distributed here.

Use this lane to answer three questions:

- Does the Java policy execute inside the real server?
- Does the server provide the state and action evidence expected by the policy?
- Does behavior survive the move from the fast twin to the authoritative game?

A policy replay is not a full environment-fidelity result. A locomotion result
is not a combat result. Every native result must state its server version, task
setup, accommodations, and evidence type.

## Boundaries

- The current policy interface is structured state, not vision.
- Native evidence is bounded to specific fixtures and accommodations.
- Full sim-to-server certification has not been issued.
- Hytale server installs, assets, captured worlds, private fixtures, and runtime
  jars are not distributed.
- Decompiled Hytale source code is not distributed.

## License

MIT. Hytale's own assets, server files, and decompiled code are not distributed
under this license.
