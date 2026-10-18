# Training and evaluation on Server 0.7.2

How to run the repo's training and evaluation flows against a 0.7.2 server, what
was checked live (2026-10-17), and the gameplay changes that matter for
training. The bridge port itself is described in [`PORT-0.7.2.md`](PORT-0.7.2.md).

## Switching a run from 0.5 to 0.7.2

1. Start the server with this folder's bridge:
   `python tools/launch_bridge_072.py --run-dir <new dir> --bridge-port <p> --game-port <g>`
   (or pass `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar` and the
   0.7.2 `HytaleServer.jar` / `Assets.zip` to whatever launcher you use).
2. Set `HYTALERL_SERVER_VERSION=0.7.2` for Python processes that check bridge
   identity before they connect (region capture, `jax_arsenal_train` bridge
   stamp). Checks made on a live connection read the version the server
   reports, so they need nothing.
3. For the policy agent, build and deploy
   [`experimental/jvm-agent-0.7.2`](../../experimental/jvm-agent-0.7.2) instead of
   `experimental/jvm-agent` (see its `PORT-0.7.2.md`).

Version-specific values live in `hytalegym/hytalegym/server_versions/`, one
module per release line (`v0_5.py`, `v0_7.py`). The 0.5 profile overrides
nothing, so 0.5 runs the original code paths. A new release line is one new
module plus one registry entry. The gates that read it:

| Gate | 0.5 | 0.7 |
|---|---|---|
| live bridge hash (`region/stability.py`, used by ~30 callers) | `CURRENT_NATIVE_EVIDENCE_JAR_SHA256` | the port jar's SHA-256 |
| region chunk contract release line (`region/contract.py`) | 0.5.x | 0.5.x or 0.7.x |
| native crafting catalog size (`worldgen/native_crafting.py`) | 1947 | 2047, keyed on the catalog's own `server_version` |
| built bridge path (`contracts/publication.py`) | `hytale-plugin/.../HytaleRLBridge-0.1.0.jar` | `hytale-plugin-0.7.2/.../HytaleRLBridge072-0.1.0.jar` |
| native eval checkpoint label (`examples/jax_policy_evaluate.py`) | exact server version | exact, or a `0.5.7` simulator checkpoint (cross-version run) |

The 0.7 bridge hash is pinned to the current port build; rebuilding the jar
with changes means updating `v0_7.py`, as the 0.5 pin is updated per deploy.

## Flow status, same invocations on 0.5.10 and 0.7.2

| Flow | 0.5.10 + original bridge | 0.7.2 + port |
|---|---|---|
| `sb3_ppo_train.py --backend headless` (2,048 steps) | runs; reward 77.4 | runs; reward 77.4 |
| `sb3_evaluate.py --backend headless` (own and crossed models) | runs | runs |
| `jax_ppo_train.py`, `jax_arsenal_train.py --open-flat-control` (no server) | runs | n/a (simulator) |
| `jax_arsenal_train` bridge identity stamp | not verified: built `DBFC877D…` ≠ pin `7700D112…` | verified with `HYTALERL_SERVER_VERSION=0.7.2` |
| `jax_policy_evaluate.py --backend headless` | refused: server `0.5.10` ≠ checkpoint `0.5.7` (exact rule) | runs (cross-version) |
| `jax_arsenal_native_transfer.py` (fresh tiny checkpoint, `iron_sword`, 32 steps) | refused: bridge `DBFC877D…` ≠ pin `7700D112…` | `execution_valid`, identity stable, full 8272/124 surface |
| `capture_native_region.py --native-evidence-jar <jar>` | refused: jar ≠ pin `7700D112…` | 500/500 sections, labelled `686BC4BC…` |
| native crafting catalog parse | (not run) | 2047 recipes parsed |
| `adk/tools/native_motion_differential.py` with `--expected-server-version` | runs (position error 1.1e-5) | runs (1.1e-5) |
| policy agent unit suite (`run_tests.py`) | 46/46 | 46/46 |
| policy agent live boot | (not run) | loads and enables, no errors |

Native transfer needs a checkpoint trained against the current tree: its
observation and action contract hashes include the staged region library's
identity and change whenever contract sources change, so a checkpoint from an
older tree (or a tree copy without `HytaleRL/artifacts`) is refused as not
exact-current.

Failing on both servers for reasons unrelated to the version:

- `random_agent.py`, `train_base_builder.py`: random actions start two
  standard-input interaction roots in one action, which the bridge refuses
  (`AgentAction.java:243`, the same in both bridges).
- `native_motion_differential` defaults to `--expected-server-version 0.5.9`,
  so it refuses 0.5.10 too without the flag.
- Policy agent with a policy loaded: a deployable bundle needs a live profile
  exported on a CUDA GPU (`experimental/jvm-agent/tools/export/export_profile.py`,
  checked at `adk/deploy/bundle.py:754`). Without one the agent stays on frozen
  perception and refuses to arm ("requires the bridge combat facade"), the same
  on both servers, so the 0.7.2 copy's live paths (chunk pinning, block
  candidates, combat facade, line of sight, fixture spawn) are still untested.
  This Mac has no CUDA. `isolated.py` also does not treat "PolicyAgent failed to
  arm" as a failure, so such runs wait out the full timeout.
- `adk/deploy/bundle.py` and `pursuit_transfer_gate.py` need a trained
  checkpoint with the policy-agent surface; none was on disk, so deploy was
  not run on either server.
- Region capture records the claimed evidence-jar hash without comparing it to
  the live bridge, so a wrong claim is accepted on both servers.

Not version-keyed on purpose: certifications that compare the JAX simulator
with server source or assets (`worldgen/source_pins.py` and its 11 audits,
the 0.5.7 `Assets.zip` hash in `rulesets/asset_surface.py`, the packaged
`hytale_0_5_7` rulesets, the JAX crafting table capped at 1947 rows in
`jax/crafting/packing.py`). They fail on 0.7.2 because the simulator still
models 0.5.7, and passing them needs a 0.7.2 calibration of the simulator.
Checkpoints from `jax_*_train` keep `hytale_version` 0.5.7 for the same reason.

## Gameplay changes that matter for training

Measured live on both servers unless marked otherwise. "Code" means read in the
decompiled servers, not measured.

1. **Melee hits interrupt and push less (weapon content).** 31 player weapon
   attack files (daggers 9, maces 7, swords 6, battleaxes 5, three abilities)
   now inherit `DamageEntityParent_Hit_Interrupt`, which adds stun, attack
   cancel and a 0.5 s flinch; none did in 0.5.10. The sword swing's knockback
   also changed (Force 6.0 → 4.3, direction (0, 1, −1) → (0, 0.2, −1)).
   Measured: the same `Weapon_Sword_Iron` hit (10 damage on both) moves a
   Trork 0.19 m on 0.7.2 versus 0.44 m on 0.5.10 (18 and 16 clean hits), lift
   0.014 m versus 0.11 m. Swapping only the sword file reproduces each side
   exactly (0.7.2 with the 0.5.10 file: 0.435 m over 7 hits; 0.5.10 with the
   0.7.2 values: 0.195 m over 19 hits), so engine physics and the bridge are
   not the cause. Agents get less spacing from their hits; whether a hit now
   cancels an attacking Trork's swing was not measured. The 0.5.7 simulator
   models neither change.
2. **The agent's own horizontal knockback is cancelled on both versions.** The
   bridge clears the agent's motion every control tick, so 32 Trork hits on a
   standing agent produced zero horizontal movement on both servers; the vertical
   kick (1.5 m/s) is the same. A code change that would raise set-type NPC
   knockback about 23-fold in 0.7.2 (`Role.processSetVelocityInstruction`) does
   not reach the agent through the bridge; it may reach other NPCs (code, not
   measured).
3. **Mana exists.** 0.7.2 `Mana.json` gives 100/100 with +1 per 0.3 s and no
   gating (0.5: 0/0), so staffs, wands and spellbooks with mana costs become
   usable. The simulator's mana is 0.
4. **Ranged damage is cut 3–4×.** Shortbow Iron charge levels 6/9/12/16 →
   1/2/4/7, Mithril 11/17/23/31 → 2/4/7/13; crossbow combos and big arrows
   drop (Ancient Steel combo 42 → 14). Agent-side only: Trork and Kweebec
   attacks use no bows.
5. **Daggers hit harder and durability changed.** Dagger swings about +50%
   (Iron 4 → 6, Mithril 8 → 12). 33 items change max durability and 44 change
   loss per hit (`Weapon_Sword_Iron` 120 → 250). Unchanged: `Weapon_Spear_Leaf`,
   `Weapon_Sword_Iron` damage, Trork battleaxe and mace damage.
6. **NPC melee is unchanged.** Trork and Kweebec role files, motion parameters
   (walk speed, jump, gravity, knockback scale), attack chains, the 5-damage
   NPC melee default, Health, Stamina and Oxygen stats, and world speed (30
   ticks per second) are identical. Measured: 5 damage per Trork hit on both.
7. **Guarding.** Shield guard gains elemental damage and knockback modifiers;
   the stop-guard stamina regen delay changes from a set value to a minimum
   (assets; engine semantics not read).
8. **Walking down slopes and ledges** (code): 0.7.2's walk controller forces a
   minimum descent rate while descending, so NPCs may drop faster off ledges.
   Not measured; every fixture here is flat.
9. **Rules already listed in `PORT-0.7.2.md`:** line of sight passes
   non-transparent blocks unless a detail hitbox is hit; block health resets on
   any block write; crafting has 2047 recipes and tier-time changes; the camera
   ray has no y limit.
10. **Content and timing.** Block and item runtime ids are renumbered (Rock_Stone
    1351 → 1530), NPC role asset paths moved from `Aggressive/` and `Neutral/` to
    `Faction/`, 46 of 250 region sections differ, spider webs slow more (speed
    0.65 → 0.45), the first step after reset takes 34–68 ticks (0.5.10: 30), and
    world-thread bridge tasks are slower (median 57 ms vs 45 ms).
