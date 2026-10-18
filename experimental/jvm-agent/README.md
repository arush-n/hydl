# Native policy agent

`experimental/jvm-agent` contains the Java policy-agent integration used by the
native transfer and deployment tooling. It translates a deployed policy's
structured actions into server-side control and exposes native perception,
combat, movement, world, and lifecycle seams.

It depends on the HytaleRL bridge protocol and the ADK deployment contract. It
is not the JAX simulator and it is not a second PPO implementation; its role is
to execute or certify a policy at the native boundary.

## Entry points

- [`mod/manifest.json`](mod/manifest.json) identifies the native plugin bundle.
- [`src/com/hytalerlbridge/policy/PolicyAgentPlugin.java`](src/com/hytalerlbridge/policy/PolicyAgentPlugin.java)
  is the plugin entry point.
- `src/com/hytalerlbridge/policy/perception/` projects native state into policy
  inputs.
- `src/com/hytalerlbridge/policy/action/` and the combat/world packages apply
  selected actions.
- `tools/export/` contains export and contract utilities used by the native
  bundle path.

For the stable Python-side deployment surface, continue to
[`adk/deploy/README.md`](../../adk/deploy/README.md).

## Locomotion contract

The exported `world_move_direction` field keeps its historical wire name, but
the values are body-relative: `1..8` are forward, forward-right, right,
back-right, back, back-left, left, and forward-left. Java must rotate that
local vector by the commanded body yaw using the same x/z convention as the
JAX runtime. A zero-yaw smoke test is insufficient because it cannot distinguish
the old absolute-compass interpretation.

`SteeringActionSink` uses
[`PlayerLocomotionProjection.java`](src/com/hytalerlbridge/policy/PlayerLocomotionProjection.java)
to apply the authored forward/back/strafe gait multipliers. It writes the
resulting vector directly to `Steering`; it must not call
`setTranslationRelativeSpeed(1.0)`, which normalizes the vector and removes the
gait magnitude. The public NPC API exposes no exact player gait-transition or
stamina control, so the standalone path is an explicit player-policy-to-NPC
surrogate. See
[`SIM-TO-SERVER-ACTUATOR-DIAGNOSTIC-2026-09-05.md`](../../HytaleRL/docs/SIM-TO-SERVER-ACTUATOR-DIAGNOSTIC-2026-09-05.md)
and [`POLICY-PORT-STATUS.md`](../../HytaleRL/POLICY-PORT-STATUS.md) for the
current evidence and remaining transfer gates.

## Verification

Run the JVM-agent gate from this directory with the matching Hytale server JAR
and a JDK (the current 0.5.7 gate was compiled with JDK 25):

```powershell
python run_tests.py --server-jar "<path-to-server-jar>" --jdk-bin "<path-to-jdk-bin>"
```

The runner compiles into a temporary directory and does not launch the game.
Fixtures under `case/` are generated evidence, not hand-maintained source; use
[`tools/export/DEV.md`](tools/export/DEV.md) to regenerate them when contracts
change. A passing gate certifies the Java implementation against the included
JAX-derived fixtures and parity corpus. It does not by itself certify policy
behavior on a live Hytale server; see the sim-to-server plan for that boundary
and its remaining evidence requirements.

The current local result and exact scope are recorded in
[`verification/JAX-JAVA-CERTIFICATION-2026-09-26.md`](verification/JAX-JAVA-CERTIFICATION-2026-09-26.md).

For an interactive 0.5.x validation, install an empty `follow-player` marker
beside the live bundle. The policy-controlled NPC then acquires the nearest
connected player in the same world; it does not fall back to a marked NPC.
The NPC remains the controlled actor, while the player supplies live target
spatial/health evidence. Player combat lifecycle fields are intentionally
unavailable in this mode because the native lifecycle facade is NPC-specific.
Use `spawn.txt` (`Role x y z`) for an automatic NPC, or `/hydl spawn` after
joining. See [`DEV.md`](DEV.md) for the full validation layout and expected
status/log evidence. Explicit live-policy spawns equip the profile's trained
agent item before claim; an unarmed spawn is intentionally rejected by the
native lifecycle contract. For local validation, `/op self` works in-game;
the console form is `op add <username>` while online or `op add <player-uuid>`
for a persisted offline grant.
