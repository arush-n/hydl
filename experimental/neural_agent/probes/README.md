# probes — measured facts for composing `HierarchicalAgentCore` on AgentKit

## Files in this folder (map, 2026-10-18)

| File | Lines | What it measured |
|---|---:|---|
| `probe_events.py` | 359 | `DecisionEvent`s, episode ends, reward and health through the ADK hierarchy (the tables below). |
| `probe_jev_latency.py` | 95 | Decision latency of J0 and J1 inside a compiled call (`out/jev_latency.json`). |
| `probe_timing.py` | 83 | Compile versus run time of the pursuit-candidate hierarchy rollout on CPU (`out/timing.json`). |

Not a build target, no gate, not imported by the package; `out/` is gitignored. Later latency measurements live
where they are gated: `../jev/system_one/probe_latency.py` (G2) and the composition driver's JEV columns. The
rest of this document is the measured record of 2026-10-16.

Measured 2026-10-16 on CPU (`jax.default_backend() == "cpu"`), one seed
(`jax.random.PRNGKey(0)`). Every number below comes from `out/events.json` or
`out/timing.json` as written by the probes in this folder. Nothing here is
inferred from a README.

```sh
export PYTHONPATH="/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym" JAX_PLATFORMS=cpu
.venv/bin/python -m experimental.neural_agent.probes.probe_events   # 188.5 s wall -> out/events.json
.venv/bin/python -m experimental.neural_agent.probes.probe_timing   # 103.9 s wall -> out/timing.json
```

## Build names

`AgentKit.build(name)` resolves `name` via `adk/core/registry` after importing
`adk/agents/<family>/spec.py` (`adk/api.py` `_resolve_spec`, `_import_agent_families`).
Only `adk/agents/combat/spec.py` exists, so:

| call | result |
|---|---|
| `AgentKit().list_agents()` | `['combat/baseline']` (`AgentSpec(name="combat/baseline", loadout="iron_sword")`, scene defaults to `combat/fail_closed`) |
| `AgentKit().list_scenes()` | `['combat/fail_closed', 'combat/open_flat_control']` |

"Armed": no AgentKit name declares an armed opponent. Armed rungs exist only as
arena league `Difficulty(opponent_armed=True)` entries (e.g. `combat/duel`
`target_armed`, `combat/survive` `armed`, `combat/defend` `armed_dagger` /
`armed_battleaxe`, `movement/track` `mobile_armed`, `world/build` `contested`,
`world/generated` `armed`). Those are not AgentKit build names and were **not
probed**. The second scene probed here is the other built-in scene, through a
probe-registered spec: `AgentSpec(name="probe/open_flat_control", scene="combat/open_flat_control")`.
Measured below: the scripted target damages a neutral agent in **both** scenes.

## Composition used

`handle = AgentKit().build(name, batch=8).with_episode_limit(128)`; `built = handle.built`;
`HierarchicalAgentCore(belief, manager, actor, manager_cadence_seconds=1.0)`;
`compile_hierarchical_collector(built, core, record, steps, initial_state=...)` with the
default `jax_hierarchical_context` and `actor_policy_input`.

- belief: pass-through — `BeliefUpdate(state, belief=actor_input.observation, prior/posterior/uncertainty=zeros[8,1], diagnostics={})`.
- manager: constant — returns `ManagedGoal(payload=<payload it was given>, control)`, `terminate=zeros bool[8]`, `diagnostics={}`.
- actor (a) neutral: `neutral_arsenal_policy_action_factors` (yaw/body_yaw/pitch = `size//2`, rest 0), per head replaced by the first legal bin if the mask forbids it.
- actor (b) pursuit: `agent_skills/policies/pursuit_contact/policy.py` `load(batch)` → `policy(carry, actor_input, key) -> (carry, factors)`; the loader applies `pursuit_learner_action_scope(allow_jump=False).restrict` to the mask.
- initial state: `initialize_hierarchical_state(HierarchicalCarry(..., previous_action=neutral int32[8,12], goal=ManagedGoal(zeros int32[8], GoalControl(zeros f32[8], inf f32[8], zeros bool[8,14]))), manager_diagnostics={})`.
- record: events from `transition.next_decision_context` (the context produced by that env step, before autoreset); health = `legal_observation.base.self_f32[:, SELF_FLOAT_FEATURES.index("health_fraction")]` (index 3).

## Events — `probe_events.py` (batch 8 × 128 steps = 1024 lane-steps per run)

Counts are lane-steps. `known` = `events.known`; `fired` = `known & values`.

| DecisionEvent | known (all 4 runs) | fired: baseline / neutral | baseline / pursuit | open_flat / neutral | open_flat / pursuit |
|---|---:|---:|---:|---:|---:|
| ATTACK_TELEGRAPH_BEGAN | 1024 | 0 | 0 | 19 | 17 |
| PROJECTILE_THREAT_ENTERED | 0 | 0 | 0 | 0 | 0 |
| DAMAGE_RECEIVED | 1024 | 17 | 14 | 17 | 15 |
| TARGET_DIED | 0 | 0 | 0 | 0 | 0 |
| LINE_OF_SIGHT_CHANGED | 1024 | 0 | 0 | 0 | 0 |
| PATH_INVALID | 0 | 0 | 0 | 0 | 0 |
| ABILITY_COMPLETED | 0 | 0 | 0 | 0 | 0 |
| ABILITY_INTERRUPTED | 0 | 0 | 0 | 0 | 0 |
| NPC_STUCK | 0 | 0 | 0 | 0 | 0 |
| GOAL_IMPOSSIBLE | 0 | 0 | 0 | 0 | 0 |
| ACTION_COMPLETED | 0 | 0 | 0 | 0 | 0 |
| ACTION_CANCELLED | 0 | 0 | 0 | 0 | 0 |
| ACTION_FAILED | 0 | 0 | 0 | 0 | 0 |
| ACTION_REJECTED | 0 | 0 | 0 | 0 | 0 |

| run | episodes ended | terminated / truncated | done step (all lanes) | mean reward / lane-step | nonzero-reward steps | min health fraction | manager_due steps | compile s | run s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline / neutral | 8 | 0 / 8 | 127 | -0.3818 | 17 | 0.343 | 50 | 29.9 | 0.152 |
| baseline / pursuit | 8 | 0 / 8 | 127 | -0.3145 | 14 | 0.562 | 46 | 68.3 | 0.195 |
| open_flat / neutral | 8 | 0 / 8 | 127 | -0.3818 | 17 | 0.343 | 57 | 32.1 | 0.156 |
| open_flat / pursuit | 8 | 0 / 8 | 127 | -0.3369 | 15 | 0.562 | 56 | 20.9 | 0.167 |

Measured observations (from the tables above / `events.json`):

- DAMAGE_RECEIVED fires in every run; its count equals the count of steps where
  `health_after < health_before` and equals the nonzero-reward step count in all four runs.
- Only three events are ever `known`: ATTACK_TELEGRAPH_BEGAN, DAMAGE_RECEIVED,
  LINE_OF_SIGHT_CHANGED (known on all 1024 lane-steps). The six lifecycle-derived events
  had `known == 0` on every step; the other five were never known.
- ATTACK_TELEGRAPH_BEGAN fired only on `combat/open_flat_control` (0 on `combat/fail_closed`).
- With `with_episode_limit(128)` every lane ended exactly once, at step index 127, as
  `truncated` (split available); no lane `terminated`.
- `timing.delta_seconds` was known on all lane-steps; observed values were {0.033333, 0.045}.
- The masked neutral never needed its fallback: the neutral bin was legal on every head, every step.
- Pursuit `action_factors` passed the collector's `int32 [B,12]` check without a cast. It moved only
  `locomotion_gait_compass`, `yaw_delta_bins`, `body_yaw_delta_bins`, `pitch_delta_bins`; the other
  8 heads stayed at 0 on all 1024 lane-steps in both scenes.
- The neutral runs produced identical reward, damage and health numbers on both scenes (cause not measured).

## Timing — `probe_timing.py` (pursuit actor, `combat/baseline`, 64 steps)

| batch | build s | lower s | compile s | first run s | best of 3 repeat runs s | lane-steps/s (best) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5.39 | 10.95 | 38.58 | 0.111 | 0.087 | 5864 |
| 32 | 8.52 | 12.41 | 22.39 | 0.207 | 0.164 | 12455 |

Compile dominates run by more than two orders of magnitude at both batches. Compile time
varied between runs of the same composition (68.3 s for 128 steps in `probe_events`, 38.6 s for
64 steps here, both batch 8); the cause was not isolated.

## Not measured

Arena league armed rungs; more than one seed; episodes longer than 128 ticks; reward sign split;
why compile time varies; why the two scenes give identical neutral outcomes.
