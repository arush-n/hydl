# skill_agent/teaching/ — what scripted teachers share, their certificates, and the lookahead teacher

A teacher is a controller that reads the full simulator state and labels data for a student policy. A teacher is
used only after its certificate passes (`../docs/TRAINING_PLAN.md` section 2c: T1 success, T2 efficiency, T3
obedience, T4 the live replay, T5 whether a student can follow it, T7 failure sorting). The lookahead teacher plans by
simulation instead of following a rule; its gate is Stage L's step L3 (`../docs/TRAINING_PLAN.md` section 7).

| Module | Lines | What it holds |
|---|---:|---|
| `teachers.py` | 122 | What every teacher shares: the head layout and codebook sizes, `TeacherOutput`, the `TEACHERS` registry and `observed_state`. The registered teachers (v2) read `observed_state`: the actual head and body yaw, what a student sees. Each skill's teacher is in its folder: `../neural_skills/walk_to/teacher.py` (walk_to v2, and v3 with `stop_margin_blocks`, which stops a margin inside the radius) and `../neural_skills/look_at/teacher.py`. |
| `path_teacher.py` | 290 | The path teacher (T1, 2026-10-25, no training): walk_to's teacher that follows the shortest path over the world's traversal graph (`tasks/region_tasks.py`'s `RegionTraversal`). Per lane, one reversed Dijkstra from the goal node (host, scipy) gives every node its cost to the goal; per tick the node under the feet (nearest within 0.75 blocks horizontally and 1.5 vertically), the next hop as the waypoint (one hop on; `waypoint_hops=2` aims two hops on), steered by `walk_to_choices` with the registered rule and the v3 margin. Edges: walk, climb to 1.3 blocks, drop to 3.0 blocks. `PathTeacher(traversal, maximum_turn_degrees=...)` is the entry point; `waypoint_hops=1` is the default; `waypoint_hops=2` is the two-hop option, reported by the bench as `path_hop2` (`path_hop1` is the default). Not in `TEACHERS` and not certified (its numbers are `bench/region_path_teacher.py`'s). |
| `certificates.py` | 1385 | `certify_walk_to`, `certify_look_at`, `certify_come_here`, `certify_track`, `certify_follow`: T1, T2, T3 and T7-lite over seeded batches of simulated episodes (track: T1 and T3; follow: T1 to T3, T2 provisional); the CLI writes the certificate as JSON (`--walk-stop-margin` for walk_to v3; `--teacher follow` for follow). |
| `certify_live.py` | 291 | T4 for the walk_to and look_at teachers (look_at since 2026-10-20 night, `--teacher look_at`): the scripted teacher in closed loop on the live 0.7.2 server (a private bridge, port 5570), paired with the simulator from the same start and goal (stock gen 5, or gen 7 under `../runtime/gen7.py`). |
| `lookahead.py` | 245 | The lookahead teacher (Stage L, step L3): every `replan` ticks it forks the simulator and rolls 10 candidate moves (wait, the four ability slots, guard, a step in, out, left or right) `horizon` ticks ahead against the opponent, all at once, and plays the best. The CLI is the L3 measurement: the teacher beside a scripted combat priority over the skill layer's bodies, per opponent weapon. The priority strikes and punishes within the held weapon's reach (`scripted_bodies.reach`) and closes no farther than it. |

Run (the live command, through the 0.7.2 launcher, is in `certify_live.py`'s docstring):

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.skill_agent.teaching.certificates \
    --teacher walk_to --seeds 77001 --episodes 256 --out PATH --date 2026-10-18 [--walk-stop-margin 0.25]
JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python -m experimental.neural_agent.skill_agent.teaching.lookahead \
    --batch 16 --ticks 600 --horizon 24 --replan 4 --targets iron_sword,iron_battleaxe,iron_mace,iron_daggers \
    --health-scale 4 --seed 0
```

On gen 7, run the lookahead CLI through `../runtime/gen7.py` (its docstring).

Gates: `test_sa_teachers.py`, `test_sa_certificates.py`, `test_sa_lookahead.py`. `certify_live.py` needs the live
server and has no unit test; its result is README section 2.15.
