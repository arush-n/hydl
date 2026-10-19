# step_and_hop: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval; it is run C4 there).

## 1. What it does

Get up ledges and over gaps on the way to a goal: the second lesson of the navigation chain (owner, 2026-10-17:
navigation is one chained skill, walk_to, then step_and_hop, then go_around).

| | |
|---|---|
| Request | walk_to's: the target position, the radius, the gait |
| Success and done | arrived: the planar distance to a goal past the obstacle is within the radius |
| Heads | walk_to's locomotion and body yaw, plus jump |

## 2. Teacher

walk_to v3 (`../walk_to/teacher.py`) plus a jump rule: jump when a step-up or a gap lies inside the jump window ahead.
The teacher reads the terrain (privileged reading is allowed for teachers). A new teacher, so a new certificate: T1 to
T3, a check that it jumps only where the terrain needs it, and T4 live (jump timing is where the simulator was last
corrected).

## 3. World

Terrain on `npc_walk`. Confirmed: the render passes the ledge walk (1.9e-5 m) and jumps in place, held and held with
forward (`../../docs/TRAINING_PLAN.md`, S2); the landing walk resume is built (`580e37b`: the gen-7 NPC walks again on
the third tick after touchdown, within 1.5e-5 m of native). Needed: a terrain task world with ledges and gaps of
sampled sizes (fixtures), since `tasks/simple_tasks.py` is flat with no geometry provider.

## 4. Student

**The student sees no terrain today,** for two reasons, both confirmed in the code: the legal observation adapter
emits its geometry tokens as invalid, zero features (`../../model/observation_adapter.py:16-17`: "The observation's
block, inventory and geometry tokens are not read"), and the trainer passes only the entity and self tokens
(`../../training/distill.py`, `_batch_tokens`). Both are built first: geometry tokens from the observation's geometry,
then passed in training. Then the shared generalist learns it as walk_to on terrain, by continued distillation with
rehearsal of flat walk_to (H2: each verb at or above its single-skill result).

## 5. Bars

- T5: at most 10 points under the teacher, on held success;
- held out: ledge heights or gap widths not trained, at least 0.9 of seen;
- E4: pace and radius as walk_to's;
- no regression: flat walk_to and look_at within tolerance of their bars.

## 6. Stages

1. **Build:** the terrain task world, geometry tokens in the adapter and the trainer, the jump teacher and its
   certificate.
2. **Smoke**, then **the run**, with a breakdown of misses by obstacle size and approach angle.

## 7. Waits on

The terrain task world, geometry tokens (adapter and trainer), walk_to's release, the owner's approval.
