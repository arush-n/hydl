# skill_agent/neural_skills/ — one folder per neural skill

Every neural skill has its own folder: what it does, its training plan, its teacher and task code, and the scripts
that certify its teacher, train it and score it (owner, 2026-10-19). The skills share one network and one trainer
(owner, 2026-10-18: one generalist, request-conditioned policy), so a skill's `train.sh` runs the generalist's current
stage, which trains every skill in it at once. The folder is `neural_skills/`, not `skills/`, because `.gitignore:189`
ignores every `skills/` folder.

## What a skill folder holds

| File | What it is |
|---|---|
| `README.md` | what the skill does (request, success, done, heads), its status and results, its files, how to run it |
| `TRAINING.md` | its plan: the teacher and certificate, the world, the student, the bars, the stages, what it waits on |
| `teacher.py` | the scripted teacher; its registry entry is in `../teaching/teachers.py` (`TEACHERS`) |
| `task.py` | the goals: the sampler, the episode ranges and the success test |
| `certify.sh` | the teacher's certificate, T1 to T3 (`../teaching/certificates.py`) |
| `train.sh` | the generalist's current stage (`train_generalist.sh` here, which calls `../training/run_distill.sh`) |
| `evaluate.sh` | a run's bars for this skill (`../training/bars.py`) |

A planned skill has only `README.md` and `TRAINING.md` until the owner approves its build stage. A new skill starts
from walk_to's layout. Training is paused (owner, 2026-10-19): the launchers stay, and no run starts until it reopens.

## The shared parts

| Where | What it is |
|---|---|
| `train_generalist.sh` | the generalist's current stage: its size and options, the one place they are set |
| `../model/` | the network: the set encoder and the request-conditioned generalist |
| `../training/` | the trainer (`distill.py`), the run launcher (`run_distill.sh`), the bar scorer (`bars.py`) |
| `../teaching/` | the teacher registry, the observed-state rule and the certificate harness |
| `../tasks/` | the flat `npc_walk` task world and `run_episode` |
| `../docs/TRAINING_PLAN.md` | the method, the hypotheses and the staged approvals every plan follows |

## Skills with a folder

| Skill | Status (2026-10-19) | Teacher |
|---|---|---|
| [walk_to](walk_to/README.md) | trained in the generalist; passes every walk_to bar | v3, certified |
| [look_at](look_at/README.md) | trained in the generalist; the held-out look fails its bar | v2, certified |
| [come_here](come_here/README.md) | planned; next, as it needs no new mechanic | walk_to v3 aimed at an entity |
| [track](track/README.md) | planned; needs a moving entity | look_at v2 aimed at an entity |
| [follow](follow/README.md) | planned; needs come_here and a moving person | walk_to v3 behind the person |
| [step_and_hop](step_and_hop/README.md) | planned; needs a terrain world and geometry tokens | walk_to v3 plus a jump rule |
| [go_around](go_around/README.md) | planned; needs the 0.7.2 library swap, geometry tokens, step_and_hop | A* plus walk_to v3 |

## Every other catalogue skill

The catalogue (`agent_skills/training_runs/SKILL_CATALOGUE.md`, private, built from the 0.7.2 game files) has 126
rows: 86 learned skills, 33 commands and 7 activities, plus the 13 activities of its section 14. The learned skills
without a folder are below, with what each waits on. The waits-on column is this session's reading of the plan and the
simulator notes, not a measured check; the catalogue's simulator column describes gen 5, which is deprecated.

Codes:
- **terrain**: a terrain task world, and geometry tokens in the adapter and the trainer (neither exists today);
- **entity**: a moving entity (or several) in the task world;
- **sight**: occlusion, and the belief's last-seen inputs;
- **edits**: breaking and placing blocks in training (the Region executor, goal-marked candidates, the placement proof);
- **combat**: the gen-7 combat rows, teachers trained in cheat mode and an opponent pool (TRAINING_PLAN 2e);
- **mechanic**: a mechanic the simulator does not model, or models unchecked;
- **heads**: an action the 12-head surface lacks (moving items, recipes, doors, riding and gliding controls).

| Family | Skill | Waits on |
|---|---|---|
| moving | run_to, sneak_to | walk_to's gait argument: trained (run, sneak) |
| moving | sprint_to | walk_to's gait argument: the teacher sprints forward only; not sampled yet |
| moving | slide, double_jump, bounce, dive_and_breathe | mechanic |
| moving | land_safely, swim_to | mechanic, terrain |
| moving | climb | mechanic (no climbable blocks), terrain |
| moving | glide, ride | mechanic, heads |
| moving | hookshot_to | mechanic (gen 7 only) |
| moving | dodge | combat (a threat to dodge; the four dodges are on the locomotion head) |
| moving | follow_route | go_around (waypoints from the A* route) |
| looking | look_around | a coverage metric for "surroundings seen"; then look_at's recipe |
| looking | search_for | sight |
| looking | spot_threats | entity (several) |
| gathering | mine_block, mine_in_direction, mine_down_stairs, chop_tree, dig, pick_by_hand, shear_plants | edits |
| gathering | pick_up | dropped items in the task world |
| farming | till, plant, water_crops, fertilize, harvest_crops | edits, mechanic |
| farming | lure_animal, milk_shear_collect, capture_creature | mechanic (animal behaviour), entity |
| building | place_block, place_row, build_up, bridge, build_shape, fill_or_level, carry_block, place_furniture | edits |
| survival | avoid_hazards | terrain (hazard cells as a region argument) |
| survival | recover_stamina | sprint_to first (it is "stop sprinting") |
| survival | hide | sight |
| melee | sword, longsword, axe, battleaxe, mace, club, daggers, spear, claws, shield_guard, flamethrower | combat |
| ranged | shortbow, crossbow, throw_spear, throw_kunai, throw_bomb, gun, darts_or_blowgun | combat |
| magic | fireball, wind_strike, chain_hook, ground_slam, charged_frost_shot | mechanic (runes, TRAINING_PLAN 2e), combat |
| magic | staff_wand_spellbook | combat |
| tactics | guard, bash, punish, interrupt | combat (timing on visible cues) |
| tactics | keep_range, kite, flank, finish | entity, combat |
| other agents | avoid | entity (a pursuer) |
| other agents | approach_friendly | come_here, with a villager in the world |
| other agents | hide_from | sight, entity |
| other agents | signal | look_at at the thing; its success needs an ally's gaze (the middle layer) |

**Not neural skills.** Commands are scripts the executor runs (owner, 2026-10-17: crafting, cooking, equipping and
inventory are commands): use_teleporter, use_spyglass, read_map, collect_water, loot_chest, set_trap, call_hunt,
change_variant, switch_item, best_tool_for, equip_armor, set_rune, move_item, drop_or_give, repair_gear, reload,
craft, cook, smelt, brew, tan_or_salvage, trade, eat, drink_potion, sleep, sit, deploy, enrage, poison_imbue,
wait_here, give, trade_with, share_chest. Activities are the middle layer's programs over skills: build_from_plan,
fight_group, retreat_and_heal, escort, help_fight, herd, and the catalogue's 13 (get wood to escape and recover).
tame is not yet classed.
