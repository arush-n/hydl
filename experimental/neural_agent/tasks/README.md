# tasks/ — the composition task and its arenas

The one task every middle-layer instance is measured on: pursue, evade when damaged, resume pursuit
(ADR-001 P2). It exists to measure composition, not skill quality: the skills are the stand-ins of
`../skill_library/`.

| File | Lines | What it is |
|---|---:|---|
| `composition.py` | 421 | `CompositionTask`: the executive state it runs under (pursuit always, evader with the `on_damage` constraint label, `EVADER_PRIORITY` above `PURSUIT_PRIORITY`), the per-episode verdict `classify` (composed, composed_loose, pursue_only, failed), the tick budget. |
| `arenas.py` | 510 | Arena selection for the runs: one builder, four arenas (`combat_open_flat` is the standard; `combat_cover` is the pillar arena below). |
| `flat_arena.py` | 688 | The flat two-actor pursuit arena, one decision at a time. |

Gates: `test_na_tasks.py`, `test_na_arenas.py`.

Removed 2026-10-18 (owner-approved cleanup): `events.py`, the damage-event read helper and ablation injector nothing on the current path used.

Known measurement facts: `j0` on seed 0 at 512 ticks, batch 8, 8 episodes is 43/81 = 0.531 (Wilson lower
0.423); catches depend on the horizon (1 tick = 0.045 s in the gen-5 simulator; the gen-7 engine steps at
1/30 s, so horizons will be restated in ticks when the port lands).

## Arena `combat_cover` (task #98, 2026-10-20)

`build_arena("combat_cover", batch, ticks, seed, loadout=None)` in `arenas.py`; `--arena combat_cover` in `eval/run_composition.py`.
It is the `combat_open_flat` legal scene (default reset, default melee opponent, 120-degree sector, sensor range 16) with local
geometry added through `geometry_arsenal_world_capabilities` and `geometry_provider`:

- Floor: solid and opaque over the whole 9 by 9 window about the spawn block (origin `(0, 65, 0)`).
- Pillars: six 1 by 1 columns, 3 high, at block offsets `(3,0) (1,3) (-1,3) (-3,0) (-1,-3) (1,-3)`, radii 3.0 and sqrt(10) from the spawn.
  The record (`describe()["pillars"]`) gives the offsets, radii and angles; the contract digest covers the layout and the flags.
- Radius, measured on 2026-10-20 with `geometry_perception_line_of_sight_result` (`hytalegym/jax/world/perception/los.py`):
  the requested 5 to 15 blocks is beyond the local window (`hytalegym/geometry/contract.py`, `CELL_RADIUS = 4`). A clear
  line is visible at block offsets 1 to 4 and not at 5, where `geometry_exhausted` is set. The usable radius is 3 blocks
  on an axis (a pillar at 3 blocks hides a target at 4) and 4.24 (sqrt 18) on the diagonal (pillar (3,3) hides (4,4)).
- Consequence (read in `hytalegym/jax/combat/arsenal/runtime.py` 2240 and 2364-2368, not run): `geometry_exhausted` is sticky
  and truncates the episode, so an actor that leaves the window ends the episode. Measured only for a target placed at offset 5.
- Loadout: `iron_sword` (default) or `iron_shortbow`, through `build_arena(..., loadout=...)`. The CLI has no loadout flag, so
  composition runs use the default. The opponent's loadout is unchanged.
- Gate: `tests/experimental/neural_agent/test_na_arenas.py` (cover layout and digest, loadout, and the behavioural case).
