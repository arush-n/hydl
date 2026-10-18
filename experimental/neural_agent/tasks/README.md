# tasks/ — the composition task and its arenas

The one task every middle-layer instance is measured on: pursue, evade when damaged, resume pursuit
(ADR-001 P2). It exists to measure composition, not skill quality: the skills are the stand-ins of
`../skill_library/`.

| File | Lines | What it is |
|---|---:|---|
| `composition.py` | 421 | `CompositionTask`: the executive state it runs under (pursuit always, evader with the `on_damage` constraint label, `EVADER_PRIORITY` above `PURSUIT_PRIORITY`), the per-episode verdict `classify` (composed, composed_loose, pursue_only, failed), the tick budget. |
| `arenas.py` | 271 | Arena selection for the runs: one builder, two arenas (`combat_open_flat` is the standard). |
| `flat_arena.py` | 688 | The flat two-actor pursuit arena, one decision at a time. |
| `events.py` | 108 | `DAMAGE_RECEIVED` read helper and the ablation injector. Cleanup candidate (awaiting the owner). |

Gates: `test_na_tasks.py`, `test_na_arenas.py`.

Known measurement facts: `j0` on seed 0 at 512 ticks, batch 8, 8 episodes is 43/81 = 0.531 (Wilson lower
0.423); catches depend on the horizon (1 tick = 0.045 s in the gen-5 simulator; the gen-7 engine steps at
1/30 s, so horizons will be restated in ticks when the port lands).
