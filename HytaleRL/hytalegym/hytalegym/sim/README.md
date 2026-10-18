# The simulator and its generations

Owner's goal (2026-10-17): "the JAX sim should be generation agnostic, so when future generations
release we can easily update everything." So there is ONE simulator engine. A generation is only three
things: a profile, the content tables extracted from that version's game files, and the profile-gated
constructs for mechanics that exist in some versions and not others. Updating to a new release means
extracting tables, reviewing the diff, enabling or adding constructs, running the differentials, and
flipping the default. Never copying kernels.

| Part | Where | What it is |
|---|---|---|
| engine | source `hytalegym/jax/combat/...` (frozen gen-5 reference); each generation's package `sim/<gen>/engine/` is RENDERED from the committed source by `sim/common/render.py` with the generation's `sim/<gen>/overlay.py` (never committed: `sim/<gen>/.gitignore`; `sim/common/RENDER.md`) | the one JAX simulator: kernels, factored actions, observation layout machinery, events, world, contracts |
| common | `sim/common/` | content schema and loader (profile -> typed tables), the content sha for the contract stamp, the differential harness |
| gen 5 | `sim/gen5/` (profile, `data/` extracted from the 0.5.7 `Assets.zip`) | Hytale 0.5.7; deprecated by the owner. Until the engine reads the gen-5 profile bit-exactly, the code it holds today is the reference and is not edited |
| gen 7 | `sim/gen7/` (profile, `data/` from the 0.7.2 `Assets.zip`, `mechanics/` constructs new in 0.7.2, `GEN7-PLAN.md`) | Hytale 0.7.2; the training target |
| extractors | `tools/content_extract/` | one per content family; run on any version's `Assets.zip`; output tables with provenance |
| runbook | `sim/PORTING.md` | the steps for generation N+1 |
| tests | `tests/sim/` | content, extractors, adapters, constructs, differentials |

Rules:
1. **Content comes from the game's files.** Speeds, stats, weapon tables, projectiles, consumables,
   durability, block properties, recipes, NPC roles: extracted, with provenance (asset path, asset
   sha256, game version), never transcribed by hand. Hand-maintained values are listed in `PORTING.md`
   with a reason each, and the list shrinks with every port.
2. **A behaviour difference is a profile switch, implemented once.** Line-of-sight rule, dodge
   directions, knockback model, hit interrupt and flinch, mana and runes, hookshot, climbing, doors,
   swimming, gliding, riding: each is a construct in the engine behind a profile field, not a copied
   module and not a conditional spread through shared code.
3. **Fidelity is measured.** A mechanic is calibrated for a generation only after its native
   differential against a live server of that version passes its bar (gen 7:
   `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md`; the checklist in
   `experimental/neural_agent/skill_agent/BUILD.md` section 4).
4. **Older generations stay reproducible.** The engine with the gen-5 profile must reproduce today's
   gen-5 rollouts byte for byte before the reference code is retired; recorded evidence stays valid.
5. **Policies retrain on a new generation** (owner decision 2026-10-17); no re-stamped carry-over.
