# Hytale 0.5 -> 0.7.2 asset and jar diff (P0)

Generated 2026-10-17T02:00:34 by `tools/diff_assets_072.py` (stdlib only; read-only on inputs).
Measured on the zips and jars below. Simulator references come from `HytaleRL/hytalegym/{hytalegym,tests}` (`*.py`, `*.json`).

## Inputs

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| old assets | `/Users/arush/Library/Application Support/Hytale/install/v0.5/package/game/latest/Assets.zip` | 3,428,467,309 | `6D14386D23C3C4B8A68FDAC6C4F2FD4FE26291A40A710F90438AFA964D838AB6` |
| new assets | `/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Assets.zip` | 3,942,760,463 | `EE8F8D5B332874EB928144726077FA66F80C117A5412017A5CDE6FCE2D9BB2A0` |
| old jar | `/Users/arush/Library/Application Support/Hytale/install/v0.5/package/game/latest/Server/HytaleServer.jar` | 123,396,480 | `B63AE21F1F103D1494C2673FB28E6AA28B8A898159C791854C0C05646AB01816` |
| new jar | `/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Server/HytaleServer.jar` | 112,406,344 | `23F71F03763422CE7C04DF191DB49CD0E7D87D43B12BAAD75C43FDC202410AFF` |

Jar `Implementation-Version`: old `0.5.10`, new `0.7.2`.

**Pin check: MISMATCH.** Old assets sha256 `6D14386D23C3C4B8A68FDAC6C4F2FD4FE26291A40A710F90438AFA964D838AB6` does not equal the simulator pin `1B8802C284C228AE4549DAC037716C175BC6B94B0FC9AAC2B7039AC6BD2FFD5D` (`HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:34`); the pin starts `1B8802C2`, this zip starts `6D14386D`.
The 0.5 jar reports `0.5.10`, so the installed 0.5 tree is not the 0.5.7 that the pin was taken from. The TRAINING-0.7.2 measurements were also taken against 0.5.10. Treat "0.5" in this report as the installed build.

## Summary counts

- Asset entries (files): old 60,148, new 68,759. Added 9,909, removed 1,298, changed 4,286, unchanged 54,564 (compared by CRC-32 and size).

| Folder | Old | New | Added | Removed | Changed | Unchanged |
|---|---:|---:|---:|---:|---:|---:|
| `Common` | 24,914 | 28,605 | 4,284 | 593 | 1,450 | 22,871 |
| `CommonAssetsIndex.hashes` | 1 | 1 | 0 | 0 | 1 | 0 |
| `Cosmetics` | 38 | 85 | 47 | 0 | 21 | 17 |
| `Server/Abilities` | 0 | 19 | 19 | 0 | 0 | 0 |
| `Server/AssetEditor` | 0 | 2 | 2 | 0 | 0 | 0 |
| `Server/Audio` | 1,761 | 2,278 | 530 | 13 | 967 | 781 |
| `Server/BarterShops` | 2 | 3 | 1 | 0 | 0 | 2 |
| `Server/BlockTypeList` | 13 | 13 | 0 | 0 | 5 | 8 |
| `Server/Camera` | 110 | 120 | 10 | 0 | 0 | 110 |
| `Server/Drops` | 676 | 718 | 47 | 5 | 28 | 643 |
| `Server/EncounterManager` | 0 | 67 | 67 | 0 | 0 | 0 |
| `Server/Entity` | 269 | 430 | 167 | 6 | 31 | 232 |
| `Server/Environments` | 122 | 134 | 13 | 1 | 1 | 120 |
| `Server/Farming` | 5 | 5 | 0 | 0 | 0 | 5 |
| `Server/GameplayConfigs` | 5 | 8 | 3 | 0 | 5 | 0 |
| `Server/HytaleGenerator` | 219 | 351 | 202 | 70 | 40 | 109 |
| `Server/Instances` | 329 | 417 | 142 | 54 | 21 | 254 |
| `Server/Item` | 6,893 | 7,730 | 1,024 | 187 | 1,093 | 5,613 |
| `Server/Languages` | 11 | 16 | 5 | 0 | 5 | 6 |
| `Server/MacroCommands` | 11 | 20 | 9 | 0 | 3 | 8 |
| `Server/Models` | 436 | 504 | 73 | 5 | 18 | 413 |
| `Server/NPC` | 1,537 | 1,858 | 626 | 305 | 412 | 820 |
| `Server/Objective` | 19 | 18 | 0 | 1 | 0 | 18 |
| `Server/Particles` | 2,344 | 3,356 | 1,026 | 14 | 14 | 2,316 |
| `Server/PortalTypes` | 5 | 7 | 2 | 0 | 5 | 0 |
| `Server/PrefabEditorCreationSettings` | 1 | 1 | 0 | 0 | 0 | 1 |
| `Server/PrefabList` | 77 | 78 | 1 | 0 | 0 | 77 |
| `Server/Prefabs` | 7,831 | 9,290 | 1,459 | 0 | 109 | 7,722 |
| `Server/ProjectileConfigs` | 112 | 189 | 84 | 7 | 39 | 66 |
| `Server/Projectiles` | 87 | 86 | 2 | 3 | 4 | 80 |
| `Server/ResponseCurves` | 12 | 13 | 1 | 0 | 0 | 12 |
| `Server/Rewards` | 0 | 1 | 1 | 0 | 0 | 0 |
| `Server/ScriptedBrushes` | 39 | 39 | 0 | 0 | 3 | 36 |
| `Server/TagPatterns` | 4 | 11 | 7 | 0 | 0 | 4 |
| `Server/TriggerVolumes` | 1 | 10 | 9 | 0 | 1 | 0 |
| `Server/Weathers` | 87 | 106 | 19 | 0 | 6 | 81 |
| `Server/WordLists` | 1 | 1 | 0 | 0 | 0 | 1 |
| `Server/World` | 12,175 | 12,141 | 0 | 34 | 4 | 12,137 |
| `Server/WorldEvent` | 0 | 27 | 27 | 0 | 0 | 0 |
| `manifest.json` | 1 | 1 | 0 | 0 | 0 | 1 |

Simulator references (`Server/...` paths in `hytalegym/` and `tests/`):

- Source files scanned: 1,073 `.py`, 37 `.json`.
- Unique referenced file paths: 437. Present in 0.5 zip: 375. Not present in 0.5 zip (unresolved): 62.
- Of the 0.5 ones: changed 127, unchanged 240, missing at the same path in 0.7.2 8.
- Bare asset ids from `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py` (quoted `Prefix_Name` literals plus `_STANDARD_SPEAR_IDS` expanded to `Weapon_Spear_<id>`): 37 unique ids; 2 had no same-stem file in the 0.5 zip.
- Dropped as fragments or directory prefixes (approx): 273. Python files failing to parse: 0.

- Server jar classes: old 37,150, new 38,436. Added 1,875, removed 589, changed 4,813, of which in keyword packages 951.

## Simulator-referenced files changed

### `Server/Audio/SoundEvents/BlockSounds/Default/SFX_Default_Build.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_audio_source_audit.py:146`

- parse: old: strict JSON; new: strict JSON
- added `AudioCategory` = "AudioCat_BlockSounds"

### `Server/Entity/HitboxCollision/HardCollision.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/weapon_archetypes/deployables/hytale_0_5_7.py:20`, `HytaleRL/hytalegym/hytalegym/combat/assets/weapon_archetypes/deployables/hytale_0_5_7.py:29`

- parse: old: strict JSON; new: strict JSON
- no field-level differences (bytes differ only in formatting or key order)

### `Server/Entity/HitboxCollision/SoftCollision.json`

Referenced by: `HytaleRL/hytalegym/tests/test_asset_surface.py:103`, `HytaleRL/hytalegym/tests/test_asset_surface.py:171`

- parse: old: strict JSON; new: strict JSON
- no field-level differences (bytes differ only in formatting or key order)

### `Server/Entity/MovementConfig/Default.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/types.py:103`, `HytaleRL/hytalegym/hytalegym/jax/combat/types.py:137`, `HytaleRL/hytalegym/hytalegym/jax/combat/types.py:70` (+7 more)

- parse: old: strict JSON; new: strict JSON
- added `MaxSlopeAngleDegrees` = 30.0
- added `MaxWallAngleDegrees` = 15.0

### `Server/Entity/Stats/Stamina.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/mechanics/schema/contract.py:120`, `HytaleRL/hytalegym/hytalegym/jax/combat/types.py:187`, `HytaleRL/hytalegym/hytalegym/jax/combat/types.py:71` (+1 more)

- parse: old: strict JSON; new: strict JSON
- added `Regenerating[0].Conditions[2].UseIgnoreFlag` = true
- added `Regenerating[2].Conditions[2].UseIgnoreFlag` = true

### `Server/GameplayConfigs/Default.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/types.py:68`, `HytaleRL/hytalegym/tests/fidelity/surrogate_locomotion_source_audit.py:102`, `HytaleRL/hytalegym/tests/test_asset_surface.py:271`

- parse: old: strict JSON; new: strict JSON
- removed `Ping.PingDuration` = 5.0
- removed `Ping.PingCooldown` = 1.0
- removed `Ping.PingBroadcastRadius` = 100.0
- removed `Ping.PingSound` = "SFX_Ping"
- added `Plugin.Wilderness.Enabled` = true
- added `Plugin.Wilderness.OwnedHomeChunkRadius` = 3
- added `Plugin.Wilderness.OwnedHomeChunkRadiusY` = 3
- added `Plugin.Wilderness.UnownedHomeChunkRadius` = 2
- added `Plugin.Wilderness.UnownedHomeChunkRadiusY` = 2
- added `Plugin.Wilderness.PlayerTrackerChunkRadius` = 8
- added `Plugin.Wilderness.PlayerTrackerChunkRadiusY` = 2
- added `Plugin.WorldEvents.ExclusionRadius` = 512
- added `Plugin.WorldEvents.Events[0].$Comment` = "Starts the global pre-temple breach event on a universe that has not run it yet."
- added `Plugin.WorldEvents.Events[0].Event` = "Goblin_Breach_Prelude_Bootstrap"
- added `Plugin.WorldEvents.Events[0].Transient` = true
- added `Plugin.WorldEvents.GlobalEvents[0].$Comment` = "Global so Stage1 can stop it by id. The stop persists in universe/events.json, so it stays off across restarts."
- added `Plugin.WorldEvents.GlobalEvents[0].Id` = "b7d9e3a1-6c24-4f5b-8e17-9a3d2c40b8f6"
- added `Plugin.WorldEvents.GlobalEvents[0].Event` = "Goblin_Breach_Prelude_Event"
- added `Plugin.WorldEvents.GlobalEvents[1].Id` = "e3f1c5d2-4b6a-4f8e-9c3b-1a2b3c4d5e6f"
- added `Plugin.WorldEvents.GlobalEvents[1].Event` = "Goblin_Breach_First_Event"
- added `Plugin.WorldEvents.GlobalEvents[1].Transient` = true
- added `Plugin.WorldEvents.GlobalEvents[2].Id` = "60fb3b60-0403-4475-8478-b71d817aca83"
- added `Plugin.WorldEvents.GlobalEvents[2].Event` = "Goblin_Breach_Perpetual_Event"
- added `Plugin.WorldEvents.GlobalEvents[2].Transient` = true
- added `CreativeEraserInteraction` = "Creative_Eraser"

### `Server/Item/Interactions/Consumables/Consume_Charge_Potion_Fast.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:1050`

- parse: old: strict JSON; new: strict JSON
- removed `Interactions[1].Next.1.5.Next.Interactions[0].DefaultValue.Interactions[0]` = "Consumed_SFX"
- removed `Interactions[1].Next.1.5.Next.Interactions[2].RunTime` = 0.2
- removed `Interactions[1].Next.1.5.Next.Interactions[2].$Comment` = "Delay before next consume cycle can start to prevent sound overlap"
- added `Interactions[1].Next.1.5.Next.Interactions[0].DefaultValue.Interactions[0].Type` = "Simple"
- added `Interactions[1].Next.1.5.Next.Interactions[1].DefaultOk` = true
- added `Interactions[1].Next.1.5.Next.Interactions[2].DefaultValue.Interactions[0]` = "Effect"
- added `Interactions[1].Next.1.5.Next.Interactions[2].Var` = "Effect"
- added `Interactions[1].Next.1.5.Next.Interactions[3].Type` = "Simple"
- added `Interactions[1].Next.1.5.Next.Interactions[3].RunTime` = 0.2
- added `Interactions[1].Next.1.5.Next.Interactions[3].$Comment` = "Delay before next consume cycle can start to prevent sound overlap"
- changed `Interactions[1].FailOnDamage`: true -> false
- changed `Interactions[1].HorizontalSpeedMultiplier`: 0.4 -> 0.75
- changed `Interactions[1].Next.1.5.Next.Interactions[0].Var`: "ConsumedSFX" -> "ConsumeCooldown"
- changed `Interactions[1].Next.1.5.Next.Interactions[1].DefaultValue.Interactions[0]`: "Effect" -> "Consumed_SFX"
- changed `Interactions[1].Next.1.5.Next.Interactions[1].Var`: "Effect" -> "ConsumedSFX"
- changed `Interactions[1].Next.1.5.Next.Interactions[2].Type`: "Simple" -> "Replace"

### `Server/Item/Interactions/Consumables/Consume_Charge_Potion_Fastest.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:1070`

- parse: old: strict JSON; new: strict JSON
- removed `Interactions[1].Next.0.5.Next.Interactions[0].DefaultValue.Interactions[0]` = "Consumed_SFX"
- removed `Interactions[1].Next.0.5.Next.Interactions[2].RunTime` = 0.2
- removed `Interactions[1].Next.0.5.Next.Interactions[2].$Comment` = "Delay before next consume cycle can start to prevent sound overlap"
- added `Interactions[1].Next.0.5.Next.Interactions[0].DefaultValue.Interactions[0].Type` = "Simple"
- added `Interactions[1].Next.0.5.Next.Interactions[1].DefaultOk` = true
- added `Interactions[1].Next.0.5.Next.Interactions[2].DefaultValue.Interactions[0]` = "Effect"
- added `Interactions[1].Next.0.5.Next.Interactions[2].Var` = "Effect"
- added `Interactions[1].Next.0.5.Next.Interactions[3].Type` = "Simple"
- added `Interactions[1].Next.0.5.Next.Interactions[3].RunTime` = 0.2
- added `Interactions[1].Next.0.5.Next.Interactions[3].$Comment` = "Delay before next consume cycle can start to prevent sound overlap"
- changed `Interactions[1].FailOnDamage`: true -> false
- changed `Interactions[1].HorizontalSpeedMultiplier`: 0.4 -> 0.75
- changed `Interactions[1].Next.0.5.Next.Interactions[0].Var`: "ConsumedSFX" -> "ConsumeCooldown"
- changed `Interactions[1].Next.0.5.Next.Interactions[1].DefaultValue.Interactions[0]`: "Effect" -> "Consumed_SFX"
- changed `Interactions[1].Next.0.5.Next.Interactions[1].Var`: "Effect" -> "ConsumedSFX"
- changed `Interactions[1].Next.0.5.Next.Interactions[2].Type`: "Simple" -> "Replace"

### `Server/Item/Interactions/Weapons/Block_Break.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/test_block_interaction_fields.py:49`

- parse: old: strict JSON; new: strict JSON
- removed `Failed.Next.Var` = "Block_Hit_Camera_Effects"
- removed `Failed.Next.DefaultOk` = true
- removed `Failed.Next.DefaultValue.Interactions[0].Type` = "Simple"
- removed `Failed.Next.DefaultValue.Interactions[0].Effects.CameraEffect` = "Unarmed_Block_Impact"
- added `Failed.Next.Interactions[0]` = "Trigger_Explosion_State_Generic"
- added `Failed.Next.Interactions[1].Type` = "Replace"
- added `Failed.Next.Interactions[1].Var` = "Block_Hit_Camera_Effects"
- added `Failed.Next.Interactions[1].DefaultOk` = true
- added `Failed.Next.Interactions[1].DefaultValue.Interactions[0].Type` = "Simple"
- added `Failed.Next.Interactions[1].DefaultValue.Interactions[0].Effects.CameraEffect` = "Unarmed_Block_Impact"
- changed `Failed.Next.Type`: "Replace" -> "Serial"

### `Server/Item/Interactions/Weapons/DamageEntityParent.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/native_status_projection_v1.json:7`, `HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/native_status_projection_v2.json:7`

- parse: old: strict JSON; new: strict JSON
- removed `Next.Type` = "Serial"
- removed `Next.Interactions[0].Type` = "ApplyEffect"
- removed `Next.Interactions[0].EffectId` = "Red_Flash"
- removed `Next.Interactions[0].Entity` = "Target"
- removed `Next.Interactions[1].Type` = "ClearEntityEffect"
- removed `Next.Interactions[1].Entity` = "Target"
- removed `Next.Interactions[1].EntityEffectId` = "Potion_Health_Regen_Lesser"
- removed `Next.Interactions[2].Type` = "ClearEntityEffect"
- removed `Next.Interactions[2].Entity` = "Target"
- removed `Next.Interactions[2].EntityEffectId` = "Potion_Health_Regen_Small"
- removed `Next.Interactions[3].Type` = "ClearEntityEffect"
- removed `Next.Interactions[3].Entity` = "Target"
- removed `Next.Interactions[3].EntityEffectId` = "Potion_Health_Regen"
- removed `Next.Interactions[4].Type` = "ClearEntityEffect"
- removed `Next.Interactions[4].Entity` = "Target"
- removed `Next.Interactions[4].EntityEffectId` = "Potion_Health_Regen_Greater"
- removed `Next.Interactions[5].Type` = "ClearEntityEffect"
- removed `Next.Interactions[5].Entity` = "Target"
- removed `Next.Interactions[5].EntityEffectId` = "Potion_Health_Regen_Large"
- removed `Next.Interactions[6].Type` = "ClearEntityEffect"
- removed `Next.Interactions[6].Entity` = "Target"
- removed `Next.Interactions[6].EntityEffectId` = "Potion_Stamina_Regen"
- added `Next` = "DamageEntityParent_On_Hit"

### `Server/Item/Interactions/Weapons/Shield/Secondary/Guard/Guard_Bash/Weapon_Shield_Secondary_Guard_Bash.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/native_status_projection_v2.json:13`

- parse: old: strict JSON; new: strict JSON
- changed `Interactions[1].Next.Behaviour`: "Set" -> "Min"
- changed `Interactions[1].Next.StatModifiers.StaminaRegenDelay`: -1.5 -> -1

### `Server/Item/Interactions/Weapons/Shortbow/Primary/Shoot/Weapon_Shortbow_Primary_Shoot.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:892`

- parse: old: strict JSON; new: strict JSON
- no field-level differences (bytes differ only in formatting or key order)

### `Server/Item/Interactions/Weapons/Shortbow/Primary/Shoot/Weapon_Shortbow_Primary_Shoot_Charge.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/projectiles/test_shortbow_charge_trajectory.py:23`

- parse: old: strict JSON; new: strict JSON
- added `Next.0.0.Type` = "ChangeStat"
- added `Next.0.0.Behaviour` = "Set"
- added `Next.0.0.StatModifiers.StaminaRegenDelay` = -1
- added `Next.0.1.StatModifiers.StaminaRegenDelay` = -1
- added `Next.0.3.StatModifiers.StaminaRegenDelay` = -1
- added `Next.0.6.StatModifiers.StaminaRegenDelay` = -1
- added `Next.0.9.StatModifiers.StaminaRegenDelay` = -1
- added `Next.1.2.StatModifiers.StaminaRegenDelay` = -1

### `Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Thrust/Weapon_Sword_Primary_Thrust.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/native_status_projection_v2.json:11`

- parse: old: strict JSON; new: strict JSON
- removed `Next.0.65.Type` = "Serial"
- removed `Next.0.65.Interactions[0].Type` = "Replace"
- removed `Next.0.65.Interactions[0].Var` = "Thrust_StaminaCost"
- removed `Next.0.65.Interactions[0].DefaultOk` = true
- removed `Next.0.65.Interactions[0].DefaultValue.Interactions[0].Type` = "ApplyEffect"
- removed `Next.0.65.Interactions[0].DefaultValue.Interactions[0].EffectId` = "Stamina_Broken_Immune"
- removed `Next.0.65.Interactions[0].DefaultValue.Interactions[1].Type` = "ChangeStat"
- removed `Next.0.65.Interactions[0].DefaultValue.Interactions[1].StatModifiers.Stamina` = -4
- removed `Next.0.65.Interactions[1].Type` = "Replace"
- removed `Next.0.65.Interactions[1].Var` = "Thrust_StaminaRegenDelay"
- removed `Next.0.65.Interactions[1].DefaultOk` = true
- removed `Next.0.65.Interactions[1].DefaultValue.Interactions[0].Type` = "ChangeStat"
- removed `Next.0.65.Interactions[1].DefaultValue.Interactions[0].Behaviour` = "Set"
- removed `Next.0.65.Interactions[1].DefaultValue.Interactions[0].StatModifiers.StaminaRegenDelay` = -2
- removed `Next.0.65.Interactions[2].Type` = "Simple"
- removed `Next.0.65.Interactions[2].RunTime` = 0.05
- removed `Next.0.65.Interactions[2].Next.Type` = "Serial"
- removed `Next.0.65.Interactions[2].Next.Interactions[0].Type` = "Replace"
- removed `Next.0.65.Interactions[2].Next.Interactions[0].Var` = "Thrust_Force"
- removed `Next.0.65.Interactions[2].Next.Interactions[0].DefaultOk` = true
- removed `Next.0.65.Interactions[2].Next.Interactions[0].DefaultValue.Interactions[0]` = "Weapon_Sword_Primary_Thrust_Force"
- removed `Next.0.65.Interactions[2].Next.Interactions[1].Type` = "Replace"
- removed `Next.0.65.Interactions[2].Next.Interactions[1].Var` = "Thrust_Selector"
- removed `Next.0.65.Interactions[2].Next.Interactions[1].DefaultOk` = true
- removed `Next.0.65.Interactions[2].Next.Interactions[1].DefaultValue.Interactions[0]` = "Weapon_Sword_Primary_Thrust_Selector"
- added `Rules.InterruptedBy[0]` = "Secondary"
- added `Next.0.565.Type` = "Serial"
- added `Next.0.565.Interactions[0].Type` = "Replace"
- added `Next.0.565.Interactions[0].Var` = "Thrust_StaminaCost"
- added `Next.0.565.Interactions[0].DefaultOk` = true
- added `Next.0.565.Interactions[0].DefaultValue.Interactions[0].Type` = "ApplyEffect"
- added `Next.0.565.Interactions[0].DefaultValue.Interactions[0].EffectId` = "Stamina_Broken_Immune"
- added `Next.0.565.Interactions[0].DefaultValue.Interactions[1].Type` = "ChangeStat"
- added `Next.0.565.Interactions[0].DefaultValue.Interactions[1].StatModifiers.Stamina` = -4
- added `Next.0.565.Interactions[1].Type` = "Replace"
- added `Next.0.565.Interactions[1].Var` = "Thrust_StaminaRegenDelay"
- added `Next.0.565.Interactions[1].DefaultOk` = true
- added `Next.0.565.Interactions[1].DefaultValue.Interactions[0].Type` = "ChangeStat"
- added `Next.0.565.Interactions[1].DefaultValue.Interactions[0].Behaviour` = "Set"
- added `Next.0.565.Interactions[1].DefaultValue.Interactions[0].StatModifiers.StaminaRegenDelay` = -2
- added `Next.0.565.Interactions[2].Type` = "Simple"
- added `Next.0.565.Interactions[2].RunTime` = 0.043
- added `Next.0.565.Interactions[2].Next.Type` = "Serial"
- added `Next.0.565.Interactions[2].Next.Interactions[0].Type` = "Replace"
- added `Next.0.565.Interactions[2].Next.Interactions[0].Var` = "Thrust_Force"
- added `Next.0.565.Interactions[2].Next.Interactions[0].DefaultOk` = true
- added `Next.0.565.Interactions[2].Next.Interactions[0].DefaultValue.Interactions[0]` = "Weapon_Sword_Primary_Thrust_Force"
- added `Next.0.565.Interactions[2].Next.Interactions[1].Type` = "Replace"
- added `Next.0.565.Interactions[2].Next.Interactions[1].Var` = "Thrust_Selector"
- added `Next.0.565.Interactions[2].Next.Interactions[1].DefaultOk` = true
- added `Next.0.565.Interactions[2].Next.Interactions[1].DefaultValue.Interactions[0]` = "Weapon_Sword_Primary_Thrust_Selector"
- changed `HorizontalSpeedMultiplier`: 0.8 -> 0.75

### `Server/Item/Interactions/Weapons/Wand/Root_Cast.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:106`, `HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:114`

- parse: old: strict JSON; new: strict JSON
- removed `Effects.Particles[0].RotationOffset.Roll` = 180
- added `Effects.Particles[0].DetachedFromModel` = true
- added `Effects.Particles[0].PositionOffset.X` = -0.5
- added `Effects.Particles[0].PositionOffset.Z` = 3.2
- added `Effects.Particles[0].TargetEntityPart` = "Entity"
- added `Effects.Particles[0].RotationOffset.Pitch` = 180
- added `Effects.FirstPersonParticles[0].SystemId` = "NatureBeam"
- added `Effects.FirstPersonParticles[0].TargetEntityPart` = "Entity"
- added `Effects.FirstPersonParticles[0].TargetNodeName` = "Body"
- added `Effects.FirstPersonParticles[0].PositionOffset.Y` = 1.5
- added `Effects.FirstPersonParticles[0].PositionOffset.Z` = 3
- added `Effects.FirstPersonParticles[0].PositionOffset.X` = -0.2
- added `Effects.FirstPersonParticles[0].RotationOffset.Pitch` = 180
- changed `Effects.Particles[0].TargetNodeName`: "Handle" -> "Head"

### `Server/Item/Items/Deco/Deco_Trophy_Harvest.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/crafting/contract.py:16`, `HytaleRL/hytalegym/tests/fidelity/surrogate_crafting_source_audit.py:43`

- parse: old: strict JSON; new: strict JSON
- changed `BlockType.Interactions.Use.Interactions[0].Effects.LocalSoundEventId`: "SFX_Plushie_Build" -> "SFX_Plushie_Interact"

### `Server/Item/Items/Potion/Potion_Antidote.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:74 (bare id Potion_Antidote)`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Effect.Interactions[0].Next.Next.Next.EffectId` = "Antidote"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.EntityEffectId` = "Ability_Poison_Stack_A"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.Next.Type` = "ClearEntityEffect"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.Next.EntityEffectId` = "Ability_Poison_Stack_B"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.Next.Next.Type` = "ClearEntityEffect"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.Next.Next.EntityEffectId` = "Ability_Poison_Stack_C"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.Next.Next.Next.Type` = "ApplyEffect"
- added `InteractionVars.Effect.Interactions[0].Next.Next.Next.Next.Next.Next.EffectId` = "Antidote"
- changed `InteractionVars.Effect.Interactions[0].Next.Next.Next.Type`: "ApplyEffect" -> "ClearEntityEffect"

### `Server/Item/Items/Potion/Potion_Health.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:74 (bare id Potion_Health)`

- parse: old: strict JSON; new: strict JSON
- removed `MaxStack` = 5
- added `Interactions.Secondary` = "Root_Secondary_Consume_Potion_Health"
- added `InteractionVars.ConsumeCooldown.Interactions[0]` = "Consume_Potion_Health_Cooldown"

### `Server/Item/Items/Potion/Potion_Stamina.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:74 (bare id Potion_Stamina)`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Effect.Interactions[0].Interactions[1].Type` = "ApplyEffect"
- removed `InteractionVars.Effect.Interactions[0].Interactions[1].EffectId` = "Potion_Stamina_Cooldown"
- removed `MaxStack` = 5
- added `InteractionVars.ConsumeCooldown.Interactions[0]` = "Consume_Potion_Stamina_Cooldown"

### `Server/Item/Items/Weapon/Battleaxe/Template_Weapon_Battleaxe.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:140`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:146`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:152` (+2 more)

- parse: old: strict JSON; new: strict JSON
- added `Tags.Family[0]` = "Battleaxe"
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Adamantite.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:92`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Cobalt.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:98`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Copper.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:104`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 100 -> 200
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Crude.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:110`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Doomed.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:116`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Iron.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:62 (bare id Weapon_Battleaxe_Iron)`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:122`, `HytaleRL/hytalegym/tests/jax/combat/arsenal/tests/profiles/specials/continuation/test_profiles.py:76`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 120 -> 250
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Mithril.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:128`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Onyxium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:134`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `DurabilityLossOnHit`: 0.3 -> 0.18

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Stone_Trork.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:158`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Battleaxe/Weapon_Battleaxe_Thorium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:164`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Signature_Whirlwind_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.45 -> 0.27

### `Server/Item/Items/Weapon/Bomb/Weapon_Bomb.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:70 (bare id Weapon_Bomb)`

- parse: old: strict JSON; new: strict JSON
- added `Particles[0].Scale` = 2

### `Server/Item/Items/Weapon/Club/Weapon_Club_Scrap.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:248`

- parse: old: strict JSON; new: strict JSON
- changed `Icon`: "Icons/ItemsGenerated/Weapon_Club_Scrap_Goblin.png" -> "Icons/ItemsGenerated/Weapon_Club_Scrap.png"

### `Server/Item/Items/Weapon/Crossbow/Template_Weapon_Crossbow.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:921`, `HytaleRL/hytalegym/tests/jax/combat/arsenal/tests/profiles/test_profiles.py:1200`

- parse: old: strict JSON; new: strict JSON
- removed `Interactions.Ability3` = "Root_Common_StatAmmoReload_Entry"
- removed `InteractionVars.Guard_Bash.Interactions[1].Effects.WorldSoundEventId` = "SFX_Hand_Crossbow_T1_Shove"
- removed `InteractionVars.Guard_Bash.Interactions[1].Effects.LocalSoundEventId` = "SFX_Hand_Crossbow_T1_Shove_Local"
- removed `InteractionVars.Guard_Bash.Interactions[1].StatModifiers.Stamina` = -2
- added `Interactions.Ability4` = "Root_Common_StatAmmoReload_Entry"
- changed `InteractionVars.Combo_Projectile_Damage.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 9 -> 3.75
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.28 -> 0.17

### `Server/Item/Items/Weapon/Crossbow/Weapon_Crossbow_Ancient_Steel.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:326`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Standard_Projectile_Damage.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 16 -> 10.5
- changed `InteractionVars.Combo_Projectile_Damage.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 42 -> 14
- changed `InteractionVars.Signature_BigArrow_Damage.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 123 -> 82
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.28 -> 0.17

### `Server/Item/Items/Weapon/Crossbow/Weapon_Crossbow_Iron.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:65 (bare id Weapon_Crossbow_Iron)`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:332`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Combo_Projectile_Damage.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 27 -> 13
- changed `MaxDurability`: 120 -> 250
- changed `DurabilityLossOnHit`: 0.28 -> 0.17

### `Server/Item/Items/Weapon/Daggers/Template_Weapon_Daggers.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:356`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:362`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:398`

- parse: old: strict JSON; new: strict JSON
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Adamantite.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:338`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 6 -> 9
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 6 -> 9
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Adamantite_Saurian.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:344`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 6 -> 9
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 6 -> 9
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Bone.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:350`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Claw_Bone.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:368`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 3 -> 5
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 4 -> 6

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Cobalt.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:374`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Copper.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:380`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 3 -> 5
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 3 -> 5
- changed `MaxDurability`: 100 -> 200
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Crude.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:386`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 2 -> 3
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 2 -> 3
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Doomed.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:392`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Iron.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:63 (bare id Weapon_Daggers_Iron)`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:404`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 4 -> 6
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 4 -> 6
- changed `MaxDurability`: 120 -> 250
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Mithril.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:410`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 8 -> 12
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 8 -> 12
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Onyxium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:416`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 8 -> 12
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 8 -> 12
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Stone_Trork.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:422`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 4 -> 6
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 4 -> 6

### `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Thorium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:428`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Razorstrike_Slash_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Sweep_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Razorstrike_Lunge_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8
- changed `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical`: 5 -> 8
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.1 -> 0.06

### `Server/Item/Items/Weapon/Longsword/Weapon_Longsword_Katana.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:470`

- parse: old: strict JSON; new: strict JSON
- no field-level differences (bytes differ only in formatting or key order)

### `Server/Item/Items/Weapon/Longsword/Weapon_Longsword_Praetorian.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:488`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:494`

- parse: old: strict JSON; new: strict JSON
- changed `Texture`: "NPC/Undead/Skeleton_Giant/Models/Weapons/Sword/Praetorian_Texture.png" -> "NPC/Undead/Skeleton_Elite/Models/Weapons/Sword/Praetorian_Texture.png"
- changed `Model`: "NPC/Undead/Skeleton_Giant/Models/Weapons/Sword/Praetorian.blockymodel" -> "NPC/Undead/Skeleton_Elite/Models/Weapons/Sword/Praetorian.blockymodel"

### `Server/Item/Items/Weapon/Mace/Template_Weapon_Mace.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:578`

- parse: old: strict JSON; new: strict JSON
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Adamantite.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:536`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Cobalt.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:542`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Copper.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:548`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 100 -> 200
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Crude.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:554`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Iron.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:61 (bare id Weapon_Mace_Iron)`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:560`, `HytaleRL/hytalegym/tests/jax/combat/arsenal/tests/profiles/specials/continuation/test_groundslam.py:1` (+1 more)

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 120 -> 250
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Mithril.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:566`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Onyxium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:572`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Scrap.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:584`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:590`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Stone_Trork.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:596`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Mace/Weapon_Mace_Thorium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:602`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Groundslam_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.5 -> 0.3

### `Server/Item/Items/Weapon/Shortbow/Template_Weapon_Shortbow.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:614`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:644`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:650` (+1 more)

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Guard_Bash.Interactions[1].Effects.WorldSoundEventId` = "SFX_Bow_T1_Swing"
- removed `InteractionVars.Guard_Bash.Interactions[1].Effects.LocalSoundEventId` = "SFX_Bow_T1_Swing_Local"
- removed `InteractionVars.Guard_Bash.Interactions[1].StatModifiers.Stamina` = -2
- added `Interactions.SwapFrom.Interactions[0].Next.Next.Type` = "ChangeStat"
- added `Interactions.SwapFrom.Interactions[0].Next.Next.Behaviour` = "Set"
- added `Interactions.SwapFrom.Interactions[0].Next.Next.StatModifiers.StaminaRegenDelay` = -1
- added `Interactions.SwapFrom.Interactions[1].Costs.SignatureCharges` = 1
- added `Interactions.SwapFrom.Interactions[1].Next.Type` = "ChangeStat"
- added `Interactions.SwapFrom.Interactions[1].Next.Behaviour` = "Set"
- added `Interactions.SwapFrom.Interactions[1].Next.StatModifiers.StaminaRegenDelay` = -1
- added `Interactions.SwapFrom.Interactions[2].Type` = "ChangeActiveSlot"
- changed `Interactions.SwapFrom.Interactions[1].Type`: "ChangeActiveSlot" -> "StatsCondition"
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 6 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 10 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 12 -> 3
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 14 -> 5
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Adamantite.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:608`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 9 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 14 -> 3
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 18 -> 6
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 25 -> 10
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Cobalt.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:620`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 7 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 11 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 15 -> 5
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 20 -> 8
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Copper.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:626`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 5 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 7 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 10 -> 3
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 13 -> 5
- changed `MaxDurability`: 100 -> 200
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Crude.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:632`, `HytaleRL/hytalegym/hytalegym/jax/crafting/contract.py:20`, `HytaleRL/hytalegym/tests/fidelity/surrogate_crafting_source_audit.py:52`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 4 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 6 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 8 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 10 -> 4
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Doomed.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:638`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 7 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 12 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 16 -> 5
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 20 -> 8
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageEffects.Knockback.Force`: 4 -> 1

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Iron.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:64 (bare id Weapon_Shortbow_Iron)`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:656`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 6 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 9 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 12 -> 4
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 16 -> 7
- changed `MaxDurability`: 120 -> 250
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Iron_Rusty.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:662`

- parse: old: strict JSON; new: strict JSON
- changed `DurabilityLossOnHit`: 0.58 -> 0.35
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 6 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 8 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 11 -> 3
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 15 -> 6
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageEffects.Knockback.Force`: 4 -> 1

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Mithril.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:668`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 11 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 17 -> 4
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 23 -> 7
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 31 -> 13
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Onyxium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:674`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 11 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 17 -> 4
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 23 -> 7
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 31 -> 13
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageEffects.Knockback.Force`: 4 -> 1
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Thorium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:680`

- parse: old: strict JSON; new: strict JSON
- changed `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 7 -> 1
- changed `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 11 -> 2
- changed `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 15 -> 5
- changed `InteractionVars.Primary_Shoot_Damage_Strength_3.Interactions[0].DamageCalculator.BaseDamage.Projectile`: 20 -> 8
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.58 -> 0.35

### `Server/Item/Items/Weapon/Staff/Weapon_Staff_Bone.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:722`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Staff_Cast_Summon_Charged.Interactions[0].Parent` = "Staff_Cast_Summon_Charged"
- removed `InteractionVars.Staff_Cast_Summon_Charged.Interactions[0].Costs.Mana` = 50

### `Server/Item/Items/Weapon/Sword/Template_Weapon_Sword.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:182`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:752`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:758` (+10 more)

- parse: old: strict JSON; new: strict JSON
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Adamantite.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:740`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 180 -> 500
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Bone.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:746`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Cobalt.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:764`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Copper.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:770`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 100 -> 200
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Crude.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:776`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 80 -> 120
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Doomed.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:788`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Iron.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:60 (bare id Weapon_Sword_Iron)`, `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:800`, `HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:46`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 120 -> 250
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Mithril.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:806`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Onyxium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:818`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Scrap.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:830`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `Icon`: "Icons/ItemsGenerated/Weapon_Sword_Scrap_Goblin.png" -> "Icons/ItemsGenerated/Weapon_Sword_Scrap.png"
- changed `InteractionVars.Swing_Left_Selector.Interactions[0].Effects.Trails[0].PositionOffset.X`: 0.6 -> 0.8
- changed `InteractionVars.Swing_Left_Selector.Interactions[0].Effects.Trails[1].PositionOffset.X`: 0.6 -> 0.8
- changed `InteractionVars.Swing_Right_Selector.Interactions[0].Effects.Trails[1].PositionOffset.X`: 0.6 -> 0.8

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Stone_Trork.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:860`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []

### `Server/Item/Items/Weapon/Sword/Weapon_Sword_Thorium.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:866`

- parse: old: strict JSON; new: strict JSON
- removed `InteractionVars.Vortexstrike_Spin_Damage.Interactions[0].EntityStatsOnHit` = []
- removed `InteractionVars.Vortexstrike_Stab_Damage.Interactions[0].EntityStatsOnHit` = []
- changed `MaxDurability`: 150 -> 350
- changed `DurabilityLossOnHit`: 0.21 -> 0.13

### `Server/Item/RootInteractions/Weapons/Battleaxe/Root_Weapon_Battleaxe_Secondary_Guard.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/interaction_rules/test_authored_rule_surface.py:36`

- parse: old: strict JSON; new: strict JSON
- removed `Rules.Interrupting[0]` = "Primary"

### `Server/Item/RootInteractions/Weapons/Common/Root_Common_StatAmmoReload_Entry.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:923`

- parse: old: strict JSON; new: strict JSON
- added `HudInputBindingEntry` = "server.hud.inputBinding.reload"

### `Server/Item/RootInteractions/Weapons/Daggers/Root_Weapon_Daggers_Secondary_Guard.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/interaction_rules/test_authored_rule_surface.py:56`

- parse: old: strict JSON; new: strict JSON
- removed `Rules.Interrupting[0]` = "Primary"

### `Server/Item/RootInteractions/Weapons/Mace/Root_Weapon_Mace_Secondary_Guard.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/interaction_rules/test_authored_rule_surface.py:61`

- parse: old: strict JSON; new: strict JSON
- removed `Rules.Interrupting[0]` = "Primary"

### `Server/Item/RootInteractions/Weapons/Shortbow/Root_Weapon_Shortbow_Secondary_Guard.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/interaction_rules/test_authored_rule_surface.py:76`

- parse: old: strict JSON; new: strict JSON
- removed `Interactions[1].Var` = "Guard_Start"
- removed `Interactions[1].DefaultOk` = true
- removed `Interactions[1].DefaultValue.Interactions[0]` = "Weapon_Shortbow_Secondary_Guard"
- added `Interactions[0].Next.Next.Next.Type` = "ChangeStat"
- added `Interactions[0].Next.Next.Next.Behaviour` = "Set"
- added `Interactions[0].Next.Next.Next.StatModifiers.StaminaRegenDelay` = -1
- added `Interactions[1].Costs.SignatureCharges` = 1
- added `Interactions[1].Next.Type` = "ChangeStat"
- added `Interactions[1].Next.Behaviour` = "Set"
- added `Interactions[1].Next.StatModifiers.StaminaRegenDelay` = -1
- added `Interactions[2].Type` = "Replace"
- added `Interactions[2].Var` = "Guard_Start"
- added `Interactions[2].DefaultOk` = true
- added `Interactions[2].DefaultValue.Interactions[0]` = "Weapon_Shortbow_Secondary_Guard"
- changed `Interactions[1].Type`: "Replace" -> "StatsCondition"

### `Server/Item/RootInteractions/Weapons/Sword/Root_Weapon_Sword_Secondary_Guard.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/interaction_rules/test_authored_rule_surface.py:81`

- parse: old: strict JSON; new: strict JSON
- removed `Rules.Interrupting[0]` = "Primary"

### `Server/NPC/Balancing/Intelligent/CAE_Hedera.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_perception_source_audit.py:186`, `HytaleRL/hytalegym/tests/fidelity/surrogate_perception_source_audit.py:192`

- parse: old: strict JSON; new: strict JSON
- added `CombatActionEvaluator.AvailableActions.VineBlock.Conditions[2].Sources[0]` = "Combat"
- added `CombatActionEvaluator.AvailableActions.BackOff.Conditions[2].Sources[0]` = "Combat"

### `Server/NPC/Roles/_Core/Templates/Template_Swimming_Passive.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_locomotion_source_audit.py:108`

- parse: old: strict JSON; new: strict JSON
- added `Parameters.MemoriesUnlockedBy.Value` = "Proximity"
- added `Parameters.MemoriesUnlockedBy.Description` = "How the player records the memory, either Proximity or Kill."
- added `Parameters.MemoriesKillRange.Value` = 0
- added `Parameters.MemoriesKillRange.Description` = "Radius in blocks in which the memory is recorded when the NPC dies. 0 uses the server default."
- added `MemoriesUnlockedBy.Compute` = "MemoriesUnlockedBy"
- added `MemoriesKillRange.Compute` = "MemoriesKillRange"

### `Server/NPC/Roles/_Core/Tests/Flocking/Test_Flock_Damage.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_locomotion_source_audit.py:111`

- parse: old: strict JSON; new: strict JSON
- changed `NameTranslationKey`: "server.npcRoles.Test_Flock_Damage.name" -> "server_no_translation.npcRoles.Test_Flock_Damage.name"

### `Server/NPC/Roles/_Core/Tests/Flocking/Test_Flock_Pos.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_locomotion_source_audit.py:114`

- parse: old: strict JSON; new: strict JSON
- changed `NameTranslationKey`: "server.npcRoles.Test_Flock_Pos.name" -> "server_no_translation.npcRoles.Test_Flock_Pos.name"

### `Server/NPC/Roles/_Core/Tests/Motion/Test_Dive_Flee.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_locomotion_source_audit.py:117`

- parse: old: strict JSON; new: strict JSON
- changed `NameTranslationKey`: "server.npcRoles.Test_Dive_Flee.name" -> "server_no_translation.npcRoles.Test_Dive_Flee.name"

### `Server/NPC/Roles/_Core/Tests/Motion/Test_Walk_Leave.json`

Referenced by: `HytaleRL/hytalegym/tests/fidelity/surrogate_navigation_source_audit.py:100`

- parse: old: strict JSON; new: strict JSON
- changed `NameTranslationKey`: "server.npcRoles.Test_Walk_Leave.name" -> "server_no_translation.npcRoles.Test_Walk_Leave.name"

### `Server/ProjectileConfigs/Weapons/Arrows/Projectile_Config_Arrow_Crossbow.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/test_projectile_launch_offsets.py:45`

- parse: old: strict JSON; new: strict JSON
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Block_Break_Adventure" -> "Block_Break_Projectile"

### `Server/ProjectileConfigs/Weapons/Arrows/Projectile_Config_Arrow_Shortbow.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/test_projectile_launch_offsets.py:38`

- parse: old: strict JSON; new: strict JSON
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Block_Break_Adventure" -> "Block_Break_Projectile"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:125`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_01.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_02.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_03.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_04.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_05.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_06.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_07.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_08.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_09.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Charge_10.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:140`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Trishot.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:126`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Pull_Pullshot.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:127`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:131`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_01.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_02.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_03.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_04.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_05.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_06.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_07.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_08.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_09.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Charge_10.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:150`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Supershot.json`

Referenced by: `HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py:132`

- parse: old: strict JSON; new: strict JSON
- added `Interactions.ProjectileMiss.Interactions[2]` = "Bow_Combat_Projectile_Miss_End"
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Bow_Combat_Projectile_Miss" -> "Block_Break_Projectile"
- changed `Interactions.ProjectileMiss.Interactions[1]`: "Bow_Combat_Projectile_Miss_End" -> "Bow_Combat_Projectile_Miss"

### `Server/ProjectileConfigs/Weapons/Crossbow/Projectile_Config_Arrow_Crossbow_Signature.json`

Referenced by: `HytaleRL/hytalegym/tests/combat/assets/test_projectile_launch_offsets.py:52`

- parse: old: strict JSON; new: strict JSON
- changed `Interactions.ProjectileMiss.Interactions[0]`: "Block_Break_Adventure" -> "Block_Break_Projectile"

## Simulator-referenced files moved or missing

| 0.5 path (referenced by) | Same name in 0.7.2 | Where |
|---|---|---|
| `Server/HytaleGenerator/WorldStructures/Zone4_Volcanic1.json` (`HytaleRL/hytalegym/tests/fidelity/fixtures/world/native_environment_damage_d954_v1.json:72`) | yes | `Server/HytaleGenerator/WorldStructures/Regions/Zone4_Volcanic1.json` |
| `Server/NPC/Balancing/Intelligent/CAE_Goblin_Ogre.json` (`HytaleRL/hytalegym/tests/fidelity/surrogate_perception_source_audit.py:180`) | no | no file with this name |
| `Server/NPC/Balancing/Intelligent/CAE_Skeleton_Praetorian.json` (`HytaleRL/hytalegym/tests/fidelity/surrogate_perception_source_audit.py:198`) | yes | `Server/NPC/Balancing/Undead/CAE_Skeleton_Praetorian.json` |
| `Server/NPC/Roles/Intelligent/Aggressive/Trork/Templates/Template_Trork_Melee.json` (`HytaleRL/hytalegym/tests/fidelity/surrogate_navigation_source_audit.py:83`) | yes | `Server/NPC/Roles/Intelligent/Faction/Trork/Templates/Template_Trork_Melee.json` |
| `Server/NPC/Roles/Intelligent/Aggressive/Trork/Trork_Brawler.json` (`HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/role_initial_statuses_v1.json:45`) | yes | `Server/Models/Intelligent/Trork/Trork_Brawler.json`, `Server/NPC/Roles/Intelligent/Faction/Trork/Trork_Brawler.json`, `Server/NPC/Spawn/Markers/Intelligent/Trork/Trork_Brawler.json` |
| `Server/NPC/Roles/Intelligent/Neutral/Kweebec/Templates/Template_Kweebec_Razorleaf.json` (`HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/kweebec_razorleaf_movement_projection_v1.json:17`) | yes | `Server/NPC/Roles/Intelligent/Faction/Kweebec/Templates/Template_Kweebec_Razorleaf.json` |
| `Server/NPC/Roles/_Core/Components/Steps/Component_Instruction_Intelligent_Chase.json` (`HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json:324`) | yes | `Server/NPC/Roles/_Core/Components/Instructions/Combat/Component_Instruction_Intelligent_Chase.json` |
| `Server/NPC/Roles/_Core/Components/Steps/Component_Instruction_Target_Adjusted_Attack_Melee.json` (`HytaleRL/hytalegym/tests/fidelity/surrogate_navigation_source_audit.py:91`) | yes | `Server/NPC/Roles/_Core/Components/Instructions/Combat/Component_Instruction_Target_Adjusted_Attack_Melee.json` |

Referenced paths absent from the 0.5 zip (62; listed in the report, not diffed):

- `Server/HytaleGenerator/Assignments/Test.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:1328`)
- `Server/HytaleGenerator/Assignments/Test_B.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:159`)
- `Server/HytaleGenerator/Biomes/Test/Test.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:288`)
- `Server/HytaleGenerator/WorldStructures/Test.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:247`)
- `Server/HytaleServer.jar` (`HytaleRL/hytalegym/tests/worldgen/test_world_contract_artifact_references.py:605`)
- `Server/Item/Block/Hitboxes/Test/Closed.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:135`)
- `Server/Item/Block/Hitboxes/Test/Open.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:141`)
- `Server/Item/Block/Hitboxes/Test/OpenIn.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:138`)
- `Server/Item/Block/Hitboxes/Test/OpenOut.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:141`)
- `Server/Item/Block/Hitboxes/Test/Slab.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_palette.py:55`)
- `Server/Item/Block/Hitboxes/Test/Thin.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:133`)
- `Server/Item/Interactions/Synthetic_Damage.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:174`)
- `Server/Item/Interactions/Test/Attack.json` (`HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:62`)
- `Server/Item/Interactions/Test/Damage_Default.json` (`HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:70`)
- `Server/Item/Items/Base/Base.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:44`)
- `Server/Item/Items/Legacy/Weapon_Unique.json` (`HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:192`)
- `Server/Item/Items/Other/Cube.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:401`)
- `Server/Item/Items/Test/A.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:399`)
- `Server/Item/Items/Test/B.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:400`)
- `Server/Item/Items/Test/Child.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:61`)
- `Server/Item/Items/Test/ComplexUse.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:127`)
- `Server/Item/Items/Test/Connected.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:72`)
- `Server/Item/Items/Test/ConnectedDoor.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:113`)
- `Server/Item/Items/Test/Cube.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:71`)
- `Server/Item/Items/Test/Door.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:97`)
- `Server/Item/Items/Test/DropChild.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:211`)
- `Server/Item/Items/Test/DropParent.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:200`)
- `Server/Item/Items/Test/Glass.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_palette.py:36`)
- `Server/Item/Items/Test/Grass.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:96`)
- `Server/Item/Items/Test/Ingredient.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:234`)
- `Server/Item/Items/Test/Lantern.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:115`)
- `Server/Item/Items/Test/LiteralRoof.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:89`)
- `Server/Item/Items/Test/LiteralStair.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:79`)
- `Server/Item/Items/Test/RedirectedStair.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:104`)
- `Server/Item/Items/Test/Rock.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_palette.py:35`)
- `Server/Item/Items/Test/Rock_Stone.json` (`HytaleRL/hytalegym/tests/worldgen/test_native_action_resolution.py:702`)
- `Server/Item/Items/Test/Shard.json` (`HytaleRL/hytalegym/tests/worldgen/test_region_block_action_catalog.py:69`)
- `Server/Item/Items/Test/Slab.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_palette.py:42`)
- `Server/Item/Items/Test/Stateful.json` (`HytaleRL/hytalegym/tests/worldgen/test_native_action_resolution.py:706`)
- `Server/Item/Items/Test/Stone.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:95`)
- `Server/Item/Items/Test/UpdatedLantern.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_semantics.py:347`)
- `Server/Item/Items/Weapon/Fake/Weapon_Wand_Stoneskin.json` (`HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:127`)
- `Server/Item/RootInteractions/Test/Root_Test.json` (`HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:59`)
- `Server/Item/RootInteractions/Test/Root_World.json` (`HytaleRL/hytalegym/tests/combat/assets/test_catalog.py:201`)
- `Server/Item/Unarmed/Gathering/FutureOre.json` (`HytaleRL/hytalegym/tests/worldgen/test_region_block_action_catalog.py:219`)
- `Server/NPC/Balancing/CAE_Base.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:161`)
- `Server/NPC/Balancing/CAE_Trork.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:158`)
- `Server/NPC/Balancing/Default_Kweebec.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:167`)
- `Server/NPC/Roles/Component_Kweebec_CAE.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:140`)
- `Server/NPC/Roles/Kweebec_Razorleaf.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:133`)
- `Server/NPC/Roles/Other/Trork.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_spawn_markers.py:84`)
- `Server/NPC/Roles/Template_Kweebec_Razorleaf.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:137`)
- `Server/NPC/Roles/Template_Trork_Melee.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:150`)
- `Server/NPC/Roles/Test/Trork.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:217`)
- `Server/NPC/Roles/Trork_Brawler.json` (`HytaleRL/hytalegym/tests/test_asset_surface.py:147`)
- `Server/NPC/Spawn/Markers/Test/Encounter.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:207`)
- `Server/Prefabs/Door.prefab.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:322`)
- `Server/Prefabs/Test.prefab.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_palette.py:86`)
- `Server/Prefabs/Test/Alt/Alt.prefab.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:1566`)
- `Server/Prefabs/Test/Doors/Door.prefab.json` (`HytaleRL/hytalegym/tests/jax/world/surrogate/test_atlas.py:1567`)
- `Server/Prefabs/Test/Entities.prefab.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_entities.py:27`)
- `Server/World/Test/Entities.prefab.json` (`HytaleRL/hytalegym/tests/worldgen/surrogate/test_entities.py:28`)

Bare ids with no same-stem file in the 0.5 zip (not assets, or not under Server/): `Adamantite_Saurian`, `Stone_Trork`

## Unreferenced but likely relevant changes

Changed files under the focus folders that the simulator does not reference:

| Folder | Changed | Unreferenced changed | Added | Removed |
|---|---:|---:|---:|---:|
| `Server/Item/Interactions/` | 204 | 195 | 399 | 82 |
| `Server/Entity/MovementConfig/` | 2 | 1 | 2 | 0 |
| `Server/Entity/Stats/` | 2 | 1 | 0 | 0 |
| `Server/NPC/Roles/` | 369 | 364 | 455 | 282 |

- `Server/Entity/MovementConfig/Mount.json` (0 fields)
- `Server/Entity/Stats/Mana.json` (3 fields)
- `Server/Item/Interactions/Block/Block_Damage.json` (0 fields)
- `Server/Item/Interactions/Block/Environmental_Block_Damage.json` (0 fields)
- `Server/Item/Interactions/Block/Half_Block.json` (0 fields)
- `Server/Item/Interactions/Block_Secondary.json` (1 fields)
- `Server/Item/Interactions/Consumables/Condition_Consume_Potion_Stamina.json` (1 fields)
- `Server/Item/Interactions/Consumables/Consume_Charge.json` (2 fields)
- `Server/Item/Interactions/Consumables/Consume_Charge_Durability.json` (1 fields)
- `Server/Item/Interactions/Consumables/Consume_Charge_Food/Consume_Charge_Food_T1_Inner.json` (2 fields)
- `Server/Item/Interactions/Consumables/Consume_Charge_Food/Consume_Charge_Food_T2_Inner.json` (2 fields)
- `Server/Item/Interactions/Consumables/Consume_Charge_Food/Consume_Charge_Food_T3_Inner.json` (2 fields)
- `Server/Item/Interactions/Consumables/Consume_Charge_Potion_Slow.json` (2 fields)
- `Server/Item/Interactions/Consumables/Food/HealthRegen_TierCheck_T1.json` (2 fields)
- `Server/Item/Interactions/Consumables/Food/HealthRegen_TierCheck_T2.json` (3 fields)
- `Server/Item/Interactions/Consumables/Food/HealthRegen_TierCheck_T3.json` (4 fields)
- `Server/Item/Interactions/Crops/Seed_Place.json` (0 fields)
- `Server/Item/Interactions/Explosions/Explode_Generic.json` (0 fields)
- `Server/Item/Interactions/NPCs/Beast/Rat/Rat_Bite.json` (0 fields)
- `Server/Item/Interactions/NPCs/Beast/Spider/Spider_Bite.json` (0 fields)
- `Server/Item/Interactions/NPCs/NPC_Attack_Selector_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Reptile/Toad_Rhino_Magma/Toad_Rhino_Magma_Headbutt.json` (12 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Burnt_Praetorian/Skeleton_Burnt_Praetorian_Charge.json` (8 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Burnt_Praetorian/Skeleton_Burnt_Praetorian_Charge_Damage.json` (8 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Fighter/Skeleton_Fighter_Club_Swing_Down.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Fighter/Skeleton_Fighter_Club_Swing_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Fighter/Skeleton_Fighter_Club_Swing_Right.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Fighter/Skeleton_Fighter_Club_Swing_Up_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Knight/Skeleton_Knight_Shield_Block.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Knight/Skeleton_Knight_Sword_Swing_Down.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Knight/Skeleton_Knight_Sword_Swing_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Knight/Skeleton_Knight_Sword_Swing_Right.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Soldier/Skeleton_Soldier_Axe_Swing_Down.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Soldier/Skeleton_Soldier_Axe_Swing_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Soldier/Skeleton_Soldier_Axe_Swing_Right.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Skeleton_Soldier/Skeleton_Soldier_Axe_Swing_Up_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Zombie/Zombie_Bite.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Zombie/Zombie_Swing_Double.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Zombie/Zombie_Swing_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Undead/Zombie/Zombie_Swing_Right.json` (0 fields)
- `Server/Item/Interactions/NPCs/Void/Crawler_Void/Crawler_Void_Bite.json` (0 fields)
- `Server/Item/Interactions/NPCs/Void/Crawler_Void/Crawler_Void_Scratch_Left.json` (0 fields)
- `Server/Item/Interactions/NPCs/Void/Crawler_Void/Crawler_Void_Scratch_Right.json` (0 fields)
- `Server/Item/Interactions/NPCs/Void/Larva_Void/Larva_Void_Bite.json` (0 fields)
- `Server/Item/Interactions/Tests/Test_Charge_Damage.json` (0 fields)
- `Server/Item/Interactions/Tools/Watering_Can_Use.json` (0 fields)
- `Server/Item/Interactions/Tools/Watering_Can_Use_3x3.json` (0 fields)
- `Server/Item/Interactions/Tools/Watering_Can_Use_Charge.json` (0 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Downstrike/Weapon_Battleaxe_Primary_Downstrike.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Downstrike/Weapon_Battleaxe_Primary_Downstrike_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Downstrike/Weapon_Battleaxe_Primary_Downstrike_Force.json` (0 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Downstrike/Weapon_Battleaxe_Primary_Downstrike_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down/Weapon_Battleaxe_Primary_Swing_Down.json` (0 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down/Weapon_Battleaxe_Primary_Swing_Down_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down/Weapon_Battleaxe_Primary_Swing_Down_Selector.json` (0 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Left/Weapon_Battleaxe_Primary_Swing_Down_Left.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Left/Weapon_Battleaxe_Primary_Swing_Down_Left_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Left/Weapon_Battleaxe_Primary_Swing_Down_Left_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Right/Weapon_Battleaxe_Primary_Swing_Down_Right.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Right/Weapon_Battleaxe_Primary_Swing_Down_Right_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Right/Weapon_Battleaxe_Primary_Swing_Down_Right_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Primary/Weapon_Battleaxe_Primary.json` (0 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Secondary/Guard/Guard_Bash/Weapon_Battleaxe_Secondary_Guard_Bash.json` (3 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Secondary/Guard/Guard_Bash/Weapon_Battleaxe_Secondary_Guard_Bash_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Secondary/Guard/Weapon_Battleaxe_Secondary_Guard.json` (0 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Secondary/Guard/Weapon_Battleaxe_Secondary_Guard_Wield.json` (2 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Signature/Whirlwind/Weapon_Battleaxe_Signature_Whirlwind_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Signature/Whirlwind/Weapon_Battleaxe_Signature_Whirlwind_End_Selector.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Signature/Whirlwind/Weapon_Battleaxe_Signature_Whirlwind_Spin_Effect.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Signature/Whirlwind/Weapon_Battleaxe_Signature_Whirlwind_Spin_Selector.json` (1 fields)
- `Server/Item/Interactions/Weapons/Battleaxe/Signature/Whirlwind/Weapon_Battleaxe_Signature_Whirlwind_Windup.json` (1 fields)
- `Server/Item/Interactions/Weapons/Block_Break_Adventure.json` (0 fields)
- `Server/Item/Interactions/Weapons/Common/Guarding/Common_Guard_Wield.json` (0 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Damage/Weapon_Crossbow_Damage_Signature_Projectile.json` (0 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Primary/Overcharge/Weapon_Crossbow_Primary_Overcharge_Shoot.json` (1 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Primary/Weapon_Crossbow_Primary_Overcharge_Start.json` (2 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Primary/Weapon_Crossbow_Primary_Reload.json` (1 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Secondary/Guard/Guard_Bash/Weapon_Crossbow_Secondary_Guard_Bash.json` (2 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Secondary/Guard/Weapon_Crossbow_Secondary_Guard.json` (0 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Secondary/Guard/Weapon_Crossbow_Secondary_Guard_Wield.json` (2 fields)
- `Server/Item/Interactions/Weapons/Crossbow/Shots/Weapon_Crossbow_Shot_Standard_Projectile.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Deprecated/Attacks/Pounce/Daggers_Pounce_Stab.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Deprecated/Attacks/Stab_Left/Daggers_Stab_Left.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Deprecated/Attacks/Stab_Right/Daggers_Stab_Right.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Deprecated/Attacks/Swing_Left/Daggers_Swing_Left.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Deprecated/Attacks/Swing_Right/Daggers_Swing_Right.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Stab/Weapon_Daggers_Primary_Pounce_Stab.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Stab/Weapon_Daggers_Primary_Pounce_Stab_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Stab/Weapon_Daggers_Primary_Pounce_Stab_Force.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Stab/Weapon_Daggers_Primary_Pounce_Stab_Selector.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Sweep/Weapon_Daggers_Primary_Pounce_Sweep.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Sweep/Weapon_Daggers_Primary_Pounce_Sweep_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Sweep/Weapon_Daggers_Primary_Pounce_Sweep_Selector.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Pounce/Weapon_Daggers_Primary_Pounce.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Left/Weapon_Daggers_Primary_Stab_Left.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Left/Weapon_Daggers_Primary_Stab_Left_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Left/Weapon_Daggers_Primary_Stab_Left_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Right/Weapon_Daggers_Primary_Stab_Right.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Right/Weapon_Daggers_Primary_Stab_Right_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Right/Weapon_Daggers_Primary_Stab_Right_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Left/Weapon_Daggers_Primary_Swing_Left.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Left/Weapon_Daggers_Primary_Swing_Left_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Left/Weapon_Daggers_Primary_Swing_Left_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Right/Weapon_Daggers_Primary_Swing_Right.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Right/Weapon_Daggers_Primary_Swing_Right_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Right/Weapon_Daggers_Primary_Swing_Right_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Primary/Weapon_Daggers_Primary.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Secondary/Guard/Guard_Bash/Weapon_Daggers_Secondary_Guard_Bash.json` (3 fields)
- `Server/Item/Interactions/Weapons/Daggers/Secondary/Guard/Guard_Bash/Weapon_Daggers_Secondary_Guard_Bash_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Secondary/Guard/Weapon_Daggers_Secondary_Guard.json` (0 fields)
- `Server/Item/Interactions/Weapons/Daggers/Secondary/Guard/Weapon_Daggers_Secondary_Guard_Wield.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Lunge/Weapon_Daggers_Signature_Razorstrike_Lunge.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Lunge/Weapon_Daggers_Signature_Razorstrike_Lunge_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Lunge/Weapon_Daggers_Signature_Razorstrike_Lunge_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Slash/Weapon_Daggers_Signature_Razorstrike_Slash.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Slash/Weapon_Daggers_Signature_Razorstrike_Slash_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Slash/Weapon_Daggers_Signature_Razorstrike_Slash_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Sweep/Weapon_Daggers_Signature_Razorstrike_Sweep.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Sweep/Weapon_Daggers_Signature_Razorstrike_Sweep_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Daggers/Signature/Razorstrike/Sweep/Weapon_Daggers_Signature_Razorstrike_Sweep_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Hatchet/Attacks/Chop/Hatchet_Chop.json` (0 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left/Weapon_Mace_Primary_Swing_Left.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left/Weapon_Mace_Primary_Swing_Left_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left/Weapon_Mace_Primary_Swing_Left_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left_Charged/Weapon_Mace_Primary_Swing_Left_Charged.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left_Charged/Weapon_Mace_Primary_Swing_Left_Charged_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left_Charged/Weapon_Mace_Primary_Swing_Left_Charged_Selector.json` (3 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Left_Charged/Weapon_Mace_Primary_Swing_Left_Charged_StaminaCondition.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right/Weapon_Mace_Primary_Swing_Right.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right/Weapon_Mace_Primary_Swing_Right_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right/Weapon_Mace_Primary_Swing_Right_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right_Charged/Weapon_Mace_Primary_Swing_Right_Charged.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right_Charged/Weapon_Mace_Primary_Swing_Right_Charged_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right_Charged/Weapon_Mace_Primary_Swing_Right_Charged_Selector.json` (3 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Right_Charged/Weapon_Mace_Primary_Swing_Right_Charged_StaminaCondition.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left/Weapon_Mace_Primary_Swing_Up_Left.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left/Weapon_Mace_Primary_Swing_Up_Left_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left/Weapon_Mace_Primary_Swing_Up_Left_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left_Charged/Weapon_Mace_Primary_Swing_Up_Left_Charged.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left_Charged/Weapon_Mace_Primary_Swing_Up_Left_Charged_Damage.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left_Charged/Weapon_Mace_Primary_Swing_Up_Left_Charged_Selector.json` (3 fields)
- `Server/Item/Interactions/Weapons/Mace/Primary/Swing_Up_Left_Charged/Weapon_Mace_Primary_Swing_Up_Left_Charged_StaminaCondition.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Secondary/Guard/Guard_Bash/Weapon_Mace_Secondary_Guard_Bash.json` (3 fields)
- `Server/Item/Interactions/Weapons/Mace/Secondary/Guard/Guard_Bash/Weapon_Mace_Secondary_Guard_Bash_Selector.json` (2 fields)
- `Server/Item/Interactions/Weapons/Mace/Secondary/Guard/Weapon_Mace_Secondary_Guard.json` (0 fields)
- `Server/Item/Interactions/Weapons/Mace/Secondary/Guard/Weapon_Mace_Secondary_Guard_Wield.json` (2 fields)
- `Server/Item/Interactions/Weapons/Mace/Signature/Groundslam/Weapon_Mace_Signature_Groundslam.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Signature/Groundslam/Weapon_Mace_Signature_Groundslam_Damage.json` (3 fields)
- `Server/Item/Interactions/Weapons/Mace/Signature/Groundslam/Weapon_Mace_Signature_Groundslam_Explosion.json` (1 fields)
- `Server/Item/Interactions/Weapons/Mace/Signature/Groundslam/Weapon_Mace_Signature_Groundslam_Swing_Selector.json` (1 fields)
- ... 411 more

## Jar class changes

Classes: old 37,150, new 38,436; added 1,875, removed 589, changed 4,813.

Changed classes whose package path contains one of: movement, physics, motion, collision, damage, knockback, interaction, combat, stat, mana, stamina, sight, raycast, npc, role, crafting, block (case-insensitive substring; `stat` also matches `state`/`static`). 951 classes in 166 packages.

- `com/hypixel/hytale/server/core/modules/entity/damage`: 63 changed
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$1.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$CameraEffect.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$CommandSource.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$EntitySource.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$EnvironmentSource.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$Particles.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$ProjectileSource.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$SoundEffect.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage$Source.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/Damage.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageCalculatorSystems$Sequence.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageCalculatorSystems.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageCause.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageModule$OrderGatherFilter.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageModule.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ApplyDamage.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ApplyParticles.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ApplySoundEffects.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ArmorDamageReduction$ArmorResistanceModifiers.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ArmorDamageReduction.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ArmorKnockbackReduction.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$BreathingSystem.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$CanBreathe.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$DamageArmor.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$DamageAttackerTool.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$DamageStamina.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$EntityUIEvents.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$FallDamageNPCs.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$FallDamagePlayers.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$FilterNPCWorldConfig.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$FilterPlayerWorldConfig.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$FilterUnkillable.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$HackKnockbackValues.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$HitAnimation.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$InvulnerableBreathing.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$OutOfWorldDamage.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$PlayerDamageFilterSystem.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$PlayerHitIndicators.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$RecordLastCombat.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$ReticleEvents.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$TrackLastDamage.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$WieldingDamageReduction.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems$WieldingKnockbackReduction.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DamageSystems.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathComponent.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$1.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$ClearEntityEffects.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$ClearHealth.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$ClearInteractions.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$CorpseRemoval.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$DeathAnimation.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$DropPlayerDeathItems.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$KillFeed.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$OnDeathSystem.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$PlayerDeathMarker.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$PlayerDeathScreen.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$PlayerDropItemsConfig.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$PlayerKilledPlayer.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$RunDeathInteractions.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$SpawnedDeathAnimation.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$StopVoiceOnDeath.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems$TickCorpseRemoval.class`
  - `com/hypixel/hytale/server/core/modules/entity/damage/DeathSystems.class`
- `com/hypixel/hytale/server/npc/systems`: 55 changed
  - `com/hypixel/hytale/server/npc/systems/AvoidanceSystem.class`
  - `com/hypixel/hytale/server/npc/systems/BalancingInitialisationSystem.class`
  - `com/hypixel/hytale/server/npc/systems/BlackboardSystems$BreakBlockEventSystem.class`
  - `com/hypixel/hytale/server/npc/systems/BlackboardSystems$DamageBlockEventSystem.class`
  - `com/hypixel/hytale/server/npc/systems/BlackboardSystems$InitSystem.class`
  - `com/hypixel/hytale/server/npc/systems/BlackboardSystems$TickingSystem.class`
  - `com/hypixel/hytale/server/npc/systems/BlackboardSystems.class`
  - `com/hypixel/hytale/server/npc/systems/MovementStatesSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDamageSystems$DamageDealtSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDamageSystems$DamageReceivedEventViewSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDamageSystems$DamageReceivedSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDamageSystems$DropDeathItems.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDamageSystems$FilterDamageSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDamageSystems.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDeathSystems$EntityViewSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDeathSystems$NPCKillsEntitySystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCDeathSystems.class`
  - `com/hypixel/hytale/server/npc/systems/NPCPreTickSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$1.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$AddSpawnEntityEffectSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$AddedFromExternalSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$AddedFromWorldGenSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$AddedSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$BreathingCheckEventSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$KillFeedDecedentEventSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$KillFeedKillerEventSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$LegacyWorldGenId.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$ModelChangeSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$OnDeathSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$OnNPCAdded.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$OnTeleportSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCSystems$PrefabPlaceEntityEventSystem.class`
  - `com/hypixel/hytale/server/npc/systems/NPCVelocityInstructionSystem.class`
  - `com/hypixel/hytale/server/npc/systems/PositionCacheSystems$OnFlockJoinSystem.class`
  - `com/hypixel/hytale/server/npc/systems/PositionCacheSystems$RoleActivateSystem.class`
  - `com/hypixel/hytale/server/npc/systems/PositionCacheSystems$UpdateSystem.class`
  - `com/hypixel/hytale/server/npc/systems/PositionCacheSystems.class`
  - `com/hypixel/hytale/server/npc/systems/RoleBuilderSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleChangeSystem$RoleChangeQueue.class`
  - `com/hypixel/hytale/server/npc/systems/RoleChangeSystem$RoleChangeRequest.class`
  - `com/hypixel/hytale/server/npc/systems/RoleChangeSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleSystems$BehaviourTickSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleSystems$PostBehaviourSupportTickSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleSystems$PreBehaviourSupportTickSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleSystems$RoleActivateSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleSystems$RoleDebugSystem.class`
  - `com/hypixel/hytale/server/npc/systems/RoleSystems.class`
  - `com/hypixel/hytale/server/npc/systems/SpawnReferenceSystems$1.class`
  - `com/hypixel/hytale/server/npc/systems/SpawnReferenceSystems$BeaconAddRemoveSystem.class`
  - `com/hypixel/hytale/server/npc/systems/SpawnReferenceSystems$MarkerAddRemoveSystem.class`
  - `com/hypixel/hytale/server/npc/systems/SpawnReferenceSystems$TickingSpawnBeaconSystem.class`
  - `com/hypixel/hytale/server/npc/systems/SpawnReferenceSystems$TickingSpawnMarkerSystem.class`
  - `com/hypixel/hytale/server/npc/systems/SpawnReferenceSystems.class`
  - `com/hypixel/hytale/server/npc/systems/StateEvaluatorSystem.class`
  - `com/hypixel/hytale/server/npc/systems/SteeringSystem.class`
- `com/hypixel/hytale/server/npc/commands`: 32 changed
  - `com/hypixel/hytale/server/npc/commands/NPCAllCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCAttackCommand$ClearAttackCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCAttackCommand$SetAttackCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCAttackCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBenchmarkCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$BlockEventsCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ChunkCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ChunksCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$DropCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$EntityEventsCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ReservationCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ReserveCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ResourceViewCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ResourceViewsCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ViewCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand$ViewsCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCBlackboardCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCCleanCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCCommand$1.class`
  - `com/hypixel/hytale/server/npc/commands/NPCCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCDebugCommand$SetCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCDebugCommand$ToggleCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCDumpCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCMessageCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCMultiSelectCommandBase.class`
  - `com/hypixel/hytale/server/npc/commands/NPCPathCommand$SetPathCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCRunTestsCommand$NPCTestData.class`
  - `com/hypixel/hytale/server/npc/commands/NPCRunTestsCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCSensorStatsCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCSpawnCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCTestCommand$ProbeTestCommand.class`
  - `com/hypixel/hytale/server/npc/commands/NPCTestCommand.class`
- `com/hypixel/hytale/server/npc/corecomponents/world`: 26 changed
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionMakePath.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionPlaceBlock.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionResetBlockSensors.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionResetPath.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionResetSearchRays.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionSetBlockToPlace.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionSetLeashPosition.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionStorePosition.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/ActionTriggerSpawners.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/BodyMotionPath.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/HeadMotionObserve.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorBlock.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorBlockType.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorCanPlace.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorEntityEvent$1.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorEntityEvent.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorEvent$EventSearchType.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorEvent.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorInWater.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorLeash.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorLight.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorPath.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorReadPosition.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorSearchRay.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorTime.class`
  - `com/hypixel/hytale/server/npc/corecomponents/world/SensorWeather.class`
- `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client`: 24 changed
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ApplyForceInteraction$1.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ApplyForceInteraction$Force.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ApplyForceInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/BlockConditionInteraction$BlockIdMatcher.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/BlockConditionInteraction$BlockMatcher.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/BlockConditionInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/BreakBlockInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChainingInteraction$ChainingTag.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChainingInteraction$Data.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChainingInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChangeBlockInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChangeStateInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChargingInteraction$ChargingTag.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ChargingInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/CycleBlockGroupInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/DestroyBlockInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ExplodeInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/PlaceBlockInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/PlaceFluidInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/ResetCooldownInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/SimpleBlockInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/TriggerCooldownInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/UseBlockInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/client/UseEntityInteraction.class`
- `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server`: 22 changed
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/ChangeStatBaseInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/ChangeStatInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/ChangeStatWithModifierInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DamageEntityInteraction$AngledDamage.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DamageEntityInteraction$AngledDamageTag.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DamageEntityInteraction$EntityStatOnHit.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DamageEntityInteraction$TargetedDamage.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DamageEntityInteraction$TargetedDamageTag.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DamageEntityInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DoorInteraction$DoorInfo.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/DoorInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/LaunchPadInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/LaunchProjectileInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/ModifyInventoryInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/OpenContainerInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/OpenCustomUIInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/RefillContainerInteraction$RefillState.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/RefillContainerInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/RunOnBlockTypesInteraction$1.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/RunOnBlockTypesInteraction$BlockSearchConsumer.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/RunOnBlockTypesInteraction.class`
  - `com/hypixel/hytale/server/core/modules/interaction/interaction/config/server/SpawnPrefabInteraction.class`
- `com/hypixel/hytale/server/npc/corecomponents/movement`: 21 changed
  - `com/hypixel/hytale/server/npc/corecomponents/movement/ActionCrouch.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/ActionOverrideAltitude.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/ActionRecomputePath.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionFind.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionFindBase$DebugFlags.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionFindBase.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionFindWithTarget.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionLand.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionMaintainDistance.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionMatchLook.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionMoveAway.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionTakeOff.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionTeleport.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionWander.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionWanderBase.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionWanderInCircle.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/BodyMotionWanderInRect.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/SensorInAir.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/SensorMotionController.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/SensorNav.class`
  - `com/hypixel/hytale/server/npc/corecomponents/movement/SensorOnGround.class`
- `com/hypixel/hytale/server/core/asset/type/blocktype/config`: 19 changed
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockGathering.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockMovementSettings.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$1.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$2.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$3.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$4.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$5.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$6.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType$7.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockType.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/BlockTypeTextures.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/CustomModelTexture.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/MergedBlockFaces.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/Rotation$1.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/Rotation.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/RotationTuple$1.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/RotationTuple.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/SoftBlockDropType.class`
  - `com/hypixel/hytale/server/core/asset/type/blocktype/config/StateData.class`
- `com/hypixel/hytale/server/npc/asset/builder`: 19 changed
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderBase.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderFactory.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderManager$1.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderManager$2.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderManager$BuilderAssetMonitorHandler.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderManager$TestType.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderManager.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderModifier$ExpressionHolder.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderModifier$SchemaGenerator.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderModifier.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderObjectMapHelper.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderObjectReferenceHelper.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderParameters$Parameter.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderParameters.class`
  - `com/hypixel/hytale/server/npc/asset/builder/BuilderSupport.class`
  - `com/hypixel/hytale/server/npc/asset/builder/EmptyBuilderModifier.class`
  - `com/hypixel/hytale/server/npc/asset/builder/InstructionContextHelper.class`
  - `com/hypixel/hytale/server/npc/asset/builder/InstructionType.class`
  - `com/hypixel/hytale/server/npc/asset/builder/InternalReferenceResolver.class`
- `com/hypixel/hytale/server/npc/corecomponents/entity/filters`: 19 changed
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterAltitude.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterAnd.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterAttitude.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterCombat.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterEntityEffect.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterHeightDifference.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterInsideBlock.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterInventory.class`
  - `com/hypixel/hytale/server/npc/corecomponents/entity/filters/EntityFilterItemInHand.class`
- ... output capped at 300 lines

Added classes (first 60 of 1,875): `com/hypixel/hytale/builtin/abilities/AbilitiesPlugin.class`, `com/hypixel/hytale/builtin/abilities/command/AbilitiesCommand$AbilitiesGiveCommand.class`, `com/hypixel/hytale/builtin/abilities/command/AbilitiesCommand$AbilitiesResetIntroCommand.class`, `com/hypixel/hytale/builtin/abilities/command/AbilitiesCommand.class`, `com/hypixel/hytale/builtin/abilities/component/AbilityBenchIntro.class`, `com/hypixel/hytale/builtin/abilities/element/Element$Reaction.class`, `com/hypixel/hytale/builtin/abilities/element/Element.class`, `com/hypixel/hytale/builtin/abilities/element/ReactionDamageSystem.class`, `com/hypixel/hytale/builtin/abilities/entity/AbilityEntityRepeat.class`, `com/hypixel/hytale/builtin/abilities/entity/AbilityEntityRepeatSystem.class`, `com/hypixel/hytale/builtin/abilities/entity/AbilityEntitySpawner.class`, `com/hypixel/hytale/builtin/abilities/interaction/ApplyAbilityEffectsInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/GrantAbilityChargesInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/LaunchAbilityProjectileInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/MarkAbilityBenchIntroSeenInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/OpenAbilityBenchInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/PullEntityAbilityInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/SpawnAbilityEntityInteraction$RepeatConfig.class`, `com/hypixel/hytale/builtin/abilities/interaction/SpawnAbilityEntityInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/SplitAbilityProjectileInteraction$1.class`, `com/hypixel/hytale/builtin/abilities/interaction/SplitAbilityProjectileInteraction.class`, `com/hypixel/hytale/builtin/abilities/interaction/TriggerAbilityFXInteraction.class`, `com/hypixel/hytale/builtin/abilities/keyword/AbilityKeywordAudit$ChainSummary.class`, `com/hypixel/hytale/builtin/abilities/keyword/AbilityKeywordAudit.class`, `com/hypixel/hytale/builtin/abilities/page/AbilityBenchIntroPage$PageAction.class`, `com/hypixel/hytale/builtin/abilities/page/AbilityBenchIntroPage$PageEventData.class`, `com/hypixel/hytale/builtin/abilities/page/AbilityBenchIntroPage.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityChargeActivator.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityChargeHitSystem.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityChargeMeleeAttackSystem.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityChargeRangedAttackSystem.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityChargeTrigger.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityCharges$Entry.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityCharges.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityProjectileConfig.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityProjectileLaunchData.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityProjectileLauncher.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityProjectileSplit.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityProjectileTrail.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityProjectileTrailSystem.class`, `com/hypixel/hytale/builtin/abilities/trigger/AbilityPullSystem.class`, `com/hypixel/hytale/builtin/abilities/trigger/PulledByAbility.class`, `com/hypixel/hytale/builtin/abilities/trigger/RuneEffectDamageSystem.class`, `com/hypixel/hytale/builtin/abilities/window/AbilityBenchWindow.class`, `com/hypixel/hytale/builtin/adventure/camera/CinematicPath.class`, `com/hypixel/hytale/builtin/adventure/camera/CinematicTeleport$1.class`, `com/hypixel/hytale/builtin/adventure/camera/CinematicTeleport.class`, `com/hypixel/hytale/builtin/adventure/camera/command/TpCinematicCommand.class`, `com/hypixel/hytale/builtin/adventure/farming/FarmingSystems$Ticking$TickContext.class`, `com/hypixel/hytale/builtin/adventure/farming/config/modifiers/WaterGrowthModifierAsset$Soil.class`, `com/hypixel/hytale/builtin/adventure/goblinlab/GoblinLabConsolePage$Action.class`, `com/hypixel/hytale/builtin/adventure/goblinlab/GoblinLabConsolePage$ConsoleSetting.class`, `com/hypixel/hytale/builtin/adventure/goblinlab/GoblinLabConsolePage$OpeningTimes.class`, `com/hypixel/hytale/builtin/adventure/goblinlab/GoblinLabConsolePage$PageData.class`, `com/hypixel/hytale/builtin/adventure/goblinlab/GoblinLabConsolePage.class`, `com/hypixel/hytale/builtin/adventure/goblinlab/GoblinLabPlugin.class`, `com/hypixel/hytale/builtin/adventure/memories/memories/npc/NPCMemory$AwardResult.class`, `com/hypixel/hytale/builtin/adventure/memories/memories/npc/NPCMemory$KillMemoriesSystem.class`, `com/hypixel/hytale/builtin/adventure/wilderness/WildernessConfig.class`, `com/hypixel/hytale/builtin/adventure/wilderness/WildernessPlugin.class`

Removed classes (first 60 of 589): `com/hypixel/hytale/builtin/adventure/objectives/blockstates/TreasureChestBlock.class`, `com/hypixel/hytale/builtin/adventure/objectives/config/task/TreasureMapObjectiveTaskAsset$ChestConfig.class`, `com/hypixel/hytale/builtin/adventure/objectives/config/task/TreasureMapObjectiveTaskAsset.class`, `com/hypixel/hytale/builtin/adventure/objectives/config/worldlocationproviders/CheckTagWorldHeightRadiusProvider.class`, `com/hypixel/hytale/builtin/adventure/objectives/config/worldlocationproviders/LookBlocksBelowProvider.class`, `com/hypixel/hytale/builtin/adventure/objectives/events/TreasureChestOpeningEvent.class`, `com/hypixel/hytale/builtin/adventure/objectives/interactions/DestroyTreasureConditionInteraction.class`, `com/hypixel/hytale/builtin/adventure/objectives/interactions/OpenTreasureContainerInteraction.class`, `com/hypixel/hytale/builtin/adventure/objectives/task/TreasureMapObjectiveTask.class`, `com/hypixel/hytale/builtin/adventure/objectives/transaction/SpawnTreasureChestTransactionRecord.class`, `com/hypixel/hytale/builtin/beds/sleep/components/PlayerSleep$MorningWakeUp.class`, `com/hypixel/hytale/builtin/blockphysics/PrefabBufferValidator$2.class`, `com/hypixel/hytale/builtin/blockphysics/PrefabBufferValidator$ValidateBlockEvent.class`, `com/hypixel/hytale/builtin/blockphysics/WorldValidationUtil.class`, `com/hypixel/hytale/builtin/buildertools/BuilderToolsPlugin$BuilderState$2.class`, `com/hypixel/hytale/builtin/buildertools/BuilderToolsPlugin$CachedAccessor.class`, `com/hypixel/hytale/builtin/buildertools/scriptedbrushes/BrushConfigChunkAccessor.class`, `com/hypixel/hytale/builtin/fluid/FluidPlugin$PreprocesorAccessor.class`, `com/hypixel/hytale/builtin/hytalegenerator/assets/interpolationasset/BiomeFrontierAsset.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/AbsDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/AmplitudeConstantDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/AmplitudeDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/AnchorDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/AngleDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/AxisDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/BaseHeightDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CacheDensity$Cache.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CacheDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CeilingDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CellWallDistanceDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/ClampDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/ConstantValueDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CubeDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CurveMapperDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/CylinderDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/DistanceDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/DistanceToBiomeEdgeDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/FastGradientWarpDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/FloorDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/FunctionDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/GradientDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/GradientWarpDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/InverterDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MaxDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MinDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MixDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiCacheDensity$Cache.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiCacheDensity$Entry.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiCacheDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiMixDensity$Key.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiMixDensity$Segment$GaugeSegmentComparator.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiMixDensity$Segment.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiMixDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/MultiplierDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/Noise2dDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/Noise3dDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/NormalizerDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/OffsetConstantDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/OffsetDensity.class`, `com/hypixel/hytale/builtin/hytalegenerator/density/nodes/PlaneDensity.class`

## Cross-check against measured 0.7.2 changes

Source of the 10 items: `HytaleRL/hytale-plugin-0.7.2/TRAINING-0.7.2.md` ("Gameplay changes that matter"). Status is what this diff shows; the `Evidence` column names the zip path and flattened JSON field (`a.b[0].c`).

| # | Measured change | Diff shows it | Evidence |
|---|---|---|---|
| 1 | Weapon_Sword_Iron knockback Force 6.0 -> 4.3, direction (0,1,-1) -> (0,0.2,-1) | SHOWN | `Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Left/Weapon_Sword_Primary_Swing_Left_Damage.json`: `DamageEffects.Knockback.Force` 6.0 -> 4.3; `DamageEffects.Knockback.Direction.Y` 1 -> 0.2; Direction.Z -1 -> -1 (unchanged) |
| 2 | Weapon attacks inherit DamageEntityParent_Hit_Interrupt (TRAINING says 31 weapon attack files) | SHOWN | 27 Server/Item/Interactions/Weapons/ files now have `Parent` = DamageEntityParent_Hit_Interrupt (was DamageEntityParent; this count covers Weapons/ only; TRAINING states 31 but its own breakdown sums to 30); files containing the id under Server/Item/Interactions/: 0.5=0, 0.7.2=37; e.g. `Server/Item/Interactions/Weapons/Battleaxe/Primary/Downstrike/Weapon_Battleaxe_Primary_Downstrike_Damage.json`, `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down/Weapon_Battleaxe_Primary_Swing_Down_Damage.json`, `Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Left/Weapon_Battleaxe_Primary_Swing_Down_Left_Damage.json` |
| 3 | Mana.json 0/0 -> 100/100, +1 per 0.3 s | SHOWN | `Server/Entity/Stats/Mana.json`: `Max` 0 -> 100; `InitialValue` 0 -> 100; `Regenerating[0].Interval` 0.2 -> 0.3 |
| 4 | Shortbow Iron charge damage 6/9/12/16 -> 1/2/4/7 | SHOWN | 8 bow-path fields match a pair; distinct pairs 4/4. `Server/Item/Items/Weapon/Shortbow/Template_Weapon_Shortbow.json` `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile` 6 -> 1; `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Bronze.json` `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile` 9 -> 2; `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Crude.json` `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile` 6 -> 1; `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Iron.json` `InteractionVars.Primary_Shoot_Damage_Strength_0.Interactions[0].DamageCalculator.BaseDamage.Projectile` 6 -> 1; `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Iron.json` `InteractionVars.Primary_Shoot_Damage_Strength_1.Interactions[0].DamageCalculator.BaseDamage.Projectile` 9 -> 2; `Server/Item/Items/Weapon/Shortbow/Weapon_Shortbow_Iron.json` `InteractionVars.Primary_Shoot_Damage_Strength_2.Interactions[0].DamageCalculator.BaseDamage.Projectile` 12 -> 4 |
| 5 | Dagger damage Iron 4 -> 6 | SHOWN | 5 dagger `Physical` fields 4 -> 6. `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Claw_Bone.json` `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical` 4 -> 6; `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Iron.json` `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical` 4 -> 6; `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Iron.json` `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical` 4 -> 6; `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Stone_Trork.json` `InteractionVars.Swing_Left_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical` 4 -> 6; `Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Stone_Trork.json` `InteractionVars.Swing_Right_Damage.Interactions[0].DamageCalculator.BaseDamage.Physical` 4 -> 6 |
| 6 | Weapon_Sword_Iron max durability 120 -> 250 | SHOWN | `Server/Item/Items/Weapon/Sword/Weapon_Sword_Iron.json`: `MaxDurability` 120 -> 250; `DurabilityLossOnHit` 0.21 -> 0.13 |
| 7 | Spider web speed 0.65 -> 0.45 | SHOWN | `Server/Item/Items/Deco/Deco_SpiderWeb.json`: `BlockType.MovementSettings.HorizontalSpeedMultiplier` 0.65 -> 0.45; same block also changed: `BlockType.MovementSettings.TerminalVelocityModifier` 0.5 -> 0.45; `BlockType.MovementSettings.JumpForceMultiplier` 0.5 -> 0.45; `BlockType.MovementSettings.Drag` 0.45 -> 0.82; new effect-level HorizontalSpeedMultiplier: `BlockType.Interactions.Collision.Interactions[0].EffectId.ApplicationEffects.HorizontalSpeedMultiplier` = 0.2 |
| 8 | NPC role paths moved from Aggressive/ and Neutral/ to Faction/ | PARTIAL | 5 referenced Roles files are absent at their 0.5 path. To Faction/: 3 (`Server/NPC/Roles/Intelligent/Aggressive/Trork/Templates/Template_Trork_Melee.json` -> `Server/NPC/Roles/Intelligent/Faction/Trork/Templates/Template_Trork_Melee.json`; `Server/NPC/Roles/Intelligent/Aggressive/Trork/Trork_Brawler.json` -> `Server/NPC/Roles/Intelligent/Faction/Trork/Trork_Brawler.json`; `Server/NPC/Roles/Intelligent/Neutral/Kweebec/Templates/Template_Kweebec_Razorleaf.json` -> `Server/NPC/Roles/Intelligent/Faction/Kweebec/Templates/Template_Kweebec_Razorleaf.json`). Elsewhere: 2 (`Server/NPC/Roles/_Core/Components/Steps/Component_Instruction_Intelligent_Chase.json` -> `Server/NPC/Roles/_Core/Components/Instructions/Combat/Component_Instruction_Intelligent_Chase.json`; `Server/NPC/Roles/_Core/Components/Steps/Component_Instruction_Target_Adjusted_Attack_Melee.json` -> `Server/NPC/Roles/_Core/Components/Instructions/Combat/Component_Instruction_Target_Adjusted_Attack_Melee.json`). |
| 9 | Trork and Kweebec role files unchanged | NOT SHOWN (value changes) | 88 Trork/Kweebec Roles files: 75 byte-identical (name match), 6 differ in keys or format only, 7 with value changes, 0 missing. Value changes: `Server/NPC/Roles/_Core/Tests/Beacons/Test_Beacon_Trork_Hunter.json`: `NameTranslationKey` "server.npcRoles.Test_Beacon_Trork_Hunter.name" -> "server_no_translation.npcRoles.Test_Beacon_Trork_Hunter.name"; `Server/NPC/Roles/_Core/Tests/Beacons/Test_Beacon_Trork_Leader.json`: `NameTranslationKey` "server.npcRoles.Test_Beacon_Trork_Leader.name" -> "server_no_translation.npcRoles.Test_Beacon_Trork_Leader.name"; `Server/NPC/Roles/_Core/Tests/Beacons/Test_Beacon_Trork_Soldier.json`: `NameTranslationKey` "server.npcRoles.Test_Beacon_Trork_Soldier.name" -> "server_no_translation.npcRoles.Test_Beacon_Trork_Soldier.name"; `Server/NPC/Roles/_Core/Tests/Beacons/Test_Beacon_Trork_Soldier_Path.json`: `NameTranslationKey` "server.npcRoles.Test_Beacon_Trork_Soldier_Path.name" -> "server_no_translation.npcRoles.Test_Beacon_Trork_Soldier_Path.name". Key-only examples: `Server/NPC/Roles/Intelligent/Aggressive/Trork/Templates/Template_Trork_Melee.json` (6 keys added, 0 removed); `Server/NPC/Roles/Intelligent/Neutral/Kweebec/Kweebec_Merchant.json` (1 keys added, 1 removed); `Server/NPC/Roles/Intelligent/Neutral/Kweebec/Kweebec_Prisoner.json` (1 keys added, 1 removed). |
| 10 | MovementConfig/Default.json (player walk/run/sprint/jump) | CHANGED (2 added, 0 removed, 0 changed) | `Server/Entity/MovementConfig/Default.json`: added `MaxSlopeAngleDegrees` = 30.0; added `MaxWallAngleDegrees` = 15.0 |

## Caveats

- Flattened keys use `.` for objects and `[i]` for arrays. A field that moves to another path shows as a removed key plus an added key; cross-check items match exact flattened keys.
- Path matching for moved NPC roles and identity checks is by file name (with extension) within `Server/NPC/Roles/`, not by full path.
- Placeholder expansion in `catalog.py` is limited to `{index:02d}` (range 1..10) and `{spear_id}` (from `_STANDARD_SPEAR_IDS`); other f-string paths stay unresolved.
- A pattern in the diff only shows that a value changed in the assets. It does not show engine semantics (for example, whether the knockback or interrupt changes the agent's motion); those stay with TRAINING-0.7.2.md.
