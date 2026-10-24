"""Declared sources of the gen-7 overlay rows, one entry per row id (checked by sim/common/drift.py).

A tuple entry is an AssetSource (a value in the release's Assets.zip at a JSON pointer), a ClassSource (a
server-jar method's bytecode digest) or several of them, every part of which must hold. A Derived entry
states why the value is not re-read by the tool. Asset values are the 0.7.2 archive's values on 2026-10-23
(Assets.zip sha256 EE8F8D5B...); the expected value is the one the overlay row was written against.
"""

from hytalegym.sim.common.drift import AssetSource, ClassSource, Derived

DECLARATIONS = {
    "step_clock:combat/arsenal/profiles/hytale_0_5_7.py:2465": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/arsenal/profiles/hytale_0_5_7.py:2652": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/arsenal/profiles/hytale_0_5_7.py:4244": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/arsenal/profiles/hytale_0_5_7.py:4254": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/arsenal/profiles/hytale_0_5_7.py:4784": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/arsenal/programs/timing.py:6": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/clock.py:19": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/mechanics/factory.py:77": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/types.py:602": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/types.py:603": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/types.py:604": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/types.py:606": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "step_clock:combat/types.py:613": Derived(
        "owner ruling 2026-10-17: the agent operates at 30 Hz (RENDER.md, the step clock); no release fact"
    ),
    "player_locomotion:PLAYER_JUMP_FORCE": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_VARIABLE_JUMP_FALL_FORCE": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_BASE_SPEED": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_GAIT_SPEED_MULTIPLIERS": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_DRAG_MIN": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_DRAG_MAX": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_DRAG_MIN_SPEED": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_DRAG_MAX_SPEED": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_CONTROL_MIN_SPEED": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_CONTROL_MAX_SPEED": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_CONTROL_MIN_MULTIPLIER": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_locomotion:PLAYER_AIR_CONTROL_MAX_MULTIPLIER": Derived(
        "content table via adapter player_locomotion (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "walk_steering_sprint_multiplier:installed_default_forward_multiplier": Derived(
        "content table via adapter walk_steering_sprint_multiplier (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "sword_selectors:_SWORD_SWING_LEFT_SELECTOR": (
        # table weapons/interaction:Weapon_Sword_Primary_Swing_Left_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Left/Weapon_Sword_Primary_Swing_Left_Selector.json",
            "/RunTime",
            0.048,
        ),
    ),
    "sword_selectors:_SWORD_SWING_RIGHT_SELECTOR": (
        # table weapons/interaction:Weapon_Sword_Primary_Swing_Right_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Right/Weapon_Sword_Primary_Swing_Right_Selector.json",
            "/RunTime",
            0.048,
        ),
    ),
    "sword_selectors:_SWORD_SWING_DOWN_SELECTOR": (
        # table weapons/interaction:Weapon_Sword_Primary_Swing_Down_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Down/Weapon_Sword_Primary_Swing_Down_Selector.json",
            "/RunTime",
            0.056,
        ),
    ),
    "sword_selectors:_SWORD_VORTEX_SPIN_SELECTOR": (
        # table weapons/interaction:Weapon_Sword_Signature_Vortexstrike_Spin_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Signature/VortexStrike/Weapon_Sword_Signature_Vortexstrike_Spin_Selector.json",
            "/RunTime",
            0.13,
        ),
    ),
    "sword_selectors:_SWORD_THRUST_SELECTOR": (
        # table weapons/interaction:Weapon_Sword_Primary_Thrust_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Thrust/Weapon_Sword_Primary_Thrust_Selector.json",
            "/RunTime",
            0.267,
        ),
    ),
    "sword_selectors:_SWORD_VORTEX_STAB_SELECTOR": (
        # table weapons/interaction:Weapon_Sword_Signature_Vortexstrike_Stab_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Signature/VortexStrike/Weapon_Sword_Signature_Vortexstrike_Stab_Selector.json",
            "/RunTime",
            0.267,
        ),
    ),
    "melee_hold_constants:_NESTED_OUTER_GATE_SECONDS": Derived(
        "content table via adapter melee_hold_constants (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "melee_hold_constants:_SWORD_THRUST_HOLD_SECONDS": Derived(
        "content table via adapter melee_hold_constants (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "melee_hold_constants:_DOWNSTRIKE_HOLD_SECONDS": Derived(
        "content table via adapter melee_hold_constants (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "melee_hold_constants:_MACE_CHARGED_HOLD_SECONDS": Derived(
        "content table via adapter melee_hold_constants (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "iron_template_damage:_iron_shortbow defaults": Derived(
        "content table via adapter shortbow_tiers (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "shortbow_tiers:shortbow draw ladder launch speeds": Derived(
        "content table via adapter shortbow_tiers (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "iron_template_damage:_iron_crossbow defaults": (
        # table weapons/item:Weapon_Crossbow_Iron damage.Combo_Projectile_Damage.base_first (fidelity 6c)
        AssetSource(
            "Server/Item/Items/Weapon/Crossbow/Weapon_Crossbow_Iron.json",
            "/InteractionVars/Combo_Projectile_Damage/Interactions/0/DamageCalculator/BaseDamage/Projectile",
            13,
        ),
    ),
    "sword_thrust_timing:dash": Derived(
        "content table via adapter sword_thrust_timing (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "sword_thrust_timing:stab": Derived(
        "content table via adapter sword_thrust_timing (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "melee_hold_constants:sword_thrust_hold_speed_multiplier": Derived(
        "content table via adapter melee_hold_constants (swap plan, active): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "melee_timing:Weapon_Daggers_Primary_Swing_Left": (
        # table weapons/interaction:Weapon_Daggers_Primary_Swing_Left document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Daggers/Primary/Swing_Left/Weapon_Daggers_Primary_Swing_Left.json",
            "/RunTime",
            0.066,
        ),
    ),
    "melee_timing:Weapon_Daggers_Primary_Stab_Left_Selector": (
        # table weapons/interaction:Weapon_Daggers_Primary_Stab_Left_Selector document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Daggers/Primary/Stab_Left/Weapon_Daggers_Primary_Stab_Left_Selector.json",
            "/RunTime",
            0.066,
        ),
    ),
    "melee_timing:Weapon_Battleaxe_Primary_Swing_Down_Left": (
        # table weapons/interaction:Weapon_Battleaxe_Primary_Swing_Down_Left document.RunTime (fidelity 5f)
        AssetSource(
            "Server/Item/Interactions/Weapons/Battleaxe/Primary/Swing_Down_Left/Weapon_Battleaxe_Primary_Swing_Down_Left.json",
            "/RunTime",
            0.348,
        ),
    ),
    "player_stamina:PLAYER_STAMINA_MAXIMUM": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_stamina:PLAYER_STAMINA_MINIMUM": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_stamina:PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_stamina:PLAYER_STAMINA_REGEN_PER_SECOND": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_stamina:PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_stamina:PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "player_stamina:PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS": Derived(
        "content table via adapter player_stamina (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_MINIMUM": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_MAXIMUM": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_REGEN_AMOUNT": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_REGEN_INTERVAL_SECONDS": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_REGEN_DELAY_AMOUNT": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_REGEN_DELAY_INTERVAL_SECONDS": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:STAMINA_BROKEN_REGEN_DELAY_SECONDS": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:OXYGEN_MINIMUM": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:OXYGEN_MAXIMUM": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:OXYGEN_BREATHABLE_REGEN_AMOUNT": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:OXYGEN_SUFFOCATING_REGEN_AMOUNT": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_resource_constants:OXYGEN_REGEN_INTERVAL_SECONDS": Derived(
        "content table via adapter mechanics_resource_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "control_immunity:CONTROL_IMMUNITY_MAXIMUM": Derived(
        "content table via adapter control_immunity (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "control_immunity:CONTROL_IMMUNITY_REGEN_AMOUNT": Derived(
        "content table via adapter control_immunity (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "control_immunity:CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS": Derived(
        "content table via adapter control_immunity (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_AUTHORED_ACTION_MASK": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_STAMINA_ADMISSION_COST": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_STAMINA_SPEND_COST": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_FORCE": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_INVULNERABILITY_SECONDS": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_REGEN_DELAY_SECONDS": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_AIR_RESISTANCE": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_AIR_RESISTANCE_MAX": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_GROUND_RESISTANCE": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_GROUND_RESISTANCE_MAX": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "dodge_constants:DODGE_RESISTANCE_THRESHOLD": Derived(
        "content table via adapter dodge_constants (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_factory_resources:mana_maximum default": Derived(
        "content table via adapter mechanics_factory_resources (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_factory_resources:mana regen 1.0 / 0.2 s": Derived(
        "content table via adapter mechanics_factory_resources (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "mechanics_factory_resources:magic charges regen 1.0 / 2.0 s": Derived(
        "content table via adapter mechanics_factory_resources (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "iron_template_damage:_iron_sword defaults": Derived(
        "content table via adapter iron_template_damage (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "iron_template_damage:_iron_mace defaults": Derived(
        "content table via adapter iron_template_damage (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "iron_template_damage:_iron_battleaxe defaults": Derived(
        "content table via adapter iron_template_damage (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "iron_template_damage:_iron_daggers defaults": Derived(
        "content table via adapter iron_template_damage (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "sword_knockback:sword melee force tuples": Derived(
        "content table via adapter sword_knockback (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "shortbow_tiers:shortbow draw ladder": Derived(
        "content table via adapter shortbow_tiers (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "spear_rows:standard spear partials": Derived(
        "content table via adapter spear_rows (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "potions:_potions values": Derived(
        "content table via adapter potions (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_MODEL_HALF_EXTENT": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_MUZZLE_VELOCITY": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_TERMINAL_VELOCITY": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_GRAVITY": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_DAMAGE": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_DEAD_TIME_SECONDS": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_AUTHORED_LIFETIME_SECONDS": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_DEPTH_OFFSET": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "outlander_arrow:OUTLANDER_ARROW_VERTICAL_OFFSET": Derived(
        "content table via adapter outlander_arrow (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "crafting_capacities:RECIPE_TABLE_CAPACITY": Derived(
        "content table via adapter crafting_capacities (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "crafting_capacities:MAX_INGREDIENTS": Derived(
        "content table via adapter crafting_capacities (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "crafting_capacities:MAX_OUTPUTS": Derived(
        "content table via adapter crafting_capacities (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "crafting_capacities:MAX_BENCH_REQUIREMENTS": Derived(
        "content table via adapter crafting_capacities (swap plan, todo): the content re-extraction and diff check it (UPGRADE.md steps 2-3)"
    ),
    "augment_context_field": Derived(
        "structural or behavioural edit (fidelity row 11x): its differential is in the Patch row"
    ),
    "augment_assets_field": Derived(
        "structural or behavioural edit (fidelity row 11x): its differential is in the Patch row"
    ),
    "augment_bench_context": Derived(
        "structural or behavioural edit (fidelity row 11x): its differential is in the Patch row"
    ),
    "sword_swing_left_knockback": (
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Left/Weapon_Sword_Primary_Swing_Left_Damage.json",
            "/DamageEffects/Knockback/Force",
            4.3,
        ),
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Left/Weapon_Sword_Primary_Swing_Left_Damage.json",
            "/DamageEffects/Knockback/Direction/Y",
            0.2,
        ),
    ),
    "sword_swing_right_knockback": (
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Right/Weapon_Sword_Primary_Swing_Right_Damage.json",
            "/DamageEffects/Knockback/Force",
            5.0,
        ),
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Right/Weapon_Sword_Primary_Swing_Right_Damage.json",
            "/DamageEffects/Knockback/Direction/Y",
            0.2,
        ),
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Right/Weapon_Sword_Primary_Swing_Right_Damage.json",
            "/DamageEffects/Knockback/VelocityType",
            "Add",
        ),
    ),
    "sword_swing_down_knockback": (
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Down/Weapon_Sword_Primary_Swing_Down_Damage.json",
            "/DamageEffects/Knockback/Force",
            10.8,
        ),
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Down/Weapon_Sword_Primary_Swing_Down_Damage.json",
            "/DamageEffects/Knockback/Direction/Y",
            0.2,
        ),
    ),
    "sword_swing_left_velocity_add": (
        AssetSource(
            "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Left/Weapon_Sword_Primary_Swing_Left_Damage.json",
            "/DamageEffects/Knockback/VelocityType",
            "Add",
        ),
    ),
    "iron_daggers_swing_damage": (
        AssetSource(
            "Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Iron.json",
            "/InteractionVars/Swing_Left_Damage/Interactions/0/DamageCalculator/BaseDamage/Physical",
            6,
        ),
        AssetSource(
            "Server/Item/Items/Weapon/Daggers/Weapon_Daggers_Iron.json",
            "/InteractionVars/Swing_Right_Damage/Interactions/0/DamageCalculator/BaseDamage/Physical",
            6,
        ),
    ),
    "void_scythe_spin_hold_table": (
        AssetSource(
            "Server/Item/Interactions/Weapons/Battleaxe/Deprecated/Variants/Battleaxe_Scythe_Void_Attack.json",
            "/Next/1.670",
            "Battleaxe_Swing_Left_Charged",
        ),
        AssetSource(
            "Server/Item/Interactions/Weapons/Battleaxe/Deprecated/Variants/Battleaxe_Scythe_Void_Attack.json",
            "/AllowIndefiniteHold",
            False,
        ),
    ),
    "guard_sword_no_primary_interrupt": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "guard_mace_no_primary_interrupt": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "guard_battleaxe_no_primary_interrupt": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "guard_daggers_no_primary_interrupt": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "slope_wall_limits_parameters": (
        AssetSource(
            "Server/Entity/MovementConfig/Default.json", "/MaxSlopeAngleDegrees", 30.0
        ),
        AssetSource(
            "Server/Entity/MovementConfig/Default.json", "/MaxWallAngleDegrees", 15.0
        ),
    ),
    "perception_detail_hitbox_los": Derived(
        "structural or behavioural edit (fidelity row 2b): its differential is in the Patch row"
    ),
    "npc_rejump_import": Derived(
        "structural or behavioural edit (fidelity row 1c): its differential is in the Patch row"
    ),
    "npc_rejump_after_landing": Derived(
        "structural or behavioural edit (fidelity row 1c): its differential is in the Patch row"
    ),
    "guard_release_closes_guard_choice": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "structural_bench_import": Derived(
        "structural or behavioural edit (fidelity row 11): its differential is in the Patch row"
    ),
    "structural_bench_types": Derived(
        "structural or behavioural edit (fidelity row 11): its differential is in the Patch row"
    ),
    "structural_bench_admission": Derived(
        "structural or behavioural edit (fidelity row 11): its differential is in the Patch row"
    ),
    "crafting_table_capacity": Derived(
        "structural or behavioural edit (fidelity row 11): its differential is in the Patch row"
    ),
    "dodge_starts_one_drain_later": Derived(
        "structural or behavioural edit (fidelity row 5f): its differential is in the Patch row"
    ),
    "dodge_effects_on_acceptance": Derived(
        "structural or behavioural edit (fidelity row 5f): its differential is in the Patch row"
    ),
    "potion_stamina_072_drink": (
        AssetSource(
            "Server/Entity/Effects/Potion/Potion_Stamina_Instant_Small.json",
            "/StatModifiers/Stamina",
            45,
        ),
        AssetSource(
            "Server/Item/Interactions/Consumables/Consume_Potion_Stamina_Cooldown.json",
            "/Cooldown/Cooldown",
            30,
        ),
    ),
    "jump_carry_state_field": Derived(
        "structural or behavioural edit (fidelity row 1c): its differential is in the Patch row"
    ),
    "jump_carry_state_init": Derived(
        "structural or behavioural edit (fidelity row 1c): its differential is in the Patch row"
    ),
    "landing_walk_state_field": Derived(
        "structural or behavioural edit (fidelity row 1c): its differential is in the Patch row"
    ),
    "landing_walk_state_init": Derived(
        "structural or behavioural edit (fidelity row 1c): its differential is in the Patch row"
    ),
    "guard_reentry_seconds_field": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "guard_reentry_seconds_init": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "guard_reentry_closes_new_claim": Derived(
        "structural or behavioural edit (fidelity row 5d): its differential is in the Patch row"
    ),
    "bench_tier_admission": Derived(
        "structural or behavioural edit (fidelity row 11b): its differential is in the Patch row"
    ),
    "memories_state_field": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_admission": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_resolution": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_fieldcraft_signature": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_fieldcraft_body": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_bench_context": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_provider_context": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "memories_commit_context": Derived(
        "structural or behavioural edit (fidelity row 11m): its differential is in the Patch row"
    ),
    "timed_job_fields": Derived(
        "structural or behavioural edit (fidelity row 11j): its differential is in the Patch row"
    ),
    "timed_job_init": Derived(
        "structural or behavioural edit (fidelity row 11j): its differential is in the Patch row"
    ),
    "timed_resolved_time": Derived(
        "structural or behavioural edit (fidelity row 11j): its differential is in the Patch row"
    ),
    "timed_executor_import": Derived(
        "structural or behavioural edit (fidelity row 11j): its differential is in the Patch row"
    ),
    "timed_executor_start": Derived(
        "structural or behavioural edit (fidelity row 11j): its differential is in the Patch row"
    ),
    "craft_pre_inventory": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "processing_fields": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "processing_init": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "diagram_fields": Derived(
        "structural or behavioural edit (fidelity row 11q): its differential is in the Patch row"
    ),
    "diagram_init": Derived(
        "structural or behavioural edit (fidelity row 11q): its differential is in the Patch row"
    ),
    "processing_bench_import": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "processing_bench_types": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "diagram_bench_types": Derived(
        "structural or behavioural edit (fidelity row 11q): its differential is in the Patch row"
    ),
    "processing_executor_import": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "diagram_executor_import": Derived(
        "structural or behavioural edit (fidelity row 11q): its differential is in the Patch row"
    ),
    "processing_executor_start": Derived(
        "structural or behavioural edit (fidelity row 11p): its differential is in the Patch row"
    ),
    "hit_interrupt_config_field": Derived(
        "structural or behavioural edit (fidelity row 5a): its differential is in the Patch row"
    ),
    "hit_interrupt_state_field": Derived(
        "structural or behavioural edit (fidelity row 5a): its differential is in the Patch row"
    ),
    "hit_interrupt_state_init": Derived(
        "structural or behavioural edit (fidelity row 5a): its differential is in the Patch row"
    ),
    "hit_interrupt_loadout_argument": Derived(
        "structural or behavioural edit (fidelity row 5a): its differential is in the Patch row"
    ),
    "perception_role_config_field": Derived(
        "structural or behavioural edit (fidelity row PV2): its differential is in the Patch row"
    ),
    "trork_shaman_rubble_builder": (
        AssetSource(
            "Server/Item/Interactions/NPCs/NPC_Rubble_Throw.json", "/RunTime", 0.2
        ),
        AssetSource(
            "Server/Item/Interactions/NPCs/NPC_Rubble_Throw.json",
            "/Next/RunTime",
            0.217,
        ),
        AssetSource(
            "Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/MuzzleVelocity", 30
        ),
        AssetSource(
            "Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/TerminalVelocity", 30
        ),
        AssetSource("Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/Gravity", 20),
        AssetSource(
            "Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/Bounciness", 0.5
        ),
        AssetSource("Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/TimeToLive", 5),
        AssetSource("Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/Damage", 16),
        AssetSource("Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/DeadTime", 0),
        AssetSource(
            "Server/Models/Projectiles/Items/Rubble/Rubble_Stone.json",
            "/HitBox/Max/X",
            0.15,
        ),
    ),
    "trork_shaman_rubble_registry": Derived(
        "structural or behavioural edit (fidelity row Shaman): its differential is in the Patch row"
    ),
    "trork_shaman_rubble_profile_name": Derived(
        "structural or behavioural edit (fidelity row Shaman): its differential is in the Patch row"
    ),
    "trork_shaman_rubble_launch": (
        AssetSource(
            "Server/Projectiles/NPCs/NPC_Rubble_Stone.json",
            "/HorizontalCenterShot",
            0.1,
        ),
        AssetSource("Server/Projectiles/NPCs/NPC_Rubble_Stone.json", "/DepthShot", 1),
    ),
    "trork_shaman_rubble_kind": Derived(
        "structural or behavioural edit (fidelity row Shaman): its differential is in the Patch row"
    ),
    "trork_shaman_rubble_kind_bound": Derived(
        "structural or behavioural edit (fidelity row Shaman): its differential is in the Patch row"
    ),
    "trork_shaman_rubble_kind_name": Derived(
        "structural or behavioural edit (fidelity row Shaman): its differential is in the Patch row"
    ),
    "skeleton_archer_turret_bow_builder": (
        AssetSource(
            "Server/Item/Interactions/NPCs/Undead/Skeleton_Archer/Skeleton_Archer_Bow_Shoot.json",
            "/RunTime",
            1,
        ),
        AssetSource(
            "Server/Item/Interactions/NPCs/Undead/Skeleton_Archer/Skeleton_Archer_Bow_Shoot.json",
            "/Next/RunTime",
            0.2,
        ),
        AssetSource(
            "Server/Projectiles/NPCs/Undead/Skeleton_Archer/Skeleton_Archer_Arrow.json",
            "/Damage",
            16,
        ),
        AssetSource("Server/Projectiles/Arrow_FullCharge.json", "/MuzzleVelocity", 50),
        AssetSource(
            "Server/Projectiles/Arrow_FullCharge.json", "/TerminalVelocity", 50
        ),
        AssetSource("Server/Projectiles/Arrow_FullCharge.json", "/Gravity", 10),
        AssetSource("Server/Projectiles/Arrow_FullCharge.json", "/TimeToLive", 20),
        AssetSource("Server/Projectiles/Arrow_FullCharge.json", "/DeadTime", 0.1),
        AssetSource(
            "Server/Models/Projectiles/Weapons/Arrow/Arrow_Iron.json",
            "/HitBox/Max/X",
            0.075,
        ),
    ),
    "skeleton_archer_turret_bow_registry": Derived(
        "structural or behavioural edit (fidelity row Skeleton_Archer_Turret): its differential is in the Patch row"
    ),
    "skeleton_archer_turret_bow_profile_name": Derived(
        "structural or behavioural edit (fidelity row Skeleton_Archer_Turret): its differential is in the Patch row"
    ),
    "crafting_augment_assets": Derived(
        "construct (fidelity row 11x): its differential is in the Hook row"
    ),
    "crafting_augment_gate": (
        # javap -c -p of CraftingManager.isValidBenchForRecipe/3, normalised (sim/common/drift.py)
        ClassSource(
            "com.hypixel.hytale.builtin.crafting.component.CraftingManager",
            "isValidBenchForRecipe/3",
            "dd94d12cb6b05f404146b62d14c2398f0c624262ead40a190180f684bf6645cd",
        ),
    ),
    "hit_interrupt_roles": Derived(
        "construct (fidelity row 5a): its differential is in the Hook row"
    ),
    "hit_interrupt_reset": Derived(
        "construct (fidelity row 5a): its differential is in the Hook row"
    ),
    "hit_interrupt_clock": Derived(
        "construct (fidelity row 5a): its differential is in the Hook row"
    ),
    "hit_interrupt_flinch": Derived(
        "construct (fidelity row 5a): its differential is in the Hook row"
    ),
    "mana_pool": Derived(
        "construct (fidelity row 7a): its differential is in the Hook row"
    ),
    "mana_pool_loadout": Derived(
        "construct (fidelity row 7a): its differential is in the Hook row"
    ),
    "magic_costs": Derived(
        "construct (fidelity row 7c): its differential is in the Hook row"
    ),
    "rune_casting": Derived(
        "construct (fidelity row 7b): its differential is in the Hook row"
    ),
    "detail_hitbox_line_of_sight": Derived(
        "construct (fidelity row 2b): its differential is in the Hook row"
    ),
    "npc_walk_jump_velocity": Derived(
        "construct (fidelity row 1c): its differential is in the Hook row"
    ),
    "npc_jump_arc_previous_delta": Derived(
        "construct (fidelity row 1c): its differential is in the Hook row"
    ),
    "auto_step_up": Derived(
        "construct (fidelity row 1f): its differential is in the Hook row"
    ),
    "npc_fluid_walk": (
        # javap -c -p of MotionControllerWalk.computeMove/6, normalised (sim/common/drift.py)
        ClassSource(
            "com.hypixel.hytale.server.npc.movement.controllers.MotionControllerWalk",
            "computeMove/6",
            "102b4b7450eadc2f7be6cf13665b957f67824441c35f6312080ee05d59eafd66",
        ),
    ),
    "npc_jump_arc_fall_speed": Derived(
        "construct (fidelity row 1c): its differential is in the Hook row"
    ),
    "dagger_swing_damage": Derived(
        "construct (fidelity row 5c): its differential is in the Hook row"
    ),
    "crossbow_scalar_variant": Derived(
        "construct (fidelity row 6c): its differential is in the Hook row"
    ),
    "shortbow_scalar_variant": Derived(
        "construct (fidelity row 6a): its differential is in the Hook row"
    ),
    "guard_cancel_window_roots": Derived(
        "construct (fidelity row 5d): its differential is in the Hook row"
    ),
    "guard_cancel_window_operations": Derived(
        "construct (fidelity row 5d): its differential is in the Hook row"
    ),
    "native_binding_types": Derived(
        "construct (fidelity row 6c): its differential is in the Hook row"
    ),
    "bench_crafting_commit": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "bench_crafting_first_candidates": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "crafting_materials_item_table": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "crafting_materials_admission": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "crafting_materials_resolution": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "crafting_materials_inventory": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "crafting_materials_commit": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "crafting_materials_exclusions": Derived(
        "construct (fidelity row 11): its differential is in the Hook row"
    ),
    "dodge_action_head": Derived(
        "construct (fidelity row 9): its differential is in the Hook row"
    ),
    "weapon_durability_catalog": Derived(
        "construct (fidelity row 8d): its differential is in the Hook row"
    ),
    "npc_walk_sprint_learner": Derived(
        "construct (fidelity row 1a): its differential is in the Hook row"
    ),
    "npc_jump_carry": Derived(
        "construct (fidelity row 1c): its differential is in the Hook row"
    ),
    "landing_walk_resume": Derived(
        "construct (fidelity row 1c): its differential is in the Hook row"
    ),
    "bench_tier_assets": Derived(
        "construct (fidelity row 11b): its differential is in the Hook row"
    ),
    "crafting_memories_assets": Derived(
        "construct (fidelity row 11m): its differential is in the Hook row"
    ),
    "timed_crafting_admission": Derived(
        "construct (fidelity row 11j): its differential is in the Hook row"
    ),
    "timed_crafting_execution": Derived(
        "construct (fidelity row 11j): its differential is in the Hook row"
    ),
    "timed_crafting_executor": Derived(
        "construct (fidelity row 11j): its differential is in the Hook row"
    ),
    "processing_admission": Derived(
        "construct (fidelity row 11p): its differential is in the Hook row"
    ),
    "diagram_admission": Derived(
        "construct (fidelity row 11q): its differential is in the Hook row"
    ),
    "processing_executor": Derived(
        "construct (fidelity row 11p): its differential is in the Hook row"
    ),
    "diagram_executor": Derived(
        "construct (fidelity row 11q): its differential is in the Hook row"
    ),
    "guard_reentry_cooldown": Derived(
        "construct (fidelity row 5d): its differential is in the Hook row"
    ),
    "ability_start_split": Derived(
        "construct (fidelity row 5f): its differential is in the Hook row"
    ),
    "replace_default_hop": Derived(
        "construct (fidelity row 5f): its differential is in the Hook row"
    ),
    "shortbow_reset_ammo": Derived(
        "construct (fidelity row 5f): its differential is in the Hook row"
    ),
    "shield_bash_root": Derived(
        "construct (fidelity row 5d): its differential is in the Hook row"
    ),
    "reset_head_heading": Derived(
        "construct (fidelity row 5g): its differential is in the Hook row"
    ),
    "ability_ground_continuation": Derived(
        "construct (fidelity row 5h): its differential is in the Hook row"
    ),
    "status_clock": Derived(
        "construct (fidelity row none): its differential is in the Hook row"
    ),
    "perception_v2": Derived(
        "construct (fidelity row none): its differential is in the Hook row"
    ),
    "perception_roles": Derived(
        "construct (fidelity row none): its differential is in the Hook row"
    ),
    "legal_hidden_held_item_and_abilities": Derived(
        "patch (fidelity row 14): legal observation masks a hidden body's weapon and ability rows; its differential is "
        "tests/sim/test_legal_observation_privilege.py (no 0.7.2 source value: a contract rule)"
    ),
    "legal_hidden_target_attack_phase": Derived(
        "patch (fidelity row 14): legal observation keeps a hidden target's base attack values; its differential is "
        "tests/sim/test_legal_observation_privilege.py (no 0.7.2 source value: a contract rule)"
    ),
    "game_clock": (
        # S14 (owner 2026-10-23): the 0.7.2 day and night lengths; SECONDS_PER_DAY 86400 and the 0.6 day split are
        # WorldTimeResource constants (javap), checked by tests/sim/test_game_clock.py.
        AssetSource("Server/GameplayConfigs/Default.json", "/World/DaytimeDurationSeconds", 1728),
        AssetSource("Server/GameplayConfigs/Default.json", "/World/NighttimeDurationSeconds", 1152),
    ),
}
