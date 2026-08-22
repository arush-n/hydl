package com.hytalerlbridge.combat;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;
import com.hytalerlbridge.combat.ruleset.Agent;
import com.hytalerlbridge.combat.ruleset.AgentAttack;
import com.hytalerlbridge.combat.ruleset.AgentCollisionMode;
import com.hytalerlbridge.combat.ruleset.Chase;
import com.hytalerlbridge.combat.ruleset.DamageInteraction;
import com.hytalerlbridge.combat.ruleset.DirectionalKnockback;
import com.hytalerlbridge.combat.ruleset.Engine;
import com.hytalerlbridge.combat.ruleset.Fixture;
import com.hytalerlbridge.combat.ruleset.KnockbackType;
import com.hytalerlbridge.combat.ruleset.KnockbackVelocityType;
import com.hytalerlbridge.combat.ruleset.MaintainDistance;
import com.hytalerlbridge.combat.ruleset.Matchup;
import com.hytalerlbridge.combat.ruleset.ObservationNormalization;
import com.hytalerlbridge.combat.ruleset.Regeneration;
import com.hytalerlbridge.combat.ruleset.Reward;
import com.hytalerlbridge.combat.ruleset.Target;
import com.hytalerlbridge.combat.ruleset.TargetAttack;
import com.hytalerlbridge.combat.ruleset.TargetVelocityControl;
import com.hytalerlbridge.combat.ruleset.WalkController;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.checkedBounds;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.checkedVector;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/**
 * Strict Java view of the single combat ruleset also consumed by JAX.
 *
 * <p>The JSON lives in the Python package and is added to the plugin resources
 * by Gradle. This prevents the standalone Java simulator and JAX environment
 * from silently drifting through duplicated literals.</p>
 */
public record CombatRuleset(
    String schema,
    int version,
    String hytaleServerVersion,
    Matchup matchup,
    Engine engine,
    Fixture fixture,
    Agent agent,
    Target target,
    DamageInteraction damageInteraction,
    Regeneration regeneration,
    Reward reward,
    ObservationNormalization observationNormalization
) {
    public static final String RESOURCE =
        "/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json";
    public static final String SCHEMA = "hytalerl_combat_ruleset_v6";
    public static final int VERSION = 9;
    public static final String HYTALE_SERVER_VERSION = "0.5.7";

    private static final CombatRuleset DEFAULT = load();

    public static CombatRuleset defaultRules() {
        return DEFAULT;
    }

    private static CombatRuleset load() {
        LoaderOptions options = new LoaderOptions();
        options.setAllowDuplicateKeys(false);
        options.setMaxAliasesForCollections(0);
        try (InputStream input = CombatRuleset.class.getResourceAsStream(RESOURCE)) {
            if (input == null) {
                throw new IllegalStateException(
                    "Missing packaged combat ruleset " + RESOURCE
                );
            }
            Object loaded = new Yaml(new SafeConstructor(options)).load(input);
            Map<String, Object> root = map(loaded, "ruleset");
            require(SCHEMA.equals(string(root, "schema")), "ruleset schema");
            require(integer(root, "version") == VERSION, "ruleset version");
            require(
                HYTALE_SERVER_VERSION.equals(string(root, "hytale_server_version")),
                "Hytale server version"
            );
            Map<String, Object> matchup = child(root, "matchup");
            Map<String, Object> engine = child(root, "engine");
            Map<String, Object> fixture = child(root, "fixture");
            Map<String, Object> agent = child(root, "agent");
            Map<String, Object> agentWalkController =
                child(agent, "walk_controller");
            Map<String, Object> target = child(root, "target");
            Map<String, Object> targetChase = child(target, "chase");
            Map<String, Object> targetMaintainDistance =
                child(target, "maintain_distance");
            Map<String, Object> damageInteraction =
                child(root, "damage_interaction");
            Map<String, Object> knockback =
                child(damageInteraction, "knockback");
            require(
                knockback.containsKey("velocity_config")
                    && knockback.get("velocity_config") == null,
                "knockback velocity config"
            );
            Map<String, Object> regeneration = child(root, "regeneration");
            Map<String, Object> reward = child(root, "reward");
            Map<String, Object> observationNormalization =
                child(root, "observation_normalization");
            List<AgentAttack> agentAttacks = new ArrayList<>();
            for (Map<String, Object> attack : children(agent, "attacks")) {
                agentAttacks.add(new AgentAttack(
                    integer(attack, "hit_delay_ticks"),
                    number(attack, "range"),
                    number(attack, "half_angle_degrees")
                ));
            }
            require(agentAttacks.size() == 3, "Razorleaf attack count");
            List<TargetAttack> targetAttacks = new ArrayList<>();
            for (Map<String, Object> attack : children(target, "attacks")) {
                targetAttacks.add(new TargetAttack(
                    string(attack, "interaction_id"),
                    integer(attack, "windup_ticks"),
                    integer(attack, "sweep_ticks"),
                    integer(attack, "recovery_ticks"),
                    number(attack, "selector_runtime_seconds"),
                    number(attack, "start_distance"),
                    number(attack, "end_distance"),
                    number(attack, "arc_degrees"),
                    MeleeAttackProfile.SweepDirection.valueOf(
                        string(attack, "sweep_direction")
                    ),
                    number(attack, "yaw_start_offset_degrees"),
                    number(attack, "pitch_offset_degrees"),
                    number(attack, "roll_offset_degrees"),
                    number(attack, "extend_top"),
                    number(attack, "extend_bottom"),
                    bool(attack, "requires_line_of_sight"),
                    number(attack, "damage")
                ));
            }
            require(targetAttacks.size() == 5, "Brawler attack count");
            return new CombatRuleset(
                string(root, "schema"),
                integer(root, "version"),
                string(root, "hytale_server_version"),
                new Matchup(
                    string(matchup, "agent_role"),
                    string(matchup, "target_role")
                ),
                new Engine(
                    integer(engine, "ticks_per_second"),
                    number(engine, "nominal_delta_seconds"),
                    number(engine, "loaded_delta_seconds"),
                    integer(engine, "motion_timing_profile_count"),
                    number(engine, "gravity"),
                    number(engine, "steering_slowdown_falloff"),
                    number(engine, "horizontal_selector_pi"),
                    number(engine, "legacy_horizontal_knockback_scale"),
                    number(
                        engine,
                        "legacy_motion_controller_horizontal_factor"
                    )
                ),
                new Fixture(
                    vector(fixture, "agent_spawn"),
                    vector(fixture, "target_offset"),
                    number(fixture, "floor_y")
                ),
                new Agent(
                    number(agent, "max_health"),
                    number(agent, "asset_max_walk_speed"),
                    number(agent, "max_speed"),
                    number(agent, "acceleration"),
                    number(agent, "jump_velocity_gravity_floor"),
                    number(agent, "jump_height_parameter"),
                    number(agent, "knockback_scale"),
                    number(agent, "movement_velocity_resistance"),
                    number(agent, "min_walk_speed"),
                    number(agent, "min_hit_slowdown"),
                    number(agent, "landing_velocity_scale"),
                    number(agent, "steering_relative_turn_speed"),
                    number(agent, "turn_degrees_per_second"),
                    new WalkController(
                        number(agentWalkController, "gravity"),
                        number(
                            agentWalkController,
                            "fall_acceleration_multiplier"
                        ),
                        number(
                            agentWalkController,
                            "gravity_drag_exponent"
                        ),
                        number(agentWalkController, "max_fall_speed"),
                        number(
                            agentWalkController,
                            "max_sink_speed_fluid"
                        ),
                        number(agentWalkController, "max_climb_height"),
                        number(agentWalkController, "max_drop_height")
                    ),
                    vector6(agent, "bounding_box"),
                    number(agent, "damage"),
                    number(agent, "attack_pause_min_seconds"),
                    number(agent, "attack_pause_max_seconds"),
                    List.copyOf(agentAttacks)
                ),
                new Target(
                    number(target, "max_health"),
                    number(target, "asset_max_walk_speed"),
                    number(target, "relative_chase_speed"),
                    number(target, "chase_speed"),
                    number(target, "acceleration"),
                    TargetVelocityControl.valueOf(
                        string(target, "velocity_control")
                    ),
                    AgentCollisionMode.valueOf(
                        string(target, "agent_collision_mode")
                    ),
                    number(target, "asset_eye_height"),
                    number(target, "model_scale"),
                    number(target, "effective_eye_height"),
                    vector6(target, "bounding_box"),
                    number(
                        target,
                        "max_head_rotation_degrees_per_second"
                    ),
                    number(target, "head_aim_relative_turn_speed"),
                    number(target, "head_default_relative_turn_speed"),
                    number(target, "head_yaw_min_degrees"),
                    number(target, "head_yaw_max_degrees"),
                    number(target, "head_pitch_min_degrees"),
                    number(target, "head_pitch_max_degrees"),
                    new Chase(
                        number(targetChase, "stop_distance"),
                        number(targetChase, "slowdown_distance")
                    ),
                    new MaintainDistance(
                        number(targetMaintainDistance, "activation_range"),
                        number(
                            targetMaintainDistance,
                            "desired_distance_min"
                        ),
                        number(
                            targetMaintainDistance,
                            "desired_distance_max"
                        ),
                        number(targetMaintainDistance, "move_threshold"),
                        number(
                            targetMaintainDistance,
                            "target_distance_factor"
                        ),
                        number(
                            targetMaintainDistance,
                            "move_towards_slowdown_distance"
                        ),
                        number(
                            targetMaintainDistance,
                            "relative_forward_speed"
                        ),
                        number(
                            targetMaintainDistance,
                            "relative_backward_speed"
                        ),
                        number(
                            targetMaintainDistance,
                            "strafing_duration_min_seconds"
                        ),
                        number(
                            targetMaintainDistance,
                            "strafing_duration_max_seconds"
                        ),
                        number(
                            targetMaintainDistance,
                            "strafing_frequency_min_seconds"
                        ),
                        number(
                            targetMaintainDistance,
                            "strafing_frequency_max_seconds"
                        ),
                        number(
                            targetMaintainDistance,
                            "strafing_yaw_offset_degrees"
                        ),
                        number(
                            targetMaintainDistance,
                            "strafing_translation_offset_degrees"
                        )
                    ),
                    integer(target, "chase_reaction_ticks"),
                    number(target, "turn_degrees_per_second"),
                    integer(target, "activation_min_tick"),
                    integer(target, "activation_max_tick"),
                    integer(target, "decision_delay_ticks"),
                    number(target, "sensor_range"),
                    number(target, "attack_pause_min_seconds"),
                    number(target, "attack_pause_max_seconds"),
                    List.copyOf(targetAttacks)
                ),
                new DamageInteraction(
                    string(damageInteraction, "interaction_id"),
                    new DirectionalKnockback(
                        KnockbackType.valueOf(string(knockback, "type")),
                        number(knockback, "force"),
                        number(knockback, "relative_x"),
                        number(knockback, "relative_z"),
                        number(knockback, "velocity_y"),
                        KnockbackVelocityType.valueOf(
                            string(knockback, "velocity_type")
                        ),
                        number(knockback, "duration_seconds")
                    )
                ),
                new Regeneration(
                    integer(regeneration, "delay_ticks"),
                    integer(regeneration, "interval_ticks"),
                    number(regeneration, "fraction")
                ),
                new Reward(
                    number(reward, "target_damage_scale"),
                    number(reward, "agent_damage_scale"),
                    number(reward, "completion"),
                    number(reward, "death")
                ),
                new ObservationNormalization(
                    number(observationNormalization, "wire_fixed_point_scale"),
                    number(observationNormalization, "vertical_speed_scale"),
                    number(
                        observationNormalization,
                        "facing_error_degrees_scale"
                    ),
                    number(
                        observationNormalization,
                        "head_pitch_degrees_scale"
                    )
                )
            );
        } catch (IOException exception) {
            throw new IllegalStateException("Unable to read combat ruleset", exception);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value, String name) {
        if (!(value instanceof Map<?, ?> raw)) {
            throw new IllegalStateException(name + " must be an object");
        }
        return (Map<String, Object>) raw;
    }

    private static Map<String, Object> child(Map<String, Object> map, String key) {
        return map(map.get(key), key);
    }

    private static List<Map<String, Object>> children(
        Map<String, Object> map,
        String key
    ) {
        Object value = map.get(key);
        if (!(value instanceof List<?> list)) {
            throw new IllegalStateException(key + " must be an array");
        }
        return list.stream().map(item -> map(item, key)).toList();
    }

    private static String string(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (!(value instanceof String result) || result.isBlank()) {
            throw new IllegalStateException(key + " must be a non-empty string");
        }
        return result;
    }

    private static int integer(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (!(value instanceof Number number)) {
            throw new IllegalStateException(key + " must be numeric");
        }
        double exact = number.doubleValue();
        if (!Double.isFinite(exact)
            || exact != Math.rint(exact)
            || exact < Integer.MIN_VALUE
            || exact > Integer.MAX_VALUE) {
            throw new IllegalStateException(key + " must be an exact int");
        }
        return (int) exact;
    }

    private static double number(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (!(value instanceof Number number)) {
            throw new IllegalStateException(key + " must be numeric");
        }
        double result = number.doubleValue();
        if (!Double.isFinite(result)) {
            throw new IllegalStateException(key + " must be finite");
        }
        return result;
    }

    private static boolean bool(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (!(value instanceof Boolean result)) {
            throw new IllegalStateException(key + " must be boolean");
        }
        return result;
    }

    private static double[] vector(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (!(value instanceof List<?> list) || list.size() != 3) {
            throw new IllegalStateException(key + " must contain three values");
        }
        double[] result = new double[3];
        for (int index = 0; index < result.length; index++) {
            Object item = list.get(index);
            if (!(item instanceof Number number)) {
                throw new IllegalStateException(key + " must be numeric");
            }
            result[index] = number.doubleValue();
        }
        return checkedVector(result, key);
    }

    private static double[] vector6(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (!(value instanceof List<?> list) || list.size() != 6) {
            throw new IllegalStateException(key + " must contain six values");
        }
        double[] result = new double[6];
        for (int index = 0; index < result.length; index++) {
            Object item = list.get(index);
            if (!(item instanceof Number number)) {
                throw new IllegalStateException(key + " must be numeric");
            }
            result[index] = number.doubleValue();
        }
        return checkedBounds(result, key);
    }

}
