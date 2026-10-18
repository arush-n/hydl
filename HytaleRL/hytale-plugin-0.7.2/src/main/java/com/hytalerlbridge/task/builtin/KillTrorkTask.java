package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.combat.simulation.ActiveMeleeSimulator;

/**
 * Built-in task adapter for the reusable native-calibrated melee simulator.
 *
 * <p>The combat model lives under {@code combat.simulation}; keeping this
 * class in {@code task.builtin} preserves the public {@code kill_trork} task
 * registration without coupling the model to the task registry.</p>
 */
public final class KillTrorkTask extends ActiveMeleeSimulator {}
