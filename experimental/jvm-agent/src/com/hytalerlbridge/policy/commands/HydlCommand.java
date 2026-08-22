package com.hytalerlbridge.policy.commands;

import com.hypixel.hytale.server.core.command.system.basecommands.AbstractCommandCollection;
import com.hytalerlbridge.policy.PolicyAgentPlugin;

/**
 * {@code /hydl} -- everything this mod exposes in game, under one name.
 *
 * <p>Deliberately <em>not</em> hung off {@code /npc}. That command belongs to
 * Hytale's own NPC plugin and carries twenty-odd subcommands about roles,
 * flocks, paths and blackboards; adding ours to it would mean our verbs and
 * theirs sharing a namespace where neither owns the other, and every future
 * Hytale update could collide with us. A separate root also means the help
 * output for {@code /hydl} is only our surface, which is the thing an operator
 * actually wants to read.
 *
 * <p>What the mod does <em>not</em> reimplement is spawning itself. The NPC is
 * created through {@code NPCPlugin.spawnEntity} exactly as
 * {@code NPCSpawnCommand} does, because a second spawn path would drift from
 * the engine's -- role validation, model selection and bounding-box placement
 * are all subtle, and getting them subtly wrong produces an NPC that exists but
 * behaves unlike every other NPC in the world.
 */
public final class HydlCommand extends AbstractCommandCollection {

    public HydlCommand(PolicyAgentPlugin plugin) {
        super("hydl", "Policy-driven agents: spawn them and inspect them.");
        addSubCommand(new HydlSpawnCommand(plugin));
        addSubCommand(new HydlStatusCommand(plugin));
    }
}
