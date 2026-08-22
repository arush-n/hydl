package com.hytalerlbridge.policy.commands;

import com.hypixel.hytale.server.core.Message;
import com.hypixel.hytale.server.core.command.system.CommandContext;
import com.hypixel.hytale.server.core.command.system.basecommands.CommandBase;
import com.hytalerlbridge.policy.PolicyAgentPlugin;
import java.util.Map;
import javax.annotation.Nonnull;

/**
 * {@code /hydl status} -- is the policy armed, and is it actually driving anything.
 *
 * <p>Extends {@link CommandBase} rather than a player command so the
 * <em>server console</em> can run it. That matters: the console has no player
 * reference, and anything extending {@code AbstractPlayerCommand} is refused
 * outright there. Status is exactly the thing you want to read from a headless
 * terminal or a log stream, so it is kept senderless on purpose.
 *
 * <p>The load-bearing line is <b>claimed</b>. An armed policy with zero claimed
 * NPCs and a populated <b>seen but not claimed</b> list means agents exist and
 * the role name does not match -- which otherwise presents as "the policy does
 * nothing" with no error anywhere, because nothing failed.
 */
public final class HydlStatusCommand extends CommandBase {

    private final PolicyAgentPlugin plugin;

    public HydlStatusCommand(PolicyAgentPlugin plugin) {
        super("status", "Whether the policy is armed and what it is driving.");
        this.plugin = plugin;
    }

    @Override
    protected void executeSync(@Nonnull CommandContext context) {
        StringBuilder out = new StringBuilder("=== HydlRL policy agent ===\n");
        if (!plugin.isArmed()) {
            out.append("armed:    NO -- ").append(plugin.inertReason()).append('\n');
            context.sendMessage(Message.raw(out.toString()));
            return;
        }

        out.append("armed:    yes\n")
           .append("role:     ").append(plugin.controlledRole()).append('\n')
           .append("claimed:  ").append(plugin.claimedCount()).append(" NPC(s)\n")
           .append("decision: every ").append(plugin.decisionPeriod())
           .append(" tick(s) (~")
           .append(Math.round(plugin.decisionPeriod() / 30.0 * 1000.0))
           .append(" ms at 30 TPS)\n")
           .append("jump:     ").append(plugin.jumpAllowed() ? "enabled" : "suppressed (nojump)")
           .append('\n')
           .append("contract: observation=").append(plugin.observationSize())
           .append(" actions=").append(plugin.actionSize()).append('\n')
           .append("perception: ").append(plugin.perceptionDescription())
           .append('\n')
           .append("live test: ")
           .append(plugin.liveActionControlDescription())
           .append('\n');

        long[] combat = plugin.liveCombatCounters();
        if (combat.length == 11) {
            out.append("combat:  bound=").append(combat[0])
               .append(" bind_rejected=").append(combat[1])
               .append(" unavailable=").append(combat[2])
               .append(" attack_accepted=").append(combat[3])
               .append(" ability_accepted=").append(combat[4])
               .append(" dodge_accepted=").append(combat[5])
               .append(" guard_started=").append(combat[6])
               .append('\n');
        }
        long[] world = plugin.liveWorldVerbCounters();
        if (world.length == 9) {
            out.append("world:   requested=").append(world[0])
               .append(" accepted=").append(world[1])
               .append(" finished=").append(world[2])
               .append(" failed=").append(world[3])
               .append(" binding_rejected=").append(world[4])
               .append(" busy=").append(world[5])
               .append(" unsupported=").append(world[6])
               .append('\n');
        }

        // The comprehension line. Contract width says what the policy *can*
        // read; this says what actually carried signal when it last looked. A
        // wide contract with almost nothing non-zero is an agent reading
        // padding, and every other diagnostic looks healthy while it happens.
        int nonZero = plugin.captureNonZeroColumns();
        if (nonZero >= 0) {
            out.append("observed: ").append(nonZero).append(" / ")
               .append(plugin.observationSize())
               .append(" columns non-zero, ")
               .append(plugin.captureLegalActions())
               .append(" legal action(s)\n");
        }

        Map<String, Integer> unclaimed = plugin.unclaimedRoles();
        if (unclaimed != null && !unclaimed.isEmpty()) {
            out.append("seen but not claimed:\n");
            unclaimed.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(8)
                .forEach(e -> out.append("  ").append(e.getKey())
                                 .append(" x").append(e.getValue()).append('\n'));
        }
        if (plugin.claimedCount() == 0) {
            out.append("nothing claimed yet -- /hydl spawn to create one.\n");
        }
        context.sendMessage(Message.raw(out.toString()));
    }
}
