package com.hytalerlbridge.policy;

import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Resolve every native ability binding once, off the world thread.
 *
 * <p><b>Why this class exists.</b> Without it the server deadlocks the first
 * time a policy-driven NPC is claimed, and it deadlocks <em>permanently</em>:
 * the world thread parks and the whole world stops — no ticks, no entity
 * updates, joins never complete, and the process stays alive looking healthy.
 * Measured with a thread dump on 2026-08-08:
 *
 * <pre>
 * "WorldThread - default"  WAITING (parking)
 *   parking to wait for &lt;0x40f09b528&gt; ReentrantReadWriteLock$NonfairSync
 *   ReentrantReadWriteLock$WriteLock.lock()
 *   AssetStore.loadAssets0(AssetStore.java:746)
 *   InteractionSupport.registeredCombatProbeRoot(:120)
 *   NativePolicyLifecycleFacade.resolve(:482) / capture(:90)
 *   ReflectiveBridgeCombatLifecycleEvidenceSource.capture(:106)
 *   LivePolicyPerception.sample(:50)
 *   PolicyControlSystem.tick(:155)
 *   World.tick(World.java:463)
 * </pre>
 *
 * <p>The world thread is already inside a tick holding a <em>read</em> lock on
 * the asset store, and {@code loadAssets} asks for the <em>write</em> lock.
 * {@link java.util.concurrent.locks.ReentrantReadWriteLock} does not permit
 * upgrading read to write, so the thread waits on a lock it is itself blocking.
 * Nothing ever releases it.
 *
 * <p><b>Why warming is sufficient, rather than a bridge fix.</b>
 * {@code InteractionSupport.registeredCombatProbeRoot} is memoised — it returns
 * early when the probe root is already in the asset map and only calls
 * {@code loadAssets} on the very first resolution of a given interaction id:
 *
 * <pre>
 * RootInteraction existing = RootInteraction.getAssetMap().getAsset(rootId);
 * if (existing != null) return existing;          // no lock taken
 * var result = RootInteraction.getAssetStore().loadAssets(...);   // first time only
 * </pre>
 *
 * <p>So the write lock is not inherent to the tick path; it is only ever taken
 * by whichever thread resolves an id first. Doing that here, during plugin
 * setup on the setup thread, means the world thread always hits the memo. The
 * bridge jar is not modified, which matters: its SHA-256 is pinned by the Gym
 * as the jar its native evidence corpus was recorded against, and replacing it
 * would invalidate that corpus.
 *
 * <p><b>Failure is not fatal.</b> A binding that cannot be resolved is counted
 * and reported, not thrown. The agent can run without any given ability, and
 * refusing to boot over one unresolvable id would trade a working agent for a
 * dead one. What must not happen — a silent skip that leaves the deadlock armed
 * — is prevented by reporting the count.
 */
public final class NativeBindingWarmup {

    /** Bridge-side resolver. Reflective: the bridge is an optional dependency. */
    private static final String SUPPORT_CLASS =
        "com.hytalerlbridge.nativebackend.support.InteractionSupport";

    private NativeBindingWarmup() {
    }

    /** What warming did, for the caller to log. */
    public record Result(int resolved, int failed, String detail) {

        public boolean warmed() {
            return resolved > 0 && failed == 0;
        }

        @Override
        public String toString() {
            if (!detail.isEmpty()) {
                return resolved + " resolved, " + failed + " failed: " + detail;
            }
            return resolved + " resolved, " + failed + " failed";
        }
    }

    /**
     * Resolve every ability binding the profile declares, for both actors.
     *
     * <p>Both actors, not just the agent: the target's abilities are resolved
     * by the same {@code capture} call on the same world thread, so warming
     * only the agent's would leave the deadlock armed behind the first
     * opponent action rather than removing it.
     *
     * @return what happened, never null; a bridge without the resolver yields
     *         zero resolved and an explanatory detail rather than an exception
     */
    public static Result warm(PerceptionProfile profile) {
        Class<?> support;
        try {
            support = Class.forName(SUPPORT_CLASS);
        } catch (ReflectiveOperationException absent) {
            return new Result(0, 0, "bridge resolver absent: " + absent);
        }
        return warm(profile, support);
    }

    /** Server-free seam for verifying the exact resolver coverage. */
    static Result warm(PerceptionProfile profile, Class<?> support) {
        Method resolveAbility;
        Method resolveInteraction;
        try {
            resolveAbility = support.getMethod(
                "resolveNativeAbilityBinding",
                String.class, String.class, String.class);
            resolveInteraction = support.getMethod(
                "resolveNativeInteractionBinding",
                String.class, String.class);
        } catch (ReflectiveOperationException absent) {
            return new Result(0, 0, "bridge resolver absent: " + absent);
        }

        // Deduplicated: the same interaction id appears once per actor row that
        // uses it, and resolving twice is wasted asset work on the setup path.
        Set<List<String>> triples = new LinkedHashSet<>();
        for (PerceptionProfile.AbilityBinding binding
                : profile.agentAbilityBindings()) {
            triples.add(List.of(binding.itemId(), binding.interactionId(),
                binding.interactionType()));
        }
        for (PerceptionProfile.AbilityBinding binding
                : profile.targetAbilityBindings()) {
            triples.add(List.of(binding.itemId(), binding.interactionId(),
                binding.interactionType()));
        }

        int resolved = 0;
        List<String> failures = new ArrayList<>();
        for (List<String> triple : triples) {
            try {
                resolveAbility.invoke(
                    null, triple.get(0), triple.get(1), triple.get(2));
                resolved++;
            } catch (ReflectiveOperationException | RuntimeException failure) {
                Throwable cause = failure.getCause() == null
                    ? failure : failure.getCause();
                failures.add(triple.get(1) + " (" + cause + ")");
            }
        }

        // Combat binding resolves Guard through the direct interaction path,
        // not the item-root ability path. Warm both actor rows so a future
        // policy-controlled target cannot reintroduce the same world-thread
        // lock upgrade through its Guard root.
        Set<List<String>> guards = new LinkedHashSet<>();
        addGuard(guards, profile.agentGuardBinding());
        addGuard(guards, profile.targetGuardBinding());
        for (List<String> pair : guards) {
            try {
                resolveInteraction.invoke(null, pair.get(0), pair.get(1));
                resolved++;
            } catch (ReflectiveOperationException | RuntimeException failure) {
                Throwable cause = failure.getCause() == null
                    ? failure : failure.getCause();
                failures.add(pair.get(0) + " (" + cause + ")");
            }
        }
        return new Result(resolved, failures.size(),
            String.join("; ", failures));
    }

    private static void addGuard(
        Set<List<String>> guards,
        PerceptionProfile.GuardBinding guard
    ) {
        if (guard != null) {
            guards.add(List.of(guard.interactionId(), guard.interactionType()));
        }
    }
}
