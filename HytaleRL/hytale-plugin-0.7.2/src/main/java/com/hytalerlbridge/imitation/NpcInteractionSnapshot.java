package com.hytalerlbridge.imitation;

import java.util.UUID;

/** Immutable public lifecycle state for one native interaction chain. */
public record NpcInteractionSnapshot(
    String source,
    String type,
    String baseType,
    int chainId,
    String initialRootId,
    String rootId,
    String serverState,
    String clientState,
    String finalState,
    float timeSeconds,
    float timeShift,
    int operationCounter,
    int simulatedOperationCounter,
    int operationIndex,
    int clientOperationIndex,
    int callDepth,
    int simulatedCallDepth,
    boolean predicted,
    boolean requiresClient,
    boolean firstRun,
    boolean preTicked,
    boolean desynced,
    UUID targetUuid
) {
    public static final String LAYOUT =
        "source:utf8,type:utf8,base_type:utf8,chain_id:i32,initial_root_id:utf8,root_id:utf8,"
            + "server_state:utf8,client_state:utf8,final_state:utf8,time_seconds:f32,"
            + "time_shift:f32,operation_counter:i32,simulated_operation_counter:i32,"
            + "operation_index:i32,client_operation_index:i32,call_depth:i32,"
            + "simulated_call_depth:i32,predicted:bool,requires_client:bool,"
            + "first_run:bool,pre_ticked:bool,desynced:bool,target_uuid:uuid?";
    public static final int WIDTH = 23;

    public NpcInteractionSnapshot {
        source = text(source);
        if (
            !source.equals("interaction_manager")
                && !source.equals("combat_support")
                && !source.equals("interaction_manager+combat_support")
        ) {
            throw new IllegalArgumentException("unknown interaction snapshot source");
        }
        type = text(type);
        baseType = text(baseType);
        initialRootId = text(initialRootId);
        rootId = text(rootId);
        serverState = text(serverState);
        clientState = text(clientState);
        finalState = text(finalState);
        if (!Float.isFinite(timeSeconds) || !Float.isFinite(timeShift)) {
            throw new IllegalArgumentException("interaction times must be finite");
        }
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
