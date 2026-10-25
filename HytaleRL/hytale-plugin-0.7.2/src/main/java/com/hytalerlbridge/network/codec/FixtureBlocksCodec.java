package com.hytalerlbridge.network.codec;

import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getField;

import com.hytalerlbridge.environment.EnvironmentOptions.FixtureBlock;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.msgpack.value.Value;

/** Decode the fixture-only reset option fixture_blocks: [x, y, z, "AssetId"] entries. */
public final class FixtureBlocksCodec {

    public static final String FIELD = "fixture_blocks";

    private FixtureBlocksCodec() {}

    /** Returns null when the option is absent; ranges are checked by {@link com.hytalerlbridge.environment.EnvironmentOptions}. */
    public static List<FixtureBlock> decode(Map<Value, Value> options) {
        Value value = getField(options, FIELD);
        if (value == null || value.isNilValue()) return null;
        if (!value.isArrayValue()) {
            throw new IllegalArgumentException(FIELD + " must be an array");
        }
        ArrayList<FixtureBlock> blocks = new ArrayList<>(value.asArrayValue().size());
        for (Value raw : value.asArrayValue()) {
            if (!raw.isArrayValue() || raw.asArrayValue().size() != 4) {
                throw new IllegalArgumentException(
                    FIELD + " entries must be [x, y, z, \"AssetId\"]"
                );
            }
            List<Value> parts = raw.asArrayValue().list();
            int[] cell = new int[3];
            for (int i = 0; i < 3; i++) {
                if (!parts.get(i).isIntegerValue()) {
                    throw new IllegalArgumentException(FIELD + " coordinates must be integers");
                }
                cell[i] = checkedInt(
                    parts.get(i).asIntegerValue().asLong(),
                    FIELD + " coordinate"
                );
            }
            if (!parts.get(3).isStringValue()) {
                throw new IllegalArgumentException(FIELD + " asset ids must be strings");
            }
            blocks.add(new FixtureBlock(
                cell[0],
                cell[1],
                cell[2],
                parts.get(3).asStringValue().asString()
            ));
        }
        return List.copyOf(blocks);
    }
}
