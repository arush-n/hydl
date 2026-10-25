package com.hytalerlbridge.soundprobe;

import com.hypixel.hytale.plugin.early.ClassTransformer;

import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.Label;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Early-plugin transformer for {@code SoundUtil}. It adds one static field and, at the entry of the two
 * terminal {@code playSoundEvent3d} overloads, inserts a JDK-only hand-off:
 *
 * <pre>
 * try {
 *   Consumer c = SoundUtil.hytalerl$soundProbe;
 *   if (c != null) c.accept(new Object[]{index, category, x, y, z, volume, pitch, predicate-or-null});
 * } catch (Throwable ignored) { }
 * </pre>
 *
 * <p>Only JDK types and the target's own field are referenced, so the inserted code links from any classloader.
 * The bridge owns the consumer. Every other class is returned unchanged.
 *
 * <p>Argument convention (CONFIRMED by javap of {@code TransformingClassLoader.transformAndDefine}, server
 * 0.7.2): {@code transform(name, internalName, bytes)} receives the dotted class name as {@code name}.
 */
public final class SoundProbeTransformer implements ClassTransformer {

    static final String DEFAULT_TARGET = "com.hypixel.hytale.server.core.universe.world.SoundUtil";
    /** Field added to the target: {@code public static volatile java.util.function.Consumer}. */
    static final String FIELD = "hytalerl$soundProbe";

    private static final String METHOD = "playSoundEvent3d";
    private static final String DESC_NO_PREDICATE =
            "(ILcom/hypixel/hytale/protocol/SoundCategory;DDDFFLcom/hypixel/hytale/component/ComponentAccessor;)V";
    private static final String DESC_WITH_PREDICATE =
            "(ILcom/hypixel/hytale/protocol/SoundCategory;DDDFFLjava/util/function/Predicate;"
                    + "Lcom/hypixel/hytale/component/ComponentAccessor;)V";

    private static final ClassDesc CONSUMER = ClassDesc.of("java.util.function.Consumer");
    private static final ClassDesc OBJECT = ConstantDescs.CD_Object;
    private static final ClassDesc INTEGER = ClassDesc.of("java.lang.Integer");
    private static final ClassDesc DOUBLE = ClassDesc.of("java.lang.Double");
    private static final ClassDesc FLOAT = ClassDesc.of("java.lang.Float");
    private static final int PAYLOAD_LENGTH = 8;

    // Local slots of the static parameters. int=0, category=1, x=2-3, y=4-5, z=6-7, volume=8, pitch=9,
    // then the predicate (overload 2 only) at 10; the accessor is not passed on.
    private static final int SLOT_INDEX = 0;
    private static final int SLOT_CATEGORY = 1;
    private static final int SLOT_X = 2;
    private static final int SLOT_Y = 4;
    private static final int SLOT_Z = 6;
    private static final int SLOT_VOLUME = 8;
    private static final int SLOT_PITCH = 9;
    private static final int SLOT_PREDICATE = 10;

    private static volatile boolean loggedFailure;

    private final String targetDotted;

    public SoundProbeTransformer() {
        this(DEFAULT_TARGET);
    }

    /** Test seam: target a class other than SoundUtil (a stub with the same method signatures). */
    SoundProbeTransformer(String targetDotted) {
        this.targetDotted = targetDotted;
    }

    @Override
    public byte[] transform(String name, String internalName, byte[] bytes) {
        if (!targetDotted.equals(name)) {
            return bytes;
        }
        try {
            // One pass per edit. Chaining transforms with ClassTransform.andThen silently dropped the
            // insertion when tested (JDK 25), so each pass re-parses the output of the previous one.
            ClassFile cf = ClassFile.of();
            ClassDesc self = ClassDesc.of(targetDotted);
            byte[] out = cf.transformClass(cf.parse(bytes), addProbeField());
            out = cf.transformClass(cf.parse(out), probeTransform(self, DESC_NO_PREDICATE, false));
            return cf.transformClass(cf.parse(out), probeTransform(self, DESC_WITH_PREDICATE, true));
        } catch (Throwable t) {
            if (!loggedFailure) {
                loggedFailure = true;
                System.err.println("[HytaleRLSoundProbe] could not transform " + name
                        + "; leaving it unchanged: " + t);
                t.printStackTrace();
            }
            return bytes;
        }
    }

    private static ClassTransform addProbeField() {
        return ClassTransform.endHandler(builder -> builder.withField(FIELD, CONSUMER,
                ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC | ClassFile.ACC_VOLATILE));
    }

    /** Inserts the hand-off at the entry of each {@code playSoundEvent3d} body with this descriptor. */
    private static ClassTransform probeTransform(ClassDesc self, String descriptor, boolean withPredicate) {
        return ClassTransform.transformingMethodBodies(
                method -> METHOD.equals(method.methodName().stringValue())
                        && descriptor.equals(method.methodType().stringValue()),
                insertProbeCall(self, withPredicate));
    }

    /** Inserts the hand-off before the first element of each matched method body. */
    private static CodeTransform insertProbeCall(ClassDesc self, boolean withPredicate) {
        return CodeTransform.ofStateful(() -> {
            boolean[] inserted = {false};
            return (builder, element) -> {
                if (!inserted[0]) {
                    inserted[0] = true;
                    emitProbeCall(builder, self, withPredicate);
                }
                builder.with(element);
            };
        });
    }

    /**
     * Emits the hand-off from the method's own arguments. All of it sits in a catch-all, so a failing or
     * throwing consumer falls through to the original body.
     */
    private static void emitProbeCall(CodeBuilder builder, ClassDesc self, boolean withPredicate) {
        builder.trying(
                block -> {
                    block.getstatic(self, FIELD, CONSUMER);
                    int consumerSlot = block.allocateLocal(TypeKind.REFERENCE);
                    block.astore(consumerSlot);
                    Label skip = block.newLabel();
                    block.aload(consumerSlot);
                    block.ifnull(skip);

                    block.aload(consumerSlot);
                    block.loadConstant(PAYLOAD_LENGTH);
                    block.anewarray(OBJECT);
                    block.dup();
                    block.loadConstant(0);
                    block.loadLocal(TypeKind.INT, SLOT_INDEX);
                    box(block, INTEGER, ConstantDescs.CD_int);
                    block.aastore();
                    block.dup();
                    block.loadConstant(1);
                    block.loadLocal(TypeKind.REFERENCE, SLOT_CATEGORY);
                    block.aastore();
                    block.dup();
                    block.loadConstant(2);
                    block.loadLocal(TypeKind.DOUBLE, SLOT_X);
                    box(block, DOUBLE, ConstantDescs.CD_double);
                    block.aastore();
                    block.dup();
                    block.loadConstant(3);
                    block.loadLocal(TypeKind.DOUBLE, SLOT_Y);
                    box(block, DOUBLE, ConstantDescs.CD_double);
                    block.aastore();
                    block.dup();
                    block.loadConstant(4);
                    block.loadLocal(TypeKind.DOUBLE, SLOT_Z);
                    box(block, DOUBLE, ConstantDescs.CD_double);
                    block.aastore();
                    block.dup();
                    block.loadConstant(5);
                    block.loadLocal(TypeKind.FLOAT, SLOT_VOLUME);
                    box(block, FLOAT, ConstantDescs.CD_float);
                    block.aastore();
                    block.dup();
                    block.loadConstant(6);
                    block.loadLocal(TypeKind.FLOAT, SLOT_PITCH);
                    box(block, FLOAT, ConstantDescs.CD_float);
                    block.aastore();
                    block.dup();
                    block.loadConstant(7);
                    if (withPredicate) {
                        block.loadLocal(TypeKind.REFERENCE, SLOT_PREDICATE);
                    } else {
                        block.aconst_null();
                    }
                    block.aastore();
                    block.invokeinterface(CONSUMER, "accept", MethodTypeDesc.of(ConstantDescs.CD_void, OBJECT));

                    block.labelBinding(skip);
                },
                catches -> catches.catchingAll(handler -> handler.pop()));
    }

    /** Boxes the primitive on the stack with its wrapper's {@code valueOf}. */
    private static void box(CodeBuilder block, ClassDesc wrapper, ClassDesc primitive) {
        block.invokestatic(wrapper, "valueOf", MethodTypeDesc.of(wrapper, primitive));
    }
}
