package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.AddReason;
import com.hypixel.hytale.component.ComponentAccessor;
import com.hypixel.hytale.component.Holder;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.server.core.event.events.ecs.DamageBlockEvent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.item.ItemComponent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;
import org.joml.Vector3i;

/** Records fixture-created drops and cancels target redirects outside it. */
final class ExplosionMutationAccessor {

    private final ComponentAccessor<EntityStore> accessor;
    private final List<ExplosionMutationDrops.ItemReading> createdItems =
        new ArrayList<>();
    private boolean redirected;

    @SuppressWarnings("unchecked")
    ExplosionMutationAccessor(
        ComponentAccessor<EntityStore> delegate,
        Vector3i controlledTarget
    ) {
        accessor = (ComponentAccessor<EntityStore>) Proxy.newProxyInstance(
            ComponentAccessor.class.getClassLoader(),
            new Class<?>[] {ComponentAccessor.class},
            (proxy, method, arguments) -> invoke(
                delegate,
                controlledTarget,
                method,
                arguments
            )
        );
    }

    ComponentAccessor<EntityStore> accessor() {
        return accessor;
    }

    boolean redirected() {
        return redirected;
    }

    List<ExplosionMutationDrops.ItemReading> createdItems() {
        return List.copyOf(createdItems);
    }

    private Object invoke(
        ComponentAccessor<EntityStore> delegate,
        Vector3i controlledTarget,
        Method method,
        Object[] arguments
    ) throws Throwable {
        Object result;
        try {
            result = method.invoke(delegate, arguments);
        } catch (InvocationTargetException exception) {
            throw exception.getCause();
        }
        if (
            method.getName().equals("addEntities")
                && arguments != null
                && arguments.length == 2
                && arguments[0] instanceof Holder<?>[] holders
                && arguments[1] == AddReason.SPAWN
                && result instanceof Ref<?>[] references
        ) {
            recordCreatedItems(holders, references);
        }
        if (
            method.getName().equals("invoke")
                && arguments != null
                && arguments.length == 1
                && arguments[0] instanceof DamageBlockEvent event
        ) {
            Vector3i target = event.getTargetBlock();
            if (
                target.x != controlledTarget.x
                    || target.y != controlledTarget.y
                    || target.z != controlledTarget.z
            ) {
                redirected = true;
                event.setCancelled(true);
            }
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private void recordCreatedItems(
        Holder<?>[] holders,
        Ref<?>[] references
    ) {
        if (holders.length != references.length) {
            throw new IllegalStateException(
                "native addEntities returned a mismatched reference count"
            );
        }
        for (int index = 0; index < holders.length; index++) {
            Holder<EntityStore> holder = (Holder<EntityStore>) holders[index];
            ItemComponent item = holder.getComponent(
                ItemComponent.getComponentType()
            );
            TransformComponent transform = holder.getComponent(
                TransformComponent.getComponentType()
            );
            if (item == null || transform == null || references[index] == null) {
                continue;
            }
            Ref<EntityStore> reference = (Ref<EntityStore>) references[index];
            if (!reference.isValid()) {
                continue;
            }
            ItemStack stack = item.getItemStack();
            Vector3d position = transform.getPosition();
            createdItems.add(new ExplosionMutationDrops.ItemReading(
                reference,
                stack,
                position.x,
                position.y,
                position.z
            ));
        }
    }
}
