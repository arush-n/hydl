package com.hytalerlbridge.nativebackend;

import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;

import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.component.Component;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Holder;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.modules.entity.player.ChunkTracker;
import com.hypixel.hytale.server.core.modules.entity.tracker.EntityTrackerSystems;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import org.bson.BsonDocument;
import org.bson.BsonString;

/**
 * Installs the smallest server-side Adventure client context required by
 * native player interaction chains.
 *
 * <p>This context is opt-in at reset. It is not an authenticated network
 * player and must never be advertised as one. Its inventory components are
 * the controlled entity's existing containers, so native interactions and
 * crafting mutate the same authoritative item state observed by the bridge.</p>
 */
final class NativeHeadlessWorldVerbContext {

    static final String HEADLESS_USERNAME = "HytaleRLHeadlessWorldVerbs";

    private NativeHeadlessWorldVerbContext() {}

    static PlayerRef install(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        PlayerRef packetAnchor
    ) {
        if (reference == null || !reference.isValid()) {
            throw new IllegalStateException(
                "Headless World verbs have no controlled entity"
            );
        }
        if (
            store.getComponent(reference, PlayerRef.getComponentType()) != null
                || store.getComponent(reference, Player.getComponentType())
                    != null
        ) {
            throw new IllegalStateException(
                "Controlled entity already has a Player context"
            );
        }
        if (packetAnchor == null) {
            throw new IllegalStateException(
                "Headless World verbs have no packet sink"
            );
        }

        UUIDComponent identity = store.getComponent(
            reference,
            UUIDComponent.getComponentType()
        );
        if (identity == null) {
            throw new IllegalStateException(
                "Headless World verbs have no entity identity"
            );
        }

        Player player = Player.CODEC.decode(
            new BsonDocument(
                "GameMode",
                new BsonString(
                    com.hypixel.hytale.protocol.GameMode.Adventure.name()
                )
            )
        );
        if (
            player == null
                || player.getGameMode()
                    != com.hypixel.hytale.protocol.GameMode.Adventure
        ) {
            throw new IllegalStateException(
                "Headless World Player did not decode in Adventure mode"
            );
        }

        PlayerRef playerRef = new PlayerRef(
            null,
            identity.getUuid(),
            HEADLESS_USERNAME,
            "en-US",
            packetAnchor.getPacketHandler(),
            new ChunkTracker()
        );
        player.init(identity.getUuid(), playerRef);
        store.addComponent(
            reference,
            PlayerRef.getComponentType(),
            playerRef
        );

        Holder<EntityStore> inventoryHolder =
            store.getRegistry().newHolder();
        copyInventoryComponent(
            reference,
            store,
            inventoryHolder,
            InventoryComponent.Storage.getComponentType()
        );
        copyInventoryComponent(
            reference,
            store,
            inventoryHolder,
            InventoryComponent.Armor.getComponentType()
        );
        copyInventoryComponent(
            reference,
            store,
            inventoryHolder,
            InventoryComponent.Hotbar.getComponentType()
        );
        copyInventoryComponent(
            reference,
            store,
            inventoryHolder,
            InventoryComponent.Utility.getComponentType()
        );
        copyInventoryComponent(
            reference,
            store,
            inventoryHolder,
            InventoryComponent.Tool.getComponentType()
        );
        copyInventoryComponent(
            reference,
            store,
            inventoryHolder,
            InventoryComponent.Backpack.getComponentType()
        );
        player.getInventory().backwardsCompatHook(inventoryHolder);

        if (
            store.getComponent(
                reference,
                EntityTrackerSystems.EntityViewer.getComponentType()
            ) == null
        ) {
            store.addComponent(
                reference,
                EntityTrackerSystems.EntityViewer.getComponentType(),
                new EntityTrackerSystems.EntityViewer(
                    0,
                    packetAnchor.getPacketHandler()
                )
            );
        }
        store.addComponent(reference, Player.getComponentType(), player);
        if (
            store.getComponent(reference, CraftingManager.getComponentType())
                == null
        ) {
            store.addComponent(
                reference,
                CraftingManager.getComponentType(),
                new CraftingManager()
            );
        }

        InteractionManager manager = interactionManager(reference, store);
        if (manager == null) {
            throw new IllegalStateException(
                "Headless World Player has no InteractionManager"
            );
        }
        // A Player component changes several authored operations (notably
        // Charging and Select) to their client-synchronised server path.  The
        // bridge supplies that sync in NativeSyntheticClientSystem.  Running
        // the NPC simulator at the same time is invalid: it predicts the
        // player-specific operation graph differently and Hytale aborts the
        // entity on the first operation-counter mismatch.
        manager.setHasRemoteClient(headlessHasRemoteClient());
        return playerRef;
    }

    static boolean headlessHasRemoteClient() {
        return true;
    }

    static void uninstall(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        if (reference == null || !reference.isValid()) return;
        InteractionManager manager = interactionManager(reference, store);
        if (manager != null) {
            for (var chain : java.util.List.copyOf(
                manager.getChains().values()
            )) {
                manager.cancelChains(chain);
            }
            manager.setHasRemoteClient(false);
        }
        // Remove the legacy Player identity before the NPC entity is removed
        // with its world. Leaving both legacy entity types attached makes
        // Hytale run both teardown paths against the same entity.
        store.removeComponentIfExists(
            reference,
            Player.getComponentType()
        );
        store.removeComponentIfExists(
            reference,
            PlayerRef.getComponentType()
        );
    }

    /** Whether this exact bridge-owned synthetic Adventure context is live. */
    static boolean installed(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        if (reference == null || !reference.isValid() || store == null) {
            return false;
        }
        PlayerRef playerRef = store.getComponent(
            reference, PlayerRef.getComponentType());
        Player player = store.getComponent(
            reference, Player.getComponentType());
        return playerRef != null
            && HEADLESS_USERNAME.equals(playerRef.getUsername())
            && player != null
            && player.getGameMode()
                == com.hypixel.hytale.protocol.GameMode.Adventure;
    }

    private static <T extends Component<EntityStore>>
    void copyInventoryComponent(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        Holder<EntityStore> holder,
        ComponentType<EntityStore, T> componentType
    ) {
        T component = store.getComponent(reference, componentType);
        if (component != null) {
            holder.putComponent(componentType, component);
        }
    }
}
