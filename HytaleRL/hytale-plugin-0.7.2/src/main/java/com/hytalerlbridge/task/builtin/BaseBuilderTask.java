package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.observation.ObservationEncoding;
import com.hytalerlbridge.task.RLTask;
import com.hytalerlbridge.world.*;

import java.util.*;

/**
 * NPC Base Builder task: an autonomous agent that gathers resources,
 * crafts items, and constructs a defended base in Hytale's Zone 1
 * (Emerald Wilds).
 *
 * Hierarchical reward structure:
 *   - Resource gathering: +0.5 per resource collected
 *   - Crafting:          +2.0 per successful craft
 *   - Block placement:   +1.0 per structural block placed
 *   - Shelter score:     +5.0 when shelter criteria met (walls + roof)
 *   - Defense bonus:     +3.0 for defensive structures (fences, height)
 *   - Survival:          +0.1 per tick alive
 *   - Damage penalty:    -2.0 per HP lost
 *   - Night survival:    +10.0 for surviving a full night cycle
 *   - Completion:        +100.0 when base score threshold reached
 */
public class BaseBuilderTask implements RLTask {

    // Scoring thresholds
    private static final int BASE_SCORE_THRESHOLD = 50;
    private static final int DAY_LENGTH = 24000; // ticks per day cycle
    private static final int NIGHT_START = 13000;
    private static final int NIGHT_END = 23000;
    private static final double MOVE_SPEED = 0.15;
    private static final int NEARBY_BLOCK_RADIUS = 4;
    private static final int NEARBY_ENTITY_RADIUS = 16;
    private static final int MAX_NEARBY_ENTITIES = 20;

    // Agent state
    private double x, y, z;
    private double vx, vy, vz;
    private double yaw, pitch;
    private double health, maxHealth;
    private int foodBuffTimer;  // Hytale has no hunger bar; food gives temporary buffs
    private double stamina;     // base 10, not 100 (Hytale Early Access)
    private double mana;        // placeholder — not yet functional in EA
    private int[] inventory;

    // World
    private WorldGrid world;
    private List<SimEntity> entities;
    private CraftingRegistry crafting;
    private Random rng;

    // Tracking
    private int tickCount;
    private int timeOfDay;
    private double prevHealth;
    private boolean shelterBuilt;
    private boolean shelterRewardAwarded;
    private int nightsSurvived;
    private boolean wasNight;
    private double baseScore;
    private double pendingReward;
    private int curriculumPhase;
    private boolean completionAwarded;

    // Structure tracking
    private Map<BlockPos, BlockType> structuralBlocks;
    private int wallBlockCount;
    private int roofBlockCount;
    private boolean hasDoor;
    private boolean hasWorkbench;
    private boolean hasFurnace;
    private boolean hasChest;
    private boolean hasTorch;

    @Override
    public String description() {
        return "Build an autonomous base: gather resources, craft, build shelter, survive nights.";
    }

    @Override
    public void reset(long seed) {
        reset(seed, -1);
    }

    @Override
    public void reset(long seed, int requestedCurriculumPhase) {
        rng = new Random(seed);
        curriculumPhase = requestedCurriculumPhase < 0
            ? 3
            : clamp(requestedCurriculumPhase, 0, 3);

        // Agent starts at ground level
        x = 0; y = 65; z = 0;
        vx = 0; vy = 0; vz = 0;
        yaw = 0; pitch = 0;
        health = 100; maxHealth = 100;
        foodBuffTimer = 0;
        stamina = 10;
        mana = 100;
        inventory = new int[36];

        // Initialize world with terrain
        world = new WorldGrid(64);
        generateTerrain();

        // Spawn entities
        entities = new ArrayList<>();
        crafting = new CraftingRegistry();

        // Tracking
        tickCount = 0;
        timeOfDay = 0; // start at dawn
        prevHealth = health;
        shelterBuilt = false;
        shelterRewardAwarded = false;
        nightsSurvived = 0;
        wasNight = false;
        baseScore = 0;
        pendingReward = 0;
        completionAwarded = false;
        structuralBlocks = new HashMap<>();
        wallBlockCount = 0;
        roofBlockCount = 0;
        hasDoor = false;
        hasWorkbench = false;
        hasFurnace = false;
        hasChest = false;
        hasTorch = false;
    }

    private void generateTerrain() {
        // Scatter trees (Zone 1 Emerald Wilds softwood trees)
        for (int i = 0; i < 15; i++) {
            int tx = rng.nextInt(40) - 20;
            int tz = rng.nextInt(40) - 20;
            world.generateTree(rng, tx, tz);
        }
        // Generate ore underground following Hytale's tiered progression
        world.generateOre(rng, BlockType.COPPER_ORE, 8, 50, 62);
        world.generateOre(rng, BlockType.IRON_ORE, 5, 40, 58);
        world.generateOre(rng, BlockType.THORIUM_ORE, 3, 30, 50);
        world.generateOre(rng, BlockType.COBALT_ORE, 2, 25, 45);
        world.generateOre(rng, BlockType.ADAMANTITE_ORE, 1, 15, 35);
        world.generateOre(rng, BlockType.MITHRIL_ORE, 1, 10, 25);
    }

    @Override
    public void applyAction(AgentAction action) {
        // Movement (yaw-relative)
        double rad = Math.toRadians(yaw);
        double dx = 0, dz = 0;
        if (action.forward()) { dx -= Math.sin(rad) * MOVE_SPEED; dz -= Math.cos(rad) * MOVE_SPEED; }
        if (action.back())    { dx += Math.sin(rad) * MOVE_SPEED; dz += Math.cos(rad) * MOVE_SPEED; }
        if (action.left())    { dx -= Math.cos(rad) * MOVE_SPEED; dz += Math.sin(rad) * MOVE_SPEED; }
        if (action.right())   { dx += Math.cos(rad) * MOVE_SPEED; dz -= Math.sin(rad) * MOVE_SPEED; }

        double newX = x + dx;
        double newZ = z + dz;

        // Collision: check if the destination block is solid at agent's feet level
        int footY = (int) Math.floor(y);
        if (!world.isSolid((int) Math.floor(newX), footY, (int) Math.floor(newZ))) {
            x = newX;
            z = newZ;
        } else if (!world.isSolid((int) Math.floor(newX), footY + 1, (int) Math.floor(newZ))) {
            // Auto-step up one block
            x = newX;
            z = newZ;
            y += 1.0;
        }

        // Gravity
        int belowY = (int) Math.floor(y) - 1;
        if (!world.isSolid((int) Math.floor(x), belowY, (int) Math.floor(z)) && y > 1) {
            vy -= 0.08; // gravity
        } else {
            vy = 0;
        }
        y += vy;
        y = Math.max(1, y);

        // Jump (consumes stamina — base stamina is 10 in Hytale)
        if (action.jump() && vy == 0 && stamina >= 1) {
            vy = 0.4;
            stamina = Math.max(0, stamina - 1);
        }

        // Camera
        yaw += action.cameraDeltaYaw();
        pitch = Math.max(-90, Math.min(90, pitch + action.cameraDeltaPitch()));

        // --- Targeted block placement ---
        if (action.hasPlacement()) {
            int bx = (int) Math.floor(x) + clamp(action.placeBlockX(), -3, 3);
            int by = (int) Math.floor(y) + clamp(action.placeBlockY(), -3, 3);
            int bz = (int) Math.floor(z) + clamp(action.placeBlockZ(), -3, 3);
            BlockType bt = BlockType.fromId(action.placeBlockType());

            if (bt != BlockType.AIR && hasItemInInventory(bt.id()) && world.getBlock(bx, by, bz) == BlockType.AIR) {
                world.setBlock(bx, by, bz, bt);
                removeFromInventory(bt.id());
                trackStructuralBlock(bt, bx, by, bz);
                addPhaseReward(1.0, 2);
            }
        }

        // --- Legacy ray-cast placement (use action) ---
        if (action.use() && !action.hasPlacement()) {
            double lookRad = Math.toRadians(yaw);
            double pitchRad = Math.toRadians(pitch);
            int bx = (int) Math.round(x - Math.sin(lookRad) * 2);
            int by = (int) Math.round(y - Math.sin(pitchRad) * 2);
            int bz = (int) Math.round(z + Math.cos(lookRad) * 2);

            int selectedItem = action.hotbarSlot() < inventory.length ? inventory[action.hotbarSlot()] : 0;
            if (selectedItem > 0 && world.getBlock(bx, by, bz) == BlockType.AIR) {
                BlockType bt = BlockType.fromId(selectedItem);
                world.setBlock(bx, by, bz, bt);
                removeFromInventory(selectedItem);
                trackStructuralBlock(bt, bx, by, bz);
                addPhaseReward(1.0, 2);
            }
        }

        // --- Targeted block break ---
        if (action.hasBreak()) {
            int bx = (int) Math.floor(x) + clamp(action.breakBlockX(), -3, 3);
            int by = (int) Math.floor(y) + clamp(action.breakBlockY(), -3, 3);
            int bz = (int) Math.floor(z) + clamp(action.breakBlockZ(), -3, 3);
            BlockType existing = world.getBlock(bx, by, bz);
            if (existing != BlockType.AIR) {
                untrackStructuralBlock(bx, by, bz);
            }

            if (existing == BlockType.STONE) {
                // Mining stone gives quartzite
                addToInventory(BlockType.QUARTZITE.id());
                world.setBlock(bx, by, bz, BlockType.AIR);
                addPhaseReward(0.3, 0);
            } else if (existing == BlockType.LOG || existing == BlockType.WOOD) {
                // Felling trees drops log + resin + sticks (Hytale tree collapse)
                addToInventory(existing.id());
                addToInventory(BlockType.RESIN.id());
                addToInventory(BlockType.STICKS.id());
                world.setBlock(bx, by, bz, BlockType.AIR);
                addPhaseReward(0.5, 0);
            } else if (existing == BlockType.LEAVES) {
                // Leaves drop plant fiber
                addToInventory(BlockType.PLANT_FIBER.id());
                world.setBlock(bx, by, bz, BlockType.AIR);
                addPhaseReward(0.2, 0);
            } else if (existing == BlockType.GRASS) {
                // Grass drops plant fiber
                addToInventory(BlockType.PLANT_FIBER.id());
                world.setBlock(bx, by, bz, BlockType.AIR);
                addPhaseReward(0.1, 0);
            } else if (existing == BlockType.DIRT) {
                // Dirt can be excavated to reach stone and ore layers.
                world.setBlock(bx, by, bz, BlockType.AIR);
                addPhaseReward(0.1, 0);
            } else if (existing != BlockType.AIR && existing != BlockType.DIRT) {
                // All other blocks: collect as-is
                addToInventory(existing.id());
                world.setBlock(bx, by, bz, BlockType.AIR);
                addPhaseReward(0.5, 0);
            }
        }

        // --- Attack nearby hostile ---
        if (action.attack()) {
            for (SimEntity entity : entities) {
                if (entity.isAlive() && entity.distanceTo(x, z) < 3.0) {
                    double damage = 9 + rng.nextDouble() * 17; // iron sword: 9-26 dmg (left swing to thrust)
                    entity.takeDamage(damage);
                    addPhaseReward(1.5, 2);
                    break;
                }
            }
        }

        // --- Crafting ---
        if (action.hasCraft()) {
            CraftingRecipe recipe = crafting.get(action.craftRecipeId());
            if (recipe != null && recipe.canCraft(inventory)) {
                recipe.consumeInputs(inventory);
                recipe.addOutput(inventory);
                addPhaseReward(2.0, 1);
            }
        }
    }

    @Override
    public void tick() {
        prevHealth = health;

        tickCount++;
        timeOfDay = (tickCount * 10) % DAY_LENGTH; // 10x speed for faster cycles

        // Stamina regeneration (~1/sec at 20 ticks/sec = 0.05/tick)
        stamina = Math.min(10, stamina + 0.05);

        // Mana regeneration (placeholder — not functional in EA)
        mana = Math.min(100, mana + 0.2);

        // Food buff timer countdown (Hytale has no hunger bar;
        // food gives temporary buffs like health regen)
        if (foodBuffTimer > 0) {
            foodBuffTimer--;
            // Active food buff: accelerated health regen
            if (tickCount % 30 == 0 && health < maxHealth) {
                health = Math.min(maxHealth, health + 2.0);
            }
        }

        // Natural health regeneration (slow, always active in Hytale)
        if (health < maxHealth && health > 0 && tickCount % 100 == 0) {
            health = Math.min(maxHealth, health + 0.5);
        }

        // Night cycle: spawn hostile creatures
        boolean isNight = timeOfDay >= NIGHT_START && timeOfDay < NIGHT_END;
        if (isNight && !wasNight && curriculumPhase >= 2) {
            spawnNightCreatures();
        }
        if (!isNight && wasNight) {
            // Survived the night
            nightsSurvived++;
            addPhaseReward(10.0, 3);
            // Despawn remaining night creatures
            entities.removeIf(e -> e.getType().isHostile());
        }
        wasNight = isNight;

        // Tick entities
        for (SimEntity entity : entities) {
            entity.tick(x, z, rng);
            double damage = entity.tryAttack(x, z, rng);
            if (damage > 0) {
                health = Math.max(0, health - damage);
            }
        }
        entities.removeIf(e -> !e.isAlive());

        // Shelter detection
        checkShelter();

        // Survival reward
        if (health > 0) {
            addPhaseReward(0.1, 3);
        }

        // Damage penalty
        double damageTaken = prevHealth - health;
        if (damageTaken > 0) {
            if (curriculumPhase >= 2) pendingReward -= damageTaken * 2.0;
        }

        // Update base score
        updateBaseScore();
    }

    @Override
    public Observation observe() {
        int ix = (int) Math.floor(x);
        int iy = (int) Math.floor(y);
        int iz = (int) Math.floor(z);

        // Nearby blocks
        List<int[]> nearbyBlocks = world.getBlocksInRadius(ix, iy, iz, NEARBY_BLOCK_RADIUS);

        // Nearby entities
        List<int[]> nearbyEntities = new ArrayList<>();
        entities.stream()
            .filter(e -> e.isAlive() && e.distanceTo(x, z) < NEARBY_ENTITY_RADIUS)
            .limit(MAX_NEARBY_ENTITIES)
            .forEach(e -> nearbyEntities.add(new int[]{
                e.getType().id(),
                ObservationEncoding.encodeNearbyEntityScalar(e.getX() - x),
                ObservationEncoding.encodeNearbyEntityScalar(e.getZ() - z),
                ObservationEncoding.encodeNearbyEntityScalar(e.getHealth())
            }));

        // Count craftable recipes
        int craftable = 0;
        for (CraftingRecipe recipe : crafting.getAll()) {
            if (recipe.canCraft(inventory)) craftable++;
        }

        return new Observation(
            x, y, z, vx, vy, vz, yaw, pitch,
            health, maxHealth, foodBuffTimer, stamina, mana,
            inventory.clone(),
            nearbyBlocks, nearbyEntities,
            timeOfDay, craftable
        );
    }

    @Override
    public double computeReward() {
        double reward = pendingReward;
        pendingReward = 0;

        // Completion bonus
        if (curriculumPhase >= 2
            && baseScore >= BASE_SCORE_THRESHOLD
            && !completionAwarded) {
            completionAwarded = true;
            reward += 100.0;
        }

        return reward;
    }

    @Override
    public boolean isTerminated() {
        return health <= 0
            || (curriculumPhase >= 2 && baseScore >= BASE_SCORE_THRESHOLD);
    }

    @Override
    public Map<String, Object> info() {
        return Map.of(
            "base_score", baseScore,
            "curriculum_phase", curriculumPhase,
            "nights_survived", nightsSurvived,
            "shelter_built", shelterBuilt
        );
    }

    // --- Helper methods ---

    private void spawnNightCreatures() {
        int count = curriculumPhase == 2 ? 1 + rng.nextInt(2) : 3 + rng.nextInt(4);
        for (int i = 0; i < count; i++) {
            double angle = rng.nextDouble() * Math.PI * 2;
            double dist = 15 + rng.nextDouble() * 15;
            double ex = x + Math.cos(angle) * dist;
            double ez = z + Math.sin(angle) * dist;
            EntityType type = rng.nextDouble() < 0.6 ? EntityType.TRORK : EntityType.OUTLANDER;
            entities.add(SimEntity.withDefaultHP(type, ex, 65, ez, 0.06));
        }
    }

    private void trackStructuralBlock(BlockType bt, int bx, int by, int bz) {
        structuralBlocks.put(new BlockPos(bx, by, bz), bt);
        recomputeStructureState();
    }

    private void untrackStructuralBlock(int bx, int by, int bz) {
        if (structuralBlocks.remove(new BlockPos(bx, by, bz)) != null) {
            recomputeStructureState();
        }
    }

    private void recomputeStructureState() {
        wallBlockCount = 0;
        roofBlockCount = 0;
        hasDoor = false;
        hasWorkbench = false;
        hasFurnace = false;
        hasChest = false;
        hasTorch = false;

        int groundY = world.getGroundLevel() + 1;
        structuralBlocks.forEach((position, type) -> {
            if (position.y() >= groundY && position.y() <= groundY + 3
                && (type == BlockType.SOFTWOOD_PLANKS
                    || type == BlockType.QUARTZITE
                    || type == BlockType.LOG)) {
                wallBlockCount++;
            }
            if (position.y() > groundY + 3
                && (type == BlockType.SOFTWOOD_PLANKS || type == BlockType.ROOF_BLOCK)) {
                roofBlockCount++;
            }
            if (type == BlockType.DOOR) hasDoor = true;
            if (type == BlockType.WORKBENCH) hasWorkbench = true;
            if (type == BlockType.FURNACE) hasFurnace = true;
            if (type == BlockType.CHEST) hasChest = true;
            if (type == BlockType.TORCH) hasTorch = true;
        });
    }

    private void checkShelter() {
        // Simple shelter check: enough walls + roof + door
        shelterBuilt = wallBlockCount >= 12 && roofBlockCount >= 4 && hasDoor;
        if (shelterBuilt && !shelterRewardAwarded) {
            shelterRewardAwarded = true;
            addPhaseReward(5.0, 2);
        }
    }

    private void updateBaseScore() {
        baseScore = 0;
        baseScore += Math.min(wallBlockCount, 20); // up to 20 pts for walls
        baseScore += Math.min(roofBlockCount, 10);  // up to 10 pts for roof
        if (hasDoor) baseScore += 3;
        if (hasWorkbench) baseScore += 3;
        if (hasFurnace) baseScore += 3;
        if (hasChest) baseScore += 3;
        if (hasTorch) baseScore += 2;
        baseScore += nightsSurvived * 3;           // 3 pts per night survived
    }

    private boolean hasItemInInventory(int itemId) {
        for (int slot : inventory) {
            if (slot == itemId) return true;
        }
        return false;
    }

    private void addToInventory(int itemId) {
        for (int i = 0; i < inventory.length; i++) {
            if (inventory[i] == 0) {
                inventory[i] = itemId;
                return;
            }
        }
    }

    private void removeFromInventory(int itemId) {
        for (int i = 0; i < inventory.length; i++) {
            if (inventory[i] == itemId) {
                inventory[i] = 0;
                return;
            }
        }
    }

    private void addPhaseReward(double reward, int minimumPhase) {
        if (curriculumPhase >= minimumPhase) {
            pendingReward += reward;
        }
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    private record BlockPos(int x, int y, int z) {
    }
}
