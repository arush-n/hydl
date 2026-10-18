package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;

/** Native-only runtime-resolved crafting catalog capability. */
public interface NativeCraftingCatalogSource {

    NativeCraftingCatalogEvidence captureCraftingCatalogEvidence();
}
