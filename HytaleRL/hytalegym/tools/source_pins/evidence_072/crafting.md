# crafting (lead: no claim re-derived; the analysis agent stopped on the weekly API limit). Every claim is C in this pass.
Rule result: C. Claims are grouped by SOURCE_CONTRACT top-level key (lines 135-249).

note: the previous pass names CraftingManager.isValidBenchForRecipe as changed. In 0.7.2 that method (package com.hypixel.hytale.builtin.crafting.component.CraftingManager) checks the player's known recipes (Player.getPlayerConfigData, KnownRecipes, isKnowledgeRequired), not the bench match, so the bench-match predicate the audit cites (old CraftingManager.java:372-403) was not located in this pass.
claim: recipe_identity.* (String asset key; BenchRecipeRegistry and CraftRecipeAction java file:line citations) | verdict: C | old decompile file:line citations, not bytecode | not checkable from the 0.7.2 bytecode
claim: bench_gate.* (window attachment; execution_predicate; fieldcraft; page types) | verdict: C | bench-match predicate not located (see note) | not re-derived
claim: execution_boundary.* (player_requirement assert; non_player_output; certificate) | verdict: C | CraftingManager.craftItem and giveOutput | not re-derived
claim: learning.* (PlayerConfigData codecs; execution key; interaction key; unverified consistency) | verdict: C | PlayerConfigData, LearnRecipeInteraction | not re-derived
claim: craftable_count.* (simulator counts; native writes zero) | verdict: C | repo-side simulator (BaseBuilderTask.java and NativeEnvironmentSession.java file:line) | repo and old citations; not re-derived
claim: inventory_boundary.* (getInputMaterials clone; resource selectors; simple input priority; crafted stack; pickup priority; overflow) | verdict: C | CraftingManager.getInputMaterials, getOutputItemStack; InventoryUtils.moveItem, smartMoveItem, getSectionById (changed, previous pass); SimpleItemContainer.addOrDropItemStack | not re-derived
payload: recipe corpus expectation (EXPECTED_ASSET_METRICS, 1947 recipes) | verdict: C | asset metrics | the previous pass records 2,047 recipes on 0.7.2 (FIDELITY (b)); not recounted in this pass, so a B cannot be stated here
