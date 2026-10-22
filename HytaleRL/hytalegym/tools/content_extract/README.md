# content_extract

Game files to versioned JSON tables with provenance, one module per family (blocks, consumables, crafting, crafting
augment tags, durability, hit interrupt, movement, NPC roles, projectiles, stats, weapons, step sites, redirects).
`archive.py` reads one `Assets.zip` read-only and records where each value came from; `manifest.py` writes the table
manifest; `diff.py` compares two versions' tables.

    cd HytaleRL/hytalegym/tools
    python -m content_extract extract --assets <Assets.zip> --out ../hytalegym/sim/genN/data
    python -m content_extract diff ../hytalegym/sim/genPREV/data ../hytalegym/sim/genN/data --json <out.json> --markdown <out.md>
    python -m content_extract verify --assets <Assets.zip> --data ../hytalegym/sim/genN/data

`--server-jar` defaults to `<assets dir>/Server/HytaleServer.jar`; `--families` limits the families. The porting
steps that use it are `../hytalegym/sim/PORTING.md` steps 2 and 3. `evidence_057.py` holds the 0.5.7 values that
were measured rather than extracted. Asset rules learned so far: `SKILL.md` (local working notes, gitignored).
