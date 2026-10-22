# hytalegym tools

Command-line tools for porting the simulator to a new Hytale server line, measuring it, and building its world
products. Each subfolder has its own README. Run them from this folder (`HytaleRL/hytalegym/tools`) with
`PYTHONPATH=<repo>:<repo>/HytaleRL/hytalegym` unless a README says otherwise.

| Folder | Purpose | Main document |
|---|---|---|
| `content_extract/` | game files (Assets.zip, server jar) to versioned JSON tables the simulator reads, with provenance; `extract`, `diff`, `verify` | `../hytalegym/sim/PORTING.md` steps 2 and 3 |
| `sim_port/` | the swap plan for the gen-7 render: `regen_swap_plan.py` re-pins step-clock sites after an engine commit | `../hytalegym/sim/PORTING.md` |
| `sim_bench/` | render baselines: compile time, throughput, fingerprint (`rollout`), real-time factor (`realtime`) | `../hytalegym/sim/BASELINES.md` |
| `source_pins/` | the eleven source-pin audits of a server line and their re-baselining | `source_pins/README.md` |
| `worldgen/` | Region capture and recapture, recertification, promotion, task zoo, path tables | `worldgen/README.md` |

Live native probes (the receipts behind the fidelity record) are not here: they live in
`../tests/fidelity/combat/gen7_probes/` (gitignored; README there) and write to `HytaleRL/artifacts/fidelity_evidence/`.
