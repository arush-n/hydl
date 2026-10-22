# sim_port

`regen_swap_plan.py` regenerates the gen-7 render's swap plan (`../hytalegym/sim/common/content/redirects.json`): the
redirect records and the step-clock sites, with their line numbers in the committed engine. After any commit that
touches a module under `hytalegym/jax/` (even a comment) those line numbers can go stale, so run it from
`HytaleRL/hytalegym/tools`:

    PYTHONPATH=<repo>:<repo>/HytaleRL/hytalegym python -m sim_port.regen_swap_plan --dry-run   # report only
    PYTHONPATH=<repo>:<repo>/HytaleRL/hytalegym python -m sim_port.regen_swap_plan             # write

The dry run reports the new head, the record counts and every moved `head_line`. The write also re-pins
`tests/sim/test_render.py` when the counts change and checks that `hytalegym.sim.gen7.overlay` reads the plan; exit 0
on success. If lines moved, commit the rewritten `redirects.json` alone ("swap plan re-pinned after <commit>"); if only
the recorded head changed, revert the file. Background and the version-bump checklist: `../hytalegym/sim/PORTING.md`.
