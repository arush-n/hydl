# source_pins: the World source-pin audit tool

`rebaseline.py` is the one-command path for a new release line. This README is the
runbook; it lives here (tracked) because `**/DEV.md` and `**/SKILL.md` are gitignored.
Pointers from the worldgen and tests DEV.md files lead here.

The gate is `verify_installed_world_source_pins` in `source_pins.py`. It runs the
eleven audits in `tests/fidelity/surrogate_*_source_audit.py` (gitignored, on disk)
against the installed `Server/HytaleServer.jar` and `Assets.zip`. The 0.5 line keeps
its module-constant pins; the 0.7 line reads `source_pin_baselines_072.json`. The
verifier stops at the first failing audit and never reads an audit's
`matches_0_5_7_baseline` flag, so audit-level consistency needs its own check.

1. Observe: `python tools/source_pins/rebaseline.py observe --install <install dir>
   --out-dir <scratch>/observe` runs the audits with the gate's own arguments and
   `HYTALERL_SERVER_VERSION` set to the build; per-audit stdout, stderr and exit go
   to the scratch directory.
2. Candidate: `python tools/source_pins/rebaseline.py candidate --install <install
   dir> --out <scratch>/candidate.json [--moved OLD=NEW ...]` writes the observed
   digests only. Pass `--moved` for every asset that moved (navigation's trace assets
   did on 0.7.2). The package JSON is never written by this step.
3. Verify by hand, audit by audit. List each audit's `SOURCE_CONTRACT` and
   `payload` claims, then read the installed bytes each claim cites:
   `rebaseline.py changed <0.5 jar> <build jar> <fqcn> [--method NAME]` (whitespace-
   normalised body diff against a 0.5.x control install: the 0.5.7 jar is not on every
   machine, and the phase-5 identity run found the pinned items byte-equal on 0.5.10),
   `javap -cp <jar> -p -constants <fqcn>` for constant values, and the bridge source
   for any transport claim.
4. Record only what you verified: `rebaseline.py accept --install <install dir>
   --audit NAME --semantic <observed hash> --evidence "<claim, class or file:line>"`.
   It refuses unless the hash equals the audit's observed semantic hash, the pinned
   digests still match, and an evidence line is given. An audit you did not verify
   keeps `EXPECTED_SEMANTIC_SHA256` null and names its gaps in `open_gaps`.
5. Gate: run `tests/worldgen/test_source_pins.py` and
   `tests/worldgen/test_source_pin_baselines.py` (16 tests), then the verifier. Do not
   edit the audit scripts' semantics or the 0.5 constants to make an audit pass.

Pitfalls found on 0.7.2:
- javap pads the constant-pool column by width; a raw body diff marks spacing-only
  changes as changed. The `changed` subcommand collapses whitespace.
- Compare by name and parameters, not by signature alone: a re-signatured method
  (`ExplosionUtils.performExplosion` gained `Rotation3f`) looks removed to a
  signature diff. Check the name across overloads too.
- Inner classes (`Outer$Inner`) are not hashed unless listed. A claim that runs
  through one is not verified by the hash, so name the class in the notes.
- Abstract methods have zero instructions; check the implementations too
  (`ItemContainer.getCapacity`).
- Parse-time defaults: the audits evaluate their `--assets` argparse default from `HYTALE_ASSETS_ZIP`
  (or `APPDATA`) before parsing, so a host without either variable failed inside argparse before any
  comparison. Since 2026-10-18 the gate (`verify_installed_world_source_pins`) passes the installed
  `Assets.zip` to the subprocesses as `HYTALE_ASSETS_ZIP`; `observe` and `accept` set a placeholder
  `APPDATA` (the default is unused). The audit scripts are unchanged.
- Static text in a hashed payload can go stale without the gate noticing (inventory's
  `transport.schema` says v1 while both bridges emit v2). Read the bridge for transport
  claims before accepting.
