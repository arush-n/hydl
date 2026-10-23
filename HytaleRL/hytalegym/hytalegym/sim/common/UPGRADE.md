# Upgrading the simulator to a new Hytale release

Companion to `sim/PORTING.md` and `sim/common/RENDER.md`. Run from `HytaleRL/hytalegym`; `PY` is the venv python,
`NEW_ASSETS` the release's `Assets.zip`, `NEW_JAR` its `Server/HytaleServer.jar`.

## 0. Read what changed (no edits yet)

```sh
$PY -m hytalegym.sim.common.drift --assets "$NEW_ASSETS" --jar "$NEW_JAR" --generation gen7 --json /tmp/drift.json
```

Per row: `unchanged`; `changed` (old -> new; check first); `missing` (member, pointer or method gone);
`derived` (not re-read, the reason names its check); `undeclared` (add it to `sim/<gen>/sources.py`).
Exit 1 on changed, missing or undeclared.

## 1. Profile and extraction (PORTING.md 1-2)

Add `server_versions/vN_M.py` and `sim/genN/` (copy `sim/gen7/`), then
`cd tools && $PY -m content_extract extract --assets "$NEW_ASSETS" --out ../hytalegym/sim/genN/data`.
Gate: `tests/sim/test_extractors.py tests/sim/test_schema.py`.

## 2. Content diff (PORTING.md 3)

`cd tools && $PY -m content_extract diff <genPREV>/data <genN>/data --json F --markdown F`. Read it against the
release notes; encode each measured change in `tests/sim/test_a2_diff.py`.

## 3. Declarations

- `undeclared`: `AssetSource(zip_path, pointer, expected)` for a JSON value (the table row's `source` gives the
  path); `ClassSource(cls, member, digest)` for a jar method (`method_block` in `drift.py`); or
  `Derived(reason)` naming its check. Pin the value the row was written against, rerun, expect `unchanged`.
- `changed`: the render must change (step 4), or the declaration takes the new value with a note why.

## 4. Overlay delta (PORTING.md 4)

- `cd tools && $PY -m sim_port.regen_swap_plan --dry-run`, then without the flag; re-pin the step-clock counts
  in `tests/sim/test_render.py` if they moved.
- Activate swap-plan rows whose adapter output changed; constructs for behaviour, patches for constants. Each new
  row gets a declaration and a fidelity row.
- `$PY -m hytalegym.sim.common.render --generation genN --check`, then `--out DIR`.

## 5. Gates

`$PY -m pytest tests/sim -q --ignore=tests/sim/test_render_identity.py`; then `tests/sim/test_drift.py`
`tests/sim/test_render.py` `tests/sim/test_constructs.py`; then `tests/sim/test_render_identity.py` (gen 5, 4 passed).

## 6. Live checks and the stamp (PORTING.md 6-8)

Bridge differentials and the 11 source-pin audits (`tools/source_pins/README.md`), then `CONTENT_SHA256`,
`DEFAULT_SERVER_VERSION` and the owner's flip; record each result in `FIDELITY-N.md`.

Not seen by the drift checker: adapter formulas, the step-clock tick rate, live-measured ruleset fields and
guard-root interrupt masks. Review those by hand.
