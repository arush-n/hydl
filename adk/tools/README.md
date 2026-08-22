# ADK tools

`adk.tools` contains standalone commands for cataloging the available surface,
running a shaping minigame, inspecting replay data, exporting policies, and
checking native fidelity. These commands consume ADK interfaces; they do not
replace the Console training entry point.

Useful starting points are [`catalog.py`](catalog.py),
[`run_minigame.py`](run_minigame.py), [`port_fidelity.py`](port_fidelity.py),
and the export helpers. Native commands require the external Hytale runtime and
bridge inputs described by the HytaleRL package.

For repository publication checks, use [`tools/README.md`](../../tools/README.md).
