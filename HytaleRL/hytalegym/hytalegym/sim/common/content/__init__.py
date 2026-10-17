"""Version-keyed content: ``load(profile) -> ContentTables``.

Tables come from ``HytaleRL/hytalegym/tools/content_extract`` and live in
``sim/genN/data/``. ``adapters`` reproduces the constants and tables the engine
holds today from the gen-5 tables (the migration's mechanical swap).
"""

from hytalegym.sim.common.content.loader import (
    ContentError,
    ContentTables,
    Override,
    Table,
    content_sha256,
    load,
)
from hytalegym.sim.common.content.registry import (
    GENERATION_MODULES,
    data_directory,
    generations,
    sim_profile,
)
from hytalegym.sim.common.content.schema import (
    FAMILIES,
    MANIFEST_SCHEMA,
    OVERLAY_SCHEMA,
    TABLE_SCHEMA,
    validate_manifest,
    validate_overlay,
    validate_table,
)

__all__ = [
    "FAMILIES",
    "GENERATION_MODULES",
    "MANIFEST_SCHEMA",
    "OVERLAY_SCHEMA",
    "TABLE_SCHEMA",
    "ContentError",
    "ContentTables",
    "Override",
    "Table",
    "content_sha256",
    "data_directory",
    "generations",
    "load",
    "sim_profile",
    "validate_manifest",
    "validate_overlay",
    "validate_table",
]
