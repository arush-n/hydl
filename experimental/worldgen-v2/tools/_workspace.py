"""Make the adjacent HytaleRL checkout importable for experiment tools."""

from __future__ import annotations

from pathlib import Path
import sys


def use_workspace_hytalerl() -> Path:
    """Prepend the checked-out HytaleRL Python source and return its path."""

    source = Path(__file__).resolve().parents[3] / "HytaleRL" / "hytalegym"
    if not source.is_dir():
        raise RuntimeError(f"HytaleRL source directory not found: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return source
