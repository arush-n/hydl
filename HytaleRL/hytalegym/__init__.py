"""Source-checkout import shim for the nested ``hytalegym`` project.

The installable package lives in ``hytalegym/hytalegym``. When Python is
started from the repository root, the project directory would otherwise be
imported as an empty namespace package before the editable-install finder can
load the real package. Extending this package path keeps repository-root test
and example commands equivalent to an installed import.
"""

from pathlib import Path as _Path


_SOURCE_PACKAGE = str(_Path(__file__).with_name("hytalegym"))
if _SOURCE_PACKAGE not in __path__:
    __path__.insert(0, _SOURCE_PACKAGE)

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.envs.registration import register_envs


__version__ = "0.1.0"
__all__ = ["HytaleEnv", "register_envs"]

register_envs()
