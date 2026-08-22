"""Deployment: turning a trained checkpoint into an NPC a server can run.

Separate from `adk/tools/` on purpose. Everything in `tools/` produces evidence
for a human or a test; everything here produces an artifact that ends up on a
running server, where a mistake is not a wrong number in a log but an NPC
behaving confidently and wrongly in someone's world.

    from adk.deploy import bundle
    bundle.export(checkpoint, destination, role="Kweebec_Razorleaf")
    problems = bundle.verify(destination)   # empty list means loadable

See `DEV.md` for what the plugin supports today and what it does not.
"""

from __future__ import annotations

# Imported lazily. Eagerly importing `bundle` here makes `python -m
# adk.deploy.bundle` emit a RuntimeWarning about the module already being in
# sys.modules, and that path is how the tool is actually run.
__all__ = ["bundle", "certify_java", "export", "isolated", "verify"]


def __getattr__(name: str):
    if name in __all__:
        # `from . import bundle` here re-enters this __getattr__ through
        # _handle_fromlist and recurses until the stack dies. import_module
        # does not consult the parent package's __getattr__.
        import importlib

        module_name = name if name in {"bundle", "isolated"} else "bundle"
        module = importlib.import_module(f"{__name__}.{module_name}")
        return module if name in {"bundle", "isolated"} else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
