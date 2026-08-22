"""Execute one console rollout, in whatever interpreter is running this file.

The far side of :func:`console.core.execution.runners.dispatch`. It exists because "run
on the GPU" is not a setting in this project -- the Windows interpreter the
console lives in cannot see the GPU at all, and JAX only reaches it from inside
WSL, which is a different interpreter on a different JAX version behind a
subprocess boundary.

**Nothing is serialised across that boundary except a path.** WSL and Windows
share one filesystem (``/mnt/c/<path>`` *is* ``C:\\<path>``),
so the spec goes in as a JSON file and the result comes back as a JSON file,
both on disk where either side can read them. That avoids inventing a wire
format for a trajectory, and it means a dispatched run writes its artifact to
exactly the same place a local run does -- the console's Runs tab cannot tell
them apart, which is the point.

Run as::

    python3 -m console.tools.remote_run <spec.json> <result.json>

Exit code is the honest signal: 0 only if the rollout completed and the result
was written. The stdout marker line is a convenience for a caller that would
rather not read the file.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

#: Printed as the last line on success. Marked so a caller can find it in
#: output a login shell and CUDA have both decorated -- the WSL probe already
#: has to survive `cuda_executor.cc` errors on stderr.
MARKER = "@@remote_run "


def execute(spec_path: str, result_path: str) -> int:
    from console.core.execution.runner import RunSpec, run

    payload: dict[str, Any] = json.loads(Path(spec_path).read_text("utf-8"))

    # Unknown keys are refused rather than dropped. A spec written by a newer
    # console and run by an older checkout would otherwise execute a *different
    # configuration* than the one requested and report it as that run -- the
    # kind of silent mismatch that makes a measurement void rather than wrong.
    known = set(RunSpec.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"spec has fields this runner does not know: {', '.join(unknown)}; "
            "the dispatching console is newer than this checkout")

    spec = RunSpec(**payload)
    result = run(spec)

    out = Path(result_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written atomically: a reader polling for the file must never see half a
    # JSON document and conclude the run produced garbage.
    staging = out.with_suffix(out.suffix + ".partial")
    staging.write_text(json.dumps(result), encoding="utf-8")
    staging.replace(out)

    import jax

    print(MARKER + json.dumps({
        "ok": True,
        "result": str(out),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }), flush=True)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <spec.json> <result.json>", file=sys.stderr)
        return 2
    try:
        return execute(argv[1], argv[2])
    except Exception as error:  # noqa: BLE001 - a failed run must say why
        # Reported as a structured line *and* a traceback: the caller parses
        # the former, a human reads the latter. A dispatched run that dies with
        # only an exit code is undiagnosable from the other side of a
        # subprocess.
        traceback.print_exc()
        print(MARKER + json.dumps({
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
