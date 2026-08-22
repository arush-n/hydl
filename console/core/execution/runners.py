"""Which compute this machine can actually train on, probed rather than assumed.

The console runs as a Windows Python process, and **that process cannot see the
GPU at all**. Measured 2026-08-08:

    jax 0.9.2   default_backend: cpu   devices: [CpuDevice(id=0)]
    jax.devices("gpu") -> RuntimeError: Unknown backend: 'gpu' requested,
                          but no platforms that are instances of gpu are present

The GPU is real (RTX 4060 Ti, 16 GB) and JAX reaches it — but only from inside
WSL, which carries its own interpreter and its own JAX:

    wsl Ubuntu: jax 0.10.0   backend: gpu   devices: [CudaDevice(id=0)]

So "run on the GPU" is not a setting. It is a different process, on a different
JAX version, reached over a subprocess boundary. This module reports what is
available, what each one costs, and -- via :func:`dispatch` -- actually runs
work on one. Anything that claims a run executed somewhere this module did not
find is describing a machine that no longer exists.

How dispatch avoids inventing a protocol
----------------------------------------
WSL and Windows share one filesystem: ``/mnt/c/<path>`` *is*
``C:\\<path>``. So a dispatched run passes a spec **file** in and
reads a result **file** out, and nothing has to serialise a trajectory across a
process boundary. The dispatched run also writes its artifact to the same
directory a local run does, which is why the Runs tab cannot tell them apart --
the only difference is which device did the arithmetic.

Why the default is CPU, despite the GPU being present
-----------------------------------------------------
Identical `lax.scan` microbenchmark, 256 steps of a 64x64 matmul -- the shape
the rollout collector actually is:

    Windows CPU   0.005 s
    WSL CUDA      0.010 s

The GPU is **2x slower** here. That is not surprising: a scan is sequential by
construction, so each step is a small kernel launch and the device never gets
the parallelism it is good at. GPUs win on this project when the *batch* is
wide, not when the loop is long.

Two further cautions, both recorded from earlier runs and **not re-measured
here** -- treat as leads, not findings:

* a crash (`double free or corruption`) at **>= 32 environments** under WSL
  CUDA. The console's own default is 32 envs, which is exactly that boundary.
* JAX 0.9.2 on Windows against 0.10.0 in WSL. A checkpoint or contract stamp
  crossing that boundary has not been shown to round-trip.
"""

from __future__ import annotations

from console.core import storage

import json
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


class RunnerError(RuntimeError):
    """A dispatched run could not be started, or did not produce a result."""

#: Probe printed as JSON by whichever interpreter is being asked. Kept to one
#: line so a shell that decorates its output (CUDA emits driver warnings on
#: stderr, and login shells print banners) cannot corrupt the parse.
_PROBE = (
    "import jax,json;"
    "print('@@'+json.dumps({"
    "'jax': jax.__version__,"
    "'backend': jax.default_backend(),"
    "'devices': [str(d) for d in jax.devices()]"
    "}))"
)

# WSL work is a GPU request, not a best-effort hint. Keep the allocator from
# claiming most VRAM and put cross-process executables on WSL's native disk.
_WSL_GPU_ENV = (
    "JAX_PLATFORMS=cuda "
    "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} "
    "HYTALERL_JAX_CACHE=${HYTALERL_JAX_CACHE:-$HOME/.cache/hytalerl/jax}"
)


def _parse(output: str) -> dict[str, Any] | None:
    """Pull the probe's line out of whatever else the shell said."""

    for line in output.splitlines():
        marker = line.find("@@")
        if marker >= 0:
            try:
                return json.loads(line[marker + 2:])
            except json.JSONDecodeError:
                continue
    return None


def _local() -> dict[str, Any]:
    """This interpreter. Always present -- it is the one asking."""

    import jax

    return {
        "id": "local",
        "label": "this process",
        "available": True,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "dispatch": "in-process",
        "note": "",
    }


def _wsl(timeout: float = 90.0) -> dict[str, Any]:
    """WSL's own interpreter, if there is one and it has JAX.

    Every failure is reported as a reason rather than an exception: "no GPU
    runner" and "the probe timed out" lead to different next steps, and a
    caller that sees only `available: False` will take the wrong one.
    """

    entry: dict[str, Any] = {
        "id": "wsl",
        "label": "WSL (Ubuntu)",
        "available": False,
        "jax": None,
        "backend": None,
        "devices": [],
        # Subprocess over the shared filesystem -- see `dispatch`.
        "dispatch": "subprocess",
        "note": "",
    }
    if shutil.which("wsl") is None and shutil.which("wsl.exe") is None:
        entry["note"] = "wsl is not installed"
        return entry
    try:
        finished = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc",
             f'{_WSL_GPU_ENV} python3 -c "{_PROBE}"'],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        entry["note"] = (
            f"probe timed out after {timeout:.0f}s; WSL may be starting from "
            "a stopped state")
        return entry
    except OSError as error:  # noqa: BLE001 - a missing shell is a reason
        entry["note"] = f"could not run wsl: {error}"
        return entry

    parsed = _parse((finished.stdout or "") + "\n" + (finished.stderr or ""))
    if parsed is None:
        entry["note"] = "WSL answered but CUDA JAX did not initialize there"
        return entry
    entry.update(available=True, **parsed)
    if entry["backend"] != "gpu":
        entry["note"] = "JAX is installed but not on a GPU backend"
    return entry


@lru_cache(maxsize=1)
def _wsl_cached() -> tuple[tuple[str, Any], ...]:
    """One WSL probe per process, as an immutable pair sequence.

    Cached because starting a *stopped* distribution takes tens of seconds and
    `options()` is on the page-load path. Returned as tuples so the cache
    cannot hand out a dict a caller then mutates. Available hardware does not
    change within a process; restart the console after installing a driver.
    """

    return tuple(_wsl().items())


def scan(*, probe_wsl: bool = True) -> list[dict[str, Any]]:
    """Every runner this machine offers, cheapest probe first.

    `probe_wsl=False` skips the subprocess entirely and reports WSL as
    unprobed. Use it anywhere a slow answer is worse than a partial one.
    """

    found = [_local()]
    if probe_wsl:
        found.append(dict(_wsl_cached()))
    else:
        found.append({
            "id": "wsl", "label": "WSL (Ubuntu)", "available": None,
            "jax": None, "backend": None, "devices": [],
            "dispatch": "subprocess", "note": "not probed",
        })
    return found


#: The runner a run uses when nothing else is asked for. `local` because it is
#: measured faster on the shape the collector actually is (see the module
#: docstring); not because it is the only one that works.
ACTIVE = "local"

#: Repo root, as each side of the boundary spells it. WSL and Windows share one
#: filesystem, which is what makes dispatch cheap -- only a path crosses.
_WINDOWS_ROOT = Path(__file__).resolve().parents[3]


def to_wsl_path(path: str | Path) -> str:
    """`C:\\repo\\x` -> `/mnt/c/repo/x`.

    Written out rather than shelled to `wslpath` because this runs on the
    Windows side, once per dispatch, and spawning a WSL process to translate a
    string before spawning a WSL process to do the work is a needless second
    cold start on a distribution that may be stopped.
    """

    text = str(Path(path).resolve())
    drive, _, rest = text.partition(":")
    if len(drive) == 1 and rest:
        return f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}"
    return text.replace(chr(92), "/")


def dispatch(spec: dict[str, Any], *, runner: str = ACTIVE,
             timeout: float = 3600.0) -> dict[str, Any]:
    """Run one rollout on `runner` and return its result dictionary.

    `local` is executed in-process -- there is no reason to pay a subprocess to
    reach the interpreter already running. Anything else is a subprocess over
    the shared filesystem: the spec goes out as a file, the result comes back
    as a file, and nothing invents a wire format for a trajectory.

    Raises `RunnerError` rather than returning a partial result. A dispatched
    run that fails halfway is not a run with missing fields; treating it as one
    is how a broken measurement gets stored as a real one.
    """

    if runner == "local":
        from console.core.execution.runner import RunSpec, run

        return run(RunSpec(**spec))

    if runner != "wsl":
        raise RunnerError(f"unknown runner {runner!r}")

    entry = _wsl_entry()
    if entry.get("available") is not True:
        raise RunnerError(
            f"wsl is not usable: {entry.get('note') or 'not probed'}")

    scratch = storage.write_root("dispatch")
    scratch.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000):d}"
    spec_file = scratch / f"spec-{stamp}.json"
    result_file = scratch / f"result-{stamp}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")

    root = to_wsl_path(_WINDOWS_ROOT)
    gym = f"{root}/HytaleRL/hytalegym"
    command = (
        f"cd {root} && {_WSL_GPU_ENV} PYTHONPATH={root}:{gym} python3 -u -m "
        f"console.tools.remote_run {to_wsl_path(spec_file)} "
        f"{to_wsl_path(result_file)}"
    )
    try:
        finished = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        raise RunnerError(
            f"wsl run exceeded {timeout:.0f}s and was abandoned") from expired

    report = _parse_marked(
        (finished.stdout or "") + "\n" + (finished.stderr or ""),
        "@@remote_run ")
    if report is not None and not report.get("ok", False):
        raise RunnerError(f"wsl run failed: {report.get('error', 'no reason')}")
    if finished.returncode != 0 or not result_file.exists():
        tail = (finished.stderr or finished.stdout or "").strip().splitlines()
        raise RunnerError(
            f"wsl run exited {finished.returncode} without a result; "
            + (tail[-1] if tail else "no output"))

    return json.loads(result_file.read_text(encoding="utf-8"))


def _parse_marked(output: str, marker: str) -> dict[str, Any] | None:
    """The remote entrypoint's own line, out of a noisy shell."""

    for line in output.splitlines():
        at = line.find(marker)
        if at >= 0:
            try:
                return json.loads(line[at + len(marker):])
            except json.JSONDecodeError:
                continue
    return None


def _wsl_entry() -> dict[str, Any]:
    return dict(_wsl_cached())
