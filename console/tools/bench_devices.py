"""Wall clock and peak VRAM per environment width, on whichever backend is live.

Run the same file under Windows CPU python and under the WSL venv to compare.
The point is not a single number — it is **where the curve stops improving**,
because that is the batch width worth training at. Throughput per environment
rises with width until the device saturates; past that you are paying memory
for nothing.

    # Windows, CPU
    python -m console.tools.bench_devices 2 8 32 128

    # WSL, CUDA
    # $(wslpath -a .) resolves the repository root for the WSL side
    wsl -d Ubuntu -- bash -lc "cd $(wslpath -a .) && \\
      PYTHONPATH=.:HytaleRL/hytalegym ~/.venvs/hydl/bin/python \\
      -m console.tools.bench_devices 2 8 32 128 512"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_ROOT, _ROOT / "HytaleRL" / "hytalegym"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import jax  # noqa: E402

from console.core.execution.runner import (  # noqa: E402
    RunSpec,
    _build_policy,
    _record,
    _scene_for,
)

TICKS = 64


def peak_mb() -> str:
    """Peak device memory, or `n/a` on a backend that does not report it.

    CPU exposes no memory stats. Printing 0 there would read as "used no
    memory" rather than "this backend cannot tell you".
    """

    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - absent on CPU
        return "n/a"
    if "peak_bytes_in_use" not in stats:
        return "n/a"
    return f"{int(stats['peak_bytes_in_use']) / 2 ** 20:.0f}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    widths = [int(a) for a in argv] or [2, 8, 32, 128]

    print(f"backend={jax.default_backend()}  devices={jax.devices()}", flush=True)
    print(f"{'envs':>6}{'build s':>10}{'compile s':>11}{'run s':>9}"
          f"{'env-ticks/s':>13}{'peak MB':>10}", flush=True)

    for n in widths:
        spec = RunSpec(ticks=TICKS, world="open_flat")
        try:
            started = time.perf_counter()
            handle, _ = _scene_for(spec, n)
            policy = _build_policy(spec)
            kwargs = ({"initial_carry": policy.initial_carry}
                      if hasattr(policy, "initial_carry") else {})
            collector = handle.compile_collector(policy, _record, TICKS, **kwargs)
            build = time.perf_counter() - started

            # First call compiles. Timing it measures XLA, not the environment
            # -- the first version of this reported 101 s for both 2 and 8
            # envs, which is compile time wearing a throughput label.
            started = time.perf_counter()
            jax.block_until_ready(collector(jax.random.key(0)))
            compile_s = time.perf_counter() - started

            started = time.perf_counter()
            jax.block_until_ready(collector(jax.random.key(1)))
            run = time.perf_counter() - started

            print(f"{n:>6}{build:>10.1f}{compile_s:>11.1f}{run:>9.2f}"
                  f"{n * TICKS / run:>13.0f}{peak_mb():>10}", flush=True)
        except Exception as exc:  # noqa: BLE001 - OOM is the answer, not a crash
            print(f"{n:>6}   FAILED: {type(exc).__name__}: {str(exc)[:70]}",
                  flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
