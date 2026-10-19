"""Re-baseline the World source-pin audits for one installed Hytale release line.

Run from any directory with the repo's Python. Subcommands:

  observe    run the eleven registry audits against an installed build, with the gate's own
             arguments and environment, and keep each stdout, stderr and exit code.
  candidate  write a candidate baseline JSON for the build's release line: observed digests of
             every pinned class and asset, EXPECTED_SEMANTIC_SHA256 null, and an open gap per audit.
             Nothing is written into the package; the candidate is reviewed by hand.
  accept     record one audit's EXPECTED_SEMANTIC_SHA256 in the baseline JSON. Refused unless the
             hash given equals the audit's observed semantic hash now, the pinned digests still
             match the baseline, and at least one evidence line is given.
  changed    list the methods of one class whose whitespace-normalised bytecode differs between
             two server jars (javap column padding is collapsed; --method prints the diff).

Runbook: hytalegym/hytalegym/worldgen/DEV.md, "Re-baselining the source-pin audits".
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from hytalegym.worldgen.source_pin_baselines import (  # noqa: E402
    BASELINE_072_PATH,
    archive_digest,
    server_jar_release_line,
)
from hytalegym.worldgen.source_pins import (  # noqa: E402
    _AUDITS,
    _source_audit_environment,
    installed_hytale_server_identity,
)

FIDELITY_DIR = PACKAGE_ROOT / "tests" / "fidelity"
PINNED_ATTRIBUTES = ("CLASS_SHA256", "ASSET_SHA256", "TRACE_ASSET_SHA256")
JAR_RELATIVE = Path("Server") / "HytaleServer.jar"


def _install_paths(install: Path) -> tuple[Path, Path]:
    """Return (server jar, Assets.zip) for an install directory laid out like the launcher's."""

    jar, assets = install / JAR_RELATIVE, install / "Assets.zip"
    for path in (jar, assets):
        if not path.is_file():
            raise SystemExit(f"missing installed file: {path}")
    return jar, assets


def _load_audit(script: str):
    spec = importlib.util.spec_from_file_location(f"source_audit_{script}", FIDELITY_DIR / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_audit(script: str, uses_assets: bool, jar: Path, assets: Path, version: str,
               python: str) -> tuple[int, str, str]:
    """Run one audit exactly as source_pins.verify_installed_world_source_pins does."""

    command = [python, str(FIDELITY_DIR / script), "--server-jar", str(jar)]
    if uses_assets:
        command += ["--assets", str(assets)]
    environment = _source_audit_environment(PACKAGE_ROOT)
    environment.update({"HYTALE_ASSETS_ZIP": str(assets), "HYTALERL_SERVER_VERSION": version})
    # crafting's argparse default reads APPDATA at parse time, even when --assets is given; on
    # macOS APPDATA is unset and the audit would crash. The default is unused here, so a placeholder.
    environment.setdefault("APPDATA", str(assets.parent))
    completed = subprocess.run(command, cwd=PACKAGE_ROOT, env=environment, capture_output=True,
                               text=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def _moved_map(pairs: list[str] | None) -> dict[str, str]:
    moved = {}
    for pair in pairs or []:
        old, sep, new = pair.partition("=")
        if not sep or not old or not new:
            raise SystemExit(f"--moved expects OLD=NEW, got {pair!r}")
        moved[old] = new
    return moved


def cmd_observe(args: argparse.Namespace) -> int:
    jar, assets = _install_paths(args.install)
    identity = installed_hytale_server_identity(jar)
    version = identity["implementation_version"]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"installed_build": identity, "audits": {}}
    for name, script, uses_assets in _AUDITS:
        code, stdout, stderr = _run_audit(script, uses_assets, jar, assets, version, args.python)
        (out / f"{name}.stdout.json").write_text(stdout)
        (out / f"{name}.stderr.txt").write_text(stderr)
        (out / f"{name}.exit").write_text(f"{code}\n")
        row = {"exit": code}
        try:
            report = json.loads(stdout)
            row.update(semantic_sha256=report.get("semantic_sha256"),
                       expected_semantic_sha256=report.get("expected_semantic_sha256"),
                       matches_flag=report.get("matches_0_5_7_baseline"))
        except json.JSONDecodeError:
            row["error"] = (stderr.strip().splitlines() or ["no JSON on stdout"])[-1]
        summary["audits"][name] = row
        print(f"{name:20s} exit={code} semantic={str(row.get('semantic_sha256'))[:12]} "
              f"matches={row.get('matches_flag')} {row.get('error', '')}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}/summary.json")
    return 0


def cmd_candidate(args: argparse.Namespace) -> int:
    jar, assets = _install_paths(args.install)
    identity = installed_hytale_server_identity(jar)
    line = server_jar_release_line(jar)
    if line == (0, 5):
        raise SystemExit("the 0.5 line keeps its audit-module pins; candidate writes the 0.7-style table only")
    moved = _moved_map(args.moved)
    jar_sha = identity["server_jar_sha256"]
    version = identity["implementation_version"]
    audits: dict[str, object] = {}
    with ZipFile(jar) as jar_zip, ZipFile(assets) as asset_zip:
        for name, script, _uses_assets in _AUDITS:
            module = _load_audit(script)
            entry: dict[str, object] = {"EXPECTED_SERVER_SHA256": jar_sha,
                                        "EXPECTED_SEMANTIC_SHA256": None}
            for attribute in PINNED_ATTRIBUTES:
                if not hasattr(module, attribute):
                    continue
                archive = jar_zip if attribute == "CLASS_SHA256" else asset_zip
                table = {}
                for pinned in getattr(module, attribute):
                    path = moved.get(pinned, pinned)
                    table[path] = archive_digest(archive, path)
                entry[attribute] = table
            entry["open_gaps"] = [
                f"not re-derived on {version}: the SOURCE_CONTRACT claims of {script} are unread "
                "(candidate; replace with the named gaps after reading the bytecode)"
            ]
            audits[name] = entry
    document = {
        "schema": "hytalerl_source_pin_baselines_072_v1",
        "release_line": list(line),
        "installed_build": {
            "implementation_version": version,
            "implementation_revision": identity["implementation_revision"],
            "server_jar_sha256": jar_sha,
        },
        "recorded_from": f"candidate written by tools/source_pins/rebaseline.py from {jar}",
        "status": "candidate: observed pins only; no semantic baseline is verified",
        "audits": audits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote candidate {args.out} ({len(audits)} audits, release line {line[0]}.{line[1]})")
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    jar, assets = _install_paths(args.install)
    identity = installed_hytale_server_identity(jar)
    document = json.loads(args.baseline.read_text(encoding="utf-8"))
    entry = document["audits"].get(args.audit)
    if entry is None:
        raise SystemExit(f"no audit {args.audit!r} in {args.baseline}")
    script, uses_assets = next((s, u) for n, s, u in _AUDITS if n == args.audit)
    code, stdout, stderr = _run_audit(script, uses_assets, jar, assets,
                                      identity["implementation_version"], args.python)
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"refused: {args.audit} printed no JSON: {stderr.strip()[-300:]}")
    observed = report.get("semantic_sha256")
    if observed != args.semantic.lower():
        raise SystemExit(f"refused: observed semantic for {args.audit} is {observed}; "
                         "pass exactly that value after reading its claims")
    if "CLASS_SHA256" in entry and report.get("class_sha256") != entry["CLASS_SHA256"]:
        raise SystemExit("refused: observed class digests differ from the baseline; re-run candidate")
    if not args.evidence:
        raise SystemExit("refused: give at least one --evidence line (claim, file:line, bytecode read)")
    entry["EXPECTED_SEMANTIC_SHA256"] = observed
    entry["open_gaps"] = []
    entry["verified_claims"] = list(args.evidence)
    if args.caveat:
        entry["caveats"] = list(args.caveat)
    args.baseline.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"recorded EXPECTED_SEMANTIC_SHA256 for {args.audit} in {args.baseline}")
    return 0


def _method_bodies(jar: Path, fqcn: str, javap: str) -> dict[str, list[str]]:
    completed = subprocess.run([javap, "-cp", str(jar), "-p", "-c", fqcn], capture_output=True,
                               text=True, check=False)
    if completed.returncode:
        raise SystemExit(f"javap failed for {fqcn}: {completed.stderr.strip()[-300:]}")
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in completed.stdout.splitlines():
        if re.match(r"^  \S.*;$", line):
            current = _signature_key(line.strip()) if ("(" in line or line.strip() == "static {};") else None
            if current is not None:
                bodies[current] = []
        elif current is not None and re.match(r"^\s{4,}\d+: ", line):
            text = re.sub(r"^\s*\d+: ", "", line)
            text = re.sub(r"#\d+", "#", text)
            bodies[current].append(re.sub(r"\s+", " ", text).strip())
    return bodies


def _signature_key(header: str) -> str:
    if header == "static {};":
        return "<clinit>"
    head, _, params = header.partition("(")
    name = head.split()[-1].split(".")[-1]
    return f"{name}({params}"


def cmd_changed(args: argparse.Namespace) -> int:
    old = _method_bodies(args.jar_a, args.fqcn, args.javap)
    new = _method_bodies(args.jar_b, args.fqcn, args.javap)
    common = sorted(set(old) & set(new))
    changed = [key for key in common if old[key] != new[key]]
    print(f"{args.fqcn}: common {len(common)}, changed {len(changed)}, "
          f"only in A {len(set(old) - set(new))}, only in B {len(set(new) - set(old))}")
    for key in changed:
        print(f"  CHANGED {key}")
    if args.method:
        for key in [k for k in common if k.split("(")[0] == args.method]:
            print(f"== {key}")
            for line in difflib.unified_diff(old[key], new[key], "A", "B", lineterm="", n=2):
                print("   " + line)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def install_argument(p: argparse.ArgumentParser) -> None:
        p.add_argument("--install", type=Path, required=True,
                       help="install directory holding Server/HytaleServer.jar and Assets.zip")
        p.add_argument("--python", default=sys.executable, help="interpreter for the audit scripts")

    observe = sub.add_parser("observe", help="run the eleven audits and keep their outputs")
    install_argument(observe)
    observe.add_argument("--out-dir", type=Path, required=True)
    observe.set_defaults(func=cmd_observe)

    candidate = sub.add_parser("candidate", help="write a candidate baseline JSON (observed pins only)")
    install_argument(candidate)
    candidate.add_argument("--out", type=Path, required=True)
    candidate.add_argument("--moved", action="append", metavar="OLD=NEW",
                           help="asset or class path that moved on this line (repeatable)")
    candidate.set_defaults(func=cmd_candidate)

    accept = sub.add_parser("accept", help="record one audit's semantic baseline after verification")
    install_argument(accept)
    accept.add_argument("--baseline", type=Path, default=BASELINE_072_PATH)
    accept.add_argument("--audit", required=True)
    accept.add_argument("--semantic", required=True, help="the observed semantic_sha256 you verified")
    accept.add_argument("--evidence", action="append", help="verified claim with its evidence (repeatable)")
    accept.add_argument("--caveat", action="append", help="known open item that does not block (repeatable)")
    accept.set_defaults(func=cmd_accept)

    changed = sub.add_parser("changed", help="whitespace-normalised method diff of one class")
    changed.add_argument("jar_a", type=Path)
    changed.add_argument("jar_b", type=Path)
    changed.add_argument("fqcn")
    changed.add_argument("--method", help="also print the diff of this method name")
    changed.add_argument("--javap", default="javap")
    changed.set_defaults(func=cmd_changed)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
