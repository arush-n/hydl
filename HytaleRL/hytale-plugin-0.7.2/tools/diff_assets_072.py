#!/usr/bin/env python3
"""P0 diff inventory: Hytale 0.5 vs 0.7.2 game assets and server jar.

Read-only on every input (zip central directories and member bytes only, nothing
is extracted). Stdlib only. Writes one markdown report. Phase P0 of
HytaleRL/hytale-plugin-0.7.2/SIMULATOR-0.7.2.md.

Example:
  python tools/diff_assets_072.py \
    --old-assets ".../v0.5/package/game/latest/Assets.zip" \
    --new-assets ".../release/package/game/latest/Assets.zip" \
    --old-jar ".../v0.5/package/game/latest/Server/HytaleServer.jar" \
    --new-jar ".../release/package/game/latest/Server/HytaleServer.jar" \
    --output HytaleRL/hytale-plugin-0.7.2/ASSET-DIFF-0.7.2.md
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SIM_SUBDIRS = ("HytaleRL/hytalegym/hytalegym", "HytaleRL/hytalegym/tests")
CATALOG_REL = "HytaleRL/hytalegym/hytalegym/combat/assets/catalog.py"
PIN_RE = re.compile(r'HYTALE_0_5_7_ASSETS_SHA256\s*=\s*\(\s*"([0-9A-Fa-f]{64})"')
SERVER_PATH_RE = re.compile(r"Server/[A-Za-z0-9_/.\-]+")
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
BARE_ID_RE = re.compile(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")
# Strips // and /* */ comments and trailing commas, leaving string literals intact.
JSONC_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])', re.S)
FIELD_PREFIXES = (
    "Server/Item/",
    "Server/Entity/",
    "Server/NPC/",
    "Server/Projectiles/",
    "Server/ProjectileConfigs/",
)
FOCUS_PREFIXES = (
    "Server/Item/Interactions/",
    "Server/Entity/MovementConfig/",
    "Server/Entity/Stats/",
    "Server/NPC/Roles/",
)
JAR_KEYWORDS = (
    "movement", "physics", "motion", "collision", "damage", "knockback",
    "interaction", "combat", "stat", "mana", "stamina", "sight", "raycast",
    "npc", "role", "crafting", "block",
)
VALUE_MAX = 120
MAX_FIELD_LINES = 200
MAX_LINE_DIFF = 40
MAX_LIST_LINES = 150
MAX_JAR_LINES = 300
_MISSING = object()


@dataclass
class FileDiff:
    mode: str
    removed: list[tuple[str, Any]]
    added: list[tuple[str, Any]]
    changed: list[tuple[str, Any, Any]]
    old_flat: dict[str, Any] | None
    new_flat: dict[str, Any] | None
    line_diff: list[str] | None


@dataclass
class Ctx:
    args: argparse.Namespace
    shas: dict[str, str]
    pin: str | None
    pin_line: int | None
    stats: Counter
    old_idx: dict[str, tuple[int, int]]
    new_idx: dict[str, tuple[int, int]]
    new_by_name: dict[str, list[str]]
    refs: dict[str, set[str]]
    unmapped_ids: list[str]
    bare_id_count: int
    changed_paths: list[str]
    diffs: Differ
    field_diffs: dict[str, FileDiff]
    old_zip: zipfile.ZipFile
    new_zip: zipfile.ZipFile
    jar: dict[str, Any]
    timings: dict[str, float]
    cross_rows: list[tuple[int, str, str, str]] = field(default_factory=list)


class Differ:
    """Memoised per-file JSON diffs; reads member bytes from both zips."""

    def __init__(self, old_zip: zipfile.ZipFile, new_zip: zipfile.ZipFile) -> None:
        self.old_zip = old_zip
        self.new_zip = new_zip
        self.cache: dict[str, FileDiff] = {}

    def get(self, path: str) -> FileDiff:
        if path not in self.cache:
            self.cache[path] = diff_json(self.old_zip.read(path), self.new_zip.read(path))
        return self.cache[path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_pin(repo_root: Path) -> tuple[str | None, int | None]:
    text = (repo_root / CATALOG_REL).read_text(encoding="utf-8")
    match = PIN_RE.search(text)
    if match is None:
        return None, None
    return match.group(1).upper(), text[: match.start()].count("\n") + 1


def server_paths(text: str, stats: Counter) -> list[str]:
    """Full file paths in a string; directory prefixes and f-string fragments are dropped."""
    paths = []
    for match in SERVER_PATH_RE.finditer(text):
        path = match.group(0).rstrip(".,;:)")
        last = path.rsplit("/", 1)[-1]
        if path.endswith(("/", "_")) or "." not in last:
            stats["fragments or directory prefixes dropped (approx)"] += 1
            continue
        paths.append(path)
    return paths


def expand(template: str, values: dict[str, list[str]]) -> list[str] | None:
    """Fill f-string placeholders from known values; None if any placeholder is unknown."""
    names = set(PLACEHOLDER_RE.findall(template))
    if any(name not in values for name in names):
        return None
    texts = [template]
    for name in names:
        texts = [t.replace("{" + name + "}", v) for t in texts for v in values[name]]
    return texts


def scan_python(
    text: str,
    rel: str,
    refs: dict[str, set[str]],
    bare_ids: list[tuple[str, str]],
    stats: Counter,
    is_catalog: bool,
) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        stats["python files that failed to parse"] += 1
        return
    # catalog.py builds some paths with f-strings; these are the only placeholder
    # families expanded (index: range(1, 11) with {index:02d}; spear_id: the
    # _STANDARD_SPEAR_IDS tuple). Other placeholders stay unresolved.
    placeholders: dict[str, list[str]] = {}
    if is_catalog:
        placeholders["index"] = [f"{i:02d}" for i in range(1, 11)]
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_STANDARD_SPEAR_IDS" for t in node.targets
            ):
                spear_ids = [str(s) for s in ast.literal_eval(node.value)]
                placeholders["spear_id"] = spear_ids
                bare_ids += [
                    (f"Weapon_Spear_{s}", f"{rel}:{node.lineno} (_STANDARD_SPEAR_IDS)")
                    for s in spear_ids
                ]
    joined_parts = {
        id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            template = "".join(
                v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
                for v in node.values
            )
            texts = expand(template, placeholders) or [template]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in joined_parts:
            texts = [node.value]
            if is_catalog and BARE_ID_RE.fullmatch(node.value):
                bare_ids.append((node.value, f"{rel}:{node.lineno}"))
        else:
            continue
        for text_piece in texts:
            for path in server_paths(text_piece, stats):
                refs[path].add(f"{rel}:{node.lineno}")


def collect_refs(repo_root: Path, stats: Counter) -> tuple[dict[str, set[str]], list[tuple[str, str]]]:
    """Every Server/ path the simulator tree mentions, with repo file:line evidence.

    Two passes: a raw-line regex (covers comments and single-line strings), and an
    AST pass over Python string constants (covers implicitly concatenated and f-string
    paths split across lines). Both are needed; their evidence is merged.
    """
    refs: dict[str, set[str]] = defaultdict(set)
    bare_ids: list[tuple[str, str]] = []
    for sub in SIM_SUBDIRS:
        for path in sorted((repo_root / sub).rglob("*")):
            if path.suffix not in (".py", ".json") or not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo_root).as_posix()
            stats[f"source files {path.suffix}"] += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                for found in server_paths(line, stats):
                    refs[found].add(f"{rel}:{lineno}")
            if path.suffix == ".py":
                scan_python(text, rel, refs, bare_ids, stats, is_catalog=(rel == CATALOG_REL))
    return refs, bare_ids


def file_index(z: zipfile.ZipFile) -> dict[str, tuple[int, int]]:
    return {i.filename: (i.CRC, i.file_size) for i in z.infolist() if not i.filename.endswith("/")}


def jar_index(path: Path) -> dict[str, tuple[int, int]]:
    with zipfile.ZipFile(path) as z:
        return {
            i.filename: (i.CRC, i.file_size)
            for i in z.infolist()
            if i.filename.endswith(".class")
        }


def jar_version(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        try:
            text = z.read("META-INF/MANIFEST.MF").decode("utf-8", "replace")
        except KeyError:
            return "no MANIFEST.MF"
    match = re.search(r"^Implementation-Version:\s*(\S+)", text, re.M)
    return match.group(1) if match else "not stated"


def group_of(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "Server" and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


def same(a: Any, b: Any) -> bool:
    # Type-aware: 1 and 1.0, or True and 1, are reported as changed.
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def fmt(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= VALUE_MAX else text[: VALUE_MAX - 3] + "..."


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def parse_json(data: bytes) -> tuple[Any, str]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        return json.loads(text), "strict JSON"
    except json.JSONDecodeError:
        pass
    cleaned = JSONC_TOKEN_RE.sub(lambda m: m.group(0) if m.group(0).startswith('"') else "", text)
    try:
        return json.loads(cleaned), "JSONC (comments/trailing commas stripped)"
    except json.JSONDecodeError:
        return None, "unparseable"


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out[prefix] = {}
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list):
        if not obj and prefix:
            out[prefix] = []
        for index, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{index}]"))
    else:
        out[prefix or "$"] = obj
    return out


def diff_json(old_bytes: bytes, new_bytes: bytes) -> FileDiff:
    old_obj, old_mode = parse_json(old_bytes)
    new_obj, new_mode = parse_json(new_bytes)
    mode = f"old: {old_mode}; new: {new_mode}"
    if old_obj is None or new_obj is None:
        old_lines = old_bytes.decode("utf-8", "replace").splitlines()
        new_lines = new_bytes.decode("utf-8", "replace").splitlines()
        lines = list(difflib.unified_diff(old_lines, new_lines, "0.5", "0.7.2", n=0, lineterm=""))
        return FileDiff(mode, [], [], [], None, None, lines[:MAX_LINE_DIFF])
    old_flat, new_flat = flatten(old_obj), flatten(new_obj)
    removed = [(k, old_flat[k]) for k in old_flat if k not in new_flat]
    added = [(k, new_flat[k]) for k in new_flat if k not in old_flat]
    changed = [(k, old_flat[k], new_flat[k]) for k in old_flat if k in new_flat and not same(old_flat[k], new_flat[k])]
    return FileDiff(mode, removed, added, changed, old_flat, new_flat, None)


def describe(fd: FileDiff, key: str) -> str:
    old_v = fd.old_flat.get(key, _MISSING) if fd.old_flat is not None else _MISSING
    new_v = fd.new_flat.get(key, _MISSING) if fd.new_flat is not None else _MISSING
    def show(v: Any) -> str:
        return "absent" if v is _MISSING else fmt(v)
    if old_v is _MISSING and new_v is _MISSING:
        return "no such key in either version"
    return f"{show(old_v)} -> {show(new_v)}"


def render_diff(fd: FileDiff, out: list[str]) -> None:
    out.append(f"- parse: {fd.mode}")
    if fd.line_diff is not None:
        out.append("- not parseable as JSON; line diff (first 40 lines):")
        out.append("```diff")
        out.extend(fd.line_diff or ["(no line differences)"])
        out.append("```")
        return
    entries = (
        [f"- removed `{k}` = {fmt(v)}" for k, v in fd.removed]
        + [f"- added `{k}` = {fmt(v)}" for k, v in fd.added]
        + [f"- changed `{k}`: {fmt(o)} -> {fmt(n)}" for k, o, n in fd.changed]
    )
    if not entries:
        out.append("- no field-level differences (bytes differ only in formatting or key order)")
        return
    out.extend(entries[:MAX_FIELD_LINES])
    if len(entries) > MAX_FIELD_LINES:
        out.append(f"- ... {len(entries) - MAX_FIELD_LINES} more field differences not shown")


def expect_changes(fd: FileDiff | None, expectations: list[tuple[str, Any, Any]]) -> tuple[str, str]:
    if fd is None:
        return "NOT SHOWN", "file missing from one zip, or not JSON"
    changed = {k: (o, n) for k, o, n in fd.changed}
    found, missed = [], []
    for key, old, new in expectations:
        if key in changed and same(changed[key][0], old) and same(changed[key][1], new):
            found.append(f"`{key}` {fmt(old)} -> {fmt(new)}")
        else:
            missed.append(f"`{key}` expected {fmt(old)} -> {fmt(new)}, diff has {describe(fd, key)}")
    status = "SHOWN" if not missed else ("PARTIAL" if found else "NOT SHOWN")
    return status, "; ".join(found + (["missing: " + m for m in missed] if missed else []))


def pair_hits(ctx: Ctx, path_re: str, pairs: list[tuple[Any, Any]], key_re: str | None = None) -> list[tuple[str, str, Any, Any]]:
    hits = []
    for path, fd in ctx.field_diffs.items():
        if not re.search(path_re, path, re.I):
            continue
        for key, old, new in fd.changed:
            if key_re and not re.search(key_re, key):
                continue
            for po, pn in pairs:
                if same(old, po) and same(new, pn):
                    hits.append((path, key, old, new))
                    break
    return hits


def text_count(z: zipfile.ZipFile, prefix: str, needle: bytes) -> int:
    return sum(1 for n in z.namelist() if n.startswith(prefix) and n.endswith(".json") and needle in z.read(n))


def cross_checks(ctx: Ctx) -> list[tuple[int, str, str, str]]:
    rows: list[tuple[int, str, str, str]] = []
    diff = ctx.diffs.get

    def has(path: str) -> FileDiff | None:
        return diff(path) if path in ctx.old_idx and path in ctx.new_idx else None

    sword = "Server/Item/Interactions/Weapons/Sword/Attacks/Primary/Swing_Left/Weapon_Sword_Primary_Swing_Left_Damage.json"
    fd_sword = has(sword)
    status, ev = expect_changes(fd_sword, [
        ("DamageEffects.Knockback.Force", 6.0, 4.3),
        ("DamageEffects.Knockback.Direction.Y", 1, 0.2),
    ])
    if fd_sword is not None:
        ev += f"; Direction.Z {describe(fd_sword, 'DamageEffects.Knockback.Direction.Z')} (unchanged)"
    rows.append((1, "Weapon_Sword_Iron knockback Force 6.0 -> 4.3, direction (0,1,-1) -> (0,0.2,-1)", status, f"`{sword}`: {ev}"))

    hit_files = set()
    for path, fd in ctx.field_diffs.items():
        if not path.startswith("Server/Item/Interactions/Weapons/"):
            continue
        for key, _old, new in fd.changed + [(k, None, v) for k, v in fd.added]:
            if re.search(r"(^|[.\]])Parent$", key) and new == "DamageEntityParent_Hit_Interrupt":
                hit_files.add(path)
    old_text = text_count(ctx.old_zip, "Server/Item/Interactions/", b"DamageEntityParent_Hit_Interrupt")
    new_text = text_count(ctx.new_zip, "Server/Item/Interactions/", b"DamageEntityParent_Hit_Interrupt")
    status2 = "SHOWN" if hit_files else "NOT SHOWN"
    examples = ", ".join(f"`{p}`" for p in sorted(hit_files)[:3])
    rows.append((2, "Weapon attacks inherit DamageEntityParent_Hit_Interrupt (TRAINING says 31 weapon attack files)", status2,
                 f"{len(hit_files)} Server/Item/Interactions/Weapons/ files now have `Parent` = DamageEntityParent_Hit_Interrupt (was DamageEntityParent; this count covers Weapons/ only; TRAINING states 31 but its own breakdown sums to 30); files containing the id under Server/Item/Interactions/: 0.5={old_text}, 0.7.2={new_text}; e.g. {examples}"))

    fd_mana = has("Server/Entity/Stats/Mana.json")
    status3, ev3 = expect_changes(fd_mana, [
        ("Max", 0, 100), ("InitialValue", 0, 100), ("Regenerating[0].Interval", 0.2, 0.3),
    ])
    rows.append((3, "Mana.json 0/0 -> 100/100, +1 per 0.3 s", status3, f"`Server/Entity/Stats/Mana.json`: {ev3}"))

    bow_pairs = [(6, 1), (9, 2), (12, 4), (16, 7)]
    bow_hits = pair_hits(ctx, r"bow", bow_pairs)
    distinct = {(o, n) for _p, _k, o, n in bow_hits}
    if len(distinct) >= len(bow_pairs):
        status4 = "SHOWN"
    elif distinct:
        status4 = "PARTIAL"
    else:
        status4 = "NOT SHOWN"
    ev4 = "; ".join(f"`{p}` `{k}` {fmt(o)} -> {fmt(n)}" for p, k, o, n in bow_hits[:6])
    rows.append((4, "Shortbow Iron charge damage 6/9/12/16 -> 1/2/4/7", status4,
                 f"{len(bow_hits)} bow-path fields match a pair; distinct pairs {len(distinct)}/4. {ev4}"))

    dagger_hits = pair_hits(ctx, r"dagger", [(4, 6)], key_re=r"Physical$")
    status5 = "SHOWN" if dagger_hits else "NOT SHOWN"
    ev5 = "; ".join(f"`{p}` `{k}` {fmt(o)} -> {fmt(n)}" for p, k, o, n in dagger_hits[:6])
    rows.append((5, "Dagger damage Iron 4 -> 6", status5, f"{len(dagger_hits)} dagger `Physical` fields 4 -> 6. {ev5}"))

    status6, ev6 = expect_changes(has("Server/Item/Items/Weapon/Sword/Weapon_Sword_Iron.json"), [
        ("MaxDurability", 120, 250), ("DurabilityLossOnHit", 0.21, 0.13),
    ])
    rows.append((6, "Weapon_Sword_Iron max durability 120 -> 250", status6,
                 f"`Server/Item/Items/Weapon/Sword/Weapon_Sword_Iron.json`: {ev6}"))

    web = "Server/Item/Items/Deco/Deco_SpiderWeb.json"
    fd_web = has(web)
    status7, ev7 = expect_changes(fd_web, [("BlockType.MovementSettings.HorizontalSpeedMultiplier", 0.65, 0.45)])
    if fd_web is not None:
        other = [f"`{k}` {fmt(o)} -> {fmt(n)}" for k, o, n in fd_web.changed
                 if ".MovementSettings." in k and not k.endswith("HorizontalSpeedMultiplier")]
        new_effect = [f"`{k}` = {fmt(v)}" for k, v in fd_web.added if k.endswith("HorizontalSpeedMultiplier")]
        ev7 += "; same block also changed: " + ("; ".join(other) or "none")
        ev7 += "; new effect-level HorizontalSpeedMultiplier: " + ("; ".join(new_effect) or "none")
    rows.append((7, "Spider web speed 0.65 -> 0.45", status7, f"`{web}`: {ev7}"))

    roles_missing = sorted(
        p for p in ctx.refs if p.startswith("Server/NPC/Roles/") and p in ctx.old_idx and p not in ctx.new_idx
    )
    to_faction, elsewhere = [], []
    for p in roles_missing:
        cands = [c for c in ctx.new_by_name.get(Path(p).name, []) if c.startswith("Server/NPC/Roles/")]
        faction_cands = [c for c in cands if "/Faction/" in c]
        if faction_cands:
            to_faction.append(f"`{p}` -> `{faction_cands[0]}`")
        else:
            elsewhere.append(f"`{p}` -> " + (f"`{cands[0]}`" if cands else "no same-name file"))
    status8 = "SHOWN" if to_faction and not elsewhere else ("PARTIAL" if to_faction else "NOT SHOWN")
    rows.append((8, "NPC role paths moved from Aggressive/ and Neutral/ to Faction/", status8,
                 f"{len(roles_missing)} referenced Roles files are absent at their 0.5 path. To Faction/: {len(to_faction)} ({'; '.join(to_faction[:3])}). Elsewhere: {len(elsewhere)} ({'; '.join(elsewhere[:3])})."))

    identical_n, missing_roles, value_changes, key_only = 0, [], [], []
    total_identity = 0
    for p in sorted(ctx.old_idx):
        if not p.startswith("Server/NPC/Roles/") or not re.search(r"(?i)trork|kweebec", Path(p).name):
            continue
        total_identity += 1
        cands = [c for c in ctx.new_by_name.get(Path(p).name, []) if c.startswith("Server/NPC/Roles/")]
        if any(ctx.new_idx[c] == ctx.old_idx[p] for c in cands):
            identical_n += 1
        elif not cands:
            missing_roles.append(f"`{p}`")
        else:
            fd_role = diff_json(ctx.old_zip.read(p), ctx.new_zip.read(cands[0]))
            if fd_role.line_diff is not None or fd_role.changed:
                detail = "unparseable" if fd_role.line_diff is not None else "; ".join(
                    f"`{k}` {fmt(o)} -> {fmt(n)}" for k, o, n in fd_role.changed[:2])
                value_changes.append(f"`{p}`: {detail}")
            else:
                key_only.append(f"`{p}` ({len(fd_role.added)} keys added, {len(fd_role.removed)} removed)")
    if value_changes:
        status9 = "NOT SHOWN (value changes)"
    elif key_only or missing_roles:
        status9 = "PARTIAL (no value changes; keys or format differ)"
    else:
        status9 = "SHOWN (unchanged)"
    rows.append((9, "Trork and Kweebec role files unchanged", status9,
                 f"{total_identity} Trork/Kweebec Roles files: {identical_n} byte-identical (name match), {len(key_only)} differ in keys or format only, {len(value_changes)} with value changes, {len(missing_roles)} missing. Value changes: {'; '.join(value_changes[:4]) or 'none'}. Key-only examples: {'; '.join(key_only[:3]) or 'none'}."))

    move = "Server/Entity/MovementConfig/Default.json"
    if move in ctx.old_idx and ctx.old_idx[move] == ctx.new_idx.get(move):
        status10, ev10 = "UNCHANGED", "CRC and size equal in both zips"
    elif move in ctx.new_idx:
        fd_move = has(move)
        status10 = f"CHANGED ({len(fd_move.added)} added, {len(fd_move.removed)} removed, {len(fd_move.changed)} changed)"
        ev10 = "; ".join(
            [f"added `{k}` = {fmt(v)}" for k, v in fd_move.added]
            + [f"removed `{k}` = {fmt(v)}" for k, v in fd_move.removed]
            + [f"changed `{k}` {fmt(o)} -> {fmt(n)}" for k, o, n in fd_move.changed]
        ) or "bytes differ, no parsed field differences"
    else:
        status10, ev10 = "NOT SHOWN", "absent from one zip"
    rows.append((10, "MovementConfig/Default.json (player walk/run/sprint/jump)", status10, f"`{move}`: {ev10}"))
    return rows


def render_report(ctx: Ctx) -> str:
    a = ctx.args
    old_paths = set(ctx.old_idx)
    new_paths = set(ctx.new_idx)
    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)
    common = sorted(old_paths & new_paths)
    changed = [p for p in common if ctx.old_idx[p] != ctx.new_idx[p]]
    changed_set = set(changed)
    unchanged = len(common) - len(changed)
    out: list[str] = []
    w = out.append

    w("# Hytale 0.5 -> 0.7.2 asset and jar diff (P0)")
    w("")
    w(f"Generated {datetime.now().isoformat(timespec='seconds')} by `tools/diff_assets_072.py` (stdlib only; read-only on inputs).")
    w("Measured on the zips and jars below. Simulator references come from `HytaleRL/hytalegym/{hytalegym,tests}` (`*.py`, `*.json`).")
    w("")
    w("## Inputs")
    w("")
    w("| Role | Path | Bytes | SHA-256 |")
    w("|---|---|---:|---|")
    w(f"| old assets | `{a.old_assets}` | {a.old_assets.stat().st_size:,} | `{ctx.shas['old assets']}` |")
    w(f"| new assets | `{a.new_assets}` | {a.new_assets.stat().st_size:,} | `{ctx.shas['new assets']}` |")
    w(f"| old jar | `{a.old_jar}` | {a.old_jar.stat().st_size:,} | `{ctx.shas['old jar']}` |")
    w(f"| new jar | `{a.new_jar}` | {a.new_jar.stat().st_size:,} | `{ctx.shas['new jar']}` |")
    w("")
    w(f"Jar `Implementation-Version`: old `{ctx.jar['old_version']}`, new `{ctx.jar['new_version']}`.")
    w("")
    old_sha = ctx.shas["old assets"]
    if ctx.pin is None:
        w(f"**Pin check: could not read the pin from `{CATALOG_REL}`.**")
    elif old_sha == ctx.pin:
        w(f"**Pin check: MATCH.** Old assets sha256 equals the simulator pin `{ctx.pin}` (`{CATALOG_REL}:{ctx.pin_line}`).")
    else:
        w(f"**Pin check: MISMATCH.** Old assets sha256 `{old_sha}` does not equal the simulator pin `{ctx.pin}` (`{CATALOG_REL}:{ctx.pin_line}`); the pin starts `{ctx.pin[:8]}`, this zip starts `{old_sha[:8]}`.")
        w(f"The 0.5 jar reports `{ctx.jar['old_version']}`, so the installed 0.5 tree is not the 0.5.7 that the pin was taken from. The TRAINING-0.7.2 measurements were also taken against 0.5.10. Treat \"0.5\" in this report as the installed build.")
    w("")

    w("## Summary counts")
    w("")
    w(f"- Asset entries (files): old {len(old_paths):,}, new {len(new_paths):,}. Added {len(added):,}, removed {len(removed):,}, changed {len(changed):,}, unchanged {unchanged:,} (compared by CRC-32 and size).")
    w("")
    w("| Folder | Old | New | Added | Removed | Changed | Unchanged |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    groups = sorted({group_of(p) for p in old_paths | new_paths})
    for g in groups:
        ch = sum(1 for p in changed if group_of(p) == g)
        ad = sum(1 for p in added if group_of(p) == g)
        rm = sum(1 for p in removed if group_of(p) == g)
        o_n = sum(1 for p in old_paths if group_of(p) == g)
        n_n = sum(1 for p in new_paths if group_of(p) == g)
        un = sum(1 for p in common if group_of(p) == g and p not in changed_set)
        w(f"| `{g}` | {o_n:,} | {n_n:,} | {ad:,} | {rm:,} | {ch:,} | {un:,} |")
    w("")
    refs = ctx.refs
    ref_paths = sorted(refs)
    in_old = [p for p in ref_paths if p in old_paths]
    ref_changed = [p for p in in_old if p in changed_set]
    ref_missing = [p for p in in_old if p not in new_paths]
    ref_unchanged = [p for p in in_old if p in new_paths and p not in changed_set]
    w("Simulator references (`Server/...` paths in `hytalegym/` and `tests/`):")
    w("")
    w(f"- Source files scanned: {ctx.stats['source files .py']:,} `.py`, {ctx.stats['source files .json']:,} `.json`.")
    w(f"- Unique referenced file paths: {len(ref_paths):,}. Present in 0.5 zip: {len(in_old):,}. Not present in 0.5 zip (unresolved): {len(ref_paths) - len(in_old):,}.")
    w(f"- Of the 0.5 ones: changed {len(ref_changed):,}, unchanged {len(ref_unchanged):,}, missing at the same path in 0.7.2 {len(ref_missing):,}.")
    w(f"- Bare asset ids from `{CATALOG_REL}` (quoted `Prefix_Name` literals plus `_STANDARD_SPEAR_IDS` expanded to `Weapon_Spear_<id>`): {ctx.bare_id_count:,} unique ids; {len(ctx.unmapped_ids):,} had no same-stem file in the 0.5 zip.")
    w(f"- Dropped as fragments or directory prefixes (approx): {ctx.stats['fragments or directory prefixes dropped (approx)']:,}. Python files failing to parse: {ctx.stats['python files that failed to parse']:,}.")
    w("")
    jar = ctx.jar
    w(f"- Server jar classes: old {jar['old_count']:,}, new {jar['new_count']:,}. Added {len(jar['added']):,}, removed {len(jar['removed']):,}, changed {len(jar['changed']):,}, of which in keyword packages {len(jar['keyword_changed']):,}.")
    w("")

    w("## Simulator-referenced files changed")
    w("")
    if not ref_changed:
        w("None.")
    for p in ref_changed:
        w(f"### `{p}`")
        w("")
        w(f"Referenced by: {', '.join(f'`{e}`' for e in sorted(refs[p])[:3])}" + (f" (+{len(refs[p]) - 3} more)" if len(refs[p]) > 3 else ""))
        w("")
        if p.endswith(".json"):
            render_diff(ctx.diffs.get(p), out)
        else:
            w("- not JSON; bytes differ (CRC/size); no field diff computed")
        w("")

    w("## Simulator-referenced files moved or missing")
    w("")
    if not ref_missing:
        w("No referenced 0.5 path is absent from 0.7.2.")
    else:
        w("| 0.5 path (referenced by) | Same name in 0.7.2 | Where |")
        w("|---|---|---|")
        for p in ref_missing:
            cands = sorted(ctx.new_by_name.get(Path(p).name, []))
            where = ", ".join(f"`{c}`" for c in cands[:4]) if cands else "no file with this name"
            w(f"| `{p}` (`{sorted(refs[p])[0]}`) | {'yes' if cands else 'no'} | {where} |")
    w("")
    w(f"Referenced paths absent from the 0.5 zip ({len(ref_paths) - len(in_old)}; listed in the report, not diffed):")
    w("")
    unresolved = [p for p in ref_paths if p not in old_paths]
    for p in unresolved[:MAX_LIST_LINES]:
        w(f"- `{p}` (`{sorted(refs[p])[0]}`)")
    if len(unresolved) > MAX_LIST_LINES:
        w(f"- ... {len(unresolved) - MAX_LIST_LINES} more")
    if ctx.unmapped_ids:
        w("")
        w(f"Bare ids with no same-stem file in the 0.5 zip (not assets, or not under Server/): {', '.join(f'`{i}`' for i in ctx.unmapped_ids[:60])}" + (" ..." if len(ctx.unmapped_ids) > 60 else ""))
    w("")

    w("## Unreferenced but likely relevant changes")
    w("")
    w("Changed files under the focus folders that the simulator does not reference:")
    w("")
    w("| Folder | Changed | Unreferenced changed | Added | Removed |")
    w("|---|---:|---:|---:|---:|")
    for pre in FOCUS_PREFIXES:
        ch = [p for p in changed if p.startswith(pre)]
        w(f"| `{pre}` | {len(ch):,} | {sum(1 for p in ch if p not in refs):,} | {sum(1 for p in added if p.startswith(pre)):,} | {sum(1 for p in removed if p.startswith(pre)):,} |")
    w("")
    unref = [p for p in changed if p.startswith(FOCUS_PREFIXES) and p not in refs]
    for p in unref[:MAX_LIST_LINES]:
        fd = ctx.field_diffs.get(p)
        note = f" ({len(fd.changed)} fields)" if fd is not None and fd.line_diff is None else ""
        w(f"- `{p}`{note}")
    if len(unref) > MAX_LIST_LINES:
        w(f"- ... {len(unref) - MAX_LIST_LINES} more")
    w("")

    w("## Jar class changes")
    w("")
    w(f"Classes: old {jar['old_count']:,}, new {jar['new_count']:,}; added {len(jar['added']):,}, removed {len(jar['removed']):,}, changed {len(jar['changed']):,}.")
    w("")
    w(f"Changed classes whose package path contains one of: {', '.join(JAR_KEYWORDS)} (case-insensitive substring; `stat` also matches `state`/`static`). {len(jar['keyword_changed']):,} classes in {len(jar['keyword_pkgs'])} packages.")
    w("")
    lines = 0
    for pkg, classes in sorted(jar["keyword_pkgs"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if lines >= MAX_JAR_LINES:
            break
        w(f"- `{pkg}`: {len(classes)} changed")
        lines += 1
        for cls in sorted(classes):
            if lines >= MAX_JAR_LINES:
                break
            w(f"  - `{cls}`")
            lines += 1
    if lines >= MAX_JAR_LINES:
        w(f"- ... output capped at {MAX_JAR_LINES} lines")
    w("")
    w(f"Added classes (first 60 of {len(jar['added']):,}): " + ", ".join(f"`{c}`" for c in jar["added"][:60]))
    w("")
    w(f"Removed classes (first 60 of {len(jar['removed']):,}): " + ", ".join(f"`{c}`" for c in jar["removed"][:60]))
    w("")

    w("## Cross-check against measured 0.7.2 changes")
    w("")
    w("Source of the 10 items: `HytaleRL/hytale-plugin-0.7.2/TRAINING-0.7.2.md` (\"Gameplay changes that matter\"). Status is what this diff shows; the `Evidence` column names the zip path and flattened JSON field (`a.b[0].c`).")
    w("")
    w("| # | Measured change | Diff shows it | Evidence |")
    w("|---|---|---|---|")
    for num, label, status, evidence in ctx.cross_rows:
        w(f"| {num} | {cell(label)} | {cell(status)} | {cell(evidence)} |")
    w("")
    w("## Caveats")
    w("")
    w("- Flattened keys use `.` for objects and `[i]` for arrays. A field that moves to another path shows as a removed key plus an added key; cross-check items match exact flattened keys.")
    w("- Path matching for moved NPC roles and identity checks is by file name (with extension) within `Server/NPC/Roles/`, not by full path.")
    w("- Placeholder expansion in `catalog.py` is limited to `{index:02d}` (range 1..10) and `{spear_id}` (from `_STANDARD_SPEAR_IDS`); other f-string paths stay unresolved.")
    w("- A pattern in the diff only shows that a value changed in the assets. It does not show engine semantics (for example, whether the knockback or interrupt changes the agent's motion); those stay with TRAINING-0.7.2.md.")
    w("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 diff inventory: Hytale 0.5 vs 0.7.2 assets and server jar.")
    parser.add_argument("--old-assets", type=Path, required=True, help="0.5 Assets.zip")
    parser.add_argument("--new-assets", type=Path, required=True, help="0.7.2 Assets.zip")
    parser.add_argument("--old-jar", type=Path, required=True, help="0.5 Server/HytaleServer.jar")
    parser.add_argument("--new-jar", type=Path, required=True, help="0.7.2 Server/HytaleServer.jar")
    parser.add_argument("--repo-root", type=Path, default=Path("/Users/arush/hydl"))
    parser.add_argument("--output", type=Path, required=True, help="markdown report to write")
    args = parser.parse_args(argv)

    timings: dict[str, float] = {}
    started = time.monotonic()
    inputs = {"old assets": args.old_assets, "new assets": args.new_assets,
              "old jar": args.old_jar, "new jar": args.new_jar}
    for label, path in inputs.items():
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        shas = dict(zip(inputs, pool.map(sha256_file, inputs.values())))
    timings["sha256"] = time.monotonic() - started
    pin, pin_line = read_pin(args.repo_root)

    stats: Counter = Counter()
    refs, bare_ids = collect_refs(args.repo_root, stats)
    timings["refs"] = time.monotonic() - started

    with zipfile.ZipFile(args.old_assets) as old_z, zipfile.ZipFile(args.new_assets) as new_z:
        old_idx, new_idx = file_index(old_z), file_index(new_z)
        old_stems: dict[str, list[str]] = defaultdict(list)
        for p in old_idx:
            if p.startswith("Server/"):
                old_stems[Path(p).stem].append(p)
        new_by_name: dict[str, list[str]] = defaultdict(list)
        for p in new_idx:
            if p.startswith("Server/"):
                new_by_name[Path(p).name].append(p)
        unmapped: set[str] = set()
        for asset_id, evidence in set(bare_ids):
            hits = old_stems.get(asset_id, [])
            if not hits:
                unmapped.add(asset_id)
            for p in hits:
                refs[p].add(f"{evidence} (bare id {asset_id})")
        bare_unique = {asset_id for asset_id, _ in bare_ids}

        diffs = Differ(old_z, new_z)
        common = sorted(set(old_idx) & set(new_idx))
        changed_paths = [p for p in common if old_idx[p] != new_idx[p]]
        field_diffs = {
            p: diffs.get(p)
            for p in changed_paths
            if p.endswith(".json") and p.startswith(FIELD_PREFIXES)
        }
        timings["field diffs"] = time.monotonic() - started

        jar_old, jar_new = jar_index(args.old_jar), jar_index(args.new_jar)
        jar_common = set(jar_old) & set(jar_new)
        jar_changed = sorted(c for c in jar_common if jar_old[c] != jar_new[c])
        keyword_pkgs: dict[str, list[str]] = defaultdict(list)
        for c in jar_changed:
            pkg = c.rsplit("/", 1)[0] if "/" in c else "(default)"
            if any(kw in pkg.lower() for kw in JAR_KEYWORDS):
                keyword_pkgs[pkg].append(c)
        jar = {
            "old_version": jar_version(args.old_jar),
            "new_version": jar_version(args.new_jar),
            "old_count": len(jar_old),
            "new_count": len(jar_new),
            "added": sorted(set(jar_new) - set(jar_old)),
            "removed": sorted(set(jar_old) - set(jar_new)),
            "changed": jar_changed,
            "keyword_changed": [c for pkgs in keyword_pkgs.values() for c in pkgs],
            "keyword_pkgs": keyword_pkgs,
        }

        ctx = Ctx(
            args=args, shas=shas, pin=pin, pin_line=pin_line, stats=stats,
            old_idx=old_idx, new_idx=new_idx, new_by_name=new_by_name, refs=refs,
            unmapped_ids=sorted(unmapped), bare_id_count=len(bare_unique),
            changed_paths=changed_paths, diffs=diffs, field_diffs=field_diffs,
            old_zip=old_z, new_zip=new_z, jar=jar, timings=timings,
        )
        ctx.cross_rows = cross_checks(ctx)
        report = render_report(ctx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    timings["total"] = time.monotonic() - started
    print(f"wrote {args.output} ({len(report.splitlines())} lines) in {timings['total']:.1f}s")
    for label, value in timings.items():
        print(f"  {label}: {value:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
