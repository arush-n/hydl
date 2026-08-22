"""Run the console's real-browser smoke gate and write a JUnit receipt.

The gate starts an isolated no-reload server on a free loopback port and drives
Chromium through the Playwright CLI.  It never reuses or stops the developer's
normal console process on port 8770.

    python -m console.tools.browser_receipt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / ".codex-local" / "test-results"
    / "console-browser-ui-v1.junit.xml"
)
DEFAULT_ARTIFACTS = ROOT / "output" / "playwright" / "console-browser-ui-v1"


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any


class BrowserGate:
    def __init__(self, *, port: int, artifacts: Path) -> None:
        self.port = port or _free_port()
        self.artifacts = artifacts
        self.session = f"console-browser-receipt-{os.getpid()}"
        self.server: subprocess.Popen[str] | None = None
        self.npx = shutil.which("npx.cmd") or shutil.which("npx")
        if not self.npx:
            raise RuntimeError("npx is required to run the Playwright CLI")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.server = subprocess.Popen(
            [
                sys.executable, "-m", "console.server", "--port",
                str(self.port), "--no-reload",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=flags,
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                output = self.server.stdout.read() if self.server.stdout else ""
                raise RuntimeError(f"console server exited early: {output.strip()}")
            try:
                with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
                    f"{self.url}/api/panels", timeout=2,
                ) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError(f"console server did not become ready at {self.url}")

    def cli(self, *args: str, raw: bool = False, timeout: int = 120) -> str:
        command = [
            self.npx, "--yes", "--package", "@playwright/cli",
            "playwright-cli", f"-s={self.session}",
        ]
        if raw:
            command.append("--raw")
        command.extend(args)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            message = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Playwright CLI {args[0]!r} failed ({completed.returncode}): "
                f"{message}"
            )
        return completed.stdout.strip()

    def evaluate(self, expression: str) -> dict[str, Any]:
        # npm's Windows .cmd shim does not preserve embedded newlines in one
        # argv value; Playwright receives a truncated function and reports a
        # misleading syntax error. Whitespace is insignificant in these
        # function expressions, so send one physical command-line argument.
        expression = " ".join(line.strip() for line in expression.splitlines())
        output = self.cli("eval", expression, raw=True)
        value = json.loads(output)
        if not isinstance(value, dict):
            raise RuntimeError(f"browser evaluation returned {type(value).__name__}")
        return value

    def screenshot(self, name: str) -> Path:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        path = self.artifacts / name
        relative = path.relative_to(ROOT).as_posix()
        self.cli("screenshot", "--filename", relative)
        if not path.is_file():
            raise RuntimeError(f"Playwright did not write {relative}")
        return path

    def close(self) -> None:
        try:
            self.cli("close", timeout=30)
        except Exception:  # noqa: BLE001 - cleanup must reach the server
            pass
        if self.server is None or self.server.poll() is not None:
            return
        self.server.terminate()
        try:
            self.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait(timeout=5)


def run_gate(gate: BrowserGate) -> list[Check]:
    checks: list[Check] = []
    gate.start()
    gate.cli("open", f"{gate.url}/#evidence")
    gate.cli("resize", "1440", "1000")

    desktop = gate.evaluate("""async () => {
      const active = document.querySelector('[data-panel=evidence]');
      for (let i = 0; i < 450; i += 1) {
        if (active && active.querySelectorAll('[aria-busy=true]').length === 0 &&
            active.querySelectorAll('.panelCard').length >= 6) break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      const paths = performance.getEntriesByType('resource')
        .map(entry => new URL(entry.name).pathname);
      const requested = paths.filter(path => path.startsWith('/api/panels/'))
        .map(path => decodeURIComponent(path.split('/').at(-1)));
      const activeIds = [...active.querySelectorAll('.panelCard')]
        .map(card => card.id.replace(/^panel-/, ''));
      return {
        href: location.href,
        cards: document.querySelectorAll('.panelCard').length,
        activeCards: activeIds,
        activeBusy: active.querySelectorAll('[aria-busy=true]').length,
        failed: document.querySelectorAll('.panelCardFailed').length,
        hiddenPlaceholders: [...document.querySelectorAll('.panel[hidden] .panelBody')]
          .filter(body => body.textContent.includes('Open this workspace')).length,
        requested: [...new Set(requested)],
        unexpectedRequests: [...new Set(requested)]
          .filter(id => !activeIds.includes(id)),
        width: document.documentElement.scrollWidth,
        viewport: innerWidth,
        height: document.documentElement.scrollHeight,
        sections: active.querySelectorAll('details.panelSection').length,
        openSections: active.querySelectorAll('details.panelSection[open]').length,
        scrollRegions: active.querySelectorAll('.panelRowsScrollable').length,
      };
    }""")
    checks.extend([
        Check(
            "evidence_workspace_loads",
            desktop["activeBusy"] == 0 and len(desktop["activeCards"]) >= 6,
            desktop,
        ),
        Check("no_failed_frames", desktop["failed"] == 0, desktop),
        Check(
            "hidden_workspaces_are_lazy",
            not desktop["unexpectedRequests"] and desktop["hiddenPlaceholders"] > 0,
            desktop,
        ),
        Check(
            "desktop_has_no_page_overflow",
            desktop["width"] <= desktop["viewport"],
            desktop,
        ),
        Check(
            "long_evidence_is_progressively_disclosed",
            desktop["sections"] > desktop["openSections"] > 0
            and desktop["scrollRegions"] >= 1,
            desktop,
        ),
    ])
    desktop_shot = gate.screenshot("desktop-evidence.png")
    checks.append(Check("desktop_screenshot_written", desktop_shot.is_file(), str(desktop_shot)))

    filtering = gate.evaluate("""() => {
      const input = document.querySelector('[data-panel-filter=evidence]');
      input.value = 'multi-actor-population-checkpoint-v1-r3';
      input.dispatchEvent(new Event('input', {bubbles: true}));
      const cards = [...document.querySelectorAll('[data-panel=evidence] .panelCard')]
        .filter(card => !card.hidden).map(card => card.id);
      const rows = [...document.querySelectorAll('#panel-test-receipts .panelRow')]
        .filter(row => !row.hidden).map(row => row.textContent.replace(/\\s+/g, ' ').trim());
      const count = document.querySelector(
        '[data-panel=evidence] [data-panel-tool-count]'
      ).textContent;
      return {cards, rows, count};
    }""")
    checks.append(Check(
        "filter_finds_receipt_family_and_supersession",
        filtering["cards"] == ["panel-test-receipts"]
        and len(filtering["rows"]) == 4
        and all("multi-actor-population-checkpoint-v1-r3" in row
                for row in filtering["rows"])
        and "4 matches" in filtering["count"],
        filtering,
    ))

    reveal = gate.evaluate("""() => {
      const input = document.querySelector('[data-panel-filter=evidence]');
      input.value = 'hytalerl_native_npc_projectile_trace_probe_v1';
      input.dispatchEvent(new Event('input', {bubbles: true}));
      const matched = [...document.querySelectorAll(
        '[data-panel=evidence] details.panelSection:not([hidden])'
      )];
      return {
        matched: matched.length,
        openedFromClosed: matched.filter(section =>
          section.open && section.dataset.filterOpen === 'false'
        ).length,
      };
    }""")
    checks.append(Check(
        "filter_reveals_matches_inside_collapsed_sections",
        reveal["matched"] >= 1 and reveal["openedFromClosed"] >= 1,
        reveal,
    ))

    disclosure = gate.evaluate("""async () => {
      const input = document.querySelector('[data-panel-filter=evidence]');
      input.value = '';
      input.dispatchEvent(new Event('input', {bubbles: true}));
      document.querySelector(
        '[data-panel=evidence] [data-sections=collapse]'
      ).click();
      await new Promise(resolve => setTimeout(resolve, 0));
      const collapsed = document.querySelectorAll(
        '[data-panel=evidence] details.panelSection[open]'
      ).length;
      document.querySelector(
        '[data-panel=evidence] [data-sections=expand]'
      ).click();
      await new Promise(resolve => setTimeout(resolve, 0));
      return {
        collapsed,
        expanded: document.querySelectorAll(
          '[data-panel=evidence] details.panelSection[open]'
        ).length,
        total: document.querySelectorAll(
          '[data-panel=evidence] details.panelSection'
        ).length,
      };
    }""")
    checks.append(Check(
        "collapse_and_expand_controls_work",
        disclosure["collapsed"] == 0
        and disclosure["expanded"] == disclosure["total"] > 0,
        disclosure,
    ))

    status = gate.evaluate("""async () => {
      location.hash = 'status';
      const panel = document.querySelector('[data-panel=status]');
      for (let i = 0; i < 300; i += 1) {
        const cards = [...panel.querySelectorAll('.panelCard')];
        if (!panel.hidden && cards.length >= 4 &&
            cards.every(card => !card.querySelector('[aria-busy=true]'))) break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      const bodies = [...panel.querySelectorAll('.panelBody')];
      return {
        visible: !panel.hidden,
        cards: bodies.length,
        placeholders: bodies.filter(
          body => body.textContent.includes('Open this workspace')
        ).length,
        policyRequests: performance.getEntriesByType('resource')
          .map(entry => new URL(entry.name).pathname)
          .filter(path => path === '/api/policy-agent').length,
      };
    }""")
    checks.append(Check(
        "workspace_switch_loads_status_and_live_policy",
        status["visible"] and status["cards"] >= 4
        and status["placeholders"] == 0 and status["policyRequests"] >= 1,
        status,
    ))

    compute = gate.evaluate("""async () => {
      const before = performance.getEntriesByType('resource')
        .filter(entry => new URL(entry.name).pathname === '/api/devices').length;
      location.hash = 'compute';
      for (let i = 0; i < 300; i += 1) {
        const after = performance.getEntriesByType('resource')
          .filter(entry => new URL(entry.name).pathname === '/api/devices').length;
        if (after > before && document.querySelectorAll('#deviceFacts .ro').length) {
          return {before, after, facts: document.querySelectorAll('#deviceFacts .ro').length};
        }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      const after = performance.getEntriesByType('resource')
        .filter(entry => new URL(entry.name).pathname === '/api/devices').length;
      return {before, after, facts: document.querySelectorAll('#deviceFacts .ro').length};
    }""")
    checks.append(Check(
        "device_polling_starts_on_compute",
        compute["before"] == 0 and compute["after"] > 0 and compute["facts"] > 0,
        compute,
    ))

    mobile = gate.evaluate("""async () => {
      location.hash = 'evidence';
      for (let i = 0; i < 450; i += 1) {
        const panel = document.querySelector('[data-panel=evidence]');
        if (!panel.hidden && !panel.querySelector('[aria-busy=true]')) break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return {ready: true};
    }""")
    if not mobile["ready"]:
        raise RuntimeError("Evidence did not become ready before mobile QA")
    gate.cli("resize", "390", "844")
    mobile = gate.evaluate("""() => {
      const chrome = document.querySelector('.chrome').getBoundingClientRect();
      const tabs = document.querySelector('.tabs');
      const tools = document.querySelector('[data-panel=evidence] .panelTools');
      return {
        width: document.documentElement.scrollWidth,
        viewport: innerWidth,
        chromeHeight: chrome.height,
        tabs: tabs.querySelectorAll('[role=tab]').length,
        tabsWidth: tabs.getBoundingClientRect().width,
        toolWidth: tools.getBoundingClientRect().width,
        toolPosition: getComputedStyle(tools).position,
      };
    }""")
    checks.append(Check(
        "mobile_layout_has_no_page_overflow",
        mobile["width"] <= mobile["viewport"] and mobile["tabs"] == 8
        and mobile["tabsWidth"] <= mobile["viewport"]
        and mobile["toolWidth"] <= mobile["viewport"]
        and mobile["toolPosition"] == "static",
        mobile,
    ))
    mobile_shot = gate.screenshot("mobile-evidence.png")
    checks.append(Check("mobile_screenshot_written", mobile_shot.is_file(), str(mobile_shot)))

    console = gate.cli("console", "error", raw=True)
    checks.append(Check(
        "browser_console_has_no_errors",
        "Errors: 0" in console and "Warnings: 0" in console,
        console,
    ))
    return checks


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _write_receipt(path: Path, checks: list[Check], elapsed: float) -> None:
    failures = sum(not check.passed for check in checks)
    suite = ET.Element("testsuite", {
        "name": "console-browser-ui-v1",
        "tests": str(len(checks)),
        "failures": str(failures),
        "errors": "0",
        "skipped": "0",
        "time": f"{elapsed:.3f}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    for check in checks:
        case = ET.SubElement(suite, "testcase", {
            "classname": "console.browser",
            "name": check.name,
        })
        detail = check.detail if isinstance(check.detail, str) else json.dumps(
            check.detail, ensure_ascii=False, sort_keys=True,
        )
        if not check.passed:
            failure = ET.SubElement(case, "failure", {
                "message": f"browser check failed: {check.name}",
            })
            failure.text = detail
        output = ET.SubElement(case, "system-out")
        output.text = detail
    ET.indent(suite, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="isolated port; 0 chooses a free one")
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()

    started = time.monotonic()
    checks: list[Check] = []
    gate: BrowserGate | None = None
    try:
        gate = BrowserGate(port=args.port, artifacts=args.artifacts)
        checks = run_gate(gate)
    except Exception as exc:  # noqa: BLE001 - failure belongs in the receipt
        checks.append(Check("browser_gate_execution", False, f"{type(exc).__name__}: {exc}"))
    finally:
        if gate is not None:
            gate.close()

    elapsed = time.monotonic() - started
    _write_receipt(args.output, checks, elapsed)
    failures = sum(not check.passed for check in checks)
    print(
        f"{args.output}: tests={len(checks)} failures={failures} "
        f"errors=0 skipped=0 ({elapsed:.2f}s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
