from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print(">", " ".join(args))
    cp = subprocess.run(args, cwd=ROOT, text=True)
    if check and cp.returncode != 0:
        raise SystemExit(cp.returncode)
    return cp


def output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def fail(message: str) -> None:
    raise SystemExit(f"\\nERROR: {message}\\n")


required = [
    ROOT / "universe.json",
    ROOT / "our_agent",
    ROOT / "tests",
    ROOT / "tools",
    ROOT / ".gitignore",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    fail("Run this from starter-kit. Missing: " + repr(missing))

if output("git", "branch", "--show-current") != "our-agent":
    fail("Expected Git branch 'our-agent'.")

if subprocess.run(
    ["git", "check-ignore", "-q", ".env"],
    cwd=ROOT,
).returncode != 0:
    fail(".env must remain Git-ignored.")

print("=== ICAIF PHASE 5: SAFETY PREFLIGHT + COST FORECAST ===")
print("Competition POST/submission remains DISABLED.")
print()

files = {'our_agent/execution/preflight.py': '"""Pre-submission safety checks for ICAIF decisions.\n\nThis module validates decision intent and timing context. It never performs\nnetwork POSTs and contains no submission function.\n"""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass, field\nfrom datetime import datetime\nfrom typing import Mapping\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.intents import StrategyIntent\nfrom our_agent.validation import validate_target_weights\n\n\n@dataclass(frozen=True)\nclass PreflightContext:\n    phase: str\n    round_id: str\n    schedule_status: str\n    opens_at: datetime\n    deadline: datetime\n    now: datetime\n    data_cutoff: datetime\n    prior_attempt_exists: bool = False\n\n\n@dataclass\nclass PreflightReport:\n    ok: bool\n    errors: list[str] = field(default_factory=list)\n    warnings: list[str] = field(default_factory=list)\n\n\ndef run_preflight(\n    *,\n    intent: StrategyIntent,\n    weights: Mapping[str, object] | None,\n    context: PreflightContext,\n) -> PreflightReport:\n    errors: list[str] = []\n    warnings: list[str] = []\n\n    for name, value in {\n        "opens_at": context.opens_at,\n        "deadline": context.deadline,\n        "now": context.now,\n        "data_cutoff": context.data_cutoff,\n    }.items():\n        if value.tzinfo is None or value.utcoffset() is None:\n            errors.append(f"{name} must be timezone-aware.")\n\n    if errors:\n        return PreflightReport(ok=False, errors=errors, warnings=warnings)\n\n    if context.schedule_status.upper() == "CANCELLED":\n        errors.append("Round is CANCELLED.")\n\n    if context.prior_attempt_exists:\n        errors.append(\n            "A prior attributable attempt already exists for this round."\n        )\n\n    # Official eligibility rule: opens_at <= upload_time < deadline.\n    if context.now < context.opens_at:\n        errors.append("Decision window has not opened yet.")\n\n    if context.now >= context.deadline:\n        errors.append("Decision deadline has passed.")\n\n    if context.data_cutoff > context.now:\n        errors.append("Data cutoff is in the future relative to now.")\n\n    if context.data_cutoff >= context.deadline:\n        errors.append(\n            "Data cutoff is not safely before the exclusive deadline."\n        )\n\n    if intent == StrategyIntent.HOLD:\n        if weights is not None:\n            errors.append("HOLD must not carry target weights.")\n    else:\n        if weights is None:\n            errors.append(f"{intent.value} requires target weights.")\n        else:\n            try:\n                validate_target_weights(weights, intent)\n            except Exception as exc:\n                errors.append(f"Weight validation failed: {exc}")\n\n    if context.deadline <= context.opens_at:\n        errors.append("Schedule has non-positive decision window.")\n\n    if context.phase not in {"validation", "official"}:\n        warnings.append(\n            f"Unexpected phase {context.phase!r}; verify live schedule."\n        )\n\n    if not context.round_id:\n        errors.append("round_id is empty.")\n\n    return PreflightReport(\n        ok=not errors,\n        errors=errors,\n        warnings=warnings,\n    )\n', 'our_agent/execution/cost_forecast.py': '"""Turnover and fee forecast proxy.\n\nImportant: this is NOT claimed to reproduce the private backend target-weight\nsolver exactly. It is a transparent pre-trade estimate based on changes in\nportfolio weights.\n\nFor a fully invested portfolio, sum(abs(target-current)) approximates\nbuy-plus-sell notional divided by NAV. From cash to 80% equity, the proxy is\n0.80. Rotating between disjoint 80% equity portfolios gives about 1.60.\n"""\n\nfrom __future__ import annotations\n\nfrom typing import Mapping\n\nfrom our_agent.config import FEE_RATE, OFFICIAL_SYMBOLS\n\n\ndef gross_weight_change_proxy(\n    current: Mapping[str, float],\n    target: Mapping[str, float],\n) -> float:\n    expected = set(OFFICIAL_SYMBOLS)\n\n    if set(current) != expected:\n        raise ValueError("current must contain exactly the 30 official symbols")\n    if set(target) != expected:\n        raise ValueError("target must contain exactly the 30 official symbols")\n\n    return sum(\n        abs(float(target[symbol]) - float(current[symbol]))\n        for symbol in OFFICIAL_SYMBOLS\n    )\n\n\ndef fee_forecast_dollars(\n    *,\n    nav: float,\n    gross_turnover_proxy: float,\n) -> float:\n    if nav <= 0:\n        raise ValueError("nav must be positive")\n    if gross_turnover_proxy < 0:\n        raise ValueError("gross_turnover_proxy must be non-negative")\n\n    return nav * gross_turnover_proxy * float(FEE_RATE)\n', 'tools/preflight_dry_run.py': '"""Dry-run competition preflight.\n\nThis tool never submits a decision. It reads the Phase 4 dry-run artifact,\nconstructs an artificial open decision window for unit plumbing, and exercises\nthe same safety logic that will later guard live submission.\n\nFor a real round, live schedule and prior-attempt state must come from the\norganizer endpoints immediately before submission.\n"""\n\nfrom __future__ import annotations\n\nimport json\nimport sys\nfrom datetime import datetime, timedelta, timezone\nfrom pathlib import Path\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nif str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.execution.cost_forecast import (\n    fee_forecast_dollars,\n    gross_weight_change_proxy,\n)\nfrom our_agent.execution.preflight import (\n    PreflightContext,\n    run_preflight,\n)\nfrom our_agent.intents import StrategyIntent\n\nARTIFACT = PROJECT_ROOT / "artifacts" / "candidate_v0_dry_run.json"\n\n\ndef main() -> None:\n    if not ARTIFACT.exists():\n        raise SystemExit(\n            "Missing artifacts/candidate_v0_dry_run.json. "\n            "Run Phase 4 first."\n        )\n\n    raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))\n\n    if raw.get("artifact_type") != (\n        "DRY_RUN_ONLY_NOT_A_COMPETITION_SUBMISSION"\n    ):\n        raise SystemExit(\n            "Refusing to continue: unexpected artifact_type."\n        )\n\n    weights = raw["portfolio"]["weights"]\n\n    if set(weights) != set(OFFICIAL_SYMBOLS):\n        raise SystemExit(\n            "Dry-run artifact does not contain exactly 30 symbols."\n        )\n\n    data_cutoff = datetime.fromisoformat(\n        raw["data"]["common_information_cutoff"]\n    )\n\n    if data_cutoff.tzinfo is None:\n        raise SystemExit("Dry-run data cutoff is not timezone-aware.")\n\n    # Artificial open window solely to exercise safety plumbing.\n    now = datetime.now(timezone.utc)\n    context = PreflightContext(\n        phase="validation",\n        round_id="DRY-RUN-NOT-A-REAL-ROUND",\n        schedule_status="OPEN",\n        opens_at=now - timedelta(minutes=10),\n        deadline=now + timedelta(minutes=10),\n        now=now,\n        data_cutoff=min(data_cutoff, now - timedelta(seconds=1)),\n        prior_attempt_exists=False,\n    )\n\n    report = run_preflight(\n        intent=StrategyIntent.REBALANCE,\n        weights=weights,\n        context=context,\n    )\n\n    current = {symbol: 0.0 for symbol in OFFICIAL_SYMBOLS}\n    turnover_proxy = gross_weight_change_proxy(current, weights)\n    fee_proxy = fee_forecast_dollars(\n        nav=1_000_000.0,\n        gross_turnover_proxy=turnover_proxy,\n    )\n\n    print("=== ICAIF DRY-RUN PREFLIGHT ===")\n    print("Network POST: DISABLED")\n    print("Real competition round: NO")\n    print(f"Artifact: {ARTIFACT}")\n    print(f"Weight count: {len(weights)}")\n    print(f"Gross exposure: {sum(float(v) for v in weights.values()):.4f}")\n    print(f"Preflight OK: {report.ok}")\n\n    for warning in report.warnings:\n        print(f"WARNING: {warning}")\n\n    for error in report.errors:\n        print(f"ERROR: {error}")\n\n    print()\n    print("=== COST FORECAST PROXY ===")\n    print(f"Cash -> target gross weight change: {turnover_proxy:.4f}")\n    print(f"Estimated 10bp fee at $1m NAV: ${fee_proxy:,.2f}")\n    print(\n        "NOTE: forecast proxy is not claimed to reproduce the "\n        "private backend weight solver exactly."\n    )\n\n    if not report.ok:\n        raise SystemExit(2)\n\n    print()\n    print("PREFLIGHT DRY RUN PASSED. NO SUBMISSION WAS MADE.")\n\n\nif __name__ == "__main__":\n    if "--import-check" in sys.argv:\n        print("PREFLIGHT_IMPORT_CHECK_OK")\n    else:\n        main()\n', 'tests/test_preflight.py': 'import unittest\nfrom datetime import datetime, timedelta, timezone\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.execution.preflight import (\n    PreflightContext,\n    run_preflight,\n)\nfrom our_agent.intents import StrategyIntent\n\n\ndef zero_weights():\n    return {symbol: 0.0 for symbol in OFFICIAL_SYMBOLS}\n\n\ndef equal_weights():\n    return {symbol: 1.0 / 30.0 for symbol in OFFICIAL_SYMBOLS}\n\n\ndef context(**overrides):\n    now = datetime(2026, 10, 8, 13, 0, tzinfo=timezone.utc)\n    values = dict(\n        phase="validation",\n        round_id="validation-test-r1",\n        schedule_status="OPEN",\n        opens_at=now - timedelta(minutes=10),\n        deadline=now + timedelta(minutes=10),\n        now=now,\n        data_cutoff=now - timedelta(minutes=1),\n        prior_attempt_exists=False,\n    )\n    values.update(overrides)\n    return PreflightContext(**values)\n\n\nclass PreflightTests(unittest.TestCase):\n    def test_valid_rebalance_passes(self):\n        report = run_preflight(\n            intent=StrategyIntent.REBALANCE,\n            weights=equal_weights(),\n            context=context(),\n        )\n        self.assertTrue(report.ok, report.errors)\n\n    def test_deadline_is_exclusive(self):\n        c = context()\n        report = run_preflight(\n            intent=StrategyIntent.REBALANCE,\n            weights=equal_weights(),\n            context=context(now=c.deadline),\n        )\n        self.assertFalse(report.ok)\n\n    def test_prior_attempt_blocks(self):\n        report = run_preflight(\n            intent=StrategyIntent.REBALANCE,\n            weights=equal_weights(),\n            context=context(prior_attempt_exists=True),\n        )\n        self.assertFalse(report.ok)\n\n    def test_cancelled_round_blocks(self):\n        report = run_preflight(\n            intent=StrategyIntent.REBALANCE,\n            weights=equal_weights(),\n            context=context(schedule_status="CANCELLED"),\n        )\n        self.assertFalse(report.ok)\n\n    def test_hold_has_no_weights(self):\n        report = run_preflight(\n            intent=StrategyIntent.HOLD,\n            weights=None,\n            context=context(),\n        )\n        self.assertTrue(report.ok, report.errors)\n\n    def test_hold_with_weights_fails(self):\n        report = run_preflight(\n            intent=StrategyIntent.HOLD,\n            weights=equal_weights(),\n            context=context(),\n        )\n        self.assertFalse(report.ok)\n\n    def test_all_zero_rebalance_fails(self):\n        report = run_preflight(\n            intent=StrategyIntent.REBALANCE,\n            weights=zero_weights(),\n            context=context(),\n        )\n        self.assertFalse(report.ok)\n\n    def test_explicit_target_cash_passes(self):\n        report = run_preflight(\n            intent=StrategyIntent.TARGET_CASH,\n            weights=zero_weights(),\n            context=context(),\n        )\n        self.assertTrue(report.ok, report.errors)\n\n    def test_future_data_cutoff_fails(self):\n        c = context()\n        report = run_preflight(\n            intent=StrategyIntent.REBALANCE,\n            weights=equal_weights(),\n            context=context(\n                data_cutoff=c.now + timedelta(seconds=1)\n            ),\n        )\n        self.assertFalse(report.ok)\n\n\nif __name__ == "__main__":\n    unittest.main()\n', 'tests/test_cost_forecast.py': 'import unittest\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.execution.cost_forecast import (\n    fee_forecast_dollars,\n    gross_weight_change_proxy,\n)\n\n\nclass CostForecastTests(unittest.TestCase):\n    def zeros(self):\n        return {symbol: 0.0 for symbol in OFFICIAL_SYMBOLS}\n\n    def test_cash_to_80_percent_equity(self):\n        current = self.zeros()\n        target = self.zeros()\n\n        for symbol in OFFICIAL_SYMBOLS[:8]:\n            target[symbol] = 0.10\n\n        proxy = gross_weight_change_proxy(current, target)\n\n        self.assertAlmostEqual(proxy, 0.8)\n        self.assertAlmostEqual(\n            fee_forecast_dollars(\n                nav=1_000_000.0,\n                gross_turnover_proxy=proxy,\n            ),\n            800.0,\n        )\n\n    def test_disjoint_80_percent_rotation_is_1_6(self):\n        current = self.zeros()\n        target = self.zeros()\n\n        for symbol in OFFICIAL_SYMBOLS[:8]:\n            current[symbol] = 0.10\n\n        for symbol in OFFICIAL_SYMBOLS[8:16]:\n            target[symbol] = 0.10\n\n        self.assertAlmostEqual(\n            gross_weight_change_proxy(current, target),\n            1.6,\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n', 'tests/test_preflight_entrypoint.py': 'import subprocess\nimport sys\nimport unittest\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parents[1]\n\n\nclass PreflightEntrypointTests(unittest.TestCase):\n    def test_preflight_import_check(self):\n        result = subprocess.run(\n            [\n                sys.executable,\n                "tools/preflight_dry_run.py",\n                "--import-check",\n            ],\n            cwd=ROOT,\n            text=True,\n            capture_output=True,\n        )\n\n        self.assertEqual(\n            result.returncode,\n            0,\n            msg=f"STDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}",\n        )\n        self.assertIn("PREFLIGHT_IMPORT_CHECK_OK", result.stdout)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'}

for filename, content in files.items():
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\\n", encoding="utf-8")
    print(f"Wrote: {filename}")

print()
print("=== ENTRYPOINT CHECK ===")
run(
    sys.executable,
    "tools/preflight_dry_run.py",
    "--import-check",
)

print()
print("=== COMPLETE LOCAL TEST SUITE ===")
run(
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
)

print()
print("=== STATIC CHECKS ===")
run(
    sys.executable,
    "-m",
    "compileall",
    "-q",
    "our_agent",
    "tests",
    "tools",
)
run("git", "diff", "--check")

print()
print("=== READ-ONLY ORGANIZER SCHEDULE CHECK ===")
print(
    "The following command is GET/read-only and does not create a submission."
)
run(
    sys.executable,
    "tools/auto_submit.py",
    "schedule",
)

print()
print("=== NON-SUBMITTABLE PREFLIGHT DRY RUN ===")
run(
    sys.executable,
    "tools/preflight_dry_run.py",
)

changelog = ROOT / "CHANGELOG.md"
title = "## 2026-10-07 — Preflight safety and cost forecast"

text = (
    changelog.read_text(encoding="utf-8")
    if changelog.exists()
    else "# ICAIF Agent Changelog\\n"
)

if title not in text:
    with changelog.open("a", encoding="utf-8") as f:
        f.write(
            "\\n"
            + title
            + "\\n\\n"
            "- Added deterministic live-decision preflight checks.\\n"
            "- Enforced exclusive deadline semantics and prior-attempt blocking.\\n"
            "- Preserved explicit HOLD versus TARGET_CASH semantics.\\n"
            "- Added transparent gross-weight-change turnover/fee forecast proxy.\\n"
            "- The proxy is explicitly not claimed to reproduce the private backend execution solver.\\n"
            "- Added direct-entrypoint regression testing.\\n"
            "- Added a read-only organizer schedule check to the build phase.\\n"
            "- No competition POST/submission functionality enabled.\\n"
        )

print()
print("=== GIT CHECKPOINT ===")
run("git", "check-ignore", "-q", ".env")
run("git", "diff", "--check")

paths = [
    "CHANGELOG.md",
    "our_agent/execution/preflight.py",
    "our_agent/execution/cost_forecast.py",
    "tools/preflight_dry_run.py",
    "tests/test_preflight.py",
    "tests/test_cost_forecast.py",
    "tests/test_preflight_entrypoint.py",
]

run("git", "add", *paths)

staged = output("git", "diff", "--cached", "--name-only")

if staged:
    run(
        "git",
        "commit",
        "-m",
        "Add competition preflight safety layer",
    )
else:
    print("No Phase 5 source changes needed committing.")

print()
print("=== FINAL STATUS ===")
print(f"Branch: {output('git', 'branch', '--show-current')}")
print(f"HEAD:   {output('git', 'rev-parse', 'HEAD')}")
run("git", "status", "--short")

print()
print("PHASE 5 COMPLETE.")
print("Preflight safety and fee/turnover forecasting are test-covered.")
print("No competition submission was made.")
