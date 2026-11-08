from __future__ import annotations

import csv
import json
import math
import subprocess
import shutil
import sys
from pathlib import Path

EXPECTED_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "INTC", "CRM",
    "JPM", "BAC", "GS", "V", "PYPL",
    "LLY", "JNJ", "UNH", "PFE", "TMO",
    "AMZN", "TSLA", "WMT", "NKE", "KO",
    "CAT", "GE", "BA", "XOM", "CVX",
    "GOOGL", "META", "DIS", "T", "NEE",
)
STRATEGY_VERSION = "validation-equal30-80-hold-v1"

BASELINE_MODULE = 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nimport math\nfrom numbers import Real\nfrom typing import Any, Mapping\n\nSTRATEGY_VERSION = "validation-equal30-80-hold-v1"\nGROSS_EXPOSURE = 0.80\n\nOFFICIAL_SYMBOLS = (\n    "AAPL", "MSFT", "NVDA", "INTC", "CRM",\n    "JPM", "BAC", "GS", "V", "PYPL",\n    "LLY", "JNJ", "UNH", "PFE", "TMO",\n    "AMZN", "TSLA", "WMT", "NKE", "KO",\n    "CAT", "GE", "BA", "XOM", "CVX",\n    "GOOGL", "META", "DIS", "T", "NEE",\n)\n\nACTION_REBALANCE = "REBALANCE"\nACTION_HOLD = "HOLD"\n\n_EPS = 1e-12\n_POSITION_VALUE_FIELDS = (\n    "weight", "market_value", "marketValue", "notional", "value",\n    "quantity", "qty", "shares",\n)\n\n\nclass PortfolioStateError(ValueError):\n    pass\n\n\n@dataclass(frozen=True)\nclass BaselineDecision:\n    action: str\n    weights: dict[str, float] | None\n    reason: str\n    strategy_version: str = STRATEGY_VERSION\n\n\ndef target_weights() -> dict[str, float]:\n    each = GROSS_EXPOSURE / len(OFFICIAL_SYMBOLS)\n    weights = {symbol: each for symbol in OFFICIAL_SYMBOLS}\n\n    if set(weights) != set(OFFICIAL_SYMBOLS):\n        raise AssertionError("Target symbol set mismatch.")\n\n    if any(\n        not isinstance(weight, float)\n        or not math.isfinite(weight)\n        or weight < 0.0\n        or weight > 0.30\n        for weight in weights.values()\n    ):\n        raise AssertionError("Invalid target weight.")\n\n    if not math.isclose(\n        math.fsum(weights.values()),\n        GROSS_EXPOSURE,\n        rel_tol=0.0,\n        abs_tol=1e-12,\n    ):\n        raise AssertionError("Target gross exposure mismatch.")\n\n    return weights\n\n\ndef _as_nonzero_real(value: Any) -> bool | None:\n    if isinstance(value, bool):\n        return None\n    if isinstance(value, Real):\n        number = float(value)\n        if not math.isfinite(number):\n            raise PortfolioStateError("Non-finite portfolio value.")\n        return abs(number) > _EPS\n    return None\n\n\ndef _record_has_exposure(record: Any) -> bool | None:\n    scalar = _as_nonzero_real(record)\n    if scalar is not None:\n        return scalar\n\n    if not isinstance(record, Mapping):\n        return None\n\n    recognized = False\n    for field in _POSITION_VALUE_FIELDS:\n        if field not in record:\n            continue\n        recognized = True\n        flag = _as_nonzero_real(record[field])\n        if flag:\n            return True\n\n    if recognized:\n        return False\n    return None\n\n\ndef _collection_has_exposure(value: Any) -> bool | None:\n    if isinstance(value, Mapping):\n        if not value:\n            return False\n        saw_recognized = False\n        for record in value.values():\n            result = _record_has_exposure(record)\n            if result is None:\n                continue\n            saw_recognized = True\n            if result:\n                return True\n        return False if saw_recognized else None\n\n    if isinstance(value, (list, tuple)):\n        if not value:\n            return False\n        saw_recognized = False\n        for record in value:\n            result = _record_has_exposure(record)\n            if result is None:\n                continue\n            saw_recognized = True\n            if result:\n                return True\n        return False if saw_recognized else None\n\n    return None\n\n\ndef portfolio_has_equity_exposure(portfolio: Mapping[str, Any]) -> bool:\n    if not isinstance(portfolio, Mapping):\n        raise PortfolioStateError("Portfolio must be a mapping.")\n\n    for key in (\n        "last_decision",\n        "lastDecision",\n        "last_selected_decision",\n        "lastSelectedDecision",\n    ):\n        if key in portfolio and portfolio[key]:\n            return True\n\n    for key in ("positions", "holdings"):\n        if key not in portfolio:\n            continue\n        result = _collection_has_exposure(portfolio[key])\n        if result is not None:\n            return result\n\n    for key in ("weights", "current_weights", "currentWeights"):\n        value = portfolio.get(key)\n        if not isinstance(value, Mapping):\n            continue\n\n        recognized = False\n        for symbol in OFFICIAL_SYMBOLS:\n            if symbol not in value:\n                continue\n            recognized = True\n            flag = _as_nonzero_real(value[symbol])\n            if flag:\n                return True\n        if recognized:\n            return False\n\n    cash = portfolio.get("cash")\n    nav = None\n    for key in (\n        "nav", "portfolio_value", "portfolioValue",\n        "equity", "total_value", "totalValue",\n    ):\n        if key in portfolio:\n            nav = portfolio[key]\n            break\n\n    if isinstance(cash, Real) and not isinstance(cash, bool):\n        if isinstance(nav, Real) and not isinstance(nav, bool):\n            cash_f = float(cash)\n            nav_f = float(nav)\n            if not (\n                math.isfinite(cash_f)\n                and math.isfinite(nav_f)\n                and nav_f > 0.0\n            ):\n                raise PortfolioStateError("Invalid cash/NAV values.")\n\n            tolerance = max(1.0, 1e-6 * nav_f)\n            if abs(cash_f - nav_f) <= tolerance:\n                return False\n            if cash_f < nav_f - tolerance:\n                return True\n\n    raise PortfolioStateError(\n        "Portfolio schema is not recognized safely. "\n        "Refusing to infer an all-cash state."\n    )\n\n\ndef decide(portfolio: Mapping[str, Any]) -> BaselineDecision:\n    if portfolio_has_equity_exposure(portfolio):\n        return BaselineDecision(\n            action=ACTION_HOLD,\n            weights=None,\n            reason=(\n                "Portfolio already deployed or has prior decision history; "\n                "HOLD to avoid unnecessary fees and turnover."\n            ),\n        )\n\n    return BaselineDecision(\n        action=ACTION_REBALANCE,\n        weights=target_weights(),\n        reason="Fresh all-cash phase; deploy frozen 80% equal-weight baseline.",\n    )\n\n\ndef metadata() -> dict[str, Any]:\n    return {\n        "strategy_version": STRATEGY_VERSION,\n        "gross_exposure": GROSS_EXPOSURE,\n        "symbol_count": len(OFFICIAL_SYMBOLS),\n        "behavior": "initial equal-weight deployment, then HOLD",\n        "live_llm_used": False,\n    }\n'
TEST_MODULE = 'from __future__ import annotations\n\nimport json\nimport math\nimport unittest\n\nfrom our_agent.strategies.validation_baseline import (\n    ACTION_HOLD,\n    ACTION_REBALANCE,\n    GROSS_EXPOSURE,\n    OFFICIAL_SYMBOLS,\n    PortfolioStateError,\n    decide,\n    metadata,\n    target_weights,\n)\n\n\nclass ValidationBaselineTests(unittest.TestCase):\n    def test_target_weights_are_exact_and_valid(self):\n        weights = target_weights()\n        self.assertEqual(set(weights), set(OFFICIAL_SYMBOLS))\n        self.assertEqual(len(weights), 30)\n        self.assertTrue(\n            math.isclose(\n                math.fsum(weights.values()),\n                GROSS_EXPOSURE,\n                rel_tol=0.0,\n                abs_tol=1e-12,\n            )\n        )\n        self.assertTrue(\n            all(0.0 <= weight <= 0.30 for weight in weights.values())\n        )\n        self.assertTrue(\n            all(\n                math.isclose(\n                    weight,\n                    GROSS_EXPOSURE / 30.0,\n                    rel_tol=0.0,\n                    abs_tol=1e-15,\n                )\n                for weight in weights.values()\n            )\n        )\n\n    def test_fresh_empty_positions_deploy(self):\n        decision = decide(\n            {\n                "cash": 1_000_000.0,\n                "nav": 1_000_000.0,\n                "positions": [],\n                "last_decision": None,\n            }\n        )\n        self.assertEqual(decision.action, ACTION_REBALANCE)\n        self.assertIsNotNone(decision.weights)\n        self.assertTrue(\n            math.isclose(\n                math.fsum(decision.weights.values()),\n                0.80,\n                abs_tol=1e-12,\n            )\n        )\n\n    def test_existing_position_holds(self):\n        decision = decide(\n            {\n                "cash": 200_000.0,\n                "nav": 1_000_000.0,\n                "positions": [{"symbol": "AAPL", "shares": 100.0}],\n            }\n        )\n        self.assertEqual(decision.action, ACTION_HOLD)\n        self.assertIsNone(decision.weights)\n\n    def test_prior_decision_holds_even_if_cash(self):\n        decision = decide(\n            {\n                "cash": 1_000_000.0,\n                "nav": 1_000_000.0,\n                "positions": [],\n                "last_decision": {"round_id": "example-r1"},\n            }\n        )\n        self.assertEqual(decision.action, ACTION_HOLD)\n        self.assertIsNone(decision.weights)\n\n    def test_weight_mapping_detects_exposure(self):\n        weights = {symbol: 0.0 for symbol in OFFICIAL_SYMBOLS}\n        weights["AAPL"] = 0.10\n        decision = decide({"weights": weights})\n        self.assertEqual(decision.action, ACTION_HOLD)\n\n    def test_unknown_schema_fails_closed(self):\n        with self.assertRaises(PortfolioStateError):\n            decide({"status": "mystery"})\n\n    def test_hold_is_not_all_zero_target(self):\n        decision = decide(\n            {"positions": [{"symbol": "MSFT", "market_value": 10_000.0}]}\n        )\n        self.assertEqual(decision.action, ACTION_HOLD)\n        self.assertIsNone(decision.weights)\n\n    def test_output_is_deterministic_and_serializable(self):\n        first = target_weights()\n        second = target_weights()\n        self.assertEqual(first, second)\n        self.assertIsInstance(json.dumps(first, sort_keys=True), str)\n\n    def test_metadata_declares_no_live_llm(self):\n        info = metadata()\n        self.assertEqual(\n            info["strategy_version"],\n            "validation-equal30-80-hold-v1",\n        )\n        self.assertFalse(info["live_llm_used"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'


def fail(message: str) -> None:
    raise SystemExit("\\nERROR: " + message + "\\n")


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (
            (candidate / "universe.json").exists()
            and (candidate / "our_agent").is_dir()
            and (candidate / ".git").exists()
        ):
            return candidate.resolve()
    fail(
        "Could not locate the ICAIF starter-kit Git repository. "
        "Run this while PowerShell is in the starter-kit folder."
    )


ROOT = find_project_root()


def run(*args: str) -> None:
    print(">", " ".join(args))
    cp = subprocess.run(args, cwd=ROOT, text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def cleanup_python_caches() -> None:
    """Remove generated bytecode caches so Git status reflects source changes only."""
    for relative in ("our_agent", "tests"):
        base = ROOT / relative
        if not base.exists():
            continue
        for cache_dir in base.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)


def collect_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from collect_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from collect_strings(key)
            yield from collect_strings(item)


def verify_universe() -> None:
    payload = json.loads(
        (ROOT / "universe.json").read_text(encoding="utf-8")
    )
    strings = list(collect_strings(payload))
    found = {s for s in strings if s in EXPECTED_SYMBOLS}
    if found != set(EXPECTED_SYMBOLS):
        fail(
            "Current universe.json does not contain the expected official "
            "30-symbol universe."
        )


def verify_phase9_evidence() -> dict[str, str]:
    path = ROOT / "artifacts" / "phase9_official_core_summary.csv"
    if not path.exists():
        fail("Missing Phase 9 official-data summary: " + str(path))

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    match = next(
        (row for row in rows if row.get("strategy") == "equal30_80_hold"),
        None,
    )
    if match is None:
        fail("Phase 9 summary does not contain equal30_80_hold.")

    try:
        windows = int(float(match["windows"]))
        mean_gross = float(match["mean_gross"])
        mean_turnover = float(match["mean_turnover_proxy"])
    except (KeyError, TypeError, ValueError) as exc:
        fail("Phase 9 summary fields invalid: " + str(exc))

    if windows < 200:
        fail("Phase 9 evidence has fewer than 200 windows.")
    if not math.isclose(mean_gross, 0.80, abs_tol=1e-12):
        fail("Phase 9 baseline gross exposure is not 0.80.")
    if not (0.0 <= mean_turnover < 0.02):
        fail("Phase 9 baseline turnover is unexpectedly high.")

    return match


def append_once(path: Path, marker: str, lines: list[str]) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current:
        return
    if current and not current.endswith("\\n"):
        current += "\\n"
    block = "\\n".join(lines) + "\\n"
    path.write_text(current + block, encoding="utf-8")


def main() -> None:
    print("=== PHASE 10: FREEZE VALIDATION BASELINE ===")
    print("Competition submission: DISABLED")
    print()

    if output("git", "branch", "--show-current") != "our-agent":
        fail("Expected Git branch 'our-agent'.")

    dirty = output("git", "status", "--short")
    if dirty:
        fail("Working tree is not clean:\\n" + dirty)

    verify_universe()
    evidence = verify_phase9_evidence()

    print(
        "Phase 9 evidence verified: "
        + evidence["windows"]
        + " rolling windows; mean return="
        + evidence["mean_return"]
        + "; mean Sharpe proxy="
        + evidence["mean_sharpe_proxy"]
        + "; mean MDD proxy="
        + evidence["mean_mdd_proxy"]
        + "; mean turnover proxy="
        + evidence["mean_turnover_proxy"]
    )

    strategy_dir = ROOT / "our_agent" / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    init_path = strategy_dir / "__init__.py"
    baseline_path = strategy_dir / "validation_baseline.py"
    test_path = ROOT / "tests" / "test_validation_baseline.py"

    compile(BASELINE_MODULE, str(baseline_path), "exec")
    compile(TEST_MODULE, str(test_path), "exec")

    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    baseline_path.write_text(BASELINE_MODULE, encoding="utf-8")
    test_path.write_text(TEST_MODULE, encoding="utf-8")

    changelog = ROOT / "CHANGELOG.md"
    marker = "## 2026-10-07 — Validation baseline frozen"
    append_once(
        changelog,
        marker,
        [
            "",
            marker,
            "",
            "- Froze `validation-equal30-80-hold-v1` as the Validation baseline.",
            "- Initial action: 80% gross equal-weight across all 30 official stocks; 20% cash.",
            "- Subsequent action: explicit HOLD with `weights=None`, never all-zero weights.",
            "- Evidence: " + evidence["windows"] + " rolling 15-day organizer-data windows; mean return "
            + evidence["mean_return"] + ", mean Sharpe proxy " + evidence["mean_sharpe_proxy"]
            + ", mean MDD proxy " + evidence["mean_mdd_proxy"] + ", mean turnover proxy "
            + evidence["mean_turnover_proxy"] + ".",
            "- No live LLM and no competition submission path were added.",
        ],
    )

    experiment = ROOT / "EXPERIMENT_LOG.md"
    exp_marker = "## EXP-P10 — Freeze Validation baseline"
    append_once(
        experiment,
        exp_marker,
        [
            "",
            exp_marker,
            "",
            "- Hypothesis: low-turnover diversified anchor is the strongest current Validation foundation.",
            "- Data: organizer historical OHLCV, 2021-01-04 through 2025-12-31.",
            "- Windows: " + evidence["windows"] + " rolling 15-trading-day windows.",
            "- Strategy: `validation-equal30-80-hold-v1`.",
            "- Mean return: " + evidence["mean_return"] + ".",
            "- Mean Sharpe proxy: " + evidence["mean_sharpe_proxy"] + ".",
            "- Mean MDD proxy: " + evidence["mean_mdd_proxy"] + ".",
            "- Worst MDD proxy: " + evidence["worst_mdd_proxy"] + ".",
            "- Mean turnover proxy: " + evidence["mean_turnover_proxy"] + ".",
            "- Decision: KEEP for Validation; alpha overlays remain research-only.",
        ],
    )

    print()
    print("=== EXACT GENERATED-FILE COMPILE CHECK ===")
    for path in (baseline_path, test_path):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        print("PASS:", path.relative_to(ROOT))

    print()
    print("=== BASELINE UNIT TEST ===")
    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_validation_baseline.py",
        "-v",
    )

    print()
    print("=== FULL REGRESSION TESTS ===")
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
    run(sys.executable, "-m", "compileall", "-q", "our_agent", "tests")
    cleanup_python_caches()
    run("git", "diff", "--check")

    for path in (
        baseline_path,
        test_path,
        changelog,
        experiment,
    ):
        content = path.read_text(encoding="utf-8")
        for token in (
            "CODABENCH_TOKEN",
            "TEAM_TOKEN",
            "auto_submit.py decision",
            "auto_submit.py watch",
        ):
            if token in content:
                fail(
                    "Forbidden live-submission/secret token found in "
                    + str(path.relative_to(ROOT))
                )

    stage_paths = [
        "our_agent/strategies/validation_baseline.py",
        "tests/test_validation_baseline.py",
        "CHANGELOG.md",
        "EXPERIMENT_LOG.md",
    ]
    init_status = output(
        "git",
        "status",
        "--short",
        "--",
        "our_agent/strategies/__init__.py",
    )
    if init_status:
        stage_paths.append("our_agent/strategies/__init__.py")

    print()
    print("=== GIT CHECKPOINT ===")
    run("git", "add", *stage_paths)
    run("git", "diff", "--cached", "--check")

    staged = output("git", "diff", "--cached", "--name-only")
    if not staged:
        fail("No staged changes found.")
    print(staged)

    run(
        "git",
        "commit",
        "-m",
        "Freeze Validation equal-weight baseline",
    )

    final_status = output("git", "status", "--short")
    if final_status:
        fail("Working tree is not clean after commit:\\n" + final_status)

    print()
    print("=== FREEZE COMPLETE ===")
    print("Strategy: validation-equal30-80-hold-v1")
    print("Initial action: REBALANCE to 80% equal-weight / 20% cash")
    print("After deployment: HOLD (weights=None internally)")
    print("HEAD:", output("git", "rev-parse", "HEAD"))
    print("Working tree: CLEAN")
    print("No competition submission was made.")


if __name__ == "__main__":
    main()
