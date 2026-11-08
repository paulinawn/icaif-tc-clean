from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()


def fail(msg: str) -> None:
    raise SystemExit(f"\nERROR: {msg}\n")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print(">", " ".join(args))
    cp = subprocess.run(args, cwd=ROOT, text=True)
    if check and cp.returncode != 0:
        raise SystemExit(cp.returncode)
    return cp


required = [ROOT / "universe.json", ROOT / ".gitignore", ROOT / "our_agent"]
missing = [p.name for p in required if not p.exists()]
if missing:
    fail(
        "Run this script from the ICAIF starter-kit folder. "
        f"Missing: {missing}"
    )

branch = subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=ROOT, text=True
).strip()
if branch != "our-agent":
    fail(f"Expected Git branch 'our-agent', found '{branch}'.")

print("=== ICAIF PHASE 3: RESEARCH BOOTSTRAP ===")
print(f"Working directory: {ROOT}")
print(f"Branch: {branch}")

print("\n=== VERIFYING PHASE 2 ===")
run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
run("git", "diff", "--check")

status = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=ROOT, text=True
).strip()

if status:
    print("\n=== CREATING PHASE 2 GIT CHECKPOINT ===")
    run(
        "git", "add",
        ".gitignore",
        "CHANGELOG.md",
        "experiments",
        "our_agent",
        "tests",
    )
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
    ).strip()
    if staged:
        run("git", "commit", "-m", "Add ICAIF core safety layer")
    else:
        print("No staged Phase 2 changes to commit.")
else:
    print("Working tree already clean; no Phase 2 commit needed.")

print("\n=== INSTALLING MINIMAL RESEARCH DEPENDENCIES ===")
run(sys.executable, "-m", "pip", "install", "numpy", "pandas")

numpy_version = importlib.metadata.version("numpy")
pandas_version = importlib.metadata.version("pandas")
(ROOT / "requirements-research.txt").write_text(
    f"numpy=={numpy_version}\npandas=={pandas_version}\n",
    encoding="utf-8",
)
print(f"Pinned numpy=={numpy_version}")
print(f"Pinned pandas=={pandas_version}")

official_dir = ROOT / "data" / "official"
official_dir.mkdir(parents=True, exist_ok=True)

gitignore_path = ROOT / ".gitignore"
gitignore_text = gitignore_path.read_text(encoding="utf-8")
for entry in ["data/official/", "data/derived/"]:
    if entry not in gitignore_text:
        with gitignore_path.open("a", encoding="utf-8") as f:
            if not gitignore_text.endswith("\n"):
                f.write("\n")
            f.write(entry + "\n")
        gitignore_text += entry + "\n"

(official_dir / "README.txt").write_text(
    "Place the organizer-provided ICAIF hourly OHLCV development dataset "
    "in this folder.\n"
    "Do not commit raw competition data to Git.\n"
    "After placing the file(s), run:\n"
    "    python tools/inspect_official_data.py\n",
    encoding="utf-8",
)

backtest_dir = ROOT / "our_agent" / "backtest"
backtest_dir.mkdir(parents=True, exist_ok=True)
(backtest_dir / "__init__.py").write_text("", encoding="utf-8")
(backtest_dir / "metrics.py").write_text(
    '''"""Competition-aligned evaluation helpers."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Sequence

SHARPE_ANNUALIZATION = 42.0


def cumulative_return(initial_nav: float, final_nav: float) -> float:
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    return final_nav / initial_nav - 1.0


def period_returns(begin_navs: Sequence[float], end_navs: Sequence[float]) -> list[float]:
    if len(begin_navs) != len(end_navs):
        raise ValueError("begin_navs and end_navs must have equal length")
    out = []
    for begin, end in zip(begin_navs, end_navs):
        if begin <= 0:
            raise ValueError("Each begin NAV must be positive")
        out.append(end / begin - 1.0)
    return out


def competition_sharpe(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sigma = stdev(returns)
    if sigma == 0.0 or not math.isfinite(sigma):
        return 0.0
    value = SHARPE_ANNUALIZATION * mean(returns) / sigma
    return float(value) if math.isfinite(value) else 0.0


def maximum_drawdown(valuation_path: Sequence[float]) -> float:
    if not valuation_path:
        raise ValueError("valuation_path must not be empty")
    if any(v <= 0 or not math.isfinite(v) for v in valuation_path):
        raise ValueError("All valuation points must be positive and finite")
    peak = valuation_path[0]
    worst = 0.0
    for value in valuation_path:
        if value > peak:
            peak = value
        worst = max(worst, (peak - value) / peak)
    return worst


def competition_turnover(gross_trade_notionals: Sequence[float], pre_execution_navs: Sequence[float]) -> float:
    if len(gross_trade_notionals) != len(pre_execution_navs):
        raise ValueError("gross_trade_notionals and pre_execution_navs must have equal length")
    if not gross_trade_notionals:
        raise ValueError("At least one decision period is required")
    ratios = []
    for notional, nav in zip(gross_trade_notionals, pre_execution_navs):
        if notional < 0 or not math.isfinite(notional):
            raise ValueError("Trade notional must be finite and non-negative")
        if nav <= 0 or not math.isfinite(nav):
            raise ValueError("Pre-execution NAV must be positive and finite")
        ratios.append(notional / nav)
    return sum(ratios) / len(ratios)
''',
    encoding="utf-8",
)

strategy_dir = ROOT / "our_agent" / "strategy"
strategy_dir.mkdir(parents=True, exist_ok=True)
(strategy_dir / "baselines.py").write_text(
    '''"""Simple, transparent portfolio baselines."""

from __future__ import annotations

from typing import Mapping

from our_agent.config import MAX_WEIGHT, OFFICIAL_SYMBOLS
from our_agent.intents import StrategyIntent
from our_agent.validation import validate_target_weights


def _empty_weights() -> dict[str, float]:
    return {symbol: 0.0 for symbol in OFFICIAL_SYMBOLS}


def equal_weight(gross_exposure: float = 1.0) -> dict[str, float]:
    if not 0.0 <= gross_exposure <= 1.0:
        raise ValueError("gross_exposure must be between 0 and 1")
    each = gross_exposure / len(OFFICIAL_SYMBOLS)
    weights = {symbol: each for symbol in OFFICIAL_SYMBOLS}
    return validate_target_weights(weights, StrategyIntent.REBALANCE)


def top_k_equal(scores: Mapping[str, float], k: int = 8, gross_exposure: float = 0.8) -> dict[str, float]:
    if set(scores) != set(OFFICIAL_SYMBOLS):
        raise ValueError("scores must contain exactly the official symbols")
    if not 1 <= k <= len(OFFICIAL_SYMBOLS):
        raise ValueError("k is outside the official universe")
    if not 0.0 < gross_exposure <= 1.0:
        raise ValueError("gross_exposure must be in (0, 1]")
    per_name = gross_exposure / k
    if per_name > float(MAX_WEIGHT):
        raise ValueError("Requested exposure/k would exceed the 30% per-stock cap")
    ranked = sorted(OFFICIAL_SYMBOLS, key=lambda s: (scores[s], s), reverse=True)
    selected = set(ranked[:k])
    weights = _empty_weights()
    for symbol in selected:
        weights[symbol] = per_name
    return validate_target_weights(weights, StrategyIntent.REBALANCE)


def inverse_volatility(volatilities: Mapping[str, float], gross_exposure: float = 1.0) -> dict[str, float]:
    if set(volatilities) != set(OFFICIAL_SYMBOLS):
        raise ValueError("volatilities must contain exactly the official symbols")
    if not 0.0 < gross_exposure <= 1.0:
        raise ValueError("gross_exposure must be in (0, 1]")
    inv = {}
    for symbol, vol in volatilities.items():
        if vol <= 0:
            raise ValueError(f"{symbol}: volatility must be positive")
        inv[symbol] = 1.0 / float(vol)
    total_inv = sum(inv.values())
    weights = {symbol: gross_exposure * inv[symbol] / total_inv for symbol in OFFICIAL_SYMBOLS}
    if max(weights.values()) > float(MAX_WEIGHT):
        raise ValueError("Inverse-vol allocation breaches the 30% cap; capped optimizer required")
    return validate_target_weights(weights, StrategyIntent.REBALANCE)


def gross_weight_change(current: Mapping[str, float], target: Mapping[str, float]) -> float:
    if set(current) != set(OFFICIAL_SYMBOLS) or set(target) != set(OFFICIAL_SYMBOLS):
        raise ValueError("current and target must contain exactly the official symbols")
    return sum(abs(float(target[s]) - float(current[s])) for s in OFFICIAL_SYMBOLS)


def no_trade_decision(current: Mapping[str, float], target: Mapping[str, float], minimum_gross_change: float = 0.10) -> StrategyIntent:
    change = gross_weight_change(current, target)
    return StrategyIntent.HOLD if change < minimum_gross_change else StrategyIntent.REBALANCE
''',
    encoding="utf-8",
)

tools_dir = ROOT / "tools"
(tools_dir / "inspect_official_data.py").write_text(
    '''"""Inspect organizer-provided development data without modifying it."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "official"


def inspect_csv(path: Path) -> None:
    print(f"CSV: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(f"  row {i}: {row[:12]}")
            if i >= 3:
                break


def inspect_json(path: Path) -> None:
    print(f"JSON: {path.name}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    print(f"  top-level type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"  first keys: {list(obj)[:20]}")
    elif isinstance(obj, list):
        print(f"  rows/items: {len(obj)}")
        if obj:
            print(f"  first item: {obj[0]}")


def inspect_zip(path: Path) -> None:
    print(f"ZIP: {path.name}")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        print(f"  entries: {len(names)}")
        for name in names[:40]:
            print(f"  - {name}")


def main() -> None:
    print(f"Official-data folder: {DATA_DIR}")
    files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.name != "README.txt"]
    if not files:
        print("\nNO ORGANIZER DATASET FOUND.\nPlace the organizer-provided hourly OHLCV development data in data/official/ and run again.")
        return
    print(f"Found {len(files)} file(s).\n")
    for path in sorted(files):
        suffix = path.suffix.lower()
        print("=" * 72)
        print(f"{path.name}  ({path.stat().st_size:,} bytes)")
        try:
            if suffix == ".csv":
                inspect_csv(path)
            elif suffix == ".json":
                inspect_json(path)
            elif suffix == ".zip":
                inspect_zip(path)
            elif suffix in {".parquet", ".pq"}:
                try:
                    import pandas as pd
                    frame = pd.read_parquet(path)
                    print(f"PARQUET: {path.name}")
                    print(f"  shape: {frame.shape}")
                    print(f"  columns: {list(frame.columns)}")
                    print(frame.head(3).to_string())
                except Exception as exc:
                    print(f"  Could not read parquet yet: {exc}")
            else:
                print("  Unrecognized extension; leaving file untouched.")
        except Exception as exc:
            print(f"  Inspection error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
''',
    encoding="utf-8",
)

(ROOT / "tests" / "test_metrics.py").write_text(
    '''import unittest

from our_agent.backtest.metrics import (
    competition_sharpe,
    competition_turnover,
    cumulative_return,
    maximum_drawdown,
    period_returns,
)


class CompetitionMetricTests(unittest.TestCase):
    def test_cumulative_return(self):
        self.assertAlmostEqual(cumulative_return(1_000_000, 1_050_000), 0.05)

    def test_period_returns(self):
        values = period_returns([100.0, 100.0], [101.0, 99.0])
        self.assertAlmostEqual(values[0], 0.01)
        self.assertAlmostEqual(values[1], -0.01)

    def test_sharpe_zero_for_constant_returns(self):
        self.assertEqual(competition_sharpe([0.01, 0.01]), 0.0)

    def test_sharpe_uses_factor_42(self):
        r = [0.01, 0.02, 0.00]
        expected = 42.0 * (sum(r) / len(r)) / 0.01
        self.assertAlmostEqual(competition_sharpe(r), expected)

    def test_maximum_drawdown(self):
        path = [100.0, 110.0, 99.0, 105.0]
        self.assertAlmostEqual(maximum_drawdown(path), 0.10)

    def test_turnover_is_unhalved_buy_plus_sell(self):
        result = competition_turnover([100.0, 0.0, 50.0], [100.0, 100.0, 100.0])
        self.assertAlmostEqual(result, 0.5)


if __name__ == "__main__":
    unittest.main()
''',
    encoding="utf-8",
)

(ROOT / "tests" / "test_baselines.py").write_text(
    '''import unittest

from our_agent.config import OFFICIAL_SYMBOLS
from our_agent.intents import StrategyIntent
from our_agent.strategy.baselines import (
    equal_weight,
    gross_weight_change,
    inverse_volatility,
    no_trade_decision,
    top_k_equal,
)


class BaselineTests(unittest.TestCase):
    def test_equal_weight(self):
        weights = equal_weight(0.90)
        self.assertEqual(len(weights), 30)
        self.assertAlmostEqual(sum(weights.values()), 0.90)
        self.assertLessEqual(max(weights.values()), 0.30)

    def test_top_k_equal(self):
        scores = {symbol: float(i) for i, symbol in enumerate(OFFICIAL_SYMBOLS)}
        weights = top_k_equal(scores, k=8, gross_exposure=0.80)
        self.assertEqual(sum(v > 0 for v in weights.values()), 8)
        self.assertAlmostEqual(sum(weights.values()), 0.80)

    def test_inverse_volatility(self):
        vols = {symbol: 0.20 + i * 0.001 for i, symbol in enumerate(OFFICIAL_SYMBOLS)}
        weights = inverse_volatility(vols, gross_exposure=0.75)
        self.assertAlmostEqual(sum(weights.values()), 0.75)
        self.assertLessEqual(max(weights.values()), 0.30)

    def test_no_trade_gate_holds_small_change(self):
        current = equal_weight(0.90)
        target = dict(current)
        target["AAPL"] += 0.001
        target["MSFT"] -= 0.001
        self.assertAlmostEqual(gross_weight_change(current, target), 0.002)
        self.assertEqual(
            no_trade_decision(current, target, minimum_gross_change=0.05),
            StrategyIntent.HOLD,
        )


if __name__ == "__main__":
    unittest.main()
''',
    encoding="utf-8",
)

changelog_path = ROOT / "CHANGELOG.md"
with changelog_path.open("a", encoding="utf-8") as f:
    f.write(
        "\n## 2026-10-07 — Research bootstrap\n\n"
        f"- Added pinned research dependencies: numpy {numpy_version}, pandas {pandas_version}.\n"
        "- Added competition-aligned metric functions.\n"
        "- Added equal-weight, Top-K equal-weight, inverse-volatility, and no-trade baseline primitives.\n"
        "- Added organizer-data inspection utility.\n"
        "- Raw organizer development data remains local and Git-ignored.\n"
        "- No live submission logic enabled.\n"
    )

print("\n=== RUNNING COMPLETE TEST SUITE ===")
run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
run(sys.executable, "-m", "compileall", "-q", "our_agent", "tests", "tools")
run("git", "diff", "--check")

print("\n=== DATASET CHECK ===")
run(sys.executable, "tools/inspect_official_data.py")

print("\n=== FINAL STATUS ===")
print(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))

print("PHASE 3 BOOTSTRAP COMPLETE.")
print(
    "\nNEXT REQUIRED INPUT: place the organizer-provided Jan-2021 to Jan-2026 hourly OHLCV "
    "development dataset in starter-kit/data/official/. Then rerun: "
    "python tools/inspect_official_data.py"
)
