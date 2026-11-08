from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()

required = [ROOT / "universe.json", ROOT / "requirements.txt", ROOT / ".gitignore"]
missing = [p.name for p in required if not p.exists()]
if missing:
    raise SystemExit(
        "ERROR: Run this script from the ICAIF starter-kit folder.\n"
        f"Missing expected files: {missing}"
    )

print("=== ICAIF PHASE 2 SETUP ===")
print(f"Working directory: {ROOT}")

files: dict[str, str] = {}

files["our_agent/config.py"] = '''"""Core competition configuration.

This file contains only stable competition constraints.
Live schedule/round information must NEVER be hard-coded here.
"""

from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = PROJECT_ROOT / "universe.json"

FEE_RATE = Decimal("0.001")
MAX_WEIGHT = Decimal("0.30")
MAX_TOTAL_EXPOSURE = Decimal("1.0")
INITIAL_NAV = Decimal("1000000")

OFFICIAL_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "INTC", "CRM",
    "JPM", "BAC", "GS", "V", "PYPL",
    "LLY", "JNJ", "UNH", "PFE", "TMO",
    "AMZN", "TSLA", "WMT", "NKE", "KO",
    "CAT", "GE", "BA", "XOM", "CVX",
    "GOOGL", "META", "DIS", "T", "NEE",
)

SECTORS = {
    "Technology": ("AAPL", "MSFT", "NVDA", "INTC", "CRM"),
    "Finance": ("JPM", "BAC", "GS", "V", "PYPL"),
    "Healthcare": ("LLY", "JNJ", "UNH", "PFE", "TMO"),
    "Consumer": ("AMZN", "TSLA", "WMT", "NKE", "KO"),
    "Industrial & Energy": ("CAT", "GE", "BA", "XOM", "CVX"),
    "Communication & Utilities": ("GOOGL", "META", "DIS", "T", "NEE"),
}
'''

files["our_agent/universe.py"] = '''"""Load and verify the organizer-supplied universe."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .config import OFFICIAL_SYMBOLS, UNIVERSE_FILE


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def load_verified_universe() -> tuple[str, ...]:
    with UNIVERSE_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    expected = set(OFFICIAL_SYMBOLS)
    discovered = {x for x in _walk_strings(raw) if x in expected}

    if discovered != expected:
        missing = sorted(expected - discovered)
        raise RuntimeError(
            "Organizer universe mismatch. "
            f"Missing expected symbols: {missing}. "
            "Stop and review universe.json before trading."
        )

    return OFFICIAL_SYMBOLS
'''

files["our_agent/intents.py"] = '''"""Explicit strategy actions."""

from enum import Enum


class StrategyIntent(str, Enum):
    HOLD = "HOLD"
    REBALANCE = "REBALANCE"
    TARGET_CASH = "TARGET_CASH"
'''

files["our_agent/validation.py"] = '''"""Deterministic portfolio safety checks."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Mapping

from .config import MAX_TOTAL_EXPOSURE, MAX_WEIGHT, OFFICIAL_SYMBOLS
from .intents import StrategyIntent


class PortfolioValidationError(ValueError):
    """Raised when a candidate target portfolio is unsafe or invalid."""


def _to_decimal(value: object, symbol: str) -> Decimal:
    if isinstance(value, bool):
        raise PortfolioValidationError(f"{symbol}: boolean weights are not allowed.")

    if not isinstance(value, (int, float, Decimal)):
        raise PortfolioValidationError(
            f"{symbol}: weight must be a numeric JSON value."
        )

    if isinstance(value, float) and not math.isfinite(value):
        raise PortfolioValidationError(f"{symbol}: weight must be finite.")

    result = Decimal(str(value))
    if not result.is_finite():
        raise PortfolioValidationError(f"{symbol}: weight must be finite.")

    return result


def validate_target_weights(
    weights: Mapping[str, object],
    intent: StrategyIntent,
) -> dict[str, float]:
    if intent == StrategyIntent.HOLD:
        raise PortfolioValidationError(
            "HOLD must not create a target-weight decision."
        )

    expected = set(OFFICIAL_SYMBOLS)
    actual = set(weights)

    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PortfolioValidationError(
            f"Symbol-set mismatch. Missing={missing}; extra={extra}"
        )

    canonical: dict[str, float] = {}
    total = Decimal("0")

    for symbol in OFFICIAL_SYMBOLS:
        weight = _to_decimal(weights[symbol], symbol)

        if weight < 0:
            raise PortfolioValidationError(
                f"{symbol}: negative weight {weight} is forbidden."
            )

        if weight > MAX_WEIGHT:
            raise PortfolioValidationError(
                f"{symbol}: weight {weight} exceeds {MAX_WEIGHT}."
            )

        total += weight
        canonical[symbol] = float(weight)

    if total > MAX_TOTAL_EXPOSURE:
        raise PortfolioValidationError(
            f"Total equity exposure {total} exceeds 1.0."
        )

    all_zero = total == Decimal("0")

    if all_zero and intent != StrategyIntent.TARGET_CASH:
        raise PortfolioValidationError(
            "All-zero weights require explicit TARGET_CASH intent."
        )

    if intent == StrategyIntent.TARGET_CASH and not all_zero:
        raise PortfolioValidationError(
            "TARGET_CASH requires all 30 target weights to be zero."
        )

    return canonical
'''

files["tests/test_core.py"] = '''import unittest

from our_agent.config import OFFICIAL_SYMBOLS
from our_agent.intents import StrategyIntent
from our_agent.universe import load_verified_universe
from our_agent.validation import (
    PortfolioValidationError,
    validate_target_weights,
)


class CoreSafetyTests(unittest.TestCase):
    def equal_weight(self):
        return {symbol: 1 / 30 for symbol in OFFICIAL_SYMBOLS}

    def zero_weight(self):
        return {symbol: 0.0 for symbol in OFFICIAL_SYMBOLS}

    def test_universe_matches_organizer_file(self):
        self.assertEqual(load_verified_universe(), OFFICIAL_SYMBOLS)

    def test_valid_equal_weight_portfolio(self):
        result = validate_target_weights(
            self.equal_weight(),
            StrategyIntent.REBALANCE,
        )
        self.assertEqual(set(result), set(OFFICIAL_SYMBOLS))
        self.assertLessEqual(sum(result.values()), 1.0000000001)

    def test_missing_symbol_fails(self):
        weights = self.equal_weight()
        weights.pop("AAPL")
        with self.assertRaises(PortfolioValidationError):
            validate_target_weights(weights, StrategyIntent.REBALANCE)

    def test_weight_above_cap_fails(self):
        weights = self.zero_weight()
        weights["AAPL"] = 0.31
        with self.assertRaises(PortfolioValidationError):
            validate_target_weights(weights, StrategyIntent.REBALANCE)

    def test_total_exposure_above_one_fails(self):
        weights = {symbol: 0.04 for symbol in OFFICIAL_SYMBOLS}
        with self.assertRaises(PortfolioValidationError):
            validate_target_weights(weights, StrategyIntent.REBALANCE)

    def test_boolean_weight_fails(self):
        weights = self.zero_weight()
        weights["AAPL"] = True
        with self.assertRaises(PortfolioValidationError):
            validate_target_weights(weights, StrategyIntent.REBALANCE)

    def test_accidental_all_zero_rebalance_fails(self):
        with self.assertRaises(PortfolioValidationError):
            validate_target_weights(
                self.zero_weight(), StrategyIntent.REBALANCE
            )

    def test_explicit_target_cash_passes(self):
        result = validate_target_weights(
            self.zero_weight(), StrategyIntent.TARGET_CASH
        )
        self.assertEqual(sum(result.values()), 0.0)

    def test_hold_cannot_generate_decision(self):
        with self.assertRaises(PortfolioValidationError):
            validate_target_weights(
                self.equal_weight(), StrategyIntent.HOLD
            )


if __name__ == "__main__":
    unittest.main()
'''

files["experiments/README.md"] = '''# ICAIF Experiment Log

Every material strategy experiment should record:

- Experiment ID
- Date
- Git commit / configuration version
- Hypothesis
- Data window
- Feature set
- Model / signal definition
- Portfolio construction rule
- Transaction-cost assumptions
- Cumulative return
- Competition Sharpe
- Maximum drawdown
- Competition turnover
- Diagnostic metrics
- Ablation / baseline comparison
- Failure modes observed
- Keep / reject decision
- Reason for the decision

Validation results are operational evidence and must not be treated as
a statistically sufficient model-selection dataset.
'''

files["CHANGELOG.md"] = '''# ICAIF Agent Changelog

## 2026-10-07 — Initial competition architecture

Organizer repository base revision:

`ad237b074d31dca39c5c975b3e5cb61cf85a8feb`

Created the `our-agent` development branch.

Initial architecture separates:

1. quantitative decision generation;
2. deterministic validation / execution safety.

Added explicit HOLD, REBALANCE and TARGET_CASH semantics.
Added hard portfolio safety checks against the competition constraints.

No live submission logic has been added or enabled.
'''

for filename, body in files.items():
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")

print(f"Created/updated {len(files)} project files.")


def run(*args: str) -> None:
    print()
    print(">", " ".join(args))
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


print("\n=== RUNNING CORE SAFETY TESTS ===")
run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")

print("\n=== CHECKING PYTHON SYNTAX ===")
run(sys.executable, "-m", "compileall", "-q", "our_agent", "tests")

print("\n=== CHECKING GIT DIFF ===")
run("git", "diff", "--check")

print("\n=== FINAL STATUS ===")
branch = subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=ROOT, text=True
).strip()
base = subprocess.check_output(
    ["git", "rev-parse", "main"], cwd=ROOT, text=True
).strip()

print(f"Branch: {branch}")
print(f"Organizer base: {base}")
print(f"Python: {sys.version.split()[0]}")
subprocess.run(["git", "status", "--short"], cwd=ROOT)

print("\nPHASE 2 COMPLETE: core competition safety layer is healthy.")
