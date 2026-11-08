"""Trade/no-trade rule: hold below L1 0.03, trade up to 0.10, cap above 0.10."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent" / "strategy"))
import sleeve_engine as engine  # noqa: E402

SYMBOLS = engine.SYMBOLS


def equal_target():
    """A valid target: 80% invested, spread equally over the 30 stocks."""
    return {s: 0.8 / 30 for s in SYMBOLS}


def current_at_distance(target, l1):
    """Holdings that differ from the target by exactly `l1` in total."""
    step = l1 / 30
    return {s: target[s] - step if i < 15 else target[s] + step for i, s in enumerate(SYMBOLS)}


def l1_distance(a, b):
    return sum(abs(a[s] - b[s]) for s in SYMBOLS)


def test_small_change_holds():
    target = equal_target()
    current = current_at_distance(target, 0.02)
    assert engine.trim_target(current, target) is None


def test_medium_change_trades_to_target():
    target = equal_target()
    current = current_at_distance(target, 0.05)
    result = engine.trim_target(current, target)
    assert result == pytest.approx(target)


def test_large_change_is_capped_at_ten_percent():
    target = equal_target()
    current = current_at_distance(target, 0.15)
    result = engine.trim_target(current, target)
    assert l1_distance(result, current) == pytest.approx(0.10)

def test_initial_entry_ignores_band_and_returns_full_target():
    """On the first purchase, the 3% band must not block buying the full target."""
    target = equal_target()
    current = current_at_distance(target, 0.02)
    result = engine.trim_target(current, target, initial=True)
    assert result == pytest.approx(target)

