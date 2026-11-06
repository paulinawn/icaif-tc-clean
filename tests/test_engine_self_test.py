"""Runs the engine's built-in regression self-test as part of the test suite."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent" / "frozen_strategy"))
import icaif_official_sleeve_engine_v23 as engine  # noqa: E402


def test_engine_self_test_passes():
    engine.regression_self_test()