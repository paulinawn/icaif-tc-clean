"""Runs each module's built-in self-test, each in its own process."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "agent" / "strategy"
LIVE = ROOT / "agent" / "live"
KIT = ROOT / "agent" / "kit"

# (test id, script, arguments that start its self-test)
SELF_TESTS = [
    ("data_fetcher", LIVE / "market_data_fetcher.py", ["self-test"]),
    ("organizer_bridge", LIVE / "organizer_bridge.py",
     ["--self-test", "--v23-dir", str(STRATEGY), "--kit-dir", str(KIT)]),
    ("readonly_rehearsal", LIVE / "readonly_rehearsal.py", ["--self-test"]),
    ("account_audit", LIVE / "account_audit.py", ["--self-test"]),
    ("round_gatekeeper", LIVE / "round_gatekeeper.py", ["--self-test"]),
    ("history_refresher", STRATEGY / "history_refresher.py", ["self-test"]),
    ("strategy_engine", STRATEGY / "sleeve_engine.py", ["self-test"]),
]


@pytest.mark.parametrize("script, args", [(s, a) for _, s, a in SELF_TESTS], ids=[i for i, _, _ in SELF_TESTS])
def test_builtin_self_test_passes(script, args):
    # The strategy folder must be importable, as it was on the production server.
    env = {**os.environ, "PYTHONPATH": str(STRATEGY)}
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=ROOT, env=env)