"""Runs each module's built-in self-test, each in its own process."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "agent" / "frozen_strategy"
LIVE = ROOT / "agent" / "production_v3933"
KIT = ROOT / "agent" / "kit"

# (test id, script, arguments that start its self-test)
SELF_TESTS = [
    ("data_fetcher", LIVE / "code" / "icaif_v28_clean_alpaca_bootstrap.py", ["self-test"]),
    ("organizer_bridge", LIVE / "icaif_v29_bridge_unarmed.py",
     ["--self-test", "--v23-dir", str(STRATEGY), "--kit-dir", str(KIT)]),
    ("readonly_rehearsal", LIVE / "icaif_v30_remote_get_only.py", ["--self-test"]),
    ("account_audit", LIVE / "icaif_v32_official_account_snapshot_get_only.py", ["--self-test"]),
    ("round_gatekeeper", LIVE / "icaif_v37_production_candidate.py", ["--self-test"]),
    ("history_refresher", STRATEGY / "icaif_official_sleeve_refresher_v23.py", ["self-test"]),
    ("strategy_engine", STRATEGY / "icaif_official_sleeve_engine_v23.py", ["self-test"]),
]


@pytest.mark.parametrize("script, args", [(s, a) for _, s, a in SELF_TESTS], ids=[i for i, _, _ in SELF_TESTS])
def test_builtin_self_test_passes(script, args):
    # The strategy folder must be importable, as it was on the production server.
    env = {**os.environ, "PYTHONPATH": str(STRATEGY)}
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=ROOT, env=env)