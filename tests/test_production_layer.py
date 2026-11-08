"""Runs the production layer's original offline tests, each in its own process."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "agent" / "production_v3933"


def test_history_normalization_passes():
    script = PRODUCTION / "test_v392_history_normalization.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)


def test_postexecution_hold_passes():
    script = PRODUCTION / "test_v3933_postexecution_hold.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)


def test_operational_repair_passes():
    script = PRODUCTION / "test_v393_operational_repair.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
