"""Golden-output test: the strategy must produce exactly the saved result."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "sample_policy.json"
DATA = ROOT / "examples" / "synthetic_daily_closes_121.csv"
SCRIPT = ROOT / "tools" / "generate_policy.py"


def test_output_matches_golden(tmp_path):
    out_file = tmp_path / "output.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--daily-close-csv", str(DATA), "--out", str(out_file)],
        check=True,
    )
    actual = json.loads(out_file.read_text())
    expected = json.loads(GOLDEN.read_text())

    assert actual["action"] == expected["action"]
    assert actual["reversal_selected_five"] == expected["reversal_selected_five"]
    assert actual["target_weights"] == pytest.approx(expected["target_weights"], abs=1e-12)