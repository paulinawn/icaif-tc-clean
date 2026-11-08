"""Golden-output test for review-day mode: current holdings given, L1 rule applied."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "review_policy.json"
DATA = ROOT / "examples" / "synthetic_daily_closes_121.csv"
CURRENT = ROOT / "evidence" / "sanitized_decisions" / "2026-10-19_submission_1770_target_weights.json"
SCRIPT = ROOT / "tools" / "generate_policy.py"


def test_review_mode_matches_golden(tmp_path):
    out_file = tmp_path / "review.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--daily-close-csv", str(DATA),
         "--current-weights", str(CURRENT), "--out", str(out_file)],
        check=True,
    )
    actual = json.loads(out_file.read_text())
    expected = json.loads(GOLDEN.read_text())

    assert actual["action"] == expected["action"] == "TURNOVER_CONTROLLED_TARGET"
    assert actual["reversal_selected_five"] == expected["reversal_selected_five"]
    assert actual["target_weights"] == pytest.approx(expected["target_weights"], abs=1e-12)