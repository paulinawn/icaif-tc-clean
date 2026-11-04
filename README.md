# ICAIF Trading Agent – Clean-up and Migration POC

Clean-up and migration proof of concept of my trading strategy from the ACM ICAIF 2026 Trading Agent Competition.

The goal: make the original code readable with an AI coding agent, and prove with automated tests that its behavior did not change.

## How to run it

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); on Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python tools/generate_policy.py --daily-close-csv examples/synthetic_daily_closes_121.csv --out output.json
```

Expected output: `ICAIF_OFFLINE_POLICY_OK` and the five reversal picks `MSFT,META,CAT,GS,TMO`.

## Status

- Day 0: original strategy imported and runs locally in a fresh environment.