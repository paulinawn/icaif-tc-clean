# ICAIF Trading Agent: Agent-Assisted Refactor

Readability refactor of my trading strategy from the ACM ICAIF 2026 Trading Agent Competition, done with an AI coding agent and verified by a test harness, so the logic stays exactly the same.

See [POC_REPORT.md](POC_REPORT.md) for scope, results and lessons learned.

## What's in here

- `agent/frozen_strategy/`: the strategy (momentum, reversal, L1 turnover rule)
- `tools/generate_policy.py`: runs the strategy offline on example data
- `examples/`: synthetic price data (121 trading days × 30 stocks)
- `tests/`: golden-output test, trade-rule unit tests, engine self-test
- `.github/workflows/tests.yml`: CI, running all tests on every pull request

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); on Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run

```bash
python tools/generate_policy.py --daily-close-csv examples/synthetic_daily_closes_121.csv --out output.json
```
Expected: `ICAIF_OFFLINE_POLICY_OK` and the five picks `MSFT,META,CAT,GS,TMO`.

## Test

```bash
python -m pytest -v
```
Expected: 6 passed.

## How changes are made

Every change goes through a branch and pull request. CI must be green before merging. Agent-generated changes are reviewed line by line against the test harness.