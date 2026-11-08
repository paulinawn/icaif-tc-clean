# ICAIF Trading Agent: Agent-Assisted Refactor

Readability refactor of my trading strategy from the ACM ICAIF 2026 Trading Agent Competition, done with an AI coding agent and verified by a test harness, so the logic stays exactly the same.

See [POC_REPORT.md](POC_REPORT.md) for scope, results and lessons learned.

## What's in here

- `agent/frozen_strategy/`: the strategy (momentum, reversal, L1 turnover rule)
- `tools/generate_policy.py`: runs the strategy offline on example data
- `examples/`: synthetic price data (121 trading days × 30 stocks)
- `tests/`: golden-output test, trade-rule unit tests, engine self-test
- `.github/workflows/tests.yml`: CI, running all tests on every pull request
- `Dockerfile`: runs the test suite in a Linux container, identically on any machine

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

## Test with Docker

No local Python setup needed, only Docker:

```bash
docker build -t icaif-tests .
docker run --rm icaif-tests
```
Expected: 6 passed, on Linux with Python 3.14.

## How changes are made

Every change goes through a branch and pull request. CI must be green before merging. Agent-generated changes are reviewed line by line against the test harness.

## System map

The agent ran live during the ACM ICAIF 2026 Official phase (Oct 12–30, 2026) on a Linux server, started by a systemd timer seven times per trading day.

| Role | File(s) | What it does |
| --- | --- | --- |
| Scheduler | `deployment/systemd/` | Starts each run before a round deadline; sandboxed, never retried |
| Orchestrator (single writer) | `agent/production_v3933/icaif_v39_live_writer.py` | Runs one decision cycle; the only component allowed to submit |
| Gatekeeper | `agent/production_v3933/icaif_v37_production_candidate.py` | Decides whether the round may act; asks the strategy what to do |
| Strategy | `agent/frozen_strategy/` | Computes target weights; tracks the strategy's mode |
| Journal | `agent/production_v3933/icaif_v33_state_machine.py` | Records intent before sending; confirms execution afterwards |
| Translator | `agent/production_v3933/icaif_v29_bridge_unarmed.py` | Turns organizer data into exact values for the strategy |
| Data fetcher | `agent/production_v3933/code/icaif_v28_clean_alpaca_bootstrap.py` | Downloads and checks market data; builds the 121-day history |
| Rehearsal and audit tools | `icaif_v30_*`, `icaif_v32_*`, `v393_preflight_*` | Read-only checks, outside the live loop |
| Organizer kit | `agent/kit/` | Official API client, submission checks and scoring formulas |

**One cycle:** collect organizer state → reconcile earlier submissions → safety gates (pending? round open? scheduled review?) → refresh market data → re-collect and check nothing changed → decide → validate → write intent to the journal → submit once. Every step can end in HOLD; only the last one changes anything.

The production layer needs competition credentials and services that are no longer available. Its logic is verified here with its original offline tests (`tests/test_production_layer.py`).