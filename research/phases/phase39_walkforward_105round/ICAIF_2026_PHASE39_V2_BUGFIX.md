# ICAIF 2026 — Phase 39 v2 hotfix (research only)

**Date:** 2026-10-11. **Files:** New `icaif_phase39_105round_walkforward_v2.py`; original Phase 39 file is unchanged. The CaMeL companion file does not need an edit.

## Failure and root cause

The user's real Alpaca run exited 2 with:

```text
PHASE39_FAILED: AttributeError 'Series' object has no attribute 'normalize'
```

At `episode_groups`, the loaded `market.missing['date']` is a pandas **Series**. `pd.to_datetime(series)` is still a Series, whose datetimelike normalization accessor is `.dt.normalize()`. The old script incorrectly used `.normalize()` directly. Its original offline self-test had `market.missing` as an *empty* DataFrame, so it never entered the failing branch. The real 2026 input contains incomplete trading sessions, and the branch is required to avoid crossing an excluded session.

## Changes

1. Convert the parsed date column to a `pd.DatetimeIndex` and normalize it, safely supporting nonempty missing-session records:

   ```python
   missing=set(pd.DatetimeIndex(pd.to_datetime(market.missing['date'], errors='raise')).normalize())
   ```

2. Add a regression test that removes a session from synthetic market arrays, supplies a nonempty `missing` DataFrame, runs `episode_groups`, and rejects any 15-session episode spanning that omitted session.
3. Print a full Python traceback if any future error occurs, preserving failure exit code 2. This makes root-cause identification possible without guessing.

No strategy rules, fee formulas, provider budgets, timestamps, price calculations, or Official code were modified.

## Tests

- `python -m py_compile icaif_phase39_105round_walkforward_v2.py`: PASS.
- `python icaif_phase39_105round_walkforward_v2.py --self-test`: PASS, including `PHASE39_INCOMPLETE_SESSION_REGRESSION_PASS` and earlier 105-round/baseline checks.
- Full offline run using the Phase 37 15-minute **synthetic** CSV after deliberately deleting one AAPL bar on May 4, 2026: PASS. Loader marked one incomplete session, printed `GAP_SPLITS: 1`, replayed 8 non-overlapping 15-session episodes × 25 policies × 105 rounds = 21,000 round records, and finished `PHASE39_COMPLETE_RESEARCH_ONLY_NO_ORGANIZER_NO_LLM_CALLS`, exit 0. All seven report files were generated.
- **Actual Alpaca research data was not available to this container.** No claims about real Phase 39 performance can be drawn until the user reruns v2 on their computer.

## Reproduction

Place `icaif_phase39_105round_walkforward_v2.py` in Downloads next to Phase 30 v2, Phase 37, and Phase 38. From the Windows starter-kit folder, run:

```powershell
python "$HOME\Downloads\icaif_phase39_105round_walkforward_v2.py" --self-test
python "$HOME\Downloads\icaif_phase39_105round_walkforward_v2.py" --run --stride 15
$LASTEXITCODE
```

The script operates solely in local research mode; it does not access keys, LLM providers, Hetzner, or competition submission endpoints.
