# ICAIF 2026 — Phase 38 Research Changelog

**Project:** ACM ICAIF 2026 Trading Agent Competition. **Status:** OFFLINE RESEARCH ONLY. No Official submission, SSH, Hetzner configuration, or hosted LLM calls in this phase. User explicitly forbids Official deployment until authorization.

## Change / rationale

New file: `icaif_phase38_105slot_camel_staging.py`.

Prior Phase37 assessed some monthly/weekly risk rules across 70 sessions, despite the Official test running 15 sessions with 7 execution opportunities per day. Phase38 uses **cash-reset 15-session episodes** with *exactly* **105 rounds**, correct submission times (09:10 / 10:25 / 11:25 / 12:25 / 13:25 / 14:25 / 15:25 ET), execution timestamps (09:30 ... 15:30 ET), lagged intraday decision inputs, fee-aware share/cash simulation, and end-of-round plus daily-close marks. No monthly recalibration or rebalance restriction.

## What is tested

- Original benchmarks: Equal60 HOLD, Equal80 HOLD, Phase33 blended80 HOLD, Phase37 six-bucket lowvol80 HOLD (legacy control).
- New `lowvol80_hold`: manually defined nine-sector-like grouping across all 30 stocks; 50% stock-equal and 50% sector-equal exposure, with 60% within-sector equal and 40% within-sector inverse 20-day volatility, all computed using previous completed closes. This is a **different portfolio** from Phase37 lowvol: do not confuse results.
- Staged entry: immediate .80; `.30` equity at R1 and `.80` at R2; or `.30` equity at R1 increasing in 6 equally spaced stages to `.80` at R7. In Equal80, R1 is 1% NAV per stock and final is 2.6667% NAV per stock, before returns/fees. Staging preserves one-time 15-day cash reset and trades at round-specific fill proxies.
- Day2–15 relative-strength updates: previous session's 5d/20d relative stock returns, zero-sum +/- small tilts, submitted R1 only when trade-notional threshold met. Separate completed-bar reversal updates allowed R2 through R7. Both are **heuristic**, not calibrated positive-IC forecasts. Returns/fees determine keep/reject.
- CaMeL pairs for every non-HOLD variant: no CaMeL classification made after simulated submission deadline can be applied. Reader accepts a JSON array of verified, company-specific, non-price-reaction, material *new* facts, including source publication time, first observation, and CaMeL completion time; analysis completed before the submission deadline is mandatory. Paper-only weight tilt +/-0.3 percentage points, offset equally across the other 29 stocks. Current historical input July–October 9 predates Oct11 CaMeL inference; paired results will therefore be **identical**, and **cannot establish LLM alpha**. A Phase36 preview JSON can be provided, but is rejected as a trading event. No new CaMeL or Laya API requests are made, preserving credits.
- Nine standard historical rolling episodes expected in complete synthetic source: 6 in July–August and 3 Sep–Oct9 at stride 5. Episodes are 15 contiguous *retained* complete sessions, additionally skip episodes that span a documented incomplete date. Overlapping windows are correlated and not independent evidence. Choose `--stride 15` for nonoverlapping 15-session evaluation. Diagnostic Sep-Oct has already been inspected and is not held out.
- The 0.1% fee applies to buy **and** sell notional on every executed rebalance. All profiles long-only, <=30% stock weight, <=100% total. No official agent/server interaction.

## Causal windows

- R1: only market data up through **previous completed session close**, before 09:10 ET deadline; fill at 09:30 ET IEX bar opening price proxy.
- R2–R7: stock-specific 15-minute bar completed before respective submission deadline; fill at next allowed hourly execution.
- R7 end marked at 16:00 close proxy. Cash and shares carry across rounds and sessions; $1m reset only at episode beginning.
- Event available at `camel_completed_utc`, not article's nominal publication. Source first-observed timestamp must fall between publication and CaMeL completion, never afterward.

## Files and outputs

Requires `icaif_phase30_2026_walkforward_agent_lab_v2.py` and `icaif_phase37_risk_regime_tournament.py` beside Phase38 script (usually `Downloads`). Reads Phase29 merged historical CSV from starter-kit without modifying it. Writes:

- `artifacts/phase38_105slot_camel_staging/phase38_15day_episode_results.csv` — every policy and 15-day episode.
- `phase38_policy_summary.csv` — descriptive episode means and simulated ranks.
- `phase38_paired_vs_same_core_summary.csv` — **preferred** comparisons of staged/active vs identical static core in same episodes.
- `phase38_round_activity.csv` — number of executed trades by round, policy, segment.
- `phase38_camel_paired_deltas.csv` — CaMeL paired differences (expected zero with no timestamp-eligible verified event).
- `phase38_all_105_round_decisions.csv` — each of 105 decisions per policy/episode, actual fees, round, signal and cash/exposure.
- `phase38_manifest.json` — input SHA, dates, assumptions, provenance, risks.

Ranks: compare against only the 17 **non-CaMeL reference policies**; paired duplicated CaMeL variants never inflate the opponent pool.

## Validation completed on research host (synthetic only)

- Phase38 `--self-test`: **PASS** — 105 rounds, stage exposure sequence, transaction fees and constraints, historical leakage guard, event timestamp gate, an actually executed synthetic event paper trade, future price invariance, benchmark rank parity for unseeded CaMeL pairs.
- Full end-to-end synthetic 15-minute CSV (all 30, sessions Sep2025–Oct9 2026), 9 rolling episodes × 29 policies × 105 = **27,405 logged decisions**, **PASS**. All 29 variants match their CaMeL-less counterparts where no causally eligible events are present, as intended. This is a plumbing check, not alpha evidence.
- Independent 15-session synthetic baseline parity with Phase37: Equal60, Equal80, Blend80, and legacy pseudo-sector Lowvol80 produce *identical* cumulative return, Sharpe, MDD, turnover, fees and trades (max error exactly zero).
- **No real Alpaca 2026 Phase38 performance tested in this container:** user has authenticated Phase29 merged data on Windows; user must run and share results.

## Scientific caveats / decision rules

- The model may evaluate 105 opportunities without transacting 105 times; forced trading would needlessly increase turnover and costs. Unlike Phase37, staged rules deliberately place R1–R7 trades on Day1.
- Staging is not automatically safer/stronger: it is predominantly lower intraday exposure to market moves. Compare after-fee return, Sharpe, MDD and turnover; whether it wins varies with entry-day price path.
- July–October 2026 has been viewed; do not choose thresholds based solely on its outcome. Multiple rolling episodes overlap; prioritize consistency in nonoverlapping windows and forward paper observations.
- IEX prices not adjusted for corporate actions and are not the organizer's fill source; 36 missing/half-day sessions excluded. Manual sector mapping not a verified as-of GICS dataset. Earnings/SEC provider historical event outputs cannot be retrospectively imputed.
- In a future forward replay, CaMeL-positive/negative labels are not calibrated expected returns. Treat paper tilts as scenario evidence only until verified incrementality after fees. `source_verified` etc are human attestations, not automatically checked against primary documents.
- Reject staged/adaptive models if their paired improvement fails across distinct 15-session episodes or they materially worsen Sharpe/MDD/turnover; retain no-trade static alternatives. Zero-event CaMeL paired differences are *not* evidence CaMeL has no value.

## Immediate next actions

1. Run Phase38 `--self-test`, `--run` on actual October9 Alpaca 15-minute data; optionally run `--stride 15` as nonoverlap sensitivity.
2. Upload `phase38_paired_vs_same_core_summary.csv`, `phase38_policy_summary.csv`, and `phase38_round_activity.csv`. Compare staged/reversion/lagged momentum against holds using paired per-episode metrics.
3. Forward-only CaMeL event collection: use existing Phase35/36 flow on a truly new company-specific earnings/guidance release, store real inference completion times and human verification, then assess incremental paper performance. Keep Laya input tokens for rare adjudication.
