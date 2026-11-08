# ICAIF 2026 — Official pre-freeze release gates

**Prepared:** 2026-10-09 (Europe/Berlin)  
**Official starts:** Monday 2026-10-12  
**Interpretation of organizer email, as reported by team:** Do not modify trading strategy once Official starts. Freeze strategy source, model artifacts and parameters before the start.

## Status

- Validation Thursday R5 2026-10-08: VALID, SELECTED, EXECUTED. 30 equities purchased (~80% exposure, ~20% cash). The corrected one-observation strategy interface works when run manually.
- Unattended execution is **not yet proven**. A separate **Validation-only** automated R3 probe is available for Fri 2026-10-09 16:50 Berlin; it is not armed until the team explicitly installs and authorizes it.
- Existing phase 12 Ridge ML, phase 13 intraday ML and phase 14 long-horizon factor challenger did not pass holdout promotion gates. Baseline remains equal 30 / 80% / HOLD.
- Phase 15 HGB + sector/risk allocation experiment is newly prepared; real backtest results **pending**. No candidate promoted.
- Public SEC/GDELT/FRED/Finnhub/Alpha Vantage collectors are experimental READ ONLY; no measured incremental trading edge or complete point-in-time historical archives. Do not add them to the live allocation formula without evidence.

## Gate 1 — Freeze competition interpretation

- Confirm verbatim organizer email; record the effective freeze date/time and precisely whether model retraining, configuration changes, API provider swaps or emergency hotfixes are prohibited.
- Re-check organizer repo main hash and latest rules, live schedule, model/external source disclosure requirements.
- Choose **one** frozen strategy release ID (default if challengers fail: `official-equal30-80-hold-v1`).
- Record provider/LLM choices: no live hosted LLM by default. Developer ChatGPT use is separate. If a live LLM is selected, organizer-permitted model name, revision, prompts, inference settings and provider must be documented.

## Gate 2 — Research promotion (time series only)

- Run Phase 15 on the official historical parquet with full transaction costs.
- Save full window-level metrics, not only averages. Compare 2023–2024 development versus 2025 calendar-year tests.
- Understand 2025 has already been inspected for other project experiments; it is **not an untouched project-wide holdout**. Apply a strong multiple-testing penalty / skepticism.
- Run ablations: equal80, equal100, three-sector concentration, vol weighting, sector momentum and nonlinear HGB; compare 15-day and shorter holding periods.
- Check alternative prices/corporate actions, data coverage, and initial entry timestamp assumptions.
- Retain ML only if it materially and consistently beats baseline on fee-aware four-metric analogues across calendar years; no winner from one lucky split.
- Do not backtest new SEC or news records as though they were available historically unless event publication AND retrieval availability were actually archived in time.

## Gate 3 — Deterministic frozen strategy

- Pure `strategy(observation)` callable using official one-object interface.
- Separate data collection, time eligibility, features, prediction, portfolio construction and execution. The model cannot upload by itself.
- Exactly 30 permitted symbols, all finite numeric targets, each in [0,0.30], gross sum <=1; remaining weight is cash; no shorts.
- Apply tested single-stock and sector concentration limits, turnover threshold, fixed fee-aware rebalance criterion and daily loss/drawdown guard.
- HOLD means **no upload**. An all-zero weight vector is a liquidation and must never be used as a fallback.
- If data are stale, symbols are missing, model unavailable, news source rate-limited, or timestamps unclear: safely HOLD. The system must not convert an infrastructure failure into a trade.
- Freeze code SHA256, model SHA256, requirements lock, schedule policy, source allowlist, rule/config versions and exact live strategy import path.

## Gate 4 — Cloud autonomy, security and recoverability

- Validation-only unattended timer starts on server without user being logged in. Verify a real receipt `VALID`, then `SELECTED` and `EXECUTED`; inspect portfolio NAV/fees/weights after execution.
- `systemd` official runner uses `America/New_York` schedule, re-checks organizer current round/deadline, and never depends on laptop.
- Exactly **one** organizer submission writer. Disable/retire all older watchers/timers and Windows scheduled tasks. Use a process/file lock.
- Per-round preflight must check full 30-symbol universe, matching phase/round, timestamp before deadline with buffer, one-attempt-only history, exact code/model hashes, and risk limits.
- No blind retries after an ambiguous POST. Keep the organizer checkpoint. Use read-only decision history/receipt and same-ID recovery.
- `icaif` Linux account invokes all organizer CLI operations; never print team/platform/provider tokens. Keys remain in protected `.env` and `.icaif` files, out of Git.
- Write structured non-secret logs for preflight verdict, intended weights hash, data-as-of, model revision, submission/selection/execution status and latency.
- Test service restart, server reboot, network timeout, market-data outage, price-delay and late-window fail-closed behavior without attempting Official uploads.

## Gate 5 — Sunday, October 11: acceptance / no-go

- Official strategy, scheduler and model are frozen in a Git commit and logged in the experiment CHANGELOG.
- Test suite passes on **Windows and Hetzner under user `icaif`**, including actual organizer single-observation inputs.
- The correct Official account is separate and fresh; first entry scheduled only for Official R1, not accidentally into Validation.
- An independent read-only service checks after each selected trade and alerts/logs discrepancies. A pending price scheduler is not mistaken for failure.
- Verify Official first round with organizer live schedule; standard R1 deadline Mon Oct 12 **15:10 Berlin CEST**, execution **15:30 Berlin**, though official metadata takes precedence.
- With the strategy frozen, enable a single Official unattended runner before first submission; record its next-fire time and hash. Never tell the team “ready” until these checks have actual PASS outputs.
- **NO-GO:** if autonomous Validation fails or Official preflight cannot be reproduced, do not introduce speculative ML/news complexity; use the proven simple strategy after execution plumbing is fixed.

## Experiment log

| Experiment | Hypothesis | Window | Comparator | Status |
|---|---|---|---|---|
| Phase 12 ridge | Linear 5-day cross-sectional alpha | Historical out-of-sample | equal30_80 | Rejected: weak/negative validation signal |
| Phase 13 intraday ridge | Intraday short-horizon alpha | 240 complete days through 2026-10-06 | equal30_80 | Rejected |
| Phase 14 factors | 60/120d sector and momentum tilt | 2021–2025 15-day windows | equal30_80 | Rejected: 2025 robustness gate |
| Phase 15 HGB/sectors | Nonlinear returns and portfolio construction improve ranking after fees | 2023–2024 development, 2025 calendar-year check | equal30_80 | Pending actual data results |
| Validation Friday R3 | One cloud timer can submit unassisted and settle without manual approval | 2026-10-09 one real Validation round | Thurs manual `watch --once` R5 | Pending unattended run |

**References:** organizer competition page and [official repo](https://github.com/DeepIntoStreams/2026ICAIF_Trading_Agent_Competition), relevant docs `rules.md`, `evaluation.md`, `automatic_submission.md`, `llm_and_external_data.md`, and organizer email (verbatim still to be archived).
