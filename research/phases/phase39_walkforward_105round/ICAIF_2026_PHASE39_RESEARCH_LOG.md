# ACM ICAIF 2026 — Phase 39 research log

**Status:** OFFLINE RESEARCH, SOFTWARE TESTS COMPLETE. **Real Alpaca 2026 performance:** NOT YET RUN HERE. User's authenticated Alpaca merged data remains in the Windows starter-kit; the working container has only a synthetic-data full-replay sample. No Official deployment, submissions, VPS changes, paid model calls or keys accessed.

## Purpose, scope and version

Two new files, independent of the Official runtime:

1. `icaif_phase39_105round_walkforward.py`: 15-session cash-reset, 105 decision/valuation periods per episode, seven submission deadlines/execution price proxies/day, 0.1% buy-and-sell fees, 30 long-only symbols, 30% per-name max. Retains **Phase38 / Phase37 / Phase30 v2** code for static benchmark parity. Runs non-overlapping episodes (`--stride 15`, default) and optional overlapping sensitivity (`--stride 5`, correlated). Reports all four competition-like metrics **and** paired return/Sharpe/MDD/turnover/fee differences relative to same starting core. No network calls.
2. `icaif_phase39_camel_forward_lab.py`: bounded CaMeL-only multitask research (novelty, documented facts, issuer attribution, supplied expectations, causal business implications, counterargument, unknowns) with a maximum of three explicit, potentially billed model calls per research ledger, no Laya. Requires actual source excerpts (not just a headline), checks quoted evidence, timestamps publisher/first observation/model completion/**human review**, and outputs reviewed **paper-only** events. Has a separate future-price return scorer with a matched 30-stock control and 0.1% modeled roundtrip fee. No trades or organizer actions.

## Hypotheses and exact strategy families

**H1 — time diversification.** Buying 80% stock exposure at the opening may create avoidable first-day timing risk. Fixed seven-round entry starts at 30% in R1 and increases linearly to 80% by R7. First-day staged implementation **only buys shortfalls, never sells to rebalance**; this is a deliberate fee-saving execution hypothesis. It may underperform in trending-up markets because of cash drag.

**H2 — causal adaptive entry.** All three adaptive/constant staged policies start with 30% equities and end at approximately 80% by R7. The advance/slow rules are predeclared and apply to the *mean log return of the 30 symbols* from R1's 09:30 price through the most recently **completed** bar at the current deadline. At +0.30% aggregate day return, trend acceleration advances exposure by 15 portfolio percentage points; at -0.30%, it delays by 10 points; the contrarian policy reverses that interpretation. Targets are monotone and max 80%; R7 reaches 80%. Rules are not fitted to observed performance. Exposure after R7 may drift slightly due to fill price and buy-only behavior. No monthly trading restriction.

**H3 — differentiated event information.** Compare immediate-investment portfolios, fixed staging, and causal adaptive staging **with and without** verified event-driven paper overlays. The event-only CaMeL policies have identical control to the corresponding HOLD core when no signals exist. Historical CaMeL outputs made on/after October 11 cannot be attributed to July–October 9 decisions. Accept only first observed source and actual LLM analysis and source-human-verification timestamps preceding the deadline. Max temporary per-stock relative tilt of 0.3% NAV, offset equally over other 29 holdings; no forced signal.

**Policies:** Four static HOLD baselines: Equal60, Equal80, blended80 (Phase37/38), and Phase37 legacy low-vol80. For each of three 80% cores: `fixed7`, `accel_trend`, `accel_contrarian`, each with and without a CaMeL event overlay. Plus a CaMeL-only immediate core variant for each of the three. **25 total**, **13 unique non-CaMeL research ranking references**, zero event trades if no verified events exist. Prior Phase38 generic daily relative-strength and intraday reversal heuristics are **not carried forward**, because they had failed paired net-value tests. Their 105-round infrastructure is retained.

## Data windows and causal protocol

- Segment 1: **Early 2026 (January–June)**, but episodes may begin only when the Phase30 v2 *existing* industry core has at least 121 completed historical feature days (in the Sep2025–Oct2026 Alpaca feed, this can begin late February). We do **not** silently shorten the industry model's training window to include January.
- Segment 2: July–August 2026, previously examined in Phases 30–38.
- Segment 3: September–October 9, also previously examined; **not an untouched out-of-sample set**.
- Each episode has exactly 15 complete trading sessions, 105 rounds, $1m initial cash. Missing half-days and incomplete sessions never get filled with made-up data; episodes do not cross *documented* incomplete sessions. Entirely absent market dates not represented in the source are not guaranteed to be detected; exchange calendar audit is still necessary.
- R1 deadline 09:10 ET, execution 09:30; R2–R7 deadlines 10:25 through 15:25, execution 10:30 through 15:30. R2 may use 10:00 bar ending 10:15, not future 10:15 bar ending 10:30. Consistent with Phase38.
- Synthetic or real training/evaluation episodes share no day between **stride-15** windows, but observations in a single period remain economically correlated and do not establish truly independent test regimes. Competing strategies are not optimized using the subsequently observed episode.

## Output files and acceptance logic

Directory `artifacts/phase39_105round_walkforward`:

- `phase39_episode_metrics.csv`: full four metrics/fees per 15-day episode, with 13 non-CaMeL opponent-relative internal percentile ranks.
- `phase39_paired_episode_deltas.csv` and `phase39_paired_summary.csv`: staged against **same** immediate core, CaMeL against **identical** non-CaMeL entry and issuer event-only against immediate HOLD core; report paired delta on return, Sharpe, drawdown, turnover and fees; no cherry-picking by best rank.
- `phase39_policy_summary.csv`: segment means, trades, rank.
- `phase39_round_activity.csv`: R1–R7 actual trade counts/notional/fees.
- `phase39_105_round_audit.csv`: each proposed/actual decision and fee.
- `phase39_manifest.json`: source SHA-256, episode counts, constraints and research limitations.

Decision to promote a staged approach: incremental **after-fee** returns consistently positive in the earlier non-overlapping episodes, without material deterioration in the other three metrics. If gains are confined to an already examined segment, **do not promote**. A lower drawdown with negative return can still be acceptable under 4-metric ranking, but requires scenario-weighted reasoning, not ad hoc thresholds. Higher trading frequency without improved metrics is negative evidence.

For CaMeL: no alpha claim until a **sample of distinct, new, timestamped, verified** events is evaluated against the identical non-LLM core. The first one or two events are pipeline proof, not evidence of net predictive edge. A same-provider second pass is not an independent model; Laya is reserved for rare questions.

## CaMeL forward event workflow

- Use Phase35's existing source collector (optional read-only Finnhub collection) to select a candidate. **Reject the Oct11 NVDA futures/earnings preview** as not issuer-specific new results.
- Supply an actual reviewed public company/SEC excerpt as a UTF-8 `.txt` file (>=100 characters) to Phase39's event lab; optional separate prior consensus/expectations `.txt` makes surprise assessment possible.
- Offline `--prepare` produces one candidate packet; only explicit `--probe-camel --ack-provider-credits` makes one short provider request. Max **3 distinct** reservations locally, no retries and no automated continuous billing; remaining provider balance must be checked in dashboard. If provider omits actual USD cost, mark **unknown**, never call a zero charge proven.
- The CaMeL result quotes public evidence, identifies expected vs observed information, and offers a counterargument. By default all report claims are **unverified**. `--approve --ack-source-reviewed` is a human attestation step, making a paper event eligible **only after that review timestamp**; it cannot be backdated to the model completion time.
- `--score-forward --market-csv` waits for subsequently acquired forward market prices and compares stock vs 30-stock basket plus a hypothetical ±0.3% tilt after a conservative roundtrip fee. Outcome horizons are the **next 1 and next 5 completed trading-session closes** after the first eligible permitted execution (not the article time). The replay has a separate approved-event gate but historical Phase30 data ends Oct9; zero live event impact is expected in the current historical runs.
- The 3-call ceiling is a local request-count rule, **not** a guaranteed US-dollar spending ceiling. No Laya tokens consumed by this workflow.

## Software validation, completed

- Phase39 historical simulator self-test: PASS, exact 105 decisions, buy-only cash/fee conservation, legal weights, causal known-bar first-day staging, R1-to-R7 schedule, absence-of-event CaMeL parity, and **exact Equal60/Equal80 baseline parity with Phase38** on the identical synthetic 15-day input.
- Synthetic historical end-to-end run: PASS, 9 non-overlapping episodes (6 early, 2 Jul/Aug, 1 Sep/Oct), 25 policies = **225 policy-episode reports and 23,625 audited rounds**, including all R1–R7. Initial Equal80 HOLD one trade; fixed seven-round Equal80 seven trades; no fake CaMeL alpha, correct zero event trades.
- Separate CaMeL event lab: PASS, strict provider-response JSON/evidence source quote gate, capped-credit local ledger, source chronology checks and synthetic forward outcome scoring with after-inference first eligible submission/execution.
- `python -m py_compile` both files: PASS.
- **None of these synthetic returns are real benchmark results.** The real 2026 Alpaca dataset is not mounted in the research container. The real Phase39 score, episode counts, and strategy decision are pending the user's local run.

## Stage 40 — explicitly deferred until Stage 39 analysis

The user proposes three new hypotheses for **Stage 40 only**:

1. **Cash as an execution buffer:** portfolio exposure allowed to vary, initially test 70–90% (i.e. $100k discretionary cash/equity variation around $800k) rather than forced 80%; trade using cash before stock-for-stock switches where expected incremental return after 0.1% fees merits it. This can save sell-side fees in some transfers but higher equity exposure also raises risk; must compare like-for-like return, Sharpe, MDD and turnover.
2. **Selective holdings:** 15–25 of 30 names may beat holding all 30 if profitable/low-risk names can be identified, but selection raises concentration and sector/stock risk. Test 15, 20, 25, 30 across predeclared price momentum, short-horizon mean reversion, volatility and drawdown/liquidity filters, sector constraints and turnover gates, versus exact 30-stock cores. Do **not** assume recent losers continue to lose; contrarian rebound is a real counter-hypothesis. One-name portfolios are too concentrated to be a reasonable default.
3. **Independent alpha search on raw data:** explore new, economically motivated short-term / cross-sectional signals with anchored causal timestamps, robust universe and corporate-action handling; use many time periods and predeclared out-of-sample protocol, multiple-comparison safeguards and sensitivity to IEX data gaps. Do not tune to the already-examined Oct2026 tail.

**No Stage40 code, model tuning or historical experiment will run until real Phase39 results have been examined and the user directs Stage40 work.**
