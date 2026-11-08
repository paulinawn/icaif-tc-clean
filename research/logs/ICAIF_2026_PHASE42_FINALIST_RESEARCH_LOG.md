# ICAIF 2026 — Phase 42: Multi-source finalist architecture and bounded tournament

**11 October 2026 — RESEARCH ONLY.** Official deployment, submission, Hetzner automation, organizer credentials and Laya usage remain OFF. No Finnhub, SEC or CaMeL network request was made from this working environment. User must explicitly approve Official operations separately.

## Decision context

Stage 41's 2026 price-only strict absolute expected-return threshold left about two eligible holdings and approximately 16% invested on average. This is not the desired 70–90% dynamic equity portfolio. Across six 15-session episodes, static Blended80 generated about +1.564% return with Sharpe 3.338, $799 initial purchase fees and 30 holdings. Stage 41's reversal5_floating returned −0.317% with roughly 16.5% equity exposure. These periods have been repeatedly examined; they are **not untouched validation**.

**Phase 42 design decision:** Replace the rigid five-day expected-positive-return gate with a *relative opportunity and downside-risk framework* that preserves sufficient diversification while allowing individual stocks to fail eligibility. Stock count is NOT fixed to 15, 20, 25 or 30. Do not claim the relative opportunity score estimates expected dollar returns. Vary equity budget 70–90%, with one 70–95% stress variant. Protect against churn with a 4.5%-of-NAV L1 no-trade region, trade only every third session on ordinary active strategies, and cap subsequent decision turnover to 12% of NAV by pre-deadline marks. Initial entry is immediate, not seven-step staging. The full scenario still evaluates all 105 permitted rounds; most R2–R7 rounds make a deliberate HOLD decision. Test one bounded R2 overnight-gap BUY branch separately. Commission=0.1% on gross buys plus sells.

## Research candidates and motivations

* Equal60 HOLD, Equal80 HOLD, Blended80 HOLD, low-volatility80 HOLD: immutable comparators.
* 20-session strength: cross-sectional continuation, 5/20-session returns.
* Five-session reversal: test mild predictive rank evidence from Phase 40; deliberately reduce rebalances to every three sessions.
* Trend + short-term pullback: seek strong 20-session trends with short-term declines, volatility moderated.
* Regime blend: switch deterministic momentum-versus-reversal proportions using preceding broad-market trend/volatility. This is a *hypothesis*, not a learned optimal classifier.
* Volatility-quality: lower realized variance combined with 20-day strength.
* Sector-relative: remove part of broad sector score to avoid concentrations and sector-wide selection mistakes.
* Cash-first execution: same pullback target, but use available cash before sell-side rebalancing; separately audit realized equity differences.
* 70–95% regime stress, R2 gap adjustment: controlled portfolio/risk ablations.
* SEC+Finnhub+CaMeL forward counterpart: uses the same regime price policy, with a very small +/-0.3 percentage-point paper tilt only after a human-reviewed event is observed before a valid decision. No retroactive model signals.

These are predeclared economic hypotheses. They are not optimized on the six episodes in this run.

## What is genuinely integrated

The **offline backtest** consumes Alpaca IEX price bars only (SEC/Finnhub/LLM events were not recorded contemporaneously for this historic period). No false inference from a historical news headline or retrospectively retrieved/restated XBRL fact is inserted into a historical decision.

The separate **Phase 42 source bridge** executes public Finnhub issuer-news and SEC EDGAR filings/CompanyFacts READ-ONLY when explicitly requested. It stamps snapshots with first observation, keeps SEC *filing* and *acceptance* dates and XBRL *filed* dates distinct, records source gaps and does not print tokens or contact email. It prepares a CaMeL input combining: (i) the Finnhub-discovered issuer event, (ii) an actual user-reviewed publicly available source excerpt, (iii) SEC filing and reported fact context clearly labelled *not new-event proof and not expectations*, and (iv) independently supplied analyst expectations if available. SEC context is supporting evidence, not itself a calibrated signal. It uses existing Phase39 CaMeL processing, source-quote checking, mandatory human source review and existing max-three-call reservation ledger. No Laya use.

Following user review, Phase39 writes a `phase39_reviewed_paper_events.json` with true source publication, first observation, CaMeL completion, and review timestamps. Phase42 reads it forward-only and, if the event was actually known at a valid decision deadline, can execute a +/-0.3%-of-NAV **paper** tilt subject to cash, name, sector and exposure limits. After-hours eligible events can be considered at the next R1 only when reviewed before its 09:10 ET deadline. THIS DOES NOT ESTABLISH POSITIVE EVENT ALPHA. No Approved event exists in these six historical research episodes; the paired results are identical.

**Source-specific limitations**: SEC current CompanyFacts may include restatements and cannot be used as an archived past forecast. Finnhub headlines may be misclassified or refer to unrelated issuers. Neither SEC XBRL nor a CaMeL confidence adjective automatically proves a return surprise. Actual historical analyst expectations are often unavailable, so surprise is unknown. The source bridge does not automatically turn an LLM response into a live trading decision.

## Completed local technical checks

- Portfolio syntax and `--self-test`: PASS. Dynamic exposure responsive, no fixed count requirement, allocation within conservative 9.5%-name / 29.5%-sector target caps, and simulated one-name positive/negative event-tilt fee conservation.
- Source bridge syntax and `--self-test`: PASS. SEC CIK resolution, accepted form filtering, filed dates, compact SEC context, zero-network offline operation.
- Full real-data Alpaca IEX October 6 CSV experiment: PASS. Six 15-day non-overlapping episodes; 14 policies; 8,820 (6 × 14 × 105) decision records. All 30 official symbols are present. No real API or model requests. No recorded >10% name or >30% internally defined sector breaches at execution; hard 30%-organizer stock max is not approached.
- Historical static comparator parity: the static Equal60, Equal80, Blended80 and legacy low-vol 80 outcomes match the prior Phase41 episode set because Phase39 exact replay was reused.
- The 95% variant is a genuine alternative exposure schedule but its observed mean exposure was only about 86.6%; do **not** equate a larger allowed maximum with trading more than 90% in every round.

## Real-data Phase42 findings (October 6 input)

Means across **six previously examined** 15-session episodes, 105 rounds each, net of modeled 0.1% fees; these are NOT clean holdout outcomes. Sharpe is annualized using 7 decision periods per session and is noisy.

| Policy | Mean net return | Mean Sharpe | Mean MDD | Mean fees/episode | Mean equity exposure | Mean holdings |
|---|---:|---:|---:|---:|---:|---:|
| Equal60 HOLD | +1.131% | 3.222 | 1.196% | $600 | 60.2% | 30.0 |
| Equal80 HOLD | +1.508% | 3.229 | 1.589% | $799 | 80.2% | 30.0 |
| **Blended80 HOLD** | **+1.564%** | **3.338** | 1.664% | $799 | 80.2% | 30.0 |
| Lowvol80 HOLD | +1.425% | 3.130 | 1.532% | $799 | 80.2% | 30.0 |
| **Dynamic five-day reversal** | **+1.780%** | 3.016 | 1.555% | $1,332 | 84.1% | 27.1 |
| Dynamic sector-neutral | +1.566% | 2.714 | 1.827% | $1,328 | 84.0% | 25.3 |
| Dynamic trend-pullback | +1.376% | 2.554 | 1.615% | $1,328 | 83.9% | 26.0 |
| Dynamic regime mix | +1.406% | 2.400 | 1.919% | $1,328 | 84.1% | 24.5 |
| Dynamic volatility-quality | +1.207% | 2.243 | 1.514% | $1,299 | 83.9% | 23.4 |
| Dynamic regime wide95 | +1.466% | 2.410 | 1.993% | $1,355 | 86.6% | 24.6 |

**Five-day reversal** is the leading active return candidate but beats Equal80 and Blended80 on return in only **3/6 episodes**; it exhibits regime instability, worse Sharpe than Blended80 and approximately $533 additional fees per episode. Do **not** promote it as the default finalist. The cash-first pullback did not improve net performance relative to the stronger static baselines; its realized exposure differs from the regular pullback so attribution requires caution. The internal equal-four-metric candidate-ranking favors Equal60 and Blended80 in this particular set, but it is **not** the official opponent-relative rank distribution.

**No LLM uplift is measurable yet.** The historical `dynamic_regime_multisource_shadow` is numerically identical to `dynamic_regime_mix` because no approved contemporaneous source-model signals existed. Calling CaMeL now can only create prospective research records.

## Evidence and research limitations

1. Alpaca IEX prints and 15-minute bar starts approximate execution; competition backend prices/slippage are not known to this simulator. No explicit spread or market impact beyond 0.1% fee.
2. Missing/half-session gaps split episode windows; the 2026 periods have been repeatedly inspected. Strategy comparisons do not represent independent performance discovery.
3. Relative-alpha eligibility cutoff (entry z > -0.45, retention > -0.95) is heuristic, not a statistically calibrated probability of positive 5-day return; cross-sectional normalization can always rank some stocks favorably even in weak markets.
4. Capital risk and name/sector targets are conservative research constraints; continuously drifting prices between rounds can still change weights. Pre-submission Official check is a different consequential operation that is NOT performed in Phase42.
5. Public-source collection reads a few issuers per explicit call and does not construct an archived as-of historical multi-source dataset; this is deliberate, to avoid silently backfilling SEC/news and producing fictitious alpha.

## Default decision and final remaining experiments

**Provisional default remains Blended80 HOLD** for an auditable balanced core, with Equal60 and Equal80 as indispensable four-metric controls. For the final one/two rounds, shortlist only (A) a **small, fixed-cost, 5-day reversal sleeve** added to the core instead of replacing it and (B) a **forward-only genuinely sourced CaMeL event sleeve**. Require paired after-fee improvement on repeated 15-session episodes or accumulate forward evidence, no arbitrarily increasing the variant count. Explore one exposure modulation separately rather than assuming +/-10% is necessarily better. If no active challenger meets the threshold, freeze the best robust simple core.

Next workflow: (i) run Phase42 on Oct9 updated CSV and analyze its four metrics and activity; (ii) collect the latest SEC/Finnhub facts for one genuinely new issuer event, source-check, invoke at most one budgeted CaMeL call and paper-track any tilt; (iii) freeze two finalist candidate specifications, then test clean production correctness **only if and when the user authorizes deployment**.

## Source references

- Official competition trading rules, 30 symbols, target weights and 0.1% fee: https://github.com/DeepIntoStreams/2026ICAIF_Trading_Agent_Competition/blob/main/starter-kit/docs/rules.md
- U.S. SEC EDGAR company submissions and XBRL APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC fair access / identifiable User-Agent: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- DeMiguel, Garlappi & Uppal (2009), **Optimal Versus Naive Diversification**: https://doi.org/10.1093/rfs/hhm075
- Medhat & Schmeling (2022), **Short-term Momentum**: https://doi.org/10.1093/rfs/hhab055
- Dybvig & Pezzo (2020), **Mean-Variance Portfolio Rebalancing with Transaction Costs**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3373329

## Windows commands (from starter-kit, Downloads contains prerequisites)

```powershell
Set-Location "$HOME\Documents\ICAIF2026\2026ICAIF_Trading_Agent_Competition\starter-kit"
.\.venv\Scripts\Activate.ps1
python "$HOME\Downloads\icaif_phase42_multisource_finalist_lab.py" --self-test
python "$HOME\Downloads\icaif_phase42_source_bridge.py" --self-test
python "$HOME\Downloads\icaif_phase42_multisource_finalist_lab.py" --run --stride 15 2>&1 | Tee-Object "artifacts\handover_logs\phase42_console.txt"
$LASTEXITCODE
```

Optional forward source collection (read-only SEC + Finnhub, ZERO CaMeL calls):

```powershell
python "$HOME\Downloads\icaif_phase42_source_bridge.py" --collect --symbols JPM GS BAC --limit 3 --ack-readonly-api-requests
```

Needs existing `~\.icaif\research_sources.env` with `FINNHUB_API_KEY` and `SEC_CONTACT_EMAIL`. Source snapshots contain publicly returned finance data, source statuses, and UTC observation times; no raw credentials. For any LLM call the user MUST first inspect/validate the news and supply a genuine public article/issuer source excerpt, then explicitly acknowledge provider credits via `--probe-camel` using the prepared packet. CaMeL's existing per-phase three-call ledger prevents silent reruns. Laya remains unused. Phase39 approves an event only with separate human source review. The full process does not contact Codabench or the competition backend.
