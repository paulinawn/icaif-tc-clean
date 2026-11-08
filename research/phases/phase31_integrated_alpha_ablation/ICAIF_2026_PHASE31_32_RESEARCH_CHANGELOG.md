# ICAIF 2026 — Phase 31/32 Research Change Log

Date: 2026-10-11. Offline research, not an Official production release.

## Hypotheses

- H31: combining point-in-time price forecasts, industry strength, dated SEC fundamentals, forecast uncertainty and cost-aware target changes can outperform static policies on the *four-metric* competition proxy.
- H32: independent CaMeL and Laya interpretations of the **same** fresh headline can screen for entity relevance and directional agreement; the event still needs forward return calibration before being tradable.

## Files and scope

- `icaif_phase31_integrated_alpha_ablation.py`: independent 2026 monthly walk-forward price/industry ridge forecasts; 1d/5d relative returns, chronological internal calibration, replay with 0.1% buy/sell fees, nonnegative allocations, partial/staged/intraday decisions, optional SEC as-filed-date+2-day panel, four-metric tournament, paper LLM audit. Uses local Phase30, optionally Phase19. No network, credentials, trading or organizer API.
- `icaif_phase32_dual_llm_event_shadow.py`: one-event two-provider CaMeL/Laya research probe; `--prepare` and `--self-test` offline; `--probe-both` sends exactly one request per provider and may incur charges. Uses Phase23 storage and keys outside repo. Emits no executable trading instruction.

## Windows / status

- Training source (pending real data): Alpaca IEX RAW 15-minute Sep-2025 through Oct-09-2026 from Phase29.
- Development July-August 2026, diagnostic September-October 9 2026. September/October already exposed in research and **not pristine**.
- Tests: both Python self-tests passed; Phase31 complete 30-symbol synthetic-data replay passed without SEC and with 30 synthetic SEC XBRL caches. **Real data not run in this environment.**
- Decision: pending real Phase29, Phase30 and Phase31 outputs; no model promoted to Official.

## Limitations and release gates

- Historical Alpaca IEX bars are not the organizer fill price and are raw, not verified total-return adjusted. No corporate-action total-return correction is claimed.
- SEC CompanyFacts is a current retrieval of facts with historical filing dates; the two-day release lag reduces but does not eliminate restatement/revision biases. One or more tickers can have incomplete facts.
- The 'confidence' value is a residual-scale signal-to-noise proxy, **not** a calibrated statistical probability.
- Rankings are percentiles among this experiment's candidate strategies, not the real competition opponents.
- Live/historical event timestamps and LLM output dates cannot be backfilled. Phase32 outputs are forward-only shadow observations, no demonstrated price alpha.
- Previously rejected RL execution variant is intentionally not embedded. Do not treat module count as evidence of alpha.
- No Official submission, server edits, key printing or training on post-cutoff labels.
