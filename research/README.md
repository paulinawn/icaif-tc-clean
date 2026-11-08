# Research history: phases 2–44

The research behind the trading agent, September–October 2026: from setup and safety rehearsals, through ML, portfolio and LLM experiments, to the final four-metric strategy choice. Each folder in `phases/` holds the phase's script(s), its `results/`, and its research log where one exists (phases 31–44).

The scripts are kept exactly as written. Their file paths refer to the original working directory, and the licensed Alpaca market data and raw SEC downloads are not included, so they are a record rather than re-runnable code.

## Phases by theme

| Theme | Phases | What was explored |
| --- | --- | --- |
| Setup and safety rehearsals | 02–06, 10–11 | Environment setup, dry-run engine, preflight safety checks, shadow validation rehearsal, frozen validation baseline |
| Early strategy research | 07–09, 12–16 | Interim backtest, anchor-and-tilt allocation, core robustness, ML challenger, intraday and long-horizon factors, leakage-controlled challengers, multi-round policy search |
| Data and signals | 17–21, 28–29 | Feature extraction, SEC + Finnhub collection for all 30 stocks, SEC-based ML, multi-horizon signals, price-integrity audit, 2026 data coverage, Alpaca refresh |
| Portfolio construction and ML | 22, 24–27, 30–31, 33, 37 | Six-industry rotation with risk optimization, hierarchical ML, a narrow Q-learning rebalance controller, alpha tournaments, 2026 walk-forward tests, ablation of price/industry/SEC signals, 80% core with selective overlay, risk-regime tournament |
| LLM event research | 23, 32, 34–36, 39, 42–43 | CaMeL and Laya interpretation of company events: single-request lab, dual-model consensus, budget-gated overlay, forward news intake, earnings watch, point-in-time historical replay of SEC filings |
| Final tournaments | 38–42, 44 | 15-session × 7-round episode tournaments, floating cash, fee-aware reversal, finalist lab, final four-metric comparison |

## Final decision (phase 44)

Seven candidates over six historical 15-session episodes, scored on the competition's four metrics (fees included):

| Candidate | Return | Sharpe | Max drawdown | Turnover |
| --- | --- | --- | --- | --- |
| Equal80 hold | +1.51% | 3.33 | 1.60% | 0.76% |
| Blended80 hold | +1.56% | 3.45 | 1.66% | 0.76% |
| Full dynamic reversal | +1.78% | 3.10 | 1.59% | 1.26% |
| **Blended80 + 15% reversal sleeve (deployed)** | +1.66% | **3.57** | 1.63% | 0.95% |
| + CaMeL earnings signal | +1.66% | 3.57 | 1.63% | 0.95% |

The deployed strategy had the best Sharpe ratio and beat the Blended80 baseline on return in 5 of 6 episodes, at about 25% higher turnover. The LLM earnings signal was kept out of live trading: only two historical events could be tested, without analyst-consensus data, so the experiment could not show an effect either way. Details: `phases/phase44_final_four_metric_comparison/`.

## Documents

- `docs/ICAIF_2026_Project_Knowledge.txt`: competition rules, schedule and the four scoring metrics
- `docs/ICAIF_Official_PreFreeze_Release_Gates.md`: the checks the agent had to pass before the strategy was frozen