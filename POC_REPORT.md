# POC Report: Agent-Assisted Refactor of ICAIF Strategy Code

## 1. Summary

An AI coding agent (Devin's local agent in the Windsurf/Devin IDE) refactored the two core strategy files of my ICAIF 2026 trading agent for readability. Behavior stayed identical, verified by a test harness and CI. The agent saved substantial time but needed close human review: it bent two naming rules and made one confident factual error. Recommendation: continue, with tighter instructions and broader test coverage.

## 2. Scope and success criteria

**In scope:** two files in `agent/strategy/`
- `market_support.py` (~190 lines)
- `sleeve_engine.py` (~470 lines)

**Off-limits:** all other files, especially `tests/` and the golden output.

**Success meant:**
- Code is clearly more readable (one statement per line, descriptive names, docstrings)
- Behavior is unchanged: all tests pass, locally and in CI
- No changes outside the two files

## 3. Approach

1. **Harness first.** Before the agent touched anything, I built:
   - a golden-output test (end-to-end): the full strategy must reproduce the saved target weights
   - unit tests for the L1 trade/no-trade rule (hold, trade, cap, initial entry)
   - CI with GitHub Actions running all tests on every PR
2. **Extend the harness before the riskier file.** The engine already had a built-in self-test. I added it to the test suite unchanged, merged it as its own PR, and told the agent not to touch it.
3. **Scoped agent tasks.** Each prompt defined goal, scope, rules, constraints and a definition of done (`python -m pytest -v` green). For the engine, the agent worked in steps and ran the tests after each one.
4. **Human review.** For each PR: ran the tests myself, checked the diff for scope, read logic-heavy parts, and fact-checked the agent's summary.

## 4. Results

**What the agent did well**
- Much more readable code: descriptive names, docstrings, one statement per line
- Stayed within the allowed file each time
- Searched the repo itself to find names other files depend on
- Worked in verified steps on the engine (5 steps, tests green after each)
- Flagged real issues in the original code without "fixing" them:
  - inconsistent volatility calculation (`ddof=1` vs. `ddof=0`)
  - dead constants (`GROUPS`, `EXPOSURE`) and a stale module docstring

**What it got wrong or needed checking**
| Finding | Impact |
| --- | --- |
| Claimed US markets have a midday break (they don't) | Code correct, explanation wrong: a confident domain error |
| Renamed module-level function `z` → `zscore` despite instructions | Safe here (no other file uses it), but a rule deviation |
| Renamed parameter `r` in two functions while its summary claimed signatures were kept | Safe here (called positionally), but the summary was inaccurate |
| Restructured variables in `evaluate`, beyond pure renaming | Covered by the self-test; needed manual review |
| Wrote files with LF line endings on Windows | Inflated the diff; harmless |

**Outcome:** both refactors merged, 6/6 tests green locally and in CI, zero behavior change detected.

## 5. Effort

| | Agent-assisted | Manual estimate |
| --- | --- | --- |
| Per file (prompt to merged PR) | ~30 min | 1–2 h for an experienced developer; several hours for me |

Cost: used the agent's free local tier; no paid plan.

## 6. Limitations

- **Coverage gaps:** functions that fetch Alpaca data (`fetch_iex`, `load_alpaca_headers`) and the refresher file have no tests. Changes there are unverified.
- **Partial codebase:** only the offline strategy was extracted. The live-trading and submission code was out of scope.
- **Small sample:** two files, ~660 lines. Results may differ on a large enterprise codebase.
- **Same person as customer and reviewer:** no independent review.

## 7. Recommendation and next steps

Continue, with these changes:
- Write tests for the uncovered data functions before refactoring further
- Make naming rules even more explicit: "do not rename any function or parameter, only local variables"
- Always verify the agent's summary against the diff instead of trusting it
- Next candidates: refactor the engine's self-test in its own PR, then the refresher file