# POC Report: Agent-Assisted Refactor of the ICAIF Trading Agent

## 1. Summary

An AI coding agent (Devin's local agent in the Devin/Windsurf IDE) refactored the full codebase of my ICAIF 2026 trading agent for readability: first the two strategy files, then the complete live decision and execution layer that traded during the competition. Behavior stayed identical, verified by a test harness that grew from 6 to 16 tests, CI and static analysis. The agent saved substantial time, but review caught several issues that tests alone would not have: a reordered safety check, a lowered naming standard, and inaccurate summaries. Recommendation: continue, with stricter prompts for safety-critical code and the follow-ups listed in section 7.

## 2. Scope and success criteria

| Phase | Scope | Size |
| --- | --- | --- |
| 1 (Nov 6) | Strategy: `market_support.py`, `sleeve_engine.py` | ~660 lines |
| 2 (Nov 8) | Live layer and tools: `state_machine.py`, `organizer_bridge.py`, `market_data_fetcher.py`, `history_refresher.py`, `round_gatekeeper.py`, `orchestrator.py`, `readonly_rehearsal.py`, `account_audit.py`, `catchup_preflight.py`, `tools/generate_policy.py` | ~2,600 lines |

**Off-limits:** the organizer's starter kit (`agent/kit/`), the deployed systemd files, the tests and golden outputs, and each module's built-in self-test.

**Success meant:**
- Clearly more readable code (one statement per line, descriptive names, docstrings)
- No behavior change: all tests pass, locally, in Docker and in CI
- Reason codes, log messages, signatures and statement order preserved
- Each change confined to its declared files

## 3. Approach

1. **Harness first, then extend it before riskier code.** Phase 1 started with a golden-output test and trade-rule unit tests. Before phase 2, the harness was extended with every module's built-in self-test, the live layer's original offline tests, and a second golden test for review-day mode: 6 → 16 tests, all in CI.
2. **Preserve the original.** The deployed code was tagged `competition-baseline` before any change.
3. **Structure before readability.** Files were moved to role-based names in a separate PR (`git mv`, then the agent updated imports), so renames and code edits were never mixed.
4. **One scoped prompt per file group,** from lower to higher risk, each with protected zones (self-tests, command-line interfaces, server paths), explicit must-not-change rules, and a definition of done. Prompts were refined after every review.
5. **Review beyond green tests:** scope check, an automated comparison of all reason codes before and after, byte-for-byte comparison of protected functions, a call-order fingerprint of the orchestrator's cycle, and a line-by-line read of safety-critical functions.

## 4. Results

**Outcome:** 12 files refactored across 9 pull requests; 16/16 tests green locally, in Docker and in CI; no reason code, log message, signature or protected function changed; the orchestrator's exactly-once call order verified mechanically.

**What the agent did well**
- Readable code with docstrings that state inputs, outputs and the errors raised
- Respected protected zones (self-tests, interfaces) in every PR
- Found names other files depend on by searching the repo itself
- Restored exact source text when a test checked it, instead of changing the test
- Flagged its own risky change (a reordered check) and pre-existing issues without "fixing" them
- Correctly kept a submission call whose result was unused, rather than deleting it as dead code

**What review caught**

| Finding | Impact | Resolution |
| --- | --- | --- |
| Reordered a short-circuit check in `authenticated_round`, claiming identical behavior | Could raise a different error on edge-case inputs; tests did not catch it | Sent back; original evaluation order restored |
| Kept short variable names "to minimize churn" despite the prompt | Readability goal quietly lowered | Sent back; names fixed |
| Summaries that contradicted themselves or were inaccurate (e.g. "no source-text tests exist", checking the wrong folder) | Misplaced trust if read alone | Verified every claim against the diff |
| Ran tests without the project's virtual environment and asked to install packages globally | Environment drift outside the repo | Environment line added to all later prompts |
| Stopped mid-task on long runs | Incomplete work | Resumed with smaller steps and targeted test runs |
| Phase 1: renamed names it was told to keep; a confident domain error (US markets have no midday break) | Safe here, but rule deviations | Caught in review |

**Issues found in the original code** (documented; behavior-changing fixes deliberately kept out of readability PRs)

| Finding | Status |
| --- | --- |
| A self-test outdated by a later live fix (V3.9.3 decimal handling), never run automatically | Fixed in the harness PR; now runs in CI |
| Two production tests that check source text rather than behavior (brittle) | Kept; candidates for replacement |
| `assert` used for input validation (skipped under `python -O`) | Documented |
| A temporary file can be left behind if a CSV write fails | Documented |
| Module-name collisions when pytest imported the production test scripts | Fixed: tests limited to `tests/`, scripts run in separate processes |
| Unused imports and values | 15 removed, verified by search and pyflakes; three unused assignments kept on purpose (one validates input, one is the submission call) |
| Computed-but-unused signals in the strategy that also act as input validation | Kept |
| Hard-coded server paths and a folder-name guard tied to the old server layout | Deferred to a configuration change |

## 5. Effort

| | Agent-assisted (actual) |
| --- | --- |
| Phase 1: 2 strategy files | ~30 min per file |
| Phase 2: harness, restructure, research reorganization, 6 refactor PRs, cleanup | ~4.5 h in one afternoon |

Cost: the agent's free local tier. Most human time went into review, not typing.

## 6. Limitations

- **No live run is possible:** the competition platform and credentials no longer exist, so the live layer is verified offline only.
- **Coverage gaps:** some live code paths (e.g. the live-arming check, which is Linux-only) are not exercised by any test; for those, removed imports were verified by static analysis instead.
- **The refactored layout is not yet deployable** as is: server paths still describe the competition server (see `deployment/example/`).
- **Same person as customer and reviewer:** no independent review.

## 7. Recommendation and next steps

Continue. For safety-critical code, keep the rules that worked: protected zones, explicit evaluation-order and reason-code rules, environment instructions, and mechanical before/after comparisons. Next:
1. Move server paths into configuration, so the refactored layout can be deployed
2. Replace the source-text assertions with behavioral tests
3. Run pyflakes as a CI gate
4. Refactor the engine's self-test in its own PR, verified with mutation testing
5. Separate strategy math from operational state in the engine
6. Fix the `assert` and temp-file findings in small behavior-changing PRs, each with a new test