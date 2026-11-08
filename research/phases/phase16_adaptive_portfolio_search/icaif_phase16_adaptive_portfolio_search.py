#!/usr/bin/env python3
"""ICAIF Phase 16: OFFLINE multi-round portfolio-policy search.

This is research-only. NO credentials, network access or organizer submissions.
Run in starter-kit with phase15 v2 in Downloads, pandas/numpy/pyarrow installed.

IMPORTANT: intraday R2-R7 execution uses hourly bar PROXIES, not exact
10:30/11:30/... execution prices. Candidates need 15-minute replay before live use.
The final 2026 set is examined only with --final-test, ONCE after candidates
and scoring rules are frozen. Avoid repeated holdout-driven tweaking.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_NAV = 1_000_000.0
FEE = 0.001
DAYS = 15
ROUNDS = 7
SYMBOLS = 30


@dataclass(frozen=True)
class Policy:
    signal: str
    gross: float
    top_k: int
    cadence: str
    gate: float
    regime: str

    @property
    def name(self):
        return (f"{self.signal}_g{int(self.gross*100):03d}_k{self.top_k:02d}_"
                f"{self.cadence}_nt{int(self.gate*1000):03d}_{self.regime}")


def policies(max_candidates=180):
    # Deterministic predeclared 1008-cell space; cap samples it reproducibly.
    full = [Policy(*v) for v in itertools.product(
        ("mom20", "mom60", "mom120", "rev5", "lowvol", "sector60", "ensemble"),
        (0.6, 0.8, 1.0),
        (5, 10, 20, 30),
        ("hold", "daily", "stage3"),
        (0.02, 0.05),
        ("fixed", "regime"),
    )]
    # Stratified sampling over cadence/signal; never sample by backtest results.
    by_group = {}
    for p in full:
        by_group.setdefault((p.signal, p.cadence), []).append(p)
    picked = []
    for group in sorted(by_group):
        arr = sorted(by_group[group], key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
        by_group[group] = arr
    while len(picked) < max_candidates and any(by_group.values()):
        for group in sorted(by_group):
            if len(picked) >= max_candidates:
                break
            if by_group[group]:
                picked.append(by_group[group].pop(0))
    # Always retain the live proven baseline regardless of the sample size.
    baseline = Policy("equal", 0.8, 30, "hold", 0.02, "fixed")
    return [baseline] + [p for p in picked if p != baseline]


def crossrank(a):
    a = np.asarray(a, float)
    n = len(a)
    return np.argsort(np.argsort(a, kind="stable"), kind="stable") / max(1, n - 1)


def build_signals(phase, daily, sectors):
    features, _, s = phase.make_features(daily, sectors)
    ret = s["ret"]
    vol = s["vol60"]
    scores = {
        "mom20": ret[20],
        "mom60": ret[60],
        "mom120": ret[120],
        "rev5": -ret[5],
        "lowvol": -vol,
        "sector60": s["sector60"],
    }
    ens = np.full_like(ret[60], np.nan)
    for d in range(121, len(daily)):
        arrays = [scores[name][d] for name in ("mom20", "mom60", "mom120", "lowvol", "sector60")]
        if all(np.isfinite(a).all() for a in arrays):
            ens[d] = np.mean([crossrank(a) for a in arrays], axis=0)
    scores["ensemble"] = ens
    scores["equal"] = np.ones_like(ret[60])
    # Avoid empty-slice runtime warnings during the warmup period.
    valid = np.isfinite(ret[20])
    n_valid = valid.sum(axis=1)
    market = np.divide(
        np.where(valid, ret[20], 0.).sum(axis=1),
        n_valid,
        out=np.full(len(daily), np.nan),
        where=n_valid > 0,
    )
    scores["market20"] = np.broadcast_to(market[:, None], daily.shape)
    return scores


def target_weights(policy, signal_row, market20):
    if not np.isfinite(signal_row).all() or not math.isfinite(market20):
        raise ValueError("Stale/nonfinite signal")
    gross = policy.gross * (0.5 if policy.regime == "regime" and market20 < 0 else 1.0)
    w = np.zeros(SYMBOLS, dtype=float)
    selected = np.argsort(signal_row, kind="stable")[-policy.top_k:]
    # Smooth rank weights among selected. Concentration is parameterized.
    if policy.signal == "equal":
        w[:] = gross / SYMBOLS
    else:
        chosen = crossrank(signal_row[selected]) + .5
        w[selected] = gross * chosen / chosen.sum()
    if w.sum() > 1.00000001 or w.min() < -1e-12 or w.max() > .30000001:
        raise ValueError("Portfolio constraints violated")
    return w


def execute(nav, cash, shares, price, target):
    assert (price > 0).all()
    current_value = shares * price
    post_nav = nav
    for _ in range(60):
        new_values = target * post_nav
        notional = float(np.abs(new_values - current_value).sum())
        revised = nav - FEE * notional
        if abs(revised - post_nav) <= 1e-9:
            post_nav = revised
            break
        post_nav = revised
    new_values = target * post_nav
    notional = float(np.abs(new_values - current_value).sum())
    new_cash = post_nav - float(new_values.sum())
    return new_cash, new_values / price, notional


def evaluate_window(policy, start, opens, ends, scores):
    shares = np.zeros(SYMBOLS)
    cash = INITIAL_NAV
    nav_path = [INITIAL_NAV]
    turnover_normalized = 0.
    fees = 0.
    trades = 0
    for day_offset in range(DAYS):
        day = start + day_offset
        signal = scores[policy.signal][day - 1]
        mkt = scores["market20"][day - 1, 0]
        dest = target_weights(policy, signal, mkt)
        for rnd in range(ROUNDS):
            # Dataset: R1 observed 09:30 open; other periods use available
            # hourly close proxies. These are NOT official 10:30..15:30 fills.
            price = opens[day] if rnd == 0 else ends[day, rnd]
            nav = cash + float(np.dot(shares, price))
            if policy.cadence == "hold":
                do_rebalance = (day_offset == 0 and rnd == 0)
                wanted = dest
            elif policy.cadence == "daily":
                do_rebalance = (rnd == 0)
                wanted = dest
            else:  # stage3: 25% / 50% / 100% target on first day
                do_rebalance = (day_offset == 0 and rnd in (0, 1, 2))
                wanted = dest * ((.25, .50, 1.0)[rnd] if (day_offset == 0 and rnd in (0, 1, 2)) else 1.0)
            if do_rebalance:
                current_weights = shares * price / max(nav, 1.0)
                drift = float(np.abs(current_weights - wanted).sum())
                # Never suppress opening tranche, only later voluntary trades.
                mandatory = (day_offset == 0 and (rnd == 0 or policy.cadence == "stage3"))
                if mandatory or drift >= policy.gate:
                    pre_nav = nav
                    cash, shares, notional = execute(nav, cash, shares, price, wanted)
                    turnover_normalized += notional / max(pre_nav, 1.0)
                    fees += FEE * notional
                    trades += 1
            # Separate valuation from trade execution; approximate same proxy.
            new_nav = cash + float(np.dot(shares, ends[day, rnd]))
            nav_path.append(new_nav)
    nav = np.asarray(nav_path)
    rets = nav[1:] / nav[:-1] - 1.
    sd = float(np.std(rets, ddof=1))
    return {
        "return": float(nav[-1] / INITIAL_NAV - 1),
        "sharpe": float(rets.mean() * 42. / sd) if sd > 1e-12 else 0.0,
        "mdd": float(np.max(1. - nav / np.maximum.accumulate(nav))),
        "turnover": turnover_normalized / (DAYS * ROUNDS),
        "fees": fees,
        "trade_count": trades,
    }


def _source(path):
    spec = importlib.util.spec_from_file_location("icaif_phase15_v2_local", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def historical_data(root, module_path):
    m = _source(module_path)
    dates, opens, ends, daily, sectors = m.load_history(m.source_path(root))
    scores = build_signals(m, daily, sectors)
    return dates, opens, ends, scores


def windows_for_year(dates, year, step=15):
    idx = np.flatnonzero(dates.year == year)
    return [int(start) for start in idx[::step] if
            start >= 125 and start + DAYS <= len(dates)
            and dates[start + DAYS - 1].year == year]


def score_policy_summary(results):
    # Mean 15-day returns and drawdowns; not a forecast of actual ICAIF ranks.
    out = pd.DataFrame(results)
    return out.groupby(["year", "policy"], as_index=False)[
        ["return", "sharpe", "mdd", "turnover", "fees", "trade_count"]
    ].mean()


def screening_ranks(summary):
    years = [2023, 2024]
    subset = summary.loc[summary.year.isin(years)].copy()
    count = subset.groupby("policy").year.nunique()
    subset = subset[subset.policy.isin(count[count == 2].index)]
    for m, asc in (("return", False), ("sharpe", False), ("mdd", True), ("turnover", True)):
        subset["rank_" + m] = subset.groupby("year")[m].rank(ascending=asc, method="average", pct=True)
    rankcols = ["rank_" + k for k in ("return", "sharpe", "mdd", "turnover")]
    subset["mean_rank"] = subset[rankcols].mean(axis=1)
    ranks = subset.groupby("policy")["mean_rank"].agg(["mean", "max"]).reset_index()
    # Penalize a bad year: a smaller mean rank and smaller worst-year rank.
    ranks["conservative_score"] = .5 * ranks["mean"] + .5 * ranks["max"]
    return ranks.sort_values("conservative_score")


def self_test():
    import types
    n = 170
    op = np.full((n, SYMBOLS), 100.)
    ep = np.full((n, ROUNDS, SYMBOLS), 100.)
    scores = {name: np.tile(np.arange(1, SYMBOLS + 1, dtype=float), (n,1))
              for name in ("mom20", "mom60", "mom120", "rev5", "lowvol", "sector60", "ensemble", "equal")}
    scores["market20"] = np.ones((n,SYMBOLS))
    baseline = Policy("equal", .8, 30, "hold", .02, "fixed")
    r = evaluate_window(baseline, 140, op, ep, scores)
    assert abs(r["fees"] - 799.36051159) < .01, r
    assert r["trade_count"] == 1
    stage = Policy("equal", .8, 30, "stage3", .02, "fixed")
    s = evaluate_window(stage, 140, op, ep, scores)
    assert s["trade_count"] == 3 and abs(s["fees"]-r["fees"]) < .5, s
    assert r["turnover"] > 0 and s["turnover"] > 0
    cap = target_weights(Policy("mom60", 1., 5, "hold", .02, "fixed"), scores["mom60"][140], .05)
    assert len(cap) == 30 and cap.max() <= .3 and abs(cap.sum()-1)<1e-12
    risk = target_weights(Policy("mom60", .8, 5, "hold", .02, "regime"), scores["mom60"][140], -.01)
    assert abs(risk.sum()-.4)<1e-12
    assert len(policies(60)) > 50
    print("PHASE16_SELF_TEST_PASS")
    print("NO_ORGANIZER_API_NO_SUBMISSION")
    print("EQUAL80_INIT_FEE", round(r["fees"], 2))
    print("STAGED3_TRADES", s["trade_count"])
    print("CONCENTRATED_MAX_WEIGHT",round(cap.max(),4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--final-test", action="store_true", help="Evaluate 2026 once after policy shortlist is frozen.")
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase15", type=Path, default=Path.home()/"Downloads"/"icaif_phase15_official_challengers_v2.py")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_candidates < 21 or args.max_candidates > 1764:
        raise SystemExit("--max-candidates must be between 21 and 1764")
    if args.step < DAYS:
        raise SystemExit("--step must be at least 15; avoid overlapping windows")
    if not args.phase15.exists():
        raise SystemExit("Phase15 v2 missing: " + str(args.phase15))
    root = args.project_root.resolve()
    dates, opens, ends, scores = historical_data(root, args.phase15)
    variants = policies(args.max_candidates)
    years = (2026,) if args.final_test else (2023,2024)
    output_dir = root / "artifacts" / "phase16_adaptive"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.final_test:
        frozen = output_dir / "screen_selection.json"
        if not frozen.exists():
            raise SystemExit("ERROR: run screening first and freeze selection (screen_selection.json)")
        spec_records = json.loads(frozen.read_text(encoding="utf-8"))
        names = set(spec_records["shortlist"])
        variants = [p for p in variants if p.name in names]
        if len(variants) != len(names):
            raise SystemExit("ERROR: --max-candidates changed after screening; abort")
        result_path = output_dir / "phase16_2026_once.csv"
        if result_path.exists():
            raise SystemExit("ERROR: 2026 holdout has already been evaluated here. Do not rerun/tune.")
    else:
        result_path = output_dir / "phase16_development_2023_2024.csv"
    print("PHASE16_OFFLINE_RESEARCH_NO_SUBMISSION")
    print("CANDIDATE_POLICIES:", len(variants), "EVALUATION_YEARS:", years)
    print("INTRADAY_PRICE_WARNING: hourly bar proxies, NOT exact official round fills")
    rows = []
    for year in years:
        starts = windows_for_year(dates, year, args.step)
        print(f"YEAR {year}: {len(starts)} nonoverlapping 15-day windows", flush=True)
        if not starts:
            raise SystemExit(f"ERROR_NO_COMPLETE_WINDOWS_FOR_YEAR_{year}")
        for i, p in enumerate(variants):
            for start in starts:
                stats = evaluate_window(p, start, opens, ends, scores)
                rows.append({"year": year,"policy": p.name,
                             "window_start": str(dates[start].date()), **stats})
            if (i+1) % 40 == 0:
                print(f"  {i+1}/{len(variants)} policies checked", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(result_path,index=False)
    summary = score_policy_summary(rows)
    if args.final_test:
        summary.to_csv(output_dir / "phase16_2026_summary.csv",index=False)
        print("2026_HOLDOUT_RESULTS (do not tune using these)")
        print(summary.to_string(index=False))
    else:
        ranks = screening_ranks(summary)
        ranks.to_csv(output_dir / "phase16_development_rankings.csv",index=False)
        summary.to_csv(output_dir / "phase16_development_summary.csv",index=False)
        # Frozen shortlist: baseline plus nine strongest, picked without looking at 2026.
        choices = [Policy("equal", .8,30,"hold",.02,"fixed").name] + [x for x in ranks.policy if x != Policy("equal", .8,30,"hold",.02,"fixed").name][:9]
        output = {"selection_rule":"2023-24 mean and worst year four-metric ranks",
                  "max_candidates":args.max_candidates,"step":args.step,
                  "shortlist":choices,"test_year_unseen":2026,
                  "limitation":"hourly proxy; 15m replay mandatory before deployment"}
        (output_dir / "screen_selection.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
        print("DEVELOPMENT_TOP10 (2026 NOT evaluated)")
        print(ranks.head(10).to_string(index=False))
        print("SHORTLIST",choices)
    print("REPORT",result_path)
    print("PHASE16_COMPLETE_NO_SUBMISSION")

if __name__ == "__main__":
    main()
