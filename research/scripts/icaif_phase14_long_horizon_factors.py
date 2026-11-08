from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOLS = (
    "AAPL","MSFT","NVDA","INTC","CRM","JPM","BAC","GS","V","PYPL",
    "LLY","JNJ","UNH","PFE","TMO","AMZN","TSLA","WMT","NKE","KO",
    "CAT","GE","BA","XOM","CVX","GOOGL","META","DIS","T","NEE",
)
STANDARD_TIMES = (
    "09:30:00","10:00:00","11:00:00","12:00:00",
    "13:00:00","14:00:00","15:00:00",
)
FEE = 0.001
GROSS = 0.80
SHARPE_ANNUALIZER = 42.0


@dataclass(frozen=True)
class Spec:
    name: str
    score: str
    tilt_l1: float
    mode: str
    no_trade_l1: float


SPECS = (
    Spec("equal30_80_hold", "equal", 0.0, "hold", math.inf),
    Spec("mom60_hold_t08", "mom60", 0.08, "hold", math.inf),
    Spec("sector60_hold_t08", "sector60", 0.08, "hold", math.inf),
    Spec("ensemble_hold_t08", "ensemble", 0.08, "hold", math.inf),
    Spec("ensemble_hold_t12", "ensemble", 0.12, "hold", math.inf),
    Spec("ensemble_daily_t08_nt04", "ensemble", 0.08, "daily", 0.04),
    Spec("ensemble_daily_t12_nt05", "ensemble", 0.12, "daily", 0.05),
)


def root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "universe.json").exists() and (p / "data" / "official").is_dir():
            return p.resolve()
    raise SystemExit("ERROR: run this from starter-kit.")


def rank_center(df: pd.DataFrame) -> pd.DataFrame:
    return (df.rank(axis=1, pct=True, method="average") - 0.5) * 2.0


def target(score: np.ndarray, tilt: float) -> np.ndarray:
    base = np.full(30, GROSS / 30.0)
    if tilt <= 0:
        return base
    centered = score - float(np.mean(score))
    denom = float(np.abs(centered).sum())
    if denom <= 1e-15:
        return base
    delta = tilt * centered / denom
    w = base + delta
    if w.min() < 0:
        neg = delta < 0
        scale = min(1.0, float(np.min(base[neg] / (-delta[neg]))))
        w = base + delta * max(0.0, scale * 0.999999)
    w = np.maximum(w, 0)
    w *= GROSS / float(w.sum())
    if w.max() > 0.30 + 1e-12:
        raise ValueError("cap exceeded")
    return w


def solve_trade(nav, values, prices, w):
    post = float(nav)
    for _ in range(50):
        tv = w * post
        notional = float(np.abs(tv - values).sum())
        nxt = nav - FEE * notional
        if abs(nxt - post) <= 1e-10 * max(1.0, nav):
            post = nxt
            break
        post = nxt
    tv = w * post
    notional = float(np.abs(tv - values).sum())
    fee = FEE * notional
    shares = tv / prices
    cash = post - float(tv.sum())
    return shares, cash, notional, fee


def load_data(path: Path):
    x = pd.read_parquet(path)
    x["timestamp_et"] = pd.to_datetime(x["timestamp_et"])
    x["trading_date"] = pd.to_datetime(x["trading_date"]).dt.normalize()
    x["time"] = x["timestamp_et"].dt.strftime("%H:%M:%S")
    x = x[x["ticker"].isin(SYMBOLS) & x["time"].isin(STANDARD_TIMES)].copy()

    counts = (
        x.groupby(["trading_date","ticker"])["time"].nunique()
        .unstack("ticker").reindex(columns=SYMBOLS)
    )
    dates = counts.index[(counts == len(STANDARD_TIMES)).all(axis=1)]
    x = x[x["trading_date"].isin(dates)]

    def piv(t, field):
        z = (
            x[x["time"] == t]
            .pivot(index="trading_date", columns="ticker", values=field)
            .reindex(index=dates, columns=SYMBOLS)
            .astype(float)
        )
        if z.isna().any().any():
            raise ValueError(f"missing {t}/{field}")
        return z

    open0930 = piv("09:30:00", "open")
    endpoints = [piv(t, "adj_close") for t in STANDARD_TIMES]
    close = endpoints[-1]

    sector = (
        x[["ticker","sector_group"]].drop_duplicates()
        .groupby("ticker")["sector_group"].first()
        .reindex(SYMBOLS)
    )
    return pd.DatetimeIndex(dates), open0930, endpoints, close, sector


def build_scores(close: pd.DataFrame, sector: pd.Series):
    r1 = close.pct_change()
    mom60 = close.pct_change(60)
    mom120 = close.pct_change(120)
    vol60 = r1.rolling(60).std()
    pos60 = (r1 > 0).rolling(60).mean()

    sec60 = mom60.copy()
    sec120 = mom120.copy()
    for s in sector.dropna().unique():
        cols = sector.index[sector == s].tolist()
        sec60.loc[:, cols] = mom60.loc[:, cols].sub(mom60.loc[:, cols].mean(axis=1), axis=0)
        sec120.loc[:, cols] = mom120.loc[:, cols].sub(mom120.loc[:, cols].mean(axis=1), axis=0)

    ranks = {
        "mom60": rank_center(mom60),
        "sector60": rank_center(sec60),
    }

    ensemble = (
        rank_center(mom60)
        + rank_center(mom120)
        + rank_center(sec60)
        + rank_center(sec120)
        + rank_center(pos60)
        + rank_center(-vol60)
    ) / 6.0

    ranks["ensemble"] = ensemble
    return ranks


def sharpe(nav):
    rets = nav[1:] / nav[:-1] - 1
    if len(rets) < 2:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return 0.0
    return float(np.mean(rets) / sd * SHARPE_ANNUALIZER)


def mdd(nav):
    peak = np.maximum.accumulate(nav)
    return float(np.max(1 - nav / peak))


def simulate(start, length, spec, opens, endpoints, scores):
    shares = np.zeros(30)
    cash = 1_000_000.0
    init = cash
    nav_path = [init]
    turnover = 0.0
    fees = 0.0
    rebalances = 0

    for off in range(length):
        d = start + off
        px = opens[d]
        values = shares * px
        pre = cash + float(values.sum())

        if spec.score == "equal":
            w = np.full(30, GROSS / 30.0)
        else:
            signal_idx = d - 1
            s = scores[spec.score][signal_idx]
            if not np.isfinite(s).all():
                raise ValueError("missing signal")
            w = target(s, spec.tilt_l1)

        should = off == 0 or spec.mode == "daily"
        if should:
            current = values / pre if pre > 0 else np.zeros(30)
            l1 = float(np.abs(w - current).sum())
            if off == 0 or l1 >= spec.no_trade_l1:
                shares, cash, notional, fee = solve_trade(pre, values, px, w)
                turnover += notional / pre
                fees += fee
                rebalances += 1

        for ep in endpoints:
            nav_path.append(cash + float(np.sum(shares * ep[d])))

    arr = np.asarray(nav_path, float)
    return {
        "return": float(arr[-1] / init - 1),
        "sharpe_proxy": sharpe(arr),
        "mdd_proxy": mdd(arr),
        "turnover_proxy": turnover / (length * 7),
        "fees": fees,
        "rebalances": rebalances,
    }


def add_ranks(rows):
    f = pd.DataFrame(rows)
    f["rr"] = f["return"].rank(ascending=False, method="average")
    f["rs"] = f["sharpe_proxy"].rank(ascending=False, method="average")
    f["rm"] = f["mdd_proxy"].rank(ascending=True, method="average")
    f["rt"] = f["turnover_proxy"].rank(ascending=True, method="average")
    f["rank_proxy"] = f[["rr","rs","rm","rt"]].mean(axis=1)
    mp = f.set_index("strategy")["rank_proxy"].to_dict()
    for r in rows:
        r["rank_proxy"] = float(mp[r["strategy"]])


def evaluate(dates, opens, endpoints, scores, length, step):
    valid = np.isfinite(scores["ensemble"]).all(axis=1)
    starts = [
        s for s in range(121, len(dates)-length+1, step)
        if valid[s-1:s+length-1].all()
    ]
    rows = []
    for wid, s in enumerate(starts, 1):
        cur = []
        for spec in SPECS:
            z = simulate(
                s, length, spec,
                opens, endpoints, scores
            )
            cur.append({
                "window_id": wid,
                "start_date": str(dates[s].date()),
                "end_date": str(dates[s+length-1].date()),
                "strategy": spec.name,
                **z,
            })
        add_ranks(cur)
        rows.extend(cur)
    return pd.DataFrame(rows)


def summary(df):
    base = df[df.strategy == "equal30_80_hold"].set_index("window_id")
    rows = []
    for name, g in df.groupby("strategy", sort=False):
        a = g.set_index("window_id")
        idx = a.index.intersection(base.index)
        rows.append({
            "strategy": name,
            "windows": len(g),
            "mean_return": g["return"].mean(),
            "median_return": g["return"].median(),
            "positive_fraction": (g["return"] > 0).mean(),
            "mean_sharpe_proxy": g["sharpe_proxy"].mean(),
            "mean_mdd_proxy": g["mdd_proxy"].mean(),
            "worst_mdd_proxy": g["mdd_proxy"].max(),
            "mean_turnover_proxy": g["turnover_proxy"].mean(),
            "mean_rebalances": g["rebalances"].mean(),
            "median_rank_proxy": g["rank_proxy"].median(),
            "better_rank_vs_equal80": float(
                (a.loc[idx,"rank_proxy"] < base.loc[idx,"rank_proxy"]).mean()
            ),
            "return_delta": float((a.loc[idx,"return"] - base.loc[idx,"return"]).mean()),
            "sharpe_delta": float((a.loc[idx,"sharpe_proxy"] - base.loc[idx,"sharpe_proxy"]).mean()),
            "mdd_delta": float((a.loc[idx,"mdd_proxy"] - base.loc[idx,"mdd_proxy"]).mean()),
        })
    return pd.DataFrame(rows).sort_values(["median_rank_proxy","strategy"])


def self_check():
    s = np.linspace(-1,1,30)
    for tilt in (0,.08,.12):
        w = target(s, tilt)
        assert len(w) == 30
        assert (w >= 0).all()
        assert abs(float(w.sum()) - .8) < 1e-10
        assert w.max() <= .30 + 1e-12
    print("PHASE14_SELF_CHECK_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return

    rt = root()
    p = rt / "data" / "official" / "hourly_market_data_2021_2026.parquet"
    art = rt / "artifacts"
    art.mkdir(exist_ok=True)

    print("=== PHASE 14: LONG-HORIZON FACTOR ENSEMBLE ===")
    print("Competition submission: DISABLED")
    print("Family: 60/120d momentum + sector-relative momentum + trend consistency + low-vol")
    print("No fitted model; fixed economic-prior ensemble")
    print()

    dates, op, eps, close, sector = load_data(p)
    scores_df = build_scores(close, sector)
    scores = {k:v.to_numpy(float) for k,v in scores_df.items()}
    opens = op.to_numpy(float)
    endpoints = [x.to_numpy(float) for x in eps]

    two = evaluate(dates, opens, endpoints, scores, 2, 1)
    fifteen = evaluate(dates, opens, endpoints, scores, 15, 5)
    s2 = summary(two)
    s15 = summary(fifteen)

    cols = [
        "strategy","windows","mean_return","median_return","positive_fraction",
        "mean_sharpe_proxy","mean_mdd_proxy","worst_mdd_proxy",
        "mean_turnover_proxy","mean_rebalances","median_rank_proxy",
        "better_rank_vs_equal80","return_delta","sharpe_delta","mdd_delta",
    ]

    print("=== 2-DAY WINDOWS ===")
    print(s2[cols].to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print()
    print("=== 15-DAY WINDOWS ===")
    print(s15[cols].to_string(index=False, float_format=lambda x:f"{x:.4f}"))

    # Independent calendar-year robustness, especially 2025 holdout.
    fifteen2 = fifteen.copy()
    fifteen2["year"] = pd.to_datetime(fifteen2["start_date"]).dt.year
    year_rows = []
    for (year, strat), g in fifteen2.groupby(["year","strategy"]):
        year_rows.append({
            "year": int(year),
            "strategy": strat,
            "windows": len(g),
            "mean_return": g["return"].mean(),
            "mean_sharpe": g["sharpe_proxy"].mean(),
            "mean_mdd": g["mdd_proxy"].mean(),
        })
    years = pd.DataFrame(year_rows)
    print()
    print("=== 15-DAY YEAR ROBUSTNESS ===")
    print(years.to_string(index=False, float_format=lambda x:f"{x:.4f}"))

    merged = s2.set_index("strategy").join(
        s15.set_index("strategy"), lsuffix="_2d", rsuffix="_15d"
    )
    candidates = [x.name for x in SPECS if x.name != "equal30_80_hold"]
    promotable = []
    for name in candidates:
        r = merged.loc[name]
        y25 = years[(years.year == 2025) & (years.strategy == name)]
        b25 = years[(years.year == 2025) & (years.strategy == "equal30_80_hold")]
        if y25.empty or b25.empty:
            continue
        holdout_return_delta = float(y25.iloc[0].mean_return - b25.iloc[0].mean_return)
        holdout_sharpe_delta = float(y25.iloc[0].mean_sharpe - b25.iloc[0].mean_sharpe)
        if (
            r["better_rank_vs_equal80_15d"] >= .55
            and r["return_delta_15d"] >= 0
            and r["sharpe_delta_15d"] >= 0
            and r["mdd_delta_15d"] <= .002
            and r["mean_turnover_proxy_15d"] <= .02
            and holdout_return_delta >= 0
            and holdout_sharpe_delta >= 0
        ):
            promotable.append(name)

    print()
    print("=== KEEP / REJECT INTERPRETATION ===")
    if promotable:
        print("PROMOTION CANDIDATE(S): " + ", ".join(promotable))
        print("Do not deploy automatically; integrate and shadow-test first.")
    else:
        print("NO PROMOTION: no long-horizon factor variant clears the predefined full-sample + 2025 holdout gate.")
        print("Keep equal30_80_hold for Validation.")

    two.to_csv(art/"phase14_factor_2d_windows.csv", index=False)
    fifteen.to_csv(art/"phase14_factor_15d_windows.csv", index=False)
    years.to_csv(art/"phase14_factor_years.csv", index=False)
    (art/"phase14_factor_summary.json").write_text(
        json.dumps({
            "two_day": s2.to_dict(orient="records"),
            "fifteen_day": s15.to_dict(orient="records"),
            "year_robustness": years.to_dict(orient="records"),
            "promotable": promotable,
        }, indent=2),
        encoding="utf-8",
    )
    print()
    print("PHASE 14 COMPLETE. NO COMPETITION SUBMISSION WAS MADE.")


if __name__ == "__main__":
    main()
