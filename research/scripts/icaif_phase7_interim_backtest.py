from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, time, timezone
from pathlib import Path
from statistics import mean, median, stdev
from zoneinfo import ZoneInfo

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from our_agent.config import FEE_RATE, OFFICIAL_SYMBOLS, SECTORS
from our_agent.data.alpaca import AlpacaMarketDataClient
from our_agent.data.canonical import bars_payload_to_frame, regular_session_only

NY = ZoneInfo("America/New_York")

CACHE = ROOT / "data" / "cache" / "alpaca_iex_15m_400d.csv"
OUT_JSON = ROOT / "artifacts" / "phase7_interim_backtest.json"
OUT_CSV = ROOT / "artifacts" / "phase7_interim_backtest_summary.csv"

TIMEFRAME = "15Min"
FETCH_DAYS = 400
FEE = float(FEE_RATE)
INITIAL_NAV = 1_000_000.0

EXECUTION_TIMES = [
    time(9, 30),
    time(10, 30),
    time(11, 30),
    time(12, 30),
    time(13, 30),
    time(14, 30),
    time(15, 30),
]
DEADLINE_TIMES = [
    time(9, 10),
    time(10, 25),
    time(11, 25),
    time(12, 25),
    time(13, 25),
    time(14, 25),
    time(15, 25),
]

BARS_1H = 4
BARS_1D = 26
BARS_5D = 130
MIN_HISTORY = BARS_5D + 2

SYMBOL_TO_SECTOR = {
    symbol: sector
    for sector, symbols in SECTORS.items()
    for symbol in symbols
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    gross: float
    top_k: int | None
    no_trade_threshold: float


STRATEGIES = [
    StrategySpec("cash_hold", "cash", 0.0, None, 999.0),
    StrategySpec("equal30_80_nt10", "equal", 0.80, 30, 0.10),
    StrategySpec("invvol30_80_nt10", "invvol", 0.80, 30, 0.10),
    StrategySpec("mom5_top8_80_nt10", "mom5", 0.80, 8, 0.10),
    StrategySpec("rev1_top8_80_nt10", "rev1", 0.80, 8, 0.10),
    StrategySpec("combo_top8_80_nt10", "combo", 0.80, 8, 0.10),
    StrategySpec("combo_top8_80_nt25", "combo", 0.80, 8, 0.25),
    StrategySpec("combo_top8_60_nt25", "combo", 0.60, 8, 0.25),
    StrategySpec("combo_top10_80_nt20", "combo", 0.80, 10, 0.20),
]


def require_project() -> None:
    required = [
        ROOT / "universe.json",
        ROOT / "our_agent" / "data" / "alpaca.py",
        ROOT / ".env",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "ERROR: Run this from starter-kit. Missing: " + repr(missing)
        )


def fetch_or_load() -> pd.DataFrame:
    if CACHE.exists():
        print(f"Using cached data: {CACHE}")
        frame = pd.read_csv(CACHE)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    now = datetime.now(timezone.utc)
    end = now - timedelta(minutes=20)
    start = end - timedelta(days=FETCH_DAYS)

    print("Fetching Alpaca IEX history...")
    print(f"Window: {start.isoformat()} -> {end.isoformat()}")

    with AlpacaMarketDataClient(feed="iex") as client:
        payload = client.historical_bars(
            symbols=OFFICIAL_SYMBOLS,
            start=start.isoformat(),
            end=end.isoformat(),
            timeframe=TIMEFRAME,
            adjustment="all",
            limit=10_000,
        )

    frame = bars_payload_to_frame(payload)
    frame = regular_session_only(frame)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)
    print(f"Saved cache: {CACHE}")
    return frame


def build_market_views(frame: pd.DataFrame):
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    local = frame["timestamp"].dt.tz_convert(NY)
    frame["date_et"] = local.dt.date
    frame["time_et"] = local.dt.time

    close = frame.pivot(index="timestamp", columns="symbol", values="close")
    open_ = frame.pivot(index="timestamp", columns="symbol", values="open")

    close = close.reindex(columns=OFFICIAL_SYMBOLS).sort_index()
    open_ = open_.reindex(columns=OFFICIAL_SYMBOLS).sort_index()

    trading_dates = sorted(frame["date_et"].unique())

    close_rows = frame[frame["time_et"] == time(15, 45)].copy()
    daily_close = (
        close_rows.pivot(index="date_et", columns="symbol", values="close")
        .reindex(columns=OFFICIAL_SYMBOLS)
        .sort_index()
    )

    return close, open_, trading_dates, daily_close


def ts_et(d, t) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, t, tzinfo=NY)).tz_convert("UTC")


def rank_signal(values: pd.Series) -> pd.Series:
    return 2.0 * values.rank(method="average", pct=True) - 1.0


def features_at(close: pd.DataFrame, deadline_utc: pd.Timestamp) -> pd.DataFrame:
    # A 15-minute bar stamped at time T is considered complete at T+15m.
    eligible_start = deadline_utc - pd.Timedelta(minutes=15)
    hist = close.loc[close.index <= eligible_start]

    rows = []
    for symbol in OFFICIAL_SYMBOLS:
        s = hist[symbol].dropna()
        if len(s) < MIN_HISTORY:
            raise ValueError(
                f"{symbol}: only {len(s)} eligible bars; need {MIN_HISTORY}"
            )

        last = float(s.iloc[-1])
        ret_1h = last / float(s.iloc[-1 - BARS_1H]) - 1.0
        ret_1d = last / float(s.iloc[-1 - BARS_1D]) - 1.0
        ret_5d = last / float(s.iloc[-1 - BARS_5D]) - 1.0
        vol_1d = float(s.pct_change(fill_method=None).tail(BARS_1D).std(ddof=1))

        if not all(math.isfinite(x) for x in [ret_1h, ret_1d, ret_5d, vol_1d]):
            raise ValueError(f"{symbol}: non-finite feature")

        rows.append(
            {
                "symbol": symbol,
                "sector": SYMBOL_TO_SECTOR[symbol],
                "ret_1h": ret_1h,
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "vol_1d": vol_1d,
            }
        )

    f = pd.DataFrame(rows).set_index("symbol")
    f["sector_relative_5d"] = (
        f["ret_5d"] - f.groupby("sector")["ret_5d"].transform("mean")
    )
    f["mom_rank"] = rank_signal(f["ret_5d"])
    f["rev_rank"] = -rank_signal(f["ret_1h"])
    f["sector_rank"] = rank_signal(f["sector_relative_5d"])
    f["combo"] = (
        0.45 * f["mom_rank"]
        + 0.30 * f["rev_rank"]
        + 0.25 * f["sector_rank"]
    )
    return f


def target_from_features(spec: StrategySpec, f: pd.DataFrame) -> dict[str, float]:
    target = {s: 0.0 for s in OFFICIAL_SYMBOLS}

    if spec.kind == "cash":
        return target

    if spec.kind == "equal":
        each = spec.gross / len(OFFICIAL_SYMBOLS)
        return {s: each for s in OFFICIAL_SYMBOLS}

    if spec.kind == "invvol":
        inv = 1.0 / f["vol_1d"].clip(lower=1e-8)
        w = spec.gross * inv / inv.sum()
        if float(w.max()) > 0.30:
            # Conservative cap-and-redistribute loop.
            remaining = spec.gross
            free = set(OFFICIAL_SYMBOLS)
            result = {s: 0.0 for s in OFFICIAL_SYMBOLS}
            raw = inv.to_dict()
            while free:
                denom = sum(raw[s] for s in free)
                changed = False
                for s in list(free):
                    proposed = remaining * raw[s] / denom
                    if proposed > 0.30:
                        result[s] = 0.30
                        remaining -= 0.30
                        free.remove(s)
                        changed = True
                if not changed:
                    denom = sum(raw[s] for s in free)
                    for s in free:
                        result[s] = remaining * raw[s] / denom
                    break
            return result
        return {s: float(w.loc[s]) for s in OFFICIAL_SYMBOLS}

    score_col = {
        "mom5": "ret_5d",
        "rev1": "rev_rank",
        "combo": "combo",
    }[spec.kind]

    ranked = (
        f[score_col]
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    chosen = ranked[: int(spec.top_k)]
    each = spec.gross / int(spec.top_k)

    if each > 0.30 + 1e-12:
        raise ValueError(f"{spec.name}: per-name cap breach")

    for s in chosen:
        target[s] = each
    return target


def current_weights(
    shares: dict[str, float],
    prices: dict[str, float],
    nav: float,
) -> dict[str, float]:
    if nav <= 0:
        raise ValueError("NAV must be positive")
    return {
        s: shares[s] * prices[s] / nav
        for s in OFFICIAL_SYMBOLS
    }


def solve_fee_aware_rebalance(
    *,
    pre_nav: float,
    current_values: dict[str, float],
    target_weights: dict[str, float],
) -> tuple[float, float, dict[str, float]]:
    """Solve N_post = N_pre - fee * gross trades.

    This is a transparent local proxy for target-weight execution at a common
    execution price. It is not claimed to reproduce hidden backend details.
    """
    post_nav = pre_nav
    for _ in range(100):
        target_values = {
            s: target_weights[s] * post_nav
            for s in OFFICIAL_SYMBOLS
        }
        gross = sum(
            abs(target_values[s] - current_values[s])
            for s in OFFICIAL_SYMBOLS
        )
        new_post = pre_nav - FEE * gross
        if abs(new_post - post_nav) < 1e-8:
            post_nav = new_post
            break
        post_nav = new_post

    target_values = {
        s: target_weights[s] * post_nav
        for s in OFFICIAL_SYMBOLS
    }
    gross = sum(
        abs(target_values[s] - current_values[s])
        for s in OFFICIAL_SYMBOLS
    )
    fee = FEE * gross
    return post_nav, gross, target_values


def competition_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sigma = stdev(returns)
    if sigma == 0 or not math.isfinite(sigma):
        return 0.0
    return 42.0 * mean(returns) / sigma


def max_drawdown(path: list[float]) -> float:
    peak = path[0]
    worst = 0.0
    for x in path:
        peak = max(peak, x)
        worst = max(worst, (peak - x) / peak)
    return worst


def simulate_window(
    spec: StrategySpec,
    dates: list,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    daily_close: pd.DataFrame,
) -> dict:
    shares = {s: 0.0 for s in OFFICIAL_SYMBOLS}
    cash = INITIAL_NAV

    begin_navs: list[float] = []
    period_end_navs: list[float] = []
    turnover_ratios: list[float] = []
    valuation_path: list[float] = [INITIAL_NAV]

    prev_begin_idx = None

    for day_idx, d in enumerate(dates):
        for round_idx, (exec_t, deadline_t) in enumerate(
            zip(EXECUTION_TIMES, DEADLINE_TIMES),
            start=1,
        ):
            exec_ts = ts_et(d, exec_t)
            deadline_ts = ts_et(d, deadline_t)

            if exec_ts not in open_.index:
                raise ValueError(f"Missing execution bar at {exec_ts}")

            row = open_.loc[exec_ts]
            if row.isna().any():
                missing = row[row.isna()].index.tolist()
                raise ValueError(
                    f"Missing execution prices {d} R{round_idx}: {missing}"
                )

            prices = {s: float(row[s]) for s in OFFICIAL_SYMBOLS}
            current_values = {
                s: shares[s] * prices[s]
                for s in OFFICIAL_SYMBOLS
            }
            pre_nav = cash + sum(current_values.values())

            # Current pre-execution NAV is the end of the previous period.
            if prev_begin_idx is not None:
                period_end_navs.append(pre_nav)
                valuation_path.append(pre_nav)

            begin_navs.append(pre_nav)
            prev_begin_idx = len(begin_navs) - 1

            if spec.kind == "cash":
                turnover_ratios.append(0.0)
                continue

            f = features_at(close, deadline_ts)
            target = target_from_features(spec, f)

            cur_w = current_weights(shares, prices, pre_nav)
            change = sum(
                abs(target[s] - cur_w[s])
                for s in OFFICIAL_SYMBOLS
            )

            if change < spec.no_trade_threshold:
                turnover_ratios.append(0.0)
                continue

            post_nav, gross, target_values = solve_fee_aware_rebalance(
                pre_nav=pre_nav,
                current_values=current_values,
                target_weights=target,
            )

            turnover_ratios.append(gross / pre_nav)

            for s in OFFICIAL_SYMBOLS:
                shares[s] = target_values[s] / prices[s]

            cash = post_nav - sum(target_values.values())
            if cash < -1e-6:
                raise ValueError(
                    f"{spec.name}: negative cash after rebalance {cash}"
                )

        # Official daily-close valuation path.
        if d not in daily_close.index:
            raise ValueError(f"Missing daily close for {d}")

        close_row = daily_close.loc[d]
        if close_row.isna().any():
            missing = close_row[close_row.isna()].index.tolist()
            raise ValueError(f"Missing daily close prices {d}: {missing}")

        close_nav = cash + sum(
            shares[s] * float(close_row[s])
            for s in OFFICIAL_SYMBOLS
        )
        valuation_path.append(close_nav)

    final_nav = valuation_path[-1]

    # Replace the last R7 period endpoint with final close.
    period_end_navs.append(final_nav)

    if len(begin_navs) != len(period_end_navs):
        raise AssertionError(
            f"period mismatch {len(begin_navs)} != {len(period_end_navs)}"
        )

    returns = [
        e / b - 1.0
        for b, e in zip(begin_navs, period_end_navs)
    ]

    return {
        "strategy": spec.name,
        "days": len(dates),
        "periods": len(returns),
        "return": final_nav / INITIAL_NAV - 1.0,
        "sharpe": competition_sharpe(returns),
        "max_drawdown": max_drawdown(valuation_path),
        "turnover": mean(turnover_ratios),
        "positive_period_fraction": (
            sum(r > 0 for r in returns) / len(returns)
        ),
        "final_nav": final_nav,
    }


def rank_window(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank_return"] = ranked["return"].rank(
        ascending=False,
        method="average",
    )
    ranked["rank_sharpe"] = ranked["sharpe"].rank(
        ascending=False,
        method="average",
    )
    ranked["rank_mdd"] = ranked["max_drawdown"].rank(
        ascending=True,
        method="average",
    )
    ranked["rank_turnover"] = ranked["turnover"].rank(
        ascending=True,
        method="average",
    )
    ranked["rank_score"] = ranked[
        ["rank_return", "rank_sharpe", "rank_mdd", "rank_turnover"]
    ].mean(axis=1)
    return ranked


def main() -> None:
    require_project()

    print("=== ICAIF PHASE 7: INTERIM ALPACA PROXY BACKTEST ===")
    print("Competition submission: DISABLED")
    print("Purpose: strategy evidence while organizer history is pending")
    print("Feed: Alpaca IEX, adjusted 15-minute bars")
    print("Execution: local proxy at scheduled bar opens")
    print("Fees: 10 bp of gross buy+sell notional")
    print()

    frame = fetch_or_load()
    close, open_, trading_dates, daily_close = build_market_views(frame)

    # Keep only days with all seven execution opens and a 15:45 closing bar.
    good_dates = []
    for d in trading_dates:
        ok = True
        for exec_t in EXECUTION_TIMES:
            ts = ts_et(d, exec_t)
            if ts not in open_.index or open_.loc[ts].isna().any():
                ok = False
                break
        if d not in daily_close.index or daily_close.loc[d].isna().any():
            ok = False
        if ok:
            good_dates.append(d)

    if len(good_dates) < 80:
        raise SystemExit(
            f"ERROR: only {len(good_dates)} complete trading days available."
        )

    # Avoid early days without feature lookback.
    usable_dates = good_dates[10:]

    # Use the latest up to 180 complete trading days.
    usable_dates = usable_dates[-180:]

    # 15-day competition-length windows, stepped every 5 trading days.
    windows = []
    for start_i in range(0, len(usable_dates) - 15 + 1, 5):
        window_dates = usable_dates[start_i : start_i + 15]
        windows.append(window_dates)

    print(f"Complete trading days in research sample: {len(usable_dates)}")
    print(f"15-day walk-forward windows: {len(windows)}")
    print(
        f"Sample span: {usable_dates[0]} through {usable_dates[-1]}"
    )
    print()

    all_results = []

    for wi, dates in enumerate(windows, start=1):
        window_rows = []
        for spec in STRATEGIES:
            result = simulate_window(
                spec,
                dates,
                close,
                open_,
                daily_close,
            )
            result["window"] = wi
            result["start_date"] = str(dates[0])
            result["end_date"] = str(dates[-1])
            window_rows.append(result)

        ranked = rank_window(pd.DataFrame(window_rows))
        all_results.append(ranked)

        if wi % 5 == 0 or wi == len(windows):
            print(
                f"Completed window {wi}/{len(windows)} "
                f"({dates[0]} -> {dates[-1]})"
            )

    results = pd.concat(all_results, ignore_index=True)

    summary_rows = []
    for spec in STRATEGIES:
        x = results.loc[results["strategy"] == spec.name]
        summary_rows.append(
            {
                "strategy": spec.name,
                "windows": len(x),
                "median_rank_score": float(x["rank_score"].median()),
                "mean_rank_score": float(x["rank_score"].mean()),
                "mean_return": float(x["return"].mean()),
                "median_return": float(x["return"].median()),
                "positive_window_fraction": float((x["return"] > 0).mean()),
                "mean_sharpe": float(x["sharpe"].mean()),
                "mean_max_drawdown": float(x["max_drawdown"].mean()),
                "worst_max_drawdown": float(x["max_drawdown"].max()),
                "mean_turnover": float(x["turnover"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["median_rank_score", "mean_rank_score", "mean_return"],
        ascending=[True, True, False],
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifact_type": "INTERIM_RESEARCH_ONLY_NO_SUBMISSION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Alpaca free IEX",
        "timeframe": TIMEFRAME,
        "adjustment": "all",
        "sample_start": str(usable_dates[0]),
        "sample_end": str(usable_dates[-1]),
        "window_days": 15,
        "window_step_days": 5,
        "window_count": len(windows),
        "fee_rate": FEE,
        "important_limitations": [
            "Alpaca IEX is not the organizer execution feed.",
            "Execution is modeled at 15-minute bar opens at competition execution times.",
            "The fee-aware target solver is a transparent local proxy.",
            "Candidate comparison across several strategies creates multiple-testing risk.",
            "Organizer historical data should replace/confirm this interim evidence when received.",
        ],
        "summary": summary.to_dict(orient="records"),
        "window_results": results.to_dict(orient="records"),
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    summary.to_csv(OUT_CSV, index=False)

    print()
    print("=== INTERIM STRATEGY SUMMARY ===")
    display = summary.copy()
    for col in [
        "mean_return",
        "median_return",
        "positive_window_fraction",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_turnover",
    ]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    display["mean_sharpe"] = display["mean_sharpe"].map(
        lambda x: f"{x:.2f}"
    )
    display["median_rank_score"] = display["median_rank_score"].map(
        lambda x: f"{x:.2f}"
    )
    display["mean_rank_score"] = display["mean_rank_score"].map(
        lambda x: f"{x:.2f}"
    )

    print(display.to_string(index=False))

    print()
    print(f"JSON artifact: {OUT_JSON}")
    print(f"CSV summary:   {OUT_CSV}")
    print()
    print("PHASE 7 COMPLETE. NO COMPETITION SUBMISSION WAS MADE.")


if __name__ == "__main__":
    main()
