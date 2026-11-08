from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "INTC", "CRM",
    "JPM", "BAC", "GS", "V", "PYPL",
    "LLY", "JNJ", "UNH", "PFE", "TMO",
    "AMZN", "TSLA", "WMT", "NKE", "KO",
    "CAT", "GE", "BA", "XOM", "CVX",
    "GOOGL", "META", "DIS", "T", "NEE",
)

BAR_TIMES = tuple(
    f"{h:02d}:{m:02d}:00"
    for h, m in (
        (9, 30), (9, 45), (10, 0), (10, 15),
        (10, 30), (10, 45), (11, 0), (11, 15),
        (11, 30), (11, 45), (12, 0), (12, 15),
        (12, 30), (12, 45), (13, 0), (13, 15),
        (13, 30), (13, 45), (14, 0), (14, 15),
        (14, 30), (14, 45), (15, 0), (15, 15),
        (15, 30), (15, 45),
    )
)

EXEC_TIMES = (
    "10:30:00",
    "11:30:00",
    "12:30:00",
    "13:30:00",
    "14:30:00",
    "15:30:00",
)

CUTOFF_LAST_COMPLETE_TIMES = (
    "10:00:00",
    "11:00:00",
    "12:00:00",
    "13:00:00",
    "14:00:00",
    "15:00:00",
)

ROUND_LABELS = ("R2", "R3", "R4", "R5", "R6", "R7")

FEE_RATE = 0.001
GROSS = 0.80
MIN_TRAIN_DAYS = 60
TRAIN_WINDOW_DAYS = 100
RIDGE_LAMBDA = 50.0
SHARPE_ANNUALIZER = 42.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    tilt_l1: float
    partial: float
    no_trade_l1: float


STRATEGIES = (
    StrategySpec("equal30_80_hold", "hold", 0.0, 0.0, math.inf),
    StrategySpec("mom30_tilt04_p50_nt03", "mom30", 0.04, 0.50, 0.03),
    StrategySpec("rev15_tilt04_p50_nt03", "rev15", 0.04, 0.50, 0.03),
    StrategySpec("ridge_tilt04_p50_nt03", "ridge", 0.04, 0.50, 0.03),
    StrategySpec("ridge_tilt08_p50_nt04", "ridge", 0.08, 0.50, 0.04),
    StrategySpec("ridge_tilt08_p25_nt03", "ridge", 0.08, 0.25, 0.03),
)


def find_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (
            (candidate / "universe.json").exists()
            and (candidate / "data" / "cache").is_dir()
        ):
            return candidate.resolve()
    raise SystemExit(
        "ERROR: Could not locate starter-kit. "
        "Run this from the starter-kit directory."
    )


def detect_column(columns, choices):
    lower = {c.lower(): c for c in columns}
    for choice in choices:
        if choice in lower:
            return lower[choice]
    return None


def rank_center_vec(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    ranks = s.rank(method="average", pct=True).to_numpy(dtype=float)
    return (ranks - 0.5) * 2.0


def load_cache(path: Path):
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("Alpaca cache is empty.")

    ts_col = detect_column(
        raw.columns,
        ("timestamp", "timestamp_utc", "datetime", "time", "start"),
    )
    symbol_col = detect_column(raw.columns, ("symbol", "ticker"))
    open_col = detect_column(raw.columns, ("open",))
    close_col = detect_column(raw.columns, ("close",))
    volume_col = detect_column(raw.columns, ("volume",))

    missing = [
        name
        for name, col in (
            ("timestamp", ts_col),
            ("symbol/ticker", symbol_col),
            ("open", open_col),
            ("close", close_col),
            ("volume", volume_col),
        )
        if col is None
    ]
    if missing:
        raise ValueError(
            "Cache is missing required columns: " + ", ".join(missing)
        )

    df = raw[
        [ts_col, symbol_col, open_col, close_col, volume_col]
    ].copy()
    df.columns = ["timestamp", "symbol", "open", "close", "volume"]

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{bad} cache timestamps could not be parsed.")

    df["timestamp_et"] = ts.dt.tz_convert("America/New_York")
    df["date"] = df["timestamp_et"].dt.tz_localize(None).dt.normalize()
    df["time"] = df["timestamp_et"].dt.strftime("%H:%M:%S")
    df["symbol"] = df["symbol"].astype(str).str.upper()

    for c in ("open", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        df["symbol"].isin(SYMBOLS)
        & df["time"].isin(BAR_TIMES)
    ].copy()

    if df[["open", "close", "volume"]].isna().any().any():
        raise ValueError("Cache contains nonnumeric OHLCV values.")

    df = (
        df.sort_values("timestamp_et")
        .drop_duplicates(["date", "time", "symbol"], keep="last")
    )

    counts = (
        df.groupby(["date", "symbol"])["time"]
        .nunique()
        .unstack("symbol")
        .reindex(columns=SYMBOLS)
    )

    complete_dates = counts.index[
        (counts == len(BAR_TIMES)).all(axis=1)
    ]

    if len(complete_dates) < 100:
        raise ValueError(
            f"Only {len(complete_dates)} fully complete 15-minute days."
        )

    df = df[df["date"].isin(complete_dates)].copy()

    full_index = pd.MultiIndex.from_product(
        [complete_dates, BAR_TIMES],
        names=["date", "time"],
    )

    arrays = {}
    for field in ("open", "close", "volume"):
        piv = (
            df.pivot(
                index=["date", "time"],
                columns="symbol",
                values=field,
            )
            .reindex(index=full_index, columns=SYMBOLS)
        )
        if piv.isna().any().any():
            raise ValueError(f"Missing values after pivoting {field}.")
        arrays[field] = piv.to_numpy(dtype=float).reshape(
            len(complete_dates),
            len(BAR_TIMES),
            len(SYMBOLS),
        )

    return pd.DatetimeIndex(complete_dates), arrays


def trailing_time_of_day_surprise(
    values: np.ndarray,
    min_periods: int = 10,
    window: int = 20,
) -> np.ndarray:
    d, t, n = values.shape
    out = np.full_like(values, np.nan, dtype=float)

    logv = np.log(np.maximum(values, 1.0))

    for ti in range(t):
        frame = pd.DataFrame(logv[:, ti, :])
        hist_mean = frame.shift(1).rolling(
            window,
            min_periods=min_periods,
        ).mean()
        out[:, ti, :] = (
            frame - hist_mean
        ).to_numpy(dtype=float)

    return out


def build_event_dataset(
    dates: pd.DatetimeIndex,
    arrays: dict[str, np.ndarray],
):
    opens = arrays["open"]
    closes = arrays["close"]
    volumes = arrays["volume"]

    time_to_idx = {t: i for i, t in enumerate(BAR_TIMES)}
    exec_idx = [time_to_idx[t] for t in EXEC_TIMES]
    cutoff_idx = [time_to_idx[t] for t in CUTOFF_LAST_COMPLETE_TIMES]

    volume_surprise = trailing_time_of_day_surprise(volumes)

    cumulative_volume = np.cumsum(volumes, axis=1)
    cumulative_surprise = trailing_time_of_day_surprise(cumulative_volume)

    d_count = len(dates)
    e_count = len(EXEC_TIMES)
    n = len(SYMBOLS)

    feature_names = (
        "reversal15_rank",
        "mom30_rank",
        "since_open_rank",
        "rv_intraday_rank",
        "volume_surprise_rank",
        "cum_volume_surprise_rank",
        "gap_rank",
        "prev1_rank",
        "prev5_rank",
        "round_x_mom30",
        "round_x_since_open",
        "regime_x_mom30",
        "regime_x_reversal15",
    )

    X = np.full(
        (d_count, e_count, n, len(feature_names)),
        np.nan,
        dtype=float,
    )
    y = np.full((d_count, e_count, n), np.nan, dtype=float)
    mom30_scores = np.full((d_count, e_count, n), np.nan, dtype=float)
    rev15_scores = np.full((d_count, e_count, n), np.nan, dtype=float)

    day_close = closes[:, time_to_idx["15:45:00"], :]
    day_open = opens[:, time_to_idx["09:30:00"], :]

    prev1 = np.full((d_count, n), np.nan)
    prev5 = np.full((d_count, n), np.nan)
    gap = np.full((d_count, n), np.nan)

    for d in range(1, d_count):
        gap[d] = day_open[d] / day_close[d - 1] - 1.0

    for d in range(2, d_count):
        prev1[d] = day_close[d - 1] / day_close[d - 2] - 1.0

    for d in range(6, d_count):
        prev5[d] = day_close[d - 1] / day_close[d - 6] - 1.0

    for d in range(d_count):
        for e, (ei, ci) in enumerate(zip(exec_idx, cutoff_idx)):
            if ci < 2:
                continue

            ret15 = closes[d, ci] / closes[d, ci - 1] - 1.0
            ret30 = closes[d, ci] / closes[d, ci - 2] - 1.0
            since_open = closes[d, ci] / day_open[d] - 1.0

            intraday_rets = (
                closes[d, 1 : ci + 1]
                / closes[d, :ci]
                - 1.0
            )
            rv = np.std(intraday_rets, axis=0, ddof=1)

            pieces = [
                rank_center_vec(-ret15),
                rank_center_vec(ret30),
                rank_center_vec(since_open),
                rank_center_vec(rv),
                rank_center_vec(volume_surprise[d, ci]),
                rank_center_vec(cumulative_surprise[d, ci]),
                rank_center_vec(gap[d]),
                rank_center_vec(prev1[d]),
                rank_center_vec(prev5[d]),
            ]

            if not all(np.isfinite(p).all() for p in pieces):
                continue

            round_norm = (e - 2.5) / 2.5
            regime = float(np.median(since_open))
            regime = float(np.clip(regime / 0.02, -1.5, 1.5))

            row = np.column_stack(
                [
                    *pieces,
                    round_norm * pieces[1],
                    round_norm * pieces[2],
                    regime * pieces[1],
                    regime * pieces[0],
                ]
            )

            X[d, e] = row
            mom30_scores[d, e] = pieces[1]
            rev15_scores[d, e] = pieces[0]

            start_px = opens[d, ei]
            if e < e_count - 1:
                end_px = opens[d, exec_idx[e + 1]]
            elif d + 1 < d_count:
                end_px = day_open[d + 1]
            else:
                continue

            ret = end_px / start_px - 1.0
            y[d, e] = ret - float(np.mean(ret))

    return (
        X,
        y,
        mom30_scores,
        rev15_scores,
        feature_names,
        exec_idx,
        time_to_idx,
    )


def walk_forward_ridge(
    X: np.ndarray,
    y: np.ndarray,
):
    d_count, e_count, n, f_count = X.shape

    day_xtx = np.zeros((d_count, f_count, f_count), dtype=float)
    day_xty = np.zeros((d_count, f_count), dtype=float)

    for d in range(d_count):
        rows_x = []
        rows_y = []
        for e in range(e_count):
            mask = (
                np.isfinite(y[d, e])
                & np.isfinite(X[d, e]).all(axis=1)
            )
            if mask.any():
                rows_x.append(X[d, e, mask])
                rows_y.append(y[d, e, mask])

        if rows_x:
            xv = np.vstack(rows_x)
            yv = np.concatenate(rows_y)
            day_xtx[d] = xv.T @ xv
            day_xty[d] = xv.T @ yv

    c_xtx = np.cumsum(day_xtx, axis=0)
    c_xty = np.cumsum(day_xty, axis=0)

    scores = np.full((d_count, e_count, n), np.nan, dtype=float)
    raw_pred = np.full_like(scores, np.nan)
    beta_norm = np.full(d_count, np.nan)

    eye = np.eye(f_count)

    for d in range(MIN_TRAIN_DAYS, d_count):
        last_train = d - 1
        first_train = max(0, d - TRAIN_WINDOW_DAYS)

        xtx = c_xtx[last_train].copy()
        xty = c_xty[last_train].copy()

        if first_train > 0:
            xtx -= c_xtx[first_train - 1]
            xty -= c_xty[first_train - 1]

        beta = np.linalg.solve(
            xtx + RIDGE_LAMBDA * eye,
            xty,
        )
        beta_norm[d] = float(np.linalg.norm(beta))

        for e in range(e_count):
            x_now = X[d, e]
            if not np.isfinite(x_now).all():
                continue
            pred = x_now @ beta
            raw_pred[d, e] = pred
            scores[d, e] = rank_center_vec(pred)

    return scores, raw_pred, beta_norm


def target_from_score(score: np.ndarray, tilt_l1: float) -> np.ndarray:
    base = np.full(len(SYMBOLS), GROSS / len(SYMBOLS), dtype=float)

    if tilt_l1 <= 0.0:
        return base

    centered = score - float(np.mean(score))
    denom = float(np.abs(centered).sum())
    if denom <= 1e-15:
        return base

    delta = tilt_l1 * centered / denom
    w = base + delta

    if float(w.min()) < 0:
        neg = delta < 0
        scale = min(
            1.0,
            float(np.min(base[neg] / (-delta[neg]))),
        )
        w = base + delta * max(0.0, scale * 0.999999)

    w = np.maximum(w, 0.0)
    w *= GROSS / float(w.sum())

    if float(w.max()) > 0.30 + 1e-12:
        raise ValueError("Per-stock cap exceeded.")

    return w


def solve_trade(
    pre_nav: float,
    current_values: np.ndarray,
    prices: np.ndarray,
    target_weights: np.ndarray,
):
    post_nav = float(pre_nav)

    for _ in range(50):
        target_values = target_weights * post_nav
        gross_notional = float(
            np.abs(target_values - current_values).sum()
        )
        new_post_nav = pre_nav - FEE_RATE * gross_notional
        if abs(new_post_nav - post_nav) <= 1e-10 * max(1.0, pre_nav):
            post_nav = new_post_nav
            break
        post_nav = new_post_nav

    target_values = target_weights * post_nav
    gross_notional = float(
        np.abs(target_values - current_values).sum()
    )
    fee = FEE_RATE * gross_notional
    shares = target_values / prices
    cash = post_nav - float(target_values.sum())

    return shares, cash, gross_notional, fee


def competition_sharpe(period_returns: np.ndarray) -> float:
    if len(period_returns) < 2:
        return 0.0
    std = float(np.std(period_returns, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return 0.0
    return float(
        np.mean(period_returns) / std * SHARPE_ANNUALIZER
    )


def max_drawdown(nav_points: np.ndarray) -> float:
    peaks = np.maximum.accumulate(nav_points)
    return float(np.max(1.0 - nav_points / peaks))


def strategy_score(
    spec: StrategySpec,
    d: int,
    e: int,
    ridge_scores: np.ndarray,
    mom_scores: np.ndarray,
    rev_scores: np.ndarray,
):
    if spec.kind == "ridge":
        return ridge_scores[d, e]
    if spec.kind == "mom30":
        return mom_scores[d, e]
    if spec.kind == "rev15":
        return rev_scores[d, e]
    return None


def simulate_window(
    start: int,
    length: int,
    spec: StrategySpec,
    arrays: dict[str, np.ndarray],
    ridge_scores: np.ndarray,
    mom_scores: np.ndarray,
    rev_scores: np.ndarray,
    exec_idx: list[int],
    time_to_idx: dict[str, int],
):
    opens = arrays["open"]
    closes = arrays["close"]

    shares = np.zeros(len(SYMBOLS), dtype=float)
    cash = 1_000_000.0
    initial_nav = cash

    gross_turnover_sum = 0.0
    fees = 0.0
    rebalances = 0
    holds = 0
    period_returns = []
    mdd_points = [initial_nav]

    base = np.full(len(SYMBOLS), GROSS / len(SYMBOLS))

    first_open = opens[start, time_to_idx["09:30:00"]]
    pre_nav = initial_nav
    shares, cash, gross_notional, fee = solve_trade(
        pre_nav,
        np.zeros(len(SYMBOLS)),
        first_open,
        base,
    )
    gross_turnover_sum += gross_notional / pre_nav
    fees += fee
    rebalances += 1
    period_start_nav = pre_nav

    for offset in range(length):
        d = start + offset

        # R1 is HOLD after the initial deployment. Its period ends at R2.
        for e, ei in enumerate(exec_idx):
            px = opens[d, ei]
            pre_nav = cash + float(np.sum(shares * px))

            period_returns.append(
                pre_nav / period_start_nav - 1.0
            )
            mdd_points.append(pre_nav)

            if spec.kind == "hold":
                holds += 1
            else:
                score = strategy_score(
                    spec,
                    d,
                    e,
                    ridge_scores,
                    mom_scores,
                    rev_scores,
                )
                if score is None or not np.isfinite(score).all():
                    holds += 1
                else:
                    raw_target = target_from_score(
                        score,
                        spec.tilt_l1,
                    )

                    current_values = shares * px
                    current_weights = current_values / pre_nav

                    target = (
                        current_weights
                        + spec.partial
                        * (raw_target - current_weights)
                    )

                    # Clean tiny numerical negatives and preserve no leverage.
                    target = np.maximum(target, 0.0)
                    if float(target.sum()) > 1.0:
                        target /= float(target.sum())

                    l1 = float(
                        np.abs(target - current_weights).sum()
                    )

                    if l1 < spec.no_trade_l1:
                        holds += 1
                    else:
                        (
                            shares,
                            cash,
                            gross_notional,
                            fee,
                        ) = solve_trade(
                            pre_nav,
                            current_values,
                            px,
                            target,
                        )
                        gross_turnover_sum += (
                            gross_notional / pre_nav
                        )
                        fees += fee
                        rebalances += 1

            period_start_nav = pre_nav

        close_px = closes[d, time_to_idx["15:45:00"]]
        close_nav = cash + float(np.sum(shares * close_px))
        mdd_points.append(close_nav)

        if offset < length - 1:
            next_open = opens[
                d + 1,
                time_to_idx["09:30:00"],
            ]
            next_open_nav = cash + float(np.sum(shares * next_open))

            # R7 -> next R1 boundary.
            period_returns.append(
                next_open_nav / period_start_nav - 1.0
            )
            mdd_points.append(next_open_nav)

            # R1 itself is HOLD; its next period starts at this boundary.
            period_start_nav = next_open_nav
        else:
            # Final phase/window day: R7 period ends at official close.
            period_returns.append(
                close_nav / period_start_nav - 1.0
            )

    final_nav = mdd_points[-1]
    rounds = length * 7

    return {
        "return": float(final_nav / initial_nav - 1.0),
        "sharpe": competition_sharpe(
            np.asarray(period_returns, dtype=float)
        ),
        "mdd": max_drawdown(np.asarray(mdd_points, dtype=float)),
        "turnover": gross_turnover_sum / rounds,
        "fees": fees,
        "rebalances": rebalances,
        "holds": holds,
    }


def add_rank_proxy(rows: list[dict]):
    df = pd.DataFrame(rows)
    df["rank_return"] = df["return"].rank(
        ascending=False,
        method="average",
    )
    df["rank_sharpe"] = df["sharpe"].rank(
        ascending=False,
        method="average",
    )
    df["rank_mdd"] = df["mdd"].rank(
        ascending=True,
        method="average",
    )
    df["rank_turnover"] = df["turnover"].rank(
        ascending=True,
        method="average",
    )
    df["rank_score_proxy"] = df[
        [
            "rank_return",
            "rank_sharpe",
            "rank_mdd",
            "rank_turnover",
        ]
    ].mean(axis=1)

    lookup = df.set_index("strategy")[
        "rank_score_proxy"
    ].to_dict()

    for row in rows:
        row["rank_score_proxy"] = float(
            lookup[row["strategy"]]
        )


def summarize(results: pd.DataFrame):
    base = (
        results[results["strategy"] == "equal30_80_hold"]
        .set_index("window_id")
    )

    rows = []
    for name, g in results.groupby("strategy", sort=False):
        a = g.set_index("window_id")
        common = a.index.intersection(base.index)

        rows.append(
            {
                "strategy": name,
                "windows": len(g),
                "mean_return": g["return"].mean(),
                "median_return": g["return"].median(),
                "positive_fraction": (g["return"] > 0).mean(),
                "mean_sharpe": g["sharpe"].mean(),
                "mean_mdd": g["mdd"].mean(),
                "worst_mdd": g["mdd"].max(),
                "mean_turnover": g["turnover"].mean(),
                "mean_rebalances": g["rebalances"].mean(),
                "median_rank_proxy": g["rank_score_proxy"].median(),
                "mean_rank_proxy": g["rank_score_proxy"].mean(),
                "better_rank_vs_equal80": float(
                    (
                        a.loc[common, "rank_score_proxy"]
                        < base.loc[common, "rank_score_proxy"]
                    ).mean()
                ),
                "return_delta_vs_equal80": float(
                    (
                        a.loc[common, "return"]
                        - base.loc[common, "return"]
                    ).mean()
                ),
                "sharpe_delta_vs_equal80": float(
                    (
                        a.loc[common, "sharpe"]
                        - base.loc[common, "sharpe"]
                    ).mean()
                ),
                "mdd_delta_vs_equal80": float(
                    (
                        a.loc[common, "mdd"]
                        - base.loc[common, "mdd"]
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["median_rank_proxy", "mean_rank_proxy"]
    )


def evaluate_windows(
    dates,
    arrays,
    ridge_scores,
    mom_scores,
    rev_scores,
    exec_idx,
    time_to_idx,
    length,
    step,
):
    valid_day = np.isfinite(ridge_scores).all(axis=(1, 2))
    starts = []

    for s in range(0, len(dates) - length + 1, step):
        if valid_day[s : s + length].all():
            starts.append(s)

    if not starts:
        raise ValueError(
            f"No eligible {length}-day OOS windows."
        )

    rows = []
    for wid, s in enumerate(starts, start=1):
        current = []
        for spec in STRATEGIES:
            metrics = simulate_window(
                s,
                length,
                spec,
                arrays,
                ridge_scores,
                mom_scores,
                rev_scores,
                exec_idx,
                time_to_idx,
            )
            current.append(
                {
                    "window_id": wid,
                    "start_date": str(dates[s].date()),
                    "end_date": str(
                        dates[s + length - 1].date()
                    ),
                    "strategy": spec.name,
                    **metrics,
                }
            )
        add_rank_proxy(current)
        rows.extend(current)

    frame = pd.DataFrame(rows)
    return frame, summarize(frame)


def self_check():
    score = np.linspace(-1, 1, 30)

    for tilt in (0.0, 0.04, 0.08):
        w = target_from_score(score, tilt)
        assert len(w) == 30
        assert np.isfinite(w).all()
        assert (w >= 0).all()
        assert abs(float(w.sum()) - 0.80) < 1e-10
        assert float(w.max()) <= 0.30 + 1e-12

    prices = np.linspace(50, 150, 30)
    w = target_from_score(score, 0.04)
    sh, cash, gross, fee = solve_trade(
        1_000_000.0,
        np.zeros(30),
        prices,
        w,
    )
    nav = cash + float(np.sum(sh * prices))
    assert gross > 0
    assert fee > 0
    assert abs(nav - (1_000_000.0 - fee)) < 1e-5

    test_rets = np.array([0.001, -0.0005, 0.0008])
    assert math.isfinite(competition_sharpe(test_rets))

    print("PHASE13_SELF_CHECK_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    root = find_root()
    cache = root / "data" / "cache" / "alpaca_iex_15m_400d.csv"
    if not cache.exists():
        raise SystemExit(
            f"ERROR: Expected Alpaca cache not found: {cache}"
        )

    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 13: INTRADAY SELECTIVE-REBALANCE CHALLENGER ===")
    print("Competition submission: DISABLED")
    print("Data source: existing cached Alpaca IEX 15-minute bars")
    print("External API downloads: NONE")
    print("Training: pooled rolling ridge, prior trading days only")
    print("Live decision opportunities modeled: R2-R7")
    print(
        "Feature cutoff: last fully completed 15-minute bar "
        "before each 5-minute submission deadline"
    )
    print(
        "R7 target: 15:30 ET execution to next trading day 09:30 ET open"
    )
    print()

    dates, arrays = load_cache(cache)
    print(
        f"Complete 15-minute days: {len(dates)} "
        f"({dates.min().date()} -> {dates.max().date()})"
    )

    (
        X,
        y,
        mom_scores,
        rev_scores,
        feature_names,
        exec_idx,
        time_to_idx,
    ) = build_event_dataset(dates, arrays)

    ridge_scores, raw_pred, beta_norm = walk_forward_ridge(X, y)

    print(f"Features: {len(feature_names)}")
    print(", ".join(feature_names))

    print()
    print("=== OOS EVENT-LEVEL SIGNAL DIAGNOSTIC ===")
    all_ics = []
    round_rows = []

    for e, label in enumerate(ROUND_LABELS):
        ics = []
        for d in range(len(dates)):
            if (
                not np.isfinite(ridge_scores[d, e]).all()
                or not np.isfinite(y[d, e]).all()
            ):
                continue

            ic = np.corrcoef(
                pd.Series(ridge_scores[d, e])
                .rank(method="average")
                .to_numpy(dtype=float),
                pd.Series(y[d, e])
                .rank(method="average")
                .to_numpy(dtype=float),
            )[0, 1]

            if math.isfinite(float(ic)):
                ics.append(float(ic))
                all_ics.append(float(ic))

        round_rows.append(
            {
                "round": label,
                "events": len(ics),
                "mean_ic": float(np.mean(ics)) if ics else np.nan,
                "median_ic": float(np.median(ics)) if ics else np.nan,
                "positive_ic_fraction": (
                    float(np.mean(np.asarray(ics) > 0))
                    if ics
                    else np.nan
                ),
            }
        )

    round_df = pd.DataFrame(round_rows)
    print(round_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not all_ics:
        raise SystemExit("ERROR: No valid OOS intraday IC observations.")

    print(
        f"Overall mean IC: {np.mean(all_ics):.4f}; "
        f"median IC: {np.median(all_ics):.4f}; "
        f"positive fraction: "
        f"{np.mean(np.asarray(all_ics) > 0):.4f}"
    )

    two_results, two_summary = evaluate_windows(
        dates,
        arrays,
        ridge_scores,
        mom_scores,
        rev_scores,
        exec_idx,
        time_to_idx,
        length=2,
        step=1,
    )

    fifteen_results, fifteen_summary = evaluate_windows(
        dates,
        arrays,
        ridge_scores,
        mom_scores,
        rev_scores,
        exec_idx,
        time_to_idx,
        length=15,
        step=5,
    )

    cols = [
        "strategy",
        "windows",
        "mean_return",
        "median_return",
        "positive_fraction",
        "mean_sharpe",
        "mean_mdd",
        "worst_mdd",
        "mean_turnover",
        "mean_rebalances",
        "median_rank_proxy",
        "better_rank_vs_equal80",
        "return_delta_vs_equal80",
        "sharpe_delta_vs_equal80",
        "mdd_delta_vs_equal80",
    ]

    print()
    print("=== 2-DAY VALIDATION-LENGTH WINDOWS ===")
    print(
        two_summary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=== 15-DAY OFFICIAL-LENGTH WINDOWS ===")
    print(
        fifteen_summary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Conservative pre-declared promotion gate.
    merged = (
        two_summary.set_index("strategy")
        .join(
            fifteen_summary.set_index("strategy"),
            lsuffix="_2d",
            rsuffix="_15d",
        )
    )

    candidate_names = [
        s.name
        for s in STRATEGIES
        if s.kind == "ridge"
    ]

    promotable = []
    overall_mean_ic = float(np.mean(all_ics))

    for name in candidate_names:
        r = merged.loc[name]

        if (
            overall_mean_ic >= 0.01
            and r["better_rank_vs_equal80_2d"] >= 0.55
            and r["return_delta_vs_equal80_2d"] >= 0.0
            and r["sharpe_delta_vs_equal80_2d"] >= 0.0
            and r["better_rank_vs_equal80_15d"] >= 0.55
            and r["return_delta_vs_equal80_15d"] >= 0.0
            and r["sharpe_delta_vs_equal80_15d"] >= 0.0
            and r["mdd_delta_vs_equal80_15d"] <= 0.002
            and r["mean_turnover_15d"] <= 0.05
        ):
            promotable.append(name)

    print()
    print("=== KEEP / REJECT INTERPRETATION ===")
    if promotable:
        best = (
            fifteen_summary[
                fifteen_summary["strategy"].isin(promotable)
            ]
            .sort_values(
                ["median_rank_proxy", "mean_rank_proxy"]
            )
            .iloc[0]["strategy"]
        )
        print(
            "PROMOTION CANDIDATE: "
            f"{best} clears the predeclared 2-day + 15-day gate."
        )
        print(
            "Next step: integrate it as a small intraday overlay, "
            "then run shadow/live-data dry-run before granting submission authority."
        )
    else:
        print(
            "NO PROMOTION: no intraday ridge overlay clears the "
            "predeclared OOS gate."
        )
        print(
            "Keep equal30_80_hold as the live fallback. "
            "Do not force R2-R7 trades merely because rounds exist."
        )

    two_results.to_csv(
        artifacts / "phase13_intraday_2d_windows.csv",
        index=False,
    )
    fifteen_results.to_csv(
        artifacts / "phase13_intraday_15d_windows.csv",
        index=False,
    )
    round_df.to_csv(
        artifacts / "phase13_intraday_ic_by_round.csv",
        index=False,
    )

    payload = {
        "features": list(feature_names),
        "training": {
            "min_train_days": MIN_TRAIN_DAYS,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "ridge_lambda": RIDGE_LAMBDA,
            "fee_rate": FEE_RATE,
        },
        "overall_mean_ic": overall_mean_ic,
        "ic_by_round": round_rows,
        "two_day_summary": two_summary.to_dict(orient="records"),
        "fifteen_day_summary": fifteen_summary.to_dict(orient="records"),
        "promotable": promotable,
    }

    (
        artifacts / "phase13_intraday_challenger.json"
    ).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print()
    print("Artifacts written under artifacts/phase13_*")
    print("PHASE 13 COMPLETE. NO COMPETITION SUBMISSION WAS MADE.")


if __name__ == "__main__":
    main()
