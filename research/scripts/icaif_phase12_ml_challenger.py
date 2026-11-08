from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OFFICIAL_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "INTC", "CRM",
    "JPM", "BAC", "GS", "V", "PYPL",
    "LLY", "JNJ", "UNH", "PFE", "TMO",
    "AMZN", "TSLA", "WMT", "NKE", "KO",
    "CAT", "GE", "BA", "XOM", "CVX",
    "GOOGL", "META", "DIS", "T", "NEE",
)

STANDARD_TIMES = (
    "09:30:00",
    "10:00:00",
    "11:00:00",
    "12:00:00",
    "13:00:00",
    "14:00:00",
    "15:00:00",
)

FEE_RATE = 0.001
GROSS_EXPOSURE = 0.80
HORIZON_DAYS = 5
MIN_TRAIN_DAYS = 252
TRAIN_WINDOW_DAYS = 756
RIDGE_LAMBDA = 25.0
WINDOW_DAYS = 15
WINDOW_STEP = 5
SHARPE_ANNUALIZER = 42.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    tilt_l1: float
    no_trade_l1: float


STRATEGIES = (
    StrategySpec("equal30_80_hold", 0.0, math.inf),
    StrategySpec("ridge_tilt08_nt05", 0.08, 0.05),
    StrategySpec("ridge_tilt12_nt05", 0.12, 0.05),
    StrategySpec("ridge_tilt16_nt05", 0.16, 0.05),
    StrategySpec("ridge_tilt16_nt10", 0.16, 0.10),
)


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (
            (candidate / "universe.json").exists()
            and (candidate / "data" / "official").is_dir()
        ):
            return candidate.resolve()
    raise SystemExit(
        "ERROR: Could not locate starter-kit. "
        "Run this while PowerShell is in the starter-kit folder."
    )


def rank_center(df: pd.DataFrame) -> pd.DataFrame:
    ranks = df.rank(axis=1, pct=True, method="average")
    return (ranks - 0.5) * 2.0


def sharpe_proxy(nav_path: np.ndarray) -> float:
    returns = nav_path[1:] / nav_path[:-1] - 1.0
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return 0.0
    return float(np.mean(returns) / std * SHARPE_ANNUALIZER)


def max_drawdown(nav_path: np.ndarray) -> float:
    peaks = np.maximum.accumulate(nav_path)
    dd = 1.0 - nav_path / peaks
    return float(np.max(dd))


def solve_target_trade(
    pre_nav: float,
    current_values: np.ndarray,
    prices: np.ndarray,
    target_weights: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    post_nav = float(pre_nav)

    for _ in range(50):
        target_values = target_weights * post_nav
        gross_notional = float(np.abs(target_values - current_values).sum())
        new_post_nav = pre_nav - FEE_RATE * gross_notional
        if abs(new_post_nav - post_nav) <= 1e-10 * max(1.0, pre_nav):
            post_nav = new_post_nav
            break
        post_nav = new_post_nav

    target_values = target_weights * post_nav
    gross_notional = float(np.abs(target_values - current_values).sum())
    fee = FEE_RATE * gross_notional
    shares = target_values / prices
    cash = post_nav - float(target_values.sum())
    return shares, cash, gross_notional, fee


def target_from_score(score: np.ndarray, tilt_l1: float) -> np.ndarray:
    n = score.size
    base = np.full(n, GROSS_EXPOSURE / n, dtype=float)

    if tilt_l1 <= 0.0:
        return base

    centered = score - float(np.mean(score))
    denom = float(np.abs(centered).sum())
    if denom <= 1e-15:
        return base

    delta = tilt_l1 * centered / denom
    weights = base + delta

    if float(weights.min()) < -1e-12:
        # Scale the tilt back just enough to preserve long-only geometry.
        negative = delta < 0
        scale = 1.0
        if np.any(negative):
            scale = min(
                1.0,
                float(np.min(base[negative] / (-delta[negative]))),
            )
        weights = base + delta * max(0.0, scale * 0.999999)

    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return base

    weights *= GROSS_EXPOSURE / total

    if float(weights.max()) > 0.30 + 1e-12:
        raise ValueError("Per-stock cap exceeded by ML target.")
    if not math.isclose(
        float(weights.sum()),
        GROSS_EXPOSURE,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("ML target gross exposure mismatch.")

    return weights


def build_daily_dataset(raw: pd.DataFrame):
    df = raw.copy()

    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"], errors="raise")
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="raise").dt.normalize()
    df["time_str"] = df["timestamp_et"].dt.strftime("%H:%M:%S")

    df = df[
        df["ticker"].isin(OFFICIAL_SYMBOLS)
        & df["time_str"].isin(STANDARD_TIMES)
    ].copy()

    counts = (
        df.groupby(["trading_date", "ticker"])["time_str"]
        .nunique()
        .unstack("ticker")
        .reindex(columns=OFFICIAL_SYMBOLS)
    )
    complete_dates = counts.index[(counts == len(STANDARD_TIMES)).all(axis=1)]
    df = df[df["trading_date"].isin(complete_dates)].copy()

    def pivot_at(time_str: str, value: str) -> pd.DataFrame:
        out = (
            df[df["time_str"] == time_str]
            .pivot(index="trading_date", columns="ticker", values=value)
            .reindex(index=complete_dates, columns=OFFICIAL_SYMBOLS)
            .astype(float)
        )
        if out.isna().any().any():
            raise ValueError(f"Missing values in pivot {time_str}/{value}.")
        return out

    open_0930 = pivot_at("09:30:00", "open")
    close_1500 = pivot_at("15:00:00", "adj_close")

    endpoint_close = {
        t: pivot_at(t, "adj_close")
        for t in STANDARD_TIMES
    }

    daily = (
        df.groupby(["trading_date", "ticker"], sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("adj_close", "last"),
            volume=("volume", "sum"),
            sector=("sector_group", "first"),
        )
        .reset_index()
    )

    close = daily.pivot(
        index="trading_date",
        columns="ticker",
        values="close",
    ).reindex(index=complete_dates, columns=OFFICIAL_SYMBOLS).astype(float)

    high = daily.pivot(
        index="trading_date",
        columns="ticker",
        values="high",
    ).reindex(index=complete_dates, columns=OFFICIAL_SYMBOLS).astype(float)

    low = daily.pivot(
        index="trading_date",
        columns="ticker",
        values="low",
    ).reindex(index=complete_dates, columns=OFFICIAL_SYMBOLS).astype(float)

    volume = daily.pivot(
        index="trading_date",
        columns="ticker",
        values="volume",
    ).reindex(index=complete_dates, columns=OFFICIAL_SYMBOLS).astype(float)

    sector_map = (
        daily[["ticker", "sector"]]
        .drop_duplicates()
        .groupby("ticker")["sector"]
        .first()
        .reindex(OFFICIAL_SYMBOLS)
    )

    return (
        close.index,
        open_0930,
        close,
        high,
        low,
        volume,
        endpoint_close,
        sector_map,
    )


def build_features(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    sector_map: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ret1 = close.pct_change(1)
    mom3 = close.pct_change(3)
    mom5 = close.pct_change(5)
    mom10 = close.pct_change(10)
    mom20 = close.pct_change(20)
    vol5 = ret1.rolling(5).std()
    vol20 = ret1.rolling(20).std()
    range1 = high / low - 1.0

    log_volume = np.log(volume.replace(0.0, np.nan))
    volume_surprise = (
        log_volume
        - log_volume.rolling(20).mean()
    ) / log_volume.rolling(20).std()

    sec_mom5 = mom5.copy()
    sec_mom20 = mom20.copy()
    for sector in sector_map.dropna().unique():
        cols = sector_map.index[sector_map == sector].tolist()
        sec_mom5.loc[:, cols] = (
            mom5.loc[:, cols]
            .sub(mom5.loc[:, cols].mean(axis=1), axis=0)
        )
        sec_mom20.loc[:, cols] = (
            mom20.loc[:, cols]
            .sub(mom20.loc[:, cols].mean(axis=1), axis=0)
        )

    base_frames = [
        rank_center(-ret1),
        rank_center(mom3),
        rank_center(mom5),
        rank_center(mom10),
        rank_center(mom20),
        rank_center(vol5),
        rank_center(vol20),
        rank_center(range1),
        rank_center(volume_surprise),
        rank_center(sec_mom5),
        rank_center(sec_mom20),
    ]

    base_names = [
        "reversal1_rank",
        "mom3_rank",
        "mom5_rank",
        "mom10_rank",
        "mom20_rank",
        "vol5_rank",
        "vol20_rank",
        "range1_rank",
        "volume_surprise_rank",
        "sector_mom5_rank",
        "sector_mom20_rank",
    ]

    base = np.stack(
        [frame.to_numpy(dtype=float) for frame in base_frames],
        axis=2,
    )

    # Small, predetermined nonlinear interaction set.
    interactions = np.stack(
        [
            base[:, :, 2] * base[:, :, 6],
            base[:, :, 4] * base[:, :, 6],
            base[:, :, 2] * base[:, :, 4],
            base[:, :, 0] * base[:, :, 8],
        ],
        axis=2,
    )
    interaction_names = [
        "mom5_x_vol20",
        "mom20_x_vol20",
        "mom5_x_mom20",
        "reversal1_x_volume",
    ]

    X = np.concatenate([base, interactions], axis=2)

    future5 = close.shift(-HORIZON_DAYS) / close - 1.0
    y_df = future5.sub(future5.mean(axis=1), axis=0)
    y = y_df.to_numpy(dtype=float)

    return X, y, base_names + interaction_names


def walk_forward_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_dates, n_assets, n_features = X.shape
    daily_xtx = np.zeros((n_dates, n_features, n_features), dtype=float)
    daily_xty = np.zeros((n_dates, n_features), dtype=float)
    valid_day = np.zeros(n_dates, dtype=bool)

    for i in range(n_dates):
        xi = X[i]
        yi = y[i]
        mask = np.isfinite(yi) & np.isfinite(xi).all(axis=1)
        if int(mask.sum()) < max(10, n_features):
            continue
        xv = xi[mask]
        yv = yi[mask]
        daily_xtx[i] = xv.T @ xv
        daily_xty[i] = xv.T @ yv
        valid_day[i] = True

    c_xtx = np.cumsum(daily_xtx, axis=0)
    c_xty = np.cumsum(daily_xty, axis=0)

    scores = np.full((n_dates, n_assets), np.nan, dtype=float)
    beta_norm = np.full(n_dates, np.nan, dtype=float)

    eye = np.eye(n_features, dtype=float)

    for signal_idx in range(MIN_TRAIN_DAYS, n_dates):
        last_train = signal_idx - HORIZON_DAYS
        if last_train < 0:
            continue

        first_train = max(
            0,
            last_train - TRAIN_WINDOW_DAYS + 1,
        )

        xtx = c_xtx[last_train].copy()
        xty = c_xty[last_train].copy()
        if first_train > 0:
            xtx -= c_xtx[first_train - 1]
            xty -= c_xty[first_train - 1]

        x_now = X[signal_idx]
        if not np.isfinite(x_now).all():
            continue

        beta = np.linalg.solve(
            xtx + RIDGE_LAMBDA * eye,
            xty,
        )
        pred = x_now @ beta

        ranks = pd.Series(pred).rank(
            pct=True,
            method="average",
        ).to_numpy()
        scores[signal_idx] = (ranks - 0.5) * 2.0
        beta_norm[signal_idx] = float(np.linalg.norm(beta))

    return scores, beta_norm


def simulate_window(
    start_idx: int,
    spec: StrategySpec,
    scores: np.ndarray,
    open_0930: np.ndarray,
    endpoint_arrays: list[np.ndarray],
) -> dict:
    n_assets = open_0930.shape[1]
    shares = np.zeros(n_assets, dtype=float)
    cash = 1_000_000.0
    initial_nav = cash

    nav_points = [initial_nav]
    total_turnover_ratio = 0.0
    total_fees = 0.0
    rebalances = 0

    for day_offset in range(WINDOW_DAYS):
        day_idx = start_idx + day_offset
        prices_open = open_0930[day_idx]
        current_values = shares * prices_open
        pre_nav = cash + float(current_values.sum())

        if spec.name == "equal30_80_hold":
            should_consider = day_offset == 0
            target = np.full(
                n_assets,
                GROSS_EXPOSURE / n_assets,
                dtype=float,
            )
        else:
            signal_idx = day_idx - 1
            if signal_idx < 0 or not np.isfinite(scores[signal_idx]).all():
                raise ValueError(
                    "Missing walk-forward score before test day."
                )
            target = target_from_score(
                scores[signal_idx],
                spec.tilt_l1,
            )
            should_consider = True

        if should_consider:
            current_weights = (
                current_values / pre_nav
                if pre_nav > 0
                else np.zeros(n_assets, dtype=float)
            )
            l1 = float(np.abs(target - current_weights).sum())

            if (
                day_offset == 0
                or l1 >= spec.no_trade_l1
            ):
                (
                    shares,
                    cash,
                    gross_notional,
                    fee,
                ) = solve_target_trade(
                    pre_nav=pre_nav,
                    current_values=current_values,
                    prices=prices_open,
                    target_weights=target,
                )
                total_turnover_ratio += gross_notional / pre_nav
                total_fees += fee
                rebalances += 1

        for endpoint in endpoint_arrays:
            nav = cash + float(
                np.sum(shares * endpoint[day_idx])
            )
            nav_points.append(nav)

    nav_arr = np.asarray(nav_points, dtype=float)

    return {
        "return": float(nav_arr[-1] / initial_nav - 1.0),
        "sharpe_proxy": sharpe_proxy(nav_arr),
        "mdd_proxy": max_drawdown(nav_arr),
        "turnover_proxy": total_turnover_ratio / (WINDOW_DAYS * 7),
        "fees": total_fees,
        "rebalances": rebalances,
    }


def rank_within_window(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame["rank_return"] = frame["return"].rank(
        ascending=False,
        method="average",
    )
    frame["rank_sharpe"] = frame["sharpe_proxy"].rank(
        ascending=False,
        method="average",
    )
    frame["rank_mdd"] = frame["mdd_proxy"].rank(
        ascending=True,
        method="average",
    )
    frame["rank_turnover"] = frame["turnover_proxy"].rank(
        ascending=True,
        method="average",
    )
    frame["rank_score_proxy"] = frame[
        ["rank_return", "rank_sharpe", "rank_mdd", "rank_turnover"]
    ].mean(axis=1)

    lookup = frame.set_index("strategy")["rank_score_proxy"].to_dict()
    for row in rows:
        row["rank_score_proxy"] = float(lookup[row["strategy"]])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[
        results["strategy"] == "equal30_80_hold"
    ].set_index("window_id")

    rows = []
    for name, g in results.groupby("strategy", sort=False):
        aligned = g.set_index("window_id")
        common = aligned.index.intersection(baseline.index)

        better_rank = float(
            (
                aligned.loc[common, "rank_score_proxy"]
                < baseline.loc[common, "rank_score_proxy"]
            ).mean()
        )

        rows.append(
            {
                "strategy": name,
                "windows": len(g),
                "mean_return": g["return"].mean(),
                "median_return": g["return"].median(),
                "p10_return": g["return"].quantile(0.10),
                "positive_fraction": (g["return"] > 0).mean(),
                "mean_sharpe_proxy": g["sharpe_proxy"].mean(),
                "median_sharpe_proxy": g["sharpe_proxy"].median(),
                "mean_mdd_proxy": g["mdd_proxy"].mean(),
                "worst_mdd_proxy": g["mdd_proxy"].max(),
                "mean_turnover_proxy": g["turnover_proxy"].mean(),
                "mean_fees": g["fees"].mean(),
                "mean_rebalances": g["rebalances"].mean(),
                "median_rank_score_proxy": g["rank_score_proxy"].median(),
                "mean_rank_score_proxy": g["rank_score_proxy"].mean(),
                "better_rank_vs_equal80": better_rank,
                "mean_return_delta_vs_equal80": (
                    aligned.loc[common, "return"]
                    - baseline.loc[common, "return"]
                ).mean(),
                "mean_sharpe_delta_vs_equal80": (
                    aligned.loc[common, "sharpe_proxy"]
                    - baseline.loc[common, "sharpe_proxy"]
                ).mean(),
                "mean_mdd_delta_vs_equal80": (
                    aligned.loc[common, "mdd_proxy"]
                    - baseline.loc[common, "mdd_proxy"]
                ).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["median_rank_score_proxy", "mean_rank_score_proxy"],
        ascending=[True, True],
    )


def self_check() -> None:
    score = np.linspace(-1, 1, 30)
    for tilt in (0.0, 0.08, 0.12, 0.16):
        w = target_from_score(score, tilt)
        assert w.shape == (30,)
        assert np.isfinite(w).all()
        assert (w >= 0.0).all()
        assert float(w.max()) <= 0.30 + 1e-12
        assert abs(float(w.sum()) - 0.80) < 1e-10

    prices = np.linspace(50.0, 100.0, 30)
    current = np.zeros(30)
    w = target_from_score(score, 0.12)
    shares, cash, gross, fee = solve_target_trade(
        1_000_000.0,
        current,
        prices,
        w,
    )
    post_nav = cash + float(np.sum(shares * prices))
    assert abs(
        post_nav - (1_000_000.0 - fee)
    ) < 1e-5
    assert gross > 0.0
    assert fee > 0.0

    nav = np.array([100.0, 101.0, 100.5, 102.0])
    assert math.isfinite(sharpe_proxy(nav))
    assert 0.0 <= max_drawdown(nav) < 1.0

    print("PHASE12_SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run fast internal tests without loading the dataset.",
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    root = find_project_root()
    dataset = (
        root
        / "data"
        / "official"
        / "hourly_market_data_2021_2026.parquet"
    )
    if not dataset.exists():
        raise SystemExit(
            f"ERROR: Official dataset missing: {dataset}"
        )

    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 12: WALK-FORWARD RIDGE ALPHA CHALLENGER ===")
    print("Competition submission: DISABLED")
    print("Primary source: organizer historical dataset")
    print("Model: rolling pooled ridge regression")
    print("Target: 5-day cross-sectional excess return")
    print(
        "Execution simulation: R1 09:30 open; "
        "R2-R7 are HOLD in this slow-model experiment"
    )
    print(
        "Risk/Sharpe timing: organizer hourly-grid proxy, "
        "not exact live R2-R7 execution timing"
    )
    print()

    raw = pd.read_parquet(dataset)

    (
        dates,
        open_0930,
        close,
        high,
        low,
        volume,
        endpoint_close,
        sector_map,
    ) = build_daily_dataset(raw)

    print(
        f"Complete standard days: {len(dates):,} "
        f"({dates.min().date()} -> {dates.max().date()})"
    )

    X, y, feature_names = build_features(
        close=close,
        high=high,
        low=low,
        volume=volume,
        sector_map=sector_map,
    )

    print(f"Features: {len(feature_names)}")
    print(", ".join(feature_names))

    scores, beta_norm = walk_forward_scores(X, y)
    valid_signal = np.isfinite(scores).all(axis=1)

    # OOS information coefficient diagnostic.
    ic_rows = []
    for i in np.flatnonzero(valid_signal):
        if not np.isfinite(y[i]).all():
            continue
        pred = pd.Series(scores[i])
        target = pd.Series(y[i])
        ic = pred.corr(target, method="spearman")
        if pd.notna(ic):
            ic_rows.append(float(ic))

    if not ic_rows:
        raise SystemExit(
            "ERROR: No valid OOS information-coefficient observations."
        )

    print()
    print("=== OOS SIGNAL DIAGNOSTIC ===")
    print(f"OOS signal dates: {int(valid_signal.sum()):,}")
    print(f"Mean Spearman IC:   {np.mean(ic_rows): .4f}")
    print(f"Median Spearman IC: {np.median(ic_rows): .4f}")
    print(
        "Positive IC fraction: "
        f"{np.mean(np.asarray(ic_rows) > 0):.4f}"
    )

    open_np = open_0930.to_numpy(dtype=float)
    endpoints_np = [
        endpoint_close[t].to_numpy(dtype=float)
        for t in STANDARD_TIMES
    ]

    eligible_starts = []
    for start_idx in range(1, len(dates) - WINDOW_DAYS + 1):
        if start_idx % WINDOW_STEP != 0:
            continue
        signal_slice = scores[
            start_idx - 1 : start_idx + WINDOW_DAYS - 1
        ]
        if np.isfinite(signal_slice).all():
            eligible_starts.append(start_idx)

    if len(eligible_starts) < 100:
        raise SystemExit(
            f"ERROR: Only {len(eligible_starts)} eligible OOS windows; "
            "expected at least 100."
        )

    print()
    print(f"15-day OOS rolling windows: {len(eligible_starts)}")
    print(
        f"Window span: {dates[eligible_starts[0]].date()} "
        f"through {dates[eligible_starts[-1] + WINDOW_DAYS - 1].date()}"
    )

    results = []
    for w_idx, start_idx in enumerate(eligible_starts, start=1):
        window_rows = []
        for spec in STRATEGIES:
            metrics = simulate_window(
                start_idx=start_idx,
                spec=spec,
                scores=scores,
                open_0930=open_np,
                endpoint_arrays=endpoints_np,
            )
            row = {
                "window_id": w_idx,
                "start_date": str(dates[start_idx].date()),
                "end_date": str(
                    dates[start_idx + WINDOW_DAYS - 1].date()
                ),
                "strategy": spec.name,
                **metrics,
            }
            window_rows.append(row)

        rank_within_window(window_rows)
        results.extend(window_rows)

        if (
            w_idx % 25 == 0
            or w_idx == len(eligible_starts)
        ):
            print(
                f"Completed window {w_idx}/{len(eligible_starts)} "
                f"({dates[start_idx].date()} -> "
                f"{dates[start_idx + WINDOW_DAYS - 1].date()})"
            )

    results_df = pd.DataFrame(results)
    summary = summarize(results_df)

    summary_path = (
        artifacts / "phase12_ml_challenger_summary.csv"
    )
    results_path = (
        artifacts / "phase12_ml_challenger_windows.csv"
    )
    json_path = (
        artifacts / "phase12_ml_challenger.json"
    )

    summary.to_csv(summary_path, index=False)
    results_df.to_csv(results_path, index=False)

    payload = {
        "model": {
            "type": "rolling pooled ridge regression",
            "target": "5-day cross-sectional excess return",
            "ridge_lambda": RIDGE_LAMBDA,
            "min_train_days": MIN_TRAIN_DAYS,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "features": feature_names,
        },
        "signal_diagnostic": {
            "oos_signal_dates": int(valid_signal.sum()),
            "mean_spearman_ic": float(np.mean(ic_rows)),
            "median_spearman_ic": float(np.median(ic_rows)),
            "positive_ic_fraction": float(
                np.mean(np.asarray(ic_rows) > 0)
            ),
            "median_beta_norm": float(
                np.nanmedian(beta_norm)
            ),
        },
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "strategy",
        "windows",
        "mean_return",
        "median_return",
        "positive_fraction",
        "mean_sharpe_proxy",
        "mean_mdd_proxy",
        "worst_mdd_proxy",
        "mean_turnover_proxy",
        "mean_rebalances",
        "median_rank_score_proxy",
        "better_rank_vs_equal80",
        "mean_return_delta_vs_equal80",
        "mean_sharpe_delta_vs_equal80",
        "mean_mdd_delta_vs_equal80",
    ]

    print()
    print("=== PHASE 12 ML CHALLENGER SUMMARY ===")
    print(
        summary[display_cols].to_string(
            index=False,
            formatters={
                c: (lambda x: f"{x:.4f}")
                for c in display_cols
                if c not in ("strategy", "windows")
            },
        )
    )

    print()
    print("=== KEEP / REJECT INTERPRETATION ===")
    baseline = summary[
        summary["strategy"] == "equal30_80_hold"
    ].iloc[0]
    challengers = summary[
        summary["strategy"] != "equal30_80_hold"
    ].copy()

    promotable = challengers[
        (challengers["better_rank_vs_equal80"] >= 0.55)
        & (
            challengers["mean_return_delta_vs_equal80"]
            >= -0.0005
        )
        & (
            challengers["mean_sharpe_delta_vs_equal80"]
            >= 0.0
        )
        & (
            challengers["mean_mdd_delta_vs_equal80"]
            <= 0.005
        )
        & (
            challengers["mean_turnover_proxy"]
            <= 0.05
        )
    ]

    if promotable.empty:
        print(
            "NO PROMOTION: no ridge challenger clears the predefined "
            "historical OOS promotion gate. Keep equal30_80_hold for live use."
        )
    else:
        best = promotable.sort_values(
            ["median_rank_score_proxy", "mean_rank_score_proxy"],
            ascending=[True, True],
        ).iloc[0]
        print(
            "PROMOTION CANDIDATE: "
            f"{best['strategy']} clears the predefined historical OOS gate."
        )
        print(
            "Do NOT deploy automatically yet; next step is live-data "
            "integration + shadow comparison + exact submission safety review."
        )

    print()
    print(f"Summary CSV: {summary_path}")
    print(f"Window CSV:  {results_path}")
    print(f"JSON:        {json_path}")
    print()
    print("PHASE 12 COMPLETE. NO COMPETITION SUBMISSION WAS MADE.")


if __name__ == "__main__":
    main()
