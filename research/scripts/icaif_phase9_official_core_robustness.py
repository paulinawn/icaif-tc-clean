from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd


PARQUET_NAME = "hourly_market_data_2021_2026.parquet"
FEE_RATE = 0.001
INITIAL_NAV = 1_000_000.0

STANDARD_TIMES = [
    time(9, 30),
    time(10, 0),
    time(11, 0),
    time(12, 0),
    time(13, 0),
    time(14, 0),
    time(15, 0),
]

WINDOW_DAYS = 15
WINDOW_STEP = 5
REGIME_LOOKBACK_DAYS = 60


@dataclass(frozen=True)
class StrategySpec:
    name: str
    mode: str
    gross: float | None = None


STRATEGIES = [
    StrategySpec("cash_hold", "fixed", 0.00),
    StrategySpec("equal30_50_hold", "fixed", 0.50),
    StrategySpec("equal30_60_hold", "fixed", 0.60),
    StrategySpec("equal30_70_hold", "fixed", 0.70),
    StrategySpec("equal30_80_hold", "fixed", 0.80),
    StrategySpec("equal30_90_hold", "fixed", 0.90),
    StrategySpec("equal30_100_hold", "fixed", 1.00),
    StrategySpec("regime_60_80_100_hold", "regime_a"),
    StrategySpec("regime_50_80_90_hold", "regime_b"),
]


def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]

    for candidate in candidates:
        if (
            (candidate / "universe.json").exists()
            and (candidate / "data" / "official" / PARQUET_NAME).exists()
        ):
            return candidate.resolve()

    raise SystemExit(
        "\nERROR: Could not locate starter-kit with the official dataset. "
        "Run this command while PowerShell is in the starter-kit folder.\n"
    )


ROOT = find_project_root()
DATASET = ROOT / "data" / "official" / PARQUET_NAME
OUT_JSON = ROOT / "artifacts" / "phase9_official_core_robustness.json"
OUT_CSV = ROOT / "artifacts" / "phase9_official_core_summary.csv"
OUT_SPECIAL = ROOT / "artifacts" / "phase9_official_special_dates.csv"


def competition_sharpe(period_returns: list[float]) -> float:
    if len(period_returns) < 2:
        return 0.0

    sigma = stdev(period_returns)
    if sigma == 0.0 or not math.isfinite(sigma):
        return 0.0

    return 42.0 * mean(period_returns) / sigma


def max_drawdown(nav_path: list[float]) -> float:
    peak = nav_path[0]
    worst = 0.0

    for nav in nav_path:
        peak = max(peak, nav)
        drawdown = (peak - nav) / peak
        worst = max(worst, drawdown)

    return worst


def choose_gross(
    spec: StrategySpec,
    *,
    market60: float,
    breadth60: float,
) -> float:
    if spec.mode == "fixed":
        assert spec.gross is not None
        return float(spec.gross)

    if spec.mode == "regime_a":
        if market60 < 0.0 or breadth60 < 0.40:
            return 0.60
        if market60 > 0.05 and breadth60 > 0.65:
            return 1.00
        return 0.80

    if spec.mode == "regime_b":
        if market60 < -0.03 or breadth60 < 0.35:
            return 0.50
        if market60 > 0.04 and breadth60 > 0.60:
            return 0.90
        return 0.80

    raise ValueError(f"Unknown strategy mode: {spec.mode}")


def initial_allocation(
    *,
    gross: float,
    start_prices: pd.Series,
) -> tuple[dict[str, float], float, float]:
    if not 0.0 <= gross <= 1.0:
        raise ValueError(f"Gross exposure outside [0,1]: {gross}")

    symbols = list(start_prices.index)

    if gross == 0.0:
        return (
            {symbol: 0.0 for symbol in symbols},
            INITIAL_NAV,
            0.0,
        )

    each_weight = gross / len(symbols)
    if each_weight > 0.30 + 1e-12:
        raise ValueError("Per-stock cap exceeded.")

    if (start_prices <= 0).any():
        raise ValueError("Non-positive start price.")

    # Starting from all cash:
    # post_nav = pre_nav - fee * gross_notional
    # gross_notional = gross * post_nav
    # => post_nav = pre_nav / (1 + fee*gross)
    post_nav = INITIAL_NAV / (1.0 + FEE_RATE * gross)
    gross_notional = gross * post_nav

    target_value_each = each_weight * post_nav

    shares = {
        symbol: target_value_each / float(start_prices[symbol])
        for symbol in symbols
    }

    cash = post_nav - target_value_each * len(symbols)

    if cash < -1e-6:
        raise ValueError(f"Negative cash after deployment: {cash}")

    return shares, cash, gross_notional


def self_check() -> None:
    # Core metric sanity.
    assert competition_sharpe([0.0, 0.0]) == 0.0
    assert abs(max_drawdown([100.0, 110.0, 99.0]) - 0.10) < 1e-12

    # Exposure rule geometry.
    for gross in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        prices = pd.Series({"A": 100.0, "B": 50.0})
        shares, cash, gross_notional = initial_allocation(
            gross=gross,
            start_prices=prices,
        )
        assert len(shares) == 2
        assert cash >= -1e-9
        assert gross_notional >= 0.0

    # Regime rule outputs are legal.
    for spec in STRATEGIES:
        if spec.mode == "fixed":
            continue
        for market60, breadth60 in [
            (-0.10, 0.20),
            (0.00, 0.50),
            (0.10, 0.80),
        ]:
            gross = choose_gross(
                spec,
                market60=market60,
                breadth60=breadth60,
            )
            assert 0.0 <= gross <= 1.0


def load_and_prepare():
    print(f"Dataset: {DATASET}")

    df = pd.read_parquet(DATASET)

    required = {
        "timestamp_et",
        "trading_date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "sector_group",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"ERROR: Missing required columns: {missing}"
        )

    df = df.copy()
    df["timestamp_et"] = pd.to_datetime(
        df["timestamp_et"],
        errors="raise",
    )
    df["trading_date"] = pd.to_datetime(
        df["trading_date"],
        errors="raise",
    ).dt.date
    df["ticker"] = df["ticker"].astype(str)
    df["time_et"] = df["timestamp_et"].dt.time

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Corporate-action adjustment factor. This makes OHLC internally
    # consistent with adj_close when the organizer file contains adjusted
    # close values that differ from raw close.
    factor = df["adj_close"] / df["close"]

    if (
        factor.isna().any()
        or (~np.isfinite(factor.astype(float))).any()
        or (factor <= 0).any()
    ):
        raise SystemExit(
            "ERROR: Invalid adj_close/close adjustment factor."
        )

    df["adj_open"] = df["open"] * factor
    df["adj_bar_close"] = df["adj_close"]

    adjusted_rows = int(
        (~np.isclose(
            df["adj_close"].astype(float),
            df["close"].astype(float),
            rtol=1e-12,
            atol=1e-12,
        )).sum()
    )

    standard_set = set(STANDARD_TIMES)
    special = df.loc[
        ~df["time_et"].isin(standard_set),
        [
            "trading_date",
            "timestamp_et",
            "ticker",
            "time_et",
        ],
    ].copy()

    special_summary = (
        special.groupby(
            ["trading_date", "time_et"],
            dropna=False,
        )
        .agg(
            rows=("ticker", "size"),
            symbols=("ticker", "nunique"),
        )
        .reset_index()
        .sort_values(["trading_date", "time_et"])
    )

    OUT_SPECIAL.parent.mkdir(parents=True, exist_ok=True)
    special_summary.to_csv(OUT_SPECIAL, index=False)

    standard = df.loc[
        df["time_et"].isin(standard_set)
    ].copy()

    # A standard research date must have all 30 symbols and exactly the seven
    # expected organizer bar timestamps for each symbol.
    per_date_symbol = (
        standard.groupby(["trading_date", "ticker"])["time_et"]
        .nunique()
    )

    symbols_per_date = (
        standard.groupby("trading_date")["ticker"]
        .nunique()
    )

    good_dates = []

    for date_value in sorted(standard["trading_date"].unique()):
        if int(symbols_per_date.get(date_value, 0)) != 30:
            continue

        counts = per_date_symbol.loc[date_value]
        if len(counts) != 30:
            continue

        if not bool((counts == len(STANDARD_TIMES)).all()):
            continue

        good_dates.append(date_value)

    if len(good_dates) < 500:
        raise SystemExit(
            f"ERROR: Only {len(good_dates)} complete standard days."
        )

    standard = standard.loc[
        standard["trading_date"].isin(good_dates)
    ].copy()

    symbols = sorted(standard["ticker"].unique())
    if len(symbols) != 30:
        raise SystemExit(
            f"ERROR: Expected 30 symbols, found {len(symbols)}."
        )

    # Exact R1 historical execution proxy: 09:30 organizer bar open.
    r1_open = (
        standard.loc[
            standard["time_et"] == time(9, 30)
        ]
        .pivot(
            index="trading_date",
            columns="ticker",
            values="adj_open",
        )
        .reindex(index=good_dates, columns=symbols)
    )

    # Organizer bar-close grid used only for hourly-grid risk/Sharpe proxies.
    bar_close = (
        standard.pivot_table(
            index=["trading_date", "time_et"],
            columns="ticker",
            values="adj_bar_close",
            aggfunc="first",
        )
        .reindex(columns=symbols)
        .sort_index()
    )

    daily_close = (
        standard.loc[
            standard["time_et"] == time(15, 0)
        ]
        .pivot(
            index="trading_date",
            columns="ticker",
            values="adj_bar_close",
        )
        .reindex(index=good_dates, columns=symbols)
    )

    if (
        r1_open.isna().any().any()
        or daily_close.isna().any().any()
    ):
        raise SystemExit(
            "ERROR: Missing values remain after complete-date filtering."
        )

    print(f"Rows loaded: {len(df):,}")
    print(f"Adjusted-close differs from close on: {adjusted_rows:,} rows")
    print(f"Rare/off-grid rows: {len(special):,}")
    print(f"Rare/off-grid date/time groups: {len(special_summary):,}")
    print(f"Complete standard research days: {len(good_dates):,}")
    print(
        f"Standard sample: {good_dates[0]} through {good_dates[-1]}"
    )
    print(f"Special-date report: {OUT_SPECIAL}")

    return (
        symbols,
        good_dates,
        r1_open,
        bar_close,
        daily_close,
        adjusted_rows,
        special_summary,
    )


def regime_at_start(
    *,
    daily_close: pd.DataFrame,
    good_dates: list,
    start_index: int,
) -> tuple[float, float]:
    if start_index < REGIME_LOOKBACK_DAYS:
        raise ValueError("Insufficient regime lookback.")

    start_date = good_dates[start_index]
    prior_date = good_dates[start_index - 1]
    lag_date = good_dates[start_index - 1 - REGIME_LOOKBACK_DAYS]

    last = daily_close.loc[prior_date]
    lag = daily_close.loc[lag_date]

    ret60 = last / lag - 1.0

    market60 = float(ret60.mean())
    breadth60 = float((ret60 > 0.0).mean())

    if not math.isfinite(market60):
        raise ValueError("Non-finite market60.")

    if not 0.0 <= breadth60 <= 1.0:
        raise ValueError("Invalid breadth60.")

    return market60, breadth60


def simulate_window(
    *,
    spec: StrategySpec,
    window_dates: list,
    start_index: int,
    symbols: list[str],
    good_dates: list,
    r1_open: pd.DataFrame,
    bar_close: pd.DataFrame,
    daily_close: pd.DataFrame,
) -> dict:
    market60, breadth60 = regime_at_start(
        daily_close=daily_close,
        good_dates=good_dates,
        start_index=start_index,
    )

    gross = choose_gross(
        spec,
        market60=market60,
        breadth60=breadth60,
    )

    first_date = window_dates[0]
    start_prices = r1_open.loc[first_date, symbols]

    shares, cash, gross_notional = initial_allocation(
        gross=gross,
        start_prices=start_prices,
    )

    nav_path = [INITIAL_NAV]
    hourly_proxy_navs = []

    for date_value in window_dates:
        for tod in STANDARD_TIMES:
            prices = bar_close.loc[(date_value, tod), symbols]

            if prices.isna().any():
                raise ValueError(
                    f"Missing bar-close prices on {date_value} {tod}."
                )

            nav = cash + sum(
                shares[symbol] * float(prices[symbol])
                for symbol in symbols
            )

            if not math.isfinite(nav) or nav <= 0.0:
                raise ValueError(
                    f"Invalid NAV on {date_value} {tod}: {nav}"
                )

            hourly_proxy_navs.append(nav)
            nav_path.append(nav)

    if len(hourly_proxy_navs) != WINDOW_DAYS * len(STANDARD_TIMES):
        raise AssertionError(
            "Expected exactly 105 hourly-grid endpoints."
        )

    # This mirrors the competition's K=105 count for a normal 15-day phase,
    # but the historical endpoints are the organizer dataset's hourly grid,
    # not the exact 10:30/11:30/... live execution boundaries.
    period_returns = []

    previous_nav = INITIAL_NAV
    for nav in hourly_proxy_navs:
        period_returns.append(nav / previous_nav - 1.0)
        previous_nav = nav

    final_nav = hourly_proxy_navs[-1]

    turnover_mean = (
        (gross_notional / INITIAL_NAV)
        / len(hourly_proxy_navs)
    )

    return {
        "strategy": spec.name,
        "gross": gross,
        "market60_at_start": market60,
        "breadth60_at_start": breadth60,
        "return": final_nav / INITIAL_NAV - 1.0,
        "sharpe_hourly_grid_proxy": competition_sharpe(period_returns),
        "max_drawdown_hourly_grid_proxy": max_drawdown(nav_path),
        "turnover_mean_proxy": turnover_mean,
        "final_nav": final_nav,
    }


def add_window_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()

    x["rank_return"] = x.groupby("window")["return"].rank(
        ascending=False,
        method="average",
    )
    x["rank_sharpe"] = x.groupby("window")[
        "sharpe_hourly_grid_proxy"
    ].rank(
        ascending=False,
        method="average",
    )
    x["rank_mdd"] = x.groupby("window")[
        "max_drawdown_hourly_grid_proxy"
    ].rank(
        ascending=True,
        method="average",
    )
    x["rank_turnover"] = x.groupby("window")[
        "turnover_mean_proxy"
    ].rank(
        ascending=True,
        method="average",
    )

    x["rank_score_proxy"] = x[
        [
            "rank_return",
            "rank_sharpe",
            "rank_mdd",
            "rank_turnover",
        ]
    ].mean(axis=1)

    return x


def main() -> None:
    self_check()

    if "--self-check" in sys.argv:
        print("PHASE9_SELF_CHECK_OK")
        print(f"Project root: {ROOT}")
        return

    print(
        "=== ICAIF PHASE 9: OFFICIAL-DATA CORE ROBUSTNESS ==="
    )
    print("Competition submission: DISABLED")
    print("Primary source: organizer historical dataset")
    print(
        "Exact historical execution used: R1 09:30 open only"
    )
    print(
        "R2-R7 risk/Sharpe timing: organizer hourly-grid proxy, "
        "NOT claimed as exact live execution timing"
    )
    print()

    (
        symbols,
        good_dates,
        r1_open,
        bar_close,
        daily_close,
        adjusted_rows,
        special_summary,
    ) = load_and_prepare()

    first_start_index = REGIME_LOOKBACK_DAYS + 1

    windows: list[tuple[int, list]] = []

    for start_index in range(
        first_start_index,
        len(good_dates) - WINDOW_DAYS + 1,
        WINDOW_STEP,
    ):
        window_dates = good_dates[
            start_index : start_index + WINDOW_DAYS
        ]
        windows.append((start_index, window_dates))

    if len(windows) < 100:
        raise SystemExit(
            f"ERROR: Expected a large multi-year sample; "
            f"only {len(windows)} windows available."
        )

    print(f"15-day rolling windows: {len(windows)}")
    print(
        f"Window span: {windows[0][1][0]} through "
        f"{windows[-1][1][-1]}"
    )
    print()

    rows = []

    for window_number, (start_index, window_dates) in enumerate(
        windows,
        start=1,
    ):
        for spec in STRATEGIES:
            result = simulate_window(
                spec=spec,
                window_dates=window_dates,
                start_index=start_index,
                symbols=symbols,
                good_dates=good_dates,
                r1_open=r1_open,
                bar_close=bar_close,
                daily_close=daily_close,
            )

            result["window"] = window_number
            result["start_date"] = str(window_dates[0])
            result["end_date"] = str(window_dates[-1])
            result["start_year"] = int(window_dates[0].year)
            rows.append(result)

        if (
            window_number % 25 == 0
            or window_number == len(windows)
        ):
            print(
                f"Completed window {window_number}/{len(windows)} "
                f"({window_dates[0]} -> {window_dates[-1]})"
            )

    results = add_window_ranks(pd.DataFrame(rows))

    baseline_name = "equal30_80_hold"

    baseline = (
        results.loc[
            results["strategy"] == baseline_name,
            [
                "window",
                "return",
                "sharpe_hourly_grid_proxy",
                "max_drawdown_hourly_grid_proxy",
                "turnover_mean_proxy",
                "rank_score_proxy",
            ],
        ]
        .set_index("window")
        .add_prefix("baseline_")
    )

    summary_rows = []

    for spec in STRATEGIES:
        x = results.loc[
            results["strategy"] == spec.name
        ].copy()

        joined = x.set_index("window").join(baseline)

        summary_rows.append(
            {
                "strategy": spec.name,
                "windows": len(x),
                "mean_gross": float(x["gross"].mean()),
                "mean_return": float(x["return"].mean()),
                "median_return": float(x["return"].median()),
                "p10_return": float(x["return"].quantile(0.10)),
                "p90_return": float(x["return"].quantile(0.90)),
                "positive_window_fraction": float(
                    (x["return"] > 0.0).mean()
                ),
                "mean_sharpe_proxy": float(
                    x["sharpe_hourly_grid_proxy"].mean()
                ),
                "median_sharpe_proxy": float(
                    x["sharpe_hourly_grid_proxy"].median()
                ),
                "mean_mdd_proxy": float(
                    x["max_drawdown_hourly_grid_proxy"].mean()
                ),
                "worst_mdd_proxy": float(
                    x["max_drawdown_hourly_grid_proxy"].max()
                ),
                "mean_turnover_proxy": float(
                    x["turnover_mean_proxy"].mean()
                ),
                "median_rank_score_proxy": float(
                    x["rank_score_proxy"].median()
                ),
                "mean_rank_score_proxy": float(
                    x["rank_score_proxy"].mean()
                ),
                "better_rank_vs_equal80": float(
                    (
                        joined["rank_score_proxy"]
                        < joined["baseline_rank_score_proxy"]
                    ).mean()
                ),
                "higher_return_vs_equal80": float(
                    (
                        joined["return"]
                        > joined["baseline_return"]
                    ).mean()
                ),
                "higher_sharpe_vs_equal80": float(
                    (
                        joined["sharpe_hourly_grid_proxy"]
                        > joined["baseline_sharpe_hourly_grid_proxy"]
                    ).mean()
                ),
                "lower_mdd_vs_equal80": float(
                    (
                        joined["max_drawdown_hourly_grid_proxy"]
                        < joined["baseline_max_drawdown_hourly_grid_proxy"]
                    ).mean()
                ),
                "mean_return_delta_vs_equal80": float(
                    (
                        joined["return"]
                        - joined["baseline_return"]
                    ).mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "median_rank_score_proxy",
            "mean_rank_score_proxy",
            "mean_return",
        ],
        ascending=[True, True, False],
    )

    # Year-by-year robustness table.
    yearly = (
        results.groupby(["strategy", "start_year"])
        .agg(
            windows=("window", "size"),
            mean_return=("return", "mean"),
            median_return=("return", "median"),
            positive_fraction=("return", lambda s: float((s > 0).mean())),
            mean_sharpe_proxy=("sharpe_hourly_grid_proxy", "mean"),
            mean_mdd_proxy=("max_drawdown_hourly_grid_proxy", "mean"),
        )
        .reset_index()
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifact_type": (
            "OFFICIAL_HISTORICAL_RESEARCH_ONLY_NO_SUBMISSION"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(DATASET),
        "source_kind": "organizer-provided historical dataset",
        "window_days": WINDOW_DAYS,
        "window_step_days": WINDOW_STEP,
        "window_count": len(windows),
        "fee_rate": FEE_RATE,
        "baseline": baseline_name,
        "adjusted_rows": adjusted_rows,
        "special_date_time_groups": len(special_summary),
        "timing_note": (
            "R1 09:30 open is directly represented. "
            "R2-R7 live execution times are not directly represented; "
            "Sharpe/MDD use organizer hourly bar endpoints as an explicit proxy."
        ),
        "summary": summary.to_dict(orient="records"),
        "yearly": yearly.to_dict(orient="records"),
        "window_results": results.to_dict(orient="records"),
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    summary.to_csv(OUT_CSV, index=False)

    print()
    print("=== PHASE 9 OFFICIAL-DATA SUMMARY ===")

    display = summary.copy()

    four_decimals = [
        "mean_gross",
        "mean_return",
        "median_return",
        "p10_return",
        "p90_return",
        "positive_window_fraction",
        "mean_mdd_proxy",
        "worst_mdd_proxy",
        "mean_turnover_proxy",
        "better_rank_vs_equal80",
        "higher_return_vs_equal80",
        "higher_sharpe_vs_equal80",
        "lower_mdd_vs_equal80",
        "mean_return_delta_vs_equal80",
    ]

    for col in four_decimals:
        display[col] = display[col].map(
            lambda value: f"{value:.4f}"
        )

    for col in [
        "mean_sharpe_proxy",
        "median_sharpe_proxy",
        "median_rank_score_proxy",
        "mean_rank_score_proxy",
    ]:
        display[col] = display[col].map(
            lambda value: f"{value:.2f}"
        )

    print(display.to_string(index=False))

    print()
    print("=== YEAR-BY-YEAR: 80% VS 100% VS REGIME ===")

    focus = yearly.loc[
        yearly["strategy"].isin(
            [
                "equal30_80_hold",
                "equal30_100_hold",
                "regime_60_80_100_hold",
                "regime_50_80_90_hold",
            ]
        )
    ].copy()

    for col in [
        "mean_return",
        "median_return",
        "positive_fraction",
        "mean_mdd_proxy",
    ]:
        focus[col] = focus[col].map(
            lambda value: f"{value:.4f}"
        )

    focus["mean_sharpe_proxy"] = focus[
        "mean_sharpe_proxy"
    ].map(lambda value: f"{value:.2f}")

    print(
        focus.sort_values(
            ["start_year", "strategy"]
        ).to_string(index=False)
    )

    print()
    print(f"JSON artifact: {OUT_JSON}")
    print(f"CSV summary:   {OUT_CSV}")
    print(f"Special dates: {OUT_SPECIAL}")
    print()
    print(
        "PHASE 9 COMPLETE. NO COMPETITION SUBMISSION WAS MADE."
    )


if __name__ == "__main__":
    main()
