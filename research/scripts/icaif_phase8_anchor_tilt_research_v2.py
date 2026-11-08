from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, time, timezone
from pathlib import Path
from statistics import mean, stdev
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
OUT_JSON = ROOT / "artifacts" / "phase8_anchor_tilt_research.json"
OUT_CSV = ROOT / "artifacts" / "phase8_anchor_tilt_summary.csv"

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

BARS_PER_DAY = 26
BARS_1D = BARS_PER_DAY
BARS_5D = 5 * BARS_PER_DAY
BARS_20D = 20 * BARS_PER_DAY
MIN_HISTORY = BARS_20D + 2

SYMBOL_TO_SECTOR = {
    symbol: sector
    for sector, symbols in SECTORS.items()
    for symbol in symbols
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    signal: str
    base_gross: float
    tilt_budget: float
    rebalance_rounds: tuple[int, ...]
    no_trade_threshold: float
    partial_step: float = 1.0
    dynamic_gross: bool = False


STRATEGIES = [
    StrategySpec(
        "cash_hold",
        "none",
        0.0,
        0.0,
        (),
        999.0,
    ),
    StrategySpec(
        "equal30_80_r1",
        "none",
        0.80,
        0.0,
        (1,),
        0.10,
    ),
    StrategySpec(
        "equal30_100_r1",
        "none",
        1.00,
        0.0,
        (1,),
        0.10,
    ),
    StrategySpec(
        "slowtilt10_80_r1",
        "slow",
        0.80,
        0.10,
        (1,),
        0.05,
    ),
    StrategySpec(
        "slowtilt20_80_r1",
        "slow",
        0.80,
        0.20,
        (1,),
        0.10,
    ),
    StrategySpec(
        "slowtilt20_80_r1r4",
        "slow",
        0.80,
        0.20,
        (1, 4),
        0.10,
    ),
    StrategySpec(
        "slowtilt20_80_all7",
        "slow",
        0.80,
        0.20,
        (1, 2, 3, 4, 5, 6, 7),
        0.10,
    ),
    StrategySpec(
        "slowtilt20_100_r1",
        "slow",
        1.00,
        0.20,
        (1,),
        0.10,
    ),
    StrategySpec(
        "slowtilt15_60_r1",
        "slow",
        0.60,
        0.15,
        (1,),
        0.10,
    ),
    StrategySpec(
        "riskslow20_80_r1",
        "risk_slow",
        0.80,
        0.20,
        (1,),
        0.10,
    ),
    StrategySpec(
        "riskslow20_80_r1r4",
        "risk_slow",
        0.80,
        0.20,
        (1, 4),
        0.10,
    ),
    StrategySpec(
        "slowtilt20_80_r1_partial50",
        "slow",
        0.80,
        0.20,
        (1,),
        0.10,
        partial_step=0.50,
    ),
    StrategySpec(
        "regime_slow20_r1",
        "slow",
        0.80,
        0.20,
        (1,),
        0.10,
        dynamic_gross=True,
    ),
]


def validate_strategy_geometry() -> None:
    """Fail fast if any configured anchor/tilt geometry can go negative."""
    n_assets = len(OFFICIAL_SYMBOLS)
    n_side = 10

    for spec in STRATEGIES:
        if spec.name == "cash_hold":
            continue

        gross_values = [spec.base_gross]
        if spec.dynamic_gross:
            gross_values = [0.55, 0.75, 0.90]

        for gross in gross_values:
            base = gross / n_assets
            effective_tilt = spec.tilt_budget

            if spec.dynamic_gross:
                if spec.base_gross <= 0:
                    raise ValueError(
                        f"{spec.name}: invalid base_gross."
                    )
                effective_tilt *= gross / spec.base_gross

            subtract_each = effective_tilt / n_side
            top_weight = base + subtract_each
            bottom_weight = base - subtract_each

            if bottom_weight < -1e-12:
                raise ValueError(
                    f"{spec.name}: geometry invalid at gross={gross:.2f}; "
                    f"bottom weight={bottom_weight:.6f}."
                )

            if top_weight > 0.30 + 1e-12:
                raise ValueError(
                    f"{spec.name}: geometry breaches 30% cap at "
                    f"gross={gross:.2f}; top weight={top_weight:.6f}."
                )

            if gross > 1.0 + 1e-12:
                raise ValueError(
                    f"{spec.name}: gross exposure exceeds 1."
                )


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
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
        )
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
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )

    local = frame["timestamp"].dt.tz_convert(NY)
    frame["date_et"] = local.dt.date
    frame["time_et"] = local.dt.time

    close = frame.pivot(
        index="timestamp",
        columns="symbol",
        values="close",
    )
    open_ = frame.pivot(
        index="timestamp",
        columns="symbol",
        values="open",
    )

    close = close.reindex(
        columns=OFFICIAL_SYMBOLS,
    ).sort_index()
    open_ = open_.reindex(
        columns=OFFICIAL_SYMBOLS,
    ).sort_index()

    trading_dates = sorted(frame["date_et"].unique())

    close_rows = frame[
        frame["time_et"] == time(15, 45)
    ].copy()
    daily_close = (
        close_rows.pivot(
            index="date_et",
            columns="symbol",
            values="close",
        )
        .reindex(columns=OFFICIAL_SYMBOLS)
        .sort_index()
    )

    return close, open_, trading_dates, daily_close


def ts_et(d, t) -> pd.Timestamp:
    return pd.Timestamp(
        datetime.combine(d, t, tzinfo=NY)
    ).tz_convert("UTC")


def rank_signal(values: pd.Series) -> pd.Series:
    return 2.0 * values.rank(
        method="average",
        pct=True,
    ) - 1.0


def features_at(
    close: pd.DataFrame,
    deadline_utc: pd.Timestamp,
) -> pd.DataFrame:
    # Alpaca 15-minute bars are stamped at bar start.
    # A bar stamped T is only complete at T+15m.
    latest_safe_start = deadline_utc - pd.Timedelta(
        minutes=15
    )
    hist = close.loc[
        close.index <= latest_safe_start
    ]

    rows = []

    for symbol in OFFICIAL_SYMBOLS:
        s = hist[symbol].dropna()

        if len(s) < MIN_HISTORY:
            raise ValueError(
                f"{symbol}: only {len(s)} eligible bars; "
                f"need {MIN_HISTORY}"
            )

        last = float(s.iloc[-1])
        ret_1d = (
            last / float(s.iloc[-1 - BARS_1D]) - 1.0
        )
        ret_5d = (
            last / float(s.iloc[-1 - BARS_5D]) - 1.0
        )
        ret_20d = (
            last / float(s.iloc[-1 - BARS_20D]) - 1.0
        )

        one_bar = s.pct_change(fill_method=None)
        vol_5d = float(
            one_bar.tail(BARS_5D).std(ddof=1)
        )

        values = [ret_1d, ret_5d, ret_20d, vol_5d]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(
                f"{symbol}: non-finite feature"
            )

        rows.append(
            {
                "symbol": symbol,
                "sector": SYMBOL_TO_SECTOR[symbol],
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "vol_5d": vol_5d,
            }
        )

    f = pd.DataFrame(rows).set_index("symbol")

    f["sector_relative_20d"] = (
        f["ret_20d"]
        - f.groupby("sector")["ret_20d"].transform(
            "mean"
        )
    )

    f["rank_5d"] = rank_signal(f["ret_5d"])
    f["rank_20d"] = rank_signal(f["ret_20d"])
    f["rank_sector_20d"] = rank_signal(
        f["sector_relative_20d"]
    )
    f["rank_low_vol"] = -rank_signal(f["vol_5d"])

    f["slow_score"] = (
        0.30 * f["rank_5d"]
        + 0.50 * f["rank_20d"]
        + 0.20 * f["rank_sector_20d"]
    )

    f["risk_slow_score"] = (
        0.25 * f["rank_5d"]
        + 0.45 * f["rank_20d"]
        + 0.20 * f["rank_sector_20d"]
        + 0.10 * f["rank_low_vol"]
    )

    # Broad market regime information.
    f.attrs["breadth_positive_20d"] = float(
        (f["ret_20d"] > 0).mean()
    )
    f.attrs["mean_market_20d"] = float(
        f["ret_20d"].mean()
    )

    return f


def desired_gross(
    spec: StrategySpec,
    f: pd.DataFrame,
) -> float:
    if not spec.dynamic_gross:
        return spec.base_gross

    breadth = float(
        f.attrs["breadth_positive_20d"]
    )
    market_ret = float(
        f.attrs["mean_market_20d"]
    )

    if breadth >= 0.65 and market_ret > 0:
        return 0.90
    if breadth <= 0.40 or market_ret < -0.03:
        return 0.55
    return 0.75


def anchor_tilt_target(
    spec: StrategySpec,
    f: pd.DataFrame,
) -> dict[str, float]:
    if spec.name == "cash_hold":
        return {
            symbol: 0.0
            for symbol in OFFICIAL_SYMBOLS
        }

    gross = desired_gross(spec, f)
    base = gross / len(OFFICIAL_SYMBOLS)

    target = {
        symbol: base
        for symbol in OFFICIAL_SYMBOLS
    }

    if spec.tilt_budget <= 0:
        return target

    score_col = {
        "slow": "slow_score",
        "risk_slow": "risk_slow_score",
    }[spec.signal]

    ranked = (
        f[score_col]
        .sort_values(
            ascending=False,
            kind="stable",
        )
        .index.tolist()
    )

    n_side = 10
    top = ranked[:n_side]
    bottom = ranked[-n_side:]

    # For dynamic-gross strategies, scale the absolute tilt budget
    # proportionally with gross exposure. Example:
    # 20 percentage-point tilt at 80% gross -> 13.75 points at 55% gross.
    # This preserves the intended relative tilt strength while keeping
    # the diversified anchor long-only in risk-off regimes.
    effective_tilt_budget = spec.tilt_budget
    if spec.dynamic_gross:
        if spec.base_gross <= 0:
            raise ValueError(
                f"{spec.name}: base_gross must be positive "
                "for dynamic gross scaling."
            )
        effective_tilt_budget *= gross / spec.base_gross

    add_each = effective_tilt_budget / n_side
    subtract_each = effective_tilt_budget / n_side

    if base - subtract_each < -1e-12:
        raise ValueError(
            f"{spec.name}: effective tilt budget "
            f"{effective_tilt_budget:.6f} is too large "
            f"for gross exposure {gross:.6f}."
        )

    for symbol in top:
        target[symbol] += add_each

    for symbol in bottom:
        target[symbol] -= subtract_each

    # Guard numerical noise and competition cap.
    for symbol in OFFICIAL_SYMBOLS:
        if target[symbol] < -1e-12:
            raise ValueError(
                f"{spec.name}: negative target weight"
            )
        target[symbol] = max(
            0.0,
            float(target[symbol]),
        )
        if target[symbol] > 0.30 + 1e-12:
            raise ValueError(
                f"{spec.name}: 30% cap breach"
            )

    if sum(target.values()) > 1.0 + 1e-10:
        raise ValueError(
            f"{spec.name}: total exposure > 1"
        )

    return target


def current_weights(
    shares: dict[str, float],
    prices: dict[str, float],
    nav: float,
) -> dict[str, float]:
    return {
        symbol: (
            shares[symbol] * prices[symbol] / nav
        )
        for symbol in OFFICIAL_SYMBOLS
    }


def blend_target(
    current: dict[str, float],
    desired: dict[str, float],
    step: float,
    force_full: bool,
) -> dict[str, float]:
    if force_full or step >= 0.999999:
        return desired.copy()

    result = {
        symbol: (
            current[symbol]
            + step
            * (
                desired[symbol]
                - current[symbol]
            )
        )
        for symbol in OFFICIAL_SYMBOLS
    }

    # Partial rebalance may inherit small exposure drift.
    # It stays long-only and under the 30% cap.
    for symbol, weight in result.items():
        if weight < -1e-12:
            raise ValueError(
                f"Negative blended weight: {symbol}"
            )
        if weight > 0.30 + 1e-12:
            raise ValueError(
                f"Cap breach after blending: {symbol}"
            )

    if sum(result.values()) > 1.0 + 1e-10:
        raise ValueError(
            "Blended total exposure exceeds 1."
        )

    return result


def solve_fee_aware_rebalance(
    *,
    pre_nav: float,
    current_values: dict[str, float],
    target_weights: dict[str, float],
) -> tuple[
    float,
    float,
    dict[str, float],
]:
    post_nav = pre_nav

    for _ in range(100):
        target_values = {
            symbol: (
                target_weights[symbol]
                * post_nav
            )
            for symbol in OFFICIAL_SYMBOLS
        }

        gross_notional = sum(
            abs(
                target_values[symbol]
                - current_values[symbol]
            )
            for symbol in OFFICIAL_SYMBOLS
        )

        new_post_nav = (
            pre_nav
            - FEE * gross_notional
        )

        if (
            abs(new_post_nav - post_nav)
            < 1e-8
        ):
            post_nav = new_post_nav
            break

        post_nav = new_post_nav

    target_values = {
        symbol: (
            target_weights[symbol]
            * post_nav
        )
        for symbol in OFFICIAL_SYMBOLS
    }

    gross_notional = sum(
        abs(
            target_values[symbol]
            - current_values[symbol]
        )
        for symbol in OFFICIAL_SYMBOLS
    )

    return (
        post_nav,
        gross_notional,
        target_values,
    )


def competition_sharpe(
    returns: list[float],
) -> float:
    if len(returns) < 2:
        return 0.0

    sigma = stdev(returns)

    if sigma == 0 or not math.isfinite(sigma):
        return 0.0

    return 42.0 * mean(returns) / sigma


def max_drawdown(
    path: list[float],
) -> float:
    peak = path[0]
    worst = 0.0

    for value in path:
        peak = max(peak, value)
        worst = max(
            worst,
            (peak - value) / peak,
        )

    return worst


def simulate_window(
    spec: StrategySpec,
    dates: list,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    daily_close: pd.DataFrame,
) -> dict:
    shares = {
        symbol: 0.0
        for symbol in OFFICIAL_SYMBOLS
    }
    cash = INITIAL_NAV

    begin_navs: list[float] = []
    period_end_navs: list[float] = []
    turnover_ratios: list[float] = []
    valuation_path: list[float] = [
        INITIAL_NAV
    ]

    rebalance_count = 0
    hold_count = 0
    deployed = False

    for d in dates:
        for round_number, (
            exec_t,
            deadline_t,
        ) in enumerate(
            zip(
                EXECUTION_TIMES,
                DEADLINE_TIMES,
            ),
            start=1,
        ):
            exec_ts = ts_et(d, exec_t)
            deadline_ts = ts_et(
                d,
                deadline_t,
            )

            if exec_ts not in open_.index:
                raise ValueError(
                    f"Missing execution bar "
                    f"at {exec_ts}"
                )

            row = open_.loc[exec_ts]
            if row.isna().any():
                missing = (
                    row[
                        row.isna()
                    ]
                    .index
                    .tolist()
                )
                raise ValueError(
                    f"Missing prices "
                    f"{d} R{round_number}: "
                    f"{missing}"
                )

            prices = {
                symbol: float(row[symbol])
                for symbol in OFFICIAL_SYMBOLS
            }

            current_values = {
                symbol: (
                    shares[symbol]
                    * prices[symbol]
                )
                for symbol in OFFICIAL_SYMBOLS
            }

            pre_nav = (
                cash
                + sum(
                    current_values.values()
                )
            )

            if begin_navs:
                period_end_navs.append(
                    pre_nav
                )
                valuation_path.append(
                    pre_nav
                )

            begin_navs.append(pre_nav)

            if (
                spec.name == "cash_hold"
                or round_number
                not in spec.rebalance_rounds
            ):
                turnover_ratios.append(
                    0.0
                )
                hold_count += 1
                continue

            features = features_at(
                close,
                deadline_ts,
            )

            desired = anchor_tilt_target(
                spec,
                features,
            )

            current = current_weights(
                shares,
                prices,
                pre_nav,
            )

            force_full = not deployed
            target = blend_target(
                current,
                desired,
                spec.partial_step,
                force_full=force_full,
            )

            gross_weight_change = sum(
                abs(
                    target[symbol]
                    - current[symbol]
                )
                for symbol in OFFICIAL_SYMBOLS
            )

            if (
                not force_full
                and gross_weight_change
                < spec.no_trade_threshold
            ):
                turnover_ratios.append(
                    0.0
                )
                hold_count += 1
                continue

            (
                post_nav,
                gross_notional,
                target_values,
            ) = solve_fee_aware_rebalance(
                pre_nav=pre_nav,
                current_values=current_values,
                target_weights=target,
            )

            for symbol in OFFICIAL_SYMBOLS:
                shares[symbol] = (
                    target_values[symbol]
                    / prices[symbol]
                )

            cash = (
                post_nav
                - sum(
                    target_values.values()
                )
            )

            if cash < -1e-6:
                raise ValueError(
                    f"{spec.name}: "
                    f"negative cash {cash}"
                )

            turnover_ratios.append(
                gross_notional / pre_nav
            )

            rebalance_count += 1
            deployed = True

        if d not in daily_close.index:
            raise ValueError(
                f"Missing daily close for {d}"
            )

        close_row = daily_close.loc[d]

        if close_row.isna().any():
            missing = (
                close_row[
                    close_row.isna()
                ]
                .index
                .tolist()
            )
            raise ValueError(
                f"Missing close prices "
                f"{d}: {missing}"
            )

        close_nav = (
            cash
            + sum(
                shares[symbol]
                * float(
                    close_row[symbol]
                )
                for symbol
                in OFFICIAL_SYMBOLS
            )
        )

        valuation_path.append(
            close_nav
        )

    final_nav = valuation_path[-1]

    period_end_navs.append(
        final_nav
    )

    if (
        len(begin_navs)
        != len(period_end_navs)
    ):
        raise AssertionError(
            "Decision period mismatch."
        )

    returns = [
        end / begin - 1.0
        for begin, end in zip(
            begin_navs,
            period_end_navs,
        )
    ]

    return {
        "strategy": spec.name,
        "days": len(dates),
        "periods": len(returns),
        "return": (
            final_nav / INITIAL_NAV - 1.0
        ),
        "sharpe": competition_sharpe(
            returns
        ),
        "max_drawdown": max_drawdown(
            valuation_path
        ),
        "turnover": mean(
            turnover_ratios
        ),
        "rebalance_count": rebalance_count,
        "hold_count": hold_count,
        "final_nav": final_nav,
    }


def rank_window(
    df: pd.DataFrame,
) -> pd.DataFrame:
    ranked = df.copy()

    ranked["rank_return"] = (
        ranked["return"].rank(
            ascending=False,
            method="average",
        )
    )
    ranked["rank_sharpe"] = (
        ranked["sharpe"].rank(
            ascending=False,
            method="average",
        )
    )
    ranked["rank_mdd"] = (
        ranked["max_drawdown"].rank(
            ascending=True,
            method="average",
        )
    )
    ranked["rank_turnover"] = (
        ranked["turnover"].rank(
            ascending=True,
            method="average",
        )
    )

    ranked["rank_score"] = ranked[
        [
            "rank_return",
            "rank_sharpe",
            "rank_mdd",
            "rank_turnover",
        ]
    ].mean(axis=1)

    return ranked


def main() -> None:
    require_project()
    validate_strategy_geometry()

    print(
        "=== ICAIF PHASE 8: "
        "LOW-TURNOVER ANCHOR + ALPHA TILT ==="
    )
    print(
        "Competition submission: DISABLED"
    )
    print(
        "Strategy geometry self-test: PASSED"
    )
    print(
        "Research question: can small, "
        "persistent tilts beat the "
        "diversified equal-weight anchor?"
    )
    print()

    frame = fetch_or_load()
    (
        close,
        open_,
        trading_dates,
        daily_close,
    ) = build_market_views(frame)

    good_dates = []

    for d in trading_dates:
        ok = True

        for exec_t in EXECUTION_TIMES:
            exec_ts = ts_et(
                d,
                exec_t,
            )
            if (
                exec_ts not in open_.index
                or open_.loc[
                    exec_ts
                ].isna().any()
            ):
                ok = False
                break

        if (
            d not in daily_close.index
            or daily_close.loc[
                d
            ].isna().any()
        ):
            ok = False

        if ok:
            good_dates.append(d)

    if len(good_dates) < 120:
        raise SystemExit(
            "ERROR: not enough complete "
            "trading days."
        )

    # 20-day signal lookback needs more
    # warm-up than Phase 7.
    usable_dates = good_dates[25:]
    usable_dates = usable_dates[-160:]

    if len(usable_dates) < 100:
        raise SystemExit(
            "ERROR: insufficient usable "
            "days after warm-up."
        )

    windows = []

    for start_i in range(
        0,
        len(usable_dates) - 15 + 1,
        5,
    ):
        windows.append(
            usable_dates[
                start_i : start_i + 15
            ]
        )

    print(
        "Complete research trading days:",
        len(usable_dates),
    )
    print(
        "15-day walk-forward windows:",
        len(windows),
    )
    print(
        "Sample span:",
        usable_dates[0],
        "through",
        usable_dates[-1],
    )
    print()

    all_results = []

    for window_number, dates in enumerate(
        windows,
        start=1,
    ):
        rows = []

        for spec in STRATEGIES:
            result = simulate_window(
                spec,
                dates,
                close,
                open_,
                daily_close,
            )

            result["window"] = (
                window_number
            )
            result["start_date"] = str(
                dates[0]
            )
            result["end_date"] = str(
                dates[-1]
            )

            rows.append(result)

        ranked = rank_window(
            pd.DataFrame(rows)
        )
        all_results.append(ranked)

        if (
            window_number % 5 == 0
            or window_number
            == len(windows)
        ):
            print(
                f"Completed window "
                f"{window_number}/"
                f"{len(windows)} "
                f"({dates[0]} -> "
                f"{dates[-1]})"
            )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    baseline_name = "equal30_80_r1"

    baseline = (
        results.loc[
            results["strategy"]
            == baseline_name,
            [
                "window",
                "return",
                "sharpe",
                "max_drawdown",
                "turnover",
                "rank_score",
            ],
        ]
        .set_index("window")
        .add_prefix("baseline_")
    )

    summary_rows = []

    for spec in STRATEGIES:
        x = results.loc[
            results["strategy"]
            == spec.name
        ].copy()

        joined = x.set_index(
            "window"
        ).join(baseline)

        better_rank_fraction = float(
            (
                joined["rank_score"]
                < joined[
                    "baseline_rank_score"
                ]
            ).mean()
        )

        higher_return_fraction = float(
            (
                joined["return"]
                > joined[
                    "baseline_return"
                ]
            ).mean()
        )

        summary_rows.append(
            {
                "strategy": spec.name,
                "windows": len(x),
                "median_rank_score": float(
                    x["rank_score"].median()
                ),
                "mean_rank_score": float(
                    x["rank_score"].mean()
                ),
                "mean_return": float(
                    x["return"].mean()
                ),
                "median_return": float(
                    x["return"].median()
                ),
                "positive_window_fraction": float(
                    (x["return"] > 0).mean()
                ),
                "mean_sharpe": float(
                    x["sharpe"].mean()
                ),
                "mean_max_drawdown": float(
                    x[
                        "max_drawdown"
                    ].mean()
                ),
                "worst_max_drawdown": float(
                    x[
                        "max_drawdown"
                    ].max()
                ),
                "mean_turnover": float(
                    x["turnover"].mean()
                ),
                "mean_rebalances_15d": float(
                    x[
                        "rebalance_count"
                    ].mean()
                ),
                "better_rank_vs_equal80": (
                    better_rank_fraction
                ),
                "higher_return_vs_equal80": (
                    higher_return_fraction
                ),
                "mean_return_delta_vs_equal80": float(
                    (
                        joined["return"]
                        - joined[
                            "baseline_return"
                        ]
                    ).mean()
                ),
                "mean_turnover_delta_vs_equal80": float(
                    (
                        joined["turnover"]
                        - joined[
                            "baseline_turnover"
                        ]
                    ).mean()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "median_rank_score",
            "mean_rank_score",
            "mean_return",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "artifact_type": (
            "INTERIM_RESEARCH_ONLY_"
            "NO_SUBMISSION"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "research_question": (
            "Can a diversified anchor "
            "with slow alpha tilts and "
            "lower rebalance frequency "
            "improve on equal30_80_r1?"
        ),
        "data_source": (
            "Alpaca free IEX"
        ),
        "timeframe": TIMEFRAME,
        "window_days": 15,
        "window_step_days": 5,
        "window_count": len(windows),
        "baseline": baseline_name,
        "fee_rate": FEE,
        "limitations": [
            "Alpaca IEX is not the organizer execution feed.",
            "15-minute bar opens proxy scheduled execution prices.",
            "The local target-weight fee solver is a transparent proxy.",
            "Several candidates are compared, so multiple-testing risk remains.",
            "Organizer historical data should later confirm or reject these interim findings.",
        ],
        "summary": summary.to_dict(
            orient="records"
        ),
        "window_results": (
            results.to_dict(
                orient="records"
            )
        ),
    }

    OUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary.to_csv(
        OUT_CSV,
        index=False,
    )

    print()
    print(
        "=== PHASE 8 STRATEGY SUMMARY ==="
    )

    display = summary.copy()

    pct_cols = [
        "mean_return",
        "median_return",
        "positive_window_fraction",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_turnover",
        "better_rank_vs_equal80",
        "higher_return_vs_equal80",
        "mean_return_delta_vs_equal80",
        "mean_turnover_delta_vs_equal80",
    ]

    for col in pct_cols:
        display[col] = display[col].map(
            lambda x: f"{x:.4f}"
        )

    for col in [
        "median_rank_score",
        "mean_rank_score",
    ]:
        display[col] = display[col].map(
            lambda x: f"{x:.2f}"
        )

    display["mean_sharpe"] = (
        display["mean_sharpe"].map(
            lambda x: f"{x:.2f}"
        )
    )

    display[
        "mean_rebalances_15d"
    ] = display[
        "mean_rebalances_15d"
    ].map(
        lambda x: f"{x:.1f}"
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation columns:"
    )
    print(
        "- better_rank_vs_equal80: "
        "fraction of 15-day windows "
        "with a lower (better) "
        "4-metric rank score than "
        "equal30_80_r1."
    )
    print(
        "- higher_return_vs_equal80: "
        "fraction of windows with "
        "higher cumulative return "
        "than equal30_80_r1."
    )
    print(
        "- mean_return_delta_vs_equal80 "
        "and turnover delta show the "
        "economic tradeoff directly."
    )

    print()
    print(f"JSON artifact: {OUT_JSON}")
    print(f"CSV summary:   {OUT_CSV}")
    print()
    print(
        "PHASE 8 COMPLETE. "
        "NO COMPETITION SUBMISSION "
        "WAS MADE."
    )


if __name__ == "__main__":
    main()
