from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from our_agent.config import FEE_RATE, OFFICIAL_SYMBOLS
from our_agent.data.alpaca import AlpacaMarketDataClient
from our_agent.data.canonical import bars_payload_to_frame, regular_session_only

NY = ZoneInfo("America/New_York")

CACHE = ROOT / "data" / "cache" / "alpaca_iex_30m_2021_2026.csv"
OUT_JSON = ROOT / "artifacts" / "phase9_multiyear_exposure.json"
OUT_CSV = ROOT / "artifacts" / "phase9_multiyear_exposure_summary.csv"

START = "2021-01-01T00:00:00+00:00"
TIMEFRAME = "30Min"
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


@dataclass(frozen=True)
class StrategySpec:
    name: str
    mode: str
    gross: float | None = None


STRATEGIES = [
    StrategySpec("cash_hold", "fixed", 0.0),
    StrategySpec("equal30_50_hold", "fixed", 0.50),
    StrategySpec("equal30_60_hold", "fixed", 0.60),
    StrategySpec("equal30_70_hold", "fixed", 0.70),
    StrategySpec("equal30_80_hold", "fixed", 0.80),
    StrategySpec("equal30_90_hold", "fixed", 0.90),
    StrategySpec("equal30_100_hold", "fixed", 1.00),
    StrategySpec("regime_60_80_100_hold", "regime_a"),
    StrategySpec("regime_50_80_90_hold", "regime_b"),
]


def require_project() -> None:
    required = [
        ROOT / "universe.json",
        ROOT / "our_agent" / "data" / "alpaca.py",
        ROOT / ".env",
    ]
    missing = [
        str(path.relative_to(ROOT))
        for path in required
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "ERROR: Run this from starter-kit. Missing: "
            + repr(missing)
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

    end = datetime.now(timezone.utc)

    print("Fetching multi-year Alpaca IEX history...")
    print(f"Window: {START} -> {end.isoformat()}")
    print("This is a larger one-time research download.")

    with AlpacaMarketDataClient(feed="iex") as client:
        payload = client.historical_bars(
            symbols=OFFICIAL_SYMBOLS,
            start=START,
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


def prepare_views(frame: pd.DataFrame):
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True,
    )

    local = x["timestamp"].dt.tz_convert(NY)
    x["date_et"] = local.dt.date
    x["time_et"] = local.dt.time

    opens = (
        x.pivot(
            index="timestamp",
            columns="symbol",
            values="open",
        )
        .reindex(columns=OFFICIAL_SYMBOLS)
        .sort_index()
    )

    closes = (
        x.pivot(
            index="timestamp",
            columns="symbol",
            values="close",
        )
        .reindex(columns=OFFICIAL_SYMBOLS)
        .sort_index()
    )

    # A 30-minute bar stamped 15:30 covers 15:30-16:00.
    # Its close is our 16:00 daily-close proxy.
    daily_rows = x.loc[
        x["time_et"] == time(15, 30)
    ].copy()

    daily_close = (
        daily_rows.pivot(
            index="date_et",
            columns="symbol",
            values="close",
        )
        .reindex(columns=OFFICIAL_SYMBOLS)
        .sort_index()
    )

    trading_dates = sorted(
        set(x["date_et"])
    )

    return opens, closes, daily_close, trading_dates


def utc_ts(d, t) -> pd.Timestamp:
    return pd.Timestamp(
        datetime.combine(d, t, tzinfo=NY)
    ).tz_convert("UTC")


def complete_dates(
    opens: pd.DataFrame,
    daily_close: pd.DataFrame,
    trading_dates: list,
) -> list:
    good = []

    for d in trading_dates:
        ok = True

        for t in EXECUTION_TIMES:
            ts = utc_ts(d, t)
            if ts not in opens.index:
                ok = False
                break
            if opens.loc[ts].isna().any():
                ok = False
                break

        if d not in daily_close.index:
            ok = False
        elif daily_close.loc[d].isna().any():
            ok = False

        if ok:
            good.append(d)

    return good


def regime_features(
    daily_close: pd.DataFrame,
    start_date,
) -> dict[str, float]:
    history = daily_close.loc[
        daily_close.index < start_date
    ].dropna()

    if len(history) < 61:
        raise ValueError(
            "Need at least 61 prior complete daily closes."
        )

    last = history.iloc[-1]
    prev20 = history.iloc[-21]
    prev60 = history.iloc[-61]

    ret20 = last / prev20 - 1.0
    ret60 = last / prev60 - 1.0

    return {
        "market20": float(ret20.mean()),
        "market60": float(ret60.mean()),
        "breadth20": float((ret20 > 0).mean()),
        "breadth60": float((ret60 > 0).mean()),
    }


def choose_gross(
    spec: StrategySpec,
    features: dict[str, float],
) -> float:
    if spec.mode == "fixed":
        return float(spec.gross)

    if spec.mode == "regime_a":
        # Deliberately coarse three-state rule.
        if (
            features["market60"] < 0.0
            or features["breadth60"] < 0.40
        ):
            return 0.60

        if (
            features["market60"] > 0.05
            and features["breadth60"] > 0.65
        ):
            return 1.00

        return 0.80

    if spec.mode == "regime_b":
        if (
            features["market60"] < -0.03
            or features["breadth60"] < 0.35
        ):
            return 0.50

        if (
            features["market60"] > 0.04
            and features["breadth60"] > 0.60
        ):
            return 0.90

        return 0.80

    raise ValueError(f"Unknown mode: {spec.mode}")


def solve_initial_allocation(
    pre_nav: float,
    gross: float,
) -> tuple[float, float, dict[str, float]]:
    if not (0.0 <= gross <= 1.0):
        raise ValueError("gross must be in [0, 1]")

    if gross == 0.0:
        return pre_nav, 0.0, {
            symbol: 0.0
            for symbol in OFFICIAL_SYMBOLS
        }

    each = gross / len(OFFICIAL_SYMBOLS)

    if each > 0.30:
        raise ValueError("30% cap breach")

    # From all-cash, gross buy notional equals gross * post-fee NAV.
    # post_nav = pre_nav - fee * gross * post_nav
    post_nav = pre_nav / (1.0 + FEE * gross)
    gross_notional = gross * post_nav

    target_values = {
        symbol: each * post_nav
        for symbol in OFFICIAL_SYMBOLS
    }

    return post_nav, gross_notional, target_values


def competition_sharpe(
    returns: list[float],
) -> float:
    if len(returns) < 2:
        return 0.0

    sigma = stdev(returns)
    if sigma == 0 or not math.isfinite(sigma):
        return 0.0

    return 42.0 * mean(returns) / sigma


def max_drawdown(path: list[float]) -> float:
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
    opens: pd.DataFrame,
    daily_close: pd.DataFrame,
) -> dict:
    regime = regime_features(
        daily_close,
        dates[0],
    )

    gross = choose_gross(spec, regime)

    shares = {
        symbol: 0.0
        for symbol in OFFICIAL_SYMBOLS
    }
    cash = INITIAL_NAV

    begin_navs = []
    period_end_navs = []
    turnover_ratios = []
    valuation_path = [INITIAL_NAV]

    deployed = False

    for d_index, d in enumerate(dates):
        for r_index, t in enumerate(
            EXECUTION_TIMES,
            start=1,
        ):
            ts = utc_ts(d, t)
            row = opens.loc[ts]

            prices = {
                symbol: float(row[symbol])
                for symbol in OFFICIAL_SYMBOLS
            }

            pre_nav = cash + sum(
                shares[symbol] * prices[symbol]
                for symbol in OFFICIAL_SYMBOLS
            )

            if begin_navs:
                period_end_navs.append(pre_nav)
                valuation_path.append(pre_nav)

            begin_navs.append(pre_nav)

            if not deployed:
                (
                    post_nav,
                    gross_notional,
                    target_values,
                ) = solve_initial_allocation(
                    pre_nav,
                    gross,
                )

                for symbol in OFFICIAL_SYMBOLS:
                    shares[symbol] = (
                        target_values[symbol]
                        / prices[symbol]
                    )

                cash = (
                    post_nav
                    - sum(target_values.values())
                )

                turnover_ratios.append(
                    gross_notional / pre_nav
                )
                deployed = True
            else:
                # Autonomous HOLD.
                turnover_ratios.append(0.0)

        close_row = daily_close.loc[d]

        close_nav = cash + sum(
            shares[symbol]
            * float(close_row[symbol])
            for symbol in OFFICIAL_SYMBOLS
        )

        valuation_path.append(close_nav)

    final_nav = valuation_path[-1]
    period_end_navs.append(final_nav)

    if len(begin_navs) != len(period_end_navs):
        raise AssertionError(
            "Decision-period length mismatch."
        )

    period_returns = [
        end / begin - 1.0
        for begin, end in zip(
            begin_navs,
            period_end_navs,
        )
    ]

    return {
        "strategy": spec.name,
        "gross": gross,
        "market60_at_start": regime["market60"],
        "breadth60_at_start": regime["breadth60"],
        "return": final_nav / INITIAL_NAV - 1.0,
        "sharpe": competition_sharpe(period_returns),
        "max_drawdown": max_drawdown(valuation_path),
        "turnover": mean(turnover_ratios),
        "final_nav": final_nav,
    }


def main() -> None:
    require_project()

    print(
        "=== ICAIF PHASE 9: MULTI-YEAR EXPOSURE ROBUSTNESS ==="
    )
    print("Competition submission: DISABLED")
    print(
        "Question: how much gross exposure should the "
        "diversified autonomous buy-and-HOLD core use?"
    )
    print()

    frame = fetch_or_load()
    (
        opens,
        closes,
        daily_close,
        trading_dates,
    ) = prepare_views(frame)

    good_dates = complete_dates(
        opens,
        daily_close,
        trading_dates,
    )

    # Need 61 prior complete days for the regime features.
    if len(good_dates) < 200:
        raise SystemExit(
            f"ERROR: only {len(good_dates)} complete trading days."
        )

    usable = good_dates[70:]

    windows = []
    for start in range(
        0,
        len(usable) - 15 + 1,
        5,
    ):
        windows.append(
            usable[start : start + 15]
        )

    print(f"Complete usable trading days: {len(usable)}")
    print(f"15-day windows: {len(windows)}")
    print(
        f"Sample span: {usable[0]} through {usable[-1]}"
    )
    print()

    rows = []

    for w_no, dates in enumerate(
        windows,
        start=1,
    ):
        for spec in STRATEGIES:
            result = simulate_window(
                spec,
                dates,
                opens,
                daily_close,
            )
            result["window"] = w_no
            result["start_date"] = str(dates[0])
            result["end_date"] = str(dates[-1])
            rows.append(result)

        if (
            w_no % 25 == 0
            or w_no == len(windows)
        ):
            print(
                f"Completed window {w_no}/{len(windows)} "
                f"({dates[0]} -> {dates[-1]})"
            )

    results = pd.DataFrame(rows)

    baseline = (
        results.loc[
            results["strategy"] == "equal30_80_hold",
            [
                "window",
                "return",
                "sharpe",
                "max_drawdown",
                "turnover",
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

        joined = x.set_index("window").join(
            baseline
        )

        summary_rows.append(
            {
                "strategy": spec.name,
                "windows": len(x),
                "mean_gross": float(x["gross"].mean()),
                "mean_return": float(x["return"].mean()),
                "median_return": float(x["return"].median()),
                "p10_return": float(x["return"].quantile(0.10)),
                "positive_window_fraction": float(
                    (x["return"] > 0).mean()
                ),
                "mean_sharpe": float(x["sharpe"].mean()),
                "median_sharpe": float(x["sharpe"].median()),
                "mean_max_drawdown": float(
                    x["max_drawdown"].mean()
                ),
                "worst_max_drawdown": float(
                    x["max_drawdown"].max()
                ),
                "mean_turnover": float(
                    x["turnover"].mean()
                ),
                "higher_return_vs_equal80": float(
                    (
                        joined["return"]
                        > joined["baseline_return"]
                    ).mean()
                ),
                "higher_sharpe_vs_equal80": float(
                    (
                        joined["sharpe"]
                        > joined["baseline_sharpe"]
                    ).mean()
                ),
                "lower_mdd_vs_equal80": float(
                    (
                        joined["max_drawdown"]
                        < joined["baseline_max_drawdown"]
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
            "mean_sharpe",
            "mean_return",
            "mean_max_drawdown",
        ],
        ascending=[False, False, True],
    )

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "artifact_type": (
            "MULTIYEAR_RESEARCH_ONLY_NO_SUBMISSION"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_source": "Alpaca free IEX",
        "timeframe": TIMEFRAME,
        "start_request": START,
        "window_days": 15,
        "window_step_days": 5,
        "window_count": len(windows),
        "fee_rate": FEE,
        "baseline": "equal30_80_hold",
        "notes": [
            "Each 15-day window deploys once at R1 then autonomously HOLDs.",
            "30-minute IEX bars proxy scheduled execution boundaries and daily close.",
            "This is interim robustness evidence, not organizer-feed ground truth.",
            "Regime rules use only prior completed daily closes.",
        ],
        "summary": summary.to_dict(
            orient="records"
        ),
        "window_results": results.to_dict(
            orient="records"
        ),
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    summary.to_csv(
        OUT_CSV,
        index=False,
    )

    print()
    print(
        "=== PHASE 9 MULTI-YEAR SUMMARY ==="
    )

    display = summary.copy()

    for col in [
        "mean_gross",
        "mean_return",
        "median_return",
        "p10_return",
        "positive_window_fraction",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_turnover",
        "higher_return_vs_equal80",
        "higher_sharpe_vs_equal80",
        "lower_mdd_vs_equal80",
        "mean_return_delta_vs_equal80",
    ]:
        display[col] = display[col].map(
            lambda value: f"{value:.4f}"
        )

    for col in [
        "mean_sharpe",
        "median_sharpe",
    ]:
        display[col] = display[col].map(
            lambda value: f"{value:.2f}"
        )

    print(
        display.to_string(index=False)
    )

    print()
    print(f"JSON artifact: {OUT_JSON}")
    print(f"CSV summary:   {OUT_CSV}")
    print()
    print(
        "PHASE 9 COMPLETE. "
        "NO COMPETITION SUBMISSION WAS MADE."
    )


if __name__ == "__main__":
    main()
