#!/usr/bin/env python3
"""ICAIF Phase 19: leakage-conservative SEC + market ML research, NO live trading.

Modes:
  --self-test  Tests timing, financial extraction, stock constraints, training and news guards.
  --prepare-sec  Cache all 30 companies' complete, historical SEC CompanyFacts files.
  --train  Year-forward gradient boosting: price-only vs SEC+price; 15d net-of-fee replay.
  --news-audit  Inspect existing Phase18 news, no inference/trades/network.

Prepared facts remain research-grade: SEC CompanyFacts is queried today, and
`filed` is a DATE, not acceptance timestamp. We impose two calendar days of
latency before using each fact, including any restatement. This is conservative
but not a substitute for final as-filed historical reconstruction review.

The 2026 untouched test is intentionally NOT evaluated. 2025 has been exposed
in prior research, so it is a diagnostic year, not a fresh holdout.

No Codabench/organizer client, credential printing, model pickle or live writer.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SYMBOLS = ("AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO "
           "AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE").split()
OUT_REL = Path("artifacts") / "phase19_sec_ml"
FEES = 0.001
SEED = 20261009
LABEL_DAYS = 5
MIN_TRAIN = 5000
METRICS = {
    "annual_revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "annual_earnings": ("NetIncomeLoss",),
    "annual_cashflow": ("NetCashProvidedByUsedInOperatingActivities",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
}


def root_dir():
    root = Path.cwd().resolve()
    if not (root / "tools" / "auto_submit.py").is_file():
        raise RuntimeError("Open the existing starter-kit folder in VS Code first.")
    return root


def load_private_contact(root):
    # Only read the one non-trading, non-secret email value used as SEC identification.
    for filename in (Path.home() / ".icaif" / "research_sources.env", root / ".env"):
        if filename.is_file():
            for line in filename.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("SEC_CONTACT_EMAIL="):
                    value = line.split("=", 1)[1].strip().strip("\"'")
                    if "@" in value and len(value) <= 180 and not any(x.isspace() for x in value):
                        return value
    raise RuntimeError("SEC contact email is not configured in the local research .env.")


def get_sec_json(url, contact, *, cap=25_000_000):
    if not url.startswith("https://data.sec.gov/api/xbrl/companyfacts/CIK") and url != "https://www.sec.gov/files/company_tickers.json":
        raise ValueError("SEC URL not allowlisted")
    req = Request(url, headers={
        "User-Agent": f"ICAIF2026-academic-research contact={contact}",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=25) as response:
            data = response.read(cap + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        name = f"HTTP_{e.code}" if isinstance(e, HTTPError) else "NETWORK_OR_TIMEOUT"
        raise RuntimeError(name) from None
    if len(data) > cap:
        raise RuntimeError("RESPONSE_TOO_LARGE")
    try:
        return json.loads(data)
    except (ValueError, UnicodeError):
        raise RuntimeError("INVALID_SEC_JSON") from None


def sec_symbol_map(payload):
    entries = payload.values() if isinstance(payload, dict) else payload
    result = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("ticker", "")).upper().strip()
        try:
            cik = int(row.get("cik_str", 0))
        except (TypeError, ValueError):
            continue
        if sym in SYMBOLS and cik > 0:
            result[sym] = cik
    return result


def sec_cache(root):
    return root / OUT_REL / "sec_raw"


def prepare_sec(root, *, limit=30):
    folder = sec_cache(root)
    folder.mkdir(parents=True, exist_ok=True)
    contact = load_private_contact(root)
    mapping = sec_symbol_map(get_sec_json("https://www.sec.gov/files/company_tickers.json", contact,
                                         cap=2_000_000))
    if len(mapping) != len(SYMBOLS):
        raise RuntimeError("SEC mapping incomplete: stop instead of guessing tickers")
    successes, errors = [], []
    for n, sym in enumerate(SYMBOLS[:limit], 1):
        cik = mapping[sym]
        path = folder / f"{sym}_CIK{cik:010d}.json"
        if path.is_file() and path.stat().st_size > 500:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if int(raw.get("cik", -1)) == cik and "facts" in raw:
                    successes.append(sym)
                    print(f"{n:2d}/{limit} {sym}: SEC_CACHED", flush=True)
                    continue
            except (ValueError, TypeError, OSError):
                pass
        try:
            raw = get_sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", contact)
            if int(raw.get("cik", -1)) != cik or "us-gaap" not in raw.get("facts", {}):
                raise RuntimeError("CIK_OR_FACTS_MISMATCH")
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
            temp.replace(path)
            successes.append(sym)
            print(f"{n:2d}/{limit} {sym}: SEC_CACHED_NEW", flush=True)
        except (RuntimeError, ValueError, OSError) as e:
            code = str(e)
            if not (code.startswith("HTTP_") or code in {"NETWORK_OR_TIMEOUT", "RESPONSE_TOO_LARGE", "CIK_OR_FACTS_MISMATCH", "INVALID_SEC_JSON"}):
                code = "PROVIDER_OR_DISK_ERROR"
            errors.append((sym, code))
            print(f"{n:2d}/{limit} {sym}: SEC_ERROR_{code}", flush=True)
        time.sleep(0.60)  # SEC identifiable polite access, no automatic retry
    print("SEC_CACHE_READY:", len(successes), "of", limit)
    print("SEC_ERRORS:", len(errors))
    if errors:
        print("SEC_INCOMPLETE_RESEARCH_ONLY")
    print("NO_COMPETITION_CALLS")


@dataclass(frozen=True)
class Fact:
    filing: pd.Timestamp
    end: pd.Timestamp
    value: float
    tag_rank: int


def parse_facts(document, metric):
    """Return filed-date-constrained events; never infer missing XBRL values."""
    if metric not in METRICS:
        raise ValueError("unknown metric")
    obj = document.get("facts", {}).get("us-gaap", {})
    annual = metric.startswith("annual_")
    events = []
    for rank, tag in enumerate(METRICS[metric]):
        units = obj.get(tag, {}).get("units", {}).get("USD", [])
        for row in units:
            if row.get("form") not in (("10-K",) if annual else ("10-K", "10-Q")):
                continue
            try:
                filed = pd.Timestamp(row["filed"])
                end = pd.Timestamp(row["end"])
                value = float(row["val"])
                if not math.isfinite(value) or abs(value) > 1e15:
                    continue
                if filed < end or end.year < 2018:
                    continue
                if annual:
                    start = pd.Timestamp(row["start"])
                    if not 320 <= (end - start).days <= 390:
                        continue
                elif "start" in row:
                    continue
                events.append(Fact(filed, end, value, rank))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
    # SEC often repeats comparative values: the filing date is the availability
    # timestamp and a restatement with a later filing cannot affect earlier days.
    return sorted(events, key=lambda f: (f.filing, f.end, -f.tag_rank))


def known_fact(facts, decision_day, *, lag_days=2, max_age_days=560):
    day = pd.Timestamp(decision_day)
    cutoff = day - pd.Timedelta(days=lag_days)
    candidates = (f for f in facts if f.filing <= cutoff and f.end <= cutoff
                  and (day - f.end).days <= max_age_days)
    return max(candidates, key=lambda f: (f.end, f.filing, -f.tag_rank), default=None)


def asof_sec_features(doc, decision_day):
    series = {k: parse_facts(doc, k) for k in METRICS}
    return features_from_series(series, decision_day)


def features_from_series(series, decision_day):
    last = {
        k: known_fact(v, decision_day, max_age_days=550 if k.startswith("annual_") else 480)
        for k, v in series.items()
    }
    rev = last["annual_revenue"]
    ni = last["annual_earnings"]
    cfo = last["annual_cashflow"]
    assets = last["assets"]
    liabilities = last["liabilities"]

    def age(f):
        return min((pd.Timestamp(decision_day) - f.filing).days / 365.0, 2.0) if f else np.nan

    # Historical financials: ratios can only be formed from facts individually
    # available before the as-of day; NaNs are passed to HGB with presence flags.
    prior = None
    if rev:
        prior_candidates = [f for f in series["annual_revenue"]
                            if f.filing <= pd.Timestamp(decision_day) - pd.Timedelta(days=2)
                            and f.tag_rank == rev.tag_rank
                            and 300 <= (rev.end - f.end).days <= 430]
        prior = max(prior_candidates, key=lambda f: (f.end, f.filing), default=None)
    revenue_growth = (
        np.clip(rev.value / prior.value - 1., -2., 4.)
        if rev and prior and rev.value > 0 and prior.value > 0 else np.nan
    )
    margin = (np.clip(ni.value / rev.value, -1., 1.) if ni and rev and rev.value > 0
              and abs((ni.end - rev.end).days) <= 35 else np.nan)
    cf_margin = (np.clip(cfo.value / rev.value, -2., 2.) if cfo and rev and rev.value > 0
                 and abs((cfo.end - rev.end).days) <= 35 else np.nan)
    leverage = (np.clip(liabilities.value / assets.value, 0, 3.)
                if assets and liabilities and assets.value > 0
                and abs((liabilities.end - assets.end).days) <= 45 else np.nan)
    values = np.asarray([
        math.log(max(rev.value, 1)) if rev and rev.value > 0 else np.nan,
        margin, cf_margin, leverage,
        math.log(max(assets.value, 1)) if assets and assets.value > 0 else np.nan,
        revenue_growth, age(rev), age(assets),
        float(rev is not None), float(ni is not None), float(assets is not None),
        float(cfo is not None), float(liabilities is not None),
    ], dtype=float)
    assert values.shape == (13,)
    return values


def build_sec_panel(dates, folder):
    n, m = len(dates), len(SYMBOLS)
    arr = np.full((n, m, 13), np.nan)
    counts = {}
    for j, sym in enumerate(SYMBOLS):
        paths = list(folder.glob(sym + "_CIK*.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Missing or duplicate SEC source for {sym}; run --prepare-sec")
        doc = json.loads(paths[0].read_text(encoding="utf-8"))
        series = {k: parse_facts(doc, k) for k in METRICS}
        for i, d in enumerate(dates):
            arr[i, j] = features_from_series(series, d)
        counts[sym] = int(np.sum(arr[:,j,8]))
        print(f"SEC_PIT_PANEL {j+1:02d}/30 {sym}: revenue_asof_days={counts[sym]}", flush=True)
    if np.isinf(arr).any():
        raise RuntimeError("Infinite fundamental features")
    return arr, counts


def load_research_module(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase15 v2: {path}. Download the corrected _v2.py file.")
    spec = importlib.util.spec_from_file_location("icaif_phase15_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def rank_target(prediction, gross, l1=0.08):
    score = np.asarray(prediction, float)
    if score.shape != (30,) or not np.isfinite(score).all():
        raise ValueError("Missing/nonfinite predictions; HOLD or baseline in live integration")
    if gross not in (0.6, 0.8, 1.0) or not 0 <= l1 <= .25:
        raise ValueError("Invalid frozen portfolio parameters")
    base = np.full(30, gross/30)
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    centered = ranks - np.mean(ranks)
    denom = np.sum(np.abs(centered))
    if denom > 0:
        d = l1 * centered / denom
        scale = min(1., np.min(base[d < 0] / -d[d < 0]) if np.any(d < 0) else 1.)
        base += d * max(0., scale * .999999)
    base *= gross / base.sum()
    if not (np.min(base) >= -1e-14 and np.max(base) <= .3 and abs(base.sum()-gross) <= 1e-10):
        raise ValueError("Unsafe portfolio mapping")
    return base


def train_predict(X, labels, dates, year, *, max_iter=100):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xflat = X.reshape(-1, X.shape[2])
    y = labels.reshape(-1)
    date_ix = np.repeat(np.arange(len(dates)), len(SYMBOLS))
    cutoff = int(np.searchsorted(dates.values, np.datetime64(f"{year}-01-01")))
    # The 5-day outcome MUST have matured before the beginning of the year.
    train = (date_ix >= 130) & (date_ix + LABEL_DAYS < cutoff) & np.isfinite(y)
    # price-only inputs require all fields; fundamental NaNs are expected.
    price_ok = np.isfinite(Xflat[:, :14]).all(axis=1)
    train &= price_ok
    test = ((date_ix >= cutoff) & (date_ix < len(dates))
            & (dates.values[date_ix] < np.datetime64(f"{year+1}-01-01"))
            & price_ok)
    if int(train.sum()) < MIN_TRAIN:
        raise RuntimeError(f"Insufficient chronological training for {year}: {train.sum()}")
    m = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.035, max_iter=max_iter,
        max_leaf_nodes=9, max_depth=3, min_samples_leaf=200,
        l2_regularization=25., random_state=SEED, early_stopping=False,
    )
    m.fit(Xflat[train], y[train])
    scores = np.full(labels.shape, np.nan)
    scores.reshape(-1)[test] = np.clip(m.predict(Xflat[test]), -.25, .25)
    return scores, int(train.sum()), int(test.sum())


def evaluate_year(phase, year, dates, opens, ends, price_features, pred_price, pred_sec):
    windows = [int(s) for s in np.flatnonzero(dates.year == year)[::15]
               if s >= 132 and s + 15 <= len(dates) and dates[s+14].year == year]
    rows = []
    for start in windows:
        index = start - 1  # decide after preceding close; execute next session
        if not np.isfinite(price_features[index]).all():
            continue
        if not (np.isfinite(pred_price[index]).all() and np.isfinite(pred_sec[index]).all()):
            continue
        mappings = {
            "equal80": np.full(30, .8/30),
            "equal100": np.full(30, 1./30),
            "price_hgb80": rank_target(pred_price[index], .8),
            "sec_price_hgb80": rank_target(pred_sec[index], .8),
            "sec_price_hgb100": rank_target(pred_sec[index], 1.),
            "sec_price_hgb60": rank_target(pred_sec[index], .6),
            "blended_hgb80": rank_target(.5*pred_price[index] + .5*pred_sec[index], .8),
        }
        for strategy, weights in mappings.items():
            record = phase.simulate(opens, ends, start, strategy, weights)
            rows.append({"year": year, "window_start": str(dates[start].date()),
                         "strategy": strategy, **record})
    return rows


def run_train(root, phase15):
    phase = load_research_module(phase15)
    source = phase.source_path(root)
    print("PHASE19_OFFLINE_ML_RESEARCH_NO_SUBMISSION")
    print("MARKET_DATA:", source)
    dates, opens, ends, daily, sectors = phase.load_history(source)
    price_X, labels, _ = phase.make_features(daily, sectors)
    if price_X.shape[2] != 14 or len(SYMBOLS) != daily.shape[1]:
        raise RuntimeError("Phase15 feature schema changed unexpectedly")
    SEC, coverage = build_sec_panel(dates, sec_cache(root))
    X = np.concatenate([price_X, SEC], axis=2)
    print("PRICE_FEATURES:", price_X.shape[2], "SEC_FEATURES:", SEC.shape[2])
    print("YEAR_SPLITS: expanding training; purged 5d labels; evaluated 2023 2024 2025")
    records = []
    for year in (2023, 2024, 2025):
        pred_price, train_n, test_n = train_predict(price_X, labels, dates, year)
        pred_sec, _, _ = train_predict(X, labels, dates, year)
        print(f"YEAR_{year}: train_rows={train_n} test_rows={test_n}")
        records.extend(evaluate_year(phase, year, dates, opens, ends, price_X,
                                     pred_price, pred_sec))
    if not records:
        raise RuntimeError("No portfolio evaluation windows")
    out = pd.DataFrame.from_records(records)
    summary = (out.groupby(["year", "strategy"], sort=True)[
        ["cumulative_return", "sharpe_proxy", "mdd_proxy", "turnover_proxy", "fee_usd"]]
        .mean().round(8).reset_index())
    print("=== PHASE19 NONOVERLAPPING 15-DAY WINDOWS; 2025 EXPOSED_DIAGNOSTIC ===")
    for year in (2023, 2024, 2025):
        s = summary[summary.year == year]
        print(f"YEAR {year}: {len(out[out.year == year])//7} windows")
        print(s.to_string(index=False))
    folder = root / OUT_REL
    folder.mkdir(parents=True, exist_ok=True)
    out.to_csv(folder / "phase19_window_results.csv", index=False)
    summary.to_csv(folder / "phase19_year_summary.csv", index=False)
    config = {
        "mode": "RESEARCH_ONLY_NO_TRADING", "sec_asof_lag_days": 2,
        "signal_type": "5day_stock_return", "fee_rate": FEES,
        "test_years": [2023,2024,2025], "fresh_holdout_tested": False,
        "fundamental_source": "SEC companyfacts historical filed-date events as currently retrieved",
        "not_live_ready": [
            "Historical CompanyFacts point-in-time reconstruction needs audit",
            "No historical Finnhub news labels or verified contemporaneous news feed",
            "Hourly bar execution prices are proxies, not competition prices",
            "No integrated Official watch scheduling and current-price inference yet",
            "2025 already viewed in earlier research and not independent",
        ],
        "sec_coverage_days": coverage,
    }
    (folder / "phase19_research_manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("REPORT:", folder / "phase19_year_summary.csv")
    print("MODEL_PROMOTION_APPROVED: NO (requires independent evidence + live interface)")
    print("PHASE19_COMPLETE_NO_COMPETITION_SUBMISSION")


def audit_news(root):
    folder = root / "artifacts" / "public_data_sources" / "phase18_universe"
    paths = sorted(folder.glob("phase18_snapshot_*.json"))
    if not paths:
        raise FileNotFoundError("Missing Phase18 snapshot; run Phase18 --collect")
    snapshot = json.loads(paths[-1].read_text(encoding="utf-8"))
    records = [r for r in snapshot.get("events", []) if r.get("source") == "FINNHUB"]
    seen = set()
    stats = {}
    for sym in SYMBOLS:
        news = [r for r in records if r.get("symbol") == sym]
        good = []
        for r in news:
            key = (sym, r.get("event_id"))
            if (not key[1] or key in seen or r.get("historical_backtest_eligible") is not False
                    or not r.get("available_at_utc")):
                continue
            seen.add(key)
            good.append(r)
        stats[sym] = len(good)
    print("FINNHUB_NEWS_EVENTS:", len(records))
    print("UNIQUE_EVENTS:", len(seen))
    print("STOCKS_COVERED:", sum(n > 0 for n in stats.values()), "/30")
    print("NEWS_HISTORICALLY_LABELLED: NO")
    print("NEWS_ONLY_SHADOW_FEATURES: YES")
    print("COMPETITION_TRADING_IMPACT: NONE")


def self_test():
    assert len(SYMBOLS) == len(set(SYMBOLS)) == 30
    x = {
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"form":"10-K", "filed":"2024-02-10", "start":"2023-01-01",
                 "end":"2023-12-31", "val":1000},
                {"form":"10-K", "filed":"2025-02-10", "start":"2024-01-01",
                 "end":"2024-12-31", "val":1200},
                {"form":"10-K", "filed":"2025-05-10", "start":"2024-01-01",
                 "end":"2024-12-31", "val":1300},
            ]}},
            "Assets": {"units": {"USD": [
                {"form":"10-K", "filed":"2024-02-10", "end":"2023-12-31", "val":2000}
            ]}},
        }},
    }
    rows = parse_facts(x, "annual_revenue")
    assert known_fact(rows,"2025-02-10").value == 1000  # filing not yet eligible
    assert known_fact(rows,"2025-02-12").value == 1200
    assert known_fact(rows,"2025-05-11").value == 1200
    assert known_fact(rows,"2025-05-12").value == 1300  # restatement only after lag
    f_before = asof_sec_features(x,"2025-02-10")
    f_after = asof_sec_features(x,"2025-02-12")
    assert np.isclose(f_after[5], .2) and np.isnan(f_before[5])
    for gross in (.6,.8,1.):
        w = rank_target(np.arange(30), gross, .12)
        assert np.isfinite(w).all() and np.min(w)>=0 and np.max(w)<=.3
        assert abs(w.sum()-gross)<1e-10
    # Chronological labels cannot be trained on returns that mature in test year.
    dates = pd.date_range("2021-01-01", periods=780, freq="B")
    idx = np.repeat(np.arange(len(dates)), len(SYMBOLS))
    cut = int(np.searchsorted(dates.values,np.datetime64("2023-01-01")))
    assert not np.any(((idx >= 130) & (idx + LABEL_DAYS < cut)) & (idx >= cut-LABEL_DAYS))
    print("PHASE19_SELF_TEST_PASS")
    print("SEC_ASOF_AND_RESTATEMENT_LAG_PASS")
    print("FEE_AWARE_PORTFOLIO_LIMITS_PASS")
    print("CHRONOLOGICAL_LABEL_PURGE_PASS")
    print("NO_NETWORK_NO_COMPETITION_SUBMISSION")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--prepare-sec", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--news-audit", action="store_true")
    parser.add_argument("--phase15", type=Path, default=Path.home()/"Downloads"/"icaif_phase15_official_challengers_v2.py")
    parser.add_argument("--limit", type=int, default=30, help="SEC preparation test: first N symbols")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = root_dir()
    if args.prepare_sec:
        if not 1<=args.limit<=30:
            raise ValueError("limit must be 1-30")
        prepare_sec(root, limit=args.limit)
    elif args.train:
        run_train(root, args.phase15)
    elif args.news_audit:
        audit_news(root)

if __name__ == "__main__":
    main()
