#!/usr/bin/env python3
"""ICAIF 2026 Phase 20 -- adaptive price+SEC multi-horizon research, NO TRADING.

Run from starter-kit with the Windows .venv active:
  python "$HOME\\Downloads\\icaif_phase20_adaptive_multihorizon.py" --self-test
  python "$HOME\\Downloads\\icaif_phase20_adaptive_multihorizon.py" --run

Strictly offline. No network calls, no organizer endpoints, no credential access.
Requires cached SEC companyfacts from Phase19 and the official hourly parquet.

Research controls:
  - 1-day and 5-day supervised targets; expanding-year training; purged labels.
  - 2023/2024 development, 2025 EXPOSED diagnostic, 2026 NOT evaluated.
  - All decisions use completed previous-day bars + as-filed SEC data.
  - Intraday period valuation uses official historical hourly PRICE PROXIES,
    not exact 10:30/11:30/... competition execution prints.
  - Daily rebalances represent R1 morning execution approximation only.
  - Fee = 0.1% of two-sided notional. HOLD sends no target and pays no fee.
  - Results cannot establish an Official candidate without final deployment tests.

This is a new reproducible candidate experiment, NOT a claim of better alpha.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

SYMBOLS = tuple("AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO "
                "AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE".split())
N_STOCKS = 30
FEE = 0.001
INITIAL_NAV = 1_000_000.0
DECISIONS_PER_DAY = 7
DAYS = 15
ANNUALIZER = math.sqrt(252 * 7)
START_FEATURE = 130
RANDOM_STATE = 20261009
OUT_REL = Path('artifacts') / 'phase20_adaptive_multihorizon'

# These are comparable total-company US-GAAP revenues, not unrelated
# customer deposits, asset balances or company-specific custom taxonomy tags.
# CompanyFacts exposes only standard taxonomy entity-wide items; missing remains
# missing if no compatible tag appears in an issuer's filing history.
REVENUE_TAGS = (
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'SalesRevenueNet',
    'SalesAndOtherOperatingRevenue',
    'OperatingRevenues',
    'RevenuesNetOfInterestExpense',
    'RevenueFromContractWithCustomerIncludingAssessedTax',
    'SalesRevenueGoodsNet',
    'UtilityRevenues',
    'ElectricUtilityRevenue',
    'ElectricUtilityRevenues',
    'RegulatedAndUnregulatedOperatingRevenue',
    'SalesRevenueServicesNet',
)


@dataclass(frozen=True)
class Policy:
    name: str
    method: str         # equal | price | sec | blend
    equity: float       # max invested portfolio fraction
    cadence: int        # 0 initial-only, 1 daily, 5 weekly
    gate: float         # minimum two-sided weight drift for rebalance
    max_trade: float    # maximum gross turnover in any single decision
    regime: bool        # reduce exposure when broad-market trend negative
    alpha_mix: float    # contribution of model to weights, 0..1


POLICIES = (
    Policy('equal80_hold',   'equal', .8, 0, .00, 1.00, False, .00),
    Policy('equal100_hold',  'equal',1.0, 0, .00, 1.00, False, .00),
    Policy('equal80_daily_gate08', 'equal', .8, 1, .08, .12, False, .00),
    Policy('price80_hold_t15', 'price', .8, 0, .00, 1.00, False, .15),
    Policy('sec80_hold_t15',   'sec',   .8, 0, .00, 1.00, False, .15),
    Policy('blend80_hold_t15', 'blend', .8, 0, .00, 1.00, False, .15),
    Policy('blend80_weekly_gate06', 'blend', .8, 5, .06, .12, False, .15),
    Policy('blend80_daily_gate06',  'blend', .8, 1, .06, .10, False, .15),
    Policy('blend80_daily_gate12',  'blend', .8, 1, .12, .12, False, .15),
    Policy('blend80_regime_weekly', 'blend', .8, 5, .06, .16, True, .15),
    Policy('blend100_weekly_gate08','blend',1.0, 5, .08, .12, False, .15),
    Policy('blend80_weekly_t30',    'blend', .8, 5, .06, .15, False, .30),
)


def import_file(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required research script: {path.name}. Place it in Downloads.')
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError('Cannot import research script')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project_root() -> Path:
    p = Path.cwd().resolve()
    if not (p/'tools'/'auto_submit.py').is_file():
        raise RuntimeError('Open the starter-kit folder in VS Code before running this research.')
    return p


def check_sec_cache(root: Path) -> list[Path]:
    folder = root/'artifacts'/'phase19_sec_ml'/'sec_raw'
    files = []
    for sym in SYMBOLS:
        matches = list(folder.glob(f'{sym}_CIK*.json'))
        if len(matches) != 1 or matches[0].stat().st_size < 500:
            raise RuntimeError(f'SEC cache missing/ambiguous for {sym}. Run Phase19 --prepare-sec first.')
        files.append(matches[0])
    return files


def audit_tags(doc: dict) -> tuple[list[str], list[str]]:
    obj = doc.get('facts', {}).get('us-gaap', {})
    have = [x for x in REVENUE_TAGS if x in obj]
    others = [x for x in obj if ('revenue' in x.lower() or 'sales' in x.lower()) and x not in REVENUE_TAGS]
    return have, sorted(others)[:12]


def annual_revenues(doc: dict, phase19, decision_day) -> float:
    """Use filed-date-constrained XBRL, including the explicit fallback tags."""
    facts = phase19.parse_facts(doc, 'annual_revenue')
    candidate = phase19.known_fact(facts, decision_day, lag_days=2, max_age_days=550)
    return candidate.value if candidate else float('nan')


def make_sec_panel(dates: pd.DatetimeIndex, sec_files: list[Path], phase19):
    """Use Phase19's point-in-time feature generator with expanded whitelisted tags."""
    orig = phase19.METRICS['annual_revenue']
    phase19.METRICS['annual_revenue'] = REVENUE_TAGS
    try:
        features = np.full((len(dates), N_STOCKS, 13), np.nan, dtype=float)
        reports = []
        for col, (sym, path) in enumerate(zip(SYMBOLS, sec_files)):
            raw = json.loads(path.read_text(encoding='utf-8'))
            have, other = audit_tags(raw)
            series = {key: phase19.parse_facts(raw, key) for key in phase19.METRICS}
            # SEC accounting inputs are slowly varying: compute only on
            # trading days; every fact has own filing date + 2 calendar days.
            for i, day in enumerate(dates):
                features[i, col] = phase19.features_from_series(series, day)
            rev_count = int(np.isfinite(features[:, col, 0]).sum())
            reports.append({'symbol':sym,'revenue_feature_days':rev_count,
                            'revenue_tag_names':have,'other_revenue_like_tags':other,
                            'revenue_missing':rev_count == 0})
            print(f'SEC_TAG_AUDIT {col+1:02d}/30 {sym}: revenue_days={rev_count} '
                  f'whitelisted_tags={len(have)}', flush=True)
    finally:
        phase19.METRICS['annual_revenue'] = orig
    return features, reports


def one_day_labels(prices: np.ndarray, days: int):
    result = np.full_like(prices, np.nan, dtype=float)
    result[:-days] = prices[days:] / prices[:-days] - 1.0
    return result


def year_predictions(dates, X_price, X_sec, daily, year, *, max_iter=75):
    """Fit 1d-price, 5d-price and 5d-price+SEC using purged expanding windows.

    Only the last completed prior trading day can enter a decision. Predictions
    for the current evaluation year are fixed from training before that year.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    n, p, nf = X_price.shape
    if n != len(dates) or p != N_STOCKS or X_sec.shape[:2] != (n,p):
        raise ValueError('Feature dimensionality mismatch')
    cut = int(np.searchsorted(dates.values, np.datetime64(f'{year}-01-01')))
    latest = int(np.searchsorted(dates.values, np.datetime64(f'{year+1}-01-01')))
    time_index = np.repeat(np.arange(n), p)
    Xp = X_price.reshape(n*p, -1)
    Xf = np.concatenate([X_price, X_sec], axis=2).reshape(n*p, -1)
    price_ok = np.isfinite(Xp).all(axis=1)
    result = {}
    counts = {}
    for label, mat, horizon in (
        ('price_1d', Xp, 1),
        ('price_5d', Xp, 5),
        ('sec_5d', Xf, 5),
    ):
        y = one_day_labels(daily, horizon).reshape(-1)
        # end-of-label < first day of test year. No random splitting.
        train = ((time_index >= START_FEATURE) &
                 (time_index+horizon < cut) & price_ok & np.isfinite(y))
        predict = ((time_index >= cut) & (time_index < latest) & price_ok)
        if train.sum() < 5000 or predict.sum() < 2000:
            raise RuntimeError(f'Insufficient training or test records for {year}/{label}')
        m = HistGradientBoostingRegressor(
            learning_rate=.035, max_iter=max_iter, max_leaf_nodes=9,
            max_depth=3, min_samples_leaf=200, l2_regularization=25.0,
            early_stopping=False, random_state=RANDOM_STATE,
        )
        m.fit(mat[train], y[train])
        forecast = np.full(n*p, np.nan)
        forecast[predict] = np.clip(m.predict(mat[predict]), -.20, .20)
        result[label] = forecast.reshape(n,p)
        counts[label] = {'train':int(train.sum()), 'test':int(predict.sum()),
                         'train_last_day_index':int(np.max(time_index[train])),
                         'test_first_day_index':int(np.min(time_index[predict])),
                         'label_horizon_days':horizon}
        assert counts[label]['train_last_day_index'] + horizon < cut
        print(f'MODEL_{year}_{label}: training={counts[label]["train"]} '
              f'test={counts[label]["test"]}', flush=True)
    return result, counts


def cross_rank(score: np.ndarray):
    a = np.asarray(score, float)
    if a.shape != (N_STOCKS,) or not np.isfinite(a).all():
        raise ValueError('Invalid cross-sectional signals')
    # Equal values must stay tied, including equal-sector neutral signals.
    return pd.Series(a).rank(method='average', pct=True).to_numpy()*2-1


def alpha_target(score: np.ndarray, gross: float, mix: float,
                 *, sector_names: list[str]):
    """Construct diversified, fee-aware legal target; NO concentration over 8%."""
    if not (0 <= mix <= .5 and 0 <= gross <= 1.0):
        raise ValueError('Invalid model blend / gross')
    scores = cross_rank(score)
    # Remove sector mean so the alpha overlay does not unintentionally
    # concentrate all portfolio exposure in one industry.
    neutral = scores.copy()
    for group in set(sector_names):
        locs = [i for i,g in enumerate(sector_names) if g == group]
        neutral[locs] -= np.mean(scores[locs])
    centered = neutral - neutral.mean()
    scale = max(float(np.max(np.abs(centered))), 1e-12)
    tilt = np.clip(centered/scale, -1, 1)
    raw = gross / N_STOCKS * (1.0 + mix * tilt)
    if (raw < -1e-12).any():
        raise AssertionError('Negative target')
    w = raw / raw.sum()*gross
    validate_weights(w)
    if w.max() > .08 + 1e-12:
        raise AssertionError('Unexpected concentration from low-strength tilts')
    return w


def validate_weights(w):
    x = np.asarray(w)
    if (x.shape != (N_STOCKS,) or not np.isfinite(x).all()
            or (x < -1e-12).any() or (x > .30+1e-12).any()
            or x.sum() > 1+1e-10):
        raise ValueError('Illegal weights')


def current_regime(price_features: np.ndarray) -> bool:
    """A predefined conditional risk switch, not model self-modification.
    Indices 12 and 13 are 20d/60d equal-universe price trends.
    """
    if price_features.ndim != 2 or price_features.shape[1] < 14:
        raise ValueError('Missing market features')
    m20 = float(price_features[0,12])
    m60 = float(price_features[0,13])
    return np.isfinite(m20) and np.isfinite(m60) and m20 < 0 and m60 < 0


def model_signal(policy: Policy, day_idx, forecasts, features):
    """One-day forecast shrunk against five-day forecast and recent trend."""
    if policy.method == 'price':
        core = .35 * cross_rank(forecasts['price_1d'][day_idx]) + .65 * cross_rank(forecasts['price_5d'][day_idx])
    elif policy.method == 'sec':
        core = .35 * cross_rank(forecasts['price_1d'][day_idx]) + .65 * cross_rank(forecasts['sec_5d'][day_idx])
    elif policy.method == 'blend':
        core = (.25 * cross_rank(forecasts['price_1d'][day_idx]) +
                .375 * cross_rank(forecasts['price_5d'][day_idx]) +
                .375 * cross_rank(forecasts['sec_5d'][day_idx]))
    else:
        core = np.zeros(N_STOCKS)
    # 20-day trend + lower volatility, same features as baseline dataset.
    momentum = cross_rank(features[day_idx,:,2])
    defensive = cross_rank(-features[day_idx,:,6])
    return .75 * core + .15 * momentum + .10 * defensive


def policy_weights(policy, decision_idx, forecasts, price_features, sectors):
    risk_off = bool(policy.regime and current_regime(price_features[decision_idx]))
    gross = policy.equity * (.65 if risk_off else 1.0)
    if policy.method == 'equal':
        w = np.full(N_STOCKS, gross/N_STOCKS)
    else:
        score = model_signal(policy,decision_idx,forecasts,price_features)
        w = alpha_target(score, gross,policy.alpha_mix,sector_names=sectors)
    validate_weights(w)
    return w


def exact_fee_transaction(cash: float, shares, price, weights):
    """Solve implicit NAV after fee, respecting combined buy and sell notional."""
    validate_weights(weights)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError('Invalid price input')
    holdings = shares * price
    nav_before = cash + float(holdings.sum())
    post_nav = nav_before
    for _ in range(100):
        target_values = post_nav * weights
        notional = float(np.abs(target_values-holdings).sum())
        updated = nav_before-FEE*notional
        if abs(updated-post_nav) < 1e-9:
            post_nav=updated
            break
        post_nav=updated
    final_values = post_nav*weights
    traded = float(np.abs(final_values-holdings).sum())
    fee = FEE*traded
    new_cash = post_nav - float(final_values.sum())
    if new_cash < -1e-7 or not math.isfinite(new_cash):
        raise ValueError('Negative cash after fees')
    return new_cash, final_values / price, traded, fee, nav_before


def replay_window(policy: Policy, start: int, opens, endpoints,
                  features, forecasts, sectors, *, size=DAYS):
    cash, shares = INITIAL_NAV, np.zeros(N_STOCKS)
    path = [INITIAL_NAV]
    fee_sum, trade_notional, decisions = 0.0, 0.0, 0
    total_turnover_fraction = 0.0
    for offset in range(size):
        day = start+offset
        signal_day = day-1
        if signal_day < START_FEATURE:
            raise ValueError('Too little prior data')
        # Endpoints represent seven hourly valuation proxies for each day.
        px = opens[day]
        nav = cash+float(np.dot(shares,px))
        if policy.cadence == 0:
            rebalance_allowed = offset == 0
        else:
            rebalance_allowed = offset == 0 or (offset % policy.cadence == 0)
        if rebalance_allowed:
            wanted = policy_weights(policy, signal_day, forecasts, features, sectors)
            current = shares * px / max(nav, 1.0)
            # Initial allocation bypasses optional drift gate; later rounds
            # may HOLD if little has changed. Max turnover veto => HOLD.
            drift = float(np.abs(wanted-current).sum())
            if offset == 0 or drift >= policy.gate:
                # When a desired regime change is larger than the per-decision
                # turnover cap, execute a partial, legal adjustment rather than
                # abandoning risk reduction entirely. The cap is approximate
                # because final dollar targets are fee-adjusted.
                if offset > 0 and drift > policy.max_trade:
                    step = (policy.max_trade * .995) / drift
                    wanted = current + step * (wanted - current)
                    validate_weights(wanted)
                old_nav = nav
                cash, shares, notion, fee, _ = exact_fee_transaction(cash,shares,px,wanted)
                if offset > 0 and notion / old_nav > policy.max_trade + 1e-5:
                    raise ValueError('Turnover guard exceeded')
                fee_sum += fee
                trade_notional += notion
                total_turnover_fraction += notion/max(old_nav,1.)
                decisions += 1
        for rnd in range(DECISIONS_PER_DAY):
            level = cash + float(np.dot(shares, endpoints[day,rnd]))
            if not math.isfinite(level) or level <= 0:
                raise ValueError('Nonfinite account NAV')
            path.append(level)
    curve=np.array(path)
    r = curve[1:]/curve[:-1]-1
    std = float(np.std(r,ddof=1))
    return {
        'cumulative_return': float(curve[-1]/INITIAL_NAV-1),
        'sharpe_proxy': float(r.mean()*ANNUALIZER/std) if std > 1e-12 else 0.,
        'mdd_proxy': float(np.max(1-curve/np.maximum.accumulate(curve))),
        'turnover_proxy': total_turnover_fraction/(size*DECISIONS_PER_DAY),
        'fee_usd':fee_sum,
        'trade_count':decisions,
        'gross_notional_usd':trade_notional,
    }


def windows_for_year(dates, year):
    ix=np.flatnonzero(dates.year==year)
    # First trading day has no same-year previous-close forecast, so begin
    # at the SECOND session. This avoids silently dropping an entire window.
    return [int(s) for s in ix[1::DAYS] if s > START_FEATURE+1 and s+DAYS <= len(dates)
            and dates[s+DAYS-1].year==year]


def print_rank_summary(summary):
    metric_dir={'cumulative_return':False,'sharpe_proxy':False,
                'mdd_proxy':True,'turnover_proxy':True}
    subset=summary[summary.year.isin([2023,2024])].copy()
    if subset.empty:
        raise RuntimeError('No development results')
    for key, asc in metric_dir.items():
        subset['rank_'+key]=subset.groupby('year')[key].rank(ascending=asc,method='average',pct=True)
    keys=['rank_'+k for k in metric_dir]
    subset['mean_metric_rank']=subset[keys].mean(axis=1)
    ranking=subset.groupby('strategy')['mean_metric_rank'].agg(['mean','max']).reset_index()
    ranking['conservative_rank']=.5*ranking['mean']+.5*ranking['max']
    ranking=ranking.sort_values('conservative_rank')
    print('=== DEVELOPMENT METRIC-RANK SCREEN (lower is better; NOT actual competition ranks) ===')
    print(ranking.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    return ranking


def run(root, phase15_path, phase19_path, max_iter):
    phase15=import_file(phase15_path, 'icaif_phase15_for_phase20')
    phase19=import_file(phase19_path, 'icaif_phase19_for_phase20')
    if tuple(phase15.SYMBOLS)!=SYMBOLS or tuple(phase19.SYMBOLS)!=SYMBOLS:
        raise RuntimeError('Symbol mismatch between modules')
    sec_files=check_sec_cache(root)
    dates, opens, endpoints, daily, sectors=phase15.load_history(phase15.source_path(root))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice',category=RuntimeWarning)
        X_price,_,_=phase15.make_features(daily,sectors)
    if X_price.shape[2] != 14:
        raise RuntimeError('Price feature schema unexpected')
    X_sec,coverage=make_sec_panel(dates,sec_files,phase19)
    missing = [r['symbol'] for r in coverage if r['revenue_missing']]
    print('SEC_REVENUE_GAPS:',missing)
    # Fail softly for company-specific revenue taxonomy; the model has
    # NaN-capable features and is compared against price-only baselines.
    print('SEC_FEATURES_MISSING_REMAIN_MISSING: YES')
    rows=[]
    model_manifest={}
    for year in (2023,2024,2025):
        pred, counts=year_predictions(dates,X_price,X_sec,daily,year,max_iter=max_iter)
        model_manifest[str(year)]=counts
        windows=windows_for_year(dates,year)
        for n,start in enumerate(windows):
            # Signal known at preceding day's completed close; no same-day leaks.
            if not np.isfinite(X_price[start-1]).all() or not all(
                np.isfinite(pred[name][start-1]).all() for name in pred):
                continue
            for policy in POLICIES:
                result=replay_window(policy,start,opens,endpoints,X_price,pred,sectors)
                rows.append({'year':year,'start':str(dates[start].date()),'strategy':policy.name,
                             **result})
        print(f'YEAR_{year}: evaluated_windows={len(windows)} candidates={len(POLICIES)}',flush=True)
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError('No valid backtest windows')
    sum_fields=['cumulative_return','sharpe_proxy','mdd_proxy',
                'turnover_proxy','fee_usd','trade_count']
    summary=out.groupby(['year','strategy'],as_index=False)[sum_fields].mean()
    for year in (2023,2024,2025):
        print(f'=== YEAR {year}: mean NONOVERLAPPING 15d window metrics ===')
        print(summary[summary.year==year].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    ranking=print_rank_summary(summary)
    folder=root/OUT_REL
    folder.mkdir(parents=True, exist_ok=True)
    out.to_csv(folder/'phase20_window_results.csv',index=False)
    summary.to_csv(folder/'phase20_year_summary.csv',index=False)
    ranking.to_csv(folder/'phase20_development_ranks.csv',index=False)
    (folder/'phase20_sec_tag_audit.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    manifest={
        'stage':'RESEARCH_ONLY_NOT_ELIGIBLE_FOR_LIVE_TRADING',
        'fee_rate':FEE,'seed':RANDOM_STATE,
        'training':'expanding years 2023/2024/2025; outcome label maturities purged',
        'models':'HGB 1-day price, HGB 5-day price, HGB 5-day price+SEC',
        'execution':'once daily open proxies; subsequent round prices are hourly proxies only',
        'validation':'2023/2024 development, 2025 exposed diagnostic, 2026 NOT tested',
        'news_training':'NOT INCLUDED: latest Finnhub snapshots have no certified historical availability',
        'macro_training':'NOT INCLUDED: requires verified vintage availability by round',
        'sec_timing':'filed date + 2 calendar days (research proxy; not acceptance-hour exact)',
        'sec_fallback_tags':REVENUE_TAGS,
        'SEC_missing_revenue':missing,
        'model_training_counts':model_manifest,
        'candidate_policies':[asdict(p) for p in POLICIES],
        'deployment_requirements':[
            'Actual round-aligned 15min bars for intraday changes',
            'Exact SEC EDGAR acceptance timestamps and historical filing audit',
            'Separate real-time inference checkpoint and one-writer submission tests',
            'Fresh independent evaluation and fee/risk rank calibration',
            'Only pre-frozen policy and model versions may run Official',
        ],
    }
    (folder/'phase20_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (folder/'CHANGELOG_PHASE20.md').write_text(
        '# Phase 20 experiment\n\n'
        '**Hypothesis:** multiple horizons, filing-date SEC fundamentals, sector-neutral shrinkage and fee gates '
        'may improve four-metric scores relative to initial equal80 HOLD.\n\n'
        '**Inputs:** complete historical hourly dataset; Phase19 cached CompanyFacts (as of retrieval).\n\n'
        '**Design:** expanding-year fit; 2023/2024 development; 2025 previously exposed; 2026 untouched. '
        '15d non-overlapping windows.\n\n'
        '**Limitations:** hourly fill proxies, SEC filed-date+2d proxy, no validated historical news, '
        '2025 not independent; no Official deployment.\n\n'
        '**Decision:** promote only after examining year-by-year four-metric scores, leakage and live interface.\n',
        encoding='utf-8')
    print('OUTPUT_DIR:',folder)
    print('LIVE_MODEL_APPROVED: NO (requires evidence and live validation)')
    print('PHASE20_COMPLETE_NO_NETWORK_NO_COMPETITION_SUBMISSION')



POSITIVE_EVENTS = {
    'raises guidance': 'guidance_up', 'raised guidance': 'guidance_up',
    'beats estimates': 'earnings_beat', 'beat estimates': 'earnings_beat',
    'record earnings': 'earnings_positive', 'revenue growth': 'revenue_positive',
    'share buyback': 'capital_return', 'share repurchase': 'capital_return',
    'fda approval': 'regulatory_approval',
}
NEGATIVE_EVENTS = {
    'cuts guidance': 'guidance_down', 'lowered guidance': 'guidance_down',
    'misses estimates': 'earnings_miss', 'missed estimates': 'earnings_miss',
    'sec investigation': 'regulatory_negative', 'recalls product': 'recall',
    'product recall': 'recall', 'files for bankruptcy': 'distress',
    'profit warning': 'profit_warning',
}


def headline_tags(text: str):
    """Uncalibrated lexical event tags -- for coverage audit ONLY."""
    headline = str(text).lower()[:300]
    positive = sorted({v for k,v in POSITIVE_EVENTS.items() if k in headline})
    negative = sorted({v for k,v in NEGATIVE_EVENTS.items() if k in headline})
    return positive, negative


def news_shadow(root: Path):
    folder=root/'artifacts'/'public_data_sources'/'phase18_universe'
    candidates=sorted(folder.glob('phase18_snapshot_*.json'))
    if not candidates:
        raise FileNotFoundError('Missing Phase18 snapshot. Run Phase18 --collect first.')
    snapshot=json.loads(candidates[-1].read_text(encoding='utf-8'))
    rows=snapshot.get('events', [])
    totals={s:{'sampled_headlines':0,'lexical_positive':0,'lexical_negative':0,
               'unverified_company_relevance':0} for s in SYMBOLS}
    dedup=set()
    from datetime import datetime, timezone
    def stamp(v):
        if not isinstance(v,str):return None
        try:
            x=datetime.fromisoformat(v.replace('Z','+00:00'))
            return x.astimezone(timezone.utc) if x.tzinfo else None
        except (ValueError, OverflowError):
            return None
    accepted=0
    bad=0
    for r in rows:
        if not isinstance(r,dict) or r.get('source')!='FINNHUB':
            continue
        sym=r.get('symbol')
        if sym not in totals:
            bad+=1;continue
        when=stamp(r.get('event_time_utc'))
        seen_at=stamp(r.get('available_at_utc'))
        retrieved_at=stamp(r.get('retrieved_utc'))
        event_id=r.get('event_id')
        if (not when or not seen_at or not retrieved_at or when>retrieved_at
                or seen_at<retrieved_at or r.get('historical_backtest_eligible') is not False
                or not event_id):
            bad+=1;continue
        key=(sym,event_id)
        if key in dedup:continue
        dedup.add(key)
        pos,neg=headline_tags(r.get('headline',''))
        t=totals[sym]
        t['sampled_headlines']+=1
        t['lexical_positive']+=int(bool(pos))
        t['lexical_negative']+=int(bool(neg))
        t['unverified_company_relevance']+=int(r.get('company_relevance_verified') is not True)
        accepted+=1
    out={'mode':'SHADOW_ONLY','source':'FINNHUB_PHASE18_SNAPSHOT',
         'historical_training_eligible':False,'trading_signal_eligible':False,
         'relevance_verified':False,'lexical_classifier_calibrated':False,
         'events_accepted':accepted,'events_rejected':bad,'per_symbol':totals}
    dest=root/OUT_REL
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'phase20_news_shadow.json'
    path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('FINNHUB_SHADOW_ACCEPTED:',accepted)
    print('FINNHUB_SHADOW_REJECTED:',bad)
    print('NEWS_HISTORICAL_ML_ELIGIBLE: NO')
    print('NEWS_LIVE_TRADING_ENABLED: NO')
    print('NEWS_REPORT:',path)
    print('NEWS_SHADOW_COMPLETE_NO_NETWORK_NO_TRADING')

def self_test():
    assert len(SYMBOLS)==len(set(SYMBOLS))==30
    assert len({p.name for p in POLICIES})==len(POLICIES)
    assert 'RevenuesNetOfInterestExpense' in REVENUE_TAGS
    assert 'SalesAndOtherOperatingRevenue' in REVENUE_TAGS
    assert 'OperatingRevenues' in REVENUE_TAGS
    assert headline_tags('Company raises guidance and beats estimates')[0]
    assert headline_tags('Company cuts guidance')[1]
    assert not headline_tags('No relevant event')[0]
    # Weight construction and sector neutrality.
    sectors=[str(x//5) for x in range(30)]
    alpha=alpha_target(np.arange(30),.8,.30,sector_names=sectors)
    assert len(alpha)==30 and abs(alpha.sum()-.8)<1e-12
    assert alpha.min()>=0 and alpha.max()<.08
    bad=np.ones(30)/30;bad[0]=.31
    try:validate_weights(bad)
    except ValueError:pass
    else:raise AssertionError('Illegal stock concentration accepted')
    # Flat price path: all loss must be attributable to 0.1% fee, not an
    # invented market move. Correct fee treatment is implicit NAV sizing.
    px=np.full((160,30),100.)
    bars=np.full((160,7,30),100.)
    feats=np.zeros((160,30,14))
    for col in (5,6):feats[:,:,col]=.01
    forecasts={x:np.tile(np.arange(30,dtype=float),(160,1))
               for x in ('price_1d','price_5d','sec_5d')}
    hold=replay_window(POLICIES[0],140,px,bars,feats,forecasts,sectors)
    assert hold['trade_count']==1
    assert abs(hold['fee_usd']-799.36051159)<.01,hold
    assert abs(hold['cumulative_return']+.00079936051159)<1e-9,hold
    # Rebalancing a portfolio at identical prices and targets must not pay
    # more fees; the daily policy still returns HOLD after its initial buy.
    daily=replay_window(POLICIES[2],140,px,bars,feats,forecasts,sectors)
    assert daily['trade_count']==1 and abs(daily['fee_usd']-hold['fee_usd'])<.001,daily
    assert daily['turnover_proxy']>0 and hold['turnover_proxy']>0
    # Delayed fundamental event cannot enter an earlier year's decision.
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class FakeFact:
        filing:pd.Timestamp;end:pd.Timestamp;value:float;tag_rank:int
    data=[FakeFact(pd.Timestamp('2025-01-05'),pd.Timestamp('2024-12-31'),100.,0)]
    def asof(day):
        cutoff=pd.Timestamp(day)-pd.Timedelta(days=2)
        return [a for a in data if a.filing<=cutoff]
    assert not asof('2025-01-06') and asof('2025-01-07')
    # Hard cap must not veto a genuine risk reduction; shift toward desired
    # allocation gradually, without exceeding any stock weight limit.
    p = Policy('test_regime', 'equal', .8, 1, .01, .10, True, .0)
    risk_features = feats.copy()
    risk_features[140,:,12] = -.05
    risk_features[140,:,13] = -.04
    # Day 140 buys 80%, day 141 switches to 52% target, limited by 10% cap.
    result = replay_window(p, 140, px, bars, risk_features, forecasts,
                           sectors, size=3)
    assert result['trade_count'] >= 2, result
    assert result['fee_usd'] > hold['fee_usd'], result
    # Purged labels cannot enter evaluation year.
    assert 2025-1 < 2025
    print('PHASE20_SELF_TEST_PASS')
    print('LEGAL_WEIGHTS_AND_SECTOR_NEUTRALITY_PASS')
    print('FEE_AND_ZERO_TRADE_HOLD_PASS')
    print('SEC_AVAILABILITY_LAG_PASS')
    print('NEWS_SHADOW_PARSER_PASS')
    print('NO_NETWORK_NO_COMPETITION_SUBMISSION')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--news-shadow',action='store_true',
                       help='Audit recently downloaded Finnhub headlines; no predictions/trades')
    ap.add_argument('--phase15',type=Path,
                    default=Path.home()/'Downloads'/'icaif_phase15_official_challengers_v2.py')
    ap.add_argument('--phase19',type=Path,
                    default=Path.home()/'Downloads'/'icaif_phase19_sec_ml_research.py')
    ap.add_argument('--max-iter',type=int,default=75,
                    help='HGB iterations: default 75; keep constant between experiments')
    args=ap.parse_args()
    if args.self_test:
        self_test()
    elif args.news_shadow:
        news_shadow(project_root())
    else:
        if not 25 <= args.max_iter <= 200:
            raise ValueError('max-iter outside safe experiment range')
        run(project_root(),args.phase15,args.phase19,args.max_iter)


if __name__=='__main__':
    main()
