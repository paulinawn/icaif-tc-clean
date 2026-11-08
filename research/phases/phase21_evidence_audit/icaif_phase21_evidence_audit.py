#!/usr/bin/env python3
r"""ICAIF Phase 21 evidence audit: test price-basis integrity + paired window comparisons.

Usage inside starter-kit with activated .venv:
  python "$HOME\Downloads\icaif_phase21_evidence_audit.py" --self-test
  python "$HOME\Downloads\icaif_phase21_evidence_audit.py" --audit

Research-only: local CSV + local Parquet; no network, no credential access, no submissions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

SYMBOLS = tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
METRICS = {
    'cumulative_return': False,
    'sharpe_proxy': False,
    'mdd_proxy': True,
    'turnover_proxy': True,
}
BASELINE = 'equal80_hold'


def project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / 'tools' / 'auto_submit.py').is_file():
        raise RuntimeError('Open the starter-kit folder in VS Code before running.')
    return root


def check_window_data(df: pd.DataFrame) -> None:
    cols = set(METRICS) | {'year', 'start', 'strategy'}
    if not cols.issubset(df.columns):
        raise ValueError('Phase20 per-window CSV missing: ' + ', '.join(sorted(cols - set(df.columns))))
    if df.duplicated(['year', 'start', 'strategy']).any():
        raise ValueError('Duplicate (year, start, strategy) records: cannot rank.')
    for k in METRICS:
        if not np.isfinite(pd.to_numeric(df[k], errors='coerce')).all():
            raise ValueError('Nonfinite metrics found in ' + k)
    groups = df.groupby(['year', 'start'], sort=True)
    if len(groups) < 20:
        raise ValueError('Too few windows; expected Phase20 full history.')
    for key, d in groups:
        if BASELINE not in set(d['strategy']):
            raise ValueError(f'Missing baseline for {key}')
        if len(d) != 12:
            raise ValueError(f'Expected 12 policies in {key}, got {len(d)}')


def paired_policy_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_window_data(df)
    rows = []
    for (year, start), group in df.groupby(['year', 'start'], sort=True):
        g = group.copy()
        for metric, asc in METRICS.items():
            g['rank_' + metric] = g[metric].rank(ascending=asc, method='average', pct=True)
        rank_cols = ['rank_' + m for m in METRICS]
        g['window_rank'] = g[rank_cols].mean(axis=1)
        base = g.loc[g.strategy == BASELINE].iloc[0]
        for _, x in g.iterrows():
            rows.append({
                'year': int(year), 'start': str(start), 'strategy': x['strategy'],
                'delta_return': float(x['cumulative_return'] - base['cumulative_return']),
                'delta_sharpe': float(x['sharpe_proxy'] - base['sharpe_proxy']),
                'drawdown_reduction': float(base['mdd_proxy'] - x['mdd_proxy']),
                'turnover_reduction': float(base['turnover_proxy'] - x['turnover_proxy']),
                'window_rank': float(x['window_rank']),
                'relative_rank_improvement': float(base['window_rank'] - x['window_rank']),
            })
    paired = pd.DataFrame(rows)
    summaries = []
    for period, subset in [('DEVELOPMENT_2023_2024', paired[paired.year.isin([2023, 2024])]),
                           ('EXPOSED_2025', paired[paired.year == 2025])]:
        for name, g in subset.groupby('strategy'):
            summaries.append({
                'period': period, 'strategy': name, 'windows': len(g),
                'mean_window_rank': g.window_rank.mean(),
                'mean_rank_improvement': g.relative_rank_improvement.mean(),
                'rank_win_fraction': float((g.relative_rank_improvement > 1e-10).mean()),
                'mean_return_delta_pctpt': 100*g.delta_return.mean(),
                'return_win_fraction': float((g.delta_return > 1e-10).mean()),
                'median_drawdown_saved_pctpt': 100*g.drawdown_reduction.median(),
                'median_turnover_saved_pctpt': 100*g.turnover_reduction.median(),
            })
    return paired, pd.DataFrame(summaries)


def price_basis_audit(path: Path) -> dict:
    # A purely local audit. The official parquet schema may or may not include
    # an unadjusted close, which is necessary to prove the price-base mismatch.
    frame = pd.read_parquet(path)
    needed = {'ticker', 'timestamp_et', 'open', 'adj_close'}
    if not needed.issubset(frame.columns):
        raise ValueError('Expected parquet columns missing: ' + str(sorted(needed - set(frame.columns))))
    df = frame[frame.ticker.isin(SYMBOLS)].copy()
    df['timestamp_et'] = pd.to_datetime(df['timestamp_et'], errors='coerce')
    df['year'] = df['timestamp_et'].dt.year
    df = df[df.year.isin((2022, 2023, 2024, 2025))]
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
    has_close = 'close' in df.columns
    has_adj_open = 'adj_open' in df.columns
    result = {
        'parquet_columns': [str(c) for c in frame.columns],
        'rows_checked': int(len(df)),
        'raw_close_available': has_close,
        'adjusted_open_available': has_adj_open,
        'basis_status': 'INCONCLUSIVE',
        'suspect_symbols': [],
        'examples': [],
    }
    if not has_close:
        # We can warn about gross jumps, not prove a specific adjustment factor.
        df['ratio'] = df['adj_close'] / df['open']
        for s, part in df.groupby('ticker'):
            med = float(part['ratio'].median())
            if math.isfinite(med) and not (0.95 <= med <= 1.05):
                result['suspect_symbols'].append(str(s))
                result['examples'].append({'symbol': str(s), 'median_adjclose_over_open': round(med, 5)})
        result['basis_status'] = 'SUSPECT_RAW_OPEN_ADJCLOSE_MISMATCH' if result['suspect_symbols'] else 'INCONCLUSIVE_NO_RAW_CLOSE'
        return result

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    good = (np.isfinite(df['close']) & np.isfinite(df['adj_close'])
            & (df['close'] > 0) & (df['adj_close'] > 0))
    df = df[good].copy()
    if len(df) < 5000:
        raise ValueError('Insufficient raw and adjusted close observations')
    df['factor'] = df['adj_close'] / df['close']
    for (sym, year), g in df.groupby(['ticker', 'year']):
        factor = float(g['factor'].median())
        if not math.isfinite(factor):
            continue
        # 1% threshold avoids tiny vendor rounding differences.
        if abs(factor - 1.0) > .01:
            result['examples'].append({'symbol': str(sym), 'year': int(year),
                                       'median_adjusted_to_raw_close': round(factor, 6)})
            if sym not in result['suspect_symbols']:
                result['suspect_symbols'].append(str(sym))
    # The backtest explicitly uses raw 'open' even if adj_open exists.
    result['basis_status'] = ('CONFIRMED_MIXED_RAW_OPEN_ADJ_CLOSE'
                              if result['suspect_symbols'] else 'NO_LARGE_CLOSE_ADJUSTMENT_DETECTED')
    return result


def self_test():
    policies = [BASELINE] + [f'p{i}' for i in range(11)]
    rows = []
    for yr in (2023, 2024):
        for window in range(12):
            for n, name in enumerate(policies):
                rows.append({
                    'year': yr, 'start': f'{yr}-01-{window+1:02d}', 'strategy': name,
                    'cumulative_return': .01 + .001 * (n == 1),
                    'sharpe_proxy': 1.5 + .1*(n == 1),
                    'mdd_proxy': .03 - .002*(n == 1),
                    'turnover_proxy': .01,
                })
    paired, summary = paired_policy_audit(pd.DataFrame(rows))
    better = summary[(summary.strategy == 'p0') & (summary.period == 'DEVELOPMENT_2023_2024')].iloc[0]
    assert better['mean_return_delta_pctpt'] > 0
    assert better['rank_win_fraction'] == 1.0
    assert len(paired) == 24*12
    try:
        check_window_data(pd.DataFrame(rows[:-1]))
        raise AssertionError('Incomplete policy groups should be rejected')
    except ValueError:
        pass
    # Synthetic corporate-action price-basis mismatch, no Parquet dependency.
    from unittest.mock import patch
    synthetic_prices = pd.DataFrame({
        'ticker': ['AAPL'] * 6000,
        'timestamp_et': pd.date_range('2023-01-01', periods=6000, freq='h'),
        'open': [100.0] * 6000,
        'close': [100.0] * 6000,
        'adj_close': [80.0] * 6000,
    })
    with patch.object(pd, 'read_parquet', return_value=synthetic_prices):
        result = price_basis_audit(Path('synthetic_not_read.parquet'))
    assert result['basis_status'] == 'CONFIRMED_MIXED_RAW_OPEN_ADJ_CLOSE'
    assert result['suspect_symbols'] == ['AAPL']
    print('SYNTHETIC_ADJUSTMENT_BASIS_DETECTION_PASS')
    print('PHASE21_AUDIT_SELF_TEST_PASS')
    print('PER_WINDOW_PAIRED_RANKS_PASS')
    print('INCOMPLETE_OR_DUPLICATE_WINDOWS_REJECTED')
    print('NO_NETWORK_NO_TRADING')


def audit():
    root = project_root()
    data_path = root/'artifacts'/'phase20_adaptive_multihorizon'/'phase20_window_results.csv'
    if not data_path.exists():
        raise FileNotFoundError('Phase20 results missing: first run Phase20 --run.')
    df = pd.read_csv(data_path)
    paired, summary = paired_policy_audit(df)
    development = summary[summary.period == 'DEVELOPMENT_2023_2024']
    exposed = summary[summary.period == 'EXPOSED_2025']
    print('=== PHASE21 PAIRED WINDOW AUDIT ===')
    print('DEVELOPMENT: 2023-2024; 2025 PREVIOUSLY EXPOSED')
    print('WINDOWS:', len(paired[['year','start']].drop_duplicates()))
    print('Development period by mean paired rank (lower better):')
    keys = ['strategy', 'windows', 'mean_window_rank','rank_win_fraction',
            'mean_return_delta_pctpt','return_win_fraction','median_drawdown_saved_pctpt']
    print(development.sort_values('mean_window_rank')[keys].to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('Exposed 2025 diagnostic by mean paired rank (lower better):')
    print(exposed.sort_values('mean_window_rank')[keys].to_string(index=False, float_format=lambda x:f'{x:.4f}'))

    parquet = root/'data'/'official'/'hourly_market_data_2021_2026.parquet'
    if not parquet.is_file():
        raise FileNotFoundError('Historical parquet not found: ' + str(parquet))
    basis = price_basis_audit(parquet)
    print('=== PRICE-BASIS INTEGRITY CHECK ===')
    print('PRICE_BASIS_STATUS:', basis['basis_status'])
    print('RAW_CLOSE_COLUMN:', basis['raw_close_available'])
    print('ADJUSTED_OPEN_COLUMN:', basis['adjusted_open_available'])
    print('SUSPECT_SYMBOLS:', ','.join(basis['suspect_symbols']) or 'none')
    print('EXAMPLE_FACTORS:', basis['examples'][:12])
    print('BACKTEST_VALIDATION:', 'REQUIRES_RECONCILIATION' if basis['suspect_symbols'] else 'NOT_FULLY_VERIFIED')
    folder = root/'artifacts'/'phase21_evidence_audit'
    folder.mkdir(parents=True, exist_ok=True)
    paired.to_csv(folder/'phase21_paired_windows.csv', index=False)
    summary.to_csv(folder/'phase21_comparison.csv', index=False)
    (folder/'phase21_price_basis.json').write_text(json.dumps(basis,indent=2), encoding='utf-8')
    print('REPORT_DIR:', folder)
    print('MODEL_PROMOTION:', 'NOT_AUTHORIZED_FROM_THIS_AUDIT')
    print('PHASE21_AUDIT_COMPLETE_NO_SUBMISSION')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        audit()


if __name__ == '__main__':
    main()
