#!/usr/bin/env python3
"""ICAIF Phase24 v2: combine Phase20 ML forecasts with Phase22-v2 gap-audited allocation.

RESEARCH ONLY. No network/API/trading. Runs from starter-kit with .venv active:
 python "$HOME\\Downloads\\icaif_phase24_hierarchical_ml_v2.py" --self-test
 python "$HOME\\Downloads\\icaif_phase24_hierarchical_ml_v2.py" --run

Requires existing Phase15 v2, Phase19, Phase20, Phase22 scripts in Downloads,
Phase19 SEC CompanyFacts cache and historical parquet. Expanding-year ML training
with purged 1d/5d labels; prior-day forecasts only; historical fill prices are
hourly proxies. 2025 EXPOSED, 2026 reserved. All 15d windows reset independently.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

N=30
DAYS=15
OUTPUT=Path('artifacts/phase24_hierarchical_ml_v2')

@dataclass(frozen=True)
class Policy:
    name: str
    kind: str              # baseline | industry | ml
    exposure: float
    ml_fraction: float
    risk: bool
    cadence: int
    gate: float
    max_trade: float=.12

POLICIES=(
    Policy('equal80_hold', 'baseline', .80,0,False,0,0),
    Policy('equal90_hold', 'baseline', .90,0,False,0,0),
    Policy('equal100_hold','baseline', 1.00,0,False,0,0),
    Policy('industry90_hold','industry',.90,0,False,0,0),
    Policy('industry_risk90_hold','industry',.90,0,True,0,0),
    Policy('industry_ml90_t15_hold','ml',.90,.15,False,0,0),
    Policy('industry_ml90_t30_hold','ml',.90,.30,False,0,0),
    Policy('industry_ml_risk90_t15_hold','ml',.90,.15,True,0,0),
    Policy('industry_ml_risk90_t30_hold','ml',.90,.30,True,0,0),
    Policy('industry_ml_risk90_t30_weekly','ml',.90,.30,True,5,.06),
)

def import_download(name,short):
    folder=Path.home()/'Downloads'
    target=folder/name
    if not target.is_file():raise FileNotFoundError(f'Missing {name} in Downloads.')
    spec=importlib.util.spec_from_file_location(short,target)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def model_intra_sector_weights(base, forecast, groups, mix):
    if len(forecast)!=N or not np.all(np.isfinite(forecast)):
        raise ValueError('Missing ML forecast values: fail closed')
    if not 0<=mix<=.5:raise ValueError('ML strength too large')
    weight=np.array(base,float).copy()
    for g in sorted(set(groups)):
        ind=np.array([i for i,z in enumerate(groups) if z==g])
        if len(ind)!=5:raise ValueError('Invalid group size')
        a=pd.Series(forecast[ind]).rank(method='average',pct=True).to_numpy()*2-1
        a-=np.mean(a)
        factor=1+mix*a
        weight[ind]=weight[ind]*factor
        weight[ind]*=base[ind].sum()/weight[ind].sum()
    if np.any(weight<0) or np.any(weight>.30) or abs(weight.sum()-base.sum())>1e-8:
        raise ValueError('ML adjustment produced illegal weights')
    return weight


class Engine:
    def __init__(self,p22,p20,daily,groups,opens,endpoints,features,pred):
        self.p22,self.p20=p22,p20
        self.daily,self.groups=daily,groups
        self.opens,self.endpoints=opens,endpoints
        self.features,self.pred=features,pred
        self.cache={}
        self.fallbacks=0

    def desired(self,i,pol):
        key=(i,pol.name)
        if key in self.cache:return self.cache[key].copy()
        b=self.p22.industry_prior(self.daily,self.groups,i,gross=pol.exposure,
                                  active=pol.kind!='baseline')
        if pol.kind=='ml':
            fore=self.p20.model_signal(
                self.p20.Policy('phase24_blend','blend',pol.exposure,0,0,1,False,.15),
                i,self.pred,self.features)
            b=model_intra_sector_weights(b,fore,self.groups,pol.ml_fraction)
        if pol.risk:
            cov=self.p22.shrunk_covariance(self.daily,i)
            b,fail=self.p22.risk_optimized_prior(b,cov,pol.exposure,self.groups)
            self.fallbacks+=int(fail)
        self.p22.check_weights(b,pol.exposure)
        self.cache[key]=b.copy()
        return b

    def replay(self,start,pol,size=15):
        cash=1_000_000.
        shares=np.zeros(N)
        path=[cash]
        fees=notion=turnover=0.
        trade_count=0
        overshoot_count=0
        max_actual=0.0
        max_gap=0.0
        gap_anomalies=0
        overshoot_examples=[]
        for d in range(size):
            k=start+d
            px=self.opens[k]
            nav=cash+float(np.dot(shares,px))
            # Decision/gate cannot know today's execution open; use prior close.
            prior_px=self.daily[k-1]
            prior_nav=cash+float(np.dot(shares,prior_px))
            gaps=px/prior_px-1.0
            gap=float(np.max(np.abs(gaps)))
            max_gap=max(max_gap,gap)
            gap_anomalies+=int(gap>.35)
            if d==0 or (pol.cadence>0 and d%pol.cadence==0):
                intended=self.desired(k-1,pol).copy()
                current=shares*prior_px/prior_nav
                drift=float(np.abs(intended-current).sum())
                if d==0 or drift>=pol.gate:
                    if d>0:
                        intended,estimated=self.p22.limit_planned_turnover(current,intended,pol.max_trade)
                    else:
                        estimated=float('nan')
                    cash,shares,t,f,before=self.p22.fee_trade(cash,shares,px,intended)
                    fees+=f
                    notion+=t
                    realized=t/before
                    turnover+=realized
                    trade_count+=1
                    if d>0:
                        max_actual=max(max_actual,realized)
                        if realized>pol.max_trade+1e-5:
                            overshoot_count+=1
                            if len(overshoot_examples)<10:
                                worst_j=int(np.argmax(np.abs(gaps)))
                                overshoot_examples.append({
                                    'day_index':int(k),
                                    'strategy':pol.name,
                                    'estimated':round(estimated,8),
                                    'realized':round(realized,8),
                                    'cap':pol.max_trade,
                                    'max_gap_symbol':self.p22.SYMBOLS[worst_j],
                                    'max_gap_return':round(float(gaps[worst_j]),8)})
            for j in range(7):
                nav=cash+float(np.dot(shares,self.endpoints[k,j]))
                if not np.isfinite(nav) or nav<=0:raise ValueError('Invalid NAV')
                path.append(nav)
        curve=np.array(path)
        r=curve[1:]/curve[:-1]-1
        vol=float(np.std(r,ddof=1))
        return {'cumulative_return':float(curve[-1]/curve[0]-1),
                'sharpe_proxy':float(np.mean(r)*np.sqrt(252*7)/vol) if vol>1e-12 else 0,
                'mdd_proxy':float(np.max(1-curve/np.maximum.accumulate(curve))),
                'turnover_proxy':turnover/(size*7),
                'fees_usd':fees,'trade_count':trade_count,
                'realized_cap_overshoots':overshoot_count,
                'max_realized_rebalance_turnover':max_actual,
                'max_abs_open_gap':max_gap,
                'open_gap_over35_count':gap_anomalies,
                'overshoot_examples':overshoot_examples}


def summarize(rows,root):
    df=pd.DataFrame(rows)
    metrics=['cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd','trade_count',
             'realized_cap_overshoots','max_realized_rebalance_turnover','max_abs_open_gap',
             'open_gap_over35_count']
    summary=df.groupby(['year','strategy'],as_index=False)[metrics].mean()
    dev=df[df.year.isin((2023,2024))].copy()
    for metric,ascending in [('cumulative_return',False),('sharpe_proxy',False),
                             ('mdd_proxy',True),('turnover_proxy',True)]:
        dev['rank_'+metric]=dev.groupby(['year','start'])[metric].rank(ascending=ascending,pct=True)
    dev['four_metric_rank']=dev[[c for c in dev if c.startswith('rank_')]].mean(axis=1)
    ranks=dev.groupby(['strategy','year'])['four_metric_rank'].mean().unstack()
    ranks['dev_mean']=ranks[[2023,2024]].mean(axis=1)
    ranks['conservative_rank']=(ranks.dev_mean+ranks[[2023,2024]].max(axis=1))/2
    ranks=ranks.sort_values('conservative_rank')
    print('=== DEVELOPMENT 2023/2024 (paired 15d ranks; lower better) ===')
    print(ranks.to_string(float_format=lambda x:f'{x:.5f}'))
    for year in (2023,2024,2025):
        print(f'=== YEAR_{year} (mean 15-session-window research proxies) ===')
        print(summary[summary.year==year].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    dest=root/OUTPUT
    dest.mkdir(exist_ok=True,parents=True)
    df.to_csv(dest/'phase24_window_results.csv',index=False)
    summary.to_csv(dest/'phase24_year_summary.csv',index=False)
    ranks.to_csv(dest/'phase24_paired_ranks.csv')
    return dest


def run():
    root=Path.cwd().resolve()
    if not (root/'tools'/'auto_submit.py').is_file():raise RuntimeError('Run inside starter-kit.')
    p22=import_download('icaif_phase22_industry_risk_90_v2.py','phase22_for_phase24_v2')
    p20=import_download('icaif_phase20_adaptive_multihorizon.py','phase20_for_phase24')
    p19=import_download('icaif_phase19_sec_ml_research.py','phase19_for_phase24')
    p15=p22.load_phase15()
    if tuple(p20.SYMBOLS)!=p22.SYMBOLS or tuple(p19.SYMBOLS)!=p22.SYMBOLS:
        raise ValueError('Universe mismatch')
    dates,opens,endpoints,daily,raw_groups=p15.load_history(p15.source_path(root))
    groups=p22.SYMBOL_GROUPS
    if len(set(raw_groups))!=6 or any(len(set(raw_groups[j:j+5]))!=1 for j in range(0,N,5)):
        raise ValueError('Historical industry groups mismatch')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Mean of empty slice')
        features,_,_=p15.make_features(daily,raw_groups)
    sec_files=p20.check_sec_cache(root)
    sec_panel,sec_audit=p20.make_sec_panel(dates,sec_files,p19)
    print('SEC_MISSING_REVENUE:',[r['symbol'] for r in sec_audit if r['revenue_missing']])
    rows=[]
    fallback_total=0
    for year in (2023,2024,2025):
        forecasts,counts=p20.year_predictions(dates,features,sec_panel,daily,year,max_iter=75)
        engine=Engine(p22,p20,daily,groups,opens,endpoints,features,forecasts)
        windows=p22.start_windows(dates,year)
        for n,start in enumerate(windows):
            if not all(np.isfinite(forecasts[f][start-1]).all() for f in forecasts):
                raise RuntimeError('Missing forecast for historical decision; no forward fill.')
            for policy in POLICIES:
                rows.append({'year':year,'start':str(dates[start].date()),
                             'strategy':policy.name,**engine.replay(start,policy)})
            if (n+1)%4==0:print(f'YEAR_{year}_WINDOWS_COMPLETED: {n+1}/{len(windows)}',flush=True)
        fallback_total+=engine.fallbacks
    output=summarize(rows,root)
    manifest={'research_only':True,'candidate_policies':[asdict(p) for p in POLICIES],
              'training':'expanding years with 1d/5d label purge',
              'years':'2023-24 development, 2025 previously exposed, 2026 reserved',
              'model':'Phase20 HGB 1d-price,5d-price,5d-SEC; within-six-group tilts',
              'historical_sec_qualifier':'filing date + 2 days research approximation',
              'price_basis_verified':False,'risk_optimization_fallback_count':fallback_total,
              'not_exact_round_execution':True,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'keep_gate':'paired four-metric scores and ablation vs industry90, risk90 and equal90',
              'official_model_approved':False,
              'turnover_guard_semantics':'80% buffered prior-close estimate, not guaranteed fill cap',
              'realized_cap_overshoot_total':int(sum(row['realized_cap_overshoots'] for row in rows)),
              'large_open_gap_count':int(sum(row['open_gap_over35_count'] for row in rows)),
              'overshoot_examples':[e for row in rows for e in row['overshoot_examples']][:30]}
    (output/'phase24_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'CHANGELOG_PHASE24.md').write_text(
        '# Phase24 — hierarchical ML\n\n'
        'Hypothesis: 1-day / 5-day/ SEC ML stock ranking inside six sector budgets, '
        'then covariance-aware risk optimization, improves fee-aware four-metric rank.\n\n'
        'Training: chronological expanding-year with purged labels. '
        '2023–2024 dev, exposed 2025 diagnostic, 2026 untouched.\n\n'
        'Decision: no Official promotion without actual out-of-sample and live feature equivalence.\n'
        'v2: buffered prior-close turnover projection; logs fill-price overshoots for audit.\n',
        encoding='utf-8')
    print('RISK_OPTIMIZER_FALLBACKS:',fallback_total)
    print('REALIZED_TURNOVER_CAP_OVERSHOOTS:',manifest['realized_cap_overshoot_total'])
    print('MAX_OPEN_GAP_ABSOLUTE:',max(row['max_abs_open_gap'] for row in rows))
    print('REVIEW_OVERSHOOTS:',manifest['overshoot_examples'][:10])
    print('RESEARCH_ONLY_NOT_A_STRICT_FILL_TIME_CAP')
    print('REPORT_DIR:',output)
    print('PHASE24_COMPLETE_NO_TRADING')


def self_test():
    # Zero network/mock pure-NumPy test of hierarchical ML mapping and covariance risks.
    fake=np.full(N,.9/N)
    group=tuple(g for g in ('A','B','C','D','E','F') for _ in range(5))
    forecasts=np.linspace(-.1,.1,N)
    x=model_intra_sector_weights(fake,forecasts,group,.3)
    assert abs(x.sum()-.9)<1e-10
    assert x.max()<.05 and x.min()>.01
    for g in sorted(set(group)):
        ii=[i for i,v in enumerate(group) if v==g]
        assert abs(x[ii].sum()-.15)<1e-10
    try:
        invalid=forecasts.copy();invalid[3]=np.nan
        model_intra_sector_weights(fake,invalid,group,.3)
        raise AssertionError('Missing forecast accepted')
    except ValueError:pass
    # Use same prior-mark turnover logic as Phase22; no future opening data.
    dep=Path(__file__).with_name('icaif_phase22_industry_risk_90_v2.py')
    if not dep.is_file():raise RuntimeError('Put both v2 research files together')
    spec=importlib.util.spec_from_file_location('p22_v2_for_test',dep)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    cur=np.full(N,.03)
    dst=cur.copy();dst[0]+=.06;dst[1:4]-=.02
    bounded,planned=module.limit_planned_turnover(cur,dst,.12)
    assert planned<=.096+1e-10 and abs(sum(bounded)-.9)<1e-9
    print('PHASE24_PRIOR_MARK_BUFFER_TEST_PASS')
    print('PHASE24_SELF_TEST_PASS')
    print('ML_WITHIN_INDUSTRY_GROUP_SHRINKAGE_PASS')
    print('MISSING_FORECAST_FAIL_CLOSED_PASS')
    print('NO_NETWORK_NO_ORGANIZER_SUBMISSION')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:run()
