#!/usr/bin/env python3
r"""ICAIF 2026 Phase 27: offline adaptive execution and walk-forward alpha weights.

Run from the existing starter-kit with the Python virtual environment active.
The ORIGINAL Phase 26 script must also be present in Downloads.

  python "$HOME\Downloads\icaif_phase27_adaptive_portfolio_lab.py" --self-test
  python "$HOME\Downloads\icaif_phase27_adaptive_portfolio_lab.py" --run

Research only: no trading endpoints, HTTP, credentials, server or Git changes.

IMPORTANT LIMITATIONS:
- Price-only signals. Cannot train SEC/Finnhub/LLM event coefficients without
  admissible timestamped historical event features and point-in-time archives.
- Annual WALK-FORWARD feature weights: 2023 model trained with 2021-22 only,
  2024 with 2021-23, 2025 (already exposed) with 2021-24. 2026 NEVER evaluated.
- Signal weighting is not validated alpha or an optimal ML model.
- Seven historic price marks are HOURLY PROXIES rather than precise Official
  execution/16:00 valuation prints. Staging at rounds 2/3 is indicative only.
- Independent 15-day windows, not a continuous 105-round Official account.
- No 'best strategy' selected by inspecting 2025; treat all rankings exploratory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SEED=20261011
NAV0=1_000_000.0
FEE=0.001
OUT=Path('artifacts/phase27_adaptive_portfolio_lab')
NAME='icaif_phase26_integrity_alpha_tournament.py'
SIGNALS=('mom20','mom60','mom120','rev5','lowvol60')

@dataclass(frozen=True)
class Policy:
    name: str
    signal: str  # equal, industry, momentum, learned
    equity: float = .8
    cadence: str = 'hold'  # hold, weekly, daily, staged
    gate: float = .04
    variable_equity: bool = False

POLICIES=(
    Policy('equal60_hold','equal',.6),
    Policy('equal80_hold','equal',.8),
    Policy('industry80_hold','industry',.8),
    Policy('momentum80_hold','momentum',.8),
    Policy('industry80_weekly_gate04','industry',.8,'weekly',.04),
    Policy('industry80_daily_gate04','industry',.8,'daily',.04),
    Policy('momentum80_weekly_gate04','momentum',.8,'weekly',.04),
    Policy('momentum80_daily_gate04','momentum',.8,'daily',.04),
    Policy('learned80_hold','learned',.8),
    Policy('learned80_weekly_gate04','learned',.8,'weekly',.04),
    Policy('learned80_daily_gate04','learned',.8,'daily',.04),
    Policy('learned_variable_daily_gate04','learned',.8,'daily',.04,True),
    Policy('industry80_staged_3rounds','industry',.8,'staged'),
    Policy('learned80_staged_3rounds','learned',.8,'staged'),
)


def phase26_module():
    p=Path.home()/'Downloads'/NAME
    if not p.is_file():
        raise FileNotFoundError('Phase 26 is required in Downloads: '+NAME)
    spec=importlib.util.spec_from_file_location('icaif_p26_dependency',p)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load Phase 26 dependency')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def feature_rows(p26,daily,i):
    f=p26.strength_features(daily,i)
    return {
        'mom20':f['r20'], 'mom60':f['r60'], 'mom120':f['r120'],
        'rev5':-f['r5'], 'lowvol60':-f['vol60'],
    }


def train_alpha(p26,dates,opening,ends,year):
    """Only completed pre-year targets; 1d and 5d next-open-to-future-mark IC.

    Annual rolling training mimics a model trained BEFORE Jan 1 of each test year.
    ICs are evaluated on past known returns only. Raw sign is intentionally NOT
    inverted: negative empirical IC makes a signal receive zero allocation.
    """
    cutoff=int(np.searchsorted(dates.values,np.datetime64(f'{year}-01-01')))
    if cutoff<300:
        raise ValueError(f'Not enough past training history for {year}')
    daily=ends[:,-1,:]
    table={name:{1:[],5:[]} for name in SIGNALS}
    for i in range(125,cutoff-5):
        f=feature_rows(p26,daily,i)
        for h in (1,5):
            if h==5 and i%5:continue
            # Trade from t+1 first historical mark, value at t+h final mark.
            # Both endpoints are strictly prior to the tested year.
            future=ends[i+h,-1,:]/opening[i+1,0,:]-1.
            if not np.isfinite(future).all():continue
            yrank=p26.ranks(future)
            for name,v in f.items():
                xrank=p26.ranks(v)
                ic=float(np.corrcoef(xrank,yrank)[0,1])
                if math.isfinite(ic):table[name][h].append(ic)
    raw={}
    counts={}
    for name in SIGNALS:
        if len(table[name][1])<200 or len(table[name][5])<35:
            raise ValueError(f'Insufficient matured training cases for {year}, {name}')
        means=[float(np.mean(table[name][h])) for h in (1,5)]
        raw[name]=float(np.mean(means))
        counts[name]={'one_day':len(table[name][1]),'five_day':len(table[name][5])}
    # Training-only positive IC weighting with shrinkage toward a uniform prior.
    # Shrinkage prevents tiny, noisy IC differences causing extreme all-or-nothing
    # allocations. It is NOT optimized against evaluated test periods.
    positive=np.array([max(0.,raw[name]) for name in SIGNALS])
    if float(positive.sum())>1e-12:
        learned=positive/positive.sum()
        weights=.70*learned+.30*np.full(len(SIGNALS),1./len(SIGNALS))
    else:
        weights=np.full(len(SIGNALS),1./len(SIGNALS))
    result={name:float(weights[j]) for j,name in enumerate(SIGNALS)}
    assert abs(sum(result.values())-1)<1e-12
    return {'test_year':year,'last_possible_matured_label_index':cutoff-6,
            'last_training_day':str(dates[cutoff-1].date()),'raw_train_IC':raw,
            'signal_weights':result,'counts':counts}


def exposure_regime(daily,t):
    """Timestamp-safe, deterministic exposure rule based on completed prior marks."""
    if t<121:raise ValueError('Warmup history insufficient')
    recent=daily[t-60:t+1,:]
    eq=np.mean(recent[1:]/recent[:-1]-1.,axis=1)
    vol=float(np.std(eq,ddof=1))
    trend=float(np.mean(daily[t]/daily[t-60]-1.))
    # All fixed decision thresholds; no 'fitted' 2025 levels.
    if trend<-.08 or vol>.018:return .60
    if trend>.04 and vol<.012:return .90
    return .80


def wanted(p26,pol,daily,t,weights):
    if pol.signal in ('equal','industry','momentum'):
        ans=p26.target(p26.Policy('p27',pol.equity,pol.signal),daily,t)
    elif pol.signal=='learned':
        raw=feature_rows(p26,daily,t)
        score=sum(weights[k]*p26.ranks(raw[k]) for k in SIGNALS)
        scored=p26.ranks(score)
        w=1.+.45*scored
        ans=p26.validate_weights(pol.equity*w/w.sum(),pol.equity)
    else:
        raise ValueError('Unrecognized policy')
    if pol.variable_equity:
        gross=exposure_regime(daily,t)
        ans=p26.validate_weights(ans*(gross/ans.sum()),gross)
    return ans


def simulate(p26,pol,start,opening,ends,daily,weights,size=15):
    if pol.cadence not in {'hold','daily','weekly','staged'}:raise ValueError('Bad cadence')
    cash=NAV0
    shares=np.zeros(30,dtype=float)
    nav_path=[NAV0]
    trades=0
    fee_total=0.
    traded_fraction=0.
    max_realized_rebalance=0.
    for d in range(size):
        i=start+d
        t=i-1
        # Only data available at the end of previous session enter the target.
        target=wanted(p26,pol,daily,t,weights)
        for rnd in range(7):
            if d==0 and pol.cadence=='staged':
                do_trade=rnd in (0,1,2)
                goal=target*((1./3,2./3,1.)[rnd]) if do_trade else target
            elif d==0 and rnd==0:
                do_trade=True
                goal=target
            elif rnd==0 and pol.cadence in ('daily','weekly'):
                do_trade=(pol.cadence=='daily' or d%5==0)
                goal=target
            else:
                do_trade=False
                goal=target
            if do_trade:
                # Intra-day staged fills approximate the official 10:30/11:30
                # execution slots using the historic proxy marks. Intent stays
                # frozen at the previous close, without peek at current prices.
                price=opening[i,0,:] if rnd==0 else ends[i,rnd,:]
                prior=daily[t,:] if rnd==0 else None
                if d==0 and pol.cadence=='staged':
                    pass  # mandatory preplanned tranche; no recent-price peek
                elif d>0:
                    # Decision time uses prior completed closing marks and prior
                    # holdings, never next morning's unknown execution mark.
                    pre_nav=float(cash+np.dot(shares,prior))
                    existing=shares*prior/pre_nav
                    drift=float(np.abs(goal-existing).sum())
                    if drift<pol.gate:
                        do_trade=False
                    else:
                        # 12% research cap, leave 20% buffer for unknown fill gap.
                        scale=min(1.,.12*.8/max(drift,1e-12))
                        goal=p26.validate_weights(existing+scale*(goal-existing))
                if do_trade:
                    pre_nav_real=float(cash+np.dot(shares,price))
                    cash,shares,notional=p26.execution(cash,shares,price,goal)
                    frac=notional/max(pre_nav_real,1.)
                    if d>0:max_realized_rebalance=max(max_realized_rebalance,frac)
                    traded_fraction+=frac
                    fee_total+=FEE*notional
                    trades+=1
            val=float(cash+np.dot(shares,ends[i,rnd,:]))
            if not math.isfinite(val) or val<=0:raise ValueError('Invalid NAV')
            nav_path.append(val)
    curve=np.array(nav_path)
    ret=curve[1:]/curve[:-1]-1
    std=float(ret.std(ddof=1))
    return {'cumulative_return':float(curve[-1]/NAV0-1),
            'sharpe_proxy':float(ret.mean()*np.sqrt(252*7)/std) if std>1e-12 else 0.,
            'mdd_proxy':float(np.max(1.-curve/np.maximum.accumulate(curve))),
            'turnover_proxy':float(traded_fraction/(size*7)),
            'trade_count':int(trades),'fees_usd':float(fee_total),
            'max_realized_rebalance_turnover':float(max_realized_rebalance)}


def evaluate(p26,dates,opening,ends,models,years=(2023,2024,2025)):
    daily=ends[:,-1,:]
    records=[]
    for year in years:
        windows=p26.starts(dates,year)
        weights=models[year]['signal_weights']
        for j,start in enumerate(windows):
            for pol in POLICIES:
                rec=simulate(p26,pol,start,opening,ends,daily,weights)
                records.append({'year':year,'start':str(dates[start].date()),'strategy':pol.name,**rec})
            if (j+1)%8==0:
                print(f'PHASE27_{year}_WINDOWS: {j+1}/{len(windows)}',flush=True)
    return pd.DataFrame(records)


def score(p26,results):
    rank=results.copy()
    for col,asc in [('cumulative_return',False),('sharpe_proxy',False),('mdd_proxy',True),('turnover_proxy',True)]:
        rank['rank_'+col]=rank.groupby(['year','start'])[col].rank(method='average',pct=True,ascending=asc)
    rank['four_metric_rank']=rank[['rank_'+n for n in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]].mean(axis=1)
    rows=rank.groupby(['year','strategy'],as_index=False).agg(
        mean_rank=('four_metric_rank','mean'),
        mean_return=('cumulative_return','mean'),
        mean_sharpe=('sharpe_proxy','mean'),
        mean_mdd=('mdd_proxy','mean'),
        mean_turnover=('turnover_proxy','mean'),
        mean_fees=('fees_usd','mean'),
        mean_trades=('trade_count','mean'),
        max_realized_rebalance_turnover=('max_realized_rebalance_turnover','max'))
    dev=rows[rows.year.isin((2023,2024))].groupby('strategy',as_index=False).agg(
        mean_rank=('mean_rank','mean'),worst_year_rank=('mean_rank','max'),
        mean_return=('mean_return','mean'),mean_sharpe=('mean_sharpe','mean'),
        mean_mdd=('mean_mdd','mean'),mean_turnover=('mean_turnover','mean'),
        mean_fees=('mean_fees','mean'),mean_trades=('mean_trades','mean'))
    dev['conservative_rank']=(dev.mean_rank+dev.worst_year_rank)/2
    return rank,rows,dev.sort_values('conservative_rank')


def self_test():
    # In self-test use a synthetic equivalent of Phase26, without needing a user's
    # real historical parquet file or potentially invoking network APIs.
    p26=phase26_module()
    assert len(POLICIES)==14
    import pandas as pd
    rng=np.random.default_rng(SEED)
    ix=pd.bdate_range('2021-01-01',periods=850)
    prices=100*np.exp(np.cumsum(rng.normal(.0002,.01,(len(ix),30)),axis=0))
    open_price=np.broadcast_to(prices[:,None,:],(len(ix),7,30)).copy()
    # Give every stock a nonconstant, unknown intraday change so 1-day IC
    # has a meaningful cross-sectional outcome instead of a zero-return tie.
    open_price[:,0,:]*=np.exp(rng.normal(0,.004,(len(ix),30)))
    ends=np.broadcast_to(prices[:,None,:],(len(ix),7,30)).copy()
    model=train_alpha(p26,ix,open_price,ends,2023)
    assert model['last_training_day'].startswith('2022-')
    assert abs(sum(model['signal_weights'].values())-1)<1e-10
    assert all(v>=0 for v in model['signal_weights'].values())
    # A 2023 forecast must NOT change if any 2023+ market observations change.
    modified_open=open_price.copy()
    modified_ends=ends.copy()
    future_mask=ix>=pd.Timestamp('2023-01-01')
    modified_open[future_mask]*=1.6
    modified_ends[future_mask]*=1.5
    model_with_changed_future=train_alpha(p26,ix,modified_open,modified_ends,2023)
    assert model['signal_weights']==model_with_changed_future['signal_weights'], 'Look-ahead leakage in training!'
    for pol in POLICIES:
        w=wanted(p26,pol,prices,600,model['signal_weights'])
        p26.validate_weights(w)
        data=simulate(p26,pol,601,open_price,ends,prices,model['signal_weights'],size=15)
        assert np.isfinite(np.array(list(data.values()),float)).all(),pol.name
        assert data['trade_count']>=1
        if pol.cadence=='staged':assert data['trade_count']>=3
        if pol.cadence=='hold':assert data['trade_count']==1
    # Confirm trades cost money: repeated same target without price movement
    # cannot earn money after the initial allocation.
    const=np.full((200,30),100.)
    op=np.repeat(const[:,None,:],7,axis=1)
    z=simulate(p26,POLICIES[1],130,op,op,const,model['signal_weights'])
    assert z['cumulative_return']<0 and z['fees_usd']>0
    print('PHASE27_SELF_TEST_PASS')
    print('CAUSAL_TRAINING_CUTOFF_AND_ANNUAL_WEIGHT_LEARNING_PASS')
    print('STAGED_ENTRY_DAILY_WEEKLY_HOLD_FEE_REPLAY_PASS')
    print('NO_NETWORK_NO_COMPETITION_SUBMISSION')


def run():
    p26=phase26_module()
    root=p26.root_dir()
    path=p26.source_path(root)
    if path is None:raise FileNotFoundError('Original historic Parquet missing')
    out=root/OUT
    out.mkdir(parents=True,exist_ok=True)
    report,bases,dates,_=p26.audit_data(path,out)
    if not report['adjusted_replay_possible']:
        raise RuntimeError('Price-basis audit blocked experiment')
    print('PRICE_BASIS_GATE:',report['price_replay_gate'])
    print('ADJUSTMENT_FACTOR_RANGE:',report.get('min_adjustment_factor'),report.get('max_adjustment_factor'))
    op,ends=bases['adj_open'],bases['adj_close']
    models={}
    for year in (2023,2024,2025):
        models[year]=train_alpha(p26,dates,op,ends,year)
        print('TRAINED_WEIGHTS_PRE_',year,models[year]['signal_weights'],flush=True)
    result=evaluate(p26,dates,op,ends,models)
    rank,annual,dev=score(p26,result)
    result.to_csv(out/'phase27_all_windows.csv',index=False)
    annual.to_csv(out/'phase27_yearly_summary.csv',index=False)
    dev.to_csv(out/'phase27_development_ranks.csv',index=False)
    (out/'phase27_walkforward_weights.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    print('=== DEVELOPMENT 2023-2024: LOWER RANK BETTER ===')
    print(dev[['strategy','mean_rank','conservative_rank','mean_return','mean_sharpe','mean_mdd','mean_turnover','mean_fees']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('=== EXPOSED 2025: DIAGNOSTIC ONLY ===')
    y=annual[annual.year==2025].sort_values('mean_rank')
    print(y[['strategy','mean_rank','mean_return','mean_sharpe','mean_mdd','mean_turnover','mean_fees']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('MAX_REALIZED_REBALANCE_TURNOVER:',float(annual.max_realized_rebalance_turnover.max()))
    meta={'status':'RESEARCH_ONLY_NO_DEPLOYMENT',
          'parquet_sha256':report['parquet_sha256'],
          'model_training':'strict expanding pre-year; 1d+5d IC shrinkage',
          'windows':'independent 15-session 2023-24 development; 2025 exposed',
          'signals':'price-only, no SEC/news/LLM archives',
          'fees':FEE,'metrics':['return','sharpe','drawdown','turnover'],
          'limitations':['historical bar times not exact Official fill times',
                         '2025 already exposed; no tuning based on it',
                         'price adjustment source/provenance still unverified',
                         'IC-based adaptive coefficient estimation only exploratory',
                         'noncontinuous 15d windows',
                         'no investment claim until stronger evaluation']}
    (out/'phase27_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('PHASE27_RESULT_DIR:',out)
    print('PHASE27_COMPLETE_RESEARCH_ONLY_NO_SUBMISSION')


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--run',action='store_true')
    a=parser.parse_args()
    if a.self_test:self_test()
    else:run()

if __name__=='__main__':main()
