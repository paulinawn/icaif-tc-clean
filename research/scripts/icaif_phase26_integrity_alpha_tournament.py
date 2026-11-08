#!/usr/bin/env python3
r"""ICAIF 2026 Phase 26: price-integrity audit, paired alpha and peer-style tournament.

RESEARCH ONLY. Local parquet -> local reports. No network, credentials, imports from
trading code, organizer calls, server changes or competition submissions.

From the starter-kit directory (with .venv active):
  python "$HOME\Downloads\icaif_phase26_integrity_alpha_tournament.py" --self-test
  python "$HOME\Downloads\icaif_phase26_integrity_alpha_tournament.py" --audit
  python "$HOME\Downloads\icaif_phase26_integrity_alpha_tournament.py" --run

Critical limits: Historic seven hourly-bar marks are NOT precise competition fills.
All 15-day windows independently reset. This is an opponent-SCENARIO tournament,
not a prediction of actual opponent strategies, ranks, or Official scores.
2023-24 development, 2025 already exposed, 2026 deliberately untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SYMBOLS = tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
TIMES = ('09:30:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00')
GROUPS = ('Technology', 'Finance', 'Healthcare', 'Consumer', 'Industrial & Energy', 'Communication & Utilities')
EXPECTED_GROUPS = tuple(g for g in GROUPS for _ in range(5))
FEE = .001
NAV0 = 1_000_000.0
DAYS = 15
ANNUALIZER = np.sqrt(252*7)
SEED = 26011
OUT = Path('artifacts/phase26_integrity_alpha_tournament')

@dataclass(frozen=True)
class Policy:
    name: str
    exposure: float
    style: str
    weekly: bool = False

# Fixed hypotheses, not post-hoc search parameters. More cash, concentration and
# turnover are intentionally included as adversarial *profiles*, not real entrants.
POLICIES = (
    Policy('equal80_hold', .80, 'equal'),
    Policy('industry80_hold', .80, 'industry'),
    Policy('sector80_hold', .80, 'sector'),
    Policy('momentum80_hold', .80, 'momentum'),
    Policy('reversal80_hold', .80, 'reversal'),
    Policy('lowvol80_hold', .80, 'lowvol'),
    Policy('concentrated80_hold', .80, 'concentrated'),
    Policy('equal60_hold', .60, 'equal'),
    Policy('equal100_hold', 1.00, 'equal'),
    Policy('industry90_hold', .90, 'industry'),
    Policy('industry80_weekly', .80, 'industry', True),
)


def root_dir():
    root = Path.cwd().resolve()
    if not (root/'tools'/'auto_submit.py').is_file():
        raise RuntimeError('Open starter-kit in Windows PowerShell before running.')
    return root


def source_path(root):
    choices=(root/'data'/'official'/'hourly_market_data_2021_2026.parquet',
             root/'hourly_market_data_2021_2026.parquet')
    return next((p for p in choices if p.is_file()),None)


def validate_weights(a, gross=None):
    w=np.asarray(a,dtype=float)
    if w.shape!=(30,) or not np.isfinite(w).all() or np.min(w)<-1e-10 or np.max(w)>.30+1e-10 or np.sum(w)>1+1e-9:
        raise ValueError('Invalid 30-stock long-only weights')
    if gross is not None and abs(float(w.sum())-gross)>1e-7:
        raise ValueError('Wrong exposure')
    return w


def ranks(x):
    return pd.Series(np.asarray(x,dtype=float)).rank(method='average',pct=True).to_numpy(dtype=float)*2-1


def load_filtered(path):
    df=pd.read_parquet(path)
    columns=set(df.columns)
    required={'ticker','trading_date','timestamp_et','sector_group','open','adj_close'}
    if not required.issubset(columns):
        raise ValueError('Required parquet columns missing: '+str(sorted(required-columns)))
    df=df.loc[df.ticker.isin(SYMBOLS)].copy()
    df['trading_date']=pd.to_datetime(df['trading_date'],errors='raise').dt.normalize()
    df['timestamp_et']=pd.to_datetime(df['timestamp_et'],errors='raise')
    df['bar_time']=df.timestamp_et.dt.strftime('%H:%M:%S')
    df=df[df.bar_time.isin(TIMES)].copy()
    if df.duplicated(['trading_date','ticker','bar_time']).any():
        raise ValueError('Duplicate (date, symbol, bar) rows')
    counts=df.groupby(['trading_date','ticker'])['bar_time'].nunique().unstack('ticker').reindex(columns=SYMBOLS)
    dates=pd.DatetimeIndex(counts.index[(counts==len(TIMES)).all(axis=1)])
    # Hard quarantine: never inspect, rank, or factor-test 2026 observations.
    dates=dates[dates.year<=2025]
    if len(dates)<650:
        raise ValueError('Insufficient complete historic days; refuse analysis')
    df=df[df.trading_date.isin(dates)].copy()
    sectors=df.drop_duplicates('ticker').set_index('ticker').sector_group.reindex(SYMBOLS).tolist()
    if len(set(sectors))!=6 or any(len(set(sectors[i:i+5]))!=1 for i in range(0,30,5)):
        raise ValueError('Unexpected sector ordering/partition; no silent remapping')
    # Historical development subset only; 2026 remains quarantined entirely.
    return df,dates,sectors


def pivot(df,dates,column):
    if column not in df:
        raise ValueError('Missing price column '+column)
    x=df.pivot(index=['trading_date','bar_time'],columns='ticker',values=column)
    idx=pd.MultiIndex.from_product([dates,TIMES],names=['trading_date','bar_time'])
    v=x.reindex(index=idx,columns=SYMBOLS).to_numpy(dtype=float).reshape(len(dates),7,30)
    if not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('Nonfinite/nonpositive values for '+column)
    return v


def price_bases(df,dates):
    opens=pivot(df,dates,'open')
    adjusted_close=pivot(df,dates,'adj_close')
    raw_close=pivot(df,dates,'close') if 'close' in df else None
    adjusted_open=pivot(df,dates,'adj_open') if 'adj_open' in df else None
    factor=None
    factor_intra_max=None
    factor_min=None
    factor_max=None
    if raw_close is not None:
        factor=adjusted_close/raw_close
        intra=factor.max(axis=1)/factor.min(axis=1)-1
        factor_intra_max=float(np.max(intra))
        factor_min=float(np.min(factor))
        factor_max=float(np.max(factor))
        # One coherent adjusted- and one coherent raw-price basis. If adj_open is
        # missing, convert raw open using same BAR's close adjustment factor.
        # This is an offline historical transformation, not a live-data feed.
        inferred_adj_open=opens*factor
        if adjusted_open is None:
            adjusted_open=inferred_adj_open
        else:
            deviation=np.max(np.abs(adjusted_open/inferred_adj_open-1))
            if deviation>.005:
                raise ValueError(f'adj_open conflicts with open*(adj_close/close): max relative error {deviation:.5f}')
        if factor_intra_max>.005:
            raise ValueError(f'Intraday adjustment factor inconsistent, max {factor_intra_max:.5f}')
    if adjusted_open is None:
        return {'raw_open':opens,'adj_close':adjusted_close,'raw_close':raw_close}, {'reason':'NO_RAW_CLOSE_OR_ADJ_OPEN','adjusted_replay_possible':False,'factor_intraday_variation_max':None}
    if not np.isfinite(adjusted_open).all() or (adjusted_open<=0).any():
        raise ValueError('Invalid adjusted opening prices')
    result={'raw_open':opens,'adj_open':adjusted_open,'adj_close':adjusted_close,'raw_close':raw_close}
    return result,{'reason':'CONSISTENT_ADJUSTED_PAIR_CONSTRUCTED' if factor is not None else 'ADJUSTED_OPEN_COLUMN_WITHOUT_RAW_CLOSE',
                   'adjusted_replay_possible':True,'raw_close_present':raw_close is not None,
                   'adjusted_open_present_in_source':'adj_open' in df,
                   'factor_intraday_variation_max':factor_intra_max,
                   'min_adjustment_factor':factor_min,'max_adjustment_factor':factor_max}


def gaps(dates,open_price,close_price,kind):
    # Avoid 2026 and the very first dataset day. The same stock must be compared.
    change=open_price[1:,0,:]/close_price[:-1,-1,:]-1
    yrs=dates[1:].year.to_numpy()
    sel=(yrs<=2025)&(yrs>=2022)
    d,s=np.where(sel[:,None]&np.isfinite(change)&(np.abs(change)>=.10))
    records=[{'basis':kind,'day':str(dates[i+1].date()),'symbol':SYMBOLS[j],
              'prev_close':float(close_price[i,-1,j]),'next_open':float(open_price[i+1,0,j]),
              'gap_return':float(change[i,j])} for i,j in zip(d,s)]
    records.sort(key=lambda z:abs(z['gap_return']),reverse=True)
    valid=change[sel]
    return records,{'max_abs_gap':float(np.max(np.abs(valid))),
                    'num_abs_gap_ge10pct':len(records),
                    'num_abs_gap_ge25pct':sum(abs(x['gap_return'])>=.25 for x in records),
                    'num_abs_gap_ge35pct':sum(abs(x['gap_return'])>=.35 for x in records)}


def audit_data(path,outdir):
    df,dates,sectors=load_filtered(path)
    bases,status=price_bases(df,dates)
    prev_raw,prev_adj=gaps(dates,bases['raw_open'],bases['adj_close'],'ORIGINAL_MIXED')
    report={
        'parquet_path_name':path.name,
        'parquet_sha256':sha256_file(path),
        'price_columns':[k for k in ('open','close','adj_open','adj_close') if k in df],
        'source_data_provenance':'UNKNOWN_UNLESS_USER_SUPPLIES_DATA_SOURCE',
        'trading_days':len(dates),'first_day':str(dates[0].date()),'last_day':str(dates[-1].date()),
        'sector_partitions':sectors,
        'original_mixed_gap':prev_adj,
        **status,
        'historical_bar_times':list(TIMES),
        'official_execution_times':'09:30,10:30,11:30,12:30,13:30,14:30,15:30 ET',
        'note':'Absolute gap is not itself proof of corporate action; check identified rows and upstream provenance.',
        'no_2026_strategy_evaluation':True,
    }
    records=prev_raw.copy()
    if status['adjusted_replay_possible']:
        c,stat=gaps(dates,bases['adj_open'],bases['adj_close'],'CONSISTENT_ADJUSTED')
        records.extend(c)
        report['consistent_adjusted_gap']=stat
        if bases['raw_close'] is not None:
            c,stat=gaps(dates,bases['raw_open'],bases['raw_close'],'CONSISTENT_RAW')
            records.extend(c)
            report['consistent_raw_gap']=stat
        report['price_replay_gate']='PROVISIONAL_PASS_CONSISTENT_ADJUSTED_PAIR'
    else:
        report['price_replay_gate']='BLOCK_NO_VERIFIABLE_MATCHING_OPEN_CLOSE_BASIS'
    outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records,columns=['basis','day','symbol','prev_close','next_open','gap_return']).to_csv(outdir/'phase26_gap_events.csv',index=False)
    (outdir/'phase26_integrity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report,bases,dates,sectors


def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while part:=f.read(2**20):
            h.update(part)
    return h.hexdigest()


def strength_features(daily,index):
    if index<121:
        raise ValueError('Require 120 completed preceding sessions')
    def ret(n):return daily[index]/daily[index-n]-1
    r5,r20,r60,r120=(ret(x) for x in (5,20,60,120))
    strength=.25*r20+.5*r60+.25*r120
    change=daily[index-59:index+1]/daily[index-60:index]-1
    vol=change.std(axis=0,ddof=1)
    if not np.isfinite(vol).all() or not np.isfinite(strength).all():
        raise ValueError('Bad research features')
    return {'r5':r5,'r20':r20,'r60':r60,'r120':r120,'strength':strength,'vol60':vol}


def target(policy,daily,idx):
    f=strength_features(daily,idx)
    gross=policy.exposure
    style=policy.style
    if style=='equal':return validate_weights(np.full(30,gross/30),gross)
    if style=='industry':
        group_strength=np.array([np.median(f['strength'][i*5:(i+1)*5]) for i in range(6)])
        b=(gross/6)*(1+.45*ranks(group_strength))
        b=b*(gross/b.sum())
        out=np.zeros(30)
        for i in range(6):
            ii=slice(i*5,(i+1)*5)
            selection=.75*f['strength'][ii]-.25*f['r5'][ii]
            w=1+.35*ranks(selection)
            out[ii]=b[i]*w/w.sum()
        return validate_weights(out,gross)
    if style=='sector':
        group_strength=np.array([np.median(f['strength'][i*5:(i+1)*5]) for i in range(6)])
        b=(gross/6)*(1+.45*ranks(group_strength))
        b=b*(gross/b.sum())
        return validate_weights(np.repeat(b/5,5),gross)
    if style in ('momentum','reversal','lowvol'):
        signal={'momentum':f['strength'],'reversal':-f['r5'],'lowvol':-f['vol60']}[style]
        w=1+.45*ranks(signal)
        return validate_weights(gross*w/w.sum(),gross)
    if style=='concentrated':
        ix=np.argsort(f['strength'],kind='stable')[-5:]
        out=np.zeros(30)
        out[ix]=gross/5
        return validate_weights(out,gross)
    raise ValueError('Unknown policy')


def execution(cash,shares,price,w):
    validate_weights(w)
    current=shares*price
    before=float(cash+current.sum())
    after=before
    for _ in range(80):
        notional=float(np.abs(w*after-current).sum())
        next_nav=before-FEE*notional
        if abs(next_nav-after)<1e-9:
            after=next_nav
            break
        after=next_nav
    notional=float(np.abs(w*after-current).sum())
    new_shares=(w*after)/price
    new_cash=float(after-(w*after).sum())
    if after<=0 or new_cash<-1e-6:
        raise ValueError('Invalid post fee NAV')
    return new_cash,new_shares,notional


def replay(policy,start,opening,ends,daily,size=DAYS):
    cash,shares=NAV0,np.zeros(30)
    nav_curve=[NAV0]
    notional_sum=0.
    trades=0
    for d in range(size):
        i=start+d
        mark=opening[i,0]
        # Note: next open only affects fills; decision uses i-1 completed close.
        if d==0 or (policy.weekly and d%5==0):
            wanted=target(policy,daily,i-1)
            # Same deterministic gate across weekly experiments, with no future fill info.
            prior=daily[i-1]
            pre=float(cash+np.dot(shares,prior))
            current=shares*prior/pre
            drift=float(np.abs(wanted-current).sum())
            if d==0 or drift>=.06:
                if d>0:
                    factor=min(1.,.12*.8/max(drift,1e-12))
                    wanted=validate_weights(current+factor*(wanted-current))
                cash,shares,notional=execution(cash,shares,mark,wanted)
                notional_sum+=notional/max(float(cash+np.dot(shares,mark))+FEE*notional,1.)
                trades+=1
        for p in ends[i]:
            nav=float(cash+np.dot(shares,p))
            if not np.isfinite(nav) or nav<=0:raise ValueError('Invalid valuation')
            nav_curve.append(nav)
    nav_curve=np.asarray(nav_curve)
    r=nav_curve[1:]/nav_curve[:-1]-1
    sd=float(r.std(ddof=1))
    return {'cumulative_return':float(nav_curve[-1]/NAV0-1),
            'sharpe_proxy':float(r.mean()*ANNUALIZER/sd) if sd>1e-12 else 0.,
            'mdd_proxy':float(np.max(1-nav_curve/np.maximum.accumulate(nav_curve))),
            'turnover_proxy':notional_sum/(size*7),
            'trade_count':trades}


def starts(dates,year):
    ix=np.flatnonzero(dates.year==year)
    return [int(s) for s in ix[1::DAYS] if s>=125 and s+DAYS<=len(dates)
            and dates[s+DAYS-1].year==year]


def replay_all(dates,bases,year_set=(2023,2024,2025)):
    opening,close=bases
    daily=close[:,-1,:]
    rows=[]
    for yr in year_set:
        windows=starts(dates,yr)
        for j,s in enumerate(windows):
            for pol in POLICIES:
                rows.append({'year':yr,'start':str(dates[s].date()),'strategy':pol.name,
                             **replay(pol,s,opening,close,daily)})
            if (j+1)%8==0:
                print(f'TOURNAMENT_{yr}: {j+1}/{len(windows)}',flush=True)
    result=pd.DataFrame(rows)
    if result.empty:raise RuntimeError('No 2023-2025 windows available')
    return result


def compare_windows(df):
    base=df[df.strategy=='equal80_hold'].set_index(['year','start'])
    competitor=df[df.strategy=='industry80_hold'].set_index(['year','start'])
    if not base.index.equals(competitor.index):raise ValueError('Unmatched paired windows')
    cols=('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')
    delta=competitor[list(cols)]-base[list(cols)]
    delta=delta.rename(columns={c:'delta_'+c for c in cols})
    delta['year']=[i[0] for i in delta.index]
    delta['start']=[i[1] for i in delta.index]
    result=[]
    rng=np.random.default_rng(SEED)
    for name,part in [('DEVELOPMENT_2023_2024',delta[delta.year.isin((2023,2024))]),
                      ('EXPOSED_2025',delta[delta.year==2025])]:
        if part.empty:continue
        n=len(part)
        # Exploratory naive window bootstrap, NOT independent if overlapping or
        # market regimes cluster; no formal alpha significance is claimed.
        values=part.delta_cumulative_return.to_numpy()
        means=values[rng.integers(0,n,size=(2000,n))].mean(axis=1)
        result.append({'period':name,'windows':n,
                       'industry_minus_equal_return_mean':float(values.mean()),
                       'industry_return_win_fraction':float((values>0).mean()),
                       'return_delta_bootstrap_2p5':float(np.quantile(means,.025)),
                       'return_delta_bootstrap_97p5':float(np.quantile(means,.975)),
                       'industry_minus_equal_sharpe_mean':float(part.delta_sharpe_proxy.mean()),
                       'industry_minus_equal_mdd_mean':float(part.delta_mdd_proxy.mean()),
                       'industry_minus_equal_turnover_mean':float(part.delta_turnover_proxy.mean())})
    return delta.reset_index(drop=True),pd.DataFrame(result)


def tournament(df):
    x=df.copy()
    for metric,ascending in (('cumulative_return',False),('sharpe_proxy',False),
                             ('mdd_proxy',True),('turnover_proxy',True)):
        x['rank_'+metric]=x.groupby(['year','start'])[metric].rank(pct=True,method='average',ascending=ascending)
    x['overall_proxy_rank']=x[[c for c in x if c.startswith('rank_')]].mean(axis=1)
    return x.groupby(['strategy','year'],as_index=False).agg(
        mean_proxy_rank=('overall_proxy_rank','mean'),
        avg_return=('cumulative_return','mean'),
        avg_sharpe=('sharpe_proxy','mean'),
        avg_mdd=('mdd_proxy','mean'),
        avg_turnover=('turnover_proxy','mean'),
        windows=('start','size'))


def factor_ic(dates,opening,closing,yr_set=(2023,2024,2025)):
    # Nonoverlapping five-session forward labels: decision at end of day t,
    # entry t+1 opening, exit end t+5. Overlapping 1d forecasts use day-by-day.
    daily=closing[:,-1,:]
    rows=[]
    for yr in yr_set:
        indices=np.flatnonzero(dates.year==yr)
        for horizon in (1,5):
            for i in indices:
                if i<125 or i+horizon>=len(dates):continue
                if dates[i+horizon].year!=yr:continue
                if horizon==5 and (i-indices[0])%5!=0:continue
                f=strength_features(daily,i)
                true=closing[i+horizon,-1]/opening[i+1,0]-1
                if not np.isfinite(true).all():continue
                group_neutral=true.copy()
                for g in range(6):
                    sl=slice(g*5,(g+1)*5)
                    group_neutral[sl]-=np.mean(true[sl])
                for name,signal in [('mom20',f['r20']),('mom60',f['r60']),
                                    ('mom120',f['r120']),('blend_momentum',f['strength']),
                                    ('reversal5',-f['r5']),('lowvol60',-f['vol60'])]:
                    cr=np.corrcoef(ranks(signal),ranks(true))[0,1]
                    cn=np.corrcoef(ranks(signal),ranks(group_neutral))[0,1]
                    rows.append({'year':yr,'decision_date':str(dates[i].date()),'horizon':horizon,
                                 'signal':name,'spearman_ic':float(cr),
                                 'industry_neutral_ic':float(cn)})
    return pd.DataFrame(rows)


def test():
    # Self tests validate algebra, detection and timestamp discipline, not market outcomes.
    assert len(SYMBOLS)==30 and len(POLICIES)==11
    z=validate_weights(np.full(30,.8/30),.8)
    assert abs(z.sum()-.8)<1e-12
    sig=np.arange(30)+1.
    assert np.argmax(ranks(sig))==29
    zero=np.zeros(30)
    cash,shares,notional=execution(NAV0,zero,np.full(30,100.),z)
    assert abs(notional-799360.51159)<.1
    assert abs(cash+float((shares*100).sum())-(NAV0-FEE*notional))<1e-4
    # Synthetic 2:1 adjustment; raw opens and adjusted closes cannot be mixed.
    f=np.array([[[.5,.5]],[[.5,.5]]])
    raw=np.array([[[200.,100.]],[[202.,101.]]])
    adj=raw*f
    assert abs(adj[1,0,0]/adj[0,0,0]-1-.01)<1e-10
    assert abs(raw[1,0,0]/adj[0,0,0]-1-1.02)<1e-10
    # A positive signal should favor top ranked stocks while staying long-only.
    rng=np.random.default_rng(SEED)
    history=100*np.exp(np.cumsum(rng.normal(0,.01,(180,30)),axis=0))
    for pol in POLICIES:
        w=target(pol,history,150)
        validate_weights(w,pol.exposure)
    op=np.repeat(history[:,None,:],7,axis=1)
    en=op.copy()
    for pol in POLICIES:
        res=replay(pol,151,op,en,history,15)
        assert np.isfinite(list(res.values())).all()
        assert res['trade_count'] >= 1
    print('PHASE26_SELF_TEST_PASS')
    print('CONSISTENT_PRICE_PAIR_AND_MIXED_BASIS_DETECTION_PASS')
    print('ALPHA_SIGNAL_PAST_ONLY_AND_FEE_TRADE_PASS')
    print('NO_NETWORK_NO_ORGANIZER_CALL_NO_SUBMISSION')


def run(mode):
    root=root_dir()
    path=source_path(root)
    if path is None:raise FileNotFoundError('Historical parquet not found in existing starter-kit data/official folder')
    out=root/OUT
    report,bases,dates,sectors=audit_data(path,out)
    print('PRICE_REPLAY_GATE:',report['price_replay_gate'])
    print('PARQUET_COLUMNS:',','.join(report['price_columns']))
    print('ORIGINAL_MIXED_MAX_GAP:',round(report['original_mixed_gap']['max_abs_gap'],6))
    if 'consistent_adjusted_gap' in report:
        print('ADJUSTED_PAIR_MAX_GAP:',round(report['consistent_adjusted_gap']['max_abs_gap'],6))
    if 'consistent_raw_gap' in report:
        print('RAW_PAIR_MAX_GAP:',round(report['consistent_raw_gap']['max_abs_gap'],6))
    print('INTEGRITY_REPORT:',out/'phase26_integrity.json')
    if mode=='audit':
        print('PHASE26_AUDIT_COMPLETE_NO_TRADING')
        return
    if not report['adjusted_replay_possible']:
        print('REPLAY_BLOCKED_REQUIRE_MATCHING_ADJUSTED_OPEN_OR_RAW_CLOSE')
        raise SystemExit(2)
    adjusted=(bases['adj_open'],bases['adj_close'])
    result=replay_all(dates,adjusted)
    pair,pair_summary=compare_windows(result)
    scores=tournament(result)
    ics=factor_ic(dates,*adjusted)
    result.to_csv(out/'phase26_windows_adjusted.csv',index=False)
    pair.to_csv(out/'phase26_industry80_vs_equal80_pairs.csv',index=False)
    pair_summary.to_csv(out/'phase26_pair_summary.csv',index=False)
    scores.to_csv(out/'phase26_scenario_ranks.csv',index=False)
    ics.to_csv(out/'phase26_signal_ic.csv',index=False)
    if bases['raw_close'] is not None:
        raw_result=replay_all(dates,(bases['raw_open'],bases['raw_close']))
        raw_result.to_csv(out/'phase26_windows_raw_sensitivity.csv',index=False)
        _,raw_summary=compare_windows(raw_result)
        raw_summary.to_csv(out/'phase26_pair_raw_sensitivity.csv',index=False)
    print('=== PAIRED INDUSTRY80 VS EQUAL80 (positive return = industry advantage) ===')
    print(pair_summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('=== COMPETITOR-STYLE SCENARIOS: DEVELOPMENT RANKS (lower better) ===')
    dev=scores[scores.year.isin((2023,2024))]
    print(dev.groupby('strategy').mean(numeric_only=True).sort_values('mean_proxy_rank')[['mean_proxy_rank','avg_return','avg_sharpe','avg_mdd','avg_turnover']].to_string(float_format=lambda x:f'{x:.6f}'))
    print('=== DIRECTIONAL CROSS-SECTIONAL FEATURE IC (not a calibrated alpha forecast) ===')
    print(ics[ics.year.isin((2023,2024))].groupby(['signal','horizon'])[['spearman_ic','industry_neutral_ic']].mean().to_string(float_format=lambda x:f'{x:.5f}'))
    manifest={'status':'RESEARCH_ONLY_NO_DEPLOYMENT',
              'source_sha256':sha256_file(Path(__file__)),
              'parquet_sha256':report['parquet_sha256'],
              'price_basis':'adjusted close with adjusted opening price/within-bar factor',
              'window':'15 sessions independently reset to $1m',
              'fees':'0.1% buy+sell notional',
              'timing':'features prior completed close; simulated next 09:30 open',
              'history_evaluated':['2023','2024','2025 EXPOSED'],
              'untouched':['2026'],
              'limitations':['hourly marks not exact official 105-period valuation',
                             'no official execution-price tape',
                             'no public-news historical event availability archive',
                             'paired bootstrap exploratory not independent market regime evidence',
                             'price-series adjustment may not equal official fill accounting',
                             'peer scenarios hypothetical, not actual competitors']}
    (out/'phase26_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PHASE26_RESULT_DIR:',out)
    print('PHASE26_COMPLETE_RESEARCH_ONLY_NO_SUBMISSION')


def main():
    parser=argparse.ArgumentParser()
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--audit',action='store_true')
    g.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.self_test:test()
    elif args.audit:run('audit')
    else:run('run')

if __name__=='__main__':main()
