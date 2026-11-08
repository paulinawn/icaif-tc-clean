#!/usr/bin/env python3
"""ICAIF Phase 30 v2: offline 2026 walk-forward forecast/portfolio experiment.

Requires Phase29 output, and no network or organizer API. 15-minute IEX RAW
bars are PROXIES, not official fills. A bar stamped 10:15 ends at 10:30
and MUST NOT be used in a 10:25 decision. Only earlier completed bars
enter forecasts. Missing mandatory bars lead to skipped incomplete days.

Windows: Jul-Aug 2026 research development, Sep-Oct 9 diagnostic. Monthly
refits use only labels fully matured BEFORE test month. Fees 0.1% each side.
Cash and shares persist continuously WITHIN each contiguous evaluation period.
No claims about actual leaderboard positions or production readiness.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

SYMBOLS=('AAPL','MSFT','NVDA','INTC','CRM','JPM','BAC','GS','V','PYPL',
 'LLY','JNJ','UNH','PFE','TMO','AMZN','TSLA','WMT','NKE','KO','CAT','GE',
 'BA','XOM','CVX','GOOGL','META','DIS','T','NEE')
INPUT=Path('artifacts/phase29_alpaca_refresh/alpaca_iex_15m_through_2026-10-09.csv')
OUT=Path('artifacts/phase30_2026_walkforward_v2')
SLOTS=[(pd.Timestamp('2026-10-07 09:30')+pd.Timedelta(minutes=15*j)).strftime('%H:%M') for j in range(26)]
# Seven allowed executions are at these 15-minute bar starts.
FILL_IDX=(0,4,8,12,16,20,24)
# For Round 2..7, last known complete bar ends 15 minutes before the deadline.
# e.g. Round 2 10:25 may use 10:00 bar (ends 10:15), but NOT 10:15.
KNOWN_IDX=(None,2,6,10,14,18,22)
END_VAL_IDX=(4,8,12,16,20,24,25)
FEE=.001
NAV0=1_000_000.
LAMBDA=80.
TRAIN_MIN_DAY=121
ALPHA_SIGNAL=('mom20','mom60','mom120','vol20','industry20')
POLICIES=('equal60_hold','equal80_hold','momentum80_hold','industry80_hold',
          'ridge80_hold','ridge80_weekly','ridge80_daily',
          'ridge80_costaware_daily','ridge80_costaware_intraday')


@dataclass
class Market:
    dates:pd.DatetimeIndex
    op:np.ndarray  # [trading days,26 start times,30 stocks]
    cl:np.ndarray
    daily:np.ndarray # market price at end 15:45 bar (16:00)
    missing:pd.DataFrame


def read_market(path, min_last_day='2026-10-07'):
    if not path.is_file():raise FileNotFoundError('Expected Alpaca 15m CSV not found: '+str(path))
    x=pd.read_csv(path)
    require={'symbol','timestamp','open','close'}
    if not require.issubset(x.columns):raise ValueError('Missing Alpaca columns')
    x=x[x.symbol.isin(SYMBOLS)].copy()
    ts=pd.to_datetime(x.timestamp,utc=True,errors='coerce')
    if ts.isna().any():raise ValueError('Invalid timestamps')
    local=ts.dt.tz_convert('America/New_York')
    x['date']=pd.to_datetime(local.dt.strftime('%Y-%m-%d'))
    x['clock']=local.dt.strftime('%H:%M')
    x=x[x.clock.isin(SLOTS)].copy()
    if x.duplicated(['date','symbol','clock']).any():raise ValueError('Duplicate symbol-date-clock')
    for col in ('open','close'):
        x[col]=pd.to_numeric(x[col],errors='raise')
        if not np.isfinite(x[col]).all() or (x[col]<=0).any():raise ValueError('Invalid '+col)
    dates=pd.DatetimeIndex(sorted(x.date.unique()))
    ix=pd.MultiIndex.from_product([dates,SLOTS],names=['date','clock'])
    def array(col):
        m=x.pivot(index=['date','clock'],columns='symbol',values=col).reindex(index=ix,columns=SYMBOLS)
        return m.to_numpy(float).reshape(len(dates),26,30)
    op=array('open');cl=array('close')
    good=np.isfinite(op).all(axis=(1,2))&np.isfinite(cl).all(axis=(1,2))
    bad=[{'date':str(d.date()),'missing_15min_fields':int(np.isnan(op[i]).sum()+np.isnan(cl[i]).sum())} for i,d in enumerate(dates) if not good[i]]
    clean_dates=dates[good]
    if len(clean_dates)<210:raise ValueError('Insufficient complete Alpaca 15m days')
    if clean_dates[-1]<pd.Timestamp(min_last_day):
        raise ValueError('No complete trading day through '+min_last_day+'; verify input file and missing-bar report')
    if clean_dates[-1]>pd.Timestamp('2026-10-09'):
        raise ValueError('Unexpected post-cutoff data; stop research')
    if clean_dates[0]>pd.Timestamp('2025-09-15'):
        raise ValueError('Too little training history for 120-day signals')
    op=op[good];cl=cl[good]
    if np.nanmin(op)<=0 or np.nanmin(cl)<=0:raise ValueError('Bad price')
    return Market(clean_dates,op,cl,cl[:,-1,:],pd.DataFrame(bad))


def zscore(y):
    arr=np.asarray(y,float)
    if not np.isfinite(arr).all():raise ValueError('Non-finite feature')
    std=float(arr.std())
    return np.clip((arr-arr.mean())/max(std,1e-8),-3.,3.)


def features(daily,t):
    if t<TRAIN_MIN_DAY:raise ValueError('Not enough completed daily history')
    log=np.log(daily)
    r20=log[t]-log[t-20]
    r60=log[t]-log[t-60]
    r120=log[t]-log[t-120]
    volatility=log[t-19:t+1]-log[t-20:t]
    vol=np.std(volatility,axis=0,ddof=1)
    relative=r20-np.repeat(r20.reshape(6,5).mean(axis=1),5)
    return np.stack((zscore(r20),zscore(r60),zscore(r120),
                     zscore(-vol),zscore(relative)),axis=1)


def ridge(x,y,penalty=LAMBDA):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or len(x)!=len(y) or len(x)<50:raise ValueError('Not enough training examples')
    beta=np.linalg.solve(x.T@x+np.eye(x.shape[1])*penalty,x.T@y)
    if not np.isfinite(beta).all():raise ValueError('Unstable regression')
    return beta


def known_intraday(cl,op,i,round_number):
    if round_number not in range(1,7):raise ValueError('Only rounds 2-7 have an intra-day signal')
    known=KNOWN_IDX[round_number]
    # A 10:00 bar closes at 10:15; 10:25 submission decision is allowed to see it.
    # The next bar 10:15 closes at 10:30; that must remain unobserved.
    total=np.log(cl[i,known,:]/op[i,0,:])
    last=np.log(cl[i,known,:]/cl[i,known-1,:])
    return np.stack([zscore(total),zscore(last)],axis=1)


def train_month(market,cutoff):
    """All training labels (1d,5d,round) complete strictly before cutoff day."""
    if cutoff<TRAIN_MIN_DAY+45:raise ValueError('Insufficient matured training history')
    log=np.log(market.daily)
    betas={};counts={}
    for h in (1,5):
        xx=[];yy=[]
        for t in range(TRAIN_MIN_DAY,cutoff-h):
            f=features(market.daily,t)
            r=log[t+h]-log[t]
            xx.append(f);yy.append(r-r.mean())
        if not xx:raise ValueError('No mature targets')
        x=np.concatenate(xx);y=np.concatenate(yy)
        betas[f'daily_{h}']=ridge(x,y)
        counts[f'daily_{h}']=len(y)
    for r in range(1,7):
        xx=[];yy=[]
        fill=FILL_IDX[r]
        # next assessment after fill is next permitted execution, or close for r=7
        valuation=FILL_IDX[r+1] if r<6 else None
        for i in range(TRAIN_MIN_DAY,cutoff):
            f=known_intraday(market.cl,market.op,i,r)
            future=market.op[i,valuation,:] if valuation is not None else market.cl[i,25,:]
            y=np.log(future/market.op[i,fill,:]);y-=y.mean()
            xx.append(f);yy.append(y)
        x=np.concatenate(xx);y=np.concatenate(yy)
        betas[f'intra_{r}']=ridge(x,y)
        counts[f'intra_{r}']=len(y)
    return betas,counts


def target_from_score(score,exposure=.8):
    x=zscore(score)
    prop=np.clip(1.+.35*x,.01,None)
    w=exposure*prop/prop.sum()
    if not np.isfinite(w).all() or (w<0).any() or (w>.3).any() or w.sum()>1.00000001:
        raise AssertionError('Invalid portfolio weights')
    return w


def desired(policy,market,t,models):
    f=features(market.daily,t)
    if policy.startswith('equal'):
        return np.full(30,(.60 if '60' in policy else .80)/30)
    if policy.startswith('momentum'):
        return target_from_score(.25*f[:,0]+.50*f[:,1]+.25*f[:,2])
    if policy.startswith('industry'):
        m=.25*f[:,0]+.5*f[:,1]+.25*f[:,2]
        sector=np.array([np.mean(m[k*5:(k+1)*5]) for k in range(6)])
        score=.65*zscore(np.repeat(sector,5))+.35*zscore(m)
        return target_from_score(score)
    if policy.startswith('ridge'):
        score=.4*(f@models['daily_1'])+.6*(f@models['daily_5'])/5.
        return target_from_score(score)
    raise ValueError('Unknown policy')


def trade(cash,shares,price,w):
    if np.max(w)>.3 or np.min(w)<0 or w.sum()>1+1e-9:raise ValueError('Illegal weights')
    before=cash+float(np.dot(shares,price))
    old=shares*price
    after=before
    for _ in range(70):
        amount=float(np.abs(after*w-old).sum())
        updated=before-FEE*amount
        if abs(updated-after)<1e-8:break
        after=updated
    amount=float(np.abs(after*w-old).sum())
    new_shares=w*after/price
    new_cash=after*(1.-w.sum())
    if new_cash<-1e-6 or after<=0:raise ValueError('Invalid posttrade NAV')
    return new_cash,new_shares,amount


def walkforward_models(market,first,last):
    """Fit once per test month before any test-month prices; no tuning to September diagnostics."""
    indices=np.flatnonzero((market.dates>=pd.Timestamp(first))&(market.dates<=pd.Timestamp(last)))
    if len(indices)<15:raise ValueError('Insufficient evaluation dates')
    models={};meta=[]
    for month in market.dates[indices].strftime('%Y-%m').unique():
        start=int(np.flatnonzero(market.dates.strftime('%Y-%m')==month)[0])
        beta,count=train_month(market,start)
        models[month]=beta
        meta.append({'test_month':month,'last_training_date':str(market.dates[start-1].date()),
                     'last_matured_5day_training_feature_date':str(market.dates[start-6].date()),
                     'label_counts':count,
                     'coefficients':{k:[float(z) for z in v] for k,v in beta.items()}})
    return indices,models,meta


def replay(market,indices,models,policy):
    cash=NAV0;shares=np.zeros(30)
    nav=[NAV0];notional=0.;fees=0.;trades=0;max_rebal=0.
    rows=[]
    for di,i in enumerate(indices):
        t=i-1
        month=market.dates[i].strftime('%Y-%m')
        mod=models[month]
        base=desired(policy,market,t,mod)
        for r in range(7):
            price=market.op[i,FILL_IDX[r],:]
            intent=None
            if di==0 and r==0:intent=base
            elif r==0 and policy.endswith('_weekly') and di%5==0:intent=base
            elif r==0 and (policy.endswith('_daily') or policy=='ridge80_costaware_intraday'):
                intent=base
            elif r>=1 and policy=='ridge80_costaware_intraday':
                intra=known_intraday(market.cl,market.op,i,r)@mod[f'intra_{r}']
                intent=target_from_score(intra)
            if intent is not None:
                if di==0 and r==0:
                    cash,shares,q=trade(cash,shares,price,intent)
                    notional+=q;fees+=FEE*q;trades+=1
                else:
                    # Establish today's decision from lagged market marks. At
                    # R2+, only the completed bar KNOWN_IDX[r] may be seen.
                    # No current fill or valuation price enters the decision.
                    mark=market.daily[t] if r==0 else market.cl[i,KNOWN_IDX[r],:]
                    pre_nav=cash+float(np.dot(shares,mark))
                    current=shares*mark/pre_nav
                    drift=float(np.abs(intent-current).sum())
                    if drift>=.04:
                        # Cap nominal trade to 9.6% on last-known prices; actual
                        # fill-time notional is recorded separately, not guaranteed.
                        scaled=current+min(1.,.096/drift)*(intent-current)
                        scaled=np.maximum(scaled,0.)
                        # Cost-aware forecast gate: incremental forecasted
                        # relative gain > 0.1% one-way fees + 1bp uncertainty.
                        gated=policy.startswith('ridge80_costaware')
                        if gated:
                            if r==0:
                                f=features(market.daily,t)
                                pred=f@mod['daily_5']
                            else:
                                pred=known_intraday(market.cl,market.op,i,r)@mod[f'intra_{r}']
                            expected_gain=float(np.dot(scaled-current,pred))
                            expected_fee=FEE*float(np.abs(scaled-current).sum())
                            gated=expected_gain>expected_fee+0.0001
                        if not policy.startswith('ridge80_costaware') or gated:
                            before=cash+float(np.dot(shares,price))
                            cash,shares,q=trade(cash,shares,price,scaled)
                            max_rebal=max(max_rebal,q/before)
                            notional+=q;fees+=FEE*q;trades+=1
            # Price at subsequent execution for R1-R6; for R7 at the 16:00
            # close of the 15:45-16:00 bar. This is a mark, not a future input.
            nextprice=market.op[i,END_VAL_IDX[r],:] if r<6 else market.cl[i,25,:]
            nav_now=cash+float(np.dot(shares,nextprice))
            if not np.isfinite(nav_now) or nav_now<=0:raise ValueError('Invalid valuation')
            nav.append(nav_now)
        rows.append({'date':str(market.dates[i].date()),'nav_end':nav[-1],
                     'trades_to_date':trades,'fees_to_date':fees})
    nav=np.asarray(nav)
    ret=nav[1:]/nav[:-1]-1
    sigma=float(ret.std(ddof=1))
    result={'strategy':policy,'days':len(indices),'cumulative_return':float(nav[-1]/NAV0-1),
            'sharpe_proxy':float(ret.mean()*np.sqrt(252*7)/sigma) if sigma>1e-12 else 0.,
            'mdd_proxy':float(np.max(1.-nav/np.maximum.accumulate(nav))),
            'turnover_proxy':float(notional/(NAV0*len(indices)*7)),
            'trade_count':trades,'fees_usd':fees,
            'max_realized_rebalance_turnover':max_rebal}
    return result,rows


def alpha_ic(market,indices,models):
    rows=[]
    log=np.log(market.daily)
    for i in indices:
        month=market.dates[i].strftime('%Y-%m')
        f=features(market.daily,i-1)
        m=models[month]
        for h in (1,5):
            if i+h-1>=len(market.dates):continue
            pred=f@m[f'daily_{h}']
            target=log[i+h-1]-log[i-1]
            if np.std(pred)<1e-14 or np.std(target)<1e-14:continue
            ic=float(pd.Series(pred).rank().corr(pd.Series(target).rank()))
            rows.append({'date':str(market.dates[i].date()),'horizon':f'{h}d',
                         'spearman_ic':ic})
    return pd.DataFrame(rows)


def evaluate(market,out,input_path=None):
    manifest={'source':'Alpaca IEX 15-minute CSV, historical price proxy, not organizer fill tape',
              'no_organizer_apis':True,'fee_one_way':FEE,
              'bar_availability':'bar start 10:00 means completes 10:15; 10:25 decision sees 10:00 but not 10:15',
              'validation':'2026-07-01 to 2026-08-31',
              'diagnostic':'2026-09-01 to latest available complete trading day; results exploratory',
              'model':'monthly expanding ridge 1d/5d daily cross-sectional excess return + independently trained hourly forecasts',
              'score':'paired four-metric rank across tested strategies; NOT expected actual competition rank',
              'limitations':['IEX feed is not SIP','No SEC/news or LLM features','RAW prices not total-return adjusted',
                             'Missing sessions dropped','Bar opens are execution proxies, not guaranteed fills',
                             'Monthly refit with h-day labels matured before start month',
                             'Fixed test candidates, not hyperparameter optimized',
                             'Portfolio resets at validation/diagnostic split, continuous within each split']}
    manifest['input_path']=str(input_path) if input_path is not None else 'unspecified'
    manifest['data_asof']=str(market.dates[-1].date())
    manifest['oct07_09_included']=bool(market.dates[-1]>=pd.Timestamp('2026-10-09'))
    result_rows=[];model_rows=[];ics=[];nav_rows=[]
    for name,start,end in [('development','2026-07-01','2026-08-31'),('diagnostic','2026-09-01','2026-10-09')]:
        indices,models,models_meta=walkforward_models(market,start,end)
        print(f'PHASE30_{name.upper()}_DAYS:',len(indices),flush=True)
        for x in models_meta:model_rows.append({'segment':name,**x})
        ic=alpha_ic(market,indices,models)
        ic['segment']=name;ics.append(ic)
        for policy in POLICIES:
            outcome,path=replay(market,indices,models,policy)
            outcome['segment']=name
            result_rows.append(outcome)
            for p in path:p['segment']=name;p['strategy']=policy
            nav_rows.extend(path)
    result=pd.DataFrame(result_rows)
    for col,ascending in [('cumulative_return',False),('sharpe_proxy',False),('mdd_proxy',True),('turnover_proxy',True)]:
        result['rank_'+col]=result.groupby('segment')[col].rank(pct=True,ascending=ascending,method='average')
    result['four_metric_proxy_rank']=result[['rank_'+n for n in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]].mean(axis=1)
    result.to_csv(out/'phase30_strategy_results.csv',index=False)
    pd.concat(ics).to_csv(out/'phase30_forecast_ic.csv',index=False)
    pd.DataFrame(nav_rows).to_csv(out/'phase30_daily_nav.csv',index=False)
    (out/'phase30_model_training.json').write_text(json.dumps(model_rows,indent=2),encoding='utf-8')
    market.missing.to_csv(out/'phase30_incomplete_days.csv',index=False)
    # Train one RESEARCH-ONLY candidate on every eligible pre-Official label.
    # 5-day targets require five completely observed future sessions.
    last_cutoff=len(market.dates)
    final_model,final_counts=train_month(market,last_cutoff)
    final_artifact={'status':'RESEARCH_ONLY_NOT_AN_OFFICIAL_AGENT',
                   'as_of_completed_market_day':str(market.dates[-1].date()),
                   'latest_matured_1d_feature_day':str(market.dates[-2].date()),
                   'latest_matured_5d_feature_day':str(market.dates[-6].date()),
                   'feed':'Alpaca IEX', 'adjustment':'raw',
                   'label_counts':final_counts,
                   'coefficients':{k:[float(z) for z in v] for k,v in final_model.items()}}
    (out/'phase30_preofficial_research_model.json').write_text(json.dumps(final_artifact,indent=2),encoding='utf-8')
    manifest['all_complete_days']=len(market.dates)
    manifest['removed_incomplete_days']=len(market.missing)
    manifest['first_data_date']=str(market.dates[0].date())
    manifest['last_data_date']=str(market.dates[-1].date())
    (out/'phase30_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for segment in ('development','diagnostic'):
        print('===',segment.upper(),'2026 FOUR METRIC RESEARCH RANKS: LOWER BETTER ===',flush=True)
        y=result[result.segment==segment].sort_values('four_metric_proxy_rank')
        print(y[['strategy','cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd','four_metric_proxy_rank']].to_string(index=False,float_format=lambda x:f'{x:.6f}'),flush=True)
    for segment in ('development','diagnostic'):
        ic=pd.concat(ics).query('segment==@segment')
        print('2026_'+segment.upper()+'_MEAN_FORECAST_IC:',ic.groupby('horizon').spearman_ic.mean().to_dict())
    print('COMPLETE_DAYS:',len(market.dates),'SKIPPED_INCOMPLETE_DAYS:',len(market.missing))
    print('PHASE30_RESULT_DIR:',out)
    print('PHASE30_COMPLETE_RESEARCH_ONLY_NO_SUBMISSION')


def self_test():
    rng=np.random.default_rng(20261011)
    dates=pd.bdate_range('2025-09-02',periods=275)
    # Synthetic simulation: randomized, nontrivial cross-sectional daily and
    # intraday returns. No market edge or performance is inferred from it.
    daily=100*np.exp(np.cumsum(rng.normal(.0001,.011,(len(dates),30)),axis=0))
    op=np.empty((len(dates),26,30));cl=np.empty_like(op)
    for i in range(len(dates)):
        start=daily[i-1] if i>0 else daily[0]
        path=np.exp(np.cumsum(rng.normal(0,.002,(26,30)),axis=0))*start
        op[i]=path;cl[i]=path*np.exp(rng.normal(0,.0006,(26,30)))
    # Daily marks are the fully known completed final bar.
    market=Market(dates,op,cl,cl[:,-1,:],pd.DataFrame())
    assert KNOWN_IDX[1]==2 and SLOTS[2]=='10:00' and SLOTS[3]=='10:15'
    cutoff=210
    a,c=train_month(market,cutoff)
    assert len(a['daily_5'])==5 and len(a['intra_1'])==2
    # Replace all observations on/after cutoff; earlier fitted models invariant.
    future_op=market.op.copy();future_cl=market.cl.copy()
    future_op[cutoff:]*=1.3;future_cl[cutoff:]*=1.3
    other=Market(dates,future_op,future_cl,future_cl[:,-1,:],pd.DataFrame())
    b,_=train_month(other,cutoff)
    assert all(np.array_equal(a[k],b[k]) for k in a),'Model learned from future test data!'
    before=known_intraday(market.cl,market.op,cutoff,1)
    tampered=market.cl.copy();tampered[cutoff,3:,:]*=100
    after=known_intraday(tampered,market.op,cutoff,1)
    assert np.array_equal(before,after),'Round 2 saw not-yet-complete 10:15 bar!'
    for pol in POLICIES:
        res,_=replay(market,np.arange(cutoff,cutoff+14),{'2026-06':a,'2026-07':a,'2026-08':a,'2026-09':a},pol)
        assert np.isfinite(list(res.values())[2:]).all(),pol
        assert res['trade_count']>=1
    static=np.full(30,.8/30)
    cash,shares,notional=trade(NAV0,np.zeros(30),np.full(30,100.),static)
    assert 0<notional and cash+float(np.dot(shares,np.full(30,100.)))<NAV0
    print('PHASE30_SELF_TEST_PASS')
    print('MONTHLY_MATURED_LABEL_CAUSALITY_AND_SUBMISSION_TIME_BARS_PASS')
    print('CONTINUOUS_REPLAY_FEES_7_ROUNDS_AND_HOLD_GATE_PASS')
    print('NO_NETWORK_NO_ORGANIZER_SUBMISSION')


def main():
    parser=argparse.ArgumentParser()
    ex=parser.add_mutually_exclusive_group(required=True)
    ex.add_argument('--self-test',action='store_true');ex.add_argument('--run',action='store_true')
    parser.add_argument('--input',default=None,help='Optional Alpaca 15m CSV; permits Oct 6 as-of research without Phase29')
    args=parser.parse_args()
    try:
        if args.self_test:self_test()
        else:
            root=Path.cwd()
            out=root/OUT;out.mkdir(parents=True,exist_ok=True)
            path=Path(args.input).expanduser() if args.input else root/INPUT
            if not path.is_absolute():path=root/path
            minimum='2026-10-06' if args.input else '2026-10-07'
            market=read_market(path,min_last_day=minimum)
            print('PHASE30_INPUT_ASOF:',str(market.dates[-1].date()),'OCT07_09_INCLUDED:',bool(market.dates[-1]>=pd.Timestamp('2026-10-09')),flush=True)
            evaluate(market,out,path)
    except Exception as exc:
        print('PHASE30_FAILED:',repr(exc),file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
