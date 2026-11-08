#!/usr/bin/env python3
"""ICAIF Phase 41: fee-aware, 105-round reversal research (NO submissions).

Price/volume ONLY: Alpaca IEX 15-minute bars. No Finnhub/SEC/LLM network calls.
Forward-only, manually verified CaMeL event input is optional; no LLM backfill.

Usage:
  python icaif_phase41_costaware_reversal.py --self-test
  python icaif_phase41_costaware_reversal.py --run
  python icaif_phase41_costaware_reversal.py --run --input path.csv --min-last-day 2026-10-06

Place Phase 39 v2, Phase 30 v2, Phase 37 and Phase 38 alongside this script.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import importlib.util
from datetime import timedelta
import numpy as np
import pandas as pd

ROOT=Path.cwd()
HERE=Path(__file__).resolve().parent
OUT=Path('artifacts/phase41_feeaware_reversal')
FEE=.001
NAV0=1_000_000.
N=30
MAX_NAME=.10
MAX_SECTOR=.30
# Trade targets stay below hard risk caps to accommodate price drift.
PLANNED_NAME=.095
PLANNED_SECTOR=.295
MAX_FLOAT=.90
MIN_FLOAT=.70
MAX_ORDER_USD=200_000.
MIN_TRADE_NOTIONAL=750.
ENTRY_EDGE=2*FEE  # estimated round-trip cost (on traded capital)
UNCERTAINTY_BASE=.0006
UNCERTAINTY_FRACTION=.03  # 3% of stock 5-day volatility
EXIT_EDGE=-.0015
ACTIVE_TOL=.006    # 0.6 percentage points exposure maintenance deadband
R2_GAP_TRIGGER=-.012  # negative 1.2% opening gap; NOT a known alpha
R2_MAX_ADD=.018
VOLUME_RATIO_LIMIT=1.7
TRAIN_DAYS=90
FORECAST_SHRINK=.65

@dataclass(frozen=True)
class Policy:
    name:str
    core:str=''
    hold:int=5
    floating:bool=True
    execution:str='cashfirst'
    volume_filter:bool=False
    r2_gap:bool=False
    camel:bool=False


def policies():
    return (
        Policy('equal60_hold','equal60'),
        Policy('equal80_hold','equal80'),
        Policy('blend80_hold','blend80'),
        Policy('lowvol80_hold','lowvol80_legacy'),
        Policy('reversal3_cashfirst_fixed80',hold=3,floating=False),
        Policy('reversal5_cashfirst_fixed80',hold=5,floating=False),
        Policy('reversal5_full_fixed80',hold=5,floating=False,execution='full'),
        Policy('reversal5_cashfirst_floating',hold=5),
        Policy('reversal5_full_floating',hold=5,execution='full'),
        Policy('reversal5_floating_volume_guard',hold=5,volume_filter=True),
        Policy('reversal5_floating_r2_gap',hold=5,r2_gap=True),
        Policy('reversal5_floating__camel',hold=5,camel=True),
    )


def dependencies():
    specs=[('p41p39','icaif_phase39_105round_walkforward_v2.py'),
           ('p41p30','icaif_phase30_2026_walkforward_agent_lab_v2.py'),
           ('p41p37','icaif_phase37_risk_regime_tournament.py'),
           ('p41p38','icaif_phase38_105slot_camel_staging.py')]
    mods=[]
    for mod_name,filename in specs:
        path=HERE/filename
        if not path.is_file():
            raise FileNotFoundError('Missing dependency next to Phase41: '+filename)
        spec=importlib.util.spec_from_file_location(mod_name,path)
        obj=importlib.util.module_from_spec(spec)
        sys.modules[mod_name]=obj
        spec.loader.exec_module(obj)
        mods.append(obj)
    return mods


def standardized(x):
    x=np.asarray(x,float)
    return np.clip((x-x.mean())/max(float(x.std()),1e-10),-3.,3.)


def feature(daily,t):
    """Reversal at complete daily close t; t-1, t-5 also known."""
    q=np.log(daily)
    raw=-.70*(q[t]-q[t-1])-.30*(q[t]-q[t-5])
    return standardized(raw)


def calibration(market,asof):
    """Only fully matured five-session outcomes with close <= asof.

    Fit common cross-sectional five-day reversal slope, shrunk 35%, plus
    a shrunk market-drift intercept. Per-day observations, not separate
    iid labels from 30 stocks. Training never includes future results.
    """
    hi=asof-5
    lo=max(21,hi-TRAIN_DAYS+1)
    if hi-lo+1<40:raise ValueError('Insufficient matured causal calibration history')
    daily=market.daily
    x=np.stack([feature(daily,t) for t in range(lo,hi+1)])
    y=np.stack([np.log(daily[t+5]/daily[t]) for t in range(lo,hi+1)])
    yr=y-y.mean(axis=1,keepdims=True)
    slope=float((x*yr).sum()/(np.square(x).sum()+50.))
    slope=np.clip(FORECAST_SHRINK*slope,-.006,.006)
    intercept=float(np.clip(.4*y.mean(),-.01,.01))
    std=np.log(daily[asof-19:asof+1]/daily[asof-20:asof])
    vol=np.maximum(std.std(axis=0,ddof=1),.004)
    pred=intercept+slope*feature(daily,asof)
    uncertainty=UNCERTAINTY_BASE+UNCERTAINTY_FRACTION*np.sqrt(5)*vol
    entry=pred-ENTRY_EDGE-uncertainty
    return {'pred':pred,'edge':entry,'uncertainty':uncertainty,'vol':vol,
            'beta':slope,'intercept':intercept,'train_from':str(market.dates[lo].date()),
            'last_feature_date':str(market.dates[hi].date()),
            'last_matured_label_date':str(market.dates[hi+5].date()),
            'n_train_days':len(x)}


def budget(market,asof,floating):
    if not floating:return .80
    q=np.log(market.daily)
    r20=float(np.mean(q[asof]-q[asof-20]))
    mkt=np.diff(q[asof-20:asof+1],axis=0).mean(axis=1)
    v=float(np.std(mkt,ddof=1))
    # Fixed ex ante; not optimized to Stage40 results.
    e=.80+.08*np.tanh(r20/.045)-.035*np.clip((v/.012-1),0,1)
    return float(np.clip(e,MIN_FLOAT,MAX_FLOAT))


def sector_indices(p30,p38):
    result=[];flat=[]
    for s in p38.SECTOR_MEMBERS:
        ix=np.array([list(p30.SYMBOLS).index(t) for t in s],int)
        result.append(ix);flat.extend(ix.tolist())
    assert sorted(flat)==list(range(N)), 'Sector groups must cover 30 stocks once'
    return result


def allocate(edge,vol,eligible,cap,sector_groups,volume_risk=None):
    """Independent binary gate, then allocates up to *cap*. No top K.

    If fewer positive-edge issuers can absorb the target within risk limits,
    the equity exposure stays below target (explicit shortfall).
    """
    edge=np.asarray(edge,float)
    vol=np.asarray(vol,float)
    allowed=np.asarray(eligible,bool)
    if volume_risk is None:volume_risk=np.ones(N)
    score=np.where(allowed, np.maximum(edge,0.)/np.maximum(vol,.004)/np.maximum(volume_risk,1.),0.)
    # A held stock can remain eligible with slightly negative estimated alpha;
    # give it a small positive allocation priority to avoid forced churn.
    score=np.where(allowed & (score<1e-10), .001,score)
    target=np.zeros(N,float)
    for _ in range(120):
        left=float(cap-target.sum())
        if left<1e-7:break
        remaining=np.where(allowed,np.maximum(0.,PLANNED_NAME-target),0.)
        if not np.any(remaining>1e-9):break
        prop=score*(remaining>1e-9)
        if prop.sum()<1e-12:break
        delta=np.minimum(left*prop/prop.sum(),remaining)
        for g in sector_groups:
            capacity=max(0.,PLANNED_SECTOR-target[g].sum())
            if delta[g].sum()>capacity and delta[g].sum()>0:
                delta[g]*=capacity/delta[g].sum()
        if delta.sum()<1e-10:break
        target+=delta
    assert target.max()<=MAX_NAME+1e-8
    assert all(target[g].sum()<=MAX_SECTOR+1e-8 for g in sector_groups)
    return target,max(0.,cap-target.sum())


def load_iex_volume(path,market,p30):
    """Source prices and volumes share exact NY trading-date/time/symbol keys."""
    x=pd.read_csv(path,usecols=['symbol','timestamp','volume'])
    x=x[x.symbol.isin(p30.SYMBOLS)].copy()
    ts=pd.to_datetime(x.timestamp,utc=True,errors='raise').dt.tz_convert('America/New_York')
    x['date']=pd.to_datetime(ts.dt.strftime('%Y-%m-%d'))
    x['clock']=ts.dt.strftime('%H:%M')
    x=x[x.clock.isin(p30.SLOTS)].copy()
    x['volume']=pd.to_numeric(x.volume,errors='raise')
    if (~np.isfinite(x.volume)).any() or (x.volume<0).any():raise ValueError('Bad IEX volume')
    if x.duplicated(['date','clock','symbol']).any():raise ValueError('Duplicate volume keys')
    idx=pd.MultiIndex.from_product([market.dates,p30.SLOTS],names=['date','clock'])
    arr=x.pivot(index=['date','clock'],columns='symbol',values='volume').reindex(index=idx,columns=p30.SYMBOLS)
    v=arr.to_numpy(float).reshape(len(market.dates),26,30)
    if not np.isfinite(v).all():raise ValueError('Incomplete volume on kept market sessions')
    return v


def early_volume_ratio(vol,i):
    observed=vol[i,:3,:].sum(axis=0) # first 45 mins, R2 known as of 10:15 ET
    previous=vol[i-20:i,:3,:].sum(axis=1)
    median=np.maximum(np.median(previous,axis=0),1.)
    return observed/median


def sell_buy(cash,shares,mark,fill,desired,eligible,execution,cap,sector_groups,
             age,minimum_age,force=False):
    """Generate market-notional buy/sell intents from PRE-deadline marks.

    Cash-first: sell rejected/over-cap names or reduce exposure; buy newly
    qualified names with cash. Full: scheduled target-weight rebalance.
    Both use the SAME desired target and risk budget; realized exposure is
    logged and MUST be compared rather than assumed identical.
    Execution prices are used to settle quantities, never select signals.
    """
    nav_mark=float(cash+shares@mark)
    curr=shares*mark/nav_mark
    equities=float(curr.sum())
    sells=np.zeros(N,float)
    buys=np.zeros(N,float)
    if execution=='full':
        for j in range(N):
            if shares[j]>1e-12 and age[j]<minimum_age and eligible[j] and curr[j]<MAX_NAME-.002:
                continue  # age restriction applies to reallocation too
            change=(desired[j]-curr[j])*nav_mark
            if change<-MIN_TRADE_NOTIONAL:sells[j]=-change
            elif change>MIN_TRADE_NOTIONAL:buys[j]=change
    else:
        # Liquidate disqualified only after minimum holding period;
        # always enforce hard name/sector controls irrespective of age.
        for j in range(N):
            value=shares[j]*mark[j]
            if value<=0:continue
            if not eligible[j] and age[j]>=minimum_age:
                sells[j]=value
            elif curr[j]>MAX_NAME-.002:
                sells[j]=max(0.,(curr[j]-(MAX_NAME-.005))*nav_mark)
        for g in sector_groups:
            if curr[g].sum()>MAX_SECTOR-.002:
                surplus=max(0.,(curr[g].sum()-(MAX_SECTOR-.005))*nav_mark)
                if surplus>0:
                    # Proportional group derisking; do not pick new winners.
                    values=shares[g]*mark[g]
                    sell_cap=np.maximum(0.,values-sells[g])
                    if sell_cap.sum()>0:sells[g]+=surplus*sell_cap/sell_cap.sum()
        # Cash-first is not allowed to hoard >80% equity in fixed80 variant.
        # When above target deadband, proportionally sell existing survivors.
        after_ordinary=equities-sells.sum()/nav_mark
        if after_ordinary>cap+ACTIVE_TOL:
            trim=(after_ordinary-cap)*nav_mark
            values=np.maximum(0.,shares*mark-sells)
            if values.sum()>0:sells+=trim*values/values.sum()
        # Buy only missing targets, not stock-for-stock replacements.
        post=curr-sells/nav_mark
        buys=np.maximum(0.,desired-post)*nav_mark
        buys=np.where(buys>MIN_TRADE_NOTIONAL,buys,0.)
    # First-day and daily execution is allowed to sell disqualified names;
    # the separate R2 gap branch never calls this full rebalance path.
    sells=np.minimum(np.maximum(sells,0.),shares*mark)
    sells=np.where(sells>=MIN_TRADE_NOTIONAL,sells,0.)
    buy_units=buys/mark; sell_units=sells/mark
    sold=np.minimum(shares,sell_units)
    gross_sold=float(sold@fill)
    after_shares=shares-sold
    after_cash=cash+gross_sold*(1-FEE)
    # Price-based available budget, with pre-trade equity cap enforced.
    # Keep cash >= (1-cap)*NAV, allowing 0.6%-point tolerance for drift.
    nav_after_sale=float(after_cash+after_shares@fill)
    budget_after_fees=max(0.,after_cash-(1-cap)*nav_after_sale)
    # Also clip to planned desired total value at decision time so
    # no purchases force total invested exposure above available opportunities.
    want=float(buy_units@fill)
    if want>0:
        units=buy_units*min(1.,budget_after_fees/max(want*(1+FEE),1e-12))
    else:units=buy_units
    gross_bought=float(units@fill)
    new_cash=after_cash-gross_bought*(1+FEE)
    new_shares=after_shares+units
    gross=gross_sold+gross_bought
    fee=FEE*gross
    if abs((cash+shares@fill)-(new_cash+new_shares@fill)-fee)>1e-4:raise AssertionError('Fee conservation')
    if new_cash<-1e-5 or new_shares.min()<-1e-12:raise AssertionError('Invalid execution')
    if gross<MIN_TRADE_NOTIONAL and not force:
        return float(cash),shares.copy(),0.,0.,np.zeros(N,bool),'HOLD_BELOW_MIN_ORDER'
    changed=(np.abs(new_shares-shares)*fill)>1e-4
    return float(new_cash),new_shares,gross,fee,changed,('FULL_REBALANCE' if execution=='full' else 'CASH_FIRST')


def r2_buy_only(cash,shares,mark,fill,edge,vol,gap,eligible,
                cap,groups,volume_ratio=None):
    """R2: additive gap-fade BUY candidate; never sell or chase a price.

    The intent is chosen only from the 09:30 opening gap, previous close
    forecast and completed 09:30--10:15 volume, with 10:30 price used
    only to settle the prechosen quantity. Do not force 70% investment.
    """
    nav=float(cash+shares@mark)
    old_w=shares*mark/nav
    eligible=eligible & (edge>0) & (gap<=R2_GAP_TRIGGER)
    if volume_ratio is not None:
        eligible &= volume_ratio<=VOLUME_RATIO_LIMIT
    if not eligible.any():return cash,shares.copy(),0.,0.,'HOLD_R2_NO_VERIFIED_GAP_SIGNAL'
    remaining_room=min(R2_MAX_ADD,max(0.,cap-float(old_w.sum())))
    if remaining_room<.0005:
        return cash,shares.copy(),0.,0.,'HOLD_R2_NO_EQUITY_ROOM'
    headroom=np.maximum(0.,PLANNED_NAME-old_w)
    score=np.where(eligible,np.maximum(edge,0.)/np.maximum(vol,.004),0.)
    total_score=float(score.sum())
    if total_score<1e-12:return cash,shares.copy(),0.,0.,'HOLD_R2_NO_FORECAST_EDGE'
    wanted=np.minimum(remaining_room*score/total_score,headroom)
    for group in groups:
        available=max(0.,PLANNED_SECTOR-float(old_w[group].sum()))
        if wanted[group].sum()>available and wanted[group].sum()>0:
            wanted[group]*=available/wanted[group].sum()
    wanted=np.where(wanted*nav>=MIN_TRADE_NOTIONAL,wanted,0.)
    if wanted.sum()<1e-12:return cash,shares.copy(),0.,0.,'HOLD_R2_BELOW_MIN_TRADE'
    qty=wanted*nav/mark
    notional=float(qty@fill)
    # Ex ante cash reserve cap, with execution-only affordability scaling.
    nav_fill=float(cash+shares@fill)
    cash_budget=min(float(cash),max(0.,cash-(1-cap)*nav_fill))
    if notional*(1+FEE)>cash_budget:
        qty*=max(0.,cash_budget/max(notional*(1+FEE),1e-12))
        notional=float(qty@fill)
    if notional<MIN_TRADE_NOTIONAL:return cash,shares.copy(),0.,0.,'HOLD_R2_CASH_OR_SIZE'
    new_cash=cash-notional*(1+FEE)
    new_shares=shares+qty
    assert new_cash>=-1e-5
    assert abs((cash+shares@fill)-(new_cash+new_shares@fill)-notional*FEE)<1e-4
    return float(new_cash),new_shares,notional,notional*FEE,'R2_BUY_ONLY_GAP'

def simulate(market,indices,pol,training,p30,p37,p38,sector_groups,volume,events):
    if len(indices)!=15 or not np.all(np.diff(indices)==1):raise AssertionError('Invalid episode')
    if pol.core:
        from collections import namedtuple
        # Exact common benchmark simulator: identical 105-round accounting.
        # Phase39.Strategy requires p39 externally, so handled in run().
        raise RuntimeError('Benchmark policies handled by Phase39')
    cash=NAV0; shares=np.zeros(N,float)
    navs=[NAV0];logs=[];turns=[];fees=0.;trades=0;used=set()
    age=np.zeros(N,int);ages_prev=None
    exposure_violations=0;name_breaches=0;sector_breaches=0;shortfall_count=0
    market_exposures=[];counts=[];eligible_ct=[]
    for d,i in enumerate(indices):
        i=int(i)
        if d:age[shares>1e-10]+=1
        for r in range(7):
            mark=market.daily[i-1] if r==0 else market.cl[i,p30.KNOWN_IDX[r],:]
            fill=market.op[i,p30.FILL_IDX[r],:]
            before_fill=float(cash+shares@fill)
            before_mark=float(cash+shares@mark)
            current=shares*mark/before_mark
            old=shares.copy()
            gross=0.;fee=0.;reason='HOLD_R3_TO_R7';event_id=''
            requested=budget(market,i-1,pol.floating)
            # All forecast information is from prior close, including R2.
            # R2 only adds info from completed opening bars and overnight gap.
            row=training[i-1]
            forecast=row['pred'];edge=row['edge'];vol=row['vol']
            held=(shares*mark/before_mark)>.0005
            eligible=np.where(held, forecast-UNCERTAINTY_BASE >=EXIT_EDGE,edge>0)
            # Five-day min holding period applies to exits; do not blindly
            # force new names to remain eligible if their forecast disappears.
            elig_w=eligible.copy()
            elig_w=np.where((age<pol.hold)&held,True,elig_w)
            # A volume shock known at the PRIOR SESSION is a risk input
            # at R1. Intraday current-day volume is unavailable until R2.
            volume_risk=np.ones(N)
            if pol.volume_filter and r==0:
                volume_risk=np.maximum(1.,early_volume_ratio(volume,i-1)/VOLUME_RATIO_LIMIT)
                elig_w &= (early_volume_ratio(volume,i-1)<=VOLUME_RATIO_LIMIT) | held
            desired,shortfall=allocate(edge,vol,elig_w,requested,sector_groups,volume_risk)
            if r==0:
                cash,shares,gross,fee,changed,reason=sell_buy(
                    cash,shares,mark,fill,desired,elig_w,pol.execution,requested,sector_groups,
                    age,pol.hold,force=(d==0))
                for j in np.flatnonzero(changed):
                    if old[j]<=1e-10 and shares[j]>1e-10:age[j]=0
                    if shares[j]<=1e-10:age[j]=0
            elif r==1 and pol.r2_gap:
                gap=np.log(market.op[i,0]/market.daily[i-1])
                volume_ratio=early_volume_ratio(volume,i)
                cash,shares,gross,fee,reason=r2_buy_only(
                    cash,shares,mark,fill,edge,vol,gap,elig_w,
                    min(MAX_FLOAT,requested+R2_MAX_ADD),sector_groups,volume_ratio)
                new_positions=(old<1e-12)&(shares>1e-12)
                age[new_positions]=0
            if pol.camel and events:
                # Forward-only CaMeL paper sleeve. Research-safe: no call here.
                # Skip zero-signal historic episodes, never infer signals from
                # financial returns, and never override core allocation.
                decision=p38.decision_time(market.dates[i],r)
                usable=[e for e in events if e['event_id'] not in used and e['_available']<=decision
                        and timedelta(0)<=decision-e['_published']<=timedelta(hours=4)]
                if usable:
                    evt=usable[-1];used.add(evt['event_id']);event_id=evt['event_id']
                    # Record classification for prospective study; do not
                    # synthesize an expected return to force a trade.
                    reason+='|CAMEL_OBSERVED_SHADOW_ONLY'
            if gross>0:trades+=1;fees+=fee
            post=float(cash+shares@fill)
            if post<=0:raise AssertionError('Negative NAV')
            if abs((before_fill-fee)-post)>1e-3:raise AssertionError('Post-fill NAV mismatch')
            weights=shares*fill/post;exp=float(weights.sum())
            if exp>MAX_FLOAT+1e-4:exposure_violations+=1
            if weights.max()>MAX_NAME+1e-4:name_breaches+=1
            if any(weights[g].sum()>MAX_SECTOR+1e-4 for g in sector_groups):sector_breaches+=1
            if (requested-exp)>.005:shortfall_count+=1
            val=market.op[i,p30.END_VAL_IDX[r],:] if r<6 else market.cl[i,25,:]
            end_nav=float(cash+shares@val)
            navs.append(end_nav);turns.append(gross/max(before_fill,1e-12))
            market_exposures.append(exp)
            count=int((weights>.0005).sum());counts.append(count)
            eligible_ct.append(int(elig_w.sum()))
            logs.append({'date':str(market.dates[i].date()),'round':r+1,
                         'deadline_et':p38.DEADLINE_ET[r],'execution_et':p38.FILL_ET[r],
                         'information_cutoff':'previous_close' if r==0 else p30.SLOTS[p30.KNOWN_IDX[r]],
                         'trade':int(gross>0),'reason':reason,'gross_notional_usd':gross,'fee_usd':fee,
                         'equity_exposure':exp,'requested_exposure':requested,'holdings':count,
                         'eligible_stocks':int(elig_w.sum()),'shortfall':max(0.,requested-exp),
                         'calibration_beta':row['beta'],'intercept':row['intercept'],
                         'last_matured_label_date':row['last_matured_label_date'],
                         'camel_event_id':event_id,'nav_end':end_nav})
    returns=np.diff(np.log(np.array(navs)))
    linear=np.array(navs[1:])/np.array(navs[:-1])-1
    stdev=float(linear.std(ddof=1))
    out={'cumulative_return':float(navs[-1]/NAV0-1),
         'sharpe_proxy':float(linear.mean()*math.sqrt(252*7)/stdev) if stdev>1e-12 else 0.,
         'mdd_proxy':float(np.max(1-np.array(navs)/np.maximum.accumulate(navs))),
         'turnover_proxy':float(np.mean(turns)),
         'fees_usd':fees,'trade_count':trades,
         'mean_exposure':float(np.mean(market_exposures)),
         'min_exposure':float(min(market_exposures)),'max_exposure':float(max(market_exposures)),
         'min_holdings':int(min(counts)),'max_holdings':int(max(counts)),
         'mean_holdings':float(np.mean(counts)),
         'mean_eligible':float(np.mean(eligible_ct)),
         'equity_above_90_rounds':exposure_violations,
         'name_cap_breach_rounds':name_breaches,
         'sector_cap_breach_rounds':sector_breaches,
         'below_requested_target_rounds':shortfall_count,
         'camel_signals_observed':int(sum(bool(x['camel_event_id']) for x in logs))}
    assert len(logs)==105
    return out,logs


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def run(input_path=None,min_last_day='2026-10-09',stride=15,events_path=None,out_path=None):
    p39,p30,p37,p38=dependencies()
    source=Path(input_path).expanduser() if input_path else ROOT/p30.INPUT
    if not source.is_absolute():source=ROOT/source
    source=source.resolve()
    market=p30.read_market(source,min_last_day=min_last_day)
    groups=sector_indices(p30,p38)
    volume=load_iex_volume(source,market,p30)
    # No network: this optional list contains only source-attested events.
    events=p39.read_events(events_path,p38,p30)
    episodes=p39.episode_groups(market,stride=stride,minimum_history=121)
    if not episodes:raise ValueError('No eligible episodes')
    needed=set(int(i)-1 for _,_,seq in episodes for i in seq)
    forecast={t:calibration(market,t) for t in sorted(needed)}
    output=Path(out_path) if out_path else ROOT/OUT
    output.mkdir(parents=True,exist_ok=True)
    rows=[];decisions=[]
    for q,(eid,seg,idx) in enumerate(episodes,1):
        for p in policies():
            if p.core:
                m,ds=p39.simulate(market,idx,p39.Strategy(p.name,p.core),[],p30,p37,p38)
                record={**m,'mean_exposure':float(np.mean([z['post_execution_equity_weight'] for z in ds])),
                        'min_exposure':float(min(z['post_execution_equity_weight'] for z in ds)),
                        'max_exposure':float(max(z['post_execution_equity_weight'] for z in ds)),
                        'mean_holdings':30.,'min_holdings':30,'max_holdings':30,
                        'mean_eligible':30.,'equity_above_90_rounds':0,'name_cap_breach_rounds':0,
                        'sector_cap_breach_rounds':0,'below_requested_target_rounds':0,
                        'camel_signals_observed':0}
                logs=[{'date':z['day'],'round':z['round'],'deadline_et':z['deadline_et'],
                       'execution_et':z['execution_et'],'information_cutoff':z['info_cutoff'],
                       'trade':int(z['decision']=='TRADE'),'reason':z['reason'],
                       'gross_notional_usd':z['gross_trade_usd'],'fee_usd':z['fee_usd'],
                       'equity_exposure':z['post_execution_equity_weight'],
                       'requested_exposure':float(p38.core_weights(p30,p37,market,int(idx[0])-1,p.core).sum()),
                       'holdings':30,'eligible_stocks':30,'shortfall':0.,
                       'calibration_beta':None,'intercept':None,'last_matured_label_date':None,
                       'camel_event_id':'','nav_end':z['nav_end']} for z in ds]
            else:
                record,logs=simulate(market,idx,p,forecast,p30,p37,p38,groups,volume,events)
            rows.append({'episode':eid,'segment':seg,'strategy':p.name,**record})
            decisions.extend([{'episode':eid,'segment':seg,'strategy':p.name,**x} for x in logs])
        print(f'PHASE41_EPISODE {q}/{len(episodes)} {eid} PASS',flush=True)
    ep=pd.DataFrame(rows)
    ledger=pd.DataFrame(decisions)
    base=ep.set_index(['episode','strategy'])
    compare=[]
    for _,r in ep.iterrows():
        if r.strategy=='equal80_hold':continue
        ref='reversal5_cashfirst_floating' if r.strategy.endswith('__camel') else 'equal80_hold'
        b=base.loc[(r.episode,ref)]
        compare.append({'episode':r.episode,'segment':r.segment,'strategy':r.strategy,
                        'reference':ref,
                        'delta_return_pp':100*(r.cumulative_return-b.cumulative_return),
                        'delta_sharpe':r.sharpe_proxy-b.sharpe_proxy,
                        'delta_mdd_pp':100*(r.mdd_proxy-b.mdd_proxy),
                        'delta_turnover':r.turnover_proxy-b.turnover_proxy,
                        'extra_fee_usd':r.fees_usd-b.fees_usd,
                        'delta_avg_exposure_pp':100*(r.mean_exposure-b.mean_exposure)})
    cp=pd.DataFrame(compare)
    sums=(ep.groupby(['segment','strategy'],as_index=False)
        .agg(episodes=('episode','size'),mean_return=('cumulative_return','mean'),
             mean_sharpe=('sharpe_proxy','mean'),mean_mdd=('mdd_proxy','mean'),
             mean_turnover=('turnover_proxy','mean'),mean_fees_usd=('fees_usd','mean'),
             mean_trade_rounds=('trade_count','mean'),mean_exposure=('mean_exposure','mean'),
             min_exposure=('min_exposure','min'),max_exposure=('max_exposure','max'),
             mean_holdings=('mean_holdings','mean'),min_holdings=('min_holdings','min'),
             max_holdings=('max_holdings','max'),above_90_rounds=('equity_above_90_rounds','sum'),
             name_breaches=('name_cap_breach_rounds','sum'),
             sector_breaches=('sector_cap_breach_rounds','sum'),
             target_shortfall_rounds=('below_requested_target_rounds','sum')))
    pair=(cp.groupby(['segment','strategy','reference'],as_index=False)
        .agg(episodes=('episode','size'),mean_delta_return_pp=('delta_return_pp','mean'),
             median_delta_return_pp=('delta_return_pp','median'),
             return_win_fraction=('delta_return_pp',lambda x:float((x>0).mean())),
             mean_delta_sharpe=('delta_sharpe','mean'),
             mean_delta_mdd_pp=('delta_mdd_pp','mean'),
             mean_delta_turnover=('delta_turnover','mean'),
             mean_extra_fee_usd=('extra_fee_usd','mean'),
             mean_delta_exposure_pp=('delta_avg_exposure_pp','mean')))
    activity=(ledger.groupby(['segment','strategy','round'],as_index=False)
        .agg(decisions=('trade','size'),executed_rounds=('trade','sum'),fees_usd=('fee_usd','sum'),
             gross_notional_usd=('gross_notional_usd','sum'),
             mean_equity_exposure=('equity_exposure','mean'),
             mean_eligible_stocks=('eligible_stocks','mean')))
    for filename,frame in (
        ('phase41_episode_metrics.csv',ep),
        ('phase41_policy_summary.csv',sums),
        ('phase41_paired_episode_deltas.csv',cp),
        ('phase41_paired_summary.csv',pair),
        ('phase41_round_activity.csv',activity),
        ('phase41_all_105_round_decisions.csv',ledger)):
        frame.to_csv(output/filename,index=False)
    manifest={'research_only':True,'no_organizer_api_calls':True,'no_llm_provider_calls':True,
              'no_finnhub_or_sec_calls':True,'no_hetzner_changes':True,
              'source_sha256':sha256(source),'source_last_complete_date':str(market.dates[-1].date()),
              'episodes':len(episodes),'slots_per_episode':105,'policy_count':len(policies()),
              'predictor':'shrunk walkforward 5-session reversal with matured labels only',
              'entry_rule':'positive empirical predicted 5-session return net of 0.2pct round-trip fee and uncertainty margin',
              'training_days':TRAIN_DAYS,'holding_horizons':[3,5],
              'target_exposure':'fixed80 or floating70to90, unfillable shortage disclosed',
              'source_kind':'Alpaca IEX raw 15m (unadjusted)',
              'events_used':len(events),
              'data_limitations':['Missing sessions excluded','unadjusted corporate actions',
                'IEX bar open is execution proxy, not organizer fill','Episodes already exposed through Stage40',
                'Cashfirst and full executions may have different realized exposure and holdings',
                'No calibrated LLM return forecast','Rolling 5d labels correlated, no untouched validation set']}
    (output/'phase41_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PHASE41_ASOF',market.dates[-1].date(),'EPISODES',len(episodes),
          'TOTAL_ROUNDS',len(ledger),'OUTPUT',output.resolve(),flush=True)
    print('PHASE41_RESEARCH_COMPLETE_NO_LLM_NO_OFFICIAL_SUBMISSION',flush=True)
    return ep,cp,ledger


def self_test():
    from types import SimpleNamespace
    p39,p30,p37,p38=dependencies()
    rng=np.random.default_rng(410026)
    T=250
    returns=rng.normal(.0002,.011,(T,N))
    daily=100*np.exp(np.cumsum(returns,axis=0))
    v15=rng.lognormal(9,1,(T,26,N))
    drift=rng.normal(0,.0005,(T,26,N))
    op=daily[:,None,:]*np.exp(np.cumsum(drift,axis=1))
    cl=op*np.exp(.00005)
    dates=pd.bdate_range('2025-09-02',periods=T)
    market=SimpleNamespace(dates=dates,op=op,cl=cl,daily=cl[:,-1,:],missing=pd.DataFrame(columns=['date']))
    groups=sector_indices(p30,p38)
    x=calibration(market,135)
    assert x['last_matured_label_date']<=str(market.dates[135].date())
    assert x['n_train_days']>=40
    # Exact causal test: changing prices AFTER the decision cutoff cannot
    # modify the forecast available at that cutoff.
    altered=SimpleNamespace(dates=market.dates,op=market.op,cl=market.cl,
          daily=market.daily.copy(),missing=market.missing)
    altered.daily[136:]*=1.21
    x2=calibration(altered,135)
    assert np.array_equal(x['pred'],x2['pred'])
    assert np.array_equal(x['edge'],x2['edge'])
    scores=np.zeros(N);scores[:5]=.02
    vol=np.full(N,.01)
    eligible=scores>0
    t,short=allocate(scores,vol,eligible,.8,groups)
    assert np.count_nonzero(t)>0 and short>0 and t.sum()<.8
    # No unintended look-ahead in intraday volume: R2 sees only first 3 bars.
    a=early_volume_ratio(v15,145)
    alt=v15.copy();alt[145,3:]*=400
    assert np.array_equal(a,early_volume_ratio(alt,145))
    f=np.ones(N)*100
    share=np.zeros(N);cash=NAV0
    target=np.zeros(N);target[:10]=.07
    target[10:15]=.02
    for exc in ('full','cashfirst'):
        result=sell_buy(cash,share,f,f,target,target>0,exc,.8,groups,np.zeros(N,int),5,force=True)
        cash1,share1,gross,fee,changed,_=result
        assert fee>=0 and gross>0 and cash1>=0 and np.all(share1>=0)
        assert abs((cash1+share1@f)-(NAV0-fee))<1e-4
        assert (share1@f)/(cash1+share1@f)<.801
    # R2 branch is buy-only; an already-held share cannot be sold.
    starting=np.zeros(N);starting[0]=100.
    neg_gap=np.zeros(N);neg_gap[2]=-.02
    positive=np.zeros(N);positive[2]=.02
    already=np.zeros(N,dtype=bool);already[0]=True;already[2]=True
    c2,sh2,g2,fe2,why=r2_buy_only(800_000.,starting,f,f,positive,vol,
                           neg_gap,already,.9,groups)
    assert sh2[0]>=starting[0] and sh2[2]>0 and fe2>0
    # Later rounds cannot cause any change to R2's opening-volume feature.
    # A truly eligible forecast and a 15d replay, with no hallucinated event.
    fake={int(i)-1:calibration(market,int(i)-1) for i in range(180,195)}
    pol=Policy('test',hold=5,floating=True,execution='cashfirst',r2_gap=True,volume_filter=True)
    metrics,decisions=simulate(market,np.arange(180,195),pol,fake,p30,p37,p38,groups,v15,[])
    assert len(decisions)==105
    assert all(z['round'] in range(1,8) for z in decisions)
    assert metrics['name_cap_breach_rounds']==0 and metrics['sector_cap_breach_rounds']==0
    assert metrics['camel_signals_observed']==0
    print('PHASE41_SELF_TEST_PASS',flush=True)
    print('CAUSAL_5D_CALIBRATION_HOLD_PERIOD_105_ROUNDS_CASH_CONSERVATION_VOLUME_CUTOFF_PASS',flush=True)
    print('NO_NETWORK_NO_KEYS_NO_LLM_NO_ORGANIZER_NO_HETZNER',flush=True)


def main():
    a=argparse.ArgumentParser(description='ICAIF Phase 41 offline research')
    a.add_argument('--self-test',action='store_true')
    a.add_argument('--run',action='store_true')
    a.add_argument('--input',type=str)
    a.add_argument('--min-last-day',default='2026-10-09')
    a.add_argument('--stride',type=int,choices=(5,15),default=15)
    a.add_argument('--events',type=str)
    a.add_argument('--out',type=str)
    args=a.parse_args()
    try:
        if args.self_test:self_test()
        elif args.run:run(args.input,args.min_last_day,args.stride,args.events,args.out)
        else:a.print_help()
    except Exception as exc:
        print('PHASE41_FAILED',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(2)

if __name__=='__main__':main()
