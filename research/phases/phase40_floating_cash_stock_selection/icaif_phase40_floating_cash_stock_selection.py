#!/usr/bin/env python3
"""ICAIF Phase 40: independent 15-session x 105-slot research, NO submission.

Per-stock eligibility => variable number of holdings; exposure budget varies
70-90%; cash-first vs fully funded rebalancing; fixed feature hypotheses.
Historical CaMeL variants are identically inactive without *contemporaneous*
verified outputs. Uses existing Phase 30 v2 / 37 / 38 / 39 v2 machinery.

This is NOT a fitted/validated trading strategy. Fees 0.1% both sides,
IEX open-bar proxies, no slippage/spread/corporate action adjustment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
import pandas as pd
import importlib.util
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd()
OUT=Path('artifacts/phase40_floating_portfolio')
N=30
FEE=.001
NAV0=1_000_000.
MAX_NAME=.10  # diversification constraint, NOT a holding count target
MAX_SECTOR=.30
MIN_EQUITY=.70
MAX_EQUITY=.90
ENTRY_THRESHOLD=-.50
KEEP_THRESHOLD=-.85
MIN_DELTA_WEIGHT=.007
FULL_REBALANCE_THRESHOLD=.050
MIN_TRADE_USD=500.

@dataclass(frozen=True)
class Policy:
    name: str
    factor: str=''
    execution: str='cashfirst'  # cashfirst or full
    floating: bool=True
    intraday: bool=False
    camel: bool=False


def policies():
    return (
        Policy('equal60_hold'),Policy('equal80_hold'),
        Policy('blend80_hold'),Policy('lowvol80_legacy_hold'),
        Policy('momentum5_cashfirst_floating','momentum'),
        Policy('reversal1_cashfirst_floating','reversal'),
        Policy('voladjusted_cashfirst_floating','voladjusted'),
        Policy('hybrid_cashfirst_fixed80','hybrid',floating=False),
        Policy('hybrid_cashfirst_floating','hybrid'),
        Policy('hybrid_fullrebalance_floating','hybrid',execution='full'),
        Policy('hybrid_intraday_cashfirst_floating','hybrid',intraday=True),
        Policy('hybrid_intraday_cashfirst_floating__camel','hybrid',intraday=True,camel=True),
    )


def dependencies():
    files=(('p40p39','icaif_phase39_105round_walkforward_v2.py'),)
    found=[]
    for name, filename in files:
        file=HERE/filename
        if not file.is_file():raise FileNotFoundError('Download and place next to Phase40: '+filename)
        spec=importlib.util.spec_from_file_location(name,file)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod;spec.loader.exec_module(mod);found.append(mod)
    p39=found[0];p30,p37,p38=p39.dependencies()
    return p39,p30,p37,p38


def z(x):
    x=np.asarray(x,float)
    s=float(x.std())
    return np.clip((x-x.mean())/max(s,1e-9),-3,3)


def factors(market,i,r,p30):
    """As-of prior close R1; only fully completed bars R2--R7."""
    t=i-1
    if t<21:raise ValueError('Insufficient causal lookback')
    past=np.log(market.daily[t-20:t+1])
    ret=np.diff(past,axis=0)
    v=np.maximum(ret.std(axis=0,ddof=1),.004)
    r1=past[-1]-past[-2];r5=past[-1]-past[-6];r20=past[-1]-past[0]
    feats={
        'momentum':z(.70*r5+.30*r20),
        'reversal':z(-.70*r1-.30*r5),
        'voladjusted':z((.60*r5+.40*r20)/v)+.20*z(-v),
        'hybrid':z(.48*z(r20)+.32*z(r5)-.25*z(r1)-.15*z(v)),
    }
    if r>0:
        k=p30.KNOWN_IDX[r]
        if k is None:raise AssertionError('No completed bar')
        within=np.log(market.cl[i,k,:]/market.daily[t])
        feats['hybrid_intraday']=z(feats['hybrid']+.35*z(within))
    else:feats['hybrid_intraday']=feats['hybrid'].copy()
    basket20=float(np.mean(r20))
    basketvol=float(np.std(ret.mean(axis=1),ddof=1))
    return feats,v,basket20,basketvol


def equity_budget(floating,basket20,basketvol):
    if not floating:return .80
    # Predeclared volatility-aware exposure response; no tuning to test outcomes.
    value=.80+.10*math.tanh(basket20/.035)-.025*min(1.,max(0.,basketvol/.014-1.))
    return float(np.clip(value,MIN_EQUITY,MAX_EQUITY))


def sector_groups(p30,p38):
    names=list(p30.SYMBOLS)
    groups=[]
    for sec in p38.SECTOR_MEMBERS:
        groups.append(np.array([names.index(s) for s in sec],int))
    if sorted(int(i) for g in groups for i in g)!=list(range(N)):
        raise AssertionError('Sector taxonomy not a partition')
    return groups


def allocate(scores,vol,held,exposure,groups):
    """Each symbol is independently eligible; no top-K count or target N.

    Existing holdings use a weaker exit threshold than new candidates (hysteresis).
    Capped pro-rata risk weights are repeatedly redistributed within the
    independently eligible set. A lack of eligible capacity is transparently
    reported as a band shortfall, NOT hidden by purchasing rejected stocks.
    """
    scores=np.asarray(scores,float);held=np.asarray(held,bool)
    eligible=np.where(held,scores>=KEEP_THRESHOLD,scores>=ENTRY_THRESHOLD)
    desired=np.zeros(N,float)
    priority=np.maximum(scores+1.35,.08)*np.power(.012/np.maximum(vol,.004),.25)
    priority=np.where(eligible,priority,0.)
    budget=float(exposure)
    for _ in range(160):
        left=budget-desired.sum()
        if left<1e-9:break
        avail=np.maximum(0.,MAX_NAME-desired)
        for group in groups:
            group_left=max(0.,MAX_SECTOR-desired[group].sum())
            # Use an allocation limit shared across the sector. Excess is
            # redistributed next iteration, not quietly concentrated.
            if avail[group].sum()>group_left and avail[group].sum()>0:
                avail[group]*=group_left/avail[group].sum()
        avail=np.where(eligible,avail,0.)
        take=(avail>1e-10)
        if not take.any():break
        prop=np.where(take,priority,0.)
        if prop.sum()<1e-12:break
        incr=left*prop/prop.sum()
        incr=np.minimum(incr,avail)
        # Enforce aggregate sector cap on proposed increments.
        for group in groups:
            rem=max(0.,MAX_SECTOR-desired[group].sum())
            if incr[group].sum()>rem and incr[group].sum()>0:
                incr[group]*=rem/incr[group].sum()
        if incr.sum()<1e-10:break
        desired+=incr
    if desired.max()>MAX_NAME+1e-9 or any(desired[g].sum()>MAX_SECTOR+1e-8 for g in groups):
        raise AssertionError('Risk allocation failed')
    return desired,eligible, max(0.,budget-desired.sum())


def fee_cash_trade(cash,shares,mark,fill,desired,score,execution,day0=False):
    """Target established at deadline-mark, fills modeled at following open.

    Cash-first: only sell failed eligibility/position-cap excess or when broad
    exposure has to be reduced. Purchases draw on real cash, not forced sales
    of an unrelated winner. Full mode is a controlled fee-inclusive ablation.
    """
    if day0:
        # For the initial acquisition use the exact previously validated
        # fee-adjusted target-weight trade, provided via the caller.
        raise ValueError('Use initial_target() for day-zero entry')
    pre_mark=cash+float(shares@mark)
    curr=shares*mark/max(pre_mark,1.)
    gap=desired-curr
    if execution=='full':
        if float(np.abs(gap).sum())<FULL_REBALANCE_THRESHOLD:
            return cash,shares.copy(),0.,0.,'HOLD_FULL_DEADBAND'
        # Executable target rebalancing uses the caller's existing p30.trade.
        return None,None,None,None,'FULL_TARGET_ORDER'
    if execution!='cashfirst':raise ValueError('Bad execution')
    # Calculate sizes at last known marks; never inspect fill when choosing
    # whether/how much a stock deserves. Fill is used only to settle orders.
    sellval=np.zeros(N)
    for j in range(N):
        if shares[j]<=1e-12:continue
        if desired[j]<1e-9:
            sellval[j]=shares[j]*mark[j]
        elif curr[j]>MAX_NAME-.005:
            sellval[j]=max(0.,curr[j]-(MAX_NAME-.012))*pre_mark
        elif curr.sum()>desired.sum()+.025 and score[j]<0:
            sellval[j]=max(0.,curr[j]-desired[j])*pre_mark
    sellval=np.where(sellval>=MIN_DELTA_WEIGHT*pre_mark,sellval,0.)
    qty_sell=np.minimum(shares,sellval/mark)
    post_shares=shares-qty_sell
    sell_n=float(qty_sell@fill)
    post_cash=cash+sell_n*(1-FEE)
    after_sell_nav=post_cash+float(post_shares@fill)
    post_weight=post_shares*mark/max(post_cash+float(post_shares@mark),1.)
    buy_gap=np.maximum(0.,desired-post_weight)
    buyval=np.where(buy_gap>=MIN_DELTA_WEIGHT,buy_gap*pre_mark,0.)
    # Desired is precommitted at decision time; do not use fill to re-rank.
    # Reserve 10% liquid cash, avoiding >90% target exposure at execution.
    allowed=max(0.,post_cash-.10*after_sell_nav)
    demand=float(buyval.sum())
    if demand>0 and allowed>0:
        buyval*=min(1.,allowed/max(demand*(1+FEE),1e-12))
    else:buyval[:]=0.
    qty_buy=buyval/mark
    # Actual fill may gap relative to predecision marks: scale quantity only
    # for financing/risk, never change its cross-sectional ordering.
    buy_n=float(qty_buy@fill)
    actual_budget=max(0.,post_cash-.10*after_sell_nav)
    if buy_n*(1+FEE)>actual_budget and buy_n>0:
        scale=actual_budget/(buy_n*(1+FEE));qty_buy*=scale;buy_n=float(qty_buy@fill)
    result_cash=post_cash-buy_n*(1+FEE)
    result_shares=post_shares+qty_buy
    gross=sell_n+buy_n
    if gross<MIN_TRADE_USD:return cash,shares.copy(),0.,0.,'HOLD_MIN_NOTIONAL'
    if result_cash<-1e-5 or result_shares.min()<-1e-12:
        raise AssertionError('Long-only/cash violation')
    fee=FEE*gross
    old_nav=cash+shares@fill
    new_nav=result_cash+result_shares@fill
    if abs((old_nav-new_nav)-fee)>1e-5:raise AssertionError('Fees not conserved')
    return float(max(0.,result_cash)),result_shares,gross,fee,'CASH_FIRST'


def initial_target(cash,shares,fill,target,p30):
    new_cash,new_shares,gross=p30.trade(cash,shares,fill,target)
    fee=FEE*gross
    if abs((cash+shares@fill)-(new_cash+new_shares@fill)-fee)>1e-5:
        raise AssertionError('Initial fee accounting')
    return new_cash,new_shares,gross,fee


def event_adjust(cash,shares,mark,fill,events,used,date,round_index,p38,p30):
    """Paper-only timestamped LLM overlay; only after manual source review.

    For every event, use only a single +/-.003 stock position change. No
    historical LLM backfill and no trade if event would breach cash/risk.
    """
    if not events:return cash,shares,0.,0.,None
    at=p38.decision_time(date,round_index)
    available=[e for e in events if e['event_id'] not in used and e['_available']<=at and
               timedelta(0)<=at-e['_published']<=timedelta(hours=4)]
    if not available:return cash,shares,0.,0.,None
    e=available[-1];used.add(e['event_id'])
    idx=list(p30.SYMBOLS).index(e['symbol'])
    groups=sector_groups(p30,p38)
    issuer_group=next(g for g in groups if idx in g)
    nav=cash+shares@mark
    value=.003*nav
    if e['direction']=='positive':
        curr_w=shares[idx]*mark[idx]/max(nav,1.)
        equity=float(shares@mark)/max(nav,1.)
        group_weight=float(shares[issuer_group]@mark[issuer_group])/max(nav,1.)
        if (curr_w+.003>MAX_NAME or equity+.003>MAX_EQUITY or
                group_weight+.003>MAX_SECTOR or cash<value*1.002):
            return cash,shares,0.,0.,None
        qty=value/mark[idx]
        charge=qty*fill[idx]
        if cash<charge*(1+FEE):return cash,shares,0.,0.,None
        after=shares.copy();after[idx]+=qty
        return cash-charge*(1+FEE),after,charge,charge*FEE,e['event_id']
    if shares[idx]*mark[idx]<value:return cash,shares,0.,0.,None
    qty=min(shares[idx],value/mark[idx]);gross=qty*fill[idx]
    after=shares.copy();after[idx]-=qty
    return cash+gross*(1-FEE),after,gross,gross*FEE,e['event_id']


def simulate_dynamic(market,ix,policy,p30,p38,events,groups):
    if len(ix)!=15 or not np.all(np.diff(ix)==1):raise AssertionError('Bad window')
    cash=NAV0;shares=np.zeros(N);fees=0.;num_trades=0
    marks=[NAV0];turn=[];logs=[];used=set()
    target_count=[];actual_count=[];band_shortfalls=0;unfilled=[];out_of_band=0
    for day_idx,i in enumerate(ix):
        i=int(i)
        for r in range(7):
            mark=market.daily[i-1] if r==0 else market.cl[i,p30.KNOWN_IDX[r]]
            fill=market.op[i,p30.FILL_IDX[r]]
            before_cash=float(cash);before_shares=shares.copy()
            pre_nav=float(cash+shares@fill)
            equity_before=float(shares@mark/(cash+shares@mark))
            active=(day_idx==0 and r==0) or r==0 or policy.intraday
            reason='HOLD_NOT_SCHEDULED';gross=0.;fee=0.;event_id=None
            target=equity_before;eligible=np.zeros(N,bool);shortfall=0.
            score=np.zeros(N)
            if active:
                feat,vol,r20,volm=factors(market,i,r,p30)
                selected_factor='hybrid_intraday' if r>0 and policy.intraday else policy.factor
                score=feat[selected_factor]
                held=(shares*mark/(cash+shares@mark))>.0005
                target=equity_budget(policy.floating,r20,volm)
                desired,eligible,shortfall=allocate(score,vol,held,target,groups)
                if day_idx==0 and r==0:
                    cash,shares,gross,fee=initial_target(cash,shares,fill,desired,p30)
                    reason='INITIAL_DYNAMIC_SELECTION'
                elif policy.execution=='full':
                    if (abs(float(desired.sum()-equity_before))<.003 and
                            np.abs(desired-(shares*mark/(cash+shares@mark))).sum()<FULL_REBALANCE_THRESHOLD):
                        reason='HOLD_FULL_DEADBAND'
                    else:
                        cash,shares,gross= p30.trade(cash,shares,fill,desired)
                        fee=FEE*gross
                        reason='FULL_TARGET_REBALANCE'
                else:
                    cash,shares,gross,fee,reason=fee_cash_trade(
                        cash,shares,mark,fill,desired,score,policy.execution)
                if shortfall>1e-6:band_shortfalls+=1
            if policy.camel and events:
                cash,shares,evgross,evfee,event_id=event_adjust(
                    cash,shares,mark,fill,events,used,market.dates[i],r,p38,p30)
                gross+=evgross;fee+=evfee
            if gross>1e-8:num_trades+=1;fees+=fee
            post_nav=float(cash+shares@fill)
            if post_nav<=0 or cash<-1e-5 or np.min(shares)<-1e-10:
                raise AssertionError('Invalid portfolio')
            if abs(post_nav-(pre_nav-fee))>1e-4:raise AssertionError('Fee accounting mismatch')
            expo=float(shares@fill/post_nav)
            w=shares*fill/post_nav
            if max(w)>MAX_NAME+.03 or expo>MAX_EQUITY+.03:
                raise AssertionError('Price gap/risk limit breach; inspect')
            if (expo<MIN_EQUITY-1e-4 or expo>MAX_EQUITY+1e-4):
                out_of_band+=1
            val=market.op[i,p30.END_VAL_IDX[r]] if r<6 else market.cl[i,25]
            end_nav=float(cash+shares@val)
            marks.append(end_nav)
            turn.append(gross/max(pre_nav,1.))
            act=int(np.sum(shares*fill/post_nav>.0005))
            actual_count.append(act);target_count.append(int(eligible.sum()))
            unfilled.append(shortfall)
            logs.append({'date':str(market.dates[i].date()),'round':r+1,
                  'deadline_et':p38.DEADLINE_ET[r],'execution_et':p38.FILL_ET[r],
                  'information_cutoff':'prior_close' if r==0 else p30.SLOTS[p30.KNOWN_IDX[r]],
                  'strategy':policy.name,'trade':int(gross>1e-8),'decision':reason,
                  'gross_notional_usd':gross,'fee_usd':fee,'equity_exposure':expo,
                  'cash_weight':cash/post_nav,'holdings':act,
                  'held_symbols':';'.join(p30.SYMBOLS[j] for j in np.where(w>.0005)[0]),
                  'eligible_symbols':int(eligible.sum()) if active else None,
                  'requested_exposure':target if active else None,
                  'unfilled_target_exposure':shortfall if active else None,
                  'event_id':event_id,'nav_end':end_nav})
    if len(logs)!=105:raise AssertionError('Must evaluate 105 slots')
    nav=np.asarray(marks);rr=nav[1:]/nav[:-1]-1
    sharpe=0. if rr.std(ddof=1)<1e-12 else float(rr.mean()*np.sqrt(252*7)/rr.std(ddof=1))
    metrics={'cumulative_return':float(nav[-1]/NAV0-1),
       'sharpe_proxy':sharpe,
       'mdd_proxy':float(np.max(1-nav/np.maximum.accumulate(nav))),
       'turnover_proxy':float(np.mean(turn)),
       'fees_usd':fees,'trade_count':num_trades,
       'camel_event_trades':int(sum(x['event_id'] is not None for x in logs)),
       'avg_exposure':float(np.mean([x['equity_exposure'] for x in logs])),
       'min_exposure':float(np.min([x['equity_exposure'] for x in logs])),
       'max_exposure':float(np.max([x['equity_exposure'] for x in logs])),
       'mean_holdings':float(np.mean(actual_count)),
       'min_holdings':int(np.min(actual_count)), 'max_holdings':int(np.max(actual_count)),
       'eligible_shortfall_rounds':band_shortfalls,
       'realized_exposure_outside_band_rounds':out_of_band}
    return metrics,logs


def factor_ic(market,episodes,p30):
    """Diagnostics only: scores computed at prior close, labels realized later.

    1h = R1 to R2 fill, day = R1 to same-day close, five-session =
    R1 through fifth later close. These are three *different* horizons.
    No future label ever enters the trading strategy.
    """
    rec=[]
    seen=set()
    for eid,seg,ix in episodes:
        for local,i in enumerate(ix):
            i=int(i)
            if i in seen:continue
            seen.add(i)
            features,_,_,_=factors(market,i,0,p30)
            y={'1hour':np.log(market.op[i,p30.FILL_IDX[1]]/market.op[i,p30.FILL_IDX[0]]),
               'same_day_close':np.log(market.daily[i]/market.op[i,p30.FILL_IDX[0]])}
            if local+4<len(ix):
                y['5sessions']=np.log(market.daily[i+4]/market.op[i,p30.FILL_IDX[0]])
            for k,x in features.items():
                if k=='hybrid_intraday':continue
                for horizon,label in y.items():
                    corr=pd.Series(x).corr(pd.Series(label),method='spearman')
                    rec.append({'segment':seg,'decision_day':str(market.dates[i].date()),
                                'factor':k,'horizon':horizon,'spearman_ic':corr})
    return pd.DataFrame(rec)


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def run(csv_path=None,stride=15,events_file=None):
    p39,p30,p37,p38=dependencies()
    src=Path(csv_path).expanduser() if csv_path else ROOT/p30.INPUT
    if not src.is_absolute():src=ROOT/src
    src=src.resolve()
    market=p30.read_market(src,min_last_day='2026-10-09')
    events=p39.read_events(events_file,p38,p30)
    groups=sector_groups(p30,p38)
    episodes=p39.episode_groups(market,stride=stride,minimum_history=121)
    if not episodes:raise ValueError('No eligible complete 15-session episodes')
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=True)
    res=[];dec=[]
    print('PHASE40_ASOF',market.dates[-1].date(),'EPISODES',len(episodes),
        'POLICIES',len(policies()),'ROUNDS_PER_EPISODE',105,flush=True)
    for k,(eid,segment,ix) in enumerate(episodes,1):
        for pol in policies():
            if not pol.factor:
                core={'equal60_hold':'equal60','equal80_hold':'equal80',
                      'blend80_hold':'blend80','lowvol80_legacy_hold':'lowvol80_legacy'}[pol.name]
                stats,logs=p39.simulate(market,ix,p39.Strategy(pol.name,core),[],p30,p37,p38)
                m={**stats,'avg_exposure':float(np.mean([d['post_execution_equity_weight'] for d in logs])),
                   'min_exposure':min(d['post_execution_equity_weight'] for d in logs),
                   'max_exposure':max(d['post_execution_equity_weight'] for d in logs),
                   'mean_holdings':30.,'min_holdings':30,'max_holdings':30,
                   'eligible_shortfall_rounds':0,
                   'realized_exposure_outside_band_rounds':0}
                rows=[{'date':d['day'],'round':d['round'],'deadline_et':d['deadline_et'],
                      'execution_et':d['execution_et'],'information_cutoff':d['info_cutoff'],
                      'strategy':pol.name,'trade':int(d['decision']=='TRADE'),
                      'decision':d['reason'],'gross_notional_usd':d['gross_trade_usd'],
                      'fee_usd':d['fee_usd'],'equity_exposure':d['post_execution_equity_weight'],
                      'cash_weight':1-d['post_execution_equity_weight'],
                      'holdings':30,'held_symbols':';'.join(p30.SYMBOLS),
                      'eligible_symbols':None,'requested_exposure':None,
                      'unfilled_target_exposure':None,'event_id':None,'nav_end':d['nav_end']}
                      for d in logs]
            else:
                m,rows=simulate_dynamic(market,ix,pol,p30,p38,events,groups)
            res.append({'episode':eid,'segment':segment,'strategy':pol.name,**m})
            dec.extend([{'episode':eid,'segment':segment,**row} for row in rows])
        print('PHASE40_EPISODE',k,'OF',len(episodes),eid,'COMPLETE',flush=True)
    metrics=pd.DataFrame(res);actions=pd.DataFrame(dec)
    base=metrics.set_index(['episode','strategy'])
    paired=[]
    for _,line in metrics.iterrows():
        if line['strategy']=='equal80_hold':continue
        ref='hybrid_intraday_cashfirst_floating' if line['strategy'].endswith('__camel') else 'equal80_hold'
        other=base.loc[(line.episode,ref)]
        paired.append({'episode':line.episode,'segment':line.segment,'strategy':line.strategy,
                      'reference':ref,'delta_return_pp':100*(line.cumulative_return-other.cumulative_return),
                      'delta_sharpe':line.sharpe_proxy-other.sharpe_proxy,
                      'delta_mdd_pp':100*(line.mdd_proxy-other.mdd_proxy),
                      'delta_turnover':line.turnover_proxy-other.turnover_proxy,
                      'delta_fees_usd':line.fees_usd-other.fees_usd,
                      'delta_mean_holdings':line.mean_holdings-other.mean_holdings})
    pair=pd.DataFrame(paired)
    policy_summary=(metrics.groupby(['segment','strategy'],as_index=False)
         .agg(episodes=('episode','size'),mean_return=('cumulative_return','mean'),
              median_return=('cumulative_return','median'),mean_sharpe=('sharpe_proxy','mean'),
              mean_mdd=('mdd_proxy','mean'),mean_turnover=('turnover_proxy','mean'),
              mean_fees_usd=('fees_usd','mean'),mean_trade_count=('trade_count','mean'),
              avg_equity_exposure=('avg_exposure','mean'),avg_holdings=('mean_holdings','mean'),
              min_holdings=('min_holdings','min'),max_holdings=('max_holdings','max'),
              min_equity_exposure=('min_exposure','min'),max_equity_exposure=('max_exposure','max'),
              eligibility_shortfall_rounds=('eligible_shortfall_rounds','sum'),
              realized_exposure_outside_band_rounds=('realized_exposure_outside_band_rounds','sum')))
    pair_summary=(pair.groupby(['segment','strategy','reference'],as_index=False)
        .agg(episodes=('episode','size'),mean_delta_return_pp=('delta_return_pp','mean'),
             median_delta_return_pp=('delta_return_pp','median'),
             return_win_fraction=('delta_return_pp',lambda x:float((x>0).mean())),
             mean_delta_sharpe=('delta_sharpe','mean'),
             mean_delta_mdd_pp=('delta_mdd_pp','mean'),
             mean_delta_turnover=('delta_turnover','mean'),
             mean_extra_fee_usd=('delta_fees_usd','mean')))
    activity=(actions.groupby(['segment','strategy','round'],as_index=False)
        .agg(slots=('trade','size'),executed_rounds=('trade','sum'),
             fee_usd=('fee_usd','sum'),gross_notional_usd=('gross_notional_usd','sum'),
             mean_holdings=('holdings','mean'),mean_exposure=('equity_exposure','mean')))
    ic=factor_ic(market,episodes,p30)
    for name,df in [('phase40_episode_metrics.csv',metrics),
                    ('phase40_policy_summary.csv',policy_summary),
                    ('phase40_paired_episode_deltas.csv',pair),
                    ('phase40_paired_summary.csv',pair_summary),
                    ('phase40_round_activity.csv',activity),
                    ('phase40_all_105_round_actions.csv',actions),
                    ('phase40_feature_ic.csv',ic)]:df.to_csv(out/name,index=False)
    manifest={'research_only':True,'no_llm_provider_calls':True,'no_organizer_api_calls':True,
              'no_official_deployment':True,'source_sha256':sha256(src),
              'data_end':str(market.dates[-1].date()),'episodes':len(episodes),
              'trading_sessions_per_episode':15,'decision_rounds_per_episode':105,
              'policies':len(policies()),'stride':stride,
              'fixed_target_holding_counts':False,
              'emergent_holding_count_from_individual_stock_gates':True,
              'dynamic_exposure_band':[MIN_EQUITY,MAX_EQUITY],
              'name_cap':MAX_NAME,'sector_cap':MAX_SECTOR,
              'min_exposure_not_forced_when_no_eligible_stocks':True,
              'known_limitations':['Alpaca IEX open/close price execution proxy and unadjusted corporate actions',
                   '15-day Sharpe annualization is noisy','No untouched 2026 holdout after extensive hypothesis testing',
                   'No expected-return calibration, so heuristics are hypotheses only',
                   'Incomplete US trading sessions skipped by Phase30 loader',
                   'CaMeL outcomes cannot be backdated; zero historical event trades is normal',
                   'Exposure may drift outside planning band between decisions due to price movements',
                   'Sector taxonomy is manual and needs formal auditing',
                   'Cross-sectional IC is exploratory, not independently validated']}
    (out/'phase40_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if policy_summary['realized_exposure_outside_band_rounds'].sum()>0:
        print('PHASE40_EXPOSURE_BAND_DIAGNOSTIC_CHECK_POLICY_SUMMARY',
              int(policy_summary['realized_exposure_outside_band_rounds'].sum()),flush=True)
    print('PHASE40_DATA_DIAGNOSTIC_FACTOR_IC_ROWS',len(ic),'EVENTS',len(events),flush=True)
    print('PHASE40_EPISODE_OUTCOME_DIR',out.resolve(),flush=True)
    print('PHASE40_RESEARCH_COMPLETE_NO_LLM_NO_SUBMISSION',flush=True)
    return metrics,policy_summary,pair,actions,ic


def self_test():
    from types import SimpleNamespace
    rng=np.random.default_rng(400011)
    days=265
    daily=100*np.exp(np.cumsum(rng.normal(0,.009,(days,N)),axis=0))
    intra=rng.normal(0,.00085,(days,26,N))
    op=daily[:,None,:]*np.exp(np.cumsum(intra,axis=1));cl=op*np.exp(.00008)
    dates=pd.bdate_range('2025-09-02',periods=days)
    market=SimpleNamespace(dates=dates,op=op,cl=cl,daily=cl[:,-1,:],missing=pd.DataFrame(columns=['date']))
    p39,p30,p37,p38=dependencies()
    groups=sector_groups(p30,p38)
    ix=np.arange(138,153)
    for pol in policies():
        if not pol.factor:continue
        m,logs=simulate_dynamic(market,ix,pol,p30,p38,[],groups)
        assert len(logs)==105
        assert np.isfinite(m['cumulative_return']) and m['fees_usd']>=0
        assert all(x['holdings']<=N and x['holdings']>=0 for x in logs)
        if pol.camel:
            base=next(p for p in policies() if p.name==pol.name.removesuffix('__camel'))
            normal,_=simulate_dynamic(market,ix,base,p30,p38,[],groups)
            for field in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd'):
                assert abs(normal[field]-m[field])<1e-11
    # Signal never changes when a future intraday bar is modified.
    before=factors(market,138,1,p30)[0]['hybrid_intraday']
    copy=SimpleNamespace(dates=dates,op=op.copy(),cl=cl.copy(),daily=market.daily.copy())
    copy.cl[138,p30.KNOWN_IDX[1]+1:,:]*=3
    after=factors(copy,138,1,p30)[0]['hybrid_intraday']
    assert np.array_equal(before,after)
    # Stock count not prescribed: changing ONE eligibility input can change N.
    scores=np.linspace(-1.5,1.5,N);vol=np.full(N,.015)
    a,_,_=allocate(scores,vol,np.zeros(N,bool),.8,groups)
    scores[0]=+2.
    b,_,_=allocate(scores,vol,np.zeros(N,bool),.8,groups)
    assert (a>1e-9).sum()!=(b>1e-9).sum()
    # Low-eligibility failure must be surfaced rather than stuffing the
    # portfolio with names that failed the per-stock signal.
    scarce=np.full(N,-3.);scarce[0]=2.
    scarce_weights,scarce_eligible,scarce_gap=allocate(scarce,vol,np.zeros(N,bool),.8,groups)
    assert int(scarce_eligible.sum())==1 and scarce_gap>.6
    assert (scarce_weights>1e-9).sum()==1
    # Profit/loss and cash fee conservation of cash-first order.
    mark=np.full(N,100.);fill=mark*1.001
    cash=250_000.;shares=np.full(N,200.)
    w=np.full(N,.8/N);score=np.zeros(N)
    result=fee_cash_trade(cash,shares,mark,fill,w,score,'cashfirst')
    assert result[0] is not None and result[0]>=-1e-6
    print('PHASE40_ELIGIBILITY_SHORTFALL_AND_NO_FORCED_WEAK_STOCK_PURCHASE_PASS')
    print('PHASE40_SELF_TEST_PASS')
    print('105_ROUNDS_15_SESSIONS_FLOATING_HOLDINGS_70_90_CASH_FIRST_FEE_AND_ASOF_GUARDS_PASS')
    print('NO_NETWORK_NO_LLM_CREDITS_NO_HETZNER_NO_ORGANIZER_SUBMISSION')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--stride',type=int,default=15,choices=(5,15))
    ap.add_argument('--input',default=None)
    ap.add_argument('--events',default=None,help='Only contemporaneously verified source+CaMeL events')
    args=ap.parse_args()
    try:
        if args.self_test:self_test()
        elif args.run:run(args.input,args.stride,args.events)
        else:ap.print_help()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('PHASE40_FAILED',type(e).__name__,str(e),file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
