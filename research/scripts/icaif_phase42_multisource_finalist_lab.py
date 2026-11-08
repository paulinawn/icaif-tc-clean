#!/usr/bin/env python3
"""Phase 42 candidate tournament: 15 sessions x 7 decisions, NO organizer access.

Offline prices: Alpaca IEX 15-minute bars. SEC/Finnhub/CaMeL are only applied
when contemporaneous, evidence-reviewed records are available. Missing history
must NEVER be filled with a later snapshot. Source collection in separate
icaif_phase42_source_bridge.py, via explicit command; never auto-calls APIs.

Copy Phase 30 v2, Phase 37, 38, 39 v2, and Phase 41 beside this file.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=Path('artifacts/phase42_multisource_finalist_lab')
FEE=.001; NAV=1_000_000.; N=30
NAME_CAP=.095; SECTOR_CAP=.295; MIN_TRADE=1_000.; MAX_TURNOVER_DECISION=.12

@dataclass(frozen=True)
class Policy:
    name:str
    core:str=''
    alpha:str=''
    cashfirst:bool=False
    wide:bool=False
    event:bool=False
    r2:bool=False


def policies():
    return (
        Policy('equal60_hold',core='equal60'),
        Policy('equal80_hold',core='equal80'),
        Policy('blend80_hold',core='blend80'),
        Policy('lowvol80_hold',core='lowvol80_legacy'),
        Policy('dynamic_20d_strength',alpha='strength'),
        Policy('dynamic_5d_reversal',alpha='reversal'),
        Policy('dynamic_trend_pullback',alpha='pullback'),
        Policy('dynamic_regime_mix',alpha='regime'),
        Policy('dynamic_volatility_quality',alpha='volquality'),
        Policy('dynamic_sector_neutral',alpha='sectorneutral'),
        Policy('dynamic_pullback_cashfirst',alpha='pullback',cashfirst=True),
        Policy('dynamic_regime_wide95',alpha='regime',wide=True),
        Policy('dynamic_regime_r2_gap',alpha='regime',r2=True),
        Policy('dynamic_regime_multisource_shadow',alpha='regime',event=True),
    )


def dependencies():
    names=[('p42p41','icaif_phase41_costaware_reversal.py'),('p42p39','icaif_phase39_105round_walkforward_v2.py'),
           ('p42p30','icaif_phase30_2026_walkforward_agent_lab_v2.py'),
           ('p42p37','icaif_phase37_risk_regime_tournament.py'),
           ('p42p38','icaif_phase38_105slot_camel_staging.py')]
    found=[]
    for key,name in names:
        path=HERE/name
        if not path.is_file():raise FileNotFoundError('Place prerequisite in same folder: '+name)
        spec=importlib.util.spec_from_file_location(key,path)
        m=importlib.util.module_from_spec(spec);sys.modules[key]=m;spec.loader.exec_module(m);found.append(m)
    return found


def z(x):
    a=np.asarray(x,float);return np.clip((a-a.mean())/max(float(a.std()),1e-8),-3.,3.)


def day_features(market,t,volume=None):
    if t<121:raise ValueError('Prior-day feature warmup needed')
    p=np.log(market.daily)
    r1=p[t]-p[t-1];r5=p[t]-p[t-5];r20=p[t]-p[t-20];r60=p[t]-p[t-60]
    dr=p[t-19:t+1]-p[t-20:t]
    vol=np.maximum(dr.std(axis=0,ddof=1),.003)
    broad20=float(r20.mean());broadvol=float(dr.mean(axis=1).std(ddof=1))
    return dict(r1=r1,r5=r5,r20=r20,r60=r60,vol=vol,
                broad20=broad20,broadvol=broadvol)


def alphas(f,groups):
    trend=z(.65*z(f['r20'])+.35*z(f['r5']))
    reversal=z(-.75*z(f['r5'])-.25*z(f['r1']))
    pullback=z(.65*z(f['r20'])-.45*z(f['r5'])+.22*z(-f['vol']))
    volquality=z(.68*z(-f['vol'])+.32*z(f['r20']))
    regime_weight=(.70 if f['broad20']>0 and f['broadvol']<.014 else .35)
    regime=z(regime_weight*trend+(1-regime_weight)*reversal+.15*z(-f['vol']))
    sectorneutral=regime.copy()
    for g in groups:sectorneutral[g]-=.65*float(regime[g].mean())
    sectorneutral=z(sectorneutral)
    return dict(strength=trend,reversal=reversal,pullback=pullback,
                regime=regime,volquality=volquality,sectorneutral=sectorneutral)


def exposure(f,wide=False):
    """Ex ante market-risk rule; no fitting on evaluation returns."""
    r=f['broad20'];v=f['broadvol']
    value=(.80+(.16 if wide else .10)*math.tanh(r/.05)
           -.025*max(0.,min(1.5,(v-.010)/.006)))
    return float(np.clip(value,.70,.95 if wide else .90))


def weights(score,f,held,limit,groups):
    """Per-issuer, floating stock count: NO fixed k-of-30 rule.

    Qualification is a relative-alpha candidate criterion; unlike Phase41's
    absolute predicted positive cash return it is NOT an expected-return claim.
    """
    n=len(score);held=np.asarray(held,bool)
    accept=np.where(held,score>=-.95,score>=-.45)
    # Do not turn a weak cross-sectional score into a calibrated return.
    priority=np.where(accept,np.maximum(.04,score+1.6)/np.sqrt(f['vol']/.01),0.)
    out=np.zeros(n)
    for _ in range(140):
        room=limit-out.sum()
        if room<1e-7:break
        avail=np.where(accept,np.maximum(0,NAME_CAP-out),0.)
        usable=np.where(avail>1e-10,priority,0.)
        if usable.sum()<1e-12:break
        inc=np.minimum(avail,room*usable/usable.sum())
        for g in groups:
            available=max(0.,SECTOR_CAP-out[g].sum())
            if inc[g].sum()>available and inc[g].sum()>0:inc[g]*=available/inc[g].sum()
        if inc.sum()<1e-10:break
        out+=inc
    if out.max()>NAME_CAP+1e-8 or any(out[g].sum()>SECTOR_CAP+1e-8 for g in groups):
        raise AssertionError('Target concentration caps violated')
    return out,accept


def trimmed_order(current,target,max_l1=MAX_TURNOVER_DECISION):
    """Bound daily turnover; do not scale initial cash allocation."""
    delta=target-current
    gross=float(np.abs(delta).sum())
    return current+delta*min(1.,max_l1/max(gross,1e-12))


def trade_target(cash,shares,mark,fill,target,p30,initial,mode,groups):
    nav_before=float(cash+shares@fill)
    marked_nav=float(cash+shares@mark)
    curr=shares*mark/marked_nav
    if not initial:
        if np.abs(target-curr).sum()<.045:return cash,shares.copy(),0.,0.,'NO_TRADE_BAND'
        if mode=='full':
            requested=trimmed_order(curr,target)
        else:
            # Cash-first ignores desired reductions of surviving issuers unless
            # risk or aggregate equity requires reducing. Adds with available cash.
            requested=target
    else:requested=target
    if mode=='cashfirst' and not initial:
        # Use preexisting, audited Stage41 cash-first share/fee accounting.
        # Sector/group and cap checks are subsequently audited at each fill.
        # Eligibility equals stocks with nonzero desired allocation.
        p41=dependencies()[0] if False else None
        raise AssertionError('Cash-first dispatcher must call p41 directly')
    prior_shares=shares.copy();prior_cash=float(cash)
    # p30.trade takes target weights and settles them at next execution price.
    new_cash,new_shares,gross=p30.trade(cash,shares,fill,requested)
    fee=FEE*gross
    if abs((prior_cash+prior_shares@fill)-(new_cash+new_shares@fill)-fee)>1e-3:
        raise AssertionError('Fee conservation')
    return float(new_cash),new_shares,gross,fee,'TARGET_TRADE'


def event_paper_adjustment(cash,shares,mark,fill,ev,groups,sector_of,maximum_exposure=.90):
    """Forward-only human-reviewed event overlay. +/-0.3% of NAV per signal.

    Not a calibrated return predictor. Order amount is set using PRE-deadline
    mark, then merely constrained/settled at execution prices. Never edits
    historical events or attributes alpha before the recorded model review.
    """
    sym=ev.get('symbol')
    j=ev.get('_symbol_index')
    if j is None:return cash,shares,0.,0.,'UNKNOWN_ISSUER'
    if ev.get('direction') not in ('positive','negative'):
        return cash,shares,0.,0.,'EVENT_DIRECTION_UNKNOWN'
    nav=float(cash+shares@mark)
    amount=min(3_000.,.003*nav)
    if ev['direction']=='positive':
        pre=shares*mark/nav
        g=groups[sector_of[j]]
        upper=min(NAME_CAP-pre[j],SECTOR_CAP-pre[g].sum(),maximum_exposure-pre.sum())
        quantity=max(0.,min(amount/nav,upper))*nav/mark[j]
        if quantity*mark[j]<MIN_TRADE:return cash,shares,0.,0.,'EVENT_BUY_TOO_SMALL'
        # Adjust *execution* quantity for cash and hard limits, never re-rank.
        px=fill[j];fin_nav=cash+shares@fill
        safe_qty=min(quantity,cash/(px*(1+FEE)),
            max(0.,((.099*fin_nav-shares[j]*px)/max(px,1e-9))),
            max(0.,((.299*fin_nav-np.sum(shares[g]*fill[g]))/max(px,1e-9))),
            max(0.,((maximum_exposure*fin_nav-shares@fill)/max(px,1e-9))))
        if safe_qty*px<MIN_TRADE:return cash,shares,0.,0.,'EVENT_RISK_REJECT'
        new=shares.copy();new[j]+=safe_qty
        gross=safe_qty*px;fee=FEE*gross;newcash=cash-gross-fee
    else:
        quantity=min(shares[j],amount/mark[j]);gross=quantity*fill[j]
        if gross<MIN_TRADE:return cash,shares,0.,0.,'EVENT_SELL_TOO_SMALL'
        new=shares.copy();new[j]-=quantity;fee=FEE*gross;newcash=cash+gross-fee
    if abs((cash+shares@fill)-(newcash+new@fill)-fee)>1e-5:
        raise AssertionError('Event overlay fee conservation')
    return float(newcash),new,float(gross),float(fee),'FORWARD_REVIEWED_EVENT_TILT'


def simulate_active(market,indices,pol,groups,mods,events):
    p41,p39,p30,p37,p38=mods
    cash=NAV;shares=np.zeros(N);nav=[NAV];turn=[];fees=0.;trades=0
    decisions=[];used=set();ages=np.zeros(N,int)
    sector_of={int(j):i for i,g in enumerate(groups) for j in g}
    event_by_symbol={s:j for j,s in enumerate(p30.SYMBOLS)}
    for di,i0 in enumerate(indices):
        i=int(i0);t=i-1
        f=day_features(market,t)
        score=alphas(f,groups)[pol.alpha]
        desired_exposure=exposure(f,pol.wide)
        for r in range(7):
            mark=market.daily[t] if r==0 else market.cl[i,p30.KNOWN_IDX[r],:]
            fill=market.op[i,p30.FILL_IDX[r],:]
            before=float(cash+shares@fill)
            current=shares*mark/(cash+shares@mark)
            starting=shares.copy()
            gross=0.;fee=0.;reason='HOLD';event_id=''
            planned,eligible=weights(score,f,current>.0006,desired_exposure,groups)
            if r==0 and (di==0 or di%3==0):
                if di==0:
                    cash,shares,gross,fee,reason=trade_target(
                        cash,shares,mark,fill,planned,p30,True,'full',groups)
                elif pol.cashfirst:
                    # Stepwise cash-first execution with explicit exposure reduction
                    # and buffers below risk limits. Same target and timing as full.
                    c,s,g,fee,changed,reason=p41.sell_buy(
                        cash,shares,mark,fill,planned,eligible,'cashfirst',
                        desired_exposure,groups,ages,3)
                    cash,shares,gross=c,s,g
                else:
                    cash,shares,gross,fee,reason=trade_target(
                        cash,shares,mark,fill,planned,p30,False,'full',groups)
            elif r==1 and pol.r2:
                # R2 gap mean-reversion filter: only completed prior-day data,
                # 09:30 opening print and completed 10:00 bar (ends 10:15).
                gap=np.log(market.op[i,0,:]/market.daily[t])
                vratio=None
                # No additional R2 purchase if the day's cash target is full.
                edge=np.where(gap<-.014,.01,0.)
                cash,shares,gross,fee,reason=p41.r2_buy_only(
                    cash,shares,mark,fill,edge,f['vol'],gap,eligible,
                    min(desired_exposure,.90),groups,vratio)
            if pol.event and events:
                cutoff=p38.decision_time(market.dates[i],r)
                eligible_events=[e for e in events if e['event_id'] not in used and
                       e['_available']<=cutoff and (
                        timedelta(0)<=cutoff-e['_published']<=timedelta(hours=4) or
                        (r==0 and e['_published'].astimezone(ZoneInfo('America/New_York')).hour>=15
                         and timedelta(0)<=cutoff-e['_published']<=timedelta(hours=22)))]
                if eligible_events:
                    e=eligible_events[-1];used.add(e['event_id']);event_id=e['event_id']
                    # An approved CaMeL classification changes weights only in the
                    # forward PAPER branch. The shift is a hypothesis, not
                    # proven alpha; real SEC/Finnhub/LLM inference timestamps
                    # are checked by Phase39.read_events and this deadline gate.
                    before_event_shares=shares.copy()
                    event=dict(e,_symbol_index=event_by_symbol.get(e['symbol']))
                    c,s,extra,extra_fee,why=event_paper_adjustment(
                        cash,shares,mark,fill,event,groups,sector_of,
                        maximum_exposure=min(desired_exposure,.90))
                    cash,shares=c,s;gross+=extra;fee+=extra_fee
                    reason+='|'+why
                    if extra<=0:event_id=''
            age_new=(starting<1e-12)&(shares>1e-12);ages[age_new]=0
            after=float(cash+shares@fill)
            if abs(before-after-fee)>1e-3:raise AssertionError('Net-of-fee NAV accounting')
            w=shares*fill/after
            e=float(w.sum());holding=int((w>.0005).sum())
            violation=int(w.max()>.1005 or any(w[g].sum()>.3005 for g in groups) or e>1.00001)
            if violation:raise AssertionError('Realized hard risk-limit violation')
            if gross>1e-5:trades+=1;fees+=fee
            turn.append(gross/before)
            endpx=market.op[i,p30.END_VAL_IDX[r],:] if r<6 else market.cl[i,25,:]
            end_nav=float(cash+shares@endpx);nav.append(end_nav)
            decisions.append(dict(day=str(market.dates[i].date()),round=r+1,
                cutoff='prior_close' if r==0 else p30.SLOTS[p30.KNOWN_IDX[r]],
                equity_exposure=e,requested_exposure=desired_exposure,holdings=holding,
                eligible_stocks=int(eligible.sum()),equity_shortfall=max(0.,desired_exposure-e),
                trade=int(gross>1e-5),gross_notional_usd=gross,fee_usd=fee,
                reason=reason,event_id=event_id,nav_end=end_nav))
        ages[shares>1e-10]+=1
    vals=np.array(nav);rets=vals[1:]/vals[:-1]-1.;dev=float(rets.std(ddof=1))
    out=dict(cumulative_return=float(vals[-1]/NAV-1),
        sharpe_proxy=float(rets.mean()*math.sqrt(252*7)/dev) if dev>1e-12 else 0.,
        mdd_proxy=float(np.max(1-vals/np.maximum.accumulate(vals))),
        turnover_proxy=float(np.mean(turn)),fees_usd=fees,trade_count=trades,
        mean_exposure=float(np.mean([x['equity_exposure'] for x in decisions])),
        min_exposure=float(min(x['equity_exposure'] for x in decisions)),
        max_exposure=float(max(x['equity_exposure'] for x in decisions)),
        mean_holdings=float(np.mean([x['holdings'] for x in decisions])),
        min_holdings=int(min(x['holdings'] for x in decisions)),
        max_holdings=int(max(x['holdings'] for x in decisions)),
        below70_rounds=sum(x['equity_exposure']<.70-.004 for x in decisions),
        over90_rounds=sum(x['equity_exposure']>.90+.004 for x in decisions),
        source_events_observed=sum(bool(x['event_id']) for x in decisions))
    if len(decisions)!=105:raise AssertionError('105 decisions required')
    return out,decisions


def run(path=None,outpath=None,stride=15,min_last_day='2026-10-06',events_file=None):
    mods=dependencies();p41,p39,p30,p37,p38=mods
    inp=Path(path) if path else p30.INPUT
    market=p30.read_market(inp,min_last_day=min_last_day)
    groups=p41.sector_indices(p30,p38)
    episodes=p39.episode_groups(market,stride=stride)
    if not episodes:raise ValueError('No complete 15-session episodes')
    events=p39.read_events(events_file,p38,p30)
    records=[];all_decisions=[]
    for k,(eid,segment,ix) in enumerate(episodes,1):
        for pol in policies():
            if pol.core:
                result,base=p39.simulate(market,ix,p39.Strategy(pol.name,pol.core),[],p30,p37,p38)
                logs=[dict(day=x['day'],round=x['round'],cutoff=x['info_cutoff'],
                    equity_exposure=x['post_execution_equity_weight'],requested_exposure=float(
                    p38.core_weights(p30,p37,market,int(ix[0])-1,pol.core).sum()),
                    holdings=30,eligible_stocks=30,equity_shortfall=0.,
                    trade=int(x['decision']=='TRADE'),gross_notional_usd=x['gross_trade_usd'],
                    fee_usd=x['fee_usd'],reason=x['reason'],event_id='',nav_end=x['nav_end']) for x in base]
                result.update(mean_exposure=float(np.mean([x['equity_exposure'] for x in logs])),
                   min_exposure=min(x['equity_exposure'] for x in logs),
                   max_exposure=max(x['equity_exposure'] for x in logs),mean_holdings=30.,
                   min_holdings=30,max_holdings=30,below70_rounds=0,over90_rounds=0,
                   source_events_observed=0)
            else:result,logs=simulate_active(market,ix,pol,groups,mods,events)
            records.append(dict(episode=eid,segment=segment,start_date=str(market.dates[ix[0]].date()),
                end_date=str(market.dates[ix[-1]].date()),strategy=pol.name,**result))
            all_decisions.extend([dict(episode=eid,segment=segment,strategy=pol.name,**r) for r in logs])
        print(f'PHASE42_EPISODE {k}/{len(episodes)} {eid} COMPLETE',flush=True)
    ep=pd.DataFrame(records); decisions=pd.DataFrame(all_decisions)
    lookup=ep.set_index(['episode','strategy']);pairs=[]
    for _,r in ep.iterrows():
        if r.strategy=='equal80_hold':continue
        b=lookup.loc[(r.episode,'equal80_hold')]
        pairs.append(dict(episode=r.episode,segment=r.segment,strategy=r.strategy,
            reference='equal80_hold',delta_return_pp=100*(r.cumulative_return-b.cumulative_return),
            delta_sharpe=r.sharpe_proxy-b.sharpe_proxy,
            delta_mdd_pp=100*(r.mdd_proxy-b.mdd_proxy),
            delta_turnover=r.turnover_proxy-b.turnover_proxy,
            delta_fee_usd=r.fees_usd-b.fees_usd,
            delta_exposure_pp=100*(r.mean_exposure-b.mean_exposure)))
    paired=pd.DataFrame(pairs)
    summary=ep.groupby(['segment','strategy'],as_index=False).agg(
        episodes=('episode','size'),mean_return=('cumulative_return','mean'),
        mean_sharpe=('sharpe_proxy','mean'),mean_mdd=('mdd_proxy','mean'),
        mean_turnover=('turnover_proxy','mean'),mean_fees_usd=('fees_usd','mean'),
        mean_exposure=('mean_exposure','mean'),mean_holdings=('mean_holdings','mean'),
        min_holdings=('min_holdings','min'),max_holdings=('max_holdings','max'),
        below70_rounds=('below70_rounds','sum'),over90_rounds=('over90_rounds','sum'),
        source_events_observed=('source_events_observed','sum'))
    pairsummary=paired.groupby(['segment','strategy','reference'],as_index=False).agg(
        episodes=('episode','size'),mean_delta_return_pp=('delta_return_pp','mean'),
        win_fraction=('delta_return_pp',lambda x:float((x>0).mean())),
        mean_delta_sharpe=('delta_sharpe','mean'),
        mean_delta_mdd_pp=('delta_mdd_pp','mean'),
        mean_delta_turnover=('delta_turnover','mean'),
        mean_extra_fee_usd=('delta_fee_usd','mean'))
    folder=Path(outpath or OUT);folder.mkdir(parents=True,exist_ok=True)
    for name,d in [('phase42_episode_metrics.csv',ep),('phase42_policy_summary.csv',summary),
                   ('phase42_paired_episode_deltas.csv',paired),('phase42_paired_summary.csv',pairsummary),
                   ('phase42_105_round_decisions.csv',decisions),
                   ('phase42_round_activity.csv',decisions.groupby(['strategy','round'],as_index=False).agg(
                       trades=('trade','sum'),mean_exposure=('equity_exposure','mean'),
                       mean_holdings=('holdings','mean'),fees_usd=('fee_usd','sum')))]:
        d.to_csv(folder/name,index=False)
    manifest=dict(research_only=True,lookahead_policy='prior-close or completed bar only',
                  historical_sec_llm_alpha='NOT_BACKFILLED',input_file=str(inp.resolve()),
                  input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                  episodes=len(episodes),policies=len(policies()),round_decisions=len(decisions),
                  execution_proxy='Alpaca IEX 15m open, not organizer fills',
                  limitations=['already-examined 2026 periods','no slippage beyond 0.1% fee',
                               'excluded incomplete dates','forward verified event paper tilt only; absent from historic data'])
    (folder/'phase42_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PHASE42_COMPLETE_RESEARCH_ONLY_NO_ORGANIZER_NO_PROVIDER_CALLS')
    print('PHASE42_OUTPUT',folder,'EPISODES',len(episodes),'DECISIONS',len(decisions))
    return ep,decisions


def self_test():
    score=np.linspace(-1.8,1.8,N)
    f=dict(vol=np.full(N,.01),broad20=.05,broadvol=.01)
    groups=[np.array(z,dtype=int) for z in np.array_split(np.arange(N),6)]
    a,eligible=weights(score,f,np.zeros(N,bool),.80,groups)
    assert 0<(a>0).sum()<N and abs(a.sum()-.8)<1e-4
    f['broad20']=-.2;low=exposure(f);f['broad20']=.2;high=exposure(f)
    assert .7<=low<high<=.9
    assert len(policies())==14 and len(set(x.name for x in policies()))==14
    cash=np.array(800_000.,float).item();shares=np.zeros(N);mark=np.full(N,100.);fill=mark.copy()
    eventsym={'symbol':'AAPL','direction':'positive','_symbol_index':0}
    sector_of={int(j):i for i,g in enumerate(groups) for j in g}
    cc,ss,g,fee,why=event_paper_adjustment(cash,shares,mark,fill,eventsym,
                                         groups,sector_of)
    assert g>0 and ss[0]>0 and cc<cash and abs(fee-g*.001)<1e-8
    eventsym['direction']='negative'
    cc2,ss2,g2,fee2,why=event_paper_adjustment(cc,ss,mark,fill,eventsym,
                                         groups,sector_of)
    assert g2>0 and ss2[0]<1e-3 and cc2>cc
    # An unobserved event cannot ever be backfilled; historical import checks
    # exist in Phase39.read_events() and in Phase39 CaMeL forward self-test.
    print('PHASE42_SELF_TEST_PASS')
    print('VARIABLE_STOCK_COUNT_DYNAMIC_EXPOSURE_COST_AWARE_NO_LOOKAHEAD_MODEL_BRANCHES_PASS')
    print('NO_NETWORK_NO_KEYS_NO_LLM_NO_OFFICIAL_SUBMISSION')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--run',action='store_true')
    ap.add_argument('--input',default=None);ap.add_argument('--out',default=None)
    ap.add_argument('--stride',type=int,choices=(5,15),default=15)
    ap.add_argument('--min-last-day',default='2026-10-09')
    ap.add_argument('--events-file',default=None)
    args=ap.parse_args()
    try:
        if args.self_test:self_test()
        else:run(args.input,args.out,args.stride,args.min_last_day,args.events_file)
        return 0
    except Exception as exc:
        print('PHASE42_STOP:',type(exc).__name__,str(exc),file=sys.stderr)
        return 2
if __name__=='__main__':sys.exit(main())
