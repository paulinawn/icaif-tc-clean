#!/usr/bin/env python3
"""ICAIF Phase38: OFFLINE 15-session x 7-round cash-reset episode tournament.

No network, API credentials, live trades, organizer calls or server changes.
CaMeL = strictly timestamped *forward-only* human-verified event overlay. There
is no historical CaMeL replay when source/model outputs were recorded later.
Current July-Oct9 2026 IEX history precedes Phase36 October11 classification.
Thus unseeded '+camel' pairs are intentionally identical: not evidence for/against
CaMeL alpha. Scenario-only event tests in self-test verify integration mechanics.

IEX open/close 15-min bars provide imperfect fill proxies; corporate actions not
adjusted, half-days and incomplete days excluded by Phase30 v2 loader.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=Path('artifacts/phase38_105slot_camel_staging')
FEE=.001
CAP=.30
N=30
NAV0=1_000_000.
EXPOSURE=.80
RAMP_INITIAL=.30
TILT=.003
MAX_EVENT_AGE_MINUTES=240
MAX_DAILY_TRADE_L1=.08
MAX_INTRADAY_TRADE_L1=.06
MIN_DAILY_TRADE_L1=.025
MIN_INTRADAY_TRADE_L1=.035
EPISODE=15
DEADLINE_ET=('09:10','10:25','11:25','12:25','13:25','14:25','15:25')
FILL_ET=('09:30','10:30','11:30','12:30','13:30','14:30','15:30')
REACTION_TERMS=('shares surged','shares plunged','stock soared','stock fell',
                'shares tumbled','stock jumped','stock dropped','market reacts')
MATERIAL_TYPES={'earnings','guidance','revenue','regulatory','legal','merger','product'}

@dataclass(frozen=True)
class Policy:
    name:str
    core:str
    entry:str='immediate'  # immediate, ramp2, ramp7
    adaptation:str='hold'   # hold, daily, intraday
    camel:bool=False


def policies():
    basic=[
       Policy('equal60_hold','equal60'),Policy('equal80_hold','equal80'),
       Policy('blend80_hold','blend80'),Policy('lowvol80_hold','lowvol80'),
       Policy('lowvol80_legacy_hold','lowvol80_legacy'),
       Policy('equal80_ramp2','equal80','ramp2'),
       Policy('equal80_ramp7','equal80','ramp7'),
       Policy('blend80_ramp7','blend80','ramp7'),
       Policy('lowvol80_ramp7','lowvol80','ramp7'),
       Policy('equal80_daily_relative','equal80','immediate','daily'),
       Policy('blend80_daily_relative','blend80','immediate','daily'),
       Policy('lowvol80_daily_relative','lowvol80','immediate','daily'),
       Policy('blend80_intraday_reversal','blend80','immediate','intraday'),
       Policy('lowvol80_intraday_reversal','lowvol80','immediate','intraday'),
       Policy('equal80_ramp7_daily','equal80','ramp7','daily'),
       Policy('blend80_ramp7_daily','blend80','ramp7','daily'),
       Policy('lowvol80_ramp7_daily','lowvol80','ramp7','daily'),
    ]
    result=[]
    for p in basic:
        result.append(p)
        if p.entry!='immediate' or p.adaptation!='hold':
            result.append(Policy(p.name+'__camel',p.core,p.entry,p.adaptation,True))
    return tuple(result)


def phase30():
    path=HERE/'icaif_phase30_2026_walkforward_agent_lab_v2.py'
    if not path.is_file():raise FileNotFoundError('Place Phase30 v2 next to Phase38: '+str(path))
    spec=importlib.util.spec_from_file_location('phase38_p30',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
    assert spec.loader;spec.loader.exec_module(m)
    return m


def phase37():
    path=HERE/'icaif_phase37_risk_regime_tournament.py'
    if not path.is_file():raise FileNotFoundError('Place Phase37 next to Phase38: '+str(path))
    spec=importlib.util.spec_from_file_location('phase38_p37',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
    assert spec.loader;spec.loader.exec_module(m)
    return m


def check_w(w):
    w=np.asarray(w,float)
    if w.shape!=(N,) or not np.isfinite(w).all() or w.min() < -1e-10 or w.max()>CAP+1e-10 or w.sum()>1+1e-10:
        raise AssertionError('Illegal portfolio weights')
    return np.maximum(w,0.)


# Manual GICS-like sector taxonomy for the 30 known issuers; not a scraped as-of
# sector database. The legacy 6x5 grouping is retained as a separate control.
SECTOR_MEMBERS=(
 ('AAPL','MSFT','NVDA','INTC','CRM'),
 ('JPM','BAC','GS','V','PYPL'),
 ('LLY','JNJ','UNH','PFE','TMO'),
 ('AMZN','TSLA','NKE'),
 ('WMT','KO'),
 ('CAT','GE','BA'),
 ('XOM','CVX'),
 ('GOOGL','META','DIS','T'),
 ('NEE',),
)


def gics_like_lowvol_core(p30,market,t):
    """80% blended equal-stock / equal-sector budgets, 20-day lagged vol.

    Blending prevents a lone NEE sector from receiving a full 1/9 of NAV;
    legacy six-by-five low-vol remains available for an exact Phase37 control.
    """
    if t<20:raise ValueError('Insufficient vol estimation history')
    r=np.diff(np.log(market.daily[t-20:t+1]),axis=0)
    sig=np.maximum(np.std(r,axis=0,ddof=1),.004)
    names=list(p30.SYMBOLS)
    allmembers=[s for sec in SECTOR_MEMBERS for s in sec]
    if len(allmembers)!=N or set(allmembers)!=set(names):raise AssertionError('Bad sector map')
    w=.5*np.full(N,EXPOSURE/N)
    for group in SECTOR_MEMBERS:
        ix=[names.index(s) for s in group]
        iv=1/sig[ix];iv=iv/iv.sum()
        within=.60/len(ix)+.40*iv
        w[ix]+=.5*(EXPOSURE/len(SECTOR_MEMBERS))*within
    if abs(float(w.sum())-EXPOSURE)>1e-10:raise AssertionError('Sector exposure changed')
    return check_w(w)


def core_weights(p30,p37,market,t,core):
    if core=='equal60':return check_w(np.full(N,.6/N))
    if core=='equal80':return check_w(np.full(N,.8/N))
    if core=='blend80':return check_w(p37.base_core(p30,market,t,'core80_blend_hold'))
    if core=='lowvol80':return gics_like_lowvol_core(p30,market,t)
    if core=='lowvol80_legacy':return check_w(p37.sector_lowvol_core(market,t))
    raise ValueError('Unknown core')


def staged_fraction(entry,r):
    """Pro rata fraction of eventual equity exposure at round r in first session."""
    if entry=='immediate':return 1.
    if entry=='ramp2':return RAMP_INITIAL/EXPOSURE if r==0 else 1.
    if entry=='ramp7':return RAMP_INITIAL/EXPOSURE+(1.-RAMP_INITIAL/EXPOSURE)*r/6.
    raise ValueError('Unknown staged entry')


def daily_tilt(market,t):
    """Cross-sectional rank-based rule only using yesterday and earlier closes."""
    if t<21:raise ValueError('Insufficient lagged history')
    lag=np.log(market.daily)
    r5=lag[t]-lag[t-5];r20=lag[t]-lag[t-20]
    score=.7*r5+.3*r20
    score=score-score.mean()
    v=float(np.std(score))
    z=np.clip(score/max(v,1e-10),-2.,2.)
    return .0020*(z-z.mean())


def intraday_tilt(market,i,r,p30):
    """Cross-sectional reversal with last bar KNOWN at submission deadline."""
    if r<1:raise ValueError('R1 no intraday completed bar')
    last=p30.KNOWN_IDX[r]
    # KNOWN_IDX[r] is completed before the specified deadline; bar fill is later.
    period=np.log(market.cl[i,last,:]/market.op[i,0,:])
    period=period-period.mean()
    z=np.clip(period/max(float(period.std()),1e-10),-2.,2.)
    return -.0017*(z-z.mean())


def project_tilt(base,delta):
    """Zero-sum tilts, preserve exposure, long-only and <=30% per symbol."""
    if not np.isfinite(delta).all() or len(delta)!=N:raise ValueError('Bad delta')
    x=base+delta
    # Bounded reserve in these diversified cores means clipping normally inactive.
    if x.min()<0 or x.max()>CAP:
        raise ValueError('Tilt breaches legal portfolio weights; abstain rather than clip silently')
    if abs(float(x.sum()-base.sum()))>1e-8:raise AssertionError('Tilt altered net exposure')
    return check_w(x)


def parse_time(s):
    if not isinstance(s,str):raise ValueError('Expected UTC ISO timestamp')
    try:d=datetime.fromisoformat(s.replace('Z','+00:00'))
    except ValueError:raise ValueError('Invalid UTC event time') from None
    if d.tzinfo is None:raise ValueError('Naive event time')
    return d.astimezone(timezone.utc)


def load_events(path,symbols):
    """Load ONLY independently verified CaMeL model outputs created at the time.

    No live provider call, no inference on historic articles. A Phase36 earnings
    preview is intentionally NOT an eligible directional event.
    """
    if not path:return []
    obj=json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
    if isinstance(obj,dict) and 'events' in obj:obj=obj['events']
    if isinstance(obj,dict) and 'extracted_watchlist' in obj:
        print('PHASE38_CAMEL_PHASE36_PREVIEW_NOT_TRADE_ELIGIBLE')
        return []
    if not isinstance(obj,list):raise ValueError('CaMeL events must be a JSON list')
    result=[];ident=set()
    for e in obj:
        if not isinstance(e,dict):raise ValueError('CaMeL event not an object')
        sid=str(e.get('event_id',''))
        if not sid or sid in ident:raise ValueError('Missing/duplicate event ID')
        ident.add(sid)
        pub=parse_time(e.get('published_utc'));obs=parse_time(e.get('first_observed_utc'))
        inf=parse_time(e.get('camel_completed_utc'))
        if not pub<=obs<=inf:raise ValueError('Non-causal event source/model timestamps')
        if e.get('symbol') not in symbols:raise ValueError('Unknown CaMeL symbol')
        if e.get('source_verified') is not True or e.get('issuer_verified') is not True:
            continue
        if e.get('material_new_fact') is not True or e.get('is_company_specific') is not True:
            continue
        if e.get('event_type') not in MATERIAL_TYPES or e.get('materiality') not in ('medium','high'):
            continue
        if e.get('direction') not in ('positive','negative'):
            continue
        if e.get('is_preview') is True or e.get('price_reaction_story') is True:
            continue
        if any(x in str(e.get('headline','')).lower() for x in REACTION_TERMS):continue
        result.append(dict(e, _published=pub, _observed=obs,_available=inf))
    return sorted(result,key=lambda x:(x['_available'],x['event_id']))


def event_tilt(events,at,core,idx):
    """One latest eligible fresh event. No inference-to-trade time travel."""
    eligible=[e for e in events if e['_available']<=at and
        timedelta(0)<=at-e['_published']<=timedelta(minutes=MAX_EVENT_AGE_MINUTES)]
    if not eligible:return np.zeros(N),None
    ev=eligible[-1]
    direction=1 if ev['direction']=='positive' else -1
    j=idx[ev['symbol']]
    delta=np.full(N,-direction*TILT/(N-1),float)
    delta[j]=direction*TILT
    candidate=core+delta
    if candidate.min()<0 or candidate.max()>CAP:return np.zeros(N),None
    return delta,ev['event_id']


def decision_time(date,r):
    """ET local schedule, converted to UTC. Avoid summer/winter timezone mistakes."""
    return pd.Timestamp(f'{pd.Timestamp(date).date()} {DEADLINE_ET[r]}',tz='America/New_York').tz_convert('UTC').to_pydatetime()


def simulate(p30,p37,market,indices,policy,events):
    if len(indices)!=EPISODE:raise AssertionError('Exactly 15 trading dates required')
    if not all(int(b)==int(a)+1 for a,b in zip(indices[:-1],indices[1:])):
        raise AssertionError('Episode indices must be contiguous in retained trading dates')
    cash=NAV0;shares=np.zeros(N);nav=[NAV0];turns=[];ledger=[]
    fees=0.;notional_total=0.;trades=0;accepted_overlay=0
    first_t=int(indices[0])-1
    base=core_weights(p30,p37,market,first_t,policy.core)
    symidx={s:i for i,s in enumerate(p30.SYMBOLS)}
    prev_core=base.copy()
    for dayn,didx in enumerate(indices):
        i=int(didx);t=i-1
        for r in range(7):
            # Decision price: previous session close for R1, most recent fully
            # completed bar at R2-R7, *never* the forthcoming execution bar.
            mark=market.daily[t] if r==0 else market.cl[i,p30.KNOWN_IDX[r],:]
            pre_nav=cash+np.dot(shares,mark)
            current=shares*mark/pre_nav
            intent=None;reason='HOLD_NO_NEW_SIGNAL';desired=None
            info_source='previous_session_close' if r==0 else p30.SLOTS[p30.KNOWN_IDX[r]]
            core=prev_core
            # Fixed episode-start portfolio for true static / staging variants.
            # Only active variants change stock weights using causal indicators.
            if dayn==0:
                exposure=float(base.sum())
                fraction=staged_fraction(policy.entry,r)
                desired=base*fraction
                if policy.entry=='immediate':
                    if r==0:intent=desired;reason='INITIAL_FULL_ENTRY'
                elif r==0 or (policy.entry=='ramp7' and r<=6) or (policy.entry=='ramp2' and r==1):
                    intent=desired;reason='STAGED_ENTRY'
            else:
                # Day 2-15: daily relative strength at R1 and intraday completed
                # bar reversal at R2..7. No monthly refit or monthly condition.
                if policy.adaptation in ('daily','intraday') and r==0:
                    try:core=project_tilt(base,daily_tilt(market,t))
                    except ValueError:core=base
                    prev_core=core.copy()
                    desired=core;intent=desired;reason='DAILY_RELATIVE_PROPOSAL'
                elif policy.adaptation=='intraday' and r>0:
                    try:desired=project_tilt(prev_core,intraday_tilt(market,i,r,p30))
                    except ValueError:desired=prev_core
                    intent=desired;reason='INTRADAY_REVERSAL_PROPOSAL'
                else:desired=prev_core
            evt=None
            if policy.camel:
                # No archival CaMeL event backfill. Event must have source
                # publication+observation+inference before round deadline.
                delta,evt=event_tilt(events,decision_time(market.dates[i],r),desired,symidx)
                if evt is not None:
                    desired=check_w(desired+delta)
                    intent=desired
                    reason='VERIFIED_CAUSAL_CAMEL_PAPER_PROPOSAL'
            proposed_l1=float(np.abs((desired if desired is not None else current)-current).sum())
            if intent is not None and not (dayn==0 and r==0) and reason!='STAGED_ENTRY':
                min_l1=(.005 if evt is not None else
                        (MIN_INTRADAY_TRADE_L1 if r>0 else MIN_DAILY_TRADE_L1))
                max_l1=MAX_INTRADAY_TRADE_L1 if r>0 else MAX_DAILY_TRADE_L1
                if proposed_l1<min_l1:
                    intent=None;reason='HOLD_BELOW_COST_AND_MIN_TURNOVER_GATE'
                else:
                    # Bounded increment in target weights so active variants do
                    # not assume unconstrained instant readjustments.
                    scale=min(1.,max_l1/proposed_l1)
                    intent=check_w(current+scale*(intent-current))
            fee=0.;qty=0.;round_turn=0.
            if intent is not None:
                # Trading only at actual eligible execution-bar price proxy.
                fill=market.op[i,p30.FILL_IDX[r],:]
                fill_nav=float(cash+np.dot(shares,fill))
                cash,shares,qty=p30.trade(cash,shares,fill,intent)
                fee=FEE*qty
                if qty>1e-6:
                    trades+=1;fees+=fee;notional_total+=qty
                    round_turn=qty/max(fill_nav,1e-12)
                    if evt is not None:accepted_overlay+=1
                    decision='TRADE'
                else:decision='HOLD';reason='ZERO_EXECUTION_NOTIONAL'
            else:decision='HOLD'
            ending=market.op[i,p30.END_VAL_IDX[r],:] if r<6 else market.cl[i,25,:]
            closing_nav=float(cash+np.dot(shares,ending))
            if not np.isfinite(closing_nav) or closing_nav<=0:raise ValueError('Invalid portfolio NAV')
            nav.append(closing_nav);turns.append(round_turn)
            ledger.append({'date':str(market.dates[i].date()),'round':r+1,
                'deadline_et':DEADLINE_ET[r],'execution_et':FILL_ET[r],
                'information_cutoff':info_source,
                'decision':decision,'reason':reason,
                'proposed_l1':proposed_l1,'predecision_exposure':float(current.sum()),
                'after_trade_equity_weight':float(np.sum(shares*fill/(cash+np.dot(shares,fill)))) if intent is not None else None,
                'target_exposure':float(desired.sum()) if desired is not None else float(current.sum()),
                'causal_camel_event_id':evt,'notional_usd':qty,'fee_usd':fee,
                'turnover_this_round':round_turn,'nav_end':closing_nav})
    if len(ledger)!=105:raise AssertionError('Did not evaluate all 105 rounds')
    v=np.asarray(nav)
    ret=v[1:]/v[:-1]-1
    sigma=float(ret.std(ddof=1))
    metrics={'cumulative_return':float(v[-1]/NAV0-1),
             'sharpe_proxy':float(ret.mean()*math.sqrt(252*7)/sigma) if sigma>1e-12 else 0.,
             'mdd_proxy':float(np.max(1.-v/np.maximum.accumulate(v))),
             'turnover_proxy':float(np.mean(turns)),'fees_usd':fees,
             'trade_count':trades,'camel_trade_count':accepted_overlay,
             'mean_equity_exposure':float(np.mean([x['predecision_exposure'] for x in ledger])),
             'rounds_simulated':len(ledger)}
    return metrics,ledger


def select_episodes(market,stride=5):
    if stride not in (1,5,15):raise ValueError('Stride must be 1,5,15 trading sessions')
    ranges=[('development','2026-07-01','2026-08-31'),
            ('exposed_diagnostic','2026-09-01','2026-10-09')]
    episodes=[]
    incomplete=set()
    if hasattr(market,'missing') and isinstance(market.missing,pd.DataFrame) and 'date' in market.missing:
        incomplete={pd.Timestamp(x) for x in market.missing['date'].tolist()}
    skipped_gaps=0
    for label,start,end in ranges:
        ix=np.flatnonzero((market.dates>=pd.Timestamp(start))&(market.dates<=pd.Timestamp(end)))
        if len(ix)<EPISODE:raise ValueError('Not enough '+label+' sessions')
        for offset in range(0,len(ix)-EPISODE+1,stride):
            block=ix[offset:offset+EPISODE]
            if len(block)!=EPISODE:raise AssertionError
            left=market.dates[int(block[0])];right=market.dates[int(block[-1])]
            if any(left < d < right for d in incomplete):
                skipped_gaps+=1
                continue
            episodes.append((f'{label}_{market.dates[int(block[0])].date()}',label,block))
    if skipped_gaps:print('PHASE38_SKIPPED_15SESSION_EPISODES_SPANNING_INCOMPLETE_DATES:',skipped_gaps,flush=True)
    return episodes


def metric_ranks(frame):
    """Rank against NON-CaMeL reference policies only.

    Duplicated paired CaMeL variants must not distort the benchmark opponent
    percentile when historical CaMeL output is unavailable.
    """
    data=frame.copy()
    base=data[~data.strategy.str.endswith('__camel')]
    for col,asc in [('cumulative_return',False),('sharpe_proxy',False),
                    ('mdd_proxy',True),('turnover_proxy',True)]:
        dest='rank_'+col
        data[dest]=np.nan
        data.loc[base.index,dest]=base.groupby('episode_id')[col].rank(
            pct=True,method='average',ascending=asc)
        for episode,group in data[data.strategy.str.endswith('__camel')].groupby('episode_id'):
            values=base[base.episode_id==episode][col].to_numpy(float)
            n=len(values)
            if n==0:raise AssertionError('Missing base policies')
            for idx,row in group.iterrows():
                score=float(row[col])
                better=int(np.sum(values < score) if asc else np.sum(values > score))
                equal=int(np.sum(values == score))
                data.at[idx,dest]=max(1.,better+(equal+1)/2)/n
    data['four_metric_proxy_rank']=data[['rank_'+c for c in
        ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]].mean(axis=1)
    return data


def pairs_summary(result):
    allrows=[]
    # Episode averages are descriptive and overlapping; never independent p-values.
    for policy in policies():
        ss=result[result.strategy==policy.name]
        for segment,g in ss.groupby('segment',sort=False):
            gb=g.copy()
            allrows.append({'strategy':policy.name,'segment':segment,'episode_count':len(g),
                'mean_return':gb.cumulative_return.mean(),'median_return':gb.cumulative_return.median(),
                'mean_sharpe':gb.sharpe_proxy.mean(),'mean_mdd':gb.mdd_proxy.mean(),
                'mean_turnover':gb.turnover_proxy.mean(),'mean_fees_usd':gb.fees_usd.mean(),
                'mean_trades':gb.trade_count.mean(),
                'mean_camel_trades':gb.camel_trade_count.mean(),
                'mean_four_metric_proxy_rank':gb.four_metric_proxy_rank.mean()})
    return pd.DataFrame(allrows).sort_values(['segment','mean_four_metric_proxy_rank'])


def paired_deltas(result):
    data=[];lookup=result.set_index(['episode_id','strategy'])
    for policy in policies():
        if policy.name.endswith('__camel'):ref=policy.name.removesuffix('__camel')
        else:continue
        if policy.name.endswith('__camel'):
            for eid in result.episode_id.unique():
                curr=lookup.loc[(eid,policy.name)];base=lookup.loc[(eid,ref)]
                data.append({'episode_id':eid,'segment':curr.segment,'variant':policy.name,
                    'compared_to':ref,'delta_return_pp':100*(curr.cumulative_return-base.cumulative_return),
                    'delta_sharpe':curr.sharpe_proxy-base.sharpe_proxy,
                    'delta_mdd_pp':100*(curr.mdd_proxy-base.mdd_proxy),
                    'delta_turnover':curr.turnover_proxy-base.turnover_proxy,
                    'delta_fees_usd':curr.fees_usd-base.fees_usd,
                    'delta_trade_count':curr.trade_count-base.trade_count})
    return pd.DataFrame(data)


def paired_vs_core(result):
    """Absolute four-metric and fee differences within same 15-day episode."""
    lookup=result.set_index(['episode_id','strategy'])
    rows=[]
    for policy in policies():
        if policy.camel or (policy.entry=='immediate' and policy.adaptation=='hold'):
            continue
        reference={'equal80':'equal80_hold','blend80':'blend80_hold',
                   'lowvol80':'lowvol80_hold'}.get(policy.core)
        if reference is None:continue
        for episode in result.episode_id.unique():
            a=lookup.loc[(episode,policy.name)];b=lookup.loc[(episode,reference)]
            rows.append({'episode_id':episode,'segment':a.segment,
                'strategy':policy.name,'reference':reference,
                'delta_return_pp':100*(a.cumulative_return-b.cumulative_return),
                'delta_sharpe':a.sharpe_proxy-b.sharpe_proxy,
                'delta_mdd_pp':100*(a.mdd_proxy-b.mdd_proxy),
                'delta_turnover':a.turnover_proxy-b.turnover_proxy,
                'delta_fees_usd':a.fees_usd-b.fees_usd,
                'delta_trade_count':a.trade_count-b.trade_count})
    details=pd.DataFrame(rows)
    if details.empty:return details,details
    s=[]
    for (segment,name),g in details.groupby(['segment','strategy']):
        s.append({'segment':segment,'strategy':name,'episodes':len(g),
            'mean_delta_return_pp':g.delta_return_pp.mean(),
            'median_delta_return_pp':g.delta_return_pp.median(),
            'return_win_fraction':float((g.delta_return_pp>0).mean()),
            'mean_delta_sharpe':g.delta_sharpe.mean(),
            'mean_delta_mdd_pp':g.delta_mdd_pp.mean(),
            'mean_delta_turnover':g.delta_turnover.mean(),
            'mean_extra_fees_usd':g.delta_fees_usd.mean(),
            'mean_extra_trades':g.delta_trade_count.mean()})
    return details,pd.DataFrame(s).sort_values(['segment','mean_delta_return_pp'],ascending=[True,False])


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(root,input_csv,events_path,stride):
    p30=phase30();p37=phase37()
    source=Path(input_csv) if input_csv else p30.INPUT
    if not source.is_absolute():source=root/source
    source=source.resolve()
    market=p30.read_market(source,min_last_day='2026-10-09')
    events=load_events(events_path,p30.SYMBOLS)
    out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    episodes=select_episodes(market,stride)
    all_metrics=[];trace=[]
    print('PHASE38_MARKET_LAST_DAY:',market.dates[-1].date(),'INCOMPLETE_DAYS_EXCLUDED:',len(market.missing),flush=True)
    print('PHASE38_15_SESSION_EPISODES:',len(episodes),'ROUND_DECISIONS_PER_POLICY_EPISODE: 105',flush=True)
    print('PHASE38_CAUSALLY_ELIGIBLE_EVENT_RECORDS:',len(events),flush=True)
    print('PHASE38_NO_BACKDATED_LLMS_CAMEL_PAIRS_MAY_BE_IDENTICAL',flush=True)
    specs=policies()
    for k,(eid,segment,idx) in enumerate(episodes,1):
        for p in specs:
            stats,details=simulate(p30,p37,market,idx,p,events)
            all_metrics.append({'episode_id':eid,'segment':segment,
                'first_date':str(market.dates[int(idx[0])].date()),
                'last_date':str(market.dates[int(idx[-1])].date()),
                'strategy':p.name,**stats})
            trace.extend([{'episode_id':eid,'segment':segment,'strategy':p.name,**d} for d in details])
        print(f'PHASE38_EPISODE_{k:02d}_OF_{len(episodes)} {eid} EVALUATED_POLICIES:{len(specs)}',flush=True)
    frame=metric_ranks(pd.DataFrame(all_metrics))
    dec=pd.DataFrame(trace)
    comparison=pairs_summary(frame)
    deltas=paired_deltas(frame)
    core_delta,core_summary=paired_vs_core(frame)
    activity=(dec.assign(is_trade=dec.decision.eq('TRADE').astype(int))
              .groupby(['segment','strategy','round'],as_index=False)
              .agg(decision_slots=('is_trade','size'),executed_trades=('is_trade','sum'),
                   gross_notional_usd=('notional_usd','sum'),fees_usd=('fee_usd','sum')))
    frame.to_csv(out/'phase38_15day_episode_results.csv',index=False)
    comparison.to_csv(out/'phase38_policy_summary.csv',index=False)
    dec.to_csv(out/'phase38_all_105_round_decisions.csv',index=False)
    deltas.to_csv(out/'phase38_camel_paired_deltas.csv',index=False)
    core_delta.to_csv(out/'phase38_paired_vs_same_core.csv',index=False)
    core_summary.to_csv(out/'phase38_paired_vs_same_core_summary.csv',index=False)
    activity.to_csv(out/'phase38_round_activity.csv',index=False)
    total_live_camel=int(dec.causal_camel_event_id.notna().sum())
    print('PHASE38_CAMEL_DECISION_SLOTS_WITH_VERIFIED_EVENTS:',total_live_camel,flush=True)
    print('=== PHASE38 PAIRED ACTIVE/STAGED VS UNCHANGED CORE (MEAN RETURN PP) ===',flush=True)
    print(core_summary[['segment','strategy','episodes','mean_delta_return_pp','mean_delta_mdd_pp','mean_extra_fees_usd']].to_string(index=False),flush=True)
    print('=== PHASE38 DEVELOPMENT DESCRIPTIVE RANK (LOWER BETTER) ===',flush=True)
    print(comparison[comparison.segment=='development'][['strategy','mean_return','mean_sharpe','mean_mdd','mean_turnover','mean_fees_usd','mean_trades','mean_four_metric_proxy_rank']].sort_values('mean_four_metric_proxy_rank').to_string(index=False),flush=True)
    print('=== PHASE38 EXPOSED_DIAGNOSTIC DESCRIPTIVE RANK (LOWER BETTER) ===',flush=True)
    print(comparison[comparison.segment=='exposed_diagnostic'][['strategy','mean_return','mean_sharpe','mean_mdd','mean_turnover','mean_fees_usd','mean_trades','mean_four_metric_proxy_rank']].sort_values('mean_four_metric_proxy_rank').to_string(index=False),flush=True)
    (out/'phase38_manifest.json').write_text(json.dumps({
        'research_only':True,'network_calls':0,'llm_calls':0,'organizer_calls':0,'official_deployment':False,
        'data_end':str(market.dates[-1].date()),'data_sha256':sha256(source),
        'source_file':str(source),'skipped_incomplete_dates':len(market.missing),
        'episode_sessions':EPISODE,'decision_rounds':105,'rolling_stride':stride,
        'overlapping_windows_not_independent':stride<15,'period_segments':list(sorted(set(comparison.segment))),
        'policy_count':len(specs),'research_reference_policy_count':sum(not p.camel for p in specs),
        'verified_event_records':len(events),
        'event_decision_rows':total_live_camel,
        'camel_backtest_note':'No model inference before decision => no CaMeL alpha attributed; identical +/- variants expected.',
        'models':['static 60/80','price/industry blend','20-day GICS-like sector lowvol',
                  '5+20day relative momentum','15min intraday reversal','staged execution','causal CaMeL events'],
        'source_limitations':['Unadjusted IEX market prices / incomplete sessions skipped',
                'Legacy six-by-five groups remain only for Phase37 baseline control; GICS-like map is manual',
                'Fees modeled 0.1% buys plus sells, no impact/slippage beyond IEX open proxy',
                'All September/October market results already viewed; no untouched test',
                'Overlap windows correlate strongly, no independent error bars',
                'Stage R1 30pct means cash drag, not extra alpha',
                'Daily and intraday tilts are heuristics with no separately validated expected-return model',
                'CaMeL calls not made; hypothetical overlay requires real timestamped source AND inference'],
        'stage_equity_targets_round_1_to_7':[round(.8*staged_fraction('ramp7',r),8) for r in range(7)],
        'schedule_et':{'deadlines':DEADLINE_ET,'executions':FILL_ET}
    },indent=2),encoding='utf-8')
    print('PHASE38_RESULTS_DIR:',out.resolve())
    print('PHASE38_COMPLETE_RESEARCH_ONLY_NO_API_NO_LLM_NO_OFFICIAL_SUBMISSION')
    return frame,comparison,dec,deltas


def self_test():
    from types import SimpleNamespace
    rng=np.random.default_rng(20261011)
    px=100*np.exp(np.cumsum(rng.normal(0,.01,(280,30)),axis=0))
    dates=pd.bdate_range('2025-08-01',periods=280)
    # coherent bar path for deterministic, synthetic round-accurate checks
    intra=np.exp(np.linspace(0,.002,26))[None,:,None]
    op=px[:,None,:]*intra;cl=op*np.exp(.00003)
    market=SimpleNamespace(dates=dates,op=op,cl=cl,daily=cl[:,-1,:])
    class P:
        SYMBOLS=tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
        KNOWN_IDX=(None,2,6,10,14,18,22)
        FILL_IDX=(0,4,8,12,16,20,24)
        END_VAL_IDX=(4,8,12,16,20,24,25)
        SLOTS=[f'{9+(30+15*j)//60:02d}:{(30+15*j)%60:02d}' for j in range(26)]
        @staticmethod
        def desired(name,m,t,models):return np.full(30,.8/30)
        @staticmethod
        def trade(cash,shares,price,w):
            before=cash+shares@price;old=shares*price;after=before
            for _ in range(70):
                amt=np.abs(after*w-old).sum();up=before-.001*amt
                if abs(up-after)<1e-8:break
                after=up
            amt=np.abs(after*w-old).sum()
            return after*(1-w.sum()),w*after/price,float(amt)
    class S:
        @staticmethod
        def base_core(p,m,t,name):return np.full(30,.8/30)
        @staticmethod
        def sector_lowvol_core(m,t):return np.full(30,.8/30)
    idx=np.arange(150,165)
    assert abs(float(gics_like_lowvol_core(P,market,149).sum())-.8)<1e-12
    assert abs(staged_fraction('ramp7',0)-.375)<1e-10
    assert abs(staged_fraction('ramp7',6)-1)<1e-10
    for p in policies():
        v,rows=simulate(P,S,market,idx,p,[])
        assert len(rows)==105 and v['rounds_simulated']==105
        assert abs(sum(x['fee_usd'] for x in rows)-v['fees_usd'])<1e-6
        assert all(0<=x['round']<=7 for x in rows)
        if p.name=='equal80_ramp7':
            assert [x['decision'] for x in rows[:7]]==['TRADE']*7
            assert np.allclose([x['target_exposure'] for x in rows[:7]],[.3,.3833333333333333,.4666666666666667,.55,.6333333333333333,.7166666666666667,.8])
        if p.name=='equal80_hold':
            assert v['trade_count']==1
        if p.camel:
            base=Policy(p.name.removesuffix('__camel'),p.core,p.entry,p.adaptation)
            b,_=simulate(P,S,market,idx,base,[])
            assert all(abs(v[k]-b[k])<1e-9 for k in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd'))
    example=[]
    for pol in policies():
        stats,_=simulate(P,S,market,idx,pol,[])
        example.append(dict(episode_id='toy',strategy=pol.name,**stats))
    ranked=metric_ranks(pd.DataFrame(example)).set_index('strategy')
    for pol in policies():
        if pol.camel:
            assert abs(float(ranked.loc[pol.name,'four_metric_proxy_rank'])-
                       float(ranked.loc[pol.name.removesuffix('__camel'),'four_metric_proxy_rank']))<1e-10
    at=decision_time(market.dates[150],2)
    ev={'event_id':'toy-test','symbol':'NVDA',
        'published_utc':(at-timedelta(minutes=30)).isoformat(),
        'first_observed_utc':(at-timedelta(minutes=25)).isoformat(),
        'camel_completed_utc':(at-timedelta(minutes=5)).isoformat(),
        'source_verified':True,'issuer_verified':True,'material_new_fact':True,
        'is_company_specific':True,'event_type':'earnings','materiality':'high',
        'direction':'positive','is_preview':False,'price_reaction_story':False,
        'headline':'Nvidia releases earnings and changes forward outlook'}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'events.json';path.write_text(json.dumps([ev]))
        x=load_events(path,P.SYMBOLS)
        z,eid=event_tilt(x,at,np.full(30,.8/30),{s:j for j,s in enumerate(P.SYMBOLS)})
        assert eid=='toy-test' and np.isclose(z.sum(),0)
        assert event_tilt(x,at-timedelta(minutes=10),np.full(30,.8/30),{s:j for j,s in enumerate(P.SYMBOLS)})[1] is None
        evtpolicy=Policy('test_camel','equal80','immediate','daily',True)
        rr,record=simulate(P,S,market,idx,evtpolicy,x)
        assert any(z['causal_camel_event_id']=='toy-test' and z['decision']=='TRADE' for z in record), 'Confirmed event should create paper-only rebalance'
        ev_bad=dict(ev);ev_bad['camel_completed_utc']=(at-timedelta(minutes=35)).isoformat()
        path.write_text(json.dumps([ev_bad]))
        try:load_events(path,P.SYMBOLS)
        except ValueError:pass
        else:raise AssertionError('Causal timestamp violation not blocked')
    # Future price edits cannot affect opening core, relative features or R2 signal.
    now=160
    before=daily_tilt(market,now)
    future=SimpleNamespace(daily=market.daily.copy())
    future.daily[now+1:]*=100
    assert np.allclose(before,daily_tilt(future,now))
    altered=SimpleNamespace(op=market.op.copy(),cl=market.cl.copy())
    r=2
    first=intraday_tilt(market,150,r,P)
    altered.cl[150,P.KNOWN_IDX[r]+1:,:]*=30
    assert np.allclose(first,intraday_tilt(altered,150,r,P))
    assert len(set(FILL_ET))==7 and len(set(DEADLINE_ET))==7
    print('PHASE38_SELF_TEST_PASS')
    print('EXACT_15_DAY_105_DECISION_SLOTS_STAGED_R1_TO_R7_AND_FEES_PASS')
    print('FORWARD_ONLY_CAMEL_TIMESTAMP_GATE_NO_HISTORICAL_BACKFILL_PASS')
    print('NO_NETWORK_NO_LLM_CREDITS_NO_ORGANIZER_CALLS_NO_HETZNER_CHANGES')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--self-test',action='store_true')
    a.add_argument('--run',action='store_true')
    a.add_argument('--input',default=None)
    a.add_argument('--camel-events',default=None,help='Forward-only verified CaMeL events JSON. Never backdate model inference.')
    a.add_argument('--stride',type=int,default=5,choices=(1,5,15))
    args=a.parse_args()
    if not (args.self_test or args.run):a.error('Choose --self-test or --run')
    try:
        if args.self_test:self_test()
        if args.run:run(Path.cwd(),args.input,args.camel_events,args.stride)
    except Exception as e:
        print(f'PHASE38_FAILED: {type(e).__name__}: {e}',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
