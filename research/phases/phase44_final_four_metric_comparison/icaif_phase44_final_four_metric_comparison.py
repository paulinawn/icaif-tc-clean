#!/usr/bin/env python3
"""ICAIF 2026 final offline four-metric portfolio comparison (research only).

No network, no LLM requests, no credentials, no organizer connections, no submissions.
This is a retrospective *illustrative* CaMeL ablation. The two responses were
actually generated in October and may have training-memory contamination.
Official-round Sharpe: R7 continues overnight to next-day R1 fill; each 16:00
close is an extra MDD observation, not an extra Sharpe return.

Prerequisites in this script's folder: phase30 v2, phase37, phase38,
phase39 v2, phase41 and phase42. Source CSV and JSON supplied by CLI flags.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ET=ZoneInfo('America/New_York')
FEE=.001; NAV0=1_000_000.0; ROUNDS=105; TILT=.003
# Verified against the *original* historical SEC accession timestamps in ET.
# These are not inference timestamps and must never be presented as such.
FILING_ET={
    '0000886982-26-000294':('GS','2026-07-14T07:30:40'),
    '0001628280-26-048086':('JPM','2026-07-14T09:52:21'),
}
JPM_EARLIER='0001628280-26-048078'  # earlier 06:30:38 ET release
POLICIES=(
    'equal80_hold','blend80_hold','dynamic_5d_reversal',
    'blend80_reversal_sleeve','blend80_camel_GS_illustrative',
    'blend80_camel_GS_JPM_repeat_sensitivity',
    'blend80_reversal_sleeve_camel_GS',
)

def modules():
    def load(name,filename):
        p=HERE/filename
        if not p.is_file(): raise FileNotFoundError('Put required file beside Phase44: '+str(p))
        s=importlib.util.spec_from_file_location(name,p)
        m=importlib.util.module_from_spec(s)
        sys.modules[name]=m;s.loader.exec_module(m)
        return m
    p30=load('phase44_p30','icaif_phase30_2026_walkforward_agent_lab_v2.py')
    p37=load('phase44_p37','icaif_phase37_risk_regime_tournament.py')
    p38=load('phase44_p38','icaif_phase38_105slot_camel_staging.py')
    p39=load('phase44_p39','icaif_phase39_105round_walkforward_v2.py')
    p41=load('phase44_p41','icaif_phase41_costaware_reversal.py')
    p42=load('phase44_p42','icaif_phase42_multisource_finalist_lab.py')
    return p30,p37,p38,p39,p41,p42


def evidence_predictions(path,symbols):
    """Only two pinned classifications with source hashes, reviewed for an as-if test.
    Constrain to exact accession + symbol; no generic arbitrary timestamp injection.
    """
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('version')!=3 or not obj.get('retrospective_only'):
        raise ValueError('Expected Phase43 v3 retrospective_predictions.json')
    events=[]
    for item in obj.get('events',[]):
        c=item.get('case',{});v=item.get('classification',{})
        acc=c.get('sec_accession','');sym=c.get('symbol')
        if acc not in FILING_ET or sym!=FILING_ET[acc][0] or sym not in symbols:
            continue
        if not item.get('retrospective_research_only') or not c.get('source_quality')=='SEC_ORIGINAL_ACCESSION_EXHIBIT':
            continue
        if item.get('case_evidence_sha256')!=c.get('evidence_sha256') or not c.get('evidence_text'):
            continue
        if not v.get('material_new_fact') or not v.get('issuer_specific') or v.get('direction') not in ('positive','negative'):
            continue
        if v.get('source_quote','') not in c['evidence_text']:
            raise ValueError('Unverified quote: '+str(sym))
        if v.get('strength') not in ('strong','medium'):continue
        # True SEC acceptance times in America/New_York. Real-time collection and
        # processing was NOT observed in 2026; the 60min lag is a hypothetical proxy.
        release=datetime.fromisoformat(FILING_ET[acc][1]).replace(tzinfo=ET)
        ready=release+timedelta(minutes=60)
        # JPM earnings release accepted at 06:30:38, before this 09:52 presentation.
        first_disclosure=(sym!='JPM')
        events.append(dict(event_id=c['event_id'],symbol=sym,accession=acc,
             pub=release,ready=ready,first_disclosure=first_disclosure,
             direction=v['direction'],strength=v['strength'],
             expectation_known=bool(v.get('expectation_known')),
             research_inferred_at=item.get('inferred_at_utc'),
             evidence_sha256=c['evidence_sha256']))
    if len(events)<2 or set(x['symbol'] for x in events)!={'GS','JPM'}:
        raise ValueError('Expected the two verified GS/JPM research cases; do not silently replace')
    return events


def cutoff_et(day,round0):
    deadlines=('09:10','10:25','11:25','12:25','13:25','14:25','15:25')
    return datetime.fromisoformat(f'{pd.Timestamp(day).date()}T{deadlines[round0]}:00').replace(tzinfo=ET)


def allowed_events(policy,events):
    if 'camel' not in policy:return []
    if 'GS_JPM_repeat_sensitivity' in policy:return events
    return [e for e in events if e['symbol']=='GS' and e['first_disclosure']]


def core_sleeve_weights(market,day_before,p30,p37,p38,p42):
    """Predeclared 15% of the equity sleeve in strongest 5-day reversal names.

    Targets all 30 holdings, deliberately unlike the free-floating full dynamic
    reversal comparator. No post-event data is used.
    """
    base=p38.core_weights(p30,p37,market,day_before,'blend80')
    f=p42.day_features(market,day_before)
    z=p42.alphas(f,[] )['reversal']
    chosen=np.argsort(-z,kind='stable')[:5]
    concentrated=np.zeros(30);concentrated[chosen]=float(base.sum())/5
    desired=.85*base+.15*concentrated
    if desired.max()>.10 or abs(desired.sum()-.8)>1e-8:
        raise AssertionError('Invalid predeclared sleeve')
    return desired


def execute_target(cash,shares,price,target,p30,initial=False,band=.03,max_l1=.10):
    nav0=float(cash+shares@price)
    w0=shares*price/nav0
    if not initial:
        distance=float(np.abs(target-w0).sum())
        if distance<band:return cash,shares,0.
        if distance>max_l1: target=w0+(target-w0)*max_l1/distance
    newcash,newshares,gross=p30.trade(cash,shares,price,target)
    if not np.isclose(nav0-(newcash+newshares@price),FEE*gross,atol=.0001):
        raise AssertionError('Cash conservation at target trade')
    return float(newcash),newshares,float(gross)


def partial_event_order(cash,shares,price,event,direction,turn='enter'):
    """Long-only event sleeve; uses cash when positive, never shorts stock.
    Size is 0.3 percentage point of current NAV. For the fixed GS/JPM positives
    buy once, then liquidate only the added fractional shares after five sessions.
    """
    nav=float(cash+shares@price)
    j=event
    if direction=='positive' and turn=='enter':
        max_by_cap=max(0.,.295*nav-shares[j]*price[j])
        # Keep total invested at or under .90 at event entries.
        cap_equity=max(0.,.90*nav-shares@price)
        notional=max(0.,min(.003*nav,cash/(1+FEE),max_by_cap,cap_equity))
        if notional<500:return cash,shares,0.,0.
        newshares=shares.copy();newshares[j]+=notional/price[j]
        newcash=cash-notional*(1+FEE)
        return float(newcash),newshares,notional,float(notional/price[j])
    raise ValueError('Only sourced positive long event orders used for this comparison')


def liquidate_increment(cash,shares,price,j,qty):
    use=min(float(qty),float(shares[j]))
    if use<=0:return cash,shares,0.
    gross=use*price[j];new=shares.copy();new[j]-=use
    return float(cash+gross*(1-FEE)),new,float(gross)


def one_episode(market,ix,pol,events,mods):
    p30,p37,p38,p39,p41,p42=mods
    if len(ix)!=15 or any(int(ix[k+1])!=int(ix[k])+1 for k in range(14)):
        raise AssertionError('Need 15 complete sequential sessions')
    baseline=p38.core_weights(p30,p37,market,int(ix[0])-1,'blend80')
    cash=NAV0;shares=np.zeros(30,float)
    rets=[];valuation=[NAV0];turn=[];trade_rounds=0;fees=0.;holdings=[];exposures=[]
    logs=[];active={};executed=[];symbols={s:i for i,s in enumerate(p30.SYMBOLS)}
    candidate=allowed_events(pol,events)
    for d,i0 in enumerate(ix):
        i=int(i0);before_day=i-1
        for r in range(7):
            price=market.op[i,p30.FILL_IDX[r],:]
            nav_before=float(cash+shares@price)
            trade_notional=0.;action=[]
            if d==0 and r==0:
                if pol=='equal80_hold': w=np.full(30,.8/30)
                elif pol=='dynamic_5d_reversal':
                    f=p42.day_features(market,before_day)
                    groups=p41.sector_indices(p30,p38)
                    score=p42.alphas(f,groups)['reversal']
                    w,elig=p42.weights(score,f,np.zeros(30,bool),p42.exposure(f),groups)
                elif 'reversal_sleeve' in pol:
                    w=core_sleeve_weights(market,before_day,p30,p37,p38,p42)
                else:w=baseline
                if pol=='dynamic_5d_reversal':
                    groups=p41.sector_indices(p30,p38)
                    cash,shares,g,fee,why=p42.trade_target(cash,shares,market.daily[before_day],price,
                                         w,p30,True,'full',groups)
                else:
                    cash,shares,g=execute_target(cash,shares,price,w,p30,initial=True)
                trade_notional+=g;action.append('INITIAL_BUY')
            elif r==0 and pol=='dynamic_5d_reversal' and d%3==0:
                f=p42.day_features(market,before_day)
                groups=p41.sector_indices(p30,p38)
                score=p42.alphas(f,groups)['reversal']
                held=(shares*price/max(cash+shares@price,1.)>.0006)
                w,elig=p42.weights(score,f,held,p42.exposure(f),groups)
                # The Phase42 full engine uses its own .045 gate and .12 round cap.
                cash,shares,g,fee,why=p42.trade_target(cash,shares,market.daily[before_day],price,
                                         w,p30,False,'full',groups)
                trade_notional+=g
                if g>0:action.append('REVERSAL_REBALANCE')
            elif r==0 and 'reversal_sleeve' in pol and d%5==0:
                w=core_sleeve_weights(market,before_day,p30,p37,p38,p42)
                cash,shares,g=execute_target(cash,shares,price,w,p30,
                                             band=.03,max_l1=.10)
                trade_notional+=g
                if g>0:action.append('SLEEVE_REBALANCE')
            # Scheduled exit of an event sleeve after 5 full trading sessions.
            if r==0:
                for key,v in list(active.items()):
                    if d>=v['day_index']+5:
                        cash,shares,g=liquidate_increment(cash,shares,price,v['j'],v['qty'])
                        trade_notional+=g
                        action.append('EVENT_EXIT_'+key);del active[key]
            if candidate:
                deadline=cutoff_et(market.dates[i],r)
                for e in candidate:
                    k=e['event_id']
                    if k in active or k in executed or deadline<e['ready']:
                        continue
                    if deadline.date()!=e['pub'].date():
                        continue
                    j=symbols[e['symbol']]
                    if e['direction']=='positive':
                        cash,shares,g,qty=partial_event_order(cash,shares,price,j,'positive')
                        if g>0:
                            active[k]=dict(j=j,qty=qty,day_index=d)
                            executed.append(k);action.append('RETROSPECTIVE_EVENT_'+e['symbol'])
                            trade_notional+=g
                    # A bearish event would only allow a long-only reduction.
            nav_after=float(cash+shares@price)
            if abs((nav_before-nav_after)-FEE*trade_notional)>.001:
                raise AssertionError('Trade-fee/NAV conservation')
            if cash<-.001 or shares.min()<-1e-10:
                raise AssertionError('Negative cash or short position')
            w=shares*price/nav_after
            if max(w)>0.3000001 or w.sum()>1.0000001:
                raise AssertionError('Official long-only weight breach')
            turn.append(float(trade_notional/nav_before));fees+=FEE*trade_notional
            trade_rounds+=int(trade_notional>1e-6)
            holdings.append(int((w>0.0005).sum()));exposures.append(float(w.sum()))
            # Official M2 valuation: R7 runs to *NEXT* session 09:30,
            # with 16:00 close recorded separately for MDD.
            if r<6:
                endpx=market.op[i,p30.FILL_IDX[r+1],:]
            elif d<14:
                close_nav=float(cash+shares@market.daily[i,:])
                valuation.append(close_nav)  # intermediate M3-only observation
                endpx=market.op[int(ix[d+1]),0,:]
            else:
                endpx=market.daily[i,:]
            nav_end=float(cash+shares@endpx)
            valuation.append(nav_end)
            rets.append(nav_end/nav_before-1.)
            logs.append(dict(policy=pol,day=str(market.dates[i].date()),
                             round=r+1,deadline_et=cutoff_et(market.dates[i],r).isoformat(),
                             nav_before=nav_before,nav_end=nav_end,
                             return_round=rets[-1],traded_notional=trade_notional,
                             official_turnover_round=turn[-1],
                             fee_usd=FEE*trade_notional,exposure=float(w.sum()),
                             holdings=holdings[-1],actions=';'.join(action) or 'HOLD'))
    if len(rets)!=ROUNDS or len(turn)!=ROUNDS or len(valuation)!=1+ROUNDS+14:
        raise AssertionError('Official 105 returns + fourteen extra closes required')
    values=np.array(valuation,float)
    rr=np.array(rets,float)
    # Verify product of the 105 returns equals the beginning-to-end NAV change.
    end_nav=float(logs[-1]['nav_end'])
    if abs(np.prod(1+rr)-end_nav/NAV0)>2e-8:
        raise AssertionError('Compounding failed; overnight period likely duplicated')
    std=float(rr.std(ddof=1))
    result=dict(strategy=pol,
        M1_cumulative_return=end_nav/NAV0-1.,
        M2_annualized_sharpe=(math.sqrt(252*7)*rr.mean()/std if std>1e-14 else 0.),
        M3_max_drawdown=float(np.max(1-values/np.maximum.accumulate(values))),
        M4_avg_round_turnover=float(np.mean(turn)),
        fees_usd=fees,traded_notional_usd=sum(x['traded_notional'] for x in logs),
        trade_rounds=trade_rounds,mean_exposure=float(np.mean(exposures)),
        min_holdings=min(holdings),max_holdings=max(holdings),
        mean_holdings=float(np.mean(holdings)),
        event_entries=int(sum('RETROSPECTIVE_EVENT' in x['actions'] for x in logs)),
        event_exits=int(sum('EVENT_EXIT' in x['actions'] for x in logs)),
        cutoff_convention='R7_TO_NEXT_R1; 16:00_EXTRA_MDD_ONLY',
    )
    return result,logs


def run(bars,predictions,out,stride=15):
    mods=modules();p30,p37,p38,p39,p41,p42=mods
    market=p30.read_market(Path(bars),min_last_day='2026-10-06')
    episodes=p39.episode_groups(market,stride=stride)
    events=evidence_predictions(predictions,p30.SYMBOLS)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    rows=[];logs=[]
    for z,(ep,segment,ix) in enumerate(episodes,1):
        for pol in POLICIES:
            values,history=one_episode(market,ix,pol,events,mods)
            rows.append(dict(episode=ep,segment=segment,
                start=str(market.dates[ix[0]].date()),end=str(market.dates[ix[-1]].date()),**values))
            logs.extend([dict(episode=ep,segment=segment,**x) for x in history])
        print('P44_EPISODE',z,'OF',len(episodes),ep,'COMPLETE',flush=True)
    ep=pd.DataFrame(rows)
    periods=pd.DataFrame(logs)
    metric=['M1_cumulative_return','M2_annualized_sharpe','M3_max_drawdown','M4_avg_round_turnover']
    summary=ep.groupby('strategy',as_index=False).agg(
        episodes=('episode','size'),mean_M1=(metric[0],'mean'),mean_M2=(metric[1],'mean'),
        mean_M3=(metric[2],'mean'),mean_M4=(metric[3],'mean'),
        mean_exposure=('mean_exposure','mean'),mean_holdings=('mean_holdings','mean'),
        min_holdings=('min_holdings','min'),max_holdings=('max_holdings','max'),
        event_entries=('event_entries','sum'),event_exits=('event_exits','sum'))
    pairs=[]
    by=ep.set_index(['episode','strategy'])
    for _,r in ep.iterrows():
        if r.strategy=='equal80_hold':continue
        for reference in ('equal80_hold','blend80_hold'):
            if r.strategy==reference:continue
            base=by.loc[(r.episode,reference)]
            pairs.append(dict(episode=r.episode,segment=r.segment,
              strategy=r.strategy,reference=reference,
              **{'delta_'+m:float(r[m]-base[m]) for m in metric}))
    paired=pd.DataFrame(pairs)
    eventaudit=[]
    for ev in events:
        first=cutoff_et(pd.Timestamp(ev['pub'].date()),0)
        first_round=next((i+1 for i in range(7)
                          if cutoff_et(pd.Timestamp(ev['pub'].date()),i)>=ev['ready']),None)
        eventaudit.append(dict(symbol=ev['symbol'],accession=ev['accession'],
             true_sec_acceptance_et=ev['pub'].isoformat(),
             hypothetical_available_et=ev['ready'].isoformat(),
             first_eligible_round=first_round,
             counted_in_primary=int(ev['first_disclosure']),
             reason='FIRST_DISCLOSURE_CANDIDATE' if ev['first_disclosure'] else
                 'REPEATED_PRESENTATION_AFTER_EARLIER_06:30_ET_EARNINGS_RELEASE',
             retrospective_actual_inference_utc=ev['research_inferred_at'],
             analyst_expectations_known=ev['expectation_known'],
             source_hash=ev['evidence_sha256']))
    for filename,data in [('phase44_episode_M1_M4.csv',ep),('phase44_policy_mean_M1_M4.csv',summary),
        ('phase44_paired_M1_M4.csv',paired),('phase44_all_105_rounds.csv',periods),
        ('phase44_event_provenance.csv',pd.DataFrame(eventaudit))]:
        data.to_csv(folder/filename,index=False)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest=dict(research_only=True,official_deployment=False,episodes=len(episodes),
      episode_stride=stride,policies=list(POLICIES),
      total_decisions=len(periods),total_model_calls=0,
      market_input=str(Path(bars).resolve()),market_sha256=digest(bars),
      event_input=str(Path(predictions).resolve()),event_sha256=digest(predictions),
      original_SEC_times_verified_from_organizer_agnostic_SEC_archives=True,
      retrospective_LLM_training_memory_uncontrolled=True,
      source_observed_in_October_not_July=True,
      important_limitations=[
         'six previously examined episodes; NOT out of sample',
         'historical CaMeL model output generated in Oct; not actual July execution',
         '60-minute source processing delay is hypothetical',
         'no verified July analyst consensus surprise or genuine novelty check for GS',
         'JPM later presentation is excluded from primary signal',
         'IEX 15-minute open and close prices are not organizer execution prices',
         'missing sessions filtered by Phase30/39; full market holiday calendar not verified',
         'market simulations ignore wider market impact and liquidity constraints',
         'returns are not rankings against entrants',
      ])
    (folder/'phase44_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('P44_RESEARCH_COMPLETE_NO_NETWORK_NO_LLM_NO_OFFICIAL')
    print('P44_COUNTS',len(episodes),'EPISODES',len(POLICIES),'POLICIES',len(periods),'DECISIONS')
    print('P44_OUTPUT',folder)
    print(summary[['strategy','episodes','mean_M1','mean_M2','mean_M3','mean_M4','event_entries']].to_string(index=False))
    return ep,summary,paired,periods


def self_test():
    assert abs(math.sqrt(252*7)-42.0)<1e-12
    assert len(POLICIES)==7
    import tempfile
    # Verify correct 7-round deadline mapping including daylight saving.
    assert cutoff_et(pd.Timestamp('2026-07-14'),0).astimezone(timezone.utc).hour==13
    assert cutoff_et(pd.Timestamp('2026-07-14'),2).astimezone(timezone.utc).hour==15
    assert datetime.fromisoformat(FILING_ET['0000886982-26-000294'][1]).hour==7
    # Long-only, cash-first fee conservation for incremental buys and exits.
    p=np.full(30,100.);h=np.full(30,2.);cc=940000.
    c,s,g,qty=partial_event_order(cc,h,p,7,'positive')
    assert g>0 and qty>0 and c>=0 and s[7]>h[7]
    assert abs((cc+h@p)-(c+s@p)-g*FEE)<1e-5
    c2,s2,g2=liquidate_increment(c,s,p,7,qty)
    assert abs((c+s@p)-(c2+s2@p)-g2*FEE)<1e-5
    assert abs(s2[7]-h[7])<1e-10
    # A daily R7 evaluation includes overnight, with no additional Sharpe return
    # at a 16:00 intermediate MDD-only mark.
    pre=1000.;sixteen=900.;nextopen=1100.
    assert abs((nextopen/pre-1)-.10)<1e-9
    assert abs(max(1-np.array([pre,sixteen,nextopen])/np.maximum.accumulate([pre,sixteen,nextopen]))-.10)<1e-12
    print('P44_SELF_TEST_PASS')
    print('M1_M2_M3_M4_R7_OVERNIGHT_MDD_EXTRA_CLOSE_AND_LONG_ONLY_CASH_PASS')
    print('ZERO_NETWORK_ZERO_LLM_ZERO_ORGANIZER_ZERO_SUBMISSION')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--bars',default='artifacts/phase29_alpaca_refresh/alpaca_iex_15m_through_2026-10-09.csv')
    ap.add_argument('--predictions',default='artifacts/phase43_camel_research_batch_v3/retrospective_predictions.json')
    ap.add_argument('--out',default='artifacts/phase44_final_four_metric_comparison')
    ap.add_argument('--stride',type=int,choices=(15,5),default=15)
    args=ap.parse_args()
    try:
        if args.self_test:self_test()
        if args.run:run(args.bars,args.predictions,args.out,args.stride)
        if not args.run and not args.self_test:ap.print_help()
    except Exception as e:
        print('P44_FAILED',type(e).__name__,str(e),file=sys.stderr)
        sys.exit(2)
