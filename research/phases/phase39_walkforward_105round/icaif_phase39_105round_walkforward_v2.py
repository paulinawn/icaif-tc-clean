#!/usr/bin/env python3
"""ICAIF Phase 39: offline, 15 sessions x 7 decisions, research only.

Broad non-overlapping 2026 episodes. Fixed and causal adaptive FIRST-DAY staging.
Strictly timestamped CaMeL paper events accepted only if independently verified;
there is no historical CaMeL backfill. Never connects to organizer or LLMs.

Requires Phase30 v2, Phase37, and Phase38 next to this file.
Requires user's refreshed Alpaca IEX CSV in starter-kit for real experiment.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path.cwd()
OUTPUT=Path('artifacts/phase39_105round_walkforward')
NAV0=1_000_000.
FEE=.001
SLOTS=105
TARGET=.80
START=.30
THRESHOLD=.0030  # 0.30% intraday basket return, predeclared
STRONG_ACCEL=.15
WEAK_DELAY=.10
MIN_BUY_USD=50.
MAX_OVERLAY_ABS=.003
SEGS=(('early_2026','2026-01-02','2026-06-30'),
      ('examined_jul_aug','2026-07-01','2026-08-31'),
      ('examined_sep_oct','2026-09-01','2026-10-09'))

@dataclass(frozen=True)
class Strategy:
    name:str
    core:str
    entry:str='immediate'  # immediate, fixed7, accel_trend, accel_contrarian
    use_camel:bool=False


def specs():
    output=[Strategy('equal60_hold','equal60'),
            Strategy('equal80_hold','equal80'),
            Strategy('blend80_hold','blend80'),
            Strategy('lowvol80_legacy_hold','lowvol80_legacy')]
    for core in ('equal80','blend80','lowvol80_legacy'):
        for entry in ('fixed7','accel_trend','accel_contrarian'):
            name=f'{core}_{entry}'
            output.extend((Strategy(name,core,entry),Strategy(name+'__camel',core,entry,True)))
        output.append(Strategy(f'{core}_event_only__camel',core,'immediate',True))
    return tuple(output)


def dependencies():
    import importlib.util
    basedir=Path(__file__).resolve().parent
    modules=[]
    for label,filename in (('p39_p30','icaif_phase30_2026_walkforward_agent_lab_v2.py'),
                           ('p39_p37','icaif_phase37_risk_regime_tournament.py'),
                           ('p39_p38','icaif_phase38_105slot_camel_staging.py')):
        loc=basedir/filename
        if not loc.exists():raise FileNotFoundError('Place this dependency in Downloads beside Phase39: '+filename)
        spec=importlib.util.spec_from_file_location(label,loc)
        module=importlib.util.module_from_spec(spec)
        sys.modules[label]=module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def first_day_exposure(entry,round_index,market=None,day_index=None,p30=None,previous=0):
    """Target equity weight known BEFORE this round's submission deadline.

    Momentum acceleration: bring purchases forward when broad same-day return
    is above +0.30%, delay when below -0.30%. Contrarian does the reverse.
    Triggers refer only to fully completed 15-minute bars. Every staging
    schedule finishes near 80% at R7; targets never reverse early purchases.
    """
    if entry=='immediate':return TARGET
    if entry not in ('fixed7','accel_trend','accel_contrarian'):raise ValueError('Unknown entry')
    if round_index==0:return START
    if round_index==6:return TARGET
    normal=START+(TARGET-START)*round_index/6.
    bias=0.
    if entry!='fixed7':
        if market is None or p30 is None or day_index is None:raise ValueError('Market needed')
        known=p30.KNOWN_IDX[round_index]
        if known is None:raise AssertionError('Intraday bar not known')
        intraday=np.log(market.cl[day_index,known,:]/market.op[day_index,0,:])
        avg=float(intraday.mean())
        positive=avg>=THRESHOLD
        negative=avg<=-THRESHOLD
        if entry=='accel_trend':
            bias=STRONG_ACCEL if positive else -WEAK_DELAY if negative else 0.
        else:
            bias=STRONG_ACCEL if negative else -WEAK_DELAY if positive else 0.
    return min(TARGET,max(previous,normal+bias))


def buy_only_to_target(cash,shares,fill,weights):
    """Buy deficits only, do not sell existing shares in first-day entry stages.

    All purchases and fees are paid from actual cash; small drift differences
    are not converted into unnecessary round-trip stock reallocations.
    """
    weights=np.asarray(weights,float)
    if weights.ndim!=1 or not np.isfinite(weights).all() or weights.min()<0 or weights.max()>.30+1e-12 or weights.sum()>1+1e-10:
        raise ValueError('Illegal weights')
    pre=cash+float(np.dot(shares,fill))
    old=shares*fill
    after=pre
    for _ in range(100):
        amount=np.maximum(0,weights*after-old)
        update=pre-FEE*float(amount.sum())
        if abs(update-after)<1e-9:break
        after=update
    amount=np.maximum(0,weights*after-old)
    gross=float(amount.sum())
    if gross<MIN_BUY_USD:return cash,shares.copy(),0.,0.
    cost=gross*(1+FEE)
    if cost>cash:
        # Preserve legality: no borrowed cash or leverage.
        scale=max(0.,(cash-1e-7)/cost)
        amount=amount*scale;gross=float(amount.sum());cost=gross*(1+FEE)
    new_cash=cash-cost
    new_shares=shares+amount/fill
    if new_cash<-1e-4 or new_shares.min()<-1e-10:raise AssertionError('Invalid cash/position')
    return max(0.,new_cash),new_shares,gross,FEE*gross


def episode_groups(market,stride=15,starting='2026-01-02',minimum_history=121):
    """No 15-day window is allowed to jump over a documented incomplete day.

    The 15-day requirement is the number of exchange sessions as present in
    the raw price records; the source does not independently certify a full
    official exchange calendar. All completed sessions have 26 bars x30.
    """
    if stride not in (5,15):raise ValueError('stride must be 5 or 15')
    missing=set()
    if market.missing is not None and not market.missing.empty:
        missing=set(pd.DatetimeIndex(pd.to_datetime(market.missing['date'], errors='raise')).normalize())
    frames=[];short=0;skipped=0
    for segment,s,e in SEGS:
        s=max(pd.Timestamp(s),pd.Timestamp(starting))
        index=np.flatnonzero((market.dates>=s)&(market.dates<=pd.Timestamp(e)) &
                            (np.arange(len(market.dates))>minimum_history))
        if len(index)<15:continue
        run=[];runs=[]
        for i in index:
            if run:
                a=market.dates[int(run[-1])]
                b=market.dates[int(i)]
                if any(a<d<b for d in missing):
                    runs.append(run);run=[];skipped+=1
            run.append(int(i))
        if run:runs.append(run)
        for group in runs:
            if len(group)<15:short+=len(group)
            for j in range(0,len(group)-14,stride):
                ix=np.array(group[j:j+15],int)
                if len(ix)!=15:raise AssertionError
                frames.append((f'{segment}_{market.dates[ix[0]].date()}',segment,ix))
    print('PHASE39_WINDOWS:',len(frames),'GAP_SPLITS:',skipped,'SHORT_RUN_DAYS:',short,
          'MIN_PRE_EPISODE_FEATURE_HISTORY:',minimum_history,flush=True)
    return frames


def read_events(path,p38,p30):
    if not path:return []
    ev=p38.load_events(path,p30.SYMBOLS)
    # SOURCE REVIEW is itself information: no replay before attestation.
    approved=[]
    for e in ev:
        if not str(e.get('source_url','')).startswith('https://'):continue
        if not e.get('manual_review_attested_at_utc'):continue
        reviewed=p38.parse_time(e['manual_review_attested_at_utc'])
        if reviewed<e['_available']:continue
        e=dict(e,_available=reviewed)
        approved.append(e)
    return approved


def simulate(market,index,policy,events,p30,p37,p38):
    """Each 15-day episode resets NAV and evaluates ALL 105 decision slots."""
    if len(index)!=15 or np.any(np.diff(index)!=1):raise AssertionError('Bad episode length')
    base=p38.core_weights(p30,p37,market,int(index[0])-1,policy.core)
    cash=NAV0;shares=np.zeros(30);nav=[NAV0];turn=[];logs=[];fee_total=0.;ntrades=0;events_used=set()
    symbols={x:k for k,x in enumerate(p30.SYMBOLS)}
    prev_firstday_exposure=0.
    for d,i in enumerate(index):
        i=int(i)
        for r in range(7):
            fill=market.op[i,p30.FILL_IDX[r],:]
            mark=market.daily[i-1] if r==0 else market.cl[i,p30.KNOWN_IDX[r],:]
            marked_nav=float(cash+np.dot(shares,mark))
            before_fill=float(cash+np.dot(shares,fill))
            before_weight=shares*mark/max(marked_nav,1e-12)
            reason='HOLD';gross=0.;fee=0.;event_id=None
            if d==0 and (r==0 or policy.entry!='immediate'):
                requested=(float(base.sum()) if policy.entry=='immediate' else
                           first_day_exposure(policy.entry,r,market,i,p30,prev_firstday_exposure))
                prev_firstday_exposure=requested
                desired=base*(requested/max(float(base.sum()),1e-12))
                cash,shares,gross,fee=buy_only_to_target(cash,shares,fill,desired)
                reason='INITIAL_BUY' if r==0 else 'STAGED_BUY' if gross>=MIN_BUY_USD else 'HOLD_STAGING_ALREADY_FILLED'
            if policy.use_camel and events:
                at=p38.decision_time(market.dates[i],r)
                eligible=[e for e in events if e['event_id'] not in events_used and
                          e['_available']<=at and
                          (timedelta(0)<=at-e['_published']<=timedelta(hours=4) or
                           (e['_published'].astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).hour>=15
                            and r==0 and timedelta(0)<=at-e['_published']<=timedelta(hours=22)))]
                if eligible:
                    e=eligible[-1];events_used.add(e['event_id'])
                    j=symbols[e['symbol']]
                    direction=1 if e['direction']=='positive' else -1
                    now_nav=float(cash+np.dot(shares,fill))
                    cur=shares*fill/now_nav
                    # A small, zero-sum relative tilt on the current portfolio,
                    # with no forced market exposure changes.
                    shift=np.full(30,-direction*MAX_OVERLAY_ABS/29.)
                    shift[j]=direction*MAX_OVERLAY_ABS
                    w=cur+shift
                    if w.min()>-1e-12 and w.max()<.30 and w.sum()<=1.0000000001:
                        prev_shares=shares.copy();prev_cash=cash
                        cash,shares,additional=p30.trade(cash,shares,fill,p38.check_w(w))
                        gross+=additional;fee+=FEE*additional
                        event_id=e['event_id'];reason='CAUSAL_CAMEL_PAPER_TILT'
                        if abs((cash+shares@fill)-(prev_cash+prev_shares@fill)+FEE*additional)>1e-5:
                            raise AssertionError('Event fee conservation error')
                    else:reason='HOLD_CAMEL_RISK_REJECT'
            equity_post=float(shares@fill)
            post_nav=float(cash+equity_post)
            if post_nav<=0 or not np.isfinite(post_nav):raise AssertionError('Invalid NAV')
            post_exposure=equity_post/post_nav
            if post_exposure>1+1e-9 or (shares*fill/post_nav).max()>.30+1e-8:
                raise AssertionError('Portfolio constraints exceeded')
            if gross>1e-6:
                ntrades+=1;fee_total+=fee;decision='TRADE'
            else:decision='HOLD'
            round_turn=gross/max(before_fill,1e-12);turn.append(round_turn)
            endpx=market.op[i,p30.END_VAL_IDX[r],:] if r<6 else market.cl[i,25,:]
            endnav=float(cash+shares@endpx)
            nav.append(endnav)
            logs.append({'day':str(market.dates[i].date()),'round':r+1,
                'deadline_et':p38.DEADLINE_ET[r],'execution_et':p38.FILL_ET[r],
                'info_cutoff':('prior_close' if r==0 else p30.SLOTS[p30.KNOWN_IDX[r]]),
                'strategy':policy.name,'decision':decision,'reason':reason,
                'requested_entry_exposure':prev_firstday_exposure if d==0 else None,
                'predecision_equity_weight':float(before_weight.sum()),
                'post_execution_equity_weight':post_exposure,
                'gross_trade_usd':gross,'fee_usd':fee,
                'round_turnover':round_turn,'camel_event_id':event_id,
                'nav_end':endnav})
    if len(logs)!=105:raise AssertionError('Missing round decisions')
    v=np.array(nav);ret=v[1:]/v[:-1]-1
    sig=float(ret.std(ddof=1))
    out={'cumulative_return':float(v[-1]/NAV0-1),
         'sharpe_proxy':float(ret.mean()*math.sqrt(252*7)/sig) if sig>1e-12 else 0.,
         'mdd_proxy':float(np.max(1-v/np.maximum.accumulate(v))),
         'turnover_proxy':float(np.mean(turn)),
         'fees_usd':fee_total,'trade_count':ntrades,
         'camel_event_trades':sum(bool(x['camel_event_id']) for x in logs),
         'avg_equity_exposure':float(np.mean([x['post_execution_equity_weight'] for x in logs])),
         'rounds':105}
    return out,logs


def add_four_metric_ranks(df):
    """Internal episode percentile ranks using NON-CaMeL candidates only.

    Identical +CaMeL duplicates must never alter opponent ranks. This is NOT
    the competition's unknown competitor-relative percentile distribution.
    """
    out=df.copy()
    baseline=out[~out.strategy.str.endswith('__camel')]
    for col,small in (('cumulative_return',False),('sharpe_proxy',False),
                      ('mdd_proxy',True),('turnover_proxy',True)):
        rc='rank_'+col
        out[rc]=np.nan
        out.loc[baseline.index,rc]=baseline.groupby('episode')[col].rank(
            pct=True,method='average',ascending=small)
        for episode,group in out[out.strategy.str.endswith('__camel')].groupby('episode'):
            b=baseline[baseline.episode==episode][col].to_numpy(float)
            if len(b)==0:raise AssertionError('No ranking benchmark')
            for ridx,row in group.iterrows():
                v=float(row[col]);better=int(np.sum(b<v) if small else np.sum(b>v))
                equal=int(np.sum(np.isclose(b,v,rtol=0,atol=1e-12)))
                out.at[ridx,rc]=(better+(equal+1)/2)/len(b)
    out['four_metric_research_rank']=out[['rank_'+z for z in
         ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]].mean(axis=1)
    return out


def paired_comparison(df):
    lookup=df.set_index(['episode','strategy'])
    rows=[]
    for spec in specs():
        if spec.name.endswith('__camel'):
            target=spec.name[:-7]
            if target.endswith('_event_only'):
                core={'equal80':'equal80_hold','blend80':'blend80_hold',
                      'lowvol80_legacy':'lowvol80_legacy_hold'}[spec.core]
            else:core=target
        elif spec.entry!='immediate':
            core={'equal80':'equal80_hold','blend80':'blend80_hold',
                  'lowvol80_legacy':'lowvol80_legacy_hold'}[spec.core]
        else:continue
        for ep in df.episode.unique():
            a=lookup.loc[(ep,spec.name)];b=lookup.loc[(ep,core)]
            rows.append({'episode':ep,'segment':a.segment,'strategy':spec.name,
                'reference':core,'delta_return_pp':100*(a.cumulative_return-b.cumulative_return),
                'delta_sharpe':a.sharpe_proxy-b.sharpe_proxy,
                'delta_mdd_pp':100*(a.mdd_proxy-b.mdd_proxy),
                'delta_turnover':a.turnover_proxy-b.turnover_proxy,
                'extra_fee_usd':a.fees_usd-b.fees_usd,'extra_trades':a.trade_count-b.trade_count,
                'event_trades':a.camel_event_trades})
    detail=pd.DataFrame(rows)
    summary=(detail.groupby(['segment','strategy','reference'],as_index=False)
        .agg(episodes=('episode','size'),mean_return_delta_pp=('delta_return_pp','mean'),
             median_return_delta_pp=('delta_return_pp','median'),
             positive_episode_fraction=('delta_return_pp',lambda x:(x>0).mean()),
             mean_sharpe_delta=('delta_sharpe','mean'),
             mean_mdd_delta_pp=('delta_mdd_pp','mean'),
             mean_turnover_delta=('delta_turnover','mean'),
             mean_extra_fee_usd=('extra_fee_usd','mean'),
             mean_extra_trades=('extra_trades','mean'),
             total_camel_trades=('event_trades','sum')))
    return detail,summary


def hashfile(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(input_csv,stride,events_path,output=OUTPUT):
    p30,p37,p38=dependencies()
    source=Path(input_csv) if input_csv else p30.INPUT
    if not source.is_absolute():source=ROOT/source
    source=source.resolve()
    market=p30.read_market(source,min_last_day='2026-10-09')
    episodes=episode_groups(market,stride)
    if not episodes:raise ValueError('No 15-day continuous episodes; inspect missing data')
    ev=read_events(events_path,p38,p30)
    rows=[];decision_rows=[]
    for n,(eid,seg,ix) in enumerate(episodes,1):
        for policy in specs():
            stat,dec=simulate(market,ix,policy,ev,p30,p37,p38)
            rows.append(dict(episode=eid,segment=seg,start_date=str(market.dates[ix[0]].date()),
                end_date=str(market.dates[ix[-1]].date()),strategy=policy.name,**stat))
            decision_rows.extend([dict(episode=eid,segment=seg,**x) for x in dec])
        print('PHASE39_EPISODE:',n,'/',len(episodes),eid,'POLICIES:',len(specs()),flush=True)
    result=add_four_metric_ranks(pd.DataFrame(rows))
    decisions=pd.DataFrame(decision_rows)
    detail,summary=paired_comparison(result)
    outcome=(result.groupby(['segment','strategy'],as_index=False)
             .agg(episodes=('episode','size'),mean_return=('cumulative_return','mean'),
                  median_return=('cumulative_return','median'),mean_sharpe=('sharpe_proxy','mean'),
                  mean_mdd=('mdd_proxy','mean'),mean_turnover=('turnover_proxy','mean'),
                  mean_fees_usd=('fees_usd','mean'),mean_trades=('trade_count','mean'),
                  mean_camel_trades=('camel_event_trades','mean'),
                  mean_four_metric_research_rank=('four_metric_research_rank','mean')))
    activity=(decisions.assign(executed=decisions.decision.eq('TRADE').astype(int))
        .groupby(['segment','strategy','round'],as_index=False)
        .agg(decision_slots=('executed','size'),trades=('executed','sum'),
             fees_usd=('fee_usd','sum'),gross_notional_usd=('gross_trade_usd','sum')))
    out=ROOT/output;out.mkdir(exist_ok=True,parents=True)
    result.to_csv(out/'phase39_episode_metrics.csv',index=False)
    summary.to_csv(out/'phase39_paired_summary.csv',index=False)
    detail.to_csv(out/'phase39_paired_episode_deltas.csv',index=False)
    outcome.to_csv(out/'phase39_policy_summary.csv',index=False)
    activity.to_csv(out/'phase39_round_activity.csv',index=False)
    decisions.to_csv(out/'phase39_105_round_audit.csv',index=False)
    manifest={'stage':39,'research_only':True,'official_trades':0,'llm_calls':0,
        'algo':'immediate / fixed7 / trend-accelerated / contrarian-accelerated',
        'first_day_start_exposure':START,'final_target_exposure':TARGET,
        'trigger_log_return_threshold':THRESHOLD,'acceleration':STRONG_ACCEL,
        'delay':WEAK_DELAY,'code_sha256':hashfile(Path(__file__).resolve()),
        'dependency_sha256':{name:hashfile(Path(__file__).resolve().parent/name) for name in
             ('icaif_phase30_2026_walkforward_agent_lab_v2.py',
              'icaif_phase37_risk_regime_tournament.py',
              'icaif_phase38_105slot_camel_staging.py')},
        'input_sha256':hashfile(source),
        'last_complete_data_date':str(market.dates[-1].date()),
        'incomplete_sessions':len(market.missing),'minimum_feature_history_days':121,'episode_days':15,'rounds_per_episode':105,
        'stride':stride,'overlap':stride<15,'episodes':len(episodes),
        'policy_count':len(specs()),'unpaired_reference_count':sum(not p.use_camel for p in specs()),
        'rank_scope':'internal candidate-relative per 15-session episode, not actual competition rank',
        'eligible_historical_camel_events':len(ev),
        'executed_camel_paper_trades':int(decisions.camel_event_id.notna().sum()),
        'forward_only_rule':'No signal enters before source publication, observation, CaMeL completed time and human review.',
        'limitations':['Already viewed Jul-Oct 2026 cannot be a pristine holdout',
            'Earlier 2026 is retrospective, not prospective proof',
            'Incomplete market sessions and half-days are not imputed',
            'IEX 15-minute prices are unadjusted and execution is a proxy',
            'Round Sharpe annualized from only 15 sessions and is noisy',
            'There is no historical LLM alpha absent contemporaneous model calls',
            'Sector buckets in legacy lowvol are not true GICS',
            'First-day buy-only staging can finish slightly off 80% due to intraday drift']}
    (out/'phase39_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('=== PHASE39 PAIRED MEAN RETURN DIFFERENCES (PERCENTAGE POINTS) ===',flush=True)
    print(summary[['segment','strategy','episodes','mean_return_delta_pp','mean_mdd_delta_pp','mean_extra_fee_usd']].to_string(index=False),flush=True)
    print('=== PHASE39 FOUR METRIC RESEARCH RANKS AMONG NON-CAMEL POLICIES ===',flush=True)
    print(outcome[~outcome.strategy.str.endswith('__camel')][['segment','strategy','episodes',
          'mean_return','mean_sharpe','mean_mdd','mean_turnover',
          'mean_four_metric_research_rank']].sort_values(['segment','mean_four_metric_research_rank']).to_string(index=False),flush=True)
    print('PHASE39_RESULT_DIR:',out.resolve(),flush=True)
    print('PHASE39_COMPLETE_RESEARCH_ONLY_NO_ORGANIZER_NO_LLM_CALLS')
    return result,summary,decisions


def self_test():
    from types import SimpleNamespace
    rng=np.random.default_rng(391111)
    names=tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
    days=260
    px=100*np.exp(np.cumsum(rng.normal(0,.009,size=(days,30)),axis=0))
    # 15m bars following a realistic continuous path, not 26 identical prices.
    increments=rng.normal(0,.001,size=(days,26,30))
    op=px[:,None,:]*np.exp(np.cumsum(increments,axis=1));cl=op*np.exp(.0001)
    date=pd.bdate_range('2025-12-01',periods=days)
    market=SimpleNamespace(dates=date,op=op,cl=cl,daily=cl[:,-1,:],missing=pd.DataFrame(columns=['date']))
    class P:
        SYMBOLS=names
        FILL_IDX=(0,4,8,12,16,20,24)
        KNOWN_IDX=(None,2,6,10,14,18,22)
        END_VAL_IDX=(4,8,12,16,20,24,25)
        SLOTS=[(pd.Timestamp('2026-10-07 09:30')+pd.Timedelta(minutes=15*j)).strftime('%H:%M') for j in range(26)]
        @staticmethod
        def trade(cash,shares,fill,weights):
            old=shares*fill;before=cash+shares@fill;after=before
            for _ in range(70):
                amt=float(np.abs(after*weights-old).sum())
                new=before-.001*amt
                if abs(new-after)<1e-9:break
                after=new
            amt=float(np.abs(after*weights-old).sum())
            return after*(1-weights.sum()),weights*after/fill,amt
    class P37: pass
    class P38:
        @staticmethod
        def core_weights(p,p37,m,t,which):
            return np.full(30,(.60 if which=='equal60' else .80)/30.)
        @staticmethod
        def check_w(x):return np.maximum(np.asarray(x),0.)
        @staticmethod
        def decision_time(date,r):
            return pd.Timestamp(f'{pd.Timestamp(date).date()} {("09:10","10:25","11:25","12:25","13:25","14:25","15:25")[r]}',tz='America/New_York').tz_convert('UTC').to_pydatetime()
        MAX_EVENT_AGE_MINUTES=240
        DEADLINE_ET=('09:10','10:25','11:25','12:25','13:25','14:25','15:25')
        FILL_ET=('09:30','10:30','11:30','12:30','13:30','14:30','15:30')
    # Initial entry and no-event CaMeL counterpart parity.
    ep=np.arange(98,113)
    for pol in specs():
        a,logs=simulate(market,ep,pol,[],P,P37,P38)
        assert len(logs)==105 and a['rounds']==105
        assert abs(sum(z['fee_usd'] for z in logs)-a['fees_usd'])<1e-6
        assert all(x['post_execution_equity_weight']<=1+1e-10 for x in logs)
        if pol.name.endswith('__camel') and not pol.name.endswith('_event_only__camel'):
            b,_=simulate(market,ep,Strategy(pol.name[:-7],pol.core,pol.entry),[],P,P37,P38)
            for col in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd'):
                assert abs(a[col]-b[col])<1e-11
        if pol.name=='equal80_hold':
            assert a['trade_count']==1
        if pol.name=='equal80_fixed7':
            assert a['trade_count']==7
            assert np.allclose([z['requested_entry_exposure'] for z in logs[:7]],np.linspace(.3,.8,7))
        if pol.entry!='immediate':
            assert all(z['post_execution_equity_weight']>=0 for z in logs[:7])
    # Regression: the REAL Alpaca loader returns a DataFrame of incomplete days.
    # A pandas Series has no .normalize(); use a DatetimeIndex and verify
    # missing sessions split an episode rather than silently jumping the gap.
    absent=pd.Timestamp('2026-03-16')
    keep=(market.dates!=absent)
    market_gap=SimpleNamespace(dates=market.dates[keep],op=market.op[keep],
                               cl=market.cl[keep],daily=market.daily[keep],
                               missing=pd.DataFrame({'date':[str(absent.date())],
                                                     'missing_15min_fields':[2]}))
    gap_episodes=episode_groups(market_gap,stride=15,minimum_history=3)
    assert len(gap_episodes)>0
    assert all(not (market_gap.dates[ix[0]]<absent<market_gap.dates[ix[-1]])
               for _,_,ix in gap_episodes), 'A 15-session episode crossed an incomplete day'
    print('PHASE39_INCOMPLETE_SESSION_REGRESSION_PASS',flush=True)
    # Causality: changing bars not completed by the deadline cannot change schedule.
    for r in (1,2,3,4,5):
        target=first_day_exposure('accel_trend',r,market,98,P,0)
        altered=SimpleNamespace(op=op.copy(),cl=cl.copy())
        altered.cl[98,P.KNOWN_IDX[r]+1:,:]*=100
        assert first_day_exposure('accel_trend',r,altered,98,P,0)==target
    # Predeclared trend and counter-trend react opposite to a strong known positive move.
    clone=SimpleNamespace(op=op.copy(),cl=cl.copy())
    clone.cl[98,P.KNOWN_IDX[2],:]=clone.op[98,0,:]*1.025
    assert first_day_exposure('accel_trend',2,clone,98,P,0)>first_day_exposure('accel_contrarian',2,clone,98,P,0)
    # Future manual approval never backdates the signal to the LLM response.
    synthetic_event_time=pd.Timestamp(f'{date[98].date()} 11:20',tz='America/New_York').tz_convert('UTC').to_pydatetime()
    event={'event_id':'synth39','symbol':'AAPL','direction':'positive',
           '_published':synthetic_event_time-timedelta(minutes=25),
           '_available':synthetic_event_time}
    cstats,clog=simulate(market,ep,Strategy('test_event','equal80','immediate',True),[event],P,P37,P38)
    assert cstats['camel_event_trades']==1
    assert all(row['camel_event_id'] is None for row in clog[:2])
    assert any(row['camel_event_id']=='synth39' for row in clog[2:])
    # Hold parity with Phase38 on identical synthetic prices/15-day window.
    p30,p37,p38=dependencies()
    # Build synthetic test variant for Phase38 with actual dependency classes.
    market2=SimpleNamespace(dates=date,op=op,cl=cl,daily=cl[:,-1,:],missing=market.missing)
    for core,label in (('equal60','equal60_hold'),('equal80','equal80_hold')):
        phase38stats,_=p38.simulate(p30,p37,market2,ep,p38.Policy(label,core),[])
        own,_=simulate(market2,ep,Strategy(label,core),[],p30,p37,p38)
        for col in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd'):
            assert abs(own[col]-phase38stats[col])<1e-8,('Phase38 parity fail',label,col,own[col],phase38stats[col])
    print('PHASE39_SELF_TEST_PASS')
    print('15_DAYS_105_ROUNDS_BUY_ONLY_STAGING_CAUSAL_INTRADAY_FEES_AND_STATIC_BASELINE_PARITY_PASS')
    print('NO_NETWORK_NO_LLM_NO_OFFICIAL_DEPLOYMENT_OR_SUBMISSION')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--input',default=None)
    parser.add_argument('--stride',type=int,choices=(5,15),default=15)
    parser.add_argument('--camel-events',default=None)
    args=parser.parse_args()
    if not (args.self_test or args.run):parser.error('Choose --self-test or --run')
    try:
        if args.self_test:self_test()
        if args.run:run(args.input,args.stride,args.camel_events)
    except Exception as exc:
        import traceback
        print('PHASE39_FAILED:',type(exc).__name__,str(exc),file=sys.stderr)
        traceback.print_exc()
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
