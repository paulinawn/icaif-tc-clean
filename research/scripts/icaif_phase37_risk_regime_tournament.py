#!/usr/bin/env python3
"""ICAIF Phase 37: OFFLINE risk/exposure tournament, no organizer/network/LLM calls.

Hypothesis: cost-aware 80% equity core with modest low-volatility stock selection
and/or predeclared market stress de-risking can outperform static 60/80 cores on
risk/return/drawdown/turnover. NO alpha or leaderboard improvements are assumed.

Data: Phase29 Alpaca IEX 15m 2025/26, via Phase30 v2 read_market.
Competition constraints modeled: 30 long-only symbols, <=30% each, sum<=1,
fee=0.1% buy+sell, seven daily periods. Historical raw IEX (no dividends/
corporate-action adjustment) is only a price/fill proxy. No API calls.

All regime and low-vol inputs end no later than previous trading day's close.
Test periods: Jul-Aug development and ALREADY-EXPOSED Sep-Oct9 diagnostic;
independent resets. Contiguous Jul-Oct is a descriptive sensitivity only.
This tool is research-only and NEVER creates/submits Official weights.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT = Path('artifacts/phase37_risk_regime_tournament')
NAV0 = 1_000_000.0
FEE = .001
CAP = .30
N = 30
CORE_EXPOSURE = .80
DEFENSIVE_EXPOSURE = .60
UPSIDE_EXPOSURE = .90

# Fixed, predeclared research thresholds. Not selected using Oct 2026 outcomes.
HIGH_VOL_RATIO = 1.25
LOW_VOL_RATIO = 1.10
WEAK_20D_LOG_RETURN = -.025
STRONG_20D_LOG_RETURN = .025
STRONG_60D_LOG_RETURN = .050
WEEKLY_REBALANCE_INTERVAL = 5
MIN_EXPOSURE_TRADE_L1 = .10
MIN_STOCK_ROTATION_L1 = .04

PROFILES = (
 'equal60_hold', 'equal80_hold', 'industry80_hold', 'core80_blend_hold',
 'core80_sector_lowvol_hold', 'core80_riskmix_hold',
 'core80_sector_lowvol_monthly',
 'core80_vol_guard_weekly', 'core80_trend_guard_weekly',
 'core80_joint_guard_weekly', 'core80_joint_guard_sticky',
 'core80_riskmix_joint_guard', 'core80_opportunistic_weekly',
)

@dataclass(frozen=True)
class Regime:
    vol_ratio: float
    ret20: float
    ret60: float
    highvol: bool
    weaktrend: bool
    strongtrend: bool


def phase30():
    p = HERE/'icaif_phase30_2026_walkforward_agent_lab_v2.py'
    if not p.is_file():
        raise FileNotFoundError('Place Phase30 v2 script next to Phase37: '+str(p))
    spec=importlib.util.spec_from_file_location('phase37_dep30',p)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _safe_weights(w, expected=None):
    x=np.asarray(w,float)
    if x.shape != (N,) or not np.isfinite(x).all() or x.min() < -1e-12 or x.max() > CAP + 1e-12 or x.sum()>1+1e-10:
        raise AssertionError('Illegal competition research weights')
    if expected is not None and abs(float(x.sum())-expected)>1e-9:
        raise AssertionError('Exposure not as intended')
    return np.maximum(x,0.)


def _logret(market,t,days):
    if t < days: raise ValueError('Insufficient lagged market data')
    a=market.daily[t-days:t+1,:]
    if not np.isfinite(a).all() or (a<=0).any(): raise ValueError('Bad historical price')
    return np.diff(np.log(a),axis=0)


def market_regime(market,t):
    """Uses observations up to and including t (R1: t = yesterday)."""
    d60=_logret(market,t,60).mean(axis=1)
    v20=float(np.std(d60[-20:],ddof=1))
    v60=float(np.std(d60,ddof=1))
    ratio=v20/max(v60,1e-8)
    r20=float(d60[-20:].sum());r60=float(d60.sum())
    return Regime(ratio,r20,r60,ratio>HIGH_VOL_RATIO,
                  r20<WEAK_20D_LOG_RETURN and r60<0.,
                  r20>STRONG_20D_LOG_RETURN and r60>STRONG_60D_LOG_RETURN and ratio<1.0)


def sector_lowvol_core(market,t):
    """Six predeclared five-stock buckets; equal sector dollars, shrink toward equal within bucket."""
    lr=_logret(market,t,20)
    sig=np.std(lr,axis=0,ddof=1)
    # Floors prevent accidental 100% weight on a zero-volatility/illiquid symbol.
    sigma=np.maximum(sig,.004)
    w=np.zeros(N)
    for g in range(6):
        j=slice(g*5,(g+1)*5)
        inv=1./sigma[j]
        inv=inv/inv.sum()
        w[j]=(CORE_EXPOSURE/6.)*(.60/5.+.40*inv)
    return _safe_weights(w,CORE_EXPOSURE)


def base_core(p30,market,t,policy):
    if policy=='equal60_hold':return _safe_weights(np.full(N,.60/N),.60)
    if policy=='equal80_hold':return _safe_weights(np.full(N,.80/N),.80)
    if policy=='industry80_hold':return _safe_weights(p30.desired('industry80_hold',market,t,{}),.80)
    industry=p30.desired('industry80_hold',market,t,{})
    blend=.5*industry+.5*np.full(N,.80/N)
    if 'sector_lowvol' in policy:return sector_lowvol_core(market,t)
    if 'riskmix' in policy:return _safe_weights(.5*blend+.5*sector_lowvol_core(market,t),.80)
    return _safe_weights(blend,.80)


def target_exposure(policy,regime,previous_exposure=None,streak=0):
    """Predeclared signal policies, no future prices and no forecasting model."""
    if policy=='core80_vol_guard_weekly':risk=regime.highvol
    elif policy=='core80_trend_guard_weekly':risk=regime.weaktrend
    elif policy in ('core80_joint_guard_weekly','core80_riskmix_joint_guard'):
        risk=regime.highvol or regime.weaktrend
    elif policy=='core80_joint_guard_sticky':
        risk=regime.highvol or regime.weaktrend
        # 2 consecutive weekly risk-off observations before switching, and
        # recovery requires vol ratio below low-vol hysteresis threshold.
        if previous_exposure is not None and abs(previous_exposure-DEFENSIVE_EXPOSURE)<.001:
            return DEFENSIVE_EXPOSURE if (risk or regime.vol_ratio>=LOW_VOL_RATIO) else CORE_EXPOSURE
        return DEFENSIVE_EXPOSURE if risk and streak>=2 else CORE_EXPOSURE
    elif policy=='core80_opportunistic_weekly':
        if regime.highvol and regime.weaktrend:return DEFENSIVE_EXPOSURE
        return UPSIDE_EXPOSURE if regime.strongtrend else CORE_EXPOSURE
    else:return CORE_EXPOSURE
    return DEFENSIVE_EXPOSURE if risk else CORE_EXPOSURE


def units(shares,cash,mark):
    nav=float(cash+np.dot(shares,mark))
    if nav<=0 or not np.isfinite(nav):raise AssertionError('Bad NAV')
    return shares*mark/nav,nav


def sim(p30,market,indices,policy):
    cash=NAV0;shares=np.zeros(N);path=[NAV0];ledger=[]
    total_fees=0.;total_notional=0.;trade_count=0;turns=[]
    sticky_exposure=None;sticky_streak=0
    last_target=None
    for dayn,i in enumerate(indices):
        i=int(i);t=i-1
        core=base_core(p30,market,t,policy) if dayn==0 or policy.endswith('_monthly') else None
        r=market_regime(market,t)
        is_hold=policy.endswith('_hold')
        is_monthly=policy.endswith('_monthly')
        is_weekly=not is_hold and not is_monthly
        due=dayn==0 or (is_weekly and dayn%WEEKLY_REBALANCE_INTERVAL==0) or (is_monthly and market.dates[i].month!=market.dates[int(indices[dayn-1])].month)
        if due and core is None:
            # Hold blend fixed throughout the episode unless the low-vol composition
            # variant is the explicitly predeclared monthly rule.
            core=initial_core
        if dayn==0:
            initial_core=core.copy()
        if due and policy=='core80_joint_guard_sticky':
            active=r.highvol or r.weaktrend
            sticky_streak=(sticky_streak+1 if active else 0)
        chosen=None
        if due:
            if dayn==0 or is_hold or is_monthly:
                exposure=float(core.sum())
            else:
                exposure=target_exposure(policy,r,sticky_exposure,sticky_streak)
            chosen=_safe_weights(core*(exposure/float(core.sum())),exposure)
            sticky_exposure=exposure
            last_target=chosen.copy()
        for rnd in range(7):
            mark=market.daily[t] if rnd==0 else market.cl[i,p30.KNOWN_IDX[rnd],:]
            state,nav_prior=units(shares,cash,mark)
            decision='HOLD';reason='NO_DECISION_SLOT';l1_proposed=0.
            fee=0.;notional=0.;turn=0.
            if rnd==0:
                reason='NOT_REBALANCE_DATE'
                if dayn==0:
                    intent=chosen
                    decision='INITIAL';reason='INITIAL_BASELINE';l1_proposed=float(intent.sum())
                elif due:
                    l1_proposed=float(np.abs(chosen-state).sum())
                    floor=MIN_STOCK_ROTATION_L1 if is_monthly else MIN_EXPOSURE_TRADE_L1
                    if l1_proposed>=floor:
                        intent=chosen;decision='REBALANCE';reason='PREDECLARED_REGIME_OR_RISK_RULE'
                    else:
                        intent=None;reason='HYSTERESIS_NO_TRADE'
                else:intent=None
                if intent is not None:
                    # Fill at next permitted execution bar, never read for decision.
                    fill=market.op[i,p30.FILL_IDX[rnd],:]
                    nav_fill=float(cash+np.dot(shares,fill))
                    cash,shares,notional=p30.trade(cash,shares,fill,intent)
                    fee=FEE*notional
                    total_notional+=notional;total_fees+=fee;trade_count+=1
                    turn=notional/max(nav_fill,1e-8)
            period_end=market.op[i,p30.END_VAL_IDX[rnd],:] if rnd<6 else market.cl[i,25,:]
            nav_end=float(cash+np.dot(shares,period_end))
            if nav_end<=0 or not np.isfinite(nav_end):raise AssertionError('Bad NAV period')
            path.append(nav_end);turns.append(turn)
            ledger.append(dict(strategy=policy,date=str(market.dates[i].date()),round=rnd+1,
                               information_cutoff='previous_close' if rnd==0 else p30.SLOTS[p30.KNOWN_IDX[rnd]],
                               regime_vol_ratio=r.vol_ratio,regime_logret20=r.ret20,regime_logret60=r.ret60,
                               high_vol=r.highvol,weak_trend=r.weaktrend,strong_trend=r.strongtrend,
                               target_exposure=float(last_target.sum()) if last_target is not None else 0.,
                               decision=decision,reason=reason,proposed_l1=l1_proposed,
                               predecision_exposure=float(state.sum()),notional_usd=notional,
                               fee_usd=fee,turnover_this_round=turn,nav_end=nav_end))
    v=np.asarray(path)
    ret=v[1:]/v[:-1]-1.
    sd=float(ret.std(ddof=1))
    return ({'strategy':policy,'days':len(indices),'cumulative_return':float(v[-1]/NAV0-1.),
             'sharpe_proxy':float(ret.mean()*np.sqrt(1764)/sd) if sd>1e-12 else 0.,
             'mdd_proxy':float(np.max(1.-v/np.maximum.accumulate(v))),
             'turnover_proxy':float(np.mean(turns)),'fees_usd':total_fees,
             'trade_count':trade_count,'risk_off_days':sum(x['target_exposure']<.75 for x in ledger[::7]),
             'mean_exposure':float(np.mean([x['predecision_exposure'] for x in ledger]))},ledger)


def rank_result(data):
    result=pd.DataFrame(data)
    for name,asc in [('cumulative_return',False),('sharpe_proxy',False),('mdd_proxy',True),('turnover_proxy',True)]:
        result['rank_'+name]=result.groupby('segment')[name].rank(ascending=asc,pct=True,method='average')
    result['four_metric_research_rank']=result[['rank_'+x for x in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]].mean(axis=1)
    return result


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()


def run(root,source):
    p30=phase30()
    source=Path(source).expanduser() if source else p30.INPUT
    if not source.is_absolute():source=root/source
    market=p30.read_market(source,min_last_day=('2026-10-06' if source!=root/p30.INPUT else '2026-10-09'))
    out=root/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    tests=[('development','2026-07-01','2026-08-31'),
           ('exposed_diagnostic','2026-09-01','2026-10-09'),
           ('contiguous_exposed_sensitivity','2026-07-01','2026-10-09')]
    rows=[];decisions=[]
    for label,begin,end in tests:
        ix=np.flatnonzero((market.dates>=pd.Timestamp(begin))&(market.dates<=pd.Timestamp(end)))
        if len(ix)<15:raise ValueError('Not enough test dates '+label)
        for policy in PROFILES:
            metrics,trace=sim(p30,market,ix,policy)
            rows.append({'segment':label,**metrics})
            decisions.extend([{'segment':label,**r} for r in trace])
            print('PHASE37_EVALUATED:',label,policy,f'return={metrics["cumulative_return"]:+.5f}',
                  f'fees={metrics["fees_usd"]:.1f}',f'trades={metrics["trade_count"]}',flush=True)
    result=rank_result(rows)
    result.sort_values(['segment','four_metric_research_rank']).to_csv(out/'phase37_four_metric_tournament.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'phase37_all_seven_round_decisions.csv',index=False)
    for segment in result.segment.unique():
        group=result[result.segment==segment]
        base=group.set_index('strategy').loc['core80_blend_hold']
        benchmark=group.set_index('strategy').loc['equal80_hold']
        print('===',segment.upper(),'FOUR METRIC PROXY LOWER BETTER ===',flush=True)
        print(group.sort_values('four_metric_research_rank')[['strategy','cumulative_return','sharpe_proxy','mdd_proxy',
              'turnover_proxy','fees_usd','trade_count','four_metric_research_rank']].to_string(index=False),flush=True)
        for candidate in ('core80_sector_lowvol_hold','core80_joint_guard_weekly',
                          'core80_riskmix_joint_guard','core80_opportunistic_weekly'):
            row=group.set_index('strategy').loc[candidate]
            print('PAIRWISE',candidate,'vs_core80_blend_hold return_pp',round((row.cumulative_return-base.cumulative_return)*100,5),
                  'mdd_pp',round((row.mdd_proxy-base.mdd_proxy)*100,5),
                  'fee_delta_usd',round(row.fees_usd-base.fees_usd,3),flush=True)
    (out/'phase37_manifest.json').write_text(json.dumps({
        'asof':str(market.dates[-1].date()),'input_csv':str(source),'input_sha256':sha256(source),
        'excluded_incomplete_sessions':len(market.missing),'network_requests':0,
        'organizer_submissions':0,'live_agent_activated':False,'llm_requests':0,
        'policies':PROFILES,'data_provenance':'unadjusted Alpaca IEX 15-min raw prices',
        'price_fill_proxy':'seven decision rounds with fill bar open, prior close as R1 information',
        'evaluation_style':'independent Jul-Aug and exposed Sep-Oct resets; overlapping contiguous sensitivity',
        'nonrobustness':['Unadjusted prices omit dividends and corporate actions',
            'IEX feed is not organizer execution price',
            '36 incomplete dates/early close dates skipped by previous loader',
            'Sep-Oct 2026 is already exposed and cannot justify parameter optimization',
            'Four-metric ranks depend on these synthetic opponent policies; not actual leaderboard',
            'Stress regime thresholds predeclared heuristics, not independently fitted or optimized',
            'Contiguous test overlaps two other segments and is not independent evidence'],
        'parameters':{'fee':FEE,'vol_guard_ratio':HIGH_VOL_RATIO,'low_vol_hysteresis':LOW_VOL_RATIO,
            'weak20_logret':WEAK_20D_LOG_RETURN,'strong20_logret':STRONG_20D_LOG_RETURN,
            'strong60_logret':STRONG_60D_LOG_RETURN,'rebalance_weekly_interval':WEEKLY_REBALANCE_INTERVAL,
            'min_exposure_l1':MIN_EXPOSURE_TRADE_L1,'min_stock_rotation_l1':MIN_STOCK_ROTATION_L1,
            'equity_low':DEFENSIVE_EXPOSURE,'equity_baseline':CORE_EXPOSURE,'equity_high':UPSIDE_EXPOSURE}
    },indent=2,default=str),encoding='utf-8')
    print('PHASE37_RESULT_DIR:',out.resolve(),flush=True)
    print('PHASE37_COMPLETE_RESEARCH_ONLY_NO_ORGANIZER_SUBMISSION',flush=True)
    return result


def self_test():
    from types import SimpleNamespace
    rng=np.random.default_rng(20261011)
    r=rng.normal(.00015,.011,size=(280,N))
    prices=np.vstack([np.full((1,N),100.),100*np.exp(np.cumsum(r,axis=0))])
    m=SimpleNamespace(daily=prices)
    # Past-only invariance: future price modification cannot affect historical regime.
    a=market_regime(m,175)
    m2=SimpleNamespace(daily=prices.copy())
    m2.daily[176:]*=10.
    assert market_regime(m2,175)==a
    c=sector_lowvol_core(m,175)
    assert abs(c.sum()-.8)<1e-10 and c.min()>=0 and c.max()<=CAP
    for risk in [Regime(1.3,-.04,-.06,True,True,False),Regime(.8,.04,.1,False,False,True)]:
        for profile in PROFILES:
            e=target_exposure(profile,risk)
            if not .59<=e<=.91:raise AssertionError('Out of bounds risk exposure')
            _safe_weights(c*e/.8,e)
    # Reconstruct 7-slot mark/fill simulator with synthetic prices.
    class P:
        KNOWN_IDX=(None,2,6,10,14,18,22)
        FILL_IDX=(0,4,8,12,16,20,24)
        END_VAL_IDX=(4,8,12,16,20,24,25)
        SLOTS=[f'{9+(30+15*j)//60:02d}:{(30+15*j)%60:02d}' for j in range(26)]
        @staticmethod
        def desired(policy,market,t,mods):
            return np.full(N,.8/N)
        @staticmethod
        def trade(cash,shares,price,w):
            before=cash+float(np.dot(shares,price));old=shares*price
            after=before
            for _ in range(100):
                q=float(np.abs(after*w-old).sum())
                update=before-FEE*q
                if abs(update-after)<1e-8:break
                after=update
            q=float(np.abs(after*w-old).sum())
            return after*(1.-w.sum()),after*w/price,q
    dates=pd.bdate_range('2025-09-01',periods=281)
    dayprice=prices[1:]
    # intraday synthetic with varying (but deterministic) within-day prices
    ix=np.linspace(0,.001,26).reshape(1,26,1)
    op=dayprice[:,None,:]*np.exp(ix)
    cl=op*np.exp(.00005)
    fake=SimpleNamespace(daily=dayprice,op=op,cl=cl,dates=dates)
    idx=np.arange(175,220)
    for p in PROFILES:
        met,records=sim(P,fake,idx,p)
        assert len(records)==len(idx)*7
        assert np.isfinite(met['cumulative_return'])
        assert all(x['round']>1 or x['decision'] in ('HOLD','INITIAL','REBALANCE') for x in records)
        assert all(x['round']==1 for x in records if x['decision']!='HOLD')
        assert abs(sum(x['fee_usd'] for x in records)-met['fees_usd'])<1e-6
        assert met['mdd_proxy']>=0 and met['turnover_proxy']>=0
    # Stress-market end-to-end test: a sudden market drawdown must cause
    # weekly de-risking, then a subsequent risk-on recovery after the shock
    # exits the 20-day realized-volatility window. All marks/fills coherent.
    shock=np.zeros(len(dayprice))
    shock[184]=-.13
    scale=np.exp(np.cumsum(shock))[:,None,None]
    stressed=SimpleNamespace(daily=dayprice*scale[:,0,:],
                             op=op*scale,cl=cl*scale,dates=dates)
    stress_ix=np.arange(175,247)
    guarded,guard_trace=sim(P,stressed,stress_ix,'core80_vol_guard_weekly')
    changes=[r for r in guard_trace if r['decision']=='REBALANCE']
    assert changes, 'Volatility stress guard never traded in constructed shock scenario'
    assert any(r['target_exposure'] < .75 for r in changes), 'Stress guard failed to reduce exposure'
    assert any(r['target_exposure'] > .75 for r in changes), 'Stress guard failed to recover exposure'
    assert len(changes) < 10, 'Excessive flipping during synthetic single-shock recovery'
    assert guarded['fees_usd'] > FEE*NAV0*.80
    stress_t=190
    observed=market_regime(stressed,stress_t)
    future_changed=SimpleNamespace(daily=stressed.daily.copy())
    future_changed.daily[stress_t+1:]*=3
    assert market_regime(future_changed,stress_t)==observed
    print('PHASE37_SYNTHETIC_STRESS_DE_RISK_AND_RECOVERY_PASS')
    # Check future modification after a decision does not change its target signals.
    old=base_core(P,fake,175,'core80_riskmix_hold')
    fake2=SimpleNamespace(daily=dayprice.copy(),op=op,cl=cl,dates=dates)
    fake2.daily[176:]*=100.
    new=base_core(P,fake2,175,'core80_riskmix_hold')
    assert np.allclose(old,new)
    print('PHASE37_SELF_TEST_PASS')
    print('ASOF_REGIME_LEAKAGE_GUARD_7_ROUNDS_FEES_AND_EXPOSURE_CAPS_PASS')
    print('NO_NETWORK_NO_LLM_NO_HETZNER_NO_ORGANIZER_SUBMISSIONS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--run',action='store_true')
    p.add_argument('--input',default=None,help='Optional 15m Alpaca CSV; default Phase29 merged Oct09')
    args=p.parse_args()
    if not args.self_test and not args.run: p.error('Use --self-test or --run')
    if args.self_test:self_test()
    if args.run:run(Path.cwd(),args.input)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('PHASE37_RESEARCH_STOP:',type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
