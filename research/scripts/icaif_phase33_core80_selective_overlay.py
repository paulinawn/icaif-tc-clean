#!/usr/bin/env python3
"""ICAIF Phase33: bounded 80% core + fee-aware selective alpha overlay, offline only.

Test hypotheses (predeclared, do NOT tune to exposed Sep/Oct 2026):
  - 80% fixed diversified core is a robust anchor;
  - calibrated 1d/5d price + industry (+ optional SEC) forecasts can justify
    BOUNDED shifts when expected horizon alpha exceeds fees, risk and uncertainty;
  - HOLD when incremental utility is weak, optionally trade part of desired shift.

Input: Phase29 merged 15m IEX raw file. Requires Phase30 v2 and Phase31 v3
next to this script. SEC optional: Phase19 file and its CompanyFacts cache.

Outputs are experimental, NOT the Official execution/scoring contract. IEX
opening bar prices are fill proxies; no distributions/corporate action simulation;
July-Aug development and Sep-Oct *EXPOSED* diagnostics independently reset NAV.
SEC date+2d is a proxy, not precise acceptance time. No historic LLM backfill.

Commands:
  python icaif_phase33_core80_selective_overlay.py --self-test
  python icaif_phase33_core80_selective_overlay.py --run --with-sec
  python icaif_phase33_core80_selective_overlay.py --run --with-sec --input data/cache/alpaca_iex_15m_400d.csv

Never calls Alpaca, organizer endpoints, LLMs or any other network source.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTDIR = Path('artifacts/phase33_core80_selective_overlay')
FEE = 0.001
NAV0 = 1_000_000.0
CORE_EXP = 0.8
MAX_STOCK_W = .30
MAX_STOCK_TILT = .012  # Max absolute tilt from initial 80% core, each stock
MAX_TOTAL_CORE_L1 = .16  # Max gross core overlay deviation, fraction of NAV
MAX_REBALANCE_L1 = .08  # Max planned one-decision gross turnover, last-known mark
MIN_REBALANCE_L1 = .008
UNCERTAINTY_COEFF = .0004  # Per unit gross turnover when evidence is weak
MIN_UTILITY = .000025    # $25 expected net improvement per $1m NAV
MKT_RISK_PENALTY = .25

POLICIES = (
 'equal60_hold', 'equal80_hold', 'momentum80_hold', 'industry80_hold',
 'core80_blend_hold', 'core80_price_weekly', 'core80_price_daily',
 'core80_industry_weekly', 'core80_industry_daily',
 'core80_consensus_weekly', 'core80_consensus_daily',
 'core80_consensus_partial', 'core80_consensus_intraday',
)
SEC_POLICIES=('core80_sec_weekly','core80_sec_daily')


def research_modules():
    import importlib.util
    def get(name, file):
        p=HERE/file
        if not p.is_file(): raise FileNotFoundError('Place dependency beside Phase33 in Downloads: '+file)
        spec=importlib.util.spec_from_file_location(name,p)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        assert spec.loader
        spec.loader.exec_module(module)
        return module
    p30=get('phase33_dep30','icaif_phase30_2026_walkforward_agent_lab_v2.py')
    p31=get('phase33_dep31','icaif_phase31_integrated_alpha_ablation_v3.py')
    return p30,p31,get


def normalize_core(w):
    w=np.asarray(w,float).copy()
    if w.shape!=(30,) or not np.isfinite(w).all() or np.min(w)<0:
        raise ValueError('Invalid core')
    w=w*(CORE_EXP/float(w.sum()))
    if np.max(w)>MAX_STOCK_W+1e-12:raise ValueError('Core violates stock maximum')
    return w


def core_weight(p30,market,t,profile):
    if profile=='equal60_hold':return np.full(30,.6/30.)
    if profile=='equal80_hold':return np.full(30,.8/30.)
    if profile=='momentum80_hold':return p30.desired('momentum80_hold',market,t,{})
    if profile=='industry80_hold':return p30.desired('industry80_hold',market,t,{})
    industry=p30.desired('industry80_hold',market,t,{})
    return normalize_core(.5*industry+.5*np.full(30,.8/30.))


def forecast(p31,p30,market,t,model,policy,sec):
    """Daily relative log return & uncertainty; all features end no later than t."""
    if '_price_' in policy:
        p,c=p31.forecast(p30,market,t,model,'price',sec)
    elif '_industry_' in policy:
        p,c=p31.forecast(p30,market,t,model,'industry',sec)
    else:
        p0,c0=p31.forecast(p30,market,t,model,'price',sec)
        p1,c1=p31.forecast(p30,market,t,model,'industry',sec)
        if '_sec_' in policy:
            if sec is None:raise ValueError('SEC policy without SEC cache')
            p2,c2=p31.forecast(p30,market,t,model,'sec',sec)
            # Predeclared fixed weights, not optimized on Sep-Oct.
            p=.30*p0+.40*p1+.30*p2
            c=np.minimum(np.minimum(c0,c1),c2)
            c=np.where((np.sign(p1)==np.sign(p2))&(np.sign(p0)==np.sign(p1)),c,0.)
        else:
            p=.50*p0+.50*p1
            c=np.minimum(c0,c1)
            c=np.where(np.sign(p0)==np.sign(p1),c,0.)
    p=np.asarray(p,float);c=np.asarray(c,float)
    if not np.isfinite(p).all() or not np.isfinite(c).all():raise ValueError('Invalid forecast')
    p=p-p.mean()  # Cross-sectional forecast only, NOT market-exposure forecast.
    return p,np.clip(c,0.,1.)


def bounded_target(core,expected_daily,confidence,forecast_horizon=5):
    """Translate forecast strength/confidence to small cash-neutral equity tilts.

    Fixed non-fitted sizing: 1% portfolio weight shift per 1% predicted advantage
    over planned holding horizon, times confidence, max +/- 1.2%/stock.
    This is an HYPOTHESIS to test, not an optimal portfolio mapping.
    """
    daily=np.asarray(expected_daily,float)
    conf=np.asarray(confidence,float)
    if daily.shape!=(30,) or conf.shape!=(30,) or not np.isfinite(daily).all():
        raise ValueError('Bad forecast shape')
    raw=1.0*(daily*forecast_horizon)*np.clip(conf,0.,1.)
    raw-=raw.mean()
    raw=np.clip(raw,-MAX_STOCK_TILT,MAX_STOCK_TILT)
    # Enforce zero-sum shift and per-stock bounds with conservative scaling.
    raw-=raw.mean()
    maxabs=float(np.max(np.abs(raw)))
    if maxabs>MAX_STOCK_TILT:raw*=MAX_STOCK_TILT/maxabs
    l1=float(np.sum(np.abs(raw)))
    if l1>MAX_TOTAL_CORE_L1:raw*=MAX_TOTAL_CORE_L1/l1
    proposed=core+raw
    if np.min(proposed)<0 or np.max(proposed)>MAX_STOCK_W:
        # Scale back to a legal core+tilt without changing baseline exposure.
        scales=[1.]
        for wi,di in zip(core,raw):
            if di<0:scales.append(float(wi/-di))
            elif di>0:scales.append(float((MAX_STOCK_W-wi)/di))
        raw*=max(0.,min(scales,default=1.)*0.999999)
        proposed=core+raw
    if not np.isfinite(proposed).all() or np.min(proposed)<-1e-10 or np.max(proposed)>MAX_STOCK_W+1e-9:
        raise AssertionError('Illegal proposal')
    if abs(proposed.sum()-core.sum())>1e-9:raise AssertionError('Target equity exposure changed')
    return proposed


def decision_utility(p30,market,i,current,proposed,expected_return,confidence,days_to_hold):
    delta=proposed-current
    l1=float(np.abs(delta).sum())
    gain=float(np.dot(delta,expected_return)*days_to_hold)
    fee=FEE*l1
    # Prior completed closes only; no fill or today's future market prices.
    lr=np.diff(np.log(market.daily[i-22:i]),axis=0)
    cov=np.cov(lr,rowvar=False)
    risk_change=float(proposed@cov@proposed-current@cov@current)
    risk=max(0.,risk_change)*MKT_RISK_PENALTY*days_to_hold
    mean_conf=float(np.mean(np.clip(confidence,0.,1.)))
    uncertainty=UNCERTAINTY_COEFF*l1*(1.-mean_conf)
    benefit=gain-fee-risk-uncertainty
    return {'expected_gain':gain,'estimated_fee':fee,'risk_penalty':risk,
            'uncertainty_penalty':uncertainty,'net_expected_utility':benefit,
            'proposed_l1':l1,'mean_forecast_confidence':mean_conf}


def choose_action(p30,market,i,mark,cash,shares,base_target,expected,confidence,
                  days_to_hold,partial=False):
    current,nav=weights(shares,cash,mark)
    delta=base_target-current
    gross=float(np.abs(delta).sum())
    if gross<MIN_REBALANCE_L1:
        return None,{'decision':'HOLD','reason':'SMALL_SHIFT','proposed_l1':gross,
                     'net_expected_utility':0.,'estimated_fee':0.,'expected_gain':0.,
                     'risk_penalty':0.,'uncertainty_penalty':0.,'mean_forecast_confidence':float(np.mean(confidence))}
    delta*=min(1.,MAX_REBALANCE_L1/gross)
    if partial:delta*=.5
    target=current+delta
    # Current weights from mark can drift from 80% exposure; do not force a market bet.
    # A target never uses the unknown fill-time price.
    if np.min(target)<-1e-10 or np.max(target)>MAX_STOCK_W+1e-9 or target.sum()>1.+1e-9:
        return None,{'decision':'HOLD','reason':'RISK_CAP','proposed_l1':gross,
                     'net_expected_utility':0.,'estimated_fee':0.,'expected_gain':0.,
                     'risk_penalty':0.,'uncertainty_penalty':0.,'mean_forecast_confidence':float(np.mean(confidence))}
    stats=decision_utility(p30,market,i,current,target,expected,confidence,days_to_hold)
    if stats['net_expected_utility']<=MIN_UTILITY:
        return None,{'decision':'HOLD','reason':'INSUFFICIENT_INCREMENTAL_ALPHA',**stats}
    return target,{'decision':'PARTIAL' if partial else 'REBALANCE','reason':'EXPECTED_UTILITY_PASS',**stats}


def weights(shares,cash,mark):
    nav=float(cash+np.dot(shares,mark))
    if not np.isfinite(nav) or nav<=0:raise ValueError('Invalid NAV')
    return np.asarray(shares*mark/nav,float),nav


def replay(p30,p31,market,indices,models,profile,sec=None):
    cash=NAV0;shares=np.zeros(30,float);nav_path=[NAV0]
    records=[];fees=0.;notional=0.;trades=0;trade_days=0
    initial_core=core_weight(p30,market,int(indices[0])-1,profile)
    for dayn,i in enumerate(indices):
        t=int(i)-1
        model=models[market.dates[i].strftime('%Y-%m')]
        traded_today=False
        daily_pred=None;daily_conf=None
        if profile.startswith('core80_') and not profile.endswith('_hold'):
            daily_pred,daily_conf=forecast(p31,p30,market,t,model,profile,sec)
        for rnd in range(7):
            mark=(market.daily[t] if rnd==0 else market.cl[i,p30.KNOWN_IDX[rnd],:])
            fill=market.op[i,p30.FILL_IDX[rnd],:]
            state,prior_nav=weights(shares,cash,mark)
            intent=None
            meta={'decision':'HOLD','reason':'NO_SCHEDULED_SIGNAL','proposed_l1':0.,
                  'net_expected_utility':0.,'estimated_fee':0.,'expected_gain':0.,
                  'risk_penalty':0.,'uncertainty_penalty':0.,'mean_forecast_confidence':0.}
            if dayn==0 and rnd==0:
                intent=initial_core
                meta={'decision':'INITIAL_CORE','reason':'BASELINE_PURCHASE','proposed_l1':float(initial_core.sum()),
                      'net_expected_utility':0.,'estimated_fee':FEE*initial_core.sum(),
                      'expected_gain':0.,'risk_penalty':0.,'uncertainty_penalty':0.,
                      'mean_forecast_confidence':0.}
            elif daily_pred is not None:
                is_weekly=profile.endswith('_weekly')
                is_daily=profile.endswith(('_daily','_partial','_intraday'))
                if rnd==0 and ((is_weekly and dayn%5==0) or is_daily):
                    days=5 if is_weekly else 1
                    proposed=bounded_target(initial_core,daily_pred,daily_conf,days)
                    intent,meta=choose_action(p30,market,int(i),mark,cash,shares,proposed,daily_pred,
                                              daily_conf,days,profile.endswith('_partial'))
                elif rnd>0 and profile.endswith('_intraday'):
                    # Intraday coefficients were fitted on previous months.
                    known=p30.known_intraday(market.cl,market.op,int(i),rnd)
                    intraday_pred=known@model.coefficients[f'intra_{rnd}']
                    intraday_pred-=intraday_pred.mean()
                    # Uncalibrated intraday forecast => stricter predetermined confidence.
                    intraday_conf=np.full(30,.20)
                    proposed=bounded_target(initial_core,intraday_pred,intraday_conf,1)
                    intent,meta=choose_action(p30,market,int(i),mark,cash,shares,proposed,
                                              intraday_pred,intraday_conf,1,False)
            fill_notional=0.
            if intent is not None:
                if not np.isfinite(intent).all() or min(intent)<0 or max(intent)>MAX_STOCK_W or sum(intent)>1.+1e-8:
                    raise AssertionError('Illegal simulation target')
                cash,shares,fill_notional=p30.trade(cash,shares,fill,intent)
                fees+=FEE*fill_notional;notional+=fill_notional
                trades+=1;traded_today=True
            period_end=(market.op[i,p30.END_VAL_IDX[rnd],:] if rnd<6 else market.cl[i,25,:])
            nav_path.append(cash+float(np.dot(shares,period_end)))
            records.append({'strategy':profile,'date':str(market.dates[i].date()),'round':rnd+1,
                            'known_price_cutoff':'previous_close' if rnd==0 else p30.SLOTS[p30.KNOWN_IDX[rnd]],
                            'actual_fill_proxy_time':p30.SLOTS[p30.FILL_IDX[rnd]],
                            'pretrade_exposure':float(state.sum()),'pretrade_nav':prior_nav,
                            'actual_trade_notional':fill_notional,'actual_fee':FEE*fill_notional,**meta})
        trade_days+=int(traded_today)
    v=np.asarray(nav_path,float)
    if np.min(v)<=0 or not np.isfinite(v).all():raise AssertionError('Invalid NAV history')
    returns=v[1:]/v[:-1]-1
    std=float(np.std(returns,ddof=1))
    return {'strategy':profile,'days':len(indices),
            'cumulative_return':float(v[-1]/NAV0-1.),
            'sharpe_proxy':float(returns.mean()*np.sqrt(252*7)/std) if std>1e-12 else 0.,
            'mdd_proxy':float(np.max(1-v/np.maximum.accumulate(v))),
            'turnover_proxy':float(notional/(NAV0*len(indices)*7)),
            'fees_usd':fees,'trade_count':trades,'trade_days':trade_days,
            'gate_passes':sum(r['decision'] in ('REBALANCE','PARTIAL') for r in records),
            'rejected_signals':sum(r['reason'] in ('SMALL_SHIFT','RISK_CAP','INSUFFICIENT_INCREMENTAL_ALPHA') for r in records)
            },records


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(root,with_sec=False,input_file=None):
    p30,p31,get=research_modules()
    source=Path(input_file).expanduser() if input_file else root/p30.INPUT
    if not source.is_absolute():source=root/source
    market=p30.read_market(source,min_last_day=('2026-10-06' if input_file else '2026-10-07'))
    output=root/OUTDIR;output.mkdir(parents=True,exist_ok=True)
    print('PHASE33_DATA_ASOF:',str(market.dates[-1].date()),'INCOMPLETE_EXCLUDED:',len(market.missing),flush=True)
    sec=None;coverage=None
    if with_sec:
        p19=get('phase33_dep19','icaif_phase19_sec_ml_research.py')
        sfolder=root/p31.SEC_FOLDER
        if not sfolder.is_dir():raise FileNotFoundError('SEC cache missing: '+str(sfolder))
        sec,coverage=p19.build_sec_panel(market.dates,sfolder)
    profiles=POLICIES+(SEC_POLICIES if sec is not None else tuple())
    metrics=[];decisions=[];provenance=[]
    for label,start,end in [('development','2026-07-01','2026-08-31'),
                            ('exposed_diagnostic','2026-09-01','2026-10-09')]:
        indices=np.flatnonzero((market.dates>=pd.Timestamp(start))&(market.dates<=pd.Timestamp(end)))
        if len(indices)<15:raise RuntimeError('Too few test sessions: '+label)
        models,metadata=p31.monthly_models(p30,market,indices,sec)
        provenance.extend(metadata)
        for profile in profiles:
            row,trace=replay(p30,p31,market,indices,models,profile,sec)
            metrics.append({'segment':label,**row})
            for item in trace:decisions.append({'segment':label,**item})
            print('PHASE33_EVALUATED:',label,profile,
                  f'return={row["cumulative_return"]:.5f}',f'fees={row["fees_usd"]:.1f}',flush=True)
    result=pd.DataFrame(metrics)
    for metric,asc in [('cumulative_return',False),('sharpe_proxy',False),('mdd_proxy',True),('turnover_proxy',True)]:
        result['rank_'+metric]=result.groupby('segment')[metric].rank(pct=True,ascending=asc)
    result['four_metric_research_rank']=result[['rank_'+k for k in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]].mean(axis=1)
    result.sort_values(['segment','four_metric_research_rank']).to_csv(output/'phase33_four_metric_comparison.csv',index=False)
    decisions_df=pd.DataFrame(decisions)
    decisions_df.to_csv(output/'phase33_all_round_decisions.csv',index=False)
    stats=decisions_df.groupby(['segment','strategy','decision','reason'],dropna=False).agg(
        decisions=('round','size'),notional=('actual_trade_notional','sum'),fees=('actual_fee','sum'),
        mean_predicted_gain=('expected_gain','mean'),mean_net_utility=('net_expected_utility','mean')).reset_index()
    stats.to_csv(output/'phase33_decision_counts.csv',index=False)
    (output/'phase33_monthly_model_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (output/'phase33_manifest.json').write_text(json.dumps({
        'research_only':True,'network_calls':False,'competition_submissions':False,
        'original_input_file':str(source),'sha256':digest(source),
        'last_market_day':str(market.dates[-1].date()),'excluded_incomplete_sessions':len(market.missing),
        'sec_enabled':sec is not None,'sec_coverage':coverage,
        'llm_signals_historical_training':'NOT_AVAILABLE_NOT_USED',
        'core_target_equity_fraction':CORE_EXP,'core_blend':'50% equal80 plus 50% industry80 at initial decision',
        'forecast_predeclared_mix':'price: 100%; industry:100%; consensus: 50/50; SEC:30/40/30',
        'max_stock_tilt':MAX_STOCK_TILT,'max_total_core_deviation':MAX_TOTAL_CORE_L1,
        'max_rebalance_planned_l1':MAX_REBALANCE_L1,'min_rebalance_l1':MIN_REBALANCE_L1,
        'fee_rate':FEE,'min_utility_fraction':MIN_UTILITY,
        'important_caveats':['IEX 15m bars are NOT organizer fills',
           'Sep-Oct exposed diagnostic, not untouched holdout',
           'L1 turnover and risk computed at prior observable mark; actual fill unknown',
           'Corporate actions not explicitly modeled',
           'SEC availability approximated by filed date plus 2 days',
           'Months have limited independent fit/calibration dates',
           'The research ranking depends on these particular candidate policies',
           '30-stock raw IEX prices may have exchange-feed gaps',
           'Nonstationary forecast calibration often turns model off',
           'No fitted market-beta predictor; overlay keeps 80% target exposure']
    },indent=2,default=str),encoding='utf-8')
    for segment in result.segment.unique():
        print('===',segment.upper(),'FOUR METRICS LOWER BETTER ===',flush=True)
        a=result[result.segment==segment].sort_values('four_metric_research_rank')
        print(a[['strategy','cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd','trade_count','four_metric_research_rank']].to_string(index=False),flush=True)
    print('PHASE33_RESULTS_DIR:',output.resolve())
    print('PHASE33_COMPLETE_RESEARCH_ONLY_NO_SUBMISSION')
    return result,decisions_df


def self_test():
    core=np.full(30,CORE_EXP/30)
    zero=np.zeros(30)
    w=bounded_target(core,zero,np.ones(30))
    assert np.allclose(w,core)
    alpha=np.linspace(-.02,.02,30)
    target=bounded_target(core,alpha,np.ones(30))
    assert np.isclose(target.sum(),np.sum(core)) and np.min(target)>=0 and max(target)<=MAX_STOCK_W
    assert np.max(np.abs(target-core))<=MAX_STOCK_TILT+1e-12
    assert np.abs(target-core).sum()<=MAX_TOTAL_CORE_L1+1e-12
    # A future-only change cannot change current target, since only passed arrays appear.
    assert np.allclose(target,bounded_target(core,alpha,np.ones(30)))
    shares=np.zeros(30)
    curr,nav=weights(shares,NAV0,np.full(30,100.))
    assert np.allclose(curr,0.) and nav==NAV0
    # Zero-signal HOLD gate must reject a mere core maintenance rebalance when
    # observed weights already equal intended core.
    from types import SimpleNamespace
    daily=np.tile(np.linspace(100,130,30),(30,1)).astype(float)
    market=SimpleNamespace(daily=daily)
    mark=np.full(30,100.)
    sh=core*NAV0/mark
    c=NAV0*.2
    action,meta=choose_action(None,market,22,mark,c,sh,core,zero,np.ones(30),5)
    assert action is None and meta['reason']=='SMALL_SHIFT'
    # Every proposed decision must include fee and uncertainty in its utility.
    delta=.03*np.array([1.,-1.]+[0.]*28)
    u=decision_utility(None,market,22,core,core+delta,zero,np.zeros(30),5)
    assert u['net_expected_utility']<0 and u['estimated_fee']>0
    print('PHASE33_SELF_TEST_PASS')
    print('CORE_80_BOUNDED_TILTS_FEES_UNCERTAINTY_AND_HOLD_GATES_PASS')
    print('NO_NETWORK_NO_LLM_NO_ORGANIZER_CALLS_NO_TRADING')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--run',action='store_true')
    parser.add_argument('--with-sec',action='store_true')
    parser.add_argument('--input',type=str,default=None)
    args=parser.parse_args()
    try:
        if args.self_test:self_test()
        else:run(Path.cwd(),args.with_sec,args.input)
    except Exception as e:
        # Never echo unknown sensitive input paths or file contents.
        print('PHASE33_STOP:',type(e).__name__,str(e)[:180] if type(e) in (ValueError,FileNotFoundError,AssertionError,RuntimeError) else '',file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
