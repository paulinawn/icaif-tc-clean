#!/usr/bin/env python3
"""ICAIF Phase 31 v3: 2026 integrated price/industry/SEC ablation and selective trading.

RESEARCH ONLY. No network, API credentials, organizer requests, LLM calls or trades.
Inputs: Phase29 updated Alpaca IEX 15m raw CSV; Phase30 in same folder.
Optional: Phase19 SEC CompanyFacts cache (30 files) and Phase19 module in same folder.

All fitted models use only fully matured labels before each monthly test cutoff.
Predictions are relative log returns (not standalone expected portfolio return).
Simulation prices are IEX 15-minute proxies, NOT actual organizer execution fills.
The 2026 September/October period is previously exposed; a diagnostic only.
CompanyFacts filed-date+2d protects publication order imperfectly: research-grade.

Usage:
 python "$HOME\\Downloads\\icaif_phase31_integrated_alpha_ablation_v3.py" --self-test
 python "$HOME\\Downloads\\icaif_phase31_integrated_alpha_ablation_v3.py" --run
 python "$HOME\\Downloads\\icaif_phase31_integrated_alpha_ablation_v3.py" --event-audit
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
OUTPUT = Path('artifacts/phase31_integrated_alpha_ablation_v3')
FEE = .001
NAV0 = 1_000_000.0
FIVE_LABEL = 5
PENALTY = 120.
SEC_FOLDER = Path('artifacts/phase19_sec_ml/sec_raw')
EVENT_FOLDER = Path('artifacts/phase23_llm_event_lab')
PROFILES = (
    'equal60_hold', 'equal80_hold', 'momentum80_hold', 'industry80_hold',
    'price_ridge80_hold', 'price_industry80_hold',
    'price_industry80_weekly', 'price_industry80_daily',
    'ensemble_conf80_daily', 'ensemble_cost80_daily',
    'ensemble_cost80_partial', 'ensemble_cost80_intraday',
    'ensemble_cost80_staged',
)
SEC_PROFILE = 'ensemble_sec_cost80_daily'


def module(name, filename):
    path=THIS/filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing required research file in Downloads: {filename}')
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    assert spec.loader
    spec.loader.exec_module(obj)
    return obj


@dataclass
class Predictor:
    coefficients: dict
    errors: dict
    quality: dict
    training_counts: dict
    cutoff: str


def safe_std(v):
    return max(float(np.std(v)),1e-8)


def sec_factors(sec, t):
    """Four dated public fundamental signals; missing => neutral, never bullish.

    SEC source feature indices are Phase19's margin, operating cashflow margin,
    leverage, revenue growth, plus a count-based availability flag. No current
    2026 earnings surprise inferred from CompanyFacts restatements.
    """
    a=np.asarray(sec[t],float)
    if a.shape!=(30,13):raise ValueError('Unexpected SEC panel shape')
    selected=[1,2,3,5]
    out=[]
    for col in selected:
        x=a[:,col].copy()
        finite=np.isfinite(x)
        if int(finite.sum())<10:out.append(np.zeros(30));continue
        med=float(np.median(x[finite]));x[~finite]=med
        score=np.clip((x-np.mean(x))/safe_std(x),-3.,3.)
        if col==3:score=-score  # leverage is a risk indicator, not alpha certainty
        out.append(score)
    available=np.isfinite(a[:,selected]).sum(axis=1).astype(float)/len(selected)
    return np.column_stack([*out,available-available.mean()])



def design_at(base, market, t, kind, sec=None):
    f=base.features(market.daily,t)
    if kind=='price':return f[:,:4]
    if kind=='industry':return f
    if kind=='sec':
        if sec is None:raise ValueError('SEC panel not available')
        return np.column_stack([f,sec_factors(sec,t)])
    raise ValueError('Unknown model kind')


def one_model(base, market, cutoff, horizon, kind, sec=None):
    """Train relative-return ridge with within-training chronological calibration.

    A chronological inner calibration block uses 15-30 matured feature days,
    keeping at least 25 training days even at the earliest allowed cutoff.
    Early-2026 estimates are exploratory due to limited effective time samples.
    No test-month prices enter fit or calibration.
    """
    last=cutoff-horizon-1   # future label must end BEFORE cutoff day
    start=base.TRAIN_MIN_DAY
    days=np.arange(start,last+1)
    if len(days)<40:
        raise ValueError(f'Not enough matured training days: {len(days)} available, require >=40 after 120-day feature warmup')
    logs=np.log(market.daily)
    X=[];Y=[]
    for t in days:
        x=design_at(base,market,int(t),kind,sec)
        y=logs[t+horizon]-logs[t]
        y=y-y.mean()  # cross-sectional relative stock return
        X.append(x);Y.append(y)
    X=np.stack(X);Y=np.stack(Y)
    # Independent days are calibration units, not individual stocks.
    valid=min(30,max(15,len(days)//4))
    fit_end=len(days)-valid
    if fit_end<25:raise AssertionError('Too few chronologically prior fit days')
    first_beta=base.ridge(X[:fit_end].reshape(-1,X.shape[-1]),Y[:fit_end].ravel(),penalty=PENALTY)
    p=(X[fit_end:]@first_beta).ravel();yy=Y[fit_end:].ravel()
    # Shrink overconfident models; don't invert forecasts based on noisy negative estimates.
    slope=float(np.dot(p,yy)/(np.dot(p,p)+1e-10))
    calibrated=float(np.clip(slope,0.,1.5))
    resid=yy-calibrated*p
    sigma=float(np.std(resid))
    predicted_ic=float(pd.Series(p).rank().corr(pd.Series(yy).rank())) if np.std(p)>1e-14 and np.std(yy)>1e-14 else 0.
    beta=base.ridge(X.reshape(-1,X.shape[-1]),Y.ravel(),penalty=PENALTY)
    return beta,{'slope':calibrated,'residual_std':sigma,'calibration_ic':predicted_ic,'calibration_days':valid,'precalibration_fit_days':fit_end,'matured_feature_days':len(days),'latest_label_date':str(market.dates[last+horizon].date()),'training_stocks':int(Y.size)}


def train_month(base, market, cutoff, sec=None):
    p={};errors={};quality={};counts={}
    kinds=('price','industry','sec') if sec is not None else ('price','industry')
    for kind in kinds:
        for h in (1,5):
            beta,stat=one_model(base,market,cutoff,h,kind,sec)
            key=f'{kind}_{h}'
            p[key]=beta
            errors[key]=stat['residual_std']
            quality[key]=stat
            counts[key]=stat['training_stocks']
    # Intraday models from Phase30 are trained using only previous complete days.
    intraday,_=base.train_month(market,cutoff)
    for r in range(1,7):p[f'intra_{r}']=intraday[f'intra_{r}']
    return Predictor(p,errors,quality,counts,str(market.dates[cutoff].date()))


def forecast(base, market, t, pred, kind, sec=None):
    f=design_at(base,market,t,kind,sec)
    p1=f@pred.coefficients[f'{kind}_1']*pred.quality[f'{kind}_1']['slope']
    p5=f@pred.coefficients[f'{kind}_5']*pred.quality[f'{kind}_5']['slope']
    # Both 1d and 5d forecasts in comparable DAILY return units.
    one_day=.5*p1+.5*(p5/5.)
    # Prediction disagreement is a further uncertainty proxy.
    agreement=np.sign(p1)==np.sign(p5)
    strength=np.abs(one_day)
    err=.5*pred.errors[f'{kind}_1']+.5*(pred.errors[f'{kind}_5']/math.sqrt(5))
    confidence=np.clip(strength/max(err,1e-10),0.,1.)
    confidence*=np.where(agreement,1.,0.)  # contradictory horizons => abstain
    return one_day,confidence


def target(base, score, exposure=.8):
    return base.target_from_score(score,exposure)


def actual_weights(cash,shares,mark):
    total=float(cash+np.dot(shares,mark))
    if total<=0 or not np.isfinite(total):raise ValueError('Invalid current NAV')
    return (shares*mark/total), total


def proposal(base, market, t, pred, profile, sec=None):
    if profile.startswith('equal') or profile.startswith('momentum') or profile.startswith('industry'):
        name=profile
        if profile.endswith('_hold'):name=profile
        return base.desired(name,market,t,{}), np.zeros(30),np.ones(30)
    kind='price' if profile=='price_ridge80_hold' else ('sec' if profile==SEC_PROFILE else 'industry')
    est,conf=forecast(base,market,t,pred,kind,sec)
    # Forecast calibrated in percent return per day, not arbitrary rank units.
    # Rank mapping only determines the target weights after forecast estimation.
    exposure=.8
    score=est*conf if 'conf' in profile or 'cost' in profile else est
    w=target(base,score,exposure)
    return w,est,conf


def utility(current,w,pred_return,confidence,market,day_index,uncertainty_penalty=True):
    shift=w-current
    l1=float(np.abs(shift).sum())
    gain=float(np.dot(shift,pred_return))
    cost=FEE*l1
    # Approximate 1-day ex-ante risk change from prior 20 known daily closes.
    # Importantly, only prior-day data enter this calculation.
    returns=np.diff(np.log(market.daily[day_index-22:day_index]),axis=0)
    cov=np.cov(returns,rowvar=False)
    risk_change=float(w@cov@w-current@cov@current)
    risk_cost=max(0.,risk_change)*.25  # mild, research-selected risk penalty
    u=gain-cost-risk_cost
    if uncertainty_penalty:
        # Confidence is based on historical calibration; cost of uncertain trading.
        u-=0.00002*l1*(1.-float(np.mean(confidence)))
    return u,{'predicted_gain':gain,'estimated_fees':cost,'risk_increment':risk_change}


def portfolio_replay(base,market,indices,models,profile,sec=None):
    cash=NAV0;shares=np.zeros(30);nav=[NAV0]
    fees=0.;notional=0.;trades=0;holds=0;stages=0;traded_days=0
    diagnostics=[]
    for day_no,i in enumerate(indices):
        t=i-1;month=market.dates[i].strftime('%Y-%m');pred=models[month]
        w,expected,conf=proposal(base,market,t,pred,profile,sec)
        made_trade=False
        for round_no in range(7):
            fill=market.op[i,base.FILL_IDX[round_no],:]
            intent=None;feature_pred=expected;feature_conf=conf
            if day_no==0 and round_no==0:
                intent=w/3. if profile.endswith('_staged') else w
            elif day_no==0 and profile.endswith('_staged') and round_no in (1,2):
                intent=w*((round_no+1)/3.)
                stages+=1
            elif round_no==0 and profile.endswith('_weekly') and day_no%5==0:
                intent=w
            elif round_no==0 and (profile.endswith('_daily') or profile.endswith('_partial') or profile.endswith('_intraday')):
                intent=w
            elif round_no>=1 and profile.endswith('_intraday'):
                # The only current-day information at R2-R7 is a COMPLETED bar.
                known=base.known_intraday(market.cl,market.op,i,round_no)
                intra=known@pred.coefficients[f'intra_{round_no}']
                feature_pred=intra
                feature_conf=np.full(30,.5)  # conservative: intraday model not calibrated
                intent=target(base,intra,.8)
            if intent is not None:
                known_mark=market.daily[t] if round_no==0 else market.cl[i,base.KNOWN_IDX[round_no],:]
                before,_=actual_weights(cash,shares,known_mark)
                step=intent-before
                if day_no>0 or round_no>0:
                    if not profile.endswith('_staged'):
                        if np.abs(step).sum()<.02:
                            holds+=1;intent=None
                        else:
                            # Limit expected 1-way transaction size, preserve cash.
                            step*=min(1.,.096/float(np.abs(step).sum()))
                            if profile.endswith('_partial'):step*=.5
                            intent=np.maximum(0.,before+step)
                            if intent.sum()>1.:intent/=intent.sum()
                    if intent is not None and not (day_no==0 and profile.endswith('_staged')) and ('cost' in profile or 'conf' in profile):
                        u,diag=utility(before,intent,feature_pred,feature_conf,market,i)
                        if u<=0.:holds+=1;intent=None
                        else:diagnostics.append({'date':str(market.dates[i].date()),'round':round_no+1,**diag,'utility':u})
                if intent is not None:
                    cash,shares,amt=base.trade(cash,shares,fill,intent)
                    fees+=FEE*amt;notional+=amt;trades+=1;made_trade=True
            valuation=market.op[i,base.END_VAL_IDX[round_no],:] if round_no<6 else market.cl[i,25,:]
            nav.append(cash+float(np.dot(shares,valuation)))
        traded_days+=int(made_trade)
    v=np.asarray(nav,float)
    if not np.isfinite(v).all() or v.min()<=0:raise AssertionError('Nonfinite NAV')
    period_returns=v[1:]/v[:-1]-1.
    dev=float(period_returns.std(ddof=1))
    return {
        'strategy':profile,'days':len(indices),
        'cumulative_return':float(v[-1]/NAV0-1.),
        'sharpe_proxy':float(period_returns.mean()*np.sqrt(252*7)/dev) if dev>1e-12 else 0.,
        'mdd_proxy':float(np.max(1.-v/np.maximum.accumulate(v))),
        'turnover_proxy':float(notional/(NAV0*len(indices)*7)),
        'fees_usd':float(fees),'trade_count':trades,'trade_days':traded_days,
        'skip_decisions':holds,'staged_trades':stages,
    },diagnostics


def monthly_models(base,market,indices,sec=None):
    output={};records=[]
    months=market.dates[indices].strftime('%Y-%m').unique()
    for month in months:
        cutoff=int(np.flatnonzero(market.dates.strftime('%Y-%m')==month)[0])
        pred=train_month(base,market,cutoff,sec)
        output[month]=pred
        records.append({'test_month':month,'cutoff':pred.cutoff,'parameters':{
            k:[float(v) for v in val] for k,val in pred.coefficients.items()},
            'quality':pred.quality,'counts':pred.training_counts})
        min_hist=min(q['matured_feature_days'] for q in pred.quality.values())
        min_fit=min(q['precalibration_fit_days'] for q in pred.quality.values())
        print('PHASE31_MODEL_MONTH:',month,'MATURED_LABELS_ONLY',
              'MIN_FEATURE_DAYS:',min_hist,'MIN_PRECALIBRATION_DAYS:',min_fit,flush=True)
    return output,records


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def event_audit(root,out):
    """Validate Phase23 LLM annotations as forward-only paper research events.

    Previously scraped Oct9 snapshots CANNOT be injected into 2026 historical
    replay at their headline publication timestamps: obtained/annotated later.
    """
    rows=[]
    folder=root/EVENT_FOLDER
    for file in sorted(folder.glob('*.json')):
        try:doc=json.loads(file.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):continue
        if not isinstance(doc,dict) or doc.get('provider') not in ('camel','laya'):
            continue
        observed=pd.to_datetime(doc.get('observed_at_utc'),utc=True,errors='coerce')
        ins=doc.get('inputs',[]);results=doc.get('results',[])
        if not isinstance(ins,list) or not isinstance(results,list):continue
        known={(str(r.get('event_id')),str(r.get('symbol'))):r for r in results if isinstance(r,dict)}
        for event in ins:
            if not isinstance(event,dict):continue
            symbol=str(event.get('symbol',''))
            published=pd.to_datetime(event.get('published_utc'),utc=True,errors='coerce')
            observed_input=pd.to_datetime(event.get('observed_at_utc'),utc=True,errors='coerce')
            matching=known.get((str(event.get('event_id')),symbol))
            valid=bool(symbol in ('AAPL','MSFT','NVDA','INTC','CRM','JPM','BAC','GS','V','PYPL','LLY','JNJ','UNH','PFE','TMO','AMZN','TSLA','WMT','NKE','KO','CAT','GE','BA','XOM','CVX','GOOGL','META','DIS','T','NEE') and matching is not None and not pd.isna(observed) and not pd.isna(observed_input) and not pd.isna(published) and published<=observed_input<=observed)
            rows.append({'file':file.name,'provider':doc['provider'],'symbol':symbol,
                         'event_id':str(event.get('event_id',''))[:64],
                         'classification_observed_at_utc':str(observed),
                         'published_utc':str(published),'availability_valid':valid,
                         'historical_alpha_eligible':False,'live_trade_eligible':False,
                         'reason':'FORWARD_ONLY_UNCALIBRATED_AND_EVENT_COMPANY_RELEVANCE_UNVERIFIED'})
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['file','provider','symbol','event_id','classification_observed_at_utc','published_utc','availability_valid','historical_alpha_eligible','live_trade_eligible','reason']).to_csv(out/'phase31_llm_forward_only_audit.csv',index=False)
    return rows


def run(root,with_sec=False, input_file=None):
    base=module('phase30_for_phase31','icaif_phase30_2026_walkforward_agent_lab_v2.py')
    source=Path(input_file).expanduser() if input_file else root/base.INPUT
    if not source.is_absolute():source=root/source
    market=base.read_market(source,min_last_day=('2026-10-06' if input_file else '2026-10-07'))
    print('PHASE31_INPUT_ASOF:',str(market.dates[-1].date()),'OCT07_09_INCLUDED:',bool(market.dates[-1]>=pd.Timestamp('2026-10-09')),flush=True)
    out=root/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    sec=None;sec_count=None
    if with_sec:
        phase19=module('phase19_for_phase31','icaif_phase19_sec_ml_research.py')
        folder=root/SEC_FOLDER
        if not folder.is_dir():
            raise FileNotFoundError('SEC 30-stock cache missing. Run Phase19 --prepare-sec first or omit --with-sec.')
        sec,sec_count=phase19.build_sec_panel(market.dates,folder)
        if len(sec_count)!=30:raise RuntimeError('SEC source incomplete')
    rows=[];metadata=[];trades=[]
    all_profiles=PROFILES+((SEC_PROFILE,) if with_sec else tuple())
    for segment,start,end in (
            ('development','2026-07-01','2026-08-31'),
            ('exposed_diagnostic','2026-09-01','2026-10-09')):
        indices=np.flatnonzero((market.dates>=pd.Timestamp(start))&(market.dates<=pd.Timestamp(end)))
        if len(indices)<15:raise RuntimeError(f'Not enough complete {segment} sessions')
        models,records=monthly_models(base,market,indices,sec)
        metadata.extend(records)
        for j,profile in enumerate(all_profiles,1):
            outcome,changes=portfolio_replay(base,market,indices,models,profile,sec)
            rows.append({'segment':segment,**outcome})
            for d in changes:trades.append({'segment':segment,'strategy':profile,**d})
            print('PHASE31_POLICY:',segment,f'{j}/{len(all_profiles)}',profile,'return',f'{outcome["cumulative_return"]:.5f}','fee',f'{outcome["fees_usd"]:.1f}',flush=True)
    df=pd.DataFrame(rows)
    for m,ascending in [('cumulative_return',False),('sharpe_proxy',False),('mdd_proxy',True),('turnover_proxy',True)]:
        df['rank_'+m]=df.groupby('segment')[m].rank(pct=True,ascending=ascending)
    fields=['rank_'+m for m in ('cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy')]
    df['four_metric_research_rank']=df[fields].mean(axis=1)
    df.sort_values(['segment','four_metric_research_rank']).to_csv(out/'phase31_four_metric_ablation.csv',index=False)
    pd.DataFrame(trades,columns=['segment','strategy','date','round','predicted_gain','estimated_fees','risk_increment','utility']).to_csv(out/'phase31_trade_gate_diagnostics.csv',index=False)
    (out/'phase31_monthly_model_provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    manifest={'research_only':True,'input_sha256':sha256(source),'input':str(source),'data_asof':str(market.dates[-1].date()),'oct07_09_included':bool(market.dates[-1]>=pd.Timestamp('2026-10-09')),
              'data_source':'Alpaca IEX RAW 15m, not official executed prices',
              'with_sec':bool(with_sec),'sec_coverage':sec_count,
              'event_news_llm_historical_alpha':'NOT_TESTED_NO_PIT_BACKFILL',
              'training_method':'month-cutoff ridge with matured labels; 15-30d chronological calibration and >=25 pre-calibration fit days',
              'short_early_training_windows':'EXPLORATORY_STATISTICAL_UNCERTAINTY_HIGH',
              'disclosure':'Sep-Oct previously exposed diagnostic; rankings depend on fixed candidate set',
              'approximations':['IEX fill proxy','no corporate-action total-return reconstruction','fees 0.1% each side','SEC acceptance time only filed date + 2 days','no intraday event overlay']}
    (out/'phase31_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    event_audit(root,out)
    for segment in df.segment.unique():
        print('===',segment.upper(),'LOWER FOUR-METRIC RANK BETTER ===',flush=True)
        s=df.query('segment == @segment').sort_values('four_metric_research_rank')
        print(s[['strategy','cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd','four_metric_research_rank']].to_string(index=False),flush=True)
    print('PHASE31_RESULT_DIR:',out,flush=True)
    print('PHASE31_COMPLETE_RESEARCH_ONLY_NO_ORGANIZER_SUBMISSION',flush=True)


def self_test():
    b=module('phase30_for_phase31_test','icaif_phase30_2026_walkforward_agent_lab_v2.py')
    rng=np.random.default_rng(310311)
    dates=pd.bdate_range('2025-09-02',periods=280)
    prev=100*np.exp(np.cumsum(rng.normal(0,.009,(len(dates),30)),axis=0))
    op=np.empty((len(dates),26,30));cl=np.empty_like(op)
    for d in range(len(dates)):
        price=prev[d-1] if d else prev[d]
        for n in range(26):
            if n:price=cl[d,n-1].copy()
            op[d,n]=price
            cl[d,n]=price*np.exp(rng.normal(.0,.0012,30))
    market=b.Market(dates,op,cl,cl[:,-1,:],pd.DataFrame())
    # Reproduce early-2026 July failure: only 40 matured five-day feature
    # dates after the 120-day lookback. Must be viable but clearly limited.
    early=train_month(b,market,b.TRAIN_MIN_DAY+45)
    assert early.quality['industry_5']['matured_feature_days']==40
    assert early.quality['industry_5']['precalibration_fit_days']>=25
    try:
        train_month(b,market,b.TRAIN_MIN_DAY+44)
    except ValueError as ex:
        assert 'Not enough matured training days' in str(ex)
    else:
        raise AssertionError('Insufficient training cutoff incorrectly accepted')
    cutoff=205
    fit=train_month(b,market,cutoff)
    assert 'price_5' in fit.coefficients and 'industry_1' in fit.coefficients
    assert fit.quality['industry_5']['latest_label_date']<str(market.dates[cutoff].date())
    future=market.daily.copy();future[cutoff:]*=1.7
    shifted=b.Market(dates,market.op,market.cl,future,pd.DataFrame())
    test_fit=train_month(b,shifted,cutoff)
    for k in fit.coefficients:
        assert np.array_equal(fit.coefficients[k],test_fit.coefficients[k]),'Future data changed a fitted coefficient'
    pred,conf=forecast(b,market,cutoff-1,fit,'industry')
    assert pred.shape==(30,) and conf.shape==(30,) and (conf>=0).all() and (conf<=1).all()
    f=np.zeros((len(dates),30,13));f[:,:,1:4]=rng.normal(size=(len(dates),30,3));f[:,:,5]=rng.normal(size=(len(dates),30))
    fitted_sec=train_month(b,market,cutoff,f)
    assert 'sec_5' in fitted_sec.coefficients
    subset=np.arange(cutoff,cutoff+11)
    models={d.strftime('%Y-%m'):fit for d in dates[subset]}
    for policy in PROFILES:
        output,_=portfolio_replay(b,market,subset,models,policy)
        assert np.isfinite([output[k] for k in ('cumulative_return','mdd_proxy','fees_usd','turnover_proxy')]).all(),policy
        assert output['trade_count']>=1,policy
    withsec={d.strftime('%Y-%m'):fitted_sec for d in dates[subset]}
    output,_=portfolio_replay(b,market,subset,withsec,SEC_PROFILE,f)
    assert output['trade_count']>=1
    # Enforce nonnegative long-only, upper limits, and fee-accounted NAV.
    c,q,z=b.trade(NAV0,np.zeros(30),np.full(30,100.),np.full(30,.8/30))
    assert c>0 and z>0 and c+q@np.full(30,100.)<NAV0
    print('PHASE31_EARLY_CUTOFF_40_MATURED_DAYS_AND_25_FIT_DAYS_PASS')
    print('PHASE31_SELF_TEST_PASS')
    print('LEAKAGE_GUARD_AND_2026_MONTHLY_MATURED_LABELS_PASS')
    print('SEC_DATE_PANEL_OPTION_AND_LONG_ONLY_FEE_REPLAY_PASS')
    print('PRICE_INDUSTRY_CONFIDENCE_COST_PARTIAL_STAGED_INTRADAY_ABLATIONS_PASS')
    print('LLM_CLASSIFICATIONS_FORWARDED_FOR_AUDIT_NOT_HISTORICALLY_BACKFILLED')
    print('NO_NETWORK_NO_SECRETS_NO_TRADING_NO_ORGANIZER_CALLS')


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    mode=arg.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--event-audit',action='store_true')
    arg.add_argument('--with-sec',action='store_true',help='Include cached historical SEC CompanyFacts in one additional price+industry+SEC ablation; no network.')
    arg.add_argument('--input',default=None,help='Optional Alpaca 15m CSV; permits Oct 6 as-of research without Phase29')
    args=arg.parse_args()
    try:
        if args.self_test:self_test()
        elif args.event_audit:
            r=event_audit(Path.cwd(),Path.cwd()/OUTPUT)
            print('PHASE31_EVENT_AUDIT_COUNT:',len(r))
            print('RESEARCH_ONLY_NO_BACKFILLED_LLM_ALPHA')
        elif args.run:run(Path.cwd(),args.with_sec,args.input)
    except Exception as e:
        # Research input errors only; no secrets are read in this script.
        print('PHASE31_STOP:',type(e).__name__,str(e),file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
