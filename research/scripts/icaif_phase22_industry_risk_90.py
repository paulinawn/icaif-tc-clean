#!/usr/bin/env python3
"""ICAIF 2026 Phase22: six-group rotation, 90% exposure, shrinkage risk optimization.

RESEARCH ONLY: no network, credentials, organizer calls, service writes or submissions.
Run inside starter-kit with the existing Python .venv activated.

  python "$HOME\\Downloads\\icaif_phase22_industry_risk_90.py" --self-test
  python "$HOME\\Downloads\\icaif_phase22_industry_risk_90.py" --run

Requires existing Phase15 script in Downloads and the official historical parquet.
Signal at preceding *completed* trading close; trade at next daily open proxy.
Historical 7 endpoints are hourly proxies, NOT genuine official fill data.
2023-2024 development, 2025 EXPOSED diagnostic, 2026 untouched.
15-session windows independently reset, thus NOT a continuous Official backtest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SYMBOLS = tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
GROUPS = (
    'Technology', 'Finance', 'Healthcare', 'Consumer',
    'Industrial & Energy', 'Communication & Utilities',
)
SYMBOL_GROUPS = tuple(g for g in GROUPS for _ in range(5))
FEE = .001
N = 30
DAYS = 15
INITIAL_NAV = 1_000_000.
ANNUALIZER = math.sqrt(252 * 7)
OUTPUT = Path('artifacts/phase22_industry_risk_90')

@dataclass(frozen=True)
class Candidate:
    name: str
    exposure: float
    industry: bool
    risk: bool
    cadence: int             # 0=buy/hold for 15 sessions, 5=weekly
    gate: float              # two-sided turnover target threshold, e.g. 0.06
    cap: float = .12         # max per-decision turnover fraction (not initial)
    regime: bool = False     # reduce to 60% invested in defensive regime

POLICIES = (
    Candidate('equal80_hold', .8, False, False, 0, .0),
    Candidate('equal90_hold', .9, False, False, 0, .0),
    Candidate('equal100_hold', 1.0, False, False, 0, .0),
    Candidate('industry80_hold', .8, True, False, 0, .0),
    Candidate('industry90_hold', .9, True, False, 0, .0),
    Candidate('industry100_hold', 1.0, True, False, 0, .0),
    Candidate('risk90_hold', .9, False, True, 0, .0),
    Candidate('industry_risk90_hold', .9, True, True, 0, .0),
    Candidate('industry90_weekly_gate06', .9, True, False, 5, .06),
    Candidate('industry_risk90_weekly_gate06', .9, True, True, 5, .06),
    Candidate('industry_risk90_weekly_gate10', .9, True, True, 5, .10),
    Candidate('industry_risk90_regime', .9, True, True, 5, .06, regime=True),
)

def ensure_root():
    root = Path.cwd().resolve()
    if not (root / 'tools' / 'auto_submit.py').is_file():
        raise RuntimeError('Run inside the starter-kit folder in VS Code.')
    return root


def load_phase15():
    folder = Path.home() / 'Downloads'
    candidates = (folder / 'icaif_phase15_official_challengers_v2.py',
                  folder / 'icaif_phase15_official_challengers.py')
    target = next((p for p in candidates if p.is_file()), None)
    if target is None:
        raise FileNotFoundError('Download the Phase15 v2 script into Downloads first (existing project dependency).')
    spec = importlib.util.spec_from_file_location('icaif_phase15_for_phase22', target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if tuple(module.SYMBOLS) != SYMBOLS:
        raise RuntimeError('Phase15 symbol list does not match the official universe.')
    return module


def check_weights(weights, gross=None):
    w = np.asarray(weights, dtype=float)
    if w.shape != (N,) or not np.all(np.isfinite(w)) or np.any(w < -1e-10) or np.any(w > .30+1e-10):
        raise ValueError('Noncompliant 30-stock allocation')
    if w.sum() > 1.0 + 1e-9:
        raise ValueError('Leverage')
    if gross is not None and abs(float(w.sum())-gross) > 1e-7:
        raise ValueError('Wrong equity exposure')
    return w


def safe_return(data, period, i):
    if i < period:
        raise ValueError('Not enough lagged observations')
    p0 = data[i-period]
    p1 = data[i]
    if not np.all(np.isfinite(p0)) or np.any(p0 <= 0) or not np.all(np.isfinite(p1)):
        raise ValueError('Bad source prices')
    return p1/p0 - 1.0


def standard_rank(a):
    v = np.asarray(a, float)
    if v.shape != (N,) or not np.isfinite(v).all():
        raise ValueError('Invalid ranking signal')
    return pd.Series(v).rank(method='average',pct=True).to_numpy(dtype=float)*2.0-1.0


def market_state(daily, i):
    """Fixed defensive state; no future reads or learned thresholds."""
    market_20 = float(np.median(safe_return(daily,20,i)))
    market_60 = float(np.median(safe_return(daily,60,i)))
    return market_20 < 0 and market_60 < 0


def industry_prior(daily, groups, idx, *, gross, active):
    """Top-level sector relative strength + stock momentum/reversal, shrunk toward equal."""
    if idx < 121:
        raise ValueError('Require 120 prior completed sessions')
    if not active:
        return np.full(N, gross/N)
    r5 = safe_return(daily,5,idx)
    r20 = safe_return(daily,20,idx)
    r60 = safe_return(daily,60,idx)
    r120 = safe_return(daily,120,idx)
    group_idx = [np.array([i for i,g in enumerate(groups) if g==name],int) for name in GROUPS]
    if any(len(ii)!=5 for ii in group_idx):
        raise ValueError('Not exactly five companies per competition group')
    # 20/60/120-session momentum with more weight on intermediate trend.
    strength = .25*r20 + .50*r60 + .25*r120
    sector_s = np.array([float(np.median(strength[ii])) for ii in group_idx])
    sector_rank = pd.Series(sector_s).rank(method='average',pct=True).to_numpy()*2-1
    budget = (gross/6)*(1+.45*sector_rank)
    budget *= gross/budget.sum()
    # Preserve groups in roughly 9%-24% for gross=90%, rather than all-in chasing a leader.
    result=np.zeros(N,float)
    for ii, grp_budget in zip(group_idx,budget):
        stock_s = (.75*strength[ii] - .25*r5[ii])
        stock_rank = pd.Series(stock_s).rank(method='average',pct=True).to_numpy()*2-1
        x = 1+.35*stock_rank
        result[ii]=grp_budget*x/x.sum()
    check_weights(result,gross)
    return result


def shrunk_covariance(daily, idx):
    """Ledoit-Wolf covariance of 80 historically completed daily returns."""
    from sklearn.covariance import LedoitWolf
    if idx < 82:
        raise ValueError('Insufficient covariance history')
    history = daily[idx-80:idx+1]
    ret=history[1:]/history[:-1]-1.0
    if not np.all(np.isfinite(ret)):
        raise ValueError('Bad volatility history')
    # Clipping explicitly limits sensitivity to gross corporate action anomalies.
    ret=np.clip(ret,-.40,.40)
    cov=LedoitWolf(store_precision=False,assume_centered=False).fit(ret).covariance_
    return cov + np.eye(N)*1e-10


def risk_optimized_prior(prior, cov, gross, groups):
    """Constrained quadratic shrinkage optimizer, not unconstrained RL.

    Minimize deviation from alpha prior + covariance risk; preserve moderate
    group diversification and 8% stock limit. Falls back to prior on solver failure.
    """
    from scipy.optimize import minimize
    p=check_weights(prior,gross)
    if cov.shape != (N,N) or not np.all(np.isfinite(cov)):
        raise ValueError('Invalid covariance matrix')
    scale=max(float(np.median(np.diag(cov))),1e-10)
    C=cov/scale
    # Quadratic risk penalty is a research setting, not an optimized alpha estimate.
    risk_coeff=.14
    def fun(w):
        diff=w-p
        return .5*np.dot(diff,diff)+.5*risk_coeff*np.dot(w,C@w)
    def jac(w):
        return (w-p)+risk_coeff*(C@w)
    a=np.ones(N)
    constraints=[{'type':'eq','fun':lambda w: float(np.sum(w)-gross), 'jac':lambda w: a}]
    for name in GROUPS:
        indices=np.array([k for k,g in enumerate(groups) if g==name],int)
        if len(indices)!=5: raise ValueError('Sector taxonomy error')
        bound=np.zeros(N);bound[indices]=1
        lo,hi=.065/.9*gross,.235/.9*gross
        constraints.append({'type':'ineq','fun':lambda w,b=bound,l=lo:float(np.dot(w,b)-l),
                            'jac':lambda w,b=bound,l=lo:b})
        constraints.append({'type':'ineq','fun':lambda w,b=bound,h=hi:float(h-np.dot(w,b)),
                            'jac':lambda w,b=bound,h=hi:-b})
    res=minimize(fun,p,jac=jac,method='SLSQP',bounds=[(0,.08)]*N,
                 constraints=constraints,options={'maxiter':100,'ftol':1e-10,'disp':False})
    if not res.success:
        # A failed risk optimization is *not* hidden: analysis later counts fallbacks.
        return p.copy(),True
    ans=np.asarray(res.x,float)
    check_weights(ans,gross)
    return ans,False


def fee_trade(cash,shares,price,target):
    check_weights(target)
    if not np.isfinite(price).all() or (price<=0).any():raise ValueError('Invalid execution price')
    nav_before=float(cash+np.dot(shares,price))
    held=shares*price
    nav=nav_before
    for _ in range(120):
        notional=float(np.abs(target*nav-held).sum())
        nxt=nav_before-FEE*notional
        if abs(nxt-nav)<1e-8:
            nav=nxt
            break
        nav=nxt
    wanted=target*nav
    traded=float(np.abs(wanted-held).sum())
    fee=traded*FEE
    new_cash=float(nav-wanted.sum())
    if new_cash < -1e-6 or nav <= 0:
        raise ValueError('Insufficient cash')
    return new_cash, wanted/price, traded, fee, nav_before


class Engine:
    def __init__(self, dates, opens, endpoints, daily, groups):
        self.dates=dates
        self.opens=opens
        self.endpoints=endpoints
        self.daily=daily
        self.groups=groups
        if len(groups)!=30 or len(set(groups))!=6:
            raise ValueError('Six sector groups required')
        if sorted(tuple(groups)) != sorted(SYMBOL_GROUPS):
            raise ValueError('Industry symbols not in competition group ordering')
        self.solver_fallbacks=set()

    @lru_cache(maxsize=5000)
    def weight(self, idx:int, name:str):
        policy=next(p for p in POLICIES if p.name==name)
        gross=policy.exposure
        if policy.regime and market_state(self.daily,idx):gross=.60
        base=industry_prior(self.daily,self.groups,idx,gross=gross,active=policy.industry)
        if policy.risk:
            cov=shrunk_covariance(self.daily,idx)
            base,failed=risk_optimized_prior(base,cov,gross,self.groups)
            if failed:self.solver_fallbacks.add((idx,name))
        return check_weights(base,gross)

    def replay(self,start,policy:Candidate,size=15):
        cash=INITIAL_NAV
        shares=np.zeros(N)
        path=[INITIAL_NAV]
        fee_total,notion_total,trade_count,turnover=0.,0.,0,0.
        for d in range(size):
            k=start+d
            px=self.opens[k]
            nav=float(cash+np.dot(shares,px))
            # IMPORTANT: the current session opening fill is not observable before R1.
            # Gate/trade size uses holdings marked at the PREVIOUS completed close.
            prior_mark=self.daily[k-1]
            prior_nav=float(cash+np.dot(shares,prior_mark))
            should_check=d==0 or (policy.cadence and d%policy.cadence==0)
            if should_check:
                target=self.weight(k-1,policy.name).copy()
                current=shares*prior_mark/prior_nav
                drift=float(np.abs(target-current).sum())
                if d==0 or drift >= policy.gate:
                    if d>0 and drift > policy.cap:
                        target=current + ((policy.cap*.90)/drift)*(target-current)
                        check_weights(target)
                    cash,shares,notion,fee,old_nav=fee_trade(cash,shares,px,target)
                    fee_total+=fee
                    notion_total+=notion
                    turnover+=notion/old_nav
                    trade_count+=1
                    if d>0 and notion/old_nav>policy.cap+1e-5:
                        raise RuntimeError('Turnover cap violated')
            for rnd in range(7):
                level=float(cash+np.dot(shares,self.endpoints[k,rnd]))
                if not np.isfinite(level) or level<=0:
                    raise RuntimeError('Invalid valuation')
                path.append(level)
        curve=np.asarray(path)
        ret=curve[1:]/curve[:-1]-1
        vol=float(np.std(ret,ddof=1))
        return {
            'cumulative_return':float(curve[-1]/INITIAL_NAV-1),
            'sharpe_proxy':float(ret.mean()*ANNUALIZER/vol) if vol>1e-12 else 0.,
            'mdd_proxy':float(np.max(1-curve/np.maximum.accumulate(curve))),
            'turnover_proxy':turnover/(size*7),
            'fees_usd':fee_total,
            'trade_count':trade_count,
            'notional_usd':notion_total,
        }


def start_windows(dates,year):
    ix=np.flatnonzero(dates.year==year)
    return [int(s) for s in ix[1::DAYS] if s>=125 and s+DAYS<=len(dates)
            and dates[s+DAYS-1].year==year]


def rankings(out):
    dev=out[out.year.isin((2023,2024))].copy()
    for metric,asc in [('cumulative_return',False),('sharpe_proxy',False),
                       ('mdd_proxy',True),('turnover_proxy',True)]:
        # Rank each candidate within the exact same 15-day period.
        dev['rank_'+metric]=dev.groupby(['year','start'])[metric].rank(ascending=asc,pct=True,method='average')
    keys=[c for c in dev.columns if c.startswith('rank_')]
    dev['mean_four_metric_rank']=dev[keys].mean(axis=1)
    summary=dev.groupby('strategy',as_index=False).agg(
        mean_window_rank=('mean_four_metric_rank','mean'),
        worst_year_rank=('mean_four_metric_rank',lambda x:float(np.max([
            np.mean(x[dev.loc[x.index,'year']==y]) for y in (2023,2024)]))),
        windows=('mean_four_metric_rank','size'),
    )
    summary['conservative_rank']=(summary.mean_window_rank+summary.worst_year_rank)/2
    return summary.sort_values('conservative_rank')


def run(root):
    phase15=load_phase15()
    dates,opens,endpoints,daily,groups=phase15.load_history(phase15.source_path(root))
    # The organizer's machine-readable sector label spelling may vary; the
    # five-stock GROUPS ordering is fixed by the verified competition universe.
    if len(set(groups))!=6 or any(len(set(groups[i:i+5]))!=1 for i in range(0,N,5)):
        raise RuntimeError('Historical sector partitions do not match six consecutive groups of five stocks')
    engine=Engine(dates,opens,endpoints,daily,SYMBOL_GROUPS)
    rows=[]
    for yr in (2023,2024,2025):
        windows=start_windows(dates,yr)
        for n,s in enumerate(windows):
            for policy in POLICIES:
                result=engine.replay(s,policy,DAYS)
                rows.append({'year':yr,'start':str(dates[s].date()),
                             'strategy':policy.name, **result})
            if (n+1)%4==0:
                print(f'YEAR_{yr}_WINDOWS_COMPLETED: {n+1}/{len(windows)}',flush=True)
        print(f'YEAR_{yr}_WINDOW_COUNT: {len(windows)}',flush=True)
    out=pd.DataFrame(rows)
    metrics=['cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd','trade_count']
    summary=out.groupby(['year','strategy'],as_index=False)[metrics].mean()
    ranking=rankings(out)
    print('=== YEARLY 15-DAY WINDOW MEANS: research proxies, not Official results ===')
    for yr in (2023,2024,2025):
        print(f'YEAR {yr}:')
        print(summary[summary.year==yr].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('=== DEVELOPMENT 2023-2024: paired four-metric ranks, lower is better ===')
    print(ranking.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('=== 2025: exposed diagnostic only, not a pristine holdout ===')
    print(summary[summary.year==2025][['strategy','cumulative_return','sharpe_proxy',
                                        'mdd_proxy','turnover_proxy']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    out_folder=root/OUTPUT
    out_folder.mkdir(parents=True,exist_ok=True)
    out.to_csv(out_folder/'phase22_window_results.csv',index=False)
    summary.to_csv(out_folder/'phase22_year_summary.csv',index=False)
    ranking.to_csv(out_folder/'phase22_paired_ranks.csv',index=False)
    manifest={
        'status':'RESEARCH_ONLY_NOT_PRODUCTION',
        'strategy_count':len(POLICIES),
        'strategies':[asdict(p) for p in POLICIES],
        'source':'official historical hourly parquet, with hourly execution proxies',
        'signal_availability':'prior completed session only',
        'window':'15 sessions independently reset; not continuous Official path',
        'development':[2023,2024],
        'diagnostic_exposed':[2025],
        '2026':'reserved_not_tested',
        'cost':'0.1% fee each buy/sell notional; turnover separate competition metric',
        'risk_estimation':'80-session historical covariance LedoitWolf, constrained SLSQP',
        'risk_optimizer_fallback_count':len(engine.solver_fallbacks),
        'risk_optimizer_fallbacks':[list(z) for z in sorted(engine.solver_fallbacks)[:50]],
        'known_gaps':['price-basis not fully verified','hourly bar vs official fill mismatch',
                      'not seven round-aware decisions','previous years already inspected in prior projects'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'decision':'Compare candidate by four-metric paired ranks and robust holdout before any promotion',
    }
    (out_folder/'phase22_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out_folder/'CHANGELOG_PHASE22.md').write_text(
        '# Phase 22 — industry/risk/exposure research\n\n'
        '**Hypothesis:** six-group momentum rotation and shrinkage-covariance risk optimization improve '
        'the four competition metrics relative to equal80/90/100.\n\n'
        '**Code:** SHA256 '+manifest['source_sha256']+'\n\n'
        '**Sample:** development 2023–2024, exposed 2025, 2026 unused; hourly historical data.\n\n'
        '**Result:** inspect phase22_year_summary.csv and phase22_paired_ranks.csv.\n\n'
        '**Caveat:** prices/proxies and multiple prior research trials preclude claiming verified alpha.\n',encoding='utf-8')
    print('RISK_OPTIMIZATION_FALLBACKS:',len(engine.solver_fallbacks))
    print('RESULT_DIR:',out_folder)
    print('NO_TRADING_NO_NETWORK_NO_LIVE_PROMOTION')
    print('PHASE22_COMPLETE')


def self_test():
    np.random.seed(2244)
    groups=SYMBOL_GROUPS
    n=380
    rng=np.random.default_rng(2323)
    ret=rng.normal(.0003,.011,size=(n,N))
    daily=100*np.cumprod(1+ret,axis=0)
    dates=pd.bdate_range('2023-01-01',periods=n)
    opens=daily.copy()
    endpoints=np.stack([daily*(1+(j-3)*.0001) for j in range(7)],axis=1)
    engine=Engine(dates,opens,endpoints,daily,groups)
    equal=engine.weight(250,'equal90_hold')
    assert abs(equal.sum()-.90)<1e-9
    assert np.allclose(equal, .03)
    ind=engine.weight(250,'industry90_hold')
    assert abs(ind.sum()-.90)<1e-9 and np.ptp(ind)>.005
    rsk=engine.weight(250,'industry_risk90_hold')
    check_weights(rsk,.90)
    assert max(rsk)<.08+1e-9
    assert all(.065/.9*.90-1e-5 <= rsk[np.array(groups)==g].sum()<=.235/.9*.90+1e-5 for g in GROUPS)
    p=next(x for x in POLICIES if x.name=='industry_risk90_weekly_gate06')
    replay=engine.replay(255,p)
    assert replay['fees_usd']>0 and replay['trade_count']>=1
    # Market data after the decision must not influence the corresponding target.
    copy=daily.copy()
    copy[251:]*=2
    engine_2=Engine(dates,opens,endpoints,copy,groups)
    assert np.allclose(ind,engine_2.weight(250,'industry90_hold'))
    assert np.allclose(rsk,engine_2.weight(250,'industry_risk90_hold'))
    # There is no network or organizer code in this experiment.
    print('PHASE22_SELF_TEST_PASS')
    print('SIX_INDUSTRY_ALLOCATION_AND_90_PCT_EXPOSURE_PASS')
    print('SHRUNK_COVARIANCE_CONSTRAINED_OPTIMIZER_PASS')
    print('PRIOR_DAY_NO_LOOKAHEAD_AND_FEES_PASS')
    print('NO_NETWORK_NO_ORGANIZER_SUBMISSIONS')


def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:run(ensure_root())

if __name__=='__main__':
    main()
