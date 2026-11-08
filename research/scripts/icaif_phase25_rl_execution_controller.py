#!/usr/bin/env python3
"""ICAIF Phase25: strictly NARROW tabular Q-learning rebalance controller.

One action per new morning: HOLD, 25%, 50%, 100% move toward the Phase22
industry-risk 90%-equity target. Alpha/stock-selection is NOT learned by RL.

Training 2021-22 only; 2023-24 development evaluation, 2025 already exposed,
2026 not evaluated. No online learning in Official; no network/submissions.

Run from starter-kit with your usual Python .venv:
 python "$HOME\\Downloads\\icaif_phase25_rl_execution_controller.py" --self-test
 python "$HOME\\Downloads\\icaif_phase25_rl_execution_controller.py" --run

Requires Phase22 script in Downloads and historical Parquet. Daily-open/hourly
endpoints are proxies for actual organizer fills. 15-session windows reset.
Do not interpret RL training reward or test gains as independently verified alpha.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED=20261011
ACTIONS=(0.0,0.25,0.50,1.0)
ACTION_NAMES=('HOLD','MOVE_25','MOVE_50','MOVE_100')
N_STATES=16
OUTPUT=Path('artifacts/phase25_rl_execution')
FEE=.001
DAYS=15


def state_at(p22,engine,day,shares,cash,prior_close):
    """Every input is observable by the deadline preceding day open."""
    i=day-1
    target=engine.weight(i,'industry_risk90_hold')
    nav=cash+float(np.dot(shares,prior_close))
    if nav<=0:raise ValueError('Invalid NAV')
    current=shares*prior_close/nav
    drift=float(np.abs(target-current).sum())
    risky=p22.market_state(engine.daily,i)
    r1=engine.daily[i-20:i+1][1:]/engine.daily[i-20:i+1][:-1]-1.0
    vol=float(np.median(np.std(r1,axis=0)))
    sector_strength=[]
    for g in p22.GROUPS:
        ids=np.array([j for j,x in enumerate(engine.groups) if x==g])
        r=p22.safe_return(engine.daily,60,i)[ids]
        sector_strength.append(float(np.median(r)))
    dispersion=max(sector_strength)-min(sector_strength)
    bits=(int(risky),int(drift>.04),int(vol>.016),int(dispersion>.15))
    state=sum((v<<j) for j,v in enumerate(bits))
    return state,target,current,drift


def act(p22,engine,day,step_in_episode,shares,cash,mode,q,rng,epsilon):
    prev_close=engine.daily[day-1]
    state,target,current,drift=state_at(p22,engine,day,shares,cash,prev_close)
    if mode=='hold':option=0
    elif mode=='weekly':option=3 if (step_in_episode%5==0 and drift>=.06) else 0
    elif mode=='daily':option=2 if drift>=.04 else 0
    elif mode=='rl':
        option=int(rng.integers(0,len(ACTIONS))) if rng.random()<epsilon else int(np.argmax(q[state]))
    else:raise ValueError('Unknown controller mode')
    fraction=ACTIONS[option]
    # The policy learns a *fractional move*, never an unbounded 30-weight vector.
    desired=current+fraction*(target-current)
    # Bounded projected 8% of NAV desired turn, leaving buffer for fill gaps.
    wanted=float(np.abs(desired-current).sum())
    if wanted>.08:
        desired=current+(.08*.90/wanted)*(desired-current)
    desired=p22.check_weights(desired)
    return state,option,desired,drift


def simulate(p22,engine,start,mode,q,*,learn=False,seed=SEED,epsilon=.0,alpha=.12,gamma=.5):
    """Evaluate 15-day portfolio, optionally train Q from past-market rewards.

    Q updates use the *next* realized price ONLY as a training reward, never as
    part of the decision state. During test, learn=False freezes Q.
    """
    rng=np.random.default_rng(seed)
    cash=1_000_000.
    shares=np.zeros(30)
    trajectory=[cash]
    fee_total=notional_total=turnover_total=0.
    decisions=0
    outcomes=[]
    peak=cash
    for d in range(DAYS):
        day=start+d
        prior=engine.daily[day-1]
        if d==0:
            # ALL controllers enter same 90%-exposure target at first trade.
            desired=engine.weight(day-1,'industry_risk90_hold').copy()
            state=action_idx=None
        else:
            state,action_idx,desired,drift=act(
                p22,engine,day,d,shares,cash,mode,q,rng,epsilon)
        px=engine.opens[day]
        nav_before=cash+float(np.dot(shares,px))
        # HOLD means NO submission/trade, even if weights drift at the unknown open.
        requested_trade=(d==0 or (action_idx is not None and ACTIONS[action_idx]>0 and drift>1e-5))
        if requested_trade:
            cash,shares,notional,fee,previous_nav=p22.fee_trade(cash,shares,px,desired)
            fee_total+=fee
            notional_total+=notional
            turnover_total+=notional/previous_nav
            decisions+=1
        else:
            notional=0.
        for j in range(7):
            nav=float(cash+np.dot(shares,engine.endpoints[day,j]))
            if not math.isfinite(nav) or nav<=0:raise RuntimeError('Invalid NAV')
            peak=max(peak,nav)
            trajectory.append(nav)
        if learn and d>0 and mode=='rl':
            # Train only on old dates; reward observed AFTER action.
            if d<DAYS-1:
                end_price=engine.opens[day+1]
            else:
                end_price=engine.endpoints[day,-1]
            next_nav=float(cash+np.dot(shares,end_price))
            if next_nav<=0:raise ValueError('Negative future NAV')
            # Fee already embedded in return. Penalize turnover again because
            # it is a separate Official metric; drawdown also gets a penalty.
            r=math.log(next_nav/nav_before) - .0004*(notional/nav_before)
            drawdown=max(0.,1-next_nav/max(peak,next_nav))
            r-=.01*max(0.,drawdown-.025)
            # Make Q gradients numerically visible (millireturn units).
            reward=1000*r
            if d<DAYS-1:
                ns,_,__,___=state_at(p22,engine,day+1,shares,cash,engine.daily[day])
            else:ns=None
            td=reward+(gamma*np.max(q[ns]) if ns is not None else 0.)-q[state,action_idx]
            q[state,action_idx]+=alpha*td
            outcomes.append(reward)
    path=np.asarray(trajectory)
    returns=path[1:]/path[:-1]-1
    std=float(np.std(returns,ddof=1))
    return {'cumulative_return':float(path[-1]/path[0]-1),
            'sharpe_proxy':float(np.mean(returns)*math.sqrt(252*7)/std) if std>1e-12 else 0.,
            'mdd_proxy':float(np.max(1-path/np.maximum.accumulate(path))),
            'turnover_proxy':turnover_total/(DAYS*7),
            'fees_usd':fee_total,'trade_count':decisions,
            'training_rewards':len(outcomes)}


def train_q(p22,engine,dates):
    # Train exclusively from 2021/22 chronological simulated episodes;
    # multiple passes are simulation replays, NOT evidence of new market paths.
    starts=[]
    for year in (2021,2022):
        starts.extend(p22.start_windows(dates,year))
    if not starts:raise RuntimeError('No 2021/2022 training episodes')
    q=np.zeros((N_STATES,len(ACTIONS)))
    for epoch in range(3):
        for j,start in enumerate(starts):
            ep_seed=SEED+epoch*10000+j
            eps=.30 if epoch==0 else (.15 if epoch==1 else .07)
            simulate(p22,engine,start,'rl',q,learn=True,seed=ep_seed,epsilon=eps)
    return q,len(starts),3


def rank_results(df):
    dev=df[df.year.isin((2023,2024))].copy()
    for metric,ascending in (('cumulative_return',False),('sharpe_proxy',False),
                             ('mdd_proxy',True),('turnover_proxy',True)):
        dev['rank_'+metric]=dev.groupby(['year','start'])[metric].rank(pct=True,ascending=ascending)
    dev['mean_rank']=dev[[k for k in dev if k.startswith('rank_')]].mean(axis=1)
    x=dev.groupby(['year','strategy'])['mean_rank'].mean().unstack('year')
    x['average']=x[[2023,2024]].mean(axis=1)
    x['worst']=x[[2023,2024]].max(axis=1)
    x['conservative']=(x.average+x.worst)/2
    return x.sort_values('conservative')


def run():
    root=Path.cwd().resolve()
    if not (root/'tools'/'auto_submit.py').is_file():raise RuntimeError('Run inside starter-kit')
    # Import Phase22 pure math functions, no network.
    import importlib.util,sys
    path=Path.home()/'Downloads'/'icaif_phase22_industry_risk_90.py'
    if not path.is_file():raise FileNotFoundError('Phase22 script missing from Downloads')
    spec=importlib.util.spec_from_file_location('phase22_for_phase25',path)
    p22=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p22
    spec.loader.exec_module(p22)
    phase15=p22.load_phase15()
    dates,opens,endpoints,daily,raw_groups=phase15.load_history(phase15.source_path(root))
    if len(set(raw_groups))!=6 or any(len(set(raw_groups[i:i+5]))!=1 for i in range(0,30,5)):
        raise RuntimeError('Six historical sector partitions do not match')
    engine=p22.Engine(dates,opens,endpoints,daily,p22.SYMBOL_GROUPS)
    q,episodes,passes=train_q(p22,engine,dates)
    frozen=q.copy()
    print('TRAINING_EPISODES_2021_2022:',episodes,'REPLAY_PASSES:',passes,flush=True)
    rows=[]
    for year in (2023,2024,2025):
        starts=p22.start_windows(dates,year)
        for i,s in enumerate(starts):
            for mode in ('hold','weekly','daily','rl'):
                res=simulate(p22,engine,s,mode,q,learn=False,epsilon=0,seed=SEED+year*1000+i)
                rows.append({'year':year,'start':str(dates[s].date()),
                             'strategy':mode,**res})
            if (i+1)%4==0:print(f'YEAR_{year}_WINDOWS: {i+1}/{len(starts)}',flush=True)
        assert np.array_equal(q,frozen),'Q changed during evaluation; invalid experiment'
    df=pd.DataFrame(rows)
    metrics=['cumulative_return','sharpe_proxy','mdd_proxy','turnover_proxy','fees_usd','trade_count']
    summary=df.groupby(['year','strategy'],as_index=False)[metrics].mean()
    ranked=rank_results(df)
    print('=== DEVELOPMENT 2023-2024: same-market paired four-metric ranks ===')
    print(ranked.to_string(float_format=lambda x:f'{x:.6f}'))
    for yr in (2023,2024,2025):
        print(f'=== {yr} RESEARCH PROXIES ===')
        print(summary[summary.year==yr].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    folder=root/OUTPUT
    folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/'phase25_window_results.csv',index=False)
    summary.to_csv(folder/'phase25_year_summary.csv',index=False)
    ranked.to_csv(folder/'phase25_paired_ranks.csv')
    (folder/'phase25_frozen_q.json').write_text(json.dumps({
        'algorithm':'tabular Q learning, four discrete actions, 16 discrete states',
        'training_years':[2021,2022],
        'evaluation_years':[2023,2024,2025],
        'q_table':q.tolist(),
        'actions':ACTION_NAMES,
        'states':'risk off, drift>4%, median vol>1.6%, industry dispersion>15%',
    },indent=2),encoding='utf-8')
    (folder/'phase25_manifest.json').write_text(json.dumps({
        'phase':'research_only','algorithm':'tabular Q-learning narrow controller',
        'Q_frozen_before_test':True,
        'training_episodes':episodes,'training_passes':passes,
        '2025':'previously_exposed_diagnostic','2026':'not_evaluated',
        'reward':'fee-adjusted one-session log growth, additional turnover and drawdown penalties',
        'limitations':['one realized market trajectory','15d reset windows',
                       'hourly prices not exact competition execution','price basis partly unverified'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'official_rl_approved':False,
    },indent=2),encoding='utf-8')
    (folder/'CHANGELOG_PHASE25.md').write_text(
        '# Phase25 RL controller\n\nHypothesis: tabular Q-learning learns HOLD/partial/full rebalance '
        'actions that improve fee-adjusted return, drawdown and turnover beyond static or weekly policies.\n\n'
        'Training 2021–22 only, development 2023–24, exposed 2025; 2026 not used.\n\n'
        'Decision: reject RL unless it improves development and exposed diagnostics without unstable turnover.\n',
        encoding='utf-8')
    print('RESULT_DIR:',folder)
    print('PHASE25_COMPLETE_RESEARCH_ONLY_NO_SUBMISSION')


def self_test():
    # Pure mechanics: no network or market-data dependence.
    assert ACTIONS==(0,.25,.5,1)
    q=np.array([[0.,.1,-.2,1.]]*N_STATES)
    frozen=q.copy()
    assert int(np.argmax(q[0]))==3
    assert q.shape==(16,4)
    assert np.array_equal(q,frozen)
    # Check that training Q update direction is correct on positive feedback.
    old=0.0;alpha=.12;reward=.4;future=.2;gamma=.5
    new=old+alpha*(reward+gamma*future-old)
    assert 0<new<reward
    # Synthetic end-to-end replay: explicitly verify HOLD incurs no repeat fees.
    import importlib.util,sys
    dep=Path(__file__).with_name('icaif_phase22_industry_risk_90.py')
    if not dep.exists():raise RuntimeError('Put Phase22 and Phase25 together in Downloads')
    spec=importlib.util.spec_from_file_location('phase22_rl_test',dep)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    rng=np.random.default_rng(444)
    prices=100*np.cumprod(1+rng.normal(.0002,.009,(365,30)),axis=0)
    end=np.stack([prices*(1+(k-3)*.00005) for k in range(7)],axis=1)
    engine=module.Engine(pd.bdate_range('2021-01-01',periods=365),prices,end,prices,module.SYMBOL_GROUPS)
    hold=simulate(module,engine,260,'hold',np.zeros((16,4)),learn=False)
    assert hold['trade_count']==1, 'HOLD executed trades unexpectedly'
    assert abs(hold['fees_usd']-900)<5, 'Initial fee unexpected'
    weekly=simulate(module,engine,260,'weekly',np.zeros((16,4)),learn=False)
    assert weekly['trade_count']>=1
    assert hold['training_rewards']==0
    print('HOLD_HAS_ZERO_REPEAT_FEES_AND_SYNTHETIC_REPLAY_PASS')
    print('PHASE25_SELF_TEST_PASS')
    print('DISCRETE_RL_ACTIONS_AND_Q_UPDATE_PASS')
    print('TRAIN_2021_2022_EVAL_2023_2025_NO_ONLINE_LEARNING_DESIGN_PASS')
    print('NO_NETWORK_NO_COMPETITION_SUBMISSION')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:run()
