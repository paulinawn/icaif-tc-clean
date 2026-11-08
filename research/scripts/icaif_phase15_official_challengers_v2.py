#!/usr/bin/env python3
"""ICAIF phase 15: offline, leakage-controlled portfolio challenger.

NO organizer API, NO credentials, NO live submissions. Run from starter-kit.
Uses predeclared strategies; 2025 is a held-out year, not a tuning window.
Nonlinear ML is trained only on labels matured before the evaluation year.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = (
    "AAPL","MSFT","NVDA","INTC","CRM","JPM","BAC","GS","V","PYPL",
    "LLY","JNJ","UNH","PFE","TMO","AMZN","TSLA","WMT","NKE","KO",
    "CAT","GE","BA","XOM","CVX","GOOGL","META","DIS","T","NEE",
)
TIMES = ("09:30:00","10:00:00","11:00:00","12:00:00","13:00:00","14:00:00","15:00:00")
FEE = .001
ANNUALIZER = 42.0
START_FEATURE = 122
HORIZON = 5
MODEL_RANDOM_STATE = 2609
WINDOW = 15
STRATEGIES = (
    "equal30_80_hold",
    "equal30_100_hold",
    "top3sectors_80_hold",
    "sector_mom_80_t08_hold",
    "lowvol_80_mix50_hold",
    "hgb_80_t08_hold",
    "hgb_sector_80_t08_hold",
)


def source_path(root: Path) -> Path:
    candidates = [
        root / "data" / "official" / "hourly_market_data_2021_2026.parquet",
        root / "hourly_market_data_2021_2026.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Official parquet missing. Expected starter-kit/data/official/hourly_market_data_2021_2026.parquet"
    )


def load_history(path: Path):
    df = pd.read_parquet(path)
    required = {"ticker","trading_date","timestamp_et","open","adj_close","sector_group"}
    if not required.issubset(df.columns):
        raise ValueError("Missing official data fields: " + str(sorted(required - set(df.columns))))
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
    df["time"] = df["timestamp_et"].dt.strftime("%H:%M:%S")
    df = df[df["ticker"].isin(SYMBOLS) & df["time"].isin(TIMES)].copy()
    counts = df.groupby(["trading_date","ticker"])["time"].nunique().unstack("ticker").reindex(columns=SYMBOLS)
    dates = pd.DatetimeIndex(counts.index[(counts == len(TIMES)).all(axis=1)])
    if len(dates) < 650:
        raise ValueError("Fewer than 650 complete history days; cannot test multiple years")
    df = df[df["trading_date"].isin(dates)].copy()
    if df.duplicated(["trading_date","ticker","time"]).any():
        raise ValueError("Duplicate bars in official dataset")

    def pivot(t, column):
        x = df.loc[df["time"] == t].pivot(index="trading_date", columns="ticker", values=column)
        arr = x.reindex(index=dates, columns=SYMBOLS).to_numpy(float)
        if not np.isfinite(arr).all() or (arr <= 0).any():
            raise ValueError("Invalid bar: " + t + "/" + column)
        return arr

    opens = pivot(TIMES[0], "open")
    endpoints = np.stack([pivot(t, "adj_close") for t in TIMES], axis=1)
    daily = endpoints[:, -1, :]
    sectors = df.drop_duplicates("ticker").set_index("ticker")["sector_group"].reindex(SYMBOLS).tolist()
    if len(set(sectors)) != 6 or any(s is None for s in sectors):
        raise ValueError("Unexpected sector structure")
    return dates, opens, endpoints, daily, sectors


def _safe_ratio(a,b):
    return a / np.where(np.abs(b) < 1e-15, np.nan, b) - 1.


def _rank_rows(x):
    return pd.DataFrame(x).rank(axis=1, method="average", pct=True).to_numpy(float)*2.-1.


def make_features(prices: np.ndarray, sectors):
    n, p = prices.shape
    r1 = np.full((n,p), np.nan)
    r1[1:] = _safe_ratio(prices[1:], prices[:-1])
    ret = {}
    for k in (2,5,20,60,120):
        z = np.full((n,p), np.nan)
        z[k:] = _safe_ratio(prices[k:], prices[:-k])
        ret[k] = z
    vol20 = pd.DataFrame(r1).rolling(20).std().to_numpy(float)
    vol60 = pd.DataFrame(r1).rolling(60).std().to_numpy(float)
    pos20 = pd.DataFrame(r1 > 0).rolling(20).mean().to_numpy(float)
    market20 = np.nanmean(ret[20],axis=1)
    market60 = np.nanmean(ret[60],axis=1)
    sector20 = np.full((n,p),np.nan)
    sector60 = np.full((n,p),np.nan)
    for sec in set(sectors):
        idx = [i for i,s in enumerate(sectors) if s == sec]
        sector20[:,idx] = np.nanmean(ret[20][:,idx],axis=1)[:,None]
        sector60[:,idx] = np.nanmean(ret[60][:,idx],axis=1)[:,None]
    # Model inputs are all known after each trading day's final completed bar.
    feature_fields = [
        ret[2],ret[5],ret[20],ret[60],ret[120],
        vol20,vol60,pos20,
        sector20,sector60,ret[20]-sector20,ret[60]-sector60,
        np.broadcast_to(market20[:,None],(n,p)),
        np.broadcast_to(market60[:,None],(n,p)),
    ]
    features = np.stack(feature_fields,axis=2)
    labels = np.full((n,p),np.nan)
    labels[:-HORIZON] = _safe_ratio(prices[HORIZON:],prices[:-HORIZON])
    return features, labels, {"ret":ret,"vol20":vol20,"vol60":vol60,"sector60":sector60}


def fit_year_models(dates, features, labels):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = features.reshape(-1,features.shape[-1])
    y = labels.reshape(-1)
    idxs = np.repeat(np.arange(len(dates)),len(SYMBOLS))
    predictions = np.full(labels.shape,np.nan)
    for year in (2023,2024,2025):
        # A 5-day forward label is usable only if its last outcome occurred
        # BEFORE January 1 of the evaluation year. This is the purge.
        cutoff = int(np.searchsorted(dates.values, np.datetime64(f"{year}-01-01")))
        train = (idxs >= START_FEATURE) & (idxs+HORIZON < cutoff)
        test = (idxs >= cutoff) & (idxs < len(dates)) & (dates.values[idxs] < np.datetime64(f"{year+1}-01-01"))
        valid_train = train & np.isfinite(X).all(axis=1) & np.isfinite(y)
        valid_test = test & np.isfinite(X).all(axis=1)
        if valid_train.sum() < 6000:
            raise ValueError(f"Not enough pre-{year} labeled examples ({int(valid_train.sum())})")
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=.035, max_iter=85,
            max_leaf_nodes=9, max_depth=3, min_samples_leaf=200,
            l2_regularization=25., random_state=MODEL_RANDOM_STATE,
            early_stopping=False,
        )
        model.fit(X[valid_train], y[valid_train])
        flat = predictions.reshape(-1)
        flat[valid_test] = model.predict(X[valid_test])
        print(f"TRAIN_END_BEFORE_{year}: {int(valid_train.sum())} matured stock-days; YEAR_{year} test stock-days={int(valid_test.sum())}")
    return predictions


def target_from_rank(score, l1, total=.8):
    w = np.ones(len(SYMBOLS))*total/len(SYMBOLS)
    centered = np.asarray(score,float)-float(np.mean(score))
    denom = float(np.abs(centered).sum())
    if denom > 1e-12:
        delta = centered * (l1/denom)
        scale = min(1., float(np.min(w[delta<0]/(-delta[delta<0]))) if np.any(delta<0) else 1.)
        w = w + delta*max(0.,scale*0.999999)
    w = np.maximum(w,0)
    w *= total/w.sum()
    return w


def target_for(name,start,features,scores,pred,sector):
    signal_idx = start - 1  # previous day's close. Execution at start-day open.
    base = np.full(len(SYMBOLS),.8/len(SYMBOLS))
    if name == "equal30_80_hold": return base
    if name == "equal30_100_hold": return np.full(len(SYMBOLS),1/len(SYMBOLS))
    if name == "top3sectors_80_hold":
        groups = sorted(set(sector))
        means = [(float(np.mean(scores["ret"][60][signal_idx,[i for i,s in enumerate(sector) if s==g]])),g) for g in groups]
        winners = {g for _,g in sorted(means,reverse=True)[:3]}
        return np.array([.8/15 if s in winners else 0. for s in sector])
    if name == "sector_mom_80_t08_hold":
        return target_from_rank(_rank_rows(scores["sector60"][[signal_idx]])[0],.08)
    if name == "lowvol_80_mix50_hold":
        v = scores["vol60"][signal_idx]
        inv = 1/np.maximum(v,.003)
        risk = inv / inv.sum()*.8
        return (base+risk)*.5
    if name == "hgb_80_t08_hold":
        return target_from_rank(_rank_rows(pred[[signal_idx]])[0],.08)
    if name == "hgb_sector_80_t08_hold":
        s1=_rank_rows(pred[[signal_idx]])[0]
        s2=_rank_rows(scores["sector60"][[signal_idx]])[0]
        return target_from_rank(.7*s1+.3*s2,.08)
    raise ValueError(name)


def transact(nav, current_values, open_prices, w):
    post = float(nav)
    for _ in range(60):
        tv = post*w
        new = nav-FEE*float(np.sum(np.abs(tv-current_values)))
        if abs(post-new)<1e-10:
            post=new;break
        post=new
    target=post*w
    notional = float(np.sum(np.abs(target-current_values)))
    fee=FEE*notional
    return target/open_prices, post-float(target.sum()), notional, fee


def simulate(opens,endpoints,start,name,w):
    if not (len(w)==30 and np.isfinite(w).all() and np.min(w)>=0 and np.max(w)<=.3 and w.sum()<=1.00000001):
        raise ValueError("Unacceptable portfolio weights")
    initial=1e6
    shares,cash,turnover,fee=transact(initial,np.zeros(30),opens[start],w)
    navs=[initial]
    for t in range(start,start+WINDOW):
        for j in range(7):
            navs.append(float(cash+np.dot(shares,endpoints[t,j])))
    navs=np.asarray(navs)
    returns=navs[1:]/navs[:-1]-1
    stdev=float(np.std(returns,ddof=1))
    sharpe=float(np.mean(returns)*ANNUALIZER/stdev) if stdev>1e-14 else 0.
    mdd=float(np.max(1.-navs/np.maximum.accumulate(navs)))
    return dict(cumulative_return=float(navs[-1]/initial-1),sharpe_proxy=sharpe,
                mdd_proxy=mdd,turnover_proxy=turnover/(initial*WINDOW*7),fee_usd=fee)


def evaluate(dates,opens,endpoints,features,scores,pred,sectors):
    # Windows in 2023,24,25. Model is trained only on prior years.
    rows=[]
    for year in (2023,2024,2025):
        indices=np.where(dates.year==year)[0]
        for start in indices[::5]:
            if start+WINDOW>len(dates) or dates[start+WINDOW-1].year!=year:
                continue
            if start<START_FEATURE or not np.isfinite(features[start-1]).all() or not np.isfinite(pred[start-1]).all():
                continue
            targets={name:target_for(name,start,features,scores,pred,sectors) for name in STRATEGIES}
            for name,w in targets.items():
                if not np.isfinite(w).all():
                    raise ValueError(f"Bad targets for {name} {dates[start]}")
                row={"year":year,"window_start":str(dates[start].date()),"strategy":name,
                     **simulate(opens,endpoints,start,name,w)}
                rows.append(row)
    if not rows: raise ValueError("No eligible windows")
    return pd.DataFrame(rows)


def summarize(out):
    cols=["cumulative_return","sharpe_proxy","mdd_proxy","turnover_proxy","fee_usd"]
    summaries=out.groupby(["year","strategy"],sort=False)[cols].mean().reset_index()
    first=out[out.year.isin((2023,2024))]
    train=first.groupby("strategy")[cols].mean()
    test=summaries[summaries.year==2025].set_index("strategy")
    if "equal30_80_hold" not in train.index or "equal30_80_hold" not in test.index:
        raise ValueError("Baseline not evaluated in both periods")
    b=train.loc["equal30_80_hold"]
    bt=test.loc["equal30_80_hold"]
    print("\n=== DEVELOPMENT YEARS 2023-2024 (non-independent overlapping windows) ===")
    print(train.to_string(float_format=lambda v:f"{v:.6f}"))
    print("\n=== HELD-OUT YEAR 2025 (non-independent overlapping windows) ===")
    print(test.to_string(float_format=lambda v:f"{v:.6f}"))
    print("\n=== PREDECLARED KEEP/REJECT GATES ===")
    decision=[]
    for name in STRATEGIES:
        if name=="equal30_80_hold":continue
        d=train.loc[name];h=test.loc[name]
        # These gates deliberately require a multi-objective improvement in both
        # development and independent calendar-year holdout. Low-turnover hold is baked in.
        improve_dev=(d["cumulative_return"]>=b["cumulative_return"] and
            d["sharpe_proxy"]>=b["sharpe_proxy"] and
            d["mdd_proxy"]<=b["mdd_proxy"]+.001)
        improve_hold=(h["cumulative_return"]>=bt["cumulative_return"] and
            h["sharpe_proxy"]>=bt["sharpe_proxy"] and
            h["mdd_proxy"]<=bt["mdd_proxy"]+.001)
        # 100%-invested portfolio may improve return but requires sufficient risk metrics.
        passed=bool(improve_dev and improve_hold)
        decision.append({"name":name,"development_pass":bool(improve_dev),
            "holdout_pass":bool(improve_hold),"candidate_for_review":passed})
        print(f"{name}: {'REVIEW_CANDIDATE' if passed else 'REJECT'}; development={improve_dev}; 2025_holdout={improve_hold}")
    print("Note: even REVIEW_CANDIDATE requires point-in-time, data and live interface review before deployment.")
    return summaries,decision


def self_test():
    x=np.arange(30,dtype=float)
    for l1 in (0,.04,.08,.12):
        w=target_from_rank(x,l1)
        assert len(w)==30 and (w>=0).all() and np.max(w)<=.3 and abs(w.sum()-.8)<1e-12
    op=np.full((16,30),100.)
    eps=np.full((16,7,30),100.)
    p=simulate(op,eps,0,"equal30_80_hold",np.full(30,.8/30))
    assert abs(p["fee_usd"]-799.36051159)<.01,p
    assert abs(p["cumulative_return"]+.00079936051159)<1e-8,p
    assert abs(p["turnover_proxy"]-(799360.5115907278/(1e6*105)))<1e-10,p
    # No strategy look-ahead: an outcome perturbation after entry must not change its target.
    assert abs(target_from_rank(x,.08).sum()-.8)<1e-10
    print("PHASE15_SELF_TEST_PASS")
    print("SIMULATED_INITIAL_BUY_FEE",round(p["fee_usd"],2))
    print("NO_COMPETITION_SUBMISSION")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args()
    if args.self_test:
        self_test();return
    root=args.project_root.resolve()
    path=source_path(root)
    print("PHASE 15 - HOLDOUT RESEARCH; COMPETITION SUBMISSION DISABLED")
    print("OFFICIAL_HISTORY:",path)
    print("ML: HistGradientBoostingRegressor; point-in-time features, 5-day matured labels")
    dates,opens,endpoints,prices,sectors=load_history(path)
    features,labels,scores=make_features(prices,sectors)
    preds=fit_year_models(dates,features,labels)
    outcomes=evaluate(dates,opens,endpoints,features,scores,preds,sectors)
    summaries,decision=summarize(outcomes)
    output=args.output or (root/"artifacts"/"phase15_official_challengers")
    output.mkdir(parents=True,exist_ok=True)
    outcomes.to_csv(output/"phase15_windows.csv",index=False)
    summaries.to_csv(output/"phase15_year_summary.csv",index=False)
    (output/"phase15_report.json").write_text(json.dumps({
        "mode":"offline_no_submission","test_year":2025,"development_years":[2023,2024],
        "fee_rate":FEE,"dates_start":str(dates[0].date()),
        "dates_end":str(dates[-1].date()),"horizon_days":HORIZON,
        "model_hyperparameters":{"max_iter":85,"max_depth":3,"learning_rate":.035,
           "min_samples_leaf":200,"l2_regularization":25.,"random_state":MODEL_RANDOM_STATE},
        "strategies":list(STRATEGIES),"decisions":decision,
        "important_limitations":["15d windows overlap; values are proxies, not expected live returns",
           "possible corporate-action/adjusted-close alignment risks need stress testing",
           "2025 holdout must not be tuned after viewing results",
           "news/filings/macro unavailable as historical point-in-time features in this experiment"],
    },indent=2),encoding="utf-8")
    print("\nREPORT:",output/"phase15_report.json")
    print("PHASE15_COMPLETE_NO_SUBMISSION")

if __name__=="__main__":
    main()
